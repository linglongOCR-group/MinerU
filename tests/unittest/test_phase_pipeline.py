import asyncio
import json
from pathlib import Path
from PIL import Image

from mineru.cli import document_ocr


def _make_image(path: Path, size=(16, 12)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 255, 255)).save(path)


def test_ocr_phase_enum_values():
    assert document_ocr.OcrPhase.FULL.value == "full"
    assert document_ocr.OcrPhase.LAYOUT.value == "layout"
    assert document_ocr.OcrPhase.RECOGNIZE.value == "recognize"


def test_document_ocr_options_has_phase_field():
    opts = document_ocr.DocumentOcrOptions(
        input_path=Path("/tmp/in"),
        output_dir=Path("/tmp/out"),
        phase=document_ocr.OcrPhase.LAYOUT,
    )
    assert opts.phase == document_ocr.OcrPhase.LAYOUT


def test_document_ocr_options_accept_explicit_layout_paths(tmp_path):
    opts = document_ocr.DocumentOcrOptions(
        input_path=tmp_path / "input",
        output_dir=tmp_path / "recognition_out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
        layout_input_path=tmp_path / "layout_in",
        layout_output_dir=tmp_path / "layout_out",
        max_http_concurrency_per_window=3,
        per_window_timeout=42.0,
    )

    assert opts.layout_input_path == tmp_path / "layout_in"
    assert opts.layout_output_dir == tmp_path / "layout_out"
    assert opts.max_http_concurrency_per_window == 3
    assert opts.per_window_timeout == 42.0


def test_document_ocr_options_rejects_positional_arguments(tmp_path):
    try:
        document_ocr.DocumentOcrOptions(tmp_path / "input", tmp_path / "out")
    except TypeError:
        pass
    else:
        raise AssertionError("DocumentOcrOptions should be keyword-only")


def test_document_ocr_options_phase_defaults_to_full():
    opts = document_ocr.DocumentOcrOptions(
        input_path=Path("/tmp/in"),
        output_dir=Path("/tmp/out"),
    )
    assert opts.phase == document_ocr.OcrPhase.FULL


def test_layout_cache_write_and_read(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    job = document_ocr.DocumentJob.from_path(
        path=image_path,
        document_type="image",
        stem="page",
        parse_dir=tmp_path / "out" / "page" / "vlm",
        order=0,
        page_count=1,
        start_page_id=0,
        end_page_id=0,
    )
    window = document_ocr.WindowJob.from_document(
        job, document_index=0, window_index=0, start_page_id=0, end_page_id=0,
    )
    blocks_by_page = [[{"type": "text", "bbox": [0, 0, 1, 1], "content": "hello"}]]
    page_sizes = [[16, 12]]

    document_ocr.write_layout_window_cache(window, blocks_by_page, page_sizes, 0.5)

    cached = document_ocr.read_valid_layout_window_cache(window)
    assert cached is not None
    assert cached["blocks_by_page"] == blocks_by_page

    # Stale cache should be rejected
    stale_job = document_ocr.DocumentJob(
        **{**job.__dict__, "source_size": job.source_size + 1}
    )
    stale_window = document_ocr.WindowJob.from_document(
        stale_job, document_index=0, window_index=0, start_page_id=0, end_page_id=0,
    )
    assert document_ocr.read_valid_layout_window_cache(stale_window) is None


def test_process_layout_window_only_calls_layout(monkeypatch, tmp_path):
    """process_layout_window calls layout_detect and writes layout cache, not recognition."""
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    job = document_ocr.DocumentJob.from_path(
        image_path, "image", "page",
        tmp_path / "out" / "page" / "vlm", 0, 1, 0, 0,
    )
    window = document_ocr.WindowJob.from_document(job, 0, 0, 0, 0)

    layout_called = []
    recognition_called = []

    class FakeClient:
        async def aio_layout_detect(self, image, priority=None, semaphore=None, scored=None):
            layout_called.append(True)
            from mineru_vl_utils.structs import ContentBlock, ExtractResult
            return ExtractResult([ContentBlock("text", [0, 0, 1, 1], content="layout-only")])

        async def aio_recognize_from_layout(self, *args, **kwargs):
            recognition_called.append(True)
            raise AssertionError("recognize_from_layout should not be called in layout phase")

    options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "out",
        phase=document_ocr.OcrPhase.LAYOUT,
    )

    asyncio.run(document_ocr.process_layout_window(FakeClient(), window, options))

    assert layout_called, "layout_detect was not called"
    assert not recognition_called, "recognize_from_layout was called but should not have been"
    cached = document_ocr.read_valid_layout_window_cache(window)
    assert cached is not None
    assert cached["blocks_by_page"][0][0]["content"] == "layout-only"


def test_process_recognition_window_uses_layout_cache(monkeypatch, tmp_path):
    """process_recognition_window reads layout cache and calls recognize_from_layout."""
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    job = document_ocr.DocumentJob.from_path(
        image_path, "image", "page",
        tmp_path / "out" / "page" / "vlm", 0, 1, 0, 0,
    )
    window = document_ocr.WindowJob.from_document(job, 0, 0, 0, 0)

    # Pre-write layout cache
    document_ocr.write_layout_window_cache(
        window,
        blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1], "content": None}]],
        page_sizes=[[16, 12]],
        elapsed_seconds=0.1,
    )

    recognition_args = []

    class FakeClient:
        async def aio_recognize_from_layout(self, image, layout_blocks, **kwargs):
            recognition_args.append((image, layout_blocks))
            from mineru_vl_utils.structs import ExtractResult
            if isinstance(layout_blocks, ExtractResult):
                for block in layout_blocks:
                    block.content = "recognized"
                return layout_blocks
            return ExtractResult([])

    options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
    )

    asyncio.run(document_ocr.process_recognition_window(FakeClient(), window, options))

    assert recognition_args, "recognize_from_layout was not called"
    cached = document_ocr.read_valid_window_cache(window)
    assert cached is not None
    assert cached["blocks_by_page"][0][0]["content"] == "recognized"


def test_process_recognition_window_uses_layout_artifact_without_layout_cache(monkeypatch, tmp_path):
    """Recognition accepts the document layout artifact as the phase boundary."""
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    job = document_ocr.DocumentJob.from_path(
        image_path, "image", "page",
        tmp_path / "out" / "page" / "vlm", 0, 1, 0, 0,
    )
    window = document_ocr.WindowJob.from_document(job, 0, 0, 0, 0)

    document_ocr.write_layout_window_cache(
        window,
        blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1], "content": None}]],
        page_sizes=[[16, 12]],
        elapsed_seconds=0.1,
    )
    document_ocr.write_document_layout_artifact(job, [window])
    document_ocr.layout_cache_path(window).unlink()

    class FakeClient:
        async def aio_recognize_from_layout(self, image, layout_blocks, **kwargs):
            for block in layout_blocks:
                block.content = "recognized-from-artifact"
            return layout_blocks

    options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
    )

    asyncio.run(document_ocr.process_recognition_window(FakeClient(), window, options))

    cached = document_ocr.read_valid_window_cache(window)
    assert cached is not None
    assert cached["blocks_by_page"][0][0]["content"] == "recognized-from-artifact"


def test_layout_artifact_path_uses_layout_input_root(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    job = document_ocr.DocumentJob.from_path(
        image_path,
        "image",
        "page",
        tmp_path / "recognition_out" / "page" / "vlm",
        0,
        1,
        0,
        0,
    )
    options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "recognition_out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
        layout_input_path=tmp_path / "layout_root",
    )

    path = document_ocr.resolve_layout_artifact_path(job, options)

    assert path == tmp_path / "layout_root" / "page" / "vlm" / "page_layout.json"


def test_layout_artifact_path_accepts_direct_file_for_single_document(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    artifact_path = tmp_path / "page_layout.json"
    artifact_path.write_text("{}", encoding="utf-8")
    job = document_ocr.DocumentJob.from_path(
        image_path,
        "image",
        "page",
        tmp_path / "recognition_out" / "page" / "vlm",
        0,
        1,
        0,
        0,
    )
    options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "recognition_out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
        layout_input_path=artifact_path,
    )

    path = document_ocr.resolve_layout_artifact_path(job, options)

    assert path == artifact_path


def test_recognize_requires_layout_input_path(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "recognition_out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
    )

    try:
        document_ocr.validate_phase_options(options, [image_path])
    except Exception as exc:
        assert "--layout-input is required for recognize" in str(exc)
    else:
        raise AssertionError("Expected recognize without layout_input_path to fail")


def test_direct_layout_artifact_rejects_multiple_documents(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _make_image(image_a)
    _make_image(image_b)
    artifact_path = tmp_path / "layout.json"
    artifact_path.write_text("{}", encoding="utf-8")
    options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "recognition_out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
        layout_input_path=artifact_path,
    )

    try:
        document_ocr.validate_phase_options(options, [image_a, image_b])
    except Exception as exc:
        assert "direct --layout-input artifact can only be used with one source document" in str(exc)
    else:
        raise AssertionError("Expected direct layout artifact with multiple documents to fail")


def test_process_full_window_calls_both_stages(monkeypatch, tmp_path):
    """process_full_window calls layout then recognition, writing both caches."""
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    job = document_ocr.DocumentJob.from_path(
        image_path, "image", "page",
        tmp_path / "out" / "page" / "vlm", 0, 1, 0, 0,
    )
    window = document_ocr.WindowJob.from_document(job, 0, 0, 0, 0)

    from mineru_vl_utils.structs import ContentBlock, ExtractResult

    class FakeClient:
        async def aio_layout_detect(self, image, priority=None, semaphore=None, scored=None):
            return ExtractResult([ContentBlock("text", [0, 0, 1, 1])])

        async def aio_recognize_from_layout(self, image, layout_blocks, **kwargs):
            for block in layout_blocks:
                block.content = "full-pipeline"
            return layout_blocks

    options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "out",
        phase=document_ocr.OcrPhase.FULL,
    )

    asyncio.run(document_ocr.process_full_window(FakeClient(), window, options))

    layout_cached = document_ocr.read_valid_layout_window_cache(window)
    assert layout_cached is not None

    recognition_cached = document_ocr.read_valid_window_cache(window)
    assert recognition_cached is not None
    assert recognition_cached["blocks_by_page"][0][0]["content"] == "full-pipeline"


def _make_pdf_placeholder(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\n%placeholder\n")


def test_run_layout_phase_writes_layout_artifact(monkeypatch, tmp_path):
    """Layout phase produces _layout.json but NOT _model.json or .md."""
    from mineru.cli import layout_artifact

    image_path = tmp_path / "doc.png"
    _make_image(image_path)

    jobs = [
        document_ocr.DocumentJob.from_path(
            image_path, "image", "doc",
            tmp_path / "out" / "doc" / "vlm", 0, 1, 0, 0,
        ),
    ]
    monkeypatch.setattr(document_ocr, "collect_document_jobs", lambda *a, **kw: jobs)

    async def fake_process_layout_window(client, window, options):
        document_ocr.write_layout_window_cache(
            window,
            blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1]}]],
            page_sizes=[[16, 12]],
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(document_ocr, "process_layout_window", fake_process_layout_window)

    class FakeClient:
        pass

    options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "out",
        phase=document_ocr.OcrPhase.LAYOUT,
        layout_output_dir=tmp_path / "out",
        progress=False,
    )
    results = asyncio.run(
        document_ocr.run_document_ocr(options, client_factory=lambda _o: FakeClient())
    )

    assert results[0].status == "completed"
    layout_path = tmp_path / "out" / "doc" / "vlm" / "doc_layout.json"
    assert layout_path.exists(), "Layout artifact should exist"
    assert not (tmp_path / "out" / "doc" / "vlm" / "doc_model.json").exists()
    assert not (tmp_path / "out" / "doc" / "vlm" / "doc.md").exists()


def test_layout_phase_writes_artifact_to_layout_output_dir(monkeypatch, tmp_path):
    image_path = tmp_path / "doc.png"
    _make_image(image_path)

    async def fake_process_layout_window(client, window, options):
        document_ocr.write_layout_window_cache(
            window,
            blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1]}]],
            page_sizes=[[16, 12]],
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(document_ocr, "process_layout_window", fake_process_layout_window)

    options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "unused_layout_work",
        phase=document_ocr.OcrPhase.LAYOUT,
        layout_output_dir=tmp_path / "layout_artifacts",
        progress=False,
    )

    results = asyncio.run(document_ocr.run_document_ocr(options, client_factory=lambda _opts: object()))

    assert results[0].status == "completed"
    assert (tmp_path / "layout_artifacts" / "doc" / "vlm" / "doc_layout.json").exists()
    assert not (tmp_path / "unused_layout_work" / "doc" / "vlm" / "doc_layout.json").exists()


def test_run_phase_writes_layout_artifact_to_default_nested_dir(monkeypatch, tmp_path):
    image_path = tmp_path / "doc.png"
    _make_image(image_path)

    async def fake_process_full_window(client, window, options):
        document_ocr.write_layout_window_cache(
            window,
            blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1]}]],
            page_sizes=[[16, 12]],
            elapsed_seconds=0.1,
        )
        document_ocr.write_window_cache(
            window,
            blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1], "content": "hello"}]],
            page_sizes=[[16, 12]],
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(document_ocr, "process_full_window", fake_process_full_window)

    options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "full_out",
        phase=document_ocr.OcrPhase.FULL,
        progress=False,
    )

    results = asyncio.run(document_ocr.run_document_ocr(options, client_factory=lambda _opts: object()))

    assert results[0].status == "completed"
    assert (tmp_path / "full_out" / "_layout_artifacts" / "doc" / "vlm" / "doc_layout.json").exists()


def test_run_full_phase_writes_both_outputs(monkeypatch, tmp_path):
    """Full phase produces layout artifact AND standard OCR outputs."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)

    jobs = [
        document_ocr.DocumentJob.from_path(
            image_path, "image", "doc",
            tmp_path / "out" / "doc" / "vlm", 0, 1, 0, 0,
        ),
    ]
    monkeypatch.setattr(document_ocr, "collect_document_jobs", lambda *a, **kw: jobs)

    async def fake_process_full_window(client, window, options):
        document_ocr.write_layout_window_cache(
            window,
            blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1]}]],
            page_sizes=[[16, 12]],
            elapsed_seconds=0.1,
        )
        document_ocr.write_window_cache(
            window,
            blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1], "content": "hello"}]],
            page_sizes=[[16, 12]],
            elapsed_seconds=0.2,
        )

    monkeypatch.setattr(document_ocr, "process_full_window", fake_process_full_window)
    monkeypatch.setattr(
        document_ocr,
        "build_middle_json_for_document",
        lambda doc, pages, sizes: {"pdf_info": [{"page_idx": 0}]},
    )
    monkeypatch.setattr(document_ocr, "render_outputs", lambda mj: ("hello", []))

    class FakeClient:
        pass

    options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "out",
        phase=document_ocr.OcrPhase.FULL,
        progress=False,
    )
    results = asyncio.run(
        document_ocr.run_document_ocr(options, client_factory=lambda _o: FakeClient())
    )

    assert results[0].status == "completed"
    assert (tmp_path / "out" / "_layout_artifacts" / "doc" / "vlm" / "doc_layout.json").exists()
    assert (tmp_path / "out" / "doc" / "vlm" / "doc_model.json").exists()
    assert (tmp_path / "out" / "doc" / "vlm" / "doc.md").exists()


def test_run_full_repairs_missing_layout_cache_when_recognition_cache_exists(monkeypatch, tmp_path):
    """FULL reruns must not fail when old recognition caches lack layout caches."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)

    job = document_ocr.DocumentJob.from_path(
        image_path, "image", "doc",
        tmp_path / "out" / "doc" / "vlm", 0, 1, 0, 0,
    )
    window = document_ocr.WindowJob.from_document(job, 0, 0, 0, 0)
    document_ocr.write_window_cache(
        window,
        blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1], "content": "cached"}]],
        page_sizes=[[16, 12]],
        elapsed_seconds=0.2,
    )

    monkeypatch.setattr(document_ocr, "collect_document_jobs", lambda *a, **kw: [job])

    layout_repairs = []

    async def fake_process_layout_window(client, repair_window, options):
        layout_repairs.append(repair_window.window_index)
        document_ocr.write_layout_window_cache(
            repair_window,
            blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1]}]],
            page_sizes=[[16, 12]],
            elapsed_seconds=0.1,
        )

    async def fail_process_full_window(client, repair_window, options):
        raise AssertionError("recognition cache should only need layout repair")

    monkeypatch.setattr(document_ocr, "process_layout_window", fake_process_layout_window)
    monkeypatch.setattr(document_ocr, "process_full_window", fail_process_full_window)
    monkeypatch.setattr(
        document_ocr,
        "build_middle_json_for_document",
        lambda doc, pages, sizes: {"pdf_info": [{"page_idx": 0}]},
    )
    monkeypatch.setattr(document_ocr, "render_outputs", lambda mj: ("cached", []))

    options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "out",
        phase=document_ocr.OcrPhase.FULL,
        progress=False,
    )
    results = asyncio.run(
        document_ocr.run_document_ocr(options, client_factory=lambda _o: object())
    )

    assert results[0].status == "completed"
    assert layout_repairs == [0]
    assert document_ocr.read_valid_layout_window_cache(window) is not None
    assert (tmp_path / "out" / "_layout_artifacts" / "doc" / "vlm" / "doc_layout.json").exists()


def test_run_full_rehydrates_layout_cache_from_default_layout_artifact(monkeypatch, tmp_path):
    image_path = tmp_path / "doc.png"
    _make_image(image_path)

    job = document_ocr.DocumentJob.from_path(
        image_path, "image", "doc",
        tmp_path / "out" / "doc" / "vlm", 0, 1, 0, 0,
    )
    window = document_ocr.WindowJob.from_document(job, 0, 0, 0, 0)
    document_ocr.write_layout_window_cache(
        window,
        blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1]}]],
        page_sizes=[[16, 12]],
        elapsed_seconds=0.1,
    )
    document_ocr.write_document_layout_artifact(
        job,
        [window],
        layout_output_dir=tmp_path / "out" / "_layout_artifacts",
    )
    document_ocr.write_window_cache(
        window,
        blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1], "content": "cached"}]],
        page_sizes=[[16, 12]],
        elapsed_seconds=0.2,
    )
    document_ocr.layout_cache_path(window).unlink()

    async def fail_process_layout_window(client, repair_window, options):
        raise AssertionError("layout should not run")

    monkeypatch.setattr(document_ocr, "process_layout_window", fail_process_layout_window)
    monkeypatch.setattr(
        document_ocr,
        "build_middle_json_for_document",
        lambda doc, pages, sizes: {"pdf_info": [{"page_idx": 0}]},
    )
    monkeypatch.setattr(document_ocr, "render_outputs", lambda mj: ("cached", []))

    options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "out",
        phase=document_ocr.OcrPhase.FULL,
        progress=False,
    )
    results = asyncio.run(
        document_ocr.run_document_ocr(options, client_factory=lambda _opts: object())
    )

    assert results[0].status == "completed"
    assert document_ocr.read_valid_layout_window_cache(window) is not None

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


import asyncio


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

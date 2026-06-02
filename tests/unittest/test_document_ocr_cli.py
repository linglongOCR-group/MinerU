import asyncio
import json
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from mineru.cli import document_ocr


def _make_image(path: Path, size=(16, 12)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 255, 255)).save(path)


def _make_pdf_placeholder(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\n%placeholder\n")


def test_collect_document_jobs_discovers_mixed_inputs_recursively(monkeypatch, tmp_path):
    _make_image(tmp_path / "same.png")
    _make_pdf_placeholder(tmp_path / "nested" / "same.pdf")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    monkeypatch.setattr(document_ocr, "probe_pdf_page_count", lambda path: 7)

    jobs = document_ocr.collect_document_jobs(
        tmp_path,
        tmp_path / "out",
        start_page_id=1,
        end_page_id=5,
        resume=False,
    )

    assert [
        (job.path.name, job.document_type, job.stem, job.start_page_id, job.end_page_id)
        for job in jobs
    ] == [
        ("same.pdf", "pdf", "same", 1, 5),
        ("same.png", "image", "same_2", 0, 0),
    ]
    assert jobs[0].parse_dir == tmp_path / "out" / "same" / "vlm"
    assert jobs[0].selected_pages == 5


def test_build_pdf_windows_and_interleave_by_document(monkeypatch, tmp_path):
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    image = tmp_path / "c.png"
    _make_pdf_placeholder(pdf_a)
    _make_pdf_placeholder(pdf_b)
    _make_image(image)
    page_counts = {pdf_a: 9, pdf_b: 5}
    monkeypatch.setattr(document_ocr, "probe_pdf_page_count", lambda path: page_counts[path])

    jobs = document_ocr.collect_document_jobs(
        tmp_path,
        tmp_path / "out",
        start_page_id=0,
        end_page_id=None,
        resume=False,
    )
    windows = document_ocr.build_window_jobs(jobs, page_window_size=4)

    assert [
        (window.document_index, window.window_index, window.start_page_id, window.end_page_id)
        for window in windows
    ] == [
        (0, 0, 0, 3),
        (0, 1, 4, 7),
        (0, 2, 8, 8),
        (1, 0, 0, 3),
        (1, 1, 4, 4),
        (2, 0, 0, 0),
    ]
    assert [
        window.document_index for window in document_ocr.interleave_window_jobs(windows)
    ] == [0, 1, 2, 0, 1, 0]


def test_resume_skips_completed_documents(monkeypatch, tmp_path):
    pdf = tmp_path / "done.pdf"
    image = tmp_path / "todo.png"
    _make_pdf_placeholder(pdf)
    _make_image(image)
    monkeypatch.setattr(document_ocr, "probe_pdf_page_count", lambda path: 2)
    status_dir = tmp_path / "out" / "done" / "vlm"
    status_dir.mkdir(parents=True)
    (status_dir / "done_status.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )

    jobs = document_ocr.collect_document_jobs(
        tmp_path,
        tmp_path / "out",
        start_page_id=0,
        end_page_id=None,
        resume=True,
    )

    assert [(job.path.name, job.stem) for job in jobs] == [("todo.png", "todo")]


def test_window_cache_rejects_stale_metadata(tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)
    job = document_ocr.DocumentJob.from_path(
        path=image,
        document_type="image",
        stem="page",
        parse_dir=tmp_path / "out" / "page" / "vlm",
        order=0,
        page_count=1,
        start_page_id=0,
        end_page_id=0,
    )
    window = document_ocr.WindowJob.from_document(job, document_index=0, window_index=0, start_page_id=0, end_page_id=0)
    document_ocr.write_window_cache(
        window,
        blocks_by_page=[[{"type": "text", "content": "hello", "bbox": [0, 0, 1, 1]}]],
        page_sizes=[[16, 12]],
        elapsed_seconds=1.0,
    )

    assert document_ocr.read_valid_window_cache(window) is not None
    stale_job = document_ocr.DocumentJob(
        **{**job.__dict__, "source_size": job.source_size + 1}
    )
    stale_window = document_ocr.WindowJob.from_document(
        stale_job,
        document_index=0,
        window_index=0,
        start_page_id=0,
        end_page_id=0,
    )
    assert document_ocr.read_valid_window_cache(stale_window) is None


def test_assembly_sorts_windows_and_writes_document_outputs(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf_placeholder(pdf)
    job = document_ocr.DocumentJob.from_path(
        path=pdf,
        document_type="pdf",
        stem="doc",
        parse_dir=tmp_path / "out" / "doc" / "vlm",
        order=0,
        page_count=3,
        start_page_id=0,
        end_page_id=2,
    )
    first = document_ocr.WindowJob.from_document(job, 0, 0, 0, 1)
    second = document_ocr.WindowJob.from_document(job, 0, 1, 2, 2)
    document_ocr.write_window_cache(second, [[{"content": "p2"}]], [[10, 12]], 0.2)
    document_ocr.write_window_cache(first, [[{"content": "p0"}], [{"content": "p1"}]], [[10, 10], [10, 11]], 0.1)
    monkeypatch.setattr(
        document_ocr,
        "build_middle_json_for_document",
        lambda document, model_pages, page_sizes: {
            "pdf_info": [
                {"page_idx": page_index, "page_size": page_sizes[page_index]}
                for page_index in range(len(model_pages))
            ]
        },
    )
    monkeypatch.setattr(
        document_ocr,
        "render_outputs",
        lambda middle_json: (
            "\n".join(f"page-{page['page_idx']}" for page in middle_json["pdf_info"]),
            [{"page_idx": page["page_idx"]} for page in middle_json["pdf_info"]],
        ),
    )

    result = document_ocr.assemble_document_outputs(job, [second, first], elapsed_seconds=3.0)

    assert result.status == "completed"
    assert json.loads((job.parse_dir / "doc_model.json").read_text(encoding="utf-8")) == [
        [{"content": "p0"}],
        [{"content": "p1"}],
        [{"content": "p2"}],
    ]
    assert (job.parse_dir / "doc.md").read_text(encoding="utf-8") == "page-0\npage-1\npage-2"
    status = json.loads((job.parse_dir / "doc_status.json").read_text(encoding="utf-8"))
    assert status["document_type"] == "pdf"
    assert status["completed_windows"] == 2
    assert status["total_windows"] == 2


def test_run_document_ocr_continues_after_document_failure(monkeypatch, tmp_path):
    bad = tmp_path / "bad.png"
    good = tmp_path / "good.png"
    _make_image(bad)
    _make_image(good)
    jobs = [
        document_ocr.DocumentJob.from_path(bad, "image", "bad", tmp_path / "out" / "bad" / "vlm", 0, 1, 0, 0),
        document_ocr.DocumentJob.from_path(good, "image", "good", tmp_path / "out" / "good" / "vlm", 1, 1, 0, 0),
    ]
    monkeypatch.setattr(document_ocr, "collect_document_jobs", lambda *args, **kwargs: jobs)

    async def fake_process_window(client, window, options):
        if window.document_stem == "bad":
            raise RuntimeError("bad window")
        document_ocr.write_window_cache(
            window,
            blocks_by_page=[[{"content": "ok"}]],
            page_sizes=[[16, 12]],
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(document_ocr, "process_window_job", fake_process_window)
    monkeypatch.setattr(
        document_ocr,
        "build_middle_json_for_document",
        lambda document, model_pages, page_sizes: {"pdf_info": [{"page_idx": 0, "page_size": page_sizes[0]}]},
    )
    monkeypatch.setattr(document_ocr, "render_outputs", lambda middle_json: ("ok", []))

    results = asyncio.run(
        document_ocr.run_document_ocr(
            document_ocr.DocumentOcrOptions(input_path=tmp_path, output_dir=tmp_path / "out", progress=False),
            client_factory=lambda _options: object(),
        )
    )

    assert {result.job.stem: result.status for result in results} == {
        "bad": "failed",
        "good": "completed",
    }
    assert (tmp_path / "out" / "good" / "vlm" / "good.md").exists()
    assert not (tmp_path / "out" / "bad" / "vlm" / "bad.md").exists()


def test_cli_parses_document_options(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)
    captured = {}

    async def fake_run_document_ocr(options):
        captured.update(options.__dict__)
        job = document_ocr.DocumentJob.from_path(
            image,
            "image",
            "page",
            tmp_path / "out" / "page" / "vlm",
            0,
            1,
            0,
            0,
        )
        return [document_ocr.DocumentJobResult(job=job, status="completed", elapsed_seconds=0.1)]

    monkeypatch.setattr(document_ocr, "run_document_ocr", fake_run_document_ocr)

    result = CliRunner().invoke(
        document_ocr.main,
        [
            "-p",
            str(image),
            "-o",
            str(tmp_path / "out"),
            "-u",
            "http://shared",
            "--layout-url",
            "http://layout",
            "--recognition-url",
            "http://recognition",
            "--resume",
            "--page-window-size",
            "4",
            "--max-windows",
            "7",
            "--max-http-concurrency-per-window",
            "11",
            "--per-window-timeout",
            "12",
            "--http-timeout",
            "13",
            "--connect-timeout",
            "3",
            "--max-retries",
            "2",
            "--retry-backoff-factor",
            "0.25",
            "--start",
            "1",
            "--end",
            "5",
            "--no-progress",
            "--no-formula",
            "--no-table",
            "--no-image-analysis",
            "--dissection",
            "--stream",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["input_path"] == image
    assert captured["output_dir"] == tmp_path / "out"
    assert captured["server_url"] == "http://shared"
    assert captured["layout_server_url"] == "http://layout"
    assert captured["recognition_server_url"] == "http://recognition"
    assert captured["resume"] is True
    assert captured["page_window_size"] == 4
    assert captured["max_windows"] == 7
    assert captured["max_http_concurrency_per_window"] == 11
    assert captured["per_window_timeout"] == 12
    assert captured["http_timeout"] == 13
    assert captured["connect_timeout"] == 3
    assert captured["max_retries"] == 2
    assert captured["retry_backoff_factor"] == 0.25
    assert captured["start_page_id"] == 1
    assert captured["end_page_id"] == 5
    assert captured["progress"] is False
    assert captured["formula_enable"] is False
    assert captured["table_enable"] is False
    assert captured["image_analysis"] is False
    assert captured["dissection_enable"] is True
    assert captured["stream"] is True


def test_cli_rejects_stream_without_dissection(tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)

    result = CliRunner().invoke(
        document_ocr.main,
        ["-p", str(image), "-o", str(tmp_path / "out"), "--stream"],
    )

    assert result.exit_code != 0
    assert "--stream requires --dissection" in result.output

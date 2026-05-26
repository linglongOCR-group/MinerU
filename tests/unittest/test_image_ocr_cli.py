import asyncio
import json
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from mineru.cli import image_ocr


def _make_image(path: Path, size=(16, 12)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 255, 255)).save(path)


def test_collect_image_jobs_skips_completed_outputs_when_resuming(tmp_path):
    first = tmp_path / "a.png"
    second = tmp_path / "nested" / "a.jpg"
    ignored = tmp_path / "notes.txt"
    _make_image(first)
    _make_image(second)
    ignored.write_text("not an image", encoding="utf-8")

    status_dir = tmp_path / "out" / "a" / "vlm"
    status_dir.mkdir(parents=True)
    (status_dir / "a_status.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )

    jobs = image_ocr.collect_image_jobs(tmp_path, tmp_path / "out", resume=True)

    assert [(job.path.name, job.stem) for job in jobs] == [("a.jpg", "a_2")]
    assert jobs[0].parse_dir == tmp_path / "out" / "a_2" / "vlm"


def test_run_image_jobs_limits_concurrency_and_continues_after_failure(tmp_path):
    jobs = [
        image_ocr.ImageJob(tmp_path / "a.png", "a", tmp_path / "out" / "a" / "vlm"),
        image_ocr.ImageJob(tmp_path / "bad.png", "bad", tmp_path / "out" / "bad" / "vlm"),
        image_ocr.ImageJob(tmp_path / "c.png", "c", tmp_path / "out" / "c" / "vlm"),
    ]
    active = 0
    max_active = 0

    async def worker(job):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.01)
            if job.stem == "bad":
                raise RuntimeError("model returned invalid output")
            return image_ocr.ImageJobResult(job=job, status="completed", elapsed_seconds=0.01)
        finally:
            active -= 1

    results = asyncio.run(
        image_ocr.run_image_jobs(jobs, worker, max_concurrency=2, progress=False)
    )

    status_by_stem = {result.job.stem: result.status for result in results}
    assert status_by_stem == {"a": "completed", "bad": "failed", "c": "completed"}
    assert max_active == 2
    failed = next(result for result in results if result.job.stem == "bad")
    assert "model returned invalid output" in failed.error


def test_write_success_outputs_mineru_like_files(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    job = image_ocr.ImageJob(image_path, "page", tmp_path / "out" / "page" / "vlm")
    blocks = [{"type": "text", "bbox": [0, 0, 1, 1], "content": "hello"}]
    middle_json = {"pdf_info": [{"page_idx": 0, "page_size": [16, 12]}]}
    content_list = [{"type": "text", "text": "hello"}]

    image_ocr.write_success_outputs(
        job,
        blocks=blocks,
        middle_json=middle_json,
        markdown="hello",
        content_list=content_list,
        elapsed_seconds=1.25,
    )

    assert (job.parse_dir / "page.md").read_text(encoding="utf-8") == "hello"
    assert json.loads((job.parse_dir / "page_model.json").read_text(encoding="utf-8")) == blocks
    assert json.loads((job.parse_dir / "page_middle.json").read_text(encoding="utf-8")) == middle_json
    assert json.loads((job.parse_dir / "page_content_list.json").read_text(encoding="utf-8")) == content_list
    status = json.loads((job.parse_dir / "page_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["image_path"] == str(image_path)
    assert status["elapsed_seconds"] == 1.25


def test_cli_parses_http_scheduler_options(monkeypatch, tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    captured = {}

    async def fake_run_image_ocr(options):
        captured.update(options.__dict__)
        job = image_ocr.ImageJob(image_path, "page", tmp_path / "out" / "page" / "vlm")
        return [image_ocr.ImageJobResult(job=job, status="completed", elapsed_seconds=0.1)]

    monkeypatch.setattr(image_ocr, "run_image_ocr", fake_run_image_ocr)

    result = CliRunner().invoke(
        image_ocr.main,
        [
            "-p",
            str(image_path),
            "-o",
            str(tmp_path / "out"),
            "-u",
            "http://shared",
            "--layout-url",
            "http://layout",
            "--recognition-url",
            "http://recognition",
            "--resume",
            "--max-concurrency",
            "7",
            "--per-image-timeout",
            "12",
            "--http-timeout",
            "13",
            "--connect-timeout",
            "3",
            "--max-retries",
            "2",
            "--retry-backoff-factor",
            "0.25",
            "--no-progress",
            "--no-formula",
            "--no-table",
            "--no-image-analysis",
            "--dissection",
            "--stream",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["input_path"] == image_path
    assert captured["output_dir"] == tmp_path / "out"
    assert captured["server_url"] == "http://shared"
    assert captured["layout_server_url"] == "http://layout"
    assert captured["recognition_server_url"] == "http://recognition"
    assert captured["resume"] is True
    assert captured["max_concurrency"] == 7
    assert captured["per_image_timeout"] == 12
    assert captured["http_timeout"] == 13
    assert captured["connect_timeout"] == 3
    assert captured["max_retries"] == 2
    assert captured["retry_backoff_factor"] == 0.25
    assert captured["progress"] is False
    assert captured["formula_enable"] is False
    assert captured["table_enable"] is False
    assert captured["image_analysis"] is False
    assert captured["dissection_enable"] is True
    assert captured["stream"] is True


def test_cli_rejects_stream_without_dissection(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)

    result = CliRunner().invoke(
        image_ocr.main,
        [
            "-p",
            str(image_path),
            "-o",
            str(tmp_path / "out"),
            "--stream",
        ],
    )

    assert result.exit_code != 0
    assert "--stream requires --dissection" in result.output

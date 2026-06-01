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


def test_write_success_outputs_writes_origin_and_layout_images(tmp_path):
    image_path = tmp_path / "page.jpg"
    _make_image(image_path, size=(100, 80))
    job = image_ocr.ImageJob(image_path, "page", tmp_path / "out" / "page" / "vlm")
    blocks = [{"type": "text", "bbox": [0.1, 0.1, 0.6, 0.5], "content": "hello"}]
    middle_json = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [100, 80],
                "para_blocks": [
                    {"type": "text", "bbox": [10, 10, 60, 40]},
                    {"type": "table", "bbox": [65, 10, 95, 45], "blocks": [
                        {"type": "table_body", "bbox": [65, 10, 95, 45]}
                    ]},
                    {"type": "text", "bbox": [90, 70, 80, 75]},
                ],
            }
        ]
    }

    image_ocr.write_success_outputs(
        job,
        blocks=blocks,
        middle_json=middle_json,
        markdown="hello",
        content_list=[],
        elapsed_seconds=1.25,
    )

    origin_path = job.parse_dir / "page_origin.jpg"
    layout_path = job.parse_dir / "page_layout.png"
    assert origin_path.read_bytes() == image_path.read_bytes()
    assert layout_path.is_file()

    with Image.open(layout_path) as layout_image:
        assert layout_image.size == (100, 80)
        assert layout_image.getpixel((12, 12)) != (255, 255, 255)


def test_process_image_job_records_output_generation_stage(monkeypatch, tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    job = image_ocr.ImageJob(image_path, "page", tmp_path / "out" / "page" / "vlm")
    options = image_ocr.ImageOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "out",
        dissection_enable=True,
    )

    class FakeClient:
        async def aio_two_step_extract(
            self,
            image,
            semaphore=None,
            image_analysis=None,
            dissection_recorder=None,
            dissection_stream=False,
            page_idx=0,
        ):
            dissection_recorder.record_pipeline_started()
            dissection_recorder.record_stage_started("layout_detection")
            dissection_recorder.record_stage_finished("layout_detection", "completed")
            return [{"type": "text", "bbox": [0, 0, 1, 1], "content": "hello"}]

    monkeypatch.setattr(
        image_ocr,
        "build_middle_json_from_blocks",
        lambda blocks, image_size: {"pdf_info": [{"page_idx": 0, "page_size": list(image_size)}]},
    )
    monkeypatch.setattr(image_ocr, "render_outputs", lambda middle_json: ("hello", []))

    result = asyncio.run(
        image_ocr.process_image_job(
            FakeClient(),
            job,
            options,
        )
    )

    assert result.status == "completed"
    manifest = json.loads((job.parse_dir / "dissection" / "manifest.json").read_text(encoding="utf-8"))
    stages = {stage["name"]: stage for stage in manifest["pipeline"]["stages"]}
    assert manifest["pipeline"]["status"] == "completed"
    assert stages["output_generation"]["status"] == "completed"


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
            "--max-http-concurrency-per-image",
            "11",
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
    assert captured["max_http_concurrency_per_image"] == 11
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


def test_cli_defaults_http_concurrency_per_image_to_one(monkeypatch, tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    captured = {}

    async def fake_run_image_ocr(options):
        captured.update(options.__dict__)
        return []

    monkeypatch.setattr(image_ocr, "run_image_ocr", fake_run_image_ocr)

    result = CliRunner().invoke(
        image_ocr.main,
        [
            "-p",
            str(image_path),
            "-o",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["max_http_concurrency_per_image"] == 1


def test_create_client_uses_http_concurrency_per_image(monkeypatch, tmp_path):
    import mineru_vl_utils

    captured = {}

    class FakeMinerUClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(mineru_vl_utils, "MinerUClient", FakeMinerUClient)
    options = image_ocr.ImageOcrOptions(
        input_path=tmp_path / "page.png",
        output_dir=tmp_path / "out",
        max_concurrency=8,
        max_http_concurrency_per_image=16,
    )

    image_ocr.create_mineru_client(options)

    assert captured["max_concurrency"] == 16


def test_cli_rejects_invalid_http_concurrency_per_image(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)

    result = CliRunner().invoke(
        image_ocr.main,
        [
            "-p",
            str(image_path),
            "-o",
            str(tmp_path / "out"),
            "--max-http-concurrency-per-image",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "--max-http-concurrency-per-image must be at least 1" in result.output


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

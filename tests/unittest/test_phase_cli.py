from pathlib import Path
from click.testing import CliRunner
from PIL import Image
from mineru.cli import phase


def _make_image(path: Path, size=(16, 12)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 255, 255)).save(path)


def test_phase_layout_parses_first_principles_options(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)
    captured = {}

    async def fake_run(options):
        captured.update(options.__dict__)
        return []

    monkeypatch.setattr(phase, "run_document_ocr", fake_run)

    result = CliRunner().invoke(phase.main, [
        "layout",
        "--source", str(image),
        "--layout-output", str(tmp_path / "layout_artifacts"),
        "--layout-url", "http://layout",
        "--page-window-size", "2",
        "--max-model-concurrency", "4",
        "--timeout", "12",
    ])

    assert result.exit_code == 0, result.output
    assert captured["phase"].value == "layout"
    assert captured["input_path"] == image
    assert captured["output_dir"] == tmp_path / "layout_artifacts"
    assert captured["layout_output_dir"] == tmp_path / "layout_artifacts"
    assert captured["layout_server_url"] == "http://layout"
    assert captured["page_window_size"] == 2
    assert captured["max_http_concurrency_per_window"] == 4
    assert captured["per_window_timeout"] == 12


def test_phase_recognize_parses_layout_input_and_output(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)
    layout_input = tmp_path / "layout_artifacts"
    layout_input.mkdir()
    captured = {}

    async def fake_run(options):
        captured.update(options.__dict__)
        return []

    monkeypatch.setattr(phase, "run_document_ocr", fake_run)

    result = CliRunner().invoke(phase.main, [
        "recognize",
        "--source", str(image),
        "--layout-input", str(layout_input),
        "--output", str(tmp_path / "recognition_out"),
        "--recognition-url", "http://recognition",
        "--max-model-concurrency", "3",
        "--timeout", "13",
    ])

    assert result.exit_code == 0, result.output
    assert captured["phase"].value == "recognize"
    assert captured["input_path"] == image
    assert captured["layout_input_path"] == layout_input
    assert captured["output_dir"] == tmp_path / "recognition_out"
    assert captured["recognition_server_url"] == "http://recognition"
    assert captured["max_http_concurrency_per_window"] == 3
    assert captured["per_window_timeout"] == 13


def test_phase_run_parses_output_and_optional_layout_output(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)
    captured = {}

    async def fake_run(options):
        captured.update(options.__dict__)
        return []

    monkeypatch.setattr(phase, "run_document_ocr", fake_run)

    result = CliRunner().invoke(phase.main, [
        "run",
        "--source", str(image),
        "--output", str(tmp_path / "full_out"),
        "--layout-output", str(tmp_path / "layout_artifacts"),
        "--layout-url", "http://layout",
        "--recognition-url", "http://recognition",
    ])

    assert result.exit_code == 0, result.output
    assert captured["phase"].value == "full"
    assert captured["output_dir"] == tmp_path / "full_out"
    assert captured["layout_output_dir"] == tmp_path / "layout_artifacts"


def test_layout_help_excludes_recognition_options():
    result = CliRunner().invoke(phase.main, ["layout", "--help"])

    assert result.exit_code == 0
    assert "--recognition-url" not in result.output
    assert "--formula" not in result.output
    assert "--table" not in result.output
    assert "--image-analysis" not in result.output
    assert "--stream" not in result.output


def test_recognize_help_excludes_layout_options():
    result = CliRunner().invoke(phase.main, ["recognize", "--help"])

    assert result.exit_code == 0
    assert "--layout-url" not in result.output
    assert "--layout-output" not in result.output
    assert "--page-window-size" not in result.output


def test_phase_recognize_requires_layout_input(tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)

    result = CliRunner().invoke(phase.main, [
        "recognize",
        "--source", str(image),
        "--output", str(tmp_path / "out"),
        "--recognition-url", "http://recognition",
    ])

    assert result.exit_code != 0
    assert "Missing option '--layout-input'" in result.output


def test_phase_rejects_stream_without_dissection(tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)

    result = CliRunner().invoke(phase.main, [
        "run",
        "--source", str(image),
        "--output", str(tmp_path / "out"),
        "--stream",
    ])

    assert result.exit_code != 0
    assert "--stream requires --dissection" in result.output

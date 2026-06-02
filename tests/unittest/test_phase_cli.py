from pathlib import Path
from click.testing import CliRunner
from PIL import Image
from mineru.cli import phase


def _make_image(path: Path, size=(16, 12)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 255, 255)).save(path)


def test_phase_run_parses_options(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)
    captured = {}

    async def fake_run(options):
        captured.update(options.__dict__)
        return []

    monkeypatch.setattr(phase, "run_document_ocr", fake_run)

    result = CliRunner().invoke(phase.main, [
        "run", "-p", str(image), "-o", str(tmp_path / "out"),
        "--layout-url", "http://layout",
        "--recognition-url", "http://recognition",
        "--page-window-size", "2", "--max-windows", "5",
    ])

    assert result.exit_code == 0, result.output
    assert captured["phase"].value == "full"
    assert captured["layout_server_url"] == "http://layout"
    assert captured["recognition_server_url"] == "http://recognition"
    assert captured["page_window_size"] == 2


def test_phase_layout_does_not_accept_recognition_url(tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)

    result = CliRunner().invoke(phase.main, [
        "layout", "-p", str(image), "-o", str(tmp_path / "out"),
        "--recognition-url", "http://recognition",
    ])

    assert result.exit_code != 0


def test_phase_layout_parses_options(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)
    captured = {}

    async def fake_run(options):
        captured.update(options.__dict__)
        return []

    monkeypatch.setattr(phase, "run_document_ocr", fake_run)

    result = CliRunner().invoke(phase.main, [
        "layout", "-p", str(image), "-o", str(tmp_path / "out"),
        "--layout-url", "http://layout",
    ])

    assert result.exit_code == 0, result.output
    assert captured["phase"].value == "layout"
    assert captured["layout_server_url"] == "http://layout"
    assert captured["recognition_server_url"] is None


def test_phase_recognize_does_not_accept_layout_url(tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)

    result = CliRunner().invoke(phase.main, [
        "recognize", "-p", str(image), "-o", str(tmp_path / "out"),
        "--layout-url", "http://layout",
    ])

    assert result.exit_code != 0


def test_phase_recognize_parses_options(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)
    captured = {}

    async def fake_run(options):
        captured.update(options.__dict__)
        return []

    monkeypatch.setattr(phase, "run_document_ocr", fake_run)

    result = CliRunner().invoke(phase.main, [
        "recognize", "-p", str(image), "-o", str(tmp_path / "out"),
        "--recognition-url", "http://recognition",
    ])

    assert result.exit_code == 0, result.output
    assert captured["phase"].value == "recognize"
    assert captured["recognition_server_url"] == "http://recognition"
    assert captured["layout_server_url"] is None


def test_phase_rejects_stream_without_dissection(tmp_path):
    image = tmp_path / "page.png"
    _make_image(image)

    result = CliRunner().invoke(phase.main, [
        "run", "-p", str(image), "-o", str(tmp_path / "out"),
        "--stream",
    ])

    assert result.exit_code != 0
    assert "--stream requires --dissection" in result.output

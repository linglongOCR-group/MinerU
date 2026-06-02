"""Integration tests for phased pipeline: layout->recognize == full."""

import asyncio
import json
from pathlib import Path

from PIL import Image

from mineru.cli import document_ocr
from mineru_vl_utils.structs import ContentBlock, ExtractResult


def _make_image(path: Path, size=(100, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 255, 255)).save(path)


def test_layout_then_recognize_equals_full(tmp_path):
    """Two-step (layout then recognize) produces identical output to single full run."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)

    call_log = []

    class FakeClient:
        async def aio_layout_detect(self, image, priority=None, semaphore=None, scored=None):
            call_log.append("layout_detect")
            return ExtractResult([
                ContentBlock("text", [0.0, 0.0, 0.5, 1.0]),
                ContentBlock("text", [0.5, 0.0, 1.0, 1.0]),
            ])

        async def aio_recognize_from_layout(self, image, layout_blocks, **kwargs):
            call_log.append("recognize_from_layout")
            for block in layout_blocks:
                block.content = f"content-{block.type}"
            return layout_blocks

    client = FakeClient()

    # --- Run FULL phase ---
    full_job = document_ocr.DocumentJob.from_path(
        image_path, "image", "doc",
        tmp_path / "full" / "doc" / "vlm", 0, 1, 0, 0,
    )
    full_window = document_ocr.WindowJob.from_document(full_job, 0, 0, 0, 0)

    full_options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "full",
        phase=document_ocr.OcrPhase.FULL,
        progress=False,
    )
    asyncio.run(document_ocr.process_full_window(client, full_window, full_options))

    full_recognition_cache = document_ocr.read_valid_window_cache(full_window)
    assert full_recognition_cache is not None
    full_blocks = full_recognition_cache["blocks_by_page"]

    # --- Run LAYOUT phase ---
    call_log.clear()
    layout_job = document_ocr.DocumentJob.from_path(
        image_path, "image", "doc",
        tmp_path / "phased" / "doc" / "vlm", 0, 1, 0, 0,
    )
    layout_window = document_ocr.WindowJob.from_document(layout_job, 0, 0, 0, 0)

    layout_options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "phased",
        phase=document_ocr.OcrPhase.LAYOUT,
        progress=False,
    )
    asyncio.run(document_ocr.process_layout_window(client, layout_window, layout_options))

    layout_cache = document_ocr.read_valid_layout_window_cache(layout_window)
    assert layout_cache is not None
    assert "layout_detect" in call_log

    # --- Run RECOGNIZE phase ---
    call_log.clear()
    recognize_options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "phased",
        phase=document_ocr.OcrPhase.RECOGNIZE,
        progress=False,
    )
    asyncio.run(document_ocr.process_recognition_window(client, layout_window, recognize_options))

    recognize_cache = document_ocr.read_valid_window_cache(layout_window)
    assert recognize_cache is not None
    assert "recognize_from_layout" in call_log

    # Both should produce the same blocks
    assert full_blocks == recognize_cache["blocks_by_page"]


def test_stale_layout_rejected(tmp_path):
    """Modified source file causes layout cache to be rejected."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)

    job = document_ocr.DocumentJob.from_path(
        image_path, "image", "doc",
        tmp_path / "out" / "doc" / "vlm", 0, 1, 0, 0,
    )
    window = document_ocr.WindowJob.from_document(job, 0, 0, 0, 0)

    # Write layout cache
    document_ocr.write_layout_window_cache(
        window,
        blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1]}]],
        page_sizes=[[100, 80]],
        elapsed_seconds=0.1,
    )

    # Modify the source file (changes size and mtime)
    image_path.write_bytes(b"modified content")

    # Create a new job from the modified file — its size/mtime differ
    new_job = document_ocr.DocumentJob.from_path(
        image_path, "image", "doc",
        tmp_path / "out" / "doc" / "vlm", 0, 1, 0, 0,
    )
    new_window = document_ocr.WindowJob.from_document(new_job, 0, 0, 0, 0)

    # Stale cache should be rejected
    cached = document_ocr.read_valid_layout_window_cache(new_window)
    assert cached is None, "Stale layout cache should be rejected"


def test_old_document_ocr_cli_still_works(monkeypatch, tmp_path):
    """mineru-ocr-documents still functions with deprecation notice."""
    from click.testing import CliRunner

    image_path = tmp_path / "page.png"
    _make_image(image_path)
    captured = {}

    async def fake_run_document_ocr(options):
        captured.update(options.__dict__)
        job = document_ocr.DocumentJob.from_path(
            image_path, "image", "page",
            tmp_path / "out" / "page" / "vlm", 0, 1, 0, 0,
        )
        return [document_ocr.DocumentJobResult(job=job, status="completed", elapsed_seconds=0.1)]

    monkeypatch.setattr(document_ocr, "run_document_ocr", fake_run_document_ocr)

    result = CliRunner().invoke(
        document_ocr.main,
        ["-p", str(image_path), "-o", str(tmp_path / "out")],
    )

    assert result.exit_code == 0, result.output
    assert "deprecated" in result.output.lower()
    assert captured["phase"] == document_ocr.OcrPhase.FULL

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

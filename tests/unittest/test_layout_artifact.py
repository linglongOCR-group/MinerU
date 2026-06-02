# Copyright (c) Opendatalab. All rights reserved.
import json
from pathlib import Path
from PIL import Image

from mineru.cli import layout_artifact


def _make_image(path: Path, size=(16, 12)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 255, 255)).save(path)


def test_layout_artifact_round_trip(tmp_path):
    """Write then read preserves all fields."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    stat = image_path.stat()

    artifact = layout_artifact.LayoutArtifact(
        schema="mineru.vlm.layout.v1",
        stage="layout",
        backend="vlm",
        source=layout_artifact.LayoutSource(
            path=str(image_path),
            type="image",
            size=stat.st_size,
            mtime=stat.st_mtime,
            page_count=1,
            start_page_id=0,
            end_page_id=0,
        ),
        layout=layout_artifact.LayoutMeta(
            model="test-model",
            layout_image_size=(1036, 1036),
            coordinate="normalized_xyxy",
        ),
        pages=[
            layout_artifact.LayoutPage(
                page_idx=0,
                page_size=[16, 12],
                blocks=[
                    layout_artifact.LayoutBlock(
                        id="p0000-b000000",
                        index=0,
                        type="text",
                        bbox=[0.1, 0.2, 0.8, 0.5],
                        angle=0,
                        merge_prev=False,
                    ),
                ],
            ),
        ],
    )

    out_path = tmp_path / "doc_layout.json"
    layout_artifact.write_layout_artifact(artifact, out_path)
    loaded = layout_artifact.read_layout_artifact(out_path)

    assert loaded.schema == "mineru.vlm.layout.v1"
    assert loaded.source.path == str(image_path)
    assert loaded.source.type == "image"
    assert loaded.source.size == stat.st_size
    assert loaded.layout.model == "test-model"
    assert len(loaded.pages) == 1
    assert loaded.pages[0].blocks[0].type == "text"
    assert loaded.pages[0].blocks[0].bbox == [0.1, 0.2, 0.8, 0.5]


def test_layout_artifact_validates_matching_source(tmp_path):
    """validate_against_source succeeds when source matches."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    stat = image_path.stat()

    artifact = layout_artifact.LayoutArtifact(
        schema="mineru.vlm.layout.v1",
        stage="layout",
        backend="vlm",
        source=layout_artifact.LayoutSource(
            path=str(image_path),
            type="image",
            size=stat.st_size,
            mtime=stat.st_mtime,
            page_count=1,
            start_page_id=0,
            end_page_id=0,
        ),
        layout=layout_artifact.LayoutMeta(
            model="test-model",
            layout_image_size=(1036, 1036),
        ),
        pages=[
            layout_artifact.LayoutPage(
                page_idx=0,
                page_size=[16, 12],
                blocks=[],
            ),
        ],
    )

    # Should not raise
    layout_artifact.validate_against_source(artifact, image_path, start_page_id=0, end_page_id=0)


def test_layout_artifact_rejects_wrong_size(tmp_path):
    """validate_against_source raises ValueError when file size differs."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    stat = image_path.stat()

    artifact = layout_artifact.LayoutArtifact(
        schema="mineru.vlm.layout.v1",
        stage="layout",
        backend="vlm",
        source=layout_artifact.LayoutSource(
            path=str(image_path),
            type="image",
            size=stat.st_size + 999,  # wrong size
            mtime=stat.st_mtime,
            page_count=1,
            start_page_id=0,
            end_page_id=0,
        ),
        layout=layout_artifact.LayoutMeta(
            model="test-model",
            layout_image_size=(1036, 1036),
        ),
        pages=[],
    )

    try:
        layout_artifact.validate_against_source(artifact, image_path, start_page_id=0, end_page_id=0)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "size" in str(exc).lower()


def test_layout_artifact_rejects_wrong_mtime(tmp_path):
    """validate_against_source raises ValueError when mtime differs."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    stat = image_path.stat()

    artifact = layout_artifact.LayoutArtifact(
        schema="mineru.vlm.layout.v1",
        stage="layout",
        backend="vlm",
        source=layout_artifact.LayoutSource(
            path=str(image_path),
            type="image",
            size=stat.st_size,
            mtime=stat.st_mtime + 9999.0,  # wrong mtime
            page_count=1,
            start_page_id=0,
            end_page_id=0,
        ),
        layout=layout_artifact.LayoutMeta(
            model="test-model",
            layout_image_size=(1036, 1036),
        ),
        pages=[],
    )

    try:
        layout_artifact.validate_against_source(artifact, image_path, start_page_id=0, end_page_id=0)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "mtime" in str(exc).lower()


def test_layout_artifact_rejects_unsupported_schema(tmp_path):
    """read_layout_artifact raises ValueError for unsupported schema."""
    data = {
        "schema": "mineru.vlm.layout.v999",
        "stage": "layout",
        "backend": "vlm",
        "source": {},
        "layout": {},
        "pages": [],
    }
    path = tmp_path / "bad_layout.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    try:
        layout_artifact.read_layout_artifact(path)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "schema" in str(exc).lower()


def test_layout_artifact_rejects_wrong_page_range(tmp_path):
    """validate_against_source raises ValueError when page range differs."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    stat = image_path.stat()

    artifact = layout_artifact.LayoutArtifact(
        schema="mineru.vlm.layout.v1",
        stage="layout",
        backend="vlm",
        source=layout_artifact.LayoutSource(
            path=str(image_path),
            type="image",
            size=stat.st_size,
            mtime=stat.st_mtime,
            page_count=10,
            start_page_id=0,
            end_page_id=9,
        ),
        layout=layout_artifact.LayoutMeta(
            model="test-model",
            layout_image_size=(1036, 1036),
        ),
        pages=[],
    )

    try:
        layout_artifact.validate_against_source(artifact, image_path, start_page_id=2, end_page_id=5)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "page" in str(exc).lower()

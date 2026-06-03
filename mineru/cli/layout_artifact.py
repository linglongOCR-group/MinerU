# Copyright (c) Opendatalab. All rights reserved.
"""Layout artifact schema — the stable contract between layout and recognize phases.

Provides dataclasses for the layout artifact format, JSON serialization /
deserialization helpers, and source-file validation.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

SUPPORTED_SCHEMAS = {"mineru.vlm.layout.v1"}
CURRENT_SCHEMA = "mineru.vlm.layout.v1"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutSource:
    path: str
    type: str
    size: int
    mtime: float
    page_count: int
    start_page_id: int
    end_page_id: int


@dataclass(frozen=True)
class LayoutMeta:
    model: str
    layout_image_size: tuple
    coordinate: str = "normalized_xyxy"


@dataclass(frozen=True)
class LayoutBlock:
    id: str
    index: int
    type: str
    bbox: list
    angle: int = 0
    merge_prev: bool = False


@dataclass(frozen=True)
class LayoutPage:
    page_idx: int
    page_size: list
    blocks: List[LayoutBlock] = field(default_factory=list)


@dataclass(frozen=True)
class LayoutArtifact:
    schema: str
    stage: str
    backend: str
    source: LayoutSource
    layout: LayoutMeta
    pages: List[LayoutPage] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dict serialization / deserialization
# ---------------------------------------------------------------------------


def _source_to_dict(src: LayoutSource) -> dict:
    return {
        "path": src.path,
        "type": src.type,
        "size": src.size,
        "mtime": src.mtime,
        "page_count": src.page_count,
        "start_page_id": src.start_page_id,
        "end_page_id": src.end_page_id,
    }


def _source_from_dict(d: dict) -> LayoutSource:
    return LayoutSource(
        path=d["path"],
        type=d["type"],
        size=d["size"],
        mtime=d["mtime"],
        page_count=d["page_count"],
        start_page_id=d["start_page_id"],
        end_page_id=d["end_page_id"],
    )


def _meta_to_dict(meta: LayoutMeta) -> dict:
    return {
        "model": meta.model,
        "layout_image_size": list(meta.layout_image_size),
        "coordinate": meta.coordinate,
    }


def _meta_from_dict(d: dict) -> LayoutMeta:
    return LayoutMeta(
        model=d["model"],
        layout_image_size=tuple(d["layout_image_size"]),
        coordinate=d.get("coordinate", "normalized_xyxy"),
    )


def _block_to_dict(block: LayoutBlock) -> dict:
    return {
        "id": block.id,
        "index": block.index,
        "type": block.type,
        "bbox": block.bbox,
        "angle": block.angle,
        "merge_prev": block.merge_prev,
    }


def _block_from_dict(d: dict) -> LayoutBlock:
    return LayoutBlock(
        id=d["id"],
        index=d["index"],
        type=d["type"],
        bbox=d["bbox"],
        angle=d.get("angle", 0),
        merge_prev=d.get("merge_prev", False),
    )


def _page_to_dict(page: LayoutPage) -> dict:
    return {
        "page_idx": page.page_idx,
        "page_size": page.page_size,
        "blocks": [_block_to_dict(b) for b in page.blocks],
    }


def _page_from_dict(d: dict) -> LayoutPage:
    return LayoutPage(
        page_idx=d["page_idx"],
        page_size=d["page_size"],
        blocks=[_block_from_dict(b) for b in d.get("blocks", [])],
    )


def _artifact_to_dict(artifact: LayoutArtifact) -> dict:
    return {
        "schema": artifact.schema,
        "stage": artifact.stage,
        "backend": artifact.backend,
        "source": _source_to_dict(artifact.source),
        "layout": _meta_to_dict(artifact.layout),
        "pages": [_page_to_dict(p) for p in artifact.pages],
    }


def _artifact_from_dict(d: dict) -> LayoutArtifact:
    return LayoutArtifact(
        schema=d["schema"],
        stage=d["stage"],
        backend=d["backend"],
        source=_source_from_dict(d["source"]),
        layout=_meta_from_dict(d["layout"]),
        pages=[_page_from_dict(p) for p in d.get("pages", [])],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_layout_artifact(artifact: LayoutArtifact, path: Path | str) -> None:
    """Write a layout artifact to *path* as JSON using an atomic temp-file write."""
    path = Path(path)
    data = _artifact_to_dict(artifact)
    encoded = json.dumps(data, ensure_ascii=False, indent=2)

    # Atomic write: write to a temp file in the same directory, then rename.
    fd, tmp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=".layout_artifact_",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(encoded)
        # os.replace is atomic on POSIX
        os.replace(tmp_name, str(path))
    except BaseException:
        # Clean up the temp file on failure
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_layout_artifact(path: Path | str) -> LayoutArtifact:
    """Read a layout artifact from *path*, validating the schema version."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    schema = raw.get("schema", "")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(
            f"Unsupported schema '{schema}'. "
            f"Supported schemas: {sorted(SUPPORTED_SCHEMAS)}"
        )

    return _artifact_from_dict(raw)


def validate_against_source(
    artifact: LayoutArtifact,
    source_path: Path | str,
    *,
    start_page_id: int,
    end_page_id: int,
    page_count: int | None = None,
    page_sizes: list[list[int]] | None = None,
) -> None:
    """Validate that *artifact* still matches *source_path* on disk.

    Checks source identity, file metadata, page range, and optional page
    metadata. Raises
    ``ValueError`` on any mismatch.
    """
    source_path = Path(source_path)
    stat = source_path.stat()
    expected_path = str(source_path)

    if artifact.source.path != expected_path:
        raise ValueError(
            f"Source path mismatch: artifact={artifact.source.path}, "
            f"actual={expected_path}"
        )

    if artifact.source.size != stat.st_size:
        raise ValueError(
            f"Source file size mismatch: artifact={artifact.source.size}, "
            f"actual={stat.st_size}"
        )

    if artifact.source.mtime != stat.st_mtime:
        raise ValueError(
            f"Source file mtime mismatch: artifact={artifact.source.mtime}, "
            f"actual={stat.st_mtime}"
        )

    if artifact.source.start_page_id != start_page_id or artifact.source.end_page_id != end_page_id:
        raise ValueError(
            f"Source page range mismatch: artifact=[{artifact.source.start_page_id}, "
            f"{artifact.source.end_page_id}], requested=[{start_page_id}, {end_page_id}]"
        )

    if page_count is not None and artifact.source.page_count != page_count:
        raise ValueError(
            f"Source page count mismatch: artifact={artifact.source.page_count}, "
            f"actual={page_count}"
        )

    if page_sizes is not None:
        pages_by_idx = {page.page_idx: page for page in artifact.pages}
        for offset, page_idx in enumerate(range(start_page_id, end_page_id + 1)):
            if offset >= len(page_sizes):
                raise ValueError(f"Source page size missing for requested page {page_idx}")
            page = pages_by_idx.get(page_idx)
            if page is None:
                raise ValueError(f"Artifact page size missing for page {page_idx}")
            expected_page_size = list(page_sizes[offset])
            if list(page.page_size) != expected_page_size:
                raise ValueError(
                    f"Source page size mismatch for page {page_idx}: "
                    f"artifact={page.page_size}, actual={expected_page_size}"
                )

# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

import click
import pypdfium2 as pdfium
from PIL import Image

from mineru.cli.common import pdf_suffixes, uniquify_task_stems
from mineru.cli.image_ocr import IMAGE_SUFFIXES
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import ImageType
from mineru.utils.pdf_image_tools import (
    aio_load_images_from_pdf_bytes_range,
    load_images_from_pdf_doc,
)
from mineru.utils.pdf_page_id import get_end_page_id
from mineru.utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
)


PDF_SUFFIXES = {f".{suffix}" for suffix in pdf_suffixes}
DOCUMENT_SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES


class OcrPhase(Enum):
    FULL = "full"
    LAYOUT = "layout"
    RECOGNIZE = "recognize"


@dataclass(frozen=True)
class DocumentJob:
    path: Path
    document_type: str
    stem: str
    parse_dir: Path
    order: int
    page_count: int
    start_page_id: int
    end_page_id: int
    source_size: int
    source_mtime: float

    @classmethod
    def from_path(
        cls,
        path: Path,
        document_type: str,
        stem: str,
        parse_dir: Path,
        order: int,
        page_count: int,
        start_page_id: int,
        end_page_id: int,
    ) -> "DocumentJob":
        stat = path.stat()
        return cls(
            path=path,
            document_type=document_type,
            stem=stem,
            parse_dir=parse_dir,
            order=order,
            page_count=page_count,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            source_size=stat.st_size,
            source_mtime=stat.st_mtime,
        )

    @property
    def selected_pages(self) -> int:
        return self.end_page_id - self.start_page_id + 1


@dataclass(frozen=True)
class WindowJob:
    document: DocumentJob
    document_index: int
    document_stem: str
    window_index: int
    start_page_id: int
    end_page_id: int

    @classmethod
    def from_document(
        cls,
        document: DocumentJob,
        document_index: int,
        window_index: int,
        start_page_id: int,
        end_page_id: int,
    ) -> "WindowJob":
        return cls(
            document=document,
            document_index=document_index,
            document_stem=document.stem,
            window_index=window_index,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )

    @property
    def page_count(self) -> int:
        return self.end_page_id - self.start_page_id + 1


@dataclass(frozen=True, kw_only=True)
class DocumentOcrOptions:
    input_path: Path
    output_dir: Path
    phase: OcrPhase = OcrPhase.FULL
    server_url: str | None = None
    layout_server_url: str | None = None
    recognition_server_url: str | None = None
    layout_input_path: Path | None = None
    layout_output_dir: Path | None = None
    resume: bool = False
    formula_enable: bool = True
    table_enable: bool = True
    image_analysis: bool = True
    page_window_size: int = 4
    max_windows: int = 16
    max_http_concurrency_per_window: int = 1
    per_window_timeout: float = 600.0
    http_timeout: int = 600
    connect_timeout: int = 10
    max_retries: int = 3
    retry_backoff_factor: float = 0.5
    start_page_id: int = 0
    end_page_id: int | None = None
    progress: bool = True
    dissection_enable: bool = False
    stream: bool = False


@dataclass(frozen=True)
class DocumentJobResult:
    job: DocumentJob
    status: str
    elapsed_seconds: float
    error: str | None = None


def _json_default(value: Any):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def document_status_path(job: DocumentJob) -> Path:
    return job.parse_dir / f"{job.stem}_status.json"


def window_cache_path(window: WindowJob) -> Path:
    return window.document.parse_dir / ".windows" / f"window_{window.window_index:05d}.json"


def _document_fingerprint(job: DocumentJob) -> dict[str, Any]:
    return {
        "source_path": str(job.path),
        "source_size": job.source_size,
        "source_mtime": job.source_mtime,
        "document_type": job.document_type,
        "page_count": job.page_count,
        "start_page_id": job.start_page_id,
        "end_page_id": job.end_page_id,
    }


def _window_metadata(window: WindowJob, page_sizes: list[list[int]] | None = None) -> dict[str, Any]:
    metadata = {
        **_document_fingerprint(window.document),
        "document_stem": window.document_stem,
        "window_index": window.window_index,
        "window_start_page_id": window.start_page_id,
        "window_end_page_id": window.end_page_id,
    }
    if page_sizes is not None:
        metadata["page_sizes"] = page_sizes
    return metadata


def _metadata_matches(window: WindowJob, metadata: dict[str, Any]) -> bool:
    expected = _window_metadata(window)
    return all(metadata.get(key) == value for key, value in expected.items())


def _is_completed_status(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return _read_json(path).get("status") == "completed"
    except Exception:
        return False


def probe_pdf_page_count(path: Path) -> int:
    pdf_doc = open_pdfium_document(pdfium.PdfDocument, str(path))
    try:
        return get_pdfium_document_page_count(pdf_doc)
    finally:
        close_pdfium_document(pdf_doc)


def _discover_documents(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in DOCUMENT_SUFFIXES:
            raise click.ClickException(f"Unsupported document suffix: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise click.ClickException(f"Input path does not exist: {input_path}")
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
    )


def _resolve_pdf_range(
    path: Path,
    page_count: int,
    start_page_id: int,
    end_page_id: int | None,
) -> tuple[int, int]:
    if page_count <= 0:
        raise click.ClickException(f"PDF has no pages: {path}")
    effective_end_page_id = get_end_page_id(end_page_id, page_count)
    if start_page_id < 0:
        raise click.ClickException("--start must be greater than or equal to 0")
    if start_page_id > effective_end_page_id:
        raise click.ClickException(
            f"Requested page range is empty for PDF {path}: "
            f"start={start_page_id}, end={end_page_id}"
        )
    return start_page_id, effective_end_page_id


def collect_document_jobs(
    input_path: Path,
    output_dir: Path,
    *,
    start_page_id: int,
    end_page_id: int | None,
    resume: bool = False,
) -> list[DocumentJob]:
    paths = _discover_documents(input_path)
    if not paths:
        raise click.ClickException(f"No supported documents found under {input_path}")

    stems, _renamed = uniquify_task_stems([path.stem for path in paths])
    jobs: list[DocumentJob] = []
    for order, (path, stem) in enumerate(zip(paths, stems)):
        suffix = path.suffix.lower()
        if suffix in PDF_SUFFIXES:
            page_count = probe_pdf_page_count(path)
            effective_start, effective_end = _resolve_pdf_range(
                path,
                page_count,
                start_page_id,
                end_page_id,
            )
            document_type = "pdf"
        else:
            page_count = 1
            effective_start = 0
            effective_end = 0
            document_type = "image"
        job = DocumentJob.from_path(
            path=path,
            document_type=document_type,
            stem=stem,
            parse_dir=output_dir / stem / "vlm",
            order=order,
            page_count=page_count,
            start_page_id=effective_start,
            end_page_id=effective_end,
        )
        if resume and _is_completed_status(document_status_path(job)):
            continue
        jobs.append(job)
    return jobs


def build_window_jobs(jobs: list[DocumentJob], page_window_size: int) -> list[WindowJob]:
    if page_window_size < 1:
        raise click.ClickException("--page-window-size must be at least 1")
    windows: list[WindowJob] = []
    for document_index, job in enumerate(jobs):
        if job.document_type == "image":
            windows.append(WindowJob.from_document(job, document_index, 0, 0, 0))
            continue
        window_index = 0
        page_start = job.start_page_id
        while page_start <= job.end_page_id:
            page_end = min(job.end_page_id, page_start + page_window_size - 1)
            windows.append(
                WindowJob.from_document(
                    job,
                    document_index,
                    window_index,
                    page_start,
                    page_end,
                )
            )
            page_start = page_end + 1
            window_index += 1
    return windows


def interleave_window_jobs(windows: list[WindowJob]) -> list[WindowJob]:
    grouped: dict[int, list[WindowJob]] = {}
    for window in windows:
        grouped.setdefault(window.document_index, []).append(window)
    for group in grouped.values():
        group.sort(key=lambda item: item.window_index)

    result: list[WindowJob] = []
    while grouped:
        for document_index in sorted(list(grouped)):
            group = grouped[document_index]
            result.append(group.pop(0))
            if not group:
                del grouped[document_index]
    return result


def write_window_cache(
    window: WindowJob,
    blocks_by_page: list[list[dict[str, Any]]],
    page_sizes: list[list[int]],
    elapsed_seconds: float,
) -> None:
    _write_json(
        window_cache_path(window),
        {
            "metadata": _window_metadata(window, page_sizes=page_sizes),
            "blocks_by_page": blocks_by_page,
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
    )


def read_valid_window_cache(window: WindowJob) -> dict[str, Any] | None:
    path = window_cache_path(window)
    if not path.exists():
        return None
    try:
        cache = _read_json(path)
    except Exception:
        return None
    metadata = cache.get("metadata")
    if not isinstance(metadata, dict) or not _metadata_matches(window, metadata):
        return None
    return cache


def layout_cache_path(window: WindowJob) -> Path:
    return window.document.parse_dir / ".cache" / "layout" / f"window_{window.window_index:05d}.json"


def write_layout_window_cache(
    window: WindowJob,
    blocks_by_page: list[list[dict[str, Any]]],
    page_sizes: list[list[int]],
    elapsed_seconds: float,
) -> None:
    _write_json(
        layout_cache_path(window),
        {
            "metadata": _window_metadata(window, page_sizes=page_sizes),
            "blocks_by_page": blocks_by_page,
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
    )


def read_valid_layout_window_cache(window: WindowJob) -> dict[str, Any] | None:
    path = layout_cache_path(window)
    if not path.exists():
        return None
    try:
        cache = _read_json(path)
    except Exception:
        return None
    metadata = cache.get("metadata")
    if not isinstance(metadata, dict) or not _metadata_matches(window, metadata):
        return None
    return cache


def document_layout_artifact_path(job: DocumentJob, layout_root: Path | None = None) -> Path:
    if layout_root is None:
        return job.parse_dir / f"{job.stem}_layout.json"
    return layout_root / job.stem / "vlm" / f"{job.stem}_layout.json"


def _is_direct_layout_artifact(path: Path) -> bool:
    return path.is_file() or path.suffix.lower() == ".json"


def resolve_layout_artifact_path(job: DocumentJob, options: DocumentOcrOptions) -> Path:
    if options.layout_input_path is not None:
        layout_input_path = Path(options.layout_input_path)
        if _is_direct_layout_artifact(layout_input_path):
            return layout_input_path
        return document_layout_artifact_path(job, layout_input_path)
    return document_layout_artifact_path(job)


def read_layout_window_cache_from_artifact(
    window: WindowJob,
    options: DocumentOcrOptions | None = None,
) -> dict[str, Any] | None:
    path = (
        resolve_layout_artifact_path(window.document, options)
        if options is not None
        else document_layout_artifact_path(window.document)
    )
    if not path.exists():
        return None

    from mineru.cli.layout_artifact import read_layout_artifact, validate_against_source

    try:
        artifact = read_layout_artifact(path)
        validate_against_source(
            artifact,
            window.document.path,
            start_page_id=window.document.start_page_id,
            end_page_id=window.document.end_page_id,
        )
    except Exception as exc:
        raise RuntimeError(f"Invalid layout artifact for {window.document_stem}: {exc}") from exc

    pages_by_idx = {page.page_idx: page for page in artifact.pages}
    blocks_by_page: list[list[dict[str, Any]]] = []
    page_sizes: list[list[int]] = []
    for page_idx in range(window.start_page_id, window.end_page_id + 1):
        page = pages_by_idx.get(page_idx)
        if page is None:
            raise RuntimeError(
                f"Layout artifact for {window.document_stem} is missing page {page_idx}"
            )
        page_sizes.append(list(page.page_size))
        blocks_by_page.append([
            {
                "type": block.type,
                "bbox": block.bbox,
                "angle": block.angle,
                "merge_prev": block.merge_prev,
            }
            for block in sorted(page.blocks, key=lambda block: block.index)
        ])

    return {
        "metadata": _window_metadata(window, page_sizes=page_sizes),
        "blocks_by_page": blocks_by_page,
        "elapsed_seconds": 0.0,
    }


def read_valid_layout_window_cache_or_artifact(
    window: WindowJob,
    *,
    options: DocumentOcrOptions | None = None,
    rehydrate: bool = False,
) -> dict[str, Any] | None:
    cache = read_valid_layout_window_cache(window)
    if cache is not None:
        return cache

    cache = read_layout_window_cache_from_artifact(window, options)
    if cache is None:
        return None

    if rehydrate:
        write_layout_window_cache(
            window,
            blocks_by_page=cache["blocks_by_page"],
            page_sizes=cache["metadata"].get("page_sizes", []),
            elapsed_seconds=cache.get("elapsed_seconds", 0.0),
        )
    return cache


def write_document_status(
    job: DocumentJob,
    *,
    status: str,
    elapsed_seconds: float,
    completed_windows: int,
    total_windows: int,
    error: str | None = None,
) -> None:
    payload = {
        "status": status,
        "document_type": job.document_type,
        "source_path": str(job.path),
        "page_count": job.page_count,
        "start_page_id": job.start_page_id,
        "end_page_id": job.end_page_id,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "completed_windows": completed_windows,
        "total_windows": total_windows,
    }
    if error is not None:
        payload["error"] = error
    _write_json(document_status_path(job), payload)


def _collect_cached_pages(windows: Iterable[WindowJob]) -> tuple[list[list[dict[str, Any]]], list[list[int]]]:
    model_pages: list[list[dict[str, Any]]] = []
    page_sizes: list[list[int]] = []
    for window in sorted(windows, key=lambda item: item.window_index):
        cache = read_valid_window_cache(window)
        if cache is None:
            raise RuntimeError(f"Missing valid cache for {window.document_stem} window {window.window_index}")
        model_pages.extend(cache["blocks_by_page"])
        page_sizes.extend(cache["metadata"].get("page_sizes", []))
    return model_pages, page_sizes


def _close_images(images_list: list[dict[str, Any]]) -> None:
    for image_dict in images_list or []:
        image = image_dict.get("img_pil")
        if image is not None:
            try:
                image.close()
            except Exception:
                pass


def build_middle_json_for_document(
    document: DocumentJob,
    model_pages: list[list[dict[str, Any]]],
    page_sizes: list[list[int]],
) -> dict[str, Any]:
    if document.document_type == "image":
        from mineru.cli.image_ocr import build_middle_json_from_blocks

        return build_middle_json_from_blocks(model_pages[0], tuple(page_sizes[0]))

    from mineru.backend.vlm.model_output_to_middle_json import (
        append_page_blocks_to_middle_json,
        finalize_middle_json,
        init_middle_json,
    )

    pdf_bytes = document.path.read_bytes()
    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    images_list: list[dict[str, Any]] = []
    try:
        images_list = load_images_from_pdf_doc(
            pdf_doc,
            start_page_id=document.start_page_id,
            end_page_id=document.end_page_id,
            image_type=ImageType.PIL,
            pdf_bytes=pdf_bytes,
        )
        image_writer = FileBasedDataWriter(str(document.parse_dir / "images"))
        middle_json = init_middle_json()
        append_page_blocks_to_middle_json(
            middle_json,
            model_pages,
            images_list,
            pdf_doc,
            image_writer,
            page_start_index=document.start_page_id,
        )
        finalize_middle_json(middle_json["pdf_info"])
        return middle_json
    finally:
        _close_images(images_list)
        close_pdfium_document(pdf_doc)


def render_outputs(middle_json: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make
    from mineru.utils.enum_class import MakeMode

    pdf_info = middle_json["pdf_info"]
    markdown = union_make(pdf_info, MakeMode.MM_MD, "images") or ""
    content_list = union_make(pdf_info, MakeMode.CONTENT_LIST, "images") or []
    return markdown, content_list


def assemble_document_outputs(
    job: DocumentJob,
    windows: list[WindowJob],
    *,
    elapsed_seconds: float,
) -> DocumentJobResult:
    model_pages, page_sizes = _collect_cached_pages(windows)
    middle_json = build_middle_json_for_document(job, model_pages, page_sizes)
    markdown, content_list = render_outputs(middle_json)
    job.parse_dir.mkdir(parents=True, exist_ok=True)
    (job.parse_dir / f"{job.stem}.md").write_text(markdown, encoding="utf-8")
    _write_json(job.parse_dir / f"{job.stem}_model.json", model_pages)
    _write_json(job.parse_dir / f"{job.stem}_middle.json", middle_json)
    _write_json(job.parse_dir / f"{job.stem}_content_list.json", content_list)
    write_document_status(
        job,
        status="completed",
        elapsed_seconds=elapsed_seconds,
        completed_windows=len(windows),
        total_windows=len(windows),
    )
    return DocumentJobResult(job=job, status="completed", elapsed_seconds=round(elapsed_seconds, 3))


def create_mineru_client(options: DocumentOcrOptions):
    from mineru_vl_utils import MinerUClient

    return MinerUClient(
        backend="http-client",
        server_url=options.server_url,
        layout_server_url=options.layout_server_url,
        recognition_server_url=options.recognition_server_url,
        http_timeout=options.http_timeout,
        connect_timeout=options.connect_timeout,
        max_retries=options.max_retries,
        retry_backoff_factor=options.retry_backoff_factor,
        max_concurrency=options.max_http_concurrency_per_window,
        image_analysis=options.image_analysis,
        enable_table_formula_eq_wrap=True,
    )


async def process_window_job(client: Any, window: WindowJob, options: DocumentOcrOptions) -> None:
    start = time.monotonic()
    recorder = None
    try:
        if options.dissection_enable:
            from mineru_vl_utils.dissection import DissectionRecorder

            recorder = DissectionRecorder(
                window.document.parse_dir / "dissection" / f"window_{window.window_index:05d}",
                document_stem=window.document_stem,
            )
        if window.document.document_type == "image":
            with Image.open(window.document.path) as src_image:
                src_image.load()
                image = src_image.convert("RGB")
            try:
                blocks = await client.aio_two_step_extract(
                    image,
                    image_analysis=options.image_analysis,
                    dissection_recorder=recorder,
                    dissection_stream=options.stream,
                    page_idx=0,
                )
                write_window_cache(
                    window,
                    blocks_by_page=[[dict(block) for block in blocks]],
                    page_sizes=[list(image.size)],
                    elapsed_seconds=time.monotonic() - start,
                )
            finally:
                image.close()
            return

        pdf_bytes = window.document.path.read_bytes()
        images_list = await aio_load_images_from_pdf_bytes_range(
            pdf_bytes,
            start_page_id=window.start_page_id,
            end_page_id=window.end_page_id,
            image_type=ImageType.PIL,
        )
        try:
            images = [image_dict["img_pil"] for image_dict in images_list]
            blocks_by_page = await client.aio_batch_two_step_extract(
                images=images,
                image_analysis=options.image_analysis,
                dissection_recorder=recorder,
                dissection_stream=options.stream,
                page_start_index=window.start_page_id,
            )
            write_window_cache(
                window,
                blocks_by_page=[[dict(block) for block in page_blocks] for page_blocks in blocks_by_page],
                page_sizes=[list(image.size) for image in images],
                elapsed_seconds=time.monotonic() - start,
            )
        finally:
            _close_images(images_list)
    finally:
        if recorder is not None:
            recorder.finalize()


def write_document_layout_artifact(
    job: DocumentJob,
    windows: list[WindowJob],
) -> None:
    """Collect layout window caches and write a document-level layout artifact."""
    from mineru.cli.layout_artifact import (
        LayoutArtifact, LayoutBlock, LayoutMeta, LayoutPage, LayoutSource,
        write_layout_artifact,
    )

    pages = []
    for window in sorted(windows, key=lambda w: w.window_index):
        cache = read_valid_layout_window_cache(window)
        if cache is None:
            raise RuntimeError(
                f"Missing valid layout cache for {job.stem} window {window.window_index}"
            )
        page_sizes = cache["metadata"].get("page_sizes", [])
        for page_idx_in_window, (page_blocks, page_size) in enumerate(
            zip(cache["blocks_by_page"], page_sizes)
        ):
            global_page_idx = window.start_page_id + page_idx_in_window
            blocks = [
                LayoutBlock(
                    id=f"p{global_page_idx:04d}-b{block_idx:06d}",
                    index=block_idx,
                    type=block.get("type", "text"),
                    bbox=block.get("bbox", [0, 0, 1, 1]),
                    angle=block.get("angle", 0),
                    merge_prev=block.get("merge_prev", False),
                )
                for block_idx, block in enumerate(page_blocks)
            ]
            pages.append(LayoutPage(
                page_idx=global_page_idx,
                page_size=page_size,
                blocks=blocks,
            ))

    stat = job.path.stat()
    artifact = LayoutArtifact(
        schema="mineru.vlm.layout.v1",
        stage="layout",
        backend="vlm",
        source=LayoutSource(
            path=str(job.path),
            type=job.document_type,
            size=stat.st_size,
            mtime=stat.st_mtime,
            page_count=job.page_count,
            start_page_id=job.start_page_id,
            end_page_id=job.end_page_id,
        ),
        layout=LayoutMeta(
            model="vlm",
            layout_image_size=(1036, 1036),
        ),
        pages=pages,
    )
    write_layout_artifact(artifact, document_layout_artifact_path(job))


async def process_layout_window(
    client: Any,
    window: WindowJob,
    options: DocumentOcrOptions,
) -> None:
    """Run layout detection only for a window. Writes layout cache."""
    start = time.monotonic()
    recorder = None
    try:
        if options.dissection_enable:
            from mineru_vl_utils.dissection import DissectionRecorder

            recorder = DissectionRecorder(
                window.document.parse_dir / "dissection" / "layout" / f"window_{window.window_index:05d}",
                document_stem=window.document_stem,
            )
        if recorder is not None:
            recorder.record_pipeline_started()
            recorder.record_stage_started("layout_detection")

        try:
            if window.document.document_type == "image":
                with Image.open(window.document.path) as src_image:
                    src_image.load()
                    image = src_image.convert("RGB")
                try:
                    layout_result = await client.aio_layout_detect(image)
                    write_layout_window_cache(
                        window,
                        blocks_by_page=[[dict(block) for block in layout_result]],
                        page_sizes=[list(image.size)],
                        elapsed_seconds=time.monotonic() - start,
                    )
                finally:
                    image.close()
            else:
                # PDF window — batch layout detect
                pdf_bytes = window.document.path.read_bytes()
                images_list = await aio_load_images_from_pdf_bytes_range(
                    pdf_bytes,
                    start_page_id=window.start_page_id,
                    end_page_id=window.end_page_id,
                    image_type=ImageType.PIL,
                )
                try:
                    images = [d["img_pil"] for d in images_list]
                    layout_results = await client.aio_batch_layout_detect(images)
                    write_layout_window_cache(
                        window,
                        blocks_by_page=[
                            [dict(block) for block in page_blocks]
                            for page_blocks in layout_results
                        ],
                        page_sizes=[list(img.size) for img in images],
                        elapsed_seconds=time.monotonic() - start,
                    )
                finally:
                    _close_images(images_list)

            if recorder is not None:
                recorder.record_stage_finished("layout_detection")
                recorder.record_pipeline_finished("completed")
        except Exception as exc:
            if recorder is not None:
                recorder.record_stage_finished("layout_detection", "failed", exc)
                recorder.record_pipeline_finished("failed", exc)
            raise
    finally:
        if recorder is not None:
            recorder.finalize()


async def process_recognition_window(
    client: Any,
    window: WindowJob,
    options: DocumentOcrOptions,
) -> None:
    """Run recognition from layout cache for a window. Writes window cache."""
    from mineru_vl_utils.structs import ContentBlock, ExtractResult

    start = time.monotonic()
    recorder = None
    try:
        if options.dissection_enable:
            from mineru_vl_utils.dissection import DissectionRecorder

            recorder = DissectionRecorder(
                window.document.parse_dir / "dissection" / "recognition" / f"window_{window.window_index:05d}",
                document_stem=window.document_stem,
            )
        if recorder is not None:
            recorder.record_pipeline_started()
            recorder.record_stage_started("recognition")

        try:
            layout_cache = read_valid_layout_window_cache_or_artifact(
                window,
                options=options,
                rehydrate=True,
            )
            if layout_cache is None:
                raise RuntimeError(
                    f"No valid layout cache or artifact for {window.document_stem} window {window.window_index}. "
                    "Run layout detection first."
                )

            cached_blocks_by_page = layout_cache["blocks_by_page"]
            page_sizes = layout_cache["metadata"].get("page_sizes", [])

            if window.document.document_type == "image":
                with Image.open(window.document.path) as src_image:
                    src_image.load()
                    image = src_image.convert("RGB")
                try:
                    layout_blocks = ExtractResult([
                        ContentBlock(
                            b.get("type", "text"),
                            b.get("bbox", [0, 0, 1, 1]),
                            angle=b.get("angle", 0),
                            merge_prev=b.get("merge_prev", False),
                        )
                        for b in cached_blocks_by_page[0]
                    ])
                    result = await client.aio_recognize_from_layout(image, layout_blocks)
                    write_window_cache(
                        window,
                        blocks_by_page=[[dict(block) for block in result]],
                        page_sizes=[list(image.size)],
                        elapsed_seconds=time.monotonic() - start,
                    )
                finally:
                    image.close()
            else:
                # PDF window — batch recognition from layout
                pdf_bytes = window.document.path.read_bytes()
                images_list = await aio_load_images_from_pdf_bytes_range(
                    pdf_bytes,
                    start_page_id=window.start_page_id,
                    end_page_id=window.end_page_id,
                    image_type=ImageType.PIL,
                )
                try:
                    images = [d["img_pil"] for d in images_list]
                    layout_blocks_by_page = []
                    for page_blocks in cached_blocks_by_page:
                        layout_blocks_by_page.append(ExtractResult([
                            ContentBlock(
                                b.get("type", "text"),
                                b.get("bbox", [0, 0, 1, 1]),
                                angle=b.get("angle", 0),
                                merge_prev=b.get("merge_prev", False),
                            )
                            for b in page_blocks
                        ]))
                    results = await client.aio_batch_recognize_from_layout(images, layout_blocks_by_page)
                    write_window_cache(
                        window,
                        blocks_by_page=[
                            [dict(block) for block in page_blocks]
                            for page_blocks in results
                        ],
                        page_sizes=[list(img.size) for img in images],
                        elapsed_seconds=time.monotonic() - start,
                    )
                finally:
                    _close_images(images_list)

            if recorder is not None:
                recorder.record_stage_finished("recognition")
                recorder.record_pipeline_finished("completed")
        except Exception as exc:
            if recorder is not None:
                recorder.record_stage_finished("recognition", "failed", exc)
                recorder.record_pipeline_finished("failed", exc)
            raise
    finally:
        if recorder is not None:
            recorder.finalize()


async def process_full_window(
    client: Any,
    window: WindowJob,
    options: DocumentOcrOptions,
) -> None:
    """Full pipeline: layout then recognition. Writes both caches."""
    await process_layout_window(client, window, options)
    await process_recognition_window(client, window, options)


@contextmanager
def _temporary_env(updates: dict[str, str]):
    original = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def run_document_ocr(
    options: DocumentOcrOptions,
    client_factory: Callable[[DocumentOcrOptions], Any] = create_mineru_client,
) -> list[DocumentJobResult]:
    if options.stream and not options.dissection_enable:
        raise click.ClickException("--stream requires --dissection")
    if options.max_windows < 1:
        raise click.ClickException("--max-windows must be at least 1")
    if options.max_http_concurrency_per_window < 1:
        raise click.ClickException("--max-http-concurrency-per-window must be at least 1")
    jobs = collect_document_jobs(
        options.input_path,
        options.output_dir,
        start_page_id=options.start_page_id,
        end_page_id=options.end_page_id,
        resume=options.resume,
    )
    if not jobs:
        return []
    windows = interleave_window_jobs(build_window_jobs(jobs, options.page_window_size))
    windows_by_document: dict[int, list[WindowJob]] = {}
    for window in windows:
        windows_by_document.setdefault(window.document_index, []).append(window)

    # Choose processing function based on phase
    if options.phase == OcrPhase.FULL:
        process_fn = process_full_window
    elif options.phase == OcrPhase.LAYOUT:
        process_fn = process_layout_window
    elif options.phase == OcrPhase.RECOGNIZE:
        process_fn = process_recognition_window
    else:
        raise ValueError(f"Unknown phase: {options.phase}")

    client = client_factory(options)
    semaphore = asyncio.Semaphore(options.max_windows)
    failed_documents: dict[int, str] = {}
    document_starts = {index: time.monotonic() for index, _job in enumerate(jobs)}

    async def run_one(window: WindowJob) -> None:
        recognition_cache = read_valid_window_cache(window)
        if options.phase == OcrPhase.RECOGNIZE:
            if recognition_cache is not None:
                return
        elif options.phase == OcrPhase.FULL and recognition_cache is not None:
            try:
                if read_valid_layout_window_cache_or_artifact(
                    window,
                    options=options,
                    rehydrate=True,
                ) is not None:
                    return
            except RuntimeError:
                pass
        # Skip if layout cache already exists (for LAYOUT phase)
        if options.phase == OcrPhase.LAYOUT:
            if read_valid_layout_window_cache(window) is not None:
                return
        async with semaphore:
            if window.document_index in failed_documents:
                return
            try:
                if (
                    options.phase == OcrPhase.FULL
                    and recognition_cache is not None
                    and read_valid_layout_window_cache(window) is None
                ):
                    coro = process_layout_window(client, window, options)
                else:
                    coro = process_fn(client, window, options)
                await asyncio.wait_for(
                    coro,
                    timeout=options.per_window_timeout,
                )
            except asyncio.TimeoutError:
                failed_documents[window.document_index] = (
                    f"window {window.window_index} timed out after {options.per_window_timeout} seconds"
                )
            except Exception as exc:
                failed_documents[window.document_index] = str(exc)

    with _temporary_env(
        {
            "MINERU_VLM_FORMULA_ENABLE": str(options.formula_enable),
            "MINERU_VLM_TABLE_ENABLE": str(options.table_enable),
        }
    ):
        pending: Iterable[asyncio.Task[None]] = [asyncio.create_task(run_one(window)) for window in windows]
        if options.progress:
            try:
                from tqdm import tqdm

                pending = tqdm(asyncio.as_completed(pending), total=len(windows), desc="OCR document windows")
            except Exception:
                pending = asyncio.as_completed(pending)
        else:
            pending = asyncio.as_completed(pending)

        for task in pending:
            await task

    results: list[DocumentJobResult] = []
    for index, job in enumerate(jobs):
        document_windows = windows_by_document.get(index, [])
        elapsed = time.monotonic() - document_starts[index]
        if index in failed_documents:
            completed = sum(read_valid_window_cache(window) is not None for window in document_windows)
            error = failed_documents[index]
            write_document_status(
                job,
                status="failed",
                elapsed_seconds=elapsed,
                completed_windows=completed,
                total_windows=len(document_windows),
                error=error,
            )
            results.append(
                DocumentJobResult(
                    job=job,
                    status="failed",
                    elapsed_seconds=round(elapsed, 3),
                    error=error,
                )
            )
            continue
        try:
            if options.phase == OcrPhase.LAYOUT:
                # Layout-only: write layout artifact and status, no assembly
                write_document_layout_artifact(job, document_windows)
                write_document_status(
                    job,
                    status="completed",
                    elapsed_seconds=elapsed,
                    completed_windows=len(document_windows),
                    total_windows=len(document_windows),
                )
                results.append(
                    DocumentJobResult(
                        job=job,
                        status="completed",
                        elapsed_seconds=round(elapsed, 3),
                    )
                )
            else:
                # FULL or RECOGNIZE: assemble standard outputs
                results.append(
                    assemble_document_outputs(
                        job,
                        document_windows,
                        elapsed_seconds=elapsed,
                    )
                )
                # FULL also writes layout artifact for later re-runs
                if options.phase == OcrPhase.FULL:
                    write_document_layout_artifact(job, document_windows)
        except Exception as exc:
            write_document_status(
                job,
                status="failed",
                elapsed_seconds=elapsed,
                completed_windows=sum(read_valid_window_cache(window) is not None for window in document_windows),
                total_windows=len(document_windows),
                error=str(exc),
            )
            results.append(
                DocumentJobResult(
                    job=job,
                    status="failed",
                    elapsed_seconds=round(elapsed, 3),
                    error=str(exc),
                )
            )
    return results


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-p", "--path", "input_path", required=True, type=click.Path(path_type=Path), help="Document image, PDF, or directory.")
@click.option("-o", "--output", "output_dir", required=True, type=click.Path(path_type=Path), help="Output directory.")
@click.option("-u", "--url", "server_url", default=None, help="Shared MinerU VLM server URL.")
@click.option("--layout-url", "layout_server_url", default=None, help="Layout stage server URL.")
@click.option("--recognition-url", "recognition_server_url", default=None, help="Recognition stage server URL.")
@click.option("-r", "--resume", is_flag=True, help="Skip documents with completed status files.")
@click.option("--formula/--no-formula", "formula_enable", default=True, show_default=True, help="Render formula content.")
@click.option("--table/--no-table", "table_enable", default=True, show_default=True, help="Render table HTML content.")
@click.option("--image-analysis/--no-image-analysis", default=True, show_default=True, help="Recognize standalone image/chart blocks.")
@click.option("--page-window-size", default=4, show_default=True, type=int, help="PDF pages per scheduled window.")
@click.option("--max-windows", default=16, show_default=True, type=int, help="Maximum in-flight document windows.")
@click.option("--max-http-concurrency-per-window", default=1, show_default=True, type=int, help="Maximum in-flight VLM HTTP requests per window.")
@click.option("--per-window-timeout", default=600.0, show_default=True, type=float, help="Timeout per image/PDF window in seconds.")
@click.option("--http-timeout", default=600, show_default=True, type=int, help="HTTP read timeout passed to MinerUClient.")
@click.option("--connect-timeout", default=10, show_default=True, type=int, help="HTTP connect timeout passed to MinerUClient.")
@click.option("--max-retries", default=3, show_default=True, type=int, help="HTTP retry count passed to MinerUClient.")
@click.option("--retry-backoff-factor", default=0.5, show_default=True, type=float, help="HTTP retry backoff factor passed to MinerUClient.")
@click.option("-s", "--start", "start_page_id", default=0, show_default=True, type=int, help="Starting PDF page, beginning from 0.")
@click.option("-e", "--end", "end_page_id", default=None, type=int, help="Ending PDF page, beginning from 0.")
@click.option("--progress/--no-progress", default=True, show_default=True, help="Show a progress bar.")
@click.option("--dissection/--no-dissection", "dissection_enable", default=False, show_default=True, help="Write VLM dissection artifacts.")
@click.option("--stream/--no-stream", "stream", default=False, show_default=True, help="Use streaming recognition to capture partial dissection output.")
def main(
    input_path: Path,
    output_dir: Path,
    server_url: str | None,
    layout_server_url: str | None,
    recognition_server_url: str | None,
    resume: bool,
    formula_enable: bool,
    table_enable: bool,
    image_analysis: bool,
    page_window_size: int,
    max_windows: int,
    max_http_concurrency_per_window: int,
    per_window_timeout: float,
    http_timeout: int,
    connect_timeout: int,
    max_retries: int,
    retry_backoff_factor: float,
    start_page_id: int,
    end_page_id: int | None,
    progress: bool,
    dissection_enable: bool,
    stream: bool,
) -> None:
    click.echo(
        "WARNING: mineru-ocr-documents is deprecated. Use 'mineru-phase run' instead.",
        err=True,
    )
    options = DocumentOcrOptions(
        input_path=input_path,
        output_dir=output_dir,
        server_url=server_url,
        layout_server_url=layout_server_url,
        recognition_server_url=recognition_server_url,
        resume=resume,
        formula_enable=formula_enable,
        table_enable=table_enable,
        image_analysis=image_analysis,
        page_window_size=page_window_size,
        max_windows=max_windows,
        max_http_concurrency_per_window=max_http_concurrency_per_window,
        per_window_timeout=per_window_timeout,
        http_timeout=http_timeout,
        connect_timeout=connect_timeout,
        max_retries=max_retries,
        retry_backoff_factor=retry_backoff_factor,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        progress=progress,
        dissection_enable=dissection_enable,
        stream=stream,
    )
    results = asyncio.run(run_document_ocr(options))
    completed = sum(result.status == "completed" for result in results)
    failed = len(results) - completed
    click.echo(f"Processed {len(results)} document(s): {completed} completed, {failed} failed.")
    if failed:
        raise click.ClickException(f"{failed} document(s) failed.")


if __name__ == "__main__":
    main()

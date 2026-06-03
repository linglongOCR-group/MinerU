# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import click
from PIL import Image, ImageDraw

from mineru.cli.common import image_suffixes


IMAGE_SUFFIXES = {
    suffix if suffix.startswith(".") else f".{suffix}" for suffix in image_suffixes
}


@dataclass(frozen=True)
class ImageJob:
    path: Path
    stem: str
    parse_dir: Path


@dataclass(frozen=True)
class ImageOcrOptions:
    input_path: Path
    output_dir: Path
    server_url: str | None = None
    layout_server_url: str | None = None
    recognition_server_url: str | None = None
    resume: bool = False
    formula_enable: bool = True
    table_enable: bool = True
    image_analysis: bool = True
    max_concurrency: int = 16
    max_http_concurrency_per_image: int = 1
    per_image_timeout: float = 600.0
    http_timeout: int = 600
    connect_timeout: int = 10
    max_retries: int = 3
    retry_backoff_factor: float = 0.5
    progress: bool = True
    dissection_enable: bool = False
    stream: bool = False


@dataclass(frozen=True)
class ImageJobResult:
    job: ImageJob
    status: str
    elapsed_seconds: float
    error: str | None = None


ImageWorker = Callable[[ImageJob], Awaitable[ImageJobResult]]

_LAYOUT_FILL_ALPHA = 76
_LAYOUT_OUTLINE_ALPHA = 230

_LAYOUT_COLORS: dict[str, tuple[int, int, int]] = {
    "code_body": (102, 0, 204),
    "code_caption": (204, 153, 255),
    "code_footnote": (229, 204, 255),
    "table_body": (204, 204, 0),
    "table_caption": (255, 255, 102),
    "table_footnote": (229, 255, 204),
    "image_body": (153, 255, 51),
    "chart_body": (153, 255, 51),
    "image_caption": (102, 178, 255),
    "chart_caption": (102, 178, 255),
    "image_footnote": (255, 178, 102),
    "chart_footnote": (255, 178, 102),
    "title": (102, 102, 255),
    "text": (153, 0, 76),
    "ref_text": (153, 0, 76),
    "abstract": (153, 0, 76),
    "interline_equation": (0, 255, 0),
    "list": (40, 169, 92),
    "index": (40, 169, 92),
    "seal": (153, 255, 51),
}

_NESTED_LAYOUT_TYPES = {"image", "chart", "code", "table"}


def _json_default(value: Any):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _origin_image_path(job: ImageJob) -> Path:
    return job.parse_dir / f"{job.stem}_origin{job.path.suffix}"


def _layout_image_path(job: ImageJob) -> Path:
    return job.parse_dir / f"{job.stem}_layout.png"


def _model_layout_image_path(job: ImageJob) -> Path:
    return job.parse_dir / f"{job.stem}_model_layout.png"


def _coerce_bbox(
    bbox: Any,
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    width, height = image_size
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height

    x0 = max(0, min(width, round(x0)))
    y0 = max(0, min(height, round(y0)))
    x1 = max(0, min(width, round(x1)))
    y1 = max(0, min(height, round(y1)))
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def _iter_layout_blocks(
    middle_json: dict[str, Any],
    *,
    image_size: tuple[int, int],
) -> Iterable[tuple[str, tuple[int, int, int, int]]]:
    pdf_info = middle_json.get("pdf_info")
    if not isinstance(pdf_info, list) or not pdf_info:
        return
    page = pdf_info[0]
    if not isinstance(page, dict):
        return

    para_blocks = page.get("para_blocks") or []
    if not isinstance(para_blocks, list):
        return

    for block in para_blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", ""))
        if block_type in _NESTED_LAYOUT_TYPES and isinstance(block.get("blocks"), list):
            for nested_block in block["blocks"]:
                if not isinstance(nested_block, dict):
                    continue
                bbox = _coerce_bbox(nested_block.get("bbox"), image_size=image_size)
                if bbox is not None:
                    yield str(nested_block.get("type", block_type)), bbox
            continue

        bbox = _coerce_bbox(block.get("bbox"), image_size=image_size)
        if bbox is not None:
            yield block_type, bbox


def _iter_model_blocks(
    blocks: list[dict[str, Any]],
    *,
    image_size: tuple[int, int],
) -> Iterable[tuple[str, tuple[int, int, int, int]]]:
    for block in blocks:
        if not isinstance(block, dict):
            continue
        bbox = _coerce_bbox(block.get("bbox"), image_size=image_size)
        if bbox is not None:
            yield str(block.get("type", "")), bbox


def _draw_label(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    image_size: tuple[int, int],
) -> None:
    width, height = image_size
    label = label or "unknown"
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    pad_x = 5
    pad_y = 2
    label_width = min(max(16, text_width + pad_x * 2), width)
    label_height = max(14, text_height + pad_y * 2)
    x0 = min(max(bbox[0], 0), max(width - label_width, 0))
    if bbox[1] >= label_height:
        y0 = bbox[1] - label_height
    else:
        y0 = min(max(bbox[1], 0), max(height - label_height, 0))
    x1 = x0 + label_width
    y1 = y0 + label_height
    draw.rectangle((x0, y0, x1, y1), fill=(*color, 240))
    draw.text(
        (x0 + pad_x, y0 + (label_height - text_height) / 2 - 1),
        label,
        fill=(255, 255, 255, 255),
    )


def render_annotated_layout_image(
    source_image: Image.Image,
    annotations: Iterable[tuple[str, tuple[int, int, int, int]]],
) -> Image.Image:
    annotated = source_image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    annotations = list(annotations)

    for block_type, bbox in annotations:
        rgb = _LAYOUT_COLORS.get(block_type, (255, 0, 0))
        overlay_draw.rectangle(
            bbox,
            fill=(*rgb, _LAYOUT_FILL_ALPHA),
            outline=(*rgb, _LAYOUT_OUTLINE_ALPHA),
            width=2,
        )

    annotated = Image.alpha_composite(annotated, overlay)
    label_draw = ImageDraw.Draw(annotated)
    for block_type, bbox in annotations:
        rgb = _LAYOUT_COLORS.get(block_type, (255, 0, 0))
        _draw_label(
            label_draw,
            bbox=bbox,
            label=block_type,
            color=rgb,
            image_size=annotated.size,
        )

    return annotated.convert("RGB")


def render_layout_image(
    source_image: Image.Image,
    middle_json: dict[str, Any],
) -> Image.Image:
    return render_annotated_layout_image(
        source_image,
        _iter_layout_blocks(middle_json, image_size=source_image.size),
    )


def render_model_layout_image(
    source_image: Image.Image,
    blocks: list[dict[str, Any]],
) -> Image.Image:
    return render_annotated_layout_image(
        source_image,
        _iter_model_blocks(blocks, image_size=source_image.size),
    )


def write_image_visual_outputs(
    job: ImageJob,
    *,
    blocks: list[dict[str, Any]],
    middle_json: dict[str, Any],
) -> None:
    job.parse_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(job.path, _origin_image_path(job))
    with Image.open(job.path) as source_image:
        source_image.load()
        render_model_layout_image(source_image, blocks).save(_model_layout_image_path(job))
        render_layout_image(source_image, middle_json).save(_layout_image_path(job))


def _status_path(job: ImageJob) -> Path:
    return job.parse_dir / f"{job.stem}_status.json"


def _is_completed_status(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except Exception:
        return False


def _discover_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise click.ClickException(f"Unsupported image suffix: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise click.ClickException(f"Input path does not exist: {input_path}")
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _unique_stems(paths: Iterable[Path]) -> dict[Path, str]:
    counts: dict[str, int] = {}
    stems: dict[Path, str] = {}
    for path in paths:
        base = path.stem
        counts[base] = counts.get(base, 0) + 1
        stems[path] = base if counts[base] == 1 else f"{base}_{counts[base]}"
    return stems


def collect_image_jobs(input_path: Path, output_dir: Path, resume: bool = False) -> list[ImageJob]:
    image_paths = _discover_images(input_path)
    stems = _unique_stems(image_paths)
    jobs = [
        ImageJob(path=path, stem=stem, parse_dir=output_dir / stem / "vlm")
        for path, stem in stems.items()
    ]
    if not resume:
        return jobs
    return [job for job in jobs if not _is_completed_status(_status_path(job))]


async def run_image_jobs(
    jobs: list[ImageJob],
    worker: ImageWorker,
    max_concurrency: int,
    progress: bool = True,
) -> list[ImageJobResult]:
    if max_concurrency < 1:
        raise click.ClickException("--max-concurrency must be at least 1")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(job: ImageJob) -> ImageJobResult:
        start = time.monotonic()
        async with semaphore:
            try:
                return await worker(job)
            except Exception as exc:
                return ImageJobResult(
                    job=job,
                    status="failed",
                    elapsed_seconds=round(time.monotonic() - start, 3),
                    error=str(exc),
                )

    tasks = [asyncio.create_task(run_one(job)) for job in jobs]
    pending = asyncio.as_completed(tasks)
    if progress:
        try:
            from tqdm import tqdm

            pending = tqdm(pending, total=len(tasks), desc="OCR images")
        except Exception:
            pass

    results = []
    for task in pending:
        results.append(await task)
    return results


def build_middle_json_from_blocks(blocks: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, Any]:
    from mineru.backend.utils.para_block_utils import (
        build_para_blocks_from_preproc,
        cleanup_internal_para_block_metadata,
        merge_para_text_blocks,
    )
    from mineru.backend.vlm.vlm_magic_model import MagicModel
    from mineru.version import __version__

    width, height = image_size
    magic_model = MagicModel(blocks, width, height)
    page_blocks = []
    page_blocks.extend(
        [
            *magic_model.get_image_blocks(),
            *magic_model.get_table_blocks(),
            *magic_model.get_chart_blocks(),
            *magic_model.get_code_blocks(),
            *magic_model.get_ref_text_blocks(),
            *magic_model.get_phonetic_blocks(),
            *magic_model.get_title_blocks(),
            *magic_model.get_text_blocks(),
            *magic_model.get_interline_equation_blocks(),
            *magic_model.get_list_blocks(),
        ]
    )
    page_blocks.sort(key=lambda block: block.get("index", float("inf")))

    pdf_info = [
        {
            "preproc_blocks": page_blocks,
            "discarded_blocks": magic_model.get_discarded_blocks(),
            "page_size": [width, height],
            "page_idx": 0,
        }
    ]
    build_para_blocks_from_preproc(pdf_info)
    merge_para_text_blocks(pdf_info, allow_cross_page=False)
    cleanup_internal_para_block_metadata(pdf_info)

    return {"pdf_info": pdf_info, "_backend": "vlm", "_version_name": __version__}


def render_outputs(middle_json: dict[str, Any], img_bucket_path: str = "") -> tuple[str, list[dict[str, Any]]]:
    from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make
    from mineru.utils.enum_class import MakeMode

    pdf_info = middle_json["pdf_info"]
    markdown = union_make(pdf_info, MakeMode.MM_MD, img_bucket_path) or ""
    content_list = union_make(pdf_info, MakeMode.CONTENT_LIST, img_bucket_path) or []
    return markdown, content_list


def write_success_outputs(
    job: ImageJob,
    *,
    blocks: list[dict[str, Any]],
    middle_json: dict[str, Any],
    markdown: str,
    content_list: list[dict[str, Any]],
    elapsed_seconds: float,
) -> None:
    job.parse_dir.mkdir(parents=True, exist_ok=True)
    (job.parse_dir / f"{job.stem}.md").write_text(markdown, encoding="utf-8")
    _write_json(job.parse_dir / f"{job.stem}_model.json", blocks)
    _write_json(job.parse_dir / f"{job.stem}_middle.json", middle_json)
    _write_json(job.parse_dir / f"{job.stem}_content_list.json", content_list)
    write_image_visual_outputs(job, blocks=blocks, middle_json=middle_json)
    _write_json(
        _status_path(job),
        {
            "status": "completed",
            "image_path": str(job.path),
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
    )


def write_failed_status(job: ImageJob, error: str, elapsed_seconds: float) -> None:
    _write_json(
        _status_path(job),
        {
            "status": "failed",
            "image_path": str(job.path),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "error": error,
        },
    )


def create_mineru_client(options: ImageOcrOptions):
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
        max_concurrency=options.max_http_concurrency_per_image,
        image_analysis=options.image_analysis,
        enable_table_formula_eq_wrap=True,
    )


async def process_image_job(
    client: Any,
    job: ImageJob,
    options: ImageOcrOptions,
) -> ImageJobResult:
    start = time.monotonic()
    recorder = None
    try:
        if options.dissection_enable:
            from mineru_vl_utils.dissection import DissectionRecorder

            recorder = DissectionRecorder(job.parse_dir / "dissection", document_stem=job.stem)
        with Image.open(job.path) as src_image:
            src_image.load()
            image = src_image.convert("RGB")
        try:
            blocks_result = await asyncio.wait_for(
                client.aio_two_step_extract(
                    image,
                    image_analysis=options.image_analysis,
                    dissection_recorder=recorder,
                    dissection_stream=options.stream,
                    page_idx=0,
                ),
                timeout=options.per_image_timeout,
            )
        finally:
            image.close()
        blocks = [dict(block) for block in blocks_result]
        if recorder is not None:
            recorder.record_stage_started("output_generation")
        try:
            middle_json = build_middle_json_from_blocks(blocks, image.size)
            markdown, content_list = render_outputs(middle_json)
            elapsed = time.monotonic() - start
            write_success_outputs(
                job,
                blocks=blocks,
                middle_json=middle_json,
                markdown=markdown,
                content_list=content_list,
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            if recorder is not None:
                recorder.record_stage_finished("output_generation", "failed", exc)
                recorder.record_pipeline_finished("failed", exc)
            raise
        if recorder is not None:
            recorder.record_stage_finished("output_generation", "completed")
            recorder.record_pipeline_finished("completed")
        return ImageJobResult(job=job, status="completed", elapsed_seconds=round(elapsed, 3))
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        error = f"timed out after {options.per_image_timeout} seconds"
        if recorder is not None:
            timeout_error = TimeoutError(error)
            recorder.record_pipeline_finished("timeout", timeout_error)
        write_failed_status(job, error, elapsed)
        return ImageJobResult(job=job, status="timeout", elapsed_seconds=round(elapsed, 3), error=error)
    except Exception as exc:
        elapsed = time.monotonic() - start
        error = str(exc)
        if recorder is not None:
            recorder.record_pipeline_finished("failed", exc)
        write_failed_status(job, error, elapsed)
        return ImageJobResult(job=job, status="failed", elapsed_seconds=round(elapsed, 3), error=error)
    finally:
        if recorder is not None:
            recorder.finalize()


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


async def run_image_ocr(
    options: ImageOcrOptions,
    client_factory: Callable[[ImageOcrOptions], Any] = create_mineru_client,
) -> list[ImageJobResult]:
    if options.stream and not options.dissection_enable:
        raise click.ClickException("--stream requires --dissection")
    if options.max_http_concurrency_per_image < 1:
        raise click.ClickException("--max-http-concurrency-per-image must be at least 1")
    jobs = collect_image_jobs(options.input_path, options.output_dir, resume=options.resume)
    if not jobs:
        return []

    client = client_factory(options)

    async def worker(job: ImageJob) -> ImageJobResult:
        return await process_image_job(client, job, options)

    with _temporary_env(
        {
            "MINERU_VLM_FORMULA_ENABLE": str(options.formula_enable),
            "MINERU_VLM_TABLE_ENABLE": str(options.table_enable),
        }
    ):
        return await run_image_jobs(
            jobs,
            worker,
            max_concurrency=options.max_concurrency,
            progress=options.progress,
        )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-p", "--path", "input_path", required=True, type=click.Path(path_type=Path), help="Image file or directory.")
@click.option("-o", "--output", "output_dir", required=True, type=click.Path(path_type=Path), help="Output directory.")
@click.option("-u", "--url", "server_url", default=None, help="Shared MinerU VLM server URL.")
@click.option("--layout-url", "layout_server_url", default=None, help="Layout stage server URL.")
@click.option("--recognition-url", "recognition_server_url", default=None, help="Recognition stage server URL.")
@click.option("-r", "--resume", is_flag=True, help="Skip images with completed status files.")
@click.option("--formula/--no-formula", "formula_enable", default=True, show_default=True, help="Render formula content.")
@click.option("--table/--no-table", "table_enable", default=True, show_default=True, help="Render table HTML content.")
@click.option("--image-analysis/--no-image-analysis", default=True, show_default=True, help="Recognize standalone image/chart blocks.")
@click.option("--max-concurrency", default=16, show_default=True, type=int, help="Maximum in-flight image jobs.")
@click.option("--max-http-concurrency-per-image", default=1, show_default=True, type=int, help="Maximum in-flight VLM HTTP requests per image job.")
@click.option("--per-image-timeout", default=600.0, show_default=True, type=float, help="Timeout per image in seconds.")
@click.option("--http-timeout", default=600, show_default=True, type=int, help="HTTP read timeout passed to MinerUClient.")
@click.option("--connect-timeout", default=10, show_default=True, type=int, help="HTTP connect timeout passed to MinerUClient.")
@click.option("--max-retries", default=3, show_default=True, type=int, help="HTTP retry count passed to MinerUClient.")
@click.option("--retry-backoff-factor", default=0.5, show_default=True, type=float, help="HTTP retry backoff factor passed to MinerUClient.")
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
    max_concurrency: int,
    max_http_concurrency_per_image: int,
    per_image_timeout: float,
    http_timeout: int,
    connect_timeout: int,
    max_retries: int,
    retry_backoff_factor: float,
    progress: bool,
    dissection_enable: bool,
    stream: bool,
) -> None:
    click.echo(
        "WARNING: mineru-ocr-images is deprecated. Use 'mineru-phase run' instead.",
        err=True,
    )
    if stream and not dissection_enable:
        raise click.ClickException("--stream requires --dissection")
    if max_http_concurrency_per_image < 1:
        raise click.ClickException("--max-http-concurrency-per-image must be at least 1")
    options = ImageOcrOptions(
        input_path=input_path,
        output_dir=output_dir,
        server_url=server_url,
        layout_server_url=layout_server_url,
        recognition_server_url=recognition_server_url,
        resume=resume,
        formula_enable=formula_enable,
        table_enable=table_enable,
        image_analysis=image_analysis,
        max_concurrency=max_concurrency,
        max_http_concurrency_per_image=max_http_concurrency_per_image,
        per_image_timeout=per_image_timeout,
        http_timeout=http_timeout,
        connect_timeout=connect_timeout,
        max_retries=max_retries,
        retry_backoff_factor=retry_backoff_factor,
        progress=progress,
        dissection_enable=dissection_enable,
        stream=stream,
    )
    results = asyncio.run(run_image_ocr(options))
    completed = sum(result.status == "completed" for result in results)
    failed = len(results) - completed
    click.echo(f"Processed {len(results)} image(s): {completed} completed, {failed} failed.")


if __name__ == "__main__":
    main()

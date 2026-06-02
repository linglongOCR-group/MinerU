# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import click
from PIL import Image

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
    per_image_timeout: float = 600.0
    http_timeout: int = 600
    connect_timeout: int = 10
    max_retries: int = 3
    retry_backoff_factor: float = 0.5
    progress: bool = True


@dataclass(frozen=True)
class ImageJobResult:
    job: ImageJob
    status: str
    elapsed_seconds: float
    error: str | None = None


ImageWorker = Callable[[ImageJob], Awaitable[ImageJobResult]]


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
        max_concurrency=options.max_concurrency,
        image_analysis=options.image_analysis,
        enable_table_formula_eq_wrap=True,
    )


async def process_image_job(
    client: Any,
    job: ImageJob,
    options: ImageOcrOptions,
    request_semaphore: asyncio.Semaphore,
) -> ImageJobResult:
    start = time.monotonic()
    try:
        with Image.open(job.path) as src_image:
            src_image.load()
            image = src_image.convert("RGB")
        blocks_result = await asyncio.wait_for(
            client.aio_two_step_extract(
                image,
                semaphore=request_semaphore,
                image_analysis=options.image_analysis,
            ),
            timeout=options.per_image_timeout,
        )
        blocks = [dict(block) for block in blocks_result]
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
        return ImageJobResult(job=job, status="completed", elapsed_seconds=round(elapsed, 3))
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        error = f"timed out after {options.per_image_timeout} seconds"
        write_failed_status(job, error, elapsed)
        return ImageJobResult(job=job, status="timeout", elapsed_seconds=round(elapsed, 3), error=error)
    except Exception as exc:
        elapsed = time.monotonic() - start
        error = str(exc)
        write_failed_status(job, error, elapsed)
        return ImageJobResult(job=job, status="failed", elapsed_seconds=round(elapsed, 3), error=error)


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
    jobs = collect_image_jobs(options.input_path, options.output_dir, resume=options.resume)
    if not jobs:
        return []

    client = client_factory(options)
    request_semaphore = asyncio.Semaphore(options.max_concurrency)

    async def worker(job: ImageJob) -> ImageJobResult:
        return await process_image_job(client, job, options, request_semaphore)

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
@click.option("--max-concurrency", default=16, show_default=True, type=int, help="Maximum in-flight image jobs and server requests.")
@click.option("--per-image-timeout", default=600.0, show_default=True, type=float, help="Timeout per image in seconds.")
@click.option("--http-timeout", default=600, show_default=True, type=int, help="HTTP read timeout passed to MinerUClient.")
@click.option("--connect-timeout", default=10, show_default=True, type=int, help="HTTP connect timeout passed to MinerUClient.")
@click.option("--max-retries", default=3, show_default=True, type=int, help="HTTP retry count passed to MinerUClient.")
@click.option("--retry-backoff-factor", default=0.5, show_default=True, type=float, help="HTTP retry backoff factor passed to MinerUClient.")
@click.option("--progress/--no-progress", default=True, show_default=True, help="Show a progress bar.")
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
    per_image_timeout: float,
    http_timeout: int,
    connect_timeout: int,
    max_retries: int,
    retry_backoff_factor: float,
    progress: bool,
) -> None:
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
        per_image_timeout=per_image_timeout,
        http_timeout=http_timeout,
        connect_timeout=connect_timeout,
        max_retries=max_retries,
        retry_backoff_factor=retry_backoff_factor,
        progress=progress,
    )
    results = asyncio.run(run_image_ocr(options))
    completed = sum(result.status == "completed" for result in results)
    failed = len(results) - completed
    click.echo(f"Processed {len(results)} image(s): {completed} completed, {failed} failed.")


if __name__ == "__main__":
    main()

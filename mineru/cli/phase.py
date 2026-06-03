"""mineru-phase: Phase-aware VLM OCR pipeline CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from mineru.cli.document_ocr import DocumentOcrOptions, OcrPhase, run_document_ocr


def _path_value(value):
    """Ensure click Path values are Path objects."""
    if value is not None and not isinstance(value, Path):
        return Path(value)
    return value


def _source_option(func):
    return click.option(
        "--source",
        "input_path",
        required=True,
        type=click.Path(exists=True),
        help="Document image, PDF, or directory.",
    )(func)


def _output_option(func):
    return click.option(
        "--output",
        "output_dir",
        required=True,
        type=click.Path(),
        help="OCR output directory.",
    )(func)


def _layout_output_option(*, required: bool):
    def decorator(func):
        return click.option(
            "--layout-output",
            "layout_output_dir",
            required=required,
            type=click.Path(),
            help="Layout artifact output directory.",
        )(func)

    return decorator


def _layout_url_option(func):
    return click.option("--layout-url", "layout_server_url", default=None, help="Layout stage server URL.")(func)


def _recognition_url_option(func):
    return click.option("--recognition-url", "recognition_server_url", default=None, help="Recognition stage server URL.")(func)


def _layout_input_option(func):
    return click.option(
        "--layout-input",
        "layout_input_path",
        required=True,
        type=click.Path(exists=True),
        help="Layout artifact root, or one layout artifact JSON for a single source document.",
    )(func)


def _execution_options(*, include_page_window_size: bool):
    def decorator(func):
        if include_page_window_size:
            func = click.option("--page-window-size", default=4, show_default=True, type=int, help="PDF pages per scheduled window.")(func)
        func = click.option("--max-windows", default=16, show_default=True, type=int, help="Maximum in-flight document windows.")(func)
        func = click.option("--max-model-concurrency", "max_http_concurrency_per_window", default=1, show_default=True, type=int, help="Maximum in-flight VLM requests per window.")(func)
        func = click.option("--timeout", "per_window_timeout", default=600.0, show_default=True, type=float, help="Timeout per window in seconds.")(func)
        func = click.option("--http-timeout", default=600, show_default=True, type=int, help="HTTP read timeout.")(func)
        func = click.option("--connect-timeout", default=10, show_default=True, type=int, help="HTTP connect timeout.")(func)
        func = click.option("--max-retries", default=3, show_default=True, type=int, help="HTTP retry count.")(func)
        func = click.option("--retry-backoff-factor", default=0.5, show_default=True, type=float, help="HTTP retry backoff factor.")(func)
        func = click.option("-r", "--resume", is_flag=True, help="Skip documents with completed status files.")(func)
        func = click.option("-s", "--start", "start_page_id", default=0, show_default=True, type=int, help="Starting PDF page, beginning from 0.")(func)
        func = click.option("-e", "--end", "end_page_id", default=None, type=int, help="Ending PDF page, beginning from 0.")(func)
        func = click.option("--progress/--no-progress", default=True, show_default=True, help="Show a progress bar.")(func)
        return func

    return decorator


def _recognition_options(func):
    func = click.option("--formula/--no-formula", "formula_enable", default=True, show_default=True, help="Render formula content.")(func)
    func = click.option("--table/--no-table", "table_enable", default=True, show_default=True, help="Render table HTML content.")(func)
    func = click.option("--image-analysis/--no-image-analysis", default=True, show_default=True, help="Recognize standalone image/chart blocks.")(func)
    return func


def _layout_dissection_option(func):
    return click.option("--dissection/--no-dissection", "dissection_enable", default=False, show_default=True, help="Write VLM dissection artifacts.")(func)


def _dissection_options(func):
    func = click.option("--dissection/--no-dissection", "dissection_enable", default=False, show_default=True, help="Write VLM dissection artifacts.")(func)
    func = click.option("--stream/--no-stream", "stream", default=False, show_default=True, help="Use streaming recognition.")(func)
    return func


def _build_options(phase, **kwargs):
    """Build DocumentOcrOptions from CLI kwargs."""
    input_path = _path_value(kwargs.pop("input_path"))
    output_dir = _path_value(kwargs.pop("output_dir"))
    layout_input_path = _path_value(kwargs.pop("layout_input_path", None))
    layout_output_dir = _path_value(kwargs.pop("layout_output_dir", None))
    if phase == OcrPhase.LAYOUT and layout_output_dir is None:
        layout_output_dir = output_dir
    return DocumentOcrOptions(
        input_path=input_path,
        output_dir=output_dir,
        phase=phase,
        layout_input_path=layout_input_path,
        layout_output_dir=layout_output_dir,
        **kwargs,
    )


def _build_layout_options(**kwargs):
    layout_output_dir = kwargs["layout_output_dir"]
    kwargs["output_dir"] = layout_output_dir
    return _build_options(OcrPhase.LAYOUT, **kwargs)


@click.group()
def main():
    """Phase-aware VLM OCR pipeline."""
    pass


@main.command()
@_source_option
@_output_option
@_layout_output_option(required=False)
@_layout_url_option
@_recognition_url_option
@_execution_options(include_page_window_size=True)
@_recognition_options
@_dissection_options
def run(**kwargs):
    """Full OCR: layout -> recognition -> assemble."""
    if kwargs.get("stream") and not kwargs.get("dissection_enable"):
        raise click.ClickException("--stream requires --dissection")
    options = _build_options(OcrPhase.FULL, **kwargs)
    results = asyncio.run(run_document_ocr(options))
    completed = sum(r.status == "completed" for r in results)
    failed = len(results) - completed
    click.echo(f"Processed {len(results)} document(s): {completed} completed, {failed} failed.")
    if failed:
        raise click.ClickException(f"{failed} document(s) failed.")


@main.command()
@_source_option
@_layout_output_option(required=True)
@_layout_url_option
@_execution_options(include_page_window_size=True)
@_layout_dissection_option
def layout(**kwargs):
    """Layout detection only. Outputs <stem>_layout.json and <stem>_status.json per document."""
    options = _build_layout_options(**kwargs)
    results = asyncio.run(run_document_ocr(options))
    completed = sum(r.status == "completed" for r in results)
    failed = len(results) - completed
    click.echo(f"Layout completed: {completed} succeeded, {failed} failed.")
    if failed:
        raise click.ClickException(f"{failed} document(s) failed.")


@main.command()
@_source_option
@_layout_input_option
@_output_option
@_recognition_url_option
@_execution_options(include_page_window_size=False)
@_recognition_options
@_dissection_options
def recognize(**kwargs):
    """Recognition from existing layout artifact. Produces full OCR output."""
    if kwargs.get("stream") and not kwargs.get("dissection_enable"):
        raise click.ClickException("--stream requires --dissection")
    options = _build_options(OcrPhase.RECOGNIZE, **kwargs)
    results = asyncio.run(run_document_ocr(options))
    completed = sum(r.status == "completed" for r in results)
    failed = len(results) - completed
    click.echo(f"Recognized {len(results)} document(s): {completed} completed, {failed} failed.")
    if failed:
        raise click.ClickException(f"{failed} document(s) failed.")

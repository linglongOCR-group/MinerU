# mineru-phase CLI Redesign Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `mineru-phase` so layout artifacts are first-class inputs/outputs and each subcommand exposes only phase-native CLI arguments.

**Architecture:** Keep the existing document/window OCR pipeline in `mineru/cli/document_ocr.py`, but decouple layout artifact read/write paths from `DocumentJob.parse_dir` and `output_dir`. Rebuild `mineru/cli/phase.py` around explicit `--source`, `--layout-input`, `--layout-output`, and `--output` arguments; legacy aliases are left out of this refactor and can be added later if needed.

**Tech Stack:** Python, Click, dataclasses, pytest, Click `CliRunner`, existing MinerU phase tests.

---

## File Structure

- Modify `mineru/cli/phase.py`
  - Owns user-facing `mineru-phase` command groups and option validation.
  - Should define smaller option decorator groups instead of one broad `_shared_options`.
  - Maps `--source` to `DocumentOcrOptions.input_path`.
  - Maps `--timeout` to `DocumentOcrOptions.per_window_timeout`.
  - Maps `--max-model-concurrency` to `DocumentOcrOptions.max_http_concurrency_per_window`.

- Modify `mineru/cli/document_ocr.py`
  - Owns phase execution and artifact lookup/write behavior.
  - Add `layout_input_path` and `layout_output_dir` to `DocumentOcrOptions`.
  - Add helper functions to resolve artifact paths from explicit layout roots/files.
  - Keep final OCR outputs under `output_dir`.
  - Write layout artifacts under `layout_output_dir` when provided.

- Modify `tests/unittest/test_phase_cli.py`
  - Tests user-facing argument parsing and illegal option rejection.
  - Should prefer new explicit long flags in all new tests.

- Modify `tests/unittest/test_phase_pipeline.py`
  - Tests artifact read/write behavior in document pipeline.
  - Add coverage for explicit layout artifact roots and direct artifact files.

- Modify `tests/unittest/test_phase_integration.py`
  - Tests end-to-end phase behavior at pipeline level.
  - Add repeated recognition using one layout root and separate recognition output roots.

---

## Task 1: Add Option Fields for Explicit Layout Artifact Paths

**Files:**
- Modify: `mineru/cli/document_ocr.py`
- Test: `tests/unittest/test_phase_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add this test near `test_document_ocr_options_has_phase_field` in `tests/unittest/test_phase_pipeline.py`:

```python
def test_document_ocr_options_accept_explicit_layout_paths(tmp_path):
    opts = document_ocr.DocumentOcrOptions(
        input_path=tmp_path / "input",
        output_dir=tmp_path / "recognition_out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
        layout_input_path=tmp_path / "layout_in",
        layout_output_dir=tmp_path / "layout_out",
        max_http_concurrency_per_window=3,
        per_window_timeout=42.0,
    )

    assert opts.layout_input_path == tmp_path / "layout_in"
    assert opts.layout_output_dir == tmp_path / "layout_out"
    assert opts.max_http_concurrency_per_window == 3
    assert opts.per_window_timeout == 42.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/unittest/test_phase_pipeline.py::test_document_ocr_options_accept_explicit_layout_paths -q
```

Expected: FAIL with `TypeError: DocumentOcrOptions.__init__() got an unexpected keyword argument 'layout_input_path'`.

- [ ] **Step 3: Add minimal dataclass fields**

In `mineru/cli/document_ocr.py`, update `DocumentOcrOptions`:

```python
@dataclass(frozen=True)
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
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
python -m pytest tests/unittest/test_phase_pipeline.py::test_document_ocr_options_accept_explicit_layout_paths -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mineru/cli/document_ocr.py tests/unittest/test_phase_pipeline.py
git commit -m "refactor: add explicit layout artifact options"
```

---

## Task 2: Resolve Layout Artifacts from Explicit Root or File

**Files:**
- Modify: `mineru/cli/document_ocr.py`
- Test: `tests/unittest/test_phase_pipeline.py`

- [ ] **Step 1: Write failing tests for artifact path resolution**

Add these tests near `test_process_recognition_window_uses_layout_artifact_without_layout_cache`:

```python
def test_layout_artifact_path_uses_layout_input_root(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    job = document_ocr.DocumentJob.from_path(
        image_path,
        "image",
        "page",
        tmp_path / "recognition_out" / "page" / "vlm",
        0,
        1,
        0,
        0,
    )
    options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "recognition_out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
        layout_input_path=tmp_path / "layout_root",
    )

    path = document_ocr.resolve_layout_artifact_path(job, options)

    assert path == tmp_path / "layout_root" / "page" / "vlm" / "page_layout.json"


def test_layout_artifact_path_accepts_direct_file_for_single_document(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    artifact_path = tmp_path / "page_layout.json"
    artifact_path.write_text("{}", encoding="utf-8")
    job = document_ocr.DocumentJob.from_path(
        image_path,
        "image",
        "page",
        tmp_path / "recognition_out" / "page" / "vlm",
        0,
        1,
        0,
        0,
    )
    options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "recognition_out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
        layout_input_path=artifact_path,
    )

    path = document_ocr.resolve_layout_artifact_path(job, options)

    assert path == artifact_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest \
  tests/unittest/test_phase_pipeline.py::test_layout_artifact_path_uses_layout_input_root \
  tests/unittest/test_phase_pipeline.py::test_layout_artifact_path_accepts_direct_file_for_single_document \
  -q
```

Expected: FAIL with `AttributeError: module 'mineru.cli.document_ocr' has no attribute 'resolve_layout_artifact_path'`.

- [ ] **Step 3: Implement artifact path helpers**

In `mineru/cli/document_ocr.py`, replace the current `document_layout_artifact_path` helper with these helpers:

```python
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
```

Update `read_layout_window_cache_from_artifact` signature and first lines:

```python
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
```

Update `read_valid_layout_window_cache_or_artifact` signature and artifact read:

```python
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
```

Update callers in `process_recognition_window` and `run_document_ocr`:

```python
layout_cache = read_valid_layout_window_cache_or_artifact(
    window,
    options=options,
    rehydrate=True,
)
```

```python
if read_valid_layout_window_cache_or_artifact(window, options=options, rehydrate=True) is not None:
    return
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest \
  tests/unittest/test_phase_pipeline.py::test_layout_artifact_path_uses_layout_input_root \
  tests/unittest/test_phase_pipeline.py::test_layout_artifact_path_accepts_direct_file_for_single_document \
  tests/unittest/test_phase_pipeline.py::test_process_recognition_window_uses_layout_artifact_without_layout_cache \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mineru/cli/document_ocr.py tests/unittest/test_phase_pipeline.py
git commit -m "refactor: resolve layout artifacts explicitly"
```

---

## Task 3: Validate Direct Artifact File Usage and Required Layout Input

**Files:**
- Modify: `mineru/cli/document_ocr.py`
- Test: `tests/unittest/test_phase_pipeline.py`

- [ ] **Step 1: Write failing validation tests**

Add these tests near the artifact path tests:

```python
def test_recognize_requires_layout_input_path(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path)
    options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "recognition_out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
    )

    try:
        document_ocr.validate_phase_options(options, [image_path])
    except Exception as exc:
        assert "--layout-input is required for recognize" in str(exc)
    else:
        raise AssertionError("Expected recognize without layout_input_path to fail")


def test_direct_layout_artifact_rejects_multiple_documents(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _make_image(image_a)
    _make_image(image_b)
    artifact_path = tmp_path / "layout.json"
    artifact_path.write_text("{}", encoding="utf-8")
    options = document_ocr.DocumentOcrOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "recognition_out",
        phase=document_ocr.OcrPhase.RECOGNIZE,
        layout_input_path=artifact_path,
    )

    try:
        document_ocr.validate_phase_options(options, [image_a, image_b])
    except Exception as exc:
        assert "direct --layout-input artifact can only be used with one source document" in str(exc)
    else:
        raise AssertionError("Expected direct layout artifact with multiple documents to fail")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest \
  tests/unittest/test_phase_pipeline.py::test_recognize_requires_layout_input_path \
  tests/unittest/test_phase_pipeline.py::test_direct_layout_artifact_rejects_multiple_documents \
  -q
```

Expected: FAIL with `AttributeError: module 'mineru.cli.document_ocr' has no attribute 'validate_phase_options'`.

- [ ] **Step 3: Implement validation helper**

In `mineru/cli/document_ocr.py`, add this helper after `_discover_documents`:

```python
def validate_phase_options(options: DocumentOcrOptions, source_paths: list[Path]) -> None:
    if options.phase == OcrPhase.RECOGNIZE:
        if options.layout_input_path is None:
            raise click.ClickException("--layout-input is required for recognize")
        layout_input_path = Path(options.layout_input_path)
        if _is_direct_layout_artifact(layout_input_path) and len(source_paths) != 1:
            raise click.ClickException(
                "direct --layout-input artifact can only be used with one source document"
            )
    if options.phase == OcrPhase.LAYOUT and options.layout_output_dir is None:
        raise click.ClickException("--layout-output is required for layout")
```

Modify `collect_document_jobs` to accept resolved source paths, so validation can run before job construction:

```python
def collect_document_jobs(
    input_path: Path,
    output_dir: Path,
    *,
    start_page_id: int,
    end_page_id: int | None,
    resume: bool = False,
    source_paths: list[Path] | None = None,
) -> list[DocumentJob]:
    paths = source_paths if source_paths is not None else _discover_documents(input_path)
```

Modify `run_document_ocr` before `collect_document_jobs`:

```python
    source_paths = _discover_documents(options.input_path)
    validate_phase_options(options, source_paths)
    jobs = collect_document_jobs(
        options.input_path,
        options.output_dir,
        start_page_id=options.start_page_id,
        end_page_id=options.end_page_id,
        resume=options.resume,
        source_paths=source_paths,
    )
```

Keep the existing empty-source behavior in `collect_document_jobs`.

- [ ] **Step 4: Run validation tests**

Run:

```bash
python -m pytest \
  tests/unittest/test_phase_pipeline.py::test_recognize_requires_layout_input_path \
  tests/unittest/test_phase_pipeline.py::test_direct_layout_artifact_rejects_multiple_documents \
  -q
```

Expected: PASS.

- [ ] **Step 5: Run existing phase pipeline tests**

Run:

```bash
python -m pytest tests/unittest/test_phase_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mineru/cli/document_ocr.py tests/unittest/test_phase_pipeline.py
git commit -m "fix: require explicit layout input for recognition"
```

---

## Task 4: Write Layout Artifacts to Explicit Layout Output Root

**Files:**
- Modify: `mineru/cli/document_ocr.py`
- Test: `tests/unittest/test_phase_pipeline.py`

- [ ] **Step 1: Write failing tests for explicit layout output**

Add these tests near `test_run_layout_phase_writes_layout_artifact`:

```python
def test_layout_phase_writes_artifact_to_layout_output_dir(monkeypatch, tmp_path):
    image_path = tmp_path / "doc.png"
    _make_image(image_path)

    async def fake_process_layout_window(client, window, options):
        document_ocr.write_layout_window_cache(
            window,
            blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1]}]],
            page_sizes=[[16, 12]],
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(document_ocr, "process_layout_window", fake_process_layout_window)

    options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "unused_layout_work",
        phase=document_ocr.OcrPhase.LAYOUT,
        layout_output_dir=tmp_path / "layout_artifacts",
        progress=False,
    )

    results = asyncio.run(document_ocr.run_document_ocr(options, client_factory=lambda _opts: object()))

    assert results[0].status == "completed"
    assert (tmp_path / "layout_artifacts" / "doc" / "vlm" / "doc_layout.json").exists()
    assert not (tmp_path / "unused_layout_work" / "doc" / "vlm" / "doc_layout.json").exists()


def test_run_phase_writes_layout_artifact_to_default_nested_dir(monkeypatch, tmp_path):
    image_path = tmp_path / "doc.png"
    _make_image(image_path)

    async def fake_process_full_window(client, window, options):
        document_ocr.write_layout_window_cache(
            window,
            blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1]}]],
            page_sizes=[[16, 12]],
            elapsed_seconds=0.1,
        )
        document_ocr.write_window_cache(
            window,
            blocks_by_page=[[{"type": "text", "bbox": [0, 0, 1, 1], "content": "hello"}]],
            page_sizes=[[16, 12]],
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(document_ocr, "process_full_window", fake_process_full_window)

    options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "full_out",
        phase=document_ocr.OcrPhase.FULL,
        progress=False,
    )

    results = asyncio.run(document_ocr.run_document_ocr(options, client_factory=lambda _opts: object()))

    assert results[0].status == "completed"
    assert (tmp_path / "full_out" / "_layout_artifacts" / "doc" / "vlm" / "doc_layout.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest \
  tests/unittest/test_phase_pipeline.py::test_layout_phase_writes_artifact_to_layout_output_dir \
  tests/unittest/test_phase_pipeline.py::test_run_phase_writes_layout_artifact_to_default_nested_dir \
  -q
```

Expected: FAIL because artifacts are still written under `job.parse_dir`.

- [ ] **Step 3: Implement explicit artifact output path**

In `mineru/cli/document_ocr.py`, add:

```python
def resolve_layout_output_dir(options: DocumentOcrOptions) -> Path:
    if options.layout_output_dir is not None:
        return Path(options.layout_output_dir)
    if options.phase == OcrPhase.FULL:
        return options.output_dir / "_layout_artifacts"
    return options.output_dir
```

Change `write_document_layout_artifact` signature:

```python
def write_document_layout_artifact(
    job: DocumentJob,
    windows: list[WindowJob],
    *,
    layout_output_dir: Path | None = None,
) -> None:
```

Change its final write:

```python
    write_layout_artifact(
        artifact,
        document_layout_artifact_path(job, layout_output_dir),
    )
```

Update both `run_document_ocr` call sites:

```python
write_document_layout_artifact(
    job,
    document_windows,
    layout_output_dir=resolve_layout_output_dir(options),
)
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest \
  tests/unittest/test_phase_pipeline.py::test_layout_phase_writes_artifact_to_layout_output_dir \
  tests/unittest/test_phase_pipeline.py::test_run_phase_writes_layout_artifact_to_default_nested_dir \
  -q
```

Expected: PASS.

- [ ] **Step 5: Run phase pipeline suite**

Run:

```bash
python -m pytest tests/unittest/test_phase_pipeline.py -q
```

Expected: PASS.

Update existing artifact path assertions to the new expected default:

```python
tmp_path / "out" / "_layout_artifacts" / "doc" / "vlm" / "doc_layout.json"
```

for `FULL` phase tests, and to `layout_output_dir` for `LAYOUT` phase tests.

- [ ] **Step 6: Commit**

```bash
git add mineru/cli/document_ocr.py tests/unittest/test_phase_pipeline.py
git commit -m "refactor: separate layout artifact outputs"
```

---

## Task 5: Refactor mineru-phase CLI Option Groups

**Files:**
- Modify: `mineru/cli/phase.py`
- Test: `tests/unittest/test_phase_cli.py`

- [ ] **Step 1: Replace CLI tests with first-principles argument tests**

In `tests/unittest/test_phase_cli.py`, update the parsing tests to the new vocabulary:

```python
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
    captured = {}

    async def fake_run(options):
        captured.update(options.__dict__)
        return []

    monkeypatch.setattr(phase, "run_document_ocr", fake_run)

    result = CliRunner().invoke(phase.main, [
        "recognize",
        "--source", str(image),
        "--layout-input", str(tmp_path / "layout_artifacts"),
        "--output", str(tmp_path / "recognition_out"),
        "--recognition-url", "http://recognition",
        "--max-model-concurrency", "3",
        "--timeout", "13",
    ])

    assert result.exit_code == 0, result.output
    assert captured["phase"].value == "recognize"
    assert captured["input_path"] == image
    assert captured["layout_input_path"] == tmp_path / "layout_artifacts"
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
```

Add help-surface tests:

```python
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
```

- [ ] **Step 2: Run CLI tests to verify failures**

Run:

```bash
python -m pytest tests/unittest/test_phase_cli.py -q
```

Expected: FAIL because `--source`, `--layout-input`, `--layout-output`, `--timeout`, and `--max-model-concurrency` do not exist yet and broad shared options still leak across subcommands.

- [ ] **Step 3: Replace broad option decorators in `phase.py`**

Refactor `mineru/cli/phase.py` so the option groups look like this:

```python
def _source_option(func):
    return click.option(
        "--source",
        "input_path",
        required=True,
        type=click.Path(exists=True),
        help="Source document image, PDF, or directory.",
    )(func)


def _output_option(func):
    return click.option(
        "--output",
        "output_dir",
        required=True,
        type=click.Path(),
        help="Output directory for products produced by this invocation.",
    )(func)


def _layout_output_option(required: bool):
    def decorator(func):
        return click.option(
            "--layout-output",
            "layout_output_dir",
            required=required,
            type=click.Path(),
            help="Directory for produced layout artifacts.",
        )(func)
    return decorator


def _layout_input_option(func):
    return click.option(
        "--layout-input",
        "layout_input_path",
        required=True,
        type=click.Path(exists=True),
        help="Layout artifact root, or one layout artifact JSON for a single source document.",
    )(func)


def _execution_options(include_page_window_size: bool):
    def decorator(func):
        func = click.option("--progress/--no-progress", default=True, show_default=True, help="Show a progress bar.")(func)
        func = click.option("-e", "--end", "end_page_id", default=None, type=int, help="Ending PDF page, beginning from 0.")(func)
        func = click.option("-s", "--start", "start_page_id", default=0, show_default=True, type=int, help="Starting PDF page, beginning from 0.")(func)
        func = click.option("--retry-backoff-factor", default=0.5, show_default=True, type=float, help="HTTP retry backoff factor.")(func)
        func = click.option("--max-retries", default=3, show_default=True, type=int, help="HTTP retry count.")(func)
        func = click.option("--connect-timeout", default=10, show_default=True, type=int, help="HTTP connect timeout.")(func)
        func = click.option("--http-timeout", default=600, show_default=True, type=int, help="HTTP read timeout.")(func)
        func = click.option("--timeout", "per_window_timeout", default=600.0, show_default=True, type=float, help="Phase window timeout in seconds.")(func)
        func = click.option("--max-model-concurrency", "max_http_concurrency_per_window", default=1, show_default=True, type=int, help="Maximum in-flight model requests.")(func)
        func = click.option("--max-windows", default=16, show_default=True, type=int, help="Maximum in-flight document windows.")(func)
        if include_page_window_size:
            func = click.option("--page-window-size", default=4, show_default=True, type=int, help="PDF pages per scheduled layout window.")(func)
        return func
    return decorator


def _recognition_options(func):
    func = click.option("--image-analysis/--no-image-analysis", default=True, show_default=True, help="Recognize standalone image/chart blocks.")(func)
    func = click.option("--table/--no-table", "table_enable", default=True, show_default=True, help="Render table HTML content.")(func)
    func = click.option("--formula/--no-formula", "formula_enable", default=True, show_default=True, help="Render formula content.")(func)
    return func
```

Keep `_layout_url_option`, `_recognition_url_option`, and `_dissection_options`, but remove `_shared_options`.

Update `_build_options`:

```python
def _build_options(phase, **kwargs):
    input_path = _path_value(kwargs.pop("input_path"))
    output_dir = _path_value(kwargs.pop("output_dir"))
    layout_input_path = _path_value(kwargs.pop("layout_input_path", None))
    layout_output_dir = _path_value(kwargs.pop("layout_output_dir", None))
    return DocumentOcrOptions(
        input_path=input_path,
        output_dir=output_dir,
        phase=phase,
        layout_input_path=layout_input_path,
        layout_output_dir=layout_output_dir,
        **kwargs,
    )
```

For `layout`, build `output_dir` from `layout_output_dir` because the lower-level pipeline still needs an output root:

```python
def _build_layout_options(**kwargs):
    layout_output_dir = _path_value(kwargs.get("layout_output_dir"))
    kwargs["output_dir"] = layout_output_dir
    return _build_options(OcrPhase.LAYOUT, **kwargs)
```

Decorate commands like this:

```python
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
    ...


@main.command()
@_source_option
@_layout_output_option(required=True)
@_layout_url_option
@_execution_options(include_page_window_size=True)
@click.option("--dissection/--no-dissection", "dissection_enable", default=False, show_default=True, help="Write layout dissection artifacts.")
def layout(**kwargs):
    options = _build_layout_options(**kwargs)
    ...


@main.command()
@_source_option
@_layout_input_option
@_output_option
@_recognition_url_option
@_execution_options(include_page_window_size=False)
@_recognition_options
@_dissection_options
def recognize(**kwargs):
    ...
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
python -m pytest tests/unittest/test_phase_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mineru/cli/phase.py tests/unittest/test_phase_cli.py
git commit -m "refactor: redesign mineru-phase arguments"
```

---

## Task 6: Add Recognition-from-Layout Integration Coverage

**Files:**
- Modify: `tests/unittest/test_phase_integration.py`

- [ ] **Step 1: Write repeated-recognition integration test**

Add this test to `tests/unittest/test_phase_integration.py`:

```python
def test_repeated_recognition_uses_one_layout_root_and_separate_outputs(tmp_path):
    image_path = tmp_path / "doc.png"
    Image.new("RGB", (16, 12), (255, 255, 255)).save(image_path)
    calls = []

    class FakeClient:
        async def aio_layout_detect(self, image, priority=None, semaphore=None, scored=None):
            calls.append("layout")
            return ExtractResult([
                ContentBlock("text", [0, 0, 1, 1]),
            ])

        async def aio_recognize_from_layout(self, image, layout_blocks, **kwargs):
            calls.append("recognize")
            suffix = len([call for call in calls if call == "recognize"])
            for block in layout_blocks:
                block.content = f"recognized-{suffix}"
            return layout_blocks

    layout_options = document_ocr.DocumentOcrOptions(
        input_path=image_path,
        output_dir=tmp_path / "layout_work",
        phase=document_ocr.OcrPhase.LAYOUT,
        layout_output_dir=tmp_path / "layout_artifacts",
        progress=False,
    )
    asyncio.run(document_ocr.run_document_ocr(layout_options, client_factory=lambda _opts: FakeClient()))

    for output_name in ["rec_a", "rec_b"]:
        recognize_options = document_ocr.DocumentOcrOptions(
            input_path=image_path,
            output_dir=tmp_path / output_name,
            phase=document_ocr.OcrPhase.RECOGNIZE,
            layout_input_path=tmp_path / "layout_artifacts",
            progress=False,
        )
        result = asyncio.run(document_ocr.run_document_ocr(recognize_options, client_factory=lambda _opts: FakeClient()))
        assert result[0].status == "completed"

    assert calls.count("layout") == 1
    assert calls.count("recognize") == 2
    assert (tmp_path / "layout_artifacts" / "doc" / "vlm" / "doc_layout.json").exists()
    assert (tmp_path / "rec_a" / "doc" / "vlm" / "doc_model.json").exists()
    assert (tmp_path / "rec_b" / "doc" / "vlm" / "doc_model.json").exists()
    assert not (tmp_path / "rec_a" / "doc" / "vlm" / "doc_layout.json").exists()
    assert not (tmp_path / "rec_b" / "doc" / "vlm" / "doc_layout.json").exists()
```

- [ ] **Step 2: Run the integration test**

Run:

```bash
python -m pytest tests/unittest/test_phase_integration.py::test_repeated_recognition_uses_one_layout_root_and_separate_outputs -q
```

Expected: PASS.

- [ ] **Step 3: Run phase integration suite**

Run:

```bash
python -m pytest tests/unittest/test_phase_integration.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/unittest/test_phase_integration.py mineru/cli/document_ocr.py
git commit -m "test: cover repeated recognition from layout root"
```

---

## Task 7: Final Verification and Spec Coverage

**Files:**
- Verify: `docs/superpowers/specs/2026-06-03-mineru-phase-cli-redesign-design.md`
- Verify: `mineru/cli/phase.py`
- Verify: `mineru/cli/document_ocr.py`
- Verify: `tests/unittest/test_phase_cli.py`
- Verify: `tests/unittest/test_phase_pipeline.py`
- Verify: `tests/unittest/test_phase_integration.py`

- [ ] **Step 1: Run focused phase test suite**

Run:

```bash
python -m pytest \
  tests/unittest/test_phase_cli.py \
  tests/unittest/test_phase_pipeline.py \
  tests/unittest/test_phase_integration.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run neighboring document CLI regression tests**

Run:

```bash
python -m pytest tests/unittest/test_document_ocr_cli.py tests/unittest/test_image_ocr_cli.py -q
```

Expected: PASS.

- [ ] **Step 3: Check help text manually**

Run:

```bash
python - <<'PY'
from click.testing import CliRunner
from mineru.cli.phase import main

runner = CliRunner()
for command in ("layout", "recognize", "run"):
    result = runner.invoke(main, [command, "--help"])
    print(f"===== {command} =====")
    print(result.output)
    assert result.exit_code == 0
PY
```

Expected:

- `layout` help contains `--source`, `--layout-output`, `--layout-url`.
- `layout` help does not contain `--recognition-url`, `--formula`, `--table`, `--image-analysis`, or `--stream`.
- `recognize` help contains `--source`, `--layout-input`, `--output`, `--recognition-url`.
- `recognize` help does not contain `--layout-url`, `--layout-output`, or `--page-window-size`.
- `run` help contains `--source`, `--output`, optional `--layout-output`, `--layout-url`, and `--recognition-url`.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff --stat HEAD
git diff -- mineru/cli/phase.py mineru/cli/document_ocr.py tests/unittest/test_phase_cli.py tests/unittest/test_phase_pipeline.py tests/unittest/test_phase_integration.py
```

Expected: only intentional refactor and tests are present.

- [ ] **Step 5: Confirm no uncommitted intended changes remain**

Run:

```bash
git status --short
```

Expected: no modified tracked files. Unrelated pre-existing untracked files, such as `.coverage` or `.superpowers/`, may remain untouched.

---

## Self-Review

### Spec coverage

- Purpose and phase-native inputs: Tasks 5 and 7.
- Artifact inputs are not outputs: Tasks 2, 4, 5, and 6.
- Stable layout artifacts as first-class objects: Tasks 2, 4, and 6.
- Source validation and fail-fast recognition: Tasks 2 and 3.
- `layout` argument list and exclusions: Task 5.
- `recognize` argument list and exclusions: Tasks 3 and 5.
- `run` argument list and default `_layout_artifacts` behavior: Tasks 4 and 5.
- Execution controls renaming: Task 5.
- Resume phase-local behavior: covered by existing phase resume tests and Task 6 final suite.
- Error handling: Tasks 3, 5, and 7.
- Migration aliases: not implemented in this refactor; the spec allows them as optional future compatibility support, while this plan implements the explicit first-principles CLI.
- Test coverage requirements: Tasks 5, 6, and 7.

### Placeholder scan

The plan contains no `TBD`, no open-ended implementation placeholders, and no "similar to previous" references. Each implementation task includes exact files, test code, commands, expected failures, and commit commands.

### Type consistency

The plan consistently uses:

- `DocumentOcrOptions.layout_input_path`
- `DocumentOcrOptions.layout_output_dir`
- `resolve_layout_artifact_path(job, options)`
- `resolve_layout_output_dir(options)`
- `--source` mapped to `input_path`
- `--timeout` mapped to `per_window_timeout`
- `--max-model-concurrency` mapped to `max_http_concurrency_per_window`

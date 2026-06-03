# mineru-phase CLI Redesign

**Date:** 2026-06-03

**Scope:** `mineru-phase` command surface in `MinerU`.

**Status:** Design proposal.

## 1. Purpose

`mineru-phase` is the command-line interface for running the VLM OCR phase graph over stable artifacts. It is not a compatibility wrapper for `mineru`, `mineru-ocr-images`, or `mineru-ocr-documents`.

The command exists to make the following workflow explicit:

1. Run layout detection once.
2. Persist reusable layout artifacts.
3. Run recognition one or more times against those fixed layout artifacts.
4. Write each recognition run to its own output directory.

This supports recognition-model experiments where layout is intentionally held constant.

## 2. Design Principles

### Phase-native inputs

Each subcommand exposes only the inputs needed by that phase. A layout-only command does not accept recognition model controls. A recognition-only command does not accept a layout endpoint.

### Artifact inputs are not outputs

`--output` always means "where this invocation writes its outputs." It must not also mean "where to find required upstream artifacts."

Recognition consumes layout artifacts through `--layout-input`, while writing recognition products through `--output`.

### Stable layout artifacts are first-class

Layout artifacts are reusable pipeline inputs, not hidden cache files. They should be easy to store, name, validate, copy, and reference independently of OCR output runs.

### Source validation remains mandatory

Recognition still receives the original source documents. It validates layout artifacts against the source fingerprint before running recognition. On mismatch, it fails fast and does not silently rerun layout.

## 3. Core Terms

- **Source:** Original image, PDF, or directory of documents.
- **Layout artifact:** Persisted layout detection result for a document.
- **Layout artifact root:** Directory containing per-document layout artifacts.
- **OCR output:** Final recognition products: `_model.json`, `_middle.json`, Markdown, content list, status, and optional dissection artifacts.
- **Run output:** Output directory for one command invocation.

## 4. Command Tree

```text
mineru-phase layout      # source -> layout artifacts
mineru-phase recognize   # source + layout artifacts -> OCR outputs
mineru-phase run         # source -> layout artifacts + OCR outputs
```

`layout` and `recognize` are the primary commands. `run` is a convenience command for full OCR.

## 5. Shared Argument Rules

### Naming

Prefer explicit phase vocabulary over legacy compatibility flags:

- Use `--source`, not `-p` or `--path`.
- Use `--layout-input` for consumed layout artifacts.
- Use `--layout-output` for produced layout artifacts.
- Use `--output` only for products produced by this invocation.

Short aliases are optional and should be added only when they do not obscure meaning.

### Source shape

`--source` accepts:

- a single image,
- a single PDF,
- a directory containing images and PDFs.

Directory traversal and stem disambiguation should match the document pipeline behavior so artifact lookup remains deterministic.

### Layout artifact shape

`--layout-input` and `--layout-output` support an artifact root by default:

```text
<layout-root>/<stem>/vlm/<stem>_layout.json
```

For a single-document source, `--layout-input` may also point directly to one layout artifact file:

```text
doc_layout.json
```

Direct artifact-file input is invalid when `--source` resolves to more than one document.

## 6. `mineru-phase layout`

### Purpose

Run layout detection only and write reusable layout artifacts.

### Arguments

```bash
mineru-phase layout \
  --source input_docs \
  --layout-output layout_artifacts \
  --layout-url http://layout:8000 \
  [--page-window-size 4] \
  [--max-windows 16] \
  [--max-model-concurrency 1] \
  [--timeout 600] \
  [--http-timeout 600] \
  [--connect-timeout 10] \
  [--max-retries 3] \
  [--retry-backoff-factor 0.5] \
  [--start 0] \
  [--end N] \
  [--dissection] \
  [--resume] \
  [--progress/--no-progress]
```

### Not accepted

`layout` does not accept:

- `--recognition-url`
- `--layout-input`
- `--output`
- `--formula` / `--no-formula`
- `--table` / `--no-table`
- `--image-analysis` / `--no-image-analysis`
- `--stream`

### Outputs

For each source document, write:

```text
<layout-output>/<stem>/vlm/<stem>_layout.json
<layout-output>/<stem>/vlm/<stem>_status.json
```

Optional layout dissection artifacts may be written under the layout output tree.

Layout-only mode does not write final Markdown, `_middle.json`, `_model.json`, or `_content_list.json`.

## 7. `mineru-phase recognize`

### Purpose

Run recognition using existing layout artifacts and write one OCR output run.

### Arguments

```bash
mineru-phase recognize \
  --source input_docs \
  --layout-input layout_artifacts \
  --output recognition_outputs/run_a \
  --recognition-url http://recognition:8000 \
  [--max-windows 16] \
  [--max-model-concurrency 1] \
  [--timeout 600] \
  [--http-timeout 600] \
  [--connect-timeout 10] \
  [--max-retries 3] \
  [--retry-backoff-factor 0.5] \
  [--start 0] \
  [--end N] \
  [--formula/--no-formula] \
  [--table/--no-table] \
  [--image-analysis/--no-image-analysis] \
  [--dissection] \
  [--stream] \
  [--resume] \
  [--progress/--no-progress]
```

### Not accepted

`recognize` does not accept:

- `--layout-url`
- `--layout-output`
- `--page-window-size`

Recognition windowing is derived from the layout artifact pages. It must not require the user to repeat the layout window size used during layout detection.

### Validation

Before recognition, the command must:

1. Resolve the source document set.
2. Resolve one layout artifact per source document.
3. Validate layout artifact schema.
4. Validate source fingerprint: path, size, mtime, page count, and requested page range.
5. Validate page sizes against rendered source pages.
6. Fail with a clear error if any artifact is missing or stale.

Recognition must not call layout detection or silently repair missing layout artifacts.

### Outputs

For each source document, write OCR products under:

```text
<output>/<stem>/vlm/
```

This includes final recognition outputs and recognition status. It does not rewrite the layout artifact unless a future implementation adds an explicit copy option.

### Example repeated-recognition workflow

```bash
mineru-phase layout \
  --source input_docs \
  --layout-output artifacts/layout_v1 \
  --layout-url http://layout:8000

mineru-phase recognize \
  --source input_docs \
  --layout-input artifacts/layout_v1 \
  --output outputs/recognition_model_a \
  --recognition-url http://recognition-a:8000

mineru-phase recognize \
  --source input_docs \
  --layout-input artifacts/layout_v1 \
  --output outputs/recognition_model_b \
  --recognition-url http://recognition-b:8000
```

## 8. `mineru-phase run`

### Purpose

Run layout, recognition, and assembly in one command. This is a convenience path for full OCR, not the primary experiment workflow.

### Arguments

```bash
mineru-phase run \
  --source input_docs \
  --output full_outputs \
  --layout-url http://layout:8000 \
  --recognition-url http://recognition:8000 \
  [--layout-output layout_artifacts] \
  [--page-window-size 4] \
  [--max-windows 16] \
  [--max-model-concurrency 1] \
  [--timeout 600] \
  [--http-timeout 600] \
  [--connect-timeout 10] \
  [--max-retries 3] \
  [--retry-backoff-factor 0.5] \
  [--start 0] \
  [--end N] \
  [--formula/--no-formula] \
  [--table/--no-table] \
  [--image-analysis/--no-image-analysis] \
  [--dissection] \
  [--stream] \
  [--resume] \
  [--progress/--no-progress]
```

### Layout artifact behavior

`run` always produces layout artifacts.

If `--layout-output` is provided, write layout artifacts there. If omitted, write layout artifacts under the run output tree:

```text
<output>/_layout_artifacts/<stem>/vlm/<stem>_layout.json
```

The nested default prevents the final OCR output root from being overloaded as the implicit input location for later recognition runs.

### Outputs

Write final OCR products under:

```text
<output>/<stem>/vlm/
```

Write layout artifacts under either `--layout-output` or the default `_layout_artifacts` directory.

## 9. Execution Controls

### `--page-window-size`

Accepted by `layout` and `run`.

It controls how many PDF pages are grouped into one scheduled layout window. It is not accepted by `recognize`, because recognition consumes the page grouping represented by the layout artifacts.

### `--max-windows`

Accepted by all commands.

It limits concurrently processed document windows for the active phase.

### `--max-model-concurrency`

Accepted by all commands.

It limits concurrent in-flight requests made by the MinerU model client. It replaces the older `--max-http-concurrency-per-window` name, which exposed an implementation detail and incorrectly tied the control to window scheduling.

### `--timeout`

Accepted by all commands.

It is the phase-window timeout for layout, recognition, or full run processing. It replaces the older `--per-window-timeout` name in the phase CLI.

### HTTP retry controls

Accepted by all commands that call a server:

- `--http-timeout`
- `--connect-timeout`
- `--max-retries`
- `--retry-backoff-factor`

### `--stream`

Accepted by `recognize` and `run` only.

`--stream` requires `--dissection`.

## 10. Resume Semantics

Resume behavior is phase-local.

- `layout --resume` skips completed layout artifacts.
- `recognize --resume` skips completed recognition outputs for the selected output directory.
- `run --resume` skips completed full OCR outputs, while still requiring valid layout artifacts when recognition work is skipped or resumed.

Resume must not infer success from artifacts written for another phase unless that phase explicitly owns them.

## 11. Error Handling

The command should fail fast for configuration errors:

- unsupported source path,
- missing required endpoint,
- missing layout input for recognition,
- direct layout artifact file used with multiple source documents,
- layout-only command given recognition-only options,
- recognition command given layout-only options.

Per-document processing failures should be recorded in each document status file. Batch behavior may continue processing other documents, but the command exits nonzero if any document fails.

## 12. Migration From Current CLI

The current `mineru-phase recognize -o <output>` behavior implicitly looks for layout artifacts inside `<output>`. That should be retired.

Migration path:

```bash
# Current coupled shape
mineru-phase recognize -p input_docs -o previous_layout_output --recognition-url ...

# New explicit shape
mineru-phase recognize \
  --source input_docs \
  --layout-input previous_layout_output \
  --output new_recognition_output \
  --recognition-url ...
```

Compatibility aliases such as `-p`, `-o`, and `--per-window-timeout` may be preserved temporarily, but help text should present the first-principles names first.

## 13. Test Coverage Requirements

Add or update tests for:

1. `layout --help` excludes recognition-only options.
2. `recognize --help` excludes layout-only options.
3. `recognize` requires `--layout-input`.
4. `recognize` accepts a layout artifact root and writes outputs to a separate `--output`.
5. `recognize` accepts a direct artifact file for one source document.
6. `recognize` rejects a direct artifact file for multiple source documents.
7. `recognize` fails on stale layout artifacts and does not call layout detection.
8. repeated recognition runs can share one layout artifact root and produce separate output trees.
9. `run` writes final outputs and layout artifacts to distinct locations.
10. `--stream` requires `--dissection` for `recognize` and `run`.

## 14. Additional Decisions

### Visual layout outputs

`layout` should write the stable JSON artifact by default. Visual layout images should require an explicit visualization flag in a future command surface, because the phase artifact contract should not depend on optional debug renderings.

### Provenance in recognition output

`recognize` should record the consumed layout artifact path and fingerprint in recognition status metadata. It should not copy layout artifacts into the recognition output by default. A future `--copy-layout-artifact` flag may be added if reproducibility workflows need self-contained output directories.

### Short aliases

The first redesign should expose explicit long flags in help text. Legacy aliases such as `-p`, `-o`, and `--per-window-timeout` may be accepted temporarily for migration, but they should be hidden or clearly marked as compatibility aliases.

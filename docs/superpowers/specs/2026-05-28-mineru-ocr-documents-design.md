# mineru-ocr-documents Design

## Context

`mineru-ocr-images` remains the quick image-only VLM OCR entry point. It avoids
`mineru-api` scheduling and gives operators a direct path for document images.

The new requirement is a unified VLM HTTP document entry point named
`mineru-ocr-documents`. It should support both document images and PDFs while
preserving document-level outputs. PDFs are rendered to page images internally,
but final output is one document-level result for the whole PDF.

The command is intended to avoid the performance bottlenecks caused by
`mineru-api` scheduling. It may reuse lower-level MinerU utilities for PDF
rendering, page metadata, middle-json conversion, and output rendering, but it
must own scheduling, cache, status, and finalization.

## Goals

- Keep `mineru-ocr-images` as an image-only quick path.
- Add `mineru-ocr-documents` for mixed image and PDF batches.
- Accept a file or recursive mixed directory input.
- Use only the VLM HTTP backend in the first version.
- Avoid `mineru-api` orchestration.
- Schedule image jobs and PDF page windows through one global scheduler.
- Interleave PDF page windows across documents.
- Preserve document-level final outputs for both images and PDFs.
- Keep PDF output atomic: if any PDF window fails, the document fails.
- Continue the whole batch after per-document failures.
- Support resumable PDF work through internal window caches.
- Support dissection and streaming options for debugging.

## Non-Goals

- Local VLM engine backend support.
- MinerU visualization outputs.
- Partial public final outputs for failed PDFs.
- A separate PDF-only command.
- Changing the existing `mineru-ocr-images` output contract.

## CLI Contract

`mineru-ocr-documents` accepts the same core VLM HTTP endpoint options as
`mineru-ocr-images`:

- `--url`
- `--layout-url`
- `--recognition-url`
- `--http-timeout`
- `--connect-timeout`
- `--max-retries`
- `--retry-backoff-factor`
- `--image-analysis / --no-image-analysis`
- `--formula / --no-formula`
- `--table / --no-table`
- `--dissection / --no-dissection`
- `--stream / --no-stream`
- `--progress / --no-progress`
- `--resume`

New or renamed document scheduling options:

- `--page-window-size`, default `4`
- `--max-windows`, default `16`
- `--max-http-concurrency-per-window`, default `1`
- `--per-window-timeout`, default `600.0`
- `--start`
- `--end`

`--start` and `--end` apply to PDFs and are ignored for image inputs.

## Architecture

The implementation should use small units with explicit boundaries:

- `DocumentJob`: one input image or PDF, with normalized output paths under
  `<output>/<stem>/vlm/`.
- `WindowJob`: the scheduler unit. An image contributes one window. A PDF
  contributes one or more page windows.
- `DocumentState`: tracks expected windows, completed cached windows, failed
  windows, document status, and finalization readiness.
- `WindowCache`: private per-window raw VLM outputs and metadata.
- `DocumentAssembler`: creates final document-level model, middle, markdown,
  content-list, and status outputs after all windows complete.

`mineru-ocr-images` should not be removed or converted. Shared helpers can be
extracted only where that reduces duplication without changing its behavior.

## Discovery And Output Layout

The command discovers supported image files and PDFs recursively when `--path`
points to a directory. Stems must be unique within the run, consistent with the
existing MinerU CLI behavior.

Every input document writes under:

```text
<output>/<document_stem>/vlm/
```

Final successful outputs:

```text
<stem>.md
<stem>_model.json
<stem>_middle.json
<stem>_content_list.json
<stem>_status.json
```

Images and PDFs use the same document-level output layout. Image documents are
represented as one-page documents.

## PDF Rendering And Windowing

PDF rendering should use existing MinerU defaults and utilities. The first
version should not expose PDF DPI or max-side options.

For PDFs:

1. Open the PDF and resolve page count.
2. Apply `--start` and `--end`.
3. Split the selected page range into windows of `--page-window-size`.
4. Schedule those windows globally with all other document windows.
5. Render only the pages needed for each active window.
6. Run VLM HTTP extraction for the rendered page images.
7. Cache raw blocks and window metadata.

The VLM extraction call must preserve page identity by passing the correct
page-start index or equivalent metadata so final assembly is independent of
completion order.

## Scheduling And Concurrency

The scheduler owns one global queue of `WindowJob`s across all documents.
Interleaving is required so large PDFs do not monopolize the batch. A
round-robin queue by document is sufficient for the first version.

Concurrency controls:

- `--max-windows` caps active windows across the whole batch.
- `--max-http-concurrency-per-window` is passed to
  `MinerUClient(max_concurrency=...)`.

`--per-window-timeout` wraps the full window operation: rendering, VLM HTTP
recognition, and cache write.

Completion order must not affect final outputs. Assembly must sort by document
window index and page index.

## Status, Resume, And Failure Handling

Public status lives at:

```text
<output>/<stem>/vlm/<stem>_status.json
```

It records:

- `status`: `completed`, `failed`, or `timeout`
- document type
- source path
- selected page range for PDFs
- elapsed time
- completed window count
- total window count
- error details when applicable

`--resume` skips only documents whose public status is `completed`.

Private window cache lives under:

```text
<output>/<stem>/vlm/.windows/
```

Each completed window writes raw VLM blocks and metadata atomically. Metadata
must include enough information to reject stale cache entries, including source
path, source size, source mtime, document type, selected page range, page count,
window index, page range, and page sizes.

Failure behavior:

- A window timeout or exception marks its parent document failed.
- Pending windows for a failed document are skipped or cancelled.
- Other documents continue.
- Cached successful windows are retained for future resume.
- Failed documents do not write final `.md`, `_model.json`, `_middle.json`, or
  `_content_list.json`.
- The command finishes the batch, prints a summary, and exits nonzero if any
  document failed.

## Assembly

The assembler runs only after all expected windows for a document are complete.

For PDFs, `_model.json` is one combined document-level model output, ordered by
page and preserving page identity. `_middle.json`, Markdown, and content list
are produced in one deterministic pass using existing MinerU middle-json and
content rendering utilities where practical.

For images, the same assembler path should be used with a single page/window.

## Testing Plan

Unit tests:

- Mixed recursive discovery finds PDFs and images and assigns unique stems.
- PDF page range and `--page-window-size=4` produce expected windows.
- Round-robin queue construction interleaves windows across documents.
- Completed public status skips a document under `--resume`.
- Cached window metadata is accepted only when it matches the current document.
- Failed windows mark the document failed and prevent final public outputs.
- Out-of-order successful windows assemble into deterministic final outputs.
- Image inputs through `mineru-ocr-documents` produce unified document status.

Integration or smoke tests:

- One image and one short PDF with a fake VLM client produce final outputs.
- Two PDFs with out-of-order window completion assemble in page order.
- One failed PDF window does not stop another document from completing.
- Resume from a partially cached PDF avoids re-running completed windows.

Validation after implementation:

- Existing `mineru-ocr-images` tests still pass.
- New `mineru-ocr-documents` tests pass.
- CLI help for both commands works.
- A small live VLM HTTP run can be used when endpoints are available.

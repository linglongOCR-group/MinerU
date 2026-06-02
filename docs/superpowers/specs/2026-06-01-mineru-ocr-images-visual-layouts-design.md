# MinerU OCR Images Visual Layouts Design

## Context

`mineru-ocr-images` currently writes `<stem>_layout.png` as a visual overlay for
successful image OCR jobs. The current overlay is generated from post-processed
`middle_json` layout data, and each bbox is marked with a numeric badge.

This hides two useful debugging signals:

- the original VLM model layout before MinerU post-processing;
- the semantic category associated with each bbox.

## Goals

- Preserve the existing post-processed visual layout output.
- Add a raw model layout visualization for direct inspection of `_model.json`.
- Place category labels directly on bbox boundaries.
- Use raw model `type` and raw model `bbox` for the raw model visualization.
- Keep image OCR core output files and success status behavior unchanged.

## Non-Goals

- Change Markdown, content-list, `_model.json`, or `_middle.json` generation.
- Add a CLI flag for choosing a visualization mode.
- Change document OCR or PDF visualization behavior.
- Rework MinerU layout post-processing.

## Output Contract

Successful `mineru-ocr-images` jobs will write two visual layout PNGs:

- `<stem>_model_layout.png`
- `<stem>_layout.png`

`<stem>_model_layout.png` visualizes raw model output:

- source data: the same raw `blocks` written to `<stem>_model.json`;
- bbox source: each raw block `bbox`;
- label source: each raw block `type`, exactly as returned by the model;
- ordering: the raw block list order passed to `write_success_outputs`, matching
  the serialized `_model.json` order.

`<stem>_layout.png` remains the post-processed layout visualization:

- source data: `<stem>_middle.json` / in-memory `middle_json`;
- bbox source: post-processed layout blocks;
- label source: post-processed block type;
- filename remains unchanged for compatibility.

## Rendering Behavior

Both visualizations will use the original source image as the canvas.

For each valid bbox:

1. Coerce the bbox into pixel coordinates.
2. If all bbox coordinates are in normalized range `[0, 1]`, scale by image
   width and height.
3. Clamp coordinates to the image bounds.
4. Skip invalid or empty bboxes.
5. Draw a translucent colored rectangle with a visible outline.
6. Draw a small solid label attached to the bbox boundary.

Preferred label placement:

- just above the top-left bbox boundary when there is enough space;
- otherwise inside the top-left corner of the bbox.

Colors should reuse the existing category color mapping where possible. Unknown
types should fall back to the current red default.

## Data Flow

`process_image_job` already receives raw model blocks, builds `middle_json`, then
writes success outputs.

The visual output writer should receive both data sources:

- `blocks` for raw model visualization;
- `middle_json` for post-processed visualization.

The writer should copy the origin image once, then render both layout images from
the same loaded source image.

## Error Handling

Visualization should stay tolerant:

- malformed blocks are skipped rather than failing the OCR job;
- missing or invalid bboxes are skipped;
- unknown types use fallback color and raw string label;
- if the source image cannot be opened or copied, the existing output-generation
  failure behavior can remain unchanged.

## Testing

Unit tests should cover:

- success output writes `<stem>_model_layout.png` and `<stem>_layout.png`;
- raw model visualization uses raw `blocks`, not `middle_json`;
- post-processed visualization still uses `middle_json`;
- raw labels are based on raw block `type`;
- invalid raw bboxes are skipped without preventing valid boxes from drawing.

The most important regression test should pass intentionally different raw and
post-processed bboxes, then assert pixel changes appear at the expected raw-only
and middle-only regions in the two output images.

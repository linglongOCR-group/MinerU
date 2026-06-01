# MinerU OCR Images Visual Layouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a raw model layout PNG for `mineru-ocr-images` while preserving the existing post-processed layout PNG, with bbox category labels drawn on bbox boundaries.

**Architecture:** Keep all image-only visualization logic in `mineru/cli/image_ocr.py`. Split annotation extraction from rendering so raw model blocks and post-processed middle-json blocks share one renderer but keep separate data-source iterators and output filenames.

**Tech Stack:** Python, Pillow, Click CLI, pytest.

---

## File Structure

- Modify `mineru/cli/image_ocr.py`
  - Add `_model_layout_image_path(job)`.
  - Add `_iter_model_blocks(blocks, image_size=...)`.
  - Replace numeric badge drawing with a category label helper.
  - Add shared `render_annotated_layout_image(source_image, annotations)`.
  - Keep `render_layout_image(source_image, middle_json)` for post-processed layout.
  - Add `render_model_layout_image(source_image, blocks)` for raw model layout.
  - Change `write_image_visual_outputs(job, blocks, middle_json)` to write both PNGs.
  - Change `write_success_outputs(...)` to pass `blocks` into the visual writer.

- Modify `tests/unittest/test_image_ocr_cli.py`
  - Extend the existing visual-output test to assert both files are written.
  - Add a regression test proving raw model and middle-json bboxes are rendered into different files.
  - Add a focused label-position test for the raw model renderer.

## Task 1: Add Failing Visual Output Tests

**Files:**
- Modify: `tests/unittest/test_image_ocr_cli.py`
- Test: `tests/unittest/test_image_ocr_cli.py`

- [ ] **Step 1: Add a helper for comparing changed pixels**

Add this helper after `_make_image`:

```python
def _pixel_is_changed(path: Path, xy: tuple[int, int]) -> bool:
    with Image.open(path) as image:
        return image.convert("RGB").getpixel(xy) != (255, 255, 255)
```

- [ ] **Step 2: Update the existing visual-output test to expect the raw model layout file**

In `test_write_success_outputs_writes_origin_and_layout_images`, add:

```python
    model_layout_path = job.parse_dir / "page_model_layout.png"
```

Then add this assertion near the existing `layout_path.is_file()` assertion:

```python
    assert model_layout_path.is_file()
```

Expected result before implementation: this test fails because `page_model_layout.png` is not written.

- [ ] **Step 3: Add a regression test for raw versus post-processed layout sources**

Add this test near the existing visual-output test:

```python
def test_visual_outputs_separate_raw_model_and_postprocessed_layout(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path, size=(100, 80))
    job = image_ocr.ImageJob(image_path, "page", tmp_path / "out" / "page" / "vlm")
    blocks = [
        {"type": "title", "bbox": [10, 10, 30, 30], "content": "raw title"},
        {"type": "text", "bbox": [0.4, 0.1, 0.6, 0.3], "content": "raw text"},
        {"type": "broken", "bbox": [70, 20, 60, 30], "content": "skip me"},
    ]
    middle_json = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [100, 80],
                "para_blocks": [
                    {"type": "table", "bbox": [70, 45, 95, 70]},
                ],
            }
        ]
    }

    image_ocr.write_success_outputs(
        job,
        blocks=blocks,
        middle_json=middle_json,
        markdown="hello",
        content_list=[],
        elapsed_seconds=1.25,
    )

    model_layout_path = job.parse_dir / "page_model_layout.png"
    layout_path = job.parse_dir / "page_layout.png"

    assert _pixel_is_changed(model_layout_path, (12, 12))
    assert _pixel_is_changed(model_layout_path, (42, 12))
    assert not _pixel_is_changed(model_layout_path, (72, 47))

    assert _pixel_is_changed(layout_path, (72, 47))
    assert not _pixel_is_changed(layout_path, (12, 12))
    assert not _pixel_is_changed(layout_path, (42, 12))
```

Expected result before implementation: this test fails because `page_model_layout.png` is not written.

- [ ] **Step 4: Add a raw label boundary placement test**

Add this test after the source-separation test:

```python
def test_model_layout_draws_raw_type_label_on_bbox_boundary(tmp_path):
    image_path = tmp_path / "page.png"
    _make_image(image_path, size=(120, 90))
    with Image.open(image_path) as source_image:
        rendered = image_ocr.render_model_layout_image(
            source_image,
            [{"type": "raw_custom_type", "bbox": [20, 25, 80, 60], "content": "hello"}],
        )

    rendered_path = tmp_path / "rendered.png"
    rendered.save(rendered_path)

    assert _pixel_is_changed(rendered_path, (22, 12))
    assert _pixel_is_changed(rendered_path, (22, 27))
```

Expected result before implementation: this test fails because `render_model_layout_image` does not exist.

- [ ] **Step 5: Run the focused tests and verify failure**

Run:

```bash
pytest tests/unittest/test_image_ocr_cli.py::test_write_success_outputs_writes_origin_and_layout_images tests/unittest/test_image_ocr_cli.py::test_visual_outputs_separate_raw_model_and_postprocessed_layout tests/unittest/test_image_ocr_cli.py::test_model_layout_draws_raw_type_label_on_bbox_boundary -q
```

Expected: FAIL, with missing `page_model_layout.png` and/or missing `render_model_layout_image`.

- [ ] **Step 6: Commit the failing tests**

```bash
git add tests/unittest/test_image_ocr_cli.py
git commit -m "test: cover image OCR raw visual layouts"
```

## Task 2: Implement Dual Layout Rendering

**Files:**
- Modify: `mineru/cli/image_ocr.py:116-243`
- Modify: `mineru/cli/image_ocr.py:387-403`
- Test: `tests/unittest/test_image_ocr_cli.py`

- [ ] **Step 1: Add the raw model layout path helper**

Add after `_layout_image_path`:

```python
def _model_layout_image_path(job: ImageJob) -> Path:
    return job.parse_dir / f"{job.stem}_model_layout.png"
```

- [ ] **Step 2: Add the raw model block iterator**

Add after `_iter_layout_blocks`:

```python
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
```

- [ ] **Step 3: Replace numeric badges with category labels**

Replace `_draw_badge` with:

```python
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
```

- [ ] **Step 4: Add the shared renderer and update public render helpers**

Replace `render_layout_image` with these functions:

```python
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
```

- [ ] **Step 5: Write both visual outputs**

Replace `write_image_visual_outputs` with:

```python
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
```

- [ ] **Step 6: Pass raw blocks from success output writing**

In `write_success_outputs`, replace:

```python
    write_image_visual_outputs(job, middle_json)
```

with:

```python
    write_image_visual_outputs(job, blocks=blocks, middle_json=middle_json)
```

- [ ] **Step 7: Run the focused tests**

Run:

```bash
pytest tests/unittest/test_image_ocr_cli.py::test_write_success_outputs_writes_origin_and_layout_images tests/unittest/test_image_ocr_cli.py::test_visual_outputs_separate_raw_model_and_postprocessed_layout tests/unittest/test_image_ocr_cli.py::test_model_layout_draws_raw_type_label_on_bbox_boundary -q
```

Expected: PASS.

- [ ] **Step 8: Run all image OCR CLI tests**

Run:

```bash
pytest tests/unittest/test_image_ocr_cli.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit implementation**

```bash
git add mineru/cli/image_ocr.py tests/unittest/test_image_ocr_cli.py
git commit -m "feat: add raw image OCR layout visualization"
```

## Task 3: Final Verification

**Files:**
- Verify: `mineru/cli/image_ocr.py`
- Verify: `tests/unittest/test_image_ocr_cli.py`
- Verify: `docs/superpowers/specs/2026-06-01-mineru-ocr-images-visual-layouts-design.md`

- [ ] **Step 1: Run the unit test file**

```bash
pytest tests/unittest/test_image_ocr_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Inspect output contract in code**

Run:

```bash
rg -n "_model_layout|_layout\\.png|write_image_visual_outputs|render_model_layout_image|render_layout_image" mineru/cli/image_ocr.py tests/unittest/test_image_ocr_cli.py
```

Expected: output shows `_model_layout.png` and `_layout.png`, plus both render helpers and tests that assert both files.

- [ ] **Step 3: Confirm only intended files changed**

Run:

```bash
git status --short
```

Expected after Task 2 commit: no modified `mineru/cli/image_ocr.py` or `tests/unittest/test_image_ocr_cli.py`. Existing unrelated local files such as `pyproject.toml` or `.superpowers/` may still appear and must not be included in implementation commits.

## Self-Review

- Spec coverage: raw model output, post-processed output, raw labels, raw bboxes, tolerant invalid bbox handling, and unchanged core OCR outputs are all covered by Tasks 1 and 2.
- Placeholder scan: no unresolved marker text or unspecified implementation steps remain.
- Type consistency: `render_model_layout_image`, `render_layout_image`, `write_image_visual_outputs`, `_iter_model_blocks`, and `_draw_label` signatures are consistent across tests and implementation steps.

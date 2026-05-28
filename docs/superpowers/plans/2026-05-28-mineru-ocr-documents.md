# mineru-ocr-documents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `mineru-ocr-documents`, a VLM HTTP OCR command that handles mixed image/PDF batches with interleaved PDF page-window scheduling and document-level outputs.

**Architecture:** Keep `mineru-ocr-images` intact. Add `mineru.cli.document_ocr` with document/window dataclasses, discovery, round-robin scheduling, private window cache, document status, assembly, and CLI parsing. Reuse existing image output helpers and MinerU PDF rendering/middle-json utilities where practical.

**Tech Stack:** Python, Click, asyncio, pypdfium2, PIL, pytest, Click CliRunner.

---

### Task 1: Discovery And Window Planning

**Files:**
- Create: `mineru/cli/document_ocr.py`
- Test: `tests/unittest/test_document_ocr_cli.py`

- [ ] **Step 1: Write failing tests for mixed discovery, page windows, and queue interleaving**

Add tests that create image and fake PDF paths, monkeypatch PDF page counts, and assert:

```python
jobs = document_ocr.collect_document_jobs(tmp_path, tmp_path / "out", start_page_id=1, end_page_id=5)
assert [(job.path.name, job.document_type, job.stem) for job in jobs] == [...]
windows = document_ocr.build_window_jobs(jobs, page_window_size=4)
assert [(w.document_index, w.window_index, w.start_page_id, w.end_page_id) for w in windows] == [...]
queue = document_ocr.interleave_window_jobs(windows)
assert [item.document_index for item in queue] == [0, 1, 0]
```

- [ ] **Step 2: Run the failing tests**

Run: `pytest tests/unittest/test_document_ocr_cli.py::test_collect_document_jobs_discovers_mixed_inputs -q`

Expected: import failure for `mineru.cli.document_ocr`.

- [ ] **Step 3: Implement dataclasses and discovery/window helpers**

Implement `DocumentType`, `DocumentJob`, `WindowJob`, `DocumentOcrOptions`, `collect_document_jobs`, `build_window_jobs`, and `interleave_window_jobs`.

- [ ] **Step 4: Run the discovery/window tests**

Run: `pytest tests/unittest/test_document_ocr_cli.py -q`

Expected: discovery/window tests pass; later tests may not exist yet.

### Task 2: Cache, Status, And Failure Isolation

**Files:**
- Modify: `mineru/cli/document_ocr.py`
- Test: `tests/unittest/test_document_ocr_cli.py`

- [ ] **Step 1: Write failing tests for public status and private window cache**

Assert completed document status is skipped under `--resume`; matching cached window metadata is accepted; mismatched source size or page range is rejected; failed documents do not write final public outputs.

- [ ] **Step 2: Run failing cache/status tests**

Run: `pytest tests/unittest/test_document_ocr_cli.py -q`

Expected: failures for missing cache/status functions.

- [ ] **Step 3: Implement cache and status helpers**

Implement `window_cache_path`, `write_window_cache`, `read_valid_window_cache`, `write_document_status`, `is_completed_status`, and failure result handling.

- [ ] **Step 4: Run cache/status tests**

Run: `pytest tests/unittest/test_document_ocr_cli.py -q`

Expected: cache/status tests pass.

### Task 3: Assembly

**Files:**
- Modify: `mineru/cli/document_ocr.py`
- Test: `tests/unittest/test_document_ocr_cli.py`

- [ ] **Step 1: Write failing tests for out-of-order assembly**

Use cached window blocks with page metadata out of completion order and assert final `_model.json`, `_middle.json`, Markdown, content list, and status are document-level and page ordered.

- [ ] **Step 2: Run failing assembly tests**

Run: `pytest tests/unittest/test_document_ocr_cli.py -q`

Expected: failures for missing assembly functions.

- [ ] **Step 3: Implement document assembly**

Implement `assemble_document_outputs` using the same deterministic ordering for images and PDFs. Use existing `image_ocr.render_outputs` and middle-json builders where possible.

- [ ] **Step 4: Run assembly tests**

Run: `pytest tests/unittest/test_document_ocr_cli.py -q`

Expected: assembly tests pass.

### Task 4: Scheduler And CLI

**Files:**
- Modify: `mineru/cli/document_ocr.py`
- Modify: `pyproject.toml`
- Test: `tests/unittest/test_document_ocr_cli.py`

- [ ] **Step 1: Write failing tests for CLI parsing and batch continuation**

Assert CLI parses VLM HTTP options, rejects `--stream` without `--dissection`, continues after a document failure, and exits nonzero only after all runnable documents finish.

- [ ] **Step 2: Run failing CLI tests**

Run: `pytest tests/unittest/test_document_ocr_cli.py -q`

Expected: CLI and scheduler failures.

- [ ] **Step 3: Implement scheduler, VLM window processing, and Click command**

Implement `run_document_ocr`, `process_window_job`, client creation, timeout handling, progress handling, and `main`.

- [ ] **Step 4: Register console script**

Add `mineru-ocr-documents = "mineru.cli.document_ocr:main"` under `[project.scripts]`.

- [ ] **Step 5: Run CLI tests**

Run: `pytest tests/unittest/test_document_ocr_cli.py -q`

Expected: all document CLI tests pass.

### Task 5: Regression Verification And Commit

**Files:**
- Modify: implementation and tests from previous tasks

- [ ] **Step 1: Run focused regression tests**

Run: `pytest tests/unittest/test_document_ocr_cli.py tests/unittest/test_image_ocr_cli.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run CLI help checks**

Run: `python -m mineru.cli.document_ocr --help`

Expected: help includes `--page-window-size`, `--max-windows`, `--max-http-concurrency-per-window`, and `--per-window-timeout`.

- [ ] **Step 3: Inspect diff**

Run: `git diff --stat` and `git diff -- pyproject.toml`.

Expected: only intended console-script addition in `pyproject.toml`, plus new CLI/tests/plan.

- [ ] **Step 4: Commit intended changes only**

Run:

```bash
git add mineru/cli/document_ocr.py tests/unittest/test_document_ocr_cli.py docs/superpowers/plans/2026-05-28-mineru-ocr-documents.md
git add -p pyproject.toml
git commit -m "Add document OCR CLI"
```

Expected: implementation commit excludes unrelated pre-existing `pyproject.toml` edits.


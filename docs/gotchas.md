# Gotchas

Common pitfalls, debugging tips, and non-obvious behaviour worth knowing.

## Multi-worker uvicorn breaks the GPU

```bash
uvicorn ytcc_pipeline.api.app:app --workers 4   # DO NOT
```

Each worker is a fresh process with its own lifespan handler, so every worker loads its own copy of every resident model and contends for VRAM. The service is designed for **one** uvicorn worker per GPU; serialise concurrency at the asyncio lock layer (already done).

## `process_pdf` is blocking; never call from asyncio without `to_thread`

`process_pdf` uses `multiprocessing.spawn` pools internally. Calling it from an async function blocks the event loop:

```python
# BAD -- blocks the loop
async def handler():
    return process_pdf(path, language="en")

# GOOD -- runs on the threadpool
async def handler():
    return await asyncio.to_thread(process_pdf, path, language="en")
```

The FastAPI service does the right thing internally. Custom async wrappers need to follow suit.

## `spawn`, not `fork`

Workers use `multiprocessing.get_context("spawn")` everywhere. The parent process may hold a CUDA context (the resident layout analyzer in service mode), and `fork` corrupts that. `spawn` re-imports the worker module from scratch -- which is also why `pdf_oxide` is imported **inside** `worker_digital_born` rather than at module top.

## Workers can spawn dozens of OpenBLAS threads each

On high-core hosts (72+), unbounded `render_workers` or `digital_born_workers` cause OpenBLAS to spawn pthread-per-worker stacks that exhaust the kernel's pthread slots. The render pool defaults to `min(16, os.cpu_count())` for this reason; `digital_born_workers=16` is the production cap on a 72-core host. Don't push higher without measuring.

## OCR worker count vs VRAM

Each OCR worker is one `RapidOCR` engine plus its CUDA context, ~2 GiB VRAM. On a 24 GiB 3090 with the resident formula recognizer (~1.8 GiB), 6 workers is empirically stable. Pushing to 8 lets ONNXRuntime's BFC allocator surface stochastic `Failed to allocate memory` exceptions **inside** worker forward passes -- which then look like generic MISS fallbacks in the log. If you see that, drop `ocr_workers` first.

## `PYTORCH_CUDA_ALLOC_CONF` is set at import time

The `ytcc_pipeline` package's import-time code calls `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` before any torch import. This is what keeps long-running scanned services stable across many sequential PDFs -- the default CUDA caching allocator fragments enough that the 3rd+ scanned PDF's OCR engines can fail to allocate. `setdefault` preserves an operator's explicit override.

> [!IMPORTANT]
> This works because the line runs before `from .config import ...` etc., which in turn triggers torch import. Reordering the file is a footgun.

## MISS semantics

`miss=True` means **the primary extraction failed**. Three flavours:

| Block kind | Primary | MISS means |
|------------|---------|------------|
| TEXT, REFERENCE | text in `block.text` | OCR / pdf_oxide returned nothing |
| FORMULA | LaTeX in `block.text` | PP-FormulaNet generated empty / crashed |
| IMAGE | crop in `block.image_path` | never -- IMAGE blocks always have a crop |
| TABLE | structured grid in `block.cells` | never -- TABLE blocks fall back to image-only with `cells=None`, NOT `miss=True` |

Surfacing matters: `BlockType.TABLE` is **not** affected by `bundle_miss_images_for`, because TABLE failures don't produce `miss=True`. To detect a TABLE that fell back to image-only, check `cells is None` rather than `miss`.

## `formula` truncated warnings

A `formula truncated: path=... n_tokens=N` WARNING means generation stopped at `max_new_tokens` rather than EOS. The LaTeX is **still bundled** but is likely incomplete past that point.

Fix by raising the matching bucket's `*_tokens`:

| If most truncations are in bucket... | Raise this |
|---|---|
| small (area < 2500 px²) | `formula_bucket_small_tokens` |
| medium (2500-15000 px²) | `formula_bucket_medium_tokens` |
| large (>= 15000 px²) | `formula_max_new_tokens` |

Default caps were tuned to give 0 truncations on the project's reference Turkish thesis -- if your corpus has unusually long display equations, expect to raise the large cap or switch to `PaddlePaddle/PP-FormulaNet_plus-L_safetensors` (2560-token decoder).

## Frozen dataclasses + immutability

Every schema dataclass is `@dataclass(slots=True, frozen=True)`. Mutating a `Block` after assembly raises:

```python
block.text = "..."           # FrozenInstanceError
```

Use `dataclasses.replace`:

```python
from dataclasses import replace
new_block = replace(block, text="...", miss=False)
```

The pipeline itself does this in every stage that rewrites blocks (`run_formula_stage`, `run_table_stage`, `run_reference_stage`). The reading order / bbox stay intact, only the modified fields change.

## `PipelineConfig.__post_init__` coerces iterables

TOML serialises sets and tuples as arrays, env vars deliver comma-separated strings, callers pass `list`, `tuple`, `frozenset` interchangeably. `PipelineConfig.__post_init__` coerces:

- `bundle_miss_images_for` -> `frozenset[BlockType]` (validating each entry).
- `reference_labels` -> `tuple[str, ...]`.

This is why `frozenset(BlockType)` is the **default factory** and not a literal -- the post-init coercion converts whatever input comes in. Unknown `BlockType` values in `bundle_miss_images_for` raise `ValueError`.

## `digital_born=None` is auto-detect, not "ignored"

`process_pdf(pdf, language="en", digital_born=None)` is the **same** as omitting the parameter -- both trigger auto-detect. There is no "ignore my override" sentinel; pass `True` or `False` to force.

## `retain_temp_dir=True` for debugging

Drops the auto-cleanup of `/tmp/ytcc_pipeline_*` after the pipeline finishes. The temp dir contains:

- `pages/` -- every rendered page image. Useful when layout misbehaves.
- `images/` -- every saved crop including ones the formula stage deleted post-recognition. Useful when checking exactly which crop the model saw.

Never enable in production; `/tmp` fills up quickly across requests.

```python
cfg = PipelineConfig(retain_temp_dir=True)
```

## `pdf_info` may have weird keys

XMP fields come directly from the PDF. Older theses often have:

- BOM-prefixed UTF-16 strings -> decoded back to UTF-8 by `_clean_xmp_value`.
- Null bytes -> stripped.
- Lists -> joined with `"; "`.
- Non-string values -> coerced via `str()`.

If a downstream consumer chokes on `pdf_info`, it's almost always one of these flavours surviving the cleanup. Most other metadata layers (XMP RDF, encrypted PDFs, IPTC) are out of scope -- `pdf_oxide.xmp_metadata()` is the only source today.

## Bundle filenames don't sort by reading order

`images/0014-formula-{uuid}.png` sorts by page first, then by label, then by random uuid. There is no on-disk encoding of reading order within a page -- consumers that need it must read `document.json` and look up each block's `reading_order`. This is intentional: the bundle is a content-addressed store with `document.json` as the index.

## Render DPI != Page DPI in the schema

After auto-DPI selection (digital-born uses `render_dpi_digital_born=150`), the orchestrator overwrites `cfg.render_dpi` in place via `dataclasses.replace`. The resulting `Page.width_px` / `Page.height_px` and every `bbox` are in **that effective DPI**, not in the original `cfg.render_dpi` you passed.

Consumers re-rendering against the page image must use the **effective** DPI. If you want it documented in the bundle, check the page dimensions against the PDF's `MediaBox` or read it from the logs (`stage render: ... dpi=N`).

## `formula_number` is a TEXT block, not a FORMULA

Layout labels like `formula_number` (the `(3.2)` next to an equation) are routed to `Route.TEXT` -- it's plain text, not a math expression. Doing it via OCR / `pdf_oxide` is faster and more accurate than pushing through a 700 MB transformer.

If you see equation numbers extracted as text instead of LaTeX, that's the intended behaviour.

## Reading the logs

Each successful run emits a predictable trail (INFO level):

```text
pipeline start: pdf=paper.pdf language=en bytes=...
pipeline detect: pdf=paper.pdf digital_born=True (auto)
stage render: pdf=paper.pdf pages=120 dpi=150 format=jpeg elapsed_s=...
analyzer ready: model=... device=cuda:0 fp16=True fast_preproc=True
stage layout: pdf=paper.pdf detections=... elapsed_s=... analyzer=owned
stage blocks: pdf=paper.pdf text=N image=N reference=N formula=N table=N miss=N elapsed_s=...
stage table: pdf=paper.pdf skipped reason=disabled
stage formula start: pdf=paper.pdf crops=N batch_size=8 bucketed=True
formula bucketed: small=N medium=N large=N (small_cap=192 medium_cap=512 large_cap=1536)
stage formula: pdf=paper.pdf formulas=N miss=N truncated=N crops_dropped=N elapsed_s=...
stage reference: pdf=paper.pdf skipped reason=disabled
stage bundle: pdf=paper.pdf path=paper.tar bytes=... elapsed_s=...
pipeline done: pdf=paper.pdf elapsed_s=... bundle=paper.tar bundle_bytes=...
```

Things to grep for when investigating:

- `pipeline miss:` -- WARNING when any block missed. Reports `miss/total (%)`.
- `formula MISS:` -- per-crop reason (`load_failed`, `empty_output`).
- `formula truncated:` -- bucket cap too tight; raise the matching `*_tokens` field.
- `table fallback degenerate:` -- SLANet+ returned <2 cells, image-only fallback.
- `table batch failed:` -- whole-batch RapidTable exception; check exc_info trace.
- `OCR call failed on crop` -- RapidOCR raised on a specific crop.
- `lifespan: GROBID unreachable` -- start the server.
- `request rejected: ... reason=scanned_disabled` -- caller sent a scanned PDF to a digital-born-only deployment.

DEBUG adds thousands of per-block lines per PDF. Useful when chasing a specific block's behaviour; impossible to read in aggregate.

## XPC: `pdf_oxide` is chatty at INFO

`pdf_oxide` emits per-font internals at INFO level -- ~50 lines per PDF, useless for operators. `LoggingSettings` pins it to WARNING:

```toml
[logging]
pdf_oxide_level = "WARNING"
```

If you set `LoggingSettings.apply()` from your own script, the library default (`WARNING`) carries through. Without calling `apply()` the library never installs handlers and `pdf_oxide`'s emissions go nowhere -- which is also fine.

## Output path defaults next to the input PDF

`process_pdf("paper.pdf", language="en")` writes to `paper.tar` next to the input. To control the location:

```python
process_pdf("paper.pdf", language="en", output_path="/tmp/out/paper.tar")
```

`Path(output_path).parent` is created if it doesn't exist. If the path already exists, it is **overwritten** without warning.

## `process_pdf` mutates `cfg.render_dpi` via `dataclasses.replace`

Subtle: the orchestrator does

```python
if digital_born and cfg.render_dpi_digital_born != cfg.render_dpi:
    cfg = replace(cfg, render_dpi=cfg.render_dpi_digital_born)
```

The outer caller's `cfg` is unchanged (`PipelineConfig` is frozen), but logs and downstream stages within this call see the new value. If you read `cfg.render_dpi` outside `process_pdf` expecting the auto-DPI value, you'll see the **original** field. Read `Page.width_px / pdf_page_width_pt * 72` instead, or just rely on the bbox space (always page-pixel at the effective DPI).

## `BlockType` is a `StrEnum`

`block.type == "formula"` works (StrEnum comparison falls back to the string). The schema's JSON serialisation drops the enum and ships the string value directly. When matching block types in your consumer, either `from ytcc_pipeline.schema import BlockType` and compare against the enum, or compare against the string literal -- both work, the string literal version is more portable for non-Python consumers.

## Per-test isolation when forking is involved

Tests that spin up the pipeline must not fork inside an event loop that already holds a CUDA context. `pytest -m gpu` markers are intentionally separate from the default suite for this reason. See `pyproject.toml` `markers` declaration.

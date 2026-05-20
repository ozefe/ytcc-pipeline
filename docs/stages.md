# Stages

Every `process_pdf` call runs the same eight stages in fixed order. This page documents what each stage does, the knobs it honours, and the skip / failure semantics.

```text
render -> metadata -> layout -> blocks -> table -> formula -> reference -> bundle
```

Each stage emits one INFO log line on completion. Disabled or no-op stages emit `skipped reason=<reason>` and short-circuit.

## render

Decode every page of the PDF to an image file inside the pipeline's temp dir (`pages/`).

- Entry point: `render_pages`.
- Parallelism: `ProcessPoolExecutor` with `spawn` start method. Pool size: `cfg.render_workers`, or `min(16, os.cpu_count())` when unset.
- Each worker reopens its own `pdf_oxide.PdfDocument` -- the handle is not picklable.

**Knobs:** `render_dpi`, `render_dpi_digital_born`, `page_format`, `jpeg_quality`, `render_workers`.

**Output files:** `pages/page-{i:04d}.{jpg|png}` in 0-based page order.

> [!NOTE]
> The default worker cap (16) exists to avoid OpenBLAS pthread storms on high-core hosts. Bump `render_workers` if your host has fewer cores than 16; raising the cap on >16-core hosts didn't move wall in benchmarks but starves the rest of the pipeline of pthread slots.

## metadata

Compute sha256, byte size, and read XMP from the PDF.

- Entry point: `extract_metadata`.
- Cheap. Run early so I/O failures surface before model load.
- XMP parse failures are caught and yield an empty `pdf_info`. The bundle still ships -- losing XMP is graceful degradation.

UTF-16 BOMs in XMP values (common in older PDFs) are unwrapped to UTF-8. Null bytes are stripped.

## layout

Run PP-DocLayoutV3 over the rendered pages.

- Entry points: the `LayoutAnalyzer` Protocol + `make_analyzer` factory; the shipped backend is `SafeTensorsLayoutAnalyzer` (PyTorch + SafeTensors via HuggingFace Transformers).
- Loaded via `make_analyzer_from_config(cfg)`; loaded once per call in library mode, once per process in service mode (injected by FastAPI lifespan).

**Knobs:** `layout_device`, `layout_batch_size`, `layout_confidence`, `layout_fp16`, `layout_fast_preproc`.

**Output:** `{page_idx: list[LayoutDetection]}` with detections sorted in reading order. Each `LayoutDetection` carries `label`, `confidence`, `bbox`, `reading_order`.

`layout_confidence=0.5` is the default. Drop it to surface noise (marginal text, page edges); raise it to be aggressive about dropping low-confidence detections.

### fp16

`layout_fp16=True` converts model + inputs to half precision. ~1.7x speedup on the layout stage in isolation; <1% detection delta on the test set. SafeTensors backend only.

### fast_preproc

`layout_fast_preproc=True` swaps HF's PIL-based `AutoImageProcessor` (~120 ms/page CPU) for a cv2 path **and** starts a producer thread that prepares batch N+1 while the GPU runs batch N. ~2.3x speedup on the layout phase in isolation. SafeTensors backend only.

### Reading order

PP-DocLayoutV3 emits an `order_seq` tensor alongside the detections. The backend converts that to per-detection 0..N-1 ranks via double-argsort and sorts the output by rank. Unknown corpora may produce strange reading orders -- the field is preserved as-is from the model.

## blocks

Per-page block assembly. Three execution paths share one output shape:

| Path | Trigger | Workers |
|------|---------|---------|
| serial | both worker counts are 1 | main process |
| parallel-digital | `digital_born=True` and `digital_born_workers > 1` | `cfg.digital_born_workers` |
| parallel-scanned | `digital_born=False` and `ocr_workers > 1` | `cfg.ocr_workers` |

- Entry points: `process_pages` (the dispatcher), `worker_digital_born` / `worker_scanned` (spawn-pool worker bodies), and `build_block` (the per-detection decision point).
- All paths use `multiprocessing.spawn` -- never `fork`, because the parent may hold a CUDA context.

Each detection is routed via `routing.route_for(label)`:

- `IGNORE` -> dropped.
- `TEXT` / `REFERENCE` -> text extraction (pdf_oxide for digital-born, RapidOCR for scanned). MISS path on failure.
- `IMAGE` -> crop + save.
- `FORMULA` -> crop + save (LaTeX filled in later by the formula stage).
- `TABLE` -> crop + save (cell grid filled in later by the table stage).

**Knobs:** `digital_born_workers`, `ocr_workers`, `ocr_batch_size`, `ocr_min_score`, `ocr_use_cuda`, `crop_format`, `jpeg_quality`, `bundle_miss_images_for`.

### Scanned text batching

On a scanned page the worker pre-collects every TEXT/REFERENCE bbox into a list and calls `OcrExtractor.extract_batch` once -- RapidOCR's internal `Rec.batch_size` flushes for free this way. Per-detection routing happens after the batch returns; failed crops degrade to MISS.

### Worker OCR engine lifecycle

Each scanned worker process lazily constructs **one** `OcrExtractor` (RapidOCR engine + CUDA context) and caches it for every page dispatched to that worker. The cache is a module-level singleton (`_worker_ocr`) -- `spawn` re-imports the worker module per process, so the global is private per worker. Engine load is ~2 GiB VRAM each; `ocr_workers=6` is the production value on a 24 GiB card alongside the resident formula recognizer.

## table

RapidTable SLANet+ recovers cell grids for `TABLE` blocks.

- Entry point: `run_table_stage`.
- Opt-in via `cfg.table_enabled` (default off).
- Skips quietly if disabled, if no `TableEngine` was loaded (caller didn't inject and `table_enabled=False`), or if no TABLE blocks exist.

**Knobs:** `table_enabled`, `table_device`, `table_batch_size`, `table_min_side_px`.

**Per-table flow:**

1. Skip tables with either dimension below `cfg.table_min_side_px` (default 120). Keeps spurious `table` detections on inline elements from going through structure recognition.
2. RapidTable SLANet+ on the crop -> cell polygons + per-cell `[row_start, row_end, col_start, col_end]`. Batched `cfg.table_batch_size` at a time.
3. Translate cell polygons from crop coords to page coords.
4. Per-cell text extraction: `pdf_oxide` (digital-born) or per-cell RapidOCR (scanned). Empty / failed extractions leave `Cell.text=None`.
5. Replace the TABLE block with a structured version carrying `n_rows`, `n_cols`, `cells`. The original crop in `image_path` stays as a fallback for renderers.

Degenerate output (< 2 cells, unreadable crop, RapidTable exception) falls back to image-only: crop stays bundled, `cells` stays `None`.

> [!WARNING]
> The RapidTable engine has no `close()` method -- VRAM is reclaimed at process exit. In long-lived service processes this is fine; in scripts that loop over many `process_pdf` calls, inject one `TableEngine` and reuse it.

## formula

PP-FormulaNet-L runs across every `FORMULA` block's crop to recover LaTeX.

- Entry point: `run_formula_stage`.
- Opt-in via `cfg.formula_enabled` (default **on**).
- Skips if disabled, if no recognizer was injected and `formula_enabled=False`, or if no FORMULA blocks exist.

**Knobs:** `formula_enabled`, `formula_model_id`, `formula_device`, `formula_dtype`, `formula_batch_size`, `formula_max_new_tokens`, `formula_torch_compile`, `formula_bucketed`, `formula_bucket_small_threshold`, `formula_bucket_medium_threshold`, `formula_bucket_small_tokens`, `formula_bucket_medium_tokens`.

**Per-formula flow:**

1. Collect every FORMULA block with `image_path` set, recording `(page_idx, block_idx, crop_path, bbox_area)`.
2. Run the recognizer: bucketed (`formula_bucketed=True`, default) or flat batching.
3. For each result:
   - **Success**: rewrite the block with `text=<LaTeX>`, `image_path=None`, `miss=False`. Delete the on-disk crop.
   - **MISS**: rewrite the block with `text=None`, `miss=True`. Rename the crop with the `-MISS-` marker (if `BlockType.FORMULA in cfg.bundle_miss_images_for`) or delete it.

### Bucketed batching

Greedy generation runs the whole batch for as many steps as the slowest row needs. A batch of seven inline formulas mixed with one display equation pays the display equation's full token budget on all eight rows. Bucketing collapses that overhead:

1. Sort crops by bbox area (a proxy for output length).
2. Bucket into `small` / `medium` / `large` by `formula_bucket_small_threshold` / `formula_bucket_medium_threshold` (in source-page px²).
3. Run each bucket with its own `max_new_tokens` cap (`formula_bucket_small_tokens`, `formula_bucket_medium_tokens`; large reuses the global `formula_max_new_tokens`).
4. Splice results back into the original `crop_paths` order.

Measured 1.63x speedup vs flat batching on a Turkish digital-born sample. Default thresholds are tuned for the 150 DPI digital-born render; scale ~4x for 300 DPI scanned crops.

> [!TIP]
> If you observe `formula truncated` WARNINGs in the logs, your bucket caps are too tight. Raise the matching `*_tokens` field. Truncation is reported per-call by the recognizer (`FormulaResult.truncated`); the LaTeX is still kept and bundled.

### `torch_compile`

`formula_torch_compile=True` wraps the model with `torch.compile` and warms up Inductor at construction time (~20-30 s extra at load). Subsequent batches run kernel-fused. Off by default; opt in per deployment after measuring on representative documents. Compile mode is `"default"` -- `reduce-overhead` (CUDA Graphs) recompiles every token because the KV-cache grows.

### Model variants

The default is `PaddlePaddle/PP-FormulaNet-L_safetensors` (1024-token decoder, ~1.8 GiB at fp16). For unusually long display equations, switch to `PaddlePaddle/PP-FormulaNet_plus-L_safetensors` (2560-token decoder, same processor config, drop-in replacement):

```toml
[pipeline]
formula_model_id = "PaddlePaddle/PP-FormulaNet_plus-L_safetensors"
```

## reference

Batched HTTP call to GROBID to enrich `REFERENCE` blocks.

- Entry point: `run_reference_stage`.
- Opt-in via `cfg.references_enabled` (default off).
- Externally-managed dependency: you start the GROBID server separately. The references guide walks through Docker and bare-metal setup.

**Knobs:** `references_enabled`, `grobid_url`, `grobid_timeout_s`, `reference_labels`.

**Flow:**

1. Collect `(page_idx, block_idx, text)` for every block whose `label` is in `cfg.reference_labels` **and** whose `text` is non-empty.
2. Send all texts in **one batched POST** to `/api/processCitationList`.
3. Splice each parsed `Reference` into the block's `reference` field. The raw text stays on `Block.text` either way.

**Failure modes** -- server unreachable, timeout, HTTP error, malformed XML -- are logged at WARNING and the page list flows through unchanged. References are an enrichment; pipeline correctness doesn't depend on them.

`Block.reference` is `None` in three cases: stage disabled, GROBID parse returned nothing usable, or the block's label was not in `cfg.reference_labels`. The first two are indistinguishable from the schema alone -- check the logs.

## bundle

Pack `document.json` + `images/*` into one uncompressed tar.

- Entry point: `create_bundle`.
- `document.json` is the first archive member; consumers can stream-parse it before image bytes arrive.
- Images are added in alphabetical filename order under `images/`.
- The output path defaults to `<pdf-stem>.tar` next to the input.

Failures here are surface I/O issues (full disk, permission). The temp dir is removed after `process_pdf` returns unless `cfg.retain_temp_dir=True`.

> [!TIP]
> When debugging extraction issues, set `retain_temp_dir=True` and inspect the temp dir (`/tmp/ytcc_pipeline_*`) after the call. You'll find the rendered pages, all intermediate crops (including ones the formula stage deleted in production), and can rerun stages in isolation.

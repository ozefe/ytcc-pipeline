# Benchmark suite summary

20 knob sweeps × 73 measured runs on an RTX 3090 (24 GiB, CUDA 12.6). Each sweep varies one `PipelineConfig` knob across a range of values while every other knob is held constant against a shared base. Speed, resources (CPU% / RSS / VRAM at min / median / p95 / max), and quality metrics are captured per row; OOMs and process-pool crashes are recorded as failed rows rather than aborting the sweep.

Source PDFs:

- `904599.pdf` -- 1.7 MB, 155-page English digital-born thesis with ~570 text blocks, 37 image blocks, 900 formula blocks, 36 tables, and 44 reference blocks. Used by every digital-born sweep.
- `084016.pdf` -- 7 MB, 135-page English scanned thesis. Used by the scanned render-DPI / OCR sweeps.
- `101123.pdf` -- 11 MB, 120-page Turkish scanned thesis. Used by `jpeg_quality` because Turkish diacritics (ç, ğ, ı, ş, ü, ö) make OCR character recall the most sensitive signal for that knob.

All three PDFs live under `samples/` at the project root. To re-run on your own corpus, drop PDFs into `samples/` and edit the filenames at the top of `benchmarks/sweeps.py`.

## Headline results

| Knob | Range | Wall (s) | Notes |
|---|---|---|---|
| `layout_batch_size` | 1 -> 24 | 35.5 -> 32.4 | flat after bs=4; minor noise |
| `layout_fp16` | False -> True | 30.7 -> 30.2 | same wall; **VRAM 44% lower** (12.3 -> 6.9 GiB) -- pure win on tensor-core GPUs |
| `layout_fast_preproc` | False -> True | 33.5 -> 30.0 | cv2 preprocess + producer thread saves ~3.5 s |
| `render_dpi_digital_born` | 100 -> 300 | 28.9 -> 39.8 | 38% wall increase end-to-end |
| `page_format` | png / jpeg | 32.9 -> 30.0 | JPEG temp pages shave ~3 s of render |
| `jpeg_quality` | 40 -> 100 | 114.9 -> 115.4 | flat wall **and** flat OCR output (178612 chars, 824 text blocks, 1 MISS at every value) on the Turkish scanned PDF -- RapidOCR PP-OCRv5 is robust down to q=40 |
| `digital_born_workers` | 1 -> 16 | 21.0 -> 30.0 | **1-worker (serial path) is fastest** at 21s; 2-16 plateau at ~30s; 32+ crash with `BrokenProcessPool` |
| `formula_enabled` | False -> True | 30.1 -> 498.9 | recovers LaTeX for 879 of 900 crops |
| `formula_model_id` | L / plus-L | 508 -> 331 | plus-L is **1.54× faster** but outputs **19% fewer LaTeX chars** (106k -> 86k) -- inspect samples before swapping |
| `formula_dtype` | fp32 -> fp16 | 518 -> 498 | similar speed, **13%** VRAM reduction (6.4 -> 5.5 GiB) -- far less than the theoretical half because decoder KV-cache + activations dominate at bs=4 |
| `formula_batch_size` | 4 -> 16 | 487 -> 229 | 2.1× speedup from bs=4 to bs=16 |
| `formula_bucketed` | False -> True | 600 -> 483 | 1.24× speedup from bucketing |
| `formula_torch_compile` | False -> True | 482 -> 486 | compile cost barely paid back on 900 crops |
| `table_enabled` | False -> True | 29.9 -> 38.2 | +8 s for 36 tables (~0.23 s / table) |
| `table_batch_size` | 1 -> 16 | 39.0 -> 37.1 | minor; SLANet+ already amortizes well |
| `references_enabled` | False -> True | 29.9 -> 61.5 | +31 s for one batched POST to GROBID over 44 refs |
| `render_dpi_scanned` | 100 -> 300 | 105 -> 119 | OCR dominates; render is a small fraction |
| `ocr_batch_size` | 4 -> 128 | 119 -> 119 | flat -- RapidOCR's `Rec.batch_size` doesn't shift wall on this PDF |
| `ocr_use_cuda` | True | 114 | CUDA baseline; CPU value omitted (1 h+ wall) |
| `ocr_workers` | 1 -> 8 | 410 -> 101 | 4× speedup at 8 workers |

Failures across the suite:

- `digital_born_workers=32, 64` -- spawn workers crash on resource exhaustion (`BrokenProcessPool`). The 16-worker default is the highest stable value on the bench host.
- `ocr_use_cuda=False` -- skipped from the automated suite because the CPU OCR path is ~50× slower than CUDA on this PDF; the warmup row alone exceeded one hour of wall.

## Apples-to-apples discipline

Two shared base configs hold the non-swept knobs constant:

- `DIGITAL_BORN_BASE` -- every digital-born sweep starts here. Formula, table, and reference stages are off by default; the sweeps that test them flip the relevant toggle. `layout_fp16=True` and `layout_fast_preproc=True` mirror the recommended production config.
- `SCANNED_BASE` -- every scanned sweep starts here. `ocr_workers=6` and `ocr_use_cuda=True` mirror the production setup, with `OMP_NUM_THREADS=2` to avoid OpenBLAS thread storms across spawn workers.

Each sweep:

1. Warm-up run on the first value (not recorded). Loads HF model files and primes the CUDA context.
2. Measured rows in order. Resources sampled at 250 ms cadence over the entire `process_pdf` call.
3. CUDA cache cleared between rows so an OOM in row N cannot leak into row N+1.
4. Incremental CSV writes -- a crash mid-sweep preserves earlier rows.

## How to read the plots

Every per-sweep dashboard has four panels:

- **Wall time** (top-left) -- a line for numeric knobs (DPI, batch size, worker counts) or a bar for categorical / boolean knobs.
- **Per-stage wall** (top-right) -- stacked bars breaking down the wall into render / layout / blocks / table / formula / reference / bundle. Stages that contributed zero in this sweep are omitted from the legend.
- **Resources** (bottom-left) -- VRAM max + RSS max as side-by-side bars; CPU% median as a line on the secondary axis.
- **Quality** (bottom-right) -- sweep-specific counts. For OCR sweeps it's text-character count + MISS; for formula sweeps it's the count of crops with recovered LaTeX + total LaTeX chars; for layout sweeps it's the total detected block count.

## Reports

| Sweep | csv | md | plot |
|---|---|---|---|
| digital_born_workers | [csv](sweeps/digital_born_workers.csv) | [md](sweeps/digital_born_workers.md) | [png](plots/digital_born_workers.png) |
| formula_batch_size | [csv](sweeps/formula_batch_size.csv) | [md](sweeps/formula_batch_size.md) | [png](plots/formula_batch_size.png) |
| formula_bucketed | [csv](sweeps/formula_bucketed.csv) | [md](sweeps/formula_bucketed.md) | [png](plots/formula_bucketed.png) |
| formula_dtype | [csv](sweeps/formula_dtype.csv) | [md](sweeps/formula_dtype.md) | [png](plots/formula_dtype.png) |
| formula_enabled | [csv](sweeps/formula_enabled.csv) | [md](sweeps/formula_enabled.md) | [png](plots/formula_enabled.png) |
| formula_model_id | [csv](sweeps/formula_model_id.csv) | [md](sweeps/formula_model_id.md) | [png](plots/formula_model_id.png) |
| formula_torch_compile | [csv](sweeps/formula_torch_compile.csv) | [md](sweeps/formula_torch_compile.md) | [png](plots/formula_torch_compile.png) |
| jpeg_quality | [csv](sweeps/jpeg_quality.csv) | [md](sweeps/jpeg_quality.md) | [png](plots/jpeg_quality.png) |
| layout_batch_size | [csv](sweeps/layout_batch_size.csv) | [md](sweeps/layout_batch_size.md) | [png](plots/layout_batch_size.png) |
| layout_fast_preproc | [csv](sweeps/layout_fast_preproc.csv) | [md](sweeps/layout_fast_preproc.md) | [png](plots/layout_fast_preproc.png) |
| layout_fp16 | [csv](sweeps/layout_fp16.csv) | [md](sweeps/layout_fp16.md) | [png](plots/layout_fp16.png) |
| ocr_batch_size | [csv](sweeps/ocr_batch_size.csv) | [md](sweeps/ocr_batch_size.md) | [png](plots/ocr_batch_size.png) |
| ocr_use_cuda | [csv](sweeps/ocr_use_cuda.csv) | [md](sweeps/ocr_use_cuda.md) | [png](plots/ocr_use_cuda.png) |
| ocr_workers | [csv](sweeps/ocr_workers.csv) | [md](sweeps/ocr_workers.md) | [png](plots/ocr_workers.png) |
| page_format | [csv](sweeps/page_format.csv) | [md](sweeps/page_format.md) | [png](plots/page_format.png) |
| references_enabled | [csv](sweeps/references_enabled.csv) | [md](sweeps/references_enabled.md) | [png](plots/references_enabled.png) |
| render_dpi_digital_born | [csv](sweeps/render_dpi_digital_born.csv) | [md](sweeps/render_dpi_digital_born.md) | [png](plots/render_dpi_digital_born.png) |
| render_dpi_scanned | [csv](sweeps/render_dpi_scanned.csv) | [md](sweeps/render_dpi_scanned.md) | [png](plots/render_dpi_scanned.png) |
| table_batch_size | [csv](sweeps/table_batch_size.csv) | [md](sweeps/table_batch_size.md) | [png](plots/table_batch_size.png) |
| table_enabled | [csv](sweeps/table_enabled.csv) | [md](sweeps/table_enabled.md) | [png](plots/table_enabled.png) |

## Reproducing the suite

```bash
# Run every sweep (skips ones with cached CSVs):
python benchmarks/run_all.py

# Limit scope:
python benchmarks/run_all.py --only formula
python benchmarks/run_all.py --only ocr

# Re-run a sweep that's already cached:
python benchmarks/run_all.py --only formula_dtype --force

# Recommended env-vars for the scanned sweeps to avoid OpenBLAS thread storms:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python benchmarks/run_all.py --only ocr
```

Plots are regenerated from CSVs whenever `--only` is omitted, or via `python -c "from benchmarks.plot import plot_all; ..."`. CSV is the source of truth; the Markdown summaries and PNGs are byproducts.

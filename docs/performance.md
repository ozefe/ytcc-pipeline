# Performance

The shipped `config.toml` is tuned for an RTX 3090 host with ~24 GiB VRAM and a 24+ core CPU. This document explains the per-knob impact, why the defaults are what they are, and how to retune for different hardware.

## Recommended production config

The values below are all already in `config.toml`. End-to-end ~15s for a 150-page English digital-born thesis on an RTX 3090.

```toml
[pipeline]
render_dpi = 300
render_dpi_digital_born = 150
page_format = "jpeg"
jpeg_quality = 90
crop_format = "png"

layout_batch_size = 8
layout_fp16 = true
layout_fast_preproc = true

ocr_batch_size = 64
ocr_use_cuda = true

digital_born_workers = 16
ocr_workers = 6

formula_enabled = true
formula_dtype = "fp16"
formula_batch_size = 8
formula_bucketed = true
```

An equivalent `PipelineConfig(...)` literal ships in the bundled `super_fast` example script.

## The biggest lever: auto-DPI

`render_dpi_digital_born = 150` (default) vs `300` halves render wall on digital-born PDFs. The layout detector downsamples to 800x800 internally and `pdf_oxide` text extraction is resolution-independent, so character recall stays at ~99.9%. The orchestrator picks the effective DPI at runtime in `_run_stages`:

```python
if digital_born and cfg.render_dpi_digital_born != cfg.render_dpi:
    cfg = replace(cfg, render_dpi=cfg.render_dpi_digital_born)
```

Scanned PDFs keep `render_dpi=300` -- OCR loses ~3% of text blocks at 150 DPI on real PDFs.

## Per-knob impact

Benchmark wall-time deltas are measured in isolation against the `DIGITAL_BORN_BASE` and `SCANNED_BASE` configs from the benchmark sweeps. Mileage varies per corpus.

### Render

| Knob | Effect |
|------|--------|
| `page_format="jpeg"` | ~30% faster end-to-end on digital-born. Layout downsamples to 800x800 so JPEG artefacts erase before they could affect detection. |
| `jpeg_quality=90` | Drop to 85 for a few percent more at slight quality risk on Turkish/Arabic edge glyphs. |
| `render_workers` | Pool size. `None` -> `min(16, os.cpu_count())`. Cap exists to avoid OpenBLAS pthread storms. |

### Layout

| Knob | Effect |
|------|--------|
| `layout_fp16=true` | ~1.7x isolated layout speedup, <1% detection delta. SafeTensors backend only. |
| `layout_fast_preproc=true` | ~2.3x isolated layout speedup. Swaps PIL `AutoImageProcessor` for cv2 + producer thread overlapping batch N+1 preproc with batch N inference. |
| `layout_batch_size=8` | Production sweet spot. bs=24 gave a 1.07-1.15x layout-stage speedup but didn't move end-to-end wall meaningfully and pushed VRAM peak from ~4 GiB to ~13 GiB. |

> [!NOTE]
> Both performance flags are SafeTensors-only and currently the only shipped backend. They are off in `PipelineConfig()` defaults because library-mode callers expect zero surprises -- flip them on in production.

### OCR (scanned only)

| Knob | Effect |
|------|--------|
| `ocr_use_cuda=true` | ~3.7x speedup vs CPU on the OCR stage. Requires `onnxruntime-gpu`. |
| `ocr_batch_size=64` | RapidOCR docs note 6 is usually optimal; 64 is fine because empty crops short-circuit early. |
| `ocr_workers=6` | Production value on a 24 GiB 3090 alongside the resident formula recognizer (~1.8 GiB). Each worker is ~2 GiB. Raise to 8 on cards >=32 GiB, drop to 4 on cards <=16 GiB. |

> [!CAUTION]
> Pushing `ocr_workers` to 8 on a 24 GiB card saturates the device and lets the ONNXRuntime BFC allocator surface stochastic `Failed to allocate memory` exceptions inside worker forward passes. 6 is empirically stable.

### Block extraction (digital-born only)

| Knob | Effect |
|------|--------|
| `digital_born_workers=16` | Process-pool size for per-page `pdf_oxide` extraction. 16 is the bench-selected value on a 72-core host; higher exhausts OpenBLAS threads. |

### Formula

| Knob | Effect |
|------|--------|
| `formula_dtype="fp16"` | Halves VRAM (~1.8 GiB at fp16 vs ~3.6 at fp32). 20/20 exact-match against fp32 on real PDF crops. |
| `formula_batch_size=8` | Sweet spot. bs=4->8 gave 1.31x on a Turkish thesis (79.7s -> 60.6s formula stage); bs=8->16 adds ~4% throughput but doubles peak VRAM and bundles easy crops with hard ones (inflates truncation). |
| `formula_bucketed=true` | 1.63x speedup vs flat batching on the same Turkish thesis. Routes crops by bbox area into small/medium/large buckets with tighter `max_new_tokens` caps. |
| `formula_torch_compile=true` | First call after enabling pays ~20-30s Inductor compilation; subsequent batches run kernel-fused. Opt in per deployment after measuring. |
| `formula_max_new_tokens=1536` | Per-crop generation cap. The L variant's decoder caps at 1024 internally; plus-L at 2560. Real crops never approach either ceiling. |

#### Bucket tuning

Defaults are calibrated for the 150 DPI digital-born render. Scale ~4x for 300 DPI scanned crops:

| Knob | Default (150 DPI) | For 300 DPI scanned |
|------|-------------------|---------------------|
| `formula_bucket_small_threshold` | 2500 | ~10000 |
| `formula_bucket_medium_threshold` | 15000 | ~60000 |
| `formula_bucket_small_tokens` | 192 | unchanged |
| `formula_bucket_medium_tokens` | 512 | unchanged |

Token caps are unit-free (token counts), so they don't scale with DPI. Watch the WARNING-level `formula truncated` log line: when it fires, raise the matching bucket's `*_tokens`.

### Table

| Knob | Effect |
|------|--------|
| `table_enabled=true` | Adds ~1 GiB VRAM for SLANet+. Off by default; flip on when consumers need the cell grid rather than just the table crop. |
| `table_batch_size=8` | Tables per SLANet+ forward pass. |
| `table_min_side_px=120` | Skip tables with either dimension below this. Keeps spurious `table` detections on inline elements out of structure recognition. |

### Reference

| Knob | Effect |
|------|--------|
| `references_enabled=true` | Adds ~50-200 ms per PDF (one batched HTTP call). Network-bound. |
| `grobid_timeout_s=60` | Single POST per PDF, so this caps total stage wall regardless of bibliography size. 60s comfortably covers 200 refs on a CRF server. |

## VRAM budget on a 24 GiB 3090

Production config peaks:

| Resident model | Approx VRAM |
|----------------|-------------|
| Layout analyzer (PP-DocLayoutV3, fp16) | ~3 GiB |
| Formula recognizer (PP-FormulaNet-L, fp16) | ~1.8 GiB |
| Table engine (SLANet+, when enabled) | ~1 GiB |
| Per OCR worker (RapidOCR + CUDA context) | ~2 GiB |

Digital-born request: layout + formula + table = ~5.8 GiB. Plenty of headroom.

Scanned request peak: 6 OCR workers + formula = 6\*2 + 1.8 = ~13.8 GiB. Workers spawn while the layout analyzer is closed (the FastAPI service does this temporarily inside the GPU lock, and library mode closes the analyzer per-call by default).

> [!IMPORTANT]
> The `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` env var is set at package-import time (via `os.environ.setdefault`) before any torch import resolves. This is what keeps long-running scanned services stable -- the default allocator fragments enough that the 3rd+ scanned PDF's OCR engines can fail to allocate. Override only if you have a good reason.

## Tuning checklist for a new host

1. **GPU VRAM** -- set `formula_dtype`, `ocr_workers`, `table_enabled` according to budget table above.
2. **CPU cores** -- set `render_workers` and `digital_born_workers` to `min(cores, 16)` initially; benchmark up from there.
3. **Run the sweeps** -- the `benchmarks` package reproduces every knob sweep against the shipped samples. CSVs and plots land under a `results/` subdirectory.
4. **Watch for MISS rates** -- if `pipeline miss: ... %` exceeds ~1-2% in logs, check the per-stage stage MISS counts (`stage formula: ... miss=N truncated=M`) to localise.
5. **VRAM pressure** -- watch `nvidia-smi` during a representative run. If you see allocator failures inside OCR workers, drop `ocr_workers`.

## When to skip flags

- **`layout_fp16=true`** -- skip if you observe detection drops on your corpus. fp32 is safe.
- **`layout_fast_preproc=true`** -- skip if you ever swap in a non-SafeTensors layout backend. Both flags assume the SafeTensors path.
- **`formula_torch_compile=true`** -- skip in short-lived processes. The 20-30s warmup eats most of the speedup unless you're processing many PDFs in the same Python process.
- **`page_format="jpeg"`** -- skip if you also intend to use the saved page images for anything beyond layout (e.g. reusing them downstream as figures). JPEG artefacts are erased before they affect layout but are still in the file on disk.

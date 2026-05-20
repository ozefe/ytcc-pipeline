# Benchmarks

Knob-sweep benchmarks for the pipeline. Each sweep varies one
`PipelineConfig` field across a range of values and records:

- **Speed** -- total wall plus per-stage wall (render, layout, blocks,
  table, formula, reference, bundle).
- **Resources** -- process-tree CPU%, host RSS, and device VRAM, each as
  min / median / p95 / max sampled every 250 ms during the run.
- **Quality** -- block counts, text characters, MISS fallbacks, formula
  LaTeX recovery, table cells recovered, references parsed. The
  most-relevant subset is surfaced per sweep.

Apples-to-apples: every sweep starts from a fixed base `PipelineConfig`
(`benchmarks/sweeps.py:DIGITAL_BORN_BASE` or `SCANNED_BASE`) with every
non-essential stage disabled, and overrides only the knob under test.

## Layout

```text
benchmarks/results/
├── summary.md                   # index of every sweep
├── sweeps/
│   ├── <sweep>.csv              # one row per (knob value, run)
│   └── <sweep>.md               # human-readable summary
├── plots/
│   └── <sweep>.png              # speed / stages / resources / quality dashboard
├── bundles/                     # transient; one .tar per row, deleted post-inspection
└── api_smoke/                   # transient; output of api_smoke.py
```

The committed reference results (`summary.md`, `sweeps/`, `plots/`) live in git; `bundles/`, `api_smoke/`, and `*.log` are gitignored transients regenerated on every run.

## Running

```bash
# Every sweep (skips ones with cached CSVs):
python benchmarks/run_all.py

# Just the sweeps whose name contains "formula":
python benchmarks/run_all.py --only formula

# Force re-run even if the CSV exists:
python benchmarks/run_all.py --force

# List defined sweeps:
python benchmarks/run_all.py --list

# After CSVs are written, generate plots:
python -m benchmarks.plot
```

The runner is resume-friendly: a sweep with an existing
`benchmarks/results/sweeps/<name>.csv` is skipped unless `--force` is
passed.

## Smoke harness (`api_smoke.py`)

`benchmarks/api_smoke.py` is the end-to-end FastAPI smoke harness, kept
separate from the knob sweeps. It spawns the service as a subprocess,
posts each entry in `TARGETS`, captures the same resource / per-stage
metrics, and writes a single JSON report. Use it to validate a
deployment, not to measure individual knobs.

## Sweep catalogue

### Standard knob sweeps

| Sweep | Knob | Values | PDF |
|---|---|---|---|
| `layout_batch_size` | `layout_batch_size` | 1, 2, 4, 8, 16, 24 | digital-born |
| `layout_fp16` | `layout_fp16` | False, True | digital-born |
| `layout_fast_preproc` | `layout_fast_preproc` | False, True | digital-born |
| `layout_confidence` | `layout_confidence` | 0.3, 0.4, 0.5, 0.6, 0.7 | digital-born |
| `render_dpi_digital_born` | `render_dpi_digital_born` | 100, 150, 200, 250, 300 | digital-born |
| `render_workers` | `render_workers` | 1, 2, 4, 8, 16 | digital-born |
| `page_format` | `page_format` | png, jpeg | digital-born |
| `crop_format` | `crop_format` | png, jpeg | digital-born + formula |
| `bundle_miss_images_for` | `bundle_miss_images_for` | {}, {formula}, all | digital-born + formula |
| `jpeg_quality` | `jpeg_quality` | 40, 70, 90, 100 | Turkish scanned |
| `digital_born_workers` | `digital_born_workers` | 1, 2, 4, 8, 16, 32, 64 | digital-born |
| `formula_enabled` | `formula_enabled` | False, True | digital-born |
| `formula_model_id` | `formula_model_id` | L, plus-L | digital-born |
| `formula_dtype` | `formula_dtype` | fp32, fp16 | digital-born |
| `formula_batch_size` | `formula_batch_size` | 4, 8, 16, 24, 32 | digital-born |
| `formula_bucketed` | `formula_bucketed` | False, True | digital-born |
| `formula_torch_compile` | `formula_torch_compile` | False, True | digital-born |
| `formula_bucket_thresholds` | `bucket_thresholds` (multi-knob) | default, wider_small, wider_medium, tighter, scaled_2x | digital-born + formula |
| `dpi_bucket_interaction` | `dpi_x_buckets` (multi-knob) | 150_default, 300_default, 150_scaled_4x, 300_scaled_4x | digital-born + formula |
| `table_enabled` | `table_enabled` | False, True | digital-born |
| `table_batch_size` | `table_batch_size` | 1, 2, 4, 8, 16 | digital-born |
| `table_min_side_px` | `table_min_side_px` | 60, 120, 180, 240 | digital-born + table |
| `references_enabled` | `references_enabled` | False, True | digital-born (needs GROBID) |
| `render_dpi_scanned` | `render_dpi` | 100, 150, 200, 250, 300 | scanned |
| `ocr_batch_size` | `ocr_batch_size` | 4, 6, 16, 64, 128 | scanned |
| `ocr_use_cuda` | `ocr_use_cuda` | True | scanned (CPU value omitted; ~1 h projected wall) |
| `ocr_workers` | `ocr_workers` | 1, 2, 4, 6, 8 | scanned |
| `ocr_min_score` | `ocr_min_score` | 0.3, 0.4, 0.5, 0.6, 0.7 | scanned |

### Methodology sweeps

These run via the same `run_all.py` machinery but cross multiple PDFs or repeat one config.

| Sweep | Knob | Values | PDF |
|---|---|---|---|
| `production_realistic_digital_born` | `stages` (multi-knob) | minimal, all_on | digital-born (needs GROBID for `all_on`) |
| `production_realistic_scanned` | `stages` (multi-knob) | minimal, all_on | scanned (needs GROBID for `all_on`) |
| `variance` | `repeat` (sentinel) | 1-10 | digital-born (needs GROBID) |
| `cross_corpus` | `pdf` (sentinel) | 904599, 995802, 1005465, 084016, 101123, 823568 | mixed digital-born + scanned, en/tr/ar (needs GROBID) |
| `language_matrix` | `language` (sentinel) | en, tr, ar | scanned en/tr/ar (needs GROBID) |

### Standalone benchmark scripts

These don't fit the per-knob `run_all.py` orchestrator (out-of-band concerns: model init, sustained load, API concurrency, GROBID payload scaling, torch.compile amortisation). Each writes a sweep-shaped CSV + Markdown to `benchmarks/results/sweeps/<name>.{csv,md}` so `summary.md` picks them up like any other sweep.

| Script | What it measures |
|---|---|
| `python -m benchmarks.cold_start` | Wall + cumulative VRAM after each of `LayoutAnalyzer` / `FormulaRecognizer` / `TableEngine` `__init__`. Pins the FastAPI `boot_timeout_s` claim. |
| `python -m benchmarks.sustained_load --calls 15` | N sequential `process_pdf` calls in one Python process; two arms (`PYTORCH_CUDA_ALLOC_CONF` legacy vs `expandable_segments:True`). Characterises allocator drift across a long-lived service. |
| `python -m benchmarks.api_concurrency` | N=1, 2, 4, 8 concurrent clients posting the same PDF to the FastAPI service. Throughput + p50/p95/p99 server-side wall. Validates the `gpu_lock` serialisation design. |
| `python -m benchmarks.grobid_payload_scaling` | Citation lists of 10 / 50 / 200 / 500 entries posted to GROBID directly. Right-sizes `grobid_timeout_s` for large bibliographies. |
| `python -m benchmarks.torch_compile_amortization --calls 25` | 25 sequential PDFs in two arms (`formula_torch_compile=False` vs `True`). Reports the crossover N where compile overtakes eager. |

## OOM handling

Each row is wrapped: `torch.OutOfMemoryError`, CUDA-OOM-shaped
`RuntimeError`, and any other exception are caught and the row records
`status="oom"` or `status="error"` plus the exception message. The
sweep continues with the next value. CUDA cache is cleared between
rows so OOM in row N doesn't leak into row N+1.

## Test PDFs

The sweep PDFs live under `samples/` at the project root (committed):
`904599.pdf` (digital-born English thesis), `084016.pdf` (scanned
English thesis), `101123.pdf` (Turkish scanned thesis), etc. -- see
`samples/README.md` for the full index with language / quality notes.
Replace with your own corpus by dropping PDFs into `samples/` and
editing the filenames at the top of `benchmarks/sweeps.py`.

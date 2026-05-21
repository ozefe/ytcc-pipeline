# Configuration

The pipeline is configured at three layers, listed from broadest to narrowest:

1. **`config.toml`** at the project root -- the single source of truth for the FastAPI service and the benchmarks. Loaded by `load_service_config()`.
2. **`YTCC_*` environment variables** -- per-field overrides, useful for tests and quick deployment tweaks.
3. **`PipelineConfig(...)` keyword arguments** -- explicit, per-call. Library mode talks to the pipeline through this.

> [!IMPORTANT]
> The three layers don't compose automatically. The TOML and env vars are read **only** by `load_service_config()` (TOML) and `PipelineConfig.from_env()` (env). If you call `process_pdf(config=PipelineConfig(...))` directly, neither TOML nor env vars apply. Use `dataclasses.replace(loaded.pipeline, ...)` if you want to layer overrides on top of a TOML-loaded config.

## The two config types

```python
from ytcc_pipeline import PipelineConfig, ServiceConfig, load_service_config
```

- **`PipelineConfig`** -- frozen slots dataclass. Every knob `process_pdf` honours lives here. Defaults are **library-safe** (PNG output, single worker, no fp16, formula on but conservative). Used directly in library mode.
- **`ServiceConfig`** -- aggregates `PipelineConfig`, `ApiSettings`, and `LoggingSettings`. Loaded from `config.toml` by `load_service_config()`. Used by the service and the benchmark harness.

`ApiSettings` carries `host`, `port`, `boot_timeout_s`. `LoggingSettings` carries `level`, `pdf_oxide_level`, `format` and exposes `.apply()` to install handlers at the application boundary.

## TOML resolution order

`load_service_config()` walks the following sequence:

1. The `path` argument, if explicit.
2. `YTCC_CONFIG=/path/to/config.toml` environment variable.
3. `./config.toml` (current working directory).
4. `config.toml` next to the installed package's project root (only when installed editable; falls through for wheel installs).
5. Built-in dataclass defaults.

```python
cfg = load_service_config()                              # auto
cfg = load_service_config("/etc/ytcc/prod.toml")         # explicit
```

> [!CAUTION]
> Unknown TOML keys in `[pipeline]`, `[api]`, or `[logging]` raise `ValueError`. Typos don't pass silently.

Missing sections fall back to dataclass defaults. TOML has no `null`, so any optional-with-`None` field (notably `render_workers`) is expressed by **omitting the key**.

## Environment variables

```python
cfg = PipelineConfig.from_env()
```

Every `PipelineConfig` field has a corresponding `YTCC_<UPPERCASE_NAME>` env var. Unset variables keep the dataclass default. The full table:

| Field | Env var | Type |
|-------|---------|------|
| `render_dpi` | `YTCC_RENDER_DPI` | int |
| `render_dpi_digital_born` | `YTCC_RENDER_DPI_DIGITAL_BORN` | int |
| `page_format` | `YTCC_PAGE_FORMAT` | str (`jpeg` / `png`) |
| `crop_format` | `YTCC_CROP_FORMAT` | str (`jpeg` / `png`) |
| `jpeg_quality` | `YTCC_JPEG_QUALITY` | int (1-100) |
| `layout_confidence` | `YTCC_LAYOUT_CONFIDENCE` | float |
| `layout_batch_size` | `YTCC_LAYOUT_BATCH_SIZE` | int |
| `layout_device` | `YTCC_LAYOUT_DEVICE` | str |
| `layout_fp16` | `YTCC_LAYOUT_FP16` | bool |
| `layout_fast_preproc` | `YTCC_LAYOUT_FAST_PREPROC` | bool |
| `ocr_batch_size` | `YTCC_OCR_BATCH_SIZE` | int |
| `ocr_min_score` | `YTCC_OCR_MIN_SCORE` | float |
| `ocr_use_cuda` | `YTCC_OCR_USE_CUDA` | bool |
| `ocr_workers` | `YTCC_OCR_WORKERS` | int |
| `digital_born_sample_pages` | `YTCC_DIGITAL_BORN_SAMPLE_PAGES` | int |
| `digital_born_text_ratio` | `YTCC_DIGITAL_BORN_TEXT_RATIO` | float |
| `digital_born_min_text_chars` | `YTCC_DIGITAL_BORN_MIN_TEXT_CHARS` | int |
| `digital_born_workers` | `YTCC_DIGITAL_BORN_WORKERS` | int |
| `render_workers` | `YTCC_RENDER_WORKERS` | int |
| `retain_temp_dir` | `YTCC_RETAIN_TEMP_DIR` | bool |
| `scanned_enabled` | `YTCC_SCANNED_ENABLED` | bool |
| `bundle_miss_images_for` | `YTCC_BUNDLE_MISS_IMAGES_FOR` | comma list |
| `formula_enabled` | `YTCC_FORMULA_ENABLED` | bool |
| `formula_model_id` | `YTCC_FORMULA_MODEL_ID` | str |
| `formula_device` | `YTCC_FORMULA_DEVICE` | str |
| `formula_dtype` | `YTCC_FORMULA_DTYPE` | str (`fp16`/`fp32`) |
| `formula_batch_size` | `YTCC_FORMULA_BATCH_SIZE` | int |
| `formula_max_new_tokens` | `YTCC_FORMULA_MAX_NEW_TOKENS` | int |
| `formula_torch_compile` | `YTCC_FORMULA_TORCH_COMPILE` | bool |
| `formula_bucketed` | `YTCC_FORMULA_BUCKETED` | bool |
| `formula_bucket_small_threshold` | `YTCC_FORMULA_BUCKET_SMALL_THRESHOLD` | int |
| `formula_bucket_medium_threshold` | `YTCC_FORMULA_BUCKET_MEDIUM_THRESHOLD` | int |
| `formula_bucket_small_tokens` | `YTCC_FORMULA_BUCKET_SMALL_TOKENS` | int |
| `formula_bucket_medium_tokens` | `YTCC_FORMULA_BUCKET_MEDIUM_TOKENS` | int |
| `table_enabled` | `YTCC_TABLE_ENABLED` | bool |
| `table_batch_size` | `YTCC_TABLE_BATCH_SIZE` | int |
| `table_device` | `YTCC_TABLE_DEVICE` | str |
| `table_min_side_px` | `YTCC_TABLE_MIN_SIDE_PX` | int |
| `references_enabled` | `YTCC_REFERENCES_ENABLED` | bool |
| `grobid_url` | `YTCC_GROBID_URL` | str |
| `grobid_timeout_s` | `YTCC_GROBID_TIMEOUT_S` | float |
| `reference_labels` | `YTCC_REFERENCE_LABELS` | comma list |

Bool parser accepts `1`/`true`/`yes`/`on` (case-insensitive). Anything else is `False`.

Comma-list fields strip whitespace and drop empties: `"text, , formula,"` is `{"text", "formula"}`.

## Field reference (selected)

Per-field tuning guidance and benchmark history live alongside each field in `PipelineConfig` and the inline `config.toml` comments. The narrative ones below are common sources of surprise.

### `render_dpi` vs `render_dpi_digital_born`

The pipeline applies different render resolutions to scanned vs digital-born PDFs:

- **Scanned** (`render_dpi`, default 300): OCR loses ~3% of text blocks at 150 DPI, so the scanned path stays at 300.
- **Digital-born** (`render_dpi_digital_born`, default 150): the layout detector downsamples to 800x800 internally and `pdf_oxide` text extraction is resolution-independent, so 150 DPI matches 300 DPI on character recall (~99.9%) while halving wall-clock.

The orchestrator picks at runtime in `_run_stages`:

```python
if digital_born and cfg.render_dpi_digital_born != cfg.render_dpi:
    cfg = replace(cfg, render_dpi=cfg.render_dpi_digital_born)
```

After that point, `cfg.render_dpi` is the effective DPI. Downstream code -- `pdf_io.text` for pixel->point conversion, `Page.width_px`/`height_px`, every `bbox` field -- all read this single field.

### `page_format` vs `crop_format`

| Knob | Controls | Default |
|------|----------|---------|
| `page_format` | throwaway temp page renders consumed by the layout model | `png` (library) / `jpeg` (service) |
| `crop_format` | crops bundled in `images/` | `png` |
| `jpeg_quality` | shared quality knob; ignored for PNG | 95 (library) / 90 (service) |

JPEG temp renders are ~30% faster on the digital-born path because the layout model downsamples to 800x800 -- the JPEG artefacts get erased before they could affect detection. PNG crops keep the bundle lossless. Don't switch `crop_format` to JPEG unless you accept quality loss.

### `bundle_miss_images_for`

Block types for which a MISS fallback crop is bundled when extraction fails. Defaults to every type. Set to `[]` to disable MISS images entirely -- the block's JSON entry still appears with `miss=true`, `text=null`, `image_path=null` so consumers retain its reading order and bbox.

### `scanned_enabled`

Master toggle. When `False`, any PDF that resolves to scanned (explicit `digital_born=False` **or** auto-detected as scanned) is rejected before any OCR worker spawns:

- `process_pdf` raises `ValueError`.
- The FastAPI service returns HTTP 415.

Off keeps the ~2 GiB-per-worker OCR engines from ever loading and lets the resident layout analyzer stay loaded across requests. Total VRAM peak on a 3090 drops from ~25 GiB to ~5 GiB.

### `references_enabled` + `grobid_url`

References require an externally-managed GROBID server. The pipeline never spawns the JVM. When the stage is on, the lifespan health-probes `cfg.grobid_url`. Probe failure is logged at WARNING but does not block startup; the reference stage logs-and-skips per request until the server comes back.

### `reference_labels`

Layout labels routed to GROBID. Default `("reference", "reference_content")`. Narrow to `("reference_content",)` if `reference` blocks tend to be multi-reference blobs in your corpus (GROBID returns a degraded single parse on those). Extend if a new bibliography-shaped label appears.

## Logging

The library never configures handlers. The application boundary -- FastAPI lifespan, your own script -- calls `LoggingSettings.apply()` once:

```python
from ytcc_pipeline import load_service_config

service = load_service_config()
service.logging.apply()                  # installs root handler + per-logger levels
```

`pdf_oxide` emits per-font internals at INFO (~50 lines per PDF, useless for operators), so `LoggingSettings` pins its tree to WARNING separately. The pipeline itself logs one INFO line per stage; DEBUG adds per-block detail (thousands of lines per PDF).

## Override patterns

Layer the TOML, then patch per-call:

```python
from dataclasses import replace
from ytcc_pipeline import load_service_config, process_pdf

service = load_service_config()
cfg = service.pipeline

# Want to process this specific PDF without formulas:
cfg_no_formulas = replace(cfg, formula_enabled=False)
bundle = process_pdf("paper.pdf", language="en", config=cfg_no_formulas)
```

Or override fully at the env level:

```bash
YTCC_FORMULA_ENABLED=false \
YTCC_LAYOUT_FP16=true \
python -c "from ytcc_pipeline import PipelineConfig, process_pdf; \
process_pdf('paper.pdf', language='en', config=PipelineConfig.from_env())"
```

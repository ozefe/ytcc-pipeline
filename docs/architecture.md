# Architecture

`ytcc-pipeline` runs the same eight stages on every PDF, with three stages opt-in. This document maps the package layout to those stages and explains the design decisions that hold the pipeline together.

## Stage flow

```text
render -> metadata -> layout -> blocks -> table -> formula -> reference -> bundle
```

1. **render** -- decode every page of the PDF to an image file. Parallelised by `cfg.render_workers`.
2. **metadata** -- sha256, byte size, and XMP fields for `document.json`. Cheap; runs early so I/O failures surface before model load.
3. **layout** -- PP-DocLayoutV3 over rendered pages, emits one `LayoutDetection` per detected block.
4. **blocks** -- per-page dispatcher that routes each detection through extract / crop / save. Three execution paths share one output shape (see the per-page execution paths section below).
5. **table** (opt-in via `cfg.table_enabled`) -- RapidTable SLANet+ recovers cell grids for `TABLE` blocks.
6. **formula** (opt-in via `cfg.formula_enabled`, on by default) -- PP-FormulaNet-L runs over every `FORMULA` block's crop to recover LaTeX.
7. **reference** (opt-in via `cfg.references_enabled`) -- batched HTTP call to an external GROBID server enriches `REFERENCE` blocks with parsed `Reference` objects.
8. **bundle** -- pack `document.json` + `images/*` into one uncompressed tar.

Each stage emits one INFO log line on completion and skip cases short-circuit with a `skipped reason=...` line. Disabled stages never load their model.

## Package layout

The package is split into thematic subpackages:

- **Top level.** Public-API factories (`process_pdf`, `PipelineConfig`, `FormulaRecognizer`, `load_service_config`), the `BlockType` enum, the routing layer that maps PP-DocLayoutV3 labels to `Route` values, the `Document` / `Page` / `Block` / `Cell` / `Reference` schema, the tar bundler, and the cv2 image-I/O wrapper.
- **`pdf_io`.** Every `pdf_oxide` call site: page rendering, text-by-bbox extraction with pixel-to-PDF-point conversion, sha256 + XMP metadata, and the digital-born / scanned heuristic.
- **`models`.** ML model wrappers, one module per external library: PP-DocLayoutV3 (the SafeTensors backend behind the `LayoutAnalyzer` Protocol), RapidOCR (`OcrExtractor`), PP-FormulaNet-L (`FormulaRecognizer` plus the `BucketSpec` and `FormulaResult` dataclasses), and the GROBID HTTP client plus TEI parser.
- **`processors`.** Per-block-type orchestration: TEXT / REFERENCE text extraction, IMAGE / FORMULA / TABLE crop-and-save, plus the three opt-in stage runners (`run_formula_stage`, `run_table_stage`, `run_reference_stage`).
- **`pipeline`.** The orchestrator (`process_pdf` entry point + `_run_stages` sequencer), the per-page dispatcher that chooses serial vs parallel-digital vs parallel-scanned, the spawn-pool worker entry points, the `build_block` decision-point, and the picklable IPC payload tuples.
- **`api`.** Optional FastAPI app exposing `/process` and `/health`, with a lifespan handler that loads the resident models once.

## Per-page execution paths

The block stage has three implementations chosen by `cfg.digital_born_workers`, `cfg.ocr_workers`, and the `digital_born` verdict:

| Path | When | What runs |
|------|------|-----------|
| serial | both worker counts are 1 | single-process loop in the main process |
| parallel-digital | `digital_born=True` and `digital_born_workers > 1` | spawn pool; each worker opens its own `pdf_oxide.PdfDocument` |
| parallel-scanned | `digital_born=False` and `ocr_workers > 1` | spawn pool; each worker lazily owns one `OcrExtractor` (RapidOCR + CUDA context) |

All three produce the same shape -- one `Page` per source page, blocks in per-page reading order.

> [!IMPORTANT]
> Workers use `multiprocessing.spawn`, not `fork`. The parent process may hold a CUDA context (the resident layout analyzer in service mode), and `fork` corrupts that. `spawn` re-imports the worker module from scratch, which is why `pdf_oxide` is imported inside the digital-born worker entry rather than at module top -- it keeps the parent from paying the import cost just to spawn workers.

## Routing

Each PP-DocLayoutV3 label (`"abstract"`, `"display_formula"`, `"table"`, etc.) is mapped to one of six `Route` values:

| Route | Labels | Treatment |
|-------|--------|-----------|
| `TEXT` | `abstract`, `content`, `doc_title`, `formula_number`, `paragraph_title`, `text`, ... | Extract text into `block.text` |
| `IMAGE` | `algorithm`, `chart`, `image` | Crop and save; `text=None` |
| `TABLE` | `table` | Crop and save; cell grid filled by table stage |
| `FORMULA` | `formula`, `display_formula`, `inline_formula` | Crop and save; LaTeX filled by formula stage |
| `REFERENCE` | `reference`, `reference_content` | Extract text into `block.text`; enriched by reference stage |
| `IGNORE` | `header`, `footer`, `number`, `seal`, ... | Dropped before block assembly |

Unknown labels fall back to `IMAGE` and are logged once per process at WARNING. The label set lives in the `routing` module -- update it if you fine-tune the layout model on a new corpus.

`formula_number` (equation labels like `"(3.2)"`) is routed to `TEXT`, not `FORMULA`. Short text strings are faster and more accurate to extract via `pdf_oxide` / `RapidOCR` than to push through a 700 MB transformer.

## Resource lifecycle

Three external resources have a meaningful load cost: the layout analyzer (~5s), the formula recognizer (~3s + 20-30s if `torch_compile=True`), and the RapidTable engine (~1s). Each is **injectable**:

```python
from ytcc_pipeline.models.layout import make_analyzer_from_config
from ytcc_pipeline.models.formula import make_formula_recognizer
from ytcc_pipeline.processors.table import make_table_engine

analyzer = make_analyzer_from_config(cfg)
formula = make_formula_recognizer(cfg)
table = make_table_engine(device=cfg.table_device, batch_size=cfg.table_batch_size)

for pdf in batch_of_pdfs:
    process_pdf(pdf, language="en", config=cfg,
                analyzer=analyzer, formula_recognizer=formula, table_engine=table)
```

When **injected**, `process_pdf` reuses the resource across calls and leaves cleanup to the caller. When **`None`**, a fresh instance is loaded per call and closed before return. The FastAPI service injects all three from `app.state`; library callers either inject manually (batch case above) or let the orchestrator own the per-call lifecycle (simple case).

The `close()` methods on `SafeTensorsLayoutAnalyzer` and `FormulaRecognizer` are idempotent: a second call is a no-op rather than raising.

> [!CAUTION]
> The FastAPI lifespan loads models on import. Running `uvicorn` with `--workers N > 1` multi-loads them and contends for VRAM. Stick to one uvicorn worker per GPU and serialise concurrency at the lock (already done).

## Why three stages stay opt-in

- **`formula_enabled`** (default `True`) -- always wanted in production. The default is `True` because the LaTeX is the only useful representation of an equation downstream.
- **`table_enabled`** (default `False`) -- SLANet+ adds ~1 GiB VRAM and recovers a structured grid; many consumers only need the table crop. Off by default to keep the minimum-VRAM footprint small.
- **`references_enabled`** (default `False`) -- requires running a GROBID server externally. Off by default so the pipeline has no service dependency in the simple case.

## Bundle stability

`document.json` is written **first** inside the tar. Streaming consumers can parse the index without buffering image bytes, and the bundle is one-pass-readable on a pipe. PNG / JPEG crops follow in alphabetical order. Both `document.json` and the per-block JSON entries are sorted deterministically -- byte-for-byte identical bundles across runs of the same input + config are not guaranteed (`uuid4` filenames, dict iteration order in `pdf_info`), but the schema-level content is.

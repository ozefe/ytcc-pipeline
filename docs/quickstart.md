# Quickstart

## Install

`ytcc-pipeline` requires Python 3.14+ and a CUDA GPU for any non-trivial throughput. CPU is supported but neither tested nor recommended.

```bash
pip install -e .                # library only
pip install -e ".[api]"         # + FastAPI service
pip install -e ".[dev]"         # + tests + lint + typing + benchmarks
```

> [!NOTE]
> The `dev` dependency group transitively includes `api`, `test`, `lint`, `typing`, and `benchmarks`.

Production wheels pin `onnxruntime-gpu` and CUDA-enabled `torch`. Drop or substitute these dependencies if you target CPU-only environments -- nothing else in the codebase assumes CUDA at import time, but every model defaults to `cuda:0` and the recommended config enables CUDA-only flags (`layout_fp16`, `ocr_use_cuda`).

## Library mode

```python
from ytcc_pipeline import process_pdf

bundle_path = process_pdf("paper.pdf", language="en")
print(bundle_path)              # -> paper.tar
```

`process_pdf` is the single public entry point. It is **synchronous** and **blocking** -- internally it uses `multiprocessing.spawn` pools, not asyncio. Calling it from an event loop requires offloading via `asyncio.to_thread` (the bundled FastAPI service does exactly this).

The function signature in full:

```python
process_pdf(
    pdf_path,                   # Path or str
    language,                   # ISO 639-1, e.g. "en", "tr", "ar"
    *,
    digital_born=None,          # None=auto-detect, True=text layer, False=OCR
    output_path=None,           # defaults to <pdf-stem>.tar
    config=None,                # PipelineConfig; defaults to PipelineConfig()
    analyzer=None,              # pre-loaded LayoutAnalyzer for service reuse
    formula_recognizer=None,    # pre-loaded FormulaRecognizer
    table_engine=None,          # pre-loaded TableEngine
) -> Path
```

Runnable examples are bundled in the `examples/` directory at the repo root.

## Service mode

```bash
uvicorn ytcc_pipeline.api.app:app --host 0.0.0.0 --port 8000
```

The service loads its config from the project TOML at the repo root (override with the `YTCC_CONFIG` environment variable). One process, one GPU, one PDF at a time -- the layout analyzer is loaded once during the FastAPI lifespan and reused across requests.

```bash
curl -X POST http://localhost:8000/process \
  -F "pdf=@paper.pdf" \
  -F "language=en" \
  -o paper.tar
```

## Reading the bundle back

The bundle is a plain uncompressed tar with `document.json` first and `images/*` after it:

```python
import json, tarfile

with tarfile.open("paper.tar") as tf:
    doc = json.loads(tf.extractfile("document.json").read())

for page in doc["pages"]:
    for block in page["blocks"]:
        print(block["reading_order"], block["type"], block["text"][:60] if block["text"] else "")
```

## Recommended performance config

```python
from ytcc_pipeline import PipelineConfig, process_pdf

config = PipelineConfig(
    layout_fp16=True,
    layout_fast_preproc=True,
    ocr_use_cuda=True,
    digital_born_workers=16,
    ocr_workers=6,
    page_format="jpeg",
    jpeg_quality=90,
)

bundle = process_pdf("paper.pdf", language="en", config=config)
```

Expect ~15s end-to-end for a 150-page English digital-born thesis on an RTX 3090 (vs ~85s with `PipelineConfig()`). The auto-DPI lever (150 DPI for digital-born) is on by default and contributes most of the win.

## Sanity check

The repo ships six sample PDFs under `samples/` covering English / Turkish / Arabic, digital-born and scanned, good and bad quality. Try a digital-born thesis first -- it exercises every stage except OCR.

A typical good run logs one INFO line per stage (`stage render`, `stage layout`, `stage blocks`, `stage table`, `stage formula`, `stage reference`, `stage bundle`) plus a per-PDF summary.

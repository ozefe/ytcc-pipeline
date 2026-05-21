# ytcc-pipeline

A synchronous Python library that turns an academic-thesis PDF into a structured JSON document plus a tar bundle of cropped figures, tables, and formulas. Ships an optional FastAPI wrapper for service deployments.

## What it does

Given a PDF, the pipeline:

1. Renders every page to an image.
2. Detects layout blocks (text, image, table, formula, reference, header/footer, ...) with PP-DocLayoutV3.
3. Routes each block: pulls text from the PDF text layer for digital-born documents or runs RapidOCR on the rendered crop for scanned ones; saves cropped images for figures, tables, and formulas.
4. Optionally recognises LaTeX from formula crops with PP-FormulaNet-L.
5. Optionally recovers table cell grids with RapidTable SLANet+.
6. Optionally enriches bibliography blocks with parsed `Reference` records via an externally-managed GROBID server.
7. Packs `document.json` and every saved crop into one streaming-first uncompressed tar.

## Quickstart

```bash
pip install -e .                # library only
pip install -e ".[api]"         # + FastAPI service
pip install -e ".[dev]"         # + tests + lint + typing + benchmarks
```

```python
from ytcc_pipeline import process_pdf

bundle_path = process_pdf("paper.pdf", language="en")
# -> paper.tar  (contains document.json + images/*)
```

To run the FastAPI service:

```bash
uvicorn ytcc_pipeline.api.app:app --host 0.0.0.0 --port 8000
```

## Docker

Pre-built images for four deployment profiles (`scanned`, `digital-born`, `digital-born-a100`, and the text-only `text-extract-a100`) are published to GitHub Container Registry, each with a slim and a pre-baked variant. The fastest path to a running service is the bundled compose file:

```bash
docker compose -f docker/compose.scanned.yml up -d
```

Image matrix, configuration overrides, and per-profile guidance live under `docker/`.

## Requirements

- Python 3.14+.
- A CUDA GPU for any non-trivial throughput. CPU is supported but neither tested nor recommended.

## Documentation

The `docs/` directory holds reference material: quickstart, architecture, output format, configuration, per-stage behaviour, performance tuning, the digital-born vs scanned distinction, API service usage, GROBID setup, and a gotchas index.

## License

MIT.

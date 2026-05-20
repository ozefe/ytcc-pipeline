"""FastAPI wrapper around `process_pdf`.

This package is optional. Install the `api` extra (`pip install -e ".[api]"`) to pick up
FastAPI, uvicorn, and python-multipart.

Run with:
    uvicorn ytcc_pipeline.api.app:app --host 0.0.0.0 --port 8000

The lifespan handler loads the `LayoutAnalyzer`, the `FormulaRecognizer` (when
`formula_enabled`), and the `TableEngine` (when `table_enabled`) once and reuses them
across requests; concurrent requests are serialized through an `asyncio.Lock` to keep
VRAM contention out of the GPU. When `references_enabled`, the lifespan also probes the
configured GROBID URL once and logs whether it's reachable -- the server itself is
external and is never spawned by this process.
"""

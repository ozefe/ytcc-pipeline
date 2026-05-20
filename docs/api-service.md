# API service

The `ytcc_pipeline.api.app:app` FastAPI module wraps `process_pdf` behind two HTTP endpoints. This page documents the request/response contract, the lifespan model-loading dance, and the concurrency model.

## Install + run

```bash
pip install -e ".[api]"
uvicorn ytcc_pipeline.api.app:app --host 0.0.0.0 --port 8000
```

The service reads its configuration from `config.toml` -- override the path with `YTCC_CONFIG=/path/to/config.toml`. The bound host/port can be set in `[api]` (`host`, `port`), but uvicorn's CLI flags take precedence at process start.

> [!CAUTION]
> Run **one** uvicorn worker per GPU. Multi-worker uvicorn (`--workers N>1`) multi-loads every resident model and contends for VRAM. The service is designed for a single-worker, single-GPU deployment with concurrency handled at the lock layer.

## Endpoints

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true
}
```

Liveness + readiness probe. `model_loaded` reflects whether the resident layout analyzer is currently loaded (`False` while the scanned-path handler is between analyzer close and reload).

### `POST /process`

`multipart/form-data` with:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pdf` | file | yes | The source PDF. |
| `language` | string | yes | ISO 639-1 code, e.g. `"en"`, `"tr"`, `"ar"`. Selects the OCR model for scanned PDFs. |
| `digital_born` | bool | no | `true` / `false` to override auto-detect. Omit for auto. |

**Response:** `application/x-tar` body, filename `<stem>.tar` matching the uploaded PDF.

**Response headers:**

- `X-Processing-Time: <seconds>` -- server-side processing wall (not including upload streaming or download). Clients use this to separate end-to-end vs server-side timing without re-instrumenting.

**Error responses:**

| Status | Cause |
|--------|-------|
| 400 | Unsupported `language`; `process_pdf` raised `FileNotFoundError` or `ValueError`. |
| 415 | Scanned-disabled deployment (`scanned_enabled=false`) and the PDF resolves to scanned. |
| 500 | Any other `process_pdf` exception. The traceback is logged via `logger.exception`. |

```bash
curl -X POST http://localhost:8000/process \
  -F "pdf=@paper.pdf" \
  -F "language=en" \
  -F "digital_born=true" \
  -D headers.txt \
  -o paper.tar

grep X-Processing-Time headers.txt
```

Or in Python with `httpx`:

```python
import httpx

with open("paper.pdf", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/process",
        files={"pdf": ("paper.pdf", f, "application/pdf")},
        data={"language": "en"},
        timeout=300,    # default 5s is too short for non-trivial PDFs
    )

response.raise_for_status()
with open("paper.tar", "wb") as out:
    out.write(response.content)
print("server processing time:", response.headers["X-Processing-Time"])
```

> [!IMPORTANT]
> `httpx` defaults to a 5-second timeout. Set `timeout=` to something realistic (300s+) for any non-trivial PDF.

## Concurrency model

The service is **a polite one-PDF-at-a-time queue**:

1. Requests arrive on the asyncio event loop. The event loop is free to accept and buffer uploads while processing is in progress.
2. The handler awaits `app.state.gpu_lock` (an `asyncio.Lock`) before touching the GPU.
3. Inside the lock, the synchronous `process_pdf` runs on a thread (`asyncio.to_thread`) so the event loop can keep accepting new uploads.
4. When the response has been written, the lock releases and the next queued request proceeds.

If lock wait exceeds 500 ms an INFO line is logged (`request queued: rid=... lock_wait_s=...`). Each request gets an 8-char hex correlation id (`rid=`) on every log line so concurrent transcripts can be untangled.

## Lifespan: model lifecycle

The `lifespan` asynccontextmanager runs once at startup and once at shutdown.

**Startup:**

1. Load `config.toml` via `load_service_config()`.
2. Apply logging (`service_config.logging.apply()`).
3. Construct the layout analyzer (`make_analyzer_from_config`).
4. Construct the formula recognizer if `formula_enabled` -- else `None`.
5. Construct the table engine if `table_enabled` -- else `None`.
6. Probe GROBID if `references_enabled` -- log WARNING on failure but **never** block startup.
7. Allocate the `asyncio.Lock`.

All three resources are stored in `app.state.{analyzer, formula_recognizer, table_engine, gpu_lock, config}` and reused across every request.

**Shutdown:**

- Close the analyzer + formula recognizer (releases VRAM).
- Drop the table engine reference -- RapidTable has no `close()`, VRAM is reclaimed at process exit.

## The scanned-path VRAM dance

The trickiest part. For scanned PDFs the handler temporarily closes the resident analyzer so OCR workers can claim its VRAM:

```python
if digital_born:
    bundle = await asyncio.to_thread(process_pdf, ..., analyzer=app.state.analyzer)
else:
    app.state.analyzer.close()
    try:
        bundle = await asyncio.to_thread(process_pdf, ..., digital_born=False)
        # ^ note: no analyzer= kwarg; process_pdf builds + closes a fresh one
        # before block processing so OCR workers can claim its VRAM.
    finally:
        app.state.analyzer = await asyncio.to_thread(make_analyzer_from_config, cfg)
```

The reload happens **before** the lock releases, so the next request finds a warm analyzer. The formula recognizer stays loaded for both paths -- its ~1.8 GiB footprint fits alongside the OCR workers.

When `scanned_enabled=false`, this dance never runs because the handler 415s scanned requests before reaching the dispatch.

## Background cleanup

Temp directories (`/tmp/ytcc_api_*`) are deleted after the response has streamed via FastAPI's `BackgroundTasks`. Cleanup failures are logged at WARNING but never re-raised -- a leaked temp dir is preferable to a 500 on a request that already succeeded.

## Smoke harness

The repo ships an API smoke harness under the `benchmarks` package. It spawns the service as a subprocess and POSTs each configured target PDF, capturing the same resource / per-stage metrics as the knob sweeps but end-to-end through HTTP. Use it to validate a deployment, not to measure individual knobs.

## Operational notes

- **Boot time** -- cold model load takes ~5 s on an idle 3090 (analyzer load dominates). With `formula_torch_compile=true` add ~20-30 s warmup. The `boot_timeout_s` in `[api]` is generous (120 s).
- **Idle VRAM** -- once loaded, the analyzer + formula recognizer hold ~5 GiB resident. `scanned_enabled=false` keeps total idle peak at ~5 GiB; with scanned support enabled, transient peak during a scanned request hits ~14 GiB.
- **Log noise** -- DEBUG adds thousands of per-block lines per request. Keep `[logging] level = "INFO"` in production.
- **Restart on config change** -- `config.toml` is only read in `lifespan` at startup. SIGHUP / hot reload is not supported; restart the uvicorn process to pick up TOML changes.

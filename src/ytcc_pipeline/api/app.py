"""FastAPI app exposing `process_pdf` as a single HTTP endpoint.

Design notes
------------

- One uvicorn worker: The lifespan handler loads the layout analyzer into GPU memory and
  keeps it there for the life of the process. Multi-worker uvicorn would multi-load the
  model and contend for VRAM.
- `asyncio.Lock` serializes GPU access: Two `process_pdf` calls in parallel would OOM
  the GPU; the lock turns the service into a polite one-PDF-at-a-time queue at the HTTP
  layer.
- Sync handler offloaded with `asyncio.to_thread`: `process_pdf` is blocking; the
  threadpool lets the event loop keep accepting and queueing next requests' uploads
  while one is processing.
- `FileResponse` streams the bundle back: `BackgroundTasks` cleans up the temp files
  after the response has been sent.
"""

import asyncio
import logging
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from ytcc_pipeline import PipelineConfig, __version__, load_service_config, process_pdf
from ytcc_pipeline.models.formula import FormulaRecognizer, make_formula_recognizer
from ytcc_pipeline.models.grobid import is_grobid_alive
from ytcc_pipeline.models.layout import make_analyzer_from_config
from ytcc_pipeline.models.ocr import LANG_TO_RAPIDOCR_LANG
from ytcc_pipeline.pdf_io.digital_born import detect_digital_born
from ytcc_pipeline.processors.table import TableEngine, make_table_engine

logger = logging.getLogger(__name__)

# Lock-wait threshold for the "queued" INFO line (sub-second waits are normal; longer
# ones mean operators should care).
_LOCK_WAIT_LOG_S = 0.5


def _build_formula_recognizer(cfg: PipelineConfig) -> FormulaRecognizer | None:
    """Construct a `FormulaRecognizer` if formulas are enabled, else None.

    Disabled formulas (`cfg.formula_enabled=False`) skip the model load entirely: the
    formula stage will be a no-op and FORMULA blocks ship as crop-only.
    """
    if not cfg.formula_enabled:
        logger.info("lifespan: formula recognition disabled; skipping model load")
        return None

    return make_formula_recognizer(cfg)


def _grobid_startup_probe(cfg: PipelineConfig) -> None:
    """One-shot GROBID health check for the FastAPI lifespan.

    Logs INFO on success, WARNING on failure. Never raises -- a missing or down GROBID
    is not a service-startup failure; the reference stage will keep logging-and-skipping
    until the server comes back. Extracted from the lifespan body to keep the probe path
    testable without spinning up the full FastAPI app.
    """
    if not cfg.references_enabled:
        return

    if is_grobid_alive(cfg.grobid_url):
        logger.info("lifespan: GROBID reachable at %s", cfg.grobid_url)
    else:
        logger.warning(
            "lifespan: GROBID unreachable at %s -- reference parsing will be a "
            "no-op until the server comes back",
            cfg.grobid_url,
        )


def _build_table_engine(cfg: PipelineConfig) -> TableEngine | None:
    """Construct a `TableEngine` if tables are enabled, else None.

    Disabled tables (`cfg.table_enabled=False`) skip the model load -- the table stage
    will be a no-op and TABLE blocks ship with `cells=None` (image-only fallback).
    """
    if not cfg.table_enabled:
        logger.info("lifespan: table recognition disabled; skipping model load")
        return None

    return make_table_engine(
        device=cfg.table_device,
        batch_size=cfg.table_batch_size,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load models at startup; close them on shutdown.

    The layout analyzer, the formula recognizer, and the table engine are held in
    `app.state` and reused across every request via the corresponding `process_pdf`
    kwargs. When the corresponding `X_enabled` flag is off, the load is skipped and
    `app.state.<X>` is `None`.

    For scanned PDFs the handler temporarily closes and reloads the layout analyzer
    inside the GPU lock -- the pre-loaded model would otherwise compete with the OCR
    worker engines (`cfg.ocr_workers`) for VRAM and produce mass MISS fallbacks. The
    formula recognizer is NOT closed for OCR (its footprint fits alongside the OCR
    workers).

    When `cfg.scanned_enabled` is `False`, the close+reload dance never runs because the
    handler short-circuits scanned PDFs with a 415 response -- no OCR worker ever spawns
    and the analyzer stays resident indefinitely.

    GROBID is externally managed; nothing is loaded for it. When
    `cfg.references_enabled` is true, the lifespan probes the configured `grobid_url`
    once and logs a WARNING if the server is unreachable. The service still starts; the
    reference stage logs and skips per request until GROBID comes back.
    """
    service_config = load_service_config()

    # Apply logging at the application boundary; library code never does this.
    service_config.logging.apply()

    cfg = service_config.pipeline
    logger.info(
        "lifespan startup: loading analyzer (fp16=%s fast_preproc=%s) "
        "scanned_enabled=%s formula_enabled=%s table_enabled=%s references_enabled=%s",
        cfg.layout_fp16,
        cfg.layout_fast_preproc,
        cfg.scanned_enabled,
        cfg.formula_enabled,
        cfg.table_enabled,
        cfg.references_enabled,
    )

    t0 = time.perf_counter()
    app.state.config = cfg
    app.state.api_settings = service_config.api
    app.state.analyzer = make_analyzer_from_config(cfg)
    analyzer_load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    app.state.formula_recognizer = _build_formula_recognizer(cfg)
    formula_load_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    app.state.table_engine = _build_table_engine(cfg)
    table_load_s = time.perf_counter() - t2
    app.state.gpu_lock = asyncio.Lock()

    # GROBID lives outside our process; we don't load anything for it, but a one-shot
    # health probe surfaces a clear log line if the URL is wrong / the server is down
    # before the first request hits the reference stage. The probe never blocks startup.
    _grobid_startup_probe(cfg)
    logger.info(
        "lifespan ready: analyzer_load_s=%.2f formula_load_s=%.2f table_load_s=%.2f"
        " total_s=%.2f",
        analyzer_load_s,
        formula_load_s,
        table_load_s,
        time.perf_counter() - t0,
    )

    yield

    logger.info("lifespan shutdown: closing analyzer + formula + table")
    app.state.analyzer.close()
    if app.state.formula_recognizer is not None:
        app.state.formula_recognizer.close()
    # rapid_table doesn't expose a close(); dropping the ref + letting CUDA reclaim VRAM
    # at process exit is enough for shutdown.
    app.state.table_engine = None


app = FastAPI(
    title="ytcc-pipeline",
    version=__version__,
    description="Thin HTTP wrapper around process_pdf. One PDF per request.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, object]:
    """Cheap liveness probe.

    Returns:
        ```json
        {
            "status": "ok",
            "model_loaded": bool
        }```

        Where `model_loaded` reflects whether the analyzer is currently resident.

    """
    loaded = getattr(app.state, "analyzer", None) is not None

    # liveness probes can be high-frequency and we don't want them flooding the log.
    logger.debug("health probe: model_loaded=%s", loaded)
    return {"status": "ok", "model_loaded": loaded}


@app.post("/process")
async def process(  # noqa: PLR0915 -- linear request flow; the path/branch detect/dispatch/cleanup steps belong together
    background_tasks: BackgroundTasks,
    pdf: Annotated[
        UploadFile,
        File(description="The source PDF (multipart/form-data)."),
    ],
    language: Annotated[
        str,
        Form(description="ISO 639-1 code, e.g. 'en', 'tr', 'ar'."),
    ],
    digital_born: Annotated[
        bool | None,
        Form(
            description=(
                "Override auto-detection. None = auto, True = text layer, False = OCR."
            )
        ),
    ] = None,
) -> FileResponse:
    """Run the pipeline on the uploaded PDF and stream back the  bundle.

    The response sets an `X-Processing-Time` header carrying the server-side processing
    wall in seconds; clients use it to break out end-to-end vs. server-side timing
    without re-instrumenting.

    Args:
        background_tasks: Injected by FastAPI; used to schedule temp-file cleanup after
            the response has streamed.
        pdf: Multipart upload of the source PDF.
        language: ISO 639-1 code matching a key in `LANG_TO_RAPIDOCR_LANG`.
        digital_born: Pass `True`/`False` to override the pipeline's text-layer
            auto-detection; omit for auto.

    Returns:
        `application/x-tar` containing `document.json` + `images/`. The bundle filename
        mirrors the uploaded PDF name (with `.tar` suffix).

    Raises:
        HTTPException(400): unsupported `language`.
        HTTPException(500): unexpected pipeline failure.

    """
    # Short per-request correlation id so concurrent requests' log lines interleave
    # readably. 8 hex chars is enough at our scale.
    request_id = uuid.uuid4().hex[:8]
    upload_name = Path(pdf.filename or "upload.pdf").name
    upload_size = pdf.size if pdf.size is not None else -1

    logger.info(
        "request received: rid=%s pdf=%s language=%s digital_born=%s upload_bytes=%d",
        request_id,
        upload_name,
        language,
        digital_born,
        upload_size,
    )

    if language not in LANG_TO_RAPIDOCR_LANG:
        logger.warning(
            "request rejected: rid=%s pdf=%s reason=unsupported_language value=%s",
            request_id,
            upload_name,
            language,
        )
        raise HTTPException(
            status_code=400,
            detail=f"unsupported language {language!r}; pick from "
            f"{sorted(LANG_TO_RAPIDOCR_LANG)}",
        )

    # Short-circuit explicit scanned requests when scanned support is off. The
    # auto-detect branch (`digital_born is None`) is handled later, after the upload is
    # staged and the text-layer probe has run.
    if digital_born is False and not app.state.config.scanned_enabled:
        logger.warning(
            "request rejected: rid=%s pdf=%s reason=scanned_disabled (caller)",
            request_id,
            upload_name,
        )
        raise HTTPException(
            status_code=415,
            detail=(
                "scanned PDF processing is disabled in this deployment "
                "(scanned_enabled=False); resubmit without digital_born=False or "
                "contact the operator to enable scanned support"
            ),
        )

    # The asyncio.Lock prevents two requests from hitting the GPU in parallel.
    # Subsequent requests await this lock; the event loop is free to receive and buffer
    # their uploads while we work.
    t_wait = time.perf_counter()
    async with app.state.gpu_lock:
        lock_wait_s = time.perf_counter() - t_wait
        if lock_wait_s > _LOCK_WAIT_LOG_S:
            logger.info(
                "request queued: rid=%s pdf=%s lock_wait_s=%.2f",
                request_id,
                upload_name,
                lock_wait_s,
            )

        # Stage the upload to a temp file: process_pdf wants a Path, and the upload may
        # be larger than the multipart body buffer. The temp file lives only for the
        # duration of this handler.
        t_start = time.perf_counter()
        tmp_dir = Path(tempfile.mkdtemp(prefix="ytcc_api_"))
        tmp_pdf = tmp_dir / upload_name
        tmp_pdf.write_bytes(await pdf.read())

        bundle_path = tmp_dir / f"{Path(upload_name).stem}.tar"
        cfg = app.state.config
        try:
            # Determine digital_born upfront so we can decide whether to keep the
            # pre-loaded analyzer resident (digital-born path) or free its VRAM for OCR
            # workers (scanned path).
            if digital_born is None:
                digital_born = await asyncio.to_thread(
                    detect_digital_born,
                    tmp_pdf,
                    sample_pages=cfg.digital_born_sample_pages,
                    text_ratio=cfg.digital_born_text_ratio,
                    min_text_chars=cfg.digital_born_min_text_chars,
                )
                logger.info(
                    "request detected: rid=%s pdf=%s digital_born=%s",
                    request_id,
                    upload_name,
                    digital_born,
                )

                if not digital_born and not cfg.scanned_enabled:
                    logger.warning(
                        "request rejected: rid=%s pdf=%s "
                        "reason=scanned_disabled (auto)",
                        request_id,
                        upload_name,
                    )
                    background_tasks.add_task(_cleanup_dir, tmp_dir)
                    # TRY301: the inner `raise` is intentional. It exits the GPU-lock
                    # block and is caught immediately by the `except HTTPException` arm
                    # just below, which re-raises without wrapping.
                    raise HTTPException(  # noqa: TRY301
                        status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                        detail=(
                            "PDF appears to be scanned but scanned processing is "
                            "disabled in this deployment (scanned_enabled=False); "
                            "submit a digital-born PDF or contact the operator"
                        ),
                    )

            if digital_born:
                logger.info(
                    "request dispatch: rid=%s pdf=%s path=digital_born "
                    "analyzer=injected",
                    request_id,
                    upload_name,
                )
                bundle = await asyncio.to_thread(
                    process_pdf,
                    tmp_pdf,
                    language=language,
                    digital_born=True,
                    output_path=bundle_path,
                    config=cfg,
                    analyzer=app.state.analyzer,
                    formula_recognizer=app.state.formula_recognizer,
                    table_engine=app.state.table_engine,
                )
            else:
                # Free analyzer VRAM so the OCR worker engines can allocate. The
                # ONNXRuntime BFC allocator would otherwise OOM and fall blocks back to
                # MISS. The formula recognizer stays loaded footprint fits alongside the
                # OCR workers.
                logger.info(
                    "request dispatch: rid=%s pdf=%s path=scanned analyzer=freeing",
                    request_id,
                    upload_name,
                )
                app.state.analyzer.close()
                try:
                    bundle = await asyncio.to_thread(
                        process_pdf,
                        tmp_pdf,
                        language=language,
                        digital_born=False,
                        output_path=bundle_path,
                        config=cfg,
                        formula_recognizer=app.state.formula_recognizer,
                        table_engine=app.state.table_engine,
                    )
                finally:
                    # Reload before releasing the lock so the next request already finds
                    # a warm analyzer.
                    t_reload = time.perf_counter()
                    app.state.analyzer = await asyncio.to_thread(
                        make_analyzer_from_config,
                        cfg,
                    )
                    logger.info(
                        "request reload: rid=%s pdf=%s analyzer_reload_s=%.2f",
                        request_id,
                        upload_name,
                        time.perf_counter() - t_reload,
                    )
        except HTTPException:
            # Explicit HTTPException (e.g. the 415 from the scanned-disabled auto-detect
            # path) already carries the right status + detail and has been logged at the
            # raise site. Let FastAPI surface it as-is.
            raise
        except (FileNotFoundError, ValueError) as exc:
            # Both map to HTTP 400: `process_pdf` raises FileNotFoundError for a missing
            # path and ValueError for unsupported language or scanned-disabled
            # rejection.
            logger.warning(
                "request error: rid=%s pdf=%s status=400 reason=%s",
                request_id,
                upload_name,
                exc,
            )
            background_tasks.add_task(_cleanup_dir, tmp_dir)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "request error: rid=%s pdf=%s status=500",
                request_id,
                upload_name,
            )
            background_tasks.add_task(_cleanup_dir, tmp_dir)
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=f"pipeline failure: {exc}",
            ) from exc

        processing_s = time.perf_counter() - t_start

    logger.info(
        "request done: rid=%s pdf=%s processing_s=%.2f bundle_bytes=%d",
        request_id,
        upload_name,
        processing_s,
        bundle.stat().st_size,
    )

    # Schedule cleanup AFTER the response has streamed. Note that this fires outside the
    # GPU lock so the next request can proceed while we transmit. Clients read
    # `X-Processing-Time` to attribute server-side wall vs. end-to-end wall.
    background_tasks.add_task(_cleanup_dir, tmp_dir)
    return FileResponse(
        bundle,
        media_type="application/x-tar",
        filename=bundle.name,
        background=background_tasks,
        headers={"X-Processing-Time": f"{processing_s:.3f}"},
    )


def _cleanup_dir(path: Path) -> None:
    """Best-effort recursive delete of a temp dir.

    Runs in a background task after the response has been sent. Failures are logged but
    never re-raised -- a leaked temp file is preferable to a 500 on a request that
    already succeeded.
    """
    try:
        shutil.rmtree(path)
        logger.debug("cleanup: removed temp dir %s", path)
    except Exception:  # noqa: BLE001
        # Cleanup runs post-response; any failure here is a leaked temp dir, not a
        # request failure.
        logger.warning("cleanup: failed to remove temp dir %s", path, exc_info=True)

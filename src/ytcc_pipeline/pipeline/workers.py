"""Spawn-pool worker entry points.

`worker_digital_born` runs in `digital_born_workers` processes; each opens its own
`pdf_oxide.PdfDocument` and processes its assigned pages, delegating text extraction to
`extract_text_digital_born`.

`worker_scanned` runs in `ocr_workers` processes; each lazily constructs exactly one
`OcrExtractor` (RapidOCR engine + CUDA context) and reuses it across all pages
dispatched to that worker, batching text-route crops through
`extract_text_scanned_batch`.

Both must be module-level so `spawn` can pickle them. `pdf_oxide` is imported inside the
digital-born worker body so the parent process doesn't pay for it just to spawn workers.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from ytcc_pipeline.image_io import read_rgb
from ytcc_pipeline.models.ocr import OcrExtractor
from ytcc_pipeline.processors.text import (
    extract_text_digital_born,
    extract_text_scanned_batch,
)
from ytcc_pipeline.routing import Route, route_for

from .blocks import TEXT_ROUTES, build_block

if TYPE_CHECKING:
    from ytcc_pipeline.schema import Block

    from .payloads import DigitalBornWorkerArgs, ScannedWorkerArgs, WorkerResult

logger = logging.getLogger(__name__)


# Per-worker-process OCR engine cache. The `spawn` start method gives each worker its
# own re-imported module, so this global is effectively a per-process singleton
# populated lazily inside `worker_scanned`. Lowercase name signals mutable module-level
# state distinct from the all-caps constants elsewhere in the package.
_worker_ocr: OcrExtractor | None = None


def worker_digital_born(args: DigitalBornWorkerArgs) -> WorkerResult:
    """Run pdf_oxide text extraction + per-block crop/save for one page.

    The function is invoked once per page across the spawn pool. Each worker reopens its
    own `pdf_oxide.PdfDocument` (pdf_oxide handles are not picklable) and dispatches
    each detection through the routing layer.

    Args:
        args: Picklable IPC payload for this page -- a `DigitalBornWorkerArgs` tuple.

    Returns:
        `(page_idx, height, width, blocks)` consumed by `pages_from_blocks` to rebuild
            the parent-side page list.
    """
    # `pdf_oxide` is a heavy PyO3 wheel that opens a file table per import. Deferring
    # the import to inside the worker entry keeps the parent process from paying for it
    # when spawning the pool.
    import pdf_oxide  # noqa: PLC0415

    (
        page_idx,
        page_path,
        detections,
        pdf_path,
        images_dir,
        render_dpi,
        crop_format,
        jpeg_quality,
        bundle_miss_images_for,
    ) = args
    pdf_name = Path(pdf_path).name

    # Wrap the worker body so any crash logs with worker context BEFORE the spawn worker
    # dies. Without this the parent only sees `BrokenProcessPool` with no clue which
    # page / PDF triggered the failure. Spawn workers don't inherit the parent's log
    # handlers, but `logger.exception` is at ERROR level so the default handler writes
    # it to inherited stderr.
    try:
        page_image = read_rgb(page_path)
        height, width = page_image.shape[:2]
        doc = pdf_oxide.PdfDocument(pdf_path)

        blocks: list[Block] = []
        for det in detections:
            route = route_for(det.label)
            if route is Route.IGNORE:
                continue

            text = (
                extract_text_digital_born(
                    doc,
                    page_idx,
                    det.bbox,
                    source_dpi=render_dpi,
                )
                if route in TEXT_ROUTES
                else None
            )
            block = build_block(
                det,
                route=route,
                page_image=page_image,
                page_no=page_idx + 1,
                images_dir=Path(images_dir),
                crop_format=crop_format,
                jpeg_quality=jpeg_quality,
                text=text,
                bundle_miss_images_for=bundle_miss_images_for,
            )
            if block.miss:
                # Orchestrator already emits one summary per PDF. A page-by-page warning
                # here floods the log on noisy scanned/digital-born runs (hundreds per
                # PDF) without adding operator-visible signal beyond the summary.
                logger.debug(
                    "block miss: pdf=%s page=%d label=%s",
                    pdf_name,
                    page_idx + 1,
                    det.label,
                )

            blocks.append(block)
    except Exception:
        logger.exception(
            "worker_digital_born crashed: pdf=%s page_idx=%d page_path=%s",
            pdf_name,
            page_idx,
            page_path,
        )
        raise
    else:
        return page_idx, height, width, blocks


def worker_scanned(args: ScannedWorkerArgs) -> WorkerResult:
    """Run RapidOCR text extraction + per-block crop/save for one page.

    Each worker lazily constructs one `OcrExtractor` (RapidOCR engine + CUDA context)
    and caches it across all pages dispatched to that worker, batching every text-route
    crop through `extract_text_scanned_batch` so the recognizer's internal batch
    (`Rec.batch_size`) flushes for free.

    Args:
        args: Picklable IPC payload for this page -- a `ScannedWorkerArgs` tuple.

    Returns:
        `(page_idx, height, width, blocks)` consumed by `pages_from_blocks` to rebuild
            the parent-side page list.
    """
    (
        page_idx,
        page_path,
        detections,
        images_dir,
        crop_format,
        jpeg_quality,
        language,
        ocr_batch_size,
        ocr_min_score,
        ocr_use_cuda,
        bundle_miss_images_for,
    ) = args

    # Wrap the worker body so any crash (OCR engine load failure, image decode error,
    # ...) logs with worker context BEFORE the spawn worker dies. Without this the
    # parent only sees `BrokenProcessPool` with no clue which page / language triggered
    # the failure.
    try:
        # One OCR engine per worker process, cached across this worker's tasks via a
        # module-level singleton. Spawn workers each re-import this module, so the
        # global is private per worker process -- a class cache would force every
        # worker's OcrExtractor to coordinate, which is the opposite of the
        # spawn-isolation we want here.
        global _worker_ocr  # noqa: PLW0603
        if _worker_ocr is None:
            logger.info(
                "ocr init: worker_pid=%d language=%s use_cuda=%s batch_size=%d",
                os.getpid(),
                language,
                ocr_use_cuda,
                ocr_batch_size,
            )
            _worker_ocr = OcrExtractor(
                language=language,
                batch_size=ocr_batch_size,
                min_score=ocr_min_score,
                use_cuda=ocr_use_cuda,
            )
        ocr = _worker_ocr

        page_image = read_rgb(page_path)
        height, width = page_image.shape[:2]

        text_by_det_id = extract_text_scanned_batch(
            ocr,
            page_image,
            (d for d in detections if route_for(d.label) in TEXT_ROUTES),
        )

        blocks: list[Block] = []
        for det in detections:
            route = route_for(det.label)
            if route is Route.IGNORE:
                continue

            text = text_by_det_id.get(id(det)) if route in TEXT_ROUTES else None
            block = build_block(
                det,
                route=route,
                page_image=page_image,
                page_no=page_idx + 1,
                images_dir=Path(images_dir),
                crop_format=crop_format,
                jpeg_quality=jpeg_quality,
                text=text,
                bundle_miss_images_for=bundle_miss_images_for,
            )
            if block.miss:
                # Orchestrator's per-PDF MISS summary at WARNING is the operator-visible
                # signal.
                logger.debug(
                    "block miss: page=%d label=%s",
                    page_idx + 1,
                    det.label,
                )

            blocks.append(block)
    except Exception:
        logger.exception(
            "worker_scanned crashed: page_idx=%d page_path=%s language=%s",
            page_idx,
            page_path,
            language,
        )
        raise
    else:
        return page_idx, height, width, blocks

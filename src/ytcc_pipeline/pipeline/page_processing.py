"""Per-page dispatch -- pick serial or parallel path based on `cfg` + `digital_born`.

Three exec modes share a single output shape (`list[Page]`):

- `_process_pages_serial`: single-process loop, used when both
  `cfg.digital_born_workers` and `cfg.ocr_workers` are `1`.
- `_process_pages_parallel_digital`: `cfg.digital_born_workers`-wide spawn pool, each
  worker opens its own `pdf_oxide.PdfDocument`.
- `_process_pages_parallel_scanned`: `cfg.ocr_workers`-wide spawn pool, each worker
  lazily owns a `RapidOCR` engine.

The top-level `process_pages` chooses among them.
"""

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from ytcc_pipeline.image_io import read_rgb
from ytcc_pipeline.models.ocr import make_ocr_extractor
from ytcc_pipeline.pdf_io.text import open_pdf_for_text
from ytcc_pipeline.processors.text import (
    extract_text_digital_born,
    extract_text_scanned_batch,
)
from ytcc_pipeline.routing import Route, route_for
from ytcc_pipeline.schema import Block, Page

from .blocks import TEXT_ROUTES, build_block, pages_from_blocks
from .payloads import (
    DigitalBornWorkerArgs,
    ScannedWorkerArgs,
    WorkerResult,
)
from .workers import worker_digital_born, worker_scanned

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from ytcc_pipeline.config import PipelineConfig
    from ytcc_pipeline.models.layout import LayoutDetection

logger = logging.getLogger(__name__)


def process_pages(  # noqa: PLR0913  -- dispatcher routes to serial / digital-parallel / scanned-parallel paths; each arg is mandatory for at least one
    *,
    page_paths: list[Path],
    layout_results: dict[int, list[LayoutDetection]],
    digital_born: bool,
    images_dir: Path,
    cfg: PipelineConfig,
    pdf_path: Path,
    language: str,
) -> list[Page]:
    """Dispatch per-page block processing to the right execution path.

    Picks serial, digital-born-parallel, or scanned-parallel based on
    `cfg.digital_born_workers`, `cfg.ocr_workers`, and `digital_born`. All three paths
    emit the same shape -- one `Page` per source page in page order, blocks in per-page
    reading order.

    Args:
        page_paths: Rendered page image paths in 0-based page order.
        layout_results: `{page_idx: [detection, ...]}` from the layout analyzer.
        digital_born: Whether the PDF should be treated as digital-born (pdf_oxide text
            extraction) or scanned (OCR).
        images_dir: Directory where crop files are written.
        cfg: Pipeline config; `digital_born_workers` and `ocr_workers` select between
            serial and parallel paths.
        pdf_path: Source PDF -- needed by the digital-born path so each worker can
            reopen its own `pdf_oxide.PdfDocument`.
        language: ISO 639-1 code; routes through the OCR workers on the scanned path,
            ignored on the digital-born path.

    Returns:
        Pages in source-order with blocks attached.
    """
    if cfg.digital_born_workers > 1 and digital_born:
        return _process_pages_parallel_digital(
            page_paths=page_paths,
            layout_results=layout_results,
            images_dir=images_dir,
            cfg=cfg,
            pdf_path=pdf_path,
        )

    if cfg.ocr_workers > 1 and not digital_born:
        return _process_pages_parallel_scanned(
            page_paths=page_paths,
            layout_results=layout_results,
            images_dir=images_dir,
            cfg=cfg,
            language=language,
        )

    return _process_pages_serial(
        page_paths=page_paths,
        layout_results=layout_results,
        digital_born=digital_born,
        images_dir=images_dir,
        cfg=cfg,
        pdf_path=pdf_path,
        language=language,
    )


def _process_pages_serial(  # noqa: PLR0913  -- mirrors the dispatcher's signature so callers can pass the same kwargs through
    *,
    page_paths: list[Path],
    layout_results: dict[int, list[LayoutDetection]],
    digital_born: bool,
    images_dir: Path,
    cfg: PipelineConfig,
    pdf_path: Path,
    language: str,
) -> list[Page]:
    pdf_text_doc = open_pdf_for_text(pdf_path) if digital_born else None
    ocr = make_ocr_extractor(cfg, language) if not digital_born else None

    pages: list[Page] = []
    for page_idx, page_path in enumerate(page_paths):
        detections = layout_results.get(page_idx, [])
        page_image = read_rgb(page_path)
        page_no = page_idx + 1

        # Pre-batch text-route OCR for the scanned path -- lets RapidOCR's recognizer
        # flush its internal batch (`Rec.batch_size`).
        text_by_det_id: dict[int, str | None] = {}
        if not digital_born and ocr is not None:
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

            text: str | None = None
            if route in TEXT_ROUTES:
                if digital_born and pdf_text_doc is not None:
                    text = extract_text_digital_born(
                        pdf_text_doc,
                        page_idx,
                        det.bbox,
                        source_dpi=cfg.render_dpi,
                    )
                else:
                    text = text_by_det_id.get(id(det))

            block = build_block(
                det,
                route=route,
                page_image=page_image,
                page_no=page_no,
                images_dir=images_dir,
                crop_format=cfg.crop_format,
                jpeg_quality=cfg.jpeg_quality,
                text=text,
                bundle_miss_images_for=cfg.bundle_miss_images_for,
            )
            if block.miss:
                # The orchestrator emits one aggregated line per PDF; flooding the log
                # per-block here adds no operator-visible signal beyond that summary.
                logger.debug(
                    "block miss: pdf=%s page=%d label=%s",
                    pdf_path.name,
                    page_no,
                    det.label,
                )
            blocks.append(block)

        height, width = page_image.shape[:2]
        pages.append(
            Page(
                page_no=page_no,
                width_px=width,
                height_px=height,
                blocks=blocks,
            ),
        )

    return pages


def _process_pages_parallel_digital(
    *,
    page_paths: list[Path],
    layout_results: dict[int, list[LayoutDetection]],
    images_dir: Path,
    cfg: PipelineConfig,
    pdf_path: Path,
) -> list[Page]:
    """Digital-born parallel path: workers each open their own pdf_oxide doc."""
    tasks = [
        DigitalBornWorkerArgs(
            page_idx=page_idx,
            page_path=str(page_path),
            detections=layout_results.get(page_idx, []),
            pdf_path=str(pdf_path),
            images_dir=str(images_dir),
            render_dpi=cfg.render_dpi,
            crop_format=cfg.crop_format,
            jpeg_quality=cfg.jpeg_quality,
            bundle_miss_images_for=cfg.bundle_miss_images_for,
        )
        for page_idx, page_path in enumerate(page_paths)
    ]

    return _run_worker_pool(worker_digital_born, tasks, cfg.digital_born_workers)


def _process_pages_parallel_scanned(
    *,
    page_paths: list[Path],
    layout_results: dict[int, list[LayoutDetection]],
    images_dir: Path,
    cfg: PipelineConfig,
    language: str,
) -> list[Page]:
    """Scanned parallel path: each worker lazily owns one RapidOCR engine."""
    tasks = [
        ScannedWorkerArgs(
            page_idx=page_idx,
            page_path=str(page_path),
            detections=layout_results.get(page_idx, []),
            images_dir=str(images_dir),
            crop_format=cfg.crop_format,
            jpeg_quality=cfg.jpeg_quality,
            language=language,
            ocr_batch_size=cfg.ocr_batch_size,
            ocr_min_score=cfg.ocr_min_score,
            ocr_use_cuda=cfg.ocr_use_cuda,
            bundle_miss_images_for=cfg.bundle_miss_images_for,
        )
        for page_idx, page_path in enumerate(page_paths)
    ]

    return _run_worker_pool(worker_scanned, tasks, cfg.ocr_workers)


def _run_worker_pool[A](
    worker_fn: Callable[[A], WorkerResult],
    tasks: Sequence[A],
    max_workers: int,
) -> list[Page]:
    """Map `worker_fn` across `tasks` in a spawn pool, return ordered pages.

    Both parallel paths (digital-born and scanned) share the same outer shape --
    different task type, different worker callable, different pool size -- so the
    spawn-context + collect-by-page_idx + page rebuild machinery lives here.

    Args:
        worker_fn: Module-level callable picklable by `spawn`; receives one task and
            returns `(page_idx, height, width, payloads)`.
        tasks: One task per source page in page order.
        max_workers: Pool size -- typically `cfg.digital_born_workers` or
            `cfg.ocr_workers`.

    Returns:
        Pages in source order with blocks attached.
    """
    logger.debug(
        "worker pool dispatch: worker_fn=%s workers=%d tasks=%d",
        worker_fn.__name__,
        max_workers,
        len(tasks),
    )
    ctx = multiprocessing.get_context("spawn")
    results: dict[int, tuple[int, int, list[Block]]] = {}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        for page_idx, h, w, blocks in ex.map(worker_fn, tasks):
            results[page_idx] = (h, w, blocks)

    logger.debug(
        "worker pool drained: worker_fn=%s pages=%d",
        worker_fn.__name__,
        len(results),
    )
    return pages_from_blocks(results, len(tasks))

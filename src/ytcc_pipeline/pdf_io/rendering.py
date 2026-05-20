"""Render PDF pages to image files in parallel."""

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pdf_oxide

if TYPE_CHECKING:
    from ytcc_pipeline.config import ImageFormat

__all__ = ["render_pages"]

logger = logging.getLogger(__name__)


class _RenderWorkerArgs(NamedTuple):
    """Picklable IPC payload for a single page-render task."""

    pdf_path: str
    page_idx: int
    out_path: str
    dpi: int
    image_format: ImageFormat
    jpeg_quality: int


_EXT_BY_FORMAT: dict[ImageFormat, str] = {"jpeg": "jpg", "png": "png"}

# On 72-core boxes, unbounded worker counts cause OpenBLAS to spawn dozens of threads
# per worker and run out of pthread slots -- keep the default sane.
_DEFAULT_WORKER_CAP = 16


def render_pages(  # noqa: PLR0913 -- one entry point that exposes every render knob; bundling into a config struct would push noise onto every caller
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 300,
    image_format: ImageFormat = "jpeg",
    jpeg_quality: int = 95,
    max_workers: int | None = None,
) -> list[Path]:
    """Render every page of `pdf_path` to image files in `output_dir`.

    Each worker re-opens the PDF locally -- `pdf_oxide.PdfDocument` is not picklable.
    Workers run under the `spawn` start method so a parent process holding a CUDA
    context (e.g. mid-pipeline) won't break fork.

    Args:
        pdf_path: Source PDF file.
        output_dir: Existing directory that will receive page image files.
        dpi: Render resolution in dots per inch.
        image_format: Output container, `"jpeg"` or `"png"`.
        jpeg_quality: JPEG quality (1-100); ignored when `image_format="png"`.
        max_workers: Process-pool size. `None` uses `min(16, os.cpu_count())` -- capped
            to avoid OpenBLAS pthread exhaustion on high-core hosts.

    Returns:
        Paths to `page-{n:04d}.{jpg|png}` files in 0-based page order.

    Raises:
        FileNotFoundError: `pdf_path` does not exist.
        NotADirectoryError: `output_dir` is not an existing directory.
        ValueError: `image_format` is not `"jpeg"` or `"png"`.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.is_file():
        msg = f"PDF not found: {pdf_path}"
        raise FileNotFoundError(msg)

    if not output_dir.is_dir():
        msg = f"output_dir is not a directory: {output_dir}"
        raise NotADirectoryError(msg)

    if image_format not in _EXT_BY_FORMAT:
        msg = f"image_format must be 'jpeg' or 'png', got {image_format!r}"
        raise ValueError(msg)

    page_count = pdf_oxide.PdfDocument(str(pdf_path)).page_count()
    if page_count == 0:
        logger.warning("render skipped: pdf=%s reason=zero_pages", pdf_path.name)
        return []

    ext = _EXT_BY_FORMAT[image_format]
    tasks = [
        _RenderWorkerArgs(
            pdf_path=str(pdf_path),
            page_idx=i,
            out_path=str(output_dir / f"page-{i:04d}.{ext}"),
            dpi=dpi,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )
        for i in range(page_count)
    ]

    if max_workers is None:
        max_workers = min(_DEFAULT_WORKER_CAP, os.cpu_count() or 1)

    logger.debug(
        "render dispatch: pdf=%s pages=%d dpi=%d format=%s workers=%d",
        pdf_path.name,
        page_count,
        dpi,
        image_format,
        max_workers,
    )

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        return [Path(p) for p in executor.map(_render_one, tasks)]


def _render_one(args: _RenderWorkerArgs) -> str:
    doc = pdf_oxide.PdfDocument(args.pdf_path)
    if args.image_format == "jpeg":
        data = doc.render_page(
            args.page_idx,
            dpi=args.dpi,
            format="jpeg",
            jpeg_quality=args.jpeg_quality,
        )
    else:
        data = doc.render_page(args.page_idx, dpi=args.dpi, format="png")

    Path(args.out_path).write_bytes(data)
    return args.out_path

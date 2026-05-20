"""Per-block assembly + summary helpers shared by every execution path.

Per-block-type processing (text extraction, image crop+save, formula recognition, table
structure, reference parsing) lives in the processors layer. This module is the thin
layer that:

- routes each `LayoutDetection` to the right processor via `build_block`,
- summarizes the assembled pages for the stage-complete log line via `summarize_pages`,
- reassembles `Page` lists from worker output via `pages_from_blocks`,
- re-binds the blocks on a page list after a stage rewrote some of them, via
  `replace_blocks`.

Everything here is module-level so spawn workers can re-import it cleanly.

This module has no logger of its own -- per-block lifecycle is logged at the call sites
(workers, serial loop, formula stage) where the PDF / worker / stage context is in
scope.
"""

from dataclasses import replace
from typing import TYPE_CHECKING

from ytcc_pipeline.processors.image import process_image_block
from ytcc_pipeline.routing import Route
from ytcc_pipeline.schema import Block, BlockType, Page

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from ytcc_pipeline.config import ImageFormat
    from ytcc_pipeline.models.layout import LayoutDetection

# Maps each non-IGNORE route to the `BlockType` value that ends up in `document.json`.
# `Route.IGNORE` is dropped before this map is consulted.
ROUTE_TO_TYPE: dict[Route, BlockType] = {
    Route.TEXT: BlockType.TEXT,
    Route.IMAGE: BlockType.IMAGE,
    Route.REFERENCE: BlockType.REFERENCE,
    Route.FORMULA: BlockType.FORMULA,
    Route.TABLE: BlockType.TABLE,
}

# Routes whose payload is always crop-only (no text extraction): the bundle ships an
# image, no `text` field. FORMULA blocks join this set at the worker layer -- their
# LaTeX is filled in later by the formula stage running on the main process (workers
# can't afford the extra VRAM per process for the model). TABLE blocks behave the same
# way: the worker saves the crop, the table stage fills in cells later.
CROP_ONLY_ROUTES: frozenset[Route] = frozenset(
    {Route.IMAGE, Route.FORMULA, Route.TABLE},
)

# Routes that need text extraction (pdf_oxide for digital-born, RapidOCR for scanned).
# Used both for the per-page text-OCR batch and for deciding whether to consult the
# extracted text when assembling the block.
TEXT_ROUTES: frozenset[Route] = frozenset({Route.TEXT, Route.REFERENCE})


def build_block(  # noqa: PLR0913 -- single decision point every worker path funnels into; splitting would just re-pass the same args
    det: LayoutDetection,
    *,
    route: Route,
    page_image: np.ndarray,
    page_no: int,
    images_dir: Path,
    crop_format: ImageFormat,
    jpeg_quality: int,
    text: str | None,
    bundle_miss_images_for: frozenset[BlockType],
) -> Block:
    """Assemble one `Block` from a detection, dispatching on its route.

    Crop-only routes (IMAGE, FORMULA, TABLE) save the crop via the image processor and
    leave `text=None` -- FORMULA's LaTeX is filled in later by the formula stage on the
    main process; TABLE's cell grid is filled in later by the table stage. Text /
    reference routes return a populated `Block` when extraction succeeded, or fall back
    to a MISS block -- by default with the saved crop attached (`image_path` set,
    `miss=True`), or with `image_path=None` when the block's type isn't in
    `bundle_miss_images_for`.

    Args:
        det: Single layout detection from the analyzer.
        route: Routing decision pre-computed by `route_for`. `Route.IGNORE` must be
            filtered upstream.
        page_image: HWC uint8 RGB page image, already loaded.
        page_no: 1-based page number for the filename prefix.
        images_dir: Destination directory for any crop file produced.
        crop_format: Output container for the crop (`"png"` or `"jpeg"`).
        jpeg_quality: 1-100 quality knob when `crop_format="jpeg"`.
        text: Extracted text for TEXT / REFERENCE routes (`None` if extraction failed or
            was not attempted).
        bundle_miss_images_for: Block types for which a MISS fallback image is written
            to the bundle. Types not in this set get a MISS entry with
            `image_path=None`.

    Returns:
        A picklable `Block` ready to ship across the worker pool.
    """
    block_type = ROUTE_TO_TYPE[route]
    if route in CROP_ONLY_ROUTES:
        return Block(
            reading_order=det.reading_order,
            label=det.label,
            type=block_type,
            bbox=det.bbox,
            confidence=det.confidence,
            text=None,
            image_path=process_image_block(
                bbox=det.bbox,
                label=det.label,
                page_image=page_image,
                page_no=page_no,
                images_dir=images_dir,
                miss=False,
                crop_format=crop_format,
                jpeg_quality=jpeg_quality,
            ),
            miss=False,
        )

    if text:
        return Block(
            reading_order=det.reading_order,
            label=det.label,
            type=block_type,
            bbox=det.bbox,
            confidence=det.confidence,
            text=text,
            image_path=None,
            miss=False,
        )

    # MISS path. Skip the crop save when the block type is not opted in, leaving
    # `image_path=None` in the bundle's JSON entry. The block still appears so consumers
    # retain its reading-order + bbox.
    miss_image_path: str | None = None
    if block_type in bundle_miss_images_for:
        miss_image_path = process_image_block(
            bbox=det.bbox,
            label=det.label,
            page_image=page_image,
            page_no=page_no,
            images_dir=images_dir,
            miss=True,
            crop_format=crop_format,
            jpeg_quality=jpeg_quality,
        )

    return Block(
        reading_order=det.reading_order,
        label=det.label,
        type=block_type,
        bbox=det.bbox,
        confidence=det.confidence,
        text=None,
        image_path=miss_image_path,
        miss=True,
    )


def summarize_pages(pages: list[Page]) -> dict[str, int]:
    """Count blocks by type + MISS fallbacks across all pages.

    Used to emit one stage-level summary INFO + a WARNING when any MISS fallback fired.
    Walking the assembled `Page` list keeps both parallel and serial paths summarized
    the same way.

    Returns:
        A dict with one entry per `BlockType` value, a `miss` key counting blocks where
        text/formula extraction failed, and a `total` key counting every block across
        every page.
    """
    counts: dict[str, int] = {bt.value: 0 for bt in BlockType}
    counts["miss"] = 0
    counts["total"] = 0
    for page in pages:
        for block in page.blocks:
            counts[block.type.value] += 1
            counts["total"] += 1
            if block.miss:
                counts["miss"] += 1

    return counts


def pages_from_blocks(
    blocks_by_idx: dict[int, tuple[int, int, list[Block]]],
    n_pages: int,
) -> list[Page]:
    """Convert `{page_idx: (height, width, blocks)}` into ordered `Page` objects."""
    pages: list[Page] = []
    for page_idx in range(n_pages):
        height, width, blocks = blocks_by_idx[page_idx]
        pages.append(
            Page(
                page_no=page_idx + 1,
                width_px=width,
                height_px=height,
                blocks=blocks,
            ),
        )

    return pages


def replace_blocks(
    pages: list[Page],
    blocks_by_page: dict[int, list[Block]],
) -> list[Page]:
    """Return a new page list with blocks replaced by `blocks_by_page[idx]`.

    Used by stages that rewrite some blocks after the per-page workers have produced the
    initial assembly -- formula, table, and reference each rebuild the page list this
    way.

    Args:
        pages: The page list emitted by the previous stage.
        blocks_by_page: `{page_idx: [block, ...]}`. Must have an entry for every index
            in `range(len(pages))`.

    Returns:
        A new list of `Page` objects, one per input page.
    """
    return [replace(page, blocks=blocks_by_page[idx]) for idx, page in enumerate(pages)]

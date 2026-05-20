"""Text-block processing -- TEXT + REFERENCE routes.

Two distinct call shapes share this module:

- `extract_text_digital_born`: per-block, one `pdf_oxide` query at a time. Cheap;
  bbox-aware; runs inside the digital-born worker.
- `extract_text_scanned_batch`: per-page, batched. RapidOCR is efficient when fed many
  crops at once (its internal `Rec.batch_size` flushes), so the scanned worker
  pre-collects every text-route crop on a page and calls this once.

Different shapes, same role: "given the layout bbox(es), give me the text inside." The
dispatcher in the worker layer picks based on `cfg.digital_born`.
"""

import logging
from typing import TYPE_CHECKING

from ytcc_pipeline.image_io import crop_from_page
from ytcc_pipeline.pdf_io.text import extract_text_in_bbox

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    import pdf_oxide

    from ytcc_pipeline.models.layout import LayoutDetection
    from ytcc_pipeline.models.ocr import OcrExtractor

logger = logging.getLogger(__name__)


def extract_text_digital_born(
    doc: pdf_oxide.PdfDocument,
    page_idx: int,
    bbox: tuple[float, float, float, float],
    *,
    source_dpi: int,
) -> str | None:
    """Extract text from a single bbox via `pdf_oxide.extract_spans`.

    Per-block call; the worker invokes this once per TEXT / REFERENCE detection on a
    page. Returns `None` on extraction failure so the caller falls back to a MISS crop.

    Args:
        doc: An open `pdf_oxide.PdfDocument` for the source PDF.
        page_idx: 0-based page index.
        bbox: `(x1, y1, x2, y2)` in pixel coords at `source_dpi`.
        source_dpi: DPI the bbox was measured at (= the render DPI).

    Returns:
        Concatenated, whitespace-normalized text or `None` when `pdf_oxide` returned
        nothing or raised.
    """
    try:
        return extract_text_in_bbox(doc, page_idx, bbox, source_dpi=source_dpi)
    except Exception:  # noqa: BLE001
        # Recoverable: caller falls back to a MISS image crop.
        #
        # `exc_info=True` keeps the underlying pdf_oxide error diagnosable.
        logger.warning(
            "pdf_text failed page=%d bbox=%s",
            page_idx,
            bbox,
            exc_info=True,
        )
        return None


def extract_text_scanned_batch(
    ocr: OcrExtractor,
    page_image: np.ndarray,
    text_detections: Iterable[LayoutDetection],
) -> dict[int, str | None]:
    """Batch-OCR every text-route crop on a single page.

    Crops every bbox out of the already-loaded page array and feeds the list to
    `OcrExtractor.extract_batch` in one call. RapidOCR's internal recognition batch
    flushes for free this way (`Rec.batch_size`).

    Args:
        ocr: A worker-owned `OcrExtractor` (one per worker process).
        page_image: HWC uint8 RGB page image, already loaded by the worker.
        text_detections: The TEXT + REFERENCE detections on this page. Other routes
            (IMAGE / FORMULA / IGNORE) must be filtered out upstream.

    Returns:
        Mapping `id(det) -> text`. Lookup by Python `id` so the caller can correlate
        against the original detection objects without restructuring their loop.
    """
    text_detections = list(text_detections)
    if not text_detections:
        return {}

    crops = [crop_from_page(page_image, d.bbox) for d in text_detections]
    logger.debug("OCR batch dispatch: crops=%d", len(crops))
    texts = ocr.extract_batch(crops)
    n_hits = sum(1 for t in texts if t)
    logger.debug(
        "OCR batch result: crops=%d hits=%d miss=%d",
        len(crops),
        n_hits,
        len(crops) - n_hits,
    )

    return {id(d): t for d, t in zip(text_detections, texts, strict=True)}

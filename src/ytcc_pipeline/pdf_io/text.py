"""Extract text from a pixel-space bbox using `pdf_oxide`.

The layout detector emits bboxes in image pixel coords at the render DPI (origin
top-left, y down). `pdf_oxide.extract_spans` wants region coords in PDF points (origin
bottom-left, y up). This module bridges the two.
"""

import re
from pathlib import Path

import pdf_oxide

__all__ = ["extract_text_in_bbox", "open_pdf_for_text"]

_WHITESPACE_RUN = re.compile(r"\s+")


def open_pdf_for_text(pdf_path: Path) -> pdf_oxide.PdfDocument:
    """Open a PDF for text-by-bbox queries.

    Callers should keep the returned instance for the duration of bbox queries on the
    same PDF -- every `PdfDocument(...)` call reparses the file table.

    Args:
        pdf_path: Source PDF file.

    Returns:
        An open `pdf_oxide.PdfDocument`.

    Raises:
        FileNotFoundError: `pdf_path` does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        msg = f"PDF not found: {pdf_path}"
        raise FileNotFoundError(msg)

    return pdf_oxide.PdfDocument(str(pdf_path))


def extract_text_in_bbox(
    doc: pdf_oxide.PdfDocument,
    page_idx: int,
    bbox: tuple[float, float, float, float],
    *,
    source_dpi: int,
) -> str | None:
    """Extract text from a pixel-space bbox region of a page.

    Args:
        doc: An open `pdf_oxide.PdfDocument`.
        page_idx: 0-based page index -- i.e. `Page.page_no - 1`.
        bbox: `(x1, y1, x2, y2)` in pixels at `source_dpi`, origin top-left.
        source_dpi: DPI at which the page was rendered for layout analysis.

    Returns:
        Concatenated, whitespace-normalized text. `None` when the bbox is empty or
        yields only whitespace.
    """
    media_box = doc.page_media_box(page_idx)
    page_height_pt = media_box[3] - media_box[1]
    region = _pixel_bbox_to_pdf_region(
        bbox,
        source_dpi=source_dpi,
        page_height_pt=page_height_pt,
    )

    spans = doc.extract_spans(page_idx, region=region)
    joined = " ".join(s.text for s in spans)
    normalized = _WHITESPACE_RUN.sub(" ", joined).strip()

    return normalized or None


def _pixel_bbox_to_pdf_region(
    bbox: tuple[float, float, float, float],
    *,
    source_dpi: int,
    page_height_pt: float,
) -> tuple[float, float, float, float]:
    """Map a top-left-origin pixel bbox to a bottom-left-origin PDF region.

    Args:
        bbox: `(x1, y1, x2, y2)` in pixels at `source_dpi`.
        source_dpi: DPI the bbox was measured at.
        page_height_pt: PDF page height in points.

    Returns:
        `(x, y, w, h)` in PDF points where `y` is the bottom edge.

    """
    x1, y1, x2, y2 = bbox
    px_to_pt = 72.0 / source_dpi

    x_pt = x1 * px_to_pt
    w_pt = (x2 - x1) * px_to_pt
    h_pt = (y2 - y1) * px_to_pt
    y_pt = page_height_pt - y2 * px_to_pt

    return (x_pt, y_pt, w_pt, h_pt)

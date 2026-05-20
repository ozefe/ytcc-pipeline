"""I/O boundary with `pdf_oxide`.

Every module in this subpackage is a thin wrapper around a `pdf_oxide.PdfDocument` call:

- `render_pages`: renders pages to image files via a `ProcessPoolExecutor`.
- `extract_text_in_bbox` / `open_pdf_for_text`: extract text from a pixel-space bbox
  using `extract_spans`.
- `extract_metadata`: sha256 + XMP fields for the bundled `document.json`.
- `detect_digital_born`: heuristic probe that decides scanned vs digital-born.
"""

from .digital_born import detect_digital_born
from .metadata import PdfMetadata, extract_metadata
from .rendering import render_pages
from .text import extract_text_in_bbox, open_pdf_for_text

__all__ = [
    "PdfMetadata",
    "detect_digital_born",
    "extract_metadata",
    "extract_text_in_bbox",
    "open_pdf_for_text",
    "render_pages",
]

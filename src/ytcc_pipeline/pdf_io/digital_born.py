"""Heuristic detection of digital-born vs scanned PDFs."""

import logging
import statistics
from pathlib import Path

import pdf_oxide

__all__ = ["detect_digital_born"]

logger = logging.getLogger(__name__)


def detect_digital_born(
    pdf_path: Path,
    *,
    sample_pages: int = 5,
    text_ratio: float = 0.3,
    min_text_chars: int = 100,
) -> bool:
    """Decide whether a PDF has a usable embedded text layer.

    Probes `sample_pages` pages distributed evenly across the document and applies two
    signals. The PDF is treated as digital-born if either fires:

    1. Strict majority: at least `text_ratio` of probed pages clear `min_text_chars`
       non-whitespace characters. Catches standard text-rich academic theses.
    2. Sparse-but-consistent: the median probed page has at least `min_text_chars / 2`
       characters and at least one probed page clears the full `min_text_chars` bar.
       Catches gazette-style PDFs with thin-but-real text layers that the strict signal
       misses.

    Args:
        pdf_path: Source PDF file.
        sample_pages: Number of pages to probe.
        text_ratio: Minimum share of probed pages that must be text-rich for signal (1)
            to fire.
        min_text_chars: Per-page non-whitespace character count that defines
            "text-rich"; the sparse signal uses half this value as its median threshold.

    Returns:
        True if the PDF should be treated as digital-born.

    Raises:
        FileNotFoundError: `pdf_path` does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        msg = f"PDF not found: {pdf_path}"
        raise FileNotFoundError(msg)

    doc = pdf_oxide.PdfDocument(str(pdf_path))
    page_count = doc.page_count()
    if page_count == 0:
        return False

    indices = _sample_indices(page_count, sample_pages)
    chars_per_page: list[int] = []
    for idx in indices:
        spans = doc.extract_spans(idx)
        non_whitespace_count = sum(1 for s in spans for c in s.text if not c.isspace())
        chars_per_page.append(non_whitespace_count)

    rich = sum(1 for c in chars_per_page if c >= min_text_chars)
    strict_ratio = rich / len(indices)
    strict_pass = strict_ratio >= text_ratio

    median_chars = statistics.median(chars_per_page)
    sparse_threshold = min_text_chars / 2
    has_rich_page = any(c >= min_text_chars for c in chars_per_page)
    sparse_pass = median_chars >= sparse_threshold and has_rich_page

    verdict = strict_pass or sparse_pass
    logger.debug(
        "digital_born probe: pdf=%s chars_per_page=%s rich=%d/%d strict_ratio=%.2f "
        "median=%.0f sparse_threshold=%.0f strict_pass=%s sparse_pass=%s verdict=%s",
        pdf_path.name,
        chars_per_page,
        rich,
        len(indices),
        strict_ratio,
        median_chars,
        sparse_threshold,
        strict_pass,
        sparse_pass,
        verdict,
    )
    return verdict


def _sample_indices(page_count: int, sample_pages: int) -> list[int]:
    """Pick up to `sample_pages` distinct page indices evenly distributed."""
    n = min(max(1, sample_pages), page_count)
    if n == 1:
        return [0]

    step = (page_count - 1) / (n - 1)

    return sorted({round(i * step) for i in range(n)})

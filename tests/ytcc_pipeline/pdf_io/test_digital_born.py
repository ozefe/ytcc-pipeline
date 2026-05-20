"""Tests for ytcc_pipeline.pdf_io.digital_born."""

from typing import TYPE_CHECKING, Any, cast

import pdf_oxide
import pytest

from ytcc_pipeline.pdf_io.digital_born import detect_digital_born

if TYPE_CHECKING:
    from pathlib import Path


def _make_pdf(path: Path, per_page_chars: list[int]) -> None:
    """Build a deterministic A4 PDF whose pages each carry `chars` x-glyphs.

    The digital-born detector probes the text layer and counts non-whitespace characters
    per page. Passing a list like `[200, 60, 60, 60, 60]` lets a test pin the exact
    distribution the heuristic needs to see -- the page text itself isn't meaningful,
    only its non-whitespace character count.

    `pdf_oxide`'s PyO3 stubs annotate every fluent-builder method with a spurious
    `slf_handle` first parameter, so the builder is cast to `Any` to keep the chain
    readable.
    """
    builder = cast("Any", pdf_oxide.DocumentBuilder())
    for chars in per_page_chars:
        page = builder.a4_page()
        if chars > 0:
            page.at(72.0, 742.0).text("x" * chars)
        page.done()
    path.write_bytes(builder.build())


def test_digital_born_pdf_returns_true(tiny_pdf: Path) -> None:
    assert detect_digital_born(tiny_pdf) is True


def test_blank_pdf_returns_false(blank_pdf: Path) -> None:
    assert detect_digital_born(blank_pdf) is False


def test_min_text_chars_threshold_filters_low_text(tiny_pdf: Path) -> None:
    """A very strict char threshold should reject low-text content."""
    assert detect_digital_born(tiny_pdf, min_text_chars=100_000) is False


def test_sample_pages_capped_by_doc_length(blank_pdf: Path) -> None:
    """Asking for more samples than pages must not error."""
    assert detect_digital_born(blank_pdf, sample_pages=50) is False


def test_text_ratio_below_threshold_returns_false(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A PDF where only 1 of 5 pages is rich should be classified scanned at 0.5
    ratio."""
    path = tmp_path_factory.mktemp("mixed") / "mixed.pdf"
    _make_pdf(path, [500, 0, 0, 0, 0])

    assert detect_digital_born(path, sample_pages=5, text_ratio=0.5) is False
    assert detect_digital_born(path, sample_pages=5, text_ratio=0.1) is True


def test_sparse_but_consistent_text_classifies_digital_born(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A PDF with thin-but-real text on every page should beat the strict ratio.

    The strict signal fires for only 1 of 5 pages but the median page has more than half
    `min_text_chars` and at least one page clears the bar -- the sparse signal must
    catch it.
    """
    path = tmp_path_factory.mktemp("sparse") / "sparse.pdf"

    # Page 0 carries 200 chars (above min_text_chars=100); pages 1-4 carry 60 chars
    # (above sparse threshold 50, below strict 100).
    _make_pdf(path, [200, 60, 60, 60, 60])

    # Strict ratio alone would flunk this: 1/5 = 0.2 < 0.5. Sparse signal saves it.
    assert detect_digital_born(path, sample_pages=5, text_ratio=0.5) is True


def test_cover_page_only_text_does_not_trigger_sparse_pass(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A scanned PDF with a single text-bearing cover page should stay scanned.

    Guards against the sparse signal over-classifying -- median chars must actually
    clear `min_text_chars / 2`, not just "any text anywhere".
    """
    path = tmp_path_factory.mktemp("cover") / "cover.pdf"
    _make_pdf(path, [200, 0, 0, 0, 0])

    assert detect_digital_born(path, sample_pages=5, text_ratio=0.5) is False


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        detect_digital_born(tmp_path / "missing.pdf")

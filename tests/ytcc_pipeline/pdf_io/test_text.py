"""Tests for ytcc_pipeline.pdf_io.text."""

from typing import TYPE_CHECKING

import pytest

from ytcc_pipeline.pdf_io.text import (
    _pixel_bbox_to_pdf_region,
    extract_text_in_bbox,
    open_pdf_for_text,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestPixelBboxToPdfRegion:
    def test_full_page_at_72_dpi_yields_full_pdf_region(self) -> None:
        region = _pixel_bbox_to_pdf_region(
            (0.0, 0.0, 595.0, 842.0),
            source_dpi=72,
            page_height_pt=842.0,
        )
        assert region == pytest.approx((0.0, 0.0, 595.0, 842.0))

    def test_dpi_scaling_preserves_pdf_region(self) -> None:
        region = _pixel_bbox_to_pdf_region(
            (0.0, 0.0, 1190.0, 1684.0),
            source_dpi=144,
            page_height_pt=842.0,
        )
        assert region == pytest.approx((0.0, 0.0, 595.0, 842.0))

    @pytest.mark.parametrize(
        ("pixel_bbox", "expected_y", "expected_h"),
        [
            ((0.0, 0.0, 100.0, 50.0), 792.0, 50.0),  # top of image -> top of PDF
            ((0.0, 800.0, 100.0, 842.0), 0.0, 42.0),  # bottom of image -> bottom of PDF
        ],
    )
    def test_pixel_y_flips_to_pdf_y(
        self,
        pixel_bbox: tuple[float, float, float, float],
        expected_y: float,
        expected_h: float,
    ) -> None:
        _, y, _, h = _pixel_bbox_to_pdf_region(
            pixel_bbox,
            source_dpi=72,
            page_height_pt=842.0,
        )
        assert y == pytest.approx(expected_y)
        assert h == pytest.approx(expected_h)


class TestTextInBbox:
    def test_extracts_known_text_at_72_dpi(self, tiny_pdf: Path) -> None:
        doc = open_pdf_for_text(tiny_pdf)

        text = extract_text_in_bbox(doc, 0, (60.0, 75.0, 535.0, 115.0), source_dpi=72)

        assert text is not None
        assert "Page 1 title" in text

    def test_extracts_known_text_at_300_dpi(self, tiny_pdf: Path) -> None:
        doc = open_pdf_for_text(tiny_pdf)
        # scale all coordinates by 300/72 ≈ 4.1666...
        scale = 300 / 72
        bbox = (60.0 * scale, 75.0 * scale, 535.0 * scale, 115.0 * scale)

        text = extract_text_in_bbox(doc, 0, bbox, source_dpi=300)

        assert text is not None
        assert "Page 1 title" in text

    def test_returns_none_for_empty_region(self, tiny_pdf: Path) -> None:
        doc = open_pdf_for_text(tiny_pdf)

        text = extract_text_in_bbox(doc, 0, (0.0, 0.0, 30.0, 30.0), source_dpi=72)

        assert text is None

    def test_normalizes_internal_whitespace(self, tiny_pdf: Path) -> None:
        doc = open_pdf_for_text(tiny_pdf)

        text = extract_text_in_bbox(doc, 0, (60.0, 75.0, 535.0, 175.0), source_dpi=72)

        assert text is not None
        assert "  " not in text
        assert text == text.strip()

    def test_open_pdf_for_text_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            open_pdf_for_text(tmp_path / "missing.pdf")

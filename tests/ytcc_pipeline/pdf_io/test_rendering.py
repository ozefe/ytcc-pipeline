"""Tests for ytcc_pipeline.pdf_io.rendering."""

import struct
from typing import TYPE_CHECKING

import pytest

from ytcc_pipeline.pdf_io.rendering import render_pages

if TYPE_CHECKING:
    from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SOI = b"\xff\xd8"


def _read_png_dims(path: Path) -> tuple[int, int]:
    """Return (width, height) of a PNG by parsing its IHDR chunk."""
    with path.open("rb") as f:
        assert f.read(8) == PNG_SIGNATURE

        f.read(4)
        assert f.read(4) == b"IHDR"

        width, height = struct.unpack(">II", f.read(8))
        return width, height


def test_render_pages_writes_one_file_per_page(tiny_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "renders"
    out.mkdir()

    paths = render_pages(tiny_pdf, out, dpi=72, max_workers=2)

    assert len(paths) == 2
    assert all(p.stat().st_size > 0 for p in paths)
    assert paths[0].name == "page-0000.jpg"
    assert paths[1].name == "page-0001.jpg"


def test_render_pages_png_extension_and_signature(
    tiny_pdf: Path, tmp_path: Path
) -> None:
    out = tmp_path / "renders"
    out.mkdir()

    paths = render_pages(tiny_pdf, out, dpi=72, image_format="png", max_workers=2)

    assert paths[0].name == "page-0000.png"
    assert paths[0].read_bytes()[:8] == PNG_SIGNATURE


def test_render_pages_jpeg_extension_and_signature(
    tiny_pdf: Path, tmp_path: Path
) -> None:
    out = tmp_path / "renders"
    out.mkdir()

    paths = render_pages(tiny_pdf, out, dpi=72, image_format="jpeg", max_workers=2)

    assert paths[0].name == "page-0000.jpg"
    assert paths[0].read_bytes()[:2] == JPEG_SOI


def test_render_pages_dpi_scales_image_dimensions(
    tiny_pdf: Path, tmp_path: Path
) -> None:
    out_150 = tmp_path / "r150"
    out_300 = tmp_path / "r300"
    out_150.mkdir()
    out_300.mkdir()

    p150 = render_pages(tiny_pdf, out_150, dpi=150, image_format="png", max_workers=2)
    p300 = render_pages(tiny_pdf, out_300, dpi=300, image_format="png", max_workers=2)

    w150, h150 = _read_png_dims(p150[0])
    w300, h300 = _read_png_dims(p300[0])

    assert w300 == pytest.approx(w150 * 2, abs=2)
    assert h300 == pytest.approx(h150 * 2, abs=2)


def test_render_pages_rejects_missing_output_dir(
    tiny_pdf: Path, tmp_path: Path
) -> None:
    nonexistent = tmp_path / "does-not-exist"

    with pytest.raises(NotADirectoryError):
        render_pages(tiny_pdf, nonexistent, dpi=72, max_workers=2)


def test_render_pages_rejects_invalid_format(tiny_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "renders"
    out.mkdir()

    with pytest.raises(ValueError, match="image_format"):
        render_pages(tiny_pdf, out, image_format="webp", max_workers=2)  # type: ignore[arg-type]

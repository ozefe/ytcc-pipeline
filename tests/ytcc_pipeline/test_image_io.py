"""Tests for ytcc_pipeline.image_io."""

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from ytcc_pipeline.image_io import crop_from_page, save_crop

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def page_image(tmp_path: Path) -> Path:
    """A 400x600 PNG with three colored stripes; used to verify cropping coords."""
    img = Image.new("RGB", (400, 600), color=(255, 255, 255))
    arr = np.array(img)
    arr[0:200, :] = (255, 0, 0)  # red top
    arr[200:400, :] = (0, 255, 0)  # green middle
    arr[400:600, :] = (0, 0, 255)  # blue bottom
    path = tmp_path / "page.png"

    Image.fromarray(arr).save(path)
    return path


@pytest.mark.parametrize(
    ("bbox", "expected_channel"),
    [
        ((0.0, 0.0, 400.0, 200.0), 0),  # red top stripe
        ((0.0, 200.0, 400.0, 400.0), 1),  # green middle stripe
        ((0.0, 400.0, 400.0, 600.0), 2),  # blue bottom stripe
    ],
)
def test_crop_from_page_extracts_correct_stripe(
    page_image: Path,
    bbox: tuple[float, float, float, float],
    expected_channel: int,
) -> None:
    crop = crop_from_page(page_image, bbox=bbox)

    assert crop.shape == (200, 400, 3)
    for channel in range(3):
        value = 255 if channel == expected_channel else 0
        assert np.all(crop[:, :, channel] == value)


def test_crop_from_page_clamps_to_image_bounds(page_image: Path) -> None:
    crop = crop_from_page(page_image, bbox=(-50.0, -50.0, 450.0, 700.0))

    assert crop.shape == (600, 400, 3)


def test_crop_from_page_rejects_inverted_bbox(page_image: Path) -> None:
    with pytest.raises(ValueError, match="bbox"):
        crop_from_page(page_image, bbox=(100.0, 100.0, 50.0, 50.0))


def test_save_crop_png_format(tmp_path: Path) -> None:
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[..., 0] = 255
    out = tmp_path / "crop.png"

    save_crop(arr, out, image_format="png")

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_crop_jpeg_format(tmp_path: Path) -> None:
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[..., 2] = 255
    out = tmp_path / "crop.jpg"

    save_crop(arr, out, image_format="jpeg")

    assert out.read_bytes()[:2] == b"\xff\xd8"


def test_save_crop_round_trip_preserves_pixels(tmp_path: Path) -> None:
    arr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    out = tmp_path / "crop.png"

    save_crop(arr, out, image_format="png")
    loaded = np.array(Image.open(out))

    np.testing.assert_array_equal(loaded, arr)


def test_save_crop_rejects_invalid_format(tmp_path: Path) -> None:
    arr = np.zeros((5, 5, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="image_format"):
        save_crop(arr, tmp_path / "x.bmp", image_format="bmp")  # pyright: ignore[reportArgumentType]

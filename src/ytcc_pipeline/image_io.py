"""All cv2 image I/O for the pipeline: read, write, crop.

Pairs with the pdf_io layer: each names a single external-library boundary. Anything
that touches cv2 lives in this module; anything that touches pdf_oxide lives next door.
Three responsibilities here:

- `read_rgb`: decode a PNG/JPEG file to an HWC uint8 RGB ndarray. Wraps `cv2.imread` and
  `cv2.cvtColor`, raising `OSError` on read failure instead of silently returning `None`
- `crop_from_page`: slice a bbox region out of an HWC ndarray.
- `save_crop`: encode an HWC uint8 RGB ndarray to PNG / JPEG.

Cropping is technically a transform not strictly I/O, but every load/save path in the
codebase needs the same RGB-ordered ndarray contract this module enforces, so it belongs
here.
"""

import logging
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from .config import ImageFormat

__all__ = ["crop_from_page", "read_rgb", "save_crop"]

logger = logging.getLogger(__name__)


def read_rgb(path: Path | str) -> np.ndarray:
    """Decode an image file as an HWC uint8 RGB numpy array.

    `cv2.imread` returns BGR and returns `None` on read failure; this helper swaps to
    RGB and turns the failure case into a loud `OSError`.

    Args:
        path: Image file path. PNG / JPEG / WebP etc. -- anything cv2 decodes.

    Returns:
        HWC uint8 RGB numpy array.

    Raises:
        OSError: cv2 could not decode the file.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        msg = f"cv2.imread failed: {path}"
        raise OSError(msg)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_from_page(
    page_image: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Crop the given bbox region out of a page image.

    Args:
        page_image: HWC uint8 RGB numpy array. Load files via `read_rgb` first.
        bbox: `(x1, y1, x2, y2)` in pixel coords, origin top-left. Floats are rounded to
            the nearest integer; out-of-bounds coords are clamped to the page edges.

    Returns:
        HWC uint8 RGB numpy array.

    Raises:
        ValueError: bbox has non-positive area, or falls entirely outside the page after
            clamping.
    """
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        msg = f"bbox has non-positive area: {bbox!r}"
        raise ValueError(msg)

    h, w = page_image.shape[:2]

    x1_i = max(0, round(x1))
    y1_i = max(0, round(y1))
    x2_i = min(w, round(x2))
    y2_i = min(h, round(y2))

    if x2_i - x1_i <= 0 or y2_i - y1_i <= 0:
        msg = f"bbox falls entirely outside page image: {bbox!r}"
        raise ValueError(msg)

    # Surface the case where the bbox went meaningfully outside the page -- this is
    # layout-detector noise we want to know about, not raise on.
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        logger.debug(
            "bbox clamped: orig=(%.1f,%.1f,%.1f,%.1f) "
            "clamped=(%d,%d,%d,%d) page=(%d,%d)",
            x1,
            y1,
            x2,
            y2,
            x1_i,
            y1_i,
            x2_i,
            y2_i,
            w,
            h,
        )
    return page_image[y1_i:y2_i, x1_i:x2_i]


def save_crop(
    crop: np.ndarray,
    output_path: Path,
    *,
    image_format: ImageFormat,
    jpeg_quality: int = 95,
) -> None:
    """Encode an HWC uint8 RGB crop to disk via cv2.

    Args:
        crop: HWC uint8 RGB numpy array (e.g. as returned by `crop_from_page`).
        output_path: Destination file path.
        image_format: `"jpeg"` or `"png"`.
        jpeg_quality: 1-100; ignored for PNG.

    Raises:
        ValueError: `image_format` is not one of the supported values.
        OSError: cv2 returned a failure result.
    """
    if image_format == "jpeg":
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            jpeg_quality,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        ]
    elif image_format == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    else:
        msg = f"image_format must be 'jpeg' or 'png', got {image_format!r}"
        raise ValueError(msg)

    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), bgr, params):
        logger.error(
            "crop save failed: path=%s shape=%s format=%s",
            output_path,
            crop.shape,
            image_format,
        )
        msg = f"cv2.imwrite failed: {output_path}"
        raise OSError(msg)

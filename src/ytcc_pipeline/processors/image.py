"""Block-crop saving -- the crop-and-save step shared by every crop-bearing route.

Single entry point for "given a layout bbox, save the pixels to disk and give me a
bundle-relative path I can put in `document.json`." Used by the IMAGE / FORMULA / TABLE
routes upfront and by the TEXT / REFERENCE MISS fallbacks.

This is also the natural home for any future preprocessing on the saved crops
(grayscale, denoise, deskew, contrast normalization, ...) -- all such transforms slot
between the crop step and the save step inside `save_block_crop`.

The `add_miss_marker_to_filename` helper is here because it operates on names produced
by this module's `_image_filename` -- it splices a `MISS-` marker into an already-saved
crop's filename when the formula stage decides recognition failed.

This module has no logger of its own -- per-block save lifecycle is logged at the call
sites (workers, serial loop) where the PDF / worker context is in scope. Disk I/O errors
surface from the image-I/O layer, which logs them with file-level context.
"""

import uuid
from typing import TYPE_CHECKING

from ytcc_pipeline.image_io import crop_from_page, save_crop

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from ytcc_pipeline.config import ImageFormat


def save_block_crop(  # noqa: PLR0913 -- every arg is mandatory per-crop state; bundling would just rename the call site
    *,
    bbox: tuple[float, float, float, float],
    label: str,
    page_image: np.ndarray,
    page_no: int,
    images_dir: Path,
    miss: bool,
    crop_format: ImageFormat,
    jpeg_quality: int,
) -> str:
    """Crop the bbox region out of `page_image`, save it, return the bundle path.

    Args:
        bbox: `(x1, y1, x2, y2)` in pixel coords at the page's render DPI.
        label: Layout label (e.g. `"formula"`, `"table"`) -- used as a filename infix so
            operators can grep by block kind.
        page_image: HWC uint8 RGB page image, already loaded.
        page_no: 1-based page number -- used as a filename prefix.
        images_dir: Destination directory (typically the pipeline temp dir's `images/`).
        miss: `True` to insert the `-MISS-` marker into the filename (text/reference
            fallbacks where extraction returned empty).
        crop_format: `"png"` or `"jpeg"`.
        jpeg_quality: 1-100. Ignored for PNG.

    Returns:
        Bundle-relative path (e.g. `"images/0014-formula-{uuid}.png"`)
        suitable for the block's `image_path` field in `document.json`.
    """
    crop = crop_from_page(page_image, bbox)
    crop_filename = _image_filename(page_no, label, miss=miss, ext=crop_format)
    save_crop(
        crop,
        images_dir / crop_filename,
        image_format=crop_format,
        jpeg_quality=jpeg_quality,
    )
    return f"images/{crop_filename}"


def add_miss_marker_to_filename(filename: str) -> str:
    """Splice `MISS-` into an existing crop filename before its uuid segment.

    Workers can't predict the future: every FORMULA crop is saved with `miss=False`
    upfront, and only after the model runs do we know which crops to mark as MISS.
    Renaming after the fact keeps the on-disk and in-JSON conventions identical to text
    / reference MISS fallbacks (`-MISS-` immediately before the uuid hex).

    Args:
        filename: Plain crop filename as produced by `_image_filename` with `miss=False`
            e.g. `"0014-formula-{32hex}.png"`.

    Returns:
        The same filename with `-MISS-` inserted before the uuid hex and extension.
        Idempotent: returns input unchanged if the marker is already present.
    """
    if "-MISS-" in filename:
        return filename

    name, dot, ext = filename.rpartition(".")

    # uuid4().hex carries no hyphens, so the rightmost hyphen separates the
    # {page}-{label} prefix from the uuid hex.
    prefix, _, uuid_part = name.rpartition("-")

    return f"{prefix}-MISS-{uuid_part}{dot}{ext}"


def _image_filename(page_no: int, label: str, *, miss: bool, ext: str) -> str:
    """Compose the per-crop filename."""
    marker = "MISS-" if miss else ""
    return f"{page_no:04d}-{label}-{marker}{uuid.uuid4().hex}.{ext}"

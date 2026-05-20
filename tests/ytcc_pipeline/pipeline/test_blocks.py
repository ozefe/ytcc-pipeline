"""Unit tests for `build_block` MISS-fallback handling.

The function is the single decision point for whether a MISS block gets its crop saved
to disk + referenced from the bundle. Exercising it directly with an in-memory page
image keeps these tests fast and independent of the worker / multiprocessing layer.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ytcc_pipeline.models.layout import LayoutDetection
from ytcc_pipeline.pipeline.blocks import build_block
from ytcc_pipeline.routing import Route
from ytcc_pipeline.schema import BlockType

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def page_image() -> np.ndarray:
    """A 400x600 RGB page image with a single solid stripe."""
    arr = np.full((600, 400, 3), 255, dtype=np.uint8)
    arr[100:200, 50:300] = (200, 200, 200)
    return arr


def _detection(label: str = "text", reading_order: int = 0) -> LayoutDetection:
    return LayoutDetection(
        label=label,
        label_id=1,
        confidence=0.9,
        bbox=(50.0, 100.0, 300.0, 200.0),
        reading_order=reading_order,
    )


def test_text_miss_with_default_set_writes_image(
    page_image: np.ndarray,
    tmp_path: Path,
) -> None:
    """Default `bundle_miss_images_for` keeps the legacy behavior -- crop saved."""
    block = build_block(
        _detection(label="text"),
        route=Route.TEXT,
        page_image=page_image,
        page_no=1,
        images_dir=tmp_path,
        crop_format="png",
        jpeg_quality=90,
        text=None,
        bundle_miss_images_for=frozenset(BlockType),
    )
    assert block.miss is True
    assert block.text is None
    assert block.image_path is not None
    assert "-MISS-" in block.image_path

    # A real file landed on disk under tmp_path with the same name.
    saved = list(tmp_path.iterdir())
    assert len(saved) == 1
    assert "-MISS-" in saved[0].name


def test_text_miss_with_text_excluded_skips_image(
    page_image: np.ndarray,
    tmp_path: Path,
) -> None:
    """Excluding TEXT from the set drops the crop save; entry still in JSON."""
    block = build_block(
        _detection(label="text"),
        route=Route.TEXT,
        page_image=page_image,
        page_no=1,
        images_dir=tmp_path,
        crop_format="png",
        jpeg_quality=90,
        text=None,
        bundle_miss_images_for=frozenset(),
    )
    assert block.miss is True
    assert block.text is None
    assert block.image_path is None

    # No crop file was written for this MISS.
    assert list(tmp_path.iterdir()) == []


def test_reference_miss_honors_per_type_setting(
    page_image: np.ndarray,
    tmp_path: Path,
) -> None:
    """REFERENCE MISS respects the per-type set independently of TEXT."""
    # TEXT-only set: REFERENCE MISS skips the image even though TEXT MISS wouldn't.
    block = build_block(
        _detection(label="reference_content"),
        route=Route.REFERENCE,
        page_image=page_image,
        page_no=1,
        images_dir=tmp_path,
        crop_format="png",
        jpeg_quality=90,
        text=None,
        bundle_miss_images_for=frozenset({BlockType.TEXT}),
    )
    assert block.miss is True
    assert block.image_path is None


def test_text_success_path_unaffected_by_setting(
    page_image: np.ndarray,
    tmp_path: Path,
) -> None:
    """Successful text extraction never depends on the MISS-image setting."""
    block = build_block(
        _detection(label="text"),
        route=Route.TEXT,
        page_image=page_image,
        page_no=1,
        images_dir=tmp_path,
        crop_format="png",
        jpeg_quality=90,
        text="extracted content",
        bundle_miss_images_for=frozenset(),
    )
    assert block.miss is False
    assert block.text == "extracted content"
    assert block.image_path is None


def test_crop_only_routes_always_save_image(
    page_image: np.ndarray,
    tmp_path: Path,
) -> None:
    """IMAGE / FORMULA crop saves on success ignore `bundle_miss_images_for`.

    The setting only affects the MISS branch. Non-MISS crops (an image block on a real
    figure) always land on disk.
    """
    block = build_block(
        _detection(label="figure"),
        route=Route.IMAGE,
        page_image=page_image,
        page_no=1,
        images_dir=tmp_path,
        crop_format="png",
        jpeg_quality=90,
        text=None,
        bundle_miss_images_for=frozenset(),
    )
    assert block.miss is False
    assert block.image_path is not None
    assert "-MISS-" not in block.image_path


def test_table_route_saves_crop_and_emits_table_type(
    page_image: np.ndarray,
    tmp_path: Path,
) -> None:
    """TABLE route saves the table crop and emits a TABLE-typed block.

    The table stage fills in cells / n_rows / n_cols later; the worker's job is just to
    get the crop on disk so the table stage can hand it to RapidTable.
    """
    block = build_block(
        _detection(label="table"),
        route=Route.TABLE,
        page_image=page_image,
        page_no=14,
        images_dir=tmp_path,
        crop_format="png",
        jpeg_quality=90,
        text=None,
        bundle_miss_images_for=frozenset(),
    )
    assert block.type is BlockType.TABLE
    assert block.miss is False
    assert block.text is None
    assert block.image_path is not None
    assert "-table-" in block.image_path
    assert "-MISS-" not in block.image_path

    # The crop landed on disk.
    saved = list(tmp_path.iterdir())
    assert len(saved) == 1
    assert "0014-table-" in saved[0].name

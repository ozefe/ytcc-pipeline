"""Unit tests for `processors.table`.

The RapidTable + analyzer integration is covered by the API smoke test and the
benchmark; here we exercise the small pure-Python pieces (engine wrapper init shape) and
one end-to-end shape test of `run_table_stage` with a stub engine.
"""

from typing import TYPE_CHECKING

import cv2
import numpy as np

from ytcc_pipeline.config import PipelineConfig
from ytcc_pipeline.processors.table import (
    make_table_engine,
    run_table_stage,
)
from ytcc_pipeline.schema import Block, BlockType, Page

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _stub_table_crop(tmp_path: Path, image_path: str) -> None:
    crop_abs = tmp_path / image_path
    crop_abs.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_abs), np.full((100, 100, 3), 255, dtype=np.uint8))


class _StubEngine:
    """Returns a deterministic 2x2 cell grid for every crop in the batch."""

    def __init__(self) -> None:
        self.calls: list[int] = []
        # 4 cells laid out as (10..50, 10..50), (50..90, 10..50), (10..50, 50..90),
        # (50..90, 50..90) -- a uniform 2x2 grid.
        self.cells = np.array(
            [
                [10, 10, 50, 10, 50, 50, 10, 50],
                [50, 10, 90, 10, 90, 50, 50, 50],
                [10, 50, 50, 50, 50, 90, 10, 90],
                [50, 50, 90, 50, 90, 90, 50, 90],
            ],
            dtype=np.float32,
        )
        self.logic = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int64,
        )

    def structure(self, crops: list[np.ndarray]):
        self.calls.append(len(crops))
        return [(self.cells.copy(), self.logic.copy()) for _ in crops]


def test_make_table_engine_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """`make_table_engine` returns a `TableEngine` whose `.recognize_structure` is
    callable.

    The inner ONNX session is replaced with a stub so the test runs without
    rapid_table's heavy init.
    """
    from ytcc_pipeline.processors import table as table_mod

    class _StubInnerEngine:
        def __call__(
            self,
            crops: list[np.ndarray],
            batch_size: int = 1,
        ) -> object:
            del batch_size  # unused; signature-compat with rapid_table
            cell_bboxes = [np.zeros((0, 8), dtype=np.float32) for _ in crops]
            logic_points = [np.zeros((0, 4), dtype=np.int64) for _ in crops]

            class _Out:
                pass

            out = _Out()
            out.cell_bboxes = cell_bboxes  # type: ignore[attr-defined]
            out.logic_points = logic_points  # type: ignore[attr-defined]
            return out

    def _stub_build(self: object) -> None:
        # Bypass `TableEngine._build_engine`'s rapid_table import + ONNX init.
        self._engine = _StubInnerEngine()  # type: ignore[attr-defined]

    monkeypatch.setattr(table_mod.TableEngine, "_build_engine", _stub_build)
    engine = make_table_engine(device="cpu", batch_size=2)
    out = engine.recognize_structure([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert len(out) == 1

    cell_bboxes, logic_points = out[0]
    assert cell_bboxes.shape == (0, 8)
    assert logic_points.shape == (0, 4)


def test_run_table_stage_fills_in_cells(tmp_path: Path) -> None:
    """`run_table_stage` replaces TABLE placeholders with a structured cell grid."""
    image_path = "images/0001-table-stub.png"
    _stub_table_crop(tmp_path, image_path)
    table = Block(
        reading_order=0,
        label="table",
        type=BlockType.TABLE,
        bbox=(200.0, 300.0, 400.0, 500.0),  # 200x200, above default min_side_px
        confidence=0.9,
        image_path=image_path,
    )
    pages = [Page(page_no=1, width_px=1240, height_px=1754, blocks=[table])]

    new_pages = run_table_stage(
        pages,
        pdf_path=tmp_path / "fake.pdf",
        digital_born=False,
        work_dir=tmp_path,
        table_engine=_StubEngine(),  # type: ignore[arg-type]
        scanned_ocr=None,
        cfg=PipelineConfig(table_enabled=True),
    )

    block = new_pages[0].blocks[0]
    assert block.type is BlockType.TABLE
    assert block.n_rows == 2
    assert block.n_cols == 2
    assert block.cells is not None
    assert len(block.cells) == 4

    # Cells are emitted in (row, col) order.
    assert [(c.row_start, c.col_start) for c in block.cells] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]

    # No OCR engine + scanned path ⇒ every cell's text is None.
    assert all(c.text is None for c in block.cells)


def test_run_table_stage_skips_tiny_tables(tmp_path: Path) -> None:
    """Tables below `table_min_side_px` fall back to image-only (cells=None)."""
    image_path = "images/0001-table-tiny.png"
    _stub_table_crop(tmp_path, image_path)
    tiny = Block(
        reading_order=0,
        label="table",
        type=BlockType.TABLE,
        bbox=(0.0, 0.0, 50.0, 50.0),  # well below default 120
        confidence=0.9,
        image_path=image_path,
    )
    pages = [Page(page_no=1, width_px=1240, height_px=1754, blocks=[tiny])]
    new_pages = run_table_stage(
        pages,
        pdf_path=tmp_path / "fake.pdf",
        digital_born=False,
        work_dir=tmp_path,
        table_engine=_StubEngine(),  # type: ignore[arg-type]
        scanned_ocr=None,
        cfg=PipelineConfig(table_enabled=True),
    )
    assert new_pages[0].blocks[0].cells is None


def test_run_table_stage_disabled_is_noop(tmp_path: Path) -> None:
    """`table_enabled=False` returns the input page list unchanged."""
    image_path = "images/0001-table-stub.png"
    _stub_table_crop(tmp_path, image_path)
    table = Block(
        reading_order=0,
        label="table",
        type=BlockType.TABLE,
        bbox=(200.0, 300.0, 400.0, 500.0),
        confidence=0.9,
        image_path=image_path,
    )
    pages = [Page(page_no=1, width_px=1240, height_px=1754, blocks=[table])]
    cfg = PipelineConfig(table_enabled=False)
    new_pages = run_table_stage(
        pages,
        pdf_path=tmp_path / "fake.pdf",
        digital_born=False,
        work_dir=tmp_path,
        table_engine=_StubEngine(),  # type: ignore[arg-type]
        scanned_ocr=None,
        cfg=cfg,
    )

    assert new_pages is pages
    assert new_pages[0].blocks[0].cells is None

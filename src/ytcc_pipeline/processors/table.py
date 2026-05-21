"""Table stage: SLANet+ structure recovery + per-cell text extraction.

Runs after the per-page workers. For each TABLE block (saved as a crop by the worker,
`cells=None`):

1. Skip the block if its bbox is below `cfg.table_min_side_px`: keeps the image-only
   fallback the worker already wrote.
2. Run RapidTable SLANet+ on the table crop -> cell polygons and per-cell
   `[row_start, row_end, col_start, col_end]`. Tables are batched `cfg.table_batch_size`
   at a time.
3. Translate cell polygons from crop coords to page coords.
4. Extract cell text: pdf_oxide on digital-born, RapidOCR per cell on scanned. Empty /
   failed extractions leave `Cell.text=None`.
5. Replace the TABLE block with a structured version carrying `n_rows`, `n_cols`,
   `cells`. The original crop in `image_path` stays as a fallback for renderers.

Degenerate output (< 2 cells, unreadable crop, or RapidTable raising) falls back to
image-only: the crop stays bundled, `cells` stays `None`.
"""

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from ytcc_pipeline.block_type import BlockType
from ytcc_pipeline.image_io import crop_from_page, read_rgb
from ytcc_pipeline.pdf_io.text import extract_text_in_bbox, open_pdf_for_text
from ytcc_pipeline.pipeline.blocks import replace_blocks
from ytcc_pipeline.schema import Block, Cell, Page

if TYPE_CHECKING:
    from pathlib import Path

    from ytcc_pipeline.config import PipelineConfig
    from ytcc_pipeline.models.ocr import OcrExtractor

__all__ = ["TableEngine", "make_table_engine", "run_table_stage"]

logger = logging.getLogger(__name__)

type _Bbox = tuple[float, float, float, float]


class _TableTarget(NamedTuple):
    """One TABLE block queued for structure recognition.

    `page_idx` / `block_idx` locate the block in the page list so the recovered cell
    grid can be spliced back into the original slot. `crop_path` is the absolute path
    of the saved crop (worker output); `bbox` is the table's bbox in page-pixel coords
    at the render DPI, kept so the per-cell text extraction can translate crop-space
    cell polygons to page-space coords.
    """

    page_idx: int
    block_idx: int
    crop_path: Path
    bbox: _Bbox


# SLANet+ returns at least header + one body cell on a real table. Anything below this
# collapses to image-only fallback.
_MIN_CELLS_FOR_STRUCTURE = 2


class TableEngine:
    """RapidTable SLANet+ wrapper, structure-only (`use_ocr=False`).

    rapid_table's bundled engine_cfg.yaml emits `gpu_id` under `cuda_ep_cfg` and
    ONNXRuntime expects `device_id`. We patch the class-level config once before
    constructing the inner engine.
    """

    def __init__(self, *, device: str, batch_size: int) -> None:
        self._device = device
        self._batch_size = batch_size
        self._engine: Any | None = None
        self._build_engine()

    def _build_engine(self) -> None:
        # `rapid_table` pulls in a heavy ONNXRuntime stack (~1 GiB import cost). Defer
        # the import to constructor time so importing this module for typing remains
        # cheap.
        from rapid_table import (  # pyright: ignore[reportMissingImports] # noqa: PLC0415
            ModelType,
            RapidTable,
            RapidTableInput,
        )
        from rapid_table.inference_engine.base import (  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
            InferSession,
        )

        ep_cfg = InferSession.engine_cfg["onnxruntime"]["cuda_ep_cfg"]
        if "gpu_id" in ep_cfg:
            ep_cfg["device_id"] = ep_cfg.pop("gpu_id")

        use_cuda = self._device.startswith("cuda")
        gpu_id = (
            int(self._device.split(":", 1)[1])
            if use_cuda and ":" in self._device
            else 0
        )
        engine_cfg = {"use_cuda": use_cuda, "cuda_ep_cfg.device_id": gpu_id}
        t0 = time.perf_counter()
        try:
            self._engine = RapidTable(
                RapidTableInput(
                    model_type=ModelType.SLANETPLUS,
                    engine_cfg=engine_cfg,
                    use_ocr=False,
                ),
            )
        except Exception:
            logger.exception(
                "table engine load failed: device=%s batch_size=%d",
                self._device,
                self._batch_size,
            )
            raise

        logger.info(
            "table engine ready: device=%s batch_size=%d load_elapsed_s=%.2f",
            self._device,
            self._batch_size,
            time.perf_counter() - t0,
        )

    def recognize_structure(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return `[(cell_polygons, logic_points)]` for each crop."""
        # `_build_engine` runs in `__init__` and unconditionally assigns `self._engine`;
        # the assert documents that invariant for the type checker without raising in a
        # real failure path.
        assert self._engine is not None  # noqa: S101

        out = self._engine(crops, batch_size=self._batch_size)
        return list(zip(out.cell_bboxes, out.logic_points, strict=True))


def make_table_engine(*, device: str, batch_size: int) -> TableEngine:
    """Build a `TableEngine` (RapidTable SLANet+, structure-only).

    Thin wrapper kept symmetric with `models.layout.make_analyzer` so every
    cross-process owner (the FastAPI lifespan, the per-call library path) uses the same
    factory.

    Args:
        device: ONNXRuntime device string, e.g. `"cuda:0"` or `"cpu"`.
        batch_size: Tables per SLANet+ structure forward pass.

    Returns:
        A ready-to-use `TableEngine`.
    """
    return TableEngine(device=device, batch_size=batch_size)


def run_table_stage(  # noqa: PLR0913  -- stage threads through PDF + work dir + engines + config; refactor would just move the args into a wrapper
    pages: list[Page],
    *,
    pdf_path: Path,
    digital_born: bool,
    work_dir: Path,
    table_engine: TableEngine | None,
    scanned_ocr: OcrExtractor | None,
    cfg: PipelineConfig,
) -> list[Page]:
    """Fill in `Cell` grids on every TABLE block.

    Skip cases (`table_enabled=False`, no engine, no TABLE blocks) leave the page list
    unchanged.
    """
    pdf_name = pdf_path.name

    if not cfg.table_enabled:
        logger.info("stage table: pdf=%s skipped reason=disabled", pdf_name)
        return pages

    if table_engine is None:
        logger.info("stage table: pdf=%s skipped reason=no_engine", pdf_name)
        return pages

    targets, skipped_small = _collect_table_targets(
        pages,
        work_dir=work_dir,
        min_side_px=cfg.table_min_side_px,
        pdf_name=pdf_name,
    )

    if not targets:
        logger.info(
            "stage table: pdf=%s skipped reason=no_blocks skipped_small=%d",
            pdf_name,
            skipped_small,
        )
        return pages

    logger.info(
        "stage table start: pdf=%s tables=%d batch_size=%d skipped_small=%d "
        "digital_born=%s",
        pdf_name,
        len(targets),
        cfg.table_batch_size,
        skipped_small,
        digital_born,
    )

    t0 = time.perf_counter()
    pdf_doc = open_pdf_for_text(pdf_path) if digital_born else None
    blocks_by_page: dict[int, list[Block]] = {
        idx: list(page.blocks) for idx, page in enumerate(pages)
    }

    fallback = 0
    for start in range(0, len(targets), cfg.table_batch_size):
        batch = targets[start : start + cfg.table_batch_size]
        crops = [_read_or_placeholder(t.crop_path) for t in batch]

        try:
            structures = table_engine.recognize_structure(crops)
        except Exception:  # noqa: BLE001  -- one bad batch falls back to image-only; the rest of the stage continues
            logger.warning(
                "table batch failed: pdf=%s n=%d",
                pdf_name,
                len(batch),
                exc_info=True,
            )
            fallback += len(batch)
            continue

        for target, (cell_polygons, logic_points), crop_img in zip(
            batch,
            structures,
            crops,
            strict=True,
        ):
            page = pages[target.page_idx]
            if len(cell_polygons) < _MIN_CELLS_FOR_STRUCTURE:
                logger.warning(
                    "table fallback degenerate: pdf=%s page=%d bbox=%s cells=%d",
                    pdf_name,
                    page.page_no,
                    target.bbox,
                    len(cell_polygons),
                )
                fallback += 1
                continue

            cells = _build_cells(
                cell_polygons=cell_polygons,
                logic_points=logic_points,
                crop_img=crop_img,
                table_bbox=target.bbox,
                page_width_px=page.width_px,
                page_height_px=page.height_px,
                page_idx=target.page_idx,
                digital_born=digital_born,
                pdf_doc=pdf_doc,
                scanned_ocr=scanned_ocr,
                source_dpi=cfg.render_dpi,
            )

            n_rows = max((c.row_end for c in cells), default=-1) + 1
            n_cols = max((c.col_end for c in cells), default=-1) + 1
            n_text_cells = sum(1 for c in cells if c.text)
            original = blocks_by_page[target.page_idx][target.block_idx]
            blocks_by_page[target.page_idx][target.block_idx] = replace(
                original,
                n_rows=n_rows,
                n_cols=n_cols,
                cells=tuple(cells),
            )

            logger.debug(
                "table cell grid: pdf=%s page=%d %dx%d cells=%d cells_with_text=%d",
                pdf_name,
                page.page_no,
                n_rows,
                n_cols,
                len(cells),
                n_text_cells,
            )

    logger.info(
        "stage table: pdf=%s tables=%d fallback=%d elapsed_s=%.2f",
        pdf_name,
        len(targets),
        fallback,
        time.perf_counter() - t0,
    )
    return replace_blocks(pages, blocks_by_page)


def _collect_table_targets(
    pages: list[Page],
    *,
    work_dir: Path,
    min_side_px: int,
    pdf_name: str,
) -> tuple[list[_TableTarget], int]:
    """Walk pages and emit one `_TableTarget` per TABLE block big enough to process.

    Args:
        pages: Pages emitted by the block stage.
        work_dir: Pipeline temp dir. Used to resolve bundle-relative `image_path`
            strings to absolute on-disk paths.
        min_side_px: Minimum bbox side (in source-page pixels). Tables with either
            dimension below this skip structure recognition and keep the worker's
            image-only crop. Filters out spurious `table` detections on small inline
            elements.
        pdf_name: Source PDF filename for log correlation.

    Returns:
        `(targets, skipped_small)` where `targets` is the list of TABLE blocks to
        process and `skipped_small` counts how many TABLE blocks were filtered out for
        being below `min_side_px`.
    """
    targets: list[_TableTarget] = []
    skipped_small = 0
    for page_idx, page in enumerate(pages):
        for block_idx, block in enumerate(page.blocks):
            if block.type is not BlockType.TABLE or block.image_path is None:
                continue

            x1, y1, x2, y2 = block.bbox
            if x2 - x1 < min_side_px or y2 - y1 < min_side_px:
                skipped_small += 1
                logger.debug(
                    "table skip small: pdf=%s page=%d bbox=%s min_side_px=%d",
                    pdf_name,
                    page.page_no,
                    block.bbox,
                    min_side_px,
                )
                continue

            targets.append(
                _TableTarget(
                    page_idx=page_idx,
                    block_idx=block_idx,
                    crop_path=work_dir / block.image_path,
                    bbox=block.bbox,
                ),
            )

    return targets, skipped_small


def _build_cells(  # noqa: PLR0913  -- consolidates per-table state into one helper; splitting would just spread the params
    *,
    cell_polygons: np.ndarray,
    logic_points: np.ndarray,
    crop_img: np.ndarray,
    table_bbox: _Bbox,
    page_width_px: int,
    page_height_px: int,
    page_idx: int,
    digital_born: bool,
    pdf_doc: Any | None,  # noqa: ANN401  -- pdf_oxide.PdfDocument is a PyO3 type without static stubs
    scanned_ocr: OcrExtractor | None,
    source_dpi: int,
) -> list[Cell]:
    """Translate cells to page coords and emit them sorted by (row, col)."""
    table_x, table_y = table_bbox[0], table_bbox[1]
    crop_h, crop_w = crop_img.shape[:2]

    triples: list[tuple[_Bbox, _Bbox, np.ndarray]] = []
    for polygon, logic in zip(cell_polygons, logic_points, strict=True):
        rect_crop = _polygon_rect(polygon, max_w=crop_w, max_h=crop_h)
        rect_page = _clamp(
            (
                rect_crop[0] + table_x,
                rect_crop[1] + table_y,
                rect_crop[2] + table_x,
                rect_crop[3] + table_y,
            ),
            page_width_px,
            page_height_px,
        )
        triples.append((rect_page, rect_crop, logic))

    triples.sort(key=lambda t: (int(t[2][0]), int(t[2][2])))

    out: list[Cell] = []
    for rect_page, rect_crop, logic in triples:
        text = _extract_text(
            rect_page=rect_page,
            rect_crop=rect_crop,
            crop_img=crop_img,
            page_idx=page_idx,
            digital_born=digital_born,
            pdf_doc=pdf_doc,
            scanned_ocr=scanned_ocr,
            source_dpi=source_dpi,
        )
        row_start, row_end, col_start, col_end = (int(v) for v in logic[:4])
        out.append(
            Cell(
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                bbox=rect_page,
                text=text,
            )
        )

    return out


def _extract_text(  # noqa: PLR0913  -- per-cell state threads through the same args; bundling would obscure the call site
    *,
    rect_page: _Bbox,
    rect_crop: _Bbox,
    crop_img: np.ndarray,
    page_idx: int,
    digital_born: bool,
    pdf_doc: Any | None,  # noqa: ANN401  -- pdf_oxide.PdfDocument is a PyO3 type without static stubs
    scanned_ocr: OcrExtractor | None,
    source_dpi: int,
) -> str | None:
    """pdf_oxide for digital-born, per-cell OCR for scanned. None on failure."""
    if digital_born:
        if pdf_doc is None:
            return None

        try:
            return extract_text_in_bbox(
                pdf_doc,
                page_idx,
                rect_page,
                source_dpi=source_dpi,
            )
        except Exception:  # noqa: BLE001  -- per-cell extraction failure degrades to text=None; routine for tiny / off-page cells
            logger.debug(
                "table cell text failed: page_idx=%d bbox=%s",
                page_idx,
                rect_page,
                exc_info=True,
            )
            return None

    if scanned_ocr is None:
        return None

    try:
        cell_image = crop_from_page(crop_img, rect_crop)
    except ValueError:
        logger.debug(
            "table cell crop empty: page_idx=%d bbox=%s",
            page_idx,
            rect_crop,
        )
        return None

    return scanned_ocr.extract_batch([cell_image])[0]


def _polygon_rect(polygon: np.ndarray, *, max_w: int, max_h: int) -> _Bbox:
    """Collapse a flat 8-coord polygon to an axis-aligned bbox clamped to crop."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    return _clamp(
        (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        max_w,
        max_h,
    )


def _clamp(bbox: _Bbox, width: int, height: int) -> _Bbox:
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def _read_or_placeholder(path: Path) -> np.ndarray:
    """Read a crop. Return a 1x1 placeholder on failure to keep batch alignment."""
    try:
        return read_rgb(path)
    except OSError:
        logger.warning("table crop unreadable: %s", path, exc_info=True)
        return np.zeros((1, 1, 3), dtype=np.uint8)

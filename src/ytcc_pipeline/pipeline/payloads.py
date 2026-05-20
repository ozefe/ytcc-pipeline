"""Picklable IPC contracts for the pipeline's worker pools.

`DigitalBornWorkerArgs` / `ScannedWorkerArgs` are the per-page payloads passed into
`worker_digital_born` / `worker_scanned` via `ProcessPoolExecutor`. Workers return
`WorkerResult`, a 4-tuple `(page_idx, height, width, [Block, ...])`.

Every field in these payloads is either a primitive, a `pathlib`-style path string, a
frozen+slots dataclass (`Block`, `LayoutDetection`), or a stdlib container thereof -- so
the entire IPC graph pickles cleanly across `spawn` without any intermediate dict/tuple
shape.
"""

from typing import TYPE_CHECKING, NamedTuple

from ytcc_pipeline.schema import Block, BlockType

if TYPE_CHECKING:
    from ytcc_pipeline.config import ImageFormat
    from ytcc_pipeline.models.layout import LayoutDetection


class DigitalBornWorkerArgs(NamedTuple):
    """Picklable IPC payload for `worker_digital_born`."""

    page_idx: int
    page_path: str
    detections: list[LayoutDetection]
    pdf_path: str
    images_dir: str
    render_dpi: int
    crop_format: ImageFormat
    jpeg_quality: int
    bundle_miss_images_for: frozenset[BlockType]


class ScannedWorkerArgs(NamedTuple):
    """Picklable IPC payload for `worker_scanned`."""

    page_idx: int
    page_path: str
    detections: list[LayoutDetection]
    images_dir: str
    crop_format: ImageFormat
    jpeg_quality: int
    language: str
    ocr_batch_size: int
    ocr_min_score: float
    ocr_use_cuda: bool
    bundle_miss_images_for: frozenset[BlockType]


type WorkerResult = tuple[int, int, int, list[Block]]

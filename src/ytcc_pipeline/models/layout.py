"""Layout analyzer Protocol + SafeTensors backend factory.

The shipped backend is PP-DocLayoutV3 loaded via HuggingFace Transformers (PyTorch +
SafeTensors weights). The `LayoutAnalyzer` Protocol is kept as a seam so a future
portable / CPU-only fallback can slot in without touching the call sites.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ytcc_pipeline.config import PipelineConfig

__all__ = [
    "LayoutAnalyzer",
    "LayoutDetection",
    "make_analyzer",
    "make_analyzer_from_config",
]


@dataclass(slots=True, frozen=True)
class LayoutDetection:
    """A single layout block emitted by the layout model."""

    label: str
    label_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    reading_order: int


@runtime_checkable
class LayoutAnalyzer(Protocol):
    """Backends emit per-page detection lists in per-page reading order."""

    def analyze(
        self,
        page_images: Sequence[Path],
        *,
        batch_size: int = 8,
        confidence: float = 0.5,
    ) -> dict[int, list[LayoutDetection]]:  # pyright: ignore[reportReturnType]
        """Detect layout blocks on each page image.

        Args:
            page_images: Rendered page image paths in page order.
            batch_size: Pages per GPU forward pass.
            confidence: Minimum detection confidence; lower scores are dropped before
                the result is returned.

        Returns:
            ```json
                {
                    page_idx: [
                        LayoutDetection,
                        ...
                    ]
                }
            ```

            With detections sorted by per-page reading order.
        """

    def close(self) -> None:
        """Release backend resources (VRAM, model weights, processors)."""


def make_analyzer(
    *,
    device: str = "cuda:0",
    model_id: str | None = None,
    fp16: bool = False,
    fast_preproc: bool = False,
) -> LayoutAnalyzer:
    """Create the PP-DocLayoutV3 SafeTensors layout analyzer.

    Args:
        device: Torch device string (e.g. `"cuda:0"`, `"cpu"`).
        model_id: Override the model identifier / local snapshot path.
        fp16: Load + run the model in fp16.
        fast_preproc: Replace HF's PIL-based `AutoImageProcessor` with a cv2
            preprocessing pipeline overlapped with GPU inference via a producer thread.

    Returns:
        A `LayoutAnalyzer` instance.
    """
    # `layout_safetensors` imports torch + transformers. Defer until someone actually
    # constructs an analyzer so that just holding a `LayoutAnalyzer` Protocol reference
    # stays cheap.
    from .layout_safetensors import SafeTensorsLayoutAnalyzer  # noqa: PLC0415

    return SafeTensorsLayoutAnalyzer(
        model_id=model_id or "PaddlePaddle/PP-DocLayoutV3_safetensors",
        device=device,
        fp16=fp16,
        fast_preproc=fast_preproc,
    )


def make_analyzer_from_config(cfg: PipelineConfig) -> LayoutAnalyzer:
    """Build a `LayoutAnalyzer` from a `PipelineConfig`.

    Thin wrapper over `make_analyzer` so call sites that already hold a `PipelineConfig`
    (the FastAPI lifespan, the orchestrator) don't have to spell out each constructor
    argument.

    Args:
        cfg: Pipeline config carrying the layout knobs.

    Returns:
        A ready-to-use `LayoutAnalyzer` instance.
    """
    return make_analyzer(
        device=cfg.layout_device,
        fp16=cfg.layout_fp16,
        fast_preproc=cfg.layout_fast_preproc,
    )

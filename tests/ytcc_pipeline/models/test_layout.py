"""Tests for `ytcc_pipeline.models.layout`.

The SafeTensors backend itself is loaded only in `integration`-marked tests under
`tests/ytcc_pipeline/test_pipeline.py` -- its weights are too large to pull on every
unit run. Here we cover the small factory layer that wires `PipelineConfig` to the
backend constructor without loading anything.
"""

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from ytcc_pipeline.config import PipelineConfig
from ytcc_pipeline.models.layout import (
    LayoutAnalyzer,
    LayoutDetection,
    make_analyzer_from_config,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def test_make_analyzer_from_config_forwards_layout_knobs() -> None:
    """`make_analyzer_from_config` must pass the three layout knobs through.

    Catches a future refactor of `PipelineConfig` that renames one of the fields without
    updating the factory.
    """
    cfg = PipelineConfig(
        layout_device="cuda:1",
        layout_fp16=True,
        layout_fast_preproc=True,
    )

    with patch("ytcc_pipeline.models.layout.make_analyzer") as mocked:
        make_analyzer_from_config(cfg)

    mocked.assert_called_once_with(
        device="cuda:1",
        fp16=True,
        fast_preproc=True,
    )


def test_layout_detection_is_immutable_record() -> None:
    """`LayoutDetection` is a frozen-slots dataclass: order matters, mutation rejected.

    Verifies the cross-process IPC contract -- workers pickle these instances directly,
    so accidentally making it mutable (or changing field order) would silently break the
    spawn dispatch.
    """
    det = LayoutDetection(
        label="paragraph_title",
        label_id=1,
        confidence=0.9,
        bbox=(0.0, 0.0, 100.0, 50.0),
        reading_order=3,
    )
    assert det.label == "paragraph_title"
    assert det.reading_order == 3

    with pytest.raises((AttributeError, TypeError)):
        det.label = "other"  # pyright: ignore[reportAttributeAccessIssue]


def test_layout_analyzer_is_a_runtime_checkable_protocol() -> None:
    """A duck-typed object with `analyze` + `close` satisfies the Protocol.

    Lets test code substitute a stub analyzer in `process_pdf(..., analyzer=...)`
    without inheriting from anything.
    """

    class _StubAnalyzer:
        def analyze(
            self,
            page_images: Sequence[Path],
            *,
            batch_size: int = 8,
            confidence: float = 0.5,
        ) -> dict[int, list[LayoutDetection]]:
            del page_images, batch_size, confidence
            return {}

        def close(self) -> None:
            return None

    assert isinstance(_StubAnalyzer(), LayoutAnalyzer)

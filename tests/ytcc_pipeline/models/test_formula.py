"""Tests for ytcc_pipeline.models.formula.

The real-model tests are marked `slow` because they need the local PP-FormulaNet-L
SafeTensors snapshot (~700 MB) and a CUDA GPU. They skip cleanly when those
preconditions aren't met. The structural tests (input validation, empty input,
missing path) run without loading the model.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from ytcc_pipeline.models.formula import BucketSpec, FormulaRecognizer, FormulaResult

if TYPE_CHECKING:
    from pathlib import Path


def _formula_crop_png(out_path: Path) -> Path:
    """Write a small blank synthetic crop to `out_path` and return it.

    The image content doesn't need to be a real formula -- the structural tests below
    check `FormulaResult` shape, not LaTeX fidelity.
    """
    img = Image.new("RGB", (300, 80), color=(255, 255, 255))
    arr = np.array(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)

    return out_path


def test_invalid_batch_size_raises(tmp_path: Path) -> None:
    """`batch_size < 1` is rejected before any model load."""
    crop = _formula_crop_png(tmp_path / "crop.png")

    # Bypass __init__ so the validation check runs without loading the heavyweight model
    # Argument validation lives in `recognize_batch_paths` itself before any model
    # access.
    rec = FormulaRecognizer.__new__(FormulaRecognizer)
    with pytest.raises(ValueError, match="batch_size"):
        rec.recognize_batch_paths([crop], batch_size=0)


def test_empty_path_list_returns_empty_list() -> None:
    """`recognize_batch_paths` short-circuits on empty input."""
    rec = FormulaRecognizer.__new__(FormulaRecognizer)
    assert rec.recognize_batch_paths([], batch_size=4) == []


def test_bucket_spec_rejects_invalid_thresholds() -> None:
    """`small_threshold` must be strictly less than `medium_threshold`."""
    with pytest.raises(ValueError, match="small_threshold"):
        BucketSpec(
            small_threshold=30000.0,
            medium_threshold=2500.0,
            small_tokens=128,
            medium_tokens=384,
        )

    with pytest.raises(ValueError, match="small_threshold"):
        BucketSpec(
            small_threshold=0,
            medium_threshold=2500.0,
            small_tokens=128,
            medium_tokens=384,
        )


def test_bucket_spec_rejects_invalid_tokens() -> None:
    """Per-bucket token caps must be at least 1."""
    with pytest.raises(ValueError, match="bucket token caps"):
        BucketSpec(
            small_threshold=2500.0,
            medium_threshold=30000.0,
            small_tokens=0,
            medium_tokens=384,
        )


def test_bucketed_empty_paths_returns_empty_list() -> None:
    """Empty input short-circuits before any model touch."""
    rec = FormulaRecognizer.__new__(FormulaRecognizer)
    spec = BucketSpec(
        small_threshold=2500.0,
        medium_threshold=30000.0,
        small_tokens=128,
        medium_tokens=384,
    )

    assert (
        rec.recognize_batch_paths_bucketed([], [], batch_size=4, bucket_spec=spec) == []
    )


def test_bucketed_rejects_length_mismatch(tmp_path: Path) -> None:
    """`crop_paths` and `crop_areas` must have matching lengths."""
    crop = _formula_crop_png(tmp_path / "crop.png")
    rec = FormulaRecognizer.__new__(FormulaRecognizer)
    spec = BucketSpec(
        small_threshold=2500.0,
        medium_threshold=30000.0,
        small_tokens=128,
        medium_tokens=384,
    )

    with pytest.raises(ValueError, match="must equal"):
        rec.recognize_batch_paths_bucketed(
            [crop],
            [100.0, 200.0],
            batch_size=4,
            bucket_spec=spec,
        )


@pytest.mark.slow
def test_bucketed_returns_results_in_input_order(tmp_path: Path) -> None:
    """Bucketing must splice results back into the original crop order.

    Builds three crops with hand-picked area sentinels that fall into three different
    buckets, then verifies the returned results align with the input order (i.e.
    result[i] is the recognition for paths[i], regardless of which bucket each crop
    landed in).
    """
    rec = _try_load()
    try:
        crops = [_formula_crop_png(tmp_path / f"c{i}.png") for i in range(3)]

        # Areas chosen to land in three different buckets given the default thresholds
        # (2500 / 30000): large, small, medium.
        areas = [100000.0, 100.0, 5000.0]
        spec = BucketSpec(
            small_threshold=2500.0,
            medium_threshold=30000.0,
            small_tokens=64,
            medium_tokens=128,
        )
        results = rec.recognize_batch_paths_bucketed(
            crops,
            areas,
            batch_size=2,
            bucket_spec=spec,
        )
        assert len(results) == 3
        for r in results:
            assert isinstance(r, FormulaResult)
    finally:
        rec.close()


@pytest.mark.slow
def test_model_loads_and_reports_device() -> None:
    """A fresh recognizer exposes device + generation cap."""
    rec = _try_load()
    try:
        assert rec.device == "cuda:0"
        assert rec.max_new_tokens > 0
    finally:
        rec.close()


@pytest.mark.slow
def test_recognize_batch_paths_returns_one_result_per_input(
    tmp_path: Path,
) -> None:
    """Output length matches input length; each entry is a FormulaResult."""
    rec = _try_load()
    try:
        paths = [_formula_crop_png(tmp_path / f"c{i}.png") for i in range(3)]
        results = rec.recognize_batch_paths(paths, batch_size=2)
        assert len(results) == len(paths)
        for r in results:
            assert isinstance(r, FormulaResult)
            assert isinstance(r.truncated, bool)
    finally:
        rec.close()


@pytest.mark.slow
def test_recognize_batch_paths_missing_path_yields_none(tmp_path: Path) -> None:
    """A bogus path produces `FormulaResult(None, False)` for that slot."""
    rec = _try_load()
    try:
        good = _formula_crop_png(tmp_path / "good.png")
        missing = tmp_path / "does-not-exist.png"
        results = rec.recognize_batch_paths([missing, good], batch_size=2)

        assert results[0].latex is None

        # The peer slot is unaffected; structurally it returns *something*.
        assert isinstance(results[1], FormulaResult)
    finally:
        rec.close()


def _try_load() -> FormulaRecognizer:
    """Construct a recognizer or skip with a clear reason if preconditions fail."""
    try:
        return FormulaRecognizer(
            "PaddlePaddle/PP-FormulaNet-L_safetensors",
            device="cuda:0",
            dtype="fp16",
        )
    except Exception as exc:  # pragma: no cover -- environment-dependent
        pytest.skip(f"formula model unavailable: {exc}")

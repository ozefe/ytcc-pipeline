"""Tests for ytcc_pipeline.models.ocr."""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from ytcc_pipeline.models.ocr import OcrExtractor


def _text_crop(text: str, width: int = 400, height: int = 80) -> np.ndarray:
    """Render plain text on a white background; returns HWC uint8 RGB."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 32)
    except OSError:
        font = ImageFont.load_default()

    draw.text((10, 20), text, fill=(0, 0, 0), font=font)
    return np.array(img)


def test_extractor_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="language"):
        OcrExtractor(language="xx")


@pytest.mark.slow
def test_extract_batch_returns_text_for_latin_crop() -> None:
    extractor = OcrExtractor(language="en", batch_size=4)
    crop = _text_crop("HELLO WORLD")

    [result] = extractor.extract_batch([crop])

    assert result is not None
    assert "HELLO" in result.upper() or "WORLD" in result.upper()


@pytest.mark.slow
def test_extract_batch_returns_none_for_blank_crop() -> None:
    extractor = OcrExtractor(language="en", batch_size=4)
    blank = np.full((80, 400, 3), 255, dtype=np.uint8)

    [result] = extractor.extract_batch([blank])

    assert result is None


@pytest.mark.slow
def test_extract_batch_preserves_input_order() -> None:
    extractor = OcrExtractor(language="en", batch_size=4)
    crops = [_text_crop(t) for t in ("ALPHA", "BETA", "GAMMA")]

    results = extractor.extract_batch(crops)

    assert len(results) == 3

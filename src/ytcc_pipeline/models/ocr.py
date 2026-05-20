"""RapidOCR wrapper that maps ISO 639-1 language codes to script-family models."""

import logging
import time
from typing import TYPE_CHECKING, Any, cast

from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from ytcc_pipeline.config import PipelineConfig

__all__ = ["LANG_TO_RAPIDOCR_LANG", "OcrExtractor", "make_ocr_extractor"]

logger = logging.getLogger(__name__)

LANG_TO_RAPIDOCR_LANG: dict[str, LangRec] = {
    "tr": LangRec.LATIN,
    "en": LangRec.EN,
    "ar": LangRec.ARABIC,
    "de": LangRec.LATIN,
    "fr": LangRec.LATIN,
    "es": LangRec.LATIN,
    "it": LangRec.LATIN,
    "ru": LangRec.CYRILLIC,
    "pl": LangRec.LATIN,
    "zh": LangRec.CH,
    "ku": LangRec.LATIN,
    "az": LangRec.LATIN,
    "bg": LangRec.CYRILLIC,
    "cs": LangRec.LATIN,
    "ro": LangRec.LATIN,
    "nl": LangRec.LATIN,
    "ja": LangRec.JAPAN,
    "fa": LangRec.ARABIC,
    "el": LangRec.EL,
    "sl": LangRec.LATIN,
    "mk": LangRec.CYRILLIC,
    "ady": LangRec.CYRILLIC,
    "ky": LangRec.CYRILLIC,
    "bs": LangRec.LATIN,
    # No Georgian
    "ko": LangRec.KOREAN,
    # No Armenian
    "zza": LangRec.LATIN,
    "ms": LangRec.LATIN,
    "kk": LangRec.CYRILLIC,
    "uk": LangRec.CYRILLIC,
    "mn": LangRec.CYRILLIC,
    "id": LangRec.LATIN,
    "uz": LangRec.LATIN,
    "hu": LangRec.LATIN,
    "sr": LangRec.CYRILLIC,
    "pt": LangRec.LATIN,
    "sq": LangRec.LATIN,
    "lv": LangRec.LATIN,
    "no": LangRec.LATIN,
    "th": LangRec.TH,
    "hi": LangRec.DEVANAGARI,
}
"""ISO 639-1 language code -> RapidOCR `LangRec` enum (script-family granular)."""


class OcrExtractor:
    """OCR pre-cropped block images using RapidOCR.

    The pipeline parallelises OCR by spawning one extractor per worker process
    (`config.ocr_workers`); a single extractor is therefore always serial.
    """

    def __init__(
        self,
        language: str,
        *,
        batch_size: int = 64,
        min_score: float = 0.5,
        use_cuda: bool = False,
    ) -> None:
        """Initialise the RapidOCR engine for `language`.

        Args:
            language: ISO 639-1 code; must be a key of `LANG_TO_RAPIDOCR_LANG`.
            batch_size: Recognition batch size (`Rec.batch_size`). RapidOCR's docs note
                that the default 6 is usually optimal; we leave the project default at
                64 because empty crops short-circuit early.
            min_score: Per-crop minimum mean recognition score. Crops below this return
                `None` (treated as a MISS by the pipeline).
            use_cuda: Enable the ONNX `CUDAExecutionProvider`. Requires
                `onnxruntime-gpu`.

        Raises:
            ValueError: `language` is not in `LANG_TO_RAPIDOCR_LANG`.
        """
        if language not in LANG_TO_RAPIDOCR_LANG:
            logger.error(
                "OCR unsupported language: requested=%s known=%s",
                language,
                sorted(LANG_TO_RAPIDOCR_LANG),
            )
            msg = (
                f"unsupported language {language!r}; "
                "add it to LANG_TO_RAPIDOCR_LANG or pick from "
                f"{sorted(LANG_TO_RAPIDOCR_LANG)}"
            )
            raise ValueError(msg)

        lang = LANG_TO_RAPIDOCR_LANG[language]
        self._min_score = min_score
        params: dict[str, Any] = {
            "Det.use": True,
            "Cls.use": False,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.precision": "fp16",
            "Rec.lang_type": lang,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.batch_size": batch_size,
        }

        if use_cuda:
            params["EngineConfig.onnxruntime.use_cuda"] = True

        t0 = time.perf_counter()
        try:
            self._engine = RapidOCR(params=params)
        except Exception:
            logger.exception(
                "OCR load failed: language=%s lang_rec=%s use_cuda=%s",
                language,
                lang.name,
                use_cuda,
            )
            raise

        # RapidOCR composes Det / Cls / Rec session wrappers; the leaf `.session` is the
        # ONNXRuntime InferenceSession. The wrapper types are dynamic so cast to dodge
        # the static union over backend kinds.
        providers = cast("Any", self._engine.text_rec.session).session.get_providers()
        logger.info(
            "OCR initialised: language=%s lang_rec=%s providers=%s load_elapsed_s=%.2f",
            language,
            lang.name,
            providers,
            time.perf_counter() - t0,
        )

    def extract_batch(self, crops: Sequence[np.ndarray]) -> list[str | None]:
        """OCR each crop in order and return its concatenated text (or `None`).

        Args:
            crops: List of pre-cropped block images (HWC uint8 RGB).

        Returns:
            One entry per input crop. `None` when OCR fails, returns empty, or the mean
            confidence is below `min_score`.
        """
        return [self._extract_one(c) for c in crops]

    def _extract_one(self, crop: np.ndarray) -> str | None:
        try:
            # RapidOCR's `__call__` is annotated as a union of Det/Cls/Rec/ full
            # pipeline outputs; with default mode (det + rec, no cls) the actual return
            # is `RapidOCROutput` carrying `.txts` and `.scores`. Treat the result as
            # that shape at this boundary and let `getattr` defend if the upstream
            # signature drifts.
            result = cast("Any", self._engine(crop))
        except Exception:  # noqa: BLE001
            # Recoverable: caller treats `None` as a MISS.
            #
            # `exc_info=True` keeps the underlying ONNX / RapidOCR error diagnosable.
            logger.warning(
                "OCR call failed on crop of shape %r",
                crop.shape,
                exc_info=True,
            )
            return None

        if not getattr(result, "txts", None):
            logger.debug("OCR empty: shape=%s reason=no_txts", crop.shape)
            return None

        scores: list[float] = list(getattr(result, "scores", None) or ())
        if scores:
            mean_score = sum(scores) / len(scores)
            if mean_score < self._min_score:
                logger.debug(
                    "OCR rejected: shape=%s mean_score=%.3f min_score=%.3f",
                    crop.shape,
                    mean_score,
                    self._min_score,
                )
                return None

        joined = " ".join(t for t in result.txts if t)
        return joined.strip() or None


def make_ocr_extractor(cfg: PipelineConfig, language: str) -> OcrExtractor:
    """Construct an `OcrExtractor` from a `PipelineConfig`.

    Centralises the `OcrExtractor(language=..., batch_size=..., min_score=...,
    use_cuda=...)` boilerplate so the orchestrator and the per-page dispatcher stay in
    sync.

    Args:
        cfg: Pipeline config carrying the OCR knobs.
        language: ISO 639-1 code for the extractor.

    Returns:
        A ready-to-use `OcrExtractor` instance.

    Raises:
        ValueError: `language` is not in `LANG_TO_RAPIDOCR_LANG`.
    """
    return OcrExtractor(
        language=language,
        batch_size=cfg.ocr_batch_size,
        min_score=cfg.ocr_min_score,
        use_cuda=cfg.ocr_use_cuda,
    )

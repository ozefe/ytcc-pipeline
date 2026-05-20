"""ML model wrappers -- one module per external library.

Every module in this subpackage loads exactly one model (or model family) and exposes a
small inference API plus a `close()` method to release VRAM:

- `FormulaRecognizer` -- PP-FormulaNet-L via HF transformers.
- `OcrExtractor` -- RapidOCR over ONNXRuntime.
- `LayoutAnalyzer` (Protocol) + the SafeTensors backend (`SafeTensorsLayoutAnalyzer`) --
  PP-DocLayoutV3 via HF transformers, constructed by the `make_analyzer` factory.

The "model wrapper" boundary is honest: each module owns one external library's CUDA
context, model load, and inference call shape. Grouping them surfaces the shared
VRAM-accounting concerns and the shared load/infer/close lifecycle.
"""

from .formula import BucketSpec, FormulaRecognizer, FormulaResult
from .layout import LayoutAnalyzer, LayoutDetection, make_analyzer
from .ocr import LANG_TO_RAPIDOCR_LANG, OcrExtractor

__all__ = [
    "LANG_TO_RAPIDOCR_LANG",
    "BucketSpec",
    "FormulaRecognizer",
    "FormulaResult",
    "LayoutAnalyzer",
    "LayoutDetection",
    "OcrExtractor",
    "make_analyzer",
]

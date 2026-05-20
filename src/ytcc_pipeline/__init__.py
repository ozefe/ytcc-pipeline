"""PDF processing pipeline for academic theses."""

import os as _os

# Hint PyTorch's CUDA caching allocator to use expandable segments before any downstream
# torch import resolves. With many sequential scanned requests in the same uvicorn
# worker the default allocator leaves enough fragmentation that the 3rd+ scanned PDF's
# OCR engines can fail to allocate inside their spawn-worker forward passes.
# `setdefault` preserves an operator's explicit override.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"

from .config import (
    ApiSettings,
    LoggingSettings,
    PipelineConfig,
    ServiceConfig,
    load_service_config,
)
from .models.formula import FormulaRecognizer
from .pipeline import process_pdf

__all__ = [
    "ApiSettings",
    "FormulaRecognizer",
    "LoggingSettings",
    "PipelineConfig",
    "ServiceConfig",
    "__version__",
    "load_service_config",
    "process_pdf",
]

"""Process multiple PDFs in a row with one `PipelineConfig`.

The analyzer and worker pool are re-created per call inside `process_pdf`, so the first
PDF pays the ~5s model-load cost; subsequent PDFs in the same process get a hot
HuggingFace cache and warm CUDA driver state.

Run from the project root, passing one or more PDF paths:

    python examples/batch.py path/to/a.pdf path/to/b.pdf [...]

Every PDF is processed with the recommended performance config. Bundles are written next
to the inputs as `<stem>.tar`. Edit `LANGUAGE` below to OCR scanned PDFs in a different
language, or call `process_pdf` directly when each input needs its own language code.
"""

import sys
import time
from pathlib import Path

from ytcc_pipeline import PipelineConfig, process_pdf

LANGUAGE = "en"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python examples/batch.py <pdf_path> [<pdf_path> ...]")

    config = PipelineConfig(
        layout_fp16=True,
        layout_fast_preproc=True,
        ocr_use_cuda=True,
        digital_born_workers=16,
        ocr_workers=6,
        page_format="jpeg",
        jpeg_quality=90,
    )

    for arg in sys.argv[1:]:
        pdf_path = Path(arg).resolve()
        if not pdf_path.is_file():
            print(f"{pdf_path}: not found, skipping")
            continue

        t0 = time.perf_counter()
        bundle = process_pdf(pdf_path, language=LANGUAGE, config=config)
        elapsed = time.perf_counter() - t0

        print(f"{pdf_path.name}: {elapsed:.1f}s -> {bundle.name}")


if __name__ == "__main__":
    main()

"""Production-tuned `process_pdf` run.

Enables every shipped performance flag. End-to-end on an RTX 3090 lands at ~15 s on a
150-page English digital-born thesis (vs ~85s with `PipelineConfig()`). The auto-DPI
lever (150 DPI for digital-born) is on by default and contributes most of the win.

Run from the project root, passing a PDF path:

    python examples/super_fast.py path/to/paper.pdf [language]
"""

import sys
import time
from pathlib import Path

from ytcc_pipeline import PipelineConfig, process_pdf


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python examples/super_fast.py <pdf_path> [language]")

    pdf_path = Path(sys.argv[1]).resolve()
    language = sys.argv[2] if len(sys.argv) > 2 else "en"

    config = PipelineConfig(
        layout_fp16=True,
        layout_fast_preproc=True,
        ocr_use_cuda=True,
        digital_born_workers=16,
        ocr_workers=6,
        page_format="jpeg",
        jpeg_quality=90,
    )

    t0 = time.perf_counter()
    bundle = process_pdf(pdf_path, language=language, config=config)
    elapsed = time.perf_counter() - t0
    print(f"wrote {bundle} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

"""Minimal usage of `process_pdf`.

Run from the project root, passing a PDF path:

    python examples/basic.py path/to/paper.pdf [language]

`language` defaults to `"en"`; pass any ISO 639-1 code supported by
`ytcc_pipeline.models.ocr.LANG_TO_RAPIDOCR_LANG` (only matters for scanned PDFs).

The bundle is written next to the input PDF as `<stem>.tar`.
"""

import sys
from pathlib import Path

from ytcc_pipeline import process_pdf


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python examples/basic.py <pdf_path> [language]")

    pdf_path = Path(sys.argv[1]).resolve()
    language = sys.argv[2] if len(sys.argv) > 2 else "en"

    bundle = process_pdf(pdf_path, language=language)
    print(f"wrote {bundle}")


if __name__ == "__main__":
    main()

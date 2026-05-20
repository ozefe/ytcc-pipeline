"""Extract PDF intrinsic metadata + provenance fields for the output JSON."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdf_oxide

__all__ = ["PdfMetadata", "extract_metadata"]

logger = logging.getLogger(__name__)

_SHA256_CHUNK_BYTES = 64 * 1024


@dataclass(slots=True, frozen=True)
class PdfMetadata:
    """Provenance + intrinsic metadata for a single source PDF."""

    filename: str
    sha256: str
    byte_size: int
    pdf_info: dict[str, str]


def extract_metadata(pdf_path: Path) -> PdfMetadata:
    """Compute provenance fields and read XMP metadata from a PDF.

    XMP is the only metadata source `pdf_oxide` exposes today; PDFs without an XMP block
    (most older theses) get an empty `pdf_info` dict.

    Args:
        pdf_path: Source PDF file.

    Returns:
        `PdfMetadata` with filename, sha256, byte_size, and any XMP fields (cleaned of
        UTF-16 BOMs and null bytes).

    Raises:
        FileNotFoundError: `pdf_path` does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        msg = f"PDF not found: {pdf_path}"
        raise FileNotFoundError(msg)

    pdf_info: dict[str, str] = {}
    try:
        xmp = pdf_oxide.PdfDocument(str(pdf_path)).xmp_metadata()
    except Exception:  # noqa: BLE001
        # XMP absence / parse failure is a recoverable degradation -- the output bundle
        # still ships with an empty `pdf_info` dict. the stack trace via `exc_info`
        # keeps it diagnosable.
        logger.warning("xmp_metadata read failed for %s", pdf_path, exc_info=True)
        xmp = None
    if isinstance(xmp, dict):
        for key, value in xmp.items():
            if cleaned := _clean_xmp_value(value):
                pdf_info[str(key)] = cleaned

    return PdfMetadata(
        filename=pdf_path.name,
        sha256=_sha256_file(pdf_path),
        byte_size=pdf_path.stat().st_size,
        pdf_info=pdf_info,
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_SHA256_CHUNK_BYTES):
            h.update(chunk)

    return h.hexdigest()


# UTF-16 BOM sentinels in their latin-1-decoded form (pdf_oxide returns the raw bytes
# wrapped in a Python str, so the BOM bytes \xFE\xFF show up as the characters 'þÿ' and
# \xFF\xFE as 'ÿþ').
_UTF16_BOMS: dict[str, str] = {"þÿ": "utf-16-be", "ÿþ": "utf-16-le"}


def _clean_xmp_value(value: Any) -> str:  # noqa: ANN401 -- pdf_oxide returns mixed-type XMP values (str / list / int) with no static schema
    """Normalise XMP values into printable UTF-8 strings.

    `pdf_oxide` returns XMP values as-is, so UTF-16 strings arrive as latin-1-decoded
    Python strings carrying their byte-order mark. This unwraps those back to UTF-16,
    joins lists with `"; "`, and strips nulls and a stray `﻿` BOM if present.
    """
    if isinstance(value, list):
        return "; ".join(filter(None, (_clean_xmp_value(v) for v in value)))

    if not isinstance(value, str):
        return str(value)

    for prefix, encoding in _UTF16_BOMS.items():
        if value.startswith(prefix):
            try:
                decoded = value.encode("latin-1")[2:].decode(encoding, errors="replace")
                return decoded.rstrip("\x00").replace("\x00", "")
            except UnicodeError:
                # Falling through to the BOM-strip path: the value claimed a UTF-16 BOM
                # but didn't survive latin-1 -> utf-16 round-trip.
                #
                # Surface at DEBUG so XMP edge cases are diagnosable.
                logger.debug(
                    "XMP value claimed %s BOM but failed to decode; "
                    "falling back to raw-string handling",
                    encoding,
                )
                break

    return value.lstrip("﻿").replace("\x00", "")

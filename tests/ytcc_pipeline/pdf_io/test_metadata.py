"""Tests for ytcc_pipeline.pdf_io.metadata."""

import hashlib
from typing import TYPE_CHECKING

import pytest

from ytcc_pipeline.pdf_io.metadata import (
    PdfMetadata,
    _clean_xmp_value,
    extract_metadata,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_extract_metadata_filename_and_size(tiny_pdf: Path) -> None:
    meta = extract_metadata(tiny_pdf)

    assert isinstance(meta, PdfMetadata)
    assert meta.filename == "tiny.pdf"
    assert meta.byte_size == tiny_pdf.stat().st_size
    assert meta.byte_size > 0


def test_extract_metadata_sha256_matches_hashlib(tiny_pdf: Path) -> None:
    meta = extract_metadata(tiny_pdf)
    expected = hashlib.sha256(tiny_pdf.read_bytes()).hexdigest()

    assert meta.sha256 == expected
    assert len(meta.sha256) == 64


def test_extract_metadata_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_metadata(tmp_path / "does-not-exist.pdf")


class TestCleanXmpValue:
    def test_passes_plain_ascii_through(self) -> None:
        assert _clean_xmp_value("Microsoft Word") == "Microsoft Word"

    def test_passes_plain_unicode_through(self) -> None:
        assert _clean_xmp_value("Microsoft® Word için") == "Microsoft® Word için"

    def test_decodes_utf16_be_prefixed(self) -> None:
        raw = "þÿ\x00H\x00i"
        assert _clean_xmp_value(raw) == "Hi"

    def test_decodes_utf16_le_prefixed(self) -> None:
        raw = "ÿþH\x00i\x00"
        assert _clean_xmp_value(raw) == "Hi"

    def test_joins_list_values(self) -> None:
        assert _clean_xmp_value(["a", "b", "c"]) == "a; b; c"

    def test_strips_null_bytes_from_other(self) -> None:
        assert _clean_xmp_value("Hello\x00World") == "HelloWorld"

    def test_strips_bom_when_no_encoding_marker(self) -> None:
        assert _clean_xmp_value("﻿Plain") == "Plain"

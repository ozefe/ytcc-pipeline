"""Tests for ytcc_pipeline.bundle."""

import json
import tarfile
from typing import TYPE_CHECKING

import pytest

from ytcc_pipeline.bundle import create_bundle
from ytcc_pipeline.pdf_io.metadata import PdfMetadata
from ytcc_pipeline.schema import Block, BlockType, Document, Page

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_doc() -> Document:
    meta = PdfMetadata(
        filename="thesis.pdf", sha256="a" * 64, byte_size=10, pdf_info={}
    )
    page = Page(
        page_no=1,
        width_px=2480,
        height_px=3508,
        blocks=[
            Block(
                reading_order=0,
                label="text",
                type=BlockType.TEXT,
                bbox=(0, 0, 100, 100),
                confidence=0.9,
                text="hello",
            ),
            Block(
                reading_order=1,
                label="table",
                type=BlockType.IMAGE,
                bbox=(0, 100, 100, 200),
                confidence=0.85,
                image_path="images/0001-table-aaaa.png",
            ),
        ],
    )

    return Document(
        metadata=meta,
        language="en",
        digital_born=True,
        pipeline_version="0.1.0",
        pages=[page],
    )


def _read_json(tar_path: Path, member: str) -> bytes:
    with tarfile.open(tar_path) as tf:
        fp = tf.extractfile(member)
        assert fp is not None
        return fp.read()


def _names(tar_path: Path) -> list[str]:
    with tarfile.open(tar_path) as tf:
        return tf.getnames()


def test_create_bundle_writes_parseable_document_and_images(
    sample_doc: Document,
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "0001-table-aaaa.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    output = tmp_path / "out.tar"

    result = create_bundle(sample_doc, image_dir, output)

    assert result == output
    assert set(_names(output)) == {"document.json", "images/0001-table-aaaa.png"}

    payload = json.loads(_read_json(output, "document.json"))
    assert payload["language"] == "en"
    assert payload["digital_born"] is True
    assert len(payload["pages"]) == 1


def test_bundle_includes_all_images_from_dir(
    tmp_path: Path, sample_doc: Document
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("0001-table-aaaa.png", "0001-image-bbbb.png", "0002-chart-cccc.png"):
        (image_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n")

    output = tmp_path / "out.tar"

    create_bundle(sample_doc, image_dir, output)

    image_entries = [n for n in _names(output) if n.startswith("images/")]
    assert set(image_entries) == {
        "images/0001-table-aaaa.png",
        "images/0001-image-bbbb.png",
        "images/0002-chart-cccc.png",
    }


def test_bundle_empty_image_dir_yields_only_document_json(
    sample_doc: Document,
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output = tmp_path / "out.tar"

    create_bundle(sample_doc, image_dir, output)

    assert _names(output) == ["document.json"]


def test_bundle_writes_document_json_first_for_streaming(
    sample_doc: Document,
    tmp_path: Path,
) -> None:
    """`document.json` is the first member so streaming consumers can parse the index
    before buffering crops.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "0001-table-aaaa.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    output = tmp_path / "out.tar"

    create_bundle(sample_doc, image_dir, output)

    assert _names(output)[0] == "document.json"


def test_bundle_is_uncompressed_posix_tar(
    sample_doc: Document,
    tmp_path: Path,
) -> None:
    """No outer gzip / bzip2 wrapper; the file opens as a plain tar."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "0001-table-aaaa.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    output = tmp_path / "out.tar"

    create_bundle(sample_doc, image_dir, output)

    # tarfile.is_tarfile rejects compressed wrappers when called with the default
    # `name=` parameter; force-opening with mode="r:" (no codec) confirms the bytes are
    # a bare uncompressed tar.
    with tarfile.open(output, mode="r:") as tf:
        members = tf.getmembers()

    assert any(m.name == "document.json" for m in members)

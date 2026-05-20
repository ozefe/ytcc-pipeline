"""JSON schema dataclasses for the pipeline's output document."""

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .block_type import BlockType

if TYPE_CHECKING:
    from .pdf_io.metadata import PdfMetadata

__all__ = [
    "Author",
    "Block",
    "BlockType",
    "Cell",
    "Document",
    "Page",
    "Reference",
    "to_json",
]


@dataclass(slots=True, frozen=True)
class Cell:
    """A single cell inside a TABLE block.

    Coordinates are page-space pixel coords at the page's render DPI -- same space as
    the parent table's `bbox` -- so consumers can rerender cells against the original
    page image. `text` is `None` when extraction returned nothing (empty cell or failed
    OCR).
    """

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    bbox: tuple[float, float, float, float]
    text: str | None = None


@dataclass(slots=True, frozen=True)
class Author:
    """A single contributor parsed out of a reference string.

    `name` is always set (display form, e.g. `"J Smith"`). `surname` / `forename` are
    populated when GROBID resolved them separately; either may be `None` for
    initials-only or surname-only entries.
    """

    name: str
    surname: str | None = None
    forename: str | None = None


@dataclass(slots=True, frozen=True)
class Reference:
    """Structured parse of a bibliographic reference string.

    Populated by the reference stage when `cfg.references_enabled`. Every field is
    optional -- GROBID routinely returns partial parses and pretending otherwise loses
    information. The original raw text survives on `Block.text` regardless.
    """

    title: str | None = None
    authors: tuple[Author, ...] = ()
    year: str | None = None
    venue: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    publisher: str | None = None
    doi: str | None = None
    url: str | None = None
    pmid: str | None = None
    arxiv: str | None = None


@dataclass(slots=True, frozen=True)
class Block:
    """A single layout block with its extracted content."""

    reading_order: int
    label: str
    type: BlockType
    bbox: tuple[float, float, float, float]
    confidence: float
    text: str | None = None
    image_path: str | None = None
    miss: bool = False

    # TABLE-only fields. `None` on non-table blocks and on TABLE blocks where structure
    # recognition failed (image-only fallback -- `image_path` still points at the saved
    # table crop).
    n_rows: int | None = None
    n_cols: int | None = None
    cells: tuple[Cell, ...] | None = None

    # REFERENCE-only field. `None` on non-reference blocks, on reference blocks when the
    # reference stage is disabled, and on reference blocks GROBID couldn't parse into
    # any usable fields. The raw reference string remains in `text` either way.
    reference: Reference | None = None


@dataclass(slots=True, frozen=True)
class Page:
    """A single page with all its blocks in per-page reading order."""

    page_no: int
    width_px: int
    height_px: int
    blocks: list[Block] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class Document:
    """Top-level output document."""

    metadata: PdfMetadata
    language: str
    digital_born: bool
    pipeline_version: str
    pages: list[Page] = field(default_factory=list)


def to_json(doc: Document) -> str:
    """Serialize a `Document` to indented UTF-8 JSON.

    bbox floats are rounded to 2 decimals to keep the output compact and readable; the
    raw pixel-precision is rarely useful downstream. Frozen-dataclass tuples are
    converted to JSON arrays.

    Args:
        doc: The document to serialize.

    Returns:
        Indented UTF-8 JSON with bbox floats rounded to 2 decimals.
    """
    payload = _normalize_for_json(asdict(doc))
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _normalize_for_json(
    value: Any,  # noqa: ANN401  -- recursive walk over `asdict`'s arbitrary JSON-shaped output
) -> Any:  # noqa: ANN401  -- same; preserved by recursion
    """Recursively round bbox floats and normalize tuples to lists."""
    if isinstance(value, dict):
        if "bbox" in value and isinstance(value["bbox"], list | tuple):
            value = {**value, "bbox": [round(float(c), 2) for c in value["bbox"]]}

        return {k: _normalize_for_json(v) for k, v in value.items()}

    if isinstance(value, list | tuple):
        return [_normalize_for_json(v) for v in value]

    return value

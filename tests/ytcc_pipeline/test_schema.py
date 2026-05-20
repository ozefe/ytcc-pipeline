"""Tests for ytcc_pipeline.schema."""

import json

import pytest

from ytcc_pipeline.pdf_io.metadata import PdfMetadata
from ytcc_pipeline.schema import (
    Author,
    Block,
    BlockType,
    Cell,
    Document,
    Page,
    Reference,
    to_json,
)


@pytest.fixture
def sample_metadata() -> PdfMetadata:
    return PdfMetadata(
        filename="thesis.pdf",
        sha256="a" * 64,
        byte_size=1234,
        pdf_info={"dc_creator": "Author"},
    )


@pytest.fixture
def sample_document(sample_metadata: PdfMetadata) -> Document:
    text_block = Block(
        reading_order=0,
        label="paragraph_title",
        type=BlockType.TEXT,
        bbox=(10.5, 20.999, 100.123, 30.001),
        confidence=0.95,
        text="Introduction",
    )
    image_block = Block(
        reading_order=1,
        label="figure",
        type=BlockType.IMAGE,
        bbox=(20.0, 40.0, 200.0, 300.0),
        confidence=0.92,
        image_path="images/0001-figure-abc.png",
    )
    ref_block = Block(
        reading_order=2,
        label="reference_content",
        type=BlockType.REFERENCE,
        bbox=(20.0, 800.0, 500.0, 900.0),
        confidence=0.88,
        text="[1] Smith, J. (2020).",
    )
    page = Page(
        page_no=1,
        width_px=2480,
        height_px=3508,
        blocks=[text_block, image_block, ref_block],
    )

    return Document(
        metadata=sample_metadata,
        language="en",
        digital_born=True,
        pipeline_version="0.1.0",
        pages=[page],
    )


class TestToJson:
    def test_pages_are_grouped_under_pages_key(self, sample_document: Document) -> None:
        parsed = json.loads(to_json(sample_document))
        assert isinstance(parsed["pages"], list)
        assert len(parsed["pages"]) == 1
        assert parsed["pages"][0]["page_no"] == 1
        assert isinstance(parsed["pages"][0]["blocks"], list)

    def test_block_type_serialized_as_lowercase_string(
        self, sample_document: Document
    ) -> None:
        parsed = json.loads(to_json(sample_document))
        types = [b["type"] for b in parsed["pages"][0]["blocks"]]
        assert types == ["text", "image", "reference"]

    def test_bbox_rounded_to_two_decimals(self, sample_document: Document) -> None:
        parsed = json.loads(to_json(sample_document))
        bbox = parsed["pages"][0]["blocks"][0]["bbox"]
        assert bbox == [10.5, 21.0, 100.12, 30.0]

    def test_metadata_pdf_info_is_dict(self, sample_document: Document) -> None:
        parsed = json.loads(to_json(sample_document))
        assert parsed["metadata"]["filename"] == "thesis.pdf"
        assert parsed["metadata"]["pdf_info"] == {"dc_creator": "Author"}

    def test_text_block_has_text_and_no_image_path(
        self, sample_document: Document
    ) -> None:
        parsed = json.loads(to_json(sample_document))
        text_block = parsed["pages"][0]["blocks"][0]
        assert text_block["text"] == "Introduction"
        assert text_block["image_path"] is None
        assert text_block["miss"] is False

    def test_formula_block_serializes_text_and_image_path_together(
        self,
        sample_metadata: PdfMetadata,
    ) -> None:
        """The schema permits a FORMULA block to carry both `text` and `image_path`.

        The post-pipeline state never produces this combination (success clears
        `image_path`; MISS clears `text`), but the dataclass shape permits it and
        downstream consumers that hand-construct blocks should round-trip cleanly.
        """
        formula = Block(
            reading_order=0,
            label="formula",
            type=BlockType.FORMULA,
            bbox=(20.0, 40.0, 200.0, 80.0),
            confidence=0.94,
            text="\\mathbf{V}_{\\mathrm{part}}",
            image_path="images/0014-formula-abc.png",
        )
        page = Page(page_no=1, width_px=2480, height_px=3508, blocks=[formula])
        doc = Document(
            metadata=sample_metadata,
            language="en",
            digital_born=True,
            pipeline_version="0.2.0",
            pages=[page],
        )
        parsed = json.loads(to_json(doc))

        block_json = parsed["pages"][0]["blocks"][0]
        assert block_json["type"] == "formula"
        assert block_json["text"] == "\\mathbf{V}_{\\mathrm{part}}"
        assert block_json["image_path"] == "images/0014-formula-abc.png"
        assert block_json["miss"] is False

    def test_non_table_block_omits_cell_fields_as_null(
        self,
        sample_document: Document,
    ) -> None:
        """Cells / n_rows / n_cols default to None for non-table blocks."""
        parsed = json.loads(to_json(sample_document))
        for block_json in parsed["pages"][0]["blocks"]:
            assert block_json["cells"] is None
            assert block_json["n_rows"] is None
            assert block_json["n_cols"] is None

    def test_table_block_serializes_cells(self, sample_metadata: PdfMetadata) -> None:
        """A TABLE block carries its cell grid in JSON."""
        cells = (
            Cell(
                row_start=0,
                row_end=0,
                col_start=0,
                col_end=0,
                bbox=(10.0, 20.0, 110.0, 60.0),
                text="Header",
            ),
            Cell(
                row_start=1,
                row_end=1,
                col_start=0,
                col_end=0,
                bbox=(10.001, 60.0, 110.0, 100.999),
                text="42",
            ),
        )
        table = Block(
            reading_order=0,
            label="table",
            type=BlockType.TABLE,
            bbox=(10.0, 20.0, 210.0, 220.0),
            confidence=0.93,
            image_path="images/0014-table-parent.png",
            n_rows=2,
            n_cols=1,
            cells=cells,
        )
        page = Page(page_no=1, width_px=2480, height_px=3508, blocks=[table])
        doc = Document(
            metadata=sample_metadata,
            language="en",
            digital_born=True,
            pipeline_version="0.3.0",
            pages=[page],
        )
        parsed = json.loads(to_json(doc))
        block_json = parsed["pages"][0]["blocks"][0]
        assert block_json["type"] == "table"
        assert block_json["n_rows"] == 2
        assert block_json["n_cols"] == 1
        cell_jsons = block_json["cells"]
        assert len(cell_jsons) == 2
        assert cell_jsons[0]["text"] == "Header"
        assert cell_jsons[1]["text"] == "42"
        # Cell bboxes are rounded to 2 decimals just like block bboxes.
        assert cell_jsons[1]["bbox"] == [10.0, 60.0, 110.0, 101.0]

    def test_table_fallback_with_no_cells_keeps_crop(
        self,
        sample_metadata: PdfMetadata,
    ) -> None:
        """When RapidTable can't parse a table, cells stays None and the crop
        survives."""
        fallback = Block(
            reading_order=0,
            label="table",
            type=BlockType.TABLE,
            bbox=(0, 0, 100, 100),
            confidence=0.91,
            image_path="images/0014-table-fallback.png",
        )
        page = Page(page_no=1, width_px=2480, height_px=3508, blocks=[fallback])
        doc = Document(
            metadata=sample_metadata,
            language="en",
            digital_born=True,
            pipeline_version="0.3.0",
            pages=[page],
        )
        parsed = json.loads(to_json(doc))

        block_json = parsed["pages"][0]["blocks"][0]
        assert block_json["cells"] is None
        assert block_json["n_rows"] is None
        assert block_json["n_cols"] is None
        assert block_json["image_path"] == "images/0014-table-fallback.png"

    def test_block_reference_field_defaults_to_none(
        self,
        sample_document: Document,
    ) -> None:
        """Blocks not enriched by the reference stage serialise `reference: null`."""
        parsed = json.loads(to_json(sample_document))
        for block_json in parsed["pages"][0]["blocks"]:
            assert block_json["reference"] is None

    def test_reference_block_serialises_nested_reference_object(
        self,
        sample_metadata: PdfMetadata,
    ) -> None:
        """A REFERENCE block carrying a populated `Reference` round-trips through
        JSON."""
        reference = Reference(
            title="Chemical Machining for Stainless Steel",
            authors=(
                Author(name="G Abd", surname="Abd", forename="G"),
                Author(name="E Enab", surname="Enab", forename="E"),
            ),
            year="2016",
            venue="Arab Journal of Nuclear Science and Applications",
            volume="49",
            issue="2",
            pages="132-139",
            doi="10.1234/abc",
            url="https://doi.org/10.1234/abc",
        )
        block = Block(
            reading_order=0,
            label="reference_content",
            type=BlockType.REFERENCE,
            bbox=(20.0, 800.0, 500.0, 900.0),
            confidence=0.88,
            text="Abd, G., ... 49(2), 132-139.",
            reference=reference,
        )
        page = Page(page_no=1, width_px=2480, height_px=3508, blocks=[block])
        doc = Document(
            metadata=sample_metadata,
            language="en",
            digital_born=True,
            pipeline_version="0.3.0",
            pages=[page],
        )

        block_json = json.loads(to_json(doc))["pages"][0]["blocks"][0]

        ref_json = block_json["reference"]
        assert ref_json["title"] == "Chemical Machining for Stainless Steel"
        assert ref_json["year"] == "2016"
        assert ref_json["venue"] == "Arab Journal of Nuclear Science and Applications"
        assert ref_json["volume"] == "49"
        assert ref_json["issue"] == "2"
        assert ref_json["pages"] == "132-139"
        assert ref_json["doi"] == "10.1234/abc"
        assert ref_json["url"] == "https://doi.org/10.1234/abc"

        # Authors come through as a list of dicts (frozen-tuple -> JSON array).
        assert ref_json["authors"] == [
            {"name": "G Abd", "surname": "Abd", "forename": "G"},
            {"name": "E Enab", "surname": "Enab", "forename": "E"},
        ]

    def test_reference_with_no_authors_serialises_as_empty_list(
        self,
        sample_metadata: PdfMetadata,
    ) -> None:
        """An author-less `Reference` keeps `authors` as an empty list, not null."""
        block = Block(
            reading_order=0,
            label="reference_content",
            type=BlockType.REFERENCE,
            bbox=(20.0, 800.0, 500.0, 900.0),
            confidence=0.88,
            text="Anonymous (1999). Title.",
            reference=Reference(title="Title", year="1999"),
        )
        page = Page(page_no=1, width_px=2480, height_px=3508, blocks=[block])
        doc = Document(
            metadata=sample_metadata,
            language="en",
            digital_born=True,
            pipeline_version="0.3.0",
            pages=[page],
        )

        ref_json = json.loads(to_json(doc))["pages"][0]["blocks"][0]["reference"]
        assert ref_json["authors"] == []
        assert ref_json["title"] == "Title"
        assert ref_json["year"] == "1999"
        assert ref_json["venue"] is None
        assert ref_json["doi"] is None

    def test_unicode_text_preserved(self, sample_metadata: PdfMetadata) -> None:
        block = Block(
            reading_order=0,
            label="text",
            type=BlockType.TEXT,
            bbox=(0, 0, 100, 100),
            confidence=0.9,
            text="çığşöü العربية",
        )
        page = Page(page_no=1, width_px=2480, height_px=3508, blocks=[block])
        doc = Document(
            metadata=sample_metadata,
            language="tr",
            digital_born=True,
            pipeline_version="0.1.0",
            pages=[page],
        )

        out = to_json(doc)
        parsed = json.loads(out)
        assert parsed["pages"][0]["blocks"][0]["text"] == "çığşöü العربية"

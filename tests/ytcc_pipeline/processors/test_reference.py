"""Unit tests for `processors.reference`.

All scenarios use a fake `GrobidClient` injected through the stage's `client=` kwarg --
no HTTP traffic, no GROBID dependency. An opt-in integration test against a real server
at the configured URL lives in the same module, gated on the `integration` marker.
"""

import logging
from typing import TYPE_CHECKING

import pytest

from ytcc_pipeline.config import PipelineConfig
from ytcc_pipeline.models.grobid import GrobidError
from ytcc_pipeline.processors.reference import run_reference_stage
from ytcc_pipeline.schema import (
    Author,
    Block,
    BlockType,
    Page,
    Reference,
)

if TYPE_CHECKING:
    from pathlib import Path


class _FakeClient:
    """Records calls; returns a caller-supplied list aligned to input."""

    def __init__(self, responses: list[Reference | None]) -> None:
        self._responses = responses
        self.calls: list[list[str]] = []

    def process_citation_list(self, citations: list[str]) -> list[Reference | None]:
        self.calls.append(list(citations))

        # Pad / truncate so the test fixture is forgiving about list length.
        out = list(self._responses)[: len(citations)]
        out.extend([None] * (len(citations) - len(out)))

        return out


class _RaisingClient:
    """Always raises `GrobidError` to exercise the warn-and-continue path."""

    def __init__(self, message: str = "boom") -> None:
        self._message = message
        self.calls = 0

    def process_citation_list(self, citations: list[str]) -> list[Reference | None]:  # noqa: ARG002
        self.calls += 1
        raise GrobidError(self._message)


def _ref_page(*blocks: Block) -> Page:
    """Wrap blocks into a single deterministic Page for tests."""
    return Page(page_no=1, width_px=1240, height_px=1754, blocks=list(blocks))


def _ref_block(reading_order: int, label: str, text: str) -> Block:
    return Block(
        reading_order=reading_order,
        label=label,
        type=BlockType.REFERENCE,
        bbox=(20.0, 100.0 + reading_order * 50, 500.0, 140.0 + reading_order * 50),
        confidence=0.9,
        text=text,
    )


def _text_block(reading_order: int, text: str) -> Block:
    return Block(
        reading_order=reading_order,
        label="text",
        type=BlockType.TEXT,
        bbox=(0, 0, 100, 50),
        confidence=0.9,
        text=text,
    )


# --- gating ------------------------------------------------------------------


def test_stage_disabled_returns_pages_unchanged(tmp_path: Path) -> None:
    """`references_enabled=False` is a no-op even with eligible blocks."""
    pages = [_ref_page(_ref_block(0, "reference_content", "Smith (2020). T."))]
    client = _FakeClient([Reference(title="should not be used")])
    cfg = PipelineConfig(references_enabled=False)

    out = run_reference_stage(
        pages, pdf_path=tmp_path / "doc.pdf", cfg=cfg, client=client
    )

    assert out == pages
    assert client.calls == []


def test_stage_skipped_when_label_allowlist_is_empty(tmp_path: Path) -> None:
    """Empty `reference_labels` short-circuits before any HTTP work."""
    pages = [_ref_page(_ref_block(0, "reference_content", "Smith (2020). T."))]
    client = _FakeClient([Reference(title="x")])
    cfg = PipelineConfig(references_enabled=True, reference_labels=())

    out = run_reference_stage(
        pages, pdf_path=tmp_path / "doc.pdf", cfg=cfg, client=client
    )

    assert out == pages
    assert client.calls == []


def test_stage_skipped_when_no_matching_blocks(tmp_path: Path) -> None:
    """A PDF with no reference-labeled blocks never calls GROBID."""
    pages = [_ref_page(_text_block(0, "body text"), _text_block(1, "more text"))]
    client = _FakeClient([])
    cfg = PipelineConfig(references_enabled=True)

    out = run_reference_stage(
        pages, pdf_path=tmp_path / "doc.pdf", cfg=cfg, client=client
    )

    assert out == pages
    assert client.calls == []


# --- happy path --------------------------------------------------------------


def test_stage_attaches_reference_to_matching_blocks(tmp_path: Path) -> None:
    """Every matching block ends up with `Reference` attached at the right index."""
    ref_a = Reference(
        title="A", authors=(Author(name="A One", surname="One", forename="A"),)
    )
    ref_b = Reference(title="B", year="2021")
    ref_c = Reference(title="C", venue="J3")
    pages = [
        _ref_page(
            _text_block(0, "intro"),
            _ref_block(1, "reference_content", "First raw"),
            _ref_block(2, "reference", "Second raw"),
            _ref_block(3, "reference_content", "Third raw"),
        )
    ]
    client = _FakeClient([ref_a, ref_b, ref_c])
    cfg = PipelineConfig(references_enabled=True)

    out = run_reference_stage(
        pages, pdf_path=tmp_path / "doc.pdf", cfg=cfg, client=client
    )

    # Single batched call carrying every reference text, in order.
    assert client.calls == [["First raw", "Second raw", "Third raw"]]

    blocks = out[0].blocks
    assert blocks[0].reference is None  # text block untouched
    assert blocks[1].reference is ref_a
    assert blocks[2].reference is ref_b
    assert blocks[3].reference is ref_c

    # Other Block fields survive the replace().
    assert blocks[1].text == "First raw"
    assert blocks[1].confidence == 0.9


def test_stage_leaves_block_reference_none_when_grobid_returns_none(
    tmp_path: Path,
) -> None:
    """GROBID's empty parse -> `reference` stays None; pipeline still completes."""
    pages = [
        _ref_page(
            _ref_block(0, "reference_content", "First"),
            _ref_block(1, "reference_content", "Second (gibberish)"),
        )
    ]
    client = _FakeClient([Reference(title="First"), None])
    cfg = PipelineConfig(references_enabled=True)

    out = run_reference_stage(
        pages, pdf_path=tmp_path / "doc.pdf", cfg=cfg, client=client
    )

    assert out[0].blocks[0].reference is not None
    assert out[0].blocks[0].reference.title == "First"
    assert out[0].blocks[1].reference is None


def test_stage_respects_reference_labels_filter(tmp_path: Path) -> None:
    """Only blocks whose label is in `cfg.reference_labels` reach the client."""
    pages = [
        _ref_page(
            _ref_block(0, "reference", "Section heading text"),
            _ref_block(1, "reference_content", "Real ref"),
        )
    ]
    client = _FakeClient([Reference(title="Real ref")])
    cfg = PipelineConfig(
        references_enabled=True,
        reference_labels=("reference_content",),
    )

    out = run_reference_stage(
        pages, pdf_path=tmp_path / "doc.pdf", cfg=cfg, client=client
    )

    assert client.calls == [["Real ref"]]
    # The `reference`-labeled block stays untouched.
    assert out[0].blocks[0].reference is None
    assert out[0].blocks[1].reference is not None
    assert out[0].blocks[1].reference.title == "Real ref"


def test_stage_skips_reference_block_with_empty_text(tmp_path: Path) -> None:
    """Empty / whitespace-only text never reaches GROBID."""
    pages = [
        _ref_page(
            _ref_block(0, "reference_content", "   "),
            _ref_block(1, "reference_content", "Real ref"),
        )
    ]
    client = _FakeClient([Reference(title="Real ref")])
    cfg = PipelineConfig(references_enabled=True)

    out = run_reference_stage(
        pages, pdf_path=tmp_path / "doc.pdf", cfg=cfg, client=client
    )

    assert client.calls == [["Real ref"]]
    assert out[0].blocks[0].reference is None
    assert out[0].blocks[1].reference is not None


# --- failure handling --------------------------------------------------------


def test_stage_warns_and_returns_pages_unchanged_on_grobid_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A `GrobidError` is logged at WARNING and the pages flow through unchanged."""
    pages = [_ref_page(_ref_block(0, "reference_content", "Some ref"))]
    client = _RaisingClient("upstream down")
    cfg = PipelineConfig(references_enabled=True, grobid_url="http://localhost:12345")

    with caplog.at_level(logging.WARNING, logger="ytcc_pipeline.processors.reference"):
        out = run_reference_stage(
            pages,
            pdf_path=tmp_path / "doc.pdf",
            cfg=cfg,
            client=client,
        )

    assert out == pages
    assert client.calls == 1
    assert any("grobid_error" in r.getMessage() for r in caplog.records)
    assert any("upstream down" in r.getMessage() for r in caplog.records)


def test_stage_builds_default_client_when_none_injected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `client=None`, the stage instantiates `GrobidClient` from `cfg`."""
    from ytcc_pipeline.processors import reference as ref_mod

    captured: dict[str, object] = {}

    class _Stub:
        def __init__(self, *, url: str, timeout_s: float) -> None:
            captured["url"] = url
            captured["timeout_s"] = timeout_s

        def process_citation_list(self, citations: list[str]):
            return [Reference(title=c) for c in citations]

    monkeypatch.setattr(ref_mod, "GrobidClient", _Stub)
    pages = [_ref_page(_ref_block(0, "reference_content", "Some ref"))]
    cfg = PipelineConfig(
        references_enabled=True,
        grobid_url="http://example:9090",
        grobid_timeout_s=42.0,
    )

    out = run_reference_stage(pages, pdf_path=tmp_path / "doc.pdf", cfg=cfg)

    assert captured == {"url": "http://example:9090", "timeout_s": 42.0}
    assert out[0].blocks[0].reference is not None
    assert out[0].blocks[0].reference.title == "Some ref"


# --- integration (opt-in) ----------------------------------------------------


@pytest.mark.integration
def test_integration_real_grobid_smoke(tmp_path: Path) -> None:
    """End-to-end smoke against a real GROBID at the configured URL.

    Skipped by default -- enable with `pytest -m integration`. Requires a running server
    on `http://localhost:8070`.
    """
    pages = [
        _ref_page(
            _ref_block(
                0,
                "reference_content",
                "Graff, Expert. Opin. Ther. Targets (2002) 6(1): 103-113",
            ),
            _ref_block(
                1,
                "reference_content",
                "Smith, J. (2020). Title here. Journal of Things, 5(2), 12-30.",
            ),
        )
    ]
    cfg = PipelineConfig(references_enabled=True)

    out = run_reference_stage(pages, pdf_path=tmp_path / "doc.pdf", cfg=cfg)

    enriched = [b.reference for b in out[0].blocks if b.reference is not None]
    assert len(enriched) == 2, "expected GROBID to parse both sample references"
    assert enriched[0].year == "2002"
    assert enriched[1].year == "2020"
    assert enriched[1].title == "Title here"

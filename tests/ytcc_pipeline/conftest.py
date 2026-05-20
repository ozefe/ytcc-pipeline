"""Shared pytest fixtures for ytcc_pipeline tests."""

import os
from pathlib import Path
from typing import Any, cast

# System-wide HF_HOME points at a read-only location; override before any `transformers`
# import so the cache lands under the user's home directory.
os.environ["HF_HOME"] = str(Path("~/.cache/huggingface").expanduser())

import pdf_oxide
import pytest

TEST_PDFS_DIR = Path(__file__).resolve().parents[2] / "samples"


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip `integration`-marked tests unless `-m integration` is selected."""
    if "integration" in (config.getoption("markexpr") or ""):
        return

    skipper = pytest.mark.skip(
        reason="integration tests skipped; run with `pytest -m integration`",
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skipper)


@pytest.fixture(scope="session")
def test_pdfs_dir() -> Path:
    return TEST_PDFS_DIR


def _document_builder() -> Any:
    """Return a fresh `DocumentBuilder` typed as `Any`.

    The PyO3-generated stubs annotate every instance method with an internal
    `slf_handle` first parameter that doesn't exist at runtime, which trips pyright on
    otherwise-clean calls. Casting at the boundary keeps the fluent chain
    (`.a4_page().at(...).text(...).done()`) readable in the fixture bodies.
    """
    return cast("Any", pdf_oxide.DocumentBuilder())


@pytest.fixture(scope="session")
def tiny_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a deterministic 2-page A4 PDF for fast unit tests.

    Uses `pdf_oxide.DocumentBuilder` (already a runtime dependency) so the test suite
    doesn't need a separate PDF-writer library. Coordinates are in PDF points with
    origin at the bottom-left, matching the upstream builder convention.
    """
    path = tmp_path_factory.mktemp("pdfs") / "tiny.pdf"
    body = (
        "Body paragraph: lorem ipsum dolor sit amet consectetur adipiscing elit. "
        "Some additional content to push us past the digital-born threshold."
    )
    builder = _document_builder()
    for i in range(2):
        page = builder.a4_page()
        page.at(72.0, 742.0).text(
            f"Page {i + 1} title - sample heading for digital-born detection"
        )
        page.at(72.0, 692.0).text(body)
        page.done()

    path.write_bytes(builder.build())
    return path


@pytest.fixture(scope="session")
def blank_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 3-page PDF with no text layer (simulates scanned output)."""
    path = tmp_path_factory.mktemp("blank") / "blank.pdf"
    builder = _document_builder()
    for _ in range(3):
        builder.a4_page().done()

    path.write_bytes(builder.build())
    return path

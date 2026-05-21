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
    """Apply opt-in / environment-aware skip rules.

    - `integration`: skipped unless `-m integration` is selected.
    - `slow`: skipped unless `-m slow` is selected (downloads large models / runs real
      inference; not appropriate for fast CI).
    - `gpu`: skipped when no CUDA device is reachable, regardless of marker expression.
      Honors the marker contract documented in `pyproject.toml`.

    The integration / slow checks are intentionally substring matches on the marker
    expression rather than full parses -- the existing pattern in this project.
    `pytest -m "not slow"` happens to pass through too, which is fine in practice (the
    test is skipped either way).
    """
    markexpr = config.getoption("markexpr") or ""
    cuda_available = _cuda_available()

    for item in items:
        keywords = item.keywords
        if "integration" in keywords and "integration" not in markexpr:
            item.add_marker(
                pytest.mark.skip(
                    reason="integration tests skipped; "
                    "run with `pytest -m integration`",
                ),
            )
        elif "slow" in keywords and "slow" not in markexpr:
            item.add_marker(
                pytest.mark.skip(
                    reason="slow tests skipped; run with `pytest -m slow`",
                ),
            )
        elif "gpu" in keywords and not cuda_available:
            item.add_marker(
                pytest.mark.skip(
                    reason="gpu tests skipped: no CUDA device available",
                ),
            )


def _cuda_available() -> bool:
    """Return True when torch reports a usable CUDA device.

    Tolerates the case where torch can't import so the suite doesn't fail at
    collection on a minimal harness without it.
    """
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


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

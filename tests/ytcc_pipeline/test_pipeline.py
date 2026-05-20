"""End-to-end tests for ytcc_pipeline.pipeline.

Slow / GPU-dependent -- exercise full model load and inference, so they are gated behind
the `integration` marker.
"""

import json
import tarfile
from typing import TYPE_CHECKING, Any

import pytest

from ytcc_pipeline import process_pdf
from ytcc_pipeline.config import PipelineConfig

if TYPE_CHECKING:
    from pathlib import Path


def _read_bundle_doc(tar_path: Path) -> dict[str, Any]:
    with tarfile.open(tar_path) as tf:
        fp = tf.extractfile("document.json")
        assert fp is not None
        return json.loads(fp.read())


def test_process_pdf_rejects_scanned_when_disabled(
    tiny_pdf: Path, tmp_path: Path
) -> None:
    """`scanned_enabled=False` must reject scanned input before any work happens.

    The rejection fires on the explicit `digital_born=False` caller path without needing
    any model load or temp-dir setup, so this test runs fast and stays out of the
    integration suite.
    """
    with pytest.raises(ValueError, match="scanned PDFs are disabled"):
        process_pdf(
            tiny_pdf,
            language="en",
            digital_born=False,
            output_path=tmp_path / "should_not_exist.tar",
            config=PipelineConfig(scanned_enabled=False),
        )

    assert not (tmp_path / "should_not_exist.tar").exists()


@pytest.mark.integration
def test_pipeline_runs_on_tiny_synthetic_pdf(tiny_pdf: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.tar"

    result = process_pdf(
        tiny_pdf,
        language="en",
        digital_born=True,
        output_path=output,
        config=PipelineConfig(layout_batch_size=2),
    )

    assert result == output
    assert output.is_file()

    with tarfile.open(output) as tf:
        names = tf.getnames()
        assert "document.json" in names

    doc = _read_bundle_doc(output)
    assert doc["language"] == "en"
    assert doc["digital_born"] is True
    assert doc["pipeline_version"]
    assert len(doc["pages"]) == 2
    assert doc["pages"][0]["page_no"] == 1
    assert doc["pages"][1]["page_no"] == 2


@pytest.mark.integration
@pytest.mark.parametrize(
    ("pdf_stem", "language", "expected_digital_born"),
    [
        ("904599", "en", True),
        ("1005465", "ar", True),
        ("995802", "tr", True),
    ],
)
def test_pipeline_on_real_digital_born_pdf(
    pdf_stem: str,
    language: str,
    expected_digital_born: bool,
    test_pdfs_dir: Path,
    tmp_path: Path,
) -> None:
    pdf_path = test_pdfs_dir / f"{pdf_stem}.pdf"
    if not pdf_path.is_file():
        pytest.skip(f"{pdf_path} missing")

    output = tmp_path / f"{pdf_stem}.tar"

    # Formulas are disabled here to keep the parametrised suite fast -- at ~0.5s per
    # formula crop a typical digital-born PDF would add several minutes. End-to-end
    # formula behavior is covered by the dedicated
    # `test_pipeline_recognises_formulas_on_real_pdf` test below.
    process_pdf(
        pdf_path,
        language=language,
        output_path=output,
        config=PipelineConfig(formula_enabled=False),
    )

    assert output.is_file()

    doc = _read_bundle_doc(output)
    assert doc["digital_born"] is expected_digital_born
    assert doc["language"] == language
    assert doc["pages"]

    # At least one page should have at least one block
    assert any(p["blocks"] for p in doc["pages"])


@pytest.mark.integration
def test_pipeline_recognises_formulas_on_real_pdf(
    test_pdfs_dir: Path,
    tmp_path: Path,
) -> None:
    """End-to-end formula recognition on 904599 (the smallest digital-born PDF).

    Verifies the full chain: layout -> formula routing -> crop save -> formula stage
    -> bundle. At least one `BlockType.FORMULA` block must end up successfully
    recognised: non-empty `text` (LaTeX), `image_path=None` (success path deletes the
    crop), and `miss=False`.
    """
    pdf_path = test_pdfs_dir / "904599.pdf"
    if not pdf_path.is_file():
        pytest.skip(f"{pdf_path} missing")

    output = tmp_path / "904599_with_formulas.tar"

    process_pdf(pdf_path, language="en", output_path=output)

    doc = _read_bundle_doc(output)

    formula_blocks = [
        b for p in doc["pages"] for b in p["blocks"] if b["type"] == "formula"
    ]
    assert formula_blocks, "expected at least one formula block in 904599"

    recognized = [
        b
        for b in formula_blocks
        if b.get("text") and b.get("image_path") is None and b.get("miss") is False
    ]
    assert recognized, "expected at least one successfully-recognised formula block"

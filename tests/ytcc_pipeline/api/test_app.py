"""Tests for ytcc_pipeline.api.app.

The happy-path test exercises the full pipeline through the FastAPI layer using the tiny
synthetic PDF from conftest. It loads the real layout model and is therefore marked
`integration` -- opt in with `pytest -m integration`.
"""

import json
import logging
import tarfile
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from ytcc_pipeline.config import PipelineConfig

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def client(cuda_available: bool) -> Iterator[TestClient]:
    """Build a TestClient that runs the FastAPI lifespan (loads the model).

    The lifespan unconditionally constructs the layout analyzer on the configured
    `layout_device` (production default `cuda:0`); without a reachable GPU torch raises
    before any test body runs. Skip cleanly in that case so the suite stays green on
    CPU-only hosts (e.g. GitHub Actions).

    Yields:
        A configured `TestClient`. The model load happens once at fixture setup;
        multiple tests can share the same instance via function-scoped teardown.
    """
    if not cuda_available:
        pytest.skip("FastAPI lifespan requires CUDA; no GPU available")

    from ytcc_pipeline.api.app import app

    with TestClient(app) as tc:
        yield tc


@pytest.mark.integration
def test_health_reports_model_loaded(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


@pytest.mark.integration
def test_lifespan_loads_formula_recognizer(client: TestClient) -> None:  # noqa: ARG001
    """The formula recognizer is loaded at startup and exposed on app.state.

    Mirrors the analyzer lifecycle -- both models live in app.state so the request
    handler can pass them as kwargs without per-call loads.
    """
    from ytcc_pipeline.api.app import app
    from ytcc_pipeline.models.formula import FormulaRecognizer

    cfg = app.state.config
    if cfg.formula_enabled:
        assert isinstance(app.state.formula_recognizer, FormulaRecognizer)
    else:
        assert app.state.formula_recognizer is None


def test_process_rejects_unsupported_language(
    client: TestClient, tiny_pdf: Path
) -> None:
    with tiny_pdf.open("rb") as f:
        resp = client.post(
            "/process",
            files={"pdf": ("tiny.pdf", f, "application/pdf")},
            data={"language": "zz"},
        )

    assert resp.status_code == 400
    assert "unsupported language" in resp.json()["detail"]


def test_process_rejects_scanned_when_explicit_digital_born_false_and_disabled(
    client: TestClient,
    tiny_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit `digital_born=false` plus `scanned_enabled=False` -> 415.

    The handler short-circuits before staging the upload, so this path needs no model
    work beyond the lifespan-loaded analyzer and exits fast with an explanatory message.
    """
    from dataclasses import replace

    from ytcc_pipeline.api.app import app

    monkeypatch.setattr(
        app.state,
        "config",
        replace(app.state.config, scanned_enabled=False),
    )
    with tiny_pdf.open("rb") as f:
        resp = client.post(
            "/process",
            files={"pdf": ("tiny.pdf", f, "application/pdf")},
            data={"language": "en", "digital_born": "false"},
        )

    assert resp.status_code == 415
    assert "scanned" in resp.json()["detail"].lower()


@pytest.mark.integration
def test_process_rejects_scanned_when_autodetect_resolves_scanned(
    client: TestClient,
    blank_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-detected scanned PDF plus `scanned_enabled=False` -> 415.

    Marked integration because the auto-detect probe runs after the upload is staged,
    exercising the full request path up to the model dispatcher. The blank-PDF fixture
    has no text layer so detection resolves to scanned.
    """
    from dataclasses import replace

    from ytcc_pipeline.api.app import app

    monkeypatch.setattr(
        app.state,
        "config",
        replace(app.state.config, scanned_enabled=False),
    )
    with blank_pdf.open("rb") as f:
        resp = client.post(
            "/process",
            files={"pdf": ("blank.pdf", f, "application/pdf")},
            data={"language": "en"},
        )

    assert resp.status_code == 415
    assert "scanned" in resp.json()["detail"].lower()


# --- GROBID startup probe ----------------------------------------------------


def test_grobid_startup_probe_silent_when_disabled(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`references_enabled=False` skips the probe entirely -- no log lines."""
    from ytcc_pipeline.api import app as app_mod

    call_count = 0

    def _record_call(*_args: object, **_kwargs: object) -> bool:
        nonlocal call_count
        call_count += 1
        return True

    monkeypatch.setattr(app_mod, "is_grobid_alive", _record_call)
    cfg = PipelineConfig(references_enabled=False, grobid_url="http://x")

    with caplog.at_level(logging.INFO, logger="ytcc_pipeline.api.app"):
        app_mod._grobid_startup_probe(cfg)

    assert call_count == 0
    assert not any("grobid" in r.getMessage() for r in caplog.records)


def test_grobid_startup_probe_info_when_alive(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful probe logs INFO with the configured URL."""
    from ytcc_pipeline.api import app as app_mod

    monkeypatch.setattr(app_mod, "is_grobid_alive", lambda *_a, **_kw: True)
    cfg = PipelineConfig(references_enabled=True, grobid_url="http://grobid:8070")

    with caplog.at_level(logging.INFO, logger="ytcc_pipeline.api.app"):
        app_mod._grobid_startup_probe(cfg)

    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "grobid reachable" in m.lower() and "http://grobid:8070" in m for m in msgs
    )


def test_grobid_startup_probe_warning_when_unreachable(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed probe logs WARNING but never raises -- service still starts."""
    from ytcc_pipeline.api import app as app_mod

    monkeypatch.setattr(app_mod, "is_grobid_alive", lambda *_a, **_kw: False)
    cfg = PipelineConfig(references_enabled=True, grobid_url="http://localhost:9999")

    with caplog.at_level(logging.WARNING, logger="ytcc_pipeline.api.app"):
        app_mod._grobid_startup_probe(cfg)  # must not raise

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "grobid unreachable" in r.getMessage().lower()
    ]

    assert len(warnings) == 1
    assert "http://localhost:9999" in warnings[0].getMessage()


@pytest.mark.integration
def test_process_returns_tar_bundle(client: TestClient, tiny_pdf: Path) -> None:
    with tiny_pdf.open("rb") as f:
        resp = client.post(
            "/process",
            files={"pdf": ("tiny.pdf", f, "application/pdf")},
            data={"language": "en"},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/x-tar"
    assert "X-Processing-Time" in resp.headers
    assert float(resp.headers["X-Processing-Time"]) > 0

    # Round-trip the bytes through tarfile and confirm the canonical layout.
    import io

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r") as tf:
        names = set(tf.getnames())
        assert "document.json" in names

        fp = tf.extractfile("document.json")
        assert fp is not None

        doc = json.loads(fp.read())
        assert doc["language"] == "en"
        assert isinstance(doc["pages"], list)
        assert doc["pipeline_version"]

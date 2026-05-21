"""Tests for ytcc_pipeline.config."""

import logging
import os
from dataclasses import fields
from typing import TYPE_CHECKING

import pytest

from ytcc_pipeline.config import (
    ApiSettings,
    LoggingSettings,
    PipelineConfig,
    ServiceConfig,
    load_service_config,
)
from ytcc_pipeline.schema import BlockType

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Unset every `YTCC_*` env var so `from_env` sees a clean slate.

    Returns the same `monkeypatch` instance so test bodies can chain `setenv` calls
    without requesting both fixtures.
    """
    for key in list(os.environ):
        if key.startswith("YTCC_"):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_scanned_enabled_env_override(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("YTCC_SCANNED_ENABLED", "false")
    assert PipelineConfig.from_env().scanned_enabled is False


def test_bundle_miss_images_for_coerces_string_iterable() -> None:
    """A TOML array (`list[str]`) coerces to `frozenset[BlockType]` post-init."""
    cfg = PipelineConfig(bundle_miss_images_for=["text", "formula"])  # pyright: ignore[reportArgumentType]
    assert cfg.bundle_miss_images_for == frozenset({BlockType.TEXT, BlockType.FORMULA})


def test_bundle_miss_images_for_empty_iterable_disables_all() -> None:
    """An empty iterable (e.g. TOML `[]`) coerces to a no-MISS-image config."""
    cfg = PipelineConfig(bundle_miss_images_for=[])  # pyright: ignore[reportArgumentType]
    assert cfg.bundle_miss_images_for == frozenset()


def test_bundle_miss_images_for_env_override(clean_env: pytest.MonkeyPatch) -> None:
    """Comma-separated env var selects a subset of block types."""
    clean_env.setenv("YTCC_BUNDLE_MISS_IMAGES_FOR", "text,reference")
    cfg = PipelineConfig.from_env()
    assert cfg.bundle_miss_images_for == frozenset(
        {BlockType.TEXT, BlockType.REFERENCE},
    )


def test_bundle_miss_images_for_rejects_unknown_type() -> None:
    """Unknown block type in the iterable raises at post-init."""
    with pytest.raises(ValueError, match="'wat'"):
        PipelineConfig(bundle_miss_images_for=["text", "wat"])  # pyright: ignore[reportArgumentType]


def test_from_env_returns_defaults_with_no_overrides(
    clean_env: pytest.MonkeyPatch,  # noqa: ARG001
) -> None:
    assert PipelineConfig.from_env() == PipelineConfig()


def test_from_env_picks_up_overrides(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("YTCC_RENDER_DPI", "150")
    clean_env.setenv("YTCC_LAYOUT_BATCH_SIZE", "2")
    clean_env.setenv("YTCC_RETAIN_TEMP_DIR", "true")

    config = PipelineConfig.from_env()

    assert config.render_dpi == 150
    assert config.layout_batch_size == 2
    assert config.retain_temp_dir is True


def test_from_env_picks_up_formula_overrides(clean_env: pytest.MonkeyPatch) -> None:
    """Every formula knob must be reachable via its YTCC_FORMULA_* env var."""
    clean_env.setenv("YTCC_FORMULA_ENABLED", "false")
    clean_env.setenv("YTCC_FORMULA_MODEL_ID", "custom/model")
    clean_env.setenv("YTCC_FORMULA_DEVICE", "cuda:1")
    clean_env.setenv("YTCC_FORMULA_DTYPE", "fp32")
    clean_env.setenv("YTCC_FORMULA_BATCH_SIZE", "8")
    clean_env.setenv("YTCC_FORMULA_MAX_NEW_TOKENS", "512")

    config = PipelineConfig.from_env()

    assert config.formula_enabled is False
    assert config.formula_model_id == "custom/model"
    assert config.formula_device == "cuda:1"
    assert config.formula_dtype == "fp32"
    assert config.formula_batch_size == 8
    assert config.formula_max_new_tokens == 512


def test_from_env_picks_up_bucket_overrides(clean_env: pytest.MonkeyPatch) -> None:
    """Sequence-bucketed batching settings are all env-overridable."""
    clean_env.setenv("YTCC_FORMULA_BUCKETED", "false")
    clean_env.setenv("YTCC_FORMULA_BUCKET_SMALL_THRESHOLD", "5000")
    clean_env.setenv("YTCC_FORMULA_BUCKET_MEDIUM_THRESHOLD", "60000")
    clean_env.setenv("YTCC_FORMULA_BUCKET_SMALL_TOKENS", "64")
    clean_env.setenv("YTCC_FORMULA_BUCKET_MEDIUM_TOKENS", "256")

    config = PipelineConfig.from_env()

    assert config.formula_bucketed is False
    assert config.formula_bucket_small_threshold == 5000
    assert config.formula_bucket_medium_threshold == 60000
    assert config.formula_bucket_small_tokens == 64
    assert config.formula_bucket_medium_tokens == 256


def test_from_env_picks_up_table_overrides(clean_env: pytest.MonkeyPatch) -> None:
    """Every table knob is reachable via its YTCC_TABLE_* env var."""
    clean_env.setenv("YTCC_TABLE_ENABLED", "true")
    clean_env.setenv("YTCC_TABLE_BATCH_SIZE", "4")
    clean_env.setenv("YTCC_TABLE_DEVICE", "cpu")
    clean_env.setenv("YTCC_TABLE_MIN_SIDE_PX", "80")

    config = PipelineConfig.from_env()

    assert config.table_enabled is True
    assert config.table_batch_size == 4
    assert config.table_device == "cpu"
    assert config.table_min_side_px == 80


def test_from_env_picks_up_reference_overrides(clean_env: pytest.MonkeyPatch) -> None:
    """Every reference knob is reachable via its YTCC_* env var."""
    clean_env.setenv("YTCC_REFERENCES_ENABLED", "true")
    clean_env.setenv("YTCC_GROBID_URL", "http://grobid.local:9090")
    clean_env.setenv("YTCC_GROBID_TIMEOUT_S", "30.5")
    clean_env.setenv("YTCC_REFERENCE_LABELS", "reference_content,bibliography")

    config = PipelineConfig.from_env()

    assert config.references_enabled is True
    assert config.grobid_url == "http://grobid.local:9090"
    assert config.grobid_timeout_s == 30.5
    assert config.reference_labels == ("reference_content", "bibliography")


def test_reference_labels_env_strips_whitespace_and_empties(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """The env-var parser is forgiving on whitespace / trailing commas."""
    clean_env.setenv("YTCC_REFERENCE_LABELS", " reference , , reference_content ,")
    cfg = PipelineConfig.from_env()
    assert cfg.reference_labels == ("reference", "reference_content")


def test_project_toml_carries_reference_settings() -> None:
    """The committed config.toml exposes every reference_* knob."""
    cfg = load_service_config()
    assert isinstance(cfg.pipeline.references_enabled, bool)
    assert cfg.pipeline.grobid_url.startswith("http")
    assert cfg.pipeline.grobid_timeout_s > 0
    assert len(cfg.pipeline.reference_labels) >= 1


class TestLoadServiceConfig:
    """`load_service_config` resolves TOML -> ServiceConfig."""

    def test_missing_path_falls_back_to_defaults(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("YTCC_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)

        # The project root config.toml WILL still be found via the fallback search;
        # verify we got a real config back rather than asserting exact dataclass
        # defaults.
        cfg = load_service_config()
        assert isinstance(cfg, ServiceConfig)

    def test_explicit_toml_overrides(self, tmp_path: Path) -> None:
        toml_text = """
[pipeline]
render_dpi = 200
layout_fp16 = true
digital_born_workers = 4

[api]
port = 9000

[logging]
level = "DEBUG"
"""
        path = tmp_path / "custom.toml"
        path.write_text(toml_text)

        cfg = load_service_config(path)
        assert cfg.pipeline.render_dpi == 200
        assert cfg.pipeline.layout_fp16 is True
        assert cfg.pipeline.digital_born_workers == 4

        # Untouched field stays at the dataclass default.
        assert cfg.pipeline.layout_batch_size == 8
        assert cfg.api.port == 9000
        assert cfg.logging.level == "DEBUG"

    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.toml"
        path.write_text("[pipeline]\nbogus_field = 1\n")
        with pytest.raises(ValueError, match="bogus_field"):
            load_service_config(path)

    def test_explicit_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_service_config(tmp_path / "nope.toml")

    def test_env_var_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "from_env.toml"
        path.write_text("[api]\nport = 12345\n")
        monkeypatch.setenv("YTCC_CONFIG", str(path))
        cfg = load_service_config()
        assert cfg.api.port == 12345

    def test_cwd_config_picked_when_no_explicit_or_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Step 3 of the resolution order: cwd config beats project-root fallback."""
        monkeypatch.delenv("YTCC_CONFIG", raising=False)
        (tmp_path / "config.toml").write_text("[api]\nport = 54321\n")
        monkeypatch.chdir(tmp_path)
        cfg = load_service_config()
        assert cfg.api.port == 54321

    def test_project_toml_has_expected_shape(self) -> None:
        """The committed config.toml at the project root parses cleanly."""
        cfg = load_service_config()
        assert isinstance(cfg.pipeline, PipelineConfig)
        assert isinstance(cfg.api, ApiSettings)
        assert isinstance(cfg.logging, LoggingSettings)

    def test_project_toml_exposes_formula_settings(self) -> None:
        """The committed config.toml carries every formula_* knob."""
        cfg = load_service_config()
        assert isinstance(cfg.pipeline.formula_enabled, bool)
        assert cfg.pipeline.formula_model_id
        assert cfg.pipeline.formula_device
        assert cfg.pipeline.formula_dtype in ("fp16", "fp32")
        assert cfg.pipeline.formula_batch_size >= 1
        assert cfg.pipeline.formula_max_new_tokens >= 1

    def test_omitted_render_workers_stays_none(self, tmp_path: Path) -> None:
        """TOML has no `null`; missing key -> dataclass default (None)."""
        path = tmp_path / "no_render_workers.toml"
        path.write_text("[pipeline]\ndigital_born_workers = 8\n")
        cfg = load_service_config(path)
        assert cfg.pipeline.render_workers is None
        assert cfg.pipeline.digital_born_workers == 8


def test_logging_settings_apply_sets_levels_on_named_loggers() -> None:
    """`LoggingSettings.apply()` propagates the level to `ytcc_pipeline` + `pdf_oxide`.

    Saves and restores the pre-test levels so the global mutation done by
    `basicConfig(force=True)` doesn't leak across tests run in the same process.
    """
    ytcc = logging.getLogger("ytcc_pipeline")
    pdfox = logging.getLogger("pdf_oxide")
    saved_levels = (ytcc.level, pdfox.level)

    try:
        LoggingSettings(level="WARNING", pdf_oxide_level="WARNING").apply()
        assert ytcc.level == logging.WARNING
        assert pdfox.level == logging.WARNING
    finally:
        ytcc.setLevel(saved_levels[0])
        pdfox.setLevel(saved_levels[1])


def test_env_map_covers_every_pipeline_config_field() -> None:
    """Every `PipelineConfig` field must have a row in `_ENV_MAP`.

    Guards against the common drift where a new field lands without a matching env-var
    entry, silently making the knob unreachable from deployment overrides.
    """
    from ytcc_pipeline.config import _ENV_MAP

    declared = {f.name for f in fields(PipelineConfig)}
    mapped = set(_ENV_MAP)
    missing = declared - mapped
    extra = mapped - declared
    assert not missing, f"PipelineConfig fields with no env var: {sorted(missing)}"
    assert not extra, f"_ENV_MAP has rows for unknown fields: {sorted(extra)}"

    # Convention: every env var is `YTCC_<UPPERCASE_FIELD>`.
    for field_name, (env_var, _parser) in _ENV_MAP.items():
        assert env_var == f"YTCC_{field_name.upper()}", (
            f"{field_name!r} maps to {env_var!r}; expected YTCC_{field_name.upper()}"
        )

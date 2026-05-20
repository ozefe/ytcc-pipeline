"""Top-level pipeline configuration.

This module owns two layers:

- `PipelineConfig`: the library's pure dataclass surface. Used directly by
  `process_pdf`; library-only callers never need anything else.
- `ServiceConfig`: the service / scripts surface. Bundles a `PipelineConfig` with
  `ApiSettings` and `LoggingSettings`. Loaded from `config.toml` at the project root
  (see `load_service_config`).

The TOML is the single source of truth for production tuning: the FastAPI service and
the smoke harness both load from it rather than duplicating the constants in Python.
"""

import logging
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .block_type import BlockType

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = [
    "ApiSettings",
    "LoggingSettings",
    "PipelineConfig",
    "ServiceConfig",
    "load_service_config",
]

logger = logging.getLogger(__name__)

type ImageFormat = Literal["jpeg", "png"]
type FormulaDtype = Literal["fp16", "fp32"]


@dataclass(slots=True, frozen=True)
class PipelineConfig:
    """All tunable knobs for `process_pdf`.

    Defaults represent the conservative library-mode configuration: PNG renders,
    single-threaded extraction, no FP16. The performance opt-ins near the bottom (fp16,
    fast_preproc, OCR CUDA, parallel workers) default off so library-only callers get
    behavior-preserving defaults. The service (and the project's `config.toml`)
    overrides them.
    """

    render_dpi: int = 300

    # Digital-born PDFs use a separate render DPI. At 150 DPI the layout detector and
    # pdf_oxide text-by-bbox extraction reach parity quality (~99.9% character recall)
    # while halving total wall-clock. Scanned PDFs keep `render_dpi` because OCR loses
    # content at 150 DPI.
    render_dpi_digital_born: int = 150

    page_format: ImageFormat = "png"
    crop_format: ImageFormat = "png"
    jpeg_quality: int = 95

    layout_confidence: float = 0.5
    layout_batch_size: int = 8
    layout_device: str = "cuda:0"

    ocr_batch_size: int = 64
    ocr_min_score: float = 0.5

    digital_born_sample_pages: int = 5
    digital_born_text_ratio: float = 0.3
    digital_born_min_text_chars: int = 100

    render_workers: int | None = None
    retain_temp_dir: bool = False

    layout_fp16: bool = False
    layout_fast_preproc: bool = False
    ocr_use_cuda: bool = False

    # Process-pool size for the digital-born path's per-page block extraction. Only used
    # when `digital_born=True`; the scanned path uses `ocr_workers`.
    digital_born_workers: int = 1

    # Process-pool size for the scanned path's per-page OCR. Only used when
    # `digital_born=False`; the digital-born path uses `digital_born_workers`.
    ocr_workers: int = 1

    # Master toggle for the scanned-PDF path. Disabled rejects any PDF that resolves to
    # scanned (either explicit `digital_born=False` or auto- detection) before any OCR
    # worker spawns. The 24 GiB the RapidOCR engines would otherwise reserve stays free,
    # the analyzer never has to close+reload between requests, and the service runs
    # digital-born-only.
    scanned_enabled: bool = True

    # Block types for which a MISS fallback image is written to the bundle when
    # extraction / recognition fails. Default keeps every type, the original behavior.
    # Drop entries to skip just those types; pass an empty set to never bundle MISS
    # crops. JSON entries for skipped MISS blocks still appear with `text=None`,
    # `image_path=None`, `miss=True` so downstream consumers retain the reading order
    # and bbox. IMAGE has no MISS state today so its inclusion is a no-op kept for
    # forward compatibility.
    bundle_miss_images_for: frozenset[BlockType] = field(
        default_factory=lambda: frozenset(BlockType),
    )

    # Master toggle for the formula recognition stage. Disabled keeps formula  blocks as
    # crop-only (image_path set, text=None); enabled runs each cropped formula image
    # through PP-FormulaNet-L and fills in the LaTeX.
    formula_enabled: bool = True

    # HuggingFace repo id or local snapshot path. The SafeTensors variant is loaded
    # directly via transformers; no PaddleOCR runtime is involved.
    # PaddlePaddle/PP-FormulaNet_plus-L_safetensors is a drop-in alternative with the
    # same processor config and a 2560-token decoder position limit (vs 1024 on the L
    # variant); switch to it if you observe truncation on unusually-long display
    # equations.
    formula_model_id: str = "PaddlePaddle/PP-FormulaNet-L_safetensors"

    # Independent device knob so multi-GPU hosts can split layout vs formula across
    # cards without code changes.
    formula_device: str = "cuda:0"

    # fp16 matches fp32 on real PDF crops at ~half the VRAM (~1.8 GiB vs ~3.6).
    formula_dtype: FormulaDtype = "fp16"

    formula_batch_size: int = 4

    # Per-crop generation cap. The L variant's decoder has
    # `max_position_embeddings=1024`; the plus-L variant has 2560. Production caps at
    # 1536 so the largest display-equation crops finish without truncation when running
    # plus-L; on L the cap effectively saturates at the model's own 1024-position limit
    # but per-PDF benchmarks show real crops never approach either ceiling. Also acts as
    # the `large` bucket cap when bucketing is on.
    formula_max_new_tokens: int = 1536

    # Apply `torch.compile` to the formula model. First call after enabling triggers
    # Inductor compilation (~20-30 s in the lifespan handler) but subsequent batches run
    # kernel-fused. Off by default, opt in per deployment after measuring on
    # representative documents.
    formula_torch_compile: bool = False

    # Sequence-bucketed batching. When on, the formula stage sorts crops by bbox area in
    # source-page pixels and groups them into three buckets -- small / medium / large --
    # before batched generation. Each bucket uses a tighter `max_new_tokens` cap, so a
    # batch of 8 short inline formulas doesn't wait on a long-tail display equation in
    # the same batch. Off -> flat batching with the global cap.
    formula_bucketed: bool = True

    # Upper-bound bbox area (source-page px**2) for the `small` bucket. Crops with area
    # strictly less than this go to `small`. The default 2500 ~= a 50x50-pixel crop at
    # 150 DPI -- single symbol / subscript range.
    formula_bucket_small_threshold: int = 2500

    # Upper-bound bbox area (px**2) for the `medium` bucket. Crops with area in
    # [small_threshold, medium_threshold) go to `medium`; crops at or above this go to
    # `large`. 15000 ~= a 120x125px crop -- typical inline expression at 150 DPI. Lower
    # than the initial 30000 default because the bench showed 30000 was high enough that
    # no crops on the test corpus ever reached the `large` bucket; lowering it pushes
    # genuinely-large formulas into `large` so they get the full token budget.
    formula_bucket_medium_threshold: int = 15000

    # Per-bucket `max_new_tokens` caps. The `large` bucket uses `formula_max_new_tokens`
    # (the global cap) so we don't double-encode the maximum. Smaller caps on `small` /
    # `medium` are where the bucketed-batching speedup comes from. Defaults retuned
    # after the initial bench produced 8 truncations on 995802 at small=128 / medium=384
    # the new caps gave 0 truncations on the same corpus.
    formula_bucket_small_tokens: int = 192
    formula_bucket_medium_tokens: int = 512

    # Master toggle for the table stage. Off (the default) keeps `table` blocks as
    # crop-only (image_path set, cells=None); opt in to load RapidTable SLANet+ and
    # recover the cell grid.
    table_enabled: bool = False

    # Tables per SLANet+ structure forward pass.
    table_batch_size: int = 8

    # Device for the SLANet+ ONNX session.
    table_device: str = "cuda:0"

    # Tables with either dimension below this many pixels skip the structure model and
    # fall back to image-only. Filters out spurious `table` detections on small inline
    # elements.
    table_min_side_px: int = 120

    # Master toggle for the reference-parsing stage. Off (the default) leaves
    # `Block.reference` as None on every block; opt in to forward every
    # reference-labeled block's text to an externally-managed GROBID server and attach
    # the parsed `Reference` to the block.
    references_enabled: bool = False

    # Base URL of a running GROBID server. We never spawn the JVM; users start GROBID
    # separately (Docker or `./gradlew run`). The path `/api/processCitationList` is
    # appended internally.
    grobid_url: str = "http://localhost:8070"

    # Per-request HTTP timeout. One batched POST per PDF, so this caps the wall the
    # reference stage can spend regardless of bibliography size. 60s comfortably covers
    # 200-ref bibliographies on a CRF server; raise it for very large bibliographies on
    # slower hosts.
    grobid_timeout_s: float = 60.0

    # Layout labels routed through GROBID. Defaults to every reference-shaped label the
    # layout model emits today; narrow it to e.g. `("reference_content",)` to skip
    # blocks that tend to be multi-reference blobs, or extend it if a new label appears.
    reference_labels: tuple[str, ...] = ("reference", "reference_content")

    def __post_init__(self) -> None:
        """Coerce iterable-of-strings inputs to their canonical container types.

        TOML serialises sets / tuples as arrays (`["text", "formula"]`) and env vars
        deliver comma-separated strings; both shapes reach this dataclass as plain
        Python iterables of strings. Normalising here keeps every call site free of "is
        this a list or a frozenset" guessing. `BlockType` is a `StrEnum`, so passing it
        the existing enum members is an identity round-trip.

        Raises:
            ValueError: An entry in `bundle_miss_images_for` is not a recognised
                `BlockType` value.
        """
        raw_block_types: Iterable[str] = self.bundle_miss_images_for
        coerced_block_types = frozenset(BlockType(item) for item in raw_block_types)
        object.__setattr__(self, "bundle_miss_images_for", coerced_block_types)

        # `reference_labels` is declared as `tuple[str, ...]` but TOML delivers a list.
        # Coerce so equality checks + `in` lookups behave identically regardless of
        # source.
        raw_labels: Iterable[str] = self.reference_labels
        object.__setattr__(self, "reference_labels", tuple(raw_labels))

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Build a config from `YTCC_*` environment variables.

        Each field maps to `YTCC_<UPPERCASE_FIELD>`. Unset variables keep the default.
        Unrecognised values raise (e.g. `int(...)` on a non-int).

        Returns:
            A `PipelineConfig` with the defaults overridden by env vars.
        """
        # `Any` on the parser return widens to each field's real type at the call site
        # without per-field casts. Mistakes (wrong parser for a field) would still raise
        # at dataclass __init__ on the value's actual runtime type.
        kwargs: dict[str, Any] = {
            field_name: parse(raw)
            for field_name, (env_var, parse) in _ENV_MAP.items()
            if (raw := os.environ.get(env_var)) is not None and raw != ""
        }

        if kwargs:
            logger.debug(
                "PipelineConfig.from_env applied %d override(s): %s",
                len(kwargs),
                sorted(kwargs),
            )
        else:
            logger.debug(
                "PipelineConfig.from_env: no YTCC_* env vars set; using defaults",
            )

        # Every PipelineConfig field has a default, so partial kwargs is safe.
        return cls(**kwargs)


@dataclass(slots=True, frozen=True)
class ApiSettings:
    """Settings for the FastAPI service + the smoke client."""

    # Binding 0.0.0.0 is the deployment default for the FastAPI service (containers
    # expose to the host network). Operators who want loopback-only set
    # `host = "127.0.0.1"` in `config.toml`.
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000

    # The smoke-test client waits up to this long for /health before giving up.
    boot_timeout_s: int = 120


@dataclass(slots=True, frozen=True)
class LoggingSettings:
    """Application-boundary logging knobs.

    The library never configures handlers; the FastAPI lifespan and the scripts call
    `apply()` once to set levels + format.
    """

    level: str = "INFO"

    # `pdf_oxide` emits per-font internals at INFO -- chatty for operators.
    pdf_oxide_level: str = "WARNING"
    format: str = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"

    def apply(self) -> None:
        """Configure root logging + per-logger overrides in one call."""
        logging.basicConfig(level=self.level, format=self.format, force=True)
        logging.getLogger("ytcc_pipeline").setLevel(self.level)
        logging.getLogger("pdf_oxide").setLevel(self.pdf_oxide_level)

        # Emitted AFTER `basicConfig(force=True)` so it actually reaches a handler;
        # earlier handlers (if any) are torn down by the force-reconfigure above.
        logger.debug(
            "logging configured: level=%s pdf_oxide_level=%s",
            self.level,
            self.pdf_oxide_level,
        )


@dataclass(slots=True, frozen=True)
class ServiceConfig:
    """Aggregated config: pipeline + api + logging."""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    api: ApiSettings = field(default_factory=ApiSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


def load_service_config(path: str | Path | None = None) -> ServiceConfig:
    """Load `ServiceConfig` from the project TOML.

    Resolution order:
        1. The `path` argument, if given.
        2. The `YTCC_CONFIG` environment variable.
        3. `config.toml` in the current working directory.
        4. `config.toml` at the project root (relative to this file).
        5. Built-in dataclass defaults.

    Unknown keys in any section raise `ValueError` -- typos shouldn't pass silently.
    Missing sections fall back to dataclass defaults. TOML has no `null`, so any
    optional-with-None field (e.g. `render_workers`) is expressed by omitting the key.

    Args:
        path: Explicit TOML path; overrides all auto-resolution.

    Returns:
        A `ServiceConfig` reflecting the TOML (or all-defaults if no TOML file is found)

    Raises:
        ValueError: The TOML contains an unknown key in `pipeline`, `api`, or `logging`.
        FileNotFoundError: Explicit `path` or `YTCC_CONFIG` points to a missing file.
        tomllib.TOMLDecodeError: The TOML file is malformed.
    """
    resolved = _resolve_config_path(path)
    if resolved is None:
        # `LoggingSettings.apply()` hasn't run yet on the very first call -- this DEBUG
        # line is for operators who already configured logging.
        logger.debug("config: no TOML found; using dataclass defaults")
        return ServiceConfig()

    # tomllib.load expects binary mode (the spec mandates UTF-8 + treats the bytes
    # directly). Open as 'rb' rather than reading text + decoding.
    with resolved.open("rb") as f:
        raw = tomllib.load(f)

    logger.info("config loaded: path=%s sections=%s", resolved, sorted(raw))
    return ServiceConfig(
        pipeline=_build_config_section(
            PipelineConfig,
            raw.get("pipeline", {}),
            section="pipeline",
        ),
        api=_build_config_section(
            ApiSettings,
            raw.get("api", {}),
            section="api",
        ),
        logging=_build_config_section(
            LoggingSettings,
            raw.get("logging", {}),
            section="logging",
        ),
    )


def _resolve_config_path(explicit: str | Path | None) -> Path | None:
    """Walk the resolution order and return the first existing TOML."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            msg = f"config not found: {path}"
            raise FileNotFoundError(msg)

        return path

    if env_path := os.environ.get("YTCC_CONFIG"):
        path = Path(env_path)
        if not path.is_file():
            msg = f"YTCC_CONFIG points to a missing file: {path}"
            raise FileNotFoundError(msg)

        return path

    cwd_path = Path.cwd() / "config.toml"
    if cwd_path.is_file():
        return cwd_path

    # `Path(__file__).parents[2]` is the project root (.../ytcc-pipeline-service) when
    # the library is installed editable. For a packaged wheel install this falls through
    # to None and the caller gets dataclass defaults.
    project_root = Path(__file__).resolve().parents[2]
    project_path = project_root / "config.toml"
    if project_path.is_file():
        return project_path

    return None


def _build_config_section(
    cls: type,
    data: dict[str, Any],
    *,
    section: str,
) -> Any:  # noqa: ANN401 -- returns one of three dataclasses; widening to Any avoids a verbose TypeVar-bound generic
    """Strict dataclass construction: reject unknown keys with the section name."""
    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - valid)
    if unknown:
        # Log before raising: config load happens once at startup, and the operator
        # sees this line in the log even if the ValueError is caught + reformatted by a
        # wrapping process supervisor.
        logger.error(
            "unknown config keys: section=%s unknown=%s valid=%s",
            section,
            unknown,
            sorted(valid),
        )
        msg = f"unknown keys in `{section}` config: {unknown}"
        raise ValueError(msg)

    return cls(**data)


def _parse_bool(value: str) -> bool:
    """Parse a string to a bool using shell-style truthy tokens.

    Truthy values (case-insensitive): `1`, `true`, `yes`, `on`. Anything else (including
    empty string, `0`, `false`, `no`, `off`) is `False`. Used by every bool-typed
    `YTCC_*` env-var entry in `_ENV_MAP`.

    Args:
        value: The raw env-var string.

    Returns:
        `True` if `value` matches a truthy token, `False` otherwise.
    """
    return value.lower() in ("1", "true", "yes", "on")


def _parse_block_type_set(value: str) -> frozenset[BlockType]:
    """Parse a comma-separated list of `BlockType` values.

    Empty / whitespace-only input yields an empty set (skip all MISS images). Unknown
    names raise `ValueError` with the valid list.
    """
    parts = [item.strip().lower() for item in value.split(",") if item.strip()]
    return frozenset(BlockType(part) for part in parts)


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    """Parse a comma-separated string into a tuple of stripped non-empty entries.

    Used for `reference_labels`. Empty / whitespace-only entries are dropped silently so
    users can leave trailing commas or accidental blanks in env-var lists without
    surprises.
    """
    return tuple(part.strip() for part in value.split(",") if part.strip())


# Field-name -> (env-var, parser) for every `PipelineConfig` knob. Keep every dataclass
# field covered here; the matching test enforces parity.
_ENV_MAP: dict[str, tuple[str, Callable[[str], Any]]] = {
    "render_dpi": ("YTCC_RENDER_DPI", int),
    "render_dpi_digital_born": ("YTCC_RENDER_DPI_DIGITAL_BORN", int),
    "page_format": ("YTCC_PAGE_FORMAT", str),
    "crop_format": ("YTCC_CROP_FORMAT", str),
    "jpeg_quality": ("YTCC_JPEG_QUALITY", int),
    "layout_confidence": ("YTCC_LAYOUT_CONFIDENCE", float),
    "layout_batch_size": ("YTCC_LAYOUT_BATCH_SIZE", int),
    "layout_device": ("YTCC_LAYOUT_DEVICE", str),
    "ocr_batch_size": ("YTCC_OCR_BATCH_SIZE", int),
    "ocr_min_score": ("YTCC_OCR_MIN_SCORE", float),
    "digital_born_sample_pages": ("YTCC_DIGITAL_BORN_SAMPLE_PAGES", int),
    "digital_born_text_ratio": ("YTCC_DIGITAL_BORN_TEXT_RATIO", float),
    "digital_born_min_text_chars": ("YTCC_DIGITAL_BORN_MIN_TEXT_CHARS", int),
    "render_workers": ("YTCC_RENDER_WORKERS", int),
    "retain_temp_dir": ("YTCC_RETAIN_TEMP_DIR", _parse_bool),
    "layout_fp16": ("YTCC_LAYOUT_FP16", _parse_bool),
    "layout_fast_preproc": ("YTCC_LAYOUT_FAST_PREPROC", _parse_bool),
    "ocr_use_cuda": ("YTCC_OCR_USE_CUDA", _parse_bool),
    "digital_born_workers": ("YTCC_DIGITAL_BORN_WORKERS", int),
    "ocr_workers": ("YTCC_OCR_WORKERS", int),
    "scanned_enabled": ("YTCC_SCANNED_ENABLED", _parse_bool),
    "bundle_miss_images_for": ("YTCC_BUNDLE_MISS_IMAGES_FOR", _parse_block_type_set),
    "formula_enabled": ("YTCC_FORMULA_ENABLED", _parse_bool),
    "formula_model_id": ("YTCC_FORMULA_MODEL_ID", str),
    "formula_device": ("YTCC_FORMULA_DEVICE", str),
    "formula_dtype": ("YTCC_FORMULA_DTYPE", str),
    "formula_batch_size": ("YTCC_FORMULA_BATCH_SIZE", int),
    "formula_max_new_tokens": ("YTCC_FORMULA_MAX_NEW_TOKENS", int),
    "formula_torch_compile": ("YTCC_FORMULA_TORCH_COMPILE", _parse_bool),
    "formula_bucketed": ("YTCC_FORMULA_BUCKETED", _parse_bool),
    "formula_bucket_small_threshold": ("YTCC_FORMULA_BUCKET_SMALL_THRESHOLD", int),
    "formula_bucket_medium_threshold": ("YTCC_FORMULA_BUCKET_MEDIUM_THRESHOLD", int),
    "formula_bucket_small_tokens": ("YTCC_FORMULA_BUCKET_SMALL_TOKENS", int),
    "formula_bucket_medium_tokens": ("YTCC_FORMULA_BUCKET_MEDIUM_TOKENS", int),
    "table_enabled": ("YTCC_TABLE_ENABLED", _parse_bool),
    "table_batch_size": ("YTCC_TABLE_BATCH_SIZE", int),
    "table_device": ("YTCC_TABLE_DEVICE", str),
    "table_min_side_px": ("YTCC_TABLE_MIN_SIDE_PX", int),
    "references_enabled": ("YTCC_REFERENCES_ENABLED", _parse_bool),
    "grobid_url": ("YTCC_GROBID_URL", str),
    "grobid_timeout_s": ("YTCC_GROBID_TIMEOUT_S", float),
    "reference_labels": ("YTCC_REFERENCE_LABELS", _parse_str_tuple),
}

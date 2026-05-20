"""Sweep definitions: one knob, several values, one PDF, one base config.

Two base configs hold every non-swept knob constant so the comparison stays
apples-to-apples. Sweeps that don't need the heavy stages turn them off explicitly
(formula / table / references all default false in the base config; the formula and
table sweeps flip them back on).
"""

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

from ytcc_pipeline import PipelineConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "DIGITAL_BORN_BASE",
    "SCANNED_BASE",
    "Sweep",
    "all_sweeps",
]

# Pre-compute once: which `Sweep.knob` strings correspond to real `PipelineConfig`
# fields. Anything else (e.g. `"repeat"`, `"pdf"`, `"stages"`) is a sentinel used by
# multi-dimensional sweeps and is handled entirely by `config_factory`.
_CONFIG_FIELD_NAMES: frozenset[str] = frozenset(f.name for f in fields(PipelineConfig))


# --- base configs -----------------------------------------------------------
#
# Both bases turn off every stage that isn't strictly needed. Per-sweep overrides enable
# the stage under test. This keeps runtime per row near the floor for the given path:
#
# - DIGITAL_BORN_BASE: layout + pdf_oxide text only. No formula, no table, no
#   references, no MISS crops written, no JPEG temp pages.
# - SCANNED_BASE: layout + RapidOCR with CUDA. Same stages-off discipline.
#
# Per-sweep helpers below add only the toggles required for the run.

DIGITAL_BORN_BASE = PipelineConfig(
    # Page rendering (digital-born path picks render_dpi_digital_born).
    render_dpi=300,
    render_dpi_digital_born=150,
    page_format="jpeg",
    crop_format="jpeg",
    jpeg_quality=90,
    # Layout.
    layout_confidence=0.5,
    layout_batch_size=8,
    layout_device="cuda:0",
    layout_fp16=True,
    layout_fast_preproc=True,
    # OCR: irrelevant on digital-born, but kept default.
    ocr_batch_size=64,
    ocr_min_score=0.5,
    ocr_use_cuda=False,
    # Parallelism.
    digital_born_workers=16,
    ocr_workers=1,
    render_workers=None,
    # Heavy stages off by default -- sweeps that test them flip them back on.
    formula_enabled=False,
    table_enabled=False,
    references_enabled=False,
    # Skip MISS crops to avoid disk noise during the sweeps.
    bundle_miss_images_for=frozenset(),
    retain_temp_dir=False,
)

SCANNED_BASE = PipelineConfig(
    render_dpi=300,
    render_dpi_digital_born=150,
    page_format="jpeg",
    crop_format="jpeg",
    jpeg_quality=90,
    layout_confidence=0.5,
    layout_batch_size=8,
    layout_device="cuda:0",
    layout_fp16=True,
    layout_fast_preproc=True,
    ocr_batch_size=64,
    ocr_min_score=0.5,
    ocr_use_cuda=True,
    digital_born_workers=1,
    ocr_workers=6,
    render_workers=None,
    formula_enabled=False,
    table_enabled=False,
    references_enabled=False,
    bundle_miss_images_for=frozenset(),
    retain_temp_dir=False,
    scanned_enabled=True,
)


@dataclass(slots=True, frozen=True)
class Sweep:
    """One knob-sweep definition.

    Attributes:
        name: Sweep id; used for `<name>.csv` and `<name>.md` filenames.
        knob: The `PipelineConfig` field name to sweep, OR a sentinel string (e.g.
            `"repeat"`, `"pdf"`, `"stages"`) for multi-dimensional sweeps where the
            value doesn't map to a single config field. Sentinel knobs require a
            `config_factory` that interprets the value; `build_config` skips the
            `replace(base, knob=value)` step for them.
        values: The set of values to try, in the order the report should display them.
        pdf: Default source PDF filename (under `samples/` at the project root). Used
            unless overridden per-value via `per_value_pdf`.
        language: ISO 639-1 code passed to OCR; ignored on digital-born. Used unless
            overridden per-value via `per_value_language`.
        digital_born: Override the auto-detect probe. Saves ~1 s and keeps the
            dispatcher deterministic. Use `None` for sweeps that cross both regimes
            (e.g. cross-corpus) so the dispatcher probes each PDF.
        base: Base config; the sweep overrides `knob` per value.
        config_factory: Optional adjustment to apply to the base after setting the swept
            value (e.g. `formula_enabled=True`). Receives the base config and the
            current value and returns the final `PipelineConfig`.
        title: Headline for the plot -- a human-readable phrase naming the stage and the
            dimension under test (e.g. "Layout Analysis: fp16 vs fp32 Precision"). Falls
            back to `name` if empty.
        description: One-sentence context for the Markdown summary -- describes what the
            knob is.
        findings: One-sentence summary of what the bench data actually shows. Used as
            the plot subtitle. Empty until the sweep has been run at least once.
        quality_focus: Quality columns surfaced in the per-sweep table.
        needs_grobid: `True` for sweeps that need a reachable GROBID; the runner skips
            them with a clear log line otherwise.
        per_value_pdf: Optional mapping `{value: pdf_filename}` for sweeps whose rows
            use different PDFs (e.g. `cross_corpus`, `language_matrix`). When set, the
            runner picks the per-value entry instead of `pdf`.
        per_value_language: Optional mapping `{value: language_code}` paired with
            `per_value_pdf` for multi-PDF sweeps that span multiple OCR languages.
    """

    name: str
    knob: str
    values: Sequence[Any]
    pdf: str
    language: str
    digital_born: bool | None
    base: PipelineConfig
    description: str
    title: str = ""
    findings: str = ""
    quality_focus: tuple[str, ...] = ("blocks_total", "text_chars", "miss")
    config_factory: Callable[[PipelineConfig, Any], PipelineConfig] | None = None
    needs_grobid: bool = False
    per_value_pdf: Mapping[Any, str] | None = None
    per_value_language: Mapping[Any, str] | None = None

    def build_config(self, value: Any) -> PipelineConfig:  # noqa: ANN401  -- sweep values are heterogeneous (int / bool / str / float)
        """Apply the swept value (and any extras) to the base.

        For sentinel knobs (not a `PipelineConfig` field) the `replace` step is skipped
        and the config_factory is solely responsible for translating the value into
        config overrides.
        """
        cfg = self.base
        if self.knob in _CONFIG_FIELD_NAMES:
            cfg = replace(cfg, **{self.knob: value})

        if self.config_factory is not None:
            cfg = self.config_factory(cfg, value)

        return cfg

    def resolve_pdf(self, value: Any) -> str:  # noqa: ANN401
        """Return the PDF filename for a sweep value.

        Falls back to `self.pdf` when `per_value_pdf` is unset or doesn't contain the
        given value.
        """
        if self.per_value_pdf is not None:
            return self.per_value_pdf.get(value, self.pdf)

        return self.pdf

    def resolve_language(self, value: Any) -> str:  # noqa: ANN401
        """Return the language code for a sweep value.

        Falls back to `self.language` when `per_value_language` is unset or doesn't
        contain the given value.
        """
        if self.per_value_language is not None:
            return self.per_value_language.get(value, self.language)

        return self.language

    def referenced_pdfs(self) -> set[str]:
        """Return every PDF filename this sweep may load across its values."""
        if self.per_value_pdf is None:
            return {self.pdf}

        return {self.pdf, *self.per_value_pdf.values()}


# --- per-sweep config adjustments -------------------------------------------
#
# Sweeps that need a stage enabled beyond what the base turns off declare the extra
# toggles here, so the `Sweep` rows stay one-liners. The `_value` arg matches the
# `config_factory` Protocol; individual factories ignore it, but the signature has to
# accept it. ANN401 noqa'd in each because the value type is the same heterogeneous one
# `Sweep.build_config` takes.


def _enable_formula(cfg: PipelineConfig, _value: Any) -> PipelineConfig:  # noqa: ANN401
    return replace(cfg, formula_enabled=True)


def _enable_table(cfg: PipelineConfig, _value: Any) -> PipelineConfig:  # noqa: ANN401
    return replace(cfg, table_enabled=True)


def _enable_references(cfg: PipelineConfig, _value: Any) -> PipelineConfig:  # pyright: ignore[reportUnusedFunction] # noqa: ANN401
    return replace(cfg, references_enabled=True)


def _bundle_miss_for_text(cfg: PipelineConfig, _value: Any) -> PipelineConfig:  # noqa: ANN401
    """OCR sweeps need MISS crops emitted so blocks survive in the bundle.

    Without `bundle_miss_images_for={text}` the JSON still records the block, but it has
    neither text nor a fallback image -- there's no quality signal to compare across
    values.
    """
    return replace(cfg, bundle_miss_images_for=frozenset({"text", "reference"}))


def _enable_all_stages(cfg: PipelineConfig, _value: Any) -> PipelineConfig:  # noqa: ANN401
    """Turn on formula + table + reference parsing.

    Matches the production configuration enabled in `config.toml` for hosts that run
    every stage. Used by the methodology sweeps (`variance`, `cross_corpus`) where the
    whole-pipeline wall is the headline number.
    """
    return replace(
        cfg,
        formula_enabled=True,
        table_enabled=True,
        references_enabled=True,
    )


def _apply_stages_value(cfg: PipelineConfig, value: Any) -> PipelineConfig:  # noqa: ANN401
    """Toggle every stage in lockstep based on the sweep value.

    Used by `production_realistic_*`: value `"minimal"` keeps the base config (layout +
    text only); `"all_on"` enables formula + table + references in one go.
    """
    if value == "all_on":
        return _enable_all_stages(cfg, value)

    return cfg


# Named bucket-threshold presets for `formula_bucket_thresholds`. Each tuple is
# `(small_threshold, medium_threshold, small_tokens, medium_tokens)`; the four knobs
# co-vary because shrinking a bucket without raising its `*_tokens` cap produces
# truncations, and the per-bucket caps are the only knob that translates threshold
# choice into wall savings.
_BUCKET_THRESHOLD_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "default": (2500, 15000, 192, 512),
    "wider_small": (5000, 15000, 256, 512),
    "wider_medium": (2500, 30000, 192, 768),
    "tighter": (1500, 8000, 128, 384),
    "scaled_2x": (5000, 30000, 192, 512),
}


def _apply_bucket_preset(cfg: PipelineConfig, value: Any) -> PipelineConfig:  # noqa: ANN401
    """Apply a named bucket-threshold preset and enable formula bucketing."""
    small_t, medium_t, small_tok, medium_tok = _BUCKET_THRESHOLD_PRESETS[value]
    return replace(
        cfg,
        formula_enabled=True,
        formula_bucketed=True,
        formula_bucket_small_threshold=small_t,
        formula_bucket_medium_threshold=medium_t,
        formula_bucket_small_tokens=small_tok,
        formula_bucket_medium_tokens=medium_tok,
    )


# DPI x bucket-threshold pairs for `dpi_bucket_interaction`. Bucket thresholds are
# bbox-area thresholds in source-page px**2; a 2x DPI scale is a 4x area scale, so the
# "scaled_4x" variant matches the 300 DPI render at the same effective crop size.
_DPI_BUCKET_PRESETS: dict[str, tuple[int, int, int]] = {
    "150_default": (150, 2500, 15000),
    "300_default": (300, 2500, 15000),
    "150_scaled_4x": (150, 10000, 60000),
    "300_scaled_4x": (300, 10000, 60000),
}


def _apply_dpi_bucket_preset(cfg: PipelineConfig, value: Any) -> PipelineConfig:  # noqa: ANN401
    """Apply paired render-DPI + bucket-threshold settings."""
    dpi, small_t, medium_t = _DPI_BUCKET_PRESETS[value]
    return replace(
        cfg,
        formula_enabled=True,
        formula_bucketed=True,
        render_dpi_digital_born=dpi,
        formula_bucket_small_threshold=small_t,
        formula_bucket_medium_threshold=medium_t,
    )


# Hybrid base for sweeps that cross both regimes (`cross_corpus`). Pairs the
# digital-born tuning (`digital_born_workers=16`, `layout_fp16`, ...) with the
# scanned-path tuning (`ocr_use_cuda=True`, `ocr_workers=6`) so each row picks the
# settings it actually needs without per-row base swaps.
PRODUCTION_HYBRID_BASE = replace(
    DIGITAL_BORN_BASE,
    ocr_use_cuda=True,
    ocr_workers=6,
)


# --- the full sweep list ----------------------------------------------------


# Default PDFs used unless overridden per-sweep.
DIGITAL_BORN_PDF = "904599.pdf"  # smallest digital-born English, ~15s baseline
SCANNED_PDF = "084016.pdf"  # English scanned with formulas + refs

# Turkish scanned thesis -- used by `jpeg_quality` because its OCR character recall is
# sensitive to compression artefacts around diacritics.
SCANNED_TR_PDF = "101123.pdf"

# Arabic + Turkish digital-born + scanned -- used by `cross_corpus` and
# `language_matrix` to exercise both the digital-born path and OCR on RTL scripts.
DIGITAL_BORN_TR_PDF = "995802.pdf"
DIGITAL_BORN_AR_PDF = "1005465.pdf"
SCANNED_AR_PDF = "823568.pdf"


def all_sweeps() -> list[Sweep]:
    """Return every defined sweep in execution order (fast -> slow)."""
    return [
        # ---- digital-born sweeps (fast, no OCR) -----------------------------
        Sweep(
            name="layout_batch_size",
            knob="layout_batch_size",
            title="Layout Analysis: Batch Size Sweep",
            values=[1, 2, 4, 8, 16, 24],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            description=(
                "Layout-model batch size on a digital-born PDF. Tests how many pages "
                "PP-DocLayoutV3 processes per GPU forward pass; larger batches mean "
                "fewer kernel launches but a higher VRAM peak."
            ),
            findings=(
                "Wall is essentially flat (32-36 s) across bs=1 to bs=24, but VRAM "
                "scales linearly from 1.5 GiB at bs=1 to 14.8 GiB at bs=24. Block "
                "count is stable at ~1588. The default bs=8 (6.4 GiB) is the sweet "
                "spot; larger batches buy no speed, just VRAM."
            ),
        ),
        Sweep(
            name="layout_fp16",
            knob="layout_fp16",
            title="Layout Analysis: fp16 vs fp32 Precision",
            values=[False, True],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            description=(
                "Run PP-DocLayoutV3 in fp16 vs fp32 on a digital-born PDF. Speed comes "
                "from tensor-core throughput; quality is tracked via total detected "
                "blocks."
            ),
            findings=(
                "fp16 cuts layout VRAM 44% (12.3 -> 6.9 GiB) at the same wall (~30s) "
                "and identical detection output (~1588 blocks). Pure win on "
                "tensor-core GPUs."
            ),
            quality_focus=("blocks_total", "blocks_text", "blocks_image"),
        ),
        Sweep(
            name="layout_fast_preproc",
            knob="layout_fast_preproc",
            title="Layout Analysis: cv2 Fast Preprocessing vs PIL/HF Pipeline",
            values=[False, True],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            description=(
                "cv2 preprocessing + producer-thread overlap vs HF's PIL-based "
                "AutoImageProcessor. Layout-stage isolated."
            ),
            findings=(
                "cv2 + producer thread shaves 1.12x wall (33.5 -> 30.0 s) with "
                "identical VRAM. Detection drops slightly (1602 -> 1587 blocks, ~1% "
                "fewer) from the resize-kernel change -- accept the trade if wall "
                "matters."
            ),
            quality_focus=("blocks_total", "blocks_text", "blocks_image"),
        ),
        Sweep(
            name="render_dpi_digital_born",
            knob="render_dpi_digital_born",
            title="Page Rendering: DPI Sweep on Digital-Born PDFs",
            values=[100, 150, 200, 250, 300],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            description=(
                "Page-render DPI for digital-born PDFs. Layout downsamples to 800x800 "
                "internally and pdf_oxide reads the text layer directly, so the "
                "quality delta is small but render cost scales with DPI**2."
            ),
            findings=(
                "Wall grows 38% (28.9s -> 39.8s) from 100 to 300 DPI while text chars "
                "stay flat at 167.8k (pdf_oxide reads the embedded text layer, not the "
                "render). Block count peaks at 200 DPI (1598); the default 150 "
                "captures essentially everything (1587)."
            ),
            quality_focus=("blocks_total", "text_chars", "blocks_text"),
        ),
        Sweep(
            name="page_format",
            knob="page_format",
            title="Page Rendering: PNG vs JPEG Temp Pages",
            values=["png", "jpeg"],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            description=(
                "Temp page-render container: PNG (lossless) vs JPEG. Layout "
                "downsamples to 800x800 so the JPEG artefacts are erased; the only "
                "impact is render wall."
            ),
            findings=(
                "JPEG temp pages run 1.10x faster (32.9s -> 30.0s) than PNG with "
                "byte-identical detection output (1587-1588 blocks, 167.8k chars). "
                "JPEG is the right default."
            ),
            quality_focus=("blocks_total", "blocks_text", "text_chars"),
        ),
        Sweep(
            name="jpeg_quality",
            knob="jpeg_quality",
            title="JPEG Compression Quality vs OCR Character Recall (Turkish Scanned)",
            # Four points span the range. The original seven-value sweep (40-100 in
            # steps of 10) proved RapidOCR PP-OCRv5 is bit-identical at every value on
            # this PDF, so the in-between rows added no signal. Keep one row deep in the
            # "low quality" zone (40), one near the docs floor (70), the production
            # default (90), and the lossless ceiling (100).
            values=[40, 70, 90, 100],
            pdf=SCANNED_TR_PDF,
            language="tr",
            digital_born=False,
            base=replace(SCANNED_BASE, page_format="jpeg", crop_format="jpeg"),
            config_factory=_bundle_miss_for_text,
            description=(
                "JPEG quality knob (1-100). Affects the temp page renders that the "
                "scanned-path OCR reads from disk. Benchmarked on a Turkish scanned "
                "PDF because Turkish diacritics make OCR character recall sensitive to "
                "JPEG compression artefacts; the digital-born path bypasses the "
                "rendered image entirely (pdf_oxide reads the embedded text layer) so "
                "there's no signal to measure there."
            ),
            findings=(
                "RapidOCR PP-OCRv5 is fully robust to JPEG compression from q=40 to "
                "q=100 on this Turkish scanned PDF: 178612 chars, 824 text blocks, 1 "
                "MISS -- bit-identical at every value. Wall is flat at ~115s. q=80 is "
                "a safe lower bound for this corpus."
            ),
            quality_focus=("text_chars", "miss", "blocks_text"),
        ),
        Sweep(
            name="digital_born_workers",
            knob="digital_born_workers",
            title="Digital-Born Path: Spawn-Pool Worker Count",
            values=[1, 2, 4, 8, 16, 32, 64],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            description=(
                "Process-pool size for digital-born block extraction. Each worker "
                "opens its own pdf_oxide doc; speed scales until OpenBLAS thread "
                "storms cap it on high-core boxes."
            ),
            findings=(
                "Single-worker is fastest (21s) -- the serial path skips spawn-pool "
                "overhead. Above that, wall plateaus at ~30s for 4-16 workers (no gain "
                "from parallelism on this 155-page PDF). 32 and 64 workers crash with "
                "BrokenProcessPool from resource exhaustion on the bench host."
            ),
            quality_focus=("blocks_total", "text_chars"),
        ),
        Sweep(
            name="formula_enabled",
            knob="formula_enabled",
            title="Formula Recognition Stage: On vs Off",
            values=[False, True],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            description=(
                "Master toggle for the formula stage. Off: formula crops ship "
                "image-only. On: PP-FormulaNet-L recovers LaTeX."
            ),
            findings=(
                "Enabling the formula stage adds 469s (30 -> 499 s, 16.6x) to recover "
                "all 900 formula blocks as LaTeX (~106k chars) on this thesis. Cost is "
                "per-formula; a PDF with no formulas would see no overhead."
            ),
            quality_focus=(
                "blocks_formula",
                "formulas_with_text",
                "formula_latex_chars",
                "formulas_with_image",
            ),
        ),
        Sweep(
            name="formula_model_id",
            knob="formula_model_id",
            title="Formula Recognition: PP-FormulaNet-L vs PP-FormulaNet_plus-L",
            values=[
                "PaddlePaddle/PP-FormulaNet-L_safetensors",
                "PaddlePaddle/PP-FormulaNet_plus-L_safetensors",
            ],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_enable_formula,
            description=(
                "PP-FormulaNet-L vs PP-FormulaNet_plus-L. The +L variant has a "
                "2560-token decoder position limit vs L's 1024."
            ),
            findings=(
                "plus-L runs 1.54x faster (508 -> 331s) at the same VRAM (~6.3 GiB), "
                "but emits 19% fewer LaTeX chars (106k -> 86k) on the same 900 crops "
                "-- output is shorter, not a strict improvement. Inspect samples "
                "before swapping."
            ),
            quality_focus=(
                "blocks_formula",
                "formulas_with_text",
                "formula_latex_chars",
                "formulas_with_image",
            ),
        ),
        Sweep(
            name="formula_dtype",
            knob="formula_dtype",
            title="Formula Recognition: fp16 vs fp32 Inference",
            values=["fp32", "fp16"],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_enable_formula,
            description=(
                "fp16 vs fp32 inference for PP-FormulaNet-L. fp16 should halve weight "
                "memory and exploit tensor cores."
            ),
            findings=(
                "fp16 trims wall 1.04x (518 -> 498s) and VRAM 13% (6.3 -> 5.5 GiB) -- "
                "far less than the theoretical half because decoder KV-cache and "
                "activations dominate at bs=4. LaTeX output is 99.7% identical (274860 "
                "vs 274116 chars)."
            ),
            quality_focus=(
                "formulas_with_text",
                "formula_latex_chars",
                "formulas_with_image",
            ),
        ),
        Sweep(
            name="formula_batch_size",
            knob="formula_batch_size",
            title="Formula Recognition: Batch Size Sweep",
            # bs=1 takes ~15 min on 900 crops -- pruned for runtime. Top end extended to
            # bs=24/32 specifically to map the VRAM cliff on the bench card: bs=16 used
            # 12 GiB, bs=24 should land near 18 GiB, bs=32 will likely OOM on a 24 GiB
            # 3090. An `status="oom"` row at the OOM point IS the data point -- it
            # anchors the `formula_batch_size` recommendation for high-VRAM hosts.
            values=[4, 8, 16, 24, 32],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_enable_formula,
            description=(
                "Formula-recognition batch size. Greedy generation runs the whole "
                "batch for the slowest row, so the optimum balances throughput against "
                "straggler cost."
            ),
            findings=(
                "bs=16 is 2.13x faster than bs=4 (487 -> 229 s) at 2.1x VRAM (5.6 -> "
                "12.0 GiB). LaTeX output is byte-equivalent at every batch size -- "
                "pick the largest your VRAM allows."
            ),
            quality_focus=(
                "formulas_with_text",
                "formula_latex_chars",
                "formulas_with_image",
            ),
        ),
        Sweep(
            name="formula_bucketed",
            knob="formula_bucketed",
            title="Formula Recognition: Sequence-Bucketed Batching",
            values=[False, True],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_enable_formula,
            description=(
                "Sequence-bucketed batching. Groups crops by bbox area so each batch's "
                "slowest row doesn't drag the rest through unused decode steps."
            ),
            findings=(
                "Bucketing delivers 1.24x speedup (600 -> 483s) at no VRAM cost (~6.0 "
                "GiB either way) and equivalent LaTeX output. A free win on documents "
                "with mixed-length formulas."
            ),
            quality_focus=(
                "formulas_with_text",
                "formula_latex_chars",
                "formulas_with_image",
            ),
        ),
        Sweep(
            name="formula_torch_compile",
            knob="formula_torch_compile",
            title="Formula Recognition: torch.compile vs Eager Mode",
            values=[False, True],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_enable_formula,
            description=(
                "torch.compile on the formula model. First call pays ~20-30s Inductor "
                "compilation; subsequent batches run kernel-fused. The wall in this "
                "sweep includes the compile cost."
            ),
            findings=(
                "torch.compile is a wash on this workload (482 vs 486s, ~1% noise) -- "
                "the ~20-30s Inductor compile is not amortized across enough"
                "autoregressive-generation calls. Output is byte-identical. Leave off "
                "unless you measure a real win."
            ),
            quality_focus=(
                "formulas_with_text",
                "formula_latex_chars",
                "formulas_with_image",
            ),
        ),
        Sweep(
            name="table_enabled",
            knob="table_enabled",
            title="Table Structure Recovery Stage: On vs Off",
            values=[False, True],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            description=(
                "Master toggle for the table stage. Off: table blocks ship image-only. "
                "On: SLANet+ recovers the cell grid + per-cell text."
            ),
            findings=(
                "Enabling the table stage adds 8.3s (29.9 -> 38.2s, +28%) to recover "
                "36 tables with structured cell grids on this PDF -- about 0.23s per "
                "table on average. VRAM unchanged."
            ),
            quality_focus=(
                "blocks_table",
                "tables_with_cells",
                "tables_image_only",
                "total_cells",
                "cells_with_text",
            ),
        ),
        Sweep(
            name="table_batch_size",
            knob="table_batch_size",
            title="Table Structure Recovery: SLANet+ Batch Size",
            values=[1, 2, 4, 8, 16],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_enable_table,
            description=(
                "Tables per SLANet+ structure forward pass. Quality is the "
                "tables-with-cells count; it should stay constant."
            ),
            findings=(
                "Wall is essentially flat (36.7-39.0s) across bs=1 to bs=16 on this "
                "36-table PDF -- SLANet+ already amortizes the per-table cost. VRAM "
                "grows modestly (6.6 -> 6.9 GiB). Cell count is identical at every "
                "batch size."
            ),
            quality_focus=(
                "tables_with_cells",
                "tables_image_only",
                "total_cells",
                "cells_with_text",
            ),
        ),
        Sweep(
            name="references_enabled",
            knob="references_enabled",
            title="Reference Parsing via External GROBID: On vs Off",
            values=[False, True],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            description=(
                "Master toggle for reference parsing via an external GROBID server. "
                "Off: `Block.reference` stays null. On: parsed citations attached to "
                "every reference block GROBID could parse."
            ),
            findings=(
                "Enabling reference parsing adds 31.6s (29.9 -> 61.5s, +106%) to "
                "attach  structured `Reference` records to all 44 reference blocks via "
                "one batched POST to GROBID. CPU drops during the wait (median 1604% "
                "-> 8% -- pipeline blocked on HTTP). Cost is per-PDF, not per-reference"
            ),
            quality_focus=(
                "blocks_reference",
                "references_parsed",
                "references_total",
            ),
            needs_grobid=True,
        ),
        # ---- scanned sweeps (slower, OCR-dominated) -------------------------
        Sweep(
            name="render_dpi_scanned",
            knob="render_dpi",
            title="Page Rendering: DPI Sweep on Scanned OCR Path",
            values=[100, 150, 200, 250, 300],
            pdf=SCANNED_PDF,
            language="en",
            digital_born=False,
            base=SCANNED_BASE,
            config_factory=_bundle_miss_for_text,
            description=(
                "Render DPI for scanned PDFs. OCR accuracy is sensitive to source "
                "resolution; lower DPI saves render + decode time but loses recall."
            ),
            findings=(
                "OCR character recall rises with DPI: 140.7k chars at 100, 144.2k at "
                "150, plateaus at ~146k from 200 upward. Wall grows 1.14x (105 -> "
                "119s). 100 DPI loses 4% of recall vs 300 -- 200 DPI is the cheapest "
                "setting that captures full recall."
            ),
            quality_focus=("text_chars", "miss", "blocks_text"),
        ),
        Sweep(
            name="ocr_batch_size",
            knob="ocr_batch_size",
            title="Scanned-Path OCR: RapidOCR Recognition Batch Size",
            # Skip the very-small batches (bs∈{1, 2, 8, 32}); the production range is
            # bs[4, 128] and the trend across {4, 6, 16, 64, 128} is enough to see the
            # curve. bs=6 is included specifically because RapidOCR's own docs cite it
            # as the canonical optimum -- a measured row here anchors the config default
            # of 64 against the upstream recommendation.
            values=[4, 6, 16, 64, 128],
            pdf=SCANNED_PDF,
            language="en",
            digital_born=False,
            base=SCANNED_BASE,
            config_factory=_bundle_miss_for_text,
            description=(
                "RapidOCR recognition batch size (`Rec.batch_size`). Larger batches "
                "reduce kernel-launch overhead but trade off latency."
            ),
            findings=(
                "Wall and OCR output are flat (~120s, 146.5k chars, 4 MISS) across "
                "bs=4 to bs=128 -- RapidOCR's recognition batching doesn't shift wall "
                "on this 135-page scanned PDF. Pick any value in this range."
            ),
            quality_focus=("text_chars", "miss"),
        ),
        Sweep(
            name="ocr_use_cuda",
            knob="ocr_use_cuda",
            title="Scanned-Path OCR: CUDA Execution Provider (CPU Omitted)",
            # CPU OCR is ~50x slower than CUDA on the scanned thesis (~1 h wall vs
            # ~2 min). The CPU value is left out of the automated sweep to keep total
            # bench time bounded; rerun with `--force` and a CPU-only value list if you
            # need it.
            values=[True],
            pdf=SCANNED_PDF,
            language="en",
            digital_born=False,
            base=SCANNED_BASE,
            config_factory=_bundle_miss_for_text,
            description=(
                "RapidOCR via ONNXRuntime CUDAExecutionProvider vs CPU. Output is "
                "deterministic across providers; only speed and resources change."
            ),
            findings=(
                "CUDA OCR completes in 114s for the 135-page scanned thesis (146.5k "
                "chars, 4 MISS). CPU OCR exceeded 1h projected wall in earlier testing "
                "and was excluded from the automated suite -- ~50x slowdown vs CUDA on "
                "this corpus."
            ),
            quality_focus=("text_chars", "miss"),
        ),
        Sweep(
            name="ocr_workers",
            knob="ocr_workers",
            title="Scanned-Path OCR: Spawn-Pool Worker Count",
            # {1, 2, 4, 6, 8} covers the strong-scaling curve cleanly on a 24 GiB 3090.
            # 6 is the production default and would otherwise be unmeasured; 2 lives
            # between 1 and 4 where the steepest speedup occurs. 8 saturates the 24 GiB
            # card -- higher counts trigger stochastic ONNXRuntime BFC OOMs and are
            # excluded.
            values=[1, 2, 4, 6, 8],
            pdf=SCANNED_PDF,
            language="en",
            digital_born=False,
            base=SCANNED_BASE,
            config_factory=_bundle_miss_for_text,
            description=(
                "Spawn-pool size for scanned OCR. Each worker owns its own RapidOCR "
                "engine (~2 GiB VRAM); too many saturate the device and trigger BFC "
                "allocator OOMs."
            ),
            findings=(
                "Strong parallel scaling: 8 workers is 4.05x faster than 1 (410 -> "
                "101s) at 3.9x VRAM (6.3 -> 24.6 GiB) and 1.9-> RSS (46 -> 88 GiB). "
                "OCR output is bit-identical. 8 workers consumes ~all of a 24 GiB GPU "
                "-- higher counts would OOM."
            ),
            quality_focus=("text_chars", "miss"),
        ),
        # ---- knob-coverage gaps: cheap digital-born knobs -------------------
        #
        # Tier-2 additions that close coverage gaps on PipelineConfig fields whose
        # current default is faith-based. Each runs against a digital-born PDF unless
        # OCR is the metric, so the per-row wall stays in the 30-90 s band.
        Sweep(
            name="layout_confidence",
            knob="layout_confidence",
            title="Layout Analysis: Detection Confidence Threshold",
            values=[0.3, 0.4, 0.5, 0.6, 0.7],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            description=(
                "PP-DocLayoutV3 detection confidence floor. Lower values surface more "
                "spurious blocks that downstream stages then have to process; higher "
                "values drop genuine but low-confidence detections."
            ),
            quality_focus=(
                "blocks_total",
                "blocks_text",
                "blocks_image",
                "blocks_formula",
                "blocks_table",
            ),
        ),
        Sweep(
            name="render_workers",
            knob="render_workers",
            title="Page Rendering: Process-Pool Worker Count",
            values=[1, 2, 4, 8, 16],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            description=(
                "Process-pool size for the page-render stage. Sibling knob to "
                "`digital_born_workers`; the render pool parallelises pdf_oxide page "
                "rasterisation upstream of layout."
            ),
            quality_focus=("blocks_total", "text_chars"),
        ),
        Sweep(
            name="crop_format",
            knob="crop_format",
            title="Crop Encoding: PNG vs JPEG for Per-Block Images",
            values=["png", "jpeg"],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_enable_formula,
            description=(
                "Per-block crop image format. PNG is lossless; JPEG compresses but "
                "feeds the formula recognizer slightly degraded inputs. Quality is "
                "tracked via `formula_latex_chars` (the recogniser reads the crop) and "
                "`bundle_bytes` (network cost)."
            ),
            quality_focus=(
                "formula_latex_chars",
                "formulas_with_text",
                "bundle_bytes",
            ),
        ),
        Sweep(
            name="bundle_miss_images_for",
            knob="bundle_miss_images_for",
            title="MISS Crop Inclusion: Bundle Size vs Fallback Coverage",
            # Three points span the dial: skip every MISS, keep only formula MISS
            # (the path where consumers most want a fallback image), keep everything
            # (the original behavior).
            values=[
                frozenset(),
                frozenset({"formula"}),
                frozenset({"text", "image", "reference", "formula"}),
            ],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_enable_formula,
            description=(
                "Block types for which a MISS-fallback crop is written to the bundle. "
                "Trades bundle size against downstream consumers' ability to render "
                "a fallback image when extraction failed."
            ),
            quality_focus=("bundle_bytes", "miss", "formulas_with_image"),
        ),
        Sweep(
            name="table_min_side_px",
            knob="table_min_side_px",
            title="Table Stage: Minimum-Side Pixel Filter",
            values=[60, 120, 180, 240],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_enable_table,
            description=(
                "Tables with either dimension below this many pixels skip the SLANet+ "
                "structure model and fall back to image-only. Lower values run more "
                "small detections through structure recovery; higher values skip them."
            ),
            quality_focus=(
                "tables_with_cells",
                "tables_image_only",
                "total_cells",
                "cells_with_text",
            ),
        ),
        # ---- knob-coverage gaps: scanned-path knobs -------------------------
        Sweep(
            name="ocr_min_score",
            knob="ocr_min_score",
            title="Scanned-Path OCR: Recognition Score Floor",
            values=[0.3, 0.4, 0.5, 0.6, 0.7],
            pdf=SCANNED_PDF,
            language="en",
            digital_born=False,
            base=SCANNED_BASE,
            config_factory=_bundle_miss_for_text,
            description=(
                "RapidOCR recognition score floor. Lower values keep low-confidence "
                "characters (more text recovered, more risk of garbage); higher "
                "values discard them (fewer chars, more MISS blocks)."
            ),
            quality_focus=("text_chars", "miss", "blocks_text"),
        ),
        # ---- knob-coverage gaps: multi-knob formula tuning ------------------
        Sweep(
            name="formula_bucket_thresholds",
            # Sentinel; `_apply_bucket_preset` sets four PipelineConfig fields per row.
            knob="bucket_thresholds",
            title="Formula Recognition: Bucket Threshold Presets",
            values=list(_BUCKET_THRESHOLD_PRESETS),
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_apply_bucket_preset,
            description=(
                "Named (small_threshold, medium_threshold, small_tokens, medium_tokens"
                ") presets for the bucketed-batching path. The four knobs co-vary "
                "because shrinking a bucket without raising its `*_tokens` cap "
                "produces truncations. `formula_truncated` is the quality signal."
            ),
            quality_focus=(
                "formulas_with_text",
                "formula_latex_chars",
                "formula_truncated",
            ),
        ),
        Sweep(
            name="dpi_bucket_interaction",
            # Sentinel; `_apply_dpi_bucket_preset` sets render DPI + bucket thresholds.
            knob="dpi_x_buckets",
            title="Render DPI x Bucket Thresholds: Joint Sweep",
            values=list(_DPI_BUCKET_PRESETS),
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_apply_dpi_bucket_preset,
            description=(
                "Joint sweep over digital-born render DPI and bucket thresholds. "
                "Bucket thresholds are bbox-area thresholds in px**2; a 2x DPI scale "
                "is a 4x area scale, so the `scaled_4x` variant matches the higher "
                "render at the same effective crop size."
            ),
            quality_focus=(
                "formulas_with_text",
                "formula_latex_chars",
                "formula_truncated",
            ),
        ),
        # ---- methodology gaps: multi-PDF + repeats --------------------------
        Sweep(
            name="cross_corpus",
            # Sentinel; `per_value_pdf` resolves the actual filename per row.
            knob="pdf",
            title="Cross-Corpus: Production Config Across Six PDFs",
            values=[
                "904599",
                "995802",
                "1005465",
                "084016",
                "101123",
                "823568",
            ],
            pdf=DIGITAL_BORN_PDF,  # only used as a fallback by `referenced_pdfs`
            language="en",
            digital_born=None,  # let the dispatcher auto-detect per PDF
            base=PRODUCTION_HYBRID_BASE,
            config_factory=_enable_all_stages,
            per_value_pdf={
                "904599": "904599.pdf",
                "995802": "995802.pdf",
                "1005465": "1005465.pdf",
                "084016": "084016.pdf",
                "101123": "101123.pdf",
                "823568": "823568.pdf",
            },
            per_value_language={
                "904599": "en",
                "995802": "tr",
                "1005465": "ar",
                "084016": "en",
                "101123": "tr",
                "823568": "ar",
            },
            description=(
                "Production-realistic config (formula + table + references) across "
                "every sample PDF: digital-born + scanned, English + Turkish + "
                "Arabic. Surfaces whether headline numbers from single-PDF sweeps "
                "generalise."
            ),
            quality_focus=(
                "blocks_total",
                "text_chars",
                "formulas_with_text",
                "tables_with_cells",
                "references_parsed",
            ),
            needs_grobid=True,
        ),
        Sweep(
            name="language_matrix",
            # Sentinel; the actual OCR language is resolved via `per_value_language`.
            knob="language",
            title="Scanned-Path OCR: English vs Turkish vs Arabic",
            values=["en", "tr", "ar"],
            pdf=SCANNED_PDF,
            language="en",
            digital_born=False,
            base=SCANNED_BASE,
            config_factory=_enable_all_stages,
            per_value_pdf={
                "en": SCANNED_PDF,
                "tr": SCANNED_TR_PDF,
                "ar": SCANNED_AR_PDF,
            },
            per_value_language={
                "en": "en",
                "tr": "tr",
                "ar": "ar",
            },
            description=(
                "Same scanned-path config across three languages on three different "
                "PDFs. RapidOCR is language-conditioned; this is the only way to see "
                "which scripts need a different `ocr_min_score` or `render_dpi`."
            ),
            quality_focus=("text_chars", "miss", "blocks_text"),
            needs_grobid=True,
        ),
        Sweep(
            name="variance",
            # Sentinel; the factory ignores the value -- all rows share one config.
            knob="repeat",
            title="Noise Floor: Same Config x N Repeats",
            # 10 repeats is large enough that stdev / median is meaningful; small enough
            # that a single sweep stays under ~75 min at production wall.
            values=list(range(1, 11)),
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_enable_all_stages,
            description=(
                "Run the same production-config row N times to characterise the "
                "noise floor. The aggregate stdev / min / max bounds the smallest "
                "wall delta that any other sweep can claim as real signal."
            ),
            quality_focus=("blocks_total", "text_chars", "formulas_with_text"),
            needs_grobid=True,
        ),
        Sweep(
            name="production_realistic_digital_born",
            # Sentinel; `_apply_stages_value` flips formula+table+references together.
            knob="stages",
            title="Production-Realistic: Stages-Off vs All-On (Digital-Born)",
            values=["minimal", "all_on"],
            pdf=DIGITAL_BORN_PDF,
            language="en",
            digital_born=True,
            base=DIGITAL_BORN_BASE,
            config_factory=_apply_stages_value,
            description=(
                "Two-row baseline: the bench's standard `DIGITAL_BORN_BASE` (layout + "
                "text only) vs the production configuration (formula + table + "
                "references all on). Pins the production-config headline wall that "
                "every per-stage sweep would otherwise leave implicit."
            ),
            quality_focus=(
                "blocks_total",
                "text_chars",
                "formulas_with_text",
                "tables_with_cells",
                "references_parsed",
            ),
            needs_grobid=True,
        ),
        Sweep(
            name="production_realistic_scanned",
            knob="stages",
            title="Production-Realistic: Stages-Off vs All-On (Scanned)",
            values=["minimal", "all_on"],
            pdf=SCANNED_PDF,
            language="en",
            digital_born=False,
            base=SCANNED_BASE,
            config_factory=_apply_stages_value,
            description=(
                "Scanned counterpart to `production_realistic_digital_born`. OCR runs "
                "in both rows (the base already has it on); only formula + table + "
                "references flip between minimal and all-on."
            ),
            quality_focus=(
                "blocks_total",
                "text_chars",
                "formulas_with_text",
                "tables_with_cells",
                "references_parsed",
            ),
            needs_grobid=True,
        ),
    ]

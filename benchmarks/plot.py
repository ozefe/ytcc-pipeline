"""Generate one PNG per sweep CSV.

The plot is a 2x3 grid that captures every dimension the bench records:

- Row 1: wall time, per-stage wall (or a sweep-specific replacement such as a
  worker-count speedup curve), single-metric quality.
- Row 2: CPU% / RSS MiB / VRAM MiB -- each with min, median, p95, and max drawn from the
  sampled distribution.

Every panel uses discrete x-positions labelled with the actual swept value, even for
numeric sweeps. That keeps the spacing consistent between sweeps where the values are
`(1, 2, 4, 8, 16)` and sweeps where they are `(False, True)` or `('png', 'jpeg')`.

Quality plots use a smart y-axis: when the metric is nearly flat across the sweep the
chart zooms in on the variation; when it spans a wide range it starts near zero. The
per-sweep Markdown carries the full set of quality columns so collapsing to a single
panel here is just a presentation choice, not a data loss.
"""

import logging
import warnings
from functools import cache
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from matplotlib.axes import Axes

__all__ = ["plot_all", "plot_sweep"]

logger = logging.getLogger("benchmarks.plot")

warnings.filterwarnings("ignore", category=FutureWarning)

_THEME = {
    "context": "talk",
    "style": "whitegrid",
    "palette": "deep",
    "font_scale": 0.7,
}

_STAGE_COLUMNS: tuple[str, ...] = (
    "stage_render_s",
    "stage_layout_s",
    "stage_blocks_s",
    "stage_table_s",
    "stage_formula_s",
    "stage_reference_s",
    "stage_bundle_s",
)

# Per-sweep override for the middle top-row panel. Default is `per_stage`; worker-count
# sweeps benefit more from a speedup-vs-baseline curve.
_PANEL_OVERRIDES: dict[str, str] = {
    "digital_born_workers": "speedup",
    "ocr_workers": "speedup",
}

# Rotation thresholds for x-tick labels. Picked empirically so model ids  fit
# (`PP-FormulaNet-L`) without overlapping their neighbours, and short labels (`8`,
# `True`) stay horizontal.
_XTICK_ROTATE_AT = 4
_XTICK_STEEP_AT = 12

# Speedup-curve panel needs at least two values to draw a delta.
_MIN_SPEEDUP_VALUES = 2

# Quality-axis "smart y-lim" thresholds + bar-annotation magnitudes.
_QUALITY_FLAT_REL_SPAN = 0.05
_FORMAT_K = 1_000
_FORMAT_M = 1_000_000


def plot_all(sweeps_dir: Path, plots_dir: Path) -> None:
    """Plot every `<sweep>.csv` under `sweeps_dir`."""
    sns.set_theme(**_THEME)  # pyright: ignore[reportArgumentType]
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(sweeps_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("no CSVs found under %s", sweeps_dir)
        return

    for csv_path in csv_files:
        plot_path = plots_dir / f"{csv_path.stem}.png"

        try:
            plot_sweep(csv_path, plot_path)
            logger.info("wrote %s", plot_path.relative_to(plots_dir.parent))
        except Exception as exc:  # noqa: BLE001
            # Plot generation is best-effort across the whole suite -- one bad CSV
            # shouldn't take the others down. `exc_info=True` keeps the matplotlib /
            # pandas traceback available so the author can fix the offending sweep
            # without re-running.
            logger.warning(
                "failed to plot %s: %s",
                csv_path.name,
                exc,
                exc_info=True,
            )


def plot_sweep(csv_path: Path, png_path: Path) -> None:
    """Render one PNG for one sweep CSV."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    sns.set_theme(**_THEME)  # pyright: ignore[reportArgumentType]

    name = csv_path.stem
    knob = df["benchmark"].iloc[0] if "benchmark" in df.columns else name

    is_numeric = _values_are_numeric(df["value"])
    ok = df[df["status"] == "ok"].copy()
    if is_numeric and not ok.empty:
        ok["value_num"] = pd.to_numeric(ok["value"], errors="coerce")
        ok = ok.sort_values("value_num").reset_index(drop=True)
    failed = df[df["status"] != "ok"].copy()

    x_pos = list(range(len(ok)))
    x_labels = [_shorten_label(str(v)) for v in ok["value"]]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11.5))

    title = _title_for(name) or name
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)

    # Knob field shown smaller, just under the title -- links the chart to the matching
    # `PipelineConfig` field for anyone reading the plot without the surrounding docs.
    fig.text(
        0.5,
        0.955,
        f"PipelineConfig knob: `{knob}`",
        ha="center",
        va="top",
        fontsize=9,
        color="dimgray",
    )

    description = _description_for(name)
    if description:
        wrapped = _wrap_subtitle(description, max_chars=130)
        fig.text(
            0.5,
            0.935,
            wrapped,
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
        )

    middle_panel = _PANEL_OVERRIDES.get(name, "per_stage")

    _plot_wall(axes[0, 0], ok, failed, x_pos, x_labels)
    if middle_panel == "speedup":
        _plot_speedup(axes[0, 1], ok, x_pos, x_labels)
    else:
        _plot_stages(axes[0, 1], ok, x_pos, x_labels)
    _plot_quality(axes[0, 2], ok, x_pos, x_labels, name)

    _plot_resource(
        axes[1, 0],
        ok,
        x_pos,
        x_labels,
        "cpu_pct",
        "CPU",
        "cores",
        scale=1 / 100,
        subtitle="100% -> 1 core; summed across process tree",
    )
    _plot_resource(
        axes[1, 1],
        ok,
        x_pos,
        x_labels,
        "rss_mb",
        "RSS",
        "MiB",
        subtitle=(
            "resident host memory; summed across tree (shared pages double-counted)"
        ),
    )
    _plot_resource(
        axes[1, 2],
        ok,
        x_pos,
        x_labels,
        "vram_mb",
        "VRAM",
        "MiB",
        subtitle="device-wide GPU memory (NVML)",
    )

    # Top 14% reserved for title + knob-name caption + multi-line findings.
    plt.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _wrap_subtitle(text: str, *, max_chars: int) -> str:
    """Soft-wrap the findings subtitle to roughly fit the figure width."""
    # `textwrap` is stdlib and tiny, but only this helper needs it; keep the import
    # local so importing `plot` for typing stays cheap.
    import textwrap  # noqa: PLC0415

    return "\n".join(textwrap.wrap(text, width=max_chars))


def _shorten_label(value: str) -> str:
    """Compact long x-tick values so they don't overlap on the axis.

    Model ids like `PaddlePaddle/PP-FormulaNet-L_safetensors` collapse to their final
    path component with the `_safetensors` suffix stripped, yielding `PP-FormulaNet-L`.
    Short labels (numbers, booleans, format names) are returned unchanged.
    """
    if "/" in value:
        value = value.rsplit("/", 1)[-1]

    for suffix in ("_safetensors", ".pdf", ".onnx"):
        value = value.removesuffix(suffix)

    return value


def _xtick_rotation(labels: list[str]) -> int:
    """Pick a rotation angle that keeps the longest label readable."""
    longest = max((len(label) for label in labels), default=0)

    if longest > _XTICK_STEEP_AT:
        return 35

    if longest > _XTICK_ROTATE_AT:
        return 20

    return 0


def _plot_wall(
    ax: Axes,
    ok: pd.DataFrame,
    failed: pd.DataFrame,
    x_pos: list[int],
    x_labels: list[str],
) -> None:
    if ok.empty:
        ax.text(
            0.5,
            0.5,
            "all runs failed",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("wall time")
        return

    palette = sns.color_palette("deep")
    ax.plot(
        x_pos,
        ok["wall_s"].to_numpy(),
        marker="o",
        color=palette[0],
        linewidth=2.2,
        markersize=8,
    )
    for x, y in zip(x_pos, ok["wall_s"], strict=True):
        ax.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="dimgray",
        )

    if not failed.empty:
        failure_labels = ", ".join(str(v) for v in failed["value"].tolist())
        ax.text(
            0.02,
            0.97,
            f"failures: {failure_labels}",
            transform=ax.transAxes,
            fontsize=8,
            color="firebrick",
            va="top",
            ha="left",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=_xtick_rotation(x_labels))
    ax.set_ylabel("wall (s)")
    ax.set_xlabel("value")
    ax.set_title("wall time")

    _apply_padding(ax, ok["wall_s"].to_numpy())


def _plot_stages(
    ax: Axes,
    ok: pd.DataFrame,
    x_pos: list[int],
    x_labels: list[str],
) -> None:
    """Stacked-bar of per-stage wall time per value."""
    if ok.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("per-stage wall (s)")
        return

    stage_cols = [c for c in _STAGE_COLUMNS if ok[c].max() > 0]
    if not stage_cols:
        ax.text(
            0.5,
            0.5,
            "no per-stage timings",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("per-stage wall (s)")
        return

    bottoms = pd.Series([0.0] * len(ok), index=ok.index)
    palette = sns.color_palette("deep", n_colors=len(stage_cols))
    for color, col in zip(palette, stage_cols, strict=True):
        ax.bar(
            x_pos,
            ok[col].to_numpy(),
            bottom=bottoms.to_numpy(),
            label=col.removeprefix("stage_").removesuffix("_s"),
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms = bottoms + ok[col].to_numpy()

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=_xtick_rotation(x_labels))
    ax.set_ylabel("wall (s)")
    ax.set_xlabel("value")
    ax.set_title("per-stage wall")

    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)


def _plot_speedup(
    ax: Axes,
    ok: pd.DataFrame,
    x_pos: list[int],
    x_labels: list[str],
) -> None:
    """Speedup vs the slowest run (typically the 1-worker baseline)."""
    if ok.empty or len(ok) < _MIN_SPEEDUP_VALUES:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("speedup vs baseline")
        return

    baseline = float(ok["wall_s"].max())
    speedups = [baseline / float(w) for w in ok["wall_s"]]

    palette = sns.color_palette("deep")
    ax.plot(
        x_pos,
        speedups,
        marker="o",
        color=palette[2],
        linewidth=2.2,
        markersize=8,
    )
    for x, y in zip(x_pos, speedups, strict=True):
        ax.annotate(
            f"{y:.2f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="dimgray",
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=_xtick_rotation(x_labels))
    ax.set_ylabel("speedup")
    ax.set_xlabel("value")
    ax.set_title(f"speedup vs slowest ({baseline:.1f}s)")

    _apply_padding(ax, speedups, floor=1.0)


def _plot_resource(  # noqa: PLR0913 -- all params describe one panel (axes + data + naming + format); a dataclass wrapper would hurt readability
    ax: Axes,
    ok: pd.DataFrame,
    x_pos: list[int],
    x_labels: list[str],
    key: str,
    name: str,
    unit: str,
    *,
    scale: float = 1.0,
    subtitle: str = "",
) -> None:
    """Plot min / median / p95 / max for one resource metric.

    The min-max range is drawn as a shaded band; median is the bold headline line; p95
    is a dashed reference line; min and max are light dotted lines so they sit just
    below the band.

    Args:
        ax: Axes to draw on.
        ok: The status-ok subset of the sweep DataFrame.
        x_pos: Discrete x positions, one per row of `ok`.
        x_labels: Display labels for `x_pos`.
        key: CSV column prefix (e.g. `"cpu_pct"` reads `cpu_pct_min`/`_med`/`_p95`/
            `_max`).
        name: Display name (e.g. `"CPU"`).
        unit: Display unit (e.g. `"cores"`, `"MiB"`).
        scale: Applied to every sample before display -- used to convert raw `cpu_pct`
            (where 100 = one core saturated, summed across the process tree) into
            "cores" via `scale=1/100`.
        subtitle: Optional second line under the panel title, used to disambiguate the
            metric (e.g. that RSS is summed across the process tree and double-counts
            shared pages).
    """
    cols = {stat: f"{key}_{stat}" for stat in ("min", "med", "p95", "max")}
    if ok.empty or not all(c in ok.columns for c in cols.values()):
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"{name} ({unit})")
        return

    palette = sns.color_palette("deep")
    band_color = palette[0]
    p95_color = palette[1]

    series = {stat: ok[cols[stat]].astype(float).to_numpy() * scale for stat in cols}

    ax.fill_between(
        x_pos,
        series["min"],
        series["max"],
        alpha=0.15,
        color=band_color,
        label="min-max band",
    )
    ax.plot(
        x_pos,
        series["min"],
        color=band_color,
        linestyle=":",
        linewidth=1.0,
        marker="v",
        markersize=5,
        alpha=0.7,
        label="min",
    )
    ax.plot(
        x_pos,
        series["max"],
        color=band_color,
        linestyle=":",
        linewidth=1.0,
        marker="^",
        markersize=5,
        alpha=0.7,
        label="max",
    )
    ax.plot(
        x_pos,
        series["p95"],
        color=p95_color,
        linestyle="--",
        linewidth=1.3,
        marker="s",
        markersize=5,
        label="p95",
    )
    ax.plot(
        x_pos,
        series["med"],
        color=band_color,
        linewidth=2.5,
        marker="o",
        markersize=7,
        label="median",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=_xtick_rotation(x_labels))
    ax.set_ylabel(f"{name} ({unit})")
    ax.set_xlabel("value")

    title_main = f"{name} (min / median / p95 / max)"

    if subtitle:
        ax.set_title(f"{title_main}\n{subtitle}", fontsize=10)
    else:
        ax.set_title(title_main)

    ax.legend(loc="best", fontsize=7, ncol=2, frameon=False)

    _apply_padding(ax, [*series["min"], *series["max"]])


def _plot_quality(
    ax: Axes,
    ok: pd.DataFrame,
    x_pos: list[int],
    x_labels: list[str],
    sweep_name: str,
) -> None:
    """Plot the headline quality metric for this sweep with a smart y-axis."""
    if ok.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("quality")
        return

    columns = _quality_columns_for(sweep_name)
    available = [c for c in columns if c in ok.columns and ok[c].notna().any()]
    if not available:
        ax.text(
            0.5,
            0.5,
            "no quality columns",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("quality")
        return

    primary_col = available[0]
    series = pd.to_numeric(ok[primary_col], errors="coerce").astype(float).to_numpy()
    palette = sns.color_palette("deep")
    bars = ax.bar(
        x_pos,
        series,
        color=palette[3],
        edgecolor="white",
        linewidth=0.5,
    )
    for x, y in zip(x_pos, series, strict=True):
        if y > 0:
            ax.annotate(
                _format_quality_value(float(y)),
                (x, float(y)),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
                color="dimgray",
            )

    if len(available) > 1:
        extras = ", ".join(c.removeprefix("quality_") for c in available[1:])
        ax.text(
            0.02,
            0.97,
            f"see CSV / MD for: {extras}",
            transform=ax.transAxes,
            fontsize=7,
            color="gray",
            va="top",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=_xtick_rotation(x_labels))
    ax.set_ylabel(primary_col.removeprefix("quality_"))
    ax.set_xlabel("value")
    ax.set_title(f"quality: {primary_col.removeprefix('quality_')}")

    _apply_smart_quality_ylim(ax, series, bars)


def _apply_smart_quality_ylim(
    ax: Axes,
    values: Iterable[float],
    bars: Any,  # noqa: ANN401  -- matplotlib BarContainer is a thin iterable; not statically typed
) -> None:
    """Pick a y-range that highlights the variation.

    If the values barely move (relative span < 5% of the mean) we zoom in so the
    differences are visible. Otherwise we use a small padding above the max so the
    labels fit, with a floor at 0 when the data is non-negative.
    """
    arr = [v for v in values if pd.notna(v)]
    if not arr:
        return

    val_min = min(arr)
    val_max = max(arr)
    span = val_max - val_min

    if span == 0:
        # All identical. Zoom around the value so the bar reaches the top of the panel
        # and the annotation sits visibly above.
        if val_min == 0:
            ax.set_ylim(-0.1, 1)
        else:
            magnitude = abs(val_min)
            pad = max(magnitude * 0.05, 1)
            ax.set_ylim(val_min - pad, val_min + pad)

            # Re-anchor bars from the zoomed floor so the visual bars don't shrink to
            # nothing; matplotlib will render them as full-height bars rising from the
            # new y-bottom.
            ylow, _ = ax.get_ylim()
            for bar in bars:
                bar.set_y(ylow)
                bar.set_height(val_min - ylow)
        return

    mean = sum(arr) / len(arr)
    relative_span = span / abs(mean) if mean else float("inf")
    pad = span * 0.18

    if relative_span < _QUALITY_FLAT_REL_SPAN and val_min > 0:
        # Variation is small relative to magnitude -- zoom in on the band.
        zoom_low = val_min - pad
        ax.set_ylim(zoom_low, val_max + pad)

        # Re-anchor each bar to the zoomed floor; total bar height = data - floor.
        for bar, value in zip(bars, arr, strict=False):
            bar.set_y(zoom_low)
            bar.set_height(float(value) - zoom_low)

        return

    floor = 0 if val_min >= 0 else val_min - pad
    ax.set_ylim(floor, val_max + pad)


def _apply_padding(
    ax: Axes,
    values: Iterable[float],
    *,
    floor: float | None = None,
) -> None:
    """Pad the y-axis so markers and labels don't crash against the edge."""
    arr = [float(v) for v in values if pd.notna(v)]
    if not arr:
        return

    val_min = min(arr)
    val_max = max(arr)
    span = val_max - val_min
    pad = max(span * 0.12, abs(val_max) * 0.02, 0.1)

    if floor is not None:
        ax.set_ylim(min(floor, val_min - pad), val_max + pad)
    else:
        ax.set_ylim(val_min - pad, val_max + pad)


def _format_quality_value(value: float) -> str:
    """Compact label for bar annotations (e.g. 178612 -> 178.6k)."""
    if abs(value) >= _FORMAT_M:
        return f"{value / _FORMAT_M:.1f}M"

    if abs(value) >= _FORMAT_K:
        return f"{value / _FORMAT_K:.1f}k"

    if value == int(value):
        return str(int(value))

    return f"{value:.2f}"


def _values_are_numeric(values: Iterable[object]) -> bool:
    """Treat the value column as numeric if it's pure numbers.

    Booleans round-trip from CSV as `True`/`False` and `pd.to_numeric` silently coerces
    them to 1/0 -- we want them shown as categorical labels, not a 0 -> 1 axis. Strings
    (model ids, formats) likewise stay categorical.
    """
    series = pd.Series(list(values))
    if series.dtype == bool:
        return False

    if series.dtype == object and any(isinstance(v, bool) for v in series):
        return False

    numeric = pd.to_numeric(series, errors="coerce")
    return bool(numeric.notna().all() and len(numeric) > 0)


def _quality_columns_for(name: str) -> list[str]:
    """Return the quality_* columns this sweep cares about (priority order)."""
    mapping: dict[str, list[str]] = {
        "render_dpi_scanned": [
            "quality_text_chars",
            "quality_blocks_text",
            "quality_miss",
        ],
        "render_dpi_digital_born": [
            "quality_text_chars",
            "quality_blocks_text",
            "quality_blocks_total",
        ],
        "page_format": [
            "quality_blocks_total",
            "quality_text_chars",
        ],
        "jpeg_quality": [
            "quality_text_chars",
            "quality_miss",
        ],
        "layout_batch_size": ["quality_blocks_total"],
        "layout_fp16": ["quality_blocks_total"],
        "layout_fast_preproc": ["quality_blocks_total"],
        "ocr_batch_size": [
            "quality_text_chars",
            "quality_miss",
        ],
        "ocr_use_cuda": [
            "quality_text_chars",
            "quality_miss",
        ],
        "ocr_workers": [
            "quality_text_chars",
            "quality_miss",
        ],
        "digital_born_workers": ["quality_text_chars"],
        "formula_enabled": [
            "quality_formulas_with_text",
            "quality_formula_latex_chars",
            "quality_formulas_with_image",
        ],
        "formula_model_id": [
            "quality_formulas_with_text",
            "quality_formula_latex_chars",
        ],
        "formula_dtype": [
            "quality_formulas_with_text",
            "quality_formula_latex_chars",
        ],
        "formula_batch_size": [
            "quality_formulas_with_text",
            "quality_formula_latex_chars",
        ],
        "formula_bucketed": [
            "quality_formulas_with_text",
            "quality_formula_latex_chars",
        ],
        "formula_torch_compile": [
            "quality_formulas_with_text",
            "quality_formula_latex_chars",
        ],
        "table_enabled": [
            "quality_tables_with_cells",
            "quality_cells_with_text",
            "quality_total_cells",
            "quality_tables_image_only",
        ],
        "table_batch_size": [
            "quality_tables_with_cells",
            "quality_cells_with_text",
        ],
        "references_enabled": [
            "quality_references_parsed",
            "quality_references_total",
        ],
        "layout_confidence": [
            "quality_blocks_total",
            "quality_blocks_text",
            "quality_blocks_formula",
        ],
        "render_workers": [
            "quality_blocks_total",
            "quality_text_chars",
        ],
        "crop_format": [
            "quality_formula_latex_chars",
            "quality_bundle_bytes",
        ],
        "bundle_miss_images_for": [
            "quality_bundle_bytes",
            "quality_miss",
            "quality_formulas_with_image",
        ],
        "table_min_side_px": [
            "quality_tables_with_cells",
            "quality_tables_image_only",
            "quality_total_cells",
        ],
        "ocr_min_score": [
            "quality_text_chars",
            "quality_miss",
        ],
        "formula_bucket_thresholds": [
            "quality_formula_truncated",
            "quality_formula_latex_chars",
            "quality_formulas_with_text",
        ],
        "dpi_bucket_interaction": [
            "quality_formula_truncated",
            "quality_formula_latex_chars",
            "quality_formulas_with_text",
        ],
        "cross_corpus": [
            "quality_text_chars",
            "quality_formulas_with_text",
            "quality_tables_with_cells",
            "quality_references_parsed",
        ],
        "language_matrix": [
            "quality_text_chars",
            "quality_miss",
        ],
        "variance": [
            "quality_text_chars",
            "quality_formulas_with_text",
        ],
        "production_realistic_digital_born": [
            "quality_text_chars",
            "quality_formulas_with_text",
            "quality_tables_with_cells",
            "quality_references_parsed",
        ],
        "production_realistic_scanned": [
            "quality_text_chars",
            "quality_formulas_with_text",
            "quality_tables_with_cells",
            "quality_references_parsed",
        ],
        # Out-of-band scripts that write sweep-shaped CSVs (`cold_start`,
        # `sustained_load`, `api_concurrency`, `grobid_payload_scaling`,
        # `torch_compile_amortization`). Map them to the quality column they care about
        # so `plot_all` doesn't fall back to `quality_blocks_total` (often absent or
        # zero in those CSVs).
        "cold_start": ["quality_vram_after_init_mb"],
        "sustained_load": ["quality_vram_after_call_mb"],
        "api_concurrency": ["quality_throughput_pdfs_per_min"],
        "grobid_payload_scaling": ["quality_grobid_wall_per_ref_ms"],
        "torch_compile_amortization": ["quality_cumulative_wall_s"],
    }
    return mapping.get(name, ["quality_blocks_total"])


@cache
def _load_meta() -> dict[str, tuple[str, str]]:
    """Lazy-load each sweep's `(title, subtitle)` pair from `benchmarks.sweeps`.

    The subtitle is the sweep's `findings` field (what the data shows), falling back to
    `description` (what the knob is) when findings aren't populated yet. Imported lazily
    so `benchmarks.plot` stays cheap to import for callers that only need plotting
    helpers. `functools.cache` memoises the result so the sweep module is imported at
    most once per process.
    """
    # Local import keeps top-level `import benchmarks.plot` cheap (loading
    # `benchmarks.sweeps` pulls in PipelineConfig and every sweep config).
    try:
        from benchmarks.sweeps import all_sweeps  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001  -- plotting must keep working even if sweep configs fail to import
        logger.debug("could not load sweep metadata: %s", exc)
        return {}

    return {s.name: (s.title, s.findings or s.description) for s in all_sweeps()}


def _title_for(name: str) -> str:
    """Look up the human-readable plot title for a sweep."""
    return _load_meta().get(name, ("", ""))[0]


def _description_for(name: str) -> str:
    """Look up the plot subtitle (findings) for a sweep."""
    return _load_meta().get(name, ("", ""))[1]

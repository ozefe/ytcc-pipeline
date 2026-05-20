"""CSV row writer + per-sweep Markdown summary.

Each sweep emits `<name>.csv` and `<name>.md` under `benchmarks/results/sweeps/`. The
CSV is the source of truth; the Markdown is a human-readable consequence.

The summary tables drop the slowest/fastest, identify any OOMs or errors, and surface
the most operator-relevant quality metric for each sweep -- defined per-sweep when
relevant, defaulting to total block count.
"""

import csv
import statistics
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .runner import BenchmarkResult, ResourceStats

__all__ = [
    "QUALITY_KEYS",
    "RESOURCE_KEYS",
    "STAGE_KEYS",
    "row_dict",
    "write_csv",
    "write_markdown",
]

# Thresholds for the per-sweep Markdown header. 2 rows = report a fastest vs. slowest
# spread; 3+ rows = include the min/median/max "Notes" line.
_MIN_ROWS_FOR_SPREAD = 2
_MIN_ROWS_FOR_NOTES = 3

STAGE_KEYS: tuple[str, ...] = (
    "render",
    "layout",
    "blocks",
    "table",
    "formula",
    "reference",
    "bundle",
)
RESOURCE_KEYS: tuple[str, ...] = (
    "cpu_pct",
    "rss_mb",
    "vram_mb",
)
QUALITY_KEYS: tuple[str, ...] = (
    "blocks_total",
    "blocks_text",
    "blocks_image",
    "blocks_reference",
    "blocks_formula",
    "blocks_table",
    "text_chars",
    "miss",
    "formulas_with_text",
    "formulas_with_image",
    "formula_latex_chars",
    "formula_truncated",
    "tables_with_cells",
    "tables_image_only",
    "total_cells",
    "cells_with_text",
    "references_parsed",
    "references_total",
    "bundle_bytes",
)


def row_dict(r: BenchmarkResult) -> dict[str, Any]:
    """Flatten a `BenchmarkResult` into a single CSV-ready dict."""
    row: dict[str, Any] = {
        "benchmark": r.benchmark,
        "value": r.value,
        "pdf": r.pdf,
        "language": r.language,
        "status": r.status,
        "wall_s": round(r.wall_s, 3),
    }

    for stage in STAGE_KEYS:
        row[f"stage_{stage}_s"] = round(r.stages.get(stage, 0.0), 3)

    for key, stats in (
        ("cpu_pct", r.cpu_pct),
        ("rss_mb", r.rss_mb),
        ("vram_mb", r.vram_mb),
    ):
        _add_resource(row, key, stats)

    for k in QUALITY_KEYS:
        row[f"quality_{k}"] = r.quality.get(k, "")

    row["error"] = r.error
    return row


def write_csv(path: Path, results: Sequence[BenchmarkResult]) -> None:
    """Write one CSV per sweep; called incrementally between runs."""
    rows = [row_dict(r) for r in results]
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(  # noqa: PLR0913, PLR0915  -- single template that emits the per-sweep page; splitting hurts readability
    path: Path,
    *,
    name: str,
    knob: str,
    description: str,
    pdf: str,
    results: Sequence[BenchmarkResult],
    quality_focus: Sequence[str] = ("blocks_total", "text_chars", "miss"),
) -> None:
    """Write a per-sweep Markdown summary alongside the CSV.

    Args:
        path: Destination `.md` file.
        name: Sweep name (used in the title).
        knob: The `PipelineConfig` field swept.
        description: One-paragraph context shown under the heading.
        pdf: PDF used for the sweep.
        results: Rows in sweep order.
        quality_focus: Quality keys highlighted in the comparison table. Each key reads
            from `r.quality[key]`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = [r for r in results if r.status == "ok"]
    failures = [r for r in results if r.status != "ok"]
    fastest = min(ok, key=lambda r: r.wall_s, default=None)
    slowest = max(ok, key=lambda r: r.wall_s, default=None)

    lines: list[str] = []
    lines.append(f"# {name}")
    lines.append("")
    lines.append(description)
    lines.append("")
    lines.append(f"- **Knob:** `{knob}`")
    lines.append(f"- **PDF:** `{pdf}`")
    lines.append(f"- **Runs:** {len(results)} ({len(ok)} ok, {len(failures)} failed)")

    if fastest is not None and slowest is not None and len(ok) >= _MIN_ROWS_FOR_SPREAD:
        lines.append(
            f"- **Fastest:** `{fastest.value}` @ {fastest.wall_s:.2f}s -- "
            f"**Slowest:** `{slowest.value}` @ {slowest.wall_s:.2f}s "
            f"(**{slowest.wall_s / max(fastest.wall_s, 1e-9):.2f}x** spread)",
        )
    lines.append("")

    lines.append("## Speed")
    lines.append("")
    header = ["value", "status", "wall (s)"] + [f"{s} (s)" for s in STAGE_KEYS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in results:
        cells = [str(r.value), r.status, f"{r.wall_s:.2f}"]
        for s in STAGE_KEYS:
            v = r.stages.get(s, 0.0)
            cells.append(f"{v:.2f}" if v else "--")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Resources (min / median / p95 / max)")
    lines.append("")
    lines.append(
        "| value | CPU% (min/med/p95/max) | "
        "RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |",
    )
    lines.append("|---|---|---|---|")
    lines.extend(
        f"| {r.value} | {_format_stats(r.cpu_pct)} | "
        f"{_format_stats(r.rss_mb)} | {_format_stats(r.vram_mb)} |"
        for r in results
    )
    lines.append("")

    if quality_focus and any(r.quality for r in ok):
        lines.append("## Quality")
        lines.append("")
        header = ["value", *list(quality_focus)]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in results:
            cells = [str(r.value)]
            for k in quality_focus:
                v = r.quality.get(k, "")
                cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if failures:
        lines.append("## Failures")
        lines.append("")
        lines.extend(
            f"- `{r.value}` -- **{r.status}** -- `{r.error[:200]}`" for r in failures
        )
        lines.append("")

    if len(ok) >= _MIN_ROWS_FOR_NOTES:
        lines.append("## Notes")
        lines.append("")
        wall_values = [r.wall_s for r in ok]
        median_s = statistics.median(wall_values)
        lines.append(
            f"- Wall time across {len(ok)} successful runs: "
            f"min={min(wall_values):.2f}s, median={median_s:.2f}s, "
            f"max={max(wall_values):.2f}s.",
        )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _add_resource(row: dict[str, Any], key: str, stats: ResourceStats | None) -> None:
    if stats is None:
        empty = {f"{key}_min": "", f"{key}_med": "", f"{key}_p95": "", f"{key}_max": ""}
        row.update(empty)
        return

    row[f"{key}_min"] = stats.min_
    row[f"{key}_med"] = stats.median
    row[f"{key}_p95"] = stats.p95
    row[f"{key}_max"] = stats.max_


def _format_stats(stats: ResourceStats | None) -> str:
    if stats is None or stats.samples == 0:
        return "--"

    return f"{stats.min_:.0f} / {stats.median:.0f} / {stats.p95:.0f} / {stats.max_:.0f}"


# Re-export for type hints elsewhere.
_ = asdict  # silence import linter if unused

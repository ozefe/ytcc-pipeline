"""Run every defined sweep and write CSV + Markdown + plots.

Per-sweep output lands under `benchmarks/results/sweeps/<name>.{csv,md}` and
`benchmarks/results/plots/<name>.png`. An aggregate `benchmarks/results/summary.md` ties
it all together.

Each sweep:

1. Skips if its CSV already exists (resume-friendly); pass `--force` to overwrite.
2. Performs a warm-up run on the first value (no measurement) so HF / CUDA caches are
   loaded before any timed row.
3. Iterates the values in order. Each row is OOM-safe and logged with the row's status +
   wall.
4. Writes its CSV incrementally so an aborted run never loses earlier rows.
5. Writes the Markdown summary at the end of the sweep.

CLI:
    python benchmarks/run_all.py                 # run every sweep
    python benchmarks/run_all.py --only formula  # only sweeps containing 'formula'
    python benchmarks/run_all.py --force         # ignore cached CSVs
    python benchmarks/run_all.py --list          # print sweep ids and exit
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Some dev hosts mount a read-only `/workspace/.hf_home`; redirect to the user's local
# HF cache before any `transformers` import resolves. We override unconditionally
# because the upstream value might point at a read-only path even when set.
os.environ["HF_HOME"] = str(Path("~/.cache/huggingface").expanduser())

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Imports below sit AFTER the `sys.path.insert` above so `benchmarks` is  resolvable
# when this script runs directly (`python benchmarks/run_all.py`).
from typing import TYPE_CHECKING  # noqa: E402

from benchmarks.reporting import write_csv, write_markdown  # noqa: E402
from benchmarks.runner import (  # noqa: E402
    BenchmarkResult,
    run_one,
    warmup_log_suppression,
)
from benchmarks.sweeps import Sweep, all_sweeps  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger("benchmarks.run_all")


def main(argv: list[str] | None = None) -> int:
    """Run every defined sweep (or a filtered subset) and write outputs."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )
    logging.getLogger("pdf_oxide").setLevel(logging.WARNING)
    logging.getLogger("ytcc_pipeline").setLevel(logging.INFO)

    if args.list:
        for sweep in all_sweeps():
            print(f"{sweep.name:30s}  {sweep.knob:30s}  values={list(sweep.values)}")

        return 0

    pdfs_dir = (_ROOT / "samples").resolve()
    results_root = (_ROOT / "benchmarks" / "results").resolve()
    sweeps_dir = results_root / "sweeps"
    plots_dir = results_root / "plots"
    bundles_dir = results_root / "bundles"
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir.mkdir(parents=True, exist_ok=True)

    selected = _filter_sweeps(all_sweeps(), args.only)
    if not selected:
        logger.error("no sweeps match filter %r", args.only)
        return 1

    grobid_alive = _grobid_alive()
    grand_t0 = time.perf_counter()
    completed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for sweep in selected:
        csv_path = sweeps_dir / f"{sweep.name}.csv"
        if csv_path.is_file() and not args.force:
            logger.info("[%s] cached CSV found; skipping", sweep.name)
            skipped.append(sweep.name)

            continue

        if sweep.needs_grobid and not grobid_alive:
            logger.warning(
                "[%s] needs GROBID at http://localhost:8070 -- not reachable; "
                "skipping (start the server and re-run if you want this sweep)",
                sweep.name,
            )
            skipped.append(sweep.name)

            continue

        missing_pdfs = [
            pdf
            for pdf in sorted(sweep.referenced_pdfs())
            if not (pdfs_dir / pdf).is_file()
        ]
        if missing_pdfs:
            logger.warning(
                "[%s] missing PDFs %s under %s; skipping",
                sweep.name,
                missing_pdfs,
                pdfs_dir,
            )
            skipped.append(sweep.name)

            continue

        try:
            _run_sweep(
                sweep=sweep,
                pdfs_dir=pdfs_dir,
                csv_path=csv_path,
                md_path=sweeps_dir / f"{sweep.name}.md",
                bundles_dir=bundles_dir,
            )
            completed.append(sweep.name)
        except Exception:
            # `logger.exception` includes the traceback; the sweep name pins which one
            # crashed in case the log gets concatenated.
            logger.exception("[%s] sweep crashed", sweep.name)
            failed.append(sweep.name)

    elapsed = time.perf_counter() - grand_t0
    logger.info(
        "benchmark suite done in %.1fs -- completed=%d skipped=%d failed=%d",
        elapsed,
        len(completed),
        len(skipped),
        len(failed),
    )

    if args.plot or args.only is None:
        # Local import: matplotlib + pandas take ~1s to import; defer until we know
        # plots are needed so `--list` and cached-skip paths stay snappy.
        from benchmarks.plot import plot_all  # noqa: PLC0415

        plot_all(sweeps_dir, plots_dir)

    _write_summary(
        results_root / "summary.md",
        completed=completed,
        skipped=skipped,
        failed=failed,
        total_s=elapsed,
    )
    return 0 if not failed else 2


def _run_sweep(
    *,
    sweep: Sweep,
    pdfs_dir: Path,
    csv_path: Path,
    md_path: Path,
    bundles_dir: Path,
) -> None:
    logger.info(
        "[%s] start -- knob=%s values=%s pdfs=%s",
        sweep.name,
        sweep.knob,
        list(sweep.values),
        sorted(sweep.referenced_pdfs()),
    )
    started = time.perf_counter()
    results: list[BenchmarkResult] = []

    # Warm-up: first value, no measurement, log noise muted. The warmup uses the value's
    # own PDF + language so spawn-pool / model caches reflect the actual first row.
    first_value = sweep.values[0]
    warmup_cfg = sweep.build_config(first_value)
    warmup_pdf = pdfs_dir / sweep.resolve_pdf(first_value)
    logger.info("[%s] warmup with %s=%r", sweep.name, sweep.knob, first_value)
    try:
        with warmup_log_suppression():
            run_one(
                warmup_cfg,
                warmup_pdf,
                benchmark=f"{sweep.name}_warmup",
                value=first_value,
                language=sweep.resolve_language(first_value),
                digital_born=sweep.digital_born,
                output_dir=bundles_dir,
            )
    except Exception as exc:  # noqa: BLE001
        # Warmup failures are recoverable (the measured rows still run)
        #
        # `exc_info=True` keeps the cause (typically CUDA init / model load) visible
        # without escalating to ERROR.
        logger.warning(
            "[%s] warmup raised %r; continuing with measured run",
            sweep.name,
            exc,
            exc_info=True,
        )

    for value in sweep.values:
        cfg = sweep.build_config(value)
        pdf_path = pdfs_dir / sweep.resolve_pdf(value)
        language = sweep.resolve_language(value)
        logger.info("[%s] %s=%r ...", sweep.name, sweep.knob, value)
        result = run_one(
            cfg,
            pdf_path,
            benchmark=sweep.name,
            value=value,
            language=language,
            digital_born=sweep.digital_born,
            output_dir=bundles_dir,
        )
        results.append(result)

        logger.info(
            "[%s] %s=%r -> status=%s wall=%.2fs",
            sweep.name,
            sweep.knob,
            value,
            result.status,
            result.wall_s,
        )

        # Incremental CSV write so a crash never loses earlier rows.
        write_csv(csv_path, results)

    # Sweeps that span multiple PDFs (`cross_corpus`, `language_matrix`) report the
    # whole set so readers can see which row used which document.
    pdfs_display = ", ".join(sorted(sweep.referenced_pdfs()))
    write_markdown(
        md_path,
        name=sweep.name,
        knob=sweep.knob,
        description=sweep.description,
        pdf=pdfs_display,
        results=results,
        quality_focus=sweep.quality_focus,
    )
    logger.info(
        "[%s] done -- %d rows in %.1fs",
        sweep.name,
        len(results),
        time.perf_counter() - started,
    )


def _filter_sweeps(sweeps: Iterable[Sweep], pattern: str | None) -> list[Sweep]:
    if pattern is None:
        return list(sweeps)

    return [s for s in sweeps if pattern in s.name]


def _grobid_alive() -> bool:
    # Lazy import: the runner script is a CLI; importing the library at top level slows
    # `--list` for no benefit. PLC0415 is intentional.
    try:
        from ytcc_pipeline.models.grobid import is_grobid_alive  # noqa: PLC0415
    except Exception:  # noqa: BLE001 -- swallow any import failure; the probe just returns False
        return False

    return is_grobid_alive("http://localhost:8070", timeout_s=1.0)


def _write_summary(
    path: Path,
    *,
    completed: list[str],
    skipped: list[str],
    failed: list[str],
    total_s: float,
) -> None:
    lines: list[str] = [
        "# Benchmark suite summary",
        "",
        f"Total wall: **{total_s:.1f} s** ({total_s / 60:.1f} min) across "
        f"{len(completed) + len(skipped) + len(failed)} sweeps.",
        "",
        f"- Completed: {len(completed)}",
        f"- Skipped:   {len(skipped)} (cached CSV, missing PDF, or GROBID unreachable)",
        f"- Failed:    {len(failed)}",
        "",
        "## Reports",
        "",
        "| sweep | csv | md | plot |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {name} "
        f"| [`{name}.csv`](sweeps/{name}.csv) "
        f"| [`{name}.md`](sweeps/{name}.md) "
        f"| [`{name}.png`](plots/{name}.png) |"
        for name in sorted(completed + skipped)
    )

    if failed:
        lines.append("")
        lines.append("## Crashes")
        lines.append("")
        lines.extend(f"- `{name}` -- see service log" for name in failed)

    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote summary to %s", path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the CLI flags."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="run only sweeps whose name contains this substring",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run sweeps even if their CSV exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print sweep ids and exit",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="generate plots after the runs",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

"""torch.compile amortisation benchmark.

The existing `formula_torch_compile` sweep runs N=1 PDFs and concludes compile is a
wash (the Inductor warmup isn't amortised). This bench answers the follow-up: how many
PDFs do you need to run in a single process before `torch_compile=True` overtakes
`False`?

Two arms (compile on / off), one process per arm to keep CUDA state independent. Each
arm runs N=25 sequential `process_pdf` calls and records:

- Cumulative wall after each call.
- Per-call wall (delta).

The crossover point -- the smallest N where cumulative `compile=True` wall is less than
cumulative `compile=False` wall -- is the break-even PDF count. Below it, compile is
a wash or a loss; at and above it, compile is a win.

Writes:

    benchmarks/results/sweeps/torch_compile_amortization.csv   (one row per arm call)
    benchmarks/results/sweeps/torch_compile_amortization.md    (per-arm + crossover)

Run:

    python -m benchmarks.torch_compile_amortization              # default: N=25
    python -m benchmarks.torch_compile_amortization --calls 10
    python -m benchmarks.torch_compile_amortization --arms compile  # one arm only
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", str(Path("~/.cache/huggingface").expanduser()))

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger("benchmarks.torch_compile_amortization")

_RESULTS_DIR = _ROOT / "benchmarks" / "results" / "sweeps"
_BUNDLES_DIR = _ROOT / "benchmarks" / "results" / "bundles"
_DEFAULT_PDF = "904599.pdf"
_DEFAULT_CALLS = 25


def main(argv: list[str] | None = None) -> int:
    """Run both arms, splice the rows, and write the cross-arm summary."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    pdf_path = _ROOT / "samples" / args.pdf
    if not pdf_path.is_file():
        logger.error("missing PDF: %s", pdf_path)
        return 1

    arms = _resolve_arms(args.arms)
    rows: list[dict[str, Any]] = []
    for arm in arms:
        logger.info("[torch_compile_amortization] arm=%s calls=%d", arm, args.calls)
        rows.extend(_run_arm_subprocess(arm, pdf_path, args.calls))

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RESULTS_DIR / "torch_compile_amortization.csv"
    md_path = _RESULTS_DIR / "torch_compile_amortization.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    logger.info("wrote %s and %s", csv_path, md_path)
    return 0


def _resolve_arms(arms: str) -> list[str]:
    """Map the `--arms` flag to a list of arm names."""
    if arms == "both":
        return ["eager", "compile"]
    if arms in {"eager", "compile"}:
        return [arms]

    msg = f"unknown arms value: {arms!r}"
    raise ValueError(msg)


def _run_arm_subprocess(arm: str, pdf_path: Path, calls: int) -> list[dict[str, Any]]:
    """Run one arm in a fresh Python process; parse its `ROW:` lines into dicts.

    Fresh process per arm because the formula recogniser holds compiled kernels for the
    lifetime of the interpreter; toggling the flag mid-run would just rebuild on top of
    a hot cache.
    """
    # S603 is a misfire: argv is a fixed list, no shell, no user input.
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "benchmarks.torch_compile_amortization",
            "--child",
            "--arm",
            arm,
            "--pdf",
            pdf_path.name,
            "--calls",
            str(calls),
        ],
        check=False,
        capture_output=True,
        env=os.environ,
        text=True,
    )

    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.startswith("ROW:"):
            continue

        try:
            rows.append(json.loads(line.removeprefix("ROW:")))
        except json.JSONDecodeError:
            logger.warning("child emitted unparseable row: %r", line)

    if not rows:
        logger.warning(
            "child for arm=%s emitted no rows; stderr=%s",
            arm,
            completed.stderr,
        )

    return rows


def _child_loop() -> int:
    """Inner loop: build a single config, run N sequential PDFs, emit one row each."""
    # Late import so `_run_arm_subprocess` setting `os.environ` in the parent is
    # respected; torch initialises here, in the child.
    from ytcc_pipeline import PipelineConfig, process_pdf  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--calls", type=int, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    pdf_path = _ROOT / "samples" / args.pdf
    _BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        formula_enabled=True,
        table_enabled=False,
        references_enabled=False,
        digital_born_workers=16,
        layout_fp16=True,
        layout_fast_preproc=True,
        formula_torch_compile=(args.arm == "compile"),
    )

    cumulative_wall_s = 0.0
    for call_i in range(1, args.calls + 1):
        bundle_path = _BUNDLES_DIR / f"compile_amort_{args.arm}_{call_i}.tar"
        t0 = time.perf_counter()
        status = "ok"
        error = ""
        try:
            process_pdf(
                pdf_path,
                language="en",
                digital_born=True,
                output_path=bundle_path,
                config=cfg,
            )
        except Exception as exc:  # noqa: BLE001  -- one bad call records error and keeps going
            status = "error"
            error = repr(exc)

        wall_s = time.perf_counter() - t0
        cumulative_wall_s += wall_s

        row = {
            "benchmark": "torch_compile_amortization",
            "value": f"{args.arm}_{call_i:02d}",
            "pdf": args.pdf,
            "language": "en",
            "status": status,
            "wall_s": round(wall_s, 3),
            "quality_arm": args.arm,
            "quality_call_index": call_i,
            "quality_cumulative_wall_s": round(cumulative_wall_s, 3),
            "error": error,
        }
        print(f"ROW:{json.dumps(row)}", flush=True)

        bundle_path.unlink(missing_ok=True)

    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the torch_compile_amortization CSV."""
    fieldnames = [
        "benchmark",
        "value",
        "pdf",
        "language",
        "status",
        "wall_s",
        "quality_arm",
        "quality_call_index",
        "quality_cumulative_wall_s",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    """Render a per-arm + crossover analysis."""
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(r["quality_arm"], []).append(r)

    lines: list[str] = [
        "# torch_compile_amortization",
        "",
        "Per-call wall and cumulative wall across two arms (`eager` vs `compile`). "
        "The Inductor warmup adds latency to call #1 under `compile`; subsequent "
        "calls should be faster. Crossover N is the smallest call index where "
        "cumulative compile-arm wall is lower than cumulative eager-arm wall.",
        "",
        f"- **Arms:** {', '.join(sorted(by_arm))}",
        f"- **Calls per arm:** "
        f"{max(len(rs) for rs in by_arm.values()) if by_arm else 0}",
        "",
    ]

    crossover = _find_crossover(by_arm)
    if crossover is not None:
        lines.append(
            f"## Crossover: N={crossover} -- `compile` overtakes `eager` here.\n",
        )
    elif by_arm:
        lines.append(
            "## Crossover: not reached within the measured call count.\n",
        )

    for arm, arm_rows in sorted(by_arm.items()):
        lines.append(f"## Arm: `{arm}`\n")
        lines.append("| call | wall (s) | cumulative (s) | status |")
        lines.append("|---|---|---|---|")
        lines.extend(
            f"| {r['quality_call_index']} | {r['wall_s']} | "
            f"{r['quality_cumulative_wall_s']} | {r['status']} |"
            for r in arm_rows
        )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _find_crossover(by_arm: dict[str, list[dict[str, Any]]]) -> int | None:
    """Return the smallest call index where compile-cumulative beats eager-cumulative.

    Returns `None` if either arm is empty or compile never overtakes eager within the
    measured calls.
    """
    eager = by_arm.get("eager")
    compile_ = by_arm.get("compile")
    if not eager or not compile_:
        return None

    eager_by_idx = {
        r["quality_call_index"]: float(r["quality_cumulative_wall_s"]) for r in eager
    }
    for r in compile_:
        idx = r["quality_call_index"]
        if idx not in eager_by_idx:
            continue

        if float(r["quality_cumulative_wall_s"]) < eager_by_idx[idx]:
            return idx

    return None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI flags for the parent process."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument("--calls", type=int, default=_DEFAULT_CALLS)
    parser.add_argument("--pdf", default=_DEFAULT_PDF)
    parser.add_argument(
        "--arms",
        choices=("eager", "compile", "both"),
        default="both",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    if "--child" in sys.argv:
        raise SystemExit(_child_loop())

    raise SystemExit(main())

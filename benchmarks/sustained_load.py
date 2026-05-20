"""Sustained-load benchmark: N sequential `process_pdf` calls in one process.

Production hosts run the FastAPI service for hours/days. The single-row sweeps don't
exercise the CUDA caching allocator past the first call. This bench loops the same PDF
through `process_pdf` N times in one Python process and records VRAM after each call to
characterise allocator drift.

Two arms, controlled by `--expandable / --no-expandable` (default: both):

- `expandable_segments=False`: legacy block-allocator behavior; fragmentation builds up
  across calls.
- `expandable_segments=True`: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The
  allocator returns segments to the OS more aggressively.

Each row records the per-call wall and the VRAM reading taken right after the call's
bundle is written. The headline metric is the VRAM ceiling reached after N calls.

Writes:

    benchmarks/results/sweeps/sustained_load.csv
    benchmarks/results/sweeps/sustained_load.md

Run:

    python -m benchmarks.sustained_load --calls 15
    python -m benchmarks.sustained_load --calls 25 --arms expandable
"""

import argparse
import csv
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

logger = logging.getLogger("benchmarks.sustained_load")

_RESULTS_DIR = _ROOT / "benchmarks" / "results" / "sweeps"
_BUNDLES_DIR = _ROOT / "benchmarks" / "results" / "bundles"

# Pin the PDF: digital-born production-config has the most predictable per-call wall, so
# allocator drift shows up cleanly.
_DEFAULT_PDF = "904599.pdf"
_DEFAULT_CALLS = 15


def main(argv: list[str] | None = None) -> int:
    """Run two arms (legacy + expandable) of N sequential `process_pdf` calls."""
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
        # The allocator config has to be in place when CUDA initialises -- which means
        # before the very first `torch.cuda.*` call in the child. Easiest way to enforce
        # that is to spawn a fresh subprocess per arm; `_run_arm_subprocess` writes its
        # rows to a temp CSV and we splice them in here.
        logger.info(
            "[sustained_load] arm=%s calls=%d pdf=%s",
            arm,
            args.calls,
            pdf_path.name,
        )
        arm_rows = _run_arm_subprocess(arm, pdf_path, args.calls)
        rows.extend(arm_rows)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RESULTS_DIR / "sustained_load.csv"
    md_path = _RESULTS_DIR / "sustained_load.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, calls=args.calls)
    logger.info("wrote %s and %s", csv_path, md_path)
    return 0


def _resolve_arms(arms: str) -> list[str]:
    """Map the `--arms` flag to a list of arm names."""
    if arms == "both":
        return ["legacy", "expandable"]
    if arms in {"legacy", "expandable"}:
        return [arms]

    msg = f"unknown arms value: {arms!r}"
    raise ValueError(msg)


def _run_arm_subprocess(arm: str, pdf_path: Path, calls: int) -> list[dict[str, Any]]:
    """Spawn a child Python that runs `calls` PDFs and emits its rows as JSON lines.

    `PYTORCH_CUDA_ALLOC_CONF` must be set before CUDA initialises. Spawning a fresh
    interpreter for each arm is the only way to guarantee that; a single-process loop
    that toggles the env var mid-run wouldn't actually re-initialise the allocator.
    """
    env = os.environ.copy()
    if arm == "expandable":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    # `sys.executable -m benchmarks.sustained_load --child` runs the inner loop. S603 is
    # a misfire here: argv is a fixed list, no shell, no user input.
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "benchmarks.sustained_load",
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
        env=env,
        text=True,
    )

    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.startswith("ROW:"):
            continue

        # The child emits CSV-ish rows we can parse here. Format is intentionally
        # unstructured (one prefix + a JSON-ish trailer) so debugging the child's
        # output is easy from the captured log.
        import json  # noqa: PLC0415

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
    """Inner loop executed in the spawned child process.

    Loads the pipeline once, runs `process_pdf` N times against the same PDF, and emits
    one `ROW:` line per call with wall + VRAM-after numbers. The parent process splices
    these rows into the CSV.
    """
    import json  # noqa: PLC0415

    import pynvml  # noqa: PLC0415

    # Late import: the child must respect the parent's `PYTORCH_CUDA_ALLOC_CONF`, which
    # only takes effect if torch's first import happens AFTER the env var is set.
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
        table_enabled=True,
        references_enabled=False,  # GROBID dependency would skew per-call wall
        digital_born_workers=16,
        layout_fp16=True,
        layout_fast_preproc=True,
    )

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    for call_i in range(1, args.calls + 1):
        bundle_path = _BUNDLES_DIR / f"sustained_load_{args.arm}_{call_i}.tar"
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
        vram_mb = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / (1024**2)

        row = {
            "benchmark": "sustained_load",
            "value": f"{args.arm}_{call_i:02d}",
            "pdf": args.pdf,
            "language": "en",
            "status": status,
            "wall_s": round(wall_s, 3),
            "quality_call_index": call_i,
            "quality_arm": args.arm,
            "quality_vram_after_call_mb": round(vram_mb, 1),
            "error": error,
        }
        print(f"ROW:{json.dumps(row)}", flush=True)

        bundle_path.unlink(missing_ok=True)

    pynvml.nvmlShutdown()
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the sustained-load CSV."""
    fieldnames = [
        "benchmark",
        "value",
        "pdf",
        "language",
        "status",
        "wall_s",
        "quality_call_index",
        "quality_arm",
        "quality_vram_after_call_mb",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, calls: int) -> None:
    """Render an arm-comparison Markdown summary."""
    arms = sorted({r["quality_arm"] for r in rows})

    lines: list[str] = [
        "# sustained_load",
        "",
        f"{calls} sequential `process_pdf` calls in one Python process per arm. The "
        "VRAM column is read from NVML right after each call's bundle is written; "
        "drift across calls characterises the CUDA caching allocator's behavior on "
        "this workload.",
        "",
        f"- **Arms:** {', '.join(arms)}",
        f"- **Rows:** {len(rows)}",
        "",
    ]

    for arm in arms:
        arm_rows = [r for r in rows if r["quality_arm"] == arm]
        if not arm_rows:
            continue

        ok = [r for r in arm_rows if r["status"] == "ok"]
        if ok:
            wall_first = float(ok[0]["wall_s"])
            wall_last = float(ok[-1]["wall_s"])
            vram_first = float(ok[0]["quality_vram_after_call_mb"])
            vram_last = float(ok[-1]["quality_vram_after_call_mb"])
            lines.extend(
                [
                    f"## Arm: `{arm}`",
                    "",
                    f"- Calls measured: {len(arm_rows)} ({len(ok)} ok)",
                    f"- Wall: first={wall_first:.2f}s, last={wall_last:.2f}s",
                    f"- VRAM: first={vram_first:.0f} MiB, last={vram_last:.0f} MiB, "
                    f"delta={vram_last - vram_first:+.0f} MiB",
                    "",
                ]
            )

        lines.append("| call | wall (s) | VRAM after (MiB) | status |")
        lines.append("|---|---|---|---|")
        lines.extend(
            f"| {r['quality_call_index']} | {r['wall_s']} | "
            f"{r['quality_vram_after_call_mb']} | {r['status']} |"
            for r in arm_rows
        )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI flags for the parent process."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument("--calls", type=int, default=_DEFAULT_CALLS)
    parser.add_argument("--pdf", default=_DEFAULT_PDF)
    parser.add_argument(
        "--arms",
        choices=("legacy", "expandable", "both"),
        default="both",
        help="Which allocator arm(s) to run.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    # `--child` short-circuits to the inner loop; the parent invokes us via subprocess
    # so each arm gets a fresh `PYTORCH_CUDA_ALLOC_CONF` env.
    if "--child" in sys.argv:
        raise SystemExit(_child_loop())

    raise SystemExit(main())

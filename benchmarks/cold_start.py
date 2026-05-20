"""Cold-start benchmark: time and size each resident model's `__init__`.

Measures the three model loads the FastAPI lifespan performs at boot:

1. `LayoutAnalyzer` (PP-DocLayoutV3 via `make_analyzer_from_config`).
2. `FormulaRecognizer` (PP-FormulaNet-L via `make_formula_recognizer`).
3. `TableEngine` (RapidTable SLANet+ via `make_table_engine`).

Per stage: wall time of `__init__`, VRAM delta (NVML before vs after), and host RSS
delta. The aggregate `total` row sums the three and pins the service's effective boot
SLA -- the value `boot_timeout_s` is set against in `config.toml`.

Writes:

    benchmarks/results/sweeps/cold_start.csv   (one row per stage + a `total` row)
    benchmarks/results/sweeps/cold_start.md    (human-readable summary)

Run:

    python -m benchmarks.cold_start              # default: production-config models
    python -m benchmarks.cold_start --compile    # add formula_torch_compile timing
"""

import argparse
import csv
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# `HF_HOME` redirect mirrors `benchmarks/run_all.py` -- some dev hosts mount a read-only
# `/workspace/.hf_home`; redirect before any `transformers` import lands.
os.environ.setdefault("HF_HOME", str(Path("~/.cache/huggingface").expanduser()))

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# E402: imports below the `sys.path.insert` above so `ytcc_pipeline` resolves from the
# `src/` layout when this script runs directly.
import psutil  # noqa: E402
import pynvml  # noqa: E402
import torch  # noqa: E402

from ytcc_pipeline import PipelineConfig  # noqa: E402
from ytcc_pipeline.models.formula import make_formula_recognizer  # noqa: E402
from ytcc_pipeline.models.layout import make_analyzer_from_config  # noqa: E402
from ytcc_pipeline.processors.table import make_table_engine  # noqa: E402

logger = logging.getLogger("benchmarks.cold_start")

_RESULTS_DIR = _ROOT / "benchmarks" / "results" / "sweeps"


def main(argv: list[str] | None = None) -> int:
    """Time each resident-model init and write the CSV + Markdown."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    cfg = PipelineConfig(
        formula_enabled=True,
        table_enabled=True,
        formula_torch_compile=args.compile,
    )

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    process = psutil.Process(os.getpid())

    rows: list[dict[str, Any]] = []
    total_wall = 0.0

    # Baseline VRAM / RSS before any model loads. The deltas below ride on top of the
    # parent's Python interpreter footprint, not on an empty CUDA context (CUDA isn't
    # initialised yet).
    vram_baseline_mb = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / (1024**2)
    rss_baseline_mb = process.memory_info().rss / (1024**2)

    rows.append(
        _build_row(
            "analyzer",
            lambda: make_analyzer_from_config(cfg),
            handle=handle,
            process=process,
            vram_baseline_mb=vram_baseline_mb,
            rss_baseline_mb=rss_baseline_mb,
        ),
    )

    rows.append(
        _build_row(
            "formula",
            lambda: make_formula_recognizer(cfg),
            handle=handle,
            process=process,
            vram_baseline_mb=vram_baseline_mb,
            rss_baseline_mb=rss_baseline_mb,
        ),
    )

    rows.append(
        _build_row(
            "table",
            lambda: make_table_engine(
                device=cfg.table_device,
                batch_size=cfg.table_batch_size,
            ),
            handle=handle,
            process=process,
            vram_baseline_mb=vram_baseline_mb,
            rss_baseline_mb=rss_baseline_mb,
        ),
    )

    pynvml.nvmlShutdown()

    total_wall = sum(float(r["wall_s"]) for r in rows)
    final_vram_mb = float(rows[-1]["quality_vram_after_init_mb"])
    final_rss_mb = float(rows[-1]["quality_rss_after_init_mb"])
    rows.append(
        {
            "benchmark": "cold_start",
            "value": "total",
            "pdf": "",
            "language": "",
            "status": "ok",
            "wall_s": round(total_wall, 3),
            "quality_vram_after_init_mb": round(final_vram_mb, 1),
            "quality_rss_after_init_mb": round(final_rss_mb, 1),
            "error": "",
        },
    )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RESULTS_DIR / "cold_start.csv"
    md_path = _RESULTS_DIR / "cold_start.md"

    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, compile_on=args.compile)
    logger.info("wrote %s and %s", csv_path, md_path)
    return 0


def _build_row(  # noqa: PLR0913  -- one row gathers identity + handles + two baselines; bundling them obscures the call sites
    name: str,
    constructor: Any,  # noqa: ANN401  -- callable that returns a model; type varies
    *,
    handle: Any,  # noqa: ANN401
    process: psutil.Process,
    vram_baseline_mb: float,
    rss_baseline_mb: float,
) -> dict[str, Any]:
    """Run one model `__init__` and capture wall + VRAM/RSS delta.

    Holds a reference to the constructed model until the next call so cumulative VRAM
    reflects the production sequence (all three models resident simultaneously).
    """
    logger.info("[cold_start] %s init...", name)
    t0 = time.perf_counter()
    try:
        model = constructor()
        status = "ok"
        error = ""
    except Exception as exc:
        logger.exception("[cold_start] %s init failed", name)
        model = None
        status = "error"
        error = repr(exc)

    wall_s = time.perf_counter() - t0
    vram_used_mb = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / (1024**2)
    rss_used_mb = process.memory_info().rss / (1024**2)

    # Park the model in a module-level list to defeat the GC; the next stage starts with
    # this stage's allocations still resident, matching production.
    _RESIDENT.append(model)
    gc.collect()

    return {
        "benchmark": "cold_start",
        "value": name,
        "pdf": "",
        "language": "",
        "status": status,
        "wall_s": round(wall_s, 3),
        "quality_vram_after_init_mb": round(vram_used_mb, 1),
        "quality_vram_delta_mb": round(vram_used_mb - vram_baseline_mb, 1),
        "quality_rss_after_init_mb": round(rss_used_mb, 1),
        "quality_rss_delta_mb": round(rss_used_mb - rss_baseline_mb, 1),
        "error": error,
    }


# Keeps each constructed model alive across the for-loop so VRAM accumulates the way the
# FastAPI lifespan handler accumulates it.
_RESIDENT: list[Any] = []


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the cold-start CSV with the bespoke quality columns this benchmark uses."""
    fieldnames = [
        "benchmark",
        "value",
        "pdf",
        "language",
        "status",
        "wall_s",
        "quality_vram_after_init_mb",
        "quality_vram_delta_mb",
        "quality_rss_after_init_mb",
        "quality_rss_delta_mb",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    compile_on: bool,
) -> None:
    """Render a human-readable summary table for the cold-start CSV."""
    header_cells = (
        "stage",
        "wall (s)",
        "VRAM after (MiB)",
        "VRAM delta (MiB)",
        "RSS after (MiB)",
        "RSS delta (MiB)",
    )
    row_cells = (
        "value",
        "wall_s",
        "quality_vram_after_init_mb",
        "quality_vram_delta_mb",
        "quality_rss_after_init_mb",
        "quality_rss_delta_mb",
    )
    lines: list[str] = [
        "# cold_start",
        "",
        "Wall + resident VRAM/RSS after each model `__init__`. Bench mirrors the "
        "FastAPI lifespan: each row picks up where the previous one left off, so the "
        "VRAM column is cumulative.",
        "",
        f"- **Knob:** model init sequence (`formula_torch_compile={compile_on}`)",
        f"- **Runs:** {len(rows)}",
        "",
        "## Speed + Footprint",
        "",
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(["---"] * len(header_cells)) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(col, "")) for col in row_cells) + " |"
        for row in rows
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Enable `formula_torch_compile=True` so the formula row includes the "
            "Inductor compilation cost the production config pays at boot."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    # `torch.cuda.empty_cache()` here makes sense even pre-init: if a previous process
    # left a fragmented allocator in shared memory (rare, but happens during dev), the
    # cold-start measurement should start as clean as we can make it.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raise SystemExit(main())

"""API concurrency benchmark: N clients hitting `/process` in parallel.

Extends `api_smoke.py`'s single-request flow to characterise the FastAPI service under
concurrent load. The service serialises GPU work via `asyncio.Lock`, so total wall
scales with `sum(N)` rather than `max(N)` -- but queue wait, p50/p95/p99 server-side
processing time, and throughput shift meaningfully with concurrency.

Per concurrency level:

- Spin up the uvicorn subprocess once (shared across levels).
- Launch N concurrent `httpx.Client` workers posting the same PDF.
- Record per-request: queue wait (client send -> server start, approximated via
  `X-Processing-Time` header offset), server processing wall, client total wall.
- Aggregate: p50, p95, p99 processing wall; throughput as PDFs/min.

Writes:

    benchmarks/results/sweeps/api_concurrency.csv
    benchmarks/results/sweeps/api_concurrency.md
    benchmarks/results/api_smoke/uvicorn_concurrency.log

Run:

    python -m benchmarks.api_concurrency             # default: N in (1, 2, 4, 8)
    python -m benchmarks.api_concurrency --pdf 904599.pdf --language en --max-clients 4
"""

import argparse
import csv
import logging
import os
import statistics
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", str(Path("~/.cache/huggingface").expanduser()))

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import httpx  # noqa: E402

from ytcc_pipeline import load_service_config  # noqa: E402

logger = logging.getLogger("benchmarks.api_concurrency")

_RESULTS_DIR = _ROOT / "benchmarks" / "results" / "sweeps"
_LOG_DIR = _ROOT / "benchmarks" / "results" / "api_smoke"
_DEFAULT_PDF = "904599.pdf"
_DEFAULT_LANGUAGE = "en"
_DEFAULT_LEVELS = (1, 2, 4, 8)

# Per-request timeout. Concurrent clients all contend for the same `gpu_lock`, so the
# 8-client level's worst-case wait is ~7x a single PDF. 1800s is conservative.
_REQUEST_TIMEOUT_S = 1800.0

# Percentile points -- p99 needs at least N=4 samples to be meaningful so we run
# multiple PDFs at each concurrency level if the level itself is small.
_P95 = 0.95
_P99 = 0.99


def main(argv: list[str] | None = None) -> int:
    """Spin up the service and bench every requested concurrency level."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    pdf_path = _ROOT / "samples" / args.pdf
    if not pdf_path.is_file():
        logger.error("missing PDF: %s", pdf_path)
        return 1

    levels = tuple(int(n) for n in args.levels.split(",") if n)
    if not levels:
        logger.error("--levels must be a comma-separated list of ints")
        return 1

    api = load_service_config().api
    host = "127.0.0.1"
    port = api.port
    base_url = f"http://{host}:{port}"

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIR / "uvicorn_concurrency.log"
    log_fh = log_path.open("w")

    logger.info("starting uvicorn on %s ...", base_url)
    # S603 is a misfire here: argv is a fixed list, no shell, no user input.
    server = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ytcc_pipeline.api.app:app",
            "--host",
            host,
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=os.environ,
    )

    rows: list[dict[str, Any]] = []
    try:
        _wait_for_ready(base_url, timeout_s=api.boot_timeout_s)

        for n in levels:
            logger.info("[api_concurrency] level=N=%d", n)
            row = _run_level(
                base_url=base_url,
                pdf_path=pdf_path,
                language=args.language,
                n_clients=n,
            )
            rows.append(row)
    finally:
        logger.info("stopping uvicorn...")
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RESULTS_DIR / "api_concurrency.csv"
    md_path = _RESULTS_DIR / "api_concurrency.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    logger.info("wrote %s and %s", csv_path, md_path)
    return 0


def _run_level(
    *,
    base_url: str,
    pdf_path: Path,
    language: str,
    n_clients: int,
) -> dict[str, Any]:
    """Launch N concurrent clients, return one aggregated row."""
    per_client: list[dict[str, float]] = []
    lock = threading.Lock()

    def _worker() -> None:
        result = _send_one(base_url, pdf_path, language)
        with lock:
            per_client.append(result)

    threads = [threading.Thread(target=_worker, daemon=False) for _ in range(n_clients)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    wall_s = time.perf_counter() - t0

    ok = [r for r in per_client if r["status_ok"]]
    processing_walls = [r["processing_s"] for r in ok]
    client_walls = [r["client_total_s"] for r in ok]

    median_s = statistics.median(processing_walls) if processing_walls else 0.0
    p95_s = _percentile(processing_walls, _P95) if processing_walls else 0.0
    p99_s = _percentile(processing_walls, _P99) if processing_walls else 0.0
    throughput_pdfs_per_min = (len(ok) / wall_s) * 60.0 if wall_s > 0 else 0.0

    # Queue wait approximation: total client wall minus server-side processing wall is
    # everything else -- upload + queueing + download. Upload + download is ~constant
    # per PDF; the difference across N is the queue wait.
    queue_waits = [max(r["client_total_s"] - r["processing_s"], 0.0) for r in ok]
    median_queue_s = statistics.median(queue_waits) if queue_waits else 0.0

    return {
        "benchmark": "api_concurrency",
        "value": n_clients,
        "pdf": pdf_path.name,
        "language": language,
        "status": "ok" if ok else "error",
        "wall_s": round(wall_s, 3),
        "quality_n_clients": n_clients,
        "quality_n_ok": len(ok),
        "quality_n_failed": n_clients - len(ok),
        "quality_processing_median_s": round(median_s, 3),
        "quality_processing_p95_s": round(p95_s, 3),
        "quality_processing_p99_s": round(p99_s, 3),
        "quality_client_median_s": round(
            statistics.median(client_walls) if client_walls else 0.0,
            3,
        ),
        "quality_queue_wait_median_s": round(median_queue_s, 3),
        "quality_throughput_pdfs_per_min": round(throughput_pdfs_per_min, 3),
        "error": "" if ok else "all clients failed",
    }


def _send_one(base_url: str, pdf_path: Path, language: str) -> dict[str, float]:
    """Post one PDF and return timing + ok-flag."""
    client_t0 = time.perf_counter()
    try:
        with (
            httpx.Client(timeout=_REQUEST_TIMEOUT_S) as client,
            pdf_path.open("rb") as fp,
        ):
            files = {"pdf": (pdf_path.name, fp, "application/pdf")}
            data = {"language": language}
            r = client.post(
                f"{base_url}/process",
                files=files,
                data=data,
            )
            r.raise_for_status()
            processing_s = float(r.headers.get("X-Processing-Time", 0.0))
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("concurrency client failed: %s", exc)
        return {
            "status_ok": False,
            "client_total_s": time.perf_counter() - client_t0,
            "processing_s": 0.0,
        }

    return {
        "status_ok": True,
        "client_total_s": time.perf_counter() - client_t0,
        "processing_s": processing_s,
    }


def _wait_for_ready(base_url: str, *, timeout_s: float) -> None:
    """Poll `/health` until 200 OK or timeout. Mirrors `api_smoke._wait_for_ready`."""
    deadline = time.perf_counter() + timeout_s
    with httpx.Client(timeout=2.0) as client:
        while time.perf_counter() < deadline:
            try:
                r = client.get(f"{base_url}/health")
                if r.status_code == HTTPStatus.OK and r.json().get("model_loaded"):
                    return
            except httpx.HTTPError:
                pass

            time.sleep(0.5)

    msg = f"service didn't respond within {timeout_s}s"
    raise TimeoutError(msg)


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile -- same logic as `runner._percentile`."""
    if not values:
        return 0.0

    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]

    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the api_concurrency CSV."""
    fieldnames = [
        "benchmark",
        "value",
        "pdf",
        "language",
        "status",
        "wall_s",
        "quality_n_clients",
        "quality_n_ok",
        "quality_n_failed",
        "quality_processing_median_s",
        "quality_processing_p95_s",
        "quality_processing_p99_s",
        "quality_client_median_s",
        "quality_queue_wait_median_s",
        "quality_throughput_pdfs_per_min",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    """Render the api_concurrency Markdown summary."""
    header_cells = (
        "N",
        "wall (s)",
        "throughput (pdfs/min)",
        "p50 proc (s)",
        "p95 proc (s)",
        "p99 proc (s)",
        "median queue (s)",
        "ok / total",
    )
    row_cols = (
        "value",
        "wall_s",
        "quality_throughput_pdfs_per_min",
        "quality_processing_median_s",
        "quality_processing_p95_s",
        "quality_processing_p99_s",
        "quality_queue_wait_median_s",
    )
    lines: list[str] = [
        "# api_concurrency",
        "",
        "Concurrent client load on the FastAPI service. Each row is one concurrency "
        "level N: N clients post the same PDF in parallel; the service serialises GPU "
        "work via `asyncio.Lock` so total wall scales linearly with N.",
        "",
        f"- **Levels:** {[r['value'] for r in rows]}",
        f"- **Runs:** {len(rows)}",
        "",
        "## Throughput + latency",
        "",
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(["---"] * len(header_cells)) + "|",
    ]
    lines.extend(
        "| "
        + " | ".join(str(r[c]) for c in row_cols)
        + f" | {r['quality_n_ok']}/{r['quality_n_clients']} |"
        for r in rows
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument("--pdf", default=_DEFAULT_PDF)
    parser.add_argument("--language", default=_DEFAULT_LANGUAGE)
    parser.add_argument(
        "--levels",
        default=",".join(str(n) for n in _DEFAULT_LEVELS),
        help="Comma-separated concurrency levels (default: 1,2,4,8).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

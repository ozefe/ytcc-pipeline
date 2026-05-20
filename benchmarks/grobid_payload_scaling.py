"""GROBID payload scaling benchmark.

Sends synthetic citation lists of progressively larger sizes to a running GROBID server
and records the per-call wall + per-citation latency. Validates the
`grobid_timeout_s=60` default against real bibliography sizes; documents the curve so
deployments with unusually large bibliographies can right-size the timeout up front.

Does NOT use the ytcc pipeline; just hits `GrobidClient.process_citation_list` directly.
PDFs are irrelevant -- the unit of work is the citation count.

Writes:

    benchmarks/results/sweeps/grobid_payload_scaling.csv
    benchmarks/results/sweeps/grobid_payload_scaling.md

Run:

    python -m benchmarks.grobid_payload_scaling
    python -m benchmarks.grobid_payload_scaling --sizes 10,50,200,500,1000
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# E402: imports below the `sys.path.insert` so `ytcc_pipeline` resolves under the `src/`
# layout when this script runs directly.
from ytcc_pipeline.models.grobid import (  # noqa: E402
    GrobidClient,
    GrobidError,
    is_grobid_alive,
)

logger = logging.getLogger("benchmarks.grobid_payload_scaling")

_RESULTS_DIR = _ROOT / "benchmarks" / "results" / "sweeps"
_DEFAULT_SIZES = (10, 50, 200, 500)
_DEFAULT_URL = "http://localhost:8070"

# A synthetic citation chosen because it has every field GROBID's CRF model looks for
# (author, title, journal, volume, year, pages, DOI) -- so per-citation parse time is
# representative of real-world bibliographies, not optimistic.
_SAMPLE_CITATION = (
    "Smith, J. and Doe, A. (2020). On the convergence of variance-reduced stochastic "
    "gradient descent in high-dimensional regimes. Journal of Machine Learning "
    "Research, 21(94):1-32. https://doi.org/10.1234/jmlr.2020.094"
)


def main(argv: list[str] | None = None) -> int:
    """Send N citations to GROBID for N in `--sizes` and write the CSV + Markdown."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    if not is_grobid_alive(args.url, timeout_s=2.0):
        logger.error(
            "GROBID not reachable at %s -- start it (e.g. `docker run --rm -p "
            "8070:8070 grobid/grobid:0.9.0`) and retry",
            args.url,
        )
        return 1

    sizes = tuple(int(n) for n in args.sizes.split(",") if n)

    # A high per-call timeout so the slowest size (typically 500-1000 citations) doesn't
    # spuriously fail. We're characterising the curve, not enforcing the
    # `grobid_timeout_s` default value used in production.
    timeout_s = 600.0
    client = GrobidClient(url=args.url, timeout_s=timeout_s)

    # One warmup run so JVM JIT and any per-process model state are loaded before the
    # measured rows. Cheap (10 citations) and discarded.
    logger.info("[grobid_payload_scaling] warmup with 10 citations...")
    try:
        client.process_citation_list([_SAMPLE_CITATION] * 10)
    except GrobidError as exc:
        logger.warning("warmup failed: %s; continuing", exc)

    rows: list[dict[str, Any]] = []
    for size in sizes:
        row = _measure_size(client, size)
        rows.append(row)
        logger.info(
            "[grobid_payload_scaling] size=%d wall=%.2fs",
            size,
            float(row["wall_s"]),
        )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RESULTS_DIR / "grobid_payload_scaling.csv"
    md_path = _RESULTS_DIR / "grobid_payload_scaling.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, url=args.url)
    logger.info("wrote %s and %s", csv_path, md_path)
    return 0


def _measure_size(client: GrobidClient, size: int) -> dict[str, Any]:
    """Send `size` copies of the sample citation and return a row dict."""
    citations = [_SAMPLE_CITATION] * size
    t0 = time.perf_counter()
    status = "ok"
    error = ""
    parsed = 0
    try:
        refs = client.process_citation_list(citations)
        parsed = sum(1 for r in refs if r is not None)
    except GrobidError as exc:
        status = "error"
        error = repr(exc)

    wall_s = time.perf_counter() - t0
    per_ref_ms = (wall_s / size) * 1000.0 if size else 0.0

    return {
        "benchmark": "grobid_payload_scaling",
        "value": size,
        "pdf": "",
        "language": "",
        "status": status,
        "wall_s": round(wall_s, 3),
        "quality_citations_sent": size,
        "quality_references_parsed": parsed,
        "quality_grobid_wall_per_ref_ms": round(per_ref_ms, 3),
        "error": error,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the grobid_payload_scaling CSV."""
    fieldnames = [
        "benchmark",
        "value",
        "pdf",
        "language",
        "status",
        "wall_s",
        "quality_citations_sent",
        "quality_references_parsed",
        "quality_grobid_wall_per_ref_ms",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, url: str) -> None:
    """Render a per-size table summary."""
    lines: list[str] = [
        "# grobid_payload_scaling",
        "",
        "Synthetic citation lists of progressively larger sizes posted to "
        f"`{url}/api/processCitationList`. Each row is one batched POST; per-citation "
        'latency is `wall_s / N` -- the curve answers "can the default '
        '`grobid_timeout_s=60` cover an N-citation bibliography?".',
        "",
        f"- **URL:** `{url}`",
        f"- **Sizes:** {[r['value'] for r in rows]}",
        f"- **Runs:** {len(rows)}",
        "",
        "## Wall vs payload size",
        "",
        "| citations | wall (s) | per-ref (ms) | parsed / sent | status |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r['value']} | {r['wall_s']} | "
        f"{r['quality_grobid_wall_per_ref_ms']} | "
        f"{r['quality_references_parsed']}/{r['quality_citations_sent']} | "
        f"{r['status']} |"
        for r in rows
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("GROBID_URL", _DEFAULT_URL),
        help=(
            "Base URL of the running GROBID server (no trailing slash). Falls back to "
            "`GROBID_URL` env var, then `http://localhost:8070`."
        ),
    )
    parser.add_argument(
        "--sizes",
        default=",".join(str(s) for s in _DEFAULT_SIZES),
        help="Comma-separated payload sizes (default: 10,50,200,500).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

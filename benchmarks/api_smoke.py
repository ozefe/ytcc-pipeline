"""End-to-end FastAPI smoke + benchmark.

Spins up `uvicorn ytcc_pipeline.api.app:app` as a subprocess, waits for `/health`, POSTs
each target PDF, and records:

- `upload_s`: time spent writing the multipart body.
- `processing_s`: server-side wall, parsed from `X-Processing-Time`.
- `download_s`: body-receive time after headers arrive.
- `total_s`: total client wall (handshake + everything).
- `resources`: process-tree CPU%, host RSS, and device VRAM sampled at 250ms cadence
  over the request. min / median / max per metric.
- `stages`: per-stage wall time parsed from the uvicorn log (`stage render`, `stage
  layout`, `stage blocks`, `stage table`, `stage formula`, `stage reference`, `stage
  bundle`).
- `bundle`: block counts (text / image / reference / formula / table / miss),
  formula-quality stats (LaTeX recovered, image fallbacks, character count), and
  table-stage stats (tables with structured cells, image-only fallbacks, total cells,
  cells with text vs empty) inspected from the returned tar.

Writes:

    benchmarks/results/api_smoke/api_timings.json   (machine-readable report)
    benchmarks/results/api_smoke/uvicorn.log        (raw service log)
    benchmarks/results/api_smoke/<stem>.tar         (one per target PDF)

Edit `TARGETS` below to point at your own PDFs. Run from the project root:

    python benchmarks/api_smoke.py
"""

import contextlib
import json
import os
import re
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# Imports below sit AFTER the `sys.path.insert` above so that `ytcc_pipeline` is
# resolvable from the `src/` layout when the script runs directly. E402 is intentional
# here.
import httpx  # noqa: E402
import psutil  # noqa: E402
import pynvml  # noqa: E402

from ytcc_pipeline import load_service_config  # noqa: E402

if TYPE_CHECKING:
    from types import TracebackType

# Host/port/boot_timeout come from `config.toml` so the script honors the same
# single-source-of-truth as the service. The smoke binds to 127.0.0.1 regardless of
# `api.host` (which may be 0.0.0.0 in production) so it can always talk to the
# subprocess on the loopback interface.
_API_SETTINGS = load_service_config().api
HOST = "127.0.0.1"
PORT = _API_SETTINGS.port
BASE_URL = f"http://{HOST}:{PORT}"
SERVER_BOOT_TIMEOUT_S = _API_SETTINGS.boot_timeout_s

# Per-request HTTP timeout. Formula recognition on large digital-born PDFs (hundreds of
# crops) can take several minutes on top of layout + blocks; the scanned path adds OCR
# time on top. 1800s is generous but bounded so a stuck request still fails the bench.
REQUEST_TIMEOUT_S = 1800.0

# Drop `(pdf_stem, language)` tuples here to bench other documents. The script reads
# each entry as `samples/<stem>.pdf` at the project root; adjust `_pdf_path` below if
# your PDFs live elsewhere. Mix digital-born and scanned PDFs to exercise both regimes
# in a single run.
TARGETS: list[tuple[str, str]] = []


def _pdf_path(stem: str) -> Path:
    """Resolve a target stem to its source PDF path."""
    return _ROOT / "samples" / f"{stem}.pdf"


_STAGE_LINE = re.compile(
    r"stage (?P<name>\w+): pdf=(?P<pdf>\S+) .*?elapsed_s=(?P<elapsed>[\d.]+)",
)


def _start_server() -> subprocess.Popen[bytes]:
    """Launch the uvicorn service as a subprocess.

    Inherits the parent's environment so the HF_HOME override propagates. Routes
    stdout/stderr to a log file so we can post-mortem failures and parse per-stage
    timings out of it later.
    """
    log_dir = _ROOT / "benchmarks" / "results" / "api_smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "uvicorn.log"
    log_fh = log_path.open("w")

    # `sys.executable` + the uvicorn module path is a fixed, trusted argv; no shell, no
    # user input. S603 misfires on this pattern.
    return subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ytcc_pipeline.api.app:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=os.environ,
    )


def _stat_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "median": round(statistics.median(values), 1),
    }


class _ResourceSampler:
    """Background sampler for process-tree CPU/RSS and device-wide VRAM.

    Samples on a daemon thread at `interval_s` cadence. CPU sampling uses
    `psutil.Process.cpu_percent(interval=None)` which keeps the delta-snapshot state per
    Process object -- so the sampler caches Process handles by pid and reuses the same
    object across ticks. Fresh handles created each tick would always read 0.

    Spawn-process workers (block / OCR pools) appear and disappear during a request; the
    cache adds new pids the first time they show up (priming their cpu_percent counter
    at zero) and evicts pids that no longer exist.

    VRAM is read device-wide via NVML -- the host has one GPU dedicated to this bench so
    the reading reflects the service's full footprint: PyTorch scratch from the layout
    analyzer + formula recognizer, plus ONNXRuntime scratch from the OCR worker engines.

    Use as a context manager; `summary()` returns the aggregated stats.
    """

    def __init__(
        self,
        root_pid: int,
        *,
        gpu_index: int = 0,
        interval_s: float = 0.25,
    ) -> None:
        self._root_pid = root_pid
        self._gpu_index = gpu_index
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._samples: list[tuple[float, float, float]] = []
        self._thread: threading.Thread | None = None
        self._nvml_handle: Any = None
        self._procs: dict[int, psutil.Process] = {}

    def __enter__(self) -> Self:
        pynvml.nvmlInit()
        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)

        # Prime the parent + any existing children so the first sample tick already has
        # a delta to compute against.
        self._refresh_procs()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()

    def _refresh_procs(self) -> list[psutil.Process]:
        """Sync the pid cache with the current uvicorn process tree.

        Adds any newly-spawned worker processes (priming their `cpu_percent` counter so
        the next tick computes a real delta) and evicts pids that have exited. Returns
        the live cached `Process` objects.
        """
        try:
            root = psutil.Process(self._root_pid)
            live_pids = {root.pid, *(c.pid for c in root.children(recursive=True))}
        except psutil.NoSuchProcess:
            return []

        for pid in live_pids - self._procs.keys():
            try:
                proc = psutil.Process(pid)
                proc.cpu_percent(interval=None)
                self._procs[pid] = proc
            except psutil.NoSuchProcess:
                continue

        for pid in list(self._procs):
            if pid not in live_pids:
                self._procs.pop(pid, None)

        return list(self._procs.values())

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            cpu_pct = 0.0
            rss_b = 0
            for proc in self._refresh_procs():
                try:
                    cpu_pct += proc.cpu_percent(interval=None)
                    rss_b += proc.memory_info().rss
                except psutil.NoSuchProcess, psutil.AccessDenied:
                    continue

            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            self._samples.append(
                (cpu_pct, rss_b / (1024**2), float(mem.used) / (1024**2))
            )

    def summary(self) -> dict[str, Any]:
        if not self._samples:
            return {"samples": 0}

        cpu = [s[0] for s in self._samples]
        rss = [s[1] for s in self._samples]
        vram = [s[2] for s in self._samples]

        return {
            "samples": len(self._samples),
            "interval_s": self._interval_s,
            "cpu_pct": _stat_summary(cpu),
            "rss_mb": _stat_summary(rss),
            "vram_mb": _stat_summary(vram),
        }


def _wait_for_ready() -> float:
    """Poll `/health` until 200 OK or timeout.

    Returns:
        Seconds spent waiting for the server to come up (lifespan + analyzer +
        formula-model load).

    Raises:
        TimeoutError: the server didn't respond within `SERVER_BOOT_TIMEOUT_S` seconds.
    """
    t0 = time.perf_counter()
    deadline = t0 + SERVER_BOOT_TIMEOUT_S
    with httpx.Client(timeout=2.0) as client:
        while time.perf_counter() < deadline:
            try:
                r = client.get(f"{BASE_URL}/health")
                if r.status_code == HTTPStatus.OK and r.json().get("model_loaded"):
                    return time.perf_counter() - t0
            except httpx.HTTPError:
                pass

            time.sleep(0.5)

    msg = f"server didn't respond within {SERVER_BOOT_TIMEOUT_S}s"
    raise TimeoutError(msg)


def _send_pdf(pdf_path: Path, language: str, out_tar: Path) -> dict[str, Any]:
    """Send one PDF and return a per-request timing dict.

    Streams the response -- `client.stream` returns once headers arrive, which under
    HTTP/1.1 is when the server starts flushing the body (i.e. `process_pdf` is done).
    That lets us measure body-receive separately from server processing.

    Args:
        pdf_path: Source PDF to POST.
        language: ISO 639-1 code (form field).
        out_tar: Where to write the response body.

    Returns:
        Timings + bundle size.
    """
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        t_total = time.perf_counter()
        with pdf_path.open("rb") as fp:
            files = {"pdf": (pdf_path.name, fp, "application/pdf")}
            data = {"language": language}
            t_upload = time.perf_counter()
            with client.stream(
                "POST",
                f"{BASE_URL}/process",
                files=files,
                data=data,
            ) as resp:
                upload_s = time.perf_counter() - t_upload
                resp.raise_for_status()
                t_download = time.perf_counter()

                out_tar.write_bytes(resp.read())

                download_s = time.perf_counter() - t_download
                processing_s = float(resp.headers.get("X-Processing-Time", 0.0))

        total_s = time.perf_counter() - t_total

    return {
        "upload_s": round(upload_s, 3),
        "processing_s": round(processing_s, 3),
        "download_s": round(download_s, 3),
        "total_s": round(total_s, 3),
        "bundle_size_bytes": out_tar.stat().st_size,
    }


def _extract_stage_timings(log_path: Path, pdf_name: str) -> dict[str, float]:
    """Pull `stage <name>: pdf=<pdf> ... elapsed_s=<f>` lines for one PDF.

    Returns a name -> elapsed_s mapping. Each stage appears at most once per PDF in the
    current pipeline; if the log shows multiple matches (e.g. multiple runs against the
    same bench script invocation) we keep the last one seen.
    """
    stages: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _STAGE_LINE.search(line)
        if m and m.group("pdf") == pdf_name:
            stages[m.group("name")] = float(m.group("elapsed"))

    return stages


def _inspect_bundle(tar_path: Path) -> dict[str, Any]:  # noqa: C901, PLR0912  -- single-pass walk over every block type
    """Open the bundle and pull headline block / formula / table stats.

    Counts blocks by `type` and `miss`, totals LaTeX character output from formula
    blocks, and reports per-table cell counts (recovered vs empty).
    """
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
        fp = tf.extractfile("document.json")
        if fp is None:
            msg = f"bundle missing document.json: {tar_path}"
            raise OSError(msg)

        doc = json.loads(fp.read())

    counts = {
        "text": 0,
        "image": 0,
        "reference": 0,
        "formula": 0,
        "table": 0,
        "miss": 0,
    }
    formula_with_text = 0
    formula_with_image = 0
    formula_chars = 0
    tables_with_cells = 0
    tables_fallback_image_only = 0
    total_cells = 0
    cells_with_text = 0
    cells_empty = 0

    for page in doc["pages"]:
        for block in page["blocks"]:
            counts[block["type"]] += 1
            if block.get("miss"):
                counts["miss"] += 1

            if block["type"] == "formula":
                if block.get("text"):
                    formula_with_text += 1
                    formula_chars += len(block["text"])

                if block.get("image_path"):
                    formula_with_image += 1
            elif block["type"] == "table":
                cells = block.get("cells")
                if cells:
                    tables_with_cells += 1
                    for cell in cells:
                        total_cells += 1
                        if cell.get("text"):
                            cells_with_text += 1
                        else:
                            cells_empty += 1
                else:
                    tables_fallback_image_only += 1

    return {
        "pages": len(doc["pages"]),
        "pipeline_version": doc.get("pipeline_version"),
        "digital_born": doc.get("digital_born"),
        "block_counts": counts,
        "formula_with_text": formula_with_text,
        "formula_with_image_path": formula_with_image,
        "formula_text_chars": formula_chars,
        "tables_with_cells": tables_with_cells,
        "tables_fallback_image_only": tables_fallback_image_only,
        "total_cells": total_cells,
        "cells_with_text": cells_with_text,
        "cells_empty": cells_empty,
        "bundle_files": len(names),
    }


def main() -> None:  # noqa: C901, PLR0912, PLR0915  -- linear CLI flow with several optional report sections
    """Spin up the FastAPI service, POST every TARGETS entry, write the report."""
    if not TARGETS:
        sys.exit(
            "no targets configured; edit `TARGETS` at the top of "
            "benchmarks/api_smoke.py"
        )
    out_dir = _ROOT / "benchmarks" / "results" / "api_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "uvicorn.log"

    print("starting uvicorn...", flush=True)
    server = _start_server()
    try:
        boot_s = _wait_for_ready()
        print(f"server ready in {boot_s:.1f}s\n", flush=True)

        records: list[dict[str, Any]] = []
        for stem, lang in TARGETS:
            pdf = _pdf_path(stem)
            out_tar = out_dir / f"{stem}.tar"

            if not pdf.is_file():
                # Missing source PDFs (e.g. moved between machines) are skipped rather
                # than fatal -- the bench should still produce numbers for everything
                # that IS present.
                print(
                    f"=== {stem}.pdf ({lang}) -- SKIPPED (missing source) ===",
                    flush=True,
                )
                continue

            print(f"=== {stem}.pdf ({lang}) ===", flush=True)
            # Sampler scope ends after the response body has been received and the
            # bundle written; the per-request resource summary therefore covers upload +
            # processing + download.
            with _ResourceSampler(server.pid) as sampler:
                timings = _send_pdf(pdf, lang, out_tar)

            resources = sampler.summary()
            stages = _extract_stage_timings(log_path, pdf.name)
            bundle = _inspect_bundle(out_tar)
            entry = {
                "pdf": f"{stem}.pdf",
                "language": lang,
                **timings,
                "resources": resources,
                "stages": stages,
                "bundle": bundle,
            }

            print(json.dumps(entry, indent=2), flush=True)
            records.append(entry)
            print(flush=True)

        report = {
            "server_boot_s": round(boot_s, 3),
            "results": records,
        }
        out_json = out_dir / "api_timings.json"
        out_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {out_json}", flush=True)

        print("\nSummary (seconds):")
        print(
            f"{'PDF':>14}  {'upload':>8}  {'processing':>11} {'download':>9} "
            f"{'total':>7}",
        )

        for r in records:
            print(
                f"{r['pdf']:>14}  {r['upload_s']:>8.2f}  "
                f"{r['processing_s']:>11.2f}  "
                f"{r['download_s']:>9.2f}  {r['total_s']:>7.2f}",
            )

        if any(r["stages"] for r in records):
            print("\nPer-stage wall (seconds):")
            print(
                f"{'PDF':>14}  {'render':>7}  {'layout':>7}  {'blocks':>7}  "
                f"{'table':>7}  {'formula':>8}  {'bundle':>7}",
            )
            for r in records:
                s = r["stages"]
                print(
                    f"{r['pdf']:>14}  "
                    f"{s.get('render', 0):>7.2f}  {s.get('layout', 0):>7.2f}  "
                    f"{s.get('blocks', 0):>7.2f}  {s.get('table', 0):>7.2f}  "
                    f"{s.get('formula', 0):>8.2f}  {s.get('bundle', 0):>7.2f}",
                )

        # Resource summary. CPU% is summed across the uvicorn process tree; 100% ~= one
        # logical core saturated.
        print("\nResources (min / median / max per request):")
        print(
            f"{'PDF':>14}  {'CPU% min':>9}  {'CPU% med':>9}  {'CPU% max':>9}  "
            f"{'RSS MiB med':>11}  {'RSS MiB max':>11}  "
            f"{'VRAM MiB med':>12}  {'VRAM MiB max':>12}",
        )
        for r in records:
            res = r["resources"]
            if res.get("samples", 0) == 0:
                continue

            cpu = res["cpu_pct"]
            rss = res["rss_mb"]
            vram = res["vram_mb"]
            print(
                f"{r['pdf']:>14}  "
                f"{cpu['min']:>9.0f}  {cpu['median']:>9.0f}  {cpu['max']:>9.0f}  "
                f"{rss['median']:>11.0f}  {rss['max']:>11.0f}  "
                f"{vram['median']:>12.0f}  {vram['max']:>12.0f}",
            )

        # Bundle quality summary.
        print("\nBundle (block counts + formula recovery):")
        print(
            f"{'PDF':>14}  {'text':>5}  {'image':>5}  {'ref':>4}  "
            f"{'formula':>7}  {'table':>5}  {'miss':>4}  "
            f"{'tex/tot':>9}  {'tex_chars':>9}",
        )
        for r in records:
            b = r["bundle"]
            c = b["block_counts"]
            tex_ratio = f"{b['formula_with_text']}/{c['formula']}"
            print(
                f"{r['pdf']:>14}  "
                f"{c['text']:>5}  {c['image']:>5}  {c['reference']:>4}  "
                f"{c['formula']:>7}  {c['table']:>5}  {c['miss']:>4}  "
                f"{tex_ratio:>9}  {b['formula_text_chars']:>9}",
            )

        # Table cell summary.
        if any(r["bundle"].get("total_cells", 0) > 0 for r in records):
            print("\nTable cells (structured cell grid):")
            print(
                f"{'PDF':>14}  {'tables':>6}  {'fallbk':>6}  "
                f"{'cells':>6}  {'w/text':>6}  {'empty':>6}",
            )
            for r in records:
                b = r["bundle"]
                if b.get("total_cells", 0) == 0:
                    continue

                print(
                    f"{r['pdf']:>14}  {b['tables_with_cells']:>6}  "
                    f"{b['tables_fallback_image_only']:>6}  "
                    f"{b['total_cells']:>6}  {b['cells_with_text']:>6}  "
                    f"{b['cells_empty']:>6}",
                )
    finally:
        print("\nshutting down server...", flush=True)
        server.terminate()

        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()

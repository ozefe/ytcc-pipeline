"""Core benchmark runner: run one `process_pdf` call, capture metrics.

A benchmark run consists of:

- A `PipelineConfig` (the knob under test plus an apples-to-apples base).
- A source PDF + language.
- An optional `digital_born` override (skips the auto-detect probe).

The runner samples CPU / RSS / VRAM on a background thread, captures per-stage wall
times by attaching a logging handler to the `ytcc_pipeline` tree, inspects the produced
bundle for quality metrics, and returns a `BenchmarkResult`. Each row is OOM-safe --
torch CUDA OOM, RuntimeError matching CUDA OOM strings, and arbitrary exceptions all
turn into a `status="oom"` / `status="error"` row with the exception message attached.
"""

import contextlib
import gc
import json
import logging
import os
import re
import statistics
import tarfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import psutil
import pynvml
import torch

from ytcc_pipeline import PipelineConfig, process_pdf

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import TracebackType

__all__ = [
    "BenchmarkResult",
    "ResourceSampler",
    "ResourceStats",
    "StageTimingCapture",
    "extract_quality_metrics",
    "run_one",
]

_STAGE_LINE = re.compile(
    r"stage (?P<name>\w+): pdf=(?P<pdf>\S+) .*?elapsed_s=(?P<elapsed>[\d.]+)",
)

# The formula stage emits a richer line that includes `truncated=N`, the per-PDF count
# of crops where greedy generation stopped at `max_new_tokens` instead of EOS. Captured
# alongside stage timings so sweeps that tune the bucket caps (and risk truncation) can
# track the quality signal end-to-end without bundle-schema changes.
_FORMULA_TRUNCATED_LINE = re.compile(
    r"stage formula: .*?truncated=(?P<truncated>\d+)",
)
_OOM_HINTS = ("CUDA out of memory", "out of memory", "CUBLAS_STATUS_ALLOC_FAILED")

# Below this sample count, percentile math (median / p95) is meaningless; the stats
# collapse to "all four values equal the one sample we got".
_MIN_SAMPLES_FOR_PERCENTILES = 2


@dataclass(slots=True, frozen=True)
class ResourceStats:
    """Aggregate stats over a sampled metric.

    Attributes:
        min_: Sample minimum.
        median: 50th percentile.
        p95: 95th percentile.
        max_: Sample maximum.
        samples: Number of samples included.
    """

    min_: float
    median: float
    p95: float
    max_: float
    samples: int


@dataclass(slots=True)
class BenchmarkResult:
    """One row in a sweep's CSV: timings, resources, quality, status.

    The status field is `"ok"` for successful runs, `"oom"` for runs that hit CUDA / CPU
    out-of-memory, and `"error"` for everything else; the `error` field carries the
    exception message.
    """

    benchmark: str
    value: Any
    pdf: str
    language: str
    status: str = "ok"
    wall_s: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)
    cpu_pct: ResourceStats | None = None
    rss_mb: ResourceStats | None = None
    vram_mb: ResourceStats | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def run_one(  # noqa: PLR0913  -- one row gathers config + identity + IO knobs; bundling them obscures call sites
    config: PipelineConfig,
    pdf_path: Path,
    *,
    benchmark: str,
    value: Any,  # noqa: ANN401  -- sweep values are heterogeneous (int / bool / str / float)
    language: str = "en",
    digital_born: bool | None = None,
    output_dir: Path,
    sampler_interval_s: float = 0.25,
) -> BenchmarkResult:
    """Run `process_pdf` once and return a populated `BenchmarkResult`.

    Args:
        config: The `PipelineConfig` to drive the run.
        pdf_path: Source PDF; must exist.
        benchmark: Sweep name (carried into the row).
        value: The knob value under test (carried into the row).
        language: ISO 639-1 code; only used for scanned PDFs.
        digital_born: Pass `True`/`False` to skip auto-detect. `None` lets the pipeline
            probe the text layer.
        output_dir: Where to write the bundle. The bundle is deleted after quality
            extraction.
        sampler_interval_s: Resource-sampler cadence in seconds.

    Returns:
        A `BenchmarkResult` with status `"ok"` on success, `"oom"` on out-of-memory, or
            `"error"` on any other exception.
    """
    result = BenchmarkResult(
        benchmark=benchmark,
        value=value,
        pdf=pdf_path.name,
        language=language,
    )

    stage_capture = StageTimingCapture()
    root_logger = logging.getLogger("ytcc_pipeline")
    root_logger.addHandler(stage_capture)
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)

    bundle_path = (
        output_dir / f"{benchmark}_{_sanitize_for_filename(value)}_{pdf_path.stem}.tar"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    try:
        with ResourceSampler(os.getpid(), interval_s=sampler_interval_s) as sampler:
            try:
                process_pdf(
                    pdf_path,
                    language=language,
                    digital_born=digital_born,
                    output_path=bundle_path,
                    config=config,
                )
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if _is_oom(exc):
                    result.status = "oom"
                    result.error = str(exc)
                else:
                    result.status = "error"
                    result.error = repr(exc)
            except Exception as exc:  # noqa: BLE001  -- one bad row records status="error" and the sweep continues
                result.status = "error"
                result.error = repr(exc)

        result.wall_s = time.perf_counter() - started

        if result.status == "ok" and bundle_path.is_file():
            try:
                result.quality = extract_quality_metrics(bundle_path)
            except Exception as exc:  # noqa: BLE001  -- inspection failure shouldn't abort the sweep; we just record the row as error
                result.status = "error"
                result.error = f"bundle inspection failed: {exc!r}"

        # Log-derived metrics that aren't recorded in the bundle JSON.
        # `formula_truncated` is the only one today; surfaced as a quality column so
        # plotting + reporting treat it uniformly with the bundle-derived counters.
        if stage_capture.formula_truncated is not None:
            result.quality.setdefault(
                "formula_truncated",
                stage_capture.formula_truncated,
            )

        result.stages = dict(stage_capture.stages)
        cpu, rss, vram = sampler.summarise()
        result.cpu_pct = cpu
        result.rss_mb = rss
        result.vram_mb = vram
    finally:
        root_logger.removeHandler(stage_capture)
        root_logger.setLevel(previous_level)
        with contextlib.suppress(OSError):
            bundle_path.unlink(missing_ok=True)

        _release_cuda()

    return result


def extract_quality_metrics(bundle_path: Path) -> dict[str, Any]:  # noqa: C901, PLR0912  -- single-pass tally of every block subtype
    """Inspect a produced bundle and return per-block-type stats.

    Reads `document.json` out of the tar and walks every page. Counts text/image/
    reference/formula/table blocks, MISS fallbacks, formula LaTeX recovery, table cell
    counts, and parsed references. Also records the bundle byte size -- relevant when
    sweeping knobs that change crop format or MISS-image inclusion.

    Args:
        bundle_path: Path to a bundle tar written by `process_pdf`.

    Returns:
        A flat dict of counters suitable for joining into the CSV row.
    """
    bundle_bytes = bundle_path.stat().st_size

    with tarfile.open(bundle_path) as tf:
        fp = tf.extractfile("document.json")
        if fp is None:
            msg = f"bundle missing document.json: {bundle_path}"
            raise OSError(msg)

        doc = json.loads(fp.read())

    by_type: Counter[str] = Counter()
    text_chars = 0
    miss = 0
    formulas_with_text = 0
    formulas_with_image = 0
    formula_latex_chars = 0
    tables_with_cells = 0
    tables_image_only = 0
    total_cells = 0
    cells_with_text = 0
    references_parsed = 0
    references_total = 0

    for page in doc["pages"]:
        for block in page["blocks"]:
            block_type = block["type"]
            by_type[block_type] += 1
            if block.get("text"):
                text_chars += len(block["text"])

            if block.get("miss"):
                miss += 1

            if block_type == "formula":
                if block.get("text"):
                    formulas_with_text += 1
                    formula_latex_chars += len(block["text"])

                if block.get("image_path"):
                    formulas_with_image += 1
            elif block_type == "table":
                cells = block.get("cells")
                if cells:
                    tables_with_cells += 1
                    total_cells += len(cells)
                    cells_with_text += sum(1 for c in cells if c.get("text"))
                else:
                    tables_image_only += 1
            elif block_type == "reference":
                references_total += 1
                if block.get("reference"):
                    references_parsed += 1

    return {
        "blocks_total": sum(by_type.values()),
        "blocks_text": by_type["text"],
        "blocks_image": by_type["image"],
        "blocks_reference": by_type["reference"],
        "blocks_formula": by_type["formula"],
        "blocks_table": by_type["table"],
        "text_chars": text_chars,
        "miss": miss,
        "formulas_with_text": formulas_with_text,
        "formulas_with_image": formulas_with_image,
        "formula_latex_chars": formula_latex_chars,
        "tables_with_cells": tables_with_cells,
        "tables_image_only": tables_image_only,
        "total_cells": total_cells,
        "cells_with_text": cells_with_text,
        "references_parsed": references_parsed,
        "references_total": references_total,
        "bundle_bytes": bundle_bytes,
    }


class StageTimingCapture(logging.Handler):
    """Parse `stage <name>: ... elapsed_s=N` log lines into a dict.

    Attach to the `ytcc_pipeline` logger before running `process_pdf`; the orchestrator
    emits one INFO line per completed stage matching that pattern. The formula stage
    log line additionally carries `truncated=N` -- captured separately into
    `formula_truncated` so bucket-threshold sweeps can correlate quality with config.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stages: dict[str, float] = {}

        # `None` distinguishes "never observed the formula stage" from "ran the stage
        # and saw zero truncations" once it surfaces in the quality dict.
        self.formula_truncated: int | None = None

    def emit(self, record: logging.LogRecord) -> None:
        """Parse one stage log record into `self.stages` / `self.formula_truncated`."""
        message = record.getMessage()
        match = _STAGE_LINE.search(message)
        if match:
            # Keep the last seen elapsed for each stage -- relevant when the same stage
            # logs more than once in a single run (rare; the orchestrator emits a single
            # summary line per stage today).
            self.stages[match.group("name")] = float(match.group("elapsed"))

        truncated_match = _FORMULA_TRUNCATED_LINE.search(message)
        if truncated_match:
            self.formula_truncated = int(truncated_match.group("truncated"))


class ResourceSampler:
    """Background sampler for process-tree CPU / RSS and device VRAM.

    Mirrors the harness used in `benchmarks/api_smoke.py`. CPU sampling uses
    `psutil.Process.cpu_percent(interval=None)` which keeps a delta-snapshot per
    `Process` instance, so we cache `Process` handles by pid; spawn-workers that appear
    and disappear during the run are added on first sight (with their delta primed at
    zero) and evicted once they exit.

    VRAM is read device-wide via NVML; on a single-GPU host this is the process-tree's
    full footprint (PyTorch + ONNXRuntime contexts both show up).

    Use as a context manager; `summarise()` returns `(cpu, rss, vram)` triples of
    `ResourceStats`.
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
        self._cpu_samples: list[float] = []
        self._rss_samples: list[float] = []
        self._vram_samples: list[float] = []
        self._thread: threading.Thread | None = None
        self._nvml_handle: Any = None
        self._procs: dict[int, psutil.Process] = {}

    def __enter__(self) -> Self:
        """Start NVML, prime the pid cache, and launch the sampler thread."""
        pynvml.nvmlInit()
        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)

        # Prime every currently-live pid so the first tick's reading  already has a
        # meaningful interval to divide against.
        self._add_new(self._live_pids())
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        """Stop the sampler thread and release NVML."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()

    def _evict_dead(self, live_pids: set[int]) -> None:
        """Drop cached `Process` handles whose pid no longer exists."""
        for pid in list(self._procs):
            if pid not in live_pids:
                self._procs.pop(pid, None)

    def _add_new(self, live_pids: set[int]) -> None:
        """Cache + prime any newly-spawned pid.

        The prime call returns 0 and seeds `cpu_percent`'s delta-snapshot state; the
        newly-added process must wait one full sampler tick before its
        `cpu_percent(interval=None)` returns a meaningful value. Reading it on the same
        tick we primed it would divide the new process's CPU time by an effectively-zero
        interval and produce spike values (commonly 10000-15000% -- physically larger
        than the host's logical-CPU count).
        """
        for pid in live_pids - self._procs.keys():
            try:
                proc = psutil.Process(pid)
                proc.cpu_percent(interval=None)
                self._procs[pid] = proc
            except psutil.NoSuchProcess:
                continue

    def _live_pids(self) -> set[int]:
        """Snapshot the parent + every descendant pid."""
        try:
            root = psutil.Process(self._root_pid)
            return {root.pid, *(c.pid for c in root.children(recursive=True))}
        except psutil.NoSuchProcess:
            return set()

    def _loop(self) -> None:
        # Two-phase per tick to keep new-process spikes out of the data:
        #   1. Read CPU/RSS for processes already primed in a prior tick.
        #   2. Refresh the cache -- evict dead, add+prime new -- so any new worker
        #      contributes for the first time on the next tick.
        while not self._stop.wait(self._interval_s):
            live_pids = self._live_pids()
            self._evict_dead(live_pids)

            cpu_pct = 0.0
            rss_bytes = 0
            for proc in list(self._procs.values()):
                try:
                    cpu_pct += proc.cpu_percent(interval=None)
                    rss_bytes += proc.memory_info().rss
                except psutil.NoSuchProcess, psutil.AccessDenied:
                    continue

            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                vram_mb = float(mem.used) / (1024**2)
            except pynvml.NVMLError:
                vram_mb = 0.0

            self._cpu_samples.append(cpu_pct)
            self._rss_samples.append(rss_bytes / (1024**2))
            self._vram_samples.append(vram_mb)

            # New procs primed here; their first measured reading lands on the next
            # tick, after a full `interval_s` of CPU time.
            self._add_new(live_pids)

    def summarise(self) -> tuple[ResourceStats, ResourceStats, ResourceStats]:
        """Return aggregate stats for CPU%, RSS (MiB), VRAM (MiB)."""
        return (
            _stats_or_empty(self._cpu_samples),
            _stats_or_empty(self._rss_samples),
            _stats_or_empty(self._vram_samples),
        )


def _stats_or_empty(values: list[float]) -> ResourceStats:
    if not values:
        return ResourceStats(0.0, 0.0, 0.0, 0.0, 0)

    if len(values) < _MIN_SAMPLES_FOR_PERCENTILES:
        v = values[0]
        return ResourceStats(v, v, v, v, len(values))

    return ResourceStats(
        min_=round(min(values), 2),
        median=round(statistics.median(values), 2),
        p95=round(_percentile(values, 0.95), 2),
        max_=round(max(values), 2),
        samples=len(values),
    )


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile for the small samples we see."""
    if not values:
        return 0.0

    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]

    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _is_oom(exc: BaseException) -> bool:
    """Recognise CUDA / system OOM regardless of the exact exception type."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True

    msg = str(exc)
    return any(hint in msg for hint in _OOM_HINTS)


def _release_cuda() -> None:
    """Free CUDA memory between runs so OOM in run N doesn't propagate."""
    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(RuntimeError):
            torch.cuda.empty_cache()

        with contextlib.suppress(RuntimeError):
            torch.cuda.ipc_collect()


def _sanitize_for_filename(value: Any) -> str:  # noqa: ANN401  -- mirrors run_one's `value: Any` (heterogeneous sweep values)
    """Sanitise a sweep value for use in filenames."""
    s = str(value).replace("/", "_").replace(":", "_").replace(" ", "_")
    return s[:64]


@contextlib.contextmanager
def warmup_log_suppression() -> Iterator[None]:
    """Demote the ytcc_pipeline logger to WARNING for the duration of a warmup.

    Keeps the console clean during the no-measurement warmup run that precedes each
    sweep.
    """
    logger = logging.getLogger("ytcc_pipeline")
    previous = logger.level
    logger.setLevel(logging.WARNING)

    try:
        yield
    finally:
        logger.setLevel(previous)

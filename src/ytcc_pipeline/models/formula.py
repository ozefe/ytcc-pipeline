"""PP-FormulaNet-L wrapper: formula image crop -> LaTeX string.

The model loads via `transformers.AutoModelForImageTextToText`, which resolves to
`PPFormulaNetForConditionalGeneration` for the
`PaddlePaddle/PP-FormulaNet-L_safetensors` snapshot. Image preprocessing and tokenizer
decoding go through the matching `PPFormulaNetProcessor`. No PaddleOCR runtime is
involved -- only torch + transformers.

The pipeline runs formula recognition as a single batched pass in the main process after
the block stage has saved every formula crop to disk. See `run_formula_stage` for the
call site.
"""
# pyright thinks float16 and float32 are not exported from "torch" -- they are.
# pyright: reportPrivateImportUsage=false

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from ytcc_pipeline.image_io import read_rgb

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ytcc_pipeline.config import PipelineConfig

__all__ = [
    "BucketSpec",
    "FormulaRecognizer",
    "FormulaResult",
    "make_formula_recognizer",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class FormulaResult:
    """One crop's worth of model output.

    Attributes:
        latex: Post-processed LaTeX string, or `None` when the model produced empty
            output, the forward pass raised, or the crop could not be loaded. `None` is
            treated as a MISS by the caller.
        truncated: `True` if generation stopped at `max_new_tokens` rather than at EOS.
            The LaTeX is retained (caller decides whether to keep it) but flagged so
            operators can surface the truncation as a warning.
    """

    latex: str | None
    truncated: bool


@dataclass(slots=True, frozen=True)
class BucketSpec:
    """Per-bucket settings for sequence-bucketed batched recognition.

    Three buckets (`small`, `medium`, `large`) are partitioned by the `small_threshold`
    and `medium_threshold` boundaries -- both measured in the same unit the caller
    passes for `crop_areas` (typically source-page px**2 at the render DPI). Each bucket
    carries its own `max_new_tokens` cap; crops in the `large` bucket fall through to
    the recognizer's construction-time cap.

    Attributes:
        small_threshold: Crops with area strictly less than this go to the `small`
            bucket.
        medium_threshold: Crops with area in `[small_threshold, medium_threshold)` go to
            `medium`; anything at or above goes to `large`.
        small_tokens: `max_new_tokens` override applied to the `small` bucket's batched
            generation call.
        medium_tokens: `max_new_tokens` override applied to the `medium` bucket's
            batched generation call.
    """

    small_threshold: float
    medium_threshold: float
    small_tokens: int
    medium_tokens: int

    def __post_init__(self) -> None:
        """Validate threshold ordering and per-bucket token caps.

        Raises:
            ValueError: `small_threshold` is not strictly less than `medium_threshold`,
                or either tokens cap is below 1.
        """
        if not (0 < self.small_threshold < self.medium_threshold):
            msg = (
                "small_threshold must be > 0 and < medium_threshold; "
                f"got small={self.small_threshold} medium={self.medium_threshold}"
            )
            raise ValueError(msg)

        if self.small_tokens < 1 or self.medium_tokens < 1:
            msg = (
                "bucket token caps must be >= 1; "
                f"got small={self.small_tokens} medium={self.medium_tokens}"
            )
            raise ValueError(msg)


class FormulaRecognizer:
    """Convert formula image crops to LaTeX via PP-FormulaNet-L.

    Loaded once per service worker (FastAPI lifespan) or once per `process_pdf` call
    (library mode). One torch + transformers model instance; batched inference via
    `recognize_batch_paths`.
    """

    def __init__(
        self,
        model_id: str = "PaddlePaddle/PP-FormulaNet-L_safetensors",
        *,
        device: str = "cuda:0",
        dtype: Literal["fp16", "fp32"] = "fp16",
        max_new_tokens: int = 1536,
        torch_compile: bool = False,
    ) -> None:
        """Load the model + processor onto `device`.

        Args:
            model_id: Local directory path or Hugging Face repo id pointing at a
                PP-FormulaNet-L SafeTensors snapshot.
            device: Torch device string (`"cuda:0"` or `"cpu"`).
            dtype: Inference dtype.
            max_new_tokens: Per-crop generation cap. Matches PP-FormulaNet-L's
                training-time `generation_config.max_length` of 1537; production caps at
                1536 so the largest display-equation crops finish without truncation.
            torch_compile: When `True`, wrap the model with `torch.compile` after
                loading and run one warmup forward pass so Inductor compiles the hot
                subgraphs at construction time. First-call latency grows by ~20-30s but
                subsequent batches run kernel-fused.

        Raises:
            Exception: Re-raises any error from `AutoProcessor.from_pretrained` or
                `AutoModelForImageTextToText.from_pretrained` after logging at `ERROR`
                so the failure is loud at boot time.
        """
        self._model_id = model_id
        self._device = device
        self._dtype = torch.float16 if dtype == "fp16" else torch.float32
        self._max_new_tokens = max_new_tokens

        t0 = time.perf_counter()
        try:
            self._processor = AutoProcessor.from_pretrained(model_id)
            # `attn_implementation="sdpa"` is the transformers 4.36+ default for this
            # architecture; setting it explicitly pins the kernel choice so subtle
            # version upgrades don't flip it. Flash Attention 2 is not supported by
            # PP-FormulaNet. `_BaseModelWithGenerate` is a mixin protocol that hides
            # `.generate` from pyright's AutoModel return type. Cast to `Any` at the
            # boundary so call sites stay readable.
            model = cast(
                "Any",
                AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    dtype=self._dtype,
                    attn_implementation="sdpa",
                ),
            )
            self._model: Any = model.to(device).eval()

            # The shipped `generation_config.json` sets `max_length=1537` alongside our
            # per-call `max_new_tokens`; `.generate` warns at every call when both are
            # set. Clearing `max_length` here leaves `max_new_tokens` as the sole cap
            # and silences the per-call warning.
            self._model.generation_config.max_length = None
        except Exception:
            logger.exception("formula model load failed: model_id=%s", model_id)
            raise

        load_elapsed = time.perf_counter() - t0
        logger.info(
            "formula model loaded: model_id=%s device=%s dtype=%s torch_compile=%s "
            "elapsed_s=%.2f",
            model_id,
            device,
            dtype,
            torch_compile,
            load_elapsed,
        )

        if torch_compile:
            self._apply_torch_compile()

    @property
    def device(self) -> str:
        """Device string the model lives on (e.g. `"cuda:0"`)."""
        return self._device

    @property
    def max_new_tokens(self) -> int:
        """Per-crop generation cap configured at construction."""
        return self._max_new_tokens

    def recognize_batch_paths(
        self,
        crop_paths: Sequence[Path],
        *,
        batch_size: int = 8,
        max_new_tokens: int | None = None,
    ) -> list[FormulaResult]:
        """Recognize a list of formula crops; return one result per input path.

        The list is processed in `batch_size`-sized chunks; each chunk
        runs one forward pass and one `post_process` call. A chunk that raises during
        inference returns a MISS `FormulaResult` (`latex=None`, `truncated=False`) for
        every crop in that chunk. The next chunk continues unaffected.

        Per-crop failures (empty output, unreadable image) become a MISS `FormulaResult`
        for that single position; the rest of the chunk's results are returned normally.

        Args:
            crop_paths: Filesystem paths to the saved crops. Order is preserved in the
                output.
            batch_size: Crops per forward pass.
            max_new_tokens: Optional per-call override of the construction-time cap.
                `None` (default) uses the value passed to `__init__`. Useful for callers
                that bucket crops by expected output length and want to cap
                small-formula batches tighter than long-formula batches.

        Returns:
            One `FormulaResult` per input path, in the same order.
            Returns `[]` when `crop_paths` is empty.
        """
        if not crop_paths:
            return []

        if batch_size < 1:
            msg = f"batch_size must be >= 1, got {batch_size}"
            raise ValueError(msg)

        # Stash + restore the cap so the override is scoped to this call. `_run_chunk`
        # reads `self._max_new_tokens`; passing the override through would need plumbing
        # changes for negligible gain.
        original_cap = self._max_new_tokens
        if max_new_tokens is not None:
            if max_new_tokens < 1:
                msg = f"max_new_tokens must be >= 1, got {max_new_tokens}"
                raise ValueError(msg)

            self._max_new_tokens = max_new_tokens

        try:
            results: list[FormulaResult] = []
            for start in range(0, len(crop_paths), batch_size):
                chunk = list(crop_paths[start : start + batch_size])
                results.extend(self._run_chunk(chunk))

            return results
        finally:
            self._max_new_tokens = original_cap

    def recognize_batch_paths_bucketed(
        self,
        crop_paths: Sequence[Path],
        crop_areas: Sequence[float],
        *,
        batch_size: int = 8,
        bucket_spec: BucketSpec,
    ) -> list[FormulaResult]:
        """Recognize crops grouped into size buckets with per-bucket token caps.

        The motivating cost: when greedy generation runs over a batch, the whole batch
        decodes for as many steps as the slowest row needs. A batch of seven
        inline-formula crops bundled with one display equation pays the display
        equation's full token budget for all eight rows. Sorting crops by an
        output-length proxy (here: bbox area in source-page pixels**2) and processing
        same-size crops together collapses that overhead.

        Implementation:

        1. Walk `crop_areas` in lockstep with `crop_paths` and bucket each crop into
           `small` / `medium` / `large` according to the thresholds in `bucket_spec`.
        2. For each non-empty bucket, sort its members by area (smallest first) so the
           within-bucket spread is minimised, then run `recognize_batch_paths` on just
           those crops with the bucket's `max_new_tokens` cap.
        3. Splice the per-bucket results back into a single list indexed by the original
           `crop_paths` order -- callers see one result per input path in the order they
           passed them.

        Args:
            crop_paths: Filesystem paths to the saved crops. Returned results stay
                aligned with this order.
            crop_areas: Per-crop area in the same unit the thresholds in `bucket_spec`
                use (production: source-page pixels**2 computed from each block's bbox).
                Must have the same length as `crop_paths`.
            batch_size: Crops per forward pass within each bucket. Same semantics as
                `recognize_batch_paths`.
            bucket_spec: Bucket boundaries + per-bucket token caps. The `large` bucket
                inherits `max_new_tokens` from construction.

        Returns:
            One `FormulaResult` per input path in the same order.
            Returns `[]` for an empty input.

        Raises:
            ValueError: `crop_paths` and `crop_areas` have different lengths, or any
                input parameter is invalid.
        """
        if len(crop_paths) != len(crop_areas):
            msg = (
                f"len(crop_paths)={len(crop_paths)} must equal "
                f"len(crop_areas)={len(crop_areas)}"
            )
            raise ValueError(msg)
        if not crop_paths:
            return []

        # Partition inputs into the three buckets while remembering each crop's position
        # in the original sequence -- we splice results back into that order after
        # per-bucket processing.
        buckets: dict[str, list[tuple[int, Path, float]]] = {
            "small": [],
            "medium": [],
            "large": [],
        }
        for idx, (path, area) in enumerate(zip(crop_paths, crop_areas, strict=True)):
            if area < bucket_spec.small_threshold:
                buckets["small"].append((idx, path, area))
            elif area < bucket_spec.medium_threshold:
                buckets["medium"].append((idx, path, area))
            else:
                buckets["large"].append((idx, path, area))

        # Token cap per bucket. The large bucket reuses the construction cap so the
        # global `formula_max_new_tokens` is the single source of truth for the absolute
        # maximum.
        token_caps: dict[str, int] = {
            "small": bucket_spec.small_tokens,
            "medium": bucket_spec.medium_tokens,
            "large": self._max_new_tokens,
        }

        logger.info(
            "formula bucketed: small=%d medium=%d large=%d "
            "(small_cap=%d medium_cap=%d large_cap=%d)",
            len(buckets["small"]),
            len(buckets["medium"]),
            len(buckets["large"]),
            token_caps["small"],
            token_caps["medium"],
            token_caps["large"],
        )

        # Process each non-empty bucket; sorting by area inside the bucket keeps the
        # within-batch spread tight so the slowest row in a batch isn't far from the
        # fastest.
        results: list[FormulaResult | None] = [None] * len(crop_paths)
        for bucket_name in ("small", "medium", "large"):
            entries = buckets[bucket_name]
            if not entries:
                continue

            entries.sort(key=lambda e: e[2])
            paths_in_order = [e[1] for e in entries]
            bucket_results = self.recognize_batch_paths(
                paths_in_order,
                batch_size=batch_size,
                max_new_tokens=token_caps[bucket_name],
            )
            for (original_idx, _, _), result in zip(
                entries,
                bucket_results,
                strict=True,
            ):
                results[original_idx] = result

        # `_run_chunk` always returns a `FormulaResult` per input path so every slot in
        # `results` is populated by the time we get here. The cast is just to satisfy
        # the type checker.
        return cast("list[FormulaResult]", results)

    def close(self) -> None:
        """Move the model to CPU and release its VRAM.

        Idempotent: a second call after the model is released is a no-op. The instance
        is unusable for inference after the first successful call.
        """
        if getattr(self, "_closed", False):
            return

        try:
            self._model.to("cpu")
        except RuntimeError as exc:
            # The CUDA-side weights were already released (CUDA context lost, or the
            # caller mutated the model in a way that broke `.to`). Not actionable; `del`
            # below still drops the Python reference.
            logger.debug(
                "formula close: model.to('cpu') raised RuntimeError %r; assuming "
                "already-released weights",
                exc,
            )

        del self._model
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()
        self._closed = True
        logger.info("formula model closed: model_id=%s", self._model_id)

    def _run_chunk(self, paths: list[Path]) -> list[FormulaResult]:
        """Process one chunk of crops; never raises.

        Returns one result per input. A whole-chunk inference exception produces
        all-`None` results. Per-crop load failures (missing file, unreadable image)
        produce a single `None` result without affecting peers.
        """
        # Load each crop via the package's cv2 wrapper. Keep a parallel record of which
        # input slots succeeded so failed slots stay None in the output.
        loaded: list[tuple[int, np.ndarray]] = []
        for idx, path in enumerate(paths):
            try:
                loaded.append((idx, read_rgb(path)))
            except OSError:
                # `exc_info=True` keeps the underlying OSError (missing file vs.
                # permission vs. corrupt image) diagnosable.
                logger.warning(
                    "formula MISS: path=%s reason=load_failed",
                    path,
                    exc_info=True,
                )

        if not loaded:
            return [FormulaResult(latex=None, truncated=False)] * len(paths)

        loaded_images = [img for _, img in loaded]
        try:
            t0 = time.perf_counter()
            latex_per_loaded, truncated_per_loaded = self._forward(loaded_images)
            elapsed = time.perf_counter() - t0
        except Exception:  # noqa: BLE001 -- one chunk degrades to MISS; the next chunk continues
            # Recoverable: every block in this chunk degrades to MISS so the caller can
            # keep processing the next chunk.
            #
            # WARNING + exc_info surfaces the stack trace (CUDA OOM is the typical
            # cause).
            logger.warning(
                "formula batch failed: crops=%d paths=%s",
                len(loaded_images),
                [str(p) for p in paths],
                exc_info=True,
            )
            return [FormulaResult(latex=None, truncated=False)] * len(paths)

        logger.debug(
            "formula batch: size=%d elapsed_s=%.3f",
            len(loaded_images),
            elapsed,
        )

        # Splice loaded results back into the original positions.
        results: list[FormulaResult] = [
            FormulaResult(latex=None, truncated=False)
        ] * len(paths)
        for loaded_pos, (original_idx, _) in enumerate(loaded):
            latex = latex_per_loaded[loaded_pos]
            truncated = truncated_per_loaded[loaded_pos]
            if not latex:
                logger.warning(
                    "formula MISS: path=%s reason=empty_output",
                    paths[original_idx],
                )
                results[original_idx] = FormulaResult(latex=None, truncated=truncated)
            else:
                if truncated:
                    logger.warning(
                        "formula truncated: path=%s n_tokens=%d",
                        paths[original_idx],
                        self._max_new_tokens,
                    )
                results[original_idx] = FormulaResult(latex=latex, truncated=truncated)
        return results

    def _forward(self, images: list[np.ndarray]) -> tuple[list[str], list[bool]]:
        """Run one forward pass over a list of HWC uint8 RGB images.

        Returns a parallel pair of lists: the post-processed LaTeX per crop, and a
        per-crop `truncated` flag (`True` if generation stopped at `max_new_tokens`).
        Both lists have the same length as `images`.
        """
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)

        # The processor always returns fp32 pixel values; the patch-embed Conv2d refuses
        # fp32 input when the model runs in fp16.
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self._dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        self._cuda_sync()

        with torch.no_grad():
            tokens = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        self._cuda_sync()

        cleaned = _post_process_tokens(self._processor, tokens)

        # All rows in a batched call have the same shape[1] -- generation stops the
        # whole batch at once. Truncation is therefore a per-call flag, not per-row; we
        # replicate it across the batch for the caller's convenience.
        n_tokens = int(tokens.shape[1])
        is_truncated = n_tokens >= self._max_new_tokens

        return cleaned, [is_truncated] * len(images)

    def _cuda_sync(self) -> None:
        """Block until queued CUDA work completes; no-op on CPU."""
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()

    def _apply_torch_compile(self) -> None:
        """Wrap the model with `torch.compile` and warm up Inductor.

        The warmup runs a tiny synthetic image through the full forward pass so the
        encoder + decoder + generation loop all see their compile triggers fired before
        the first real request hits.
        """
        # `mode="default"` is the right balance for autoregressive generation --
        # `reduce-overhead` (CUDA Graphs) recompiles every time the decoder's KV-cache
        # grows, which is every token.
        t0 = time.perf_counter()
        self._model = torch.compile(self._model)
        logger.info("formula compile: torch.compile wrapped (mode=default)")

        warmup_image = np.full((64, 256, 3), 255, dtype=np.uint8)
        try:
            self._forward([warmup_image])
        except Exception:  # noqa: BLE001 -- compile warmup is opt-in; failure logs + degrades to lazy compile
            # Compile is opt-in; a failed warmup shouldn't take the whole service down.
            #
            # WARNING + exc_info surfaces the cause -- the recovery is silent (next
            # forward pass triggers compilation lazily anyway).
            logger.warning("formula compile: warmup failed; continuing", exc_info=True)
            return

        logger.info(
            "formula compile: warmup done elapsed_s=%.2f",
            time.perf_counter() - t0,
        )


def _post_process_tokens(
    processor: Any,  # noqa: ANN401 -- HF `PPFormulaNetProcessor` has no exported stub
    tokens: torch.Tensor,
) -> list[str]:
    """Run `processor.post_process` and normalize the return shape.

    The processor returns `list[str]` for batched input but the API is loose enough to
    also yield a bare string for single inputs. This helper always returns a list so
    callers can index by row.

    Args:
        processor: `PPFormulaNetProcessor` instance.
        tokens: Generated token IDs from `Model.generate`.

    Returns:
        Cleaned LaTeX strings, one per row of `tokens`.
    """
    result = processor.post_process(tokens)
    if isinstance(result, list):
        return result

    return [str(result)]


def make_formula_recognizer(cfg: PipelineConfig) -> FormulaRecognizer:
    """Build a `FormulaRecognizer` from a `PipelineConfig`.

    Thin wrapper that maps the formula-related fields on `PipelineConfig` to the
    recognizer's constructor; lets the FastAPI lifespan and the orchestrator share one
    factory.

    Args:
        cfg: Pipeline config carrying the formula knobs.

    Returns:
        A ready-to-use `FormulaRecognizer` instance.

    """
    return FormulaRecognizer(
        cfg.formula_model_id,
        device=cfg.formula_device,
        dtype=cfg.formula_dtype,
        max_new_tokens=cfg.formula_max_new_tokens,
        torch_compile=cfg.formula_torch_compile,
    )

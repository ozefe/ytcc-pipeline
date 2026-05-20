"""SafeTensors / HuggingFace Transformers backend for PP-DocLayoutV3."""
# pyright thinks float16, float32, device(), and from_numpy() are not exported from
# torch -- they are.
# pyright: reportPrivateImportUsage=false

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from ytcc_pipeline.image_io import read_rgb

from .layout import LayoutDetection

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["SafeTensorsLayoutAnalyzer"]

_INPUT_RESOLUTION_PX = 800

# Sentinel queue object signalling "no more batches" to the consumer in `_batches`.
# Module-level so identity comparison (`is _STOP`) works across `cast` calls and
# producer / consumer threads.
_STOP = object()


class SafeTensorsLayoutAnalyzer:
    """PP-DocLayoutV3 layout analyzer using HF Transformers on PyTorch.

    Two optional performance modes are layered onto the base path:

    - `fp16=True`: converts the model and inputs to half precision.
    - `fast_preproc=True`: replaces HF's `AutoImageProcessor` with a direct cv2 path AND
      overlaps preprocessing with GPU inference via a single producer thread.
    """

    def __init__(
        self,
        *,
        model_id: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
        device: str = "cuda:0",
        fp16: bool = False,
        fast_preproc: bool = False,
    ) -> None:
        """Load model + processor onto `device`.

        Args:
            model_id: HuggingFace repo id or local path of the SafeTensors model.
            device: Torch device string (e.g. `"cuda:0"`, `"cpu"`).
            fp16: Load + run the model in fp16.
            fast_preproc: Use the cv2 preprocessing pipeline overlapped with GPU
                inference.

        Raises:
            Exception: Re-raises any error from `AutoImageProcessor.from_pretrained` or
                `AutoModelForObjectDetection.from_pretrained` after logging at `ERROR`
                so the failure is loud at boot time.
        """
        self._device = torch.device(device)
        t0 = time.perf_counter()
        try:
            self._model = (
                AutoModelForObjectDetection.from_pretrained(model_id)
                .to(self._device)
                .eval()
            )
            self._processor = AutoImageProcessor.from_pretrained(model_id)
        except Exception:
            logger.exception("analyzer load failed: model_id=%s", model_id)
            raise

        self._id2label = self._model.config.id2label
        self._fast_preproc = fast_preproc
        if fp16:
            self._model.half()
        self._dtype = torch.float16 if fp16 else torch.float32

        logger.info(
            "analyzer ready: model=%s device=%s fp16=%s fast_preproc=%s "
            "load_elapsed_s=%.2f",
            model_id,
            device,
            fp16,
            fast_preproc,
            time.perf_counter() - t0,
        )

    def analyze(
        self,
        page_images: Sequence[Path],
        *,
        batch_size: int = 8,
        confidence: float = 0.5,
    ) -> dict[int, list[LayoutDetection]]:
        """Detect layout blocks on each page image.

        Args:
            page_images: Paths to rendered page images, in page order.
            batch_size: Number of pages per GPU batch.
            confidence: Minimum detection confidence; lower scores are dropped.

        Returns:
            ```json
                {
                    page_index: [
                        LayoutDetection,
                        ...
                    ]
                }```

            With detections sorted by per-page reading order.
        """
        logger.debug(
            "analyze start: pages=%d batch_size=%d confidence=%.2f fast_preproc=%s",
            len(page_images),
            batch_size,
            confidence,
            self._fast_preproc,
        )

        results: dict[int, list[LayoutDetection]] = {}
        for chunk_start, target_sizes, pixel_values in self._batches(
            page_images,
            batch_size,
        ):
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self._model(pixel_values=pixel_values)
            post = self._processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=confidence,
            )

            batch_detections = 0
            for offset, result in enumerate(post):
                detections = self._build_detections(result)
                results[chunk_start + offset] = detections
                batch_detections += len(detections)

            logger.debug(
                "analyze batch: start=%d size=%d detections=%d elapsed_s=%.3f",
                chunk_start,
                len(target_sizes),
                batch_detections,
                time.perf_counter() - t0,
            )

        return results

    def close(self) -> None:
        """Release the model + processor and free CUDA memory.

        Called by the FastAPI scanned-path handler to make room for the OCR worker
        engines, and by `process_pdf` in library mode at the end of each call when the
        analyzer is owned per-call. Idempotent: a second call is a no-op.
        """
        if getattr(self, "_closed", False):
            return

        del self._model
        del self._processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._closed = True

        logger.info("analyzer closed: device=%s", self._device)

    # ---- internals -------------------------------------------------------

    def _batches(
        self,
        page_images: Sequence[Path],
        batch_size: int,
    ) -> Iterator[tuple[int, list[tuple[int, int]], torch.Tensor]]:
        """Yield `(chunk_start, target_sizes, pixel_values)` per GPU batch.

        With `fast_preproc`, a daemon thread runs preprocessing one batch ahead so the
        GPU stays busy. Without it, preprocessing happens inline.
        """
        chunks = [
            (i, list(page_images[i : i + batch_size]))
            for i in range(0, len(page_images), batch_size)
        ]
        if not self._fast_preproc:
            for start, paths in chunks:
                target_sizes, pixel_values = self._preprocess(paths)
                yield start, target_sizes, pixel_values
            return

        # Queue holds either a prepared batch `(start, target_sizes, tensor)` or the
        # `_STOP` sentinel; type as `object` since the union is opaque.
        preprocess_queue: queue.Queue[object] = queue.Queue(maxsize=2)

        def producer() -> None:
            for start, paths in chunks:
                preprocess_queue.put((start, *self._preprocess(paths)))
            preprocess_queue.put(_STOP)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        try:
            while (item := preprocess_queue.get()) is not _STOP:
                # `item` is a `(start, target_sizes, pixel_values)` triple whenever it's
                # not the sentinel -- see `producer` above.
                yield cast("tuple[int, list[tuple[int, int]], torch.Tensor]", item)
        finally:
            thread.join()

    def _preprocess(
        self,
        paths: Sequence[Path],
    ) -> tuple[list[tuple[int, int]], torch.Tensor]:
        if self._fast_preproc:
            blobs: list[np.ndarray] = []
            target_sizes: list[tuple[int, int]] = []
            for p in paths:
                img = read_rgb(p)
                target_sizes.append((img.shape[0], img.shape[1]))
                img = cv2.resize(
                    img,
                    (_INPUT_RESOLUTION_PX, _INPUT_RESOLUTION_PX),
                    interpolation=cv2.INTER_LINEAR,
                )
                blobs.append((img.astype(np.float32) / 255.0).transpose(2, 0, 1))

            tensor = torch.from_numpy(np.ascontiguousarray(np.stack(blobs))).to(
                self._device,
                dtype=self._dtype,
            )
            return target_sizes, tensor

        # HF's AutoImageProcessor accepts numpy arrays alongside PIL.Image, so we read
        # via `read_rgb` and skip the PIL dep entirely. Normalisation (mean/std) is
        # applied inside the processor, identical to the PIL-input path.
        images = [read_rgb(p) for p in paths]
        target_sizes = [(img.shape[0], img.shape[1]) for img in images]
        inputs = self._processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device, dtype=self._dtype)

        return target_sizes, pixel_values

    def _build_detections(self, result: dict[str, Any]) -> list[LayoutDetection]:
        scores = result["scores"].float().cpu().tolist()
        labels = result["labels"].cpu().tolist()
        boxes = result["boxes"].float().cpu().tolist()
        order_seq = result.get("order_seq")

        # `order_seq` ranks detections by reading position; double-argsort converts the
        # rank tensor into 0..N-1 positions per detection.
        ranks = (
            order_seq.argsort().argsort().cpu().tolist()
            if order_seq is not None
            else list(range(len(scores)))
        )

        items = [
            LayoutDetection(
                label=self._id2label.get(label_id, f"class_{label_id}"),
                label_id=int(label_id),
                confidence=float(score),
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                reading_order=int(rank),
            )
            for score, label_id, box, rank in zip(
                scores,
                labels,
                boxes,
                ranks,
                strict=True,
            )
        ]
        items.sort(key=lambda d: d.reading_order)
        return items

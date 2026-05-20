"""PP-FormulaNet-L stage -- runs on the main process after blocks.

Reads every FORMULA block crop saved by the block stage, batches them through
`FormulaRecognizer` (bucketed or flat), splices the recognized LaTeX back into the page
list, and handles crop lifecycle:

- Success: LaTeX in `text`, on-disk crop deleted, `image_path` cleared.
- MISS: crop renamed with `-MISS-` marker, `image_path` repointed, `miss=True`.

Mirrors the text-block convention: the bundle ships exactly one representation per
formula block -- either LaTeX text or a fallback image, never both.
"""

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

from ytcc_pipeline.models.formula import BucketSpec, FormulaRecognizer, FormulaResult
from ytcc_pipeline.pipeline.blocks import replace_blocks
from ytcc_pipeline.schema import Block, BlockType, Page

from .image import add_miss_marker_to_filename

if TYPE_CHECKING:
    from pathlib import Path

    from ytcc_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class _FormulaTarget(NamedTuple):
    """One FORMULA block queued for recognition.

    `page_idx` / `block_idx` locate the block in the page list so the recognized LaTeX
    can be spliced back into the original slot. `crop_path` is the absolute path of the
    saved crop (worker output). `bbox_area` is in source-page px**2 -- bucketed batching
    reads it to estimate output length and route to a small / medium / large bucket.
    """

    page_idx: int
    block_idx: int
    crop_path: Path
    bbox_area: float


def run_formula_stage(
    pages: list[Page],
    *,
    work_dir: Path,
    cfg: PipelineConfig,
    formula_recognizer: FormulaRecognizer | None,
    pdf_name: str,
) -> list[Page]:
    """Fill in LaTeX text on every FORMULA block via PP-FormulaNet-L.

    Skip cases (`formula_enabled=False`, no recognizer, no FORMULA blocks) leave the
    page list, and the saved crops, unchanged.

    Args:
        pages: Pages assembled by the block stage. FORMULA blocks come in with
            `image_path` set and `text=None`.
        work_dir: Pipeline temp dir. Used to resolve bundle-relative `image_path` values
            (`"images/..."`) to absolute paths on disk so we can delete or rename them.
        cfg: Pipeline config. `formula_enabled`, `formula_bucketed`,
            `formula_batch_size`, and the per-bucket knobs drive this stage.
        formula_recognizer: Loaded recognizer (injected by the service) or `None`. When
            `None` and `cfg.formula_enabled` is true the caller is expected to skip this
            call entirely -- we defensively no-op in that case as well.
        pdf_name: Source PDF filename for log correlation.

    Returns:
        The page list with FORMULA blocks updated. When the stage is a no-op the
        original list is returned unchanged.
    """
    if not cfg.formula_enabled:
        logger.info("stage formula: pdf=%s skipped reason=disabled", pdf_name)
        return pages

    if formula_recognizer is None:
        logger.info("stage formula: pdf=%s skipped reason=no_recognizer", pdf_name)
        return pages

    targets = _collect_formula_targets(pages, work_dir)
    if not targets:
        logger.info("stage formula: pdf=%s skipped reason=no_blocks", pdf_name)
        return pages

    logger.info(
        "stage formula start: pdf=%s crops=%d batch_size=%d bucketed=%s",
        pdf_name,
        len(targets),
        cfg.formula_batch_size,
        cfg.formula_bucketed,
    )

    t0 = time.perf_counter()
    results = _recognize(formula_recognizer, targets, cfg)

    # Page -> mutable list of blocks; replace the FORMULA slots with the recognized
    # LaTeX (or flip to miss=True when recognition failed). The cleanup helpers handle
    # the on-disk crops -- drop on success, rename with -MISS- on failure (when
    # `BlockType.FORMULA in cfg.bundle_miss_images_for`).
    blocks_by_page: dict[int, list[Block]] = {
        idx: list(page.blocks) for idx, page in enumerate(pages)
    }
    keep_miss_image = BlockType.FORMULA in cfg.bundle_miss_images_for
    miss_count = 0
    truncated_count = 0
    deleted_count = 0
    for target, result in zip(targets, results, strict=True):
        original = blocks_by_page[target.page_idx][target.block_idx]
        page_no = pages[target.page_idx].page_no

        if result.latex:
            if _drop_success_crop(target.crop_path):
                deleted_count += 1

            if result.truncated:
                truncated_count += 1

            blocks_by_page[target.page_idx][target.block_idx] = replace(
                original,
                text=result.latex,
                image_path=None,
                miss=False,
            )
            logger.debug(
                "formula block: pdf=%s page=%d reading_order=%d label=%s chars=%d "
                "truncated=%s",
                pdf_name,
                page_no,
                original.reading_order,
                original.label,
                len(result.latex),
                result.truncated,
            )
        else:
            miss_count += 1
            # Workers always set `image_path` for FORMULA blocks (the crop is saved
            # upfront so the formula stage can rename or delete it). Assert documents
            # the invariant for pyright.
            assert original.image_path is not None  # noqa: S101
            new_image_path = _handle_miss_crop(
                target.crop_path,
                keep_image=keep_miss_image,
                fallback_image_path=original.image_path,
            )
            blocks_by_page[target.page_idx][target.block_idx] = replace(
                original,
                text=None,
                image_path=new_image_path,
                miss=True,
            )
            logger.debug(
                "formula block: pdf=%s page=%d reading_order=%d label=%s miss=True "
                "image_kept=%s",
                pdf_name,
                page_no,
                original.reading_order,
                original.label,
                new_image_path is not None,
            )

    logger.info(
        "stage formula: pdf=%s formulas=%d miss=%d truncated=%d "
        "crops_dropped=%d elapsed_s=%.2f",
        pdf_name,
        len(targets),
        miss_count,
        truncated_count,
        deleted_count,
        time.perf_counter() - t0,
    )
    return replace_blocks(pages, blocks_by_page)


def _collect_formula_targets(pages: list[Page], work_dir: Path) -> list[_FormulaTarget]:
    """Walk pages and emit one `_FormulaTarget` per FORMULA block with a saved crop.

    Args:
        pages: Pages emitted by the block stage.
        work_dir: Pipeline temp dir. Used to resolve bundle-relative `image_path`
            strings to absolute on-disk paths.

    Returns:
        A list of targets in page-then-block order. `bbox_area` is computed in
        source-page px**2 so the bucketed-batching path can route each crop to a small /
        medium / large bucket without a second pass.
    """
    targets: list[_FormulaTarget] = []
    for page_idx, page in enumerate(pages):
        for block_idx, block in enumerate(page.blocks):
            if block.type is BlockType.FORMULA and block.image_path:
                x1, y1, x2, y2 = block.bbox
                targets.append(
                    _FormulaTarget(
                        page_idx=page_idx,
                        block_idx=block_idx,
                        crop_path=work_dir / block.image_path,
                        bbox_area=max(0.0, (x2 - x1) * (y2 - y1)),
                    )
                )
    return targets


def _recognize(
    recognizer: FormulaRecognizer,
    targets: list[_FormulaTarget],
    cfg: PipelineConfig,
) -> list[FormulaResult]:
    """Dispatch crops to bucketed or flat batched recognition per `cfg`.

    Args:
        recognizer: Loaded `FormulaRecognizer`.
        targets: Recognition targets in input order.
        cfg: Pipeline config. `formula_bucketed`, `formula_batch_size`, and the
            per-bucket knobs drive the dispatch.

    Returns:
        One `FormulaResult` per target, in the same order.
    """
    crop_paths = [t.crop_path for t in targets]
    if not cfg.formula_bucketed:
        return recognizer.recognize_batch_paths(
            crop_paths,
            batch_size=cfg.formula_batch_size,
        )

    # Bucket spec is built from PipelineConfig so the recognizer call stays decoupled
    # from TOML / env-var layout. Each scalar field on cfg has a corresponding slot on
    # BucketSpec; tuning guidance lives alongside the PipelineConfig field definitions.
    crop_areas = [t.bbox_area for t in targets]
    return recognizer.recognize_batch_paths_bucketed(
        crop_paths,
        crop_areas,
        batch_size=cfg.formula_batch_size,
        bucket_spec=BucketSpec(
            small_threshold=cfg.formula_bucket_small_threshold,
            medium_threshold=cfg.formula_bucket_medium_threshold,
            small_tokens=cfg.formula_bucket_small_tokens,
            medium_tokens=cfg.formula_bucket_medium_tokens,
        ),
    )


def _drop_success_crop(crop_path: Path) -> bool:
    """Delete a successfully-recognized crop; never raises.

    Args:
        crop_path: Absolute path to the crop on disk.

    Returns:
        `True` when the file was unlinked (or was already gone), `False` when an
        `OSError` blocked the delete. The bundle still ships in either case -- an
        undeletable crop is a small operational leak, not a correctness bug.
    """
    try:
        crop_path.unlink(missing_ok=True)
    except OSError as exc:
        # `exc_info=True` attaches the OSError traceback so operators can tell
        # permission / read-only / ENOSPC apart at debug time.
        logger.warning(
            "formula cleanup: failed to delete %s reason=%s",
            crop_path,
            exc,
            exc_info=True,
        )
        return False
    return True


def _handle_miss_crop(
    crop_path: Path,
    *,
    keep_image: bool,
    fallback_image_path: str,
) -> str | None:
    """Rename a MISS crop with the `-MISS-` marker, or delete it; never raises.

    Args:
        crop_path: Absolute path to the speculatively-saved crop on disk.
        keep_image: When `True` the crop is renamed with a `-MISS-` marker so it ships
            in the bundle as an image fallback. When `False` the crop is deleted.
        fallback_image_path: Bundle-relative path to return if the rename fails (keeps
            the block's `image_path` consistent with on-disk state).

    Returns:
        The bundle-relative path the block's `image_path` should be set to. `None`
        means no fallback image is bundled (caller sets `image_path=None`).
    """
    if not keep_image:
        try:
            crop_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "formula cleanup: failed to delete %s reason=%s",
                crop_path,
                exc,
                exc_info=True,
            )
        return None

    miss_filename = add_miss_marker_to_filename(crop_path.name)
    try:
        crop_path.rename(crop_path.with_name(miss_filename))
    except OSError as exc:
        # If the rename fails the original filename stays valid; we just won't get the
        # `-MISS-` marker. Surface it but keep the block's image_path consistent with
        # the on-disk state.
        logger.warning(
            "formula cleanup: failed to rename %s reason=%s",
            crop_path,
            exc,
            exc_info=True,
        )
        return fallback_image_path
    return f"images/{miss_filename}"

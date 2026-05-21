"""Reference stage: enrich reference-labeled blocks with parsed citations.

Runs after the per-page workers have populated `Block.text` for every reference-shaped
block. The stage:

1. Collects `(page_idx, block_idx, text)` for every block whose label is in
   `cfg.reference_labels` and whose text is non-empty.
2. Sends every text in one batched HTTP call to GROBID's `/api/processCitationList`.
3. Replaces each block in place with a copy that carries the parsed `Reference`. Blocks
   GROBID couldn't parse keep `reference=None`; the raw reference string always survives
   on `Block.text`.

Failure modes -- server unreachable, timeout, HTTP error, malformed payload -- are
logged at WARNING and the pages flow through unchanged. References are an enrichment;
pipeline correctness doesn't depend on them.
"""

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from ytcc_pipeline.models.grobid import GrobidClient, GrobidError
from ytcc_pipeline.pipeline.blocks import replace_blocks

if TYPE_CHECKING:
    from pathlib import Path

    from ytcc_pipeline.config import PipelineConfig
    from ytcc_pipeline.schema import Block, Page, Reference

__all__ = ["run_reference_stage"]

logger = logging.getLogger(__name__)


class _CitationClient(Protocol):
    """Subset of `GrobidClient` the stage actually needs.

    Declaring this `Protocol` makes the `client=` injection point testable with a small
    fake and keeps the production wrapper decoupled from the stage's call shape.
    """

    def process_citation_list(self, citations: list[str]) -> list[Reference | None]: ...


def run_reference_stage(  # noqa: C901  -- single-pass: skip-checks, collect refs, one POST, splice results; splitting threads more state
    pages: list[Page],
    *,
    pdf_path: Path,
    cfg: PipelineConfig,
    client: _CitationClient | None = None,
) -> list[Page]:
    """Attach parsed `Reference` objects to reference-labeled blocks.

    Args:
        pages: Pages with text already extracted.
        pdf_path: Source PDF (used only for log context).
        cfg: Pipeline knobs; gated on `cfg.references_enabled`.
        client: Optional pre-built citation client. When `None` (the production default)
            and the stage is enabled, a fresh `GrobidClient` is built from
            cfg.grobid_url` / `cfg.grobid_timeout_s`.

    Returns:
        Pages with `Block.reference` populated where parses succeeded.
        Returns `pages` unchanged when the stage is disabled, when
        `cfg.reference_labels` is empty, when no matching blocks exist, or when GROBID
        raises.
    """
    pdf_name = pdf_path.name

    if not cfg.references_enabled:
        logger.info("stage reference: pdf=%s skipped reason=disabled", pdf_name)
        return pages

    label_allow = frozenset(cfg.reference_labels)
    if not label_allow:
        logger.info("stage reference: pdf=%s skipped reason=no_labels", pdf_name)
        return pages

    targets: list[tuple[int, int, str]] = []
    for page_idx, page in enumerate(pages):
        for block_idx, block in enumerate(page.blocks):
            if block.label not in label_allow:
                continue

            text = (block.text or "").strip()
            if not text:
                continue

            targets.append((page_idx, block_idx, text))

    if not targets:
        logger.info("stage reference: pdf=%s skipped reason=no_blocks", pdf_name)
        return pages

    if client is None:
        client = GrobidClient(url=cfg.grobid_url, timeout_s=cfg.grobid_timeout_s)

    citations = [text for _, _, text in targets]
    logger.info(
        "stage reference start: pdf=%s refs=%d labels=%s url=%s",
        pdf_name,
        len(citations),
        sorted(label_allow),
        cfg.grobid_url,
    )

    t0 = time.perf_counter()
    try:
        parsed = client.process_citation_list(citations)
    except GrobidError as exc:
        # `exc_info=True` keeps the chained transport / parse cause (`__cause__`) on the
        # traceback so operators can tell DNS / refusal / timeout / malformed-XML apart
        # without re-running the request.
        logger.warning(
            "stage reference: pdf=%s skipped reason=grobid_error url=%s err=%s",
            pdf_name,
            cfg.grobid_url,
            exc,
            exc_info=True,
        )
        return pages

    # Replace blocks by page so frozen-dataclass `replace(page, blocks=...)` only fires
    # for pages that actually changed.
    blocks_by_page: dict[int, list[Block]] = {
        idx: list(p.blocks) for idx, p in enumerate(pages)
    }
    enriched = 0
    unparsed = 0
    for (page_idx, block_idx, _text), reference in zip(targets, parsed, strict=True):
        if reference is None:
            unparsed += 1
            continue

        original = blocks_by_page[page_idx][block_idx]
        blocks_by_page[page_idx][block_idx] = replace(original, reference=reference)
        enriched += 1

    logger.info(
        "stage reference: pdf=%s refs=%d enriched=%d unparsed=%d elapsed_s=%.2f",
        pdf_name,
        len(citations),
        enriched,
        unparsed,
        time.perf_counter() - t0,
    )
    return replace_blocks(pages, blocks_by_page)

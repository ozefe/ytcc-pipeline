"""Map PP-DocLayoutV3 label names to processing routes."""

import logging
from enum import StrEnum

__all__ = [
    "FORMULA_LABELS",
    "IGNORE_LABELS",
    "IMAGE_LABELS",
    "REFERENCE_LABELS",
    "TABLE_LABELS",
    "TEXT_LABELS",
    "Route",
    "route_for",
]

logger = logging.getLogger(__name__)

# Module-level set of unknown labels already warned about -- keeps the fallback warning
# at one line per distinct label per process instead of one per detection (which would
# spam hundreds of lines for a label that appears on every page).
_warned_unknown_labels: set[str] = set()

TEXT_LABELS: frozenset[str] = frozenset(
    {
        "abstract",
        "aside_text",
        "content",
        "doc_title",
        "figure_title",
        "footnote",
        "formula_number",  # "(3.2)"-style numbering -- short text, not a real formula
        "paragraph_title",
        "text",
        "vertical_text",
        "vision_footnote",
    }
)

IMAGE_LABELS: frozenset[str] = frozenset(
    {
        "algorithm",
        "chart",
        "image",
    }
)

# Tables go through the table stage (RapidTable structure recognition + per-cell text
# extraction). Kept distinct from IMAGE_LABELS so the stage can recover the cell grid
# instead of just saving the crop.
TABLE_LABELS: frozenset[str] = frozenset({"table"})

# Formula labels go through PP-FormulaNet-L to recover LaTeX text while still saving the
# original crop as a fallback. `formula_number` is NOT in this set -- equation numbering
# like "(3.2)" is plain text, faster (and more accurate) to extract via pdf_oxide or
# RapidOCR than to push through a 700 MB transformer.
FORMULA_LABELS: frozenset[str] = frozenset(
    {
        "display_formula",
        "formula",
        "inline_formula",
    }
)

# Bibliography labels are routed separately so callers can render references distinctly
# from body text downstream.
REFERENCE_LABELS: frozenset[str] = frozenset({"reference", "reference_content"})

IGNORE_LABELS: frozenset[str] = frozenset(
    {
        "footer",
        "footer_image",
        "header",
        "header_image",
        "number",
        "seal",
    }
)


class Route(StrEnum):
    """The six ways a layout block can be processed."""

    TEXT = "text"
    IMAGE = "image"
    REFERENCE = "reference"
    IGNORE = "ignore"
    FORMULA = "formula"
    TABLE = "table"


_ROUTE_BY_LABEL: dict[str, Route] = {
    label: route
    for labels, route in (
        (TEXT_LABELS, Route.TEXT),
        (IMAGE_LABELS, Route.IMAGE),
        (FORMULA_LABELS, Route.FORMULA),
        (REFERENCE_LABELS, Route.REFERENCE),
        (TABLE_LABELS, Route.TABLE),
        (IGNORE_LABELS, Route.IGNORE),
    )
    for label in labels
}


def route_for(label: str) -> Route:
    """Return the processing route for a layout label.

    Unknown labels -- anything the model emits that we haven't catalogued -- fall back
    to `Route.IMAGE` so the block is at least preserved, and the miss is logged so the
    mapping can be extended.

    Args:
        label: A label string as returned by the layout model.

    Returns:
        The `Route` to apply to blocks with this label.
    """
    route = _ROUTE_BY_LABEL.get(label)
    if route is not None:
        return route

    if label not in _warned_unknown_labels:
        _warned_unknown_labels.add(label)
        logger.warning("unknown layout label %r -- routing to IMAGE", label)
    else:
        # Subsequent detections of an already-warned label stay quiet at WARNING. DEBUG
        # keeps the per-detection trail available for operators who turn the logger up.
        logger.debug(
            "unknown layout label %r -- routing to IMAGE (already warned)",
            label,
        )

    return Route.IMAGE

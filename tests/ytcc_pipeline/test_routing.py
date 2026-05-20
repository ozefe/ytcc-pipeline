"""Tests for ytcc_pipeline.routing."""

import logging
from itertools import combinations

import pytest

from ytcc_pipeline import routing
from ytcc_pipeline.routing import (
    FORMULA_LABELS,
    IGNORE_LABELS,
    IMAGE_LABELS,
    REFERENCE_LABELS,
    TABLE_LABELS,
    TEXT_LABELS,
    Route,
    route_for,
)

_LABEL_SETS: dict[str, frozenset[str]] = {
    "TEXT": TEXT_LABELS,
    "IMAGE": IMAGE_LABELS,
    "FORMULA": FORMULA_LABELS,
    "REFERENCE": REFERENCE_LABELS,
    "TABLE": TABLE_LABELS,
    "IGNORE": IGNORE_LABELS,
}

_SET_TO_ROUTE: dict[str, Route] = {
    "TEXT": Route.TEXT,
    "IMAGE": Route.IMAGE,
    "FORMULA": Route.FORMULA,
    "REFERENCE": Route.REFERENCE,
    "TABLE": Route.TABLE,
    "IGNORE": Route.IGNORE,
}

# Every (label, expected_route) pair, materialised as test parameters so each label
# becomes its own test case with a readable ID.
_LABEL_ROUTE_CASES: list[pytest.param] = [  # pyright: ignore[reportGeneralTypeIssues]
    pytest.param(label, _SET_TO_ROUTE[set_name], id=f"{set_name}-{label}")
    for set_name, labels in _LABEL_SETS.items()
    for label in labels
]

# Every (set_name_a, set_name_b) unordered pair.
_SET_PAIRS: list[pytest.param] = [  # pyright: ignore[reportGeneralTypeIssues]
    pytest.param(name_a, name_b, id=f"{name_a}-vs-{name_b}")
    for name_a, name_b in combinations(_LABEL_SETS, 2)
]


@pytest.fixture
def reset_unknown_label_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty the module-level `_warned_unknown_labels` set for the test.

    `route_for` deduplicates the WARNING for each unknown label after the first
    sighting; subsequent calls log at DEBUG instead. Tests that assert on the WARNING
    must start with an empty cache to be order-independent.
    """
    monkeypatch.setattr(routing, "_warned_unknown_labels", set())


@pytest.mark.parametrize(("name_a", "name_b"), _SET_PAIRS)
def test_label_sets_are_pairwise_disjoint(name_a: str, name_b: str) -> None:
    """Each label appears in exactly one of the six routing sets.

    A label landing in two sets would silently shadow one route with another at module
    import; parametrising over every unordered pair pinpoints exactly which two sets
    collided when the assertion fails.
    """
    overlap = _LABEL_SETS[name_a] & _LABEL_SETS[name_b]
    assert overlap == set(), f"{name_a} and {name_b} both contain {sorted(overlap)}"


@pytest.mark.parametrize(("label", "expected"), _LABEL_ROUTE_CASES)
def test_label_routes_to_its_declared_destination(
    label: str,
    expected: Route,
) -> None:
    """Every label in each set must `route_for` back to its set's route.

    Catches accidental drift where a label is removed from its set without updating the
    dispatch dict, or vice versa. The parametrised test ID names the originating set so
    failures pinpoint the drifted label.
    """
    assert route_for(label) is expected


def test_formula_number_routes_to_text_not_formula() -> None:
    """Equation numbering goes through the text path, not the model.

    A `(3.2)`-style numbering label carries plain text; routing it through PP-FormulaNet
    would be 100-200x slower than pdf_oxide / RapidOCR for the same answer.
    """
    assert route_for("formula_number") is Route.TEXT


def test_unknown_label_routes_to_image_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
    reset_unknown_label_cache: None,  # noqa: ARG001
) -> None:
    """First sighting of an unknown label routes to IMAGE and logs WARNING once."""
    with caplog.at_level(logging.WARNING, logger="ytcc_pipeline.routing"):
        assert route_for("brand_new_label") is Route.IMAGE

    assert any("brand_new_label" in r.message for r in caplog.records)


def test_unknown_label_warning_dedupes_after_first_sighting(
    caplog: pytest.LogCaptureFixture,
    reset_unknown_label_cache: None,  # noqa: ARG001
) -> None:
    """A label already warned about drops to DEBUG on subsequent calls."""
    # Prime the cache.
    route_for("already_seen_label")

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="ytcc_pipeline.routing"):
        for _ in range(5):
            assert route_for("already_seen_label") is Route.IMAGE

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []

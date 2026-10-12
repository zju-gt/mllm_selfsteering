"""Paired steering accuracy and flip statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def accuracy_by_alpha(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[float, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[float(row["alpha"])].append(bool(row["correct"]))
    accuracy = {alpha: sum(values) / len(values) for alpha, values in grouped.items()}
    baseline = accuracy.get(0.0)
    return {
        str(alpha): {
            "count": len(grouped[alpha]),
            "accuracy": value,
            "delta": value - baseline if baseline is not None else None,
        }
        for alpha, value in sorted(accuracy.items())
    }


def flip_report(
    rows: Iterable[Mapping[str, Any]], *, alpha: float
) -> dict[str, int] | None:
    by_item: dict[str, dict[float, bool]] = defaultdict(dict)
    for row in rows:
        by_item[str(row["item_identity"])][float(row["alpha"])] = bool(row["correct"])
    pairs = [values for values in by_item.values() if 0.0 in values and alpha in values]
    if not pairs:
        return None
    wrong_to_right = sum(not values[0.0] and values[alpha] for values in pairs)
    right_to_wrong = sum(values[0.0] and not values[alpha] for values in pairs)
    return {
        "paired_items": len(pairs),
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "net_flips": wrong_to_right - right_to_wrong,
    }


def assert_complete_population(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_items: Mapping[str, Sequence[str]],
    capabilities: Sequence[str],
    alphas: Sequence[float],
) -> None:
    observed = {
        (
            str(row["dataset"]),
            str(row["item_identity"]),
            str(row["steering_capability"]),
            float(row["alpha"]),
        )
        for row in rows
        if row.get("status") == "ok"
    }
    expected = {
        (dataset, identity, capability, float(alpha))
        for dataset, identities in expected_items.items()
        for identity in identities
        for capability in capabilities
        for alpha in alphas
    }
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        sample = sorted(missing)[:3]
        raise ValueError(
            f"generation population is incomplete: missing={len(missing)}, "
            f"extra={len(extra)}, missing_sample={sample}"
        )


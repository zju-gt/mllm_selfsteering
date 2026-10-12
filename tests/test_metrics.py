import pytest

from mllm_self_steering.evaluation.metrics import (
    accuracy_by_alpha,
    assert_complete_population,
    flip_report,
)


ROWS = [
    {
        "dataset": "d",
        "item_identity": "i",
        "steering_capability": "c",
        "alpha": 0.0,
        "correct": False,
        "status": "ok",
    },
    {
        "dataset": "d",
        "item_identity": "i",
        "steering_capability": "c",
        "alpha": 1.0,
        "correct": True,
        "status": "ok",
    },
]


def test_metrics_report_delta_and_flips() -> None:
    metrics = accuracy_by_alpha(ROWS)
    assert metrics["1.0"]["delta"] == 1.0
    assert flip_report(ROWS, alpha=1.0) == {
        "paired_items": 1,
        "wrong_to_right": 1,
        "right_to_wrong": 0,
        "net_flips": 1,
    }


def test_population_check_rejects_missing_capability() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        assert_complete_population(
            ROWS,
            expected_items={"d": ["i"]},
            capabilities=["c", "missing"],
            alphas=[0.0, 1.0],
        )


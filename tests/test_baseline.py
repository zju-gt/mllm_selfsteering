from pathlib import Path

import pytest

from mllm_self_steering.baseline import (
    _metrics,
    baseline_payload,
    validate_baseline_resume,
)
from mllm_self_steering.datasets.types import MultimodalItem


def test_baseline_resume_rejects_changed_experiment(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline_1025_1200"
    existing = baseline_payload(
        {"model": {"name": "model-a"}},
        baseline_dir,
        experiment_fingerprint="fingerprint-a",
        status="running",
    )
    changed = baseline_payload(
        {"model": {"name": "model-b"}},
        baseline_dir,
        experiment_fingerprint="fingerprint-b",
        status="running",
    )

    with pytest.raises(ValueError, match="experiment_fingerprint"):
        validate_baseline_resume(existing, changed, baseline_dir)


def test_baseline_metrics_report_coverage_and_accuracy() -> None:
    first = MultimodalItem(
        item_id="one",
        dataset="mmstar",
        split="val",
        prompt="Question one",
        image_paths=["one.png"],
        gold_answer="A",
        answer_type="choice",
        choices={"A": "yes", "B": "no"},
    )
    second = MultimodalItem(
        item_id="two",
        dataset="mmstar",
        split="val",
        prompt="Question two",
        image_paths=["two.png"],
        gold_answer="B",
        answer_type="choice",
        choices={"A": "yes", "B": "no"},
    )
    records = [(first, [], "id-one"), (second, [], "id-two")]
    successful = {
        ("mmstar", "id-one"): {"baseline_correct": True},
    }

    metrics = _metrics(records, successful)

    assert metrics["coverage"] == {
        "expected": 2,
        "successful": 1,
        "missing": 1,
        "complete": False,
    }
    assert metrics["overall"]["accuracy"] == 1.0
    assert metrics["datasets"]["mmstar"]["accuracy"] == 1.0

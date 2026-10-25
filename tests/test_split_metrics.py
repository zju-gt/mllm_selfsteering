import json
from pathlib import Path

import pytest

from mllm_self_steering.split_metrics import calculate_split_metrics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _item(
    dataset: str,
    item_id: str,
    *,
    answer_type: str = "choice",
    metadata: dict | None = None,
) -> dict:
    return {
        "dataset": dataset,
        "item_id": item_id,
        "answer_type": answer_type,
        "metadata": metadata or {},
    }


def _generation(
    dataset: str,
    item_id: str,
    baseline: bool,
    steering: bool,
) -> dict:
    return {
        "dataset": dataset,
        "item_id": item_id,
        "item_identity": f"identity-{dataset}-{item_id}",
        "baseline_correct": baseline,
        "correct": steering,
        "steering_applied": True,
        "status": "ok",
    }


def _fixture(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    canonical = {
        "mathvista": [
            _item("mathvista", "mv-oe", answer_type="math"),
            _item("mathvista", "mv-mc"),
        ],
        "mathvision": [
            _item("mathvision", "mvision-oe", answer_type="math"),
            _item("mathvision", "mvision-mc"),
        ],
        "mmstar": [
            _item("mmstar", "star-p", metadata={"category": "coarse perception"}),
            _item("mmstar", "star-r", metadata={"category": "logical reasoning"}),
            _item("mmstar", "star-m", metadata={"category": "math"}),
        ],
        "mmmu": [
            _item("mmmu", "mmmu-s", metadata={"subject": "Physics"}),
            _item("mmmu", "mmmu-h", metadata={"subject": "Literature"}),
        ],
    }
    for dataset, rows in canonical.items():
        _write_jsonl(data_root / "processed" / f"{dataset}.jsonl", rows)

    generation_dir = tmp_path / "outputs" / "generation_1025_1200"
    generation_dir.mkdir(parents=True)
    config = {
        "config": {
            "data": {"enabled_datasets": list(canonical)},
            "experiment": {"paths": {"data_dir": str(data_root)}},
        }
    }
    (generation_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rows = [
        _generation("mathvista", "mv-oe", False, True),
        _generation("mathvista", "mv-mc", True, True),
        _generation("mathvision", "mvision-oe", False, False),
        _generation("mathvision", "mvision-mc", True, False),
        _generation("mmstar", "star-p", True, False),
        _generation("mmstar", "star-r", False, True),
        _generation("mmstar", "star-m", True, True),
        _generation("mmmu", "mmmu-s", False, True),
        _generation("mmmu", "mmmu-h", True, False),
    ]
    _write_jsonl(generation_dir / "generations.jsonl", rows)
    return generation_dir


def test_calculates_expected_dataset_splits_and_paired_flips(tmp_path: Path) -> None:
    generation_dir = _fixture(tmp_path)

    report = calculate_split_metrics(generation_dir, project_root=tmp_path)

    mathvista = report["datasets"]["mathvista"]["groups"]
    assert mathvista["OE"]["wrong_to_right"] == 1
    assert mathvista["MC"]["baseline_accuracy"] == 1.0
    mmstar = report["datasets"]["mmstar"]["groups"]
    assert mmstar["ALL"]["count"] == 3
    assert mmstar["Perception"]["count"] == 1
    assert mmstar["Reasoning"]["count"] == 1
    mmmu = report["datasets"]["mmmu"]["groups"]
    assert mmmu["STEM"]["steering_accuracy"] == 1.0
    assert mmmu["HASS"]["steering_accuracy"] == 0.0
    assert report["coverage"]["complete"] is True


def test_rejects_incomplete_generation_unless_explicitly_allowed(
    tmp_path: Path,
) -> None:
    generation_dir = _fixture(tmp_path)
    rows = (generation_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    (generation_dir / "generations.jsonl").write_text(
        "\n".join(rows[:-1]) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="incomplete"):
        calculate_split_metrics(generation_dir, project_root=tmp_path)

    report = calculate_split_metrics(
        generation_dir, project_root=tmp_path, allow_incomplete=True
    )
    assert report["coverage"]["missing"] == 1

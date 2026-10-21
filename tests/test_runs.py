from datetime import datetime, timezone
from pathlib import Path

import pytest

from mllm_self_steering.runs import (
    create_timestamped_run_dir,
    generation_payload,
    preparation_payload,
    preparation_vector_root,
    read_run_config,
    validate_generation_resume,
    write_run_config,
)


def test_timestamped_run_directories_are_readable_and_collision_safe(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 10, 21, 16, 54, tzinfo=timezone.utc)

    first = create_timestamped_run_dir(tmp_path, "generation", now=now)
    second = create_timestamped_run_dir(tmp_path, "generation", now=now)

    assert first.name == "generation_1021_1654"
    assert second.name == "generation_1021_1654_02"


def test_preparation_config_pins_vector_library(tmp_path: Path) -> None:
    preparation_dir = tmp_path / "preparation_1021_1654"
    vector_root = preparation_dir / "vectors" / "capture"
    vector_root.mkdir(parents=True)
    (vector_root / "capability_vectors.safetensors").write_bytes(b"vectors")
    (vector_root / "capability_vectors.json").write_text("{}", encoding="utf-8")
    payload = preparation_payload(
        {"model": {}, "experiment": {}},
        preparation_dir,
        capture_artifact_id="capture",
        vector_root=vector_root,
        status="complete",
    )
    write_run_config(preparation_dir, payload)

    loaded = read_run_config(preparation_dir, expected_kind="preparation")

    assert preparation_vector_root(preparation_dir, loaded) == vector_root.resolve()
    (vector_root / "capability_vectors.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        preparation_vector_root(preparation_dir, loaded)


def test_generation_resume_rejects_changed_experiment(tmp_path: Path) -> None:
    preparation_dir = tmp_path / "preparation_1021_1654"
    generation_dir = tmp_path / "generation_1021_1700"
    preparation = {"preparation_fingerprint": "prep-hash"}
    existing = generation_payload(
        {},
        generation_dir,
        preparation_dir,
        preparation,
        experiment_fingerprint="experiment-a",
        status="running",
    )
    changed = generation_payload(
        {},
        generation_dir,
        preparation_dir,
        preparation,
        experiment_fingerprint="experiment-b",
        status="running",
    )

    with pytest.raises(ValueError, match="experiment_fingerprint"):
        validate_generation_resume(existing, changed, generation_dir)

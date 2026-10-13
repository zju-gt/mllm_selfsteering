from pathlib import Path

import mllm_self_steering.pipeline as pipeline
from mllm_self_steering.pipeline import _limit, _runtime_limit


def test_legacy_limit_applies_to_every_scope() -> None:
    config = {"_runtime": {"limit": 2}}

    assert _runtime_limit(config, "data") == 2
    assert _runtime_limit(config, "pair") == 2
    assert _runtime_limit(config, "eval") == 2


def test_scoped_limits_are_independent() -> None:
    config = {"_runtime": {"pair_limit": 2, "eval_limit": 5}}

    assert _runtime_limit(config, "data") is None
    assert _runtime_limit(config, "pair") == 2
    assert _runtime_limit(config, "eval") == 5
    assert _limit(config, list(range(10)), scope="pair") == [0, 1]
    assert _limit(config, list(range(10)), scope="eval") == [0, 1, 2, 3, 4]


def test_scoped_limit_overrides_legacy_fallback() -> None:
    config = {"_runtime": {"limit": 7, "eval_limit": 3}}

    assert _runtime_limit(config, "pair") == 7
    assert _runtime_limit(config, "eval") == 3


def _artifact_config(runtime: dict[str, int] | None = None) -> dict:
    config = {
        "model": {
            "name": "model",
            "revision": "revision",
            "dtype": "bfloat16",
            "max_new_tokens": 8,
        },
        "data": {"enabled_datasets": ["mmstar"]},
        "experiment": {
            "target_layer": 1,
            "capabilities": [
                {"id": "capability", "modality": "text", "pair_path": "pairs.jsonl"}
            ],
            "vector_scaling": "mean_norm",
            "alphas": [0.0, 1.0],
            "generation": {"backend": "transformers"},
        },
    }
    if runtime is not None:
        config["_runtime"] = runtime
    return config


def test_scoped_limits_preserve_legacy_artifact_identity(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pipeline, "sha256_file", lambda path: "hash")
    monkeypatch.setattr(pipeline, "load_capability_pairs", lambda root, row: [])

    base = _artifact_config()
    legacy = _artifact_config({"limit": 2})
    scoped = _artifact_config({"pair_limit": 2, "eval_limit": 2})
    eval_only = _artifact_config({"eval_limit": 2})

    assert pipeline.capture_artifact_id(legacy, tmp_path) == pipeline.capture_artifact_id(
        scoped, tmp_path
    )
    assert pipeline.capture_artifact_id(base, tmp_path) == pipeline.capture_artifact_id(
        eval_only, tmp_path
    )
    assert pipeline.steering_artifact_id(
        legacy, tmp_path
    ) == pipeline.steering_artifact_id(scoped, tmp_path)
    assert pipeline.steering_artifact_id(
        base, tmp_path
    ) != pipeline.steering_artifact_id(eval_only, tmp_path)

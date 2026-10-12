from pathlib import Path

import pytest

from mllm_self_steering.config import ConfigError, capability_map, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_default_configuration_has_three_plus_three_capabilities() -> None:
    config = load_config(
        [
            ROOT / "configs/model.yaml",
            ROOT / "configs/data.yaml",
            ROOT / "configs/experiment.yaml",
        ]
    )
    capabilities = capability_map(config)
    assert len(capabilities) == 6
    assert sum(row["modality"] == "text" for row in capabilities.values()) == 3
    assert sum(row["modality"] == "visual" for row in capabilities.values()) == 3
    assert config["data"]["sources"]["mmstar"]["source_split"] == "val"
    assert config["data"]["sources"]["mmmu"]["source_split"] == "validation"


def test_override_changes_backend() -> None:
    config = load_config(
        [
            ROOT / "configs/model.yaml",
            ROOT / "configs/data.yaml",
            ROOT / "configs/experiment.yaml",
        ],
        ["experiment.generation.backend=vllm_hook"],
    )
    assert config["experiment"]["generation"]["backend"] == "vllm_hook"


def test_alpha_grid_requires_unsteered_baseline() -> None:
    with pytest.raises(ConfigError, match="baseline"):
        load_config(
            [
                ROOT / "configs/model.yaml",
                ROOT / "configs/data.yaml",
                ROOT / "configs/experiment.yaml",
            ],
            ["experiment.alphas=[-1, 1]"],
        )


def test_vllm_concurrency_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="request_concurrency"):
        load_config(
            [
                ROOT / "configs/model.yaml",
                ROOT / "configs/data.yaml",
                ROOT / "configs/experiment.yaml",
            ],
            ["experiment.vllm_hook.request_concurrency=0"],
        )


def test_max_new_tokens_remains_one_global_setting() -> None:
    config = load_config(
        [
            ROOT / "configs/model.yaml",
            ROOT / "configs/data.yaml",
            ROOT / "configs/experiment.yaml",
        ],
        ["model.max_new_tokens=777"],
    )
    assert config["model"]["max_new_tokens"] == 777


def test_vllm_server_roots_are_absolute_linux_paths() -> None:
    with pytest.raises(ConfigError, match="absolute Linux path"):
        load_config(
            [
                ROOT / "configs/model.yaml",
                ROOT / "configs/data.yaml",
                ROOT / "configs/experiment.yaml",
            ],
            ["experiment.vllm_hook.server_media_root=relative/data"],
        )


def test_vllm_probe_configuration_is_validated() -> None:
    with pytest.raises(ConfigError, match="top_logprobs"):
        load_config(
            [
                ROOT / "configs/model.yaml",
                ROOT / "configs/data.yaml",
                ROOT / "configs/experiment.yaml",
            ],
            ["experiment.vllm_hook.steering_probe_top_logprobs=21"],
        )

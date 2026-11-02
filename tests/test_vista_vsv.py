from __future__ import annotations

import pytest
import torch

from mllm_self_steering.baselines.vista_vsv import (
    compute_vsv,
    validate_vsv_config,
    vista_norm_preserving_transform,
)


def test_compute_vsv_is_positive_minus_negative_per_layer() -> None:
    positive = [torch.tensor([3.0, 4.0]), torch.tensor([1.0, -1.0])]
    negative = [torch.tensor([1.0, 1.0]), torch.tensor([0.5, -2.0])]
    vectors = compute_vsv(positive, negative)
    assert torch.equal(vectors[0], torch.tensor([2.0, 3.0]))
    assert torch.equal(vectors[1], torch.tensor([0.5, 1.0]))


def test_compute_vsv_rejects_invalid_layer_population() -> None:
    with pytest.raises(ValueError, match="equal non-zero"):
        compute_vsv([torch.ones(2)], [])
    with pytest.raises(ValueError, match="shapes differ"):
        compute_vsv([torch.ones(2)], [torch.ones(3)])


def test_vista_transform_preserves_each_token_norm() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])
    changed = vista_norm_preserving_transform(
        hidden, torch.tensor([1.0, 0.0]), 0.1
    )
    before = torch.linalg.vector_norm(hidden, dim=-1)
    after = torch.linalg.vector_norm(changed, dim=-1)
    assert torch.allclose(before, after, atol=1e-5)
    assert not torch.allclose(hidden, changed)


def test_validate_vsv_config() -> None:
    config = {
        "vista_vsv": {
            "lambda": 0.12,
            "construction": "image_minus_text",
            "injection": "official_norm_preserving_mlp",
            "storage_dtype": "float16",
            "consecutive_error_limit": 3,
        }
    }
    value = validate_vsv_config(config)
    assert value["lambda"] == 0.12
    with pytest.raises(ValueError, match="positive"):
        validate_vsv_config({"vista_vsv": {"lambda": 0.0}})

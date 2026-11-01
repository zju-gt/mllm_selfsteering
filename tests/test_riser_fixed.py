from __future__ import annotations

import pytest
import torch

from mllm_self_steering.baselines.riser_fixed import (
    compose_fixed_router_vector,
    fixed_vector_key,
    parse_component_overrides,
    resolve_fixed_router,
)


CAPABILITIES = [
    {"id": "text_logic"},
    {"id": "text_quantitative"},
    {"id": "text_relevance"},
    {"id": "vision_detail"},
    {"id": "vision_spatial"},
    {"id": "vision_grounding"},
]


def _config() -> dict:
    return {
        "experiment": {"capabilities": CAPABILITIES},
        "riser_baseline": {
            "fixed_router": {
                "components": {"text_logic": 0.75},
                "alpha_max": 1.0,
            }
        },
    }


def test_parse_component_overrides() -> None:
    assert parse_component_overrides(
        ["vision_detail=0.7", "text_quantitative=0.5"]
    ) == {"vision_detail": 0.7, "text_quantitative": 0.5}
    with pytest.raises(ValueError, match="duplicate"):
        parse_component_overrides(["text_logic=1", "text_logic=0.5"])
    with pytest.raises(ValueError, match="positive"):
        parse_component_overrides(["text_logic=0"])


def test_resolve_fixed_router_returns_dense_unormalized_strengths() -> None:
    strengths, alpha_max = resolve_fixed_router(_config())
    assert alpha_max == 1.0
    assert strengths == {
        "text_logic": 0.75,
        "text_quantitative": 0.0,
        "text_relevance": 0.0,
        "vision_detail": 0.0,
        "vision_spatial": 0.0,
        "vision_grounding": 0.0,
    }


def test_resolve_fixed_router_rejects_unknown_and_excess_strength() -> None:
    with pytest.raises(ValueError, match="unknown"):
        resolve_fixed_router(_config(), component_overrides={"missing": 1.0})
    with pytest.raises(ValueError, match=r"\(0, 1.0\]"):
        resolve_fixed_router(_config(), component_overrides={"text_logic": 1.1})


def test_compose_fixed_router_vector_is_not_normalized() -> None:
    vectors = {
        "a": torch.tensor([2.0, 0.0]),
        "b": torch.tensor([0.0, 4.0]),
    }
    mixed, diagnostics = compose_fixed_router_vector(vectors, {"a": 0.5, "b": 0.25})
    assert torch.equal(mixed, torch.tensor([1.0, 1.0]))
    assert diagnostics["normalization"] == "none"
    assert diagnostics["selection_mask"] == {"a": 1, "b": 1}


def test_fixed_vector_key_is_order_independent() -> None:
    left = fixed_vector_key(
        {"a": 1.0, "b": 0.0}, {"a": "aa", "b": "bb"}, vector_scaling="unit"
    )
    right = fixed_vector_key(
        {"b": 0.0, "a": 1.0}, {"b": "bb", "a": "aa"}, vector_scaling="unit"
    )
    assert left == right

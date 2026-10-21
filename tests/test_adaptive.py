import json
from pathlib import Path

import pytest
import torch

from mllm_self_steering.adaptive import (
    _server_adaptive_vector_path,
    adaptive_vector_key,
    build_selector_prompts,
    compose_capability_vectors,
    parse_capability_weights,
    selector_prompt_hash,
)


CAPABILITIES = {
    "text_logic": {"description": "Logical reasoning."},
    "text_quantitative": {"description": "Numerical reasoning."},
    "text_relevance": {"description": "Relevant evidence."},
    "vision_detail": {"description": "Visual details."},
    "vision_grounding": {"description": "Visual grounding."},
    "vision_spatial": {"description": "Spatial reasoning."},
}


def _response(**updates: float) -> str:
    weights = {name: 0.0 for name in CAPABILITIES}
    weights.update(updates)
    return json.dumps({"weights": weights})


def test_selector_prompt_uses_configured_capabilities_and_versioned_hash() -> None:
    system, user = build_selector_prompts(CAPABILITIES, "Which option is correct?")

    assert "text_logic: Logical reasoning." in system
    assert "do not need to sum to 1" in system
    assert "Which option is correct?" in user
    assert selector_prompt_hash(CAPABILITIES) == selector_prompt_hash(CAPABILITIES)


def test_parse_weights_is_strict_and_quantizes_half_up() -> None:
    parsed = parse_capability_weights(
        _response(text_logic=0.73, text_quantitative=0.125, vision_detail=0.99),
        list(CAPABILITIES),
        quantum=0.05,
    )

    assert parsed["text_logic"] == pytest.approx(0.75 / 1.9)
    assert parsed["text_quantitative"] == pytest.approx(0.15 / 1.9)
    assert parsed["vision_detail"] == pytest.approx(1.0 / 1.9)
    assert sum(parsed.values()) == pytest.approx(1.0)


def test_parse_weights_preserves_zero_for_no_steering() -> None:
    parsed = parse_capability_weights(_response(), list(CAPABILITIES))

    assert parsed == {name: 0.0 for name in CAPABILITIES}


@pytest.mark.parametrize(
    "payload,match",
    [
        ('```json\n{"weights": {}}\n```', "strict JSON"),
        ('{"weights": {}, "extra": 1}', "top-level"),
        ('{"weights": {}}', "keys mismatch"),
        (_response(text_logic=True), "must be numeric"),
        (_response(text_logic=1.1), "must be finite"),
        (_response(text_logic=float("nan")), "strict JSON"),
    ],
)
def test_parse_weights_rejects_malformed_values(payload: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_capability_weights(payload, list(CAPABILITIES))


def test_weighted_sum_and_mean_are_numerically_distinct() -> None:
    vectors = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([0.0, 2.0]),
    }
    weights = {"a": 1.0, "b": 1.0}

    summed, sum_metadata = compose_capability_vectors(
        vectors, weights, mode="weighted_sum", global_scale=1.0
    )
    averaged, mean_metadata = compose_capability_vectors(
        vectors, weights, mode="weighted_mean", global_scale=1.0
    )

    assert torch.equal(summed, torch.tensor([1.0, 2.0]))
    assert torch.equal(averaged, torch.tensor([0.5, 1.0]))
    assert sum_metadata["weight_sum"] == 2.0
    assert mean_metadata["combined_norm"] == pytest.approx(float(averaged.norm()))


def test_zero_weights_produce_a_finite_zero_vector() -> None:
    mixed, metadata = compose_capability_vectors(
        {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])},
        {"a": 0.0, "b": 0.0},
    )

    assert torch.equal(mixed, torch.zeros(1))
    assert metadata["combined_norm"] == 0.0
    assert metadata["norm_ratio"] == 0.0


def test_vector_key_is_stable_and_sensitive_to_semantics() -> None:
    weights = {"a": 0.5, "b": 1.0}
    fingerprints = {"a": "hash-a", "b": "hash-b"}

    first = adaptive_vector_key(
        weights, fingerprints, mode="weighted_sum", global_scale=1.0
    )
    reordered = adaptive_vector_key(
        {"b": 1.0, "a": 0.5},
        {"b": "hash-b", "a": "hash-a"},
        mode="weighted_sum",
        global_scale=1.0,
    )
    changed = adaptive_vector_key(
        weights, fingerprints, mode="weighted_mean", global_scale=1.0
    )

    assert first == reordered
    assert first != changed


def test_adaptive_vector_path_tracks_server_outputs_sibling(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    local = outputs / "adaptive_vectors" / "capture" / "weights.pt"

    assert _server_adaptive_vector_path(
        local,
        outputs=outputs,
        server_vector_root="/workspace/project/outputs/vectors",
    ) == "/workspace/project/outputs/adaptive_vectors/capture/weights.pt"

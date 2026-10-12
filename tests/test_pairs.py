import pytest

from mllm_self_steering.datasets.types import ContrastPair


def test_text_pair_must_hold_images_fixed() -> None:
    pair = ContrastPair(
        pair_id="x",
        capability="logic",
        modality="text",
        positive_prompt="positive",
        negative_prompt="negative",
        positive_image_paths=["a.png"],
        negative_image_paths=["b.png"],
    )
    with pytest.raises(ValueError, match="hold images fixed"):
        pair.validate()


def test_visual_pair_must_hold_prompt_fixed() -> None:
    pair = ContrastPair(
        pair_id="x",
        capability="detail",
        modality="visual",
        positive_prompt="one",
        negative_prompt="two",
        positive_image_paths=["a.png"],
        negative_image_paths=["b.png"],
    )
    with pytest.raises(ValueError, match="hold prompt fixed"):
        pair.validate()


def test_pair_schemas_are_normalized() -> None:
    text = ContrastPair.from_dict(
        {
            "pair_id": "t",
            "positive_prompt": "positive",
            "negative_prompt": "negative",
            "image_paths": ["same.png"],
        },
        capability="logic",
        modality="text",
    )
    visual = ContrastPair.from_dict(
        {
            "pair_id": "v",
            "prompt": "same",
            "positive_image_paths": ["a.png"],
            "negative_image_paths": ["b.png"],
        },
        capability="detail",
        modality="visual",
    )
    assert text.positive_image_paths == text.negative_image_paths
    assert visual.positive_prompt == visual.negative_prompt


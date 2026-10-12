"""Contrastive activation capture for text and visual input pairs."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from mllm_self_steering.hooks.capture import capture_last_token
from mllm_self_steering.models.inputs import prepare_transformers_inputs
from mllm_self_steering.models.loader import model_input_device


def capture_activation(
    model: Any,
    processor: Any,
    layer: nn.Module,
    *,
    prompt: str,
    image_paths: Sequence[str],
) -> torch.Tensor:
    inputs = prepare_transformers_inputs(processor, prompt, image_paths)
    device = model_input_device(model)
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    with torch.inference_mode(), capture_last_token(layer) as captured:
        model(**inputs, use_cache=False)
    if captured.value is None:
        raise RuntimeError("decoder hook captured no activation")
    return captured.value


def capture_contrast(
    model: Any,
    processor: Any,
    layer: nn.Module,
    *,
    positive_prompt: str,
    negative_prompt: str,
    positive_image_paths: Sequence[str],
    negative_image_paths: Sequence[str],
) -> torch.Tensor:
    positive = capture_activation(
        model,
        processor,
        layer,
        prompt=positive_prompt,
        image_paths=positive_image_paths,
    )
    negative = capture_activation(
        model,
        processor,
        layer,
        prompt=negative_prompt,
        image_paths=negative_image_paths,
    )
    if positive.shape != negative.shape:
        raise ValueError("positive and negative activation shapes differ")
    return positive.to(torch.float32) - negative.to(torch.float32)


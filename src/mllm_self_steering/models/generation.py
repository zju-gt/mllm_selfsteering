"""Reference Transformers generation with optional multimodal steering."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import torch
from torch import nn

from mllm_self_steering.hooks.intervention import add_steering_vector
from mllm_self_steering.models.inputs import prepare_transformers_inputs
from mllm_self_steering.models.loader import model_input_device


def _move_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def generate_transformers(
    model: Any,
    processor: Any,
    *,
    prompt: str,
    image_paths: Sequence[str],
    answer_type: str,
    layer: nn.Module,
    vector: torch.Tensor | None,
    alpha: float,
    max_new_tokens: int,
) -> str:
    inputs = prepare_transformers_inputs(
        processor, prompt, image_paths, answer_type=answer_type
    )
    inputs = _move_inputs(inputs, model_input_device(model))
    context = (
        add_steering_vector(layer, vector, alpha)
        if vector is not None and float(alpha) != 0.0
        else nullcontext()
    )
    with torch.inference_mode(), context:
        generated = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
        )
    prompt_length = int(inputs["input_ids"].shape[1])
    new_tokens = generated[:, prompt_length:]
    return processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


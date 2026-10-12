"""Load a multimodal Transformers model and resolve its language decoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "auto": "auto",
}


def load_model_and_processor(config: dict[str, Any]):
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("transformers>=4.57 is required for Qwen3-VL") from exc
    name = str(config["name"])
    shared = {
        "revision": config.get("revision"),
        "cache_dir": config.get("cache_dir"),
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    shared = {key: value for key, value in shared.items() if value is not None}
    processor = AutoProcessor.from_pretrained(name, **shared)
    dtype_name = str(config.get("dtype", "bfloat16"))
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    kwargs = {
        **shared,
        "dtype": DTYPES[dtype_name],
        "device_map": config.get("device_map", "auto"),
    }
    if config.get("attention_implementation"):
        kwargs["attn_implementation"] = config["attention_implementation"]
    model = AutoModelForMultimodalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, processor


def resolve_language_layers(model: Any) -> Any:
    candidates = (
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "model.model.layers",
        "model.layers",
    )
    for path in candidates:
        current = model
        try:
            for part in path.split("."):
                current = getattr(current, part)
        except AttributeError:
            continue
        if hasattr(current, "__len__") and len(current) > 0:
            return current
    raise ValueError(
        "could not locate language decoder layers; checked: " + ", ".join(candidates)
    )


def resolve_language_layer(model: Any, index: int) -> nn.Module:
    layers = resolve_language_layers(model)
    if not 0 <= index < len(layers):
        raise ValueError(f"target layer {index} is outside [0, {len(layers) - 1}]")
    return layers[index]


def model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


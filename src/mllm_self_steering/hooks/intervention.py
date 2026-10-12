"""Add a direction to the final token during prefill and cached decode."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any, Iterator

import torch
from torch import nn


@contextmanager
def add_steering_vector(
    layer: nn.Module, vector: torch.Tensor, alpha: float
) -> Iterator[None]:
    if vector.ndim != 1:
        raise ValueError("steering vector must be one-dimensional")
    if not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")
    source = vector.detach()
    cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise TypeError("decoder output must contain [batch, sequence, hidden]")
        if hidden.shape[-1] != source.numel():
            raise ValueError("vector hidden size does not match decoder hidden size")
        key = (hidden.device, hidden.dtype)
        scaled = cache.get(key)
        if scaled is None:
            scaled = source.to(device=hidden.device, dtype=hidden.dtype) * float(alpha)
            cache[key] = scaled
        changed = hidden.clone()
        changed[:, -1] += scaled
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


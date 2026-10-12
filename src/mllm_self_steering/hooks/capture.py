"""Capture the final-position output of a language decoder block."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn


def hidden_from_output(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("decoder output must contain [batch, sequence, hidden]")
    return hidden


@dataclass
class CaptureBuffer:
    value: torch.Tensor | None = None


@contextmanager
def capture_last_token(layer: nn.Module) -> Iterator[CaptureBuffer]:
    buffer = CaptureBuffer()

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = hidden_from_output(output)
        buffer.value = hidden[0, -1].detach().to(device="cpu", dtype=torch.float32)
        return output

    handle = layer.register_forward_hook(hook)
    try:
        yield buffer
    finally:
        handle.remove()


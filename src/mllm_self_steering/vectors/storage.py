"""Save vector libraries and vLLM-Hook-compatible direction files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import torch
from safetensors.torch import load_file

from mllm_self_steering.utils.io import atomic_save_json, atomic_save_tensors


FORMS = ("raw", "unit", "steering")


def save_library(
    tensor_path: Path,
    metadata_path: Path,
    vectors: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    metadata: Mapping[str, Any],
) -> None:
    flattened: dict[str, torch.Tensor] = {}
    norms: dict[str, dict[str, float]] = {}
    for capability, forms in vectors.items():
        norms[capability] = {}
        for form in FORMS:
            tensor = forms[form].detach().cpu().float()
            flattened[f"{capability}__{form}"] = tensor
            norms[capability][form] = float(tensor.norm())
    atomic_save_tensors(tensor_path, flattened)
    payload = dict(metadata)
    payload.update({"capabilities": sorted(vectors), "norms": norms})
    atomic_save_json(metadata_path, payload)


def load_library(
    tensor_path: Path, metadata_path: Path
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    flattened = load_file(str(tensor_path), device="cpu")
    vectors: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in flattened.items():
        capability, separator, form = key.partition("__")
        if not separator or form not in FORMS:
            raise ValueError(f"invalid vector key: {key}")
        vectors.setdefault(capability, {})[form] = tensor
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return vectors, metadata


def export_vllm_hook(
    root: Path, vectors: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, dict[str, str]]:
    """Export the current MIA steering contract: torch.save({"dir": vector})."""

    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, str]] = {}
    for capability, forms in vectors.items():
        result[capability] = {}
        for form in FORMS:
            destination = root / f"{capability}__{form}.pt"
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                torch.save({"dir": forms[form].detach().cpu().float()}, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            result[capability][form] = str(destination.resolve())
    return result


"""Mean-difference vectors with shared norm calibration."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def aggregate_vectors(
    deltas: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    if not deltas:
        raise ValueError("at least one capability is required")
    raw: dict[str, torch.Tensor] = {}
    norms: dict[str, torch.Tensor] = {}
    hidden_size: int | None = None
    for capability, values in deltas.items():
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(f"{capability} deltas must be [items, hidden]")
        if hidden_size is None:
            hidden_size = values.shape[1]
        elif values.shape[1] != hidden_size:
            raise ValueError("capability hidden sizes differ")
        value = values.detach().cpu().float().mean(0)
        norm = value.norm()
        if not torch.isfinite(norm) or norm.item() == 0:
            raise ValueError(f"{capability} produced an invalid mean vector")
        raw[capability] = value
        norms[capability] = norm
    mean_norm = torch.stack(list(norms.values())).mean()
    return {
        capability: {
            "raw": value,
            "unit": value / norms[capability],
            "steering": value / norms[capability] * mean_norm,
        }
        for capability, value in raw.items()
    }


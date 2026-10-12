"""Within-capability coherence and cross-capability cosine similarity."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional


def similarity_matrix(
    vectors: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    names = list(vectors)
    if not names:
        return {}
    values = torch.stack([vectors[name].detach().cpu().float() for name in names])
    if torch.any(values.norm(dim=1) == 0) or not torch.all(torch.isfinite(values)):
        raise ValueError("similarity requires finite non-zero vectors")
    normalized = functional.normalize(values, dim=1)
    matrix = normalized @ normalized.T
    return {
        left: {right: float(matrix[i, j]) for j, right in enumerate(names)}
        for i, left in enumerate(names)
    }


def coherence(deltas: torch.Tensor) -> float:
    if deltas.ndim != 2 or deltas.shape[0] < 2:
        raise ValueError("coherence requires at least two deltas")
    values = deltas.detach().cpu().float()
    if torch.any(values.norm(dim=1) == 0) or not torch.all(torch.isfinite(values)):
        raise ValueError("coherence requires finite non-zero deltas")
    normalized = functional.normalize(values, dim=1)
    pairwise = normalized @ normalized.T
    mask = ~torch.eye(values.shape[0], dtype=torch.bool)
    return float(pairwise[mask].mean())


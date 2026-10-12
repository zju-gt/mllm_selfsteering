"""Load user-curated contrast pairs without model/API annotation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mllm_self_steering.datasets.types import ContrastPair
from mllm_self_steering.utils.io import read_jsonl


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_capability_pairs(
    project_root: Path, capability: dict[str, Any]
) -> list[ContrastPair]:
    path = resolve_project_path(project_root, str(capability["pair_path"]))
    if not path.is_file():
        raise FileNotFoundError(
            f"contrast pair file not found for {capability['id']}: {path}"
        )
    pairs = [
        ContrastPair.from_dict(
            row,
            capability=str(capability["id"]),
            modality=str(capability["modality"]),
        )
        for row in read_jsonl(path)
    ]
    if len(pairs) < 2:
        raise ValueError(
            f"contrast pair file must contain at least two pairs for coherence analysis: {path}"
        )
    seen: set[str] = set()
    for pair in pairs:
        if pair.pair_id in seen:
            raise ValueError(f"duplicate pair_id in {path}: {pair.pair_id}")
        seen.add(pair.pair_id)
        for raw_image in pair.positive_image_paths + pair.negative_image_paths:
            image = resolve_project_path(project_root, raw_image)
            if not image.is_file():
                raise FileNotFoundError(f"pair image not found: {image}")
    return pairs

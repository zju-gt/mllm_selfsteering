"""Dataset loading with local canonical JSONL and Hugging Face sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mllm_self_steering.datasets.adapters import ADAPTERS, _materialize_images
from mllm_self_steering.datasets.types import MultimodalItem
from mllm_self_steering.utils.io import read_jsonl


def load_dataset_items(
    name: str,
    source: dict[str, Any],
    *,
    image_root: Path,
    limit: int | None = None,
) -> list[MultimodalItem]:
    if name not in ADAPTERS:
        raise KeyError(f"unsupported dataset: {name}")
    logical_split = str(source.get("split", source.get("source_split", "test")))
    local_path = source.get("local_path")
    if local_path:
        rows = list(read_jsonl(local_path))
        if rows and {
            "item_id",
            "dataset",
            "prompt",
            "image_paths",
            "gold_answer",
            "answer_type",
        }.issubset(rows[0]):
            source_root = Path(local_path).resolve().parent
            normalized = []
            for row in rows:
                value = dict(row)
                value["image_paths"] = [
                    str(
                        Path(image).resolve()
                        if Path(image).is_absolute()
                        else (source_root / image).resolve()
                    )
                    for image in row["image_paths"]
                ]
                normalized.append(value)
            normalized = normalized[:limit] if limit is not None else normalized
            items = []
            for row in normalized:
                item = MultimodalItem.from_dict(row)
                value = item.to_dict()
                value["image_paths"] = _materialize_images(
                    item.image_paths,
                    image_root=image_root / name,
                    item_id=item.item_id,
                )
                items.append(MultimodalItem.from_dict(value))
            return _validate_unique(name, items)
    else:
        rows = _load_huggingface(source, limit=limit)

    adapter = ADAPTERS[name]
    items = [
        adapter(
            row,
            split=logical_split,
            index=index,
            image_root=image_root / name,
        )
        for index, row in enumerate(rows)
    ]
    return _validate_unique(name, items)


def _validate_unique(name: str, items: list[MultimodalItem]) -> list[MultimodalItem]:
    seen: set[str] = set()
    for item in items:
        item.validate()
        if item.item_id in seen:
            raise ValueError(f"duplicate item_id in {name}: {item.item_id}")
        seen.add(item.item_id)
    return items


def _load_huggingface(source: dict[str, Any], *, limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for Hugging Face sources") from exc
    path = str(source["path"])
    split = str(source.get("source_split", source.get("split", "test")))
    cache_dir = source.get("cache_dir")
    base_kwargs = {"path": path, "split": split}
    if cache_dir:
        base_kwargs["cache_dir"] = cache_dir
    if source.get("all_configs"):
        records: list[dict[str, Any]] = []
        for config_name in get_dataset_config_names(path, cache_dir=cache_dir):
            dataset = load_dataset(**base_kwargs, name=config_name)
            for row in dataset:
                record = dict(row)
                record["_subject"] = config_name
                records.append(record)
                if limit is not None and len(records) >= limit:
                    return records
        return records
    kwargs = dict(base_kwargs)
    if source.get("name"):
        kwargs["name"] = source["name"]
    dataset = load_dataset(**kwargs)
    values = [dict(row) for row in dataset]
    return values[:limit] if limit is not None else values

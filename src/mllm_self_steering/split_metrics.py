"""Fine-grained paired metrics for adaptive generation directories."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any
import uuid


SUPPORTED_DATASETS = {"mathvista", "mathvision", "mmstar", "mmmu"}

MMSTAR_PERCEPTION = {"coarse perception", "fine-grained perception"}
MMSTAR_REASONING = {"instance reasoning", "logical reasoning"}

# Official MMMU domains collapsed to the STEM/HASS reporting convention.
MMMU_STEM_SUBJECTS = {
    # Science
    "Biology",
    "Chemistry",
    "Geography",
    "Math",
    "Physics",
    # Health and Medicine
    "Basic_Medical_Science",
    "Clinical_Medicine",
    "Diagnostics_and_Laboratory_Medicine",
    "Pharmacy",
    "Public_Health",
    # Tech and Engineering
    "Agriculture",
    "Architecture_and_Engineering",
    "Computer_Science",
    "Electronics",
    "Energy_and_Power",
    "Materials",
    "Mechanical_Engineering",
}

MMMU_HASS_SUBJECTS = {
    # Art and Design
    "Art",
    "Art_Theory",
    "Design",
    "Music",
    # Business
    "Accounting",
    "Economics",
    "Finance",
    "Manage",
    "Marketing",
    # Humanities and Social Science
    "History",
    "Literature",
    "Psychology",
    "Sociology",
}

GROUP_DEFINITIONS: dict[str, Any] = {
    "mathvista": {
        "OE": "canonical answer_type is not 'choice'",
        "MC": "canonical answer_type is 'choice'",
        "ALL": "all evaluated MathVista items",
    },
    "mathvision": {
        "OE": "canonical answer_type is not 'choice'",
        "MC": "canonical answer_type is 'choice'",
        "ALL": "all evaluated MathVision items",
    },
    "mmstar": {
        "Perception": sorted(MMSTAR_PERCEPTION),
        "Reasoning": sorted(MMSTAR_REASONING),
        "ALL": "all six official MMStar categories",
    },
    "mmmu": {
        "STEM": sorted(MMMU_STEM_SUBJECTS),
        "HASS": sorted(MMMU_HASS_SUBJECTS),
        "ALL": "all official MMMU subjects",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required file not found: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required file not found: {path}") from exc
    with handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{number}")
            rows.append(row)
    return rows


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _config_snapshot(generation_config: Mapping[str, Any]) -> dict[str, Any]:
    value = generation_config.get("config")
    if not isinstance(value, dict):
        raise ValueError("generation config.json has no configuration snapshot")
    return value


def _enabled_datasets(config: Mapping[str, Any]) -> list[str]:
    data = config.get("data")
    enabled = data.get("enabled_datasets") if isinstance(data, dict) else None
    if not isinstance(enabled, list) or not enabled:
        raise ValueError("generation configuration has no enabled datasets")
    names = [str(value) for value in enabled]
    unsupported = set(names) - SUPPORTED_DATASETS
    if unsupported:
        raise ValueError(f"unsupported datasets: {', '.join(sorted(unsupported))}")
    return names


def _eval_limit(config: Mapping[str, Any]) -> int | None:
    runtime = config.get("_runtime")
    if not isinstance(runtime, dict):
        return None
    value = runtime.get("eval_limit", runtime.get("limit"))
    return int(value) if value is not None else None


def _dataset_path(config: Mapping[str, Any], project_root: Path, dataset: str) -> Path:
    experiment = config.get("experiment")
    paths = experiment.get("paths", {}) if isinstance(experiment, dict) else {}
    data_dir = _resolve_project_path(project_root, str(paths.get("data_dir", "data")))
    canonical = data_dir / "processed" / f"{dataset}.jsonl"
    if canonical.is_file():
        return canonical

    data = config.get("data")
    sources = data.get("sources", {}) if isinstance(data, dict) else {}
    source = sources.get(dataset, {}) if isinstance(sources, dict) else {}
    local_path = source.get("local_path") if isinstance(source, dict) else None
    if isinstance(local_path, str):
        fallback = _resolve_project_path(project_root, local_path)
        if fallback.is_file():
            return fallback
    raise FileNotFoundError(f"canonical data for {dataset} was not found: {canonical}")


def _load_expected_items(
    config: Mapping[str, Any], project_root: Path, datasets: Sequence[str]
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    limit = _eval_limit(config)
    items: dict[tuple[str, str], dict[str, Any]] = {}
    paths: dict[str, str] = {}
    for dataset in datasets:
        path = _dataset_path(config, project_root, dataset)
        paths[dataset] = str(path.resolve())
        rows = _read_jsonl(path)
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            raw_dataset = row.get("dataset")
            raw_item_id = row.get("item_id")
            if not isinstance(raw_dataset, str) or not isinstance(raw_item_id, str):
                raise ValueError(f"invalid canonical item in {path}: {row!r}")
            row_dataset = raw_dataset
            item_id = raw_item_id.strip()
            if row_dataset != dataset or not item_id:
                raise ValueError(f"invalid canonical item in {path}: {row!r}")
            key = (dataset, item_id)
            if key in items:
                raise ValueError(f"duplicate canonical item: {dataset}/{item_id}")
            items[key] = row
    return items, paths


def _successful_generations(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    successful: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("status") != "ok":
            continue
        raw_dataset = row.get("dataset")
        raw_item_id = row.get("item_id")
        if not isinstance(raw_dataset, str) or not isinstance(raw_item_id, str):
            raise ValueError(f"successful generation has no dataset/item_id: {row!r}")
        dataset = raw_dataset.strip()
        item_id = raw_item_id.strip()
        if not dataset or not item_id:
            raise ValueError(f"successful generation has no dataset/item_id: {row!r}")
        successful[(dataset, item_id)] = row
    return successful


def _split_group(dataset: str, item: Mapping[str, Any]) -> str | None:
    if dataset in {"mathvista", "mathvision"}:
        return "MC" if item.get("answer_type") == "choice" else "OE"
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"canonical item has invalid metadata: {item.get('item_id')}")
    if dataset == "mmstar":
        category = str(metadata.get("category", "")).strip().casefold()
        if category in MMSTAR_PERCEPTION:
            return "Perception"
        if category in MMSTAR_REASONING:
            return "Reasoning"
        if category in {"math", "science & technology"}:
            return None
        raise ValueError(f"unknown MMStar category: {metadata.get('category')!r}")
    if dataset == "mmmu":
        subject = str(metadata.get("subject", ""))
        if subject in MMMU_STEM_SUBJECTS:
            return "STEM"
        if subject in MMMU_HASS_SUBJECTS:
            return "HASS"
        raise ValueError(f"unknown MMMU subject: {subject!r}")
    raise ValueError(f"unsupported dataset: {dataset}")


def _metric(rows: Iterable[Mapping[str, Any]]) -> dict[str, int | float | None]:
    selected = list(rows)
    count = len(selected)
    baseline_correct = sum(bool(row["baseline_correct"]) for row in selected)
    steering_correct = sum(bool(row["correct"]) for row in selected)
    wrong_to_right = sum(
        not bool(row["baseline_correct"]) and bool(row["correct"])
        for row in selected
    )
    right_to_wrong = sum(
        bool(row["baseline_correct"]) and not bool(row["correct"])
        for row in selected
    )
    baseline_accuracy = baseline_correct / count if count else None
    steering_accuracy = steering_correct / count if count else None
    accuracy_delta = (
        steering_accuracy - baseline_accuracy
        if steering_accuracy is not None and baseline_accuracy is not None
        else None
    )
    return {
        "count": count,
        "steering_applied_count": sum(
            bool(row.get("steering_applied")) for row in selected
        ),
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_accuracy,
        "baseline_accuracy_percent": (
            round(baseline_accuracy * 100.0, 6)
            if baseline_accuracy is not None
            else None
        ),
        "steering_correct": steering_correct,
        "steering_accuracy": steering_accuracy,
        "steering_accuracy_percent": (
            round(steering_accuracy * 100.0, 6)
            if steering_accuracy is not None
            else None
        ),
        "accuracy_delta": accuracy_delta,
        "accuracy_delta_percentage_points": (
            round(accuracy_delta * 100.0, 6) if accuracy_delta is not None else None
        ),
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "net_flips": wrong_to_right - right_to_wrong,
    }


def calculate_split_metrics(
    generation_dir: Path | str,
    *,
    project_root: Path | str,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Calculate paired baseline/steering metrics for one generation directory."""

    generation = Path(generation_dir).resolve()
    root = Path(project_root).resolve()
    config_path = generation / "config.json"
    generations_path = generation / "generations.jsonl"
    generation_config = _read_json(config_path)
    config = _config_snapshot(generation_config)
    datasets = _enabled_datasets(config)
    expected, data_paths = _load_expected_items(config, root, datasets)
    successful = _successful_generations(generations_path)

    expected_keys = set(expected)
    successful_keys = set(successful)
    missing = sorted(expected_keys - successful_keys)
    extra = sorted(successful_keys - expected_keys)
    if extra:
        raise ValueError(
            f"generations.jsonl contains {len(extra)} unexpected successful items; "
            f"sample={extra[:3]}"
        )
    if missing and not allow_incomplete:
        raise ValueError(
            f"generation is incomplete: {len(missing)} successful items are missing; "
            f"sample={missing[:3]}; pass --allow-incomplete to score available rows"
        )

    joined_by_dataset: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        dataset: [] for dataset in datasets
    }
    for key in sorted(expected_keys & successful_keys):
        item = expected[key]
        row = successful[key]
        for field in ("baseline_correct", "correct"):
            if not isinstance(row.get(field), bool):
                raise ValueError(
                    f"successful generation {key[0]}/{key[1]} has no boolean {field}"
                )
        joined_by_dataset[key[0]].append((item, row))

    dataset_metrics: dict[str, Any] = {}
    for dataset in datasets:
        joined = joined_by_dataset[dataset]
        group_order = (
            ["ALL", "OE", "MC"]
            if dataset in {"mathvista", "mathvision"}
            else ["ALL", "Perception", "Reasoning"]
            if dataset == "mmstar"
            else ["ALL", "STEM", "HASS"]
        )
        grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in group_order}
        grouped["ALL"] = [row for _, row in joined]
        for item, row in joined:
            group = _split_group(dataset, item)
            if group is not None:
                grouped[group].append(row)
        dataset_metrics[dataset] = {
            "definition": GROUP_DEFINITIONS[dataset],
            "groups": {name: _metric(grouped[name]) for name in group_order},
        }

    return {
        "schema_version": 1,
        "generation_dir": str(generation),
        "source": {
            "config": str(config_path.resolve()),
            "generations": str(generations_path.resolve()),
            "canonical_data": data_paths,
        },
        "coverage": {
            "expected": len(expected_keys),
            "successful": len(expected_keys & successful_keys),
            "missing": len(missing),
            "complete": not missing,
        },
        "datasets": dataset_metrics,
    }


def write_metrics(path: Path | str, metrics: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(metrics), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination

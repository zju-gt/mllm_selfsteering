"""Frozen-model Table 2 steering strategies and reporting."""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import fmean
from typing import Any, Mapping, Sequence
import uuid

from mllm_self_steering.config import _apply_override, validate_config
from mllm_self_steering.datasets.pairs import resolve_project_path
from mllm_self_steering.datasets.types import MultimodalItem
from mllm_self_steering.utils.io import atomic_save_json


MODES = ("none", "single", "all", "random", "text", "visual", "self")
ROUTED_MODES = {"text", "visual", "self"}
CONTRACT = "table2-various-steering-v1-dirichlet-random"


def publish_route(path: Path, route: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish once, returning the same winning route to all modes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        atomic_save_json(temporary, dict(route))
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    return json.loads(path.read_text(encoding="utf-8"))


def strategy_weights(
    mode: str,
    capabilities: Sequence[Mapping[str, Any]],
    item_identity: str,
    *,
    seed: int,
    routed_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return actual injection weights; random sampling never calls the model."""

    if mode not in MODES:
        raise ValueError(f"unknown steering mode: {mode}")
    ids = [str(row["id"]) for row in capabilities]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("capabilities must be non-empty and unique")
    weights = dict.fromkeys(ids, 0.0)
    if mode == "none":
        return weights
    if mode == "single":
        weights[ids[0]] = 1.0
        return weights
    if mode == "all":
        return dict.fromkeys(ids, 1.0 / len(ids))
    if mode == "random":
        digest = hashlib.sha256(f"{CONTRACT}:{seed}:{item_identity}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        # Independent Exp(1) draws normalized by their sum are Dirichlet(1,...,1).
        values = [rng.expovariate(1.0) for _ in ids]
        total = sum(values)
        return {name: value / total for name, value in zip(ids, values)}
    if routed_weights is None or set(routed_weights) != set(ids):
        raise ValueError(f"{mode} mode requires a complete six-capability route")
    for row in capabilities:
        name = str(row["id"])
        value = float(routed_weights[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError("routed weights must be finite and non-negative")
        if mode == "self" or row["modality"] == mode:
            weights[name] = value
    total = sum(weights.values())
    return (
        {name: value / total for name, value in weights.items()} if total else weights
    )


def build_various_config(
    base: Mapping[str, Any],
    preparation: Mapping[str, Any],
    *,
    mode: str,
    random_seed: int,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Inherit the bank's checkpoint and extraction settings before CLI overrides."""

    if mode not in MODES:
        raise ValueError(f"unknown steering mode: {mode}")
    if preparation.get("status") != "complete":
        raise ValueError("the preparation directory must be complete")
    config = deepcopy(dict(base))
    # Table 2 uses the full MathVision test population; explicit CLI overrides win.
    config["data"]["sources"]["mathvision"].update(
        local_path="data/processed/mathvision.jsonl", split="test", source_split="test"
    )
    saved = preparation["config"]
    config["model"] = deepcopy(saved["model"])
    experiment = config["experiment"]
    for key in ("capabilities", "target_layer", "vector_scaling"):
        experiment[key] = deepcopy(saved["experiment"][key])
    experiment.setdefault("generation", {})["backend"] = "vllm_hook"
    experiment.setdefault("vllm_hook", {})["protocol"] = "vllm_hook_v0_2"
    adaptive = experiment.setdefault("adaptive_steering", {})
    adaptive.update(
        baseline_no_answer_only=False, include_baseline=True, composition="weighted_sum"
    )
    experiment["various_steering"] = {
        "contract": CONTRACT,
        "mode": mode,
        "random_seed": random_seed if mode == "random" else None,
    }
    for override in overrides:
        _apply_override(config, override)
    if experiment["various_steering"] != {
        "contract": CONTRACT,
        "mode": mode,
        "random_seed": random_seed if mode == "random" else None,
    }:
        raise ValueError(
            "set the strategy and random seed using the CLI mode and --random-seed"
        )
    if adaptive.get("baseline_no_answer_only") or not adaptive.get("include_baseline"):
        raise ValueError(
            "Table 2 requires paired baselines and disables no-answer gating"
        )
    if adaptive.get("composition") != "weighted_sum":
        raise ValueError(
            "Table 2 uses normalized weights with weighted_sum composition"
        )
    validate_config(config)
    return config


def evaluation_path(
    config: Mapping[str, Any], project_root: Path, dataset: str
) -> Path:
    """Stage 08 honors local_path; preserve Stage 07's legacy population contract."""

    if "various_steering" in config["experiment"]:
        local = config["data"]["sources"][dataset].get("local_path")
        if local:
            return resolve_project_path(project_root, str(local))
    data_dir = resolve_project_path(
        project_root, str(config["experiment"]["paths"]["data_dir"])
    )
    return data_dir / "processed" / f"{dataset}.jsonl"


def evaluation_item(row: Mapping[str, Any], project_root: Path) -> MultimodalItem:
    """Resolve existing images or rebase a migrated data/images path in memory."""

    value = dict(row)
    paths = []
    for original in row["image_paths"]:
        path = resolve_project_path(project_root, str(original))
        if not path.is_file():
            parts = [
                part for part in str(original).replace("\\", "/").split("/") if part
            ]
            for index in range(len(parts) - 1):
                if parts[index : index + 2] == ["data", "images"]:
                    candidate = project_root.joinpath(*parts[index:])
                    if candidate.is_file():
                        path = candidate
                        break
        if not path.is_file():
            raise FileNotFoundError(f"evaluation image not found: {original}")
        paths.append(str(path.resolve()))
    value["image_paths"] = paths
    return MultimodalItem.from_dict(value)


def table2_summary(
    report: Mapping[str, Any],
    datasets: Sequence[str],
    mode: str,
) -> dict[str, Any]:
    """Macro-average datasets, never publish a final Avg for partial coverage."""

    groups = report["groups"]
    complete = bool(report["complete"]) and all(
        groups.get(name, {}).get("adaptive_accuracy") is not None for name in datasets
    )
    return {
        "mode": mode,
        "complete": complete,
        "datasets": {name: groups.get(name, {}) for name in datasets},
        "macro_accuracy": (
            fmean(groups[name]["adaptive_accuracy"] for name in datasets)
            if complete
            else None
        ),
        "macro_baseline_accuracy": (
            fmean(groups[name]["baseline_accuracy"] for name in datasets)
            if complete
            else None
        ),
    }


def write_table2_report(
    metrics_path: Path,
    config: Mapping[str, Any],
    project_root: Path,
    *,
    interrupted: bool = False,
) -> Path:
    """Persist comparable Table 2 metrics, including interrupted populations."""

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    if interrupted:
        report["complete"] = False
    strategy = config["experiment"]["various_steering"]
    datasets = config["data"]["enabled_datasets"]
    summary = table2_summary(report, datasets, strategy["mode"])
    subset = strategy.get("subset")
    summary.update(
        evaluation_scope="subset" if subset is not None else "full",
        sampling=subset,
        random_seed=strategy["random_seed"],
        evaluation={
            name: {"path": str(evaluation_path(config, project_root, name))}
            for name in datasets
        },
    )
    report["table2"] = summary
    atomic_save_json(metrics_path, report)
    csv_path = metrics_path.with_name("table2.csv")
    temporary = csv_path.with_name(f".{csv_path.name}.{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "evaluation_scope",
                    "mode",
                    "random_seed",
                    "subset_fraction",
                    "subset_seed",
                    *datasets,
                    "Avg",
                    "complete",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "evaluation_scope": summary["evaluation_scope"],
                    "mode": strategy["mode"],
                    "random_seed": strategy["random_seed"],
                    "subset_fraction": subset["fraction"] if subset else 1.0,
                    "subset_seed": subset["seed"] if subset else None,
                    **{
                        name: (
                            summary["datasets"][name]["adaptive_accuracy"] * 100
                            if summary["datasets"][name].get("adaptive_accuracy")
                            is not None
                            else None
                        )
                        for name in datasets
                    },
                    "Avg": (
                        summary["macro_accuracy"] * 100
                        if summary["macro_accuracy"] is not None
                        else None
                    ),
                    "complete": summary["complete"],
                }
            )
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    return csv_path

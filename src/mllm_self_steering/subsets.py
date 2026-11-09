"""Reproducible subset evaluation and explicitly labelled hybrid previews."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence
import uuid

from mllm_self_steering.datasets.pairs import resolve_project_path
from mllm_self_steering.datasets.types import MultimodalItem
from mllm_self_steering.evaluation.answers import extract_answer, is_correct
from mllm_self_steering.utils.io import (
    atomic_save_json,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


Record = tuple[MultimodalItem, list[str], str]
Key = tuple[str, str]
SUBSET_CONTRACT = "per-dataset-sha256-ranking-v1"


def select_subset_keys(keys: Sequence[Key], *, fraction: float, seed: int) -> set[Key]:
    """Select ceil(N*fraction) per dataset, independent of file order and mode."""

    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("subset fraction must be finite and in (0, 1]")
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate evaluation sample identity")
    grouped: dict[str, list[Key]] = {}
    for key in keys:
        grouped.setdefault(key[0], []).append(key)
    selected: set[Key] = set()
    for group in grouped.values():
        ranked = sorted(
            group,
            key=lambda key: (
                hashlib.sha256(
                    json.dumps([SUBSET_CONTRACT, seed, *key]).encode()
                ).digest(),
                key,
            ),
        )
        selected.update(ranked[: math.ceil(len(group) * fraction)])
    return selected


def configure_subset(
    config: dict[str, Any],
    project_root: Path,
    *,
    fraction: float,
    seed: int,
    merge_base: Path | None = None,
) -> None:
    """Freeze sampling parameters and the optional reference file's content hash."""

    select_subset_keys([], fraction=fraction, seed=seed)
    strategy = config["experiment"]["various_steering"]
    if fraction == 1 and merge_base is None:
        strategy.pop("subset", None)
        return
    settings: dict[str, Any] = {
        "contract": SUBSET_CONTRACT,
        "fraction": fraction,
        "seed": seed,
    }
    if merge_base is not None:
        path = resolve_project_path(project_root, str(merge_base)).resolve()
        settings.update(merge_base=str(path), base_sha256=sha256_file(path))
    strategy["subset"] = settings


def _reference_result(
    row: Mapping[str, Any], item: MultimodalItem, identity: str
) -> dict[str, Any]:
    if row.get("item_identity") != identity:
        raise ValueError(
            f"merge base sample identity mismatch: {item.dataset}/{item.item_id}"
        )
    for field in ("gold_answer", "answer_type"):
        if field in row and str(row[field]) != str(getattr(item, field)):
            raise ValueError(
                f"merge base {field} mismatch: {item.dataset}/{item.item_id}"
            )
    raw_key = next(
        (key for key in ("raw_output", "baseline_raw_output") if key in row), None
    )
    if raw_key is not None:
        raw = row[raw_key]
        if not isinstance(raw, str):
            raise ValueError(f"merge base {raw_key} must be text: {item.item_id}")
        prediction = extract_answer(raw, item.answer_type)
    else:
        prediction_key = next(
            (
                key
                for key in ("predicted_answer", "baseline_predicted_answer")
                if key in row
            ),
            None,
        )
        if prediction_key is None:
            raise ValueError(
                f"merge base has no prediction: {item.dataset}/{item.item_id}"
            )
        prediction = row[prediction_key]
        if prediction is not None and not isinstance(prediction, str):
            raise ValueError(
                f"merge base prediction must be text or null: {item.item_id}"
            )
        raw = None
    return {
        **row,
        "raw_output": raw,
        "predicted_answer": prediction,
        "gold_answer": item.gold_answer,
        "answer_type": item.answer_type,
        "correct": is_correct(
            prediction, item.gold_answer, answer_type=item.answer_type
        ),
        "status": "ok",
    }


@dataclass
class SubsetPlan:
    settings: dict[str, Any]
    selected_keys: set[Key]
    base_rows: dict[Key, dict[str, Any]]
    manifest: dict[str, Any]


def prepare_subset(
    config: Mapping[str, Any], records: Sequence[Record]
) -> SubsetPlan | None:
    """Validate the entire complement before any requests; never fill failed reruns."""

    settings = config["experiment"].get("various_steering", {}).get("subset")
    if settings is None:
        return None
    keys = [(item.dataset, identity) for item, _, identity in records]
    selected = select_subset_keys(
        keys, fraction=settings["fraction"], seed=settings["seed"]
    )
    expected = {
        (item.dataset, item.item_id): (item, identity) for item, _, identity in records
    }
    if len(expected) != len(records):
        raise ValueError("duplicate dataset/item_id in evaluation population")
    base_rows: dict[Key, dict[str, Any]] = {}
    if settings.get("merge_base"):
        path = Path(settings["merge_base"])
        if sha256_file(path) != settings["base_sha256"]:
            raise ValueError(
                "merge base contents changed since configuration was frozen"
            )
        canonical: dict[Key, dict[str, Any]] = {}
        for row in read_jsonl(path):
            key = (str(row.get("dataset")), str(row.get("item_id")))
            if key not in expected or row.get("status", "ok") != "ok":
                continue
            previous = canonical.get(key)
            if previous is not None and previous.get("run_id") != row.get("run_id"):
                raise ValueError(f"merge base contains multiple runs for sample: {key}")
            canonical[key] = row
        if sha256_file(path) != settings["base_sha256"]:
            raise ValueError("merge base contents changed while reading")
        for key, (item, identity) in expected.items():
            output_key = (item.dataset, identity)
            if key in canonical:
                base_rows[output_key] = _reference_result(
                    canonical[key], item, identity
                )
            elif output_key not in selected:
                raise ValueError(
                    f"merge base is missing a successful result for: {key}"
                )
    counts: dict[str, dict[str, int]] = {}
    manifest_rows = []
    for item, _, identity in records:
        chosen = (item.dataset, identity) in selected
        count = counts.setdefault(item.dataset, {"total": 0, "evaluated": 0, "base": 0})
        count["total"] += 1
        count["evaluated"] += int(chosen)
        count["base"] += int(not chosen and bool(settings.get("merge_base")))
        manifest_rows.append(
            {
                "dataset": item.dataset,
                "item_id": item.item_id,
                "item_identity": identity,
                "assignment": "evaluated"
                if chosen
                else ("base" if settings.get("merge_base") else "excluded"),
            }
        )
    return SubsetPlan(
        dict(settings),
        selected,
        base_rows,
        {
            **settings,
            "datasets": counts,
            "samples": manifest_rows,
        },
    )


def _consistency(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> dict[str, Any]:
    count = len(pairs)
    return {
        "compared_items": count,
        "prediction_agreement": (
            sum(
                current["predicted_answer"] == base["predicted_answer"]
                for current, base in pairs
            )
            / count
            if count
            else None
        ),
        "evaluated_accuracy": fmean(bool(current["correct"]) for current, _ in pairs)
        if count
        else None,
        "base_accuracy": fmean(bool(base["correct"]) for _, base in pairs)
        if count
        else None,
        "accuracy_delta": (
            sum(
                int(bool(current["correct"])) - int(bool(base["correct"]))
                for current, base in pairs
            )
            / count
            if count
            else None
        ),
        "wrong_to_right": sum(
            bool(current["correct"]) and not base["correct"] for current, base in pairs
        ),
        "right_to_wrong": sum(
            not current["correct"] and bool(base["correct"]) for current, base in pairs
        ),
    }


def write_hybrid_outputs(
    output_dir: Path,
    records: Sequence[Record],
    evaluated: Mapping[Key, Mapping[str, Any]],
    plan: SubsetPlan,
    *,
    mode: str,
    run_id: str,
    random_seed: int | None = None,
    interrupted: bool = False,
) -> dict[str, Path]:
    """Write source-labelled results and separate preview metrics in population order."""

    merged = []
    comparisons: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for item, _, identity in records:
        key = (item.dataset, identity)
        chosen = key in plan.selected_keys
        row = evaluated.get(key) if chosen else plan.base_rows.get(key)
        if row is None:
            continue
        if chosen and key in plan.base_rows:
            comparisons.setdefault(item.dataset, []).append((row, plan.base_rows[key]))
        merged.append(
            {
                **row,
                "result_kind": "hybrid",
                "result_source": "evaluated" if chosen else "base",
                "hybrid_run_id": run_id,
                "source_run_id": row.get("run_id"),
                "source_path": str(output_dir / "generations.jsonl")
                if chosen
                else plan.settings["merge_base"],
                "source_sha256": None if chosen else plan.settings["base_sha256"],
            }
        )
    groups = {}
    for dataset, counts in plan.manifest["datasets"].items():
        rows = [row for row in merged if row["dataset"] == dataset]
        complete = len(rows) == counts["total"] and not interrupted
        groups[dataset] = {
            "expected_items": counts["total"],
            "completed_items": len(rows),
            "complete": complete,
            "accuracy": fmean(bool(row["correct"]) for row in rows) if rows else None,
            "evaluated_items": sum(row["result_source"] == "evaluated" for row in rows),
            "base_items": sum(row["result_source"] == "base" for row in rows),
            "consistency": _consistency(comparisons.get(dataset, [])),
        }
    complete = len(merged) == len(records) and not interrupted
    report = {
        "result_kind": "hybrid",
        "mode": mode,
        "random_seed": random_seed,
        "sampling": plan.settings,
        "expected_items": len(records),
        "completed_items": len(merged),
        "complete": complete,
        "macro_accuracy": fmean(group["accuracy"] for group in groups.values())
        if complete
        else None,
        "groups": groups,
        "consistency": _consistency(
            [pair for pairs in comparisons.values() for pair in pairs]
        ),
    }
    generations_path = output_dir / "hybrid_generations.jsonl"
    metrics_path = output_dir / "hybrid_metrics.json"
    csv_path = output_dir / "hybrid_table2.csv"
    write_jsonl(generations_path, merged)
    atomic_save_json(metrics_path, report)
    temporary = csv_path.with_name(f".{csv_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "result_kind",
                    "mode",
                    "random_seed",
                    "subset_fraction",
                    "subset_seed",
                    *groups,
                    "Avg",
                    "complete",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "result_kind": "hybrid",
                    "mode": mode,
                    "random_seed": random_seed,
                    "subset_fraction": plan.settings["fraction"],
                    "subset_seed": plan.settings["seed"],
                    **{
                        name: group["accuracy"] * 100
                        if group["accuracy"] is not None
                        else None
                        for name, group in groups.items()
                    },
                    "Avg": report["macro_accuracy"] * 100 if complete else None,
                    "complete": complete,
                }
            )
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "hybrid_generations": generations_path,
        "hybrid_metrics": metrics_path,
        "hybrid_table2_csv": csv_path,
    }

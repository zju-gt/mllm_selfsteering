"""Resumable unsteered multimodal generation through an OpenAI-compatible vLLM server."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence, TypeVar

from tqdm.auto import tqdm

from mllm_self_steering.datasets.types import MultimodalItem
from mllm_self_steering.evaluation.answers import extract_answer, is_correct
from mllm_self_steering.models.vllm_hook import (
    AsyncVLLMHookClient,
    build_vllm_messages,
)
from mllm_self_steering.pipeline import (
    _image_hashes,
    _limit,
    _paths,
    _runtime_limit,
    item_identity,
)
from mllm_self_steering.runs import read_run_config, write_run_config
from mllm_self_steering.utils.io import (
    JsonlCheckpointWriter,
    atomic_save_json,
    read_jsonl,
    repair_truncated_jsonl,
    sha256_file,
    sha256_text,
)


_T = TypeVar("_T")
BASELINE_CONTRACT = "multimodal-vllm-baseline-v1"


def baseline_experiment_fingerprint(
    config: Mapping[str, Any], project_root: Path
) -> str:
    """Identify settings and inputs that can change baseline outputs."""

    data_dir, _ = _paths(dict(config), project_root)
    datasets = list(config["data"]["enabled_datasets"])
    vllm = config["experiment"].get("vllm_hook", {})
    identity = {
        "contract": BASELINE_CONTRACT,
        "model": {
            key: config["model"].get(key)
            for key in ("name", "source", "revision", "dtype", "max_new_tokens")
        },
        "server_provenance": vllm.get("server_provenance", {}),
        "seed": config["experiment"].get("seed", 42),
        "temperature": 0.0,
        "eval_limit": _runtime_limit(dict(config), "eval"),
        "datasets": datasets,
        "evaluation": {
            dataset: sha256_file(data_dir / "processed" / f"{dataset}.jsonl")
            for dataset in datasets
        },
    }
    return sha256_text(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def baseline_payload(
    config: Mapping[str, Any],
    baseline_dir: Path,
    *,
    experiment_fingerprint: str,
    status: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "baseline",
        "name": baseline_dir.name,
        "created_at": created_at or datetime.now().astimezone().isoformat(),
        "status": status,
        "experiment_fingerprint": experiment_fingerprint,
        "contract": BASELINE_CONTRACT,
        "config": deepcopy(dict(config)),
        "artifacts": {
            "baselines": "baselines.jsonl",
            "metrics_json": "metrics.json",
        },
    }


def validate_baseline_resume(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
    baseline_dir: Path,
) -> None:
    for key in ("kind", "name", "experiment_fingerprint", "contract"):
        if existing.get(key) != expected.get(key):
            raise ValueError(
                f"cannot resume {baseline_dir}: config mismatch for {key}; "
                "omit --generation-dir to start a new baseline run"
            )


def _canonical_success(
    path: Path, run_id: str
) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    repair_truncated_jsonl(path)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("run_id") != run_id or row.get("status") != "ok":
            continue
        rows[(str(row["dataset"]), str(row["item_identity"]))] = row
    return rows


def _item_seed(base_seed: int, identity: str) -> int:
    return (int(base_seed) + int(identity[:8], 16)) % (2**31 - 1)


async def _run_batched(
    values: Sequence[_T],
    *,
    concurrency: int,
    worker: Callable[[_T], Awaitable[dict[str, Any]]],
    writer: JsonlCheckpointWriter,
    progress: Any,
    consecutive_error_limit: int,
    on_success: Callable[[dict[str, Any]], None],
) -> None:
    error_streak = 0
    for start in range(0, len(values), concurrency):
        results = await asyncio.gather(
            *(worker(value) for value in values[start : start + concurrency])
        )
        for row in results:
            writer.write(row)
            progress.update(1)
            if row.get("status") == "ok":
                error_streak = 0
                on_success(row)
            else:
                error_streak += 1
            if error_streak >= consecutive_error_limit:
                writer.checkpoint()
                raise RuntimeError(
                    "aborting baseline generation after "
                    f"{error_streak} consecutive request failures; "
                    f"last error={row.get('error')}"
                )


def _answer_fields(output: str, item: MultimodalItem) -> dict[str, Any]:
    prediction = extract_answer(output, item.answer_type)
    return {
        "baseline_raw_output": output,
        "baseline_predicted_answer": prediction,
        "baseline_correct": is_correct(
            prediction, item.gold_answer, answer_type=item.answer_type
        ),
    }


def _metrics(
    records: Sequence[tuple[MultimodalItem, list[str], str]],
    successful: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    total_correct = 0
    total_successful = 0
    for dataset in dict.fromkeys(item.dataset for item, _, _ in records):
        expected = [record for record in records if record[0].dataset == dataset]
        rows = [
            successful[(item.dataset, identity)]
            for item, _, identity in expected
            if (item.dataset, identity) in successful
        ]
        correct = sum(bool(row["baseline_correct"]) for row in rows)
        count = len(rows)
        total_correct += correct
        total_successful += count
        by_dataset[dataset] = {
            "expected": len(expected),
            "successful": count,
            "correct": correct,
            "accuracy": correct / count if count else None,
        }
    expected_total = len(records)
    return {
        "schema_version": 1,
        "coverage": {
            "expected": expected_total,
            "successful": total_successful,
            "missing": expected_total - total_successful,
            "complete": total_successful == expected_total,
        },
        "overall": {
            "correct": total_correct,
            "accuracy": (
                total_correct / total_successful if total_successful else None
            ),
        },
        "datasets": by_dataset,
    }


def run_baseline(
    config: dict[str, Any],
    project_root: Path,
    *,
    baseline_dir: Path,
) -> dict[str, Path]:
    """Run or resume one unsteered generation per evaluation item."""

    if config["experiment"].get("generation", {}).get("backend") != "vllm_hook":
        raise ValueError("baseline generation requires experiment.generation.backend=vllm_hook")

    data_dir, _ = _paths(config, project_root)
    baseline_dir = baseline_dir.resolve()
    baseline_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = baseline_experiment_fingerprint(config, project_root)
    expected_metadata = baseline_payload(
        config,
        baseline_dir,
        experiment_fingerprint=fingerprint,
        status="running",
    )
    metadata_path = baseline_dir / "config.json"
    if metadata_path.is_file():
        existing = read_run_config(baseline_dir, expected_kind="baseline")
        validate_baseline_resume(existing, expected_metadata, baseline_dir)
        metadata = dict(existing)
        metadata["status"] = "running"
        metadata["config"] = expected_metadata["config"]
    else:
        if any(baseline_dir.iterdir()):
            raise ValueError(
                f"baseline directory is non-empty but has no config.json: {baseline_dir}"
            )
        metadata = expected_metadata
    write_run_config(baseline_dir, metadata)

    run_id = str(metadata["name"])
    baseline_path = baseline_dir / "baselines.jsonl"
    metrics_path = baseline_dir / "metrics.json"
    successful = _canonical_success(baseline_path, run_id)

    items: list[MultimodalItem] = []
    for dataset in config["data"]["enabled_datasets"]:
        rows = list(read_jsonl(data_dir / "processed" / f"{dataset}.jsonl"))
        items.extend(
            MultimodalItem.from_dict(row)
            for row in _limit(config, rows, scope="eval")
        )
    records = []
    for item in items:
        hashes = _image_hashes(item)
        records.append((item, hashes, item_identity(item, image_hashes=hashes)))
    missing = [
        record
        for record in records
        if (record[0].dataset, record[2]) not in successful
    ]
    progress = tqdm(
        total=len(records),
        initial=len(records) - len(missing),
        desc=f"{run_id}:baseline",
    )
    vllm = config["experiment"].get("vllm_hook", {})
    base_seed = int(config["experiment"].get("seed", 42))

    async def execute() -> None:
        client = AsyncVLLMHookClient(
            base_url=str(vllm.get("base_url", "http://localhost:8770/v1")),
            api_key=str(vllm.get("api_key", "EMPTY")),
            model_name=str(config["model"]["name"]),
            timeout_seconds=float(vllm.get("request_timeout_seconds", 600)),
            max_retries=int(vllm.get("max_retries", 1)),
            protocol=str(vllm.get("protocol", "mia")),
        )
        try:
            await client.preflight()

            async def generate_one(
                record: tuple[MultimodalItem, list[str], str],
            ) -> dict[str, Any]:
                item, image_hashes, identity = record
                base = {
                    "run_id": run_id,
                    "dataset": item.dataset,
                    "item_id": item.item_id,
                    "item_identity": identity,
                    "gold_answer": item.gold_answer,
                    "answer_type": item.answer_type,
                    "backend": "vllm",
                }
                try:
                    messages = build_vllm_messages(
                        item.prompt,
                        item.image_paths,
                        answer_type=item.answer_type,
                        media_transport=str(vllm.get("media_transport", "file")),
                        local_media_root=data_dir,
                        server_media_root=vllm.get("server_media_root"),
                        media_uuids=image_hashes,
                    )
                    output = await client.generate(
                        messages=messages,
                        vector_path=None,
                        target_layer=0,
                        alpha=0.0,
                        phase="both",
                        positions="last_token",
                        max_new_tokens=int(config["model"].get("max_new_tokens", 2048)),
                        temperature=0.0,
                        seed=_item_seed(base_seed, identity),
                    )
                    return {**base, **_answer_fields(output, item), "status": "ok"}
                except Exception as error:
                    return {
                        **base,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": repr(error),
                    }

            checkpoint_rows = int(
                config["experiment"].get("generation", {}).get(
                    "checkpoint_rows", 100
                )
            )
            with JsonlCheckpointWriter(
                baseline_path, sync_every=checkpoint_rows
            ) as writer:
                await _run_batched(
                    missing,
                    concurrency=int(vllm.get("request_concurrency", 16)),
                    worker=generate_one,
                    writer=writer,
                    progress=progress,
                    consecutive_error_limit=int(
                        vllm.get("consecutive_error_limit", 5)
                    ),
                    on_success=lambda row: successful.__setitem__(
                        (str(row["dataset"]), str(row["item_identity"])), row
                    ),
                )
        finally:
            await client.close()

    try:
        if missing:
            asyncio.run(execute())
    except BaseException:
        metadata["status"] = "incomplete"
        write_run_config(baseline_dir, metadata)
        atomic_save_json(metrics_path, _metrics(records, successful))
        raise
    finally:
        progress.close()

    metrics = _metrics(records, successful)
    atomic_save_json(metrics_path, metrics)
    metadata["status"] = "complete" if metrics["coverage"]["complete"] else "incomplete"
    write_run_config(baseline_dir, metadata)
    if not metrics["coverage"]["complete"]:
        raise RuntimeError(
            "baseline generation is incomplete: "
            f"{metrics['coverage']['missing']} successful items are missing; "
            f"resume with --generation-dir {baseline_dir}"
        )
    return {
        "directory": baseline_dir,
        "config": metadata_path,
        "baselines": baseline_path,
        "metrics": metrics_path,
    }

"""RISER-style activation steering with a deterministic fixed router.

This module intentionally implements only the reproducible inference contract of
RISER: a primitive vector library, a binary selection mask, non-negative vector
strengths, weighted-sum composition, and last-token activation injection.  It
does not claim to reproduce the paper's learned Router or its training process.
"""

from __future__ import annotations

import asyncio
import csv
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence
import uuid

import torch
from tqdm.auto import tqdm

from mllm_self_steering.adaptive import (
    _answer_fields,
    _atomic_save_vllm_vector,
    _canonical_rows,
    _item_seed,
    _load_base_vectors,
    _run_batched,
    _server_run_root,
    _validate_preparation_compatibility,
)
from mllm_self_steering.config import capability_map
from mllm_self_steering.datasets.types import MultimodalItem
from mllm_self_steering.models.vllm_hook import (
    AsyncVLLMHookClient,
    build_vllm_messages,
    map_server_path,
)
from mllm_self_steering.pipeline import (
    _image_hashes,
    _limit,
    _paths,
    _runtime_limit,
    _write_stage_manifest,
    item_identity,
)
from mllm_self_steering.runs import (
    preparation_vector_root,
    read_run_config,
    validate_generation_resume,
    write_run_config,
)
from mllm_self_steering.utils.io import (
    JsonlCheckpointWriter,
    atomic_save_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)


RISER_FIXED_CONTRACT = "riser-fixed-router-multimodal-v1"


def parse_component_overrides(values: Sequence[str]) -> dict[str, float]:
    """Parse repeated ``CAPABILITY=STRENGTH`` command-line values."""

    components: dict[str, float] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(
                f"invalid component {raw!r}; expected CAPABILITY=STRENGTH"
            )
        capability, raw_strength = (part.strip() for part in raw.split("=", 1))
        if not capability:
            raise ValueError("fixed-router capability cannot be empty")
        if capability in components:
            raise ValueError(f"duplicate fixed-router capability: {capability}")
        try:
            strength = float(raw_strength)
        except ValueError as exc:
            raise ValueError(
                f"invalid strength for {capability}: {raw_strength!r}"
            ) from exc
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError(
                f"strength for {capability} must be positive and finite"
            )
        components[capability] = strength
    if not components:
        raise ValueError("at least one fixed-router component is required")
    return components


def resolve_fixed_router(
    config: Mapping[str, Any],
    *,
    component_overrides: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], float]:
    """Return a dense strength vector in capability-library order."""

    capability_ids = list(capability_map(dict(config)))
    section = config.get("riser_baseline", {})
    if not isinstance(section, Mapping):
        raise ValueError("riser_baseline must be a mapping")
    router = section.get("fixed_router", {})
    if not isinstance(router, Mapping):
        raise ValueError("riser_baseline.fixed_router must be a mapping")
    raw_components: Any = (
        dict(component_overrides)
        if component_overrides is not None
        else router.get("components")
    )
    if not isinstance(raw_components, Mapping) or not raw_components:
        raise ValueError(
            "riser_baseline.fixed_router.components must be a non-empty mapping"
        )
    alpha_max = router.get("alpha_max", 1.0)
    if (
        isinstance(alpha_max, bool)
        or not isinstance(alpha_max, (int, float))
        or not math.isfinite(float(alpha_max))
        or float(alpha_max) <= 0
    ):
        raise ValueError(
            "riser_baseline.fixed_router.alpha_max must be positive and finite"
        )
    unknown = set(raw_components) - set(capability_ids)
    if unknown:
        raise ValueError(
            "unknown fixed-router capabilities: " + ", ".join(sorted(map(str, unknown)))
        )

    strengths = {capability: 0.0 for capability in capability_ids}
    for raw_capability, raw_strength in raw_components.items():
        capability = str(raw_capability)
        if isinstance(raw_strength, bool) or not isinstance(raw_strength, (int, float)):
            raise ValueError(f"strength for {capability} must be numeric")
        strength = float(raw_strength)
        if not math.isfinite(strength) or not 0 < strength <= float(alpha_max):
            raise ValueError(
                f"strength for {capability} must be in (0, {float(alpha_max)}]"
            )
        strengths[capability] = strength
    return strengths, float(alpha_max)


def compose_fixed_router_vector(
    vectors: Mapping[str, torch.Tensor], strengths: Mapping[str, float]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Implement RISER Eq. (1) without normalizing Router strengths."""

    if set(vectors) != set(strengths):
        raise ValueError("vector and strength capability keys must match")
    if not vectors:
        raise ValueError("at least one primitive vector is required")
    ordered = list(strengths)
    first = vectors[ordered[0]].detach().cpu().float()
    shape = first.shape
    mixed = torch.zeros_like(first)
    component_norms: dict[str, float] = {}
    for capability in ordered:
        vector = vectors[capability].detach().cpu().float()
        if vector.shape != shape:
            raise ValueError(
                f"primitive shape mismatch for {capability}: "
                f"{tuple(vector.shape)} != {tuple(shape)}"
            )
        strength = float(strengths[capability])
        if not math.isfinite(strength) or strength < 0:
            raise ValueError(
                f"invalid non-negative strength for {capability}: {strength}"
            )
        component_norms[capability] = float(vector.norm())
        mixed.add_(vector, alpha=strength)
    if not bool(torch.isfinite(mixed).all()):
        raise ValueError("fixed-router vector contains non-finite values")
    combined_norm = float(mixed.norm())
    if combined_norm == 0.0:
        raise ValueError("fixed-router composition produced a zero vector")
    active_norms = [
        component_norms[name] for name, value in strengths.items() if value > 0
    ]
    max_active_norm = max(active_norms)
    return mixed, {
        "formula": "sum_i(mask_i * strength_i * vector_i)",
        "normalization": "none",
        "selection_mask": {name: int(value > 0) for name, value in strengths.items()},
        "strengths": {name: float(value) for name, value in strengths.items()},
        "selected_capabilities": [
            name for name, value in strengths.items() if value > 0
        ],
        "component_norms": component_norms,
        "combined_norm": combined_norm,
        "max_active_component_norm": max_active_norm,
        "norm_ratio": combined_norm / max_active_norm if max_active_norm > 0 else 0.0,
    }


def fixed_vector_key(
    strengths: Mapping[str, float],
    vector_fingerprints: Mapping[str, str],
    *,
    vector_scaling: str,
) -> str:
    payload = {
        "contract": RISER_FIXED_CONTRACT,
        "strengths": dict(
            sorted((key, float(value)) for key, value in strengths.items())
        ),
        "vector_fingerprints": dict(sorted(vector_fingerprints.items())),
        "vector_scaling": vector_scaling,
        "composition": "unnormalized_weighted_sum",
        "format": "vllm-hook-dir-v1",
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))[:24]


def _experiment_fingerprint(
    config: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    strengths: Mapping[str, float],
    vector_fingerprints: Mapping[str, str],
    data_dir: Path,
) -> str:
    vllm = config["experiment"].get("vllm_hook", {})
    identity = {
        "contract": RISER_FIXED_CONTRACT,
        "preparation_fingerprint": preparation.get("preparation_fingerprint"),
        "vector_fingerprints": dict(sorted(vector_fingerprints.items())),
        "strengths": dict(
            sorted((key, float(value)) for key, value in strengths.items())
        ),
        "model": config["model"].get("name"),
        "revision": config["model"].get("revision"),
        "target_layer": config["experiment"].get("target_layer"),
        "vector_scaling": config["experiment"].get("vector_scaling"),
        "max_new_tokens": config["model"].get("max_new_tokens"),
        "seed": config["experiment"].get("seed", 42),
        "vllm_hook": {
            "protocol": vllm.get("protocol"),
            "phase": vllm.get("phase"),
            "positions": vllm.get("positions"),
            "server_provenance": vllm.get("server_provenance", {}),
        },
        "eval_limit": _runtime_limit(dict(config), "eval"),
        "evaluation": {
            name: sha256_file(data_dir / "processed" / f"{name}.jsonl")
            for name in config["data"]["enabled_datasets"]
        },
    }
    return sha256_text(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )[:16]


def _run_payload(
    config: Mapping[str, Any],
    run_dir: Path,
    preparation_dir: Path,
    preparation: Mapping[str, Any],
    *,
    experiment_fingerprint: str,
    strengths: Mapping[str, float],
    status: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "riser_fixed_router",
        "name": run_dir.name,
        "created_at": created_at or datetime.now().astimezone().isoformat(),
        "status": status,
        "implementation": RISER_FIXED_CONTRACT,
        "approximation": {
            "learned_router": False,
            "router_type": "fixed",
            "shared_across_inputs": True,
        },
        "preparation_dir": str(preparation_dir.resolve()),
        "preparation_fingerprint": preparation.get("preparation_fingerprint"),
        "experiment_fingerprint": experiment_fingerprint,
        "fixed_strengths": dict(strengths),
        "config": deepcopy(dict(config)),
        "artifacts": {
            "fixed_router": "fixed_router.json",
            "vectors": "vectors",
            "generations": "generations.jsonl",
            "metrics_json": "metrics.json",
            "metrics_csv": "metrics.csv",
        },
    }


def _write_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_items: int,
    router: Mapping[str, Any],
    json_path: Path,
    csv_path: Path,
) -> None:
    def group_metrics(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        baseline = [bool(row["baseline_correct"]) for row in selected]
        riser = [bool(row["correct"]) for row in selected]
        count = len(selected)
        wrong_to_right = sum(
            not before and after
            for before, after in zip(baseline, riser, strict=True)
        )
        right_to_wrong = sum(
            before and not after
            for before, after in zip(baseline, riser, strict=True)
        )
        return {
            "count": count,
            "baseline_accuracy": sum(baseline) / count if count else None,
            "riser_accuracy": sum(riser) / count if count else None,
            "accuracy_delta": (sum(riser) - sum(baseline)) / count if count else None,
            "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong,
            "net_flips": wrong_to_right - right_to_wrong,
        }

    by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    groups = {"overall": group_metrics(rows)}
    groups.update(
        {name: group_metrics(values) for name, values in sorted(by_dataset.items())}
    )
    norm_ratios = [float(row["composition"]["norm_ratio"]) for row in rows]
    report = {
        "implementation": RISER_FIXED_CONTRACT,
        "expected_items": expected_items,
        "completed_items": len(rows),
        "complete": len(rows) == expected_items,
        "groups": groups,
        "fixed_router": dict(router),
        "combined_norm_ratio": {
            "mean": fmean(norm_ratios) if norm_ratios else None,
            "max": max(norm_ratios) if norm_ratios else None,
        },
    }
    atomic_save_json(json_path, report)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fieldnames = [
            "dataset",
            "count",
            "baseline_accuracy",
            "riser_accuracy",
            "accuracy_delta",
            "wrong_to_right",
            "right_to_wrong",
            "net_flips",
        ]
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for name, values in groups.items():
                writer.writerow({"dataset": name, **values})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)


def run_riser_fixed_baseline(
    config: dict[str, Any],
    project_root: Path,
    *,
    preparation_dir: Path,
    run_dir: Path,
    component_overrides: Mapping[str, float] | None = None,
) -> dict[str, Path]:
    """Run the isolated RISER fixed-router approximation and paired baseline."""

    if config["experiment"].get("generation", {}).get("backend") != "vllm_hook":
        raise ValueError("RISER fixed-router baseline requires generation.backend=vllm_hook")
    data_dir, configured_outputs = _paths(config, project_root)
    vllm = config["experiment"].get("vllm_hook", {})
    if vllm.get("positions", "last_token") != "last_token":
        raise ValueError("RISER inference requires vllm_hook.positions=last_token")
    if vllm.get("phase", "both") != "both":
        raise ValueError(
            "RISER sustained decoding intervention requires vllm_hook.phase=both"
        )
    preparation_dir = preparation_dir.resolve()
    run_dir = run_dir.resolve()
    preparation = read_run_config(preparation_dir, expected_kind="preparation")
    if preparation.get("status") != "complete":
        raise ValueError(f"preparation is not complete: {preparation_dir}")
    _validate_preparation_compatibility(config, preparation, preparation_dir)
    vector_root = preparation_vector_root(preparation_dir, preparation)

    strengths, alpha_max = resolve_fixed_router(
        config, component_overrides=component_overrides
    )
    capabilities = list(strengths)
    scaling = str(config["experiment"]["vector_scaling"])
    form = {"mean_norm": "steering", "raw": "raw", "unit": "unit"}[scaling]
    base_vectors, base_paths, vector_fingerprints = _load_base_vectors(
        vector_root, capabilities, form
    )
    mixed, composition = compose_fixed_router_vector(base_vectors, strengths)
    vector_key = fixed_vector_key(
        strengths, vector_fingerprints, vector_scaling=scaling
    )
    experiment_fingerprint = _experiment_fingerprint(
        config,
        preparation=preparation,
        strengths=strengths,
        vector_fingerprints=vector_fingerprints,
        data_dir=data_dir,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    expected_metadata = _run_payload(
        config,
        run_dir,
        preparation_dir,
        preparation,
        experiment_fingerprint=experiment_fingerprint,
        strengths=strengths,
        status="running",
    )
    metadata_path = run_dir / "config.json"
    if metadata_path.is_file():
        existing = read_run_config(run_dir, expected_kind="riser_fixed_router")
        validate_generation_resume(existing, expected_metadata, run_dir)
        metadata = dict(existing)
        metadata["status"] = "running"
        metadata["config"] = expected_metadata["config"]
    else:
        if any(run_dir.iterdir()):
            raise ValueError(f"run directory is non-empty but has no config.json: {run_dir}")
        metadata = expected_metadata
    write_run_config(run_dir, metadata)
    run_id = str(metadata["name"])

    vector_path = run_dir / "vectors" / f"fixed_{vector_key}.pt"
    # Re-materialize on resume as well. This makes the composite a deterministic
    # derivative of the frozen library instead of trusting a mutable prior file.
    _atomic_save_vllm_vector(vector_path, mixed)
    warning_ratio = float(
        config.get("riser_baseline", {}).get("norm_warning_ratio", 2.0)
    )
    if not math.isfinite(warning_ratio) or warning_ratio <= 0:
        raise ValueError("riser_baseline.norm_warning_ratio must be positive and finite")
    composition.update(
        {
            "vector_hash": vector_key,
            "vector_scaling": scaling,
            "norm_warning_ratio": warning_ratio,
            "norm_warning": composition["norm_ratio"] > warning_ratio,
        }
    )
    router_payload = {
        "implementation": RISER_FIXED_CONTRACT,
        "learned_router": False,
        "router_type": "fixed",
        "shared_across_inputs": True,
        "alpha_max": alpha_max,
        "selection_mask": composition["selection_mask"],
        "strengths": composition["strengths"],
        "selected_capabilities": composition["selected_capabilities"],
        "composition": composition,
        "base_vectors": {
            name: {
                "path": str(base_paths[name]),
                "fingerprint": vector_fingerprints[name],
            }
            for name in capabilities
        },
        "composite_vector": {
            "path": str(vector_path),
            "sha256": sha256_file(vector_path),
        },
    }
    router_path = run_dir / "fixed_router.json"
    atomic_save_json(router_path, router_payload)

    items: list[MultimodalItem] = []
    for dataset in config["data"]["enabled_datasets"]:
        rows = list(read_jsonl(data_dir / "processed" / f"{dataset}.jsonl"))
        items.extend(
            MultimodalItem.from_dict(row)
            for row in _limit(config, rows, scope="eval")
        )
    records: list[tuple[MultimodalItem, list[str], str]] = []
    for item in items:
        image_hashes = _image_hashes(item)
        records.append(
            (item, image_hashes, item_identity(item, image_hashes=image_hashes))
        )

    generation_path = run_dir / "generations.jsonl"
    metrics_json = run_dir / "metrics.json"
    metrics_csv = run_dir / "metrics.csv"
    generation_success = _canonical_rows(
        generation_path, run_id, successful_only=True
    )
    generation_latest = _canonical_rows(
        generation_path, run_id, successful_only=False
    )
    section = config.get("riser_baseline", {})
    concurrency = int(section.get("generation_concurrency", 4))
    if concurrency < 1:
        raise ValueError("riser_baseline.generation_concurrency must be positive")
    checkpoint_rows = int(
        config["experiment"].get("generation", {}).get("checkpoint_rows", 100)
    )
    base_seed = int(config["experiment"].get("seed", 42))
    run_server_root = _server_run_root(
        run_dir,
        outputs_root=configured_outputs,
        server_vector_root=vllm.get("server_vector_root"),
    )
    server_vector_path = map_server_path(
        vector_path, local_root=run_dir, server_root=run_server_root
    )

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
            missing = [
                record
                for record in records
                if (record[0].dataset, record[2]) not in generation_success
            ]
            if missing and bool(vllm.get("verify_steering_effect", True)):
                probe_item, probe_hashes, _ = missing[0]
                probe_messages = build_vllm_messages(
                    probe_item.prompt,
                    probe_item.image_paths,
                    answer_type=probe_item.answer_type,
                    media_transport=str(vllm.get("media_transport", "file")),
                    local_media_root=data_dir,
                    server_media_root=vllm.get("server_media_root"),
                    media_uuids=probe_hashes,
                )
                await client.verify_steering_effect(
                    messages=probe_messages,
                    vector_path=server_vector_path,
                    target_layer=int(config["experiment"]["target_layer"]),
                    alpha=float(vllm.get("steering_probe_alpha") or 1.0),
                    phase=str(vllm.get("phase", "both")),
                    positions=str(vllm.get("positions", "last_token")),
                    top_logprobs=int(vllm.get("steering_probe_top_logprobs", 20)),
                    tolerance=float(vllm.get("steering_probe_tolerance", 1e-6)),
                )

            progress = tqdm(
                total=len(records),
                initial=len(records) - len(missing),
                desc=f"{run_id}:riser-fixed",
            )

            async def generate_one(
                record: tuple[MultimodalItem, list[str], str],
            ) -> dict[str, Any]:
                item, image_hashes, identity = record
                key = (item.dataset, identity)
                base = {
                    "run_id": run_id,
                    "implementation": RISER_FIXED_CONTRACT,
                    "capture_artifact_id": preparation["capture_artifact_id"],
                    "dataset": item.dataset,
                    "item_id": item.item_id,
                    "item_identity": identity,
                    "gold_answer": item.gold_answer,
                    "answer_type": item.answer_type,
                    "backend": "vllm_hook",
                    "vllm_hook_protocol": str(vllm.get("protocol", "mia")),
                    "vector_scaling": scaling,
                    "router_type": "fixed",
                    "selection_mask": composition["selection_mask"],
                    "strengths": composition["strengths"],
                    "selected_capabilities": composition["selected_capabilities"],
                    "composition": composition,
                    "steering_applied": True,
                }
                baseline_fields: dict[str, Any] = {}
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
                    baseline_output: str | None = None
                    partial = generation_latest.get(key)
                    if partial is not None:
                        cached = partial.get("baseline_raw_output")
                        if isinstance(cached, str) and cached:
                            baseline_output = cached
                    if baseline_output is None:
                        baseline_output = await client.generate(
                            messages=messages,
                            vector_path=None,
                            target_layer=int(config["experiment"]["target_layer"]),
                            alpha=0.0,
                            phase=str(vllm.get("phase", "both")),
                            positions=str(vllm.get("positions", "last_token")),
                            max_new_tokens=int(
                                config["model"].get("max_new_tokens", 2048)
                            ),
                            temperature=0.0,
                            seed=_item_seed(base_seed, identity),
                        )
                    parsed = _answer_fields(baseline_output, item)
                    baseline_fields = {
                        "baseline_raw_output": parsed["raw_output"],
                        "baseline_predicted_answer": parsed["predicted_answer"],
                        "baseline_correct": parsed["correct"],
                    }
                    output = await client.generate(
                        messages=messages,
                        vector_path=server_vector_path,
                        target_layer=int(config["experiment"]["target_layer"]),
                        alpha=1.0,
                        phase=str(vllm.get("phase", "both")),
                        positions=str(vllm.get("positions", "last_token")),
                        max_new_tokens=int(
                            config["model"].get("max_new_tokens", 2048)
                        ),
                        temperature=0.0,
                        seed=_item_seed(base_seed, identity),
                    )
                    return {
                        **base,
                        **baseline_fields,
                        **_answer_fields(output, item),
                        "status": "ok",
                    }
                except Exception as error:
                    return {
                        **base,
                        **baseline_fields,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": repr(error),
                    }

            try:
                with JsonlCheckpointWriter(
                    generation_path, sync_every=checkpoint_rows
                ) as writer:
                    await _run_batched(
                        missing,
                        concurrency=concurrency,
                        worker=generate_one,
                        writer=writer,
                        progress=progress,
                        consecutive_error_limit=int(
                            vllm.get("consecutive_error_limit", 5)
                        ),
                        on_success=lambda row: generation_success.__setitem__(
                            (str(row["dataset"]), str(row["item_identity"])), row
                        ),
                    )
            finally:
                progress.close()
        finally:
            await client.close()

    if len(generation_success) < len(records):
        asyncio.run(execute())

    generation_success = _canonical_rows(
        generation_path, run_id, successful_only=True
    )
    _write_metrics(
        list(generation_success.values()),
        expected_items=len(records),
        router=router_payload,
        json_path=metrics_json,
        csv_path=metrics_csv,
    )
    artifacts = {
        "fixed_router": router_path,
        "composite_vector": vector_path,
        "generations": generation_path,
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
    }
    _write_stage_manifest(config, run_dir, "run_riser_fixed_baseline", artifacts)
    complete = len(generation_success) == len(records)
    metadata["status"] = "complete" if complete else "incomplete"
    artifacts["config"] = write_run_config(run_dir, metadata)
    if not complete:
        raise RuntimeError(
            f"RISER fixed-router run is incomplete: "
            f"{len(generation_success)}/{len(records)} successful items; resume with --run-dir"
        )
    return artifacts

"""End-to-end multimodal self-steering pipeline."""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

from mllm_self_steering.config import capability_map
from mllm_self_steering.datasets.pairs import load_capability_pairs, resolve_project_path
from mllm_self_steering.datasets.registry import load_dataset_items
from mllm_self_steering.datasets.types import MultimodalItem
from mllm_self_steering.evaluation.answers import extract_answer, is_correct
from mllm_self_steering.evaluation.metrics import (
    accuracy_by_alpha,
    assert_complete_population,
    flip_report,
)
from mllm_self_steering.models.generation import generate_transformers
from mllm_self_steering.models.loader import resolve_language_layer
from mllm_self_steering.models.vllm_hook import (
    AsyncVLLMHookClient,
    build_vllm_messages,
    map_server_path,
)
from mllm_self_steering.utils.io import (
    JsonlCheckpointWriter,
    append_jsonl,
    atomic_save_json,
    atomic_save_tensors,
    read_jsonl,
    repair_truncated_jsonl,
    sha256_file,
    sha256_text,
    write_jsonl,
)
from mllm_self_steering.utils.seed import seed_everything
from mllm_self_steering.vectors.capture import capture_contrast
from mllm_self_steering.vectors.extract import aggregate_vectors
from mllm_self_steering.vectors.similarity import coherence, similarity_matrix
from mllm_self_steering.vectors.storage import export_vllm_hook, load_library, save_library


def _paths(config: dict[str, Any], project_root: Path) -> tuple[Path, Path]:
    configured = config["experiment"].get("paths", {})
    data = resolve_project_path(project_root, str(configured.get("data_dir", "data")))
    outputs = resolve_project_path(project_root, str(configured.get("outputs_dir", "outputs")))
    return data, outputs


def _limit(config: dict[str, Any], rows: list[Any]) -> list[Any]:
    value = config.get("_runtime", {}).get("limit")
    return rows[: int(value)] if value is not None else rows


def _identity(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))[:16]


def _write_stage_manifest(
    config: dict[str, Any], outputs: Path, stage: str, artifacts: Mapping[str, Path]
) -> None:
    fingerprints = {
        key: sha256_file(path) for key, path in artifacts.items() if path.is_file()
    }
    payload = {"stage": stage, "config": config, "artifacts": fingerprints}
    payload["run_id"] = _identity(payload)
    atomic_save_json(outputs / "manifests" / f"{stage}.json", payload)


def capture_artifact_id(config: dict[str, Any], project_root: Path) -> str:
    pair_hashes = {}
    image_hashes: dict[str, dict[str, str]] = {}
    for row in config["experiment"]["capabilities"]:
        path = resolve_project_path(project_root, str(row["pair_path"]))
        name = str(row["id"])
        pair_hashes[name] = sha256_file(path)
        pairs = load_capability_pairs(project_root, row)
        referenced = {
            str(resolve_project_path(project_root, image))
            for pair in pairs
            for image in pair.positive_image_paths + pair.negative_image_paths
        }
        image_hashes[name] = {
            image: sha256_file(image) for image in sorted(referenced)
        }
    identity = {
        "model": config["model"].get("name"),
        "revision": config["model"].get("revision"),
        "dtype": config["model"].get("dtype"),
        "target_layer": config["experiment"]["target_layer"],
        "capabilities": config["experiment"]["capabilities"],
        "pair_hashes": pair_hashes,
        "image_hashes": image_hashes,
        "limit": config.get("_runtime", {}).get("limit"),
        "capture_contract": "qwen3-vl-language-decoder-final-prompt-token-v1",
    }
    return _identity(identity)


def prepare_data(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    data_dir, outputs = _paths(config, project_root)
    result: dict[str, Path] = {}
    for name in config["data"]["enabled_datasets"]:
        source = dict(config["data"]["sources"][name])
        source.setdefault("cache_dir", config["data"].get("cache_dir"))
        if source.get("local_path"):
            source["local_path"] = str(resolve_project_path(project_root, source["local_path"]))
        items = load_dataset_items(
            name,
            source,
            image_root=data_dir / "images",
            limit=config.get("_runtime", {}).get("limit"),
        )
        path = data_dir / "processed" / f"{name}.jsonl"
        write_jsonl(path, (item.to_dict() for item in items))
        result[name] = path
    _write_stage_manifest(config, outputs, "00_prepare_data", result)
    return result


def validate_pairs(config: dict[str, Any], project_root: Path) -> Path:
    _, outputs = _paths(config, project_root)
    report = {"capabilities": {}}
    artifacts: dict[str, Path] = {}
    for capability in config["experiment"]["capabilities"]:
        pairs = load_capability_pairs(project_root, capability)
        pair_path = resolve_project_path(project_root, capability["pair_path"])
        artifacts[str(capability["id"])] = pair_path
        report["capabilities"][str(capability["id"])] = {
            "modality": capability["modality"],
            "count": len(pairs),
            "pair_sha256": sha256_file(pair_path),
        }
    destination = outputs / "pair_validation.json"
    atomic_save_json(destination, report)
    artifacts["report"] = destination
    _write_stage_manifest(config, outputs, "01_validate_pairs", artifacts)
    return destination


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def capture_contrasts(
    config: dict[str, Any], project_root: Path, model: Any, processor: Any
) -> list[Path]:
    _, outputs = _paths(config, project_root)
    seed_everything(int(config["experiment"].get("seed", 42)))
    layer = resolve_language_layer(model, int(config["experiment"]["target_layer"]))
    capture_id = capture_artifact_id(config, project_root)
    root = outputs / "activations" / capture_id
    index: dict[str, list[str]] = {}
    written: list[Path] = []
    for capability in config["experiment"]["capabilities"]:
        name = str(capability["id"])
        index[name] = []
        pairs = _limit(config, load_capability_pairs(project_root, capability))
        for pair in tqdm(pairs, desc=name):
            path = root / name / f"{_safe_name(pair.pair_id)}.safetensors"
            if not path.is_file():
                try:
                    delta = capture_contrast(
                        model,
                        processor,
                        layer,
                        positive_prompt=pair.positive_prompt,
                        negative_prompt=pair.negative_prompt,
                        positive_image_paths=[
                            str(resolve_project_path(project_root, value))
                            for value in pair.positive_image_paths
                        ],
                        negative_image_paths=[
                            str(resolve_project_path(project_root, value))
                            for value in pair.negative_image_paths
                        ],
                    )
                    atomic_save_tensors(path, {"delta": delta})
                except Exception as error:
                    append_jsonl(
                        root / "errors.jsonl",
                        {
                            "status": "error",
                            "capability": name,
                            "pair_id": pair.pair_id,
                            "error_type": type(error).__name__,
                            "error": repr(error),
                        },
                    )
                    continue
            relative = str(path.relative_to(root)).replace("\\", "/")
            index[name].append(relative)
            written.append(path)
    index_path = root / "index.json"
    atomic_save_json(index_path, {"capture_artifact_id": capture_id, "shards": index})
    _write_stage_manifest(config, outputs, "02_capture_contrasts", {"index": index_path})
    return written


def extract_vectors(config: dict[str, Any], project_root: Path) -> Path:
    _, outputs = _paths(config, project_root)
    capture_id = capture_artifact_id(config, project_root)
    activation_root = outputs / "activations" / capture_id
    index = json.loads((activation_root / "index.json").read_text(encoding="utf-8"))
    deltas: dict[str, torch.Tensor] = {}
    for capability in capability_map(config):
        paths = [activation_root / value for value in index["shards"].get(capability, [])]
        if not paths:
            raise ValueError(f"no successful activation shards for {capability}")
        deltas[capability] = torch.stack([load_file(str(path))["delta"] for path in paths])
    vectors = aggregate_vectors(deltas)
    root = outputs / "vectors" / capture_id
    tensor_path = root / "capability_vectors.safetensors"
    metadata_path = root / "capability_vectors.json"
    vllm_paths = export_vllm_hook(root / "vllm_hook", vectors)
    save_library(
        tensor_path,
        metadata_path,
        vectors,
        metadata={
            "capture_artifact_id": capture_id,
            "target_layer": int(config["experiment"]["target_layer"]),
            "counts": {name: int(values.shape[0]) for name, values in deltas.items()},
            "vllm_hook": vllm_paths,
        },
    )
    _write_stage_manifest(config, outputs, "03_extract_vectors", {"vectors": tensor_path})
    return tensor_path


def analyze_vectors(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    _, outputs = _paths(config, project_root)
    capture_id = capture_artifact_id(config, project_root)
    activation_root = outputs / "activations" / capture_id
    vector_root = outputs / "vectors" / capture_id
    index = json.loads((activation_root / "index.json").read_text(encoding="utf-8"))
    vectors, _ = load_library(
        vector_root / "capability_vectors.safetensors",
        vector_root / "capability_vectors.json",
    )
    coherence_values = {}
    for capability in vectors:
        values = torch.stack(
            [
                load_file(str(activation_root / relative))["delta"]
                for relative in index["shards"][capability]
            ]
        )
        coherence_values[capability] = coherence(values)
    similarity_path = outputs / "metrics" / f"{capture_id}_vector_similarity.json"
    coherence_path = outputs / "metrics" / f"{capture_id}_vector_coherence.json"
    atomic_save_json(
        similarity_path,
        similarity_matrix({name: forms["unit"] for name, forms in vectors.items()}),
    )
    atomic_save_json(coherence_path, coherence_values)
    _write_stage_manifest(
        config,
        outputs,
        "04_analyze_vectors",
        {"similarity": similarity_path, "coherence": coherence_path},
    )
    return {"similarity": similarity_path, "coherence": coherence_path}


def _image_hashes(item: MultimodalItem) -> list[str]:
    return [sha256_file(path) for path in item.image_paths]


def item_identity(
    item: MultimodalItem, *, image_hashes: list[str] | None = None
) -> str:
    return sha256_text(
        json.dumps(
            {
                "dataset": item.dataset,
                "item_id": item.item_id,
                "prompt": item.prompt,
                "gold_answer": item.gold_answer,
                "answer_type": item.answer_type,
                "image_sha256": (
                    image_hashes if image_hashes is not None else _image_hashes(item)
                ),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )


def steering_artifact_id(config: dict[str, Any], project_root: Path) -> str:
    data_dir, outputs = _paths(config, project_root)
    capture_id = capture_artifact_id(config, project_root)
    vector_root = outputs / "vectors" / capture_id
    backend = config["experiment"].get("generation", {}).get(
        "backend", "transformers"
    )
    scaling = str(config["experiment"]["vector_scaling"])
    form = {"mean_norm": "steering", "raw": "raw", "unit": "unit"}[scaling]
    if backend == "vllm_hook":
        vector_fingerprints: Any = {
            capability: sha256_file(
                vector_root / "vllm_hook" / f"{capability}__{form}.pt"
            )
            for capability in capability_map(config)
        }
    else:
        vector_fingerprints = sha256_file(
            vector_root / "capability_vectors.safetensors"
        )
    vllm_config = config["experiment"].get("vllm_hook", {})
    identity = {
        "capture_id": capture_id,
        "vector_fingerprints": vector_fingerprints,
        "model": config["model"].get("name"),
        "revision": config["model"].get("revision"),
        "target_layer": config["experiment"]["target_layer"],
        "scaling": scaling,
        "alphas": config["experiment"]["alphas"],
        "backend": backend,
        "vllm_hook": {
            **{key: vllm_config.get(key) for key in ("phase", "positions")},
            "server_provenance": vllm_config.get("server_provenance", {}),
        }
        if backend == "vllm_hook"
        else None,
        "max_new_tokens": config["model"].get("max_new_tokens"),
        "limit": config.get("_runtime", {}).get("limit"),
        "evaluation": {
            name: sha256_file(data_dir / "processed" / f"{name}.jsonl")
            for name in config["data"]["enabled_datasets"]
        },
        "generation_contract": "multimodal-final-answer-v1",
    }
    return _identity(identity)


def _canonical_success(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("run_id") != run_id:
            continue
        key = (
            str(row["dataset"]),
            str(row["item_identity"]),
            str(row["steering_capability"]),
            float(row["alpha"]),
        )
        latest[key] = row
    return list(latest.values())


GenerationKey = tuple[str, str, str, float]


@dataclass(frozen=True)
class _GenerationTarget:
    key: GenerationKey
    base: dict[str, Any]


@dataclass(frozen=True)
class _VLLMJob:
    messages: list[dict[str, Any]]
    vector_path: str | None
    alpha: float
    targets: tuple[_GenerationTarget, ...]
    cached_output: str | None = None


def _success_row(base: Mapping[str, Any], output: str) -> dict[str, Any]:
    answer_type = str(base["answer_type"])
    prediction = extract_answer(output, answer_type)
    return {
        **base,
        "raw_output": output,
        "predicted_answer": prediction,
        "correct": is_correct(
            prediction,
            base["gold_answer"],
            answer_type=answer_type,
        ),
        "status": "ok",
    }


async def _execute_vllm_jobs(
    jobs: Iterable[_VLLMJob],
    *,
    client: AsyncVLLMHookClient,
    target_layer: int,
    phase: str,
    positions: str,
    max_new_tokens: int,
    concurrency: int,
    consecutive_error_limit: int,
    writer: JsonlCheckpointWriter,
    completed: set[GenerationKey],
    progress: Any,
) -> None:
    """Consume a rolling request window so vLLM can continuously batch work."""

    async def run_one(
        job: _VLLMJob,
    ) -> tuple[_VLLMJob, str | None, Exception | None]:
        if job.cached_output is not None:
            return job, job.cached_output, None
        try:
            output = await client.generate(
                messages=job.messages,
                vector_path=job.vector_path,
                target_layer=target_layer,
                alpha=job.alpha,
                phase=phase,
                positions=positions,
                max_new_tokens=max_new_tokens,
            )
            return job, output, None
        except Exception as error:
            return job, None, error

    iterator = iter(jobs)
    pending: set[asyncio.Task[tuple[_VLLMJob, str | None, Exception | None]]] = set()
    exhausted = False

    def fill_window() -> None:
        nonlocal exhausted
        while not exhausted and len(pending) < concurrency:
            try:
                job = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending.add(asyncio.create_task(run_one(job)))

    error_streak = 0
    last_error: Exception | None = None
    try:
        fill_window()
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                job, output, error = task.result()
                if error is None and output is not None:
                    error_streak = 0
                    for target in job.targets:
                        writer.write(_success_row(target.base, output))
                        completed.add(target.key)
                        progress.update(1)
                else:
                    error_streak += 1
                    last_error = error or RuntimeError("vLLM returned no output")
                    for target in job.targets:
                        writer.write(
                            {
                                **target.base,
                                "status": "error",
                                "error_type": type(last_error).__name__,
                                "error": repr(last_error),
                            }
                        )
                        progress.update(1)
            if error_streak >= consecutive_error_limit:
                writer.checkpoint()
                raise RuntimeError(
                    "aborting vLLM generation after "
                    f"{error_streak} consecutive request failures"
                ) from last_error
            fill_window()
    finally:
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)


async def _run_vllm_steering(
    *,
    config: dict[str, Any],
    data_dir: Path,
    outputs: Path,
    vector_root: Path,
    items: list[MultimodalItem],
    capabilities: list[str],
    alphas: list[float],
    form: str,
    scaling: str,
    capture_id: str,
    run_id: str,
    completed: set[GenerationKey],
    baseline_cache: dict[tuple[str, str], str],
    writer: JsonlCheckpointWriter,
    progress: Any,
) -> None:
    vllm_config = config["experiment"].get("vllm_hook", {})
    local_vector_root = outputs / "vectors"
    server_vector_root = vllm_config.get("server_vector_root")
    vector_paths: dict[str, str] = {}
    for capability in capabilities:
        exported = vector_root / "vllm_hook" / f"{capability}__{form}.pt"
        if not exported.is_file():
            raise FileNotFoundError(
                f"missing vLLM-Hook vector {exported}; run extract-vectors first"
            )
        vector_paths[capability] = map_server_path(
            exported,
            local_root=local_vector_root,
            server_root=server_vector_root,
        )

    client = AsyncVLLMHookClient(
        base_url=str(vllm_config.get("base_url", "http://localhost:8770/v1")),
        api_key=str(vllm_config.get("api_key", "EMPTY")),
        model_name=str(config["model"]["name"]),
        timeout_seconds=float(vllm_config.get("request_timeout_seconds", 600)),
        max_retries=int(vllm_config.get("max_retries", 1)),
    )

    def base_row(
        item: MultimodalItem,
        identity: str,
        capability: str,
        alpha: float,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "capture_artifact_id": capture_id,
            "dataset": item.dataset,
            "item_id": item.item_id,
            "item_identity": identity,
            "steering_capability": capability,
            "alpha": alpha,
            "gold_answer": item.gold_answer,
            "answer_type": item.answer_type,
            "backend": "vllm_hook",
            "vector_scaling": scaling,
        }

    def jobs() -> Iterable[_VLLMJob]:
        for item in items:
            image_hashes = _image_hashes(item)
            identity = item_identity(item, image_hashes=image_hashes)
            missing = any(
                (item.dataset, identity, capability, alpha) not in completed
                for capability in capabilities
                for alpha in alphas
            )
            if not missing:
                continue
            messages = build_vllm_messages(
                item.prompt,
                item.image_paths,
                answer_type=item.answer_type,
                media_transport=str(vllm_config.get("media_transport", "file")),
                local_media_root=data_dir,
                server_media_root=vllm_config.get("server_media_root"),
                media_uuids=image_hashes,
            )
            baseline_targets = tuple(
                _GenerationTarget(
                    key=(item.dataset, identity, capability, 0.0),
                    base=base_row(item, identity, capability, 0.0),
                )
                for capability in capabilities
                if 0.0 in alphas
                and (item.dataset, identity, capability, 0.0) not in completed
            )
            if baseline_targets:
                yield _VLLMJob(
                    messages=messages,
                    vector_path=None,
                    alpha=0.0,
                    targets=baseline_targets,
                    cached_output=baseline_cache.get((item.dataset, identity)),
                )
            for capability in capabilities:
                for alpha in alphas:
                    key = (item.dataset, identity, capability, alpha)
                    if alpha == 0.0 or key in completed:
                        continue
                    yield _VLLMJob(
                        messages=messages,
                        vector_path=vector_paths[capability],
                        alpha=alpha,
                        targets=(
                            _GenerationTarget(
                                key=key,
                                base=base_row(item, identity, capability, alpha),
                            ),
                        ),
                    )

    try:
        await client.preflight()
        nonzero_alphas = [alpha for alpha in alphas if alpha != 0.0]
        if bool(vllm_config.get("verify_steering_effect", True)) and nonzero_alphas:
            configured_probe_alpha = vllm_config.get("steering_probe_alpha")
            probe_target: tuple[MultimodalItem, str, list[str], float] | None = None
            for probe_item in items:
                hashes = _image_hashes(probe_item)
                identity = item_identity(probe_item, image_hashes=hashes)
                for capability in capabilities:
                    incomplete_alphas = [
                        alpha
                        for alpha in nonzero_alphas
                        if (probe_item.dataset, identity, capability, alpha)
                        not in completed
                    ]
                    if incomplete_alphas:
                        probe_alpha = (
                            float(configured_probe_alpha)
                            if configured_probe_alpha is not None
                            else max(incomplete_alphas, key=abs)
                        )
                        probe_target = (
                            probe_item,
                            capability,
                            hashes,
                            probe_alpha,
                        )
                        break
                if probe_target is not None:
                    break
            if probe_target is not None:
                probe_item, probe_capability, hashes, probe_alpha = probe_target
                probe_messages = build_vllm_messages(
                    probe_item.prompt,
                    probe_item.image_paths,
                    answer_type=probe_item.answer_type,
                    media_transport=str(vllm_config.get("media_transport", "file")),
                    local_media_root=data_dir,
                    server_media_root=vllm_config.get("server_media_root"),
                    media_uuids=hashes,
                )
                await client.verify_steering_effect(
                    messages=probe_messages,
                    vector_path=vector_paths[probe_capability],
                    target_layer=int(config["experiment"]["target_layer"]),
                    alpha=probe_alpha,
                    phase=str(vllm_config.get("phase", "both")),
                    positions=str(vllm_config.get("positions", "last_token")),
                    top_logprobs=int(vllm_config.get("steering_probe_top_logprobs", 20)),
                    tolerance=float(vllm_config.get("steering_probe_tolerance", 1e-6)),
                )
        await _execute_vllm_jobs(
            jobs(),
            client=client,
            target_layer=int(config["experiment"]["target_layer"]),
            phase=str(vllm_config.get("phase", "both")),
            positions=str(vllm_config.get("positions", "last_token")),
            max_new_tokens=int(config["model"].get("max_new_tokens", 2048)),
            concurrency=int(vllm_config.get("request_concurrency", 16)),
            consecutive_error_limit=int(
                vllm_config.get("consecutive_error_limit", 5)
            ),
            writer=writer,
            completed=completed,
            progress=progress,
        )
    finally:
        await client.close()


def run_steering(
    config: dict[str, Any],
    project_root: Path,
    *,
    model: Any = None,
    processor: Any = None,
) -> Path:
    data_dir, outputs = _paths(config, project_root)
    seed_everything(int(config["experiment"].get("seed", 42)))
    capture_id = capture_artifact_id(config, project_root)
    vector_root = outputs / "vectors" / capture_id
    scaling = str(config["experiment"]["vector_scaling"])
    form = {"mean_norm": "steering", "raw": "raw", "unit": "unit"}[scaling]
    backend = config["experiment"].get("generation", {}).get("backend", "transformers")
    if backend == "transformers" and (model is None or processor is None):
        raise ValueError("transformers backend requires model and processor")
    vectors: dict[str, dict[str, torch.Tensor]] = {}
    if backend == "transformers":
        vectors, _ = load_library(
            vector_root / "capability_vectors.safetensors",
            vector_root / "capability_vectors.json",
        )
    layer = (
        resolve_language_layer(model, int(config["experiment"]["target_layer"]))
        if backend == "transformers"
        else None
    )
    run_id = steering_artifact_id(config, project_root)
    output_path = outputs / "generations" / f"{run_id}.jsonl"
    if output_path.is_file():
        repair_truncated_jsonl(output_path)
    existing = list(read_jsonl(output_path)) if output_path.is_file() else []
    current = _canonical_success(existing, run_id)
    completed = {
        (
            row["dataset"],
            row["item_identity"],
            row["steering_capability"],
            float(row["alpha"]),
        )
        for row in current
    }
    baseline_cache = {
        (row["dataset"], row["item_identity"]): row["raw_output"]
        for row in current
        if float(row["alpha"]) == 0.0
    }
    items: list[MultimodalItem] = []
    for dataset in config["data"]["enabled_datasets"]:
        rows = list(read_jsonl(data_dir / "processed" / f"{dataset}.jsonl"))
        items.extend(MultimodalItem.from_dict(row) for row in _limit(config, rows))
    capabilities = list(capability_map(config))
    alphas = [float(value) for value in config["experiment"]["alphas"]]
    total = len(items) * len(capabilities) * len(alphas)
    progress = tqdm(total=total, initial=len(completed), desc=run_id)
    checkpoint_rows = int(
        config["experiment"].get("generation", {}).get("checkpoint_rows", 100)
    )
    try:
        with JsonlCheckpointWriter(output_path, sync_every=checkpoint_rows) as writer:
            if len(completed) >= total:
                pass
            elif backend == "vllm_hook":
                asyncio.run(
                    _run_vllm_steering(
                        config=config,
                        data_dir=data_dir,
                        outputs=outputs,
                        vector_root=vector_root,
                        items=items,
                        capabilities=capabilities,
                        alphas=alphas,
                        form=form,
                        scaling=scaling,
                        capture_id=capture_id,
                        run_id=run_id,
                        completed=completed,
                        baseline_cache=baseline_cache,
                        writer=writer,
                        progress=progress,
                    )
                )
            else:
                for item in items:
                    identity = item_identity(item)
                    for capability in capabilities:
                        for alpha in alphas:
                            key = (item.dataset, identity, capability, alpha)
                            if key in completed:
                                continue
                            base = {
                                "run_id": run_id,
                                "capture_artifact_id": capture_id,
                                "dataset": item.dataset,
                                "item_id": item.item_id,
                                "item_identity": identity,
                                "steering_capability": capability,
                                "alpha": alpha,
                                "gold_answer": item.gold_answer,
                                "answer_type": item.answer_type,
                                "backend": backend,
                                "vector_scaling": scaling,
                            }
                            try:
                                baseline_key = (item.dataset, identity)
                                if alpha == 0.0 and baseline_key in baseline_cache:
                                    output = baseline_cache[baseline_key]
                                else:
                                    output = generate_transformers(
                                        model,
                                        processor,
                                        prompt=item.prompt,
                                        image_paths=item.image_paths,
                                        answer_type=item.answer_type,
                                        layer=layer,
                                        vector=(
                                            vectors[capability][form]
                                            if alpha != 0
                                            else None
                                        ),
                                        alpha=alpha,
                                        max_new_tokens=int(
                                            config["model"].get(
                                                "max_new_tokens", 2048
                                            )
                                        ),
                                    )
                                    if alpha == 0.0:
                                        baseline_cache[baseline_key] = output
                                writer.write(_success_row(base, output))
                                completed.add(key)
                            except Exception as error:
                                writer.write(
                                    {
                                        **base,
                                        "status": "error",
                                        "error_type": type(error).__name__,
                                        "error": repr(error),
                                    }
                                )
                            progress.update(1)
    finally:
        progress.close()
    _write_stage_manifest(config, outputs, "05_run_steering", {"generations": output_path})
    return output_path


def score_generations(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    data_dir, outputs = _paths(config, project_root)
    run_id = steering_artifact_id(config, project_root)
    generation_path = outputs / "generations" / f"{run_id}.jsonl"
    rows = _canonical_success(list(read_jsonl(generation_path)), run_id)
    expected_items: dict[str, list[str]] = {}
    for dataset in config["data"]["enabled_datasets"]:
        source = _limit(
            config, list(read_jsonl(data_dir / "processed" / f"{dataset}.jsonl"))
        )
        expected_items[dataset] = [
            item_identity(MultimodalItem.from_dict(row)) for row in source
        ]
    capabilities = list(capability_map(config))
    alphas = [float(value) for value in config["experiment"]["alphas"]]
    assert_complete_population(
        rows,
        expected_items=expected_items,
        capabilities=capabilities,
        alphas=alphas,
    )
    report: dict[str, Any] = {"run_id": run_id, "datasets": {}}
    csv_rows: list[dict[str, Any]] = []
    for dataset in config["data"]["enabled_datasets"]:
        report["datasets"][dataset] = {}
        for capability in capabilities:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["steering_capability"] == capability
            ]
            by_alpha = accuracy_by_alpha(selected)
            flips = {
                str(alpha): flip_report(selected, alpha=alpha)
                for alpha in alphas
                if alpha != 0.0
            }
            report["datasets"][dataset][capability] = {
                "by_alpha": by_alpha,
                "flips_vs_baseline": flips,
            }
            for alpha, values in by_alpha.items():
                csv_rows.append(
                    {
                        "dataset": dataset,
                        "capability": capability,
                        "alpha": alpha,
                        **values,
                    }
                )
    json_path = outputs / "metrics" / f"{run_id}_steering_metrics.json"
    csv_path = outputs / "metrics" / f"{run_id}_steering_metrics.csv"
    atomic_save_json(json_path, report)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "capability", "alpha", "count", "accuracy", "delta"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    _write_stage_manifest(
        config, outputs, "06_score_generations", {"json": json_path, "csv": csv_path}
    )
    return {"json": json_path, "csv": csv_path}

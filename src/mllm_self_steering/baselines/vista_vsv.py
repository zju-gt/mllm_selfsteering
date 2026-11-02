"""VISTA VSV-Only baseline adapted to Qwen3-VL.

VSV is input-conditioned rather than a global steering direction.  For every
evaluation item this module contrasts the last context-token residual streams
with and without visual tokens, saves one direction per language layer, and
applies those directions during generation using the norm-preserving transform
from the official VISTA implementation.  Self-Logits Augmentation is deliberately
excluded: this module implements the VSV-Only ablation.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import csv
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Iterator, Mapping, Sequence
import uuid

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from mllm_self_steering.datasets.types import MultimodalItem
from mllm_self_steering.evaluation.answers import extract_answer, is_correct
from mllm_self_steering.hooks.capture import hidden_from_output
from mllm_self_steering.models.inputs import prepare_transformers_inputs
from mllm_self_steering.models.loader import (
    load_model_and_processor,
    model_input_device,
    resolve_language_layers,
)
from mllm_self_steering.pipeline import (
    _image_hashes,
    _limit,
    _paths,
    _runtime_limit,
    _write_stage_manifest,
    item_identity,
)
from mllm_self_steering.runs import read_run_config, write_run_config
from mllm_self_steering.utils.io import (
    JsonlCheckpointWriter,
    atomic_save_json,
    atomic_save_tensors,
    read_jsonl,
    repair_truncated_jsonl,
    sha256_file,
    sha256_text,
    write_jsonl,
)
from mllm_self_steering.utils.seed import seed_everything


VISTA_VSV_CONTRACT = "vista-vsv-only-qwen3vl-v1"
VISTA_PAPER = "arXiv:2502.03628"
VISTA_OFFICIAL_REPOSITORY = "https://github.com/LzVv123456/VISTA"
_STORAGE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def validate_vsv_config(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get("vista_vsv", {})
    if not isinstance(section, Mapping):
        raise ValueError("vista_vsv must be a mapping")
    value = dict(section)
    strength = value.get("lambda", 0.1)
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or float(strength) <= 0
    ):
        raise ValueError("vista_vsv.lambda must be positive and finite")
    storage_dtype = str(value.get("storage_dtype", "float16"))
    if storage_dtype not in _STORAGE_DTYPES:
        raise ValueError(
            "vista_vsv.storage_dtype must be one of "
            + ", ".join(sorted(_STORAGE_DTYPES))
        )
    if value.get("construction", "image_minus_text") != "image_minus_text":
        raise ValueError("VSV-Only requires construction=image_minus_text")
    if value.get("injection", "official_norm_preserving_mlp") != (
        "official_norm_preserving_mlp"
    ):
        raise ValueError(
            "VSV-Only requires injection=official_norm_preserving_mlp"
        )
    error_limit = value.get("consecutive_error_limit", 3)
    if (
        not isinstance(error_limit, int)
        or isinstance(error_limit, bool)
        or error_limit < 1
    ):
        raise ValueError("vista_vsv.consecutive_error_limit must be positive")
    value.update(
        {
            "lambda": float(strength),
            "storage_dtype": storage_dtype,
            "construction": "image_minus_text",
            "injection": "official_norm_preserving_mlp",
            "consecutive_error_limit": int(error_limit),
        }
    )
    return value


def compute_vsv(
    positive_states: Sequence[torch.Tensor],
    negative_states: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Implement paper Eq. (4): image-conditioned minus text-only states."""

    if len(positive_states) != len(negative_states) or not positive_states:
        raise ValueError("positive and negative states must have equal non-zero length")
    vectors: list[torch.Tensor] = []
    for index, (positive, negative) in enumerate(
        zip(positive_states, negative_states, strict=True)
    ):
        positive = positive.detach().cpu().float()
        negative = negative.detach().cpu().float()
        if positive.ndim != 1 or negative.ndim != 1:
            raise ValueError(f"layer {index} state must be one-dimensional")
        if positive.shape != negative.shape:
            raise ValueError(f"layer {index} positive/negative shapes differ")
        vector = positive - negative
        if not bool(torch.isfinite(vector).all()):
            raise ValueError(f"layer {index} VSV contains non-finite values")
        vectors.append(vector)
    if not any(float(vector.norm()) > 0 for vector in vectors):
        raise ValueError("all VSV layer directions are zero")
    return vectors


def vista_norm_preserving_transform(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Apply the official VSV cosine-adaptive, norm-preserving transform."""

    if hidden.ndim != 3:
        raise ValueError("VSV hidden state must have [batch, sequence, hidden] shape")
    if direction.ndim != 1 or direction.numel() != hidden.shape[-1]:
        raise ValueError("VSV direction does not match hidden size")
    if not math.isfinite(float(strength)) or float(strength) <= 0:
        raise ValueError("VSV strength must be positive and finite")
    source = direction.to(device=hidden.device, dtype=torch.float32)
    source_unit = F.normalize(source, p=2, dim=-1)
    return _transform_with_unit_direction(hidden, source_unit, strength)


def _transform_with_unit_direction(
    hidden: torch.Tensor, source_unit: torch.Tensor, strength: float
) -> torch.Tensor:
    if hidden.ndim != 3 or source_unit.ndim != 1:
        raise ValueError("VSV transform expects 3D hidden states and a 1D direction")
    if source_unit.numel() != hidden.shape[-1]:
        raise ValueError("VSV direction does not match hidden size")
    original_dtype = hidden.dtype
    value = hidden.float()
    original_norm = torch.linalg.vector_norm(value, ord=2, dim=-1, keepdim=True)
    value_unit = F.normalize(value, p=2, dim=-1)
    cosine_scale = 1.0 + torch.clamp_min(
        (-value_unit * source_unit[None, None, :]).sum(dim=-1), 0.0
    ).unsqueeze(-1)
    steered = value_unit + float(strength) * cosine_scale * source_unit
    steered = F.normalize(steered, p=2, dim=-1) * original_norm
    return steered.to(dtype=original_dtype)


@contextmanager
def capture_all_layer_last_tokens(
    layers: Sequence[nn.Module],
) -> Iterator[list[torch.Tensor | None]]:
    """Capture the final context position from every language decoder block."""

    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles: list[Any] = []

    def make_hook(index: int):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = hidden_from_output(output)
            captured[index] = hidden[0, -1].detach().cpu().float()
            return output

        return hook

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(index)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def add_vsv_hooks(
    layers: Sequence[nn.Module],
    vectors: Sequence[torch.Tensor],
    strength: float,
) -> Iterator[None]:
    """Attach official-style VSV transforms after every decoder MLP."""

    if len(layers) != len(vectors):
        raise ValueError("VSV must provide exactly one direction per language layer")
    if not math.isfinite(float(strength)) or float(strength) <= 0:
        raise ValueError("VSV strength must be positive and finite")
    handles: list[Any] = []

    def make_hook(source: torch.Tensor):
        cache: dict[torch.device, torch.Tensor] = {}

        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0]
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError("MLP output tuple does not start with a tensor")
                source_unit = cache.get(hidden.device)
                if source_unit is None:
                    source_unit = F.normalize(
                        source.to(device=hidden.device, dtype=torch.float32),
                        p=2,
                        dim=-1,
                    )
                    cache[hidden.device] = source_unit
                changed = _transform_with_unit_direction(
                    hidden, source_unit, strength
                )
                return (changed, *output[1:])
            if not isinstance(output, torch.Tensor):
                raise TypeError("MLP output must be a tensor or tensor-first tuple")
            source_unit = cache.get(output.device)
            if source_unit is None:
                source_unit = F.normalize(
                    source.to(device=output.device, dtype=torch.float32),
                    p=2,
                    dim=-1,
                )
                cache[output.device] = source_unit
            return _transform_with_unit_direction(output, source_unit, strength)

        return hook

    try:
        for index, (layer, vector) in enumerate(zip(layers, vectors, strict=True)):
            if vector.ndim != 1:
                raise ValueError(f"VSV direction for layer {index} must be one-dimensional")
            mlp = getattr(layer, "mlp", None)
            if not isinstance(mlp, nn.Module):
                raise ValueError(f"language layer {index} has no .mlp module")
            handles.append(mlp.register_forward_hook(make_hook(vector.detach())))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _move_inputs(inputs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _capture_states(
    model: Any,
    layers: Sequence[nn.Module],
    inputs: Mapping[str, Any],
) -> list[torch.Tensor]:
    with torch.inference_mode(), capture_all_layer_last_tokens(layers) as captured:
        model(**dict(inputs), use_cache=False)
    missing = [index for index, value in enumerate(captured) if value is None]
    if missing:
        raise RuntimeError(f"VSV capture missed language layers: {missing}")
    return [value for value in captured if value is not None]


def _generate(
    model: Any,
    processor: Any,
    inputs: Mapping[str, Any],
    *,
    layers: Sequence[nn.Module],
    vectors: Sequence[torch.Tensor] | None,
    strength: float,
    max_new_tokens: int,
) -> str:
    context = (
        add_vsv_hooks(layers, vectors, strength)
        if vectors is not None
        else nullcontext()
    )
    with torch.inference_mode(), context:
        generated = model.generate(
            **dict(inputs),
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
        )
    prompt_length = int(inputs["input_ids"].shape[1])
    new_tokens = generated[:, prompt_length:]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _answer_fields(output: str, item: MultimodalItem) -> dict[str, Any]:
    prediction = extract_answer(output, item.answer_type)
    return {
        "raw_output": output,
        "predicted_answer": prediction,
        "correct": is_correct(
            prediction, item.gold_answer, answer_type=item.answer_type
        ),
    }


def _canonical_rows(
    path: Path, run_id: str, *, successful_only: bool
) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    repair_truncated_jsonl(path)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("run_id") != run_id:
            continue
        if successful_only and row.get("status") != "ok":
            continue
        rows[(str(row["dataset"]), str(row["item_identity"]))] = row
    return rows


def _item_seed(base_seed: int, identity: str) -> int:
    return (int(base_seed) + int(identity[:8], 16)) % (2**31 - 1)


def _vector_summary(vectors: Sequence[torch.Tensor]) -> dict[str, Any]:
    norms = [float(vector.norm()) for vector in vectors]
    return {
        "layer_count": len(vectors),
        "hidden_size": int(vectors[0].numel()),
        "norm_min": min(norms),
        "norm_mean": fmean(norms),
        "norm_max": max(norms),
        "zero_norm_layers": sum(value == 0.0 for value in norms),
    }


def _save_vsv(
    path: Path, vectors: Sequence[torch.Tensor], *, storage_dtype: str
) -> None:
    dtype = _STORAGE_DTYPES[storage_dtype]
    atomic_save_tensors(
        path,
        {
            f"layer_{index:03d}": vector.detach().cpu().to(dtype=dtype)
            for index, vector in enumerate(vectors)
        },
    )


def vista_experiment_fingerprint(
    config: Mapping[str, Any], project_root: Path, vsv: Mapping[str, Any]
) -> str:
    data_dir, _ = _paths(dict(config), project_root)
    datasets = list(config["data"]["enabled_datasets"])
    identity = {
        "contract": VISTA_VSV_CONTRACT,
        "paper": VISTA_PAPER,
        "model": {
            key: config["model"].get(key)
            for key in (
                "name",
                "revision",
                "dtype",
                "attention_implementation",
                "max_new_tokens",
            )
        },
        "vsv": dict(vsv),
        "decoding": {"do_sample": False, "temperature": 0.0},
        "seed": config["experiment"].get("seed", 42),
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


def _run_payload(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    fingerprint: str,
    vsv: Mapping[str, Any],
    status: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "vista_vsv_only",
        "name": run_dir.name,
        "created_at": created_at or datetime.now().astimezone().isoformat(),
        "status": status,
        "contract": VISTA_VSV_CONTRACT,
        "paper": VISTA_PAPER,
        "official_repository": VISTA_OFFICIAL_REPOSITORY,
        "variant": "VSV-Only",
        "adaptation": "Qwen3-VL",
        "experiment_fingerprint": fingerprint,
        "vsv": dict(vsv),
        "config": deepcopy(dict(config)),
        "artifacts": {
            "vectors": "vectors/",
            "vector_index": "vector_index.jsonl",
            "generations": "generations.jsonl",
            "metrics_json": "metrics.json",
            "metrics_csv": "metrics.csv",
        },
    }


def _validate_resume(
    existing: Mapping[str, Any], expected: Mapping[str, Any], run_dir: Path
) -> None:
    for key in ("kind", "contract", "experiment_fingerprint"):
        if existing.get(key) != expected.get(key):
            raise ValueError(
                f"cannot resume {run_dir}: config mismatch for {key}; "
                "omit --run-dir to start a new VISTA run"
            )


def _group_metrics(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = [bool(row["baseline_correct"]) for row in selected]
    vista = [bool(row["correct"]) for row in selected]
    count = len(selected)
    wrong_to_right = sum(
        not before and after for before, after in zip(baseline, vista, strict=True)
    )
    right_to_wrong = sum(
        before and not after for before, after in zip(baseline, vista, strict=True)
    )
    return {
        "count": count,
        "baseline_accuracy": sum(baseline) / count if count else None,
        "vista_vsv_accuracy": sum(vista) / count if count else None,
        "accuracy_delta": (sum(vista) - sum(baseline)) / count if count else None,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "net_flips": wrong_to_right - right_to_wrong,
    }


def _write_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_items: int,
    vsv: Mapping[str, Any],
    json_path: Path,
    csv_path: Path,
) -> None:
    by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    groups = {"overall": _group_metrics(rows)}
    groups.update(
        {name: _group_metrics(values) for name, values in sorted(by_dataset.items())}
    )
    report = {
        "implementation": VISTA_VSV_CONTRACT,
        "variant": "VSV-Only",
        "expected_items": expected_items,
        "completed_items": len(rows),
        "complete": len(rows) == expected_items,
        "vsv": dict(vsv),
        "groups": groups,
    }
    atomic_save_json(json_path, report)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fieldnames = [
            "dataset",
            "count",
            "baseline_accuracy",
            "vista_vsv_accuracy",
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


def _write_vector_index(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    write_jsonl(
        path,
        (
            {
                "dataset": row["dataset"],
                "item_id": row["item_id"],
                "item_identity": row["item_identity"],
                "vector_path": row["vector_path"],
                "vector_sha256": row["vector_sha256"],
                "vector_summary": row["vector_summary"],
            }
            for row in rows
        ),
    )


def run_vista_vsv_baseline(
    config: dict[str, Any], project_root: Path, *, run_dir: Path
) -> dict[str, Path]:
    """Run or resume paired baseline and VISTA VSV-Only generation."""

    if config["experiment"].get("generation", {}).get("backend") != "transformers":
        raise ValueError("VISTA VSV-Only requires generation.backend=transformers")
    vsv = validate_vsv_config(config)
    data_dir, _ = _paths(config, project_root)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = vista_experiment_fingerprint(config, project_root, vsv)
    expected_metadata = _run_payload(
        config,
        run_dir,
        fingerprint=fingerprint,
        vsv=vsv,
        status="running",
    )
    metadata_path = run_dir / "config.json"
    if metadata_path.is_file():
        existing = read_run_config(run_dir, expected_kind="vista_vsv_only")
        _validate_resume(existing, expected_metadata, run_dir)
        metadata = dict(existing)
        metadata["status"] = "running"
        metadata["config"] = expected_metadata["config"]
    else:
        if any(run_dir.iterdir()):
            raise ValueError(f"run directory is non-empty but has no config.json: {run_dir}")
        metadata = expected_metadata
    write_run_config(run_dir, metadata)
    run_id = str(metadata["name"])

    generation_path = run_dir / "generations.jsonl"
    vector_index_path = run_dir / "vector_index.jsonl"
    metrics_json = run_dir / "metrics.json"
    metrics_csv = run_dir / "metrics.csv"
    successful = _canonical_rows(generation_path, run_id, successful_only=True)
    latest = _canonical_rows(generation_path, run_id, successful_only=False)

    items: list[MultimodalItem] = []
    for dataset in config["data"]["enabled_datasets"]:
        rows = list(read_jsonl(data_dir / "processed" / f"{dataset}.jsonl"))
        items.extend(
            MultimodalItem.from_dict(row)
            for row in _limit(config, rows, scope="eval")
        )
    records: list[tuple[MultimodalItem, list[str], str]] = []
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
        desc=f"{run_id}:vista-vsv",
    )
    checkpoint_rows = int(
        config["experiment"].get("generation", {}).get("checkpoint_rows", 100)
    )
    base_seed = int(config["experiment"].get("seed", 42))
    max_new_tokens = int(config["model"].get("max_new_tokens", 2048))

    try:
        if missing:
            seed_everything(base_seed)
            model, processor = load_model_and_processor(config["model"])
            layers = list(resolve_language_layers(model))
            input_device = model_input_device(model)
            error_streak = 0
            with JsonlCheckpointWriter(
                generation_path, sync_every=checkpoint_rows
            ) as writer:
                for item, _, identity in missing:
                    key = (item.dataset, identity)
                    base = {
                        "run_id": run_id,
                        "implementation": VISTA_VSV_CONTRACT,
                        "variant": "VSV-Only",
                        "adaptation": "Qwen3-VL",
                        "dataset": item.dataset,
                        "item_id": item.item_id,
                        "item_identity": identity,
                        "gold_answer": item.gold_answer,
                        "answer_type": item.answer_type,
                        "backend": "transformers",
                        "vsv_lambda": vsv["lambda"],
                        "vsv_construction": vsv["construction"],
                        "vsv_injection": vsv["injection"],
                        "steering_applied": True,
                    }
                    baseline_fields: dict[str, Any] = {}
                    vector_fields: dict[str, Any] = {}
                    try:
                        positive_inputs = _move_inputs(
                            prepare_transformers_inputs(
                                processor,
                                item.prompt,
                                item.image_paths,
                                answer_type=item.answer_type,
                            ),
                            input_device,
                        )
                        baseline_output: str | None = None
                        partial = latest.get(key)
                        if partial is not None:
                            cached = partial.get("baseline_raw_output")
                            if isinstance(cached, str) and cached:
                                baseline_output = cached
                        if baseline_output is None:
                            seed_everything(_item_seed(base_seed, identity))
                            baseline_output = _generate(
                                model,
                                processor,
                                positive_inputs,
                                layers=layers,
                                vectors=None,
                                strength=float(vsv["lambda"]),
                                max_new_tokens=max_new_tokens,
                            )
                        parsed_baseline = _answer_fields(baseline_output, item)
                        baseline_fields = {
                            "baseline_raw_output": parsed_baseline["raw_output"],
                            "baseline_predicted_answer": parsed_baseline[
                                "predicted_answer"
                            ],
                            "baseline_correct": parsed_baseline["correct"],
                        }

                        negative_inputs = _move_inputs(
                            prepare_transformers_inputs(
                                processor,
                                item.prompt,
                                [],
                                answer_type=item.answer_type,
                            ),
                            input_device,
                        )
                        positive_states = _capture_states(
                            model, layers, positive_inputs
                        )
                        negative_states = _capture_states(
                            model, layers, negative_inputs
                        )
                        vectors = compute_vsv(positive_states, negative_states)
                        local_vector_path = (
                            run_dir / "vectors" / item.dataset / f"{identity}.safetensors"
                        )
                        _save_vsv(
                            local_vector_path,
                            vectors,
                            storage_dtype=str(vsv["storage_dtype"]),
                        )
                        vector_fields = {
                            "vector_path": str(
                                local_vector_path.relative_to(run_dir)
                            ).replace("\\", "/"),
                            "vector_sha256": sha256_file(local_vector_path),
                            "vector_storage_dtype": vsv["storage_dtype"],
                            "vector_summary": _vector_summary(vectors),
                        }

                        seed_everything(_item_seed(base_seed, identity))
                        output = _generate(
                            model,
                            processor,
                            positive_inputs,
                            layers=layers,
                            vectors=vectors,
                            strength=float(vsv["lambda"]),
                            max_new_tokens=max_new_tokens,
                        )
                        row = {
                            **base,
                            **baseline_fields,
                            **vector_fields,
                            **_answer_fields(output, item),
                            "status": "ok",
                        }
                    except Exception as error:
                        row = {
                            **base,
                            **baseline_fields,
                            **vector_fields,
                            "status": "error",
                            "error_type": type(error).__name__,
                            "error": repr(error),
                        }
                    writer.write(row)
                    progress.update(1)
                    if row["status"] == "ok":
                        error_streak = 0
                        successful[key] = row
                    else:
                        error_streak += 1
                    if error_streak >= int(vsv["consecutive_error_limit"]):
                        writer.checkpoint()
                        raise RuntimeError(
                            "aborting VISTA after "
                            f"{error_streak} consecutive failures; "
                            f"last error={row.get('error')}"
                        )
    except BaseException:
        successful = _canonical_rows(
            generation_path, run_id, successful_only=True
        )
        rows = list(successful.values())
        _write_vector_index(vector_index_path, rows)
        _write_metrics(
            rows,
            expected_items=len(records),
            vsv=vsv,
            json_path=metrics_json,
            csv_path=metrics_csv,
        )
        metadata["status"] = "incomplete"
        write_run_config(run_dir, metadata)
        raise
    finally:
        progress.close()

    successful = _canonical_rows(generation_path, run_id, successful_only=True)
    rows = list(successful.values())
    _write_vector_index(vector_index_path, rows)
    _write_metrics(
        rows,
        expected_items=len(records),
        vsv=vsv,
        json_path=metrics_json,
        csv_path=metrics_csv,
    )
    artifacts = {
        "vector_index": vector_index_path,
        "generations": generation_path,
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
    }
    _write_stage_manifest(config, run_dir, "run_vista_vsv_baseline", artifacts)
    complete = len(rows) == len(records)
    metadata["status"] = "complete" if complete else "incomplete"
    artifacts["config"] = write_run_config(run_dir, metadata)
    if not complete:
        raise RuntimeError(
            f"VISTA VSV-Only run is incomplete: {len(rows)}/{len(records)} items; "
            "resume with --run-dir"
        )
    return artifacts

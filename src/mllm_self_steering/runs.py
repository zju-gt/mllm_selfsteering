"""Human-readable preparation and generation run directories."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from mllm_self_steering.datasets.pairs import resolve_project_path
from mllm_self_steering.utils.io import atomic_save_json, sha256_file, sha256_text


RUN_CONFIG_NAME = "config.json"
RUN_SCHEMA_VERSION = 1


def resolve_run_path(project_root: Path, value: Path | str) -> Path:
    """Resolve a CLI run path relative to the project root."""

    return resolve_project_path(project_root, str(value)).resolve()


def create_timestamped_run_dir(
    outputs_root: Path,
    kind: str,
    *,
    now: datetime | None = None,
) -> Path:
    """Create ``<kind>_MMDD_HHMM``, adding a numeric suffix on collision."""

    outputs_root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now().astimezone()).strftime("%m%d_%H%M")
    stem = f"{kind}_{timestamp}"
    for index in range(1000):
        name = stem if index == 0 else f"{stem}_{index + 1:02d}"
        destination = outputs_root / name
        try:
            destination.mkdir()
        except FileExistsError:
            continue
        return destination.resolve()
    raise RuntimeError(f"too many {stem} run directories")


def run_config_path(run_dir: Path) -> Path:
    return run_dir / RUN_CONFIG_NAME


def read_run_config(run_dir: Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    path = run_config_path(run_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"run configuration not found: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"run configuration must be an object: {path}")
    if payload.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError(f"unsupported run configuration schema: {path}")
    if expected_kind is not None and payload.get("kind") != expected_kind:
        raise ValueError(
            f"expected a {expected_kind} directory, got {payload.get('kind')!r}: {run_dir}"
        )
    return payload


def write_run_config(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    value = dict(payload)
    value["schema_version"] = RUN_SCHEMA_VERSION
    path = run_config_path(run_dir)
    atomic_save_json(path, value)
    return path


def preparation_payload(
    config: Mapping[str, Any],
    preparation_dir: Path,
    *,
    capture_artifact_id: str,
    vector_root: Path,
    status: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    library = vector_root / "capability_vectors.safetensors"
    metadata = vector_root / "capability_vectors.json"
    payload: dict[str, Any] = {
        "kind": "preparation",
        "name": preparation_dir.name,
        "created_at": created_at or datetime.now().astimezone().isoformat(),
        "status": status,
        "capture_artifact_id": capture_artifact_id,
        "config": deepcopy(dict(config)),
        "artifacts": {
            "vector_root": str(vector_root.relative_to(preparation_dir)).replace("\\", "/"),
            "library": str(library.relative_to(preparation_dir)).replace("\\", "/"),
            "metadata": str(metadata.relative_to(preparation_dir)).replace("\\", "/"),
        },
    }
    if library.is_file():
        payload["vector_library_sha256"] = sha256_file(library)
    payload["preparation_fingerprint"] = preparation_fingerprint(payload)
    return payload


def preparation_fingerprint(payload: Mapping[str, Any]) -> str:
    identity = {
        "capture_artifact_id": payload.get("capture_artifact_id"),
        "vector_library_sha256": payload.get("vector_library_sha256"),
        "artifacts": payload.get("artifacts"),
    }
    return sha256_text(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def preparation_vector_root(
    preparation_dir: Path, payload: Mapping[str, Any]
) -> Path:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("vector_root"), str):
        raise ValueError(f"preparation config has no vector_root: {preparation_dir}")
    root = (preparation_dir / artifacts["vector_root"]).resolve()
    try:
        root.relative_to(preparation_dir.resolve())
    except ValueError as exc:
        raise ValueError("preparation vector_root escapes its run directory") from exc
    if not root.is_dir():
        raise FileNotFoundError(f"preparation vector directory not found: {root}")
    library = root / "capability_vectors.safetensors"
    expected = payload.get("vector_library_sha256")
    if expected is not None and sha256_file(library) != expected:
        raise ValueError(f"preparation vector library was modified after creation: {library}")
    return root


def generation_payload(
    config: Mapping[str, Any],
    generation_dir: Path,
    preparation_dir: Path,
    preparation: Mapping[str, Any],
    *,
    experiment_fingerprint: str,
    status: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "kind": "generation",
        "name": generation_dir.name,
        "created_at": created_at or datetime.now().astimezone().isoformat(),
        "status": status,
        "preparation_dir": str(preparation_dir.resolve()),
        "preparation_fingerprint": preparation.get("preparation_fingerprint"),
        "experiment_fingerprint": experiment_fingerprint,
        "config": deepcopy(dict(config)),
        "artifacts": {
            "baselines": "baselines.jsonl",
            "selections": "selections.jsonl",
            "generations": "generations.jsonl",
            "metrics_json": "metrics.json",
            "metrics_csv": "metrics.csv",
        },
    }


def validate_generation_resume(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
    generation_dir: Path,
) -> None:
    for key in (
        "kind",
        "preparation_dir",
        "preparation_fingerprint",
        "experiment_fingerprint",
    ):
        if existing.get(key) != expected.get(key):
            raise ValueError(
                f"cannot resume {generation_dir}: config mismatch for {key}; "
                "omit --generation-dir to start a new run"
            )

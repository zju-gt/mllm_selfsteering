"""Configuration loading and cross-stage validation."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml


DATASETS = {"mmstar", "mmmu", "mathvista", "mathvision"}
MODALITIES = {"text", "visual"}
VECTOR_SCALINGS = {"raw", "unit", "mean_norm"}
BACKENDS = {"transformers", "vllm_hook"}


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override must use key=value syntax: {override!r}")
    dotted, raw = override.split("=", 1)
    parts = [part.strip() for part in dotted.split(".") if part.strip()]
    if not parts:
        raise ConfigError("override key cannot be empty")
    current = config
    for part in parts[:-1]:
        value = current.setdefault(part, {})
        if not isinstance(value, dict):
            raise ConfigError(f"cannot set a child below non-mapping key {part!r}")
        current = value
    current[parts[-1]] = yaml.safe_load(raw)


def load_config(
    paths: Iterable[Path | str], overrides: list[str] | None = None
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"configuration file not found: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ConfigError(f"configuration root must be a mapping: {path}")
        config = _deep_merge(config, value)
    for override in overrides or []:
        _apply_override(config, override)
    validate_config(config)
    return config


def capability_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in config["experiment"]["capabilities"]}


def validate_config(config: dict[str, Any]) -> None:
    for section in ("model", "data", "experiment"):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"missing {section} configuration")

    model = config["model"]
    if not isinstance(model.get("name"), str) or not model["name"].strip():
        raise ConfigError("model.name must be non-empty")
    target_layer = config["experiment"].get("target_layer")
    if not isinstance(target_layer, int) or target_layer < 0:
        raise ConfigError("experiment.target_layer must be a non-negative integer")
    max_tokens = model.get("max_new_tokens", 2048)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ConfigError("model.max_new_tokens must be a positive integer")

    enabled = config["data"].get("enabled_datasets")
    if not isinstance(enabled, list) or not enabled:
        raise ConfigError("data.enabled_datasets must be a non-empty list")
    unknown = set(enabled) - DATASETS
    if unknown:
        raise ConfigError(f"unknown datasets: {', '.join(sorted(unknown))}")
    if len(enabled) != len(set(enabled)):
        raise ConfigError("data.enabled_datasets must not contain duplicates")
    sources = config["data"].get("sources")
    if not isinstance(sources, dict):
        raise ConfigError("data.sources must be a mapping")
    missing_sources = set(enabled) - set(sources)
    if missing_sources:
        raise ConfigError(
            f"missing dataset sources: {', '.join(sorted(missing_sources))}"
        )

    capabilities = config["experiment"].get("capabilities")
    if not isinstance(capabilities, list) or len(capabilities) != 6:
        raise ConfigError("experiment.capabilities must contain exactly six entries")
    ids: list[str] = []
    modality_counts = {"text": 0, "visual": 0}
    for row in capabilities:
        if not isinstance(row, dict):
            raise ConfigError("each capability must be a mapping")
        capability = row.get("id")
        modality = row.get("modality")
        if not isinstance(capability, str) or not capability.strip():
            raise ConfigError("capability id must be non-empty")
        if modality not in MODALITIES:
            raise ConfigError(f"invalid modality for {capability}: {modality!r}")
        if not isinstance(row.get("pair_path"), str) or not row["pair_path"].strip():
            raise ConfigError(f"capability {capability} requires pair_path")
        ids.append(capability)
        modality_counts[modality] += 1
    if len(ids) != len(set(ids)):
        raise ConfigError("capability ids must be unique")
    if modality_counts != {"text": 3, "visual": 3}:
        raise ConfigError("capabilities must contain three text and three visual entries")

    scaling = config["experiment"].get("vector_scaling")
    if scaling not in VECTOR_SCALINGS:
        raise ConfigError(f"vector_scaling must be one of {sorted(VECTOR_SCALINGS)}")
    alphas = config["experiment"].get("alphas")
    if not isinstance(alphas, list) or not alphas:
        raise ConfigError("experiment.alphas must be a non-empty list")
    normalized = []
    for alpha in alphas:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ConfigError("every alpha must be numeric")
        normalized.append(float(alpha))
    if not all(math.isfinite(alpha) for alpha in normalized):
        raise ConfigError("alphas must be finite")
    if len(normalized) != len(set(normalized)):
        raise ConfigError("alphas must be unique")
    if 0.0 not in normalized:
        raise ConfigError("experiment.alphas must include the unsteered 0.0 baseline")

    generation = config["experiment"].get("generation", {})
    if generation.get("backend", "transformers") not in BACKENDS:
        raise ConfigError(f"generation.backend must be one of {sorted(BACKENDS)}")
    checkpoint_rows = generation.get("checkpoint_rows", 100)
    if (
        not isinstance(checkpoint_rows, int)
        or isinstance(checkpoint_rows, bool)
        or checkpoint_rows < 1
    ):
        raise ConfigError("generation.checkpoint_rows must be a positive integer")
    vllm = config["experiment"].get("vllm_hook", {})
    for key, default in (
        ("request_concurrency", 16),
        ("max_retries", 1),
        ("consecutive_error_limit", 5),
    ):
        value = vllm.get(key, default)
        lower_bound = 0 if key == "max_retries" else 1
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < lower_bound
        ):
            qualifier = "non-negative" if lower_bound == 0 else "positive"
            raise ConfigError(f"vllm_hook.{key} must be a {qualifier} integer")
    timeout = vllm.get("request_timeout_seconds", 600)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        raise ConfigError("vllm_hook.request_timeout_seconds must be positive and finite")
    if vllm.get("media_transport", "file") not in {"file", "data_url"}:
        raise ConfigError("vllm_hook.media_transport must be file or data_url")
    for key in ("server_media_root", "server_vector_root"):
        value = vllm.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f"vllm_hook.{key} must be null or a non-empty string")
        if isinstance(value, str) and not PurePosixPath(value).is_absolute():
            raise ConfigError(f"vllm_hook.{key} must be an absolute Linux path")
    if vllm.get("phase", "both") not in {"both", "prefill", "decode"}:
        raise ConfigError("vllm_hook.phase must be both, prefill, or decode")
    if vllm.get("positions", "last_token") not in {"all_tokens", "last_token"}:
        raise ConfigError("vllm_hook.positions must be all_tokens or last_token")
    verify_effect = vllm.get("verify_steering_effect", True)
    if not isinstance(verify_effect, bool):
        raise ConfigError("vllm_hook.verify_steering_effect must be boolean")
    top_logprobs = vllm.get("steering_probe_top_logprobs", 20)
    if (
        not isinstance(top_logprobs, int)
        or isinstance(top_logprobs, bool)
        or not 1 <= top_logprobs <= 20
    ):
        raise ConfigError(
            "vllm_hook.steering_probe_top_logprobs must be an integer from 1 to 20"
        )
    tolerance = vllm.get("steering_probe_tolerance", 1e-6)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0
    ):
        raise ConfigError(
            "vllm_hook.steering_probe_tolerance must be non-negative and finite"
        )
    probe_alpha = vllm.get("steering_probe_alpha")
    if probe_alpha is not None and (
        isinstance(probe_alpha, bool)
        or not isinstance(probe_alpha, (int, float))
        or not math.isfinite(float(probe_alpha))
        or float(probe_alpha) == 0.0
    ):
        raise ConfigError("vllm_hook.steering_probe_alpha must be null or finite and nonzero")
    provenance = vllm.get("server_provenance", {})
    if not isinstance(provenance, dict):
        raise ConfigError("vllm_hook.server_provenance must be a mapping")
    for key in ("vllm_version", "mia_revision", "served_model_revision"):
        value = provenance.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(
                f"vllm_hook.server_provenance.{key} must be null or a non-empty string"
            )

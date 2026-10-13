"""Shared CLI configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mllm_self_steering.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    PROJECT_ROOT / "configs" / "model.yaml",
    PROJECT_ROOT / "configs" / "data.yaml",
    PROJECT_ROOT / "configs" / "experiment.yaml",
)


def parser(description: str) -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=description)
    value.add_argument("--config", action="append", type=Path)
    value.add_argument("--override", action="append", default=[])
    value.add_argument(
        "--limit",
        type=int,
        help="legacy shorthand that applies the same limit to every stage scope",
    )
    value.add_argument("--data-limit", type=int, help="limit records during stage 00")
    value.add_argument(
        "--pair-limit",
        type=int,
        help="limit contrast pairs during stages 02-04",
    )
    value.add_argument(
        "--eval-limit",
        type=int,
        help="limit evaluation items per enabled dataset during stages 05-06",
    )
    return value


def resolved(args: argparse.Namespace) -> dict[str, Any]:
    paths = args.config if args.config is not None else DEFAULT_CONFIGS
    config = load_config(paths, args.override)
    scoped = {
        "data_limit": args.data_limit,
        "pair_limit": args.pair_limit,
        "eval_limit": args.eval_limit,
    }
    if args.limit is not None and any(value is not None for value in scoped.values()):
        raise ValueError("--limit cannot be combined with scoped limit options")
    runtime = dict(config.get("_runtime", {}))
    values = {"limit": args.limit, **scoped}
    for key, value in values.items():
        if value is None:
            continue
        if value < 1:
            option = "--" + key.replace("_", "-")
            raise ValueError(f"{option} must be positive")
        runtime[key] = value
    if runtime:
        config["_runtime"] = runtime
    return config


def print_result(value: Any) -> None:
    if isinstance(value, dict):
        value = {key: str(item) for key, item in value.items()}
    elif isinstance(value, list):
        value = [str(item) for item in value]
    else:
        value = str(value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


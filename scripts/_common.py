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
    value.add_argument("--limit", type=int)
    return value


def resolved(args: argparse.Namespace) -> dict[str, Any]:
    paths = args.config if args.config is not None else DEFAULT_CONFIGS
    config = load_config(paths, args.override)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        config["_runtime"] = {"limit": args.limit}
    return config


def print_result(value: Any) -> None:
    if isinstance(value, dict):
        value = {key: str(item) for key, item in value.items()}
    elif isinstance(value, list):
        value = [str(item) for item in value]
    else:
        value = str(value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


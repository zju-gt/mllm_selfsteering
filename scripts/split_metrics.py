"""Compute fine-grained baseline and steering metrics for a generation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute OE/MC, Perception/Reasoning, and STEM/HASS metrics"
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="generation directory containing config.json and generations.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output JSON path; default: <generation_dir>/split_metrics.json",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="score only available successful rows instead of requiring complete coverage",
    )
    args = parser.parse_args()

    from mllm_self_steering.split_metrics import calculate_split_metrics, write_metrics

    generation_dir = args.generation_dir.resolve()
    output = args.output or generation_dir / "split_metrics.json"
    metrics = calculate_split_metrics(
        generation_dir,
        project_root=PROJECT_ROOT,
        allow_incomplete=args.allow_incomplete,
    )
    destination = write_metrics(output, metrics)
    print(json.dumps({"output": str(destination.resolve()), **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

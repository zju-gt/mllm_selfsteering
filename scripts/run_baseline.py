from pathlib import Path

from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    cli = parser("Run or resume unsteered multimodal baseline generation")
    cli.add_argument(
        "--generation-dir",
        "--baseline-dir",
        dest="baseline_dir",
        type=Path,
        help=(
            "existing baseline directory to resume; omit to create "
            "outputs/baseline_MMDD_HHMM"
        ),
    )
    args = cli.parse_args()
    config = resolved(args)
    config["experiment"].setdefault("generation", {})["backend"] = "vllm_hook"

    from mllm_self_steering.baseline import run_baseline
    from mllm_self_steering.pipeline import _paths
    from mllm_self_steering.runs import create_timestamped_run_dir, resolve_run_path

    if args.baseline_dir is None:
        _, outputs = _paths(config, PROJECT_ROOT)
        baseline_dir = create_timestamped_run_dir(outputs, "baseline")
    else:
        baseline_dir = resolve_run_path(PROJECT_ROOT, args.baseline_dir)
    print_result(run_baseline(config, PROJECT_ROOT, baseline_dir=baseline_dir))


if __name__ == "__main__":
    main()

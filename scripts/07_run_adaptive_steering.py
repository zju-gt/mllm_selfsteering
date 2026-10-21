from pathlib import Path

from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    cli = parser("Run two-round adaptive capability selection and vLLM-Hook steering")
    cli.add_argument(
        "--baseline-no-answer-only",
        action="store_true",
        help="select and steer only when the baseline has no parseable answer",
    )
    cli.add_argument(
        "--preparation-dir",
        type=Path,
        help="frozen preparation directory containing config.json and the vector library",
    )
    cli.add_argument(
        "--generation-dir",
        type=Path,
        help="existing directory to resume; omit to create outputs/generation_MMDD_HHMM",
    )
    cli.add_argument(
        "--legacy-layout",
        action="store_true",
        help="use the pre-run-directory outputs layout for an existing legacy experiment",
    )
    args = cli.parse_args()
    config = resolved(args)
    if args.baseline_no_answer_only:
        config["experiment"].setdefault("adaptive_steering", {})[
            "baseline_no_answer_only"
        ] = True
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.pipeline import _paths
    from mllm_self_steering.runs import create_timestamped_run_dir, resolve_run_path

    if args.legacy_layout:
        if args.preparation_dir is not None or args.generation_dir is not None:
            raise ValueError(
                "--legacy-layout cannot be combined with run-directory options"
            )
        result = run_adaptive_steering(config, PROJECT_ROOT)
    else:
        if args.preparation_dir is None:
            raise ValueError("--preparation-dir is required unless --legacy-layout is used")
        preparation_dir = resolve_run_path(PROJECT_ROOT, args.preparation_dir)
        if args.generation_dir is None:
            _, outputs = _paths(config, PROJECT_ROOT)
            generation_dir = create_timestamped_run_dir(outputs, "generation")
        else:
            generation_dir = resolve_run_path(PROJECT_ROOT, args.generation_dir)
        result = run_adaptive_steering(
            config,
            PROJECT_ROOT,
            preparation_dir=preparation_dir,
            generation_dir=generation_dir,
        )
    print_result(result)


if __name__ == "__main__":
    main()

from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    cli = parser("Run two-round adaptive capability selection and vLLM-Hook steering")
    cli.add_argument(
        "--baseline-no-answer-only",
        action="store_true",
        help="select and steer only when the baseline has no parseable answer",
    )
    args = cli.parse_args()
    config = resolved(args)
    if args.baseline_no_answer_only:
        config["experiment"].setdefault("adaptive_steering", {})[
            "baseline_no_answer_only"
        ] = True
    from mllm_self_steering.adaptive import run_adaptive_steering

    result = run_adaptive_steering(config, PROJECT_ROOT)
    print_result(result)


if __name__ == "__main__":
    main()

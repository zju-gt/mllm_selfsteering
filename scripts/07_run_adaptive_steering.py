from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    args = parser(
        "Run two-round adaptive capability selection and vLLM-Hook steering"
    ).parse_args()
    config = resolved(args)
    from mllm_self_steering.adaptive import run_adaptive_steering

    result = run_adaptive_steering(config, PROJECT_ROOT)
    print_result(result)


if __name__ == "__main__":
    main()

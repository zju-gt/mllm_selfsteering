from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    args = parser("Validate the six user-curated contrast-pair files").parse_args()
    from mllm_self_steering.pipeline import validate_pairs

    print_result(validate_pairs(resolved(args), PROJECT_ROOT))


if __name__ == "__main__":
    main()


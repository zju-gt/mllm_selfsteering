from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    args = parser("Prepare MMStar, MMMU, MathVista, and MathVision").parse_args()
    from mllm_self_steering.pipeline import prepare_data

    print_result(prepare_data(resolved(args), PROJECT_ROOT))


if __name__ == "__main__":
    main()


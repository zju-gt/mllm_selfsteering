from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    args = parser("Aggregate six multimodal capability vectors").parse_args()
    from mllm_self_steering.pipeline import extract_vectors

    print_result(extract_vectors(resolved(args), PROJECT_ROOT))


if __name__ == "__main__":
    main()


from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    args = parser("Analyze vector coherence and separation").parse_args()
    from mllm_self_steering.pipeline import analyze_vectors

    print_result(analyze_vectors(resolved(args), PROJECT_ROOT))


if __name__ == "__main__":
    main()


from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    args = parser("Score only a complete multimodal steering population").parse_args()
    from mllm_self_steering.pipeline import score_generations

    print_result(score_generations(resolved(args), PROJECT_ROOT))


if __name__ == "__main__":
    main()


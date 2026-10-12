from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    args = parser("Capture text and visual contrast activations").parse_args()
    config = resolved(args)
    from mllm_self_steering.models.loader import load_model_and_processor
    from mllm_self_steering.pipeline import capture_contrasts

    model, processor = load_model_and_processor(config["model"])
    print_result(capture_contrasts(config, PROJECT_ROOT, model, processor))


if __name__ == "__main__":
    main()


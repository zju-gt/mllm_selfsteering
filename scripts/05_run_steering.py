from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    args = parser("Run Transformers or vLLM-Hook multimodal steering").parse_args()
    config = resolved(args)
    from mllm_self_steering.pipeline import run_steering

    backend = config["experiment"].get("generation", {}).get("backend", "transformers")
    if backend == "transformers":
        from mllm_self_steering.models.loader import load_model_and_processor

        model, processor = load_model_and_processor(config["model"])
        result = run_steering(
            config, PROJECT_ROOT, model=model, processor=processor
        )
    else:
        result = run_steering(config, PROJECT_ROOT)
    print_result(result)


if __name__ == "__main__":
    main()


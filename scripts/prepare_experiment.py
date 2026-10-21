from __future__ import annotations

from pathlib import Path

from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    cli = parser("Create a frozen steering-vector preparation directory")
    cli.add_argument(
        "--preparation-dir",
        type=Path,
        help="create or continue this directory; default: outputs/preparation_MMDD_HHMM",
    )
    args = cli.parse_args()
    config = resolved(args)

    from mllm_self_steering.models.loader import load_model_and_processor
    from mllm_self_steering.pipeline import (
        _paths,
        analyze_vectors,
        capture_artifact_id,
        capture_contrasts,
        extract_vectors,
        validate_pairs,
    )
    from mllm_self_steering.runs import (
        create_timestamped_run_dir,
        preparation_payload,
        read_run_config,
        resolve_run_path,
        write_run_config,
    )

    _, configured_outputs = _paths(config, PROJECT_ROOT)
    if args.preparation_dir is None:
        preparation_dir = create_timestamped_run_dir(configured_outputs, "preparation")
        created_at = None
        existing_preparation = None
    else:
        preparation_dir = resolve_run_path(PROJECT_ROOT, args.preparation_dir)
        preparation_dir.mkdir(parents=True, exist_ok=True)
        config_path = preparation_dir / "config.json"
        existing_preparation = (
            read_run_config(preparation_dir, expected_kind="preparation")
            if config_path.is_file()
            else None
        )
        created_at = (
            str(existing_preparation["created_at"])
            if existing_preparation is not None
            else None
        )

    config["experiment"].setdefault("paths", {})["outputs_dir"] = str(preparation_dir)
    capture_id = capture_artifact_id(config, PROJECT_ROOT)
    if (
        existing_preparation is not None
        and existing_preparation.get("capture_artifact_id") != capture_id
    ):
        raise ValueError(
            f"cannot continue {preparation_dir}: preparation inputs changed; "
            "omit --preparation-dir to create a new preparation"
        )
    vector_root = preparation_dir / "vectors" / capture_id
    write_run_config(
        preparation_dir,
        preparation_payload(
            config,
            preparation_dir,
            capture_artifact_id=capture_id,
            vector_root=vector_root,
            status="running",
            created_at=created_at,
        ),
    )

    validate_pairs(config, PROJECT_ROOT)
    model, processor = load_model_and_processor(config["model"])
    capture_contrasts(config, PROJECT_ROOT, model, processor)
    del model, processor
    extract_vectors(config, PROJECT_ROOT)
    analysis = analyze_vectors(config, PROJECT_ROOT)

    run_config = write_run_config(
        preparation_dir,
        preparation_payload(
            config,
            preparation_dir,
            capture_artifact_id=capture_id,
            vector_root=vector_root,
            status="complete",
            created_at=created_at,
        ),
    )
    print_result(
        {
            "preparation_dir": preparation_dir,
            "config": run_config,
            "vector_root": vector_root,
            **analysis,
        }
    )


if __name__ == "__main__":
    main()

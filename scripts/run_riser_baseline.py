"""Run the isolated RISER fixed-router approximation."""

from pathlib import Path

from _common import DEFAULT_CONFIGS, PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    cli = parser("Run the isolated RISER fixed-router activation-steering baseline")
    cli.add_argument(
        "--preparation-dir",
        type=Path,
        required=True,
        help="frozen preparation directory containing the Stage-03 vector library",
    )
    cli.add_argument(
        "--run-dir",
        type=Path,
        help="existing run to resume; default: outputs/baselines/riser_MMDD_HHMM",
    )
    cli.add_argument(
        "--riser-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baselines" / "riser.yaml",
        help="baseline-only settings; loaded after the normal experiment configs",
    )
    cli.add_argument(
        "--component",
        action="append",
        default=[],
        metavar="CAPABILITY=STRENGTH",
        help="replace the configured fixed selection; repeat to compose several vectors",
    )
    args = cli.parse_args()
    # Load the baseline-specific file last so its required vLLM backend and
    # fixed-router defaults cannot be shadowed by the normal experiment defaults.
    args.config = [*(args.config or DEFAULT_CONFIGS), args.riser_config]
    config = resolved(args)

    from mllm_self_steering.baselines.riser_fixed import (
        parse_component_overrides,
        run_riser_fixed_baseline,
    )
    from mllm_self_steering.pipeline import _paths
    from mllm_self_steering.runs import create_timestamped_run_dir, resolve_run_path

    preparation_dir = resolve_run_path(PROJECT_ROOT, args.preparation_dir)
    if args.run_dir is None:
        _, outputs = _paths(config, PROJECT_ROOT)
        run_dir = create_timestamped_run_dir(outputs / "baselines", "riser")
    else:
        run_dir = resolve_run_path(PROJECT_ROOT, args.run_dir)
    overrides = parse_component_overrides(args.component) if args.component else None
    result = run_riser_fixed_baseline(
        config,
        PROJECT_ROOT,
        preparation_dir=preparation_dir,
        run_dir=run_dir,
        component_overrides=overrides,
    )
    print_result(result)


if __name__ == "__main__":
    main()

"""Run the isolated VISTA VSV-Only baseline."""

from pathlib import Path

from _common import DEFAULT_CONFIGS, PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    cli = parser("Run VISTA VSV-Only with per-item, all-layer visual vectors")
    cli.add_argument(
        "--run-dir",
        type=Path,
        help="existing run to resume; default: outputs/baselines/vista_vsv_MMDD_HHMM",
    )
    cli.add_argument(
        "--vista-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baselines" / "vista_vsv.yaml",
        help="baseline-only settings loaded after the normal experiment configs",
    )
    cli.add_argument(
        "--vsv-lambda",
        type=float,
        help="override the VSV strength; tune on validation data for Qwen3-VL",
    )
    args = cli.parse_args()
    args.config = [*(args.config or DEFAULT_CONFIGS), args.vista_config]
    config = resolved(args)
    if args.vsv_lambda is not None:
        config.setdefault("vista_vsv", {})["lambda"] = args.vsv_lambda

    from mllm_self_steering.baselines.vista_vsv import run_vista_vsv_baseline
    from mllm_self_steering.pipeline import _paths
    from mllm_self_steering.runs import create_timestamped_run_dir, resolve_run_path

    if args.run_dir is None:
        _, outputs = _paths(config, PROJECT_ROOT)
        run_dir = create_timestamped_run_dir(outputs / "baselines", "vista_vsv")
    else:
        run_dir = resolve_run_path(PROJECT_ROOT, args.run_dir)
    print_result(run_vista_vsv_baseline(config, PROJECT_ROOT, run_dir=run_dir))


if __name__ == "__main__":
    main()

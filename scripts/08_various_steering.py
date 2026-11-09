"""Evaluate Table 2 strategies with one frozen checkpoint and vector library."""

from copy import copy
import json
from pathlib import Path

from _common import PROJECT_ROOT, parser, print_result, resolved


def main() -> None:
    from mllm_self_steering.adaptive import (
        _load_base_vectors,
        _validate_preparation_compatibility,
        run_adaptive_steering,
    )
    from mllm_self_steering.pipeline import _limit, _paths
    from mllm_self_steering.subsets import configure_subset, prepare_subset
    from mllm_self_steering.runs import (
        create_timestamped_run_dir,
        preparation_vector_root,
        read_run_config,
        resolve_run_path,
    )
    from mllm_self_steering.utils.io import read_jsonl
    from mllm_self_steering.various import (
        MODES,
        build_various_config,
        evaluation_item,
        evaluation_path,
    )

    cli = parser(
        "Table 2: evaluate a frozen model and bank with different steering weights"
    )
    cli.add_argument("mode", nargs="?", choices=MODES)
    cli.add_argument("--mode", dest="named_mode", choices=MODES)
    cli.add_argument(
        "--preparation-dir",
        type=Path,
        default=Path("outputs/preparation_1105_0414"),
        help="frozen vector library; model settings are inherited from its config.json",
    )
    cli.add_argument(
        "--run-dir",
        "--generation-dir",
        dest="run_dir",
        type=Path,
        help="resume this directory; omit to create a new timestamped run",
    )
    cli.add_argument(
        "--random-seed",
        type=int,
        default=36,
        help="random-mode weights only; does not change the solver's seed",
    )
    cli.add_argument(
        "--datasets", nargs="+", choices=("mmstar", "mmmu", "mathvista", "mathvision")
    )
    cli.add_argument(
        "--subset-fraction",
        type=float,
        default=1.0,
        help="evaluate this fraction per dataset (0 < f <= 1), e.g. 0.3",
    )
    cli.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="sampling seed shared across methods; independent of --random-seed",
    )
    cli.add_argument(
        "--merge-base",
        type=Path,
        help="reuse complementary results from baseline.jsonl or generations.jsonl",
    )
    cli.add_argument(
        "--base-url", help="vLLM endpoint, default from the repository YAML"
    )
    cli.add_argument("--concurrency", type=int, help="router and solver concurrency")
    cli.add_argument(
        "--routing-cache-dir",
        type=Path,
        help="shared self/text/visual route cache; default: outputs/table2_routing",
    )
    cli.add_argument(
        "--dry-run",
        action="store_true",
        help="validate checkpoint, bank and evaluation files without contacting vLLM",
    )
    args = cli.parse_args()
    if args.mode and args.named_mode and args.mode != args.named_mode:
        cli.error("positional mode and --mode disagree")
    mode = args.named_mode or args.mode
    if mode is None:
        cli.error("specify a mode: " + ", ".join(MODES))
    if args.concurrency is not None and args.concurrency < 1:
        cli.error("--concurrency must be positive")

    # Apply overrides after inheriting the bank so explicit changes are validated,
    # rather than silently replacing a requested checkpoint with a different one.
    base_args = copy(args)
    base_args.override = []
    preparation_dir = resolve_run_path(PROJECT_ROOT, args.preparation_dir)
    preparation = read_run_config(preparation_dir, expected_kind="preparation")
    overrides = list(args.override)
    if args.datasets:
        overrides.append("data.enabled_datasets=" + json.dumps(args.datasets))
    if args.base_url:
        overrides.append("experiment.vllm_hook.base_url=" + json.dumps(args.base_url))
    if args.concurrency is not None:
        overrides.extend(
            f"experiment.adaptive_steering.{key}={args.concurrency}"
            for key in ("selector_concurrency", "generation_concurrency")
        )
    config = build_various_config(
        resolved(base_args),
        preparation,
        mode=mode,
        random_seed=args.random_seed,
        overrides=overrides,
    )
    configure_subset(
        config,
        PROJECT_ROOT,
        fraction=args.subset_fraction,
        seed=args.subset_seed,
        merge_base=args.merge_base,
    )
    _validate_preparation_compatibility(config, preparation, preparation_dir)
    vector_root = preparation_vector_root(preparation_dir, preparation)
    capabilities = config["experiment"]["capabilities"]
    form = {"mean_norm": "steering", "raw": "raw", "unit": "unit"}[
        config["experiment"]["vector_scaling"]
    ]
    _, _, fingerprints = _load_base_vectors(
        vector_root,
        [row["id"] for row in capabilities],
        form,
    )
    _, outputs = _paths(config, PROJECT_ROOT)
    routing_cache = (
        resolve_run_path(PROJECT_ROOT, args.routing_cache_dir)
        if args.routing_cache_dir
        else outputs / "table2_routing"
    )
    config["experiment"]["various_steering"]["routing_cache_dir"] = str(routing_cache)
    datasets = config["data"]["enabled_datasets"]
    population = {}
    subset_records = []
    subset_enabled = "subset" in config["experiment"]["various_steering"]
    for dataset in datasets:
        path = evaluation_path(config, PROJECT_ROOT, dataset)
        rows = _limit(config, list(read_jsonl(path)), scope="eval")
        if not rows:
            raise ValueError(f"evaluation dataset is empty: {path}")
        population[dataset] = {
            "path": str(path),
            "count": len(rows),
            "recorded_splits": sorted({str(row.get("split")) for row in rows}),
        }
        if subset_enabled:
            from mllm_self_steering.pipeline import _image_hashes, item_identity

            for row in rows:
                item = evaluation_item(row, PROJECT_ROOT)
                hashes = _image_hashes(item)
                subset_records.append(
                    (item, hashes, item_identity(item, image_hashes=hashes))
                )
    subset_plan = prepare_subset(config, subset_records) if subset_enabled else None
    if subset_plan is not None:
        for name, counts in subset_plan.manifest["datasets"].items():
            population[name]["subset"] = counts
    plan = {
        "mode": mode,
        "model": config["model"]["name"],
        "preparation_dir": str(preparation_dir),
        "vector_root": str(vector_root),
        "vector_fingerprints": fingerprints,
        "target_layer": config["experiment"]["target_layer"],
        "vector_scaling": config["experiment"]["vector_scaling"],
        "first_feature": capabilities[0]["id"],
        "random_seed": args.random_seed if mode == "random" else None,
        "random_distribution": "Dirichlet(1,1,1,1,1,1)" if mode == "random" else None,
        "baseline_no_answer_only": False,
        "base_url": config["experiment"]["vllm_hook"]["base_url"],
        "max_new_tokens": config["model"]["max_new_tokens"],
        "global_scale": config["experiment"]["adaptive_steering"]["global_scale"],
        "routing_cache_dir": str(routing_cache),
        "evaluation": population,
        "sampling": subset_plan.settings if subset_plan is not None else None,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    run_dir = (
        resolve_run_path(PROJECT_ROOT, args.run_dir)
        if args.run_dir
        else create_timestamped_run_dir(outputs, f"table2_{mode}")
    )
    print(
        json.dumps({"run_dir": str(run_dir), **plan}, ensure_ascii=False, indent=2),
        flush=True,
    )
    artifacts = run_adaptive_steering(
        config,
        PROJECT_ROOT,
        preparation_dir=preparation_dir,
        generation_dir=run_dir,
    )
    report = json.loads(artifacts["metrics_json"].read_text(encoding="utf-8"))
    print_result(artifacts)
    if not report["table2"]["complete"]:
        raise RuntimeError(
            "Table 2 run is incomplete; resume using the same mode and --run-dir"
        )


if __name__ == "__main__":
    main()

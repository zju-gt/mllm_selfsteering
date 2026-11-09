from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest
import torch

from mllm_self_steering.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES = [
    {"id": "text_logic", "modality": "text"},
    {"id": "text_quantitative", "modality": "text"},
    {"id": "text_relevance", "modality": "text"},
    {"id": "vision_detail", "modality": "visual"},
    {"id": "vision_spatial", "modality": "visual"},
    {"id": "vision_grounding", "modality": "visual"},
]


def various():
    return importlib.import_module("mllm_self_steering.various")


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("none", [0, 0, 0, 0, 0, 0]),
        ("single", [1, 0, 0, 0, 0, 0]),
        ("all", [1 / 6] * 6),
    ],
)
def test_fixed_modes_use_the_declared_coefficient_budget(mode, expected):
    weights = various().strategy_weights(mode, CAPABILITIES, "item-a", seed=36)
    assert list(weights.values()) == pytest.approx(expected)


def test_random_weights_are_item_seeded_and_independent_of_processing_order():
    module = various()
    first = module.strategy_weights("random", CAPABILITIES, "item-a", seed=36)
    other = module.strategy_weights("random", CAPABILITIES, "item-b", seed=36)
    repeat = module.strategy_weights("random", CAPABILITIES, "item-a", seed=36)
    changed = module.strategy_weights("random", CAPABILITIES, "item-a", seed=37)
    assert first == repeat
    assert first != other
    assert first != changed
    assert sum(first.values()) == pytest.approx(1)
    assert all(0 < weight < 1 for weight in first.values())


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("text", [2 / 3, 1 / 3, 0, 0, 0, 0]),
        ("visual", [0, 0, 0, 0.75, 0.25, 0]),
        ("self", [0.4, 0.2, 0, 0.3, 0.1, 0]),
    ],
)
def test_routed_modes_mask_and_renormalize_without_mutating_the_route(mode, expected):
    routed = dict(zip([row["id"] for row in CAPABILITIES], [0.4, 0.2, 0, 0.3, 0.1, 0]))
    before = dict(routed)
    weights = various().strategy_weights(
        mode, CAPABILITIES, "item-a", seed=36, routed_weights=routed
    )
    assert list(weights.values()) == pytest.approx(expected)
    assert routed == before


def test_removing_all_selected_features_preserves_the_zero_bypass():
    routed = dict(zip([row["id"] for row in CAPABILITIES], [0, 0, 0, 1, 0, 0]))
    weights = various().strategy_weights(
        "text", CAPABILITIES, "item-a", seed=36, routed_weights=routed
    )
    assert list(weights.values()) == [0] * 6


def test_configuration_inherits_the_preparation_model_and_disables_gating():
    base = load_config(
        [
            ROOT / "configs" / name
            for name in ("model.yaml", "data.yaml", "experiment.yaml")
        ]
    )
    preparation = {"config": deepcopy(base), "status": "complete"}
    preparation["config"]["model"]["name"] = "/trained/checkpoint"
    preparation["config"]["experiment"]["target_layer"] = 17
    base["experiment"]["adaptive_steering"]["baseline_no_answer_only"] = True
    config = various().build_various_config(
        base, preparation, mode="single", random_seed=36
    )
    assert config["model"]["name"] == "/trained/checkpoint"
    assert config["experiment"]["target_layer"] == 17
    assert config["experiment"]["adaptive_steering"]["baseline_no_answer_only"] is False
    assert config["experiment"]["generation"]["backend"] == "vllm_hook"
    assert base["experiment"]["adaptive_steering"]["baseline_no_answer_only"] is True


def test_table2_defaults_to_mathvision_test_but_allows_explicit_testmini():
    base = load_config(
        [
            ROOT / "configs" / name
            for name in ("model.yaml", "data.yaml", "experiment.yaml")
        ]
    )
    preparation = {"config": deepcopy(base), "status": "complete"}
    config = various().build_various_config(
        base, preparation, mode="single", random_seed=36
    )
    source = config["data"]["sources"]["mathvision"]
    assert source["local_path"] == "data/processed/mathvision.jsonl"
    assert source["split"] == source["source_split"] == "test"
    explicit = various().build_various_config(
        base,
        preparation,
        mode="single",
        random_seed=36,
        overrides=[
            "data.sources.mathvision.local_path=data/processed/mathvision_testmini.jsonl",
            "data.sources.mathvision.split=testmini",
            "data.sources.mathvision.source_split=testmini",
        ],
    )
    assert explicit["data"]["sources"]["mathvision"]["local_path"].endswith(
        "mathvision_testmini.jsonl"
    )
    assert explicit["data"]["sources"]["mathvision"]["split"] == "testmini"


def test_table2_shell_defaults_and_user_overrides_work_from_another_directory(tmp_path):
    import os
    import subprocess

    capture = tmp_path / "capture-python"
    capture.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "print(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:]}))\n"
    )
    capture.chmod(0o755)
    env = dict(os.environ)
    for key in (
        "MODE",
        "PREPARATION_DIR",
        "SUBSET_FRACTION",
        "SUBSET_SEED",
        "RANDOM_SEED",
        "MERGE_BASE",
        "RUN_DIR",
        "BASE_URL",
        "CONCURRENCY",
        "GLOBAL_SCALE",
        "MAX_NEW_TOKENS",
        "ROUTING_CACHE_DIR",
    ):
        env.pop(key, None)
    env["PYTHON_BIN"] = str(capture)
    result = subprocess.run(
        ["bash", str(ROOT / "table2.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    invocation = json.loads(result.stdout)
    args = invocation["args"]
    assert invocation["cwd"] == str(ROOT)
    assert args[:2] == ["scripts/08_various_steering.py", "single"]
    assert args[args.index("--subset-fraction") + 1] == "0.3"
    assert args[args.index("--concurrency") + 1] == "8"
    assert "--merge-base" not in args
    env.update(
        MERGE_BASE="/tmp/base with spaces.jsonl",
        RUN_DIR="outputs/my run",
        SUBSET_FRACTION="0.5",
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "table2.sh"),
            "random",
            "--subset-fraction",
            "0.7",
            "--dry-run",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    args = json.loads(result.stdout)["args"]
    assert args[1] == "random"
    assert args[args.index("--merge-base") + 1] == "/tmp/base with spaces.jsonl"
    assert args[args.index("--run-dir") + 1] == "outputs/my run"
    assert args[args.index("--subset-fraction") + 1] == "0.5"
    assert args[-3:] == ["--subset-fraction", "0.7", "--dry-run"]


def test_evaluation_path_honors_the_configured_mathvision_file(tmp_path):
    config = {
        "data": {
            "sources": {
                "mathvision": {"local_path": "data/processed/mathvision_testmini.jsonl"}
            }
        },
        "experiment": {"paths": {"data_dir": "data"}, "various_steering": {}},
    }
    assert various().evaluation_path(config, tmp_path, "mathvision") == (
        tmp_path / "data/processed/mathvision_testmini.jsonl"
    )
    del config["experiment"]["various_steering"]
    assert various().evaluation_path(config, tmp_path, "mathvision") == (
        tmp_path / "data/processed/mathvision.jsonl"
    )


def test_resume_identity_distinguishes_modes_seeds_and_actual_data(tmp_path):
    from mllm_self_steering.adaptive import _adaptive_run_id

    module = various()
    base = load_config(
        [
            ROOT / "configs" / name
            for name in ("model.yaml", "data.yaml", "experiment.yaml")
        ]
    )
    base["data"]["enabled_datasets"] = ["mathvision"]
    target = tmp_path / "data/processed/mathvision_testmini.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n")
    config = module.build_various_config(
        base,
        {"config": base, "status": "complete"},
        mode="single",
        random_seed=36,
        overrides=[
            "data.sources.mathvision.local_path=data/processed/mathvision_testmini.jsonl"
        ],
    )

    def identity(value):
        return _adaptive_run_id(
            value,
            tmp_path,
            capture_id="bank",
            vector_fingerprints={"a": "hash"},
            prompt_hash="prompt",
        )

    first = identity(config)
    changed = deepcopy(config)
    changed["experiment"]["various_steering"]["mode"] = "all"
    assert identity(changed) != first
    changed["experiment"]["various_steering"].update(mode="random", random_seed=36)
    random_first = identity(changed)
    changed["experiment"]["various_steering"]["random_seed"] = 37
    assert identity(changed) != random_first
    target.write_text('{"changed":true}\n')
    assert identity(config) != first


def test_table2_average_is_macro_and_unfinished_runs_have_no_final_average():
    report = {
        "complete": True,
        "groups": {
            "overall": {"count": 10, "adaptive_accuracy": 0.9},
            "mmstar": {"count": 9, "adaptive_accuracy": 1.0, "baseline_accuracy": 0.5},
            "mmmu": {"count": 1, "adaptive_accuracy": 0.0, "baseline_accuracy": 0.0},
        },
    }
    summary = various().table2_summary(report, ["mmstar", "mmmu"], "single")
    assert summary["macro_accuracy"] == 0.5
    assert summary["macro_baseline_accuracy"] == 0.25
    report["complete"] = False
    assert (
        various().table2_summary(report, ["mmstar", "mmmu"], "single")["macro_accuracy"]
        is None
    )


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """Keep the pipeline real; replace only the external GPU server."""
    from mllm_self_steering import adaptive
    from mllm_self_steering.runs import preparation_payload, write_run_config
    from mllm_self_steering.utils.io import write_jsonl
    from mllm_self_steering.vectors.storage import export_vllm_hook, save_library

    config = load_config(
        [
            ROOT / "configs" / name
            for name in ("model.yaml", "data.yaml", "experiment.yaml")
        ]
    )
    config["model"]["name"] = "/trained/checkpoint"
    config["data"]["enabled_datasets"] = ["mathvision"]
    config["experiment"]["paths"] = {"data_dir": "data", "outputs_dir": "outputs"}
    config["experiment"]["vllm_hook"].update(verify_steering_effect=False)
    data = tmp_path / "data"
    data.mkdir()
    image = data / "image.png"
    image.write_bytes(b"test-image-content")
    rows = [
        {
            "item_id": f"question-{i}",
            "dataset": "mathvision",
            "split": "testmini",
            "prompt": f"Question {i}?",
            "image_paths": [str(image)],
            "gold_answer": "A",
            "answer_type": "choice",
            "choices": {"A": "yes", "B": "no"},
        }
        for i in range(2)
    ]
    write_jsonl(data / "processed/mathvision_testmini.jsonl", rows)
    # Stage 08 must not accidentally evaluate the old full-test file.
    write_jsonl(data / "processed/mathvision.jsonl", [{"unexpected": "full-test"}])
    preparation_dir = tmp_path / "outputs/preparation"
    vector_root = preparation_dir / "vectors/bank"
    vectors = {
        row["id"]: {
            form: torch.tensor([float(i + 1), 0.0])
            for form in ("raw", "unit", "steering")
        }
        for i, row in enumerate(config["experiment"]["capabilities"])
    }
    export_vllm_hook(vector_root / "vllm_hook", vectors)
    save_library(
        vector_root / "capability_vectors.safetensors",
        vector_root / "capability_vectors.json",
        vectors,
        metadata={},
    )
    preparation = preparation_payload(
        config,
        preparation_dir,
        capture_artifact_id="bank",
        vector_root=vector_root,
        status="complete",
    )
    write_run_config(preparation_dir, preparation)
    activity = {
        "routes": 0,
        "solves": 0,
        "probes": 0,
        "invalid_route": False,
        "fail_solves": False,
    }

    class Server:
        def __init__(self, **kwargs):
            pass

        async def preflight(self):
            pass

        async def close(self):
            pass

        async def verify_steering_effect(self, **kwargs):
            activity["probes"] += 1

        async def generate(
            self,
            *,
            messages,
            vector_path,
            target_layer,
            alpha,
            phase,
            positions,
            max_new_tokens,
            temperature=0,
            seed=None,
        ):
            if temperature:
                activity["routes"] += 1
                if activity["invalid_route"]:
                    return "invalid JSON"
                return json.dumps(
                    {
                        "weights": dict(
                            zip(
                                [row["id"] for row in CAPABILITIES],
                                [0.8, 0.4, 0, 0.6, 0.2, 0],
                            )
                        )
                    }
                )
            activity["solves"] += 1
            if activity["fail_solves"]:
                raise RuntimeError("test GPU server is unavailable")
            if vector_path is not None:
                payload = torch.load(vector_path, weights_only=True)
                assert torch.isfinite(payload["dir"]).all()
            return "Final Answer: A"

    monkeypatch.setattr(adaptive, "AsyncVLLMHookClient", Server)

    def configured(mode, seed=36):
        value = various().build_various_config(
            config,
            preparation,
            mode=mode,
            random_seed=seed,
            overrides=[
                "data.sources.mathvision.local_path=data/processed/mathvision_testmini.jsonl",
                "data.sources.mathvision.split=testmini",
                "data.sources.mathvision.source_split=testmini",
            ],
        )
        value["experiment"]["various_steering"]["routing_cache_dir"] = str(
            tmp_path / "outputs/table2_routing"
        )
        return value

    return tmp_path, preparation_dir, configured, activity


@pytest.mark.parametrize(
    "mode", ["none", "single", "all", "random", "text", "visual", "self"]
)
def test_modes_generate_real_checkpoints_and_resume_without_repeating_requests(
    experiment, mode
):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.utils.io import read_jsonl

    root, preparation, configured, activity = experiment
    config = configured(mode)
    if mode == "none":
        config["experiment"]["vllm_hook"]["verify_steering_effect"] = True
    run_dir = root / f"outputs/{mode}"
    result = run_adaptive_steering(
        config, root, preparation_dir=preparation, generation_dir=run_dir
    )
    rows = list(read_jsonl(result["generations"]))
    assert len(rows) == 2
    assert all(row["correct"] and row["baseline_correct"] for row in rows)
    assert all(row["steering_mode"] == mode for row in rows)
    assert all(row["steering_applied"] == (mode != "none") for row in rows)
    assert activity["routes"] == (2 if mode in {"text", "visual", "self"} else 0)
    assert activity["probes"] == 0
    if mode == "single":
        assert rows[0]["weights"] == dict(
            zip([r["id"] for r in CAPABILITIES], [1, 0, 0, 0, 0, 0])
        )
    elif mode == "text":
        assert list(rows[0]["weights"].values()) == pytest.approx(
            [2 / 3, 1 / 3, 0, 0, 0, 0]
        )
    elif mode == "visual":
        # JSON keys are sorted on disk: compare by capability rather than iteration order.
        assert rows[0]["weights"]["vision_detail"] == pytest.approx(0.75)
        assert rows[0]["weights"]["vision_spatial"] == pytest.approx(0.25)
    metrics = json.loads(result["metrics_json"].read_text())
    assert metrics["complete"] is True
    assert metrics["expected_items"] == 2
    assert metrics["table2"]["macro_accuracy"] == 1
    assert result["table2_csv"].is_file()
    before = dict(activity)
    run_adaptive_steering(
        config, root, preparation_dir=preparation, generation_dir=run_dir
    )
    assert activity == before
    changed = configured("all" if mode != "all" else "single")
    with pytest.raises(ValueError, match="experiment_fingerprint"):
        run_adaptive_steering(
            changed, root, preparation_dir=preparation, generation_dir=run_dir
        )


def test_modal_ablations_reuse_the_exact_same_cached_full_routes(experiment):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.utils.io import read_jsonl

    root, preparation, configured, activity = experiment
    results = {}
    for mode in ("self", "text", "visual"):
        result = run_adaptive_steering(
            configured(mode),
            root,
            preparation_dir=preparation,
            generation_dir=root / f"outputs/{mode}",
        )
        results[mode] = list(read_jsonl(result["selections"]))
    assert activity["routes"] == 2
    for index in range(2):
        for mode in ("text", "visual"):
            assert results[mode][index]["routing_cache_hit"] is True
            assert (
                results[mode][index]["routed_weights"]
                == results["self"][index]["routed_weights"]
            )


def test_failed_routes_do_not_mark_an_incomplete_population_complete(experiment):
    from mllm_self_steering.adaptive import run_adaptive_steering

    root, preparation, configured, activity = experiment
    activity["invalid_route"] = True
    result = run_adaptive_steering(
        configured("self"),
        root,
        preparation_dir=preparation,
        generation_dir=root / "outputs/failed",
    )
    assert json.loads(result["metrics_json"].read_text())["complete"] is False
    assert json.loads(result["config"].read_text())["status"] == "incomplete"


def test_parallel_route_publishers_cannot_replace_an_already_shared_route(tmp_path):
    module = various()
    target = tmp_path / "cache/item.json"
    first = {"selector_raw_output": "first route", "selector_attempts": 1}
    second = {"selector_raw_output": "different route", "selector_attempts": 2}
    assert module.publish_route(target, first) == first
    assert module.publish_route(target, second) == first
    assert json.loads(target.read_text()) == first


def test_five_server_failures_save_incomplete_metrics_before_raising(experiment):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.utils.io import read_jsonl, write_jsonl

    root, preparation, configured, activity = experiment
    path = root / "data/processed/mathvision_testmini.jsonl"
    template = list(read_jsonl(path))[0]
    write_jsonl(path, [{**template, "item_id": f"item-{i}"} for i in range(6)])
    activity["fail_solves"] = True
    run_dir = root / "outputs/interrupted"
    with pytest.raises(RuntimeError, match="5 consecutive request failures"):
        run_adaptive_steering(
            configured("single"),
            root,
            preparation_dir=preparation,
            generation_dir=run_dir,
        )
    assert json.loads((run_dir / "config.json").read_text())["status"] == "incomplete"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["complete"] is False
    assert metrics["table2"]["macro_accuracy"] is None
    assert (run_dir / "table2.csv").is_file()


def test_subset_sampling_is_per_dataset_and_independent_of_order_and_mode():
    from mllm_self_steering.subsets import select_subset_keys

    keys = [("mmstar", f"item-{i}") for i in range(10)] + [
        ("mathvision", f"item-{i}") for i in range(4)
    ]
    selected = select_subset_keys(keys, fraction=0.3, seed=42)
    assert len([key for key in selected if key[0] == "mmstar"]) == 3
    assert len([key for key in selected if key[0] == "mathvision"]) == 2
    assert selected == select_subset_keys(list(reversed(keys)), fraction=0.3, seed=42)
    assert selected != select_subset_keys(keys, fraction=0.3, seed=43)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_subset_rejects_invalid_fractions(fraction):
    from mllm_self_steering.subsets import select_subset_keys

    with pytest.raises(ValueError, match="fraction"):
        select_subset_keys([("mmstar", "a")], fraction=fraction, seed=42)


def _subset_population(experiment):
    from mllm_self_steering.datasets.types import MultimodalItem
    from mllm_self_steering.pipeline import item_identity
    from mllm_self_steering.utils.io import read_jsonl, write_jsonl

    root, _, _, _ = experiment
    path = root / "data/processed/mathvision_testmini.jsonl"
    template = list(read_jsonl(path))[0]
    items = [
        {**template, "item_id": f"item-{i}", "prompt": f"Question {i}?"}
        for i in range(10)
    ]
    write_jsonl(path, items)
    base = [
        {
            "run_id": "reference-run",
            "dataset": row["dataset"],
            "item_id": row["item_id"],
            "item_identity": item_identity(MultimodalItem.from_dict(row)),
            "gold_answer": "A",
            "answer_type": "choice",
            "baseline_raw_output": "Final Answer: B",
            # Deliberately wrong cached score: the merger must recompute it.
            "baseline_correct": True,
            "status": "ok",
        }
        for row in items
    ]
    base_path = root / "base.jsonl"
    write_jsonl(base_path, list(reversed(base)))
    return base_path, base


@pytest.mark.parametrize(
    "mode", ["none", "single", "all", "random", "text", "visual", "self"]
)
def test_subset_evaluates_three_and_merges_seven_with_traceable_metrics(
    experiment, mode
):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.subsets import configure_subset
    from mllm_self_steering.utils.io import read_jsonl

    root, preparation, configured, activity = experiment
    base_path, _ = _subset_population(experiment)
    before = base_path.read_bytes()
    config = configured(mode)
    configure_subset(config, root, fraction=0.3, seed=42, merge_base=base_path)
    result = run_adaptive_steering(
        config,
        root,
        preparation_dir=preparation,
        generation_dir=root / f"outputs/{mode}",
    )
    actual = list(read_jsonl(result["generations"]))
    merged = list(read_jsonl(result["hybrid_generations"]))
    assert len(actual) == 3
    assert activity["solves"] == (3 if mode == "none" else 6)
    assert len(merged) == 10
    assert [row["item_id"] for row in merged] == [f"item-{i}" for i in range(10)]
    assert sum(row["result_source"] == "evaluated" for row in merged) == 3
    assert sum(row["result_source"] == "base" for row in merged) == 7
    assert sum(row["correct"] for row in merged) == 3
    assert all(
        row["source_run_id"] == "reference-run"
        for row in merged
        if row["result_source"] == "base"
    )
    assert base_path.read_bytes() == before
    metrics = json.loads(result["metrics_json"].read_text())
    assert metrics["expected_items"] == 3
    assert metrics["table2"]["evaluation_scope"] == "subset"
    assert metrics["table2"]["macro_accuracy"] == 1
    hybrid = json.loads(result["hybrid_metrics"].read_text())
    assert hybrid["result_kind"] == "hybrid"
    assert hybrid["complete"] is True
    assert hybrid["expected_items"] == 10
    assert hybrid["macro_accuracy"] == pytest.approx(0.3)
    assert hybrid["consistency"]["compared_items"] == 3
    assert hybrid["consistency"]["prediction_agreement"] == 0
    assert hybrid["consistency"]["wrong_to_right"] == 3
    assert result["hybrid_table2_csv"].is_file()
    manifest = json.loads(result["subset_manifest"].read_text())
    assert manifest["datasets"]["mathvision"] == {
        "total": 10,
        "evaluated": 3,
        "base": 7,
    }
    previous_activity = dict(activity)
    run_adaptive_steering(
        config,
        root,
        preparation_dir=preparation,
        generation_dir=root / f"outputs/{mode}",
    )
    assert activity == previous_activity


@pytest.mark.parametrize("problem", ["missing", "identity", "gold", "duplicate"])
def test_merge_base_is_validated_before_any_inference(experiment, problem):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.subsets import configure_subset
    from mllm_self_steering.utils.io import write_jsonl

    root, preparation, configured, activity = experiment
    base_path, base = _subset_population(experiment)
    if problem == "missing":
        base = []
    elif problem == "identity":
        base = [{**row, "item_identity": "stale"} for row in base]
    elif problem == "gold":
        base = [{**row, "gold_answer": "B"} for row in base]
    else:
        base.append({**base[0], "run_id": "different-run"})
    write_jsonl(base_path, base)
    config = configured("single")
    configure_subset(config, root, fraction=0.3, seed=42, merge_base=base_path)
    with pytest.raises(ValueError):
        run_adaptive_steering(
            config,
            root,
            preparation_dir=preparation,
            generation_dir=root / "outputs/rejected",
        )
    assert activity["solves"] == 0
    assert activity["routes"] == 0


def test_failed_subset_items_are_not_silently_filled_from_base(experiment):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.subsets import configure_subset
    from mllm_self_steering.utils.io import read_jsonl

    root, preparation, configured, activity = experiment
    base_path, _ = _subset_population(experiment)
    config = configured("self")
    configure_subset(config, root, fraction=0.3, seed=42, merge_base=base_path)
    activity["invalid_route"] = True
    result = run_adaptive_steering(
        config,
        root,
        preparation_dir=preparation,
        generation_dir=root / "outputs/partial",
    )
    hybrid = json.loads(result["hybrid_metrics"].read_text())
    assert hybrid["complete"] is False
    assert hybrid["macro_accuracy"] is None
    assert len(list(read_jsonl(result["hybrid_generations"]))) == 7
    assert json.loads(result["config"].read_text())["status"] == "incomplete"


def test_subset_resume_rejects_changed_seed_or_reference_contents(experiment):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.subsets import configure_subset
    from mllm_self_steering.utils.io import write_jsonl

    root, preparation, configured, _ = experiment
    base_path, base = _subset_population(experiment)
    config = configured("single")
    configure_subset(config, root, fraction=0.3, seed=42, merge_base=base_path)
    run_dir = root / "outputs/subset"
    run_adaptive_steering(
        config, root, preparation_dir=preparation, generation_dir=run_dir
    )
    changed = configured("single")
    configure_subset(changed, root, fraction=0.3, seed=43, merge_base=base_path)
    with pytest.raises(ValueError, match="experiment_fingerprint"):
        run_adaptive_steering(
            changed, root, preparation_dir=preparation, generation_dir=run_dir
        )
    write_jsonl(
        base_path, [{**row, "baseline_raw_output": "Final Answer: A"} for row in base]
    )
    changed = configured("single")
    configure_subset(changed, root, fraction=0.3, seed=42, merge_base=base_path)
    with pytest.raises(ValueError, match="experiment_fingerprint"):
        run_adaptive_steering(
            changed, root, preparation_dir=preparation, generation_dir=run_dir
        )


def test_subset_without_base_and_complement_only_reference_share_the_same_sample(
    experiment,
):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.subsets import configure_subset
    from mllm_self_steering.utils.io import read_jsonl, write_jsonl

    root, preparation, configured, _ = experiment
    base_path, base = _subset_population(experiment)
    config = configured("single")
    configure_subset(config, root, fraction=0.3, seed=42)
    first = run_adaptive_steering(
        config,
        root,
        preparation_dir=preparation,
        generation_dir=root / "outputs/subset-only",
    )
    selected = {row["item_identity"] for row in read_jsonl(first["generations"])}
    assert len(selected) == 3
    assert "hybrid_generations" not in first
    write_jsonl(
        base_path, [row for row in base if row["item_identity"] not in selected]
    )
    other = configured("random", seed=999)
    configure_subset(other, root, fraction=0.3, seed=42, merge_base=base_path)
    second = run_adaptive_steering(
        other,
        root,
        preparation_dir=preparation,
        generation_dir=root / "outputs/complement",
    )
    assert {
        row["item_identity"] for row in read_jsonl(second["generations"])
    } == selected
    hybrid = json.loads(second["hybrid_metrics"].read_text())
    assert hybrid["complete"] is True
    assert hybrid["consistency"]["compared_items"] == 0
    assert hybrid["consistency"]["prediction_agreement"] is None
    assert hybrid["random_seed"] == 999


@pytest.mark.parametrize("prediction_only", [False, True])
def test_merge_accepts_generation_and_prediction_only_reference_results(
    experiment, prediction_only
):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.subsets import configure_subset
    from mllm_self_steering.utils.io import write_jsonl

    root, preparation, configured, _ = experiment
    base_path, base = _subset_population(experiment)
    for row in base:
        row.pop("baseline_raw_output")
        row["predicted_answer"] = "B"
        row["correct"] = True
        if not prediction_only:
            row["raw_output"] = "Final Answer: B"
    write_jsonl(base_path, base)
    config = configured("single")
    configure_subset(config, root, fraction=0.3, seed=42, merge_base=base_path)
    result = run_adaptive_steering(
        config,
        root,
        preparation_dir=preparation,
        generation_dir=root / "outputs/reference-format",
    )
    report = json.loads(result["hybrid_metrics"].read_text())
    assert report["macro_accuracy"] == pytest.approx(0.3)
    assert report["consistency"]["accuracy_delta"] == 1


def test_aborted_subset_keeps_hybrid_incomplete_and_rejects_changed_base(experiment):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.subsets import configure_subset
    from mllm_self_steering.utils.io import read_jsonl, write_jsonl

    root, preparation, configured, activity = experiment
    base_path, base = _subset_population(experiment)
    config = configured("single")
    configure_subset(config, root, fraction=0.6, seed=42, merge_base=base_path)
    activity["fail_solves"] = True
    run_dir = root / "outputs/abort-subset"
    with pytest.raises(RuntimeError, match="5 consecutive request failures"):
        run_adaptive_steering(
            config, root, preparation_dir=preparation, generation_dir=run_dir
        )
    assert json.loads((run_dir / "config.json").read_text())["status"] == "incomplete"
    report = json.loads((run_dir / "hybrid_metrics.json").read_text())
    assert report["complete"] is False
    assert report["macro_accuracy"] is None
    assert len(list(read_jsonl(run_dir / "hybrid_generations.jsonl"))) == 4
    previous = dict(activity)
    write_jsonl(
        base_path, [{**row, "baseline_raw_output": "Final Answer: A"} for row in base]
    )
    with pytest.raises(ValueError, match="contents changed"):
        run_adaptive_steering(
            config, root, preparation_dir=preparation, generation_dir=run_dir
        )
    assert activity == previous


def test_subset_resolves_migrated_windows_image_paths_without_rewriting_data(
    experiment,
):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.subsets import configure_subset
    from mllm_self_steering.utils.io import read_jsonl, write_jsonl

    root, preparation, configured, _ = experiment
    base_path, _ = _subset_population(experiment)
    path = root / "data/processed/mathvision_testmini.jsonl"
    rows = list(read_jsonl(path))
    image = root / "data/images/mathvision_testmini/test.png"
    image.parent.mkdir(parents=True)
    image.write_bytes((root / "data/image.png").read_bytes())
    for row in rows:
        row["image_paths"] = [
            r"D:\\old-project\\data\\images\\mathvision_testmini\\test.png"
        ]
    write_jsonl(path, rows)
    original = path.read_bytes()
    config = configured("single")
    configure_subset(config, root, fraction=0.3, seed=42, merge_base=base_path)
    result = run_adaptive_steering(
        config,
        root,
        preparation_dir=preparation,
        generation_dir=root / "outputs/migrated",
    )
    assert json.loads(result["hybrid_metrics"].read_text())["complete"] is True
    assert path.read_bytes() == original


def test_subset_resume_rejects_changed_image_bytes_before_overwriting_manifest(
    experiment,
):
    from mllm_self_steering.adaptive import run_adaptive_steering
    from mllm_self_steering.subsets import configure_subset

    root, preparation, configured, activity = experiment
    _subset_population(experiment)
    config = configured("single")
    configure_subset(config, root, fraction=0.3, seed=42)
    run_dir = root / "outputs/image-resume"
    result = run_adaptive_steering(
        config, root, preparation_dir=preparation, generation_dir=run_dir
    )
    manifest_before = result["subset_manifest"].read_bytes()
    previous_activity = dict(activity)
    (root / "data/image.png").write_bytes(b"different-image-content-at-the-same-path")
    with pytest.raises(ValueError, match="experiment_fingerprint"):
        run_adaptive_steering(
            config, root, preparation_dir=preparation, generation_dir=run_dir
        )
    assert result["subset_manifest"].read_bytes() == manifest_before
    assert activity == previous_activity

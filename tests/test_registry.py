import json
from pathlib import Path

from PIL import Image

from mllm_self_steering.datasets.registry import load_dataset_items


def test_canonical_images_are_materialized_under_data_root(tmp_path: Path) -> None:
    external = tmp_path / "external" / "source.jpg"
    external.parent.mkdir()
    Image.new("RGB", (2, 2)).save(external)
    canonical = tmp_path / "canonical" / "mmstar.jsonl"
    canonical.parent.mkdir()
    canonical.write_text(
        json.dumps(
            {
                "item_id": "mmstar_1",
                "dataset": "mmstar",
                "split": "test",
                "prompt": "Which?",
                "image_paths": [str(external)],
                "gold_answer": "A",
                "answer_type": "choice",
                "choices": {"A": "one", "B": "two"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    image_root = tmp_path / "data" / "images"

    items = load_dataset_items(
        "mmstar",
        {"local_path": str(canonical), "split": "test"},
        image_root=image_root,
    )

    copied = Path(items[0].image_paths[0])
    assert copied.is_file()
    assert copied.is_relative_to((image_root / "mmstar").resolve())
    assert copied != external.resolve()

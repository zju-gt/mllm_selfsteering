from pathlib import Path

import pytest

from mllm_self_steering.utils.io import (
    JsonlCheckpointWriter,
    read_jsonl,
    repair_truncated_jsonl,
)


def test_checkpoint_writer_appends_valid_jsonl(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "rows.jsonl"
    with JsonlCheckpointWriter(destination, sync_every=2) as writer:
        writer.write({"index": 1})
        writer.write({"index": 2})
        writer.write({"index": 3})

    assert list(read_jsonl(destination)) == [
        {"index": 1},
        {"index": 2},
        {"index": 3},
    ]


def test_repair_truncated_jsonl_discards_only_final_fragment(tmp_path: Path) -> None:
    destination = tmp_path / "rows.jsonl"
    destination.write_bytes(b'{"index": 1}\n{"index":')

    assert repair_truncated_jsonl(destination)
    assert list(read_jsonl(destination)) == [{"index": 1}]
    assert destination.read_bytes() == b'{"index": 1}\n'


def test_repair_truncated_jsonl_keeps_complete_final_object(tmp_path: Path) -> None:
    destination = tmp_path / "rows.jsonl"
    destination.write_bytes(b'{"index": 1}')

    assert repair_truncated_jsonl(destination)
    assert list(read_jsonl(destination)) == [{"index": 1}]
    assert destination.read_bytes().endswith(b"\n")


def test_repair_does_not_hide_complete_non_object_row(tmp_path: Path) -> None:
    destination = tmp_path / "rows.jsonl"
    destination.write_bytes(b"[]")

    assert repair_truncated_jsonl(destination)
    with pytest.raises(ValueError, match="must be an object"):
        list(read_jsonl(destination))

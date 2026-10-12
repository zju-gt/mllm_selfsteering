import asyncio
from pathlib import Path
from typing import Any

from mllm_self_steering.pipeline import (
    _GenerationTarget,
    _VLLMJob,
    _execute_vllm_jobs,
)
from mllm_self_steering.utils.io import JsonlCheckpointWriter, read_jsonl


class _FakeClient:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def generate(self, **_: Any) -> str:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return "Final Answer: A"


class _Progress:
    def __init__(self) -> None:
        self.count = 0

    def update(self, amount: int) -> None:
        self.count += amount


def _target(index: int) -> _GenerationTarget:
    return _GenerationTarget(
        key=("dataset", f"item-{index}", "capability", 1.0),
        base={
            "dataset": "dataset",
            "item_identity": f"item-{index}",
            "steering_capability": "capability",
            "alpha": 1.0,
            "answer_type": "choice",
            "gold_answer": "A",
        },
    )


def test_scheduler_keeps_multiple_requests_in_flight(tmp_path: Path) -> None:
    jobs = [
        _VLLMJob(
            messages=[],
            vector_path=f"/vectors/{index}.pt",
            alpha=1.0,
            targets=(_target(index), _target(100)) if index == 0 else (_target(index),),
        )
        for index in range(6)
    ]
    client = _FakeClient()
    progress = _Progress()
    completed: set[tuple[str, str, str, float]] = set()
    destination = tmp_path / "rows.jsonl"

    with JsonlCheckpointWriter(destination, sync_every=3) as writer:
        asyncio.run(
            _execute_vllm_jobs(
                jobs,
                client=client,  # type: ignore[arg-type]
                target_layer=23,
                phase="both",
                positions="last_token",
                max_new_tokens=128,
                concurrency=3,
                consecutive_error_limit=2,
                writer=writer,
                completed=completed,
                progress=progress,
            )
        )

    assert client.calls == 6
    assert client.max_active == 3
    assert progress.count == 7
    assert len(completed) == 7
    assert len(list(read_jsonl(destination))) == 7

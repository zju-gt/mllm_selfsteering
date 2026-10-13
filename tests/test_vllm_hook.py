import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_self_steering.models.vllm_hook import (
    AsyncVLLMHookClient,
    _signatures_differ,
    build_steer_config,
    build_vllm_messages,
    map_server_path,
)


def test_map_server_path_uses_linux_mount(tmp_path: Path) -> None:
    local_root = tmp_path / "outputs" / "vectors"
    vector = local_root / "capture" / "vllm_hook" / "vision_detail__steering.pt"
    vector.parent.mkdir(parents=True)
    vector.touch()

    assert map_server_path(
        vector,
        local_root=local_root,
        server_root="/workspace/outputs/vectors",
    ) == "/workspace/outputs/vectors/capture/vllm_hook/vision_detail__steering.pt"


def test_map_server_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        map_server_path(
            tmp_path / "elsewhere" / "vector.pt",
            local_root=tmp_path / "vectors",
            server_root="/workspace/vectors",
        )


def test_file_transport_builds_encoded_linux_file_url(tmp_path: Path) -> None:
    media_root = tmp_path / "data"
    image = media_root / "images" / "sample image.png"
    image.parent.mkdir(parents=True)
    image.touch()

    messages = build_vllm_messages(
        "What is shown?",
        [str(image)],
        answer_type="text",
        media_transport="file",
        local_media_root=media_root,
        server_media_root="/workspace/data",
        media_uuids=["abc123"],
    )

    image_part = messages[1]["content"][0]
    assert image_part["image_url"]["url"] == (
        "file:///workspace/data/images/sample%20image.png"
    )
    assert image_part["uuid"] == "abc123"


def test_steering_probe_detects_ignored_and_changed_logprobs() -> None:
    baseline = ("A", {"A": -0.1, "B": -2.0})
    assert not _signatures_differ(
        baseline,
        ("A", {"A": -0.1, "B": -2.0}),
        tolerance=1e-6,
    )
    assert _signatures_differ(
        baseline,
        ("A", {"A": -0.2, "B": -1.9}),
        tolerance=1e-6,
    )


def test_legacy_protocol_maps_current_semantics() -> None:
    both = build_steer_config(
        protocol="vllm_hook_v0_2",
        vector_path="/vectors/test.pt",
        target_layer=23,
        alpha=0.5,
        phase="both",
        positions="last_token",
    )
    assert both["apply_at_all_positions"] is True
    assert "phase" not in both
    assert "positions" not in both

    prefill = build_steer_config(
        protocol="vllm_hook_v0_2",
        vector_path="/vectors/test.pt",
        target_layer=23,
        alpha=0.5,
        phase="prefill",
        positions="last_token",
    )
    assert prefill["apply_at_all_positions"] is False


def test_legacy_protocol_rejects_unrepresentable_semantics() -> None:
    with pytest.raises(ValueError, match="last-token"):
        build_steer_config(
            protocol="vllm_hook_v0_2",
            vector_path="/vectors/test.pt",
            target_layer=23,
            alpha=1.0,
            phase="both",
            positions="all_tokens",
        )
    with pytest.raises(ValueError, match="decode-only"):
        build_steer_config(
            protocol="vllm_hook_v0_2",
            vector_path="/vectors/test.pt",
            target_layer=23,
            alpha=1.0,
            phase="decode",
            positions="last_token",
        )


class _FakeCompletions:
    def __init__(self, logprobs: list[float]) -> None:
        self.logprobs = iter(logprobs)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        logprob = next(self.logprobs)
        entry = SimpleNamespace(
            token="A",
            top_logprobs=[SimpleNamespace(token="A", logprob=logprob)],
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(logprobs=SimpleNamespace(content=[entry]))
            ]
        )


def _probe_client(logprobs: list[float]) -> tuple[AsyncVLLMHookClient, _FakeCompletions]:
    completions = _FakeCompletions(logprobs)
    client = AsyncVLLMHookClient.__new__(AsyncVLLMHookClient)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client._model_name = "model"
    client._protocol = "mia"
    return client, completions


def test_effect_probe_fails_when_server_ignores_steering() -> None:
    client, completions = _probe_client([-0.1, -0.1])
    with pytest.raises(RuntimeError, match="identical"):
        asyncio.run(
            client.verify_steering_effect(
                messages=[],
                vector_path="/vectors/test.pt",
                target_layer=23,
                alpha=1.0,
                phase="both",
                positions="last_token",
                top_logprobs=20,
                tolerance=1e-6,
            )
        )

    assert completions.calls[0]["max_tokens"] == 1
    assert completions.calls[0]["extra_body"] == {}
    assert "steer" in completions.calls[1]["extra_body"]["vllm_xargs"]


def test_effect_probe_accepts_changed_logprobs() -> None:
    client, _ = _probe_client([-0.1, -0.2])
    asyncio.run(
        client.verify_steering_effect(
            messages=[],
            vector_path="/vectors/test.pt",
            target_layer=23,
            alpha=1.0,
            phase="both",
            positions="last_token",
            top_logprobs=20,
            tolerance=1e-6,
        )
    )


def test_effect_probe_uses_legacy_payload() -> None:
    client, completions = _probe_client([-0.1, -0.2])
    client._protocol = "vllm_hook_v0_2"
    asyncio.run(
        client.verify_steering_effect(
            messages=[],
            vector_path="/vectors/test.pt",
            target_layer=23,
            alpha=1.0,
            phase="both",
            positions="last_token",
            top_logprobs=20,
            tolerance=1e-6,
        )
    )
    encoded = completions.calls[1]["extra_body"]["vllm_xargs"]["steer"]
    steer = json.loads(encoded)
    assert steer["apply_at_all_positions"] is True
    assert "phase" not in steer

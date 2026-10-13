"""Asynchronous OpenAI-compatible client for vLLM-Hook/MIA steering."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from urllib.parse import quote

from mllm_self_steering.models.inputs import build_messages, image_data_url


VLLM_HOOK_PROTOCOLS = {"mia", "vllm_hook_v0_2"}


def build_steer_config(
    *,
    protocol: str,
    vector_path: str,
    target_layer: int,
    alpha: float,
    phase: str,
    positions: str,
) -> dict[str, Any]:
    """Build the per-request steering payload for a supported server protocol."""

    steer: dict[str, Any] = {
        "method": "add_vector",
        "coefficient": float(alpha),
        "optimal_layer": int(target_layer),
        "vector_path": vector_path,
    }
    if protocol == "mia":
        steer.update({"phase": phase, "positions": positions})
        return steer
    if protocol == "vllm_hook_v0_2":
        if positions != "last_token":
            raise ValueError(
                "vLLM-Hook v0.2 only supports last-token steering"
            )
        if phase not in {"both", "prefill"}:
            raise ValueError(
                "vLLM-Hook v0.2 supports phase=both or phase=prefill, not decode-only"
            )
        # In v0.2 the worker always changes the last row of each request. True
        # applies it to the final prompt token and every one-token decode step;
        # False applies it only to the final prompt token.
        steer["apply_at_all_positions"] = phase == "both"
        return steer
    raise ValueError(f"unsupported vLLM-Hook protocol: {protocol}")


def map_server_path(
    local_path: Path | str,
    *,
    local_root: Path | str,
    server_root: str | None,
) -> str:
    """Map a client path below ``local_root`` to an optional Linux server root."""

    path = Path(local_path).resolve()
    root = Path(local_root).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside configured local root: {path} (root: {root})") from exc
    if server_root is None:
        return str(path)
    destination = PurePosixPath(server_root)
    if not destination.is_absolute():
        raise ValueError("server root must be an absolute Linux path")
    return str(destination.joinpath(*relative.parts))


def _linux_file_uri(path: str) -> str:
    value = PurePosixPath(path)
    if not value.is_absolute():
        raise ValueError("file media path must be absolute")
    return "file://" + quote(str(value), safe="/")


def build_vllm_messages(
    prompt: str,
    image_paths: Sequence[str],
    *,
    answer_type: str,
    media_transport: str,
    local_media_root: Path,
    server_media_root: str | None,
    media_uuids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a request once, using file URLs on Linux or data URLs as fallback."""

    if media_transport == "data_url":
        urls = [image_data_url(path) for path in image_paths]
    elif media_transport == "file":
        mapped = [
            map_server_path(
                path,
                local_root=local_media_root,
                server_root=server_media_root,
            )
            for path in image_paths
        ]
        urls = (
            [_linux_file_uri(path) for path in mapped]
            if server_media_root is not None
            else [Path(path).as_uri() for path in mapped]
        )
    else:
        raise ValueError(f"unsupported media transport: {media_transport}")
    return build_messages(
        prompt,
        image_paths,
        answer_type=answer_type,
        openai_format=True,
        openai_media_urls=urls,
        openai_media_uuids=media_uuids,
    )


def _first_token_signature(response: Any) -> tuple[str, dict[str, float]]:
    """Return a stable first-token signature from an OpenAI chat response."""

    try:
        entry = response.choices[0].logprobs.content[0]
        token = str(entry.token)
        top = {str(value.token): float(value.logprob) for value in entry.top_logprobs}
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "vLLM steering verification requires first-token logprobs; ensure the "
            "server supports chat logprobs or set vllm_hook.verify_steering_effect=false"
        ) from exc
    if not top:
        raise RuntimeError("vLLM returned no top logprobs for steering verification")
    return token, top


def _signatures_differ(
    baseline: tuple[str, dict[str, float]],
    steered: tuple[str, dict[str, float]],
    *,
    tolerance: float,
) -> bool:
    if baseline[0] != steered[0] or baseline[1].keys() != steered[1].keys():
        return True
    return any(
        abs(baseline[1][token] - steered[1][token]) > tolerance
        for token in baseline[1]
    )


class AsyncVLLMHookClient:
    """Persistent async client that lets vLLM continuously batch requests."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout_seconds: float,
        max_retries: int,
        protocol: str = "mia",
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for the vllm_hook client") from exc
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self._model_name = model_name
        if protocol not in VLLM_HOOK_PROTOCOLS:
            raise ValueError(f"unsupported vLLM-Hook protocol: {protocol}")
        self._protocol = protocol

    async def preflight(self) -> None:
        """Fail before a large run when the endpoint or served model is wrong."""

        response = await self._client.models.list()
        available = {str(model.id) for model in response.data}
        if self._model_name not in available:
            shown = ", ".join(sorted(available)) or "<none>"
            raise RuntimeError(
                f"vLLM does not expose model {self._model_name!r}; available: {shown}"
            )

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        vector_path: str | None,
        target_layer: int,
        alpha: float,
        phase: str,
        positions: str,
        max_new_tokens: int,
    ) -> str:
        extra_body: dict[str, Any] = {}
        if vector_path is not None and float(alpha) != 0.0:
            steer = build_steer_config(
                protocol=self._protocol,
                vector_path=vector_path,
                target_layer=target_layer,
                alpha=alpha,
                phase=phase,
                positions=positions,
            )
            extra_body = {"vllm_xargs": {"steer": json.dumps(steer)}}
        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=int(max_new_tokens),
            temperature=0.0,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("vLLM server returned no text content")
        return content

    async def verify_steering_effect(
        self,
        *,
        messages: list[dict[str, Any]],
        vector_path: str,
        target_layer: int,
        alpha: float,
        phase: str,
        positions: str,
        top_logprobs: int,
        tolerance: float,
    ) -> None:
        """Reject a server that accepts MIA arguments but appears to ignore them."""

        async def request(vector: str | None, coefficient: float) -> Any:
            extra_body: dict[str, Any] = {}
            if vector is not None:
                steer = build_steer_config(
                    protocol=self._protocol,
                    vector_path=vector,
                    target_layer=target_layer,
                    alpha=coefficient,
                    phase=phase,
                    positions=positions,
                )
                extra_body = {"vllm_xargs": {"steer": json.dumps(steer)}}
            return await self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=int(top_logprobs),
                extra_body=extra_body,
            )

        baseline = _first_token_signature(await request(None, 0.0))
        steered = _first_token_signature(await request(vector_path, alpha))
        if not _signatures_differ(baseline, steered, tolerance=float(tolerance)):
            raise RuntimeError(
                "steering verification found identical baseline and steered first-token "
                "logprobs; the MIA worker may be inactive or the configured decoder layer "
                "may not match the served model. Check the vLLM-Hook server log."
            )

    async def close(self) -> None:
        await self._client.close()

"""Build equivalent Transformers and OpenAI-compatible multimodal prompts."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


SYSTEM_PROMPT = "You are a careful multimodal reasoning assistant."


def answer_instruction(answer_type: str) -> str:
    if answer_type == "choice":
        return "End with `Final Answer: X`, where X is only the uppercase option letter."
    return "End with `Final Answer: <answer>` and keep the final answer concise."


def user_text(prompt: str, answer_type: str | None = None) -> str:
    instruction = answer_instruction(answer_type) if answer_type else ""
    return f"{prompt.strip()}\n{instruction}".strip()


def build_messages(
    prompt: str,
    image_paths: Sequence[str],
    *,
    answer_type: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    openai_format: bool = False,
    openai_media_urls: Sequence[str] | None = None,
    openai_media_uuids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if openai_media_urls is not None:
        if not openai_format:
            raise ValueError("openai_media_urls requires openai_format=True")
        if len(openai_media_urls) != len(image_paths):
            raise ValueError("openai_media_urls must match image_paths")
    if openai_media_uuids is not None:
        if not openai_format:
            raise ValueError("openai_media_uuids requires openai_format=True")
        if len(openai_media_uuids) != len(image_paths):
            raise ValueError("openai_media_uuids must match image_paths")
    content: list[dict[str, Any]] = []
    for index, raw_path in enumerate(image_paths):
        if openai_format:
            url = (
                str(openai_media_urls[index])
                if openai_media_urls is not None
                else image_data_url(raw_path)
            )
            image_part: dict[str, Any] = {
                "type": "image_url",
                "image_url": {"url": url},
            }
            if openai_media_uuids is not None:
                image_part["uuid"] = str(openai_media_uuids[index])
            content.append(image_part)
        else:
            content.append({"type": "image", "image": str(Path(raw_path).resolve())})
    content.append({"type": "text", "text": user_text(prompt, answer_type)})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def prepare_transformers_inputs(
    processor: Any,
    prompt: str,
    image_paths: Sequence[str],
    *,
    answer_type: str | None = None,
) -> dict[str, Any]:
    messages = build_messages(prompt, image_paths, answer_type=answer_type)
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images = []
    for path in image_paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    kwargs: dict[str, Any] = {
        "text": [rendered],
        "padding": True,
        "return_tensors": "pt",
    }
    if images:
        kwargs["images"] = images
    return dict(processor(**kwargs))


def image_data_url(path: str | Path) -> str:
    source = Path(path)
    suffix = source.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"

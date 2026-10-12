"""Adapters for MMStar, MMMU, MathVista, and MathVision."""

from __future__ import annotations

import ast
import io
from pathlib import Path
import re
from string import ascii_uppercase
from typing import Any, Mapping, Sequence

from PIL import Image

from mllm_self_steering.datasets.types import MultimodalItem


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except (SyntaxError, ValueError):
                pass
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _choice_map(raw: Any) -> dict[str, str] | None:
    values = _as_sequence(raw)
    values = [value for value in values if value is not None and str(value).strip()]
    if not values:
        return None
    return {ascii_uppercase[index]: str(value) for index, value in enumerate(values)}


def _choices_from_inline_question(question: str) -> dict[str, str] | None:
    marker = re.search(r"(?is)\b(?:options|choices)\s*:\s*", question)
    tail = question[marker.end() :] if marker else question
    pattern = (
        r"(?i)(?:^|[,;]\s*|\s+)(?:\(([A-Z])\)|([A-Z])\s*[:.)])\s*"
        if marker
        else r"(?im)(?:^|\n)\s*(?:\(([A-Z])\)|([A-Z])\s*[:.)])\s*"
    )
    matches = list(re.finditer(pattern, tail))
    if not matches:
        return None
    choices: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(tail)
        label = match.group(1) or match.group(2)
        choices[label.upper()] = tail[match.end() : end].strip(" ,;")
    return choices or None


def _render_choices(question: str, choices: dict[str, str] | None) -> str:
    if (
        not choices
        or re.search(r"(?i)\b(?:options|choices)\s*:", question)
        or _choices_from_inline_question(question)
    ):
        return question.strip()
    rendered = "\n".join(f"{label}. {text}" for label, text in choices.items())
    return f"{question.strip()}\n\nOptions:\n{rendered}"


def _choice_answer(answer: str, choices: dict[str, str] | None) -> str | None:
    if not choices:
        return None
    cleaned = answer.strip()
    label_match = re.fullmatch(r"\(?([A-Za-z])\)?", cleaned)
    if label_match and label_match.group(1).upper() in choices:
        return label_match.group(1).upper()
    normalized = re.sub(r"\s+", " ", cleaned).strip().casefold()
    for label, value in choices.items():
        if re.sub(r"\s+", " ", value).strip().casefold() == normalized:
            return label
    return None


def _save_image(value: Any, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return str(destination.resolve())
    image: Image.Image
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, Mapping) and value.get("bytes") is not None:
        image = Image.open(io.BytesIO(value["bytes"]))
    elif isinstance(value, Mapping) and value.get("path"):
        image = Image.open(str(value["path"]))
    elif isinstance(value, (str, Path)) and Path(value).is_file():
        image = Image.open(value)
    else:
        raise ValueError(f"unsupported or unavailable image value: {type(value).__name__}")
    image.convert("RGB").save(destination, format="PNG")
    return str(destination.resolve())


def _materialize_images(
    values: Sequence[Any], *, image_root: Path, item_id: str
) -> list[str]:
    paths = []
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id)
    for index, value in enumerate(values):
        if value is None:
            continue
        paths.append(_save_image(value, image_root / f"{safe_id}_{index}.png"))
    if not paths:
        raise ValueError(f"no decodable images for {item_id}")
    return paths


def adapt_mmstar(
    row: Mapping[str, Any], *, split: str, index: int, image_root: Path
) -> MultimodalItem:
    item_id = f"mmstar_{row.get('index', index)}"
    question = str(row["question"])
    choices = _choices_from_inline_question(question)
    item = MultimodalItem(
        item_id=item_id,
        dataset="mmstar",
        split=split,
        prompt=question,
        image_paths=_materialize_images([row.get("image")], image_root=image_root, item_id=item_id),
        gold_answer=str(row["answer"]).strip().upper(),
        answer_type="choice",
        choices=choices,
        metadata={
            "category": row.get("category"),
            "l2_category": row.get("l2_category"),
            "source_split": row.get("meta_info", {}).get("split")
            if isinstance(row.get("meta_info"), Mapping)
            else None,
        },
    )
    item.validate()
    return item


def adapt_mmmu(
    row: Mapping[str, Any], *, split: str, index: int, image_root: Path
) -> MultimodalItem:
    source_id = row.get("id", index)
    subject = str(row.get("_subject") or "unknown")
    item_id = f"mmmu_{subject}_{source_id}"
    choices = _choice_map(row.get("options"))
    answer = str(row["answer"]).strip()
    choice_answer = _choice_answer(answer, choices)
    answer_type = "choice" if choice_answer else "text"
    images = [row.get(f"image_{number}") for number in range(1, 8)]
    item = MultimodalItem(
        item_id=item_id,
        dataset="mmmu",
        split=split,
        prompt=_render_choices(str(row["question"]), choices),
        image_paths=_materialize_images(images, image_root=image_root, item_id=item_id),
        gold_answer=choice_answer or answer,
        answer_type=answer_type,
        choices=choices,
        metadata={
            "subject": subject,
            "subfield": row.get("subfield"),
        },
    )
    item.validate()
    return item


def adapt_mathvista(
    row: Mapping[str, Any], *, split: str, index: int, image_root: Path
) -> MultimodalItem:
    source_id = row.get("pid", index)
    item_id = f"mathvista_{source_id}"
    choices = _choice_map(row.get("choices"))
    answer = str(row["answer"]).strip()
    choice_answer = _choice_answer(answer, choices)
    upstream_type = str(row.get("answer_type", "")).casefold()
    answer_type = (
        "choice"
        if choice_answer
        else "text"
        if upstream_type == "text"
        else "math"
    )
    prompt = str(row.get("query") or row.get("question") or "")
    image = row.get("decoded_image") or row.get("image")
    item = MultimodalItem(
        item_id=item_id,
        dataset="mathvista",
        split=split,
        prompt=_render_choices(prompt, choices),
        image_paths=_materialize_images([image], image_root=image_root, item_id=item_id),
        gold_answer=choice_answer or answer,
        answer_type=answer_type,
        choices=choices if answer_type == "choice" else None,
        metadata={
            key: row.get(key)
            for key in ("question_type", "answer_type", "task", "skills", "source")
            if row.get(key) is not None
        },
    )
    item.validate()
    return item


def adapt_mathvision(
    row: Mapping[str, Any], *, split: str, index: int, image_root: Path
) -> MultimodalItem:
    source_id = row.get("id", index)
    item_id = f"mathvision_{source_id}"
    choices = _choice_map(row.get("options"))
    answer = str(row["answer"]).strip()
    choice_answer = _choice_answer(answer, choices)
    answer_type = "choice" if choice_answer else "math"
    image = row.get("decoded_image") or row.get("image")
    item = MultimodalItem(
        item_id=item_id,
        dataset="mathvision",
        split=split,
        prompt=_render_choices(str(row["question"]), choices),
        image_paths=_materialize_images([image], image_root=image_root, item_id=item_id),
        gold_answer=choice_answer or answer,
        answer_type=answer_type,
        choices=choices if answer_type == "choice" else None,
        metadata={key: row.get(key) for key in ("level", "subject") if row.get(key) is not None},
    )
    item.validate()
    return item


ADAPTERS = {
    "mmstar": adapt_mmstar,
    "mmmu": adapt_mmmu,
    "mathvista": adapt_mathvista,
    "mathvision": adapt_mathvision,
}

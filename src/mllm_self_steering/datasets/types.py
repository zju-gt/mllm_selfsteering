"""Canonical multimodal benchmark and contrast-pair records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class MultimodalItem:
    item_id: str
    dataset: str
    split: str
    prompt: str
    image_paths: list[str]
    gold_answer: str
    answer_type: Literal["choice", "math", "text"]
    choices: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.item_id.strip() or not self.dataset.strip() or not self.prompt.strip():
            raise ValueError("item_id, dataset, and prompt must be non-empty")
        if not self.image_paths:
            raise ValueError(f"multimodal item has no images: {self.item_id}")
        if not self.gold_answer.strip():
            raise ValueError(f"gold answer is empty: {self.item_id}")
        if self.answer_type not in {"choice", "math", "text"}:
            raise ValueError(f"unsupported answer type: {self.answer_type}")
        if self.answer_type == "choice":
            if not self.choices:
                raise ValueError(f"choice item has no choices: {self.item_id}")
            if self.gold_answer.upper() not in self.choices:
                raise ValueError(f"gold answer is absent from choices: {self.item_id}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "MultimodalItem":
        item = cls(
            item_id=str(row["item_id"]),
            dataset=str(row["dataset"]),
            split=str(row["split"]),
            prompt=str(row["prompt"]),
            image_paths=[str(value) for value in row["image_paths"]],
            gold_answer=str(row["gold_answer"]),
            answer_type=row["answer_type"],
            choices={str(k): str(v) for k, v in row["choices"].items()}
            if row.get("choices")
            else None,
            metadata=dict(row.get("metadata", {})),
        )
        item.validate()
        return item


@dataclass(frozen=True)
class ContrastPair:
    pair_id: str
    capability: str
    modality: Literal["text", "visual"]
    positive_prompt: str
    negative_prompt: str
    positive_image_paths: list[str]
    negative_image_paths: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.pair_id.strip() or not self.capability.strip():
            raise ValueError("pair_id and capability must be non-empty")
        if not self.positive_prompt.strip() or not self.negative_prompt.strip():
            raise ValueError(f"both prompts are required: {self.pair_id}")
        if self.modality == "text":
            if self.positive_image_paths != self.negative_image_paths:
                raise ValueError(
                    f"text pair must hold images fixed: {self.pair_id}"
                )
        elif self.modality == "visual":
            if self.positive_prompt != self.negative_prompt:
                raise ValueError(
                    f"visual pair must hold prompt fixed: {self.pair_id}"
                )
            if not self.positive_image_paths or not self.negative_image_paths:
                raise ValueError(f"visual pair requires images on both sides: {self.pair_id}")
        else:
            raise ValueError(f"unsupported pair modality: {self.modality!r}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls, row: dict[str, Any], *, capability: str, modality: str
    ) -> "ContrastPair":
        if modality == "text":
            shared = [str(value) for value in row.get("image_paths", [])]
            positive_prompt = str(row["positive_prompt"])
            negative_prompt = str(row["negative_prompt"])
            positive_images = shared
            negative_images = shared
        else:
            prompt = str(row["prompt"])
            positive_prompt = prompt
            negative_prompt = prompt
            positive_images = [str(value) for value in row["positive_image_paths"]]
            negative_images = [str(value) for value in row["negative_image_paths"]]
        pair = cls(
            pair_id=str(row["pair_id"]),
            capability=capability,
            modality=modality,  # type: ignore[arg-type]
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            positive_image_paths=positive_images,
            negative_image_paths=negative_images,
            metadata=dict(row.get("metadata", {})),
        )
        pair.validate()
        return pair


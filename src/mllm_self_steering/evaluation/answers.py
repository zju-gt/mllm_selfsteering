"""Deterministic answer extraction for multimodal benchmarks."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re


def extract_answer(text: str, answer_type: str) -> str | None:
    if not text or not text.strip():
        return None
    if answer_type == "choice":
        matches = re.findall(
            r"(?im)final\s+answer\s*:\s*(?:option\s*)?\(?([A-Z])\)?\s*[.)]?\s*$",
            text,
        )
        if matches:
            return matches[-1].upper()
        terminal = re.search(r"(?i)(?:^|\n)\s*\(?([A-Z])\)?\s*[.)]?\s*$", text)
        return terminal.group(1).upper() if terminal else None
    matches = re.findall(r"(?im)final\s+answer\s*:\s*(.+?)\s*$", text)
    if matches:
        return matches[-1].strip().rstrip(".")
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    return None


def normalize(answer: str) -> str:
    value = answer.strip().replace("$", "").replace(",", "")
    value = value.replace(r"\left", "").replace(r"\right", "")
    return re.sub(r"\s+", "", value).rstrip(".").casefold()


def is_correct(predicted: str | None, gold: str, *, answer_type: str) -> bool:
    if predicted is None:
        return False
    if answer_type == "choice":
        return predicted.strip().upper() == gold.strip().upper()
    left, right = normalize(predicted), normalize(gold)
    if left == right:
        return True
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


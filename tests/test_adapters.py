from pathlib import Path

from PIL import Image

from mllm_self_steering.datasets.adapters import adapt_mathvista, adapt_mmstar


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "source.png"
    Image.new("RGB", (2, 2)).save(path)
    return path


def test_mmstar_accepts_parenthesized_choices(tmp_path: Path) -> None:
    item = adapt_mmstar(
        {
            "index": 500,
            "question": "Question: age? Choices: (A) 4 (B) 2 (C) 3 (D) 1",
            "image": str(_image(tmp_path)),
            "answer": "A",
            "category": "reasoning",
            "l2_category": "attribute",
        },
        split="test",
        index=0,
        image_root=tmp_path / "images",
    )

    assert item.choices == {"A": "4", "B": "2", "C": "3", "D": "1"}


def test_mathvista_does_not_duplicate_preformatted_choices(tmp_path: Path) -> None:
    query = "Hint: answer with a letter. Question: Is it greater? Choices: (A) Yes (B) No"
    item = adapt_mathvista(
        {
            "pid": "957",
            "query": query,
            "question": "Is it greater?",
            "decoded_image": str(_image(tmp_path)),
            "choices": ["Yes", "No"],
            "answer": "Yes",
            "question_type": "multi_choice",
            "answer_type": "text",
        },
        split="testmini",
        index=0,
        image_root=tmp_path / "images",
    )

    assert item.prompt == query
    assert item.prompt.count("Choices:") == 1
    assert "Options:" not in item.prompt
    assert item.gold_answer == "A"


def test_mmstar_accepts_newline_parenthesized_choices_without_marker(
    tmp_path: Path,
) -> None:
    item = adapt_mmstar(
        {
            "index": 776,
            "question": "Question: What is the difference?\n(A) 54\n(B) 52\n(C) 56\n(D) 57",
            "image": str(_image(tmp_path)),
            "answer": "A",
            "category": "logical reasoning",
            "l2_category": "diagram reasoning",
        },
        split="test",
        index=0,
        image_root=tmp_path / "images",
    )

    assert item.choices == {"A": "54", "B": "52", "C": "56", "D": "57"}
    assert item.prompt.count("(A) 54") == 1

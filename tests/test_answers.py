from mllm_self_steering.evaluation.answers import extract_answer, is_correct


def test_choice_and_numeric_answers() -> None:
    assert extract_answer("reasoning\nFinal Answer: C", "choice") == "C"
    assert is_correct("1,000", "1000", answer_type="math")


def test_choice_answer_accepts_markdown_bold() -> None:
    assert extract_answer("reasoning\n**Final Answer: B**", "choice") == "B"
    assert extract_answer("reasoning\nFinal Answer: **C**", "choice") == "C"
    assert extract_answer("Final Answer: A\n**Final Answer: D**", "choice") == "D"
    assert extract_answer("Final Answer: None of the above", "choice") is None


def test_numeric_answer_accepts_markdown_bold() -> None:
    assert extract_answer("**Final Answer: 1,000**", "math") == "1,000"
    assert extract_answer("Final Answer: **42**", "math") == "42"


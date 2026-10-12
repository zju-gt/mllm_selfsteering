from mllm_self_steering.evaluation.answers import extract_answer, is_correct


def test_choice_and_numeric_answers() -> None:
    assert extract_answer("reasoning\nFinal Answer: C", "choice") == "C"
    assert is_correct("1,000", "1000", answer_type="math")


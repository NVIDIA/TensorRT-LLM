from tensorrt_llm.scaffolding.math_utils import (
    extract_answer_from_boxed,
    get_digit_majority_vote_result,
    get_majority_result,
)


def test_extract_answer_from_boxed_basic():
    assert extract_answer_from_boxed("The answer is \\boxed{42}") == "42"
    assert extract_answer_from_boxed("\\boxed{0}") == "0"
    assert extract_answer_from_boxed("\\boxed{123456}") == "123456"


def test_extract_answer_from_boxed_negative():
    assert extract_answer_from_boxed("The answer is \\boxed{-50}") == "-50"
    assert extract_answer_from_boxed("\\boxed{-1}") == "-1"
    assert extract_answer_from_boxed("\\boxed{-999}") == "-999"


def test_extract_answer_from_boxed_nested_braces():
    assert extract_answer_from_boxed("\\boxed{2^{10}}") == "2^{10}"


def test_extract_answer_from_boxed_no_boxed():
    assert extract_answer_from_boxed("no boxed here") is None
    assert extract_answer_from_boxed("") is None


def test_get_majority_result_basic():
    index, answer = get_majority_result(["a", "a", "b"])
    assert answer == "a"

    index, answer = get_majority_result(["x", "y", "y"])
    assert answer == "y"


def test_get_majority_result_all_invalid():
    index, answer = get_majority_result(
        [None, None, None],
        result_validator=lambda x: x is not None,
    )
    assert index is None
    assert answer is None


def test_get_majority_result_empty():
    index, answer = get_majority_result([])
    assert index is None
    assert answer is None


def test_get_digit_majority_vote_result_positive():
    results = [
        "The answer is \\boxed{42}",
        "So we get \\boxed{42}",
        "Therefore \\boxed{42}",
    ]
    index, answer = get_digit_majority_vote_result(results)
    assert answer == "42"


def test_get_digit_majority_vote_result_positive_majority():
    results = [
        "\\boxed{42}",
        "\\boxed{42}",
        "\\boxed{99}",
    ]
    _, answer = get_digit_majority_vote_result(results)
    assert answer == "42"


def test_get_digit_majority_vote_result_negative():
    results = [
        "The answer is \\boxed{-50}",
        "So we get \\boxed{-50}",
        "Therefore \\boxed{-50}",
    ]
    index, answer = get_digit_majority_vote_result(results)
    assert answer == "-50"


def test_get_digit_majority_vote_result_negative_majority():
    results = [
        "\\boxed{-50}",
        "\\boxed{-50}",
        "\\boxed{-99}",
    ]
    _, answer = get_digit_majority_vote_result(results)
    assert answer == "-50"


def test_get_digit_majority_vote_result_mixed_sign_majority():
    results = [
        "\\boxed{-7}",
        "\\boxed{-7}",
        "\\boxed{7}",
    ]
    _, answer = get_digit_majority_vote_result(results)
    assert answer == "-7"


def test_get_digit_majority_vote_result_zero():
    results = [
        "\\boxed{0}",
        "\\boxed{0}",
        "\\boxed{1}",
    ]
    _, answer = get_digit_majority_vote_result(results)
    assert answer == "0"


def test_get_digit_majority_vote_result_invalid():
    results = [
        "\\boxed{abc}",
        "\\boxed{abc}",
        "\\boxed{abc}",
    ]
    index, answer = get_digit_majority_vote_result(results)
    assert index == 0
    assert answer is None


def test_get_digit_majority_vote_result_no_boxed():
    results = [
        "no boxed here",
        "also no boxed",
        "still nothing",
    ]
    index, answer = get_digit_majority_vote_result(results)
    assert index == 0
    assert answer is None


def test_get_digit_majority_vote_result_mixed_valid_invalid():
    results = [
        "\\boxed{-50}",
        "\\boxed{-50}",
        "\\boxed{abc}",
    ]
    _, answer = get_digit_majority_vote_result(results)
    assert answer == "-50"


def test_get_digit_majority_vote_result_decimal_rejected():
    results = [
        "\\boxed{-3.14}",
        "\\boxed{-3.14}",
        "\\boxed{-3.14}",
    ]
    index, answer = get_digit_majority_vote_result(results)
    assert index == 0
    assert answer is None

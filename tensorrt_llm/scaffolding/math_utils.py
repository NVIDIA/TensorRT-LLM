import re
from collections import Counter
from typing import List


def extract_answer_with_regex(string: str,
                              extract_regex: str = r"The final answer is (.+)$"
                              ):
    match = re.search(extract_regex, string)
    if match:
        return match.group(1)
    return None


def extract_answer_from_boxed(string: str):
    """Extract Answer String from \\boxed expression or based on regex"""

    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[:len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left):-1]
        except AssertionError:
            return None

    return None


def get_majority_result(
    results: list,
    result_extractor=lambda x: x,
    result_validator=lambda x: True,
):
    valid_answers_and_results = [(result, result_extractor(result))
                                 for result in results
                                 if result_validator(result) is True
                                 and result_extractor(result) is not None]
    if len(valid_answers_and_results) == 0:
        return None, None

    majority_result = Counter(valid_answers_and_results).most_common(1)[0][0]
    # return result and extracted result
    return majority_result[0], majority_result[1]


def get_digit_majority_vote_result(results: List[str]) -> str:

    def is_digit(result: str):
        extracted_answer = extract_answer_from_boxed(result)
        if extracted_answer is None:
            return False
        return extracted_answer.isdigit()

    vote_result = get_majority_result(
        results,
        result_extractor=extract_answer_from_boxed,
        result_validator=is_digit)[0]
    return vote_result if vote_result else results[0]

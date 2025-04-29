# This code is adapted from the Qwen2.5-Math project by the QwenLM team
# Original source: https://github.com/QwenLM/Qwen2.5-Math
# Thank you to the original authors for their valuable contribution
"""
Mathematical Expression Evaluator Module

This module provides functionality for evaluating and comparing mathematical expressions,
particularly for assessing the correctness of model-generated answers to mathematical problems.
It handles various formats of mathematical expressions, including LaTeX notation, and provides
methods to normalize, parse, and compare these expressions.
"""

import multiprocessing
import re
from math import isclose
from typing import Any, Callable, List, Optional, Union

import regex
from latex2sympy2 import latex2sympy
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from word2number import w2n


def _fix_fracs(string: str) -> str:
    """
    Fix fraction notation in LaTeX strings.

    Converts improper fraction notation (e.g., \frac12) to proper notation (e.g., \frac{1}{2}).

    Args:
        string: A LaTeX string that may contain improper fraction notation.

    Returns:
        A string with proper fraction notation.
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    """
    Convert simple fraction notation (a/b) to LaTeX fraction notation (\frac{a}{b}).

    Args:
        string: A string that may contain simple fraction notation.

    Returns:
        A string with LaTeX fraction notation.
    """
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string: str) -> str:
    """
    Fix square root notation in LaTeX strings.

    Ensures proper braces around the argument of square root.

    Args:
        string: A LaTeX string that may contain improper square root notation.

    Returns:
        A string with proper square root notation.
    """
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    """
    Convert word representations of numbers to their numerical form.

    Args:
        text: A string that may contain word representations of numbers.

    Returns:
        A string with numerical representations of numbers.
    """
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text


# units mainly from MathQA
unit_texts: List[str] = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m square",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def strip_string(string: str, skip_unit: bool = False) -> str:
    """
    Clean and normalize a mathematical expression string.

    Performs various cleaning operations on a string containing a mathematical expression,
    including removing whitespace, normalizing LaTeX notation, and handling special cases.

    Args:
        string: The input string containing a mathematical expression.
        skip_unit: If True, skip removing unit text from the string.

    Returns:
        A cleaned and normalized string.
    """
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (string.replace("\\neq",
                             "\\ne").replace("\\leq",
                                             "\\le").replace("\\geq", "\\ge"))

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2",
                                 string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in [
            "x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to",
            "z\\to"
    ]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (string.startswith("{") and string.endswith("}") and string.isalnum()
            or string.startswith("(") and string.endswith(")")
            and string.isalnum() or string.startswith("[")
            and string.endswith("]") and string.isalnum()):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_answer(pred_str: str,
                   data_name: str,
                   use_last_number: bool = True) -> str:
    """
    Extract the answer from a model's prediction string.

    This function handles various formats of answer extraction, including:
    - Boxed answers
    - Final answer markers
    - Chinese answer markers
    - Last number in the text

    Args:
        pred_str: The prediction string from the model.
        data_name: The name of the dataset being evaluated.
        use_last_number: If True, extract the last number in the text if no other format is found.

    Returns:
        The extracted answer as a string.
    """
    pred_str = pred_str.replace("\u043a\u0438", "")

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # choice answer
    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred,
                        skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


def extract_first_boxed_answer(pred_str: str, data_name: str) -> str:
    """
    Extract the first boxed answer from a model's prediction string.

    Args:
        pred_str: The prediction string from the model.
        data_name: The name of the dataset being evaluated.

    Returns:
        The first boxed answer as a string.
    """
    pred_str = pred_str.replace("\u043a\u0438", "")

    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    else:
        pred = ""

    # choice answer
    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred,
                        skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


def extract_boxed_answer(pred_str: str, data_name: str) -> str:
    """
    Extract the last boxed answer from a model's prediction string.

    Args:
        pred_str: The prediction string from the model.
        data_name: The name of the dataset being evaluated.

    Returns:
        The last boxed answer as a string.
    """
    pred_str = pred_str.replace("\u043a\u0438", "")

    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    else:
        pred = ""

    # choice answer
    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred,
                        skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


# from .parser import choice_answer_clean, strip_string
# from parser import choice_answer_clean


def choice_answer_clean(pred: str) -> str:
    """
    Clean a multiple choice answer.

    Extracts and standardizes multiple choice answers (A, B, C, D, E).

    Args:
        pred: The prediction string that may contain a multiple choice answer.

    Returns:
        A cleaned multiple choice answer.
    """
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num: str) -> Optional[float]:
    """
    Parse a string containing digits into a float.

    Handles various formats including percentages and comma-separated numbers.

    Args:
        num: A string that may contain digits.

    Returns:
        A float if the string can be parsed, None otherwise.
    """
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num: str) -> bool:
    """
    Check if a string can be parsed as a number.

    Args:
        num: A string that may contain digits.

    Returns:
        True if the string can be parsed as a number, False otherwise.
    """
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str: str) -> str:
    """
    Convert a string representation of a matrix to LaTeX pmatrix format.

    Args:
        input_str: A string containing a matrix representation.

    Returns:
        A string with LaTeX pmatrix notation.
    """
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Determine if two mathematical expressions are equal.

    This function performs a comprehensive comparison of mathematical expressions,
    handling various formats and special cases. It considers expressions equal if:
    1. They are numerically equal (both can be converted to float and are equal)
    2. They are symbolically equal (both can be converted to sympy expressions and are equal)
    3. They are multiple choice answers that match
    4. They are matrices with equal elements
    5. They are equations with equal sides

    Args:
        prediction: The predicted answer.
        reference: The reference answer.
        include_percentage: If True, consider percentage variations of the reference.
        is_close: If True, use numerical closeness for comparison.
        timeout: If True, use a timeout for symbolic comparison.

    Returns:
        True if the expressions are considered equal, False otherwise.
    """
    # print("Judge:", prediction, reference)
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (reference in ["A", "B", "C", "D", "E"]
            and choice_answer_clean(prediction) == reference):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]")
            and not reference.startswith("(")) or (
                prediction.startswith("(") and prediction.endswith(")")
                and not reference.startswith("[")):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
            and regex.match(r"(\(|\[).+(\)|\])", reference) is not None):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all([
                    math_equal(pred_parts[i], ref_parts[i], include_percentage,
                               is_close) for i in range(len(pred_parts))
            ]):
                return True
    if ((prediction.startswith("\\begin{pmatrix}")
         or prediction.startswith("\\begin{bmatrix}"))
            and (prediction.endswith("\\end{pmatrix}")
                 or prediction.endswith("\\end{bmatrix}"))
            and (reference.startswith("\\begin{pmatrix}")
                 or reference.startswith("\\begin{bmatrix}"))
            and (reference.endswith("\\end{pmatrix}")
                 or reference.endswith("\\end{bmatrix}"))):
        pred_lines = [
            line.strip()
            for line in prediction[len("\\begin{pmatrix}"
                                       ):-len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len("\\begin{pmatrix}"
                                      ):-len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all([
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            ) for i in range(len(pred_parts))
                    ]):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (prediction.count("=") == 1
          and len(prediction.split("=")[0].strip()) <= 2
          and "=" not in reference):
        if math_equal(
                prediction.split("=")[1], reference, include_percentage,
                is_close):
            return True
    elif (reference.count("=") == 1
          and len(reference.split("=")[0].strip()) <= 2
          and "=" not in prediction):
        if math_equal(prediction,
                      reference.split("=")[1], include_percentage, is_close):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process,
                             prediction,
                             reference,
                             timeout=timeout):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def count_not_empty(answers: List[str]) -> int:
    """
    Count the number of non-empty answers in a list.

    Args:
        answers: A list of answer strings.

    Returns:
        The number of non-empty answers.
    """
    return sum(1 for answer in answers if answer != "")


def equal_group(answers: List[str]) -> bool:
    """
    Check if all answers in a group are equal.

    Args:
        answers: A list of answer strings.

    Returns:
        True if all answers are equal, False otherwise.
    """
    equiv_classes = []

    for answer in answers:
        flag = 0
        for i, rep in enumerate(equiv_classes):
            if math_equal(answer, rep):
                flag = 1
                break
        if flag:
            continue
        equiv_classes.append(answer)

    return len(equiv_classes) == 1


def math_equal_process(param: List[Any]) -> bool:
    """
    Process function for parallel execution of math_equal.

    Args:
        param: A list containing the parameters for math_equal.

    Returns:
        The result of math_equal.
    """
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float) -> bool:
    """
    Check if two numbers are numerically equal within a tolerance.

    Args:
        prediction: The predicted number.
        reference: The reference number.

    Returns:
        True if the numbers are equal within a tolerance, False otherwise.
    """
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a: str, b: str) -> bool:
    """
    Check if two mathematical expressions are symbolically equal.

    This function attempts to parse the expressions using various methods and
    compares them using sympy's symbolic equality.

    Args:
        a: The first mathematical expression.
        b: The second mathematical expression.

    Returns:
        True if the expressions are symbolically equal, False otherwise.
    """

    def _parse(s: str) -> Any:
        """
        Parse a string into a sympy expression.

        Args:
            s: A string containing a mathematical expression.

        Returns:
            A sympy expression if parsing is successful, the original string otherwise.
        """
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a: str, b: str,
                           output_queue: multiprocessing.Queue) -> None:
    """
    Process function for parallel execution of symbolic_equal.

    Args:
        a: The first mathematical expression.
        b: The second mathematical expression.
        output_queue: A multiprocessing Queue to store the result.
    """
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func: Callable,
                      *args: Any,
                      timeout: int = 1,
                      **kwargs: Any) -> bool:
    """
    Call a function with a timeout.

    Args:
        func: The function to call.
        *args: Positional arguments for the function.
        timeout: The timeout in seconds.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the function call, or False if the call times out.
    """
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue, )
    process = multiprocessing.Process(target=func,
                                      args=process_args,
                                      kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()

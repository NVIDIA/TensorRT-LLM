# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the MMMU multiple-choice answer parser (HUMAN FEEDBACK #5 Directive 2).

Pins the LaTeX/markdown-wrapped ``Answer:`` extraction in
``inkling_mmmu_harness.parse_multi_choice_response`` so the asymmetric
false-negative scoring bug (SGLang ~8.6% vs TRT ~37.5% dollar-style misses, which
manufactures a spurious TRT-vs-SGLang delta) cannot regress. The MMMU prompt
instructs the model to answer ``'Answer: $LETTER'``, so ``Answer: $B$`` is the
EXPECTED output and must parse to ``B``.

Runs on CPU with no numpy/torch install: ``inkling_mmmu_harness`` imports ``numpy``
only at module scope and the parser under test is pure-``re``, so numpy is stubbed
if it is not installed. Runnable either under pytest or as
``python3 test_inkling_mmmu_parser.py``.
"""
import os
import sys
import types

# Stub numpy so the pure-re parser imports without a numpy install (login/CI CPU).
# The harness uses numpy only for module-level IMAGE_MEAN/IMAGE_STD/PAD_NORM image
# constants (lines 48-51) that ``parse_multi_choice_response`` never touches, so a
# minimal stub whose factories return 1.0 lets those constants evaluate without
# error while the REAL parser runs unchanged. The real module is preferred when
# installed.
if "numpy" not in sys.modules:
    try:  # prefer the real module when available
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001
        _np = types.ModuleType("numpy")
        _one = lambda *a, **k: 1.0  # noqa: E731  (array/full/float32/... -> 1.0)
        for _name in ("array", "asarray", "empty", "full", "broadcast_to",
                      "float32", "float", "uint"):
            setattr(_np, _name, _one)
        _np.ndarray = object
        sys.modules["numpy"] = _np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_mmmu_harness as H  # noqa: E402

CH5 = ["A", "B", "C", "D", "E"]


def _p(resp, choices=CH5):
    return H.parse_multi_choice_response(resp, choices, {c: "" for c in choices})


def test_wrapping_forms_all_extract_the_letter():
    # Directive 2.1: every wrapper form must extract the wrapped letter.
    assert _p("Reasoning ...\nAnswer: $B$") == "B"
    assert _p("Reasoning ...\nAnswer: \\(B\\)") == "B"
    assert _p("Reasoning ...\nAnswer: **B**") == "B"
    assert _p("Reasoning ...\nAnswer: $\\text{B}$") == "B"
    assert _p("Reasoning ...\nAnswer: \\boxed{B}") == "B"
    assert _p("Reasoning ...\nAnswer: $\\boxed{\\text{B}}$") == "B"
    # tolerate whitespace around the colon and inside the wrapper
    assert _p("Answer : $ B $") == "B"


def test_plain_forms_still_work_no_regression():
    assert _p("Answer: B") == "B"
    assert _p("Answer:B") == "B"
    assert _p("Answer: (C)") == "C"
    assert _p("blah blah\nAnswer: D\n") == "D"
    assert _p("Answer: B is correct because ...") == "B"


def test_cited_false_negative_records():
    # validation_Math_14: model answered C correctly; the OLD regex fell through and
    # scored it B (0.0). The fix must recover C. (real token layout, special tokens)
    math14 = "... so the total is C.\n<|content_text|>Answer: $C$<|end_message|>"
    assert _p(math14, CH5) == "C"
    # validation_Agriculture_4: the model's STATED final is B (wrong vs gold E, but
    # the PARSE must reflect the model's stated letter, not the heuristic 'D').
    agri4 = "...<|content_text|>Answer: $B$<|end_message|>"
    assert _p(agri4, CH5) == "B"


def test_last_occurrence_wins():
    # first A, then changed the mind to $C$: the FINAL answer wins.
    assert _p("Answer: A\nOn reflection, Answer: $C$") == "C"
    # dollar-wrapped repetition (collapse) -> last letter is still well-defined.
    assert _p("Answer: $B$ " * 40) == "B"


def test_parser_does_not_contradict_unambiguous_final_answer():
    # Directive 2.4: when there is one clear final 'Answer: X', return X even if
    # other option letters appear incidentally in the reasoning above it.
    resp = ("We compare option (A) and option (D) at length; (A) seems plausible "
            "but the calculation rules it out.\nAnswer: $E$")
    assert _p(resp) == "E"


def test_prompt_echo_and_words_are_not_mined_for_a_letter():
    # The isolated-letter lookahead must NOT mine a capital out of an echoed
    # instruction 'Answer: $LETTER' or a following word; such a response has no
    # explicit answer and should fall through (here -> heuristic default A).
    assert _p("Answer: $LETTER$", CH5) == "A"      # no isolated letter -> fallback
    assert _p("Answer: See table 3", CH5) == "A"   # 'See' is not an answer


def _all_tests():
    return [v for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    tests = _all_tests()
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"MMMU_PARSER_TEST_OK ({len(tests)} tests)")

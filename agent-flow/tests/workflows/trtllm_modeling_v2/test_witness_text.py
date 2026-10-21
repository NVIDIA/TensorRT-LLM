# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The prose a token-comparing witness generates to a declared length.

A counting pattern is continued by copying from the attention pages, so a
hop that drops the recurrent state still reproduces it. These pin the
properties that make the generated prompts something the model has to read.
"""

from __future__ import annotations

import re

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks.witness import filler_words
from agent_flow.workflows.trtllm_modeling_v2.checks.witness_text import (
    NATURAL_TEXT_WORDS,
    natural_text,
)
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import MAX_WITNESS_PROMPT_WORDS


@pytest.mark.parametrize("count", [1, 5, 131, 2050, 4097, NATURAL_TEXT_WORDS + 7, 16384])
def test_the_prompt_is_exactly_the_declared_number_of_words(count):
    """Every word is at least one token, so ``count`` words is at least ``count`` tokens."""
    assert len(natural_text(count).split()) == count


def test_no_words_is_an_empty_prompt():
    assert natural_text(0) == ""


def test_the_prompt_is_fixed_for_a_length():
    """Both legs, and every run of one witness, send the same words."""
    assert natural_text(2050) == natural_text(2050)


def test_the_passage_covers_the_usual_lengths_without_repeating():
    """A repeated tail is a pattern again: copying continues it without the state."""
    assert NATURAL_TEXT_WORDS > 4097
    words = natural_text(NATURAL_TEXT_WORDS).split()
    windows = [tuple(words[i : i + 8]) for i in range(len(words) - 7)]
    assert len(set(windows)) == len(windows)


def test_it_is_prose_and_not_a_counting_pattern():
    prose, pattern = natural_text(2050).split(), filler_words(2050).split()
    assert len(set(pattern)) == 10
    assert len(set(prose)) > 500
    assert sum(word.endswith(".") for word in prose) > 50


def test_two_declared_lengths_are_two_texts():
    """No generated prompt finds its prefix cached by another's prefill."""
    lengths = (131, 2050, 2051, 4097)
    openings = {natural_text(n)[:60] for n in lengths}
    assert len(openings) == len(lengths)
    for short in lengths:
        for long in lengths:
            if short < long:
                assert not natural_text(long).startswith(natural_text(short))


def test_the_text_survives_a_shell_quoted_argv():
    """Each prompt travels as one argv string, which Linux caps at 128 KiB."""
    text = natural_text(MAX_WITNESS_PROMPT_WORDS)
    assert len(text.encode()) < 128 * 1024
    assert not re.search(r"[0-9'\"\n\t`$\\]", natural_text(NATURAL_TEXT_WORDS))

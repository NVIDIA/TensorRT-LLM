# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stop-criteria helpers shared by the sampler and the disaggregated
generation server's first-token close-out."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.sampler.sampler_features import (
    meet_max_token_stop_criteria,
    meet_max_token_stop_criteria_at,
    meet_stop_token_criteria,
    stop_words_match,
)

pytestmark = pytest.mark.cpu_only


def _request(tokens, *, stop_words=None):
    return SimpleNamespace(
        py_orig_prompt_len=3,
        py_max_new_tokens=4,
        py_stop_words_list=stop_words,
        get_num_tokens=lambda beam: len(tokens),
        get_tokens=lambda beam: list(tokens),
    )


@pytest.mark.parametrize(
    ("num_tokens", "max_seq_len", "expected"),
    [
        pytest.param(6, 64, False, id="below_both_limits"),
        pytest.param(7, 64, True, id="max_new_tokens"),
        pytest.param(5, 5, True, id="max_seq_len"),
    ],
)
def test_max_token_criteria_at_a_length_matches_the_request_check(
    num_tokens, max_seq_len, expected
) -> None:
    request = _request(list(range(num_tokens)))
    num_new_tokens = num_tokens - request.py_orig_prompt_len

    assert (
        meet_max_token_stop_criteria_at(request, num_new_tokens, num_tokens, max_seq_len)
        is expected
    )
    assert meet_max_token_stop_criteria(request, max_seq_len) is expected


@pytest.mark.parametrize(
    ("stop_words", "expected"),
    [
        pytest.param([[9], [4]], True, id="single_token_word"),
        pytest.param([[9], [3]], False, id="single_token_miss"),
        pytest.param([[2, 4], [9]], True, id="multi_token_word"),
        pytest.param([[1, 4]], False, id="multi_token_miss"),
        pytest.param([[0, 1, 2, 3, 4, 5]], False, id="word_longer_than_history"),
    ],
)
def test_stop_words_match_agrees_with_the_request_check(stop_words, expected) -> None:
    """The history is read only when a multi-token stop word needs it."""
    tokens = [1, 2, 4]  # ends with the new token 4
    history_reads = []

    def history():
        history_reads.append(True)
        return tokens

    assert stop_words_match(stop_words, 4, history) is expected
    assert meet_stop_token_criteria(_request(tokens, stop_words=stop_words), 4) is expected
    assert bool(history_reads) == any(len(word) > 1 for word in stop_words)

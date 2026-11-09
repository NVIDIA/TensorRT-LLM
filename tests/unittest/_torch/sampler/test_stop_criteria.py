# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``first_stop_reason``, the stop check of a token outside the per-step
sampler path, against the sampler's ``handle_stop_criteria``."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.sampler.sampler_features import (
    first_stop_reason,
    handle_stop_criteria,
)

pytestmark = pytest.mark.cpu_only

_PROMPT = [1, 2, 3]


def _request(tokens: list, *, end_id: int = -1, stop_words=None) -> SimpleNamespace:
    """A request holding ``tokens`` with max_new_tokens 4."""
    request = SimpleNamespace(
        py_orig_prompt_len=len(_PROMPT),
        py_max_new_tokens=4,
        py_end_id=end_id,
        py_stop_words_list=stop_words,
        finished=None,
    )
    request.get_num_tokens = lambda beam: len(tokens)
    request.get_tokens = lambda beam: list(tokens)
    request.finish_by = lambda reason, beam: setattr(request, "finished", reason)
    return request


@pytest.mark.parametrize(
    ("generated", "new_token", "options", "max_seq_len"),
    [
        pytest.param([], 9, {}, 64, id="continues"),
        pytest.param([], 7, {"end_id": 7}, 64, id="end_id"),
        pytest.param([5, 6, 8], 7, {"end_id": 7}, 64, id="end_id_before_length"),
        pytest.param([5, 6, 8], 9, {}, 64, id="max_new_tokens"),
        pytest.param([], 9, {}, 4, id="max_seq_len"),
        pytest.param([5, 6, 8], 9, {"stop_words": [[9]]}, 64, id="length_before_stop_words"),
        pytest.param([], 9, {"stop_words": [[4], [9]]}, 64, id="single_token_stop_word"),
        pytest.param([5], 9, {"stop_words": [[4], [5, 9]]}, 64, id="multi_token_stop_word"),
        pytest.param([], 9, {"stop_words": [[3, 9]]}, 64, id="stop_word_spanning_the_prompt"),
        pytest.param(
            [], 9, {"stop_words": [[0, 1, 2, 3, 9]]}, 64, id="stop_word_beyond_the_sequence"
        ),
    ],
)
def test_first_stop_reason_matches_the_sampler(
    generated: list, new_token: int, options: dict, max_seq_len: int
) -> None:
    """For the same sequence the check gives the sampler's verdict, so a token
    checked off the sampler path ends a request exactly when sampling it would."""
    before = _PROMPT + generated
    sampled = _request(before + [new_token], **options)
    handle_stop_criteria(sampled, new_token, max_seq_len=max_seq_len, beam_idx=0)

    reason = first_stop_reason(
        _request(before, **options),
        new_token,
        num_generated_tokens=len(generated) + 1,
        num_tokens=len(before) + 1,
        max_seq_len=max_seq_len,
        stop_word_history=lambda count: before[-count:],
    )

    assert reason == sampled.finished


def test_first_stop_reason_can_leave_max_seq_len_unchecked() -> None:
    """A caller whose sequence length does not match the engine limit's basis
    (a Helix CP rank's slice) leaves that limit to the decode steps."""
    request = _request(_PROMPT)

    reason = first_stop_reason(
        request,
        9,
        num_generated_tokens=1,
        num_tokens=len(_PROMPT) + 1,
        max_seq_len=None,
        stop_word_history=lambda count: _PROMPT[-count:],
    )

    assert reason is None

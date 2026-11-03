# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The end token of a finished sequence is not part of its text."""

import pytest

from tensorrt_llm.executor.result import CompletionOutput, _text_token_ids
from tensorrt_llm.sampling_params import SamplingParams

_END_ID = 9


def _output(finish_reason, stop_reason=None):
    out = CompletionOutput(index=0, text="")
    out.finish_reason = finish_reason
    out.stop_reason = stop_reason
    return out


def test_end_token_is_left_out_of_the_text():
    params = SamplingParams(end_id=_END_ID)
    assert _text_token_ids(_output("stop"), [1, 2, _END_ID], params) == [1, 2]


@pytest.mark.parametrize(
    "output, params",
    [
        # Unfinished, length-capped and stop-word finishes keep every token.
        (_output(None), SamplingParams(end_id=_END_ID)),
        (_output("length"), SamplingParams(end_id=_END_ID)),
        (_output("stop", stop_reason=7), SamplingParams(end_id=_END_ID)),
        # Asked for explicitly.
        (_output("stop"), SamplingParams(end_id=_END_ID, include_stop_str_in_output=True)),
    ],
)
def test_other_sequences_keep_their_tokens(output, params):
    assert _text_token_ids(output, [1, 2, _END_ID], params) == [1, 2, _END_ID]


def test_streaming_diff_without_the_end_token_is_unchanged():
    params = SamplingParams(end_id=_END_ID)
    assert _text_token_ids(_output("stop"), [3, 4], params) == [3, 4]
    assert _text_token_ids(_output("stop"), [], params) == []

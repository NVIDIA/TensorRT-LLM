# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A run of a degenerate token stops the generation and fails the request.

A server sets the runs with ``SamplingParams._set_degenerate_token_runs`` for a
model that declares degenerate tokens. Each run is a stop word of token ids, so
the engine stops on it as on any stop word, and the result layer then fails the
request instead of reporting a stop. CPU-only: the engine's stop-word rule is
exercised through its host-side reference, ``meet_stop_token_criteria``.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.sampler.sampler_features import meet_stop_token_criteria
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.result import GenerationResultBase
from tensorrt_llm.executor.utils import (
    RequestError,
    degenerate_generation_message,
    is_context_length_exceeded_message,
    is_degenerate_generation_message,
)
from tensorrt_llm.sampling_params import DegenerateTokenRun, SamplingParams

pytestmark = pytest.mark.cpu_only

END_ID = 1
PAD = 2
PLACEHOLDER = 129279
# Servers use 20; a short run keeps the sequences readable.
RUN = 4
PROMPT = (5, 6, 7)


def _params(**kwargs) -> SamplingParams:
    params = SamplingParams(end_id=END_ID, **kwargs)
    params._set_degenerate_token_runs([PAD, PLACEHOLDER], RUN)
    return params


class _Request:
    """The request state ``meet_stop_token_criteria`` reads."""

    def __init__(self, params: SamplingParams):
        self.py_stop_words_list = params._get_stop_words()
        self.tokens = list(PROMPT)

    def get_tokens(self, beam_idx: int) -> list:
        return self.tokens


def _engine_stop(params: SamplingParams, generated: list) -> int | None:
    """Generate ``generated`` token by token; how many were written when the engine stopped."""
    request = _Request(params)
    for count, token in enumerate(generated, start=1):
        request.tokens.append(token)
        if meet_stop_token_criteria(request, token, 0):
            return count
    return None


def _finish(result: GenerationResultBase, generated: list, finish_reason) -> None:
    """Hand the result layer the engine's final response for ``generated``."""
    response = SimpleNamespace(
        output_token_ids=[list(generated)],
        cum_log_probs=None,
        log_probs=None,
        generation_logits=None,
        per_pos_drafted=None,
        per_pos_accepted=None,
        spec_dec_totals=None,
        time_breakdown_metrics=None,
        request_perf_metrics=None,
        additional_context_outputs=None,
        additional_generation_outputs=None,
    )
    result._handle_sequence([finish_reason], response, 0)


def _finished_result(params: SamplingParams) -> GenerationResultBase:
    result = GenerationResultBase(id=11, sampling_params=params)
    result._done = True
    return result


# --------------------------------------------------------------------------
# The runs are stop words, matched by id
# --------------------------------------------------------------------------


def test_runs_are_stop_words_after_the_requests_own():
    params = _params(stop_token_ids=[7])
    assert params._get_stop_words() == [[7], [PAD] * RUN, [PLACEHOLDER] * RUN]
    assert params._get_stop_reasons_and_words() == [
        (7, [7]),
        (DegenerateTokenRun(PAD, RUN), [PAD] * RUN),
        (DegenerateTokenRun(PLACEHOLDER, RUN), [PLACEHOLDER] * RUN),
    ]


def test_without_runs_the_stop_words_are_unchanged():
    params = SamplingParams(end_id=END_ID, stop_token_ids=[7])
    assert params._get_stop_words() == [[7]]
    assert params._get_stop_reasons_and_words() == [(7, [7])]
    assert SamplingParams(end_id=END_ID)._get_stop_words() == []


def test_runs_stay_ids_when_stop_strings_are_tokenized():
    """A stop string spelling a degenerate token is encoded as text; the run is not."""
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: [91, 47, 93])
    params = _params(stop="<pad>")
    params._setup(tokenizer, None, None)
    assert params._get_stop_words() == [[91, 47, 93], [PAD] * RUN, [PLACEHOLDER] * RUN]
    assert [reason for reason, _ in params._get_stop_reasons_and_words()] == [
        "<pad>",
        DegenerateTokenRun(PAD, RUN),
        DegenerateTokenRun(PLACEHOLDER, RUN),
    ]


def test_duplicate_ids_give_one_run_each():
    params = SamplingParams(end_id=END_ID)
    params._set_degenerate_token_runs([PAD, PAD, PLACEHOLDER], RUN)
    assert params._get_stop_words() == [[PAD] * RUN, [PLACEHOLDER] * RUN]


def test_a_run_must_be_at_least_one_token():
    with pytest.raises(ValueError, match="run_length"):
        SamplingParams(end_id=END_ID)._set_degenerate_token_runs([PAD], 0)


@pytest.mark.parametrize("token", [PAD, PLACEHOLDER])
def test_the_engine_stops_at_the_threshold(token):
    assert _engine_stop(_params(), [10, 11] + [token] * (RUN + 5)) == 2 + RUN


def test_a_shorter_run_does_not_stop_the_engine():
    generated = [10] + [PAD] * (RUN - 1) + [11] + [PLACEHOLDER] * (RUN - 1) + [12]
    assert _engine_stop(_params(), generated) is None


def test_text_spelling_a_degenerate_token_does_not_stop_the_engine():
    """Ordinary tokens that read like the pad token are not the pad token."""
    assert _engine_stop(_params(), [91, 47, 93] * (RUN + 1)) is None


def test_without_runs_the_engine_does_not_stop():
    assert _engine_stop(SamplingParams(end_id=END_ID), [PAD] * (RUN * 5)) is None


# --------------------------------------------------------------------------
# The result layer fails a generation that stopped on a run
# --------------------------------------------------------------------------


@pytest.mark.parametrize("token", [PAD, PLACEHOLDER])
def test_a_generation_stopped_on_a_run_fails(token):
    result = _finished_result(_params())

    with pytest.raises(RequestError) as excinfo:
        _finish(result, [10, 11] + [token] * RUN, tllm.FinishReason.STOP_WORDS)

    message = degenerate_generation_message(token, RUN)
    assert str(excinfo.value) == message
    # The failure sticks: the result reports it and every later wait raises it.
    assert result.error == message
    assert result._terminal_error is excinfo.value
    # It is the request's failure, so nothing stopped normally.
    assert result._outputs[0].stop_reason is None


def test_the_requests_own_stop_token_wins_over_a_run():
    params = SamplingParams(end_id=END_ID, stop_token_ids=[PAD])
    params._set_degenerate_token_runs([PAD], 1)
    result = _finished_result(params)

    _finish(result, [10, PAD], tllm.FinishReason.STOP_WORDS)

    output = result._outputs[0]
    assert (output.finish_reason, output.stop_reason, output.token_ids) == ("stop", PAD, [10])
    assert result.error is None


def test_a_shorter_run_completes_normally():
    result = _finished_result(_params())
    generated = [10] + [PAD] * (RUN - 1) + [11]

    _finish(result, generated, tllm.FinishReason.END_ID)

    output = result._outputs[0]
    assert (output.finish_reason, output.token_ids) == ("stop", generated)
    assert result.error is None


def test_a_failed_sequence_aborts_the_rest_of_the_request():
    """With several sequences, the others are cancelled with the failed one."""
    result = GenerationResultBase(id=11, sampling_params=_params())
    result._done = False

    with pytest.raises(RequestError):
        _finish(result, [10] + [PAD] * RUN, tllm.FinishReason.STOP_WORDS)

    assert result.aborted()


# --------------------------------------------------------------------------
# The message the server recognizes
# --------------------------------------------------------------------------


def test_message_names_the_token_and_the_run():
    message = degenerate_generation_message(PLACEHOLDER, 20)
    assert f"token id {PLACEHOLDER} (20 in a row)" in message
    assert "retry" in message


def test_detector_matches_exact_and_wrapped_messages():
    message = degenerate_generation_message(PAD, 20)
    assert is_degenerate_generation_message(message)
    # A postprocessing worker forwards it with a prefix.
    assert is_degenerate_generation_message(f"Postprocessing error: {message}")
    assert not is_context_length_exceeded_message(message)


def test_detector_rejects_other_errors():
    assert not is_degenerate_generation_message("Generation failed: request cancelled")
    assert not is_degenerate_generation_message("")

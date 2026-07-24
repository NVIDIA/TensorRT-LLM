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
"""Unit tests for OSL enforcement at the disaggregated context->gen handoff.

Regression coverage for nvbugs/6482606: under disagg + MTP/spec-decode, a request
with ``max_tokens=1`` received 2 output tokens because the context-side first token
was injected on the gen server without any stop-criteria check, so an extra
generation (MTP) step ran and emitted a second token before the sampler's stop
check fired. These tests exercise the two fix surfaces directly, without a GPU:

  * ``PyExecutor._apply_disagg_gen_first_token_stop_criteria`` — applies the same
    stop criteria (END_ID / max-token / stop-words) the aggregated sampler applies
    to the context step's first token, finishing the request before the gen step.
  * ``PyExecutor._handle_first_token_response`` — must skip an already-finished
    request so it does not emit a (final) first-token response in addition to the
    final response from ``_handle_responses`` (which would double-count the token).
"""

from types import SimpleNamespace
from unittest.mock import Mock

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.bindings.executor import FinishReason


def _make_executor(max_seq_len=8192):
    executor = object.__new__(PyExecutor)
    executor.max_seq_len = max_seq_len
    return executor


def _make_disagg_gen_req(
    *, num_tokens, orig_prompt_len, max_new_tokens, end_id=-1, beam_width=1, stop_words=None
):
    """Stub of a disagg gen request *after* the context first token was injected.

    ``num_tokens`` is the post-injection token count (prompt + generated). The same
    value is returned for every beam, which matches the state right after the
    per-beam ``add_new_token`` loop in ``_prepare_disagg_gen_transmission_complete``.
    """
    req = SimpleNamespace()
    req.py_orig_prompt_len = orig_prompt_len
    req.py_max_new_tokens = max_new_tokens
    req.py_end_id = end_id
    req.py_stop_words_list = stop_words
    req.py_beam_width = beam_width
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.get_num_tokens = lambda beam_idx=0: num_tokens
    req.set_finished_reason = Mock()
    return req


def test_first_token_completes_request_when_max_tokens_is_one():
    # The core nvbugs/6482606 case: with max_tokens=1 the injected context token
    # already exhausts the budget, so the request must finish here (LENGTH) before
    # the immediately-following MTP gen step can emit a second token.
    executor = _make_executor()
    req = _make_disagg_gen_req(num_tokens=11, orig_prompt_len=10, max_new_tokens=1)

    PyExecutor._apply_disagg_gen_first_token_stop_criteria(executor, req, [42], 1)

    assert req.state == LlmRequestState.GENERATION_COMPLETE
    req.set_finished_reason.assert_called_once_with(FinishReason.LENGTH, 0)


def test_first_token_does_not_complete_when_budget_remains():
    # max_tokens=2: one more token is allowed, so the request must NOT be finished
    # by the first token and should proceed to a normal generation step.
    executor = _make_executor()
    req = _make_disagg_gen_req(num_tokens=11, orig_prompt_len=10, max_new_tokens=2)

    PyExecutor._apply_disagg_gen_first_token_stop_criteria(executor, req, [42], 1)

    assert req.state == LlmRequestState.GENERATION_IN_PROGRESS
    req.set_finished_reason.assert_not_called()


def test_first_token_end_id_completes_request():
    # An end-id first token finishes the request even when the token budget is not
    # exhausted, matching TorchSampler._handle_stop_criteria's END_ID branch.
    executor = _make_executor()
    req = _make_disagg_gen_req(num_tokens=11, orig_prompt_len=10, max_new_tokens=16, end_id=42)

    PyExecutor._apply_disagg_gen_first_token_stop_criteria(executor, req, [42], 1)

    assert req.state == LlmRequestState.GENERATION_COMPLETE
    req.set_finished_reason.assert_called_once_with(FinishReason.END_ID, 0)


def test_first_token_stop_word_completes_request():
    # A single-token stop word completed by the first token finishes the request.
    # This is the parity case that a bare max-token/END_ID check would miss.
    executor = _make_executor()
    req = _make_disagg_gen_req(
        num_tokens=11, orig_prompt_len=10, max_new_tokens=16, stop_words=([99], [1])
    )

    PyExecutor._apply_disagg_gen_first_token_stop_criteria(executor, req, [99], 1)

    assert req.state == LlmRequestState.GENERATION_COMPLETE
    req.set_finished_reason.assert_called_once_with(FinishReason.STOP_WORDS, 0)


def test_none_max_seq_len_still_enforces_output_length():
    # max_seq_len is Optional on PyExecutor; when unset, the per-request
    # max_new_tokens budget must still be enforced.
    executor = _make_executor(max_seq_len=None)
    req = _make_disagg_gen_req(num_tokens=11, orig_prompt_len=10, max_new_tokens=1)

    PyExecutor._apply_disagg_gen_first_token_stop_criteria(executor, req, [42], 1)

    assert req.state == LlmRequestState.GENERATION_COMPLETE
    req.set_finished_reason.assert_called_once_with(FinishReason.LENGTH, 0)


def test_all_beams_complete_together_for_length():
    # With max_tokens=1 every beam is exhausted by its injected token, so the whole
    # request completes and each beam gets a finish reason.
    executor = _make_executor()
    req = _make_disagg_gen_req(num_tokens=11, orig_prompt_len=10, max_new_tokens=1, beam_width=2)

    PyExecutor._apply_disagg_gen_first_token_stop_criteria(executor, req, [42, 43], 2)

    assert req.state == LlmRequestState.GENERATION_COMPLETE
    assert req.set_finished_reason.call_count == 2


def test_request_incomplete_until_all_beams_stop():
    # Beam 0's first token is the end id but beam 1's is not and the budget is not
    # exhausted, so the request must remain in progress (agg completes a request
    # only when every beam has stopped).
    executor = _make_executor()
    req = _make_disagg_gen_req(
        num_tokens=11, orig_prompt_len=10, max_new_tokens=16, end_id=42, beam_width=2
    )

    PyExecutor._apply_disagg_gen_first_token_stop_criteria(executor, req, [42, 43], 2)

    assert req.state == LlmRequestState.GENERATION_IN_PROGRESS
    req.set_finished_reason.assert_not_called()


def _make_first_token_req(*, is_finished, request_id):
    req = SimpleNamespace()
    req.py_decoding_iter = 1
    req.is_finished = is_finished
    req.py_request_id = request_id
    req.create_response = Mock(return_value=SimpleNamespace(result=SimpleNamespace()))
    return req


def test_handle_first_token_response_skips_finished_request():
    # A request finished at injection must not emit a first-token response here:
    # _handle_responses emits its single final response, and emitting one here too
    # would produce a second final response and double-count the token
    # (nvbugs/6482606). Only the still-in-progress request gets a first-token
    # response.
    executor = object.__new__(PyExecutor)
    executor._enqueue_responses = Mock()
    executor._has_prepended_logits = Mock(return_value=False)
    executor.dist = SimpleNamespace(rank=0)

    finished = _make_first_token_req(is_finished=True, request_id=1)
    unfinished = _make_first_token_req(is_finished=False, request_id=2)
    scheduled = SimpleNamespace(generation_requests=[finished, unfinished])

    PyExecutor._handle_first_token_response(executor, scheduled)

    finished.create_response.assert_not_called()
    unfinished.create_response.assert_called_once()
    executor._enqueue_responses.assert_called_once()
    enqueued = executor._enqueue_responses.call_args.args[0]
    assert [request_id for request_id, _ in enqueued] == [2]

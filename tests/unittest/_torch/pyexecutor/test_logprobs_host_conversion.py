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

from math import prod
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.sampler.logprobs import (
    LogProbsHandler,
    LogProbsState,
    LogProbsStateList,
    store_logprobs_list_to_request,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler import (
    SampleStateTensorsHostTorch,
    SampleStateTorch,
    TorchSampler,
)

pytestmark = pytest.mark.cpu_only


def _make_logprobs_state(
    num_slots: int = 8,
    beam_width: int = 2,
    num_steps: int = 3,
    topk: int = 2,
) -> LogProbsState:
    sampled_shape = (num_slots, beam_width, num_steps)
    topk_shape = (num_slots, num_steps, topk)
    return LogProbsState(
        sampled_vals=torch.arange(prod(sampled_shape), dtype=torch.float32).reshape(sampled_shape),
        sampled_indices=torch.arange(prod(sampled_shape), dtype=torch.int32).reshape(sampled_shape),
        sampled_rank=torch.arange(prod(sampled_shape), dtype=torch.int32).reshape(sampled_shape),
        topk_vals=torch.arange(prod(topk_shape), dtype=torch.float32).reshape(topk_shape),
        topk_indices=torch.arange(prod(topk_shape), dtype=torch.int32).reshape(topk_shape),
    )


@pytest.mark.parametrize(
    ("num_topk", "simple_format", "topk_width"),
    [(0, False, 2), (0, True, 2), (2, False, 2), (0, False, 0), (0, True, 0)],
)
def test_sparse_logprobs_conversion_matches_dense(
    num_topk: int,
    simple_format: bool,
    topk_width: int,
) -> None:
    state = _make_logprobs_state(topk=topk_width)
    active_slots = [1, 6]
    dense = LogProbsStateList.from_logprobs_state(state)
    sparse = LogProbsStateList.from_logprobs_state(state, active_slots=active_slots)

    assert len(sparse.sampled_vals) == len(active_slots)
    for slot in active_slots:
        expected = store_logprobs_list_to_request(
            dense,
            req_seq_slot=slot,
            beam_width=2,
            count=3,
            num_topk_logprobs=num_topk,
            simple_format=simple_format,
        )
        actual = store_logprobs_list_to_request(
            sparse,
            req_seq_slot=slot,
            beam_width=2,
            count=3,
            num_topk_logprobs=num_topk,
            simple_format=simple_format,
        )
        assert actual == expected


def test_dense_logprobs_conversion_preserves_slot_indexing() -> None:
    state = _make_logprobs_state()
    dense = LogProbsStateList.from_logprobs_state(state)
    dense_with_slots = LogProbsStateList.from_logprobs_state(
        state,
        active_slots=[0, 1, 2, 3],
    )

    assert len(dense_with_slots.sampled_vals) == 8
    assert dense_with_slots == dense


def test_update_requests_converts_only_active_logprob_slots() -> None:
    def make_request(slot: int, return_logprobs: bool, state: LlmRequestState) -> Mock:
        return Mock(
            spec=LlmRequest,
            py_seq_slot=slot,
            py_beam_width=1,
            py_return_log_probs=return_logprobs,
            py_num_logprobs=0 if return_logprobs else None,
            py_logprobs_simple_format=True,
            py_draft_tokens=None,
            py_decoding_iter=0,
            py_stop_words_list=None,
            py_result=Mock(),
            get_num_tokens=Mock(return_value=1),
            state=state,
        )

    generating = LlmRequestState.GENERATION_IN_PROGRESS
    active = make_request(6, True, generating)
    without_logprobs = make_request(3, False, generating)
    completed = make_request(4, True, LlmRequestState.GENERATION_COMPLETE)
    other_active = make_request(1, True, generating)
    logprobs_state = _make_logprobs_state(beam_width=1, num_steps=1, topk=0)
    state = SampleStateTorch(
        requests=[active, without_logprobs, completed, other_active],
        host=SampleStateTensorsHostTorch(
            new_tokens=torch.zeros((1, 8, 1), dtype=torch.int32),
            finish_reasons=None,
            first_finish_reasons=None,
            logprobs_state=logprobs_state,
        ),
    )
    sampler = Mock(
        spec=TorchSampler,
        max_num_sequences=8,
        max_tokens=1,
        max_beam_width=1,
        _track_pending_steps=False,
        _batch_fastpath_eligible=False,
        _use_beam_search=False,
        _process_draft_tokens=Mock(return_value=0),
        _top_p_decay=Mock(),
        _penalty_handler=Mock(),
    )
    sampler._log_probs = LogProbsHandler(sampler)

    with patch.object(
        LogProbsStateList,
        "from_logprobs_state",
        wraps=LogProbsStateList.from_logprobs_state,
    ) as convert:
        TorchSampler.update_requests(sampler, state)

    convert.assert_called_once_with(logprobs_state, active_slots=[6, 1])
    active.py_result.append_log_probs.assert_called_once_with([[6.0]])
    other_active.py_result.append_log_probs.assert_called_once_with([[1.0]])
    without_logprobs.py_result.append_log_probs.assert_not_called()
    completed.py_result.append_log_probs.assert_not_called()

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

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.logprobs import (
    LogProbsState,
    LogProbsStateList,
    store_logprobs_list_to_request,
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


@pytest.mark.parametrize(("num_topk", "simple_format"), [(0, False), (0, True), (2, False)])
def test_sparse_logprobs_conversion_matches_dense(
    num_topk: int,
    simple_format: bool,
) -> None:
    state = _make_logprobs_state()
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

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
"""Host-side request updates of the one-model speculative sampler."""

import types
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative import spec_sampler_base as sampler_ops
from tensorrt_llm._torch.speculative.spec_sampler_base import (
    SampleStateSpec,
    SampleStateTensorsSpec,
    SpecSampler,
)

pytestmark = pytest.mark.cpu_only

NUM_SLOTS = 16
PATH_LEN = 4
DRAFT_LEN = 3


def _request(slot: Optional[int], state=LlmRequestState.GENERATION_IN_PROGRESS):
    request = types.SimpleNamespace(
        py_seq_slot=slot,
        state=state,
        py_draft_tokens=[99] * DRAFT_LEN,
        py_decoding_iter=0,
        tokens=[],
    )
    request.add_new_token = lambda token, beam_idx: request.tokens.append(token)
    return request


@pytest.mark.parametrize("verify_lens", [None, [3, 2, 1, 2]])
def test_update_requests_reads_each_request_slot_row(monkeypatch, verify_lens):
    """Rows of scattered slots in a larger store must reach the request that owns them."""
    monkeypatch.setattr(sampler_ops, "handle_stop_criteria", lambda *args, **kwargs: False)
    sampler = SpecSampler.__new__(SpecSampler)
    sampler.draft_len = DRAFT_LEN
    sampler.max_accepted_path_len = PATH_LEN
    sampler.max_seq_len = 128

    new_tokens = torch.arange(PATH_LEN * NUM_SLOTS, dtype=torch.int32).reshape(
        PATH_LEN, NUM_SLOTS, 1
    )
    new_tokens_lens = torch.full((NUM_SLOTS,), PATH_LEN + 1, dtype=torch.int32)
    new_tokens_lens[[11, 0, 7]] = torch.tensor([3, 1, 2], dtype=torch.int32)
    next_draft_tokens = 1000 + torch.arange(NUM_SLOTS * DRAFT_LEN, dtype=torch.int32).reshape(
        NUM_SLOTS, DRAFT_LEN
    )
    host = SampleStateTensorsSpec(
        new_tokens=new_tokens,
        new_tokens_lens=new_tokens_lens,
        next_draft_tokens=next_draft_tokens,
        verify_lens=None if verify_lens is None else torch.tensor(verify_lens, dtype=torch.int32),
    )
    # The finished request between the others shifts list positions against slot rows.
    requests = [
        _request(11),
        _request(None, LlmRequestState.GENERATION_COMPLETE),
        _request(0),
        _request(7),
    ]
    draft_lens = [3, 3, 0, 2]
    runtime_draft_len = 2
    state = SampleStateSpec(
        requests=requests,
        device=host,
        host=host,
        sampler_event=types.SimpleNamespace(synchronize=lambda: None),
        runtime_draft_len=runtime_draft_len,
        draft_lens=draft_lens,
    )

    sampler.update_requests(state)

    finished = requests[1]
    assert finished.tokens == []
    assert finished.py_draft_tokens == [99] * DRAFT_LEN
    assert finished.py_decoding_iter == 0
    for request_index in (0, 2, 3):
        request = requests[request_index]
        slot = request.py_seq_slot
        num_new_tokens = int(new_tokens_lens[slot])
        verified_len = runtime_draft_len if verify_lens is None else verify_lens[request_index] - 1
        assert request.tokens == new_tokens[:num_new_tokens, slot, 0].tolist()
        assert request.py_num_accepted_draft_tokens == num_new_tokens - 1
        assert request.py_num_draft_tokens_verified == min(draft_lens[request_index], verified_len)
        assert request.py_rewind_len == verified_len - (num_new_tokens - 1)
        assert request.py_draft_tokens == next_draft_tokens[slot, :runtime_draft_len].tolist()
        assert request.py_decoding_iter == 1

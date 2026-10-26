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
"""SpecSampler.update_requests stops real requests exactly as add_token + handle_stop_criteria."""

import random
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import add_token
from tensorrt_llm._torch.pyexecutor.sampler.sampler_features import handle_stop_criteria
from tensorrt_llm._torch.speculative.spec_sampler_base import (
    SampleStateSpec,
    SampleStateTensorsSpec,
    SpecSampler,
)
from tensorrt_llm.bindings.executor import FinishReason

pytestmark = pytest.mark.cpu_only

DRAFT_LEN = 3
MAX_SEQ_LEN = 64
# A small vocabulary, so end ids and stop words come up often.
VOCAB = 10
END_ID = 5
END_IDS = (END_ID, END_ID, None, -1)
STOP_WORDS = (
    None,
    [],
    [[7]],
    [[3, 4]],
    [[9], [2, 8, 2]],
    [[1, 2, 3, 4, 5, 6]],
)
NUM_REQUESTS = 32
NUM_STEPS = 10
# Requests 0 and 1 run through every step: no end id, no stop words, and at most
# NUM_STEPS * (DRAFT_LEN + 1) new tokens after a 1-token prompt stay below MAX_SEQ_LEN.
NUM_LONG_RUNNERS = 2


def _request(request_id, rng):
    if request_id < NUM_LONG_RUNNERS:
        prompt_len, max_new_tokens, end_id, stop_words = 1, 1000, None, None
    else:
        prompt_len = rng.choice((1, 8, 40, 60, 63, MAX_SEQ_LEN))
        max_new_tokens = rng.randint(1, 30)
        end_id = rng.choice(END_IDS)
        stop_words = rng.choice(STOP_WORDS)
    request = LlmRequest(
        request_id=request_id,
        seq_slot=request_id,
        max_new_tokens=max_new_tokens,
        input_tokens=[rng.randrange(VOCAB) for _ in range(prompt_len)],
        sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
        is_streaming=False,
        end_id=end_id,
        stop_words_list=stop_words,
    )
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    return request


def _reference_update(requests, new_tokens, new_tokens_lens):
    """The accepted-token loop as add_token + handle_stop_criteria."""
    for request in requests:
        if request.state == LlmRequestState.GENERATION_COMPLETE:
            continue
        for step in range(new_tokens_lens[request.py_seq_slot]):
            new_token = add_token(request, new_tokens, beam_idx=0, step=step)
            if handle_stop_criteria(request, new_token, max_seq_len=MAX_SEQ_LEN, beam_idx=0):
                break


def _state(requests, new_tokens, new_tokens_lens):
    host = SampleStateTensorsSpec(
        new_tokens=torch.tensor(new_tokens, dtype=torch.int32),
        new_tokens_lens=torch.tensor(new_tokens_lens, dtype=torch.int32),
        next_draft_tokens=torch.zeros(NUM_REQUESTS, DRAFT_LEN, dtype=torch.int32),
    )
    return SampleStateSpec(
        requests=requests,
        device=host,
        host=host,
        sampler_event=SimpleNamespace(synchronize=lambda: None),
        runtime_draft_len=DRAFT_LEN,
        draft_lens=[DRAFT_LEN] * len(requests),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_update_requests_matches_handle_stop_criteria(monkeypatch, seed):
    reasons = {}
    finish_by = LlmRequest.finish_by

    def recording_finish_by(request, reason, beam):
        reasons.setdefault(id(request), []).append((reason, beam))
        finish_by(request, reason, beam)

    monkeypatch.setattr(LlmRequest, "finish_by", recording_finish_by)
    sampler = SpecSampler.__new__(SpecSampler)
    sampler.max_seq_len = MAX_SEQ_LEN
    sampler.draft_len = DRAFT_LEN
    sampler.max_accepted_path_len = DRAFT_LEN + 1

    # Two copies of one workload: update_requests drives one, the reference the other.
    actual = [_request(i, random.Random(seed * 1000 + i)) for i in range(NUM_REQUESTS)]
    expected = [_request(i, random.Random(seed * 1000 + i)) for i in range(NUM_REQUESTS)]
    rng = random.Random(seed)
    # Requests that completed before this batch are skipped.
    for index in rng.sample(range(NUM_LONG_RUNNERS, NUM_REQUESTS), 3):
        actual[index].state = expected[index].state = LlmRequestState.GENERATION_COMPLETE

    for step in range(NUM_STEPS):
        # new_tokens: [DRAFT_LEN + 1, slots, beam]
        new_tokens = [
            [[rng.randrange(VOCAB)] for _ in range(NUM_REQUESTS)] for _ in range(DRAFT_LEN + 1)
        ]
        new_tokens_lens = [rng.randint(1, DRAFT_LEN + 1) for _ in range(NUM_REQUESTS)]
        sampler.update_requests(_state(actual, new_tokens, new_tokens_lens))
        _reference_update(expected, new_tokens, new_tokens_lens)
        for got, want in zip(actual, expected):
            label = f"seed {seed} step {step} request {got.py_request_id}"
            assert got.get_tokens(0) == want.get_tokens(0), label
            assert got.state == want.state, label
            assert reasons.get(id(got)) == reasons.get(id(want)), label

    # The workload reaches every finish reason; the long runners are still running.
    finished = [reasons[id(r)][0][0] for r in expected if id(r) in reasons]
    assert {FinishReason.END_ID, FinishReason.LENGTH, FinishReason.STOP_WORDS} <= set(finished)
    for runner in actual[:NUM_LONG_RUNNERS]:
        assert runner.state == LlmRequestState.GENERATION_IN_PROGRESS
        assert len(runner.get_tokens(0)) >= 1 + NUM_STEPS

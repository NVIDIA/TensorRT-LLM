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
"""Step-pairing tests for speculative-decoding acceptance stats.

py_draft_tokens is not a valid acceptance-rate denominator at response time:
it is padded to the static max for CUDA graphs (two-model flow), and in the
one-model flow SpecSampler.update_requests replaces it with the NEXT
step's buffer before responses are handled. The samplers therefore write
py_num_draft_tokens_verified — the real proposal count of the same step the
acceptance numerator describes — and PyExecutor._accumulate_spec_dec_stats
consumes it exactly once per verified step. These tests pin that pairing
(GPU-free logic; no engine needed).
"""

from types import SimpleNamespace

import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import (
    MAX_SPEC_DECODE_POSITIONS,
    LlmRequest,
    SamplingConfig,
    get_draft_token_length,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.speculative.spec_sampler_base import (
    SampleStateSpec,
    SampleStateTensorsSpec,
    SpecSampler,
)

# GPU-free logic, but importing py_executor is unproven on the CPU-only CI
# stage, so this file is wired into a GPU list (l0_a10.yml) instead of
# l0_cpu_x86.yml and left unmarked.


def _fake_request(*, verified, accepted, draft_buffer_len):
    """Request stub carrying the counters the accumulator reads/writes.

    py_draft_tokens deliberately holds a buffer of a DIFFERENT length than
    the verified count, mimicking the padded / next-step buffer present at
    accumulation time.
    """
    return SimpleNamespace(
        py_num_draft_tokens_verified=verified,
        py_num_accepted_draft_tokens=accepted,
        py_draft_tokens=[0] * draft_buffer_len,
        py_total_draft_tokens=0,
        py_total_accepted_draft_tokens=0,
        py_per_pos_drafted=[0] * MAX_SPEC_DECODE_POSITIONS,
        py_per_pos_accepted=[0] * MAX_SPEC_DECODE_POSITIONS,
    )


def _accumulate(requests, max_draft_len=0):
    executor = SimpleNamespace(max_draft_len=max_draft_len)
    PyExecutor._accumulate_spec_dec_stats(executor, SimpleNamespace(requests=requests))


class TestAccumulator:
    def test_uses_verified_count_not_draft_buffer(self):
        # The step verified 2 real proposals (1 accepted), but by
        # accumulation time py_draft_tokens holds an 8-token padded /
        # next-step buffer. The old code read the buffer length and would
        # have recorded drafted=8, biasing acceptance rate low.
        request = _fake_request(verified=2, accepted=1, draft_buffer_len=8)
        _accumulate([request])
        assert request.py_total_draft_tokens == 2
        assert request.py_total_accepted_draft_tokens == 1
        assert request.py_per_pos_drafted[:3] == [1, 1, 0]
        assert request.py_per_pos_accepted[:2] == [1, 0]

    def test_consumed_exactly_once(self):
        # A second accumulation pass (e.g. the request sat unscheduled for
        # an iteration) must not double-count the same step.
        request = _fake_request(verified=3, accepted=2, draft_buffer_len=3)
        _accumulate([request])
        _accumulate([request])
        assert request.py_num_draft_tokens_verified == 0
        assert request.py_total_draft_tokens == 3
        assert request.py_total_accepted_draft_tokens == 2
        assert request.py_per_pos_drafted[:4] == [1, 1, 1, 0]

    def test_step_without_verified_drafts_is_skipped(self):
        # Prefill steps and non-spec requests have verified == 0 even when
        # py_draft_tokens holds dummy/placeholder tokens.
        request = _fake_request(verified=0, accepted=0, draft_buffer_len=4)
        _accumulate([request])
        assert request.py_total_draft_tokens == 0
        assert request.py_per_pos_drafted[0] == 0

    def test_tree_drafting_clamps_totals_to_max_path_len(self):
        # Tree drafting verifies up to max_total_draft_tokens nodes but can
        # accept at most max_draft_len (the max path length) per step; the
        # cumulative denominator counts paths, mirroring the C++
        # updateNumTokensPerIteration clamp. Per-pos arrays keep the
        # unclamped node count.
        request = _fake_request(verified=12, accepted=3, draft_buffer_len=12)
        _accumulate([request], max_draft_len=3)
        assert request.py_total_draft_tokens == 3
        assert request.py_total_accepted_draft_tokens == 3
        assert request.py_per_pos_drafted[:13] == [1] * 12 + [0]


def _make_llm_request(request_id, seq_slot):
    return LlmRequest(
        request_id=request_id,
        seq_slot=seq_slot,
        max_new_tokens=64,
        input_tokens=[10, 11, 12, 13],
        sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
        is_streaming=False,
    )


def _run_spec_sampler_update(request, *, draft_lens, new_tokens_lens, next_draft_tokens):
    sampler = SpecSampler.__new__(SpecSampler)
    sampler.max_seq_len = 2048
    sampler.draft_len = len(next_draft_tokens[0])

    runtime_draft_len = len(next_draft_tokens[0])
    # update_requests asserts against this bound; new_tokens below is built
    # with runtime_draft_len + 1 rows, so that is the bound here.
    sampler.max_accepted_path_len = runtime_draft_len + 1
    # new_tokens: [max_new_tokens, seq_slots, beam]
    new_tokens = torch.arange(100, 100 + runtime_draft_len + 1, dtype=torch.int).reshape(
        runtime_draft_len + 1, 1, 1
    )
    host = SampleStateTensorsSpec(
        new_tokens=new_tokens,
        new_tokens_lens=torch.tensor(new_tokens_lens, dtype=torch.int),
        next_draft_tokens=torch.tensor(next_draft_tokens, dtype=torch.int),
    )
    state = SampleStateSpec(
        requests=[request],
        device=host,
        host=host,
        sampler_event=SimpleNamespace(synchronize=lambda: None),
        runtime_draft_len=runtime_draft_len,
        draft_lens=draft_lens,
    )
    sampler.update_requests(state)


class TestSpecSamplerPairing:
    """One-model flow: pair acceptance counts with the completed step.

    update_requests must pair the acceptance count with the draft count of
    the step it just processed, not with the next-step buffer it installs.
    """

    def test_verified_is_completed_step_count_not_next_buffer(self):
        request = _make_llm_request(1, 0)
        # The completed step ran with 3 real draft tokens.
        request.py_draft_tokens = [21, 22, 23]
        # 2 new tokens => 1 accepted; next-step buffer is 4 wide (padded).
        _run_spec_sampler_update(
            request, draft_lens=[3], new_tokens_lens=[2], next_draft_tokens=[[31, 32, 33, 34]]
        )

        assert request.py_num_accepted_draft_tokens == 1
        assert request.py_num_draft_tokens_verified == 3
        # update_requests has already installed the next step's buffer:
        # reading py_draft_tokens here (as the old code did) pairs this
        # step's numerator with the NEXT step's denominator.
        assert get_draft_token_length(request) == 4

    def test_prefill_step_verifies_nothing(self):
        request = _make_llm_request(2, 0)
        # sample_async snapshots draft_lens before adding dummy draft
        # tokens to finished-context requests, so the prefill step carries
        # draft_lens == 0 even though the buffer holds dummies.
        request.py_draft_tokens = [1, 1, 1, 1]
        _run_spec_sampler_update(
            request, draft_lens=[0], new_tokens_lens=[1], next_draft_tokens=[[41, 42, 43, 44]]
        )

        assert request.py_num_accepted_draft_tokens == 0
        assert request.py_num_draft_tokens_verified == 0


class TestDrafterPadRecordsEffectiveLen:
    def test_pre_padding_count_recorded(self):
        from tensorrt_llm._torch.speculative.drafter import Drafter

        class _NoopDrafter(Drafter):
            def prepare_draft_tokens(self, scheduled_requests, resource_manager=None) -> None:
                pass

        drafter = _NoopDrafter(max_concurrency=None)
        drafter._static_max_total_draft_tokens = 4
        drafter._needs_padding_kv_extension = False

        request = SimpleNamespace(py_draft_tokens=[7, 8], py_draft_tokens_effective_len=None)
        batch = SimpleNamespace(generation_requests=[request])
        drafter.pad_draft_tokens_for_cuda_graph(batch)

        assert request.py_draft_tokens_effective_len == 2
        assert request.py_draft_tokens == [7, 8, 0, 0]

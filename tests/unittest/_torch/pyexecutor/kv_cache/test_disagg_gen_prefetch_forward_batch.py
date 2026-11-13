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
"""KVCacheV2Scheduler forward batch with a disaggregated generation KV prefetch request.

With KV prefetch (disagg_gen_prefetch) a generation worker holds up to max_batch_size + N
requests. The extra one is receiving its KV (DISAGG_GENERATION_INIT / TRANS_IN_PROGRESS) or
has received it (TRANS_COMPLETE) while max_batch_size requests decode. The forward batch must
stay at max_batch_size: a receiving request does not count toward the request budget, and a
landed one is scheduled only once a running request leaves the budget.
"""

import itertools

import pytest
from _torch.pyexecutor.kv_cache.test_kv_cache_v2_scheduler import (
    DISAGG_GEN_TRANS_IN_PROGRESS,
    GEN_TO_COMPLETE,
    ids,
    make_disagg_request,
    make_gen_request,
    make_kv_cache_manager,
    make_scheduler,
)

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only

TRANS_COMPLETE = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE.value


def _landed(request_id):
    req = make_gen_request(request_id)
    req.state_value = TRANS_COMPLETE
    req.is_generation_in_progress_state = False
    return req


def _receiving(request_id):
    req = make_gen_request(request_id)
    req.state_value = DISAGG_GEN_TRANS_IN_PROGRESS
    req.is_generation_in_progress_state = False
    return req


def _retiring(request_id):
    req = make_gen_request(request_id)
    req.state_value = GEN_TO_COMPLETE
    req.is_generation_in_progress_state = False
    req.is_generation_to_complete_state = True
    return req


def _schedule(mbs, reqs):
    sched = make_scheduler(make_kv_cache_manager(), max_batch_size=mbs, max_num_tokens=120)
    return sched.schedule_request(reqs, set())


def _forward_size(out):
    return len(out.context_requests) + len(out.generation_requests)


@pytest.mark.parametrize("mbs", [1, 2])
def test_new_prefetch_request_starts_receiving_beside_a_full_batch(mbs):
    reqs = [make_gen_request(i) for i in range(mbs)] + [make_disagg_request(10)]
    out = _schedule(mbs, reqs)
    assert ids(out.generation_requests) == list(range(mbs))
    assert ids(out.fitting_disagg_gen_init_requests) == [10]
    assert _forward_size(out) == mbs


@pytest.mark.parametrize("mbs", [1, 2])
def test_receiving_request_is_not_in_the_forward_batch(mbs):
    reqs = [make_gen_request(i) for i in range(mbs)] + [_receiving(10)]
    out = _schedule(mbs, reqs)
    assert ids(out.generation_requests) == list(range(mbs))
    assert _forward_size(out) == mbs


@pytest.mark.parametrize("mbs", [1, 2])
def test_landed_request_waits_while_the_batch_is_full(mbs):
    reqs = [make_gen_request(i) for i in range(mbs)] + [_landed(10)]
    out = _schedule(mbs, reqs)
    assert ids(out.generation_requests) == list(range(mbs))
    assert _forward_size(out) == mbs


@pytest.mark.parametrize("mbs", [1, 2])
def test_landed_request_takes_the_retiring_requests_place(mbs):
    reqs = [_retiring(0)] + [make_gen_request(i) for i in range(1, mbs)] + [_landed(10)]
    out = _schedule(mbs, reqs)
    assert ids(out.generation_requests) == list(range(1, mbs)) + [10]
    assert _forward_size(out) == mbs


@pytest.mark.parametrize("mbs", [1, 2])
@pytest.mark.parametrize("prefetch", [1, 2])
def test_forward_batch_never_exceeds_max_batch_size(mbs, prefetch):
    """Every mix of running, retiring, receiving and landed requests the prefetch cap allows."""
    makers = [make_gen_request, _retiring, _receiving, _landed, make_disagg_request]
    for combo in itertools.product(range(len(makers)), repeat=mbs + prefetch):
        reqs = [makers[k](i) for i, k in enumerate(combo)]
        out = _schedule(mbs, reqs)
        assert _forward_size(out) <= mbs, combo


def test_prioritize_first_token_would_let_a_landed_request_jump_the_batch(monkeypatch):
    """Why prefetch is gated off with TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN=1."""
    monkeypatch.setenv("TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN", "1")
    landed = _landed(10)
    landed.is_generation_only_request = True
    landed.py_decoding_iter = 0
    running = make_gen_request(0)
    running.is_generation_only_request = True
    running.py_decoding_iter = 5
    out = _schedule(1, [running, landed])
    assert ids(out.generation_requests) == [10]

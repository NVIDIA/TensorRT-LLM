# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module-level performance test for the sampler update_requests path.

Measures the CPU overhead of TorchSampler.update_requests() — the #1 known
hot spot in the PyExecutor pipeline. This function runs after each forward
step and performs per-request:
  - Tensor-to-list conversion (new_tokens, finish_reasons)
  - Token appending (add_token)
  - Finish reason handling (end-of-sequence, stop words, max length)
  - Draft token processing (for speculative decoding)

The test creates a real TorchSampler instance (requires GPU for tensor
allocation) but does NOT need model weights. Synthetic requests and
pre-computed SampleStateTorch objects are used.

Scenarios:
  - Greedy decoding at BS=8 (baseline CPU overhead)
  - With stop words at BS=32 (exercises stop-word matching CUDA kernel)

Run:
    pytest tests/integration/defs/perf/host_perf/test_module_sampler.py -v -s
"""

import time
from typing import List

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.sampler import (
    SampleStateTensorsHostTorch,
    SampleStateTorch,
    TorchSampler,
)

from .regression_helper import check_regression, collect_module_result, report_latencies

# ---------------------------------------------------------------------------
# Request factory helpers
# ---------------------------------------------------------------------------


def make_request(
    request_id: int,
    seq_slot: int,
    prompt_len: int = 128,
    max_new_tokens: int = 100000,
    stop_words: list = None,
) -> LlmRequest:
    """Create a synthetic LlmRequest with a specific seq_slot."""
    sampling_config = SamplingConfig()
    req = LlmRequest(
        request_id=request_id,
        max_new_tokens=max_new_tokens,
        input_tokens=[1] * prompt_len,
        sampling_config=sampling_config,
        is_streaming=False,
    )
    req.py_seq_slot = seq_slot
    if stop_words:
        req.py_stop_words_list = stop_words
    return req


def make_generation_requests(count: int, stop_words: list = None) -> List[LlmRequest]:
    """Create a batch of generation requests with sequential seq_slots."""
    requests = []
    for i in range(count):
        req = make_request(request_id=i, seq_slot=i, stop_words=stop_words)
        req.state = LlmRequestState.GENERATION_IN_PROGRESS
        requests.append(req)
    return requests


# ---------------------------------------------------------------------------
# State factory
# ---------------------------------------------------------------------------


def make_sample_state(
    requests: List[LlmRequest],
    max_num_sequences: int,
    beam_width: int = 1,
) -> SampleStateTorch:
    """Create a SampleStateTorch with pre-computed CPU tensors.

    All tokens are set to token_id=1 (arbitrary) and finish_reasons are
    NOT_FINISHED (0), so requests remain in GENERATION_IN_PROGRESS.
    """
    max_tokens = 1  # non-speculative: 1 token per step
    # new_tokens shape: (max_tokens, max_num_sequences, beam_width)
    new_tokens = torch.ones(
        max_tokens, max_num_sequences, beam_width, dtype=torch.int, device="cpu"
    )

    # finish_reasons shape: (max_tokens, max_num_sequences, beam_width)
    # 0 = NOT_FINISHED
    finish_reasons = torch.zeros(
        max_tokens, max_num_sequences, beam_width, dtype=torch.int, device="cpu"
    )

    host = SampleStateTensorsHostTorch(
        new_tokens=new_tokens,
        finish_reasons=finish_reasons,
        first_finish_reasons=None,
        logprobs_state=None,
    )

    state = SampleStateTorch(
        requests=requests,
        host=host,
        sampler_event=None,
        beam_history_builders=None,
    )
    return state


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def benchmark_update_requests(
    sampler: TorchSampler, state: SampleStateTorch, num_iterations: int = 2000
):
    """Benchmark update_requests() and return per-call latencies in µs."""
    latencies_us = []

    # Warmup
    for _ in range(50):
        sampler.update_requests(state)

    # Benchmark
    for _ in range(num_iterations):
        start = time.perf_counter()
        sampler.update_requests(state)
        end = time.perf_counter()
        latencies_us.append((end - start) * 1e6)

    return latencies_us


# ---------------------------------------------------------------------------
# Sampler fixture
# ---------------------------------------------------------------------------

# Maximum batch size for all scenarios
MAX_NUM_SEQUENCES = 256


@pytest.fixture(scope="module")
def sampler():
    """Create a TorchSampler instance (requires GPU for tensor allocation)."""
    args = TorchSampler.Args(
        max_seq_len=4096,
        max_draft_len=0,
        max_num_sequences=MAX_NUM_SEQUENCES,
        max_beam_width=1,
        max_total_draft_tokens=0,
    )
    s = TorchSampler(args)
    return s


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

GREEDY_SCENARIOS = [
    # (name, batch_size, stop_words)
    ("greedy_bs8", 8, None),
]

STOP_WORD_SCENARIOS = [
    # 3 stop words of varying lengths, format: (flat_words, cumulative_lengths)
    # Stop words: [29871, 13], [29871, 13, 13], [29871, 13, 13, 13, 13]
    ("stopwords_bs32", 32, ([29871, 13, 29871, 13, 13, 29871, 13, 13, 13, 13], [2, 5, 10, -1])),
]


def register_requests_with_sampler(sampler: TorchSampler, requests: List[LlmRequest]):
    """Register requests with the sampler's FinishReasonsHandler.

    In production, setup_sampler_step() is called with context requests to
    initialize the FinishReasonsHandler store (max_lengths, end_ids, stop words).
    We replicate the essential setup directly to avoid ScheduledRequests C++
    binding constraints on context requests.
    """
    sampler._finish_reasons_handler.setup_new_request_handling()
    for req in requests:
        sampler._finish_reasons_handler.prepare_for_new_request(req)
        sampler._request_grouper.prepare_for_new_request(req, req.py_seq_slot)

    # Flush the temp data into the FinishReasonsHandler stores
    seq_slots = [req.py_seq_slot for req in requests]
    max_lens = sampler._finish_reasons_handler.new_max_lens
    end_ids = sampler._finish_reasons_handler.new_end_ids

    if seq_slots:
        full_list_tensor_host = torch.tensor([seq_slots, max_lens, end_ids], dtype=torch.int32)
        full_list_tensor_cuda = full_list_tensor_host.cuda()
        seq_slots_t = full_list_tensor_cuda[0]
        max_lens_t = full_list_tensor_cuda[1]
        end_ids_t = full_list_tensor_cuda[2]

        store = sampler._finish_reasons_handler.store
        store.max_lengths_cuda[seq_slots_t] = max_lens_t
        store.end_ids_cuda[seq_slots_t] = end_ids_t


@pytest.mark.parametrize(
    "scenario",
    GREEDY_SCENARIOS,
    ids=[s[0] for s in GREEDY_SCENARIOS],
)
def test_sampler_update_greedy(sampler, scenario):
    """Benchmark update_requests() with greedy decoding.

    This is the most common decode path: beam_width=1, no stop words,
    no speculative decoding. The overhead is:
    - new_tokens.tolist() + finish_reasons_list() tensor conversions
    - Per-request add_token() and _handle_finish_reasons()
    """
    name, batch_size, stop_words = scenario

    requests = make_generation_requests(batch_size, stop_words=stop_words)

    # Register requests with the sampler's internal stores
    register_requests_with_sampler(sampler, requests)

    state = make_sample_state(requests, MAX_NUM_SEQUENCES)

    latencies_us = benchmark_update_requests(sampler, state)
    median = report_latencies("SAMPLER_UPDATE_PERF", name, latencies_us)
    collect_module_result("sampler", name, latencies_us)
    check_regression("sampler", name, median)


@pytest.mark.parametrize(
    "scenario",
    STOP_WORD_SCENARIOS,
    ids=[s[0] for s in STOP_WORD_SCENARIOS],
)
def test_sampler_update_stop_words(sampler, scenario):
    """Benchmark update_requests() with stop words.

    Stop word checking is a known hot spot. The FinishReasonsHandler stores
    stop words in a CUDA tensor and checks against generated tokens each step.
    This path was optimized with a 175KB patch — regressions here would be
    significant.
    """
    name, batch_size, stop_words = scenario

    requests = make_generation_requests(batch_size, stop_words=stop_words)

    # Register requests with the sampler's internal stores
    register_requests_with_sampler(sampler, requests)

    state = make_sample_state(requests, MAX_NUM_SEQUENCES)

    latencies_us = benchmark_update_requests(sampler, state)
    median = report_latencies("SAMPLER_UPDATE_PERF", name, latencies_us)
    collect_module_result("sampler", name, latencies_us)
    check_regression("sampler", name, median)

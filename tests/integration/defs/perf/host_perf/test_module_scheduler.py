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
"""Module-level performance test for the PyExecutor scheduler.

Measures the CPU overhead of schedule_request() calls at representative batch
sizes.  This test does NOT require GPU or model weights — it creates synthetic
request objects and benchmarks the scheduler's Python-side overhead in isolation.

The scheduler is the first stage of each PyExecutor iteration and runs entirely
on the CPU.  Regressions here directly increase inter-token latency.

The test uses the **production** code path (GuaranteedNoEvictPolicy with a mock
KV cache manager) which exercises NoEvictScheduledBlocksManager with per-request
block accounting — the same path used in production serving.

Run:
    pytest tests/integration/defs/perf/host_perf/test_module_scheduler.py -v -s
"""

import math
import time
from dataclasses import dataclass
from typing import List

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import SimpleUnifiedScheduler
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

from .regression_helper import check_regression, collect_module_result, report_latencies

# ---------------------------------------------------------------------------
# Mock KV Cache Manager
# ---------------------------------------------------------------------------


@dataclass
class MockKVCacheStats:
    """Mimics the stats object returned by kv_cache_manager.get_kv_cache_stats().

    The scheduler's NoEvictScheduledBlocksManager reads
    stats.num_free_blocks_per_window_size to initialize available blocks.
    """

    num_free_blocks_per_window_size: dict


class MockKVCacheManager:
    """Mock KV cache manager that exercises GuaranteedNoEvictPolicy.

    When kv_cache_manager is not None, PyCapacityScheduler creates
    GuaranteedNoEvictPolicy instead of MaxRequestsPolicy. This mock
    provides the interface methods that GuaranteedNoEvictPolicy and
    NoEvictScheduledBlocksManager call:

    - get_kv_cache_stats() → stats with num_free_blocks_per_window_size
    - get_remaining_blocks_to_completion(req, window_size) → int
    - is_variable_window → bool (property)
    - enable_block_reuse → bool (property)

    The mock returns deterministic block counts based on request properties,
    simulating the real KV cache accounting cost (iteration over requests,
    dict lookups, arithmetic) without needing GPU memory.
    """

    def __init__(self, total_blocks: int = 4096, tokens_per_block: int = 64):
        self._total_blocks = total_blocks
        self._tokens_per_block = tokens_per_block
        # Single window size (non-VSWA) is the common case
        self._window_size = 2**31 - 1  # max_attention_window (infinite)

    @property
    def is_variable_window(self) -> bool:
        return False

    @property
    def enable_block_reuse(self) -> bool:
        return False

    def get_kv_cache_stats(self) -> MockKVCacheStats:
        return MockKVCacheStats(
            num_free_blocks_per_window_size={self._window_size: self._total_blocks}
        )

    def get_remaining_blocks_to_completion(self, req: LlmRequest, window_size: int) -> int:
        """Estimate blocks needed to complete this request.

        Mirrors the real implementation in resource_manager.py:568-581.
        This exercises the same per-request call overhead that
        NoEvictScheduledBlocksManager.decrement_reserved_blocks() incurs.
        """
        context_token_count = req.orig_prompt_len
        num_context_blocks = context_token_count // self._tokens_per_block
        remaining_tokens = (
            context_token_count + req.max_new_tokens - num_context_blocks * self._tokens_per_block
        )
        return num_context_blocks + math.ceil(remaining_tokens / self._tokens_per_block)


# ---------------------------------------------------------------------------
# Request factory helpers
# ---------------------------------------------------------------------------


def make_generation_request(request_id: int, prompt_len: int = 128) -> LlmRequest:
    """Create a synthetic LlmRequest in GENERATION_IN_PROGRESS state."""
    req = LlmRequest(
        request_id=request_id,
        max_new_tokens=128,
        input_tokens=[1] * prompt_len,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    return req


def make_context_request(request_id: int, prompt_len: int = 256) -> LlmRequest:
    """Create a synthetic LlmRequest in CONTEXT_INIT state."""
    req = LlmRequest(
        request_id=request_id,
        max_new_tokens=128,
        input_tokens=[1] * prompt_len,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    # CONTEXT_INIT is the default state after construction
    return req


def make_mixed_requests(num_gen: int, num_ctx: int, start_id: int = 0) -> List[LlmRequest]:
    """Create a mix of generation and context requests."""
    requests = []
    for i in range(num_gen):
        requests.append(make_generation_request(start_id + i))
    for i in range(num_ctx):
        requests.append(make_context_request(start_id + num_gen + i))
    return requests


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def benchmark_scheduler(scheduler, active_requests, num_iterations=2000):
    """Benchmark schedule_request() and return per-call latencies in microseconds."""
    inflight_req_ids = set()
    latencies_us = []

    # Warmup
    for _ in range(100):
        scheduler.schedule_request(active_requests, inflight_req_ids)

    # Benchmark
    for _ in range(num_iterations):
        start = time.perf_counter()
        scheduler.schedule_request(active_requests, inflight_req_ids)
        end = time.perf_counter()
        latencies_us.append((end - start) * 1e6)

    return latencies_us


# ---------------------------------------------------------------------------
# Test scenarios — production path (GuaranteedNoEvictPolicy + mock KV cache)
# ---------------------------------------------------------------------------

# (name, num_gen, num_ctx, max_batch_size, max_num_tokens, total_blocks)
PRODUCTION_SCENARIOS = [
    ("production_gen_only_bs8", 8, 0, 8, 2048, 1024),
    ("production_mixed_32gen_4ctx", 32, 4, 64, 8192, 4096),
]


@pytest.mark.parametrize(
    "scenario",
    PRODUCTION_SCENARIOS,
    ids=[s[0] for s in PRODUCTION_SCENARIOS],
)
def test_scheduler_production(scenario):
    """Benchmark scheduler with GuaranteedNoEvictPolicy (mock KV cache).

    This exercises the production code path:
    - PyCapacityScheduler sees kv_cache_manager != None → GuaranteedNoEvictPolicy
    - GuaranteedNoEvictPolicy creates NoEvictScheduledBlocksManager
    - NoEvictScheduledBlocksManager calls get_kv_cache_stats() and
      get_remaining_blocks_to_completion() for every request
    - Two-pass scheduling: generation-first, then context with block reservation

    This is significantly more expensive than MaxRequestsPolicy and represents
    the actual scheduling overhead seen in production.
    """
    name, num_gen, num_ctx, max_batch_size, max_num_tokens, total_blocks = scenario

    mock_kv_cache = MockKVCacheManager(total_blocks=total_blocks)

    scheduler = SimpleUnifiedScheduler(
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        kv_cache_manager=mock_kv_cache,
        peft_cache_manager=None,
        scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
    )

    active_requests = make_mixed_requests(num_gen, num_ctx)
    latencies_us = benchmark_scheduler(scheduler, active_requests)
    median = report_latencies("SCHEDULER_PERF", name, latencies_us)
    collect_module_result("scheduler", name, latencies_us)
    check_regression("scheduler", name, median)

    # Sanity check: with enough blocks, all requests should be scheduled
    inflight_req_ids = set()
    result = scheduler.schedule_request(active_requests, inflight_req_ids)
    total_scheduled = len(result.context_requests) + len(result.generation_requests)
    assert total_scheduled > 0, f"Scheduler produced no scheduled requests for {name}"

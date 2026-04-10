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
"""Module-level performance test for the KVCacheManager resource management path.

Measures the CPU overhead of KVCacheManager.prepare_resources() — the function
that allocates/extends KV cache blocks each iteration. The generation path
(add_token + refresh_blocks) runs every decode step and scales with batch size.

The test creates a real KVCacheManager with a C++ KVCacheManagerCpp backend
(requires GPU for memory pool allocation) but does NOT need model weights.
Small cache dimensions (4 layers, 4 KV heads, 64 head_dim) are used to minimize
GPU memory while still exercising the real C++ code path.

Scenarios:
  - Generation prepare_resources at BS=8 (steady-state decode path)
  - Context prepare_resources at BS=8 (new request allocation path)

Run:
    pytest tests/integration/defs/perf/host_perf/test_module_resource_manager.py -v -s
"""

import time
from typing import List

import pytest

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

from .regression_helper import check_regression, collect_module_result, report_latencies

CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
DataType = tensorrt_llm.bindings.DataType

# ---------------------------------------------------------------------------
# KV cache dimensions — small to minimize GPU memory (~2GB pool)
# ---------------------------------------------------------------------------
NUM_LAYERS = 4
NUM_KV_HEADS = 4
HEAD_DIM = 64
TOKENS_PER_BLOCK = 64
MAX_SEQ_LEN = 4096
MAX_BATCH_SIZE = 128
# max_tokens controls pool size:
# bytes_per_token = NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * 2(k+v) * 2(FP16) = 4096
# 262144 tokens * 4096 bytes = ~1 GB GPU memory
MAX_TOKENS = 262144
PROMPT_LEN = 128


# ---------------------------------------------------------------------------
# Request factory helpers
# ---------------------------------------------------------------------------


def make_request(
    request_id: int, seq_slot: int, prompt_len: int = PROMPT_LEN, max_new_tokens: int = 100000
) -> LlmRequest:
    """Create a synthetic LlmRequest."""
    sampling_config = SamplingConfig()
    req = LlmRequest(
        request_id=request_id,
        max_new_tokens=max_new_tokens,
        input_tokens=[1] * prompt_len,
        sampling_config=sampling_config,
        is_streaming=False,
    )
    req.py_seq_slot = seq_slot
    return req


def make_scheduled_requests(
    generation_requests: List[LlmRequest] = None,
    context_requests: List[LlmRequest] = None,
) -> ScheduledRequests:
    """Create a ScheduledRequests object."""
    sr = ScheduledRequests()
    for req in generation_requests or []:
        sr.generation_requests.append(req)
    for req in context_requests or []:
        sr.context_requests.append(req)
    return sr


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def benchmark_callable(fn, num_iterations: int = 2000, warmup: int = 50):
    """Benchmark a callable and return per-call latencies in µs."""
    latencies_us = []

    for _ in range(warmup):
        fn()

    for _ in range(num_iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        latencies_us.append((end - start) * 1e6)

    return latencies_us


# ---------------------------------------------------------------------------
# KVCacheManager fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kv_cache_manager():
    """Create a KVCacheManager with a real C++ backend.

    Uses small dimensions to minimize GPU memory (~1 GB) while still
    exercising the real add_sequence/add_token/refresh_blocks code path.
    """
    kv_cache_config = KvCacheConfig(
        max_tokens=MAX_TOKENS,
        enable_block_reuse=False,
    )
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        mapping=mapping,
        dtype=DataType.HALF,
        is_estimating_kv_cache=True,
    )

    yield manager

    manager.shutdown()


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


def test_kv_cache_prepare_generation(kv_cache_manager):
    """Benchmark prepare_resources() for the generation (decode) path at BS=8.

    This is the steady-state path that runs every decode iteration:
    - impl.sync_transfer_manager_with_buffer_manager() (sync check)
    - impl.add_token(request_id) per generation request
    - impl.refresh_blocks() to finalize allocations

    The cost scales linearly with batch size. A regression in the C++
    add_token or refresh_blocks implementation would be caught here.
    """
    name = "gen_bs8"
    batch_size = 8

    # Use request IDs offset by 10000 to avoid conflicts across tests
    base_id = 10000
    requests = []
    for i in range(batch_size):
        req_id = base_id + i
        req = make_request(request_id=req_id, seq_slot=i, prompt_len=PROMPT_LEN)
        # Add sequence to KV cache (context phase)
        kv_cache_manager.impl.add_sequence(req_id, PROMPT_LEN, 1, req)
        req.state = LlmRequestState.GENERATION_IN_PROGRESS
        requests.append(req)

    gen_sr = make_scheduled_requests(generation_requests=requests)

    latencies_us = benchmark_callable(
        lambda: kv_cache_manager.prepare_resources(gen_sr),
        num_iterations=2000,
    )
    median = report_latencies("KV_CACHE_PERF", name, latencies_us)
    collect_module_result("kv_cache", name, latencies_us)
    check_regression("kv_cache", name, median)

    # Cleanup: remove sequences so other tests can reuse the pool
    for req in requests:
        kv_cache_manager.impl.remove_sequence(req.request_id, req, False)


def test_kv_cache_prepare_context(kv_cache_manager):
    """Benchmark the context (new request) allocation path at BS=8.

    When a new request enters the system, add_sequence is called to
    allocate initial KV cache blocks for the prompt. This is more
    expensive than add_token but happens only once per request.

    Each iteration allocates and then frees sequences to measure the
    allocation/deallocation cycle cost.
    """
    name = "ctx_bs8"
    batch_size = 8

    base_id = 20000
    num_iterations = 500  # Fewer iterations since alloc/free is heavier

    latencies_us = []

    # Warmup
    for warmup_iter in range(20):
        reqs = []
        for i in range(batch_size):
            req_id = base_id + warmup_iter * batch_size + i
            req = make_request(request_id=req_id, seq_slot=i, prompt_len=PROMPT_LEN)
            kv_cache_manager.impl.add_sequence(req_id, PROMPT_LEN, 1, req)
            reqs.append(req)
        kv_cache_manager.impl.refresh_blocks()
        for req in reqs:
            kv_cache_manager.impl.remove_sequence(req.request_id, req, False)

    # Benchmark
    for bench_iter in range(num_iterations):
        reqs = []
        for i in range(batch_size):
            req_id = base_id + (20 + bench_iter) * batch_size + i
            req = make_request(request_id=req_id, seq_slot=i, prompt_len=PROMPT_LEN)
            reqs.append(req)

        start = time.perf_counter()
        for req in reqs:
            kv_cache_manager.impl.add_sequence(req.request_id, PROMPT_LEN, 1, req)
        kv_cache_manager.impl.refresh_blocks()
        end = time.perf_counter()

        latencies_us.append((end - start) * 1e6)

        # Free sequences for next iteration
        for req in reqs:
            kv_cache_manager.impl.remove_sequence(req.request_id, req, False)

    median = report_latencies("KV_CACHE_PERF", name, latencies_us)
    collect_module_result("kv_cache", name, latencies_us)
    check_regression("kv_cache", name, median)

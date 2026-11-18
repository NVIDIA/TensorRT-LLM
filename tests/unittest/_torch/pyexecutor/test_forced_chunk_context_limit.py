# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context-length limit check of the micro-batch schedulers under FORCE_CHUNK.

Under the FORCE_CHUNK policy the micro-batch schedulers require the context
length limit (``max_num_tokens``) to be at least one chunk unit and fail every
context request otherwise. Both schedulers reject that limit when they are
created, and so does ``SimpleUnifiedScheduler``, which normalizes the policy
before it builds its ``PyMicroBatchScheduler``.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    BindMicroBatchScheduler,
    ChunkingPolicy,
    ContextChunkingConfig,
    PyMicroBatchScheduler,
    SimpleUnifiedScheduler,
)
from tensorrt_llm.bindings.internal.batch_manager import ReqIdsSet
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy, ContextChunkingPolicy

pytestmark = pytest.mark.cpu_only

UNIT = 32
LIMIT_ERROR = r"max_num_tokens \(\d+\) must be at least the FORCE_CHUNK context chunk unit \(32"


def _context_request(request_id: int, prompt_len: int) -> LlmRequest:
    req = LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    req.state = LlmRequestState.CONTEXT_INIT
    return req


def _py_scheduler(max_num_tokens, policy=ChunkingPolicy.FORCE_CHUNK):
    return PyMicroBatchScheduler(
        max_batch_size=4,
        max_num_tokens=max_num_tokens,
        ctx_chunk_config=ContextChunkingConfig(policy, chunk_unit_size=UNIT),
    )


def _bind_scheduler(max_num_tokens, policy=ContextChunkingPolicy.FORCE_CHUNK):
    return BindMicroBatchScheduler(
        max_batch_size=4, max_num_tokens=max_num_tokens, ctx_chunk_config=(policy, UNIT)
    )


def _unified_scheduler(max_num_tokens, policy=ContextChunkingPolicy.FORCE_CHUNK):
    # The executor's scheduler_config.use_python_scheduler path: the policy is
    # normalized here before the inner PyMicroBatchScheduler is built.
    return SimpleUnifiedScheduler(
        max_batch_size=4,
        max_num_tokens=max_num_tokens,
        kv_cache_manager=None,
        peft_cache_manager=None,
        scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        ctx_chunk_config=(policy, UNIT),
    )


SCHEDULERS = [_py_scheduler, _bind_scheduler, _unified_scheduler]


@pytest.mark.parametrize("make_scheduler", SCHEDULERS)
@pytest.mark.parametrize("max_num_tokens", [1, 16, UNIT - 1])
def test_limit_below_one_unit_is_rejected(make_scheduler, max_num_tokens):
    with pytest.raises(ValueError, match=LIMIT_ERROR):
        make_scheduler(max_num_tokens)


@pytest.mark.parametrize("make_scheduler", SCHEDULERS)
@pytest.mark.parametrize("max_num_tokens", [None, UNIT, 8192])
def test_limit_of_at_least_one_unit_is_accepted(make_scheduler, max_num_tokens):
    make_scheduler(max_num_tokens)


@pytest.mark.parametrize(
    "make_scheduler, policy",
    [
        (_py_scheduler, ChunkingPolicy.FIRST_COME_FIRST_SERVED),
        (_py_scheduler, ChunkingPolicy.EQUAL_PROGRESS),
        (_bind_scheduler, ContextChunkingPolicy.FIRST_COME_FIRST_SERVED),
        (_bind_scheduler, ContextChunkingPolicy.EQUAL_PROGRESS),
        (_unified_scheduler, ContextChunkingPolicy.FIRST_COME_FIRST_SERVED),
        (_unified_scheduler, ContextChunkingPolicy.EQUAL_PROGRESS),
    ],
)
def test_other_policies_keep_small_limits(make_scheduler, policy):
    make_scheduler(16, policy)


@pytest.mark.parametrize(
    "policy, expected",
    [
        (ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, ChunkingPolicy.FIRST_COME_FIRST_SERVED),
        (ContextChunkingPolicy.EQUAL_PROGRESS, ChunkingPolicy.EQUAL_PROGRESS),
        (ContextChunkingPolicy.FORCE_CHUNK, ChunkingPolicy.FORCE_CHUNK),
    ],
)
def test_unified_scheduler_keeps_the_configured_policy(policy, expected):
    # A policy mapped to FIRST_COME_FIRST_SERVED by mistake would pass the limit
    # cases above, so the mapping itself is asserted.
    scheduler = _unified_scheduler(8192, policy)
    assert scheduler.micro_batch_scheduler.ctx_chunk_config.chunking_policy is expected


@pytest.mark.parametrize("make_scheduler", [_py_scheduler, _bind_scheduler])
@pytest.mark.parametrize("prompt_len", [2, UNIT, 3 * UNIT])
def test_smallest_accepted_limit_schedules_context(make_scheduler, prompt_len):
    scheduler = make_scheduler(UNIT)
    _, context_requests, _ = scheduler.schedule([_context_request(0, prompt_len)], ReqIdsSet())
    assert len(context_requests) == 1
    assert context_requests[0].context_chunk_size == min(prompt_len, UNIT)

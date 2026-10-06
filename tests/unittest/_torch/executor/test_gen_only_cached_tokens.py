# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cached_tokens`` for disaggregated generation-only requests.

A generation-only request never takes the context branch of
``_prepare_tp_inputs`` on the generation worker, so
``PyExecutor._prepare_disagg_gen_resources`` latches ``cached_tokens`` from
``prepopulated_prompt_len`` right after allocation. Driven by a real
``LlmRequest`` so the C++ ``prepopulated_prompt_len`` and the Python set-once
property are exercised together.
"""

from types import SimpleNamespace

import pytest
from _torch.executor.llm_request_factory import make_llm_request

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import executor as trtllm

pytestmark = pytest.mark.cpu_only

PROMPT = list(range(1, 17))
TOKENS_PER_BLOCK = 4


class _FakeKvCacheManager:
    """``prepare_resources`` is where ``addSequence`` settles the prefix this
    worker already holds; only that side effect matters here."""

    def __init__(self, reuse: int):
        self.reuse = reuse

    def prepare_resources(self, scheduled_batch: ScheduledRequests) -> None:
        for req in scheduled_batch.context_requests_last_chunk:
            req.set_prepopulated_prompt_len(self.reuse, TOKENS_PER_BLOCK)


def _make_executor(kv_cache_manager) -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.resource_manager = SimpleNamespace(
        resource_managers={ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager}
    )
    return executor


@pytest.mark.parametrize("reuse", [0, 8], ids=["cold_cache", "local_prefix_hit"])
def test_gen_init_latches_the_locally_reused_prefix(reuse: int) -> None:
    req = make_llm_request(
        1, trtllm.RequestType.REQUEST_TYPE_GENERATION_ONLY, input_token_ids=PROMPT
    )
    executor = _make_executor(_FakeKvCacheManager(reuse))

    executor._prepare_disagg_gen_resources([req])

    assert req.prepopulated_prompt_len == reuse
    assert req.cached_tokens == reuse

    # What a decode step used to write: the sequence length, which for a
    # generation-only request is the whole prompt. It must bounce off the latch.
    req.cached_tokens = req.max_beam_num_tokens - 1
    assert req.cached_tokens == reuse

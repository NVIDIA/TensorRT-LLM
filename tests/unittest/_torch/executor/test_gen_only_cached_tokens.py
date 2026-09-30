# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cached_tokens`` for disaggregated generation-only requests.

``LlmRequest.cached_tokens`` is latched once per request. A context request
latches it on its first context-phase forward, from the prefix the KV cache
manager matched. A generation-only request never takes that branch on the
generation worker, so ``PyExecutor._prepare_disagg_gen_resources`` latches it
right after the allocation that decided how much of the prompt this worker
serves from its own cache instead of receiving. The sequence length a decode
step could write afterwards must bounce off the latch.

Driven by a real ``LlmRequest`` so the C++ ``prepopulated_prompt_len`` and the
Python set-once property are exercised together.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, executor_request_to_llm_request
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import executor as trtllm

pytestmark = pytest.mark.cpu_only

PROMPT = list(range(1, 17))
TOKENS_PER_BLOCK = 4


def _make_gen_only_request(request_id: int = 1) -> LlmRequest:
    # A generation-only request always arrives from a context server, so it
    # carries the first generated token in its context phase params.
    executor_request = trtllm.Request(
        input_token_ids=PROMPT,
        max_tokens=8,
        type=trtllm.RequestType.REQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=trtllm.ContextPhaseParams([100], request_id, None, None, None, None),
    )
    return executor_request_to_llm_request(
        request_id,
        executor_request,
        child_req_ids=[],
        exclude_last_generation_logits=False,
    )


class _FakeKvCacheManager:
    """``prepare_resources`` is where ``addSequence`` settles the prefix this
    worker already holds; only that side effect matters here."""

    def __init__(self, reuse: int):
        self.reuse = reuse
        self.prepared: list[int] = []

    def prepare_resources(self, scheduled_batch: ScheduledRequests) -> None:
        for req in scheduled_batch.context_requests_last_chunk:
            self.prepared.append(req.py_request_id)
            if self.reuse:
                req.set_prepopulated_prompt_len(self.reuse, TOKENS_PER_BLOCK)


def _make_executor(kv_cache_manager) -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.resource_manager = SimpleNamespace(
        resource_managers={ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager}
    )
    return executor


def _write_decode_position(req: LlmRequest) -> None:
    # The value the generation branch of ``_prepare_tp_inputs`` derives its
    # KV length from. For a generation-only request it is the whole prompt.
    req.cached_tokens = req.max_beam_num_tokens - 1


@pytest.mark.parametrize("reuse", [0, 8], ids=["cold_cache", "local_prefix_hit"])
def test_gen_init_latches_the_locally_reused_prefix(reuse: int) -> None:
    req = _make_gen_only_request()
    manager = _FakeKvCacheManager(reuse)
    executor = _make_executor(manager)

    executor._prepare_disagg_gen_resources([req])

    assert manager.prepared == [req.py_request_id]
    assert req.prepopulated_prompt_len == reuse
    assert req.cached_tokens == reuse

    _write_decode_position(req)
    assert req.cached_tokens == reuse, "a decode step must not overwrite the latched prefix"


def test_unlatched_gen_only_request_would_report_the_whole_prompt() -> None:
    """The failure mode the gen-init latch closes: without it the first decode
    step is the first write, and the sequence length becomes the cache hit."""
    req = _make_gen_only_request()
    assert req.cached_tokens == 0

    _write_decode_position(req)

    assert req.cached_tokens == len(PROMPT) - 1

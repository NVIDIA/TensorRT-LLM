# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache hits of a request that the executor pauses and computes again.

The recompute prefill reuses KV cache blocks that the request computed itself,
so its prefix hit says nothing about the client's prompt: usage reports the
first prefill's hit. The KV cache-aware attention-DP router weighs the request
by the prefill it runs now, so it uses the recompute's hit.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from _torch.pyexecutor.llm_request_factory import make_llm_request

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import KVCacheAwareADPRouter
from tensorrt_llm.bindings import executor as trtllm

pytestmark = pytest.mark.cpu_only

PROMPT = list(range(1, 17))
GENERATED = [100, 101, 102]
TOKENS_PER_BLOCK = 4


def _prefill(req, reuse: int) -> None:
    """What the KV cache manager and ``DecoderInputPreparer`` do at a prefill."""
    req.set_prepopulated_prompt_len(reuse, TOKENS_PER_BLOCK)
    req.cached_tokens = req.context_current_position


def _router():
    dist = MagicMock(tp_rank=0, has_cp_helix=False)
    dist.mapping.dp_size = 1
    return KVCacheAwareADPRouter(dist=dist, kv_cache_manager=MagicMock())


def _recomputed_request(first_reuse: int, streaming: bool = False):
    req = make_llm_request(
        1,
        trtllm.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION,
        input_token_ids=PROMPT,
        streaming=streaming,
    )
    _prefill(req, first_reuse)
    for token in GENERATED:
        req.add_new_token(token, 0)

    PyExecutor._pause_recompute_request(SimpleNamespace(), req)
    assert req.prompt_len == len(PROMPT) + len(GENERATED)

    # The recompute prefill finds every full block of the prompt and the
    # generated tokens in the cache.
    _prefill(req, len(PROMPT))
    return req


@pytest.mark.parametrize("first_reuse", [0, 8], ids=["cold_cache", "prefix_hit"])
def test_recompute_reports_the_first_prefill_cached_tokens(first_reuse: int) -> None:
    req = _recomputed_request(first_reuse)

    assert req.reported_cached_tokens == first_reuse
    assert req.cached_tokens == len(PROMPT)


def _first_token_after_recompute():
    """A streaming request, prefix hit 8, at its first token after the recompute prefill."""
    req = _recomputed_request(8, streaming=True)
    req.add_new_token(GENERATED[-1] + 1, 0)
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.py_decoding_iter = 1
    assert req.cached_tokens == len(PROMPT)
    return req


def test_recompute_response_reports_the_first_prefill_cached_tokens() -> None:
    response = _first_token_after_recompute().create_response()

    assert response.result.cached_tokens == 8


@pytest.mark.parametrize("early_first_token", [False, True], ids=["handle_responses", "early"])
def test_executor_response_reports_the_first_prefill_cached_tokens(
    early_first_token: bool,
) -> None:
    req = _first_token_after_recompute()
    executor = object.__new__(PyExecutor)
    executor.active_requests = [req]
    executor.perf_manager = MagicMock()
    executor.iter_counter = 1
    executor.stream_interval = 1
    executor.first_content_emit_iters = 1
    executor.disable_overlap_scheduler = True
    executor.should_exclude_last_generation_logits = False
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.kv_cache_transceiver = None
    executor.enable_attention_dp = False
    executor._enqueue_responses = MagicMock()

    if early_first_token:
        executor._emit_first_token_responses(SimpleNamespace(all_requests=lambda: [req]))
    else:
        executor._handle_responses()

    ((request_id, response),) = executor._enqueue_responses.call_args.args[0]
    assert request_id == req.py_request_id
    assert response.result.cached_tokens == 8


@pytest.mark.parametrize("first_reuse", [0, 8], ids=["cold_cache", "prefix_hit"])
def test_kv_cache_aware_router_counts_the_recompute_prefill(first_reuse: int) -> None:
    req = _recomputed_request(first_reuse)

    state = _router().create_rank_state([req], [])

    assert state.num_active_tokens == len(GENERATED)


def test_kv_cache_aware_router_counts_a_prefill_hit_once_the_prefill_runs() -> None:
    req = make_llm_request(
        1, trtllm.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION, input_token_ids=PROMPT
    )
    router = _router()

    # The KV cache manager matched a prefix, but the prefill has not run:
    # the request may still leave the batch and match again on its next
    # admission, so the hit is not counted yet.
    req.set_prepopulated_prompt_len(8, TOKENS_PER_BLOCK)
    assert router.create_rank_state([req], []).num_active_tokens == len(PROMPT)

    req.cached_tokens = req.context_current_position
    assert router.create_rank_state([req], []).num_active_tokens == len(PROMPT) - 8

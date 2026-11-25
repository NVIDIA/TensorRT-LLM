# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import json
import time
from typing import Any, Dict, Iterator

import pytest
import torch
from test_common.session_reuse import REUSE
from utils.util import skip_ray

from tensorrt_llm import LLM
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.llmapi import KvCacheConfig, RequestOutput
from tensorrt_llm.sampling_params import SamplingParams

from .test_llm import (check_llm_capture_request_error, check_llm_get_stats,
                       check_llm_get_stats_async, check_llm_logits_processor,
                       drain_llm_stats, get_model_path, llama_model_path,
                       llm_get_stats_test_harness,
                       llm_return_logprobs_test_harness,
                       pp_sync_sample_state_env,
                       tinyllama_logits_processor_test_harness)

global_kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)


@contextlib.contextmanager
def _shared_llm(request: pytest.FixtureRequest,
                **llm_kwargs: Any) -> Iterator[LLM]:
    """Build an LLM that the tests of one class share.

    The class-scoped fixtures that use it shut the LLM down before the next
    class starts, so its worker pool returns to the session-reuse cache in
    time for the next LLM. A failed test drains that cache once the LLM is
    shut down, as the per-test failure fence does for an LLM a test owns.
    """
    failed_before = request.session.testsfailed
    try:
        with LLM(model=llama_model_path, **llm_kwargs) as llm:
            yield llm
    finally:
        if request.session.testsfailed > failed_before:
            REUSE.drain()


@skip_ray
@pytest.mark.gpu2
@pytest.mark.private_mpi_session
def test_llm_rpc_tp2() -> None:
    # RPC builds rank 0's engine on another thread. Reusing IPC workers would
    # leave asymmetric thread-local allreduce caches across the ranks.
    # Defined first: its private pool then starts before session reuse
    # prefetches any pool in the background.
    with LLM(model=llama_model_path,
             kv_cache_config=global_kv_cache_config,
             orchestrator_type="rpc",
             tensor_parallel_size=2) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)
        sampling_params = SamplingParams(max_tokens=10, end_id=-1)

        res = llm.generate("Tell me a joke", sampling_params=sampling_params)
        print(f"get result: {res}")
        assert len(res.outputs) == 1
        assert len(res.outputs[0].token_ids) == 10

        async def generate_async() -> list[RequestOutput]:
            outputs = []
            async for output in llm.generate_async(
                    "Tell me a joke", sampling_params=sampling_params):
                print(f"get result: {output}")
                outputs.append(output)
            return outputs

        outputs = asyncio.run(generate_async())
        assert outputs
        assert len(outputs[-1].outputs[0].token_ids) == 10


@pytest.mark.gpu2
class TestTp2:
    """Checks that tolerate earlier requests, on one default TP2 LLM."""

    @pytest.fixture(scope="class")
    def llm(self, request: pytest.FixtureRequest) -> Iterator[LLM]:
        with _shared_llm(request,
                         kv_cache_config=global_kv_cache_config,
                         tensor_parallel_size=2) as llm:
            # Start the threads of the request path before the tests, so
            # the threadleak check of the first test does not count them.
            llm.generate("A B C", SamplingParams(max_tokens=1))
            yield llm

    @pytest.mark.part0
    def test_logits_processor(self, llm: LLM) -> None:
        check_llm_logits_processor(llm)

    def test_capture_request_error(self, llm: LLM) -> None:
        check_llm_capture_request_error(llm)


@pytest.mark.gpu2
@pytest.mark.part0
def test_tinyllama_logits_processor_pp2() -> None:
    tinyllama_logits_processor_test_harness(backend="pytorch",
                                            pipeline_parallel_size=2)


@pytest.mark.gpu4
def test_tinyllama_logits_processor_tp2pp2():
    tinyllama_logits_processor_test_harness(backend="pytorch",
                                            tensor_parallel_size=2,
                                            pipeline_parallel_size=2)


@skip_ray
@pytest.mark.gpu2
# The class LLM's request threads start in whichever test sends its first
# request and live until the class ends.
@pytest.mark.threadleak(enabled=False)
class TestTp2IterStats:
    """get_stats and get_stats_async on one TP2 LLM with iteration stats.

    test_get_stats_async must issue the LLM's first request, so it is
    defined first and the fixture sends no warmup request.
    """

    @pytest.fixture(scope="class")
    def llm(self, request: pytest.FixtureRequest) -> Iterator[LLM]:
        with _shared_llm(request,
                         kv_cache_config=global_kv_cache_config,
                         tensor_parallel_size=2,
                         enable_iter_perf_stats=True,
                         disable_overlap_scheduler=True) as llm:
            yield llm

    def test_get_stats_async(self, llm: LLM) -> None:
        check_llm_get_stats_async(llm, pytorch_backend=True)

    def test_get_stats(self, llm: LLM) -> None:
        drain_llm_stats(llm)
        check_llm_get_stats(llm, pytorch_backend=True)


@skip_ray
@pytest.mark.gpu2
# The class LLM's request threads start in whichever test sends its first
# request and live until the class ends.
@pytest.mark.threadleak(enabled=False)
class TestPp2IterStats:
    """get_stats and get_stats_async on one PP2 LLM with request stats.

    test_get_stats_async must issue the LLM's first request, so it is
    defined first and the fixture sends no warmup request.
    """

    @pytest.fixture(scope="class")
    def llm(self, request: pytest.FixtureRequest) -> Iterator[LLM]:
        with pp_sync_sample_state_env(2), _shared_llm(
                request,
                kv_cache_config=global_kv_cache_config,
                pipeline_parallel_size=2,
                enable_iter_perf_stats=True,
                enable_iter_req_stats=True,
                disable_overlap_scheduler=True) as llm:
            yield llm

    def test_get_stats_async(self, llm: LLM) -> None:
        check_llm_get_stats_async(llm,
                                  pp_size=2,
                                  pytorch_backend=True,
                                  enable_iter_req_stats=True)

    def test_get_stats(self, llm: LLM) -> None:
        drain_llm_stats(llm)
        check_llm_get_stats(llm,
                            pp_size=2,
                            pytorch_backend=True,
                            enable_iter_req_stats=True,
                            microbatch_base=None)


@skip_ray
@pytest.mark.gpu2
def test_llm_get_stats_pp2_chunked_prefill() -> None:
    llm_get_stats_test_harness(tp_size=1,
                               pp_size=2,
                               pytorch_backend=True,
                               enable_chunked_prefill=True,
                               enable_iter_req_stats=True)


@skip_ray
@pytest.mark.gpu4
@pytest.mark.parametrize(
    "return_context_logits, enable_chunked_prefill, enable_iter_req_stats",
    [
        (False, False, True),
        (False, True, True),
    ],
)
def test_llm_get_stats_pp4(return_context_logits, enable_chunked_prefill,
                           enable_iter_req_stats):
    llm_get_stats_test_harness(
        tp_size=1,
        pp_size=4,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_iter_req_stats=enable_iter_req_stats,
    )


@skip_ray
@pytest.mark.gpu2
@pytest.mark.parametrize(
    "prompt_logprobs, logprobs, return_context_logits, return_generation_logits",
    [
        (None, 1, False,
         False),  # generation logprobs only (top-1, PyTorch limit)
    ])
def test_llm_return_logprobs_streaming_tp2(prompt_logprobs, logprobs,
                                           return_context_logits,
                                           return_generation_logits):
    llm_return_logprobs_test_harness(prompt_logprobs,
                                     logprobs,
                                     return_context_logits,
                                     return_generation_logits,
                                     streaming=True,
                                     backend="pytorch",
                                     tp_size=2)


def _context_requests_per_rank(llm: LLM) -> Dict[int, int]:
    """Context requests each attention-DP rank scheduled since the last call.

    One per request (no chunking), from the iteration stats that arrived.

    Args:
        llm: An attention-DP LLM with iteration stats on.

    Returns:
        Attention-DP rank -> context requests it ran.
    """
    time.sleep(2)  # as llm_get_stats_test_harness: let the stats arrive
    counts: Dict[int, int] = {}
    for stats in llm.get_stats(2):
        stats = json.loads(stats) if isinstance(stats, str) else stats
        rank = stats["attentionDpRank"]
        counts[rank] = counts.get(
            rank, 0) + stats["inflightBatchingStats"]["numContextRequests"]
    return {rank: n for rank, n in counts.items() if n}


@skip_ray
@pytest.mark.gpu2
def test_llm_attention_dp_home_wait_tp2():
    """Home-wait runs every turn of a conversation on its home rank.

    Conversation-affinity routing with home-wait on 2-GPU attention DP.

    With one request slot per rank, later turns of a conversation wait for
    their full home rank (the rank its first turn ran on), even while the
    other rank is free: the per-rank context requests of the turns are three
    per conversation homed there. Each turn completes with the same greedy
    output as its first turn.
    """
    from tensorrt_llm.conversation_params import ConversationParams
    from tensorrt_llm.llmapi import AttentionDpConfig

    prompts = {
        "conv-a": "The capital of France is",
        "conv-b": "One, two, three, four,"
    }
    params = SamplingParams(max_tokens=8, temperature=0, end_id=-1)
    with LLM(
            model=llama_model_path,
            kv_cache_config=global_kv_cache_config,
            tensor_parallel_size=2,
            enable_attention_dp=True,
            max_batch_size=1,
            enable_iter_perf_stats=True,
            attention_dp_config=AttentionDpConfig(
                kv_cache_routing_conversation_affinity=True,
                # More rounds than the six turns take on one rank, so that
                # no turn overflows to the free rank.
                kv_cache_routing_home_wait_rounds=1000)) as llm:
        _context_requests_per_rank(llm)  # drop any warmup rows
        first, home = {}, {}
        for conv, prompt in prompts.items():
            first[conv] = llm.generate(
                prompt,
                params,
                conversation_params=ConversationParams(
                    conversation_id=conv)).outputs[0].token_ids
            ranks = list(_context_requests_per_rank(llm))
            assert len(ranks) == 1, (conv, ranks)
            home[conv] = ranks[0]
        convs = [conv for conv in prompts for _ in range(3)]
        results = llm.generate([prompts[conv] for conv in convs],
                               params,
                               conversation_params=[
                                   ConversationParams(conversation_id=conv)
                                   for conv in convs
                               ])
        per_rank = _context_requests_per_rank(llm)
    for conv, result in zip(convs, results):
        assert len(first[conv]) == 8
        assert result.outputs[0].token_ids == first[conv], conv
    expected: Dict[int, int] = {}
    for conv in convs:
        expected[home[conv]] = expected.get(home[conv], 0) + 1
    assert per_rank == expected, (home, per_rank)


@skip_ray
@pytest.mark.gpu2
def test_scheduled_pooling_tp2():
    llm_kwargs = dict(model=get_model_path("Qwen3/Qwen3-Embedding-0.6B"),
                      model_kwargs={"architectures": ["Qwen3ForTextEmbedding"]},
                      max_seq_len=2048,
                      max_batch_size=1)
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ]
    params = SamplingParams(max_tokens=1, return_context_logits=True)

    with LLM(**llm_kwargs, tensor_parallel_size=2) as llm:
        sharded = [
            output.context_logits.float().cpu()
            for output in llm.generate(prompts, sampling_params=params)
        ]
        futures = [
            llm.generate_async(prompts[i % len(prompts)],
                               sampling_params=params) for i in range(50)
        ]
        futures[-1].abort()
        finished = [
            future.result(timeout=60).outputs[0].finish_reason
            for future in futures
        ]
        # A forward after the abort runs on both ranks only if they dropped the same request.
        after_abort = llm.generate(prompts[:1], sampling_params=params)
    with LLM(**llm_kwargs) as llm:
        single = [
            output.context_logits.float().cpu()
            for output in llm.generate(prompts, sampling_params=params)
        ]

    # The abort may land before or after the request's forward.
    assert finished[:-1] == ["length"] * 49
    assert finished[-1] in ("cancelled", "length")
    sharded.append(after_abort[0].context_logits.float().cpu())
    for got, expected in zip(sharded, single + single[:1]):
        assert torch.nn.functional.cosine_similarity(got.flatten(),
                                                     expected.flatten(),
                                                     dim=0) > 0.999

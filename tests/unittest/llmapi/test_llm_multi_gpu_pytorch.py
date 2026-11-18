# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
from typing import Dict

import pytest
import torch
from utils.util import skip_ray

from tensorrt_llm import LLM
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

from .test_llm import (_test_llm_capture_request_error, get_model_path,
                       llama_model_path, llm_get_stats_async_test_harness,
                       llm_get_stats_test_harness,
                       llm_return_logprobs_test_harness,
                       tinyllama_logits_processor_test_harness)

global_kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)


@pytest.mark.gpu2
def test_llm_capture_request_error():
    _test_llm_capture_request_error(pytorch_backend=True, tp_size=2)


@pytest.mark.gpu4
def test_tinyllama_logits_processor_tp2pp2():
    tinyllama_logits_processor_test_harness(backend="pytorch",
                                            tensor_parallel_size=2,
                                            pipeline_parallel_size=2)


@pytest.mark.gpu2
@pytest.mark.part0
@pytest.mark.parametrize("tp_size, pp_size", [(1, 2), (2, 1)])
def test_tinyllama_logits_processor_2gpu(tp_size: int, pp_size: int):
    tinyllama_logits_processor_test_harness(backend="pytorch",
                                            tensor_parallel_size=tp_size,
                                            pipeline_parallel_size=pp_size)


@skip_ray
@pytest.mark.gpu2
@pytest.mark.private_mpi_session
def test_llm_rpc_tp2() -> None:
    # RPC builds rank 0's engine on another thread. Reusing IPC workers would
    # leave asymmetric thread-local allreduce caches across the ranks.
    with LLM(model=llama_model_path,
             kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
             orchestrator_type="rpc",
             tensor_parallel_size=2) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        res = llm.generate("Tell me a joke",
                           sampling_params=SamplingParams(max_tokens=10,
                                                          end_id=-1))
        print(f"get result: {res}")

        assert len(res.outputs) == 1
        assert len(res.outputs[0].token_ids) == 10


@skip_ray
@pytest.mark.gpu2
@pytest.mark.asyncio
@pytest.mark.private_mpi_session
async def test_llm_rpc_streaming_tp2() -> None:
    with LLM(model=llama_model_path,
             kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
             orchestrator_type="rpc",
             tensor_parallel_size=2) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        async for output in llm.generate_async("Tell me a joke",
                                               sampling_params=SamplingParams(
                                                   max_tokens=10, end_id=-1)):
            print(f"get result: {output}")


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


@skip_ray
@pytest.mark.gpu2
@pytest.mark.parametrize(
    "return_context_logits, enable_chunked_prefill, enable_iter_req_stats",
    [
        (False, False, True),
        (False, True, True),
    ],
)
def test_llm_get_stats_pp2(return_context_logits, enable_chunked_prefill,
                           enable_iter_req_stats):
    llm_get_stats_test_harness(
        tp_size=1,
        pp_size=2,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_iter_req_stats=enable_iter_req_stats,
    )


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
def test_llm_get_stats_tp2():
    llm_get_stats_test_harness(tp_size=2, pytorch_backend=True)


@skip_ray
@pytest.mark.gpu2
def test_llm_get_stats_async_tp2():
    llm_get_stats_async_test_harness(tp_size=2, pytorch_backend=True)


@skip_ray
@pytest.mark.gpu2
def test_llm_get_stats_async_pp2():
    llm_get_stats_async_test_harness(pp_size=2, pytorch_backend=True)


@pytest.mark.gpu2
@pytest.mark.parametrize(
    "compare_pp1, async_relay, speculative_d2h",
    [
        pytest.param(True, False, False, id="pp1_reference"),
        pytest.param(False, False, False, id="default"),
        pytest.param(False, False, True, id="inline_speculative"),
        pytest.param(False, True, True, id="async_speculative"),
    ],
)
def test_llm_beam_search_pp2(compare_pp1: bool, async_relay: bool,
                             speculative_d2h: bool,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Beam search under pipeline parallelism returns every finalized beam.

    Only the last pipeline-parallel rank samples; the first rank, which returns
    the results, finalizes the beams from the state that rank hands over.
    """
    beam_width = 4
    max_tokens = 24
    prompts = ["A B C", "The future of the AI is"]
    monkeypatch.setenv("TLLM_PP_ASYNC_BROADCAST_SAMPLE_STATE",
                       "1" if async_relay else "0")
    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     n=beam_width,
                                     best_of=beam_width,
                                     use_beam_search=True,
                                     logprobs=True)
    llm_kwargs = dict(max_batch_size=len(prompts))
    reference_outputs = None
    if compare_pp1:
        llm_kwargs.update(max_batch_size=1,
                          disable_overlap_scheduler=True,
                          cuda_graph_config=None)
        with LLM(model=llama_model_path,
                 kv_cache_config=global_kv_cache_config,
                 pipeline_parallel_size=1,
                 max_beam_width=beam_width,
                 **llm_kwargs) as llm:
            reference_outputs = llm.generate(prompts,
                                             sampling_params=sampling_params)
    with LLM(model=llama_model_path,
             kv_cache_config=global_kv_cache_config,
             pipeline_parallel_size=2,
             max_beam_width=beam_width,
             enable_speculative_beam_history_d2h=speculative_d2h,
             **llm_kwargs) as llm:
        outputs = llm.generate(prompts, sampling_params=sampling_params)

    assert len(outputs) == len(prompts)
    if reference_outputs is not None:
        assert len(reference_outputs) == len(prompts)
        for prompt, output, reference in zip(prompts, outputs,
                                             reference_outputs):
            assert [list(beam.token_ids) for beam in output.outputs
                    ] == [list(beam.token_ids)
                          for beam in reference.outputs], prompt
            for beam, reference_beam in zip(output.outputs, reference.outputs):
                assert beam.cumulative_logprob == pytest.approx(
                    reference_beam.cumulative_logprob, rel=1e-4,
                    abs=1e-4), prompt
                assert len(beam.logprobs) == len(
                    reference_beam.logprobs), prompt
                for token_id, actual, expected in zip(beam.token_ids,
                                                      beam.logprobs,
                                                      reference_beam.logprobs):
                    assert actual[token_id].logprob == pytest.approx(
                        expected[token_id].logprob, rel=1e-4, abs=1e-4), prompt
    for prompt, output in zip(prompts, outputs):
        beams = output.outputs
        assert len(beams) == beam_width, prompt
        # Finalized beams are distinct paths, best first.
        assert len({tuple(beam.token_ids) for beam in beams}) == beam_width
        cumulative = [beam.cumulative_logprob for beam in beams]
        assert None not in cumulative, prompt
        assert cumulative == sorted(cumulative, reverse=True), prompt
        for beam in beams:
            assert 0 < len(beam.token_ids) <= max_tokens, prompt
            assert len(beam.logprobs) == len(beam.token_ids), prompt
            token_logprobs = [
                next(iter(position.values())).logprob
                for position in beam.logprobs
            ]
            assert sum(token_logprobs) == pytest.approx(
                beam.cumulative_logprob), prompt


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

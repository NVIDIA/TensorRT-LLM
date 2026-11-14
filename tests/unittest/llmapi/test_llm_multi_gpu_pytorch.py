# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from utils.util import skip_ray

from tensorrt_llm import LLM
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

from .test_llm import (_test_llm_capture_request_error, llama_model_path,
                       llm_get_stats_async_test_harness,
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
def test_llm_beam_search_pp2() -> None:
    """Beam search under pipeline parallelism returns every finalized beam.

    Only the last pipeline-parallel rank samples; the first rank, which returns
    the results, finalizes the beams from the state that rank hands over.
    """
    beam_width = 4
    max_tokens = 24
    prompts = ["A B C", "The future of the AI is"]
    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     n=beam_width,
                                     best_of=beam_width,
                                     use_beam_search=True,
                                     logprobs=True)
    with LLM(model=llama_model_path,
             kv_cache_config=global_kv_cache_config,
             pipeline_parallel_size=1,
             max_beam_width=beam_width,
             max_batch_size=1,
             disable_overlap_scheduler=True,
             cuda_graph_config=None) as llm:
        reference_outputs = llm.generate(prompts,
                                         sampling_params=sampling_params)
    with LLM(model=llama_model_path,
             kv_cache_config=global_kv_cache_config,
             pipeline_parallel_size=2,
             max_beam_width=beam_width,
             max_batch_size=1,
             disable_overlap_scheduler=True,
             cuda_graph_config=None) as llm:
        outputs = llm.generate(prompts, sampling_params=sampling_params)

    assert len(outputs) == len(reference_outputs) == len(prompts)
    for prompt, output, reference in zip(prompts, outputs, reference_outputs):
        beams = output.outputs
        assert [list(beam.token_ids) for beam in beams
                ] == [list(beam.token_ids)
                      for beam in reference.outputs], prompt
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

import json
import random
import time
from contextlib import contextmanager, nullcontext
from typing import Optional

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor import GenerationExecutorWorker, RequestError
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.llmapi import CacheTransceiverConfig, KvCacheConfig
from tensorrt_llm.llmapi.llm_args import NGramDecodingConfig, PeftCacheConfig
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.metrics import MetricNames
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from .lora_test_utils import (
    check_llama_7b_multi_lora_from_request_test_harness,
    check_llama_7b_multi_unique_lora_adapters_from_request,
    create_mock_nemo_lora_checkpoint, compare_cuda_graph_lora_params_filler,
    CUDAGraphLoRATestParams, test_lora_with_and_without_cuda_graph)
from .test_llm import (_test_llm_capture_request_error, get_model_path,
                       global_kvcache_config, llama_model_path,
                       llm_get_stats_async_test_harness,
                       llm_get_stats_test_harness,
                       llm_return_logprobs_test_harness, llm_test_harness,
                       prompts, run_llm_abort_request,
                       run_llm_with_postprocess_parallel_and_result_handler,
                       tinyllama_logits_processor_test_harness)
from utils.util import (force_ampere, similar, similarity_score,
                        skip_fp8_pre_ada, skip_gpu_memory_less_than_40gb,
                        skip_gpu_memory_less_than_80gb,
                        skip_gpu_memory_less_than_138gb, skip_ray)
from utils.llm_data import llm_models_root
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.executor.request import LoRARequest
import tempfile

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import replace

# isort: on


@force_ampere
@pytest.mark.parametrize("enable_chunked_prefill,", [False, True])
@pytest.mark.part2
def test_tinyllama_logits_processor(enable_chunked_prefill):
    tinyllama_logits_processor_test_harness(
        backend="pytorch", enable_chunked_prefill=enable_chunked_prefill)


@skip_ray
@pytest.mark.parametrize(
    "return_context_logits, use_overlap, enable_chunked_prefill, enable_iter_req_stats",
    [
        (False, False, False, True),
        (False, False, True, True),
        (False, True, False, True),
        (False, True, True, True),
    ])
@pytest.mark.part0
def test_llm_get_stats(return_context_logits, use_overlap,
                       enable_chunked_prefill, enable_iter_req_stats):
    llm_get_stats_test_harness(tp_size=1,
                               pp_size=1,
                               return_context_logits=return_context_logits,
                               pytorch_backend=True,
                               use_overlap=use_overlap,
                               enable_chunked_prefill=enable_chunked_prefill,
                               enable_iter_req_stats=enable_iter_req_stats)


@skip_ray
@pytest.mark.parametrize(
    "return_context_logits, use_overlap, enable_chunked_prefill, enable_iter_req_stats",
    [
        (False, False, False, True),
        (False, False, True, True),
        (False, True, False, True),
        (False, True, True, True),
    ])
@pytest.mark.part1
def test_llm_get_stats_async(return_context_logits, use_overlap,
                             enable_chunked_prefill, enable_iter_req_stats):
    llm_get_stats_async_test_harness(
        tp_size=1,
        pp_size=1,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        use_overlap=use_overlap,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_iter_req_stats=enable_iter_req_stats)


@pytest.mark.part1
def test_llm_capture_request_error():
    _test_llm_capture_request_error(pytorch_backend=True, tp_size=1)


@force_ampere
@pytest.mark.mpi_ray_parity
@pytest.mark.parametrize(
    "sampling_params",
    [
        SamplingParams()  # pytorch only supports n=1
    ])
@pytest.mark.part0
def test_llm_abort_request(sampling_params):
    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
    run_llm_abort_request(llm=llm, sampling_params=sampling_params)


@contextmanager
def _validate_invalid_token_error_scope():
    with pytest.raises(RuntimeError) as exc_info:
        yield
    assert "Token ID out of range" in str(exc_info.value)


@force_ampere
@pytest.mark.part1
def test_llm_invalid_input_token():
    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
    prompts = [
        [-1],
    ]
    # NB: exc_info in _validate_invalid_token_error_scope creates a reference
    #     to a traceback which outlives the scope of 'exc_info' and prevents
    #     deletion of 'llm'. However, using the context manager protocol is
    #     anyways more robust than delegating cleanup to __del__.
    with llm:
        with _validate_invalid_token_error_scope():
            llm.generate(
                prompts,
                sampling_params=SamplingParams(max_tokens=5),
            )


@force_ampere
@pytest.mark.part0
def test_llm_invalid_input_token_async():
    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
    # NB: exc_info in _validate_invalid_token_error_scope creates a reference
    #     to a traceback which outlives the scope of 'exc_info' and prevents
    #     deletion of 'llm'. However, using the context manager protocol is
    #     anyways more robust than delegating cleanup to __del__.
    with llm:
        prompts = [
            [-1],
            [42],
        ]
        fail_idx = [0]
        for submit_order in [[0, 1], [1, 0]]:
            for collect_order in [[0, 1], [1, 0]]:
                print(f"submitting {submit_order}")
                futures = [
                    llm.generate_async(
                        prompts[submit_idx],
                        sampling_params=SamplingParams(max_tokens=5),
                    ) for submit_idx in submit_order
                ]
                for collect_idx in collect_order:
                    with _validate_invalid_token_error_scope(
                    ) if submit_order[collect_idx] in fail_idx else nullcontext(
                    ):
                        print(
                            f"collect order {collect_order}, collecting {collect_idx}"
                        )
                        futures[collect_idx].result()


@pytest.mark.part2
def test_llm_reward_model():
    rm_model_path = get_model_path("Qwen2.5-Math-PRM-7B")
    tokenizer = TransformersTokenizer.from_pretrained(rm_model_path)
    tokenized_input = tokenizer(prompts, return_tensors="pt")["input_ids"]

    llm = LLM(model=rm_model_path,
              attn_backend="VANILLA",
              disable_overlap_scheduler=True)

    sampling_params = SamplingParams(return_context_logits=True)

    outputs = llm.generate(prompts, sampling_params)
    scores = outputs[0].context_logits

    print(scores)

    assert scores.shape == (tokenized_input.shape[1], 2)
    assert not outputs[0].outputs[0].text


@skip_ray
@pytest.mark.part3
def test_llm_perf_metrics():
    with LLM(model=llama_model_path,
             kv_cache_config=global_kvcache_config) as llm:
        sampling_params = SamplingParams(max_tokens=10,
                                         return_perf_metrics=True)
        outputs = llm.generate(prompts, sampling_params)
        assert outputs[0].outputs[0].request_perf_metrics is not None

        perf_metrics = outputs[0].outputs[0].request_perf_metrics

        timing_metrics = perf_metrics.timing_metrics
        assert timing_metrics.arrival_time < timing_metrics.first_scheduled_time
        assert timing_metrics.first_scheduled_time < timing_metrics.first_token_time
        assert timing_metrics.first_token_time < timing_metrics.last_token_time

        kv_cache_metrics = perf_metrics.kv_cache_metrics
        assert kv_cache_metrics.num_total_allocated_blocks == 1
        assert kv_cache_metrics.num_new_allocated_blocks == 1
        assert kv_cache_metrics.num_reused_blocks == 0
        assert kv_cache_metrics.num_missed_blocks == 1
        assert kv_cache_metrics.kv_cache_hit_rate == 0

        assert perf_metrics.first_iter is not None
        assert perf_metrics.iter - perf_metrics.first_iter == sampling_params.max_tokens - 1
        assert perf_metrics.last_iter == perf_metrics.iter


@skip_ray
@pytest.mark.part3
def test_llm_prometheus():
    test_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.8, top_p=0.95)
    llm = LLM(model=llama_model_path,
              return_perf_metrics=True,
              kv_cache_config=global_kvcache_config)
    for test_prompt in test_prompts:
        request_output = llm.generate(test_prompt, sampling_params)
        assert request_output.metrics_dict is not None
        assert MetricNames.REQUEST_QUEUE_TIME in request_output.metrics_dict
        assert MetricNames.TPOT in request_output.metrics_dict
        assert MetricNames.TTFT in request_output.metrics_dict
        assert MetricNames.E2E in request_output.metrics_dict
        assert request_output.outputs is not None


@skip_ray
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.part3
def test_llm_with_postprocess_parallel_and_result_handler(streaming):
    run_llm_with_postprocess_parallel_and_result_handler(streaming,
                                                         "pytorch",
                                                         tp_size=1)


@pytest.mark.part0
def test_embedding_bias_with_torch_sampler_strategies():
    """Test embedding bias application in TorchSampler."""
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    vocab_size_padded = 32000
    embedding_bias = torch.zeros(vocab_size_padded)
    embedding_bias[biased_word_id] = torch.finfo(torch.float32).max

    sampling_kwargs = {
        "max_tokens": 6,
        "embedding_bias": embedding_bias,
    }

    # All test cases use greedy sampling for simplicity

    sampling_params = SamplingParams(**sampling_kwargs)

    llm_test_harness(
        llama_model_path,
        prompts,
        ["Z Z Z Z Z Z"],
        sampling_params=sampling_params,
        backend="pytorch",
    )


def test_lora_cuda_graph_params_filling_kernel_special_cases():
    torch.cuda.set_device(0)

    # test all requests have the same LoRA id case
    test_params = CUDAGraphLoRATestParams(
        batch_slot_ids=[0] * 10,
        input_hidden_size=4096,
        slot_ranks=[64] * 10,
        max_lora_rank=64,
        output_hidden_sizes=[123],
        layer_module_mask=None,
        dtype=torch.bfloat16,
        seed=42,
    )
    compare_cuda_graph_lora_params_filler(test_params)

    # test no LoRA in a batch case
    test_params2 = replace(test_params,
                           batch_slot_ids=[len(test_params.slot_ranks)] * 10)
    compare_cuda_graph_lora_params_filler(test_params2)

    # test all having three modules case
    test_params3 = replace(test_params, output_hidden_sizes=[123, 456, 789])
    compare_cuda_graph_lora_params_filler(test_params3)

    # test some layer module have invalid weight pointers case
    mask = torch.full((test_params3.module_count, test_params3.slot_count),
                      True,
                      dtype=torch.bool)
    mask[0, 0] = False
    mask[1, 7] = False
    mask[2, 3] = False
    test_params4 = replace(test_params3, layer_module_mask=mask)
    compare_cuda_graph_lora_params_filler(test_params4)

    # test mixed slot ids case
    test_params5 = CUDAGraphLoRATestParams(
        batch_slot_ids=[6, 2, 0, 1, 1, 1, 5, 6],
        input_hidden_size=512,
        slot_ranks=[8, 12, 4] * 2,
        max_lora_rank=15,
        output_hidden_sizes=[123, 456, 789],
        layer_module_mask=None,
        dtype=torch.bfloat16,
        seed=42,
    )
    compare_cuda_graph_lora_params_filler(test_params5)

    # test mixed slot with invalid weight pointers
    mask = torch.full((test_params5.module_count, test_params5.slot_count),
                      True,
                      dtype=torch.bool)
    mask[1, 3] = False
    mask[2, 5] = False
    mask[-1, -4] = False
    test_params6 = replace(test_params5, layer_module_mask=mask)
    compare_cuda_graph_lora_params_filler(test_params6)


def llama_7b_lora_from_dir_test_harness(**llm_kwargs) -> None:
    lora_config = LoraConfig(
        lora_dir=[f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"],
        max_lora_rank=8,
        max_loras=2,
        max_cpu_loras=2)
    if "cuda_graph_config" not in llm_kwargs:
        llm_kwargs["cuda_graph_config"] = None
    llm = LLM(model=f"{llm_models_root()}/llama-models/llama-7b-hf",
              lora_config=lora_config,
              **llm_kwargs)
    try:
        prompts = [
            "美国的首都在哪里? \n答案:",
        ]
        references = [
            "美国的首都是华盛顿。\n\n美国的",
        ]
        sampling_params = SamplingParams(max_tokens=20)
        lora_req = LoRARequest(
            "task-0", 0, f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1")
        lora_request = [lora_req]

        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_request)
        assert similar(outputs[0].outputs[0].text, references[0])
    finally:
        llm.shutdown()


@skip_gpu_memory_less_than_40gb
@pytest.mark.part0
@test_lora_with_and_without_cuda_graph
def test_llama_7b_lora(cuda_graph_config):
    llama_7b_lora_from_dir_test_harness(cuda_graph_config=cuda_graph_config)


@skip_gpu_memory_less_than_40gb
@test_lora_with_and_without_cuda_graph
def test_llama_7b_lora_default_modules(cuda_graph_config) -> None:
    lora_config = LoraConfig(max_lora_rank=64, max_loras=2, max_cpu_loras=2)

    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"

    llm = LLM(model=hf_model_dir,
              lora_config=lora_config,
              cuda_graph_config=cuda_graph_config)

    hf_lora_dir = f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"
    try:
        prompts = [
            "美国的首都在哪里? \n答案:",
        ]
        references = [
            "美国的首都是华盛顿。\n\n美国的",
        ]
        sampling_params = SamplingParams(max_tokens=20,
                                         add_special_tokens=False)
        lora_req = LoRARequest("luotuo", 1, hf_lora_dir)
        lora_request = [lora_req]

        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_request)

        assert similar(outputs[0].outputs[0].text, references[0])
    finally:
        llm.shutdown()


def _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call: list[int], max_loras: int,
        max_cpu_loras: int, repeat_calls: int, repeats_per_call: int,
        **llm_kwargs):
    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    lora_config = LoraConfig(lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
                             max_lora_rank=8,
                             max_loras=max_loras,
                             max_cpu_loras=max_cpu_loras)
    check_llama_7b_multi_unique_lora_adapters_from_request(
        lora_adapter_count_per_call,
        repeat_calls,
        repeats_per_call,
        LLM,
        lora_config=lora_config,
        **llm_kwargs)


@skip_gpu_memory_less_than_40gb
@skip_ray  # https://nvbugs/5682551
@pytest.mark.part3
@test_lora_with_and_without_cuda_graph
def test_llama_7b_multi_lora_evict_and_reload_lora_gpu_cache(cuda_graph_config):
    """Test eviction and re-loading a previously evicted adapter from the LoRA GPU cache, within a single
    llm.generate call, that's repeated twice.
    """  # noqa: D205
    _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call=[2],
        max_loras=1,
        max_cpu_loras=2,
        repeat_calls=2,
        repeats_per_call=3,
        cuda_graph_config=cuda_graph_config)


@skip_gpu_memory_less_than_40gb
@pytest.mark.part1
@test_lora_with_and_without_cuda_graph
def test_llama_7b_multi_lora_evict_and_load_new_adapters_in_cpu_and_gpu_cache(
        cuda_graph_config):
    """Test eviction and loading of new adapters in the evicted space, over several llm.generate calls, with LoRA GPU
    cache size < LoRA CPU cache size.
    """  # noqa: D205
    _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call=[2, 2, 2],
        max_loras=1,
        max_cpu_loras=3,
        repeat_calls=1,
        repeats_per_call=1,
        cuda_graph_config=cuda_graph_config)


@skip_gpu_memory_less_than_40gb
@pytest.mark.part0
@test_lora_with_and_without_cuda_graph
def test_llama_7b_multi_lora_read_from_cache_after_insert(cuda_graph_config):
    """Test that loading and then using the same adapters loaded in cache works."""
    _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call=[3],
        max_loras=3,
        max_cpu_loras=3,
        repeat_calls=2,
        repeats_per_call=1,
        cuda_graph_config=cuda_graph_config)


@skip_gpu_memory_less_than_40gb
@pytest.mark.part3
@test_lora_with_and_without_cuda_graph
def test_llama_7b_multi_lora_evict_and_reload_evicted_adapters_in_cpu_and_gpu_cache(
        cuda_graph_config):
    """Test eviction, reloading new adapters and reloading previously evicted adapters from the LoRA CPU cache & GPU
    cache over multiple llm.generate call repeated twice (two calls with the same requests):
    At the end of the 1st llm.generate call:
      The LoRA caches should contain adapters 1, 2 and shouldn't contain adapter 0 (it should have been evicted).
    So in the 2nd call, the worker should:
    - Send req0 with adapter 0 weights (because it was previously evicted)
    - Send the other two requests without their adapter weights as they're already in LoRA CPU cache
    Then, handling of req0 that has weights but not in the cache should evict one of the other two adapters from
    the cache, causing that evicted adapter's request to again load its weights from the file system, as they
    aren't with the request and aren't in LoRA cache.
    """  # noqa: D205
    _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call=[3],
        max_loras=2,
        max_cpu_loras=2,
        repeat_calls=2,
        repeats_per_call=1,
        cuda_graph_config=cuda_graph_config)


@skip_gpu_memory_less_than_40gb
@pytest.mark.part2
@test_lora_with_and_without_cuda_graph
def test_llama_7b_peft_cache_config_affects_peft_cache_size(cuda_graph_config):
    """Tests that LLM arg of peft_cache_config affects the peft cache sizes.

    NOTE: The caller can't get the actual LoRA cache sizes, so we instead we
    test that it fails when configured with a value too small to contain a
    single adapter.
    """
    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    lora_config_no_cache_size_values = LoraConfig(
        lora_target_modules=['attn_q', 'attn_k', 'attn_v'], max_lora_rank=8)

    # Test that too small PeftCacheConfig.host_cache_size causes failure
    with pytest.raises(RuntimeError):
        check_llama_7b_multi_lora_from_request_test_harness(
            LLM,
            lora_config=lora_config_no_cache_size_values,
            peft_cache_config=PeftCacheConfig(
                host_cache_size=1),  # size in bytes
            cuda_graph_config=cuda_graph_config)

    # Test that too small PeftCacheConfig.device_cache_percent causes failure
    with pytest.raises(RuntimeError):
        check_llama_7b_multi_lora_from_request_test_harness(
            LLM,
            lora_config=lora_config_no_cache_size_values,
            peft_cache_config=PeftCacheConfig(device_cache_percent=0.0000001),
            cuda_graph_config=cuda_graph_config)


@skip_ray  # https://nvbugs/5682551
@skip_gpu_memory_less_than_40gb
@pytest.mark.part1
@test_lora_with_and_without_cuda_graph
def test_llama_7b_lora_config_overrides_peft_cache_config(cuda_graph_config):
    """Tests that cache size args in lora_config LLM arg override the cache size
    parameters in peft_cache_config LLM arg.
    """    # noqa: D205
    check_llama_7b_multi_lora_from_request_test_harness(
        LLM,
        lora_config=LoraConfig(
            lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
            max_lora_rank=8,
            max_loras=2,
            max_cpu_loras=2),
        peft_cache_config=PeftCacheConfig(
            host_cache_size=1,  # size in bytes
            device_cache_percent=0.0000001),
        cuda_graph_config=cuda_graph_config)


# TODO smor: currently Nemotron-Super-49B-v1 with LoRA memory consumption is overly high
# https://jirasw.nvidia.com/browse/TRTLLM-5045
@pytest.mark.skip(reason="https://nvbugs/5448464")
@skip_gpu_memory_less_than_138gb
@pytest.mark.part1
@test_lora_with_and_without_cuda_graph
def test_nemotron_nas_lora(cuda_graph_config) -> None:
    lora_config = LoraConfig(lora_dir=[
        f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1-lora-adapter_r64"
    ],
                             max_lora_rank=64,
                             max_loras=1,
                             max_cpu_loras=1)

    llm = LLM(
        model=
        f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1",
        lora_config=lora_config,
        cuda_graph_config=cuda_graph_config)

    prompts = [
        "Hello, how are you?",
        "Hello, how are you?",
    ]

    sampling_params = SamplingParams(max_tokens=10, add_special_tokens=False)
    lora_req = LoRARequest(
        "task-0", 0,
        f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1-lora-adapter_r64"
    )
    lora_request = [lora_req, None]

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    assert similar(outputs[0].outputs[0].text, outputs[1].outputs[0].text)


@skip_gpu_memory_less_than_80gb
@pytest.mark.part0
@test_lora_with_and_without_cuda_graph
def test_llama_3_1_8b_fp8_with_bf16_lora(cuda_graph_config) -> None:
    skip_fp8_pre_ada(use_fp8=True)
    model_dir = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8"
    lora_dir = f"{llm_models_root()}/lora/llama-3-chinese-8b-instruct-v2-lora"
    prompt = "美国的首都是哪里？"
    reference = "华盛顿特区。华盛顿特区是美国的首都和一个行政区"

    lora_config = LoraConfig(lora_dir=[lora_dir],
                             max_lora_rank=64,
                             max_loras=2,
                             max_cpu_loras=2)
    lora_req = LoRARequest("lora-chinese", 0, lora_dir)

    llm = LLM(model_dir,
              lora_config=lora_config,
              cuda_graph_config=cuda_graph_config)

    try:
        output = llm.generate(prompt,
                              SamplingParams(max_tokens=20),
                              lora_request=[lora_req])
    finally:
        llm.shutdown()
    assert similar(output.outputs[0].text, reference)


@skip_ray  # https://nvbugs/5682551
@skip_gpu_memory_less_than_80gb
def test_llama_3_3_70b_fp8_with_squad_lora_tp2() -> None:
    skip_fp8_pre_ada(use_fp8=True)

    model_dir = f"{llm_models_root()}/llama-3.3-models/Llama-3.3-70B-Instruct-FP8"
    lora_dir = f"{llm_models_root()}/llama-3.3-models/Llama-3.3-70B-Instruct-FP8-lora-adapter_NIM_r8"

    prompt = "What is the capital of the United States?"
    expected_output = " Washington, D.C.\nWhat is the capital of the United States? Washington, D.C."

    lora_config = LoraConfig(lora_dir=[lora_dir],
                             max_lora_rank=8,
                             max_loras=2,
                             max_cpu_loras=2)
    lora_req = LoRARequest("squad-lora", 0, lora_dir)

    llm = LLM(model_dir,
              tensor_parallel_size=2,
              lora_config=lora_config,
              cuda_graph_config=None)

    try:
        output = llm.generate(prompt,
                              SamplingParams(max_tokens=50, temperature=0.0),
                              lora_request=[lora_req])
        generated_text = output.outputs[0].text
        print(f"Generated output: {repr(generated_text)}")

        similarity = similarity_score(generated_text, expected_output)
        assert similar(generated_text, expected_output, threshold=0.8), \
            f"Output similarity too low (similarity={similarity:.2%})!\nExpected: {repr(expected_output)}\nGot: {repr(generated_text)}"
    finally:
        llm.shutdown()


@skip_gpu_memory_less_than_80gb
@pytest.mark.part2
@test_lora_with_and_without_cuda_graph
def test_bielik_11b_v2_2_instruct_multi_lora(cuda_graph_config) -> None:
    model_dir = f"{llm_models_root()}/Bielik-11B-v2.2-Instruct"

    target_modules = ['attn_q', 'attn_k', 'attn_v']

    # Set up temporary directory for LoRA adapters
    with tempfile.TemporaryDirectory() as lora_dir:
        print("Creating dummy LoRAs...")

        model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                     dtype=torch.bfloat16,
                                                     device_map="auto")
        hf_modules = ["q_proj", "k_proj", "v_proj"]
        peft_lora_config = PeftLoraConfig(r=8,
                                          target_modules=hf_modules,
                                          bias="none",
                                          task_type="CAUSAL_LM")
        lora_paths = []
        for i in range(2):
            lora_model = get_peft_model(model, peft_lora_config)
            for param in lora_model.parameters():
                param.data.zero_()
            lora_path = f"{lora_dir}/lora_{i}"
            lora_model.save_pretrained(lora_path)
            lora_paths.append(lora_path)

        trtllm_lora_config = LoraConfig(lora_target_modules=target_modules,
                                        max_lora_rank=8,
                                        max_loras=2,
                                        max_cpu_loras=2)
        llm = LLM(model_dir,
                  lora_config=trtllm_lora_config,
                  cuda_graph_config=cuda_graph_config)

        prompts = [
            "Kim był Mikołaj Kopernik i z czego zasłynął?",
            "Gdzie znajduje się stolica Polski?",
        ]
        lora_req1 = LoRARequest("lora-1", 0, lora_paths[0])
        lora_req2 = LoRARequest("lora-2", 1, lora_paths[1])
        lora_requests = [lora_req1, lora_req2]
        sampling_params = SamplingParams(max_tokens=200)

        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_requests)

        assert len(outputs) == 2


@pytest.mark.part2
@test_lora_with_and_without_cuda_graph
def test_gemma3_1b_instruct_multi_lora(cuda_graph_config) -> None:
    model_dir = f"{llm_models_root()}/gemma/gemma-3-1b-it"

    target_modules = ['attn_q', 'attn_k', 'attn_v']

    # Set up temporary directory for LoRA adapters
    with tempfile.TemporaryDirectory() as lora_dir:
        print("Creating dummy LoRAs...")

        model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                     dtype=torch.bfloat16,
                                                     device_map="auto")
        hf_modules = ["q_proj", "k_proj", "v_proj"]
        peft_lora_config = PeftLoraConfig(r=8,
                                          target_modules=hf_modules,
                                          bias="none",
                                          task_type="CAUSAL_LM")
        lora_paths = []
        for i in range(2):
            lora_model = get_peft_model(model, peft_lora_config)
            for param in lora_model.parameters():
                param.data.zero_()
            lora_path = f"{lora_dir}/lora_{i}"
            lora_model.save_pretrained(lora_path)
            lora_paths.append(lora_path)

        trtllm_lora_config = LoraConfig(lora_dir=lora_paths,
                                        lora_target_modules=target_modules,
                                        max_lora_rank=8,
                                        max_loras=2,
                                        max_cpu_loras=2)
        # Disabling kv cache reuse as a WAR to deal with gaps in kernel support for Gemma3's non-inclusive sliding window size.
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            enable_partial_reuse=False,
        )
        llm = LLM(model_dir,
                  lora_config=trtllm_lora_config,
                  kv_cache_config=kv_cache_config,
                  cuda_graph_config=cuda_graph_config)

        prompts = [
            "Is it ok to fill diesel in a petrol car?",
            "What is the capital of France?",
        ]
        lora_req1 = LoRARequest("lora-1", 0, lora_paths[0])
        lora_req2 = LoRARequest("lora-2", 1, lora_paths[1])
        lora_requests = [lora_req1, lora_req2]
        sampling_params = SamplingParams(max_tokens=200)

        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_requests)

        assert len(outputs) == 2


@pytest.mark.parametrize(
    "lora_rank,max_lora_rank,description",
    [
        # (lora_rank, max_lora_rank, description)
        (8, 8, "rank_8"),
        (16, 16, "rank_16"),
        (4, 8, "rank_4_max_8"),
    ])
@pytest.mark.part3
def test_load_torch_nemo_lora_function(tmp_path, lora_rank, max_lora_rank,
                                       description):
    """Test load_torch_nemo_lora function with different LoRA rank configurations."""
    from tensorrt_llm.lora_manager import load_torch_nemo_lora

    nemo_path = create_mock_nemo_lora_checkpoint(
        tmp_path,
        hidden_size=2048,
        num_layers=16,
        lora_rank=lora_rank,
    )

    lora_config = LoraConfig(
        lora_dir=[str(nemo_path)],
        lora_ckpt_source="nemo",
        max_lora_rank=max_lora_rank,
    )

    # This should not raise an error
    load_torch_nemo_lora(lora_config)

    assert lora_config.lora_target_modules == [
        "attn_qkv"
    ], f"Expected attn_qkv modules for {description}"
    assert lora_config.trtllm_modules_to_hf_modules == {
        "attn_qkv": "attn_qkv"
    }, f"Expected correct module mapping for {description}"


@pytest.mark.part0
def test_nemo_lora_unsupported_modules_validation(tmp_path):
    """Test validation of unsupported modules in NeMo LoRA."""
    from tensorrt_llm.lora_manager import load_torch_nemo_lora

    nemo_path = create_mock_nemo_lora_checkpoint(
        tmp_path,
        hidden_size=2048,
        num_layers=16,
        lora_rank=8,
    )

    # Test validation: should fail with unsupported modules
    invalid_config = LoraConfig(
        lora_dir=[str(nemo_path)],
        lora_ckpt_source="nemo",
        lora_target_modules=["attn_qkv",
                             "mlp_h_to_4h"],  # mlp_h_to_4h not supported
        max_lora_rank=8,
    )

    with pytest.raises(ValueError, match="NeMo LoRA only supports"):
        load_torch_nemo_lora(invalid_config)


@force_ampere
@pytest.mark.part1
@test_lora_with_and_without_cuda_graph
def test_gqa_nemo_lora(tmp_path, cuda_graph_config):
    """
    Test NeMo-format LoRA checkpoint loading and GQA support in TinyLlama.

    This test verifies two properties:
    1. That a NeMo-format LoRA checkpoint with GQA (grouped query attention) can be loaded and applied to a TinyLlama model,
       and that generation with this LoRA produces a deterministic, expected output for a fixed prompt and temperature=0.0.
    2. That the LoRA weights have a significant effect: generating with LoRA produces a different output than generating
       without LoRA, confirming that the LoRA adapter is actually being applied.

    The test uses a deterministic dummy LoRA checkpoint (seed=42) and checks both the positive (LoRA applied) and negative
    (no LoRA) cases for output text.
    """
    # TinyLlama's exact GQA configuration
    hidden_size = 2048
    num_layers = 22
    num_q_heads = 32  # Query attention heads
    num_kv_heads = 4  # Key/Value heads (GQA)
    lora_rank = 8

    nemo_path = create_mock_nemo_lora_checkpoint(
        tmp_path,
        hidden_size=hidden_size,
        num_layers=num_layers,
        lora_rank=lora_rank,
        num_attention_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seed=42,  # NOTE: the seed=42 is important for the test to pass.
    )
    expected_lora_text_output = "Paris. The capital of France is Paris. The"
    test_prompts = ["The capital of France is"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    lora_config = LoraConfig(
        lora_dir=[str(nemo_path)],
        lora_ckpt_source="nemo",
        max_lora_rank=lora_rank,
    )

    model_path = get_model_path("llama-models-v2/TinyLlama-1.1B-Chat-v1.0")

    llm = LLM(
        model=model_path,
        lora_config=lora_config,
        kv_cache_config=global_kvcache_config,
        cuda_graph_config=cuda_graph_config,
    )

    try:
        lora_req = LoRARequest("tinyllama-gqa-test",
                               0,
                               str(nemo_path),
                               lora_ckpt_source="nemo")

        lora_outputs = llm.generate(test_prompts,
                                    sampling_params,
                                    lora_request=[lora_req])

        # For the above deterministic dummy LoRA checkpoint,
        # with temperature=0.0,
        # the expected output text should always be the same.
        assert lora_outputs[0].outputs[0].text == expected_lora_text_output, \
            f"Expected output text: {expected_lora_text_output}, " \
            f"got: {lora_outputs[0].outputs[0].text}"
        assert len(lora_outputs) == 1

        # Generate without LoRA.
        # The LoRA weights are tuned/large enough that
        # they differ from a no-LoRA run.
        base_outputs = llm.generate(test_prompts, sampling_params)
        assert base_outputs[0].outputs[0].text != expected_lora_text_output, \
            f"No-LoRA output should differ from expected output text: {expected_lora_text_output}, " \
            f"got: {base_outputs[0].outputs[0].text}"
    finally:
        llm.shutdown()


class TestLlmError:

    @pytest.mark.part3
    def test_max_num_token_check(self):
        """ LLM should raise error when got prompt length exceed the valid range. """
        llm = LLM(llama_model_path,
                  kv_cache_config=global_kvcache_config,
                  max_num_tokens=100)

        try:
            with pytest.raises(RequestError,
                               match="should not exceed max_num_tokens"):
                ids = [random.randint(10, 100) for _ in range(101)]
                llm.generate([ids])
        finally:
            llm.shutdown()


class FailingExecutorWorker(GenerationExecutorWorker):
    """Mock worker that fails during initialization to test error handling."""

    def __init__(self, *args, **kwargs):
        # Simulate a constructor failure
        raise RuntimeError(
            "Mock GenerationExecutorWorker initialization failed")


FailingExecutor = type(
    "FailingExecutor", (), {
        "create":
        classmethod(
            lambda cls, *args, **kwargs: FailingExecutorWorker(*args, **kwargs))
    })


@skip_ray
@pytest.mark.part2
def test_llm_with_proxy_error():
    """Test that LLM properly handles GenerationExecutorWorker constructor failures.

    This test mocks the GenerationExecutorWorker to fail during __init__ and
    verifies that the LLM class properly catches and re-raises the error.
    """
    from unittest.mock import patch

    # Test that the error is properly caught and re-raised by LLM
    # We patch GenerationExecutor.create directly to return our failing worker
    with patch('tensorrt_llm.executor.executor.GenerationExecutor.create',
               side_effect=lambda *args, **kwargs: FailingExecutorWorker(
                   *args, **kwargs)):
        with pytest.raises(
                RuntimeError,
                match="Mock GenerationExecutorWorker initialization failed"):
            llm = LLM(model=llama_model_path,
                      kv_cache_config=global_kvcache_config)


@pytest.mark.part0
@pytest.mark.parametrize("use_speculative", [True, False])
def test_min_tokens(use_speculative: bool):
    """Check min_tokens is respected."""
    llm_common_config = dict(
        model=llama_model_path,
        max_batch_size=2,
        kv_cache_config=global_kvcache_config,
        max_num_tokens=2048,
    )

    if use_speculative:
        spec_config = NGramDecodingConfig(
            max_draft_len=4,
            max_matching_ngram_size=2,
            is_keep_all=True,
            is_use_oldest=True,
            is_public_pool=True,
        )
        llm = LLM(**llm_common_config, speculative_config=spec_config)
    else:
        llm = LLM(**llm_common_config)

    output_len = 2000
    sampling_params = SamplingParams(max_tokens=output_len,
                                     min_tokens=output_len,
                                     temperature=1)
    res = llm.generate("The end.", sampling_params=sampling_params)

    assert len(res.outputs) == 1
    assert len(res.outputs[0].token_ids) == output_len


@skip_ray
@pytest.mark.parametrize(
    "prompt_logprobs, logprobs, return_context_logits, return_generation_logits, backend",
    [
        (2, None, True, False,
         "pytorch"),  # prompt_logprobs with context_logits
        (None, 1, False, False,
         "pytorch"),  # generation logprobs only (top-1, PyTorch limit)
        (2, None, False, False,
         "pytorch"),  # prompt_logprobs without context_logits
        (None, None, False, False, "pytorch"),  # no logprobs at all
    ])
def test_llm_return_logprobs(prompt_logprobs: Optional[int],
                             logprobs: Optional[int],
                             return_context_logits: bool,
                             return_generation_logits: bool, backend: str):
    llm_return_logprobs_test_harness(prompt_logprobs,
                                     logprobs,
                                     return_context_logits,
                                     return_generation_logits,
                                     backend=backend)


@skip_ray
@pytest.mark.parametrize(
    "prompt_logprobs, logprobs, return_context_logits, return_generation_logits",
    [
        (None, 1, False,
         False),  # generation logprobs only (top-1, PyTorch limit)
        (2, None, True, False),  # prompt_logprobs with context_logits
        (2, None, False, False),  # prompt_logprobs only
        (2, 1, False, False),  # both prompt and generation logprobs
        (2, 3, False, False),  # both prompt and generation logprobs
    ])
def test_llm_return_logprobs_streaming(prompt_logprobs, logprobs,
                                       return_context_logits,
                                       return_generation_logits):
    llm_return_logprobs_test_harness(prompt_logprobs,
                                     logprobs,
                                     return_context_logits,
                                     return_generation_logits,
                                     streaming=True,
                                     backend="pytorch")


class TestLlmError:

    @pytest.mark.part3
    def test_max_num_token_check(self):
        """ LLM should raise error when got prompt length exceed the valid range. """
        llm = LLM(llama_model_path,
                  kv_cache_config=global_kvcache_config,
                  max_num_tokens=100)

        try:
            with pytest.raises(RequestError,
                               match="should not exceed max_num_tokens"):
                ids = [random.randint(10, 100) for _ in range(101)]
                llm.generate([ids])
        finally:
            llm.shutdown()


@skip_ray
@pytest.mark.parametrize("num_requests", [1, 5, 10])
def test_llm_rpc(num_requests: int):
    # TODO: remove the with-statement when shutdown hang issue is fixed
    with LLM(model=llama_model_path,
             kv_cache_config=global_kvcache_config,
             orchestrator_type="rpc") as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        res = llm.generate("Tell me a joke",
                           sampling_params=SamplingParams(max_tokens=10,
                                                          end_id=-1))
        print(f"get result: {res}")

        assert len(res.outputs) == 1
        assert len(res.outputs[0].token_ids) == 10


@skip_ray
@pytest.mark.asyncio
async def test_llm_rpc_streaming():
    # TODO: remove the with-statement when shutdown hang issue is fixed
    with LLM(model=llama_model_path,
             kv_cache_config=global_kvcache_config,
             orchestrator_type="rpc") as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        outputs = []
        async for output in llm.generate_async("Tell me a joke",
                                               sampling_params=SamplingParams(
                                                   max_tokens=10, end_id=-1),
                                               streaming=True):
            outputs.append(output.outputs[0].text)
        "".join(outputs)
        print(f"get result: {outputs}")


@skip_ray
def test_llm_rpc_get_stats():
    """Test that get_stats works with RPC orchestrator."""

    with LLM(model=llama_model_path,
             kv_cache_config=global_kvcache_config,
             enable_iter_perf_stats=True,
             orchestrator_type="rpc") as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        # Generate some output to produce stats
        for output in llm.generate(
                prompts, sampling_params=SamplingParams(max_tokens=5)):
            print(output)

        stats = llm.get_stats(timeout=5)

        assert len(stats) > 0, "Should have at least one stats entry"
        # Stats should be JSON strings that can be parsed
        parsed = json.loads(stats[0]) if isinstance(stats[0], str) else stats[0]
        assert "iter" in parsed, "Stats should contain 'iter' field"
        assert "cpuMemUsage" in parsed, "Stats should contain 'cpuMemUsage' field"


@skip_ray
@pytest.mark.asyncio
async def test_llm_rpc_get_stats_async():
    """Test that get_stats_async works with RPC orchestrator."""
    import json

    with LLM(model=llama_model_path,
             kv_cache_config=global_kvcache_config,
             enable_iter_perf_stats=True,
             orchestrator_type="rpc") as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        # Generate some output to produce stats
        async for output in llm.generate_async(
            prompts[0], sampling_params=SamplingParams(max_tokens=5)):
            print(output)

        # Get stats via async API
        stats_result = llm.get_stats_async(timeout=2)

        # Should be able to iterate over results
        stats_count = 0
        async for stat in stats_result:
            parsed = json.loads(stat) if isinstance(stat, str) else stat
            assert "iter" in parsed, "Stats should contain 'iter' field"
            stats_count += 1
            if stats_count >= 1:
                break  # Just verify we can get at least one

        assert stats_count > 0, "Should have received at least one stat"


@pytest.mark.threadleak(enabled=False)
@pytest.mark.part0
@skip_ray
def test_llm_context_only_timed_out():
    tp_size = 1
    use_overlap = False
    enable_iter_req_stats = False

    llm_args_extra = {}

    llm_args_extra.update(
        dict(enable_iter_perf_stats=True,
             enable_iter_req_stats=enable_iter_req_stats,
             disable_overlap_scheduler=not use_overlap))

    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kvcache_config,
              tensor_parallel_size=tp_size,
              cache_transceiver_config=CacheTransceiverConfig(
                  backend="UCX", kv_transfer_timeout_ms=1000),
              **llm_args_extra)

    max_tokens = 1
    sampling_params = SamplingParams(max_tokens=max_tokens)

    disaggregated_params = DisaggregatedParams(request_type="context_only")

    prompts0 = [
        "What is your name?",
    ]
    prompts1 = [
        "Nvidia is awesome because",
    ]

    # Send context-only request
    for output in llm.generate(prompts1,
                               sampling_params=sampling_params,
                               disaggregated_params=disaggregated_params):
        print(output)

    max_retries = 10
    for _ in range(max_retries):
        results = llm.get_stats(2)
        if len(results) == 1:
            break
        time.sleep(1)
    else:
        pytest.fail(
            f"Failed to get stats with len==1 after {max_retries} retries")

    assert len(results) == 1
    context_only_used_num_blocks = results[0]["kvCacheStats"]["usedNumBlocks"]
    print(f"Context only used num blocks: {context_only_used_num_blocks}")

    # Sleep 5 seconds to allow context only request to time out
    time.sleep(5)

    # Send regular request
    for output in llm.generate(prompts0, sampling_params=sampling_params):
        print(output)

    # Get number of allocated blocks
    results = llm.get_stats(2)
    assert len(results) == 1
    final_used_num_blocks = results[0]["kvCacheStats"]["usedNumBlocks"]

    assert final_used_num_blocks == 0


# This test is to verify that when the KV cache is exhausted and scheduled batch size is 0, the context only request will be aborted due to timeout.


@pytest.mark.threadleak(enabled=False)
@pytest.mark.part0
@skip_ray
@pytest.mark.parametrize("sender_future_timeout_ms", [100, 1000])
@pytest.mark.parametrize("backend", ["NIXL", "UCX"])
def test_llm_context_only_timed_out_kv_cache_exhausted(sender_future_timeout_ms,
                                                       backend):
    tp_size = 1
    use_overlap = False
    enable_iter_req_stats = False

    llm_args_extra = {}

    llm_args_extra.update(
        dict(enable_iter_perf_stats=True,
             enable_iter_req_stats=enable_iter_req_stats,
             disable_overlap_scheduler=not use_overlap))

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.1,
                                    max_tokens=1000,
                                    enable_block_reuse=False)
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=kv_cache_config,
        tensor_parallel_size=tp_size,
        cache_transceiver_config=CacheTransceiverConfig(
            backend=backend,
            kv_transfer_timeout_ms=1000,
            kv_transfer_sender_future_timeout_ms=sender_future_timeout_ms),
        **llm_args_extra)

    max_tokens = 1
    sampling_params = SamplingParams(max_tokens=max_tokens)

    disaggregated_params = DisaggregatedParams(request_type="context_only")

    prompts0 = [
        "What is your name?",
    ]
    prompts1 = [
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua "
        * 10
    ]

    # Send context-only request
    for output in llm.generate(prompts1 * 10,
                               sampling_params=sampling_params,
                               disaggregated_params=disaggregated_params):
        print(output)

    max_retries = 10
    all_results = []
    for _ in range(max_retries):
        results = llm.get_stats(2)
        all_results.extend(results)

    assert len(all_results) > 0

    context_only_used_num_blocks = all_results[-1]["kvCacheStats"][
        "usedNumBlocks"]
    print(f"Context only used num blocks: {context_only_used_num_blocks}")

    # Sleep 5 seconds to allow context only request to time out
    time.sleep(5)

    # Send regular request
    for output in llm.generate(prompts0, sampling_params=sampling_params):
        print(output)

    # Get number of allocated blocks
    results = llm.get_stats(2)
    assert len(results) == 1
    final_used_num_blocks = results[0]["kvCacheStats"]["usedNumBlocks"]

    assert final_used_num_blocks == 0

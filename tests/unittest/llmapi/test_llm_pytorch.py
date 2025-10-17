import random
from contextlib import contextmanager, nullcontext
from typing import Optional

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.executor import GenerationExecutorWorker
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import NGramDecodingConfig, PeftCacheConfig
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.metrics import MetricNames
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from .lora_test_utils import (
    check_llama_7b_multi_lora_from_request_test_harness,
    check_llama_7b_multi_unique_lora_adapters_from_request,
    create_mock_nemo_lora_checkpoint)
from .test_llm import (_test_llm_capture_request_error, get_model_path,
                       global_kvcache_config, llama_model_path,
                       llm_get_stats_async_test_harness,
                       llm_get_stats_test_harness,
                       llm_return_logprobs_test_harness, llm_test_harness,
                       prompts, run_llm_abort_request,
                       run_llm_with_postprocess_parallel_and_result_handler,
                       tinyllama_logits_processor_test_harness)
from utils.util import (force_ampere, similar, skip_fp8_pre_ada,
                        skip_gpu_memory_less_than_40gb,
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

# isort: on


@force_ampere
@pytest.mark.parametrize("enable_chunked_prefill,", [False, True])
def test_tinyllama_logits_processor(enable_chunked_prefill):
    tinyllama_logits_processor_test_harness(
        backend="pytorch", enable_chunked_prefill=enable_chunked_prefill)


@skip_ray
@pytest.mark.parametrize(
    "return_context_logits, use_overlap, enable_iter_req_stats", [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
    ])
def test_llm_get_stats(return_context_logits, use_overlap,
                       enable_iter_req_stats):
    llm_get_stats_test_harness(tp_size=1,
                               return_context_logits=return_context_logits,
                               pytorch_backend=True,
                               use_overlap=use_overlap,
                               enable_iter_req_stats=enable_iter_req_stats)


@skip_ray
@pytest.mark.parametrize(
    "return_context_logits, use_overlap, enable_iter_req_stats", [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
    ])
def test_llm_get_stats_async(return_context_logits, use_overlap,
                             enable_iter_req_stats):
    llm_get_stats_async_test_harness(
        tp_size=1,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        use_overlap=use_overlap,
        enable_iter_req_stats=enable_iter_req_stats)


def test_llm_capture_request_error():
    _test_llm_capture_request_error(pytorch_backend=True, tp_size=1)


@force_ampere
@pytest.mark.mpi_ray_parity
@pytest.mark.parametrize(
    "sampling_params",
    [
        SamplingParams()  # pytorch only supports n=1
    ])
def test_llm_abort_request(sampling_params):
    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
    run_llm_abort_request(llm=llm, sampling_params=sampling_params)


@contextmanager
def _validate_invalid_token_error_scope():
    with pytest.raises(RuntimeError) as exc_info:
        yield
    assert "Token ID out of range" in str(exc_info.value)


@force_ampere
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
def test_llm_perf_metrics():
    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
    sampling_params = SamplingParams(max_tokens=10, return_perf_metrics=True)
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


def llama_7b_lora_from_dir_test_harness(**llm_kwargs) -> None:
    lora_config = LoraConfig(
        lora_dir=[f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"],
        max_lora_rank=8,
        max_loras=2,
        max_cpu_loras=2)
    llm = LLM(
        model=f"{llm_models_root()}/llama-models/llama-7b-hf",
        lora_config=lora_config,
        # Disable CUDA graph
        # TODO: remove this once we have a proper fix for CUDA graph in LoRA
        cuda_graph_config=None,
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
def test_llama_7b_lora():
    llama_7b_lora_from_dir_test_harness()


@skip_gpu_memory_less_than_40gb
def test_llama_7b_lora_default_modules() -> None:
    lora_config = LoraConfig(max_lora_rank=64, max_loras=2, max_cpu_loras=2)

    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"

    llm = LLM(
        model=hf_model_dir,
        lora_config=lora_config,
        # Disable CUDA graph
        # TODO: remove this once we have a proper fix for CUDA graph in LoRA
        cuda_graph_config=None)

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
        max_cpu_loras: int, repeat_calls: int, repeats_per_call: int):
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
        # Disable CUDA graph
        # TODO: remove this once we have a proper fix for CUDA graph in LoRA
        cuda_graph_config=None)


@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora_evict_and_reload_lora_gpu_cache():
    """Test eviction and re-loading a previously evicted adapter from the LoRA GPU cache, within a single
    llm.generate call, that's repeated twice.
    """  # noqa: D205
    _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call=[2],
        max_loras=1,
        max_cpu_loras=2,
        repeat_calls=2,
        repeats_per_call=3)


@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora_evict_and_load_new_adapters_in_cpu_and_gpu_cache():
    """Test eviction and loading of new adapters in the evicted space, over several llm.generate calls, with LoRA GPU
    cache size < LoRA CPU cache size.
    """  # noqa: D205
    _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call=[2, 2, 2],
        max_loras=1,
        max_cpu_loras=3,
        repeat_calls=1,
        repeats_per_call=1)


@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora_read_from_cache_after_insert():
    """Test that loading and then using the same adapters loaded in cache works."""
    _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call=[3],
        max_loras=3,
        max_cpu_loras=3,
        repeat_calls=2,
        repeats_per_call=1)


@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora_evict_and_reload_evicted_adapters_in_cpu_and_gpu_cache(
):
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
        repeats_per_call=1)


@skip_gpu_memory_less_than_40gb
def test_llama_7b_peft_cache_config_affects_peft_cache_size():
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
            # Disable CUDA graph
            # TODO: remove this once we have a proper fix for CUDA graph in LoRA
            cuda_graph_config=None)

    # Test that too small PeftCacheConfig.device_cache_percent causes failure
    with pytest.raises(RuntimeError):
        check_llama_7b_multi_lora_from_request_test_harness(
            LLM,
            lora_config=lora_config_no_cache_size_values,
            peft_cache_config=PeftCacheConfig(device_cache_percent=0.0000001),
            # Disable CUDA graph
            # TODO: remove this once we have a proper fix for CUDA graph in LoRA
            cuda_graph_config=None)


@skip_gpu_memory_less_than_40gb
def test_llama_7b_lora_config_overrides_peft_cache_config():
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
        # Disable CUDA graph
        # TODO: remove this once we have a proper fix for CUDA graph in LoRA
        cuda_graph_config=None)


# TODO smor: currently Nemotron-Super-49B-v1 with LoRA memory consumption is overly high
# https://jirasw.nvidia.com/browse/TRTLLM-5045
@pytest.mark.skip(reason="https://nvbugs/5448464")
@skip_gpu_memory_less_than_138gb
def test_nemotron_nas_lora() -> None:
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
    )

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
def test_llama_3_1_8b_fp8_with_bf16_lora() -> None:
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

    llm = LLM(
        model_dir,
        lora_config=lora_config,
        # Disable CUDA graph
        # TODO: remove this once we have a proper fix for CUDA graph in LoRA
        cuda_graph_config=None)

    try:
        output = llm.generate(prompt,
                              SamplingParams(max_tokens=20),
                              lora_request=[lora_req])
    finally:
        llm.shutdown()
    assert similar(output.outputs[0].text, reference)


@skip_gpu_memory_less_than_80gb
def test_bielik_11b_v2_2_instruct_multi_lora() -> None:
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
        llm = LLM(
            model_dir,
            lora_config=trtllm_lora_config,
            # Disable CUDA graph
            # TODO: remove this once we have a proper fix for CUDA graph in LoRA
            cuda_graph_config=None)

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


def test_gemma3_1b_instruct_multi_lora() -> None:
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
                  kv_cache_config=kv_cache_config)

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
def test_gqa_nemo_lora(tmp_path):
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

    def test_max_num_token_check(self):
        """ LLM should raise error when got prompt length exceed the valid range. """
        llm = LLM(llama_model_path,
                  kv_cache_config=global_kvcache_config,
                  max_num_tokens=100)

        with pytest.raises(ValueError,
                           match="should not exceed max_num_tokens"):
            ids = [random.randint(10, 100) for _ in range(101)]
            llm.generate([ids])


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

    def test_max_num_token_check(self):
        """ LLM should raise error when got prompt length exceed the valid range. """
        llm = LLM(llama_model_path,
                  kv_cache_config=global_kvcache_config,
                  max_num_tokens=100)

        with pytest.raises(ValueError,
                           match="should not exceed max_num_tokens"):
            ids = [random.randint(10, 100) for _ in range(101)]
            llm.generate([ids])


@skip_ray
def test_llm_rpc():
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

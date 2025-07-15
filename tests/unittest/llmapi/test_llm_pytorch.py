from collections import OrderedDict

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from .test_llm import (
    get_model_path, global_kvcache_config, llama_model_path,
    llm_get_stats_async_test_harness, llm_get_stats_test_harness, prompts,
    run_llm_abort_request, run_llm_with_postprocess_parallel_and_result_handler,
    tinyllama_logits_processor_test_harness, _test_llm_capture_request_error)
from utils.util import EnvVarsContextManager, duplicate_list_to_length, flatten_list, force_ampere, run_function_in_sub_process, similar, skip_gpu_memory_less_than_40gb, skip_gpu_memory_less_than_80gb, skip_gpu_memory_less_than_138gb
from utils.llm_data import llm_models_root
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
import tempfile

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM

# isort: on


@force_ampere
def test_tinyllama_logits_processor():
    tinyllama_logits_processor_test_harness(backend="pytorch")


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
@pytest.mark.parametrize(
    "sampling_params",
    [
        SamplingParams()  # pytorch only supports n=1
    ])
def test_llm_abort_request(sampling_params):
    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
    run_llm_abort_request(llm=llm, sampling_params=sampling_params)


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


@pytest.mark.parametrize("streaming", [True, False])
def test_llm_with_postprocess_parallel_and_result_handler(streaming):
    run_llm_with_postprocess_parallel_and_result_handler(streaming,
                                                         "pytorch",
                                                         tp_size=1)


def llama_7b_lora_from_dir_test_harness(**llm_kwargs) -> None:
    lora_config = LoraConfig(
        lora_dir=[f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"],
        max_lora_rank=8)
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
        # TODO: remove this once we have a proper fix for CUDA graph in LoRA
        # assert similar(outputs[0].outputs[0].text, references[0])
        print(f"lora output: {outputs[0].outputs[0].text}")
        print(f"ref output: {references[0]}")
    finally:
        llm.shutdown()


def llama_7b_multi_lora_from_request_test_harness(**llm_kwargs) -> None:
    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"
    hf_lora_dir1 = f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"
    hf_lora_dir2 = f"{llm_models_root()}/llama-models/Japanese-Alpaca-LoRA-7b-v0"

    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    lora_config = LoraConfig(lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
                             max_lora_rank=8)
    # Disable CUDA graph
    # TODO: remove this once we have a proper fix for CUDA graph in LoRA
    llm = LLM(hf_model_dir,
              lora_config=lora_config,
              cuda_graph_config=None,
              **llm_kwargs)

    try:
        prompts = [
            "美国的首都在哪里? \n答案:",
            "美国的首都在哪里? \n答案:",
            "美国的首都在哪里? \n答案:",
            "アメリカ合衆国の首都はどこですか? \n答え:",
            "アメリカ合衆国の首都はどこですか? \n答え:",
            "アメリカ合衆国の首都はどこですか? \n答え:",
        ]
        references = [
            "沃尔玛\n\n## 新闻\n\n* ",
            "美国的首都是华盛顿。\n\n美国的",
            "纽约\n\n### カンファレンスの",
            "Washington, D.C.\nWashington, D.C. is the capital of the United",
            "华盛顿。\n\n英国の首都是什",
            "ワシントン\nQ1. アメリカ合衆国",
        ]
        lora_req1 = LoRARequest("luotuo", 1, hf_lora_dir1)
        lora_req2 = LoRARequest("Japanese", 2, hf_lora_dir2)
        sampling_params = SamplingParams(max_tokens=20)
        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=[
                                   None, lora_req1, lora_req2, None, lora_req1,
                                   lora_req2
                               ])
        for output, ref in zip(outputs, references):
            assert similar(output.outputs[0].text, ref)
    finally:
        llm.shutdown()


@skip_gpu_memory_less_than_40gb
def test_llama_7b_lora():
    llama_7b_lora_from_dir_test_harness()


@skip_gpu_memory_less_than_40gb
def test_llama_7b_lora_default_modules() -> None:
    lora_config = LoraConfig(max_lora_rank=64)

    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"

    llm = LLM(model=hf_model_dir, lora_config=lora_config)

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

        # assert similar(outputs[0].outputs[0].text, references[0])
        print(f"lora output: {outputs[0].outputs[0].text}")
        print(f"ref output: {references[0]}")
    finally:
        llm.shutdown()


@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora():
    llama_7b_multi_lora_from_request_test_harness()


def llama_7b_multi_unique_lora_adapters_from_request(
        lora_adapter_count_per_call: list[int], max_loras: int,
        max_cpu_loras: int, repeats: int, **llm_kwargs):
    total_lora_adapters = sum(lora_adapter_count_per_call)

    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"
    hf_lora_dirs = [
        f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1",
        f"{llm_models_root()}/llama-models/Japanese-Alpaca-LoRA-7b-v0"
    ]

    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    lora_config = LoraConfig(lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
                             max_lora_rank=8,
                             max_loras=max_loras,
                             max_cpu_loras=max_cpu_loras)
    llm = LLM(hf_model_dir, lora_config=lora_config, **llm_kwargs)

    # Each prompt should have a reference for every LoRA adapter dir (in the same order as in hf_lora_dirs)
    prompt_to_references = OrderedDict({
        "美国的首都在哪里? \n答案:": [
            "美国的首都是华盛顿。\n\n美国的",
            "纽约\n\n### カンファレンスの",
        ],
        "アメリカ合衆国の首都はどこですか? \n答え:": [
            "华盛顿。\n\n英国の首都是什",
            "ワシントン\nQ1. アメリカ合衆国",
        ],
    })

    prompts_to_generate = duplicate_list_to_length(
        flatten_list([[prompt] * len(hf_lora_dirs)
                      for prompt in prompt_to_references.keys()]),
        total_lora_adapters)
    references = duplicate_list_to_length(
        flatten_list(list(prompt_to_references.values())), total_lora_adapters)
    lora_requests = [
        LoRARequest(str(i), i, hf_lora_dirs[i % len(hf_lora_dirs)])
        for i in range(total_lora_adapters)
    ]

    # Perform repeats of the same requests to test reuse and reload of adapters previously unloaded from cache
    for i in range(repeats):
        last_idx = 0
        for adapter_count in lora_adapter_count_per_call:
            sampling_params = SamplingParams(max_tokens=20)
            outputs = llm.generate(
                prompts_to_generate[last_idx:last_idx + adapter_count],
                sampling_params,
                lora_request=lora_requests[last_idx:last_idx + adapter_count])
            for output, ref in zip(
                    outputs, references[last_idx:last_idx + adapter_count]):
                assert similar(output.outputs[0].text, ref)
            last_idx += adapter_count


@pytest.mark.parametrize(
    "lora_adapter_count_per_call, max_loras, max_cpu_loras, repeats",
    [
        # Test eviction and loading of new adapters in the evicted space, within a single llm.generate call
        ([
            5,
        ], 2, 2, 1),
        # Test eviction and re-loading a previously evicted adapter from the LoRA GPU cache, within a single
        # llm.generate call
        ([
            2,
        ], 1, 2, 2),
        # Test eviction and loading of new adapters in the evicted space, over several llm.generate calls, with LoRA GPU
        # cache size < LoRA CPU cache size
        ([2, 2, 2], 1, 3, 1),
    ])
@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call: list[int], max_loras: int,
        max_cpu_loras: int, repeats: int):
    llama_7b_multi_unique_lora_adapters_from_request(
        lora_adapter_count_per_call, max_loras, max_cpu_loras, repeats)


@pytest.mark.parametrize(
    "lora_adapter_count_per_call, max_loras, max_cpu_loras, repeats",
    [
        # Test eviction, reloading new adapters and reloading previously evicted adapters from the LoRA CPU cache & GPU
        # cache over more than a single llm.generate call
        ([1, 1], 1, 1, 2),
        # Test eviction, reloading new adapters and reloading previously evicted adapters from the LoRA CPU cache & GPU
        # cache over a single llm.generate call
        ([
            5,
        ], 2, 2, 2),
    ])
@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora_load_previously_cpu_cache_evicted_adapter_fails(
        lora_adapter_count_per_call: list[int], max_loras: int,
        max_cpu_loras: int, repeats: int):
    """Tests that trying to load a LoRA adapter after it was evicted from CPU cache fails with the expected
    message, as this feature is currently not supported in favor of the performance improvement of not
    sending the LoRA weights with every request after the first time.
    """  # noqa: D205

    def _check_contains_expected_message(stdout: str, stderr: str):
        note_in_message = "Note that currently a request with LoRA task that was already loaded is sent" \
                          " without its LoRA weights to save its serialization, copy and deserialization, so if this" \
                          " LoRA task was evicted from LoRA CPU cache, then its reuse is currently not supported."
        return note_in_message in stderr

    with EnvVarsContextManager({"TLLM_WORKER_USE_SINGLE_PROCESS": "1"}):
        child_stdout, child_stderr = run_function_in_sub_process(
            target=llama_7b_multi_unique_lora_adapters_from_request,
            args=(lora_adapter_count_per_call, max_loras, max_cpu_loras,
                  repeats),
            kwargs={},
            stop_waiting_criteria=_check_contains_expected_message)
    print("STDOUT:")
    print(child_stdout)
    print("STDERR:")
    print(child_stderr)
    assert _check_contains_expected_message(child_stdout, child_stderr)


# TODO smor: currently Nemotron-Super-49B-v1 with LoRA memory consumption is overly high
# https://jirasw.nvidia.com/browse/TRTLLM-5045
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
def test_codellama_fp8_with_bf16_lora() -> None:
    model_dir = f"{llm_models_root()}/codellama/CodeLlama-7b-Instruct-hf/"
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                               kv_cache_quant_algo=QuantAlgo.FP8)

    target_modules = ['attn_q', 'attn_k', 'attn_v']

    # Set up temporary directory for LoRA adapters
    with tempfile.TemporaryDirectory() as lora_dir:
        print("Creating dummy LoRAs...")

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        hf_modules = ["q_proj", "k_proj", "v_proj"]

        lora_config = PeftLoraConfig(r=8,
                                     target_modules=hf_modules,
                                     bias="none",
                                     task_type="CAUSAL_LM")

        lora_paths = []
        for i in range(2):
            lora_model = get_peft_model(model, lora_config)
            for param in lora_model.parameters():
                param.data.zero_()
            lora_path = f"{lora_dir}/lora_{i}"
            lora_model.save_pretrained(lora_path)
            lora_paths.append(lora_path)

        lora_config = LoraConfig(lora_dir=lora_paths,
                                 lora_target_modules=target_modules,
                                 max_lora_rank=8)

        llm = LLM(model_dir, quant_config=quant_config, lora_config=lora_config)

        prompts = [
            "Write a function that calculates the Fibonacci sequence.",
            "Convert this C++ code to Python: int x = 0; x++;",
        ]

        lora_req1 = LoRARequest("lora-1", 0, lora_paths[0])
        lora_req2 = LoRARequest("lora-2", 1, lora_paths[1])
        lora_requests = [lora_req1, lora_req2]
        sampling_params = SamplingParams(max_tokens=200)

        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_requests)

        assert len(outputs) == 2


@skip_gpu_memory_less_than_80gb
def test_bielik_11b_v2_2_instruct_multi_lora() -> None:
    model_dir = f"{llm_models_root()}/Bielik-11B-v2.2-Instruct"

    target_modules = ['attn_q', 'attn_k', 'attn_v']

    # Set up temporary directory for LoRA adapters
    with tempfile.TemporaryDirectory() as lora_dir:
        print("Creating dummy LoRAs...")

        model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                     torch_dtype=torch.bfloat16,
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
                                        max_lora_rank=8)
        llm = LLM(model_dir, lora_config=trtllm_lora_config)

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

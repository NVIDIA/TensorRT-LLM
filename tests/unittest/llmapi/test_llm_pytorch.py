import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from .lora_test_utils import check_llama_7b_multi_unique_lora_adapters_from_request, create_mock_nemo_lora_checkpoint
from .test_llm import (get_model_path, global_kvcache_config, llama_model_path,
                       llm_get_stats_async_test_harness,
                       llm_get_stats_test_harness, prompts,
                       run_llm_abort_request,
                       run_llm_with_postprocess_parallel_and_result_handler,
                       tinyllama_logits_processor_test_harness)
from utils.util import (EnvVarsContextManager, force_ampere,
                        run_function_in_sub_process, similar,
                        skip_gpu_memory_less_than_40gb,
                        skip_gpu_memory_less_than_80gb,
                        skip_gpu_memory_less_than_138gb)
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

# NeMo LoRA test data
LORA_RANK_CONFIGS = [
    # (lora_rank, max_lora_rank, description)
    (8, 8, "rank_8"),
    (16, 16, "rank_16"),
    (4, 8, "rank_4_max_8"),
]


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


@pytest.mark.parametrize(
    "lora_adapter_count_per_call, max_loras, max_cpu_loras, repeat_calls, repeats_per_call",
    [
        # Test eviction and re-loading a previously evicted adapter from the LoRA GPU cache, within a single
        # llm.generate call, that's repeated twice.
        ([
            2,
        ], 1, 2, 2, 3),
        # Test eviction and loading of new adapters in the evicted space, over several llm.generate calls, with LoRA GPU
        # cache size < LoRA CPU cache size
        ([2, 2, 2], 1, 3, 1, 1),
    ])
@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora_evict_load_new_adapters(
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


@pytest.mark.parametrize(
    "lora_adapter_count_per_call, max_loras, max_cpu_loras, repeat_calls, repeats_per_call",
    [
        # Test eviction, reloading new adapters and reloading previously evicted adapters from the LoRA CPU cache & GPU
        # cache over multiple llm.generate call repeated twice (two calls with the same requests):
        # At the end of the 1st llm.generate call:
        #   The LoRA caches should contain adapters 1, 2 and shouldn't contain adapter 0 (it should have been evicted).
        # So in the 2nd call, the worker should:
        # - Send req0 with adapter 0 weights (because it was previously evicted)
        # - Send the other two requests without their adapter weights as they're already in LoRA CPU cache
        # Then, handling of req0 that has weights but not in the cache should evict one of the other two adapters from
        # the cache, causing that evicted adapter's request to fail because its weights aren't with the request and
        # aren't in LoRA cache.
        ([
            3,
        ], 2, 2, 2, 1),
    ])
@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora_load_previously_cpu_cache_evicted_adapter_fails(
        lora_adapter_count_per_call: list[int], max_loras: int,
        max_cpu_loras: int, repeat_calls: int, repeats_per_call: int):
    """Tests that trying to load a LoRA adapter after it was evicted from CPU cache fails with the expected
    message, as this feature is currently not supported in favor of the performance improvement of not
    sending the LoRA weights with every request after the first time.
    NOTE: This test assumes the requests are handled in the order they're sent, if that's not true, then this test
          may not get any error at all, which would cause it to fail.
    """  # noqa: D205

    def _check_contains_expected_message(stdout: str, stderr: str):
        note_in_message = "Note that currently a request with LoRA task that was already loaded is sent" \
                          " without its LoRA weights to save its serialization, copy and deserialization, so if this" \
                          " LoRA task was evicted from LoRA CPU cache, then its reuse is currently not supported."
        return note_in_message in stderr

    lora_config = LoraConfig(lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
                             max_lora_rank=8,
                             max_loras=max_loras,
                             max_cpu_loras=max_cpu_loras)
    with EnvVarsContextManager({"TLLM_WORKER_USE_SINGLE_PROCESS": "1"}):
        child_stdout, child_stderr = run_function_in_sub_process(
            target=check_llama_7b_multi_unique_lora_adapters_from_request,
            args=(lora_adapter_count_per_call, repeat_calls, repeats_per_call,
                  LLM),
            kwargs={
                "lora_config": lora_config,
                # Disable CUDA graph
                # TODO: remove this once we have a proper fix for CUDA graph in LoRA
                "cuda_graph_config": None
            },
            stop_waiting_criteria=_check_contains_expected_message)

    assert _check_contains_expected_message(child_stdout, child_stderr)


# TODO smor: currently Nemotron-Super-49B-v1 with LoRA memory consumption is overly high
# https://jirasw.nvidia.com/browse/TRTLLM-5045
@pytest.mark.skip(reason="https://nvbugs/5401210")
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
    """Test NeMo LoRA with GQA using TinyLlama."""
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
    )

    lora_config = LoraConfig(
        lora_dir=[str(nemo_path)],
        lora_ckpt_source="nemo",
        max_lora_rank=lora_rank,
    )

    model_path = get_model_path("llama-models-v2/TinyLlama-1.1B-Chat-v1.0")

    # First, generate without LoRA
    llm_no_lora = LLM(
        model=model_path,
        kv_cache_config=global_kvcache_config,
    )

    try:
        test_prompts = ["The capital of France is"]
        sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

        # Generate without LoRA
        outputs_no_lora = llm_no_lora.generate(test_prompts, sampling_params)
        no_lora_text = outputs_no_lora[0].outputs[0].text
    finally:
        llm_no_lora.shutdown()

    # Now generate with LoRA
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

        outputs = llm.generate(test_prompts,
                               sampling_params,
                               lora_request=[lora_req])

        # Validate output
        assert len(outputs) == 1
        assert outputs[0].outputs[0] is not None
        assert len(outputs[0].outputs[0].token_ids) > 0

        # Compare with and without LoRA
        lora_text = outputs[0].outputs[0].text
        assert lora_text, "Generated text with LoRA should not be empty"

        # Since LoRA weights are initialized with small values (* 0.01),
        # the outputs should start similarly but may diverge
        # Check that both outputs start with the expected completion "Paris"
        assert "Paris" in lora_text or "paris" in lora_text.lower(), \
            f"LoRA output should contain 'Paris', got: {lora_text}"
        assert "Paris" in no_lora_text or "paris" in no_lora_text.lower(), \
            f"No-LoRA output should contain 'Paris', got: {no_lora_text}"

        # For very small LoRA weights, at least the first few tokens should be similar
        # Check if the outputs are at least 60% similar or start with the same word
        if not similar(lora_text, no_lora_text, threshold=0.6):
            # If not similar enough, at least check they start with the same word
            first_word_lora = lora_text.split()[0] if lora_text.split() else ""
            first_word_no_lora = no_lora_text.split()[0] if no_lora_text.split(
            ) else ""
            assert first_word_lora.lower() == first_word_no_lora.lower(), \
                f"First words should match: LoRA='{first_word_lora}' vs No-LoRA='{first_word_no_lora}'"

    finally:
        llm.shutdown()

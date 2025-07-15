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
from utils.util import force_ampere, similar, skip_gpu_memory_less_than_40gb, skip_gpu_memory_less_than_80gb, skip_gpu_memory_less_than_138gb
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

import json
import tarfile
from pathlib import Path

# isort: on

# NeMo LoRA test data
LORA_RANK_CONFIGS = [
    # (lora_rank, max_lora_rank, description)
    (8, 8, "rank_8"),
    (16, 16, "rank_16"),
    (4, 8, "rank_4_max_8"),
]


def create_mock_nemo_lora_checkpoint(
    lora_dir: Path,
    hidden_size: int = 4096,
    num_layers: int = 32,
    lora_rank: int = 8,
    tp_size: int = 1,
) -> Path:
    """Create a minimal NeMo LoRA checkpoint for testing.

    This creates a .nemo tarfile with the expected structure:
    - model_weights.ckpt containing attn_qkv adapter weights
    - model_config.yaml with basic configuration

    Args:
        lora_dir: Directory to create the checkpoint in
        hidden_size: Model hidden size
        num_layers: Number of transformer layers
        lora_rank: LoRA rank
        tp_size: Tensor parallelism size

    Returns:
        Path to the created .nemo file
    """
    nemo_path = lora_dir / "test_lora.nemo"

    # Use temporary directory context manager for safe cleanup
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Create LoRA weights dict
        weights_dict = {}

        for layer_idx in range(num_layers):
            # NeMo uses this key format for QKV adapters
            key_prefix = f"model.layers.{layer_idx}.self_attention.adapter_layer.lora_kqv_adapter"

            # Create linear_in weights [lora_rank, hidden_size] with small random values
            linear_in_key = f"{key_prefix}.linear_in.weight"
            weights_dict[linear_in_key] = torch.randn(
                lora_rank, hidden_size, dtype=torch.float16) * 0.01

            # Create linear_out weights [3 * hidden_size, lora_rank] for QKV combined
            linear_out_key = f"{key_prefix}.linear_out.weight"
            weights_dict[linear_out_key] = torch.randn(
                3 * hidden_size, lora_rank, dtype=torch.float16) * 0.01

        ckpt_path = temp_dir / "model_weights.ckpt"
        torch.save(weights_dict, ckpt_path)

        # Create minimal config
        config = {
            "precision": "fp16",
            "trainer": {
                "num_nodes": 1,
                "devices": tp_size,
            },
            "model": {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
            },
            "lora": {
                "rank": lora_rank,
                "target_modules": ["attn_qkv"],
            }
        }

        config_path = temp_dir / "model_config.yaml"
        # Using JSON for simplicity since YAML parsing isn't critical for the test
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # Create .nemo tarfile
        with tarfile.open(nemo_path, 'w') as tar:
            tar.add(ckpt_path, arcname="model_weights.ckpt")
            tar.add(config_path, arcname="model_config.yaml")

    return nemo_path


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


# NeMo LoRA tests
@pytest.mark.parametrize("lora_rank,max_lora_rank,description",
                         LORA_RANK_CONFIGS)
def test_load_torch_nemo_lora_function(lora_rank, max_lora_rank, description):
    """Test load_torch_nemo_lora function with different LoRA rank configurations."""
    from tensorrt_llm.lora_manager import load_torch_nemo_lora

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a mock NeMo checkpoint
        nemo_path = create_mock_nemo_lora_checkpoint(
            temp_path,
            hidden_size=2048,
            num_layers=16,
            lora_rank=lora_rank,
        )

        # Test load_torch_nemo_lora
        lora_config = LoraConfig(
            lora_dir=[str(nemo_path)],
            lora_ckpt_source="nemo",
            max_lora_rank=max_lora_rank,
        )

        # This should not raise an error
        load_torch_nemo_lora(lora_config)

        # Verify configuration was set correctly
        assert lora_config.lora_target_modules == [
            "attn_qkv"
        ], f"Expected attn_qkv modules for {description}"
        assert lora_config.trtllm_modules_to_hf_modules == {
            "attn_qkv": "attn_qkv"
        }, f"Expected correct module mapping for {description}"


def test_nemo_lora_unsupported_modules_validation():
    """Test validation of unsupported modules in NeMo LoRA."""
    from tensorrt_llm.lora_manager import load_torch_nemo_lora

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a mock NeMo checkpoint
        nemo_path = create_mock_nemo_lora_checkpoint(
            temp_path,
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
def test_tinyllama_nemo_lora():
    """Test end-to-end generation with NeMo LoRA checkpoint."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a mock NeMo checkpoint for TinyLlama
        # TinyLlama has hidden_size=2048, num_layers=22
        nemo_path = create_mock_nemo_lora_checkpoint(
            temp_path,
            hidden_size=2048,
            num_layers=22,
            lora_rank=8,
        )

        # Create LoRA config for NeMo checkpoint
        lora_config = LoraConfig(
            lora_dir=[str(nemo_path)],
            lora_ckpt_source="nemo",
            max_lora_rank=8,
        )

        # Verify LoRA config is set up correctly
        assert lora_config.lora_ckpt_source == "nemo"
        assert len(lora_config.lora_dir) == 1
        print(f"✓ Created NeMo LoRA config: {nemo_path}")

        # Use TinyLlama for fast testing
        model_path = get_model_path("llama-models-v2/TinyLlama-1.1B-Chat-v1.0")

        # Create LLM with NeMo LoRA
        llm = LLM(
            model=model_path,
            lora_config=lora_config,
            kv_cache_config=global_kvcache_config,
        )

        try:
            # Test prompts
            test_prompts = [
                "Hello, how are you?",
                "What is the capital of France?",
            ]

            # Create LoRA request for the NeMo checkpoint
            lora_req = LoRARequest("nemo-task",
                                   0,
                                   str(nemo_path),
                                   lora_ckpt_source="nemo")

            # Verify LoRA request is configured correctly
            assert lora_req.ckpt_source == "nemo"
            assert lora_req.path == str(nemo_path)

            # Test with and without LoRA
            sampling_params = SamplingParams(max_tokens=20, temperature=0.0)

            # Generate with LoRA
            outputs_with_lora = llm.generate(test_prompts,
                                             sampling_params,
                                             lora_request=[lora_req, lora_req])

            # Generate without LoRA
            outputs_without_lora = llm.generate(test_prompts,
                                                sampling_params,
                                                lora_request=[None, None])

            # Basic validation - outputs should be generated without errors
            assert len(outputs_with_lora) == 2
            assert len(outputs_without_lora) == 2

            # Verify that generation completed successfully (may have minimal output with mock weights)
            for i in range(2):
                # Check that we got valid completion outputs
                assert outputs_with_lora[i].outputs[0] is not None
                assert outputs_without_lora[i].outputs[0] is not None
                # Check that token_ids are present (even if just EOS token)
                assert len(outputs_with_lora[i].outputs[0].token_ids) > 0
                assert len(outputs_without_lora[i].outputs[0].token_ids) > 0

            print(f"✓ NeMo LoRA generation completed successfully")
            print(
                f"✓ LoRA output tokens: {[len(out.outputs[0].token_ids) for out in outputs_with_lora]}"
            )
            print(
                f"✓ Base output tokens: {[len(out.outputs[0].token_ids) for out in outputs_without_lora]}"
            )

            # Test passes if generation completes without errors
            # Note: With mock LoRA weights, outputs may be minimal but that's expected

        finally:
            llm.shutdown()

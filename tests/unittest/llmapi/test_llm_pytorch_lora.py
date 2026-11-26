# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import pytest
import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from utils.llm_data import llm_models_root
from utils.util import (
    force_ampere,
    similar,
    skip_gpu_memory_less_than_40gb,
    skip_gpu_memory_less_than_138gb,
)

from tensorrt_llm import LLM
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

from .llm_test_utils import global_kvcache_config
from .lora_test_utils import create_mock_nemo_lora_checkpoint, test_lora_with_and_without_cuda_graph
from .model_paths import get_model_path


@skip_gpu_memory_less_than_138gb
@test_lora_with_and_without_cuda_graph
def test_nemotron_nas_lora(cuda_graph_config) -> None:
    lora_config = LoraConfig(
        lora_dir=[
            f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1-lora-adapter_r64"
        ],
        max_lora_rank=64,
        max_loras=1,
        max_cpu_loras=1,
    )

    llm = LLM(
        model=f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1",
        lora_config=lora_config,
        kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True),
        cuda_graph_config=cuda_graph_config,
        trust_remote_code=True,
    )

    try:
        prompts = [
            "Hello, how are you?",
            "Hello, how are you?",
        ]

        sampling_params = SamplingParams(max_tokens=3, add_special_tokens=False)
        lora_req = LoRARequest(
            "task-0",
            0,
            f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1-lora-adapter_r64",
        )
        lora_request = [lora_req, None]

        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

        assert similar(outputs[0].outputs[0].text, outputs[1].outputs[0].text)
    finally:
        llm.shutdown()


@test_lora_with_and_without_cuda_graph
def test_gemma3_1b_instruct_multi_lora(cuda_graph_config) -> None:
    model_dir = f"{llm_models_root()}/gemma/gemma-3-1b-it"

    target_modules = ["attn_q", "attn_k", "attn_v"]

    # Set up temporary directory for LoRA adapters
    with tempfile.TemporaryDirectory() as lora_dir:
        print("Creating dummy LoRAs...")

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="auto"
        )
        hf_modules = ["q_proj", "k_proj", "v_proj"]
        peft_lora_config = PeftLoraConfig(
            r=8, target_modules=hf_modules, bias="none", task_type="CAUSAL_LM"
        )
        lora_paths = []
        for i in range(2):
            lora_model = get_peft_model(model, peft_lora_config)
            for param in lora_model.parameters():
                param.data.zero_()
            lora_path = f"{lora_dir}/lora_{i}"
            lora_model.save_pretrained(lora_path)
            lora_paths.append(lora_path)

        trtllm_lora_config = LoraConfig(
            lora_dir=lora_paths,
            lora_target_modules=target_modules,
            max_lora_rank=8,
            max_loras=2,
            max_cpu_loras=2,
        )
        # Gemma3's non-inclusive sliding window requires KV cache reuse to be disabled.
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            enable_partial_reuse=False,
        )
        llm = LLM(
            model_dir,
            lora_config=trtllm_lora_config,
            kv_cache_config=kv_cache_config,
            cuda_graph_config=cuda_graph_config,
        )

        prompts = [
            "Is it ok to fill diesel in a petrol car?",
            "What is the capital of France?",
        ]
        lora_req1 = LoRARequest("lora-1", 0, lora_paths[0])
        lora_req2 = LoRARequest("lora-2", 1, lora_paths[1])
        lora_requests = [lora_req1, lora_req2]
        sampling_params = SamplingParams(max_tokens=200)

        outputs = llm.generate(prompts, sampling_params, lora_request=lora_requests)

        assert len(outputs) == 2


@skip_gpu_memory_less_than_40gb
def test_lora_many_adapters_no_memory_leak() -> None:
    """Verify GPU memory stays bounded when loading many unique LoRA adapters.

    Creates 20 dummy adapters but sets max_loras=2 and max_cpu_loras=4 to
    force eviction.  Without proper cleanup, _lora_weights can accumulate
    GPU tensors for every loaded adapter, causing unbounded memory growth.
    """
    model_dir = f"{llm_models_root()}/gemma/gemma-3-1b-it"
    num_adapters = 20
    target_modules = ["attn_q", "attn_k", "attn_v"]

    with tempfile.TemporaryDirectory() as lora_dir:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="auto"
        )
        hf_modules = ["q_proj", "k_proj", "v_proj"]
        peft_lora_config = PeftLoraConfig(
            r=8, target_modules=hf_modules, bias="none", task_type="CAUSAL_LM"
        )
        lora_paths = []
        for i in range(num_adapters):
            lora_model = get_peft_model(model, peft_lora_config)
            for param in lora_model.parameters():
                param.data.zero_()
            lora_path = f"{lora_dir}/lora_{i}"
            lora_model.save_pretrained(lora_path)
            lora_paths.append(lora_path)

        del model
        torch.cuda.empty_cache()

        trtllm_lora_config = LoraConfig(
            lora_dir=lora_paths[:1],
            lora_target_modules=target_modules,
            max_lora_rank=8,
            max_loras=2,
            max_cpu_loras=4,
        )
        kv_cache_config = KvCacheConfig(enable_block_reuse=False, enable_partial_reuse=False)
        llm = LLM(model_dir, lora_config=trtllm_lora_config, kv_cache_config=kv_cache_config)

        sampling_params = SamplingParams(max_tokens=20)
        warmup_count = 5

        mem_samples = []
        for i in range(num_adapters):
            lora_req = LoRARequest(f"lora-{i}", i, lora_paths[i])
            output = llm.generate("Hello, tell me a story.", sampling_params, lora_request=lora_req)
            assert output.outputs[0].text != ""

            if i >= warmup_count:
                mem_samples.append(torch.cuda.memory_allocated())

        num_measured = len(mem_samples)
        assert num_measured >= 2, "Not enough samples to measure growth"

        total_growth = mem_samples[-1] - mem_samples[0]
        per_adapter_mb = (total_growth / (num_measured - 1)) / (1024 * 1024)

        # Each adapter is ~3 MB on GPU (r=8, 3 modules, 26 layers, bf16).
        # The C++ PeftCacheManager handles eviction and _lora_weights
        # stays empty, so per-adapter growth should be ~0.  If GPU tensors
        # leak, we would see ~3 MB/adapter of linear growth.  Threshold
        # of 1 MB/adapter catches leaks while tolerating noise from
        # allocator fragmentation averaged over many samples.
        max_per_adapter_mb = 1.0
        assert per_adapter_mb < max_per_adapter_mb, (
            f"GPU memory growing at {per_adapter_mb:.2f} MB/adapter over "
            f"{num_measured} adapters (total {total_growth / (1024**2):.1f} MB). "
            f"Possible _lora_weights leak."
        )


@pytest.mark.parametrize(
    "lora_rank,max_lora_rank,description",
    [
        # (lora_rank, max_lora_rank, description)
        (8, 8, "rank_8"),
        (16, 16, "rank_16"),
        (4, 8, "rank_4_max_8"),
    ],
)
def test_load_torch_nemo_lora_function(tmp_path, lora_rank, max_lora_rank, description):
    """Test load_torch_nemo_lora function with different LoRA rank configurations."""
    from tensorrt_llm._torch.peft.lora.manager import load_torch_nemo_lora

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

    assert lora_config.lora_target_modules == ["attn_qkv"], (
        f"Expected attn_qkv modules for {description}"
    )
    assert lora_config.trtllm_modules_to_hf_modules == {"attn_qkv": "attn_qkv"}, (
        f"Expected correct module mapping for {description}"
    )


def test_nemo_lora_unsupported_modules_validation(tmp_path):
    """Test validation of unsupported modules in NeMo LoRA."""
    from tensorrt_llm._torch.peft.lora.manager import load_torch_nemo_lora

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
        lora_target_modules=["attn_qkv", "mlp_h_to_4h"],  # mlp_h_to_4h not supported
        max_lora_rank=8,
    )

    with pytest.raises(ValueError, match="NeMo LoRA only supports"):
        load_torch_nemo_lora(invalid_config)


@force_ampere
@test_lora_with_and_without_cuda_graph
def test_gqa_nemo_lora(tmp_path, cuda_graph_config):
    """Test NeMo-format LoRA checkpoint loading and GQA support in TinyLlama.

    A NeMo-format LoRA checkpoint with grouped query attention loads into a
    TinyLlama model and produces deterministic output with temperature=0.
    The adapted output differs from the output without LoRA.

    The dummy checkpoint uses seed=42 for both output comparisons.
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
        lora_req = LoRARequest("tinyllama-gqa-test", 0, str(nemo_path), lora_ckpt_source="nemo")

        lora_outputs = llm.generate(test_prompts, sampling_params, lora_request=[lora_req])

        # For the above deterministic dummy LoRA checkpoint,
        # with temperature=0.0,
        # the expected output text should always be the same.
        assert lora_outputs[0].outputs[0].text == expected_lora_text_output, (
            f"Expected output text: {expected_lora_text_output}, "
            f"got: {lora_outputs[0].outputs[0].text}"
        )
        assert len(lora_outputs) == 1

        # Generate without LoRA.
        # The LoRA weights are tuned/large enough that
        # they differ from a no-LoRA run.
        base_outputs = llm.generate(test_prompts, sampling_params)
        assert base_outputs[0].outputs[0].text != expected_lora_text_output, (
            f"No-LoRA output should differ from expected output text: {expected_lora_text_output}, "
            f"got: {base_outputs[0].outputs[0].text}"
        )
    finally:
        llm.shutdown()

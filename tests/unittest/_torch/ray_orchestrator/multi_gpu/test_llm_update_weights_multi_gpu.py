import re
from typing import Callable

import pytest
from _torch.ray_orchestrator.single_gpu.test_llm_update_weights import (
    RefHFModelWithIPCHandles,
    compare_logits,
    run_generate,
)
from transformers import AutoTokenizer
from utils.llm_data import llm_models_root
from utils.util import skip_pre_blackwell

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


@skip_pre_blackwell
@pytest.mark.parametrize(
    "model_dir, fp8_model_dir",
    [
        ("Qwen3/Qwen3-8B", "Qwen3/Qwen3-8B-FP8"),
        ("Qwen3/Qwen3-30B-A3B", "Qwen3/Qwen3-30B-A3B-FP8"),
    ],
)
def test_llm_update_weights_with_quant_config(model_dir, fp8_model_dir):
    model_dir = str(llm_models_root() / model_dir)
    fp8_model_dir = str(llm_models_root() / fp8_model_dir)
    additional_kwargs = {}
    if "Qwen3/Qwen3-30B-A3B" in model_dir:
        additional_kwargs["moe_config"] = {
            "backend": "DEEPGEMM",
        }
    num_hidden_layers = 1
    hf_model = RefHFModelWithIPCHandles(fp8_model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(fp8_model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    llm = LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=2,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        model_kwargs={
            "num_hidden_layers": num_hidden_layers,
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        },
        **additional_kwargs,
    )

    # Generate texts from the prompts.
    prompts_texts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
    del tokenizer
    sampling_params = SamplingParams(temperature=0, return_generation_logits=True, max_tokens=1024)

    ipc_handles = hf_model.get_weight_ipc_handles_serialized([0, 1])

    llm._collective_rpc("update_weights", (ipc_handles,))
    # Finalize the update weights
    llm._collective_rpc("update_weights", (None,))

    llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
    compare_logits(llm_logits, ref_logits)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "model_dir, fp8_model_dir",
    [
        ("Qwen3/Qwen3-8B", "Qwen3/Qwen3-8B-FP8"),
        ("Qwen3/Qwen3-30B-A3B", "Qwen3/Qwen3-30B-A3B-FP8"),
    ],
)
def test_llm_partial_update_weights_with_quant_config(model_dir, fp8_model_dir):
    model_dir = str(llm_models_root() / model_dir)
    fp8_model_dir = str(llm_models_root() / fp8_model_dir)
    additional_kwargs = {}
    if "Qwen3/Qwen3-30B-A3B" in model_dir:
        additional_kwargs["moe_config"] = {
            "backend": "DEEPGEMM",
        }
    num_hidden_layers = 1
    hf_model = RefHFModelWithIPCHandles(fp8_model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(fp8_model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    llm = LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=2,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        model_kwargs={
            "num_hidden_layers": num_hidden_layers,
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        },
        **additional_kwargs,
    )

    # Generate texts from the prompts.
    prompts_texts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
    del tokenizer

    sampling_params = SamplingParams(temperature=0, return_generation_logits=True, max_tokens=1024)

    def common_filter(filter_name: str) -> Callable[[str], bool]:
        def filter_fn(name: str) -> bool:
            return name.endswith(filter_name)

        return filter_fn

    # Generate filter_list from model weight keys by removing layer prefix
    # e.g., "model.layers.41.input_layernorm.weight" -> "input_layernorm.weight"
    layer_prefix_pattern = re.compile(r"^model\.layers\.\d+\.")
    filter_set = set()
    for name, _ in hf_model.all_weights[hf_model.device_id]:
        suffix = layer_prefix_pattern.sub("", name)
        filter_set.add(suffix)
    filter_list = list(filter_set)

    for filter_name in filter_list:
        weight_filter = common_filter(filter_name=filter_name)
        ipc_handles = hf_model.get_weight_ipc_handles_serialized(
            [0, 1], weight_filter=weight_filter
        )
        llm._collective_rpc("update_weights", (ipc_handles,))
    # Finalize the update weights
    llm._collective_rpc("update_weights", (None,))

    llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
    compare_logits(llm_logits, ref_logits)

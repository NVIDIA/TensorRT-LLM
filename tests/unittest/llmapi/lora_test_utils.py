import json
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, OrderedDict, Type

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import duplicate_list_to_length, flatten_list, similar

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_params import \
    CudaGraphLoraParams
from tensorrt_llm._torch.peft.lora.layer import (GroupedGemmParamsInput,
                                                 LoraLayer,
                                                 compare_grouped_gemm_params)
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig

_RU_LORA_ADAPTER_PROMPTS = [
    "Назови главную площадь в центре Москвы.",
    "Напиши полное предложение, описывающее, что в музее не хватает женских скульптур. Используй фразу \"не хватает\".",
    "Что означает выражение \"водить за нос\"? Объясни в двух словах.",
]


def _generate_phi3_response_lora_fused_modules(llm_class: Type[BaseLLM],
                                               prompts: List[str],
                                               **extra_llm_kwargs) -> List[str]:
    """Generates responses with LoRA requests with the Phi-3-mini-4k-instruct-ru-lora adapter.
    The used LoRA adapter has fused attention QKV and fused MLP gate up proj modules.
    Returns the generated texts.
    """  # noqa: D205
    hf_model_dir = f"{llm_models_root()}/Phi-3/Phi-3-mini-4k-instruct"
    hf_lora_dir = f"{llm_models_root()}/lora/phi/Phi-3-mini-4k-instruct-ru-lora"

    lora_req = LoRARequest("ru-lora", 0, hf_lora_dir)
    sampling_params = SamplingParams(max_tokens=20)

    lora_config = LoraConfig(lora_dir=[hf_lora_dir],
                             max_lora_rank=16,
                             max_loras=2,
                             max_cpu_loras=2)

    lora_requests = [lora_req] * len(prompts)
    with llm_class(hf_model_dir, lora_config=lora_config,
                   **extra_llm_kwargs) as llm:
        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_requests)

    return [output.outputs[0].text for output in outputs]


def check_phi3_lora_fused_modules_output_tp2_identical_to_tp1(
        llm_class: Type[BaseLLM], **extra_llm_kwargs) -> None:
    """Tests the output with LoRA requests with the Phi-3-mini-4k-instruct-ru-lora adapter with TP=2 is identical to
    the output with TP=1.
    That LoRA adapter has fused attention QKV and fused MLP gate up proj modules.
    """  # noqa: D205
    extra_llm_kwargs["tensor_parallel_size"] = 1
    outputs_tp1 = _generate_phi3_response_lora_fused_modules(
        llm_class, _RU_LORA_ADAPTER_PROMPTS, **extra_llm_kwargs)

    extra_llm_kwargs["tensor_parallel_size"] = 2
    outputs_tp2 = _generate_phi3_response_lora_fused_modules(
        llm_class, _RU_LORA_ADAPTER_PROMPTS, **extra_llm_kwargs)

    assert outputs_tp1 == outputs_tp2


def check_llama_7b_multi_unique_lora_adapters_from_request(
        lora_adapter_count_per_call: List[int], repeat_calls: int,
        repeats_per_call: int, llm_class: Type[BaseLLM], **llm_kwargs):
    """Calls llm.generate s.t. for each C in lora_adapter_count_per_call, llm.generate is called with C requests
    repeated 'repeats_per_call' times, where each request is configured with a unique LoRA adapter ID.
    This entire process is done in a loop 'repeats_per_call' times with the same requests.
    Asserts the output of each llm.generate call is similar to the expected.
    """  # noqa: D205
    total_lora_adapters = sum(lora_adapter_count_per_call)
    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"
    hf_lora_dirs = [
        f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1",
        f"{llm_models_root()}/llama-models/Japanese-Alpaca-LoRA-7b-v0"
    ]
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
    llm = llm_class(hf_model_dir, **llm_kwargs)

    # Perform repeats of the same requests to test reuse and reload of adapters previously unloaded from cache
    try:
        for _ in range(repeat_calls):
            last_idx = 0
            for adapter_count in lora_adapter_count_per_call:
                sampling_params = SamplingParams(max_tokens=20)
                outputs = llm.generate(
                    prompts_to_generate[last_idx:last_idx + adapter_count] *
                    repeats_per_call,
                    sampling_params,
                    lora_request=lora_requests[last_idx:last_idx +
                                               adapter_count] *
                    repeats_per_call)
                for output, ref in zip(
                        outputs, references[last_idx:last_idx + adapter_count] *
                        repeats_per_call):
                    assert similar(output.outputs[0].text, ref)
                last_idx += adapter_count
    finally:
        llm.shutdown()


def check_llama_7b_multi_lora_from_request_test_harness(
        llm_class: Type[BaseLLM], **llm_kwargs) -> None:
    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"
    hf_lora_dir1 = f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"
    hf_lora_dir2 = f"{llm_models_root()}/llama-models/Japanese-Alpaca-LoRA-7b-v0"
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
    key_words = [
        "沃尔玛",
        "华盛顿",
        "纽约",
        "Washington",
        "华盛顿",
        "ワシントン",
    ]
    lora_req1 = LoRARequest("luotuo", 1, hf_lora_dir1)
    lora_req2 = LoRARequest("Japanese", 2, hf_lora_dir2)
    sampling_params = SamplingParams(max_tokens=20)

    llm = llm_class(hf_model_dir, **llm_kwargs)
    try:
        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=[
                                   None, lora_req1, lora_req2, None, lora_req1,
                                   lora_req2
                               ])
    finally:
        llm.shutdown()
    for output, ref, key_word in zip(outputs, references, key_words):
        assert similar(output.outputs[0].text,
                       ref) or key_word in output.outputs[0].text


def create_mock_nemo_lora_checkpoint(
        lora_dir: Path,
        hidden_size: int = 4096,
        num_layers: int = 32,
        lora_rank: int = 8,
        tp_size: int = 1,
        num_attention_heads: int = 32,
        num_kv_heads: int = None,  # If None, defaults to num_attention_heads
        dtype: torch.dtype = torch.float16,
        seed: int = None,  # For deterministic weight initialization
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
        num_attention_heads: Number of query attention heads
        num_kv_heads: Number of key/value heads (for GQA). If None, equals num_attention_heads
        dtype: Data type for the weights (default: torch.float16)

    Returns:
        Path to the created .nemo file
    """

    # Validate parameters
    if hidden_size % num_attention_heads != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by "
                         f"num_attention_heads ({num_attention_heads})")

    # Default to standard MHA if not specified
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads

    if num_attention_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_attention_heads ({num_attention_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads}) for GQA")

    nemo_path = lora_dir / "test_lora.nemo"

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Set random seed for deterministic weight initialization
        if seed is not None:
            torch.manual_seed(seed)

        weights_dict = {}

        head_dim = hidden_size // num_attention_heads
        kv_hidden_size = head_dim * num_kv_heads

        qkv_output_dim = hidden_size + 2 * kv_hidden_size

        # NOTE:
        # for seed=42, and coefficient=0.02, the expected outputs are hardcoded
        # in the test `test_llm_pytorch.py::test_gqa_nemo_lora`.
        # Therefore changing "WEIGHTS_COEFFICIENT" or the seed will break the test.
        WEIGHTS_COEFFICIENT = 0.02
        for layer_idx in range(num_layers):
            key_prefix = f"model.layers.{layer_idx}.self_attention.adapter_layer.lora_kqv_adapter"

            # Create linear_in weights [lora_rank, hidden_size] with small random values
            linear_in_key = f"{key_prefix}.linear_in.weight"
            weights_dict[linear_in_key] = torch.randn(
                lora_rank, hidden_size, dtype=dtype) * WEIGHTS_COEFFICIENT

            # Create linear_out weights [qkv_output_dim, lora_rank] for fused QKV
            # This is the key difference for GQA - the output dimension changes
            linear_out_key = f"{key_prefix}.linear_out.weight"
            weights_dict[linear_out_key] = torch.randn(
                qkv_output_dim, lora_rank, dtype=dtype) * WEIGHTS_COEFFICIENT

        ckpt_path = temp_dir / "model_weights.ckpt"
        torch.save(weights_dict, ckpt_path)

        config = {
            "precision": "fp16" if dtype == torch.float16 else "bf16",
            "trainer": {
                "num_nodes": 1,
                "devices": tp_size,
            },
            "model": {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_attention_heads": num_attention_heads,
                "num_query_groups": num_kv_heads,  # This is the key for GQA
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

        with tarfile.open(nemo_path, 'w') as tar:
            tar.add(ckpt_path, arcname="model_weights.ckpt")
            tar.add(config_path, arcname="model_config.yaml")

    return nemo_path


@dataclass
class CUDAGraphLoRATestParams:
    batch_slot_ids: list[int]
    input_hidden_size: int
    slot_ranks: list[int]
    max_lora_rank: int
    output_hidden_sizes: list[int]
    layer_module_mask: torch.Tensor | None | bool
    dtype: torch.dtype
    seed: int

    def __post_init__(self):
        assert self.layer_module_mask is None or isinstance(
            self.layer_module_mask,
            bool) or self.layer_module_mask.shape == (self.module_count,
                                                      self.slot_count)
        assert all(0 <= idx <= self.slot_count for idx in self.batch_slot_ids)
        assert all(0 <= rank <= self.max_lora_rank for rank in self.slot_ranks)
        if isinstance(self.layer_module_mask, torch.Tensor):
            self.layer_module_mask = self.layer_module_mask.to(dtype=torch.bool)
        elif self.layer_module_mask is not None:
            self.layer_module_mask = bool(self.layer_module_mask)
        else:
            self.layer_module_mask = True

    @property
    def module_count(self):
        return len(self.output_hidden_sizes)

    @property
    def slot_count(self):
        return len(self.slot_ranks)

    @property
    def batch_size(self):
        return len(self.batch_slot_ids)

    @property
    def sum_output_hidden_size(self):
        return sum(self.output_hidden_sizes)


def create_grouped_gemm_params_filler_input(
    test_params: CUDAGraphLoRATestParams = CUDAGraphLoRATestParams(
        batch_slot_ids=[0, 3, 3, 4, 5, 8],
        input_hidden_size=4096,
        slot_ranks=[8, 12, 4, 3] * 2,
        max_lora_rank=64,
        output_hidden_sizes=[4096, 4096],
        layer_module_mask=None,
        dtype=torch.bfloat16,
        seed=42,
    ),
) -> tuple[GroupedGemmParamsInput, LoraLayer]:

    with torch.random.fork_rng():
        torch.manual_seed(test_params.seed)
        shape_2d = (test_params.module_count, test_params.slot_count)

        x = torch.randn(test_params.batch_size,
                        test_params.input_hidden_size,
                        dtype=test_params.dtype,
                        device="cuda")
        output_buffer = torch.randn(test_params.batch_size,
                                    test_params.sum_output_hidden_size,
                                    dtype=test_params.dtype,
                                    device="cuda")
        b_ptrs = torch.randint(1, 1000000, shape_2d, dtype=LoraLayer.PTR_DTYPE)
        b_prime_ptrs = torch.randint(1,
                                     1000000,
                                     shape_2d,
                                     dtype=LoraLayer.PTR_DTYPE)

        b_ptrs *= test_params.layer_module_mask
        b_prime_ptrs *= test_params.layer_module_mask

        b_ptrs = b_ptrs.to(device="cuda")
        b_prime_ptrs = b_prime_ptrs.to(device="cuda")
        slot_ranks = torch.tensor(test_params.slot_ranks,
                                  dtype=LoraLayer.SIZES_DTYPE,
                                  device="cuda")

        intermediate_buffer = torch.randn(test_params.module_count,
                                          test_params.batch_size,
                                          test_params.max_lora_rank,
                                          dtype=test_params.dtype,
                                          device="cuda")
        slot_counts = CudaGraphLoraParams.get_slot_counts(
            test_params.batch_slot_ids, test_params.slot_count)
        slot_offsets_full = CudaGraphLoraParams.get_offset_from_counts(
            slot_counts, full=True)
        sorted_ids = CudaGraphLoraParams.get_sorted_indices(
            test_params.batch_slot_ids)

        slot_offsets_full = slot_offsets_full.to(device="cuda",
                                                 dtype=torch.int64)
        slot_counts = slot_counts.to(device="cuda", dtype=torch.int32)
        sorted_ids = sorted_ids.to(device="cuda", dtype=torch.int64)

        layer = LoraLayer([0] * test_params.module_count,
                          test_params.output_hidden_sizes)
        inputs = GroupedGemmParamsInput(x=x,
                                        output_buffer=output_buffer,
                                        intermediate_buffer=intermediate_buffer,
                                        max_lora_size=test_params.slot_count,
                                        max_rank=test_params.max_lora_rank,
                                        slot_counts=slot_counts,
                                        slot_ranks=slot_ranks,
                                        slot_offsets_full=slot_offsets_full,
                                        sorted_ids=sorted_ids,
                                        b_ptrs=b_ptrs,
                                        b_prime_ptrs=b_prime_ptrs)
        return inputs, layer


def compare_cuda_graph_lora_params_filler(test_params: CUDAGraphLoRATestParams):
    grouped_gemm_params_filler_input, layer = create_grouped_gemm_params_filler_input(
        test_params)
    output_fused = layer._prepare_grouped_gemm_buffers_fused(
        grouped_gemm_params_filler_input)

    assert torch.all(
        grouped_gemm_params_filler_input.intermediate_buffer == 0
    ), f"intermediate_buffer is not all zeros: {grouped_gemm_params_filler_input.intermediate_buffer}; non zero / zeros: {(grouped_gemm_params_filler_input.output_buffer != 0).sum()} / {grouped_gemm_params_filler_input.output_buffer.numel()}"
    assert torch.all(
        grouped_gemm_params_filler_input.output_buffer == 0
    ), f"output_buffer is not all zeros: {grouped_gemm_params_filler_input.output_buffer}; non zero / zeros: {(grouped_gemm_params_filler_input.output_buffer != 0).sum()} / {grouped_gemm_params_filler_input.output_buffer.numel()}"

    output_pytorch = layer.prepare_grouped_gemm_buffers(
        grouped_gemm_params_filler_input)
    compare_grouped_gemm_params(output_fused,
                                output_pytorch,
                                grouped_gemm_params_filler_input,
                                params_to_store_msg=None)


test_lora_with_and_without_cuda_graph = pytest.mark.parametrize(
    "cuda_graph_config", [CudaGraphConfig(max_batch_size=10), None])

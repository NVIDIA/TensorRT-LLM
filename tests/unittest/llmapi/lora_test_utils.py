# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator, List, Optional, OrderedDict, Tuple, Type, Union

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from utils.llm_data import llm_models_root
from utils.util import duplicate_list_to_length, flatten_list, similar

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_params import \
    CudaGraphLoraParams
from tensorrt_llm._torch.peft.lora.layer import (GroupedGemmParamsInput,
                                                 GroupedGemmParamsOutput,
                                                 LoraLayer)
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig

from .test_utils import DelayedAssert

QWEN3_MODEL_DIR = Path(llm_models_root()) / "Qwen3" / "Qwen3-0.6B"


def create_qwen3_lora_adapter(adapter_dir: Path, rank: int = 8) -> Path:
    """Create a deterministic Qwen3 attention LoRA adapter for server tests."""
    config = AutoConfig.from_pretrained(QWEN3_MODEL_DIR)
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)
    projection_output_sizes = {
        "q_proj": config.num_attention_heads * head_dim,
        "k_proj": config.num_key_value_heads * head_dim,
        "v_proj": config.num_key_value_heads * head_dim,
    }
    generator = torch.Generator().manual_seed(42)
    weights = {}
    for module_name, output_size in projection_output_sizes.items():
        prefix = f"base_model.model.model.layers.0.self_attn.{module_name}"
        weights[f"{prefix}.lora_A.weight"] = torch.randn(
            rank, config.hidden_size, generator=generator) * 0.1
        weights[f"{prefix}.lora_B.weight"] = torch.randn(
            output_size, rank, generator=generator) * 0.1

    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(weights, adapter_dir / "adapter_model.safetensors")
    adapter_config = {
        "base_model_name_or_path": str(QWEN3_MODEL_DIR),
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": list(projection_output_sizes),
        "task_type": "CAUSAL_LM",
    }
    with open(adapter_dir / "adapter_config.json", "w",
              encoding="utf-8") as config_file:
        json.dump(adapter_config, config_file)
    return adapter_dir


@contextmanager
def qwen3_lora_adapter() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield create_qwen3_lora_adapter(Path(temp_dir) / "qwen3-lora")


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
    batch_slot_ids: List[int]
    input_hidden_size: int
    slot_ranks: List[int]
    max_lora_rank: int
    output_hidden_sizes: List[int]
    layer_module_mask: Optional[Union[torch.Tensor, bool]]
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
    test_params: Optional[CUDAGraphLoRATestParams] = None
) -> Tuple[GroupedGemmParamsInput, LoraLayer]:
    if test_params is None:
        test_params = CUDAGraphLoRATestParams(
            batch_slot_ids=[0, 3, 3, 4, 5, 8],
            input_hidden_size=4096,
            slot_ranks=[8, 12, 4, 3] * 2,
            max_lora_rank=64,
            output_hidden_sizes=[4096, 4096],
            layer_module_mask=None,
            dtype=torch.bfloat16,
            seed=42,
        )

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
        b_ptrs = torch.randint(1,
                               1000000,
                               shape_2d,
                               dtype=CudaGraphLoraParams.PTR_DTYPE)
        b_prime_ptrs = torch.randint(1,
                                     1000000,
                                     shape_2d,
                                     dtype=CudaGraphLoraParams.PTR_DTYPE)

        b_ptrs *= test_params.layer_module_mask
        b_prime_ptrs *= test_params.layer_module_mask

        b_ptrs = b_ptrs.to(device="cuda")
        b_prime_ptrs = b_prime_ptrs.to(device="cuda")
        slot_ranks = torch.tensor(test_params.slot_ranks,
                                  dtype=CudaGraphLoraParams.SIZES_DTYPE,
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

        output_hidden_sizes = torch.tensor(
            test_params.output_hidden_sizes,
            dtype=CudaGraphLoraParams.SIZES_DTYPE,
            device="cuda")
        output_sizes_offset = CudaGraphLoraParams.get_offset_from_counts(
            output_hidden_sizes).to(dtype=CudaGraphLoraParams.PTR_DTYPE,
                                    device="cuda")

        layer = LoraLayer([0] * test_params.module_count,
                          test_params.output_hidden_sizes)
        inputs = GroupedGemmParamsInput(
            x=x,
            output_buffer=output_buffer,
            intermediate_buffer=intermediate_buffer,
            max_lora_size=test_params.slot_count,
            max_rank=test_params.max_lora_rank,
            slot_counts=slot_counts,
            slot_ranks=slot_ranks,
            slot_offsets_full=slot_offsets_full,
            sorted_ids=sorted_ids,
            b_ptrs=b_ptrs,
            b_prime_ptrs=b_prime_ptrs,
            output_hidden_sizes=output_hidden_sizes,
            output_sizes_offset=output_sizes_offset,
        )
        return inputs, layer


def compare_grouped_gemm_params(
    params: GroupedGemmParamsOutput,
    ref: GroupedGemmParamsOutput,
    params_input: GroupedGemmParamsInput,
    params_to_store_msg: List[str] | None = ['splitk_offsets'],
    params_exclude_msg: List[str] | None = None,
):
    assert not (params_to_store_msg and params_exclude_msg)

    bs, input_hidden_size = params.reordered_input.shape
    asserter = DelayedAssert()
    params_dict = asdict(params)
    ref_dict = asdict(ref)

    if not params_to_store_msg:
        params_to_store_msg = set(params_dict.keys())
    if params_exclude_msg:
        for name in params_exclude_msg:
            params_to_store_msg.discard(name)

    def get_msg(name: str, v: torch.Tensor, ref_v: torch.Tensor):
        is_get_msg = any(p in name or name in p for p in params_to_store_msg)
        header = f"\n\n{name=}\n"
        return f"{header} {v=}\n {ref_v=}\n diff:\n{v - ref_v}" if is_get_msg else header

    for name in params_dict.keys():
        v = params_dict[name]
        ref_v = ref_dict[name]
        if name not in ("reordered_input", "a_offset"):
            asserter.add(
                v.allclose(ref_v),
                get_msg(name, v, ref_v),
            )

    # Test a_offset separately
    offset = params.a_offset - params.reordered_input.data_ptr()
    ref_offset = ref.a_offset - ref.reordered_input.data_ptr()
    asserter.add(
        (offset == ref_offset).all(),
        # 'a_offset_fused',
        get_msg("a_offset", offset, ref_offset))

    # Test reordered_input separately
    valid_row = params_input.slot_offsets_full[-1].cpu().item()
    valid_rows = params.reordered_input[:valid_row]
    ref_valid_rows = ref.reordered_input[:valid_row]
    asserter.add(
        valid_rows.allclose(ref_valid_rows),
        get_msg(f"valid part({valid_row=}, {bs=}) of reordered_input",
                valid_rows, ref_valid_rows))

    # check intermediate buffer and output buffer are all zeros
    asserter.add(
        torch.all(params_input.intermediate_buffer == 0),
        get_msg("intermediate buffer", params_input.intermediate_buffer, 0))
    asserter.add(torch.all(params_input.output_buffer == 0),
                 get_msg("output buffer", params_input.output_buffer, 0))

    if valid_row < bs:
        invalid_rows = params.reordered_input[valid_row:]
        ref_invalid_rows = ref.reordered_input[valid_row:]
        asserter.add(
            torch.all(invalid_rows == 0),
            get_msg("invalid part of reordered_input", invalid_rows,
                    ref_invalid_rows))
    else:
        asserter.add(
            True,
            f"valid_row is full {valid_row=} v. bs: {params_dict['reordered_input'].shape[0]=}"
        )
    asserter.assert_all()


def compare_cuda_graph_lora_params_filler(test_params: CUDAGraphLoRATestParams):
    grouped_gemm_params_filler_input, layer = create_grouped_gemm_params_filler_input(
        test_params)
    output_fused = layer._prepare_grouped_gemm_buffers_fused(
        grouped_gemm_params_filler_input)

    assert torch.all(
        grouped_gemm_params_filler_input.intermediate_buffer == 0
    ), f"intermediate_buffer is not all zeros: {grouped_gemm_params_filler_input.intermediate_buffer}; non zero / zeros: {(grouped_gemm_params_filler_input.intermediate_buffer != 0).sum()} / {grouped_gemm_params_filler_input.intermediate_buffer.numel()}"
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

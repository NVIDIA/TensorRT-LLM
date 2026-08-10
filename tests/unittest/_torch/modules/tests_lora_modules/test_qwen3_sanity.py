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
"""Sanity tests for Qwen3 LoRA support (dense and MoE)."""

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file
from utils.llm_data import llm_models_root
from utils.util import skip_gpu_memory_less_than_80gb

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.lora_helper import LoraConfig

# HF module name -> block path relative to layers.{idx}.
# Attention targets work on all architectures. MLP targets only apply to
# dense layers (not MoE expert layers, which use w1/w2/w3 instead).
_ATTN_LORA_MODULES = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
}
_MLP_LORA_MODULES = {
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}

# Corresponding TRT-LLM module names
_ATTN_TRTLLM_MODULES = ["attn_q", "attn_k", "attn_v", "attn_dense"]
_MLP_TRTLLM_MODULES = ["mlp_h_to_4h", "mlp_gate", "mlp_4h_to_h"]


def _create_lora_adapter(
    output_dir, base_model_path, target_modules, lora_rank=8, dtype=torch.bfloat16
):
    """Create a dummy LoRA adapter for any decoder model."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(base_model_path, "config.json")) as f:
        cfg = json.load(f)

    hidden = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim", hidden // num_heads)
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    intermediate = cfg.get("intermediate_size", hidden * 4)
    num_layers = cfg["num_hidden_layers"]

    dim_map = {
        "q_proj": (hidden, q_dim),
        "k_proj": (hidden, kv_dim),
        "v_proj": (hidden, kv_dim),
        "o_proj": (q_dim, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }

    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "base_model_name_or_path": base_model_path,
                "bias": "none",
                "peft_type": "LORA",
                "r": lora_rank,
                "lora_alpha": 16,
                "target_modules": list(target_modules.keys()),
                "task_type": "CAUSAL_LM",
            },
            f,
        )

    weights = {}
    for layer_idx in range(num_layers):
        for module, block_path in target_modules.items():
            in_dim, out_dim = dim_map[module]
            key = f"base_model.model.model.layers.{layer_idx}.{block_path}.{module}"
            weights[f"{key}.lora_A.weight"] = (
                torch.randn(lora_rank, in_dim, dtype=torch.bfloat16) * 0.1
            ).to(dtype)
            weights[f"{key}.lora_B.weight"] = (
                torch.randn(out_dim, lora_rank, dtype=torch.bfloat16) * 0.1
            ).to(dtype)

    save_file(weights, os.path.join(output_dir, "adapter_model.safetensors"))
    return output_dir


def _run_with_and_without_lora(model_path, lora_config, lora_dir, prompts):
    """Run inference with and without LoRA, return (lora_outputs, base_outputs)."""
    with LLM(
        model=model_path,
        backend="pytorch",
        lora_config=lora_config,
        tensor_parallel_size=1,
        max_batch_size=4,
        max_num_tokens=256,
    ) as llm:
        sampling = SamplingParams(max_tokens=20, temperature=0.0, logprobs=0)
        lora_request = [LoRARequest("test-lora", 0, lora_dir)] * len(prompts)

        out_lora = llm.generate(prompts, sampling, lora_request=lora_request)
        out_base = llm.generate(prompts, sampling)

    return out_lora, out_base


def _assert_lora_changes_output(out_lora, out_base):
    """Assert that LoRA produces at least one different output (tokens or logprobs)."""
    any_differ = False
    for lora_out, base_out in zip(out_lora, out_base, strict=True):
        lora_ids = lora_out.outputs[0].token_ids
        base_ids = base_out.outputs[0].token_ids
        if lora_ids != base_ids:
            any_differ = True
            break

        # Even if tokens match, logprobs should differ
        lp_lora = lora_out.outputs[0].logprobs
        lp_base = base_out.outputs[0].logprobs
        if lp_lora and lp_base:
            for lp_w, lp_wo in zip(lp_lora, lp_base, strict=True):
                val_w = next(iter(lp_w.values())).logprob
                val_wo = next(iter(lp_wo.values())).logprob
                if abs(val_w - val_wo) > 1e-6:
                    any_differ = True
                    break

    assert any_differ, "LoRA outputs identical to base model (same tokens AND same logprobs)"


def _run_lora_test(model_path, target_modules, trtllm_modules, dtype=torch.bfloat16):
    """End-to-end helper: create adapter, run inference, assert output differs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lora_dir = _create_lora_adapter(
            os.path.join(tmpdir, "lora"), model_path, target_modules, dtype=dtype
        )
        lora_config = LoraConfig(
            lora_dir=[lora_dir],
            lora_target_modules=trtllm_modules,
            max_lora_rank=16,
            max_loras=2,
        )
        out_lora, out_base = _run_with_and_without_lora(
            model_path,
            lora_config,
            lora_dir,
            ["The capital of France is", "Hello, how are you"],
        )
        _assert_lora_changes_output(out_lora, out_base)


class TestQwen3LoRA:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = f"{llm_models_root()}/Qwen3/Qwen3-0.6B"
        if not os.path.exists(self.model_path):
            pytest.skip(f"Model not found: {self.model_path}")

    def test_qwen3_bf16_lora(self):
        _run_lora_test(
            self.model_path,
            {**_ATTN_LORA_MODULES, **_MLP_LORA_MODULES},
            _ATTN_TRTLLM_MODULES + _MLP_TRTLLM_MODULES,
        )

    def test_qwen3_fp8_lora(self):
        _run_lora_test(
            self.model_path,
            {**_ATTN_LORA_MODULES, **_MLP_LORA_MODULES},
            _ATTN_TRTLLM_MODULES + _MLP_TRTLLM_MODULES,
            dtype=torch.float8_e4m3fn,
        )


@skip_gpu_memory_less_than_80gb
class TestQwen3MoELoRA:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = f"{llm_models_root()}/Qwen3/Qwen3-30B-A3B"
        if not os.path.exists(self.model_path):
            pytest.skip(f"Model not found: {self.model_path}")

    def test_qwen3_moe_bf16_lora(self):
        _run_lora_test(
            self.model_path,
            _ATTN_LORA_MODULES,
            _ATTN_TRTLLM_MODULES,
        )

    def test_qwen3_moe_fp8_lora(self):
        _run_lora_test(
            self.model_path,
            _ATTN_LORA_MODULES,
            _ATTN_TRTLLM_MODULES,
            dtype=torch.float8_e4m3fn,
        )

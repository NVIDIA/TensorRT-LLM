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
"""Shared helpers for architecture-agnostic LoRA sanity tests.

These build a dummy LoRA adapter for any HF decoder checkpoint, run the model
with and without the adapter, and assert that the adapter actually changed the
output. That is enough to catch modules that silently drop `lora_params` or
never receive their `layer_idx`.
"""

import json
import os
import tempfile

import torch
from safetensors.torch import save_file

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi import RequestOutput
from tensorrt_llm.lora_helper import LoraConfig

# HF module name -> block path relative to layers.{idx}.
# Attention targets work on all architectures. MLP targets only apply to
# dense layers (not MoE expert layers, which use w1/w2/w3 instead).
ATTN_LORA_MODULES = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
}
MLP_LORA_MODULES = {
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}

# Corresponding TRT-LLM module names
ATTN_TRTLLM_MODULES = ["attn_q", "attn_k", "attn_v", "attn_dense"]
MLP_TRTLLM_MODULES = ["mlp_h_to_4h", "mlp_gate", "mlp_4h_to_h"]


def create_lora_adapter(
    output_dir: str,
    base_model_path: str,
    target_modules: dict[str, str],
    lora_rank: int = 8,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
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


def run_with_and_without_lora(
    model_path: str,
    lora_config: LoraConfig,
    lora_dir: str,
    prompts: list[str],
) -> tuple[list[RequestOutput], list[RequestOutput]]:
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


def assert_lora_changes_output(
    out_lora: list[RequestOutput], out_base: list[RequestOutput]
) -> None:
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


def run_lora_test(
    model_path: str,
    target_modules: dict[str, str],
    trtllm_modules: list[str],
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """End-to-end helper: create adapter, run inference, assert output differs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lora_dir = create_lora_adapter(
            os.path.join(tmpdir, "lora"), model_path, target_modules, dtype=dtype
        )
        lora_config = LoraConfig(
            lora_dir=[lora_dir],
            lora_target_modules=trtllm_modules,
            max_lora_rank=16,
            max_loras=2,
        )
        out_lora, out_base = run_with_and_without_lora(
            model_path,
            lora_config,
            lora_dir,
            ["The capital of France is", "Hello, how are you"],
        )
        assert_lora_changes_output(out_lora, out_base)

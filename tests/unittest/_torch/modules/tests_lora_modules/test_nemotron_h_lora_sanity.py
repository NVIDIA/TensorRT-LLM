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
"""Sanity tests for Nemotron-H LoRA support."""

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file
from utils.llm_data import llm_models_root
from utils.util import skip_gpu_memory_less_than_80gb, skip_pre_hopper

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.lora_helper import LoraConfig, get_default_trtllm_modules_to_hf_modules


def _create_dummy_lora_adapter(output_dir: str, base_model_path: str, lora_rank: int = 8) -> str:
    """Create a dummy LoRA adapter targeting attention layers."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(base_model_path, "config.json")) as f:
        cfg = json.load(f)

    hidden = cfg["hidden_size"]
    q_dim = cfg.get("num_attention_heads", 32) * cfg.get("head_dim", 128)
    kv_dim = cfg.get("num_key_value_heads", 2) * cfg.get("head_dim", 128)
    pattern = cfg.get("hybrid_override_pattern", "")
    attn_layers = [i for i, c in enumerate(pattern) if c == "*"]

    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "base_model_name_or_path": base_model_path,
                "bias": "none",
                "peft_type": "LORA",
                "r": lora_rank,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "task_type": "CAUSAL_LM",
            },
            f,
        )

    # Projection dims: q(hidden->q_dim), k/v(hidden->kv_dim), o(q_dim->hidden)
    proj_dims = {
        "q_proj": (hidden, q_dim),
        "k_proj": (hidden, kv_dim),
        "v_proj": (hidden, kv_dim),
        "o_proj": (q_dim, hidden),
    }
    weights = {}
    for layer_idx in attn_layers:
        for proj, (in_dim, out_dim) in proj_dims.items():
            key = f"base_model.model.backbone.layers.{layer_idx}.mixer.{proj}"
            weights[f"{key}.lora_A.weight"] = (
                torch.randn(lora_rank, in_dim, dtype=torch.bfloat16) * 0.01
            )
            weights[f"{key}.lora_B.weight"] = torch.zeros(out_dim, lora_rank, dtype=torch.bfloat16)

    save_file(weights, os.path.join(output_dir, "adapter_model.safetensors"))
    return output_dir


@skip_pre_hopper
@skip_gpu_memory_less_than_80gb
class TestNemotronHLoRASanity:
    """Sanity test for Nemotron-H LoRA support."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = f"{llm_models_root()}/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    def test_load_and_generate_with_lora(self):
        """Test loading and running inference with a dummy LoRA adapter."""
        if not os.path.exists(self.model_path):
            pytest.skip(f"Model not found: {self.model_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            lora_dir = _create_dummy_lora_adapter(os.path.join(tmpdir, "lora"), self.model_path)

            with LLM(
                model=self.model_path,
                lora_config=LoraConfig(lora_dir=[lora_dir], max_lora_rank=16, max_loras=2),
                tensor_parallel_size=1,
                max_batch_size=1,
                max_num_tokens=128,
            ) as llm:
                outputs = llm.generate(
                    ["Hello"],
                    sampling_params=SamplingParams(max_tokens=5),
                    lora_request=[LoRARequest("test-lora", 0, lora_dir)],
                )
                assert len(outputs) == 1
                assert len(outputs[0].outputs[0].token_ids) > 0


class TestNemotronHLoRAModuleMappings:
    """Test LoRA module name mappings for Nemotron-H."""

    def test_nemotron_h_modules_have_mappings(self):
        """Verify Nemotron-H specific modules have correct HF mappings."""
        mapping = get_default_trtllm_modules_to_hf_modules()
        expected = {
            "mamba_in_proj": "in_proj",
            "mamba_out_proj": "out_proj",
            "moe_latent_up": "fc1_latent_proj",
            "moe_latent_down": "fc2_latent_proj",
        }
        for trtllm_name, hf_name in expected.items():
            assert mapping.get(trtllm_name) == hf_name, f"{trtllm_name} should map to {hf_name}"

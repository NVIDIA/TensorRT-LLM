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
from utils.util import skip_gpu_memory_less_than_80gb, skip_num_gpus_less_than, skip_pre_blackwell

import tensorrt_llm.bindings as _tb
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.executor.request import LoRARequest


def _create_lora_adapter(output_dir, base_model_path, lora_rank=8):
    """Create a dummy LoRA adapter targeting attention, Mamba, shared expert,
    and MoE latent projection layers with small non-zero weights."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(base_model_path, "config.json")) as f:
        cfg = json.load(f)

    hidden = cfg["hidden_size"]
    q_dim = cfg["num_attention_heads"] * cfg["head_dim"]
    kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]
    pattern = cfg["hybrid_override_pattern"]
    shared_intermediate = cfg["moe_shared_expert_intermediate_size"]
    latent = cfg["moe_latent_size"]
    d_inner = cfg["mamba_head_dim"] * cfg["mamba_num_heads"]
    d_in_proj = 2 * d_inner + 2 * cfg["n_groups"] * cfg["ssm_state_size"] + cfg["mamba_num_heads"]

    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "base_model_name_or_path": base_model_path,
                "bias": "none",
                "peft_type": "LORA",
                "r": lora_rank,
                "lora_alpha": 16,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "in_proj",
                    "out_proj",
                    "shared_experts.up_proj",
                    "shared_experts.down_proj",
                    "fc1_latent_proj",
                    "fc2_latent_proj",
                ],
                "task_type": "CAUSAL_LM",
            },
            f,
        )

    weights = {}

    def _add(key, in_dim, out_dim):
        weights[f"{key}.lora_A.weight"] = (
            torch.randn(lora_rank, in_dim, dtype=torch.bfloat16) * 0.01
        )
        weights[f"{key}.lora_B.weight"] = (
            torch.randn(out_dim, lora_rank, dtype=torch.bfloat16) * 0.001
        )

    def _layer(idx, suffix):
        return f"base_model.model.backbone.layers.{idx}.{suffix}"

    attn_dims = {
        "q_proj": (hidden, q_dim),
        "k_proj": (hidden, kv_dim),
        "v_proj": (hidden, kv_dim),
        "o_proj": (q_dim, hidden),
    }
    mamba_dims = {"in_proj": (hidden, d_in_proj), "out_proj": (d_inner, hidden)}

    for idx, kind in enumerate(pattern):
        if kind == "*":
            for proj, (i, o) in attn_dims.items():
                _add(_layer(idx, f"mixer.{proj}"), i, o)
        elif kind == "M":
            for proj, (i, o) in mamba_dims.items():
                _add(_layer(idx, f"mixer.{proj}"), i, o)
        elif kind == "E":
            _add(_layer(idx, "mlp.shared_experts.up_proj"), hidden, shared_intermediate)
            _add(_layer(idx, "mlp.shared_experts.down_proj"), shared_intermediate, hidden)
            _add(_layer(idx, "mlp.fc1_latent_proj"), hidden, latent)
            _add(_layer(idx, "mlp.fc2_latent_proj"), latent, hidden)

    save_file(weights, os.path.join(output_dir, "adapter_model.safetensors"))
    return output_dir


def _get_lora_config(lora_dir):
    """Get model-specific LoRA config via NemotronHForCausalLM.lora_config()."""
    from tensorrt_llm._torch.models.modeling_nemotron_h import NemotronHForCausalLM

    config = NemotronHForCausalLM.lora_config("")
    config.lora_dir = [lora_dir]
    config.max_lora_rank = 16
    config.max_loras = 2
    return config


@skip_pre_blackwell
@skip_gpu_memory_less_than_80gb
class TestNemotronHLoRA:
    """E2E LoRA tests on the Super-V3 120B NVFP4 model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = f"{llm_models_root()}/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
        if not os.path.exists(self.model_path):
            pytest.skip(f"Model not found: {self.model_path}")

    def _run_generate(self, llm, lora_dir, prompts):
        sampling = SamplingParams(max_tokens=5, temperature=0.0)
        lora_req = [LoRARequest("test-lora", 0, lora_dir)] * len(prompts)
        return llm.generate(prompts, sampling, lora_request=lora_req)

    def test_lora_pp1_sanity(self):
        """LoRA inference with pp_size=1 produces tokens for base and LoRA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_dir = _create_lora_adapter(os.path.join(tmpdir, "lora"), self.model_path)
            with LLM(
                model=self.model_path,
                lora_config=_get_lora_config(lora_dir),
                tensor_parallel_size=1,
                max_batch_size=2,
                max_num_tokens=256,
            ) as llm:
                prompts = ["Hello", "The capital of France is"]
                lora = self._run_generate(llm, lora_dir, prompts)
                assert all(len(o.outputs[0].token_ids) > 0 for o in lora)

    @skip_num_gpus_less_than(2)
    def test_lora_pp2_sanity(self):
        """LoRA inference with pp_size=2 produces tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_dir = _create_lora_adapter(os.path.join(tmpdir, "lora"), self.model_path)
            with LLM(
                model=self.model_path,
                lora_config=_get_lora_config(lora_dir),
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
                max_batch_size=2,
                max_num_tokens=256,
            ) as llm:
                lora = self._run_generate(llm, lora_dir, ["Hello", "The capital of France is"])
                assert all(len(o.outputs[0].token_ids) > 0 for o in lora)


def _make_model_config(n_lora_layers):
    """Create a C++ ModelConfig with the given number of LoRA layers."""
    cfg = _tb.ModelConfig(
        vocab_size=32000,
        num_layers=n_lora_layers,
        num_attention_layers=n_lora_layers,
        num_rnn_layers=0,
        num_heads=32,
        hidden_size=4096,
        data_type=_tb.DataType.HALF,
    )
    cfg.set_num_lora_layers(n_lora_layers)
    return cfg


@pytest.mark.parametrize(
    "n_layers,pp_size",
    [
        (88, 4),  # 22 each
        (7, 2),  # 4 + 3
        (7, 3),  # 3 + 2 + 2
        (1, 2),  # Edge: 1 + 0
        (32, 1),  # Single rank
    ],
)
def test_lora_layer_distribution_no_overlap(n_layers, pp_size):
    """C++ getFirstLoraLayer()/getNbLoraLayers(): every layer assigned to exactly one PP rank."""
    cfg = _make_model_config(n_layers)
    assigned = set()
    prev_end = 0
    for rank in range(pp_size):
        first = cfg.first_lora_layer(pp_size, rank)
        count = cfg.num_lora_layers(pp_size, rank)
        assert first == prev_end, f"Rank {rank}: first_layer={first}, expected {prev_end}"
        assert count >= 0, f"Rank {rank}: negative count {count}"
        layers = set(range(first, first + count))
        assert not assigned & layers, (
            f"Rank {rank} overlaps with previous ranks on {assigned & layers}"
        )
        assigned |= layers
        prev_end = first + count
    assert assigned == set(range(n_layers)), f"Missing layers: {set(range(n_layers)) - assigned}"


def test_lora_fallback_with_empty_layer_types_and_pp():
    """Tests for correct layer distribution when no layer types are set."""
    num_layers = 32
    pp_size = 4
    layers_per_rank = num_layers // pp_size

    # No set_num_lora_layers → mNbLoraLayers=0, no layer_types → mLayerTypes empty
    cfg = _tb.ModelConfig(
        vocab_size=32000,
        num_layers=num_layers,
        num_attention_layers=num_layers,
        num_rnn_layers=0,
        num_heads=32,
        hidden_size=4096,
        data_type=_tb.DataType.HALF,
    )

    for rank in range(pp_size):
        assert cfg.first_lora_layer(pp_size, rank) == rank * layers_per_rank
        assert cfg.num_lora_layers(pp_size, rank) == layers_per_rank


def test_moe_latent_lora_modules_use_correct_dimensions():
    """Test for correct dimensions of MoE latent projection layers."""
    hidden, mlp_hidden, moe_latent, tp = 6144, 3072, 1024, 2
    up, down = _tb.LoraModule.create_lora_modules(
        lora_module_names=["moe_latent_up", "moe_latent_down"],
        hidden_size=hidden,
        mlp_hidden_size=mlp_hidden,
        num_attention_heads=32,
        num_kv_attention_heads=8,
        attention_head_size=128,
        tp_size=tp,
        moe_latent_size=moe_latent,
    )
    global_hidden = hidden * tp

    assert (up.in_dim, up.out_dim) == (global_hidden, moe_latent)
    assert (down.in_dim, down.out_dim) == (moe_latent, global_hidden)
    for m in (up, down):
        assert m.in_tp_split_dim == -1
        assert m.out_tp_split_dim == -1

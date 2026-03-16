# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Module-level test: AttentionBlock (TRT-LLM) vs GptOssAttention (HuggingFace)

This test verifies that the TRT-LLM AttentionBlock for GPT-OSS produces
numerically equivalent outputs to the HuggingFace GptOssAttention module
when loaded with the same weights from the checkpoint.
"""

import sys
import os

import pytest
import torch
import torch.nn.functional as F

# --- Fixtures and helpers ---

CHECKPOINT_PATH = "/scratch.trt_llm_data/llm-models/gpt_oss/gpt-oss-20b/"
HF_SOURCE_PATH = "/home/scratch.huig_gpu/modular_gpt_oss.py"
LAYER_IDX = 0  # Test with layer 0 (sliding_attention)


def _load_hf_config():
    """Load the HuggingFace GptOssConfig from the checkpoint."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(CHECKPOINT_PATH)
    return config


def _load_checkpoint_weights(layer_idx: int):
    """Load attention weights for a specific layer from the checkpoint."""
    from safetensors.torch import load_file
    import json

    index_path = os.path.join(CHECKPOINT_PATH, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    prefix = f"model.layers.{layer_idx}.self_attn."
    needed_keys = [
        f"{prefix}q_proj.weight", f"{prefix}q_proj.bias",
        f"{prefix}k_proj.weight", f"{prefix}k_proj.bias",
        f"{prefix}v_proj.weight", f"{prefix}v_proj.bias",
        f"{prefix}o_proj.weight", f"{prefix}o_proj.bias",
        f"{prefix}sinks",
    ]

    # Find which shard files we need
    shard_files = set()
    for key in needed_keys:
        if key in weight_map:
            shard_files.add(weight_map[key])

    weights = {}
    for shard_file in shard_files:
        shard_path = os.path.join(CHECKPOINT_PATH, shard_file)
        shard_weights = load_file(shard_path)
        for key in needed_keys:
            if key in shard_weights:
                weights[key] = shard_weights[key]

    return weights


def _create_hf_attention(config, layer_idx: int):
    """Create and return a HuggingFace GptOssAttention module."""
    # Import from the modular source
    import importlib.util
    spec = importlib.util.spec_from_file_location("modular_gpt_oss", HF_SOURCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    attn = mod.GptOssAttention(config, layer_idx=layer_idx)
    return attn, mod


def _load_hf_attention_weights(attn, weights, layer_idx: int):
    """Load checkpoint weights into HF attention module."""
    prefix = f"model.layers.{layer_idx}.self_attn."
    attn.q_proj.weight.data.copy_(weights[f"{prefix}q_proj.weight"])
    attn.q_proj.bias.data.copy_(weights[f"{prefix}q_proj.bias"])
    attn.k_proj.weight.data.copy_(weights[f"{prefix}k_proj.weight"])
    attn.k_proj.bias.data.copy_(weights[f"{prefix}k_proj.bias"])
    attn.v_proj.weight.data.copy_(weights[f"{prefix}v_proj.weight"])
    attn.v_proj.bias.data.copy_(weights[f"{prefix}v_proj.bias"])
    attn.o_proj.weight.data.copy_(weights[f"{prefix}o_proj.weight"])
    attn.o_proj.bias.data.copy_(weights[f"{prefix}o_proj.bias"])
    attn.sinks.data.copy_(weights[f"{prefix}sinks"])


def _create_trtllm_attention(config, layer_idx: int):
    """Create TRT-LLM AttentionBlock with matching config."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

    # Create a minimal ModelConfig wrapping the HF config
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, rank=0)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        attn_backend='TRTLLM',
    )

    from tensorrt_llm._torch.models.modeling_gpt_oss import AttentionBlock
    trt_attn = AttentionBlock(model_config, layer_idx=layer_idx)
    return trt_attn


def _load_trtllm_attention_weights(trt_attn, weights, layer_idx: int):
    """Load checkpoint weights into TRT-LLM AttentionBlock."""
    prefix = f"model.layers.{layer_idx}.self_attn."

    # Load QKV weights (fused)
    q_weights = {"weight": weights[f"{prefix}q_proj.weight"],
                 "bias": weights[f"{prefix}q_proj.bias"]}
    k_weights = {"weight": weights[f"{prefix}k_proj.weight"],
                 "bias": weights[f"{prefix}k_proj.bias"]}
    v_weights = {"weight": weights[f"{prefix}v_proj.weight"],
                 "bias": weights[f"{prefix}v_proj.bias"]}

    # The qkv_proj load_weights expects [q_dict, k_dict, v_dict]
    trt_attn.qkv_proj.load_weights(weights=[q_weights, k_weights, v_weights])

    # Load output projection
    o_weights = {"weight": weights[f"{prefix}o_proj.weight"],
                 "bias": weights[f"{prefix}o_proj.bias"]}
    trt_attn.o_proj.load_weights(weights=[o_weights])

    # Load sinks
    sinks_weights = {"sinks": weights[f"{prefix}sinks"]}
    trt_attn.load_weights(weights=[sinks_weights])


# --- Test Cases ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not os.path.exists(CHECKPOINT_PATH), reason="Checkpoint not found")
class TestAttentionBlockConsistency:
    """Test that TRT-LLM AttentionBlock matches HF GptOssAttention output."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.config = _load_hf_config()
        self.weights = _load_checkpoint_weights(LAYER_IDX)
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16

    def test_weight_shapes(self):
        """Verify weight shapes match expected dimensions."""
        prefix = f"model.layers.{LAYER_IDX}.self_attn."

        # Q: [num_heads * head_dim, hidden_size] = [4096, 2880]
        assert self.weights[f"{prefix}q_proj.weight"].shape == (4096, 2880), \
            f"q_proj.weight shape mismatch: {self.weights[f'{prefix}q_proj.weight'].shape}"

        # K: [num_kv_heads * head_dim, hidden_size] = [512, 2880]
        assert self.weights[f"{prefix}k_proj.weight"].shape == (512, 2880), \
            f"k_proj.weight shape mismatch: {self.weights[f'{prefix}k_proj.weight'].shape}"

        # V: [num_kv_heads * head_dim, hidden_size] = [512, 2880]
        assert self.weights[f"{prefix}v_proj.weight"].shape == (512, 2880), \
            f"v_proj.weight shape mismatch: {self.weights[f'{prefix}v_proj.weight'].shape}"

        # O: [hidden_size, num_heads * head_dim] = [2880, 4096]
        assert self.weights[f"{prefix}o_proj.weight"].shape == (2880, 4096), \
            f"o_proj.weight shape mismatch: {self.weights[f'{prefix}o_proj.weight'].shape}"

        # Sinks: [num_attention_heads] = [64]
        assert self.weights[f"{prefix}sinks"].shape == (64,), \
            f"sinks shape mismatch: {self.weights[f'{prefix}sinks'].shape}"

    def test_sliding_window_config(self):
        """Verify sliding window configuration for layer 0 (should be sliding)."""
        trt_attn = _create_trtllm_attention(self.config, layer_idx=0)
        assert trt_attn.sliding_window == 128, \
            f"Layer 0 should have sliding_window=128, got {trt_attn.sliding_window}"

        trt_attn_full = _create_trtllm_attention(self.config, layer_idx=1)
        assert trt_attn_full.sliding_window is None, \
            f"Layer 1 should have sliding_window=None, got {trt_attn_full.sliding_window}"

    def test_rope_params(self):
        """Verify RoPE parameters are correctly configured."""
        from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType

        trt_attn = _create_trtllm_attention(self.config, layer_idx=0)
        pos_params = trt_attn.pos_embd_params

        assert pos_params.type == PositionEmbeddingType.yarn, \
            f"Expected YaRN position embedding, got {pos_params.type}"
        assert pos_params.is_neox == False, \
            f"Expected is_neox=False, got {pos_params.is_neox}"
        assert pos_params.rope.dim == 64, \
            f"Expected rope dim=64 (head_dim), got {pos_params.rope.dim}"
        assert pos_params.rope.theta == 150000, \
            f"Expected theta=150000, got {pos_params.rope.theta}"
        assert pos_params.rope.scale == 32.0, \
            f"Expected scale (factor)=32.0, got {pos_params.rope.scale}"
        assert pos_params.rope.beta_fast == 32.0, \
            f"Expected beta_fast=32.0, got {pos_params.rope.beta_fast}"
        assert pos_params.rope.beta_slow == 1.0, \
            f"Expected beta_slow=1.0, got {pos_params.rope.beta_slow}"
        assert pos_params.rope.duplicate_data == False, \
            f"Expected duplicate_data=False, got {pos_params.rope.duplicate_data}"

    def test_attention_dimensions(self):
        """Verify attention dimensions handle hidden_size != num_heads * head_dim."""
        trt_attn = _create_trtllm_attention(self.config, layer_idx=0)

        assert trt_attn.hidden_size == 2880, \
            f"Expected hidden_size=2880, got {trt_attn.hidden_size}"
        assert trt_attn.num_heads == 64, \
            f"Expected num_heads=64, got {trt_attn.num_heads}"
        assert trt_attn.head_dim == 64, \
            f"Expected head_dim=64, got {trt_attn.head_dim}"
        assert trt_attn.num_key_value_heads == 8, \
            f"Expected num_kv_heads=8, got {trt_attn.num_key_value_heads}"

    def test_sinks_parameter(self):
        """Verify sinks parameter shape and loading."""
        trt_attn = _create_trtllm_attention(self.config, layer_idx=0)
        trt_attn = trt_attn.to(self.device)

        # Sinks shape should be [num_heads / tp_size] = [64] for tp_size=1
        assert trt_attn.sinks.shape == (64,), \
            f"Expected sinks shape (64,), got {trt_attn.sinks.shape}"

        # Load and verify
        _load_trtllm_attention_weights(trt_attn, self.weights, LAYER_IDX)
        expected_sinks = self.weights[f"model.layers.{LAYER_IDX}.self_attn.sinks"].to(torch.float32).to(self.device)
        assert torch.allclose(trt_attn.sinks.data, expected_sinks), \
            f"Sinks data mismatch after loading. Max diff: {(trt_attn.sinks.data - expected_sinks).abs().max()}"

    def test_bias_configuration(self):
        """Verify bias=True on all projections (Q, K, V, O)."""
        trt_attn = _create_trtllm_attention(self.config, layer_idx=0)

        # QKV projection should have bias
        assert trt_attn.qkv_proj.bias is not None or \
               (hasattr(trt_attn.qkv_proj, 'has_bias') and trt_attn.qkv_proj.has_bias), \
            "QKV projection should have bias=True"

        # Output projection should have bias
        assert trt_attn.o_proj.bias is not None or \
               (hasattr(trt_attn.o_proj, 'has_bias') and trt_attn.o_proj.has_bias), \
            "Output projection should have bias=True (dense_bias=True)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=long"])

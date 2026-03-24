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
Module-level tests for GptOssAttention.
Verifies consistency between HuggingFace GptOssAttention and TRT-LLM GptOssAttention.

Tests:
1. Weight loading: QKV fusion from separate q/k/v_proj into fused qkv_proj
2. Sinks parameter loading and dtype
3. RoPE configuration (YaRN with NeoX style)
4. Sliding window configuration per layer
5. QKV projection numerical equivalence (pre-attention)
"""

import json
import sys
import os

import pytest
import torch

# ---- Checkpoint and config paths ----
CHECKPOINT_PATH = "/scratch.trt_llm_data/llm-models/gpt_oss/gpt-oss-20b/"
HF_MODELING_PATH = "/home/scratch.huig_gpu/modeling_gpt_oss.py"

# We need to load the HF modeling code from a standalone file
# Add the parent dir so we can import configuration_gpt_oss
sys.path.insert(0, os.path.dirname(HF_MODELING_PATH))


def load_config():
    """Load the model config.json."""
    with open(os.path.join(CHECKPOINT_PATH, "config.json")) as f:
        return json.load(f)


def load_checkpoint_weights(layer_idx=0):
    """Load attention weights for a specific layer from the checkpoint."""
    from safetensors import safe_open

    index_file = os.path.join(CHECKPOINT_PATH, "model.safetensors.index.json")
    with open(index_file) as f:
        index = json.load(f)

    prefix = f"model.layers.{layer_idx}.self_attn"
    weight_names = [
        k for k in index["weight_map"].keys() if k.startswith(prefix)
    ]

    weights = {}
    loaded_files = {}
    for wn in weight_names:
        shard_file = index["weight_map"][wn]
        if shard_file not in loaded_files:
            loaded_files[shard_file] = safe_open(
                os.path.join(CHECKPOINT_PATH, shard_file),
                framework="pt",
                device="cpu",
            )
        weights[wn] = loaded_files[shard_file].get_tensor(wn)

    return weights


class TestGptOssAttentionConsistency:
    """Test weight loading and configuration consistency for GptOssAttention."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config_dict = load_config()
        self.layer_idx = 0
        self.weights = load_checkpoint_weights(self.layer_idx)

    def test_qkv_fusion_shapes(self):
        """Verify QKV weights can be fused correctly."""
        prefix = f"model.layers.{self.layer_idx}.self_attn"
        q_weight = self.weights[f"{prefix}.q_proj.weight"]
        k_weight = self.weights[f"{prefix}.k_proj.weight"]
        v_weight = self.weights[f"{prefix}.v_proj.weight"]

        q_bias = self.weights[f"{prefix}.q_proj.bias"]
        k_bias = self.weights[f"{prefix}.k_proj.bias"]
        v_bias = self.weights[f"{prefix}.v_proj.bias"]

        # Verify individual shapes match config
        num_heads = self.config_dict["num_attention_heads"]  # 64
        num_kv_heads = self.config_dict["num_key_value_heads"]  # 8
        head_dim = self.config_dict["head_dim"]  # 64
        hidden_size = self.config_dict["hidden_size"]  # 2880

        assert q_weight.shape == (num_heads * head_dim, hidden_size), \
            f"q_proj weight shape mismatch: {q_weight.shape} vs expected ({num_heads * head_dim}, {hidden_size})"
        assert k_weight.shape == (num_kv_heads * head_dim, hidden_size), \
            f"k_proj weight shape mismatch: {k_weight.shape}"
        assert v_weight.shape == (num_kv_heads * head_dim, hidden_size), \
            f"v_proj weight shape mismatch: {v_weight.shape}"

        # Verify fused QKV shape
        fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        expected_fused_size = (num_heads + 2 * num_kv_heads) * head_dim  # 4096 + 512 + 512 = 5120
        assert fused_weight.shape == (expected_fused_size, hidden_size), \
            f"Fused QKV weight shape: {fused_weight.shape} vs expected ({expected_fused_size}, {hidden_size})"

        # Verify fused bias shape
        fused_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        assert fused_bias.shape == (expected_fused_size,), \
            f"Fused QKV bias shape: {fused_bias.shape} vs expected ({expected_fused_size},)"

    def test_sinks_parameter(self):
        """Verify sinks parameter shape and existence."""
        prefix = f"model.layers.{self.layer_idx}.self_attn"
        sinks = self.weights[f"{prefix}.sinks"]

        num_heads = self.config_dict["num_attention_heads"]
        assert sinks.shape == (num_heads,), \
            f"Sinks shape mismatch: {sinks.shape} vs expected ({num_heads},)"
        # Checkpoint stores sinks as BF16
        assert sinks.dtype == torch.bfloat16, \
            f"Sinks dtype in checkpoint: {sinks.dtype}, expected bfloat16"

    def test_sliding_window_per_layer(self):
        """Verify layer_types configuration for sliding vs full attention."""
        layer_types = self.config_dict["layer_types"]
        sliding_window = self.config_dict["sliding_window"]

        assert len(layer_types) == self.config_dict["num_hidden_layers"], \
            f"layer_types length {len(layer_types)} != num_hidden_layers {self.config_dict['num_hidden_layers']}"

        # Even layers should be sliding, odd should be full
        for i, lt in enumerate(layer_types):
            if i % 2 == 0:
                assert lt == "sliding_attention", \
                    f"Layer {i} expected sliding_attention, got {lt}"
            else:
                assert lt == "full_attention", \
                    f"Layer {i} expected full_attention, got {lt}"

        assert sliding_window == 128, \
            f"sliding_window expected 128, got {sliding_window}"

    def test_rope_config(self):
        """Verify YaRN RoPE configuration."""
        rope_scaling = self.config_dict["rope_scaling"]
        rope_theta = self.config_dict["rope_theta"]

        assert rope_theta == 150000, f"rope_theta expected 150000, got {rope_theta}"
        assert rope_scaling["rope_type"] == "yarn", \
            f"rope_type expected 'yarn', got '{rope_scaling['rope_type']}'"
        assert rope_scaling["factor"] == 32.0, \
            f"factor expected 32.0, got {rope_scaling['factor']}"
        assert rope_scaling["original_max_position_embeddings"] == 4096, \
            f"original_max_position_embeddings expected 4096, got {rope_scaling['original_max_position_embeddings']}"
        assert rope_scaling["beta_fast"] == 32.0, \
            f"beta_fast expected 32.0, got {rope_scaling['beta_fast']}"
        assert rope_scaling["beta_slow"] == 1.0, \
            f"beta_slow expected 1.0, got {rope_scaling['beta_slow']}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGptOssAttentionTrtllmInit:
    """Test TRT-LLM GptOssAttention instantiation and weight loading."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config_dict = load_config()
        self.layer_idx = 0
        self.weights = load_checkpoint_weights(self.layer_idx)

    def _create_trtllm_attention(self):
        """Create a TRT-LLM GptOssAttention module."""
        from transformers import PretrainedConfig as HfPretrainedConfig

        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.models.modeling_gpt_oss import GptOssAttention
        from tensorrt_llm.mapping import Mapping

        # Create a pretrained config from the checkpoint config
        pretrained_config = HfPretrainedConfig.from_pretrained(CHECKPOINT_PATH)

        mapping = Mapping(world_size=1, tp_size=1, pp_size=1, rank=0)
        model_config = ModelConfig(pretrained_config=pretrained_config,
                                   mapping=mapping)

        attn = GptOssAttention(
            model_config=model_config,
            layer_idx=self.layer_idx,
        )
        return attn

    def test_trtllm_attention_init(self):
        """Test that TRT-LLM GptOssAttention can be instantiated."""
        attn = self._create_trtllm_attention()

        # Verify basic properties
        assert attn.num_heads == self.config_dict["num_attention_heads"]
        assert attn.num_key_value_heads == self.config_dict[
            "num_key_value_heads"]
        assert attn.head_dim == self.config_dict["head_dim"]
        assert hasattr(attn, "sinks")
        assert attn.sinks.shape == (self.config_dict["num_attention_heads"],)
        assert attn.sinks.dtype == torch.float32

        # Verify sliding window config for layer 0 (should be sliding)
        assert attn._window_size == self.config_dict["sliding_window"]

    def test_trtllm_attention_init_full_layer(self):
        """Test that TRT-LLM GptOssAttention for full-attention layer (layer 1)."""
        from transformers import PretrainedConfig as HfPretrainedConfig

        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.models.modeling_gpt_oss import GptOssAttention
        from tensorrt_llm.mapping import Mapping

        pretrained_config = HfPretrainedConfig.from_pretrained(CHECKPOINT_PATH)
        mapping = Mapping(world_size=1, tp_size=1, pp_size=1, rank=0)
        model_config = ModelConfig(pretrained_config=pretrained_config,
                                   mapping=mapping)

        attn = GptOssAttention(
            model_config=model_config,
            layer_idx=1,  # odd layer = full attention
        )
        # Verify no sliding window for layer 1
        assert attn._window_size is None, \
            f"Layer 1 should have no sliding window, got {attn._window_size}"

    def test_weight_loading_qkv_fusion(self):
        """Test that QKV weights are loaded and fused correctly into the TRT-LLM module."""
        attn = self._create_trtllm_attention()
        attn = attn.cuda()

        prefix = f"model.layers.{self.layer_idx}.self_attn"

        # Build fused QKV weight manually
        q_weight = self.weights[f"{prefix}.q_proj.weight"]
        k_weight = self.weights[f"{prefix}.k_proj.weight"]
        v_weight = self.weights[f"{prefix}.v_proj.weight"]
        fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        q_bias = self.weights[f"{prefix}.q_proj.bias"]
        k_bias = self.weights[f"{prefix}.k_proj.bias"]
        v_bias = self.weights[f"{prefix}.v_proj.bias"]
        fused_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

        # Load into TRT-LLM qkv_proj via the Linear module's load_weights
        attn.qkv_proj.load_weights(weights=[
            {"weight": q_weight, "bias": q_bias},
            {"weight": k_weight, "bias": k_bias},
            {"weight": v_weight, "bias": v_bias},
        ])

        # Verify the fused weight matches
        loaded_weight = attn.qkv_proj.weight.data.cpu().to(torch.bfloat16)
        assert torch.allclose(loaded_weight, fused_weight), \
            f"Fused QKV weight mismatch. Max diff: {(loaded_weight - fused_weight).abs().max()}"

        loaded_bias = attn.qkv_proj.bias.data.cpu().to(torch.bfloat16)
        assert torch.allclose(loaded_bias, fused_bias), \
            f"Fused QKV bias mismatch. Max diff: {(loaded_bias - fused_bias).abs().max()}"

    def test_weight_loading_dense(self):
        """Test that o_proj weight is loaded correctly into the TRT-LLM dense layer."""
        attn = self._create_trtllm_attention()
        attn = attn.cuda()

        prefix = f"model.layers.{self.layer_idx}.self_attn"
        o_weight = self.weights[f"{prefix}.o_proj.weight"]
        o_bias = self.weights[f"{prefix}.o_proj.bias"]

        # Load dense weights
        attn.o_proj.load_weights(weights={"weight": o_weight, "bias": o_bias})

        loaded_weight = attn.o_proj.weight.data.cpu().to(torch.bfloat16)
        assert torch.allclose(loaded_weight, o_weight), \
            f"Dense weight mismatch. Max diff: {(loaded_weight - o_weight).abs().max()}"

        loaded_bias = attn.o_proj.bias.data.cpu().to(torch.bfloat16)
        assert torch.allclose(loaded_bias, o_bias), \
            f"Dense bias mismatch. Max diff: {(loaded_bias - o_bias).abs().max()}"

    def test_sinks_loading(self):
        """Test that sinks parameter can be loaded from checkpoint."""
        attn = self._create_trtllm_attention()
        attn = attn.cuda()

        prefix = f"model.layers.{self.layer_idx}.self_attn"
        sinks_ckpt = self.weights[f"{prefix}.sinks"]

        # Load sinks directly (it's a plain nn.Parameter)
        attn.sinks.data.copy_(sinks_ckpt.float())

        # Verify values match (accounting for bf16 -> fp32 conversion)
        loaded_sinks = attn.sinks.data.cpu()
        expected_sinks = sinks_ckpt.float()
        assert torch.allclose(loaded_sinks, expected_sinks), \
            f"Sinks mismatch. Max diff: {(loaded_sinks - expected_sinks).abs().max()}"

    def test_qkv_projection_numerical(self):
        """Test that the fused QKV projection produces the same result as separate projections."""
        attn = self._create_trtllm_attention()
        attn = attn.cuda()

        prefix = f"model.layers.{self.layer_idx}.self_attn"

        q_weight = self.weights[f"{prefix}.q_proj.weight"]
        k_weight = self.weights[f"{prefix}.k_proj.weight"]
        v_weight = self.weights[f"{prefix}.v_proj.weight"]
        q_bias = self.weights[f"{prefix}.q_proj.bias"]
        k_bias = self.weights[f"{prefix}.k_proj.bias"]
        v_bias = self.weights[f"{prefix}.v_proj.bias"]

        # Load weights into TRT-LLM
        attn.qkv_proj.load_weights(weights=[
            {"weight": q_weight, "bias": q_bias},
            {"weight": k_weight, "bias": k_bias},
            {"weight": v_weight, "bias": v_bias},
        ])

        # Create test input
        batch_size = 1
        seq_len = 4
        hidden_size = self.config_dict["hidden_size"]
        x = torch.randn(batch_size * seq_len, hidden_size,
                         dtype=torch.bfloat16, device="cuda")

        # Compute using separate projections (HF style)
        q_out_hf = torch.nn.functional.linear(x, q_weight.cuda(), q_bias.cuda())
        k_out_hf = torch.nn.functional.linear(x, k_weight.cuda(), k_bias.cuda())
        v_out_hf = torch.nn.functional.linear(x, v_weight.cuda(), v_bias.cuda())
        hf_output = torch.cat([q_out_hf, k_out_hf, v_out_hf], dim=-1)

        # Compute using fused projection (TRT-LLM style)
        trtllm_output = attn.qkv_proj(x)

        # Compare
        assert torch.allclose(trtllm_output.cpu(), hf_output.cpu(), atol=1e-2, rtol=1e-2), \
            (f"QKV projection numerical mismatch.\n"
             f"Max diff: {(trtllm_output.cpu() - hf_output.cpu()).abs().max()}\n"
             f"TRT-LLM output shape: {trtllm_output.shape}, HF output shape: {hf_output.shape}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGptOssAttentionRoPE:
    """Test RoPE configuration and computation consistency."""

    def test_rope_params_from_config(self):
        """Verify RopeParams.from_config correctly parses the GptOss config."""
        from transformers import PretrainedConfig as HfPretrainedConfig

        from tensorrt_llm._torch.attention_backend.interface import RopeParams

        config = HfPretrainedConfig.from_pretrained(CHECKPOINT_PATH)
        rope_params = RopeParams.from_config(config)

        assert rope_params.theta == 150000.0, \
            f"rope theta: {rope_params.theta} != 150000.0"
        assert rope_params.dim == 64, \
            f"rope dim: {rope_params.dim} != 64 (head_dim)"
        assert rope_params.scale == 32.0, \
            f"rope scale (factor): {rope_params.scale} != 32.0"
        assert rope_params.original_max_positions == 4096, \
            f"rope original_max_positions: {rope_params.original_max_positions} != 4096"
        assert rope_params.beta_fast == 32.0, \
            f"rope beta_fast: {rope_params.beta_fast} != 32.0"
        assert rope_params.beta_slow == 1.0, \
            f"rope beta_slow: {rope_params.beta_slow} != 1.0"
        # Verify it parsed as yarn scaling type
        from tensorrt_llm._torch.attention_backend.interface import \
            RotaryScalingType
        assert rope_params.scale_type == RotaryScalingType.yarn, \
            f"rope scale_type: {rope_params.scale_type} != RotaryScalingType.yarn"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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
"""
Module-level tests for MiMoV2FlashAttention.

Tests verify that the TRT-LLM MiMoV2FlashAttention module's sub-operations
(QKV projection, split, V-padding, output truncation, o_proj) produce
correct results when compared against the HuggingFace MiMoV2Attention module.

Since the full TRT-LLM attention forward requires complex AttentionMetadata
infrastructure, we test the individual building blocks independently.
"""

import sys
import json
import pytest
import torch
import torch.nn as nn

# Add HF model path for imports
sys.path.insert(0, "/workspace/MiMo-V2-Flash/MiMo-V2-Flash")

from types import SimpleNamespace


def make_hf_config():
    """Create a minimal HF-style config matching the real checkpoint."""
    with open("/workspace/MiMo-V2-Flash/MiMo-V2-Flash/config.json", "r") as f:
        cfg_dict = json.load(f)
    # Build a SimpleNamespace that behaves like the HF config
    cfg = SimpleNamespace(**cfg_dict)
    cfg.torch_dtype = torch.bfloat16
    return cfg


def make_trtllm_pretrained_config():
    """Create a pretrained config object for TRT-LLM ModelConfig."""
    cfg = make_hf_config()
    # TRT-LLM reads head_dim from pretrained_config
    return cfg


def make_model_config(pretrained_config):
    """Create a minimal TRT-LLM ModelConfig wrapping pretrained_config."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, rank=0)
    model_config = ModelConfig(pretrained_config=pretrained_config,
                               mapping=mapping,
                               attn_backend="TRTLLM")
    return model_config


# ========================================================
# Test 1: QKV projection dimensions
# ========================================================
class TestQKVProjectionDimensions:
    """Verify that the QKV projection output sizes are correct for both
    full-attention and SWA layers."""

    def test_full_attention_qkv_sizes(self):
        """Full attention layer (pattern=0): layer 0.
        Expected: q=12288, k=768, v=512, total=13568
        """
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=0)

        assert attn._qk_head_dim == 192, f"Expected qk_head_dim=192, got {attn._qk_head_dim}"
        assert attn._v_head_dim == 128, f"Expected v_head_dim=128, got {attn._v_head_dim}"
        assert attn.q_size == 64 * 192, f"Expected q_size=12288, got {attn.q_size}"
        assert attn.kv_size == 4 * 192, f"Expected kv_size=768, got {attn.kv_size}"
        assert attn._v_size_actual == 4 * 128, f"Expected v_size_actual=512, got {attn._v_size_actual}"

        # Check qkv_proj output dimension
        expected_total = 12288 + 768 + 512  # 13568
        qkv_out_features = attn.qkv_proj.out_features
        assert qkv_out_features == expected_total, (
            f"Expected qkv_proj out_features={expected_total}, got {qkv_out_features}")

    def test_swa_qkv_sizes(self):
        """SWA layer (pattern=1): layer 1.
        Expected: q=12288, k=1536, v=1024, total=14848
        """
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=1)

        assert attn._qk_head_dim == 192, f"Expected qk_head_dim=192, got {attn._qk_head_dim}"
        assert attn._v_head_dim == 128, f"Expected v_head_dim=128, got {attn._v_head_dim}"
        assert attn.q_size == 64 * 192, f"Expected q_size=12288, got {attn.q_size}"
        assert attn.kv_size == 8 * 192, f"Expected kv_size=1536, got {attn.kv_size}"
        assert attn._v_size_actual == 8 * 128, f"Expected v_size_actual=1024, got {attn._v_size_actual}"

        expected_total = 12288 + 1536 + 1024  # 14848
        qkv_out_features = attn.qkv_proj.out_features
        assert qkv_out_features == expected_total, (
            f"Expected qkv_proj out_features={expected_total}, got {qkv_out_features}")


# ========================================================
# Test 2: split_qkv correctness
# ========================================================
class TestSplitQKV:
    """Verify that split_qkv correctly splits [q_size, k_size, v_size]
    with different K and V sizes."""

    def test_split_qkv_full_attention(self):
        """Full attention: split [12288, 768, 512]."""
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=0)

        num_tokens = 4
        total = 12288 + 768 + 512
        qkv = torch.randn(num_tokens, total, dtype=torch.bfloat16)

        q, k, v = attn.split_qkv(qkv)
        assert q.shape == (num_tokens, 12288), f"Q shape: {q.shape}"
        assert k.shape == (num_tokens, 768), f"K shape: {k.shape}"
        assert v.shape == (num_tokens, 512), f"V shape: {v.shape}"

        # Verify the split is correct (values match original tensor)
        assert torch.equal(q, qkv[:, :12288])
        assert torch.equal(k, qkv[:, 12288:12288 + 768])
        assert torch.equal(v, qkv[:, 12288 + 768:])

    def test_split_qkv_swa(self):
        """SWA: split [12288, 1536, 1024]."""
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=1)

        num_tokens = 4
        total = 12288 + 1536 + 1024
        qkv = torch.randn(num_tokens, total, dtype=torch.bfloat16)

        q, k, v = attn.split_qkv(qkv)
        assert q.shape == (num_tokens, 12288), f"Q shape: {q.shape}"
        assert k.shape == (num_tokens, 1536), f"K shape: {k.shape}"
        assert v.shape == (num_tokens, 1024), f"V shape: {v.shape}"


# ========================================================
# Test 3: o_proj input dimension
# ========================================================
class TestOProjDimension:
    """Verify o_proj input size = num_heads * v_head_dim = 64 * 128 = 8192."""

    def test_o_proj_input_size(self):
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention

        for layer_idx in [0, 1]:  # full and SWA
            attn = MiMoV2FlashAttention(model_config=model_config,
                                        layer_idx=layer_idx)
            o_proj_in = attn.o_proj.in_features
            expected = 64 * 128  # 8192
            assert o_proj_in == expected, (
                f"Layer {layer_idx}: Expected o_proj in_features={expected}, "
                f"got {o_proj_in}")

            o_proj_out = attn.o_proj.out_features
            assert o_proj_out == 4096, (
                f"Layer {layer_idx}: Expected o_proj out_features=4096, "
                f"got {o_proj_out}")


# ========================================================
# Test 4: V-padding and output truncation
# ========================================================
class TestVPaddingAndTruncation:
    """Verify the V-padding strategy (pad V to head_dim, then truncate output)
    is mathematically correct."""

    def test_v_padding_shapes(self):
        """V should be padded from [num_tokens, num_kv_heads * 128]
        to [num_tokens, num_kv_heads * 192]."""
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention

        # SWA layer (8 kv heads)
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=1)

        num_tokens = 4
        v = torch.randn(num_tokens, 8 * 128, dtype=torch.bfloat16)
        v_padded = attn._pad_v(v)

        assert v_padded.shape == (num_tokens, 8 * 192), (
            f"Expected padded V shape ({num_tokens}, {8 * 192}), "
            f"got {v_padded.shape}")

        # Check that original values are preserved
        v_reshaped = v_padded.view(num_tokens, 8, 192)
        v_orig = v.view(num_tokens, 8, 128)
        assert torch.equal(v_reshaped[:, :, :128].to(torch.float32),
                           v_orig.to(torch.float32)), \
            "Original V values not preserved after padding"

        # Check that padding is zero
        assert torch.all(v_reshaped[:, :, 128:] == 0), \
            "Padding should be zeros"

    def test_output_truncation_shapes(self):
        """Output should be truncated from [num_tokens, num_heads * 192]
        to [num_tokens, num_heads * 128]."""
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=1)

        num_tokens = 4
        attn_output = torch.randn(num_tokens, 64 * 192, dtype=torch.bfloat16)
        truncated = attn._truncate_output(attn_output)

        assert truncated.shape == (num_tokens, 64 * 128), (
            f"Expected truncated shape ({num_tokens}, {64 * 128}), "
            f"got {truncated.shape}")

    def test_pad_truncate_mathematical_equivalence(self):
        """Test that zero-padding V and truncating the output is mathematically
        equivalent to using the original V dimensions directly.

        For a single head: attn_output = softmax(Q @ K^T / sqrt(d)) @ V
        If V is [seq_len, v_dim] and we pad to [seq_len, d] with zeros,
        then output[:, :v_dim] == original_output.
        """
        torch.manual_seed(42)

        # Simulate a single-head attention with small dimensions
        seq_len = 4
        qk_dim = 6   # analogous to head_dim=192
        v_dim = 4     # analogous to v_head_dim=128
        pad = qk_dim - v_dim

        Q = torch.randn(1, 1, seq_len, qk_dim, dtype=torch.float32)
        K = torch.randn(1, 1, seq_len, qk_dim, dtype=torch.float32)
        V_orig = torch.randn(1, 1, seq_len, v_dim, dtype=torch.float32)

        # Method 1: Direct computation with original V dimensions
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (qk_dim ** 0.5)
        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        output_direct = torch.matmul(attn_weights, V_orig)  # [1, 1, seq_len, v_dim]

        # Method 2: Pad V, compute, then truncate
        V_padded = torch.nn.functional.pad(V_orig, (0, pad), value=0.0)
        output_padded = torch.matmul(attn_weights, V_padded)  # [1, 1, seq_len, qk_dim]
        output_truncated = output_padded[:, :, :, :v_dim]

        assert torch.allclose(output_direct, output_truncated, atol=1e-6), (
            f"V-padding strategy not mathematically equivalent!\n"
            f"Direct output: {output_direct}\n"
            f"Padded+truncated output: {output_truncated}\n"
            f"Max diff: {(output_direct - output_truncated).abs().max()}")


# ========================================================
# Test 5: Per-layer RoPE configuration
# ========================================================
class TestPerLayerRoPE:
    """Verify that per-layer RoPE uses correct rope_theta and rope_dim."""

    def test_full_attention_rope_theta(self):
        """Full attention layers should use rope_theta=5000000."""
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=0)

        rope_params = attn.pos_embd_params.rope
        assert rope_params.theta == 5000000, (
            f"Expected rope_theta=5000000, got {rope_params.theta}")

    def test_swa_rope_theta(self):
        """SWA layers should use rope_theta=10000."""
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=1)

        rope_params = attn.pos_embd_params.rope
        assert rope_params.theta == 10000, (
            f"Expected rope_theta=10000, got {rope_params.theta}")

    def test_partial_rotary_factor(self):
        """Both full and SWA should have rope_dim = int(192 * 0.334) = 64."""
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention

        for layer_idx in [0, 1]:
            attn = MiMoV2FlashAttention(model_config=model_config,
                                        layer_idx=layer_idx)
            rope_dim = attn.pos_embd_params.rope.dim
            assert rope_dim == 64, (
                f"Layer {layer_idx}: Expected rope_dim=64, got {rope_dim}")


# ========================================================
# Test 6: Attention sink bias
# ========================================================
class TestAttentionSinkBias:
    """Verify attention sink bias is only created on SWA layers."""

    def test_swa_has_sink_bias(self):
        """SWA layers should have attention sink bias."""
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=1)

        assert attn.sinks is not None, "SWA layer should have sinks parameter"
        assert attn.sinks.shape == (64,), (
            f"Expected sinks shape (64,), got {attn.sinks.shape}")

    def test_full_attention_no_sink_bias(self):
        """Full attention layers should NOT have attention sink bias
        (add_full_attention_sink_bias=false)."""
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=0)

        assert attn.sinks is None, "Full attention layer should NOT have sinks parameter"


# ========================================================
# Test 7: Per-layer num_kv_heads
# ========================================================
class TestPerLayerKVHeads:
    """Verify per-layer num_kv_heads (4 for full, 8 for SWA)."""

    def test_full_attention_kv_heads(self):
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=0)
        assert attn.num_key_value_heads == 4, (
            f"Expected num_kv_heads=4, got {attn.num_key_value_heads}")

    def test_swa_kv_heads(self):
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=1)
        assert attn.num_key_value_heads == 8, (
            f"Expected num_kv_heads=8, got {attn.num_key_value_heads}")


# ========================================================
# Test 8: QKV projection numerical comparison with HF
# ========================================================
class TestQKVProjectionNumerical:
    """Compare QKV projection outputs between HF and TRT-LLM modules
    by copying weights and verifying outputs match."""

    @pytest.fixture
    def setup_swa_modules(self):
        """Set up both HF and TRT-LLM attention modules for SWA layer 1."""
        cfg = make_hf_config()
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from modeling_mimo_v2_flash import MiMoV2Attention as HFAttention
        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention

        hf_attn = HFAttention(cfg, is_swa=True, layer_idx=1).to(
            dtype=torch.bfloat16, device='cuda')

        trt_attn = MiMoV2FlashAttention(
            model_config=model_config, layer_idx=1).to(
            dtype=torch.bfloat16, device='cuda')

        return hf_attn, trt_attn

    def test_qkv_projection_output_match(self, setup_swa_modules):
        """Copy Q, K, V weights from HF to TRT-LLM (fused QKV),
        then verify the projections produce the same outputs."""
        hf_attn, trt_attn = setup_swa_modules

        # Copy HF q/k/v weights into the fused TRT-LLM qkv_proj
        with torch.no_grad():
            q_weight = hf_attn.q_proj.weight.data  # [12288, 4096]
            k_weight = hf_attn.k_proj.weight.data  # [1536, 4096]
            v_weight = hf_attn.v_proj.weight.data  # [1024, 4096]

            # Fused QKV weight: [q_size + k_size + v_size, hidden_size]
            fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            trt_attn.qkv_proj.weight.data.copy_(fused_weight)

        # Run projections
        torch.manual_seed(42)
        x = torch.randn(1, 4, 4096, dtype=torch.bfloat16, device='cuda')

        with torch.no_grad():
            # HF: separate projections
            hf_q = hf_attn.q_proj(x)
            hf_k = hf_attn.k_proj(x)
            hf_v = hf_attn.v_proj(x)
            hf_qkv = torch.cat([hf_q, hf_k, hf_v], dim=-1)

            # TRT-LLM: fused projection
            # TRT-LLM uses [num_tokens, hidden] format (no batch dim for
            # remove_input_padding mode), but Linear should handle both.
            trt_qkv = trt_attn.qkv_proj(x.view(-1, 4096))

        assert torch.allclose(hf_qkv.view(-1, hf_qkv.shape[-1]),
                              trt_qkv,
                              atol=1e-2, rtol=1e-2), (
            f"QKV projection mismatch!\n"
            f"Max diff: {(hf_qkv.view(-1, hf_qkv.shape[-1]) - trt_qkv).abs().max()}\n"
            f"HF shape: {hf_qkv.shape}, TRT shape: {trt_qkv.shape}")

    def test_split_qkv_after_projection(self, setup_swa_modules):
        """After fused QKV projection, split should produce the same
        Q, K, V as HF's separate projections."""
        hf_attn, trt_attn = setup_swa_modules

        with torch.no_grad():
            q_weight = hf_attn.q_proj.weight.data
            k_weight = hf_attn.k_proj.weight.data
            v_weight = hf_attn.v_proj.weight.data
            fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            trt_attn.qkv_proj.weight.data.copy_(fused_weight)

        torch.manual_seed(42)
        x = torch.randn(4, 4096, dtype=torch.bfloat16, device='cuda')

        with torch.no_grad():
            hf_q = hf_attn.q_proj(x)
            hf_k = hf_attn.k_proj(x)
            hf_v = hf_attn.v_proj(x)

            trt_qkv = trt_attn.qkv_proj(x)
            trt_q, trt_k, trt_v = trt_attn.split_qkv(trt_qkv)

        assert torch.allclose(hf_q, trt_q, atol=1e-2, rtol=1e-2), (
            f"Q mismatch! Max diff: {(hf_q - trt_q).abs().max()}")
        assert torch.allclose(hf_k, trt_k, atol=1e-2, rtol=1e-2), (
            f"K mismatch! Max diff: {(hf_k - trt_k).abs().max()}")
        assert torch.allclose(hf_v, trt_v, atol=1e-2, rtol=1e-2), (
            f"V mismatch! Max diff: {(hf_v - trt_v).abs().max()}")


# ========================================================
# Test 9: o_proj numerical comparison
# ========================================================
class TestOProjNumerical:
    """Compare o_proj outputs between HF and TRT-LLM by copying weights."""

    def test_o_proj_output_match(self):
        """o_proj should produce the same output given the same weights
        and input. Input size = 64 * 128 = 8192."""
        cfg = make_hf_config()
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from modeling_mimo_v2_flash import MiMoV2Attention as HFAttention
        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention

        hf_attn = HFAttention(cfg, is_swa=True, layer_idx=1).to(
            dtype=torch.bfloat16, device='cuda')
        trt_attn = MiMoV2FlashAttention(
            model_config=model_config, layer_idx=1).to(
            dtype=torch.bfloat16, device='cuda')

        # Copy o_proj weights
        with torch.no_grad():
            trt_attn.o_proj.weight.data.copy_(hf_attn.o_proj.weight.data)

        # Input: [num_tokens, 8192]
        torch.manual_seed(42)
        x = torch.randn(4, 8192, dtype=torch.bfloat16, device='cuda')

        with torch.no_grad():
            hf_out = hf_attn.o_proj(x)
            trt_out = trt_attn.o_proj(x)

        assert torch.allclose(hf_out, trt_out, atol=1e-2, rtol=1e-2), (
            f"o_proj output mismatch!\n"
            f"Max diff: {(hf_out - trt_out).abs().max()}\n"
            f"HF shape: {hf_out.shape}, TRT shape: {trt_out.shape}")


# ========================================================
# Test 10: Window size configuration
# ========================================================
class TestWindowSize:
    """Verify window_size is set correctly for SWA and full attention layers."""

    def test_swa_window_size(self):
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=1)
        assert attn._window_size == 128, (
            f"Expected window_size=128, got {attn._window_size}")

    def test_full_attention_no_window(self):
        pretrained_config = make_trtllm_pretrained_config()
        model_config = make_model_config(pretrained_config)

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import MiMoV2FlashAttention
        attn = MiMoV2FlashAttention(model_config=model_config, layer_idx=0)
        assert attn._window_size is None, (
            f"Expected window_size=None, got {attn._window_size}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

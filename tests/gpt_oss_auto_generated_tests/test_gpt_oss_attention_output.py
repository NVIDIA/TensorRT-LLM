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
Module-level numerical comparison test: AttentionBlock (TRT-LLM) vs GptOssAttention (HF)

Tests the sub-operations of the attention module that can be compared without
requiring the full TRT-LLM attention backend (which needs KV cache, paged
attention, etc.):

1. QKV projection: fused TRT-LLM qkv_proj vs separate HF q/k/v_proj
2. RoPE application: TRT-LLM RotaryEmbedding vs HF _apply_rotary_emb
3. Output projection: TRT-LLM o_proj vs HF o_proj
4. Sinks: weight loading, dtype (float32), and TP slicing
"""

import importlib.util
import json
import os
import sys

import pytest
import torch

CHECKPOINT_PATH = "/scratch.trt_llm_data/llm-models/gpt_oss/gpt-oss-20b/"
HF_SOURCE_PATH = "/home/scratch.huig_gpu/models/modeling_gpt_oss.py"
LAYER_IDX = 0  # Layer 0 = sliding_attention


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _skip_if_no_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip("Checkpoint not found")


def _load_hf_config():
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(CHECKPOINT_PATH)


def _load_hf_module():
    """Import HF GptOssAttention from the source file."""
    spec = importlib.util.spec_from_file_location("modeling_gpt_oss_hf",
                                                   HF_SOURCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Need to add the parent transformers package context
    sys.modules["modeling_gpt_oss_hf"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_layer_weights(layer_idx: int):
    """Load attention weights for a specific layer from safetensors checkpoint."""
    from safetensors.torch import load_file

    index_path = os.path.join(CHECKPOINT_PATH, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    prefix = f"model.layers.{layer_idx}.self_attn."
    needed_keys = [
        f"{prefix}q_proj.weight",
        f"{prefix}q_proj.bias",
        f"{prefix}k_proj.weight",
        f"{prefix}k_proj.bias",
        f"{prefix}v_proj.weight",
        f"{prefix}v_proj.bias",
        f"{prefix}o_proj.weight",
        f"{prefix}o_proj.bias",
        f"{prefix}sinks",
    ]

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


def _create_trtllm_attention(hf_config, layer_idx: int):
    """Create TRT-LLM AttentionBlock."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, rank=0)
    model_config = ModelConfig(
        pretrained_config=hf_config,
        mapping=mapping,
        attn_backend='TRTLLM',
    )

    from tensorrt_llm._torch.models.modeling_gpt_oss import AttentionBlock
    trt_attn = AttentionBlock(model_config, layer_idx=layer_idx)
    return trt_attn


def _load_trtllm_weights(trt_attn, weights, layer_idx: int):
    """Load checkpoint weights into TRT-LLM AttentionBlock."""
    prefix = f"model.layers.{layer_idx}.self_attn."

    q_w = {
        "weight": weights[f"{prefix}q_proj.weight"],
        "bias": weights[f"{prefix}q_proj.bias"]
    }
    k_w = {
        "weight": weights[f"{prefix}k_proj.weight"],
        "bias": weights[f"{prefix}k_proj.bias"]
    }
    v_w = {
        "weight": weights[f"{prefix}v_proj.weight"],
        "bias": weights[f"{prefix}v_proj.bias"]
    }
    trt_attn.qkv_proj.load_weights(weights=[q_w, k_w, v_w])

    o_w = {
        "weight": weights[f"{prefix}o_proj.weight"],
        "bias": weights[f"{prefix}o_proj.bias"]
    }
    trt_attn.o_proj.load_weights(weights=[o_w])

    sinks_w = {"sinks": weights[f"{prefix}sinks"]}
    trt_attn.load_weights(weights=[sinks_w])


@pytest.fixture(scope="module")
def test_setup():
    """Module-scoped fixture: load config, weights, create both modules."""
    _skip_if_no_cuda()
    _skip_if_no_checkpoint()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    hf_config = _load_hf_config()
    weights = _load_layer_weights(LAYER_IDX)
    hf_mod = _load_hf_module()

    # Create HF attention
    hf_attn = hf_mod.GptOssAttention(hf_config, layer_idx=LAYER_IDX)
    prefix = f"model.layers.{LAYER_IDX}.self_attn."
    hf_attn.q_proj.weight.data.copy_(weights[f"{prefix}q_proj.weight"])
    hf_attn.q_proj.bias.data.copy_(weights[f"{prefix}q_proj.bias"])
    hf_attn.k_proj.weight.data.copy_(weights[f"{prefix}k_proj.weight"])
    hf_attn.k_proj.bias.data.copy_(weights[f"{prefix}k_proj.bias"])
    hf_attn.v_proj.weight.data.copy_(weights[f"{prefix}v_proj.weight"])
    hf_attn.v_proj.bias.data.copy_(weights[f"{prefix}v_proj.bias"])
    hf_attn.o_proj.weight.data.copy_(weights[f"{prefix}o_proj.weight"])
    hf_attn.o_proj.bias.data.copy_(weights[f"{prefix}o_proj.bias"])
    hf_attn.sinks.data.copy_(weights[f"{prefix}sinks"])
    hf_attn = hf_attn.to(device).to(dtype).eval()

    # Create TRT-LLM attention
    trt_attn = _create_trtllm_attention(hf_config, layer_idx=LAYER_IDX)
    trt_attn = trt_attn.to(device)
    _load_trtllm_weights(trt_attn, weights, LAYER_IDX)
    trt_attn = trt_attn.eval()

    # HF RoPE embedding
    hf_rope = hf_mod.GptOssRotaryEmbedding(hf_config, device=device)

    return {
        "hf_attn": hf_attn,
        "trt_attn": trt_attn,
        "hf_rope": hf_rope,
        "hf_mod": hf_mod,
        "device": device,
        "dtype": dtype,
        "config": hf_config,
        "weights": weights,
    }


class TestQKVProjection:
    """Compare fused TRT-LLM qkv_proj against separate HF q/k/v_proj."""

    def test_qkv_output_matches(self, test_setup):
        """Feed same input to both QKV projections and compare outputs."""
        hf_attn = test_setup["hf_attn"]
        trt_attn = test_setup["trt_attn"]
        device = test_setup["device"]
        dtype = test_setup["dtype"]

        batch, seq_len, hidden = 1, 8, 2880
        x = torch.randn(batch * seq_len, hidden, device=device, dtype=dtype)

        with torch.no_grad():
            # HF: separate projections
            hf_q = hf_attn.q_proj(x)  # [tokens, 4096]
            hf_k = hf_attn.k_proj(x)  # [tokens, 512]
            hf_v = hf_attn.v_proj(x)  # [tokens, 512]
            hf_qkv = torch.cat([hf_q, hf_k, hf_v], dim=-1)  # [tokens, 5120]

            # TRT-LLM: fused projection
            trt_qkv = trt_attn.qkv_proj(x)  # [tokens, 5120]

        assert hf_qkv.shape == trt_qkv.shape, (
            f"QKV shape mismatch: HF={hf_qkv.shape}, TRT={trt_qkv.shape}")

        if not torch.allclose(hf_qkv, trt_qkv, atol=1e-2, rtol=1e-2):
            diff = (hf_qkv - trt_qkv).abs()
            raise AssertionError(
                f"QKV projection output mismatch.\n"
                f"Max abs diff: {diff.max().item():.6f}\n"
                f"Mean abs diff: {diff.mean().item():.6f}\n"
                f"HF output sample: {hf_qkv[0, :5]}\n"
                f"TRT output sample: {trt_qkv[0, :5]}\n"
                f"HF shape: {hf_qkv.shape}, TRT shape: {trt_qkv.shape}")


class TestOutputProjection:
    """Compare TRT-LLM o_proj against HF o_proj."""

    def test_output_proj_matches(self, test_setup):
        """Feed same intermediate result to both output projections."""
        hf_attn = test_setup["hf_attn"]
        trt_attn = test_setup["trt_attn"]
        device = test_setup["device"]
        dtype = test_setup["dtype"]

        # Output projection input: [tokens, num_heads * head_dim] = [tokens, 4096]
        tokens = 8
        attn_out = torch.randn(tokens, 4096, device=device, dtype=dtype)

        with torch.no_grad():
            hf_out = hf_attn.o_proj(attn_out)  # [tokens, 2880]
            trt_out = trt_attn.o_proj(attn_out)  # [tokens, 2880]

        assert hf_out.shape == trt_out.shape, (
            f"Output proj shape mismatch: HF={hf_out.shape}, TRT={trt_out.shape}"
        )

        if not torch.allclose(hf_out, trt_out, atol=1e-2, rtol=1e-2):
            diff = (hf_out - trt_out).abs()
            raise AssertionError(
                f"Output projection mismatch.\n"
                f"Max abs diff: {diff.max().item():.6f}\n"
                f"Mean abs diff: {diff.mean().item():.6f}\n"
                f"HF output sample: {hf_out[0, :5]}\n"
                f"TRT output sample: {trt_out[0, :5]}\n"
                f"HF shape: {hf_out.shape}, TRT shape: {trt_out.shape}")


class TestRoPEApplication:
    """Compare RoPE application between HF and TRT-LLM."""

    def test_rope_cos_sin_values(self, test_setup):
        """Compare that RoPE cos/sin values agree between HF and TRT-LLM."""
        trt_attn = test_setup["trt_attn"]
        hf_rope = test_setup["hf_rope"]
        device = test_setup["device"]
        dtype = test_setup["dtype"]

        seq_len = 8
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # HF RoPE: cos, sin each [1, seq_len, head_dim/2] or [1, seq_len, head_dim]
        dummy_x = torch.randn(1, seq_len, 2880, device=device, dtype=dtype)
        with torch.no_grad():
            hf_cos, hf_sin = hf_rope(dummy_x, position_ids)
        # hf_cos shape: [1, seq_len, head_dim] = [1, 8, 64]

        # TRT-LLM RoPE: stored in rotary_cos_sin [max_positions, 2, dim]
        trt_rope = trt_attn.rotary_emb
        if trt_rope is not None:
            trt_cos_sin = trt_rope.rotary_cos_sin  # [max_positions, 2, dim]
            trt_cos = trt_cos_sin[position_ids.squeeze(0),
                                  0, :]  # [seq_len, dim]
            trt_sin = trt_cos_sin[position_ids.squeeze(0),
                                  1, :]  # [seq_len, dim]

            # HF cos is [1, seq_len, head_dim], squeeze batch
            hf_cos_sq = hf_cos.squeeze(0).float()  # [seq_len, head_dim]
            trt_cos_f = trt_cos.float()  # [seq_len, dim]

            # The dimensions should match: both should be head_dim or head_dim/2
            # TRT-LLM with is_neox=False stores full cos/sin of dim = head_dim/2
            # but the values should correspond
            min_dim = min(hf_cos_sq.shape[-1], trt_cos_f.shape[-1])
            hf_cos_cmp = hf_cos_sq[..., :min_dim]
            trt_cos_cmp = trt_cos_f[..., :min_dim]

            if not torch.allclose(hf_cos_cmp, trt_cos_cmp, atol=1e-4,
                                  rtol=1e-3):
                diff = (hf_cos_cmp - trt_cos_cmp).abs()
                raise AssertionError(
                    f"RoPE cos mismatch.\n"
                    f"Max abs diff: {diff.max().item():.6f}\n"
                    f"HF cos shape: {hf_cos_sq.shape}, TRT cos shape: {trt_cos_f.shape}\n"
                    f"HF cos sample: {hf_cos_sq[0, :5]}\n"
                    f"TRT cos sample: {trt_cos_f[0, :5]}")
        else:
            pytest.skip(
                "RoPE is fused into attention backend, cannot compare directly")

    def test_rope_applied_to_query(self, test_setup):
        """Compare RoPE-applied query states between HF and TRT-LLM."""
        hf_attn = test_setup["hf_attn"]
        trt_attn = test_setup["trt_attn"]
        hf_rope = test_setup["hf_rope"]
        hf_mod = test_setup["hf_mod"]
        device = test_setup["device"]
        dtype = test_setup["dtype"]
        config = test_setup["config"]

        batch, seq_len, hidden = 1, 8, 2880
        x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        with torch.no_grad():
            # HF path: project -> reshape -> apply_rotary_pos_emb
            hf_q = hf_attn.q_proj(x)  # [1, 8, 4096]
            head_dim = config.head_dim
            num_heads = config.num_attention_heads
            hf_q_heads = hf_q.view(batch, seq_len, num_heads,
                                   head_dim).transpose(1,
                                                       2)  # [1, 64, 8, 64]

            hf_k = hf_attn.k_proj(x)  # [1, 8, 512]
            num_kv_heads = config.num_key_value_heads
            hf_k_heads = hf_k.view(batch, seq_len, num_kv_heads,
                                   head_dim).transpose(1, 2)  # [1, 8, 8, 64]

            cos, sin = hf_rope(x, position_ids)
            hf_q_rotated, hf_k_rotated = hf_mod.apply_rotary_pos_emb(
                hf_q_heads, hf_k_heads, cos, sin)
            # hf_q_rotated: [1, 64, 8, 64]

            # TRT-LLM path: fused qkv -> split -> apply_rope
            trt_qkv = trt_attn.qkv_proj(
                x.view(-1, hidden))  # [tokens, q+k+v sizes]
            q_size = num_heads * head_dim
            kv_size = num_kv_heads * head_dim
            trt_q_flat, trt_k_flat, trt_v_flat = trt_qkv.split(
                [q_size, kv_size, kv_size], dim=-1)

        if trt_attn.rotary_emb is not None:
            with torch.no_grad():
                # Apply TRT-LLM RoPE
                flat_pos = position_ids.view(-1)  # [seq_len]
                trt_q_rope, trt_k_rope = trt_attn.rotary_emb(
                    flat_pos, [trt_q_flat, trt_k_flat])

                # Reshape for comparison: [tokens, q_size] -> [1, num_heads, seq_len, head_dim]
                trt_q_compare = trt_q_rope.view(batch, seq_len, num_heads,
                                                head_dim).transpose(1, 2)

                # Compare
                if not torch.allclose(hf_q_rotated.to(dtype),
                                      trt_q_compare.to(dtype),
                                      atol=5e-2,
                                      rtol=5e-2):
                    diff = (hf_q_rotated.to(dtype) -
                            trt_q_compare.to(dtype)).abs()
                    raise AssertionError(
                        f"RoPE-applied query mismatch.\n"
                        f"Max abs diff: {diff.max().item():.6f}\n"
                        f"Mean abs diff: {diff.mean().item():.6f}\n"
                        f"HF shape: {hf_q_rotated.shape}, TRT shape: {trt_q_compare.shape}"
                    )
        else:
            pytest.skip("RoPE is fused into attention backend")


class TestSinksWeightLoading:
    """Verify sinks parameter loading: float32 dtype, TP slicing."""

    def test_sinks_dtype_is_float32(self, test_setup):
        """Sinks should be stored in float32 regardless of model dtype."""
        trt_attn = test_setup["trt_attn"]
        assert trt_attn.sinks.dtype == torch.float32, (
            f"Expected sinks dtype float32, got {trt_attn.sinks.dtype}")

    def test_sinks_values_match_checkpoint(self, test_setup):
        """Sinks values should match checkpoint values (cast to float32)."""
        trt_attn = test_setup["trt_attn"]
        weights = test_setup["weights"]
        device = test_setup["device"]

        expected = weights[
            f"model.layers.{LAYER_IDX}.self_attn.sinks"].to(
                torch.float32).to(device)

        if not torch.allclose(trt_attn.sinks.data, expected, atol=1e-6):
            diff = (trt_attn.sinks.data - expected).abs()
            raise AssertionError(
                f"Sinks value mismatch.\n"
                f"Max abs diff: {diff.max().item():.8f}\n"
                f"TRT sinks sample: {trt_attn.sinks.data[:5]}\n"
                f"Expected sample: {expected[:5]}")

    def test_sinks_shape_tp1(self, test_setup):
        """With TP=1, sinks shape should be [num_attention_heads]."""
        trt_attn = test_setup["trt_attn"]
        assert trt_attn.sinks.shape == (64, ), (
            f"Expected sinks shape (64,), got {trt_attn.sinks.shape}")

    def test_sinks_tp_slicing(self, test_setup):
        """Verify TP slicing: TP=2 rank=1 should get second half of sinks."""
        config = test_setup["config"]
        weights = test_setup["weights"]
        device = test_setup["device"]

        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm.mapping import Mapping

        mapping_tp2 = Mapping(world_size=2, tp_size=2, pp_size=1, rank=1)
        model_config_tp2 = ModelConfig(
            pretrained_config=config,
            mapping=mapping_tp2,
            attn_backend='TRTLLM',
        )

        from tensorrt_llm._torch.models.modeling_gpt_oss import AttentionBlock
        trt_attn_tp2 = AttentionBlock(model_config_tp2, layer_idx=LAYER_IDX)
        trt_attn_tp2 = trt_attn_tp2.to(device)

        # Load weights -- the load_weights method should slice for TP rank 1
        sinks_w = {
            "sinks":
            weights[f"model.layers.{LAYER_IDX}.self_attn.sinks"]
        }
        trt_attn_tp2.load_weights(weights=[sinks_w])

        # TP rank 1 with TP=2: should get sinks[32:64]
        expected = weights[
            f"model.layers.{LAYER_IDX}.self_attn.sinks"][32:64].to(
                torch.float32).to(device)

        assert trt_attn_tp2.sinks.shape == (32, ), (
            f"TP=2 sinks shape should be (32,), got {trt_attn_tp2.sinks.shape}"
        )

        if not torch.allclose(trt_attn_tp2.sinks.data, expected, atol=1e-6):
            diff = (trt_attn_tp2.sinks.data - expected).abs()
            raise AssertionError(
                f"TP-sliced sinks mismatch for rank=1.\n"
                f"Max abs diff: {diff.max().item():.8f}\n"
                f"TRT sinks[:5]: {trt_attn_tp2.sinks.data[:5]}\n"
                f"Expected[:5]: {expected[:5]}")


class TestSlidingWindowConfig:
    """Verify sliding window is correctly applied per layer."""

    def test_layer0_has_sliding_window(self, test_setup):
        """Layer 0 (even) should have sliding_window=128."""
        trt_attn = test_setup["trt_attn"]
        assert trt_attn.sliding_window == 128, (
            f"Layer 0 sliding_window should be 128, got {trt_attn.sliding_window}"
        )

    def test_layer1_has_no_sliding_window(self, test_setup):
        """Layer 1 (odd) should have sliding_window=None."""
        config = test_setup["config"]
        trt_attn_1 = _create_trtllm_attention(config, layer_idx=1)
        assert trt_attn_1.sliding_window is None, (
            f"Layer 1 sliding_window should be None, got {trt_attn_1.sliding_window}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=long"])

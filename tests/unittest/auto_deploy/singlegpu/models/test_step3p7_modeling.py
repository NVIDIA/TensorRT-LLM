# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical equivalence tests for the StepFun Step-3.7-Flash AutoDeploy custom model.

Step-3.7-Flash uses ``trust_remote_code`` (its ``step3p7`` / ``step3p5`` modeling code is not
in transformers natively), so the reference classes (``_Ref*``) below are minimal, faithful
standalone reimplementations of the HuggingFace text-decoder math (modeling_step3p7.py).

Levels: MLP block -> Attention block (full + sliding) -> MoE block -> Decoder layer ->
Full model -> Export.
"""

from typing import Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — register ops
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_step3p7 import (
    Step3p7Attention,
    Step3p7DecoderLayer,
    Step3p7ForCausalLM,
    Step3p7MLP,
    Step3p7MoE,
    Step3p7RotaryEmbedding,
)


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Seed RNG before every test so random weights/inputs are reproducible.

    The MoE routing top-k can otherwise be sensitive to inter-test RNG state on
    near-tied expert scores, making the equivalence checks intermittently flaky.
    """
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Test-only config (minimal faithful copy of the HF Step3p7 text config)
# ---------------------------------------------------------------------------


class Step3p7TestConfig(PretrainedConfig):
    """Flat text config for testing.

    The model's ``_get_text_config`` treats a config without a ``text_config`` attribute as the
    text config itself, so no VLM wrapper is needed here.
    """

    model_type = "step3p5"

    def __init__(
        self,
        vocab_size=1000,
        hidden_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_attention_groups=2,
        attention_other_setting=None,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=None,
        moe_layers_enum=(2, 3),
        moe_num_experts=8,
        moe_top_k=2,
        moe_intermediate_size=32,
        share_expert_dim=32,
        moe_router_scaling_factor=3.0,
        rms_norm_eps=1e-5,
        sliding_window=4,
        max_position_embeddings=64,
        rope_theta=(5e6, 1e4, 5e6, 1e4),
        partial_rotary_factors=(0.5, 1.0, 0.5, 1.0),
        rope_scaling=None,
        yarn_only_types=("full_attention",),
        swiglu_limits=None,
        swiglu_limits_shared=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.attention_other_setting = attention_other_setting or {
            "num_attention_heads": 6,
            "num_attention_groups": 2,
            "head_dim": 16,
        }
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_types = layer_types or [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]
        self.moe_layers_enum = moe_layers_enum
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_intermediate_size = moe_intermediate_size
        self.share_expert_dim = share_expert_dim
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = list(rope_theta)
        self.partial_rotary_factors = list(partial_rotary_factors)
        self.rope_scaling = rope_scaling or {
            "rope_type": "llama3",
            "factor": 2.0,
            "original_max_position_embeddings": 64,
            "low_freq_factor": 1.0,
            "high_freq_factor": 32.0,
        }
        self.yarn_only_types = list(yarn_only_types)
        self.swiglu_limits = swiglu_limits
        self.swiglu_limits_shared = swiglu_limits_shared
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_rmse_close(actual, expected, rmse_ratio_tol, msg=""):
    diff = actual.float() - expected.float()
    rmse_diff = torch.sqrt(torch.mean(diff**2))
    rmse_ref = torch.sqrt(torch.mean(expected.float() ** 2))
    ratio = (rmse_diff / rmse_ref).item()
    assert ratio < rmse_ratio_tol, (
        f"{msg}RMSE ratio {ratio:.6f} exceeds tolerance {rmse_ratio_tol}. "
        f"(rmse_diff={rmse_diff.item():.6f}, rmse_ref={rmse_ref.item():.6f})"
    )


def _device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _position_ids(batch: int, seq: int, device) -> torch.Tensor:
    return torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)


def _small_config(**overrides) -> Step3p7TestConfig:
    return Step3p7TestConfig(**overrides)


def _add_one_to_norms(state_dict: dict) -> dict:
    """Mimic the AD load hook: absorb the (1 + weight) RMSNorm convention into norm weights."""
    out = {}
    for k, v in state_dict.items():
        if k.endswith("layernorm.weight") or k.endswith("norm.weight"):
            out[k] = v + 1.0
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Reference implementations (HF-faithful, plain PyTorch, bnsd layout)
# ---------------------------------------------------------------------------


class _RefRMSNorm(nn.Module):
    """Step RMSNorm with the (1 + weight) convention."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.variance_epsilon)
        normed = normed * (self.weight.float() + 1)
        return normed.to(dtype)


def _ref_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_ref_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_ref_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def _ref_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class _RefMLP(nn.Module):
    """Reference SwiGLU MLP with optional post-activation clamp (matches Step3p7MLP naming)."""

    def __init__(self, hidden_size, intermediate_size, swiglu_limit=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        return self.down_proj(gate * up)


class _RefAttention(nn.Module):
    """Reference Step attention: per-head QK norm, partial RoPE, head-wise gate, GQA, sliding."""

    def __init__(self, config: Step3p7TestConfig, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        if is_sliding:
            other = config.attention_other_setting
            self.num_heads = other["num_attention_heads"]
            self.num_kv_heads = other["num_attention_groups"]
            self.sliding_window = config.sliding_window
        else:
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_attention_groups
            self.sliding_window = None
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.q_norm = _RefRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RefRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings):
        bsz, q_len, _ = hidden_states.size()
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        gate = self.g_proj(hidden_states)  # [B, S, N]

        cos, sin = position_embeddings
        q, k = _ref_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        k = _ref_repeat_kv(k, self.num_kv_groups)
        v = _ref_repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        qpos = torch.arange(q_len, device=q.device)
        pos_diff = qpos.unsqueeze(1) - qpos.unsqueeze(0)
        if self.sliding_window is not None:
            mask = (pos_diff < 0) | (pos_diff >= self.sliding_window)
        else:
            mask = pos_diff < 0
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)  # [B, N, S, D]

        attn_output = attn_output.transpose(1, 2)  # [B, S, N, D]
        attn_output = attn_output * gate.unsqueeze(-1).sigmoid()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class _RefMoE(nn.Module):
    """Reference routed Step MoE: sigmoid routing + per-expert bias (selection only).

    The shared expert is a sibling of this module on the decoder layer (matching the checkpoint
    hierarchy), so it lives on ``_RefDecoderLayer``, not here.
    """

    def __init__(self, config: Step3p7TestConfig):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(self.num_experts, dtype=torch.float32))
        self.experts = nn.ModuleList(
            [
                _RefMLP(config.hidden_size, config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states):
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())
        probs = torch.sigmoid(router_logits)
        scores = probs + self.router_bias.unsqueeze(0)
        _, idx = torch.topk(scores, self.top_k, dim=-1)
        weights = torch.gather(probs, 1, idx)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        weights = weights.to(hidden_flat.dtype)

        final = torch.zeros_like(hidden_flat)
        expert_mask = F.one_hot(idx, num_classes=self.num_experts).permute(2, 1, 0)
        for e in range(self.num_experts):
            tok_idx, top_x = torch.where(expert_mask[e])
            if top_x.numel() == 0:
                continue
            out = self.experts[e](hidden_flat[top_x]) * weights[top_x, tok_idx, None]
            final.index_add_(0, top_x, out.to(hidden_flat.dtype))

        return final.view(bsz, seq_len, hidden_dim)


class _RefDecoderLayer(nn.Module):
    def __init__(self, config: Step3p7TestConfig, layer_idx: int, is_moe: bool):
        super().__init__()
        self.self_attn = _RefAttention(config, layer_idx)
        self.is_moe = is_moe
        if is_moe:
            self.moe = _RefMoE(config)
            self.share_expert = _RefMLP(config.hidden_size, config.share_expert_dim)
        else:
            self.mlp = _RefMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            hidden_states = self.moe(hidden_states) + self.share_expert(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# RoPE helper: build cos/sin for a given attention type using the AD module
# ---------------------------------------------------------------------------


def _build_position_embeddings(config, layer_idx, B, S, device, dtype):
    layer_type = config.layer_types[layer_idx]
    scaling = config.rope_scaling if layer_type in config.yarn_only_types else None
    rope = Step3p7RotaryEmbedding(
        head_dim=config.head_dim,
        partial_rotary_factor=config.partial_rotary_factors[layer_idx],
        base=config.rope_theta[layer_idx],
        max_position_embeddings=config.max_position_embeddings,
        rope_scaling=scaling,
    ).to(device)
    dummy = torch.zeros(1, device=device, dtype=dtype)
    return rope(dummy, _position_ids(B, S, device))


def _transfer_block(ad_module, ref_module):
    """Load reference weights into an AD block, absorbing the (1 + weight) norm convention."""
    ad_module.load_state_dict(_add_one_to_norms(ref_module.state_dict()), strict=True)


# ---------------------------------------------------------------------------
# Tests — Block equivalence
# ---------------------------------------------------------------------------


def test_mlp_equivalence():
    device, dtype = _device_and_dtype()
    config = _small_config()
    ref = _RefMLP(config.hidden_size, config.intermediate_size).to(device, dtype).eval()
    ad = Step3p7MLP(config.hidden_size, config.intermediate_size).to(device, dtype).eval()
    ad.load_state_dict(ref.state_dict())
    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        torch.testing.assert_close(ad(x), ref(x), rtol=1e-3, atol=1e-3)


def test_mlp_clamped_equivalence():
    """SwiGLU clamp (swiglu_limit) on the dense/shared MLP path matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_config()
    ref = _RefMLP(config.hidden_size, config.intermediate_size, swiglu_limit=0.5).to(device, dtype)
    ref = ref.eval()
    ad = Step3p7MLP(config.hidden_size, config.intermediate_size, swiglu_limit=0.5).to(
        device, dtype
    )
    ad = ad.eval()
    ad.load_state_dict(ref.state_dict())
    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype) * 3.0  # trigger clamp
    with torch.no_grad():
        torch.testing.assert_close(ad(x), ref(x), rtol=1e-3, atol=1e-3)


def test_attention_full_equivalence():
    """Full-attention layer (64-head config, partial RoPE w/ llama3 scaling) matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_config()
    B, S = 2, 8
    layer_idx = 0  # full_attention
    ref = _RefAttention(config, layer_idx).to(device, dtype).eval()
    ad = Step3p7Attention(config, layer_idx).to(device, dtype).eval()
    _transfer_block(ad, ref)
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    pe = _build_position_embeddings(config, layer_idx, B, S, device, dtype)
    with torch.no_grad():
        ad_out = ad(x, position_embeddings=pe)
        ref_out = ref(x, position_embeddings=pe)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.10, msg="Full attention: ")


def test_attention_sliding_equivalence():
    """Sliding-attention layer (96-head config, full RoPE, sliding window) matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_config()
    B, S = 2, 8
    layer_idx = 1  # sliding_attention
    ref = _RefAttention(config, layer_idx).to(device, dtype).eval()
    ad = Step3p7Attention(config, layer_idx).to(device, dtype).eval()
    _transfer_block(ad, ref)
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    pe = _build_position_embeddings(config, layer_idx, B, S, device, dtype)
    with torch.no_grad():
        ad_out = ad(x, position_embeddings=pe)
        ref_out = ref(x, position_embeddings=pe)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.10, msg="Sliding attention: ")


def test_moe_block_equivalence():
    device, dtype = _device_and_dtype()
    config = _small_config()
    ref = _RefMoE(config).to(device, dtype).eval()
    ad = Step3p7MoE(config).to(device, dtype).eval()
    ad.load_state_dict(ref.state_dict())
    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        ad_out = ad(x)
        ref_out = ref(x)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.02, msg="MoE block: ")


# ---------------------------------------------------------------------------
# Tests — Layer equivalence
# ---------------------------------------------------------------------------


def test_decoder_layer_moe_equivalence():
    """MoE decoder layer (full-attention variant) matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_config()
    B, S = 2, 8
    layer_idx = 2  # full_attention + moe
    ref = _RefDecoderLayer(config, layer_idx, is_moe=True).to(device, dtype).eval()
    ad = Step3p7DecoderLayer(config, layer_idx, is_moe_layer=True).to(device, dtype).eval()
    _transfer_block(ad, ref)
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    full_pe = _build_position_embeddings(config, layer_idx, B, S, device, dtype)
    with torch.no_grad():
        ad_out = ad(x, full_pe, full_pe)  # full-attention layer ignores sliding pe
        ref_out = ref(x, full_pe)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.05, msg="MoE decoder layer: ")


def test_decoder_layer_dense_sliding_equivalence():
    """Dense decoder layer (sliding-attention variant) matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_config()
    B, S = 2, 8
    layer_idx = 1  # sliding_attention + dense
    ref = _RefDecoderLayer(config, layer_idx, is_moe=False).to(device, dtype).eval()
    ad = Step3p7DecoderLayer(config, layer_idx, is_moe_layer=False).to(device, dtype).eval()
    _transfer_block(ad, ref)
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    sliding_pe = _build_position_embeddings(config, layer_idx, B, S, device, dtype)
    with torch.no_grad():
        ad_out = ad(x, sliding_pe, sliding_pe)
        ref_out = ref(x, sliding_pe)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.05, msg="Dense sliding decoder layer: ")


# ---------------------------------------------------------------------------
# Tests — Full model equivalence (also exercises the load-state-dict hooks)
# ---------------------------------------------------------------------------


class _RefForCausalLM(nn.Module):
    def __init__(self, config: Step3p7TestConfig):
        super().__init__()
        self.config = config
        moe_layers = set(config.moe_layers_enum)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                _RefDecoderLayer(config, i, is_moe=i in moe_layers)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids):
        hidden = self.embed_tokens(input_ids)
        B, S = input_ids.shape
        device, dtype = input_ids.device, hidden.dtype
        pe_cache = {
            i: _build_position_embeddings(self.config, i, B, S, device, dtype)
            for i in range(self.config.num_hidden_layers)
        }
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, pe_cache[i])
        hidden = self.norm(hidden)
        return self.lm_head(hidden).float()


def _ref_to_checkpoint_state_dict(ref: _RefForCausalLM) -> dict:
    """Convert reference weights into the on-disk checkpoint form expected by the AD load hooks.

    * norm weights stay zero-centered (the hook adds 1.0)
    * routed-expert weights are stacked into ``moe.{gate,up,down}_proj.weight``
    * everything else gets the ``model.`` prefix (lm_head stays top-level)
    """
    ref_sd = ref.state_dict()
    ckpt = {}
    num_experts = ref.config.moe_num_experts
    for k, v in ref_sd.items():
        # Stack per-expert routed weights -> moe.{proj}.weight
        import re

        m = re.match(
            r"layers\.(\d+)\.moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", k
        )
        if m:
            continue  # handled below
        if k.startswith("lm_head."):
            ckpt[k] = v
        else:
            ckpt[f"model.{k}"] = v

    for li in ref.config.moe_layers_enum:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            ws = [ref_sd[f"layers.{li}.moe.experts.{e}.{proj}.weight"] for e in range(num_experts)]
            ckpt[f"model.layers.{li}.moe.{proj}.weight"] = torch.stack(ws, 0)
    return ckpt


def test_full_model_equivalence():
    device, dtype = _device_and_dtype()
    config = _small_config()
    ref = _RefForCausalLM(config).to(device, dtype).eval()
    ad = Step3p7ForCausalLM(config).to(device, dtype).eval()

    missing, unexpected = ad.load_state_dict(_ref_to_checkpoint_state_dict(ref), strict=False)
    assert not missing, f"Missing keys: {missing[:10]}"
    assert not unexpected, f"Unexpected keys: {unexpected[:10]}"

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    pos_ids = _position_ids(B, S, device)
    with torch.no_grad():
        ref_logits = ref(input_ids, pos_ids)
        ad_out = ad(input_ids=input_ids, position_ids=pos_ids)

    assert ad_out.logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(ad_out.logits).all()
    assert_rmse_close(ad_out.logits, ref_logits, rmse_ratio_tol=0.05, msg="Full model: ")


# ---------------------------------------------------------------------------
# Tests — Export
# ---------------------------------------------------------------------------


def test_export():
    device = "cpu"
    dtype = torch.float32
    config = _small_config()
    model = Step3p7ForCausalLM(config).to(device, dtype).eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    pos_ids = _position_ids(B, S, device)
    dynamic_shapes = {
        "input_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        "position_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    }
    gm = torch_export_to_gm(
        model,
        args=(input_ids,),
        kwargs={"position_ids": pos_ids},
        dynamic_shapes=dynamic_shapes,
    )
    with torch.no_grad():
        pre = model(input_ids=input_ids, position_ids=pos_ids)
        exported = gm(input_ids, position_ids=pos_ids)
    logits = exported[0] if isinstance(exported, tuple) else getattr(exported, "logits", exported)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, pre.logits, rtol=1e-3, atol=1e-3)

    # Second shape
    B2, S2 = 1, 5
    ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    pos2 = _position_ids(B2, S2, device)
    with torch.no_grad():
        out2 = gm(ids2, position_ids=pos2)
    logits2 = out2[0] if isinstance(out2, tuple) else getattr(out2, "logits", out2)
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(logits2).all()

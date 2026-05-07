# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical equivalence tests for Gemma4 AutoDeploy custom model.

Reference classes (_Ref*) are standalone PyTorch reimplementations of the
HuggingFace Gemma4 math — no transformers>=5.3 dependency required.
"""

from types import SimpleNamespace
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim
from transformers.activations import ACT2FN

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_gemma4 import (
    ADGemma4ImageProcessor,
    Gemma4ADInputProcessor,
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4MoEBlock,
    Gemma4MultimodalEmbedder,
    Gemma4RotaryEmbedding,
    Gemma4Router,
    Gemma4TextAttention,
    Gemma4TextConfig,
    Gemma4TextDecoderLayer,
    Gemma4TextMLP,
    Gemma4VisionAttention,
    Gemma4VisionConfig,
    Gemma4VisionEncoder,
    Gemma4VisionEncoderLayer,
    Gemma4VisionMLP,
    Gemma4VisionModel,
    Gemma4VisionPatchEmbedder,
    Gemma4VisionPooler,
    Gemma4VisionRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_rmse_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rmse_ratio_tol: float,
    msg: str = "",
) -> None:
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


def _small_text_config() -> Gemma4TextConfig:
    config = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        attention_k_eq_v=True,
        sliding_window=16,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        expert_intermediate_size=16,
        final_logit_softcapping=30.0,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        use_bidirectional_attention="vision",
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    return config


def _small_dense_text_config() -> Gemma4TextConfig:
    """Small config mimicking gemma-4-31B-it (dense, no MoE)."""
    config = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        attention_k_eq_v=True,
        sliding_window=16,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        enable_moe_block=False,
        num_experts=None,
        top_k_experts=None,
        expert_intermediate_size=None,
        final_logit_softcapping=30.0,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        use_bidirectional_attention="vision",
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    return config


def _position_ids(batch_size: int, seq_len: int, device: str) -> torch.Tensor:
    return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Standalone HF-faithful reference implementations (pure PyTorch)
# These mirror the HuggingFace Gemma4 math exactly, using the same
# state_dict key names, so weights can be shared between AD and reference.
# ---------------------------------------------------------------------------


class _RefRMSNorm(nn.Module):
    """HF Gemma4RMSNorm (transformers>=5.5): norm(x) * weight."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.weight is not None:
            normed = normed * self.weight.float()
        return normed.type_as(x)


def _ref_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, udim: int = 2):
    cos = cos.unsqueeze(udim)
    sin = sin.unsqueeze(udim)
    return (x * cos) + (_ref_rotate_half(x) * sin)


def _ref_repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n, s, d = x.shape
    return x[:, :, None, :, :].expand(b, n, n_rep, s, d).reshape(b, n * n_rep, s, d)


class _RefMLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _RefAttention(nn.Module):
    """HF Gemma4TextAttention reference (eager, no cache, no shared-kv)."""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.use_k_eq_v = config.attention_k_eq_v and not self.is_sliding

        self.head_dim = (
            config.global_head_dim
            if (not self.is_sliding and config.global_head_dim)
            else config.head_dim
        )
        self.num_heads = config.num_attention_heads
        num_kv_heads = (
            config.num_global_key_value_heads if self.use_k_eq_v else config.num_key_value_heads
        )
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = self.num_heads // num_kv_heads
        self.scaling = 1.0

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = (
            None
            if self.use_k_eq_v
            else nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=False)
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = _RefRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RefRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = _RefRMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        shape = (B, S, -1, self.head_dim)
        cos, sin = position_embeddings

        q = self.q_proj(hidden_states).view(shape)
        q = self.q_norm(q)
        q = _ref_apply_rotary(q, cos, sin, udim=2)
        q = q.transpose(1, 2)  # -> [B, num_heads, S, head_dim]

        k = self.k_proj(hidden_states).view(shape)
        v = self.v_proj(hidden_states).view(shape) if self.v_proj is not None else k
        k = self.k_norm(k)
        k = _ref_apply_rotary(k, cos, sin, udim=2)
        k = k.transpose(1, 2)
        v = self.v_norm(v)
        v = v.transpose(1, 2)

        # Eager attention with GQA repeat
        k = _ref_repeat_kv(k, self.num_kv_groups)
        v = _ref_repeat_kv(v, self.num_kv_groups)
        attn_w = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_w = attn_w + attention_mask
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_w, v)
        out = out.transpose(1, 2).contiguous().reshape(B, S, -1)
        return self.o_proj(out)


class _RefRouter(nn.Module):
    """HF Gemma4Router reference."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.register_buffer("root_size", torch.tensor(config.hidden_size**-0.5), persistent=False)
        self.eps = config.rms_norm_eps
        self.top_k = config.top_k_experts

    def forward(self, hidden_states: torch.Tensor):
        normed = hidden_states.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = normed.type_as(hidden_states)
        normed = (
            normed * self.root_size.to(hidden_states.dtype) * self.scale.to(hidden_states.dtype)
        )
        probs = F.softmax(self.proj(normed), dim=-1)
        topk_w, topk_i = torch.topk(probs, k=self.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        return topk_w, topk_i


class _RefMoEBlock(nn.Module):
    """HF Gemma4MoEBlock reference with fused parameter layout."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        inter = config.expert_intermediate_size
        hidden = config.hidden_size
        self.gate_up_proj = nn.Parameter(torch.zeros(self.num_experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.zeros(self.num_experts, hidden, inter))
        self.per_expert_scale = nn.Parameter(torch.ones(self.num_experts))
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = expert_mask.sum(dim=(-1, -2)).nonzero()
        for eidx in expert_hit:
            eidx = eidx[0]
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            cur = hidden_states[token_idx]
            gate, up = F.linear(cur, self.gate_up_proj[eidx]).chunk(2, dim=-1)
            cur = self.act_fn(gate) * up
            cur = F.linear(cur, self.down_proj[eidx])
            cur = cur * self.per_expert_scale[eidx]
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur.to(final.dtype))
        return final


class _RefDecoderLayer(nn.Module):
    """HF Gemma4TextDecoderLayer reference (no cache/grad-ckpt)."""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = _RefAttention(config, layer_idx)
        self.mlp = _RefMLP(config)
        self.input_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))
        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = _RefRouter(config)
            self.moe = _RefMoEBlock(config)
            self.post_feedforward_layernorm_1 = _RefRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = _RefRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = _RefRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, position_embeddings, attention_mask=attention_mask)
        h = self.post_attention_layernorm(h)
        hidden_states = residual + h

        residual = hidden_states
        if self.enable_moe_block:
            h1 = self.pre_feedforward_layernorm(hidden_states)
            h1 = self.mlp(h1)
            h1 = self.post_feedforward_layernorm_1(h1)
            h_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            topk_w, topk_i = self.router(h_flat)
            h2 = self.pre_feedforward_layernorm_2(h_flat)
            h2 = self.moe(h2, topk_i, topk_w)
            h2 = h2.reshape(hidden_states.shape)
            h2 = self.post_feedforward_layernorm_2(h2)
            hidden_states = h1 + h2
        else:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


# ---------------------------------------------------------------------------
# Weight-transfer helpers
# ---------------------------------------------------------------------------


def _build_ref_rope(config: Gemma4TextConfig, layer_type: str, device, dtype):
    """Build reference cos/sin matching AD's Gemma4RotaryEmbedding."""
    rope = Gemma4RotaryEmbedding(config, layer_type).to(device)
    return rope


def _load_ref_into_ad(ad_module: nn.Module, ref_module: nn.Module):
    """Load reference state_dict into AD module (hooks handle weight conversion)."""
    missing, unexpected = ad_module.load_state_dict(ref_module.state_dict(), strict=False)
    # v_norm buffer (non-persistent) won't be in state_dict, that's expected
    allowed_missing = {"v_norm.weight"}
    real_missing = {k for k in missing if not any(k.endswith(s) for s in allowed_missing)}
    assert not real_missing, f"Unexpected missing keys: {real_missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"


# ---------------------------------------------------------------------------
# Tests — Block equivalence
# ---------------------------------------------------------------------------


def test_mlp_equivalence():
    """MLP block: identical math, should match exactly."""
    device, dtype = _device_and_dtype()
    config = _small_text_config()

    ref = _RefMLP(config).to(device=device, dtype=dtype).eval()
    ad = Gemma4TextMLP(config).to(device=device, dtype=dtype).eval()
    ad.load_state_dict(ref.state_dict())

    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        torch.testing.assert_close(ad(x), ref(x), rtol=1e-3, atol=1e-3)


def test_attention_sliding_equivalence():
    """Sliding attention (standard GQA) matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_text_config()
    layer_idx = 0  # sliding

    ref = _RefAttention(config, layer_idx).to(device=device, dtype=dtype).eval()
    ad = Gemma4TextAttention(config, layer_idx).to(device=device, dtype=dtype).eval()
    _load_ref_into_ad(ad, ref)

    B, S = 2, 8
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    pos_ids = _position_ids(B, S, device)
    rope = _build_ref_rope(config, "sliding_attention", device, dtype)
    cos, sin = rope(x, pos_ids)

    # Build causal mask for reference (AD uses is_causal=True internally)
    causal_mask = torch.triu(
        torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        ad_out = ad(x, (cos, sin))
        ref_out = ref(x, (cos, sin), attention_mask=causal_mask)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.10, msg="Sliding attention: ")


def test_attention_full_k_eq_v_equivalence():
    """Full attention with K=V and different head_dim matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_text_config()
    layer_idx = 2  # full_attention

    ref = _RefAttention(config, layer_idx).to(device=device, dtype=dtype).eval()
    ad = Gemma4TextAttention(config, layer_idx).to(device=device, dtype=dtype).eval()
    _load_ref_into_ad(ad, ref)

    B, S = 2, 8
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    pos_ids = _position_ids(B, S, device)
    rope = _build_ref_rope(config, "full_attention", device, dtype)
    cos, sin = rope(x, pos_ids)

    causal_mask = torch.triu(
        torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        ad_out = ad(x, (cos, sin))
        ref_out = ref(x, (cos, sin), attention_mask=causal_mask)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.10, msg="Full K=V attention: ")


def test_moe_block_equivalence():
    """MoE block (router + experts) matches reference with fused weight conversion."""
    device, dtype = _device_and_dtype()
    config = _small_text_config()

    ref_router = _RefRouter(config).to(device=device, dtype=dtype).eval()
    ref_moe = _RefMoEBlock(config).to(device=device, dtype=dtype).eval()
    # Initialize MoE fused params with random values (default is zeros → all-zero output)
    nn.init.normal_(ref_moe.gate_up_proj, std=0.02)
    nn.init.normal_(ref_moe.down_proj, std=0.02)
    nn.init.uniform_(ref_moe.per_expert_scale, 0.5, 1.5)

    ad_router = Gemma4Router(config).to(device=device, dtype=dtype).eval()
    ad_moe = Gemma4MoEBlock(config).to(device=device, dtype=dtype).eval()

    # Load router weights (same structure)
    ad_router.load_state_dict(ref_router.state_dict())
    # Manually unfuse ref MoE fused weights into per-expert format
    # (The unfusing hook is on the decoder layer, not the MoE block)
    ref_sd = ref_moe.state_dict()
    gate_up = ref_sd["gate_up_proj"]  # [E, 2*I, H]
    down = ref_sd["down_proj"]  # [E, H, I]
    scale = ref_sd["per_expert_scale"]  # [E]
    inter = config.expert_intermediate_size
    ad_sd = {}
    for e in range(config.num_experts):
        ad_sd[f"experts.{e}.gate_proj.weight"] = gate_up[e, :inter, :]
        ad_sd[f"experts.{e}.up_proj.weight"] = gate_up[e, inter:, :]
        ad_sd[f"experts.{e}.down_proj.weight"] = down[e] * scale[e]
    ad_moe.load_state_dict(ad_sd)

    T = 16  # num tokens (flattened B*S)
    x = torch.randn(T, config.hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        ref_w, ref_i = ref_router(x)
        ad_w, ad_i = ad_router(x)
        # Router outputs should match exactly (same math, no custom ops)
        torch.testing.assert_close(ad_w, ref_w, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ad_i, ref_i)

        ref_out = ref_moe(x, ref_i, ref_w)
        ad_out = ad_moe(x, ad_i, ad_w)

    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.02, msg="MoE block: ")


# ---------------------------------------------------------------------------
# Tests — Layer equivalence
# ---------------------------------------------------------------------------


def test_decoder_layer_equivalence():
    """Decoder layer (sliding + full) matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_text_config()

    for layer_idx in [0, 2]:
        layer_type = config.layer_types[layer_idx]
        ref = _RefDecoderLayer(config, layer_idx).to(device=device, dtype=dtype).eval()
        ad = Gemma4TextDecoderLayer(config, layer_idx).to(device=device, dtype=dtype).eval()
        _load_ref_into_ad(ad, ref)

        B, S = 2, 8
        x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
        pos_ids = _position_ids(B, S, device)
        rope = _build_ref_rope(config, layer_type, device, dtype)
        cos, sin = rope(x, pos_ids)

        causal_mask = (
            torch.triu(torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        with torch.no_grad():
            ad_out = ad(x, (cos, sin))
            ref_out = ref(x, (cos, sin), attention_mask=causal_mask)
        assert_rmse_close(
            ad_out, ref_out, rmse_ratio_tol=0.05, msg=f"Layer {layer_idx} ({layer_type}): "
        )


# ---------------------------------------------------------------------------
# Tests — Full model equivalence
# ---------------------------------------------------------------------------


class _RefForCausalLM(nn.Module):
    """Standalone reference CausalLM for full-model equivalence testing."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.embed_scale = config.hidden_size**0.5
        self.layers = nn.ModuleList(
            [_RefDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Tie weights like AD model (tie_word_embeddings=True)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, position_ids):
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        for i, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            rope = _build_ref_rope(
                self.config, layer_type, hidden_states.device, hidden_states.dtype
            )
            cos, sin = rope(hidden_states, position_ids)
            causal_mask = (
                torch.triu(
                    torch.full(
                        (hidden_states.shape[1], hidden_states.shape[1]),
                        float("-inf"),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    ),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            hidden_states = layer(hidden_states, (cos, sin), attention_mask=causal_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return logits


def _transfer_ref_to_ad_full_model(ad_model: Gemma4ForCausalLM, ref_model: _RefForCausalLM) -> None:
    """Transfer weights from reference full model into AD ForCausalLM.

    The ref uses fused MoE weights (gate_up_proj, down_proj, per_expert_scale)
    while the AD model uses per-expert weights. The AD decoder layer's
    _unfuse_moe_weights pre-hook handles this conversion automatically when
    the fused keys are present in the state_dict passed to load_state_dict.
    """
    ref_sd = ref_model.state_dict()
    # AD ForCausalLM has flat keys (layers.0..., embed_tokens..., lm_head...)
    # matching the ref layout — no prefix remapping needed.
    missing, unexpected = ad_model.load_state_dict(ref_sd, strict=False)
    # v_norm buffers are non-persistent, expected missing
    real_missing = {m for m in missing if "v_norm" not in m}
    assert not real_missing, f"Missing keys: {real_missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"


def test_full_model_equivalence():
    """Full CausalLM logits match standalone reference with shared weights."""
    device, dtype = _device_and_dtype()
    config = _small_text_config()

    ref = _RefForCausalLM(config).to(device=device, dtype=dtype).eval()
    ad = Gemma4ForCausalLM(config).to(device=device, dtype=dtype).eval()
    _transfer_ref_to_ad_full_model(ad, ref)

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    pos_ids = _position_ids(B, S, device)

    with torch.no_grad():
        ref_logits = ref(input_ids, pos_ids)
        ad_out = ad(input_ids=input_ids, position_ids=pos_ids)

    assert ad_out.logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(ad_out.logits).all()
    assert_rmse_close(ad_out.logits, ref_logits, rmse_ratio_tol=0.05, msg="Full model: ")


def test_conditional_generation_wrapper():
    """ConditionalGeneration wrapper loads and forwards correctly."""
    device, dtype = _device_and_dtype()
    config = Gemma4Config(
        text_config=_small_text_config(),
        vision_config=Gemma4VisionConfig(hidden_size=32),
    )
    model = Gemma4ForConditionalGeneration(config).to(device=device, dtype=dtype).eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.text_config.vocab_size, (B, S), device=device)
    pos_ids = _position_ids(B, S, device)

    with torch.no_grad():
        out = model(input_ids=input_ids, position_ids=pos_ids)
    assert out.logits is not None
    assert out.logits.shape == (B, S, config.text_config.vocab_size)
    assert torch.isfinite(out.logits).all()


# ---------------------------------------------------------------------------
# Tests — Export
# ---------------------------------------------------------------------------


def test_export():
    """Model can be exported with torch.export and produces correct output."""
    device = "cpu"
    dtype = torch.float32
    config = _small_text_config()
    config.enable_moe_block = False  # MoE expert dispatch uses data-dependent ops

    model = Gemma4ForCausalLM(config).to(device=device, dtype=dtype).eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    pos_ids = _position_ids(B, S, device)

    batch_dim = Dim("batch", min=1, max=4)
    seq_dim = Dim("seq", min=1, max=64)
    dynamic_shapes = {
        "input_ids": {0: batch_dim, 1: seq_dim},
        "position_ids": {0: batch_dim, 1: seq_dim},
    }

    gm = torch_export_to_gm(
        model,
        args=(input_ids,),
        kwargs={"position_ids": pos_ids},
        dynamic_shapes=dynamic_shapes,
    )

    with torch.no_grad():
        pre_export_out = model(input_ids=input_ids, position_ids=pos_ids)
        exported_out = gm(input_ids, position_ids=pos_ids)

    logits = (
        exported_out[0]
        if isinstance(exported_out, tuple)
        else getattr(exported_out, "logits", exported_out)
    )
    assert torch.isfinite(logits).all(), "Export produced non-finite values"
    # Exported graph should produce identical output to the original model
    torch.testing.assert_close(logits, pre_export_out.logits, rtol=1e-3, atol=1e-3)

    # Test different shape
    B2, S2 = 1, 4
    ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    pos2 = _position_ids(B2, S2, device)
    with torch.no_grad():
        out2 = gm(ids2, position_ids=pos2)
    logits2 = out2[0] if isinstance(out2, tuple) else getattr(out2, "logits", out2)
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(logits2).all()


# ---------------------------------------------------------------------------
# Tests — Dense variant (gemma-4-31B-it style, no MoE)
# ---------------------------------------------------------------------------


def test_dense_decoder_layer_equivalence():
    """Dense (non-MoE) decoder layer matches reference for sliding and full attention."""
    device, dtype = _device_and_dtype()
    config = _small_dense_text_config()

    for layer_idx in [0, 2]:
        layer_type = config.layer_types[layer_idx]
        ref = _RefDecoderLayer(config, layer_idx).to(device=device, dtype=dtype).eval()
        ad = Gemma4TextDecoderLayer(config, layer_idx).to(device=device, dtype=dtype).eval()
        _load_ref_into_ad(ad, ref)

        B, S = 2, 8
        x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
        pos_ids = _position_ids(B, S, device)
        rope = _build_ref_rope(config, layer_type, device, dtype)
        cos, sin = rope(x, pos_ids)

        causal_mask = (
            torch.triu(torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        with torch.no_grad():
            ad_out = ad(x, (cos, sin))
            ref_out = ref(x, (cos, sin), attention_mask=causal_mask)
        assert_rmse_close(
            ad_out,
            ref_out,
            rmse_ratio_tol=0.05,
            msg=f"Dense layer {layer_idx} ({layer_type}): ",
        )


def test_dense_full_model_equivalence():
    """Dense CausalLM logits (no MoE) match reference."""
    device, dtype = _device_and_dtype()
    config = _small_dense_text_config()

    ref = _RefForCausalLM(config).to(device=device, dtype=dtype).eval()
    ad = Gemma4ForCausalLM(config).to(device=device, dtype=dtype).eval()
    _transfer_ref_to_ad_full_model(ad, ref)

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    pos_ids = _position_ids(B, S, device)

    with torch.no_grad():
        ref_logits = ref(input_ids, pos_ids)
        ad_out = ad(input_ids=input_ids, position_ids=pos_ids)

    assert ad_out.logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(ad_out.logits).all()
    assert_rmse_close(ad_out.logits, ref_logits, rmse_ratio_tol=0.05, msg="Dense full model: ")


def test_dense_conditional_generation_wrapper():
    """ConditionalGeneration wrapper works with dense (non-MoE) text config."""
    device, dtype = _device_and_dtype()
    config = Gemma4Config(
        text_config=_small_dense_text_config(),
        vision_config=Gemma4VisionConfig(hidden_size=32),
    )
    model = Gemma4ForConditionalGeneration(config).to(device=device, dtype=dtype).eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.text_config.vocab_size, (B, S), device=device)
    pos_ids = _position_ids(B, S, device)

    with torch.no_grad():
        out = model(input_ids=input_ids, position_ids=pos_ids)
    assert out.logits is not None
    assert out.logits.shape == (B, S, config.text_config.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_dense_export():
    """Dense model (no MoE) can be exported with torch.export."""
    device = "cpu"
    dtype = torch.float32
    config = _small_dense_text_config()

    model = Gemma4ForCausalLM(config).to(device=device, dtype=dtype).eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    pos_ids = _position_ids(B, S, device)

    batch_dim = Dim("batch", min=1, max=4)
    seq_dim = Dim("seq", min=1, max=64)
    dynamic_shapes = {
        "input_ids": {0: batch_dim, 1: seq_dim},
        "position_ids": {0: batch_dim, 1: seq_dim},
    }

    gm = torch_export_to_gm(
        model,
        args=(input_ids,),
        kwargs={"position_ids": pos_ids},
        dynamic_shapes=dynamic_shapes,
    )

    with torch.no_grad():
        pre_export_out = model(input_ids=input_ids, position_ids=pos_ids)
        exported_out = gm(input_ids, position_ids=pos_ids)

    logits = (
        exported_out[0]
        if isinstance(exported_out, tuple)
        else getattr(exported_out, "logits", exported_out)
    )
    assert torch.isfinite(logits).all(), "Dense export produced non-finite values"
    torch.testing.assert_close(logits, pre_export_out.logits, rtol=1e-3, atol=1e-3)

    # Test different shape
    B2, S2 = 1, 4
    ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    pos2 = _position_ids(B2, S2, device)
    with torch.no_grad():
        out2 = gm(ids2, position_ids=pos2)
    logits2 = out2[0] if isinstance(out2, tuple) else getattr(out2, "logits", out2)
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(logits2).all()


# ---------------------------------------------------------------------------
# Vision tower — helpers and small config
# ---------------------------------------------------------------------------


def _small_vision_config() -> Gemma4VisionConfig:
    return Gemma4VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        attention_bias=False,
        attention_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": 100.0},
        pooling_kernel_size=3,
        patch_size=4,
        position_embedding_size=64,
        standardize=False,
    )


def _make_vision_inputs(
    config: Gemma4VisionConfig, batch_size: int, num_patches: int, device: str, dtype: torch.dtype
):
    """Create synthetic pixel_values and pixel_position_ids for the vision tower.

    num_patches should be divisible by pooling_kernel_size^2.
    """
    patch_dim = 3 * config.patch_size**2
    pixel_values = torch.rand(batch_size, num_patches, patch_dim, device=device, dtype=dtype)
    # 2D position ids: (batch, num_patches, 2) with (x, y) grid coordinates
    grid_side = int(num_patches**0.5)
    pos_x = torch.arange(grid_side, device=device).unsqueeze(1).expand(grid_side, grid_side)
    pos_y = torch.arange(grid_side, device=device).unsqueeze(0).expand(grid_side, grid_side)
    positions = torch.stack([pos_x.flatten(), pos_y.flatten()], dim=-1)  # [num_patches, 2]
    pixel_position_ids = positions.unsqueeze(0).expand(batch_size, -1, -1).long()
    return pixel_values, pixel_position_ids


# ---------------------------------------------------------------------------
# Vision tower — standalone HF-faithful reference implementations
# ---------------------------------------------------------------------------


class _RefVisionMLP(nn.Module):
    """HF Gemma4VisionMLP reference — plain nn.Linear (no ClippableLinear wrapper)."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _RefVisionRotaryEmbedding(nn.Module):
    """HF Gemma4VisionRotaryEmbedding reference."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        rope_theta = config.rope_parameters["rope_theta"]
        spatial_dim = config.head_dim // 2
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        all_cos = []
        all_sin = []
        for dim_idx in range(2):
            dim_pos = position_ids[:, None, :, dim_idx].float().to(hidden_states.device)
            freqs = (inv_freq_expanded.to(hidden_states.device) @ dim_pos).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        cos = torch.cat(all_cos, dim=-1).to(dtype=hidden_states.dtype, device=hidden_states.device)
        sin = torch.cat(all_sin, dim=-1).to(dtype=hidden_states.dtype, device=hidden_states.device)
        return cos, sin


def _ref_vision_apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """HF apply_multidimensional_rope reference."""
    ndim = position_ids.shape[-1]
    num_channels = x.shape[-1]
    num_rotated_per_dim = 2 * (num_channels // (2 * ndim))
    split_sizes = [num_rotated_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    outputs = []
    for idx in range(ndim):
        c = cos_parts[idx].unsqueeze(unsqueeze_dim)
        s = sin_parts[idx].unsqueeze(unsqueeze_dim)
        outputs.append((x_parts[idx] * c) + (_ref_rotate_half(x_parts[idx]) * s))
    return torch.cat(outputs, dim=-1)


class _RefVisionAttention(nn.Module):
    """HF Gemma4VisionAttention reference (eager, no cache)."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = _RefRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RefRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = _RefRMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = hidden_states.shape
        shape = (B, S, -1, self.head_dim)
        cos, sin = position_embeddings

        q = self.q_norm(self.q_proj(hidden_states).view(shape))
        q = _ref_vision_apply_multidimensional_rope(q, cos, sin, torch.zeros_like(cos[..., :2]))
        q = q.transpose(1, 2)

        k = self.k_norm(self.k_proj(hidden_states).view(shape))
        k = _ref_vision_apply_multidimensional_rope(k, cos, sin, torch.zeros_like(cos[..., :2]))
        k = k.transpose(1, 2)

        v = self.v_norm(self.v_proj(hidden_states).view(shape))
        v = v.transpose(1, 2)

        # GQA repeat
        k = _ref_repeat_kv(k, self.num_kv_groups)
        v = _ref_repeat_kv(v, self.num_kv_groups)

        attn_w = torch.matmul(q, k.transpose(2, 3))  # scaling=1.0 for vision
        if attention_mask is not None:
            invalid = torch.finfo(attn_w.dtype).min
            attn_w = attn_w.masked_fill(attention_mask.logical_not(), invalid)
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_w, v)
        out = out.transpose(1, 2).contiguous().reshape(B, S, -1)
        return self.o_proj(out), attn_w


class _RefVisionEncoderLayer(nn.Module):
    """HF Gemma4VisionEncoderLayer reference."""

    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        del layer_idx
        self.self_attn = _RefVisionAttention(config)
        self.mlp = _RefVisionMLP(config)
        self.input_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class _RefVisionEncoder(nn.Module):
    """HF Gemma4VisionEncoder reference."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.rotary_emb = _RefVisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [_RefVisionEncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.LongTensor,
    ):
        valid = attention_mask.to(torch.bool)
        attention_mask_4d = valid[:, None, :, None] & valid[:, None, None, :]
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
            )
        return hidden_states


class _RefVisionPatchEmbedder(nn.Module):
    """HF Gemma4VisionPatchEmbedder reference."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = nn.Linear(3 * self.patch_size**2, self.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(
        self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        clamped = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(clamped, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        pos_emb = one_hot @ self.position_embedding_table
        pos_emb = pos_emb.sum(dim=1)
        return torch.where(padding_positions.unsqueeze(-1), 0.0, pos_emb)

    def forward(self, pixel_values, pixel_position_ids, padding_positions):
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        pos_emb = self._position_embeddings(pixel_position_ids, padding_positions)
        return hidden_states + pos_emb


class _RefVisionPooler(nn.Module):
    """HF Gemma4VisionPooler reference."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(self, hidden_states, pixel_position_ids, length):
        input_seq_len = hidden_states.shape[1]
        kernel_size = int((input_seq_len // length) ** 0.5)
        clamped = pixel_position_ids.clamp(min=0)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_indices = torch.div(clamped, kernel_size, rounding_mode="floor")
        kernel_indices = kernel_indices[..., 0] + (max_x // kernel_size) * kernel_indices[..., 1]
        weights = F.one_hot(kernel_indices.long(), length).float() / (kernel_size**2)
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length=None):
        if output_length is None:
            output_length = hidden_states.shape[1]
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        hidden_states *= self.root_hidden_size
        return hidden_states, padding_positions


class _RefVisionModel(nn.Module):
    """HF Gemma4VisionModel reference (full pipeline)."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = _RefVisionPatchEmbedder(config)
        self.encoder = _RefVisionEncoder(config)
        self.pooler = _RefVisionPooler(config)

    def forward(self, pixel_values, pixel_position_ids):
        pooling_kernel_size = self.config.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (pooling_kernel_size * pooling_kernel_size)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        hidden_states = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
        )
        hidden_states, pooler_mask = self.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        return hidden_states[pooler_mask]


class _RefMultimodalEmbedder(nn.Module):
    """HF Gemma4MultimodalEmbedder reference."""

    def __init__(self, vision_config: Gemma4VisionConfig, text_config: Gemma4TextConfig):
        super().__init__()
        self.eps = vision_config.rms_norm_eps
        self.embedding_projection = nn.Linear(
            vision_config.hidden_size, text_config.hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = _RefRMSNorm(
            vision_config.hidden_size, eps=self.eps, with_scale=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))


# ---------------------------------------------------------------------------
# Vision tower — weight transfer helpers
# ---------------------------------------------------------------------------


def _transfer_vision_mlp_weights(ad_mlp: Gemma4VisionMLP, ref_mlp: _RefVisionMLP):
    """Transfer weights from ref MLP (plain nn.Linear) to AD MLP (ClippableLinear)."""
    ad_mlp.gate_proj.linear.weight.data.copy_(ref_mlp.gate_proj.weight.data)
    ad_mlp.up_proj.linear.weight.data.copy_(ref_mlp.up_proj.weight.data)
    ad_mlp.down_proj.linear.weight.data.copy_(ref_mlp.down_proj.weight.data)


def _transfer_vision_attn_weights(ad_attn: Gemma4VisionAttention, ref_attn: _RefVisionAttention):
    """Transfer weights from ref attention to AD attention (ClippableLinear + canonical norms)."""
    ad_attn.q_proj.linear.weight.data.copy_(ref_attn.q_proj.weight.data)
    ad_attn.k_proj.linear.weight.data.copy_(ref_attn.k_proj.weight.data)
    ad_attn.v_proj.linear.weight.data.copy_(ref_attn.v_proj.weight.data)
    ad_attn.o_proj.linear.weight.data.copy_(ref_attn.o_proj.weight.data)
    ad_attn.q_norm.weight.data.copy_(ref_attn.q_norm.weight.data)
    ad_attn.k_norm.weight.data.copy_(ref_attn.k_norm.weight.data)
    # v_norm has no learnable scale (with_scale=False), but AD uses a buffer
    # The ref also uses with_scale=False so no weight to copy


def _transfer_vision_encoder_layer_weights(
    ad_layer: Gemma4VisionEncoderLayer, ref_layer: _RefVisionEncoderLayer
):
    """Transfer all weights for a single encoder layer."""
    _transfer_vision_attn_weights(ad_layer.self_attn, ref_layer.self_attn)
    _transfer_vision_mlp_weights(ad_layer.mlp, ref_layer.mlp)
    for norm_name in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ]:
        getattr(ad_layer, norm_name).weight.data.copy_(getattr(ref_layer, norm_name).weight.data)


def _transfer_vision_encoder_weights(
    ad_encoder: Gemma4VisionEncoder, ref_encoder: _RefVisionEncoder
):
    """Transfer encoder weights including RoPE and all layers."""
    ad_encoder.rotary_emb.inv_freq.data.copy_(ref_encoder.rotary_emb.inv_freq.data)
    for ad_layer, ref_layer in zip(ad_encoder.layers, ref_encoder.layers):
        _transfer_vision_encoder_layer_weights(ad_layer, ref_layer)


def _transfer_vision_patch_embedder_weights(
    ad_pe: Gemma4VisionPatchEmbedder, ref_pe: _RefVisionPatchEmbedder
):
    """Transfer patch embedder weights."""
    ad_pe.input_proj.weight.data.copy_(ref_pe.input_proj.weight.data)
    ad_pe.position_embedding_table.data.copy_(ref_pe.position_embedding_table.data)


def _transfer_vision_model_weights(ad_model: Gemma4VisionModel, ref_model: _RefVisionModel):
    """Transfer all vision model weights."""
    _transfer_vision_patch_embedder_weights(ad_model.patch_embedder, ref_model.patch_embedder)
    _transfer_vision_encoder_weights(ad_model.encoder, ref_model.encoder)
    # Pooler has no learnable parameters


def _transfer_multimodal_embedder_weights(
    ad_emb: Gemma4MultimodalEmbedder, ref_emb: _RefMultimodalEmbedder
):
    """Transfer multimodal embedder weights."""
    ad_emb.embedding_projection.weight.data.copy_(ref_emb.embedding_projection.weight.data)
    # Pre-projection norm has with_scale=False on both sides, no weight to copy


# ---------------------------------------------------------------------------
# Tests — Vision tower block equivalence
# ---------------------------------------------------------------------------


def test_vision_mlp_equivalence():
    """Vision MLP: identical math (SwiGLU), should match exactly."""
    device, dtype = _device_and_dtype()
    config = _small_vision_config()

    ref = _RefVisionMLP(config).to(device=device, dtype=dtype).eval()
    ad = Gemma4VisionMLP(config).to(device=device, dtype=dtype).eval()
    _transfer_vision_mlp_weights(ad, ref)

    x = torch.randn(2, 9, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        torch.testing.assert_close(ad(x), ref(x), rtol=1e-3, atol=1e-3)


def test_vision_rotary_embedding_equivalence():
    """Vision RoPE: multidimensional cos/sin should match reference exactly."""
    device, dtype = _device_and_dtype()
    config = _small_vision_config()

    ref_rope = _RefVisionRotaryEmbedding(config).to(device=device)
    ad_rope = Gemma4VisionRotaryEmbedding(config).to(device=device)

    B, S = 2, 9
    hidden = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    # 2D position ids
    grid_side = 3
    pos_x = torch.arange(grid_side, device=device).unsqueeze(1).expand(grid_side, grid_side)
    pos_y = torch.arange(grid_side, device=device).unsqueeze(0).expand(grid_side, grid_side)
    pos_ids = torch.stack([pos_x.flatten(), pos_y.flatten()], dim=-1).unsqueeze(0).expand(B, -1, -1)

    with torch.no_grad():
        ref_cos, ref_sin = ref_rope(hidden, pos_ids)
        ad_cos, ad_sin = ad_rope(hidden, pos_ids)
    torch.testing.assert_close(ad_cos, ref_cos, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ad_sin, ref_sin, rtol=1e-5, atol=1e-5)


def test_vision_patch_embedder_equivalence():
    """Patch embedder: linear projection + position embeddings."""
    device, dtype = _device_and_dtype()
    config = _small_vision_config()

    ref = _RefVisionPatchEmbedder(config).to(device=device, dtype=dtype).eval()
    ad = Gemma4VisionPatchEmbedder(config).to(device=device, dtype=dtype).eval()
    _transfer_vision_patch_embedder_weights(ad, ref)

    B, num_patches = 2, 9
    pixel_values, pixel_position_ids = _make_vision_inputs(config, B, num_patches, device, dtype)
    padding_positions = (pixel_position_ids == -1).all(dim=-1)

    with torch.no_grad():
        ref_out = ref(pixel_values, pixel_position_ids, padding_positions)
        ad_out = ad(pixel_values, pixel_position_ids, padding_positions)
    torch.testing.assert_close(ad_out, ref_out, rtol=1e-3, atol=1e-3)


def test_image_processor_pads_to_fixed_patch_budget():
    """Image processor should pad every request to the configured patch budget."""
    config = _small_vision_config()
    processor = ADGemma4ImageProcessor(
        patch_size=config.patch_size,
        max_soft_tokens=280,
        pooling_kernel_size=config.pooling_kernel_size,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )

    image_small = torch.zeros(3, 12, 12, dtype=torch.float32)
    image_large = torch.zeros(3, 12, 24, dtype=torch.float32)

    outputs = processor([image_small, image_large])

    target_patches = 280 * config.pooling_kernel_size**2
    assert outputs["pixel_values"].shape == (2, target_patches, 3 * config.patch_size**2)
    assert outputs["image_position_ids"].shape == (2, target_patches, 2)
    assert outputs["num_soft_tokens_per_image"] == [1, 2]
    assert torch.all(outputs["image_position_ids"][0, 9:] == -1)
    assert torch.all(outputs["image_position_ids"][1, 18:] == -1)
    assert torch.all(outputs["image_position_ids"][1, :18] >= 0)


def test_ad_input_processor_emits_layout_metadata_for_boi_eoi_spans():
    class _DummyBaseProcessor:
        def __init__(self):
            self.processor = SimpleNamespace(
                image_processor=lambda images, **kwargs: {
                    "num_soft_tokens_per_image": torch.tensor([260], dtype=torch.int32)
                }
            )
            self.tokenizer = SimpleNamespace(vocab_size=1024)

        def __call__(self, inputs, sampling_params):
            del inputs, sampling_params
            return [7, 255999, 258880, 258880, 258882, 9], {
                "multimodal_data": {
                    "token_type_ids": torch.tensor([0, 1, 1, 1, 1, 0], dtype=torch.int32)
                }
            }

    processor = Gemma4ADInputProcessor(
        _DummyBaseProcessor(),
        image_token_id=258880,
        boi_token_id=255999,
        eoi_token_id=258882,
    )

    token_ids, extra = processor(inputs={}, sampling_params=None)

    assert token_ids == [7, 255999, 258880, 258880, 258882, 9]
    assert processor.get_num_tokens_per_image(image=torch.zeros(3, 8, 8)) == 262
    torch.testing.assert_close(
        processor.get_mm_token_ids(), torch.tensor([258880], dtype=torch.int32)
    )
    torch.testing.assert_close(
        processor.get_mm_special_token_ids(), torch.tensor([255999, 258882], dtype=torch.int32)
    )

    multimodal_input = extra["multimodal_input"]
    assert multimodal_input.multimodal_positions == [1]
    assert multimodal_input.multimodal_lengths == [4]

    multimodal_data = extra["multimodal_data"]
    assert "token_type_ids" not in multimodal_data
    torch.testing.assert_close(
        multimodal_data["layout_metadata"]["special_token_offsets"],
        torch.tensor([0, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        multimodal_data["layout_metadata"]["item_types"],
        torch.tensor([0], dtype=torch.int32),
    )


def test_vision_pooler_equivalence():
    """Vision pooler: avg pooling + scaling."""
    device, dtype = _device_and_dtype()
    config = _small_vision_config()

    ref = _RefVisionPooler(config).to(device=device, dtype=dtype).eval()
    ad = Gemma4VisionPooler(config).to(device=device, dtype=dtype).eval()
    # Pooler has no learnable params

    B, num_patches = 2, 9  # 9 patches → pool to 1 with kernel_size=3
    hidden = torch.randn(B, num_patches, config.hidden_size, device=device, dtype=dtype)
    _, pixel_position_ids = _make_vision_inputs(config, B, num_patches, device, dtype)
    padding_positions = torch.zeros(B, num_patches, device=device, dtype=torch.bool)
    output_length = num_patches // (config.pooling_kernel_size**2)

    with torch.no_grad():
        ref_h, ref_mask = ref(hidden, pixel_position_ids, padding_positions, output_length)
        ad_h, ad_mask = ad(hidden, pixel_position_ids, padding_positions, output_length)
    torch.testing.assert_close(ad_h, ref_h, rtol=1e-3, atol=1e-3)
    assert (ad_mask == ref_mask).all()


def test_vision_attention_equivalence():
    """Vision attention: bidirectional, multidimensional RoPE, scaling=1.0."""
    device, dtype = _device_and_dtype()
    config = _small_vision_config()

    ref = _RefVisionAttention(config).to(device=device, dtype=dtype).eval()
    ad = Gemma4VisionAttention(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    _transfer_vision_attn_weights(ad, ref)

    B, S = 2, 9
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    # 2D position ids for vision
    grid_side = 3
    pos_x = torch.arange(grid_side, device=device).unsqueeze(1).expand(grid_side, grid_side)
    pos_y = torch.arange(grid_side, device=device).unsqueeze(0).expand(grid_side, grid_side)
    pos_ids = torch.stack([pos_x.flatten(), pos_y.flatten()], dim=-1).unsqueeze(0).expand(B, -1, -1)

    rope = Gemma4VisionRotaryEmbedding(config).to(device=device)
    with torch.no_grad():
        cos, sin = rope(x, pos_ids)

    # Bidirectional mask (all True)
    attn_mask = torch.ones(B, 1, S, S, device=device, dtype=torch.bool)

    with torch.no_grad():
        ad_out, _ = ad(x, (cos, sin), attention_mask=attn_mask, position_ids=pos_ids)
        ref_out, _ = ref(x, (cos, sin), attention_mask=attn_mask, position_ids=pos_ids)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.10, msg="Vision attention: ")


# ---------------------------------------------------------------------------
# Tests — Vision tower layer equivalence
# ---------------------------------------------------------------------------


def test_vision_encoder_layer_equivalence():
    """Vision encoder layer matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_vision_config()

    ref = _RefVisionEncoderLayer(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    ad = Gemma4VisionEncoderLayer(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    _transfer_vision_encoder_layer_weights(ad, ref)

    B, S = 2, 9
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    grid_side = 3
    pos_x = torch.arange(grid_side, device=device).unsqueeze(1).expand(grid_side, grid_side)
    pos_y = torch.arange(grid_side, device=device).unsqueeze(0).expand(grid_side, grid_side)
    pos_ids = torch.stack([pos_x.flatten(), pos_y.flatten()], dim=-1).unsqueeze(0).expand(B, -1, -1)

    rope = Gemma4VisionRotaryEmbedding(config).to(device=device)
    with torch.no_grad():
        cos, sin = rope(x, pos_ids)

    attn_mask = torch.ones(B, 1, S, S, device=device, dtype=torch.bool)

    with torch.no_grad():
        ad_out = ad(x, (cos, sin), attention_mask=attn_mask, position_ids=pos_ids)
        ref_out = ref(x, (cos, sin), attention_mask=attn_mask, position_ids=pos_ids)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.05, msg="Vision encoder layer: ")


# ---------------------------------------------------------------------------
# Tests — Vision tower full model equivalence
# ---------------------------------------------------------------------------


def test_vision_encoder_equivalence():
    """Full vision encoder (all layers + RoPE) matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_vision_config()

    ref = _RefVisionEncoder(config).to(device=device, dtype=dtype).eval()
    ad = Gemma4VisionEncoder(config).to(device=device, dtype=dtype).eval()
    _transfer_vision_encoder_weights(ad, ref)

    B, num_patches = 2, 9
    x = torch.randn(B, num_patches, config.hidden_size, device=device, dtype=dtype)
    grid_side = 3
    pos_x = torch.arange(grid_side, device=device).unsqueeze(1).expand(grid_side, grid_side)
    pos_y = torch.arange(grid_side, device=device).unsqueeze(0).expand(grid_side, grid_side)
    pos_ids = torch.stack([pos_x.flatten(), pos_y.flatten()], dim=-1).unsqueeze(0).expand(B, -1, -1)
    attn_mask = torch.ones(B, num_patches, device=device, dtype=torch.bool)

    with torch.no_grad():
        ad_out = ad(inputs_embeds=x, attention_mask=attn_mask, pixel_position_ids=pos_ids)
        ref_out = ref(inputs_embeds=x, attention_mask=attn_mask, pixel_position_ids=pos_ids)
    # ad_out is ModelOutput, ref_out is tensor
    ad_hidden = ad_out.last_hidden_state
    assert_rmse_close(ad_hidden, ref_out, rmse_ratio_tol=0.05, msg="Vision encoder: ")


def test_vision_model_equivalence():
    """Full vision model (embedder + encoder + pooler) matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_vision_config()

    ref = _RefVisionModel(config).to(device=device, dtype=dtype).eval()
    ad = Gemma4VisionModel(config).to(device=device, dtype=dtype).eval()
    _transfer_vision_model_weights(ad, ref)

    B, num_patches = 2, 9
    pixel_values, pixel_position_ids = _make_vision_inputs(config, B, num_patches, device, dtype)

    with torch.no_grad():
        ad_out = ad(pixel_values=pixel_values, pixel_position_ids=pixel_position_ids)
        ref_out = ref(pixel_values=pixel_values, pixel_position_ids=pixel_position_ids)
    ad_hidden = ad_out.last_hidden_state
    assert_rmse_close(ad_hidden, ref_out, rmse_ratio_tol=0.05, msg="Vision model: ")


def test_multimodal_embedder_equivalence():
    """Multimodal embedder (norm + projection) matches reference."""
    device, dtype = _device_and_dtype()
    vision_config = _small_vision_config()
    text_config = _small_text_config()

    ref = _RefMultimodalEmbedder(vision_config, text_config).to(device=device, dtype=dtype).eval()
    ad = Gemma4MultimodalEmbedder(vision_config, text_config).to(device=device, dtype=dtype).eval()
    _transfer_multimodal_embedder_weights(ad, ref)

    x = torch.randn(2, 4, vision_config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        ad_out = ad(x)
        ref_out = ref(x)
    torch.testing.assert_close(ad_out, ref_out, rtol=1e-3, atol=1e-3)


@torch.inference_mode()
def test_gemma4_text_export_uses_semantic_multimodal_mask():
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=1,
        sliding_window=4,
        layer_types=["sliding_attention"],
        enable_moe_block=False,
        final_logit_softcapping=None,
    )
    model = Gemma4ForCausalLM(config).eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).unsqueeze(0)
    mm_item_cu_seqlen = torch.tensor([0, 1], dtype=torch.int32)
    mm_item_types = torch.tensor([0], dtype=torch.int32)
    mm_token_positions = torch.tensor([1], dtype=torch.int32)
    mm_token_lengths = torch.tensor([2], dtype=torch.int32)
    mm_special_offsets_cu_seqlen = torch.tensor([0, 2], dtype=torch.int32)
    mm_special_offsets = torch.tensor([0, 1], dtype=torch.int32)

    gm = torch_export_to_gm(
        model,
        args=(),
        kwargs={
            "input_ids": input_ids,
            "position_ids": position_ids,
            "mm_item_cu_seqlen": mm_item_cu_seqlen,
            "mm_item_types": mm_item_types,
            "mm_token_positions": mm_token_positions,
            "mm_token_lengths": mm_token_lengths,
            "mm_special_offsets_cu_seqlen": mm_special_offsets_cu_seqlen,
            "mm_special_offsets": mm_special_offsets,
        },
        clone=True,
    )

    semantic_nodes = [
        node for node in gm.graph.nodes if is_op(node, torch.ops.auto_deploy.gemma4_multimodal_mask)
    ]
    attention_nodes = [
        node for node in gm.graph.nodes if is_op(node, torch.ops.auto_deploy.torch_attention)
    ]

    assert len(semantic_nodes) == 1
    assert len(attention_nodes) == 1

    attention_mask_arg = attention_nodes[0].kwargs.get("attn_mask")
    if attention_mask_arg is None and len(attention_nodes[0].args) > 3:
        attention_mask_arg = attention_nodes[0].args[3]
    assert attention_mask_arg is semantic_nodes[0]

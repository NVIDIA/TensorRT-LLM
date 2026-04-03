# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical equivalence tests for Gemma4 AutoDeploy custom model.

Reference classes (_Ref*) are standalone PyTorch reimplementations of the
HuggingFace Gemma4 math — no transformers>=5.3 dependency required.
"""

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
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4MoEBlock,
    Gemma4RotaryEmbedding,
    Gemma4Router,
    Gemma4TextAttention,
    Gemma4TextConfig,
    Gemma4TextDecoderLayer,
    Gemma4TextMLP,
    Gemma4VisionConfig,
)

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
    """Transfer weights from reference full model into AD ForCausalLM."""
    ref_sd = ref_model.state_dict()
    ad_sd = {}
    for k, v in ref_sd.items():
        if k.startswith("lm_head."):
            ad_sd[k] = v
        else:
            ad_sd[f"model.{k}"] = v
    missing, unexpected = ad_model.load_state_dict(ad_sd, strict=False)
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

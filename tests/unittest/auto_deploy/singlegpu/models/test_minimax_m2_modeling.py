# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical equivalence tests for MiniMax-M2 AutoDeploy custom model.

Reference classes (_Ref*) are standalone PyTorch reimplementations of the
HuggingFace MiniMax M2 math. The HF model uses trust_remote_code (not in
transformers natively), so we include minimal standalone reference classes.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — register ops
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2DecoderLayer,
    MiniMaxM2ForCausalLM,
    MiniMaxM2MLP,
    MiniMaxM2RotaryEmbedding,
    MiniMaxM2SparseMoeBlock,
)

# ---------------------------------------------------------------------------
# Test-only config (minimal faithful copy of HF MiniMaxM2Config)
# ---------------------------------------------------------------------------


class MiniMaxM2TestConfig(PretrainedConfig):
    """Minimal MiniMaxM2Config for testing. Mirrors HF trust_remote_code config."""

    model_type = "minimax_m2"

    def __init__(
        self,
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rotary_dim=8,
        num_experts_per_tok=2,
        num_local_experts=4,
        use_qk_norm=True,
        qk_norm_type="per_layer",
        scoring_func="sigmoid",
        use_routing_bias=True,
        attention_dropout=0.0,
        router_jitter_noise=0.0,
        router_aux_loss_coef=0.001,
        output_router_logits=False,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.scoring_func = scoring_func
        self.use_routing_bias = use_routing_bias
        self.attention_dropout = attention_dropout
        self.router_jitter_noise = router_jitter_noise
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


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


def _position_ids(batch: int, seq: int, device) -> torch.Tensor:
    return torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)


def _small_config() -> MiniMaxM2TestConfig:
    return MiniMaxM2TestConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rotary_dim=8,
        num_experts_per_tok=2,
        num_local_experts=4,
        use_qk_norm=True,
        qk_norm_type="per_layer",
    )


# ---------------------------------------------------------------------------
# Reference implementations (HF-faithful, plain PyTorch)
# ---------------------------------------------------------------------------


class _RefRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _ref_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """HF-style RoPE application in bnsd layout (unsqueeze_dim=1)."""
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
    """Reference SwiGLU MLP with w1/w2/w3 naming."""

    def __init__(self, hidden_size, ffn_dim, hidden_act):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


class _RefAttention(nn.Module):
    """Reference GQA attention with per-layer QK norm and partial RoPE (bnsd layout)."""

    def __init__(self, config: MiniMaxM2TestConfig, layer_idx: int = 0):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = _RefRMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RefRMSNorm(self.num_kv_heads * self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(hidden_shape).transpose(1, 2)  # bnsd
        k = k.view(hidden_shape).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _ref_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        k = _ref_repeat_kv(k, self.num_kv_groups)
        v = _ref_repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        causal_mask = torch.triu(
            torch.full((q_len, q_len), float("-inf"), device=q.device, dtype=q.dtype),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class _RefSparseMoeBlock(nn.Module):
    """Reference MoE with sigmoid routing and e_score_correction_bias."""

    def __init__(self, config: MiniMaxM2TestConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                _RefMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
                for _ in range(config.num_local_experts)
            ]
        )
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_flat)
        routing_weights = torch.sigmoid(router_logits.float())
        scores = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_flat.dtype)

        # Naive per-token expert dispatch
        final = torch.zeros_like(hidden_flat)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current = hidden_flat[top_x]
            out = self.experts[expert_idx](current) * top_k_weights[top_x, idx, None]
            final.index_add_(0, top_x, out.to(hidden_flat.dtype))

        return final.view(bsz, seq_len, hidden_dim)


class _RefDecoderLayer(nn.Module):
    """Reference decoder layer."""

    def __init__(self, config: MiniMaxM2TestConfig, layer_idx: int = 0):
        super().__init__()
        self.self_attn = _RefAttention(config, layer_idx)
        self.block_sparse_moe = _RefSparseMoeBlock(config)
        self.input_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _RefForCausalLM(nn.Module):
    """Reference full model."""

    def __init__(self, config: MiniMaxM2TestConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [_RefDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = _RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)


# ---------------------------------------------------------------------------
# Weight transfer helpers
# ---------------------------------------------------------------------------


def _build_rope_embeddings(
    config: MiniMaxM2TestConfig, device, dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin from AD RotaryEmbedding for a given position range."""
    rope = MiniMaxM2RotaryEmbedding(
        dim=config.rotary_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    ).to(device)
    dummy = torch.zeros(1, device=device, dtype=dtype)
    return rope, dummy


def _get_position_embeddings(
    config: MiniMaxM2TestConfig, B, S, device, dtype
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Return (ad_pos_emb, ref_pos_emb) for testing.

    AD pos_emb: (B, S, rotary_dim) — pre-sliced by position_ids, for bsnd layout.
    Ref pos_emb: (B, S, rotary_dim) — same values, for bnsd layout.
    """
    rope = MiniMaxM2RotaryEmbedding(
        dim=config.rotary_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    ).to(device)
    pos_ids = _position_ids(B, S, device)
    dummy = torch.zeros(1, device=device, dtype=dtype)
    cos, sin = rope(dummy, pos_ids)  # (B, S, rotary_dim)

    # Both AD and ref get the same cos/sin values
    return (cos, sin), (cos, sin)


def _transfer_ref_to_ad_full_model(ad_model: MiniMaxM2ForCausalLM, ref_model: _RefForCausalLM):
    """Transfer reference weights into AD model under 'model.' prefix."""
    ref_sd = ref_model.state_dict()
    ad_sd = {}
    for k, v in ref_sd.items():
        if k.startswith("lm_head."):
            ad_sd[k] = v
        else:
            ad_sd[f"model.{k}"] = v
    missing, unexpected = ad_model.load_state_dict(ad_sd, strict=False)
    assert not missing, f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"


# ---------------------------------------------------------------------------
# Tests — Block equivalence
# ---------------------------------------------------------------------------


def test_mlp_equivalence():
    """MLP block: identical math, should match exactly."""
    device, dtype = _device_and_dtype()
    config = _small_config()

    ref = _RefMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
    ref = ref.to(device=device, dtype=dtype).eval()
    ad = MiniMaxM2MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
    ad = ad.to(device=device, dtype=dtype).eval()
    ad.load_state_dict(ref.state_dict())

    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        torch.testing.assert_close(ad(x), ref(x), rtol=1e-3, atol=1e-3)


def test_attention_equivalence():
    """GQA attention with QK norm and partial RoPE matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_config()
    B, S = 2, 8

    ref = _RefAttention(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    ad = MiniMaxM2Attention(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    ad.load_state_dict(ref.state_dict())

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    ad_pos_emb, ref_pos_emb = _get_position_embeddings(config, B, S, device, dtype)

    with torch.no_grad():
        ad_out = ad(x, position_embeddings=ad_pos_emb)
        ref_out = ref(x, position_embeddings=ref_pos_emb)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.10, msg="Attention: ")


def test_moe_block_equivalence():
    """MoE block: sigmoid routing + e_score_correction_bias matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_config()

    ref = _RefSparseMoeBlock(config).to(device=device, dtype=dtype).eval()
    ad = MiniMaxM2SparseMoeBlock(config).to(device=device, dtype=dtype).eval()
    ad.load_state_dict(ref.state_dict())

    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        ad_out = ad(x)
        ref_out = ref(x)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.02, msg="MoE block: ")


# ---------------------------------------------------------------------------
# Tests — Layer equivalence
# ---------------------------------------------------------------------------


def test_decoder_layer_equivalence():
    """Full decoder layer (attention + MoE) matches reference."""
    device, dtype = _device_and_dtype()
    config = _small_config()
    B, S = 2, 8

    ref = _RefDecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    ad = MiniMaxM2DecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    ad.load_state_dict(ref.state_dict())

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    ad_pos_emb, ref_pos_emb = _get_position_embeddings(config, B, S, device, dtype)

    with torch.no_grad():
        ad_out = ad(x, position_embeddings=ad_pos_emb)
        ref_out = ref(x, position_embeddings=ref_pos_emb)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# ---------------------------------------------------------------------------
# Tests — Full model equivalence
# ---------------------------------------------------------------------------


def test_full_model_equivalence():
    """Full CausalLM logits match standalone reference with shared weights."""
    device, dtype = _device_and_dtype()
    config = _small_config()

    ref = _RefForCausalLM(config).to(device=device, dtype=dtype).eval()
    ad = MiniMaxM2ForCausalLM(config).to(device=device, dtype=dtype).eval()
    _transfer_ref_to_ad_full_model(ad, ref)

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    pos_ids = _position_ids(B, S, device)

    # Get position embeddings for reference
    ad_pos_emb, ref_pos_emb = _get_position_embeddings(config, B, S, device, dtype)

    with torch.no_grad():
        ref_logits = ref(input_ids, position_embeddings=ref_pos_emb)
        ad_out = ad(input_ids=input_ids, position_ids=pos_ids)

    assert ad_out.logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(ad_out.logits).all()
    assert_rmse_close(ad_out.logits, ref_logits, rmse_ratio_tol=0.05, msg="Full model: ")


# ---------------------------------------------------------------------------
# Tests — Export
# ---------------------------------------------------------------------------


def test_export():
    """Model can be exported with torch.export and produces correct output."""
    device = "cpu"
    dtype = torch.float32

    # Use smaller config for export (MoE expert dispatch uses data-dependent ops,
    # but torch_moe canonical op handles this for export)
    config = _small_config()
    model = MiniMaxM2ForCausalLM(config).to(device=device, dtype=dtype).eval()

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
        pre_export_out = model(input_ids=input_ids, position_ids=pos_ids)
        exported_out = gm(input_ids, position_ids=pos_ids)

    logits = (
        exported_out[0]
        if isinstance(exported_out, tuple)
        else getattr(exported_out, "logits", exported_out)
    )
    assert torch.isfinite(logits).all(), "Export produced non-finite values"
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

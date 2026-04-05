"""Tests for GLM MoE DSA custom model implementation.

This module tests the custom GLM MoE DSA model implementation which uses
auto_deploy custom ops (torch_mla, torch_moe, etc.) for export compatibility.

Since glm_moe_dsa is not yet in the installed transformers, we use standalone
HF-faithful reference classes copied from the Glm4Moe family for equivalence tests.
"""

import pytest
import torch
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch import nn
from torch.export import Dim
from transformers.activations import ACT2FN

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_glm_moe_dsa import (
    GlmMoeDsaAttention,
    GlmMoeDsaConfig,
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaForCausalLM,
    GlmMoeDsaMLP,
    GlmMoeDsaMoE,
    GlmMoeDsaRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> GlmMoeDsaConfig:
    """Create a small GLM MoE DSA config for testing."""
    return GlmMoeDsaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,  # Layer 0 dense, layers 1-2 MoE
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        # MLA params (scaled down)
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        # MoE params (scaled down)
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        moe_layer_freq=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        first_k_dense_replace=1,
        # RoPE
        rope_theta=10000.0,
        rope_parameters=None,
        rope_interleave=True,
        # Other
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
    )


def _create_moe_layer(config: GlmMoeDsaConfig) -> GlmMoeDsaMoE:
    """Create a MoE layer from config."""
    moe = GlmMoeDsaMoE(config)
    moe.gate.weight = torch.nn.Parameter(torch.randn_like(moe.gate.weight))
    return moe


# =============================================================================
# HF-faithful reference implementations for equivalence tests
# (standalone reference classes faithful to HF Glm4Moe / DeepSeek-V3 logic)
# =============================================================================


class _HFRefRMSNorm(nn.Module):
    """Reference RMSNorm matching HF implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class _HFRefMLP(nn.Module):
    """Reference MLP matching HF Glm4Moe MLP."""

    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _HFRefTopkRouter(nn.Module):
    """Reference top-k router matching HF Glm4Moe routing logic."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(
            -1, self.n_routed_experts
        ) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class _HFRefMoE(nn.Module):
    """Reference MoE matching HF Glm4Moe logic (per-expert loop)."""

    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                _HFRefMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = _HFRefTopkRouter(config)
        self.shared_experts = _HFRefMLP(
            config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def moe(self, hidden_states, topk_indices, topk_weights):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = F.one_hot(topk_indices, num_classes=len(self.experts)).permute(2, 0, 1)
        for expert_idx in range(len(self.experts)):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = self.experts[expert_idx](expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _HFRefRotaryEmbedding(nn.Module):
    """Reference rotary embedding with standard cos/sin computation."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, position_ids):
        cos = self.cos_cached[position_ids].to(x.dtype)
        sin = self.sin_cached[position_ids].to(x.dtype)
        return cos, sin


class _HFRefAttention(nn.Module):
    """Reference MLA attention matching HF logic with standard RoPE and SDPA."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                config.hidden_size, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                config.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = _HFRefRMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = _HFRefRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.softmax_scale = self.qk_head_dim ** (-0.5)
        self.rotary_emb = _HFRefRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(self, hidden_states, position_ids):
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q[..., : self.qk_nope_head_dim], q[..., self.qk_nope_head_dim :]

        kv_a_output = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = kv_a_output[..., : self.kv_lora_rank]
        k_pe = kv_a_output[..., self.kv_lora_rank :]

        compressed_kv = self.kv_a_layernorm(compressed_kv)

        # Decompress KV
        kv_b_output = self.kv_b_proj(compressed_kv)
        kv_b_output = kv_b_output.view(
            bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope = kv_b_output[..., : self.qk_nope_head_dim]
        v = kv_b_output[..., self.qk_nope_head_dim :]

        # Apply RoPE
        k_pe = k_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        q_pe = (q_pe * cos) + (_rotate_half(q_pe) * sin)
        k_pe = (k_pe * cos) + (_rotate_half(k_pe) * sin)

        # Concatenate nope + pe for Q and K
        q_full = torch.cat([q_nope, q_pe], dim=-1)
        k_full = torch.cat([k_nope, k_pe], dim=-1)

        # Transpose for attention: [B, N, S, D]
        q_full = q_full.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v = v.transpose(1, 2)

        # Standard scaled dot-product attention
        attn_weights = torch.matmul(q_full, k_full.transpose(-2, -1)) * self.softmax_scale
        causal_mask = torch.triu(
            torch.full(
                (q_len, q_len),
                float("-inf"),
                device=attn_weights.device,
                dtype=attn_weights.dtype,
            ),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_full.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        )
        return self.o_proj(attn_output)


class _HFRefDecoderLayer(nn.Module):
    """Reference decoder layer matching HF logic (with MLA)."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = _HFRefAttention(config, layer_idx=layer_idx)
        use_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        if use_moe:
            self.mlp = _HFRefMoE(config)
        else:
            self.mlp = _HFRefMLP(config)
        self.input_layernorm = _HFRefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _HFRefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_ids):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _HFRefModel(nn.Module):
    """Reference full model for equivalence testing."""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [_HFRefDecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = _HFRefRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states).float()


# =============================================================================
# Weight conversion helpers
# =============================================================================


def _copy_attn_weights_hf_to_custom(hf_attn, custom_attn):
    """Copy attention weights from HF reference to custom model.

    Handles RoPE de-interleaving for q_b_proj and kv_a_proj_with_mqa.
    """
    d = custom_attn.qk_rope_head_dim
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    qk_head_dim = custom_attn.qk_nope_head_dim + d

    if custom_attn.q_lora_rank is not None:
        custom_attn.q_a_proj.load_state_dict(hf_attn.q_a_proj.state_dict())
        custom_attn.q_a_layernorm.load_state_dict(hf_attn.q_a_layernorm.state_dict())

        # q_b_proj — de-interleave RoPE columns
        w = hf_attn.q_b_proj.weight.clone()
        w = w.view(custom_attn.num_heads, qk_head_dim, -1)
        w_nope = w[:, : custom_attn.qk_nope_head_dim, :]
        w_rope = w[:, custom_attn.qk_nope_head_dim :, :]
        w_rope = w_rope[:, perm, :]
        w = torch.cat([w_nope, w_rope], dim=1).view(-1, w.shape[-1])
        custom_attn.q_b_proj.weight = nn.Parameter(w)
    else:
        w = hf_attn.q_proj.weight.clone()
        w = w.view(custom_attn.num_heads, qk_head_dim, -1)
        w_nope = w[:, : custom_attn.qk_nope_head_dim, :]
        w_rope = w[:, custom_attn.qk_nope_head_dim :, :]
        w_rope = w_rope[:, perm, :]
        w = torch.cat([w_nope, w_rope], dim=1).view(-1, w.shape[-1])
        custom_attn.q_proj.weight = nn.Parameter(w)

    # kv_a_proj_with_mqa — de-interleave k_pe rows
    w = hf_attn.kv_a_proj_with_mqa.weight.clone()
    w_kv = w[: custom_attn.kv_lora_rank, :]
    w_pe = w[custom_attn.kv_lora_rank :, :]
    w_pe = w_pe[perm, :]
    custom_attn.kv_a_proj_with_mqa.weight = nn.Parameter(torch.cat([w_kv, w_pe], dim=0))

    if hf_attn.kv_a_proj_with_mqa.bias is not None:
        b = hf_attn.kv_a_proj_with_mqa.bias.clone()
        b_kv = b[: custom_attn.kv_lora_rank]
        b_pe = b[custom_attn.kv_lora_rank :]
        b_pe = b_pe[perm]
        custom_attn.kv_a_proj_with_mqa.bias = nn.Parameter(torch.cat([b_kv, b_pe]))

    custom_attn.kv_a_layernorm.load_state_dict(hf_attn.kv_a_layernorm.state_dict())
    custom_attn.kv_b_proj.load_state_dict(hf_attn.kv_b_proj.state_dict())
    custom_attn.o_proj.load_state_dict(hf_attn.o_proj.state_dict())


def _copy_full_model_weights_hf_to_custom(hf_model, custom_model):
    """Copy weights from HF reference model to custom model with RoPE de-interleaving."""
    custom_model.model.embed_tokens.load_state_dict(hf_model.embed_tokens.state_dict())
    custom_model.lm_head.load_state_dict(hf_model.lm_head.state_dict())
    custom_model.model.norm.load_state_dict(hf_model.norm.state_dict())

    for hf_layer, custom_layer in zip(hf_model.layers, custom_model.model.layers):
        custom_layer.input_layernorm.load_state_dict(hf_layer.input_layernorm.state_dict())
        custom_layer.post_attention_layernorm.load_state_dict(
            hf_layer.post_attention_layernorm.state_dict()
        )
        _copy_attn_weights_hf_to_custom(hf_layer.self_attn, custom_layer.self_attn)
        custom_layer.mlp.load_state_dict(hf_layer.mlp.state_dict())


# =============================================================================
# Config and structure tests
# =============================================================================


def test_glm_moe_dsa_config_registration():
    """Test that the config is properly registered."""
    config = _create_small_config()
    assert config.model_type == "glm_moe_dsa"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "n_routed_experts")
    assert hasattr(config, "kv_lora_rank")
    assert hasattr(config, "qk_rope_head_dim")
    assert hasattr(config, "moe_layer_freq")


def test_glm_moe_dsa_config_nested_rope_parameters():
    """Test that nested rope_parameters extracts rope_theta correctly."""
    config = GlmMoeDsaConfig(
        rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
    )
    assert config.rope_theta == 500000.0

    config2 = GlmMoeDsaConfig(rope_theta=10000.0, rope_parameters=None)
    assert config2.rope_theta == 10000.0


def test_glm_moe_dsa_layer_types():
    """Test that first layers use dense MLP and later layers use MoE."""
    config = _create_small_config()
    model = GlmMoeDsaForCausalLM(config)

    # Layer 0 should be dense
    layer0_mlp = model.model.layers[0].mlp
    assert type(layer0_mlp).__name__ == "GlmMoeDsaMLP"

    # Layers 1+ should be MoE
    for i in range(config.first_k_dense_replace, config.num_hidden_layers):
        layer_mlp = model.model.layers[i].mlp
        assert type(layer_mlp).__name__ == "GlmMoeDsaMoE"


def test_glm_moe_dsa_expert_structure():
    """Test that experts have correct structure for checkpoint loading."""
    config = _create_small_config()
    moe = GlmMoeDsaMoE(config)

    assert isinstance(moe.experts, torch.nn.ModuleList)
    assert len(moe.experts) == config.n_routed_experts

    state_dict = moe.state_dict()
    for key in [
        "experts.0.gate_proj.weight",
        "experts.0.up_proj.weight",
        "experts.0.down_proj.weight",
    ]:
        assert key in state_dict


# =============================================================================
# Block-level numerical equivalence tests
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_glm_moe_dsa_mlp_numerical_equivalence(B, S, dtype):
    """Test MLP produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    hf_mlp = _HFRefMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = GlmMoeDsaMLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_glm_moe_dsa_moe_numerical_equivalence(B, S, dtype):
    """Test MoE layer produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    hf_moe = _HFRefMoE(config)
    hf_moe.gate.weight = nn.Parameter(torch.randn_like(hf_moe.gate.weight))
    hf_moe.to(device=device, dtype=dtype)
    hf_moe.eval()

    custom_moe = GlmMoeDsaMoE(config)
    custom_moe.to(device=device, dtype=dtype)
    custom_moe.load_state_dict(hf_moe.state_dict())
    custom_moe.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_moe(x)
    custom_out = custom_moe(x)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.02, msg="MoE: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_glm_moe_dsa_attention_numerical_equivalence(B, S, dtype):
    """Test MLA attention produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    hf_attn = _HFRefAttention(config)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = GlmMoeDsaAttention(config)
    custom_attn.to(device=device, dtype=dtype)
    _copy_attn_weights_hf_to_custom(hf_attn, custom_attn)
    custom_attn.eval()

    rotary_emb = GlmMoeDsaRotaryEmbedding(
        config.qk_rope_head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )
    rotary_emb.to(device=device, dtype=dtype)

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    position_embeddings = rotary_emb(x)

    hf_out = hf_attn(x, position_ids)
    custom_out = custom_attn(x, position_ids, position_embeddings)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


# =============================================================================
# Layer-level numerical equivalence test
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layer_idx", [0, 1])  # 0=dense, 1=MoE
@torch.no_grad()
def test_glm_moe_dsa_decoder_layer_numerical_equivalence(B, S, dtype, layer_idx):
    """Test decoder layer produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    hf_layer = _HFRefDecoderLayer(config, layer_idx=layer_idx)
    # Initialize any uninitialized weights (e.g., router gate uses torch.empty)
    for module in hf_layer.modules():
        if isinstance(module, _HFRefTopkRouter):
            module.weight = nn.Parameter(torch.randn_like(module.weight))
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = GlmMoeDsaDecoderLayer(config, layer_idx=layer_idx)
    custom_layer.to(device=device, dtype=dtype)

    custom_layer.input_layernorm.load_state_dict(hf_layer.input_layernorm.state_dict())
    custom_layer.post_attention_layernorm.load_state_dict(
        hf_layer.post_attention_layernorm.state_dict()
    )
    _copy_attn_weights_hf_to_custom(hf_layer.self_attn, custom_layer.self_attn)
    custom_layer.mlp.load_state_dict(hf_layer.mlp.state_dict())
    custom_layer.eval()

    rotary_emb = GlmMoeDsaRotaryEmbedding(
        config.qk_rope_head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )
    rotary_emb.to(device=device, dtype=dtype)

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    position_embeddings = rotary_emb(x)

    hf_out = hf_layer(x, position_ids)
    custom_out = custom_layer(x, position_ids, position_embeddings)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# =============================================================================
# Full model numerical equivalence test
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_glm_moe_dsa_full_model_numerical_equivalence(B, S, dtype):
    """Test full model produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    hf_model = _HFRefModel(config)
    for module in hf_model.modules():
        if isinstance(module, _HFRefTopkRouter):
            module.weight = nn.Parameter(torch.randn_like(module.weight))
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = GlmMoeDsaForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    _copy_full_model_weights_hf_to_custom(hf_model, custom_model)
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_logits = hf_model(input_ids, position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(
        custom_out.logits.float(), hf_logits.float(), rmse_ratio_tol=0.05, msg="Full model: "
    )


# =============================================================================
# Export test
# =============================================================================


def test_glm_moe_dsa_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = GlmMoeDsaForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Get reference output before export
    with torch.no_grad():
        ref_output = model(input_ids=input_ids, position_ids=position_ids)

    batch_size_dynamic = Dim.DYNAMIC
    seq_len_dynamic = Dim.DYNAMIC
    dynamic_shapes = (
        {0: batch_size_dynamic, 1: seq_len_dynamic},
        {0: batch_size_dynamic, 1: seq_len_dynamic},
    )

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    move_to_device(gm, device)

    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(logits).all()

    # Compare exported output vs original model output
    assert_rmse_close(
        logits.float(), ref_output.logits.float(), rmse_ratio_tol=0.05, msg="Export: "
    )

    # Test with different shape for dynamic shapes
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(logits2).all()

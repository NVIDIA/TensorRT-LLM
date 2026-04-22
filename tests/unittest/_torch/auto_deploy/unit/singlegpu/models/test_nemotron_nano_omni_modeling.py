# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical equivalence tests for the NemotronH Nano Omni AD custom model.

Tests cover MLP, Attention, MoE, decoder layer, full model, multimodal wrapper,
and torch.export. Reference implementations are defined inline since the HF
NemotronH model depends on mamba_ssm (unavailable in standard CI).
"""

from typing import Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim
from transformers import PretrainedConfig

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — register all ops
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_nemotron_nano_omni import (
    NemotronHAttention,
    NemotronHBlock,
    NemotronHCausalLMOutput,
    NemotronHForCausalLM,
    NemotronHMamba2Mixer,
    NemotronHMLP,
    NemotronHMOE,
    NemotronHRMSNorm,
    NemotronHTopkRouter,
    NemotronNanoOmniForConditionalGeneration,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

# ---------------------------------------------------------------------------
# Utilities
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


# ---------------------------------------------------------------------------
# Minimal config classes for testing (faithful to HF NemotronHConfig)
# ---------------------------------------------------------------------------


class _NemotronHConfig(PretrainedConfig):
    """Minimal NemotronHConfig replacement for unit tests."""

    model_type = "nemotron_h_test"

    def __init__(
        self,
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        hybrid_override_pattern="ME*",
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
        mlp_hidden_act="relu2",
        attention_bias=False,
        mlp_bias=False,
        use_bias=False,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        residual_in_fp32=False,
        rescale_prenorm_residual=False,
        use_cache=False,
        max_position_embeddings=64,
        attention_dropout=0.0,
        ssm_state_size=16,
        mamba_num_heads=4,
        n_groups=2,
        mamba_head_dim=8,
        conv_kernel=4,
        chunk_size=8,
        time_step_limit=(0.0, float("inf")),
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        use_conv_bias=True,
        mamba_hidden_act="silu",
        n_routed_experts=4,
        n_shared_experts=1,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=128,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=False, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.hybrid_override_pattern = hybrid_override_pattern
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.mlp_hidden_act = mlp_hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.n_groups = n_groups
        self.mamba_head_dim = mamba_head_dim
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.use_conv_bias = use_conv_bias
        self.mamba_hidden_act = mamba_hidden_act
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob

    @property
    def layers_block_type(self):
        return [
            "mamba" if c == "M" else "attention" if c == "*" else "mlp" if c == "-" else "moe"
            for c in self.hybrid_override_pattern
        ]


class _OmniConfig(PretrainedConfig):
    """Minimal multimodal config wrapper for testing."""

    model_type = "nemotron_nano_omni_test"

    def __init__(self, llm_config=None, **kwargs):
        super().__init__(**kwargs)
        self.llm_config = llm_config


def _small_config() -> _NemotronHConfig:
    """Small NemotronH config: M(amba) + E(MoE) + *(Attention) = 3 layers."""
    return _NemotronHConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        hybrid_override_pattern="ME*",
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
        ssm_state_size=16,
        mamba_num_heads=4,
        n_groups=2,
        mamba_head_dim=8,
        conv_kernel=4,
        chunk_size=8,
        n_routed_experts=4,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=128,
        num_experts_per_tok=2,
    )


# ---------------------------------------------------------------------------
# Reference implementations (vanilla PyTorch, no AD ops)
# ---------------------------------------------------------------------------


class _RefMLP(nn.Module):
    """Reference MLP: up_proj → relu2 → down_proj."""

    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(torch.pow(F.relu(self.up_proj(x)), 2))


class _RefAttention(nn.Module):
    """Reference attention using standard SDPA (with GQA via repeat_kv)."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

    def forward(self, x):
        bsz, q_len, _ = x.size()
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # GQA repeat
        if self.num_kv_groups > 1:
            k = (
                k[:, :, None, :, :]
                .expand(-1, -1, self.num_kv_groups, -1, -1)
                .reshape(bsz, self.num_heads, q_len, self.head_dim)
            )
            v = (
                v[:, :, None, :, :]
                .expand(-1, -1, self.num_kv_groups, -1, -1)
                .reshape(bsz, self.num_heads, q_len, self.head_dim)
            )
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Tests — Block equivalence
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_mlp_equivalence():
    """MLP block: identical math → tight tolerance."""
    config = _small_config()
    device, dtype = _device_and_dtype()

    ad_mlp = NemotronHMLP(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    ref_mlp = _RefMLP(config.hidden_size, config.intermediate_size).to(device=device, dtype=dtype)
    ref_mlp.up_proj.weight.data.copy_(ad_mlp.up_proj.weight.data)
    ref_mlp.down_proj.weight.data.copy_(ad_mlp.down_proj.weight.data)

    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(ad_mlp(x), ref_mlp(x), rtol=1e-3, atol=1e-3)


@torch.no_grad()
def test_attention_equivalence():
    """Attention block: AD torch_attention vs SDPA."""
    config = _small_config()
    device, dtype = _device_and_dtype()

    ad_attn = NemotronHAttention(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    ref_attn = _RefAttention(
        config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim
    ).to(device=device, dtype=dtype)

    ref_attn.q_proj.weight.data.copy_(ad_attn.q_proj.weight.data)
    ref_attn.k_proj.weight.data.copy_(ad_attn.k_proj.weight.data)
    ref_attn.v_proj.weight.data.copy_(ad_attn.v_proj.weight.data)
    ref_attn.o_proj.weight.data.copy_(ad_attn.o_proj.weight.data)

    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    ad_out = ad_attn(x)
    ref_out = ref_attn(x)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.10, msg="Attention: ")


@torch.no_grad()
def test_rmsnorm_equivalence():
    """RMSNorm: identical math."""
    config = _small_config()
    device, dtype = _device_and_dtype()

    norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon).to(
        device=device, dtype=dtype
    )
    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    out = norm(x)

    # Manual reference
    x_f32 = x.float()
    var = x_f32.pow(2).mean(-1, keepdim=True)
    ref = (norm.weight.float() * (x_f32 * torch.rsqrt(var + norm.variance_epsilon))).to(dtype)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@torch.no_grad()
def test_router_output_shapes():
    """Router: output shapes and value constraints."""
    config = _small_config()
    device, dtype = _device_and_dtype()

    router = NemotronHTopkRouter(config).to(device=device)
    _init_router_weights(router)
    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    topk_indices, topk_weights = router(x)

    assert topk_indices.shape == (16, config.num_experts_per_tok)
    assert topk_weights.shape == (16, config.num_experts_per_tok)
    assert topk_indices.min() >= 0
    assert topk_indices.max() < config.n_routed_experts
    assert topk_weights.min() >= 0  # sigmoid-based weights are non-negative


def _init_router_weights(module):
    """Initialize router weights that are torch.empty by default."""
    for m in module.modules():
        if isinstance(m, NemotronHTopkRouter):
            nn.init.normal_(m.weight, std=0.01)


@torch.no_grad()
def test_moe_equivalence():
    """MoE block: AD torch_moe vs manual expert dispatch with identical weights."""
    config = _small_config()
    device, dtype = _device_and_dtype()

    moe = NemotronHMOE(config, layer_idx=0).to(device=device, dtype=dtype)
    _init_router_weights(moe)
    moe.eval()
    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    ad_out = moe(x)

    # Manual reference: dispatch each expert individually
    topk_indices, topk_weights = moe.gate(x)
    x_flat = x.view(-1, config.hidden_size)
    ref_routed = torch.zeros_like(x_flat, dtype=topk_weights.dtype)
    expert_mask = F.one_hot(topk_indices, num_classes=config.n_routed_experts).permute(2, 0, 1)
    for i, expert in enumerate(moe.experts):
        token_idx, weight_idx = torch.where(expert_mask[i])
        if token_idx.numel() > 0:
            ew = topk_weights[token_idx, weight_idx]
            ref_routed.index_add_(0, token_idx, expert(x_flat[token_idx]) * ew.unsqueeze(-1))
    ref_routed = ref_routed.to(x.dtype).view_as(x)
    ref_out = moe.shared_experts(x) + ref_routed

    assert ad_out.shape == x.shape
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.02, msg="MoE: ")


@torch.no_grad()
def test_mamba_mixer_deterministic():
    """Mamba mixer: deterministic output, correct shape, and finite values."""
    config = _small_config()
    device, dtype = _device_and_dtype()

    mixer = NemotronHMamba2Mixer(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    out1 = mixer(x)
    out2 = mixer(x)

    assert out1.shape == x.shape
    assert torch.isfinite(out1).all(), "Mamba output contains NaN or Inf"
    torch.testing.assert_close(out1, out2, msg="Mamba not deterministic")


# ---------------------------------------------------------------------------
# Tests — Layer equivalence
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_decoder_layer_mamba():
    """Mamba decoder layer: compare AD layer vs manual pre-norm residual + mixer.

    No external HF reference for Mamba2 is available (requires mamba_ssm). We verify
    the pre-norm residual structure is correct by decomposing the layer manually.
    """
    config = _small_config()
    device, dtype = _device_and_dtype()
    layer = NemotronHBlock(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    assert layer.block_type == "mamba"
    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    ad_out = layer(x)

    # Manual reference: pre-norm residual using layer's own norm + mixer
    residual = x
    h = layer.norm(x.to(dtype=layer.norm.weight.dtype))
    if layer.residual_in_fp32:
        residual = residual.to(torch.float32)
    mixer_out = layer.mixer(h)
    ref_out = residual + mixer_out
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.05, msg="Mamba layer: ")


@torch.no_grad()
def test_decoder_layer_moe_equivalence():
    """MoE decoder layer: compare AD layer vs manual pre-norm residual + mixer."""
    config = _small_config()
    device, dtype = _device_and_dtype()
    layer = NemotronHBlock(config, layer_idx=1).to(device=device, dtype=dtype)
    _init_router_weights(layer)
    layer.eval()
    assert layer.block_type == "moe"
    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    ad_out = layer(x)

    # Manual reference: pre-norm residual block using layer's own components
    residual = x
    h = layer.norm(x.to(dtype=layer.norm.weight.dtype))
    if layer.residual_in_fp32:
        residual = residual.to(torch.float32)
    mixer_out = layer.mixer(h)
    ref_out = residual + mixer_out
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.05, msg="MoE layer: ")


@torch.no_grad()
def test_decoder_layer_attention_equivalence():
    """Attention decoder layer: compare AD layer vs manual pre-norm residual + attention."""
    config = _small_config()
    device, dtype = _device_and_dtype()
    layer = NemotronHBlock(config, layer_idx=2).to(device=device, dtype=dtype).eval()
    assert layer.block_type == "attention"
    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    ad_out = layer(x)

    residual = x
    h = layer.norm(x.to(dtype=layer.norm.weight.dtype))
    if layer.residual_in_fp32:
        residual = residual.to(torch.float32)
    h = layer.mixer(h)
    ref_out = residual + h

    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.05, msg="Attention layer: ")


# ---------------------------------------------------------------------------
# Tests — Full model
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_full_model_logits():
    """Full model: correct logits shape, finiteness, and wrapper↔backbone equivalence."""
    config = _small_config()
    device, dtype = _device_and_dtype()

    model = NemotronHForCausalLM(config).to(device=device, dtype=dtype).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)
    position_ids = torch.arange(8, device=device).unsqueeze(0).expand(2, -1)

    out = model(input_ids=input_ids, position_ids=position_ids)
    assert isinstance(out, NemotronHCausalLMOutput)
    assert out.logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(out.logits).all()

    # Wrapper should produce identical logits to backbone
    omni_config = _OmniConfig(config)
    wrapper = NemotronNanoOmniForConditionalGeneration(omni_config).to(device=device, dtype=dtype)
    wrapper.language_model.load_state_dict(model.state_dict())
    wrapper.eval()
    wrapper_out = wrapper(input_ids=input_ids, position_ids=position_ids)
    assert_rmse_close(
        wrapper_out.logits, out.logits, rmse_ratio_tol=0.05, msg="Wrapper vs backbone: "
    )


@torch.no_grad()
def test_full_model_deterministic():
    """Full model: same input → same output (deterministic in eval mode)."""
    config = _small_config()
    device, dtype = _device_and_dtype()

    model = NemotronHForCausalLM(config).to(device=device, dtype=dtype).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 6), device=device)
    position_ids = torch.arange(6, device=device).unsqueeze(0)

    out1 = model(input_ids=input_ids, position_ids=position_ids)
    out2 = model(input_ids=input_ids, position_ids=position_ids)
    torch.testing.assert_close(out1.logits, out2.logits)


# ---------------------------------------------------------------------------
# Tests — Multimodal wrapper
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_wrapper_forward():
    """Multimodal wrapper: text-only forward produces correct output."""
    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    device, dtype = _device_and_dtype()

    model = NemotronNanoOmniForConditionalGeneration(config).to(device=device, dtype=dtype).eval()
    input_ids = torch.randint(0, llm_config.vocab_size, (2, 8), device=device)
    position_ids = torch.arange(8, device=device).unsqueeze(0).expand(2, -1)

    out = model(input_ids=input_ids, position_ids=position_ids)
    assert isinstance(out, NemotronHCausalLMOutput)
    assert out.logits.shape == (2, 8, llm_config.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_wrapper_drop_multimodal_weights():
    """Multimodal wrapper: load hook drops vision/audio/projector weights."""
    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    model = NemotronNanoOmniForConditionalGeneration(config)

    state_dict = model.state_dict()
    # Inject fake multimodal weights
    state_dict["vision_model.fake.weight"] = torch.randn(2, 2)
    state_dict["mlp1.fake.weight"] = torch.randn(2, 2)
    state_dict["sound_encoder.fake.weight"] = torch.randn(2, 2)
    state_dict["sound_projection.fake.weight"] = torch.randn(2, 2)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert missing == []
    assert unexpected == []


def test_wrapper_text_config_alias():
    """Multimodal wrapper: config.text_config is set for TextModelExportInfo."""
    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    NemotronNanoOmniForConditionalGeneration(config)
    assert hasattr(config, "text_config")
    assert config.text_config is llm_config


def test_wrapper_get_input_embeddings():
    """Multimodal wrapper: get_input_embeddings delegates to LLM backbone."""
    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    model = NemotronNanoOmniForConditionalGeneration(config)
    emb = model.get_input_embeddings()
    assert emb is model.language_model.backbone.embeddings


def test_wrapper_requires_position_ids():
    """Multimodal wrapper: forward asserts position_ids is not None."""
    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    model = NemotronNanoOmniForConditionalGeneration(config).eval()
    input_ids = torch.randint(0, llm_config.vocab_size, (1, 4))
    with pytest.raises(AssertionError):
        model(input_ids=input_ids, position_ids=None)


# ---------------------------------------------------------------------------
# Tests — Export
# ---------------------------------------------------------------------------


def test_full_model_export():
    """Export test: torch_export_to_gm with dynamic shapes, verify finite and consistent."""
    if not torch.cuda.is_available():
        pytest.skip("Export test requires CUDA")

    device, dtype = "cuda", torch.bfloat16
    config = _small_config()
    model = NemotronHForCausalLM(config).to(device=device, dtype=dtype).eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)
    position_ids = torch.arange(8, device=device).unsqueeze(0).expand(2, -1)

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=(
            {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
            {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        ),
    )
    move_to_device(gm, device)

    with torch.inference_mode():
        eager_out = model(input_ids=input_ids, position_ids=position_ids)
        export_out = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in export_out
    assert torch.isfinite(export_out["logits"]).all()
    assert_rmse_close(export_out["logits"], eager_out.logits, rmse_ratio_tol=0.05, msg="Export: ")

    # Verify dynamic shape — different batch and sequence
    input_ids_2 = torch.randint(0, config.vocab_size, (1, 5), device=device)
    position_ids_2 = torch.arange(5, device=device).unsqueeze(0)
    with torch.inference_mode():
        export_out_2 = gm(input_ids=input_ids_2, position_ids=position_ids_2)
        eager_out_2 = model(input_ids=input_ids_2, position_ids=position_ids_2)

    assert_rmse_close(
        export_out_2["logits"], eager_out_2.logits, rmse_ratio_tol=0.05, msg="Export dynamic: "
    )

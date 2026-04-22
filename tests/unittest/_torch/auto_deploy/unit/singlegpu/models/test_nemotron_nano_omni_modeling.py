# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical equivalence tests for the NemotronH Nano Omni AD custom model.

Tests cover MLP, Attention, MoE, decoder layer, full model, multimodal wrapper,
and torch.export. Reference implementations are defined inline since the HF
NemotronH model depends on mamba_ssm (unavailable in standard CI).
"""

from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
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
    NemotronNanoOmniADInputProcessor,
    NemotronNanoOmniForConditionalGeneration,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device
from tensorrt_llm.inputs.utils import VideoData

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


class _FakeLanguageModel(nn.Module):
    """Small text model stub used to validate multimodal embedding injection."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def get_output_embeddings(self):
        return self.embedding

    def set_output_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, **kwargs):
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embedding(input_ids)
        return NemotronHCausalLMOutput(logits=inputs_embeds.float())


def _make_fake_multimodal_wrapper() -> NemotronNanoOmniForConditionalGeneration:
    llm_config = _small_config()
    model = NemotronNanoOmniForConditionalGeneration(_OmniConfig(llm_config)).eval()
    model.language_model = _FakeLanguageModel(llm_config.vocab_size, llm_config.hidden_size)
    model._vision_enabled = True
    model._audio_enabled = True
    model.img_context_token_id = 3
    model.sound_context_token_id = 5
    model.video_temporal_patch_size = 1
    return model


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


@torch.no_grad()
def test_text_model_prefers_inputs_embeds_when_both_are_provided():
    """Text model should accept both inputs and treat inputs_embeds as authoritative."""
    config = _small_config()
    device, dtype = _device_and_dtype()

    model = NemotronHForCausalLM(config).to(device=device, dtype=dtype).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 4), device=device)
    position_ids = torch.arange(4, device=device).unsqueeze(0).expand(2, -1)
    inputs_embeds = model.get_input_embeddings()(input_ids).detach().clone() + 0.125

    out_both = model(input_ids=input_ids, inputs_embeds=inputs_embeds, position_ids=position_ids)
    out_embeds_only = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
    out_ids_only = model(input_ids=input_ids, position_ids=position_ids)

    torch.testing.assert_close(out_both.logits, out_embeds_only.logits)
    assert not torch.allclose(out_both.logits, out_ids_only.logits)


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


@torch.no_grad()
def test_wrapper_image_embedding_injection():
    """Wrapper replaces image placeholder tokens with the provided image embeddings."""
    model = _make_fake_multimodal_wrapper()
    hidden_size = model.language_model.config.hidden_size
    image_embeds = torch.arange(1, hidden_size * 3 + 1, dtype=torch.float32).view(3, hidden_size)
    model.get_image_features = lambda pixel_values, image_num_patches: [image_embeds]

    input_ids = torch.tensor([[11, 3, 3, 3, 12]], dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    base_embeds = model.get_input_embeddings()(input_ids)

    out = model(
        input_ids=input_ids,
        position_ids=position_ids,
        pixel_values=torch.zeros(1, 3, 4, 4),
        image_num_patches=torch.tensor([1], dtype=torch.int32),
    )

    torch.testing.assert_close(out.logits[0, 1:4], image_embeds)
    torch.testing.assert_close(out.logits[0, 0], base_embeds[0, 0].float())
    torch.testing.assert_close(out.logits[0, 4], base_embeds[0, 4].float())


@torch.no_grad()
def test_wrapper_audio_embedding_injection():
    """Wrapper replaces sound placeholder tokens with the provided audio embeddings."""
    model = _make_fake_multimodal_wrapper()
    hidden_size = model.language_model.config.hidden_size
    audio_embeds = torch.arange(1, hidden_size * 2 + 1, dtype=torch.float32).view(2, hidden_size)
    model.get_audio_features = (
        lambda input_audio_features, feature_attention_mask, audio_num_clips: [audio_embeds]
    )

    input_ids = torch.tensor([[11, 5, 5, 12]], dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    base_embeds = model.get_input_embeddings()(input_ids)

    out = model(
        input_ids=input_ids,
        position_ids=position_ids,
        input_audio_features=torch.zeros(1, 4, 8),
        feature_attention_mask=torch.ones(1, 4, dtype=torch.int32),
        audio_num_clips=torch.tensor([1], dtype=torch.int32),
    )

    torch.testing.assert_close(out.logits[0, 1:3], audio_embeds)
    torch.testing.assert_close(out.logits[0, 0], base_embeds[0, 0].float())
    torch.testing.assert_close(out.logits[0, 3], base_embeds[0, 3].float())


@torch.no_grad()
def test_wrapper_chunked_multimodal_embedding_selection():
    """Wrapper selects only the visible multimodal embedding rows for a chunked batch."""
    model = _make_fake_multimodal_wrapper()
    hidden_size = model.language_model.config.hidden_size

    image_embeds = torch.arange(1, hidden_size * 2 + 1, dtype=torch.float32).view(2, hidden_size)
    first_video_embed = torch.full((1, hidden_size), -5.0)
    second_video_embed = torch.full((1, hidden_size), 7.0)
    model.get_image_features = lambda pixel_values, image_num_patches: [image_embeds]
    model.get_video_features = lambda pixel_values_videos, video_size: [
        first_video_embed,
        second_video_embed,
    ]

    input_ids = torch.tensor([20, 3, 3, 21, 12, 20, 3, 21], dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[0])
    base_embeds = model.get_input_embeddings()(input_ids)

    out = model(
        input_ids=input_ids,
        position_ids=position_ids,
        pixel_values=torch.zeros(1, 3, 4, 4),
        image_num_patches=torch.tensor([1], dtype=torch.int32),
        pixel_values_videos=torch.zeros(2, 3, 4, 4),
        video_size=torch.tensor([[2, 1, 4, 4]], dtype=torch.int32),
        batch_info_host=torch.tensor([2, 8], dtype=torch.int32),
        cu_seqlen=torch.tensor([0, 4, 8], dtype=torch.int32),
        input_pos=torch.tensor([1, 4], dtype=torch.int32),
        mm_item_cu_seqlen=torch.tensor([0, 1, 3], dtype=torch.int32),
        mm_item_types=torch.tensor([0, 1, 1], dtype=torch.int32),
        mm_token_positions=torch.tensor([1, 1, 5], dtype=torch.int32),
        mm_token_lengths=torch.tensor([4, 3, 3], dtype=torch.int32),
        mm_special_offsets_cu_seqlen=torch.tensor([0, 2, 6], dtype=torch.int32),
        mm_special_offsets=torch.tensor([0, 3, 0, 2, 3, 5], dtype=torch.int32),
    )

    torch.testing.assert_close(out.logits[1], image_embeds[0])
    torch.testing.assert_close(out.logits[2], image_embeds[1])
    torch.testing.assert_close(out.logits[6], second_video_embed[0])
    torch.testing.assert_close(out.logits[0], base_embeds[0].float())
    torch.testing.assert_close(out.logits[3], base_embeds[3].float())
    torch.testing.assert_close(out.logits[4], base_embeds[4].float())
    torch.testing.assert_close(out.logits[5], base_embeds[5].float())
    torch.testing.assert_close(out.logits[7], base_embeds[7].float())


@torch.no_grad()
def test_wrapper_chunked_audio_embedding_selection():
    """Chunk selection should slice sound embeddings using audio special-token offsets."""
    model = _make_fake_multimodal_wrapper()
    hidden_size = model.language_model.config.hidden_size

    audio_embeds = torch.arange(1, hidden_size * 2 + 1, dtype=torch.float32).view(2, hidden_size)
    selected = model._select_request_chunk_multimodal_embeds(
        req_input_pos=1,
        req_seq_len=3,
        req_mm_item_types=[2],
        req_mm_positions=[1],
        req_mm_lengths=[4],
        req_special_offsets=[0, 3],
        image_embeds_list=None,
        video_embeds_list=None,
        audio_embeds_list=[audio_embeds],
    )

    torch.testing.assert_close(selected, audio_embeds)


@torch.no_grad()
def test_wrapper_interleaved_multimodal_selection_preserves_prompt_order():
    """Chunk selection should preserve interleaved video/image/video prompt order."""
    model = _make_fake_multimodal_wrapper()
    hidden_size = model.language_model.config.hidden_size

    image_embed = torch.full((1, hidden_size), 2.5)
    first_video_embed = torch.full((1, hidden_size), -4.0)
    second_video_embed = torch.full((1, hidden_size), 8.0)

    selected = model._select_request_chunk_multimodal_embeds(
        req_input_pos=0,
        req_seq_len=9,
        req_mm_item_types=[1, 0, 1],
        req_mm_positions=[0, 3, 6],
        req_mm_lengths=[3, 3, 3],
        req_special_offsets=[0, 2, 3, 5, 6, 8],
        image_embeds_list=[image_embed],
        video_embeds_list=[first_video_embed, second_video_embed],
    )

    expected = torch.cat([first_video_embed, image_embed, second_video_embed], dim=0)
    torch.testing.assert_close(selected, expected)


def test_wrapper_drop_multimodal_weights():
    """Multimodal wrapper: load hook still drops audio-only weights."""
    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    model = NemotronNanoOmniForConditionalGeneration(config)

    state_dict = model.state_dict()
    state_dict["sound_encoder.fake.weight"] = torch.randn(2, 2)
    state_dict["sound_projection.fake.weight"] = torch.randn(2, 2)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert missing == []
    assert unexpected == []


def test_wrapper_keeps_audio_weights_when_enabled():
    """Multimodal wrapper keeps sound weights by remapping sound_projection into sound_encoder."""

    class _FakeSoundEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(2, 2)
            self.projection = nn.Linear(2, 2, bias=False)

    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    model = NemotronNanoOmniForConditionalGeneration(config)
    model.sound_encoder = _FakeSoundEncoder()
    model._audio_enabled = True

    state_dict = model.state_dict()
    state_dict["sound_encoder.encoder.weight"] = torch.full_like(
        state_dict["sound_encoder.encoder.weight"], 0.5
    )
    state_dict["sound_projection.weight"] = torch.full_like(
        state_dict["sound_encoder.projection.weight"], -1.25
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(
        model.sound_encoder.encoder.weight,
        torch.full_like(model.sound_encoder.encoder.weight, 0.5),
    )
    torch.testing.assert_close(
        model.sound_encoder.projection.weight,
        torch.full_like(model.sound_encoder.projection.weight, -1.25),
    )


def test_wrapper_keeps_vision_weights_when_enabled():
    """Multimodal wrapper keeps vision/projector weights when the vision path is enabled."""
    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    model = NemotronNanoOmniForConditionalGeneration(config)
    model.vision_model = nn.Sequential(nn.Linear(2, 2, bias=False))
    model.mlp1 = nn.Sequential(nn.Linear(2, 2, bias=False))
    model._vision_enabled = True

    state_dict = model.state_dict()
    state_dict["vision_model.0.weight"] = torch.full_like(state_dict["vision_model.0.weight"], 1.5)
    state_dict["mlp1.0.weight"] = torch.full_like(state_dict["mlp1.0.weight"], -2.0)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(
        model.vision_model[0].weight,
        torch.full_like(model.vision_model[0].weight, 1.5),
    )
    torch.testing.assert_close(
        model.mlp1[0].weight,
        torch.full_like(model.mlp1[0].weight, -2.0),
    )


def test_wrapper_remaps_radio_vision_checkpoint_keys():
    """Nemotron load hook should remap legacy RADIO checkpoint keys before loading."""

    class _FakeVisionAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = nn.Linear(2, 2)
            self.o_proj = nn.Linear(2, 2)

    class _FakeVisionMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(2, 2)
            self.down_proj = nn.Linear(2, 2)

    class _FakeVisionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _FakeVisionAttention()
            self.mlp = _FakeVisionMlp()

    class _FakeVisionBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_FakeVisionBlock()])

    class _FakeVisionModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.radio_model = nn.Module()
            self.radio_model.model = _FakeVisionBackbone()

    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    model = NemotronNanoOmniForConditionalGeneration(config)
    model.vision_model = _FakeVisionModule()
    model._vision_enabled = True

    state_dict = model.state_dict()
    renamed_pairs = (
        (
            "vision_model.radio_model.model.blocks.0.attn.qkv.weight",
            "vision_model.radio_model.model.blocks.0.attn.qkv_proj.weight",
        ),
        (
            "vision_model.radio_model.model.blocks.0.attn.qkv.bias",
            "vision_model.radio_model.model.blocks.0.attn.qkv_proj.bias",
        ),
        (
            "vision_model.radio_model.model.blocks.0.attn.proj.weight",
            "vision_model.radio_model.model.blocks.0.attn.o_proj.weight",
        ),
        (
            "vision_model.radio_model.model.blocks.0.attn.proj.bias",
            "vision_model.radio_model.model.blocks.0.attn.o_proj.bias",
        ),
        (
            "vision_model.radio_model.model.blocks.0.mlp.fc1.weight",
            "vision_model.radio_model.model.blocks.0.mlp.up_proj.weight",
        ),
        (
            "vision_model.radio_model.model.blocks.0.mlp.fc1.bias",
            "vision_model.radio_model.model.blocks.0.mlp.up_proj.bias",
        ),
        (
            "vision_model.radio_model.model.blocks.0.mlp.fc2.weight",
            "vision_model.radio_model.model.blocks.0.mlp.down_proj.weight",
        ),
        (
            "vision_model.radio_model.model.blocks.0.mlp.fc2.bias",
            "vision_model.radio_model.model.blocks.0.mlp.down_proj.bias",
        ),
    )

    for old_key, new_key in renamed_pairs:
        target = state_dict.pop(new_key)
        state_dict[old_key] = torch.full_like(target, 1.25)
    state_dict["vision_model.radio_model.input_conditioner.norm_mean"] = torch.zeros(3)
    state_dict["vision_model.radio_model.input_conditioner.norm_std"] = torch.ones(3)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(
        model.vision_model.radio_model.model.blocks[0].attn.qkv_proj.weight,
        torch.full_like(model.vision_model.radio_model.model.blocks[0].attn.qkv_proj.weight, 1.25),
    )
    torch.testing.assert_close(
        model.vision_model.radio_model.model.blocks[0].attn.o_proj.bias,
        torch.full_like(model.vision_model.radio_model.model.blocks[0].attn.o_proj.bias, 1.25),
    )
    torch.testing.assert_close(
        model.vision_model.radio_model.model.blocks[0].mlp.up_proj.weight,
        torch.full_like(model.vision_model.radio_model.model.blocks[0].mlp.up_proj.weight, 1.25),
    )
    torch.testing.assert_close(
        model.vision_model.radio_model.model.blocks[0].mlp.down_proj.bias,
        torch.full_like(model.vision_model.radio_model.model.blocks[0].mlp.down_proj.bias, 1.25),
    )


def test_wrapper_marks_radio_video_embedder_loaded_when_weight_is_present():
    """Nemotron load hook should restore RADIO's temporal-compression readiness flag."""

    class _FakePatchGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.video_embedder = nn.Linear(2, 2, bias=False)
            self._video_embedder_loaded = False

    class _FakeVisionBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_generator = _FakePatchGenerator()

    class _FakeVisionModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.radio_model = nn.Module()
            self.radio_model.model = _FakeVisionBackbone()

    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    model = NemotronNanoOmniForConditionalGeneration(config)
    model.vision_model = _FakeVisionModule()
    model._vision_enabled = True

    state_dict = model.state_dict()
    key = "vision_model.radio_model.model.patch_generator.video_embedder.weight"
    state_dict[key] = torch.full_like(state_dict[key], 2.5)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    assert missing == []
    assert unexpected == []
    assert model.vision_model.radio_model.model.patch_generator._video_embedder_loaded
    torch.testing.assert_close(
        model.vision_model.radio_model.model.patch_generator.video_embedder.weight,
        torch.full_like(
            model.vision_model.radio_model.model.patch_generator.video_embedder.weight, 2.5
        ),
    )


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


def test_wrapper_input_embeddings_survive_text_model_flattening():
    """Wrapper keeps a stable embedding handle if the inner text model loses its backbone path."""
    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    model = NemotronNanoOmniForConditionalGeneration(config)

    emb = model.get_input_embeddings()
    model.language_model.backbone = nn.Module()
    model.language_model.__dict__.pop("_input_embeddings_ref", None)
    assert model.get_input_embeddings() is emb


def test_language_model_input_embeddings_survive_backbone_flattening():
    """CausalLM keeps a stable embedding reference even if export strips the backbone wrapper."""
    config = _small_config()
    model = NemotronHForCausalLM(config)

    emb = model.get_input_embeddings()
    model.backbone = nn.Module()
    assert model.get_input_embeddings() is emb

    new_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    model.set_input_embeddings(new_emb)
    assert model.get_input_embeddings() is new_emb


def test_wrapper_requires_position_ids():
    """Multimodal wrapper: forward asserts position_ids is not None."""
    llm_config = _small_config()
    config = _OmniConfig(llm_config)
    model = NemotronNanoOmniForConditionalGeneration(config).eval()
    input_ids = torch.randint(0, llm_config.vocab_size, (1, 4))
    with pytest.raises(AssertionError):
        model(input_ids=input_ids, position_ids=None)


@torch.no_grad()
def test_wrapper_text_only_forward_preserves_input_ids_and_inputs_embeds():
    """Text-only wrapper path should materialize inputs_embeds without dropping input_ids."""

    class _CapturingLanguageModel(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.last_kwargs = None

        def get_input_embeddings(self):
            return self.embedding

        def set_input_embeddings(self, new_embeddings):
            self.embedding = new_embeddings

        def get_output_embeddings(self):
            return self.embedding

        def set_output_embeddings(self, new_embeddings):
            self.embedding = new_embeddings

        def forward(self, **kwargs):
            self.last_kwargs = kwargs
            return NemotronHCausalLMOutput(logits=kwargs["inputs_embeds"].float())

    llm_config = _small_config()
    model = NemotronNanoOmniForConditionalGeneration(_OmniConfig(llm_config)).eval()
    model.language_model = _CapturingLanguageModel(llm_config.vocab_size, llm_config.hidden_size)
    model.__dict__["_input_embeddings_ref"] = model.language_model.get_input_embeddings()
    model.__dict__["_output_embeddings_ref"] = model.language_model.get_output_embeddings()

    input_ids = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    expected_embeds = model.get_input_embeddings()(input_ids)

    out = model(input_ids=input_ids, position_ids=position_ids)

    assert model.language_model.last_kwargs is not None
    torch.testing.assert_close(model.language_model.last_kwargs["input_ids"], input_ids)
    torch.testing.assert_close(model.language_model.last_kwargs["inputs_embeds"], expected_embeds)
    torch.testing.assert_close(out.logits, expected_embeds.float())


@torch.no_grad()
def test_wrapper_get_image_features_flattens_patch_rows():
    """Image features are split by flattened token rows, not by patch batch dimension."""
    model = _make_fake_multimodal_wrapper()
    hidden_size = model.language_model.config.hidden_size
    model.num_image_token = 4
    model.extract_feature = lambda pixel_values: torch.arange(
        1, 2 * model.num_image_token * hidden_size + 1, dtype=torch.float32
    ).reshape(2, model.num_image_token, hidden_size)

    image_features = model.get_image_features(
        pixel_values=torch.zeros(2, 3, 4, 4),
        image_num_patches=torch.tensor([2], dtype=torch.int32),
    )

    assert len(image_features) == 1
    assert image_features[0].shape == (2 * model.num_image_token, hidden_size)
    torch.testing.assert_close(
        image_features[0],
        model.extract_feature(None).reshape(-1, hidden_size),
    )


@torch.no_grad()
def test_wrapper_chunked_multimodal_empty_slice_survives_graphmodule_text_model():
    """Empty visible multimodal slices should not rely on GraphModule.config."""
    model = _make_fake_multimodal_wrapper()
    hidden_size = model.language_model.config.hidden_size
    model.language_model = torch.fx.GraphModule(nn.Module(), torch.fx.Graph())

    empty_slice = model._select_request_chunk_multimodal_embeds(
        req_input_pos=10,
        req_seq_len=2,
        req_mm_item_types=[0],
        req_mm_positions=[1],
        req_mm_lengths=[4],
        req_special_offsets=[0, 3],
        image_embeds_list=[torch.ones(2, hidden_size)],
        video_embeds_list=None,
    )

    assert empty_slice.shape == (0, hidden_size)


@torch.no_grad()
def test_wrapper_multimodal_graphmodule_keeps_input_ids():
    """Graph-mode multimodal wrapper should keep tensor input_ids for the exported text model."""

    class _NeedsInputIds(nn.Module):
        def forward(self, input_ids, inputs_embeds, position_ids):
            del position_ids
            logits = input_ids.unsqueeze(-1).float() + inputs_embeds * 0
            return {"logits": logits}

    model = _make_fake_multimodal_wrapper()
    model.language_model = torch.fx.symbolic_trace(_NeedsInputIds())
    model.get_image_features = lambda pixel_values, image_num_patches: [
        torch.ones(2, model.__dict__["_language_model_hidden_size"], dtype=torch.float32)
    ]

    input_ids = torch.tensor([[11, 3, 3, 12]], dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    out = model(
        input_ids=input_ids,
        position_ids=position_ids,
        pixel_values=torch.zeros(1, 3, 4, 4),
        image_num_patches=torch.tensor([1], dtype=torch.int32),
    )

    expected = input_ids.unsqueeze(-1).expand_as(out.logits).float()
    torch.testing.assert_close(out.logits, expected)


@torch.no_grad()
def test_wrapper_multimodal_graphmodule_uses_inputs_embeds():
    """Graph-mode multimodal wrapper should forward merged inputs_embeds into the text graph."""

    class _NeedsMergedEmbeds(nn.Module):
        def forward(self, input_ids, inputs_embeds, position_ids):
            del input_ids
            del position_ids
            return {"logits": inputs_embeds.float()}

    model = _make_fake_multimodal_wrapper()
    model.language_model = torch.fx.symbolic_trace(_NeedsMergedEmbeds())
    merged_embeds = torch.full(
        (2, model.__dict__["_language_model_hidden_size"]), 3.5, dtype=torch.float32
    )
    model.get_image_features = lambda pixel_values, image_num_patches: [merged_embeds]

    input_ids = torch.tensor([[11, 3, 3, 12]], dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    out = model(
        input_ids=input_ids,
        position_ids=position_ids,
        pixel_values=torch.zeros(1, 3, 4, 4),
        image_num_patches=torch.tensor([1], dtype=torch.int32),
    )

    torch.testing.assert_close(out.logits[0, 1:3], merged_embeds)


@torch.no_grad()
def test_wrapper_mixed_modalities_with_layout_metadata():
    """Mixed image+video forward path should preserve multimodal item order with layout metadata."""
    model = _make_fake_multimodal_wrapper()
    hidden_size = model.language_model.config.hidden_size

    image_embeds = torch.arange(1, hidden_size * 2 + 1, dtype=torch.float32).view(2, hidden_size)
    video_embed = torch.full((1, hidden_size), -3.0, dtype=torch.float32)
    model.get_image_features = lambda pixel_values, image_num_patches: [image_embeds]
    model.get_video_features = lambda pixel_values_videos, video_size: [video_embed]

    input_ids = torch.tensor([20, 3, 21, 22, 3, 3, 23], dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[0])
    base_embeds = model.get_input_embeddings()(input_ids)

    out = model(
        input_ids=input_ids,
        position_ids=position_ids,
        pixel_values=torch.zeros(1, 3, 4, 4),
        image_num_patches=torch.tensor([1], dtype=torch.int32),
        pixel_values_videos=torch.zeros(1, 3, 4, 4),
        video_size=torch.tensor([[1, 1, 4, 4]], dtype=torch.int32),
        batch_info_host=torch.tensor([1, 7], dtype=torch.int32),
        cu_seqlen=torch.tensor([0, 7], dtype=torch.int32),
        input_pos=torch.tensor([0], dtype=torch.int32),
        mm_item_cu_seqlen=torch.tensor([0, 2], dtype=torch.int32),
        mm_item_types=torch.tensor([1, 0], dtype=torch.int32),
        mm_token_positions=torch.tensor([1, 4], dtype=torch.int32),
        mm_token_lengths=torch.tensor([3, 4], dtype=torch.int32),
        mm_special_offsets_cu_seqlen=torch.tensor([0, 4], dtype=torch.int32),
        mm_special_offsets=torch.tensor([0, 2, 3, 6], dtype=torch.int32),
    )

    torch.testing.assert_close(out.logits[1], video_embed[0])
    torch.testing.assert_close(out.logits[4], image_embeds[0])
    torch.testing.assert_close(out.logits[5], image_embeds[1])
    torch.testing.assert_close(out.logits[0], base_embeds[0].float())
    torch.testing.assert_close(out.logits[2], base_embeds[2].float())
    torch.testing.assert_close(out.logits[3], base_embeds[3].float())
    torch.testing.assert_close(out.logits[6], base_embeds[6].float())


@torch.no_grad()
def test_wrapper_mixed_modalities_require_layout_metadata():
    """Mixed image+video requests should fail fast without layout metadata."""
    model = _make_fake_multimodal_wrapper()
    hidden_size = model.language_model.config.hidden_size
    model.get_image_features = lambda pixel_values, image_num_patches: [
        torch.ones(1, hidden_size, dtype=torch.float32)
    ]
    model.get_video_features = lambda pixel_values_videos, video_size: [
        torch.full((1, hidden_size), -2.0, dtype=torch.float32)
    ]

    input_ids = torch.tensor([[11, 3, 3, 12]], dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    with pytest.raises(ValueError, match="requires layout metadata"):
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=torch.zeros(1, 3, 4, 4),
            image_num_patches=torch.tensor([1], dtype=torch.int32),
            pixel_values_videos=torch.zeros(1, 3, 4, 4),
            video_size=torch.tensor([[1, 1, 4, 4]], dtype=torch.int32),
        )


def test_input_processor_preserves_hf_image_tiling():
    """Image preprocessing should follow the HF image processor tile count."""

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            result = []
            i = 0
            while i < len(text):
                if text.startswith("hello ", i):
                    result.append(11)
                    i += len("hello ")
                elif text.startswith("<img>", i):
                    result.append(90)
                    i += len("<img>")
                elif text.startswith("</img>", i):
                    result.append(91)
                    i += len("</img>")
                elif text.startswith("<image>", i):
                    result.append(7)
                    i += len("<image>")
                else:
                    next_special = len(text)
                    for token in ("hello ", "<img>", "</img>", "<image>"):
                        candidate = text.find(token, i)
                        if candidate != -1:
                            next_special = min(next_special, candidate)
                    result.append(17)
                    i = next_special if next_special > i else i + 1
            return result

    class _FakeImageProcessor:
        def __init__(self):
            self.max_num_tiles = 12
            self.calls = []

        def __call__(self, images, return_tensors="pt"):
            self.calls.append(self.max_num_tiles)
            num_tiles = self.max_num_tiles
            return {
                "pixel_values": torch.zeros(num_tiles, 3, 8, 8),
                "num_patches": torch.tensor([num_tiles], dtype=torch.int32),
            }

    base_processor = SimpleNamespace(tokenizer=SimpleNamespace(tokenizer=_FakeTokenizer()))
    config = SimpleNamespace(
        torch_dtype=torch.float32,
        force_image_size=8,
        patch_size=4,
        downsample_ratio=0.5,
        img_context_token="<image>",
        img_context_token_id=7,
        video_context_token="<video>",
        video_context_token_id=8,
        img_start_token="<img>",
        img_end_token="</img>",
        llm_config=SimpleNamespace(vocab_size=256),
        vision_config=SimpleNamespace(video_temporal_patch_size=1),
    )
    image_processor = _FakeImageProcessor()
    processor = SimpleNamespace(image_processor=image_processor)
    input_processor = NemotronNanoOmniADInputProcessor(base_processor, processor, config)

    token_ids, payload = input_processor(
        {
            "prompt": "hello <image>",
            "multi_modal_data": {"image": [torch.zeros(3, 8, 8)]},
        },
        sampling_params=None,
    )

    assert image_processor.calls == [12]
    assert image_processor.max_num_tiles == 12
    assert tuple(payload["multimodal_data"]["pixel_values"].shape) == (12, 3, 8, 8)
    assert payload["multimodal_data"]["image_num_patches"].tolist() == [12]
    assert token_ids == [11, 90] + [7] * 12 + [91]


def test_input_processor_handles_messages_with_interleaved_images(tmp_path):
    """Interleaved multimodal messages should take the Nemotron fixed-res image path."""

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            result = []
            i = 0
            while i < len(text):
                if text.startswith("<img>", i):
                    result.append(90)
                    i += len("<img>")
                elif text.startswith("</img>", i):
                    result.append(91)
                    i += len("</img>")
                elif text.startswith("<image>", i):
                    result.append(7)
                    i += len("<image>")
                else:
                    next_special = len(text)
                    for token in ("<img>", "</img>", "<image>"):
                        candidate = text.find(token, i)
                        if candidate != -1:
                            next_special = min(next_special, candidate)
                    result.append(17)
                    i = next_special if next_special > i else i + 1
            return result

    class _BaseProcessor:
        def __init__(self):
            self.tokenizer = SimpleNamespace(tokenizer=_FakeTokenizer())
            self.called = False

        def __call__(self, inputs, sampling_params):
            self.called = True
            raise AssertionError("messages with images should not fall back to base processor")

    class _FakeImageProcessor:
        def __init__(self):
            self.max_num_tiles = 9
            self.calls = []

        def __call__(self, images, return_tensors="pt"):
            self.calls.append((self.max_num_tiles, len(images)))
            return {
                "pixel_values": torch.zeros(len(images), 3, 8, 8),
                "num_patches": torch.ones(len(images), dtype=torch.int32),
            }

    class _FakeProcessor:
        def __init__(self):
            self.image_processor = _FakeImageProcessor()
            self.chat_templates = []

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            self.chat_templates.append(messages)
            rendered = []
            for message in messages:
                content = message["content"]
                if isinstance(content, list):
                    image_idx = 0
                    parts = []
                    for part in content:
                        if part["type"] == "image":
                            image_idx += 1
                            parts.append(f"<image {image_idx}><image>")
                        elif part["type"] == "text":
                            parts.append(part["text"])
                    content = " ".join(parts)
                rendered.append(f"{message['role']}:{content}")
            if add_generation_prompt:
                rendered.append("assistant:")
            return "\n".join(rendered)

    image_path_1 = tmp_path / "image1.png"
    image_path_2 = tmp_path / "image2.png"
    Image.new("RGB", (8, 8), color="red").save(image_path_1)
    Image.new("RGB", (8, 8), color="blue").save(image_path_2)

    base_processor = _BaseProcessor()
    processor = _FakeProcessor()
    config = SimpleNamespace(
        torch_dtype=torch.float32,
        force_image_size=8,
        patch_size=4,
        downsample_ratio=0.5,
        img_context_token="<image>",
        img_context_token_id=7,
        video_context_token="<video>",
        video_context_token_id=8,
        img_start_token="<img>",
        img_end_token="</img>",
        llm_config=SimpleNamespace(vocab_size=256),
        vision_config=SimpleNamespace(video_temporal_patch_size=1, num_frames=8),
    )
    input_processor = NemotronNanoOmniADInputProcessor(base_processor, processor, config)

    token_ids, payload = input_processor(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path_1)},
                        {"type": "text", "text": "What do you see?"},
                        {"type": "image", "image": str(image_path_2)},
                    ],
                }
            ]
        },
        sampling_params=None,
    )

    assert not base_processor.called
    assert processor.image_processor.calls == [(9, 2)]
    assert processor.chat_templates[0][0]["content"][0]["type"] == "image"
    assert processor.chat_templates[0][0]["content"][1]["text"] == "What do you see?"
    assert tuple(payload["multimodal_data"]["pixel_values"].shape) == (2, 3, 8, 8)
    assert payload["multimodal_data"]["image_num_patches"].tolist() == [1, 1]
    assert payload["multimodal_data"]["layout_metadata"]["item_types"].tolist() == [0, 0]
    assert payload["multimodal_input"].multimodal_positions == sorted(
        payload["multimodal_input"].multimodal_positions
    )
    assert payload["multimodal_input"].multimodal_lengths == [3, 3]
    assert token_ids.count(7) == 2


def test_input_processor_handles_mixed_image_and_video_prompt():
    """Mixed image+video prompts should preserve span order in layout metadata."""

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            result = []
            i = 0
            while i < len(text):
                if text.startswith("<img>", i):
                    result.append(90)
                    i += len("<img>")
                elif text.startswith("</img>", i):
                    result.append(91)
                    i += len("</img>")
                elif text.startswith("<image>", i):
                    result.append(7)
                    i += len("<image>")
                else:
                    next_special = len(text)
                    for token in ("<img>", "</img>", "<image>"):
                        candidate = text.find(token, i)
                        if candidate != -1:
                            next_special = min(next_special, candidate)
                    result.append(17)
                    i = next_special if next_special > i else i + 1
            return result

    class _FakeImageProcessor:
        def __init__(self):
            self.max_num_tiles = 5
            self.calls = []

        def __call__(self, images, return_tensors="pt"):
            del return_tensors
            self.calls.append((self.max_num_tiles, len(images)))
            num_tiles = self.max_num_tiles
            return {
                "pixel_values": torch.zeros(len(images) * num_tiles, 3, 8, 8),
                "num_patches": torch.full((len(images),), num_tiles, dtype=torch.int32),
            }

    base_processor = SimpleNamespace(tokenizer=SimpleNamespace(tokenizer=_FakeTokenizer()))
    processor = SimpleNamespace(image_processor=_FakeImageProcessor())
    config = SimpleNamespace(
        torch_dtype=torch.float32,
        force_image_size=8,
        patch_size=4,
        downsample_ratio=0.5,
        img_context_token="<image>",
        img_context_token_id=7,
        video_context_token="<video>",
        video_context_token_id=8,
        img_start_token="<img>",
        img_end_token="</img>",
        llm_config=SimpleNamespace(vocab_size=256),
        vision_config=SimpleNamespace(video_temporal_patch_size=1, num_frames=2),
    )
    input_processor = NemotronNanoOmniADInputProcessor(base_processor, processor, config)

    token_ids, payload = input_processor(
        {
            "prompt": "compare <video> with <image>",
            "multi_modal_data": {
                "image": [torch.zeros(3, 8, 8)],
                "video": [
                    [Image.new("RGB", (8, 8), color="red"), Image.new("RGB", (8, 8), color="blue")]
                ],
            },
        },
        sampling_params=None,
    )

    assert processor.image_processor.calls == [(5, 1), (1, 2)]
    assert tuple(payload["multimodal_data"]["pixel_values_videos"].shape) == (2, 3, 8, 8)
    assert tuple(payload["multimodal_data"]["pixel_values"].shape) == (5, 3, 8, 8)
    assert payload["multimodal_data"]["image_num_patches"].tolist() == [5]
    assert payload["multimodal_data"]["layout_metadata"]["item_types"].tolist() == [1, 1, 0]
    assert payload["multimodal_input"].multimodal_lengths == [3, 3, 7]
    assert token_ids.count(7) == 7


def test_input_processor_handles_audio_prompt():
    """Audio prompts should expand sound placeholders and emit audio layout metadata."""

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            result = []
            i = 0
            while i < len(text):
                if text.startswith("<img>", i):
                    result.append(90)
                    i += len("<img>")
                elif text.startswith("</img>", i):
                    result.append(91)
                    i += len("</img>")
                elif text.startswith("<so_start>", i):
                    result.append(92)
                    i += len("<so_start>")
                elif text.startswith("<so_end>", i):
                    result.append(93)
                    i += len("<so_end>")
                elif text.startswith("<so_embedding>", i):
                    result.append(27)
                    i += len("<so_embedding>")
                else:
                    next_special = len(text)
                    for token in (
                        "<img>",
                        "</img>",
                        "<so_start>",
                        "<so_end>",
                        "<so_embedding>",
                    ):
                        candidate = text.find(token, i)
                        if candidate != -1:
                            next_special = min(next_special, candidate)
                    result.append(17)
                    i = next_special if next_special > i else i + 1
            return result

    base_processor = SimpleNamespace(tokenizer=SimpleNamespace(tokenizer=_FakeTokenizer()))
    processor = SimpleNamespace(image_processor=SimpleNamespace(max_num_tiles=1))
    sound_config = SimpleNamespace(
        num_mel_bins=8,
        sampling_rate=16000,
        subsampling_factor=8,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
    )
    config = SimpleNamespace(
        torch_dtype=torch.float32,
        force_image_size=8,
        patch_size=4,
        downsample_ratio=0.5,
        img_context_token="<image>",
        img_context_token_id=7,
        video_context_token="<video>",
        video_context_token_id=8,
        img_start_token="<img>",
        img_end_token="</img>",
        sound_context_token="<so_embedding>",
        sound_context_token_id=27,
        sound_config=sound_config,
        llm_config=SimpleNamespace(vocab_size=256),
        vision_config=SimpleNamespace(video_temporal_patch_size=1, num_frames=2),
    )
    input_processor = NemotronNanoOmniADInputProcessor(base_processor, processor, config)
    input_processor.get_num_tokens_per_audio = lambda *, audio, **kwargs: 4
    input_processor._prepare_audio_features = lambda audios: {
        "input_audio_features": torch.zeros(len(audios), 4, 8),
        "feature_attention_mask": torch.ones(len(audios), 4, dtype=torch.int32),
        "audio_num_clips": torch.ones(len(audios), dtype=torch.int32),
    }

    token_ids, payload = input_processor(
        {
            "prompt": "hear <so_embedding>",
            "multi_modal_data": {
                "audio": [(np.zeros(800, dtype=np.float32), 16000)],
            },
        },
        sampling_params=None,
    )

    assert payload["multimodal_data"]["layout_metadata"]["item_types"].tolist() == [2]
    assert payload["multimodal_data"]["layout_metadata"]["special_token_offsets"].tolist() == [0, 3]
    assert payload["multimodal_input"].multimodal_lengths == [4]
    assert payload["multimodal_input"].multimodal_positions == [1]
    assert token_ids == [17, 92, 27, 27, 93]


def test_input_processor_extracts_audio_from_video_metadata():
    """Video metadata audio should become a separate ordered audio span after video spans."""

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            result = []
            i = 0
            while i < len(text):
                if text.startswith("<img>", i):
                    result.append(90)
                    i += len("<img>")
                elif text.startswith("</img>", i):
                    result.append(91)
                    i += len("</img>")
                elif text.startswith("<so_start>", i):
                    result.append(92)
                    i += len("<so_start>")
                elif text.startswith("<so_end>", i):
                    result.append(93)
                    i += len("<so_end>")
                elif text.startswith("<image>", i):
                    result.append(7)
                    i += len("<image>")
                elif text.startswith("<video>", i):
                    result.append(8)
                    i += len("<video>")
                elif text.startswith("<so_embedding>", i):
                    result.append(27)
                    i += len("<so_embedding>")
                else:
                    next_special = len(text)
                    for token in (
                        "<img>",
                        "</img>",
                        "<so_start>",
                        "<so_end>",
                        "<image>",
                        "<video>",
                        "<so_embedding>",
                    ):
                        candidate = text.find(token, i)
                        if candidate != -1:
                            next_special = min(next_special, candidate)
                    result.append(17)
                    i = next_special if next_special > i else i + 1
            return result

    class _FakeImageProcessor:
        def __init__(self):
            self.max_num_tiles = 1

        def __call__(self, images, return_tensors="pt"):
            del return_tensors
            return {
                "pixel_values": torch.zeros(len(images), 3, 8, 8),
                "num_patches": torch.ones(len(images), dtype=torch.int32),
            }

    base_processor = SimpleNamespace(tokenizer=SimpleNamespace(tokenizer=_FakeTokenizer()))
    processor = SimpleNamespace(image_processor=_FakeImageProcessor())
    sound_config = SimpleNamespace(
        num_mel_bins=8,
        sampling_rate=16000,
        subsampling_factor=8,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
    )
    config = SimpleNamespace(
        torch_dtype=torch.float32,
        force_image_size=8,
        patch_size=4,
        downsample_ratio=0.5,
        img_context_token="<image>",
        img_context_token_id=7,
        video_context_token="<video>",
        video_context_token_id=8,
        img_start_token="<img>",
        img_end_token="</img>",
        sound_context_token="<so_embedding>",
        sound_context_token_id=27,
        sound_config=sound_config,
        llm_config=SimpleNamespace(vocab_size=256),
        vision_config=SimpleNamespace(video_temporal_patch_size=1, num_frames=2),
    )
    input_processor = NemotronNanoOmniADInputProcessor(base_processor, processor, config)
    input_processor.get_num_tokens_per_audio = lambda *, audio, **kwargs: 4
    input_processor._prepare_audio_features = lambda audios: {
        "input_audio_features": torch.zeros(len(audios), 4, 8),
        "feature_attention_mask": torch.ones(len(audios), 4, dtype=torch.int32),
        "audio_num_clips": torch.ones(len(audios), dtype=torch.int32),
    }

    video = VideoData(
        frames=[Image.new("RGB", (8, 8), color="red"), Image.new("RGB", (8, 8), color="blue")],
        metadata={
            "fps": 30,
            "frames_indices": [0, 1],
            "audio_samples": np.zeros(800, dtype=np.float32),
            "audio_sample_rate": 16000,
        },
    )

    token_ids, payload = input_processor(
        {
            "prompt": "watch <video>",
            "multi_modal_data": {"video": [video]},
        },
        sampling_params=None,
    )

    assert payload["multimodal_data"]["layout_metadata"]["item_types"].tolist() == [1, 1, 2]
    assert payload["multimodal_input"].multimodal_lengths == [3, 3, 4]
    assert "input_audio_features" in payload["multimodal_data"]
    assert token_ids.count(27) == 2


def test_input_processor_handles_messages_with_interleaved_image_and_video():
    """Mixed image+video messages should preserve multimodal span order after chat templating."""

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            result = []
            i = 0
            while i < len(text):
                if text.startswith("<img>", i):
                    result.append(90)
                    i += len("<img>")
                elif text.startswith("</img>", i):
                    result.append(91)
                    i += len("</img>")
                elif text.startswith("<image>", i):
                    result.append(7)
                    i += len("<image>")
                else:
                    next_special = len(text)
                    for token in ("<img>", "</img>", "<image>"):
                        candidate = text.find(token, i)
                        if candidate != -1:
                            next_special = min(next_special, candidate)
                    result.append(17)
                    i = next_special if next_special > i else i + 1
            return result

    class _BaseProcessor:
        def __init__(self):
            self.tokenizer = SimpleNamespace(tokenizer=_FakeTokenizer())
            self.called = False

        def __call__(self, inputs, sampling_params):
            del inputs
            del sampling_params
            self.called = True
            raise AssertionError("mixed multimodal messages should not fall back to base processor")

    class _FakeImageProcessor:
        def __init__(self):
            self.max_num_tiles = 4
            self.calls = []

        def __call__(self, images, return_tensors="pt"):
            del return_tensors
            self.calls.append((self.max_num_tiles, len(images)))
            num_tiles = self.max_num_tiles
            return {
                "pixel_values": torch.zeros(len(images) * num_tiles, 3, 8, 8),
                "num_patches": torch.full((len(images),), num_tiles, dtype=torch.int32),
            }

    class _FakeProcessor:
        def __init__(self):
            self.image_processor = _FakeImageProcessor()
            self.chat_templates = []

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            del tokenize
            self.chat_templates.append(messages)
            rendered = []
            for message in messages:
                parts = []
                image_idx = 0
                video_idx = 0
                for part in message["content"]:
                    if part["type"] == "image":
                        image_idx += 1
                        parts.append(f"<image {image_idx}><image>")
                    elif part["type"] == "video":
                        video_idx += 1
                        parts.append(f"<video {video_idx}><video>")
                    elif part["type"] == "text":
                        parts.append(part["text"])
                rendered.append(" ".join(parts))
            if add_generation_prompt:
                rendered.append("assistant:")
            return "\n".join(rendered)

    base_processor = _BaseProcessor()
    processor = _FakeProcessor()
    config = SimpleNamespace(
        torch_dtype=torch.float32,
        force_image_size=8,
        patch_size=4,
        downsample_ratio=0.5,
        img_context_token="<image>",
        img_context_token_id=7,
        video_context_token="<video>",
        video_context_token_id=8,
        img_start_token="<img>",
        img_end_token="</img>",
        llm_config=SimpleNamespace(vocab_size=256),
        vision_config=SimpleNamespace(video_temporal_patch_size=1, num_frames=2),
    )
    input_processor = NemotronNanoOmniADInputProcessor(base_processor, processor, config)

    video_frames = [Image.new("RGB", (8, 8), color="red"), Image.new("RGB", (8, 8), color="blue")]
    token_ids, payload = input_processor(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.new("RGB", (8, 8), color="green")},
                        {"type": "text", "text": "Compare this with"},
                        {"type": "video", "video": video_frames},
                    ],
                }
            ]
        },
        sampling_params=None,
    )

    assert not base_processor.called
    assert processor.image_processor.calls == [(4, 1), (1, 2)]
    assert tuple(payload["multimodal_data"]["pixel_values"].shape) == (4, 3, 8, 8)
    assert tuple(payload["multimodal_data"]["pixel_values_videos"].shape) == (2, 3, 8, 8)
    assert payload["multimodal_data"]["layout_metadata"]["item_types"].tolist() == [0, 1, 1]
    assert payload["multimodal_input"].multimodal_lengths == [6, 3, 3]
    assert token_ids.count(7) == 6


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


def test_full_model_export_with_inputs_embeds():
    """Exported text graph should preserve the inputs_embeds path for multimodal wrapper use."""
    if not torch.cuda.is_available():
        pytest.skip("Export test requires CUDA")

    device, dtype = "cuda", torch.bfloat16
    config = _small_config()
    model = NemotronHForCausalLM(config).to(device=device, dtype=dtype).eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 4), device=device)
    position_ids = torch.arange(4, device=device).unsqueeze(0).expand(2, -1)
    inputs_embeds = model.get_input_embeddings()(input_ids).detach().clone()

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
        },
        dynamic_shapes={
            "input_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
            "inputs_embeds": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
            "position_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        },
    )
    move_to_device(gm, device)

    with torch.inference_mode():
        eager_out = model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        export_out = gm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        export_out_shifted = gm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds + 0.25,
            position_ids=position_ids,
        )

    assert "logits" in export_out
    assert {"input_ids", "inputs_embeds", "position_ids"} <= {
        node.name for node in gm.graph.nodes if node.op == "placeholder"
    }
    assert_rmse_close(
        export_out["logits"], eager_out.logits, rmse_ratio_tol=0.05, msg="Export embeds: "
    )
    assert (export_out_shifted["logits"] - export_out["logits"]).abs().max().item() > 1e-3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical unit tests for the Mistral4 Eagle AutoDeploy layer implementation.

Tests every AD canonical op (torch_mla, torch_rope_with_explicit_cos_sin, torch_rmsnorm)
against a plain-PyTorch reference, then tests the full Mistral4EagleLayer and the
EagleDrafterForCausalLM export path.

Reference implementations below are minimal faithful copies of the checkpoint's modeling
semantics written in plain PyTorch (no AD canonical ops), used exclusively for numerical
comparison.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    EagleConfig,
    EagleDrafterForCausalLM,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_mistral3 import (
    Mistral4EagleLayer,
    Mistral4EagleMLA,
    Mistral4EagleMLP,
    Mistral4RMSNorm,
    Mistral4TextConfig,
)

# =============================================================================
# Helpers
# =============================================================================


def assert_rmse_close(
    actual: torch.Tensor, expected: torch.Tensor, rmse_ratio_tol: float, msg: str = ""
) -> None:
    actual = actual.float()
    expected = expected.float()
    rmse = torch.sqrt(torch.mean((actual - expected) ** 2))
    denom = torch.sqrt(torch.mean(expected**2)).clamp_min(1e-8)
    ratio = (rmse / denom).item()
    assert ratio <= rmse_ratio_tol, f"{msg}rmse_ratio={ratio:.6f} > {rmse_ratio_tol:.6f}"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _small_eagle_text_config() -> Mistral4TextConfig:
    """Tiny Mistral4TextConfig suitable for Eagle unit testing."""
    return Mistral4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_head_dim=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        max_position_embeddings=128,
        rope_parameters={
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 8.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "llama_4_scaling_beta": 0.1,
        },
        pad_token_id=0,
    )


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


# =============================================================================
# Reference implementations (plain PyTorch, no AD canonical ops)
# =============================================================================


class RefMistral4EagleRMSNorm(nn.Module):
    """Plain-PyTorch RMSNorm reference."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * hidden_states).to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RefMistral4EagleRotaryEmbedding(nn.Module):
    """Plain-PyTorch YaRN RoPE reference (same math as Mistral4YarnRotaryEmbedding)."""

    def __init__(self, config: Mistral4TextConfig):
        super().__init__()
        rs = config.rope_scaling
        self.dim = config.qk_rope_head_dim
        self.base = rs.get("rope_theta", 10000.0)
        self.scale = rs.get("factor", 1.0)
        self.beta_fast = rs.get("beta_fast", 32.0)
        self.beta_slow = rs.get("beta_slow", 1.0)
        self.mscale = rs.get("mscale", 1.0)
        self.mscale_all_dim = rs.get("mscale_all_dim", 1.0)
        self.original_max_position_embeddings = rs.get("original_max_position_embeddings", 8192)
        self.max_position_embeddings = config.max_position_embeddings
        self._build_cache()

    @staticmethod
    def _find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    @classmethod
    def _find_correction_range(cls, low_rot, high_rot, dim, base, max_position_embeddings):
        low = math.floor(cls._find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(cls._find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _get_mscale(scale, mscale):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _build_cache(self):
        dim = self.dim
        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freq_inter = 1.0 / (
            self.scale * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        low, high = self._find_correction_range(
            self.beta_fast, self.beta_slow, dim, self.base, self.original_max_position_embeddings
        )
        mask = 1.0 - torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3), 0, 1
        )
        inv_freq = freq_inter * (1 - mask) + freq_extra * mask
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        mscale = self._get_mscale(self.scale, self.mscale) / self._get_mscale(
            self.scale, self.mscale_all_dim
        )
        self.register_buffer("cos_cached", emb.cos() * mscale, persistent=False)
        self.register_buffer("sin_cached", emb.sin() * mscale, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        cos = self.cos_cached.to(dtype=x.dtype, device=x.device)[position_ids]
        sin = self.sin_cached.to(dtype=x.dtype, device=x.device)[position_ids]
        return cos, sin


class RefMistral4EagleMLP(nn.Module):
    """Plain-PyTorch SwiGLU MLP with w1/w2/w3 names matching Eagle checkpoint."""

    def __init__(self, config: Mistral4TextConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RefMistral4EagleMLA(nn.Module):
    """Plain-PyTorch MLA reference using standard SDPA.

    Implements the same computation as Mistral4EagleMLA but using:
    - F.scaled_dot_product_attention instead of torch_mla
    - Manual rotate_half instead of torch_rope_with_explicit_cos_sin
    - Inline variance norm instead of torch_rmsnorm

    Weight names match Mistral4EagleMLA / Eagle checkpoint naming.
    """

    def __init__(self, config: Mistral4TextConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)

        rope_parameters = config.rope_parameters
        mscale_all_dim = rope_parameters.get("mscale_all_dim", 0.0)
        if mscale_all_dim:
            scale = rope_parameters.get("factor", 1.0)
            yarn_mscale = RefMistral4EagleRotaryEmbedding._get_mscale(scale, mscale_all_dim)
            self.softmax_scale = self.softmax_scale * yarn_mscale * yarn_mscale

        rms_eps = config.rms_norm_eps
        self.wq_a = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_norm = RefMistral4EagleRMSNorm(self.q_lora_rank, eps=rms_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.wkv_a_with_mqa = nn.Linear(
            config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_norm = RefMistral4EagleRMSNorm(self.kv_lora_rank, eps=rms_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.rotary_emb = RefMistral4EagleRotaryEmbedding(config)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.wq_b(self.q_a_norm(self.wq_a(hidden_states)))
        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.wkv_a_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_norm(compressed_kv)
        k_pe = k_pe.view(batch_size, seq_len, 1, self.qk_rope_head_dim)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        # Apply RoPE via rotate_half (no unsqueeze needed — cos/sin are [B,S,rope_dim])
        cos_h = cos.unsqueeze(2)  # [B,S,1,rope_dim]
        sin_h = sin.unsqueeze(2)  # [B,S,1,rope_dim]
        q_pe = q_pe * cos_h + _rotate_half(q_pe) * sin_h
        k_pe = k_pe * cos_h + _rotate_half(k_pe) * sin_h

        # Absorb wkv_b: expand compressed_kv to full K and V
        kv_expanded = F.linear(compressed_kv, self.wkv_b.weight)
        kv_expanded = kv_expanded.view(
            batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q = torch.cat([q_nope, q_pe], dim=-1).permute(0, 2, 1, 3).float()
        k = (
            torch.cat([k_nope, k_pe.expand(-1, -1, self.num_heads, -1)], dim=-1)
            .permute(0, 2, 1, 3)
            .float()
        )
        v = v.permute(0, 2, 1, 3).float()

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v).permute(0, 2, 1, 3).to(hidden_states.dtype)
        output = output.reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        return self.wo(output)


class RefMistral4EagleLayer(nn.Module):
    """Plain-PyTorch Mistral4 Eagle transformer layer reference."""

    def __init__(self, config: Mistral4TextConfig, layer_idx: int, has_eagle_proj: bool = False):
        super().__init__()
        rms_eps = config.rms_norm_eps
        self.eagle_proj = (
            nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
            if has_eagle_proj
            else None
        )
        self.attention_norm = RefMistral4EagleRMSNorm(config.hidden_size, eps=rms_eps)
        self.attention = RefMistral4EagleMLA(config, layer_idx)
        self.ffn_norm = RefMistral4EagleRMSNorm(config.hidden_size, eps=rms_eps)
        self.feed_forward = RefMistral4EagleMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        if self.eagle_proj is not None:
            hidden_states = self.eagle_proj(torch.cat([inputs_embeds, hidden_states], dim=-1))
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(hidden_states, position_ids)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        return residual + hidden_states


# =============================================================================
# Weight-copy helpers
# =============================================================================


def _copy_weights(target: nn.Module, source: nn.Module) -> None:
    """Load source state dict into target (strict, same parameter names)."""
    target.load_state_dict(source.state_dict(), strict=True)


# =============================================================================
# Tests: per-op AD canonical op vs. reference
# =============================================================================


def test_mistral4_eagle_rmsnorm_equivalence():
    """Mistral4RMSNorm (torch_rmsnorm AD op) == RefMistral4EagleRMSNorm (plain PyTorch)."""
    device = _device()
    dtype = torch.bfloat16
    ad_mod = Mistral4RMSNorm(64, eps=1e-6).to(device=device, dtype=dtype)
    ref_mod = RefMistral4EagleRMSNorm(64, eps=1e-6).to(device=device, dtype=dtype)
    _copy_weights(ref_mod, ad_mod)
    x = torch.randn(2, 8, 64, device=device, dtype=dtype)
    actual = ad_mod(x)
    expected = ref_mod(x)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_mistral4_eagle_mlp_equivalence():
    """Mistral4EagleMLP (torch ops) == RefMistral4EagleMLP (plain PyTorch F.silu + linear)."""
    device = _device()
    dtype = torch.bfloat16
    config = _small_eagle_text_config()
    ad_mod = Mistral4EagleMLP(config).to(device=device, dtype=dtype)
    ref_mod = RefMistral4EagleMLP(config).to(device=device, dtype=dtype)
    _copy_weights(ref_mod, ad_mod)
    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    actual = ad_mod(x)
    expected = ref_mod(x)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_mistral4_eagle_mla_equivalence():
    """Mistral4EagleMLA (torch_mla + torch_rope AD ops) == RefMistral4EagleMLA (plain PyTorch SDPA).

    Tests: torch_mla, torch_rope_with_explicit_cos_sin, torch_rmsnorm (via q_a_norm / kv_a_norm).
    """
    if not torch.cuda.is_available():
        pytest.skip("torch_mla requires CUDA.")
    device = "cuda"
    dtype = torch.bfloat16
    config = _small_eagle_text_config()
    ad_mod = Mistral4EagleMLA(config, layer_idx=0).to(device=device, dtype=dtype)
    ref_mod = RefMistral4EagleMLA(config, layer_idx=0).to(device=device, dtype=dtype)
    _copy_weights(ref_mod, ad_mod)
    hidden_states = torch.randn(2, 6, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(6, device=device).unsqueeze(0).expand(2, -1)
    actual = ad_mod(hidden_states, position_ids)
    expected = ref_mod(hidden_states, position_ids)
    assert_rmse_close(actual, expected, rmse_ratio_tol=0.10, msg="MLA: ")


def test_mistral4_eagle_layer_no_proj_equivalence():
    """Mistral4EagleLayer (no eagle_proj, layer_idx>0) == Ref layer."""
    if not torch.cuda.is_available():
        pytest.skip("torch_mla requires CUDA.")
    device = "cuda"
    dtype = torch.bfloat16
    config = _small_eagle_text_config()
    ad_mod = Mistral4EagleLayer(config, layer_idx=1, has_eagle_proj=False).to(
        device=device, dtype=dtype
    )
    ref_mod = RefMistral4EagleLayer(config, layer_idx=1, has_eagle_proj=False).to(
        device=device, dtype=dtype
    )
    _copy_weights(ref_mod, ad_mod)
    hidden_states = torch.randn(2, 5, config.hidden_size, device=device, dtype=dtype)
    inputs_embeds = torch.randn(2, 5, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(5, device=device).unsqueeze(0).expand(2, -1)
    actual = ad_mod(hidden_states, inputs_embeds, position_ids)
    expected = ref_mod(hidden_states, inputs_embeds, position_ids)
    assert_rmse_close(actual, expected, rmse_ratio_tol=0.05, msg="EagleLayer (no proj): ")


def test_mistral4_eagle_layer_with_proj_equivalence():
    """Mistral4EagleLayer (with eagle_proj, layer_idx=0) == Ref layer."""
    if not torch.cuda.is_available():
        pytest.skip("torch_mla requires CUDA.")
    device = "cuda"
    dtype = torch.bfloat16
    config = _small_eagle_text_config()
    ad_mod = Mistral4EagleLayer(config, layer_idx=0, has_eagle_proj=True).to(
        device=device, dtype=dtype
    )
    ref_mod = RefMistral4EagleLayer(config, layer_idx=0, has_eagle_proj=True).to(
        device=device, dtype=dtype
    )
    _copy_weights(ref_mod, ad_mod)
    hidden_states = torch.randn(2, 5, config.hidden_size, device=device, dtype=dtype)
    inputs_embeds = torch.randn(2, 5, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(5, device=device).unsqueeze(0).expand(2, -1)
    actual = ad_mod(hidden_states, inputs_embeds, position_ids)
    expected = ref_mod(hidden_states, inputs_embeds, position_ids)
    assert_rmse_close(actual, expected, rmse_ratio_tol=0.05, msg="EagleLayer (with proj): ")


# =============================================================================
# Tests: FP8 dequant hooks
# =============================================================================


def _make_fp8_weight_entry(
    proj_name: str,
    shape: tuple,
    device: str,
) -> tuple[dict, torch.Tensor]:
    """Create a fake FP8 checkpoint entry (weight + scale) and the expected dequant result."""
    dequantized = torch.randn(*shape, dtype=torch.float32, device=device)
    scale = torch.tensor(0.5, dtype=torch.float32, device=device)
    quantized = (dequantized / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    state_dict = {
        f"{proj_name}.weight": quantized,
        f"{proj_name}.qscale_weight": scale,
        f"{proj_name}.qscale_act": torch.tensor(3.0, device=device),
    }
    return state_dict, (quantized.float() * scale)


def test_mistral4_eagle_mlp_fp8_dequant_hook():
    """Mistral4EagleMLP._dequantize_fp8_weights correctly dequantizes w1/w2/w3 FP8 weights."""
    device = _device()
    config = _small_eagle_text_config()
    mlp = Mistral4EagleMLP(config).to(device)

    state_dict = {}
    expected = {}
    for proj, shape in [
        ("w1", (config.intermediate_size, config.hidden_size)),
        ("w2", (config.hidden_size, config.intermediate_size)),
        ("w3", (config.intermediate_size, config.hidden_size)),
    ]:
        entry, deq = _make_fp8_weight_entry(proj, shape, device)
        state_dict.update(entry)
        expected[proj] = deq

    mlp._dequantize_fp8_weights(state_dict, "")

    for proj in ("w1", "w2", "w3"):
        assert f"{proj}.qscale_weight" not in state_dict, f"{proj}.qscale_weight not removed"
        assert f"{proj}.qscale_act" not in state_dict, f"{proj}.qscale_act not removed"
        weight = state_dict[f"{proj}.weight"]
        assert weight.dtype != torch.float8_e4m3fn, f"{proj}.weight not dequantized"
        torch.testing.assert_close(weight.float(), expected[proj], rtol=0, atol=1e-3)


def test_mistral4_eagle_mla_fp8_dequant_hooks():
    """Mistral4EagleMLA FP8 hooks dequantize wkv_b and attention projection weights."""
    device = _device()
    config = _small_eagle_text_config()
    mla = Mistral4EagleMLA(config, layer_idx=0).to(device)

    # Test wkv_b hook
    wkv_b_shape = (
        config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
        config.kv_lora_rank,
    )
    state_dict_wkv_b, expected_wkv_b = _make_fp8_weight_entry("wkv_b", wkv_b_shape, device)
    mla._dequantize_fp8_wkv_b(state_dict_wkv_b, "")
    assert "wkv_b.qscale_weight" not in state_dict_wkv_b
    assert "wkv_b.qscale_act" not in state_dict_wkv_b
    assert state_dict_wkv_b["wkv_b.weight"].dtype != torch.float8_e4m3fn
    torch.testing.assert_close(
        state_dict_wkv_b["wkv_b.weight"].float(), expected_wkv_b, rtol=0, atol=1e-3
    )

    # Test remaining projections hook
    proj_shapes = {
        "wq_a": (config.q_lora_rank, config.hidden_size),
        "wq_b": (
            config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
            config.q_lora_rank,
        ),
        "wkv_a_with_mqa": (config.kv_lora_rank + config.qk_rope_head_dim, config.hidden_size),
        "wo": (config.hidden_size, config.num_attention_heads * config.v_head_dim),
    }
    state_dict_proj = {}
    expected_proj = {}
    for proj, shape in proj_shapes.items():
        entry, deq = _make_fp8_weight_entry(proj, shape, device)
        state_dict_proj.update(entry)
        expected_proj[proj] = deq

    mla._dequantize_fp8_weights(state_dict_proj, "")
    for proj in proj_shapes:
        assert f"{proj}.qscale_weight" not in state_dict_proj
        assert f"{proj}.qscale_act" not in state_dict_proj
        assert state_dict_proj[f"{proj}.weight"].dtype != torch.float8_e4m3fn
        torch.testing.assert_close(
            state_dict_proj[f"{proj}.weight"].float(), expected_proj[proj], rtol=0, atol=1e-3
        )


def test_mistral4_eagle_layer_fp8_eagle_proj_hook():
    """Mistral4EagleLayer._dequantize_fp8_eagle_proj dequantizes eagle_proj."""
    device = _device()
    config = _small_eagle_text_config()
    layer = Mistral4EagleLayer(config, layer_idx=0, has_eagle_proj=True).to(device)

    proj_shape = (config.hidden_size, 2 * config.hidden_size)
    state_dict, expected = _make_fp8_weight_entry("eagle_proj", proj_shape, device)

    layer._dequantize_fp8_eagle_proj(state_dict, "")
    assert "eagle_proj.qscale_weight" not in state_dict
    assert "eagle_proj.qscale_act" not in state_dict
    assert state_dict["eagle_proj.weight"].dtype != torch.float8_e4m3fn
    torch.testing.assert_close(state_dict["eagle_proj.weight"].float(), expected, rtol=0, atol=1e-3)


def test_mistral4_eagle_layer_no_eagle_proj_hook_is_noop():
    """Layers without eagle_proj don't have the FP8 hook registered."""
    config = _small_eagle_text_config()
    layer = Mistral4EagleLayer(config, layer_idx=1, has_eagle_proj=False)
    # No pre-hooks registered on the layer itself (hooks are on submodules)
    assert layer.eagle_proj is None


# =============================================================================
# Tests: EagleDrafterForCausalLM export
# =============================================================================


def _make_eagle_config(config: Mistral4TextConfig) -> EagleConfig:
    """Wrap a Mistral4TextConfig in EagleConfig with mistral4 defaults."""
    return EagleConfig(config, "mistral4")


def test_mistral4_eagle_drafter_forward():
    """EagleDrafterForCausalLM forward runs without error and returns finite outputs."""
    if not torch.cuda.is_available():
        pytest.skip("torch_mla requires CUDA.")
    device = "cuda"
    dtype = torch.bfloat16
    config = _make_eagle_config(_small_eagle_text_config())
    model = EagleDrafterForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    batch, seq = 2, 6
    hidden_size = config.hidden_size
    inputs_embeds = torch.randn(batch, seq, hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
    hidden_states = torch.randn(batch, seq, hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        out = model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, hidden_states=hidden_states
        )

    assert out.norm_hidden_state is not None
    assert torch.isfinite(out.norm_hidden_state).all()
    assert out.norm_hidden_state.shape == (batch, seq, hidden_size)


def test_mistral4_eagle_drafter_export():
    """EagleDrafterForCausalLM can be exported with torch.export."""
    if not torch.cuda.is_available():
        pytest.skip("torch_mla requires CUDA.")
    device = "cuda"
    dtype = torch.bfloat16
    config = _make_eagle_config(_small_eagle_text_config())
    model = EagleDrafterForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    batch, seq = 1, 8
    hidden_size = config.hidden_size
    inputs_embeds = torch.randn(batch, seq, hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq, device=device).unsqueeze(0)
    hidden_states = torch.randn(batch, seq, hidden_size, device=device, dtype=dtype)

    # Note: dynamic_shapes is not passed since hidden_states arrives via **kwargs and
    # PyTorch's dynamic_shapes pytree matching doesn't support **kwargs args well.
    # We verify export succeeds and outputs are finite; shape flexibility is tested at
    # the EagleOneModelFactory level during full-pipeline smoke tests.
    gm = torch_export_to_gm(
        model,
        args=(inputs_embeds, position_ids),
        kwargs={"hidden_states": hidden_states},
    )
    assert gm is not None

    with torch.no_grad():
        out = gm(inputs_embeds, position_ids, hidden_states=hidden_states)
    assert out.norm_hidden_state is not None
    assert torch.isfinite(out.norm_hidden_state).all()
    assert out.norm_hidden_state.shape == (batch, seq, hidden_size)

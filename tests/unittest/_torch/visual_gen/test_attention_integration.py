# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test WAN Attention Integration.

Compares the new integrated attention (using TRT-LLM backend) with the original
naive implementation to ensure numerical equivalence.
"""

from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.modules.rms_norm import RMSNorm

# ============================================================================
# Flash Attention 4 availability
# ============================================================================
from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl import _cute_dsl_import_error
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import _flash_attn_fwd as _fa4_fwd
from tensorrt_llm._torch.visual_gen.attention_backend.parallel import (
    Attention2DAttention,
    RingAttention,
    UlyssesAttention,
)
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.attention_backend.vanilla import VanillaAttention
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping

# Import new integrated versions
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode, apply_rotary_emb
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    QuantAttentionConfig,
    VideoSparseAttentionConfig,
)

_flash_attn4_available = _fa4_fwd is not None
_cute_dsl_available = _cute_dsl_import_error is None

# ============================================================================
# Original naive implementations for comparison
# ============================================================================


class NaiveWanSelfAttention(nn.Module):
    """Original naive self-attention implementation (for comparison)."""

    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int, eps: float = 1e-6, dtype=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size

        # fused QKV projection
        self.to_qkv = nn.Linear(hidden_size, 3 * hidden_size, dtype=dtype)
        self.norm_q = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype, has_weights=True)
        self.norm_k = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype, has_weights=True)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, dtype=dtype)])

    def forward(self, hidden_states, freqs_cos, freqs_sin):
        B, S = hidden_states.shape[:2]

        q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)

        q = self.norm_q(q).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.norm_k(k).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if freqs_cos is not None and freqs_sin is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).flatten(2)
        out = self.to_out[0](out)
        return out


class NaiveWanCrossAttention(nn.Module):
    """Original naive cross-attention implementation (for comparison)."""

    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int, eps: float = 1e-6, dtype=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size

        self.to_q = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.to_k = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.to_v = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.norm_q = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype, has_weights=True)
        self.norm_k = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype, has_weights=True)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, dtype=dtype)])

    def forward(self, hidden_states, encoder_hidden_states):
        B, S = hidden_states.shape[:2]

        q = self.norm_q(self.to_q(hidden_states))
        k = self.norm_k(self.to_k(encoder_hidden_states))
        v = self.to_v(encoder_hidden_states)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).flatten(2)
        out = self.to_out[0](out)
        return out


# ============================================================================
# Test utilities
# ============================================================================


def create_model_config(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    attn_backend: str = "VANILLA",
    quant_attention_config: "QuantAttentionConfig | None" = None,
    sparse_attention_config=None,
    vsa_sparsity: "float | None" = None,
    *,
    visual_gen_mapping: VisualGenMapping | None = None,
    skip_create_weights_in_init: bool = False,
):
    """Create a mock DiffusionModelConfig for testing."""
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        eps=eps,
    )

    if vsa_sparsity is not None and sparse_attention_config is None:
        sparse_attention_config = VideoSparseAttentionConfig(vsa_sparsity=vsa_sparsity)

    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(
            backend=attn_backend,
            quant_attention_config=quant_attention_config,
            sparse_attention_config=sparse_attention_config,
        ),
        skip_create_weights_in_init=skip_create_weights_in_init,
    )
    config.attention_metadata_state = (
        create_attention_metadata_state() if attn_backend == "TRTLLM" else None
    )
    if visual_gen_mapping is not None:
        config.visual_gen_mapping = visual_gen_mapping
    return config


def _require_attention_backend(attn_backend: str, head_dim: Optional[int] = None) -> None:
    if attn_backend == "FA4" and not _flash_attn4_available:
        pytest.fail("FlashAttention 4 backend is required for FA4 attention test")
    if attn_backend == "CUTEDSL" and not _cute_dsl_available:
        pytest.fail("CuTe DSL backend is required for CUTEDSL attention test")
    if attn_backend == "CUTEDSL":
        compute_capability = torch.cuda.get_device_capability()
        gpu_arch = f"sm_{compute_capability[0]}{compute_capability[1]}a"
        if gpu_arch not in ("sm_100a", "sm_103a"):
            pytest.skip("CUTEDSL attention test requires a supported Blackwell-class GPU")
        if head_dim is not None and head_dim != 128:
            pytest.skip("CUTEDSL attention test requires head_dim=128")


def _make_cross_attention_with_mapping(
    visual_gen_mapping: VisualGenMapping,
    *,
    enable_sequence_parallel: bool,
    hidden_size: int = 64,
    num_heads: int = 4,
    head_dim: int = 16,
) -> Attention:
    config = create_model_config(
        hidden_size,
        num_heads,
        head_dim,
        visual_gen_mapping=visual_gen_mapping,
        skip_create_weights_in_init=True,
    )
    return Attention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        qkv_mode=QKVMode.SEPARATE_QKV,
        config=config,
        enable_sequence_parallel=enable_sequence_parallel,
    )


def copy_weights_self_attention(naive: NaiveWanSelfAttention, integrated: Attention):
    """Copy weights from naive to integrated self-attention."""
    # QKV projection: naive has to_qkv, integrated has qkv_proj
    integrated.qkv_proj.weight.data.copy_(naive.to_qkv.weight.data)
    if naive.to_qkv.bias is not None and integrated.qkv_proj.bias is not None:
        integrated.qkv_proj.bias.data.copy_(naive.to_qkv.bias.data)

    # QK norms
    integrated.norm_q.weight.data.copy_(naive.norm_q.weight.data)
    integrated.norm_k.weight.data.copy_(naive.norm_k.weight.data)

    # Output projection
    integrated.to_out[0].weight.data.copy_(naive.to_out[0].weight.data)
    if naive.to_out[0].bias is not None and integrated.to_out[0].bias is not None:
        integrated.to_out[0].bias.data.copy_(naive.to_out[0].bias.data)


def copy_weights_cross_attention(naive: NaiveWanCrossAttention, integrated: Attention):
    """Copy weights from naive to integrated cross-attention."""
    # Q, K, V projections
    integrated.to_q.weight.data.copy_(naive.to_q.weight.data)
    integrated.to_k.weight.data.copy_(naive.to_k.weight.data)
    integrated.to_v.weight.data.copy_(naive.to_v.weight.data)

    if naive.to_q.bias is not None and integrated.to_q.bias is not None:
        integrated.to_q.bias.data.copy_(naive.to_q.bias.data)
    if naive.to_k.bias is not None and integrated.to_k.bias is not None:
        integrated.to_k.bias.data.copy_(naive.to_k.bias.data)
    if naive.to_v.bias is not None and integrated.to_v.bias is not None:
        integrated.to_v.bias.data.copy_(naive.to_v.bias.data)

    # QK norms
    integrated.norm_q.weight.data.copy_(naive.norm_q.weight.data)
    integrated.norm_k.weight.data.copy_(naive.norm_k.weight.data)

    # Output projection
    integrated.to_out[0].weight.data.copy_(naive.to_out[0].weight.data)
    if naive.to_out[0].bias is not None and integrated.to_out[0].bias is not None:
        integrated.to_out[0].bias.data.copy_(naive.to_out[0].bias.data)


def generate_rope_embeddings(
    seq_len: int, head_dim: int, device: torch.device, is_HSD: bool = False
):
    """Generate RoPE embeddings with full head_dim.

    apply_rotary_emb expects freqs with full head_dim, then slices with [..., 0::2] and [..., 1::2].

    Args:
        is_HSD: If True, returns [1, 1, S, D] for broadcasting with [B, H, S, D] (naive)
                If False, returns [1, S, 1, D] for broadcasting with [B, S, H, D] (integrated)
    """
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    # Use full head_dim - apply_rotary_emb will slice with 0::2 and 1::2
    div_term = torch.exp(
        torch.arange(0, head_dim, device=device) * (-torch.log(torch.tensor(10000.0)) / head_dim)
    )

    if is_HSD:
        freqs_cos = torch.cos(position * div_term).unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
        freqs_sin = torch.sin(position * div_term).unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
    else:
        freqs_cos = torch.cos(position * div_term).unsqueeze(0).unsqueeze(2)  # [1, S, 1, D]
        freqs_sin = torch.sin(position * div_term).unsqueeze(0).unsqueeze(2)  # [1, S, 1, D]

    return freqs_cos, freqs_sin


# ============================================================================
# Sequence-parallel configuration guards
# ============================================================================


class TestSeparateQkvSequenceParallelGuard:
    def test_ring_with_separate_qkv_raises(self):
        vgm = VisualGenMapping(world_size=2, rank=0, ring_size=2)

        with pytest.raises(ValueError, match="SEPARATE_QKV cross-attention does not support"):
            _make_cross_attention_with_mapping(vgm, enable_sequence_parallel=True)

    def test_ring_with_sequence_parallel_disabled_allowed(self):
        vgm = VisualGenMapping(world_size=2, rank=0, ring_size=2)

        attn = _make_cross_attention_with_mapping(vgm, enable_sequence_parallel=False)

        assert isinstance(attn.attn, VanillaAttention)
        assert not isinstance(attn.attn, (UlyssesAttention, RingAttention, Attention2DAttention))

    def test_attn2d_with_sequence_parallel_disabled_allowed(self):
        vgm = VisualGenMapping(world_size=4, rank=0, attn2d_row_size=2, attn2d_col_size=2)

        attn = _make_cross_attention_with_mapping(vgm, enable_sequence_parallel=False)

        assert isinstance(attn.attn, VanillaAttention)


def _build_sage_routed_attention(qkv_mode: QKVMode):
    """Build a TRTLLM-SAGE-configured Attention for backend-routing checks."""
    quant_cfg = QuantAttentionConfig(
        qk_dtype="int8", q_block_size=1, k_block_size=16, v_block_size=1
    )
    config = create_model_config(
        hidden_size=512,
        num_heads=4,
        head_dim=128,
        attn_backend="TRTLLM",
        quant_attention_config=quant_cfg,
        skip_create_weights_in_init=True,
    )
    attn = Attention(
        hidden_size=512,
        num_attention_heads=4,
        head_dim=128,
        qkv_mode=qkv_mode,
        config=config,
    )
    return attn, quant_cfg


class TestSageAttentionBackendRouting:
    def test_self_attention_uses_trtllm_sage_backend(self):
        attn, quant_cfg = _build_sage_routed_attention(QKVMode.FUSE_QKV)
        assert attn.attn_backend == "TRTLLM"
        assert isinstance(attn.attn, TrtllmAttention)
        assert attn.attn.quant_attention_config == quant_cfg
        assert not attn.attn.support_fused_qkv()

    def test_cross_attention_with_sage_config_falls_back_to_vanilla(self):
        attn, _ = _build_sage_routed_attention(QKVMode.SEPARATE_QKV)
        assert attn.attn_backend == "VANILLA"
        assert isinstance(attn.attn, VanillaAttention)


# ============================================================================
# Test functions
# ============================================================================
@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize(
    ("attn_backend", "quant_attention_config"),
    [
        ("VANILLA", None),
        ("TRTLLM", None),
        ("FA4", None),
        ("CUTEDSL", None),
        ("CUTEDSL", QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8")),
    ],
)
def test_self_attention_equivalence(
    head_dim: int, attn_backend: str, quant_attention_config: "QuantAttentionConfig | None"
):
    """Test that integrated self-attention produces same output as naive."""
    _require_attention_backend(attn_backend, head_dim)

    print("\n" + "=" * 60)
    print("Testing Self-Attention Equivalence")
    print("=" * 60)

    # Config
    batch_size = 2
    seq_len = 16
    num_heads = 4
    hidden_size = head_dim * num_heads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16  # Use bf16 since flashinfer doesn't support fp32

    print(f"Config: B={batch_size}, S={seq_len}, H={hidden_size}, heads={num_heads}")
    print(f"Device: {device}, dtype: {dtype}")

    # Create models
    naive = NaiveWanSelfAttention(hidden_size, num_heads, head_dim, dtype=dtype).to(device)

    model_config = create_model_config(
        hidden_size,
        num_heads,
        head_dim,
        attn_backend=attn_backend,
        quant_attention_config=quant_attention_config,
    )
    integrated = Attention(
        hidden_size, num_heads, qkv_mode=QKVMode.FUSE_QKV, config=model_config
    ).to(device)  # self attention

    # Copy weights
    copy_weights_self_attention(naive, integrated)

    # Set to eval mode
    naive.eval()
    integrated.eval()

    # Create inputs
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    # Naive uses [1, 1, S, D] (HSD format) - broadcasts with [B, H, S, D]
    freqs_cos_HSD, freqs_sin_HSD = generate_rope_embeddings(seq_len, head_dim, device, is_HSD=True)
    # Integrated uses [1, S, 1, D] (SHD format) - broadcasts with [B, S, H, D]
    freqs_cos_SHD, freqs_sin_SHD = generate_rope_embeddings(seq_len, head_dim, device, is_HSD=False)

    # Forward pass
    with torch.no_grad():
        out_naive = naive(hidden_states, freqs_cos_HSD, freqs_sin_HSD)
        out_integrated = integrated(hidden_states, freqs=(freqs_cos_SHD, freqs_sin_SHD))

    # Compare (using looser tolerance for bf16)
    max_diff = (out_naive - out_integrated).abs().max().item()
    mean_diff = (out_naive - out_integrated).abs().mean().item()
    tol = 1e-2 if quant_attention_config is None else 2e-2
    is_close = torch.allclose(out_naive, out_integrated, rtol=tol, atol=tol)

    print("\nResults:")
    print(f"  Output shape: naive={out_naive.shape}, integrated={out_integrated.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Outputs match (rtol={tol}, atol={tol}): {is_close}")

    if is_close:
        print("  ✅ PASS: Self-attention outputs match!")
    else:
        print("  ❌ FAIL: Self-attention outputs differ!")

    assert is_close, (
        f"Self-attention outputs differ: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
    )
    return is_close


# seq_len: pow2 baselines + real WAN latent token counts (VAE 8x spatial, 4x temporal, patch [1,2,2])
# batch_size: B=1 (cfg_size=2, split across GPUs) / B=2 (cfg_size=1, single GPU)
@pytest.mark.parametrize("seq_len", [256, 512, 1560, 3600, 4096, 16384, 32760])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("qk_dtype", ["fp8", "int8"])
def test_sage_attention_self_attention(qk_dtype: str, batch_size: int, seq_len: int):
    """Test SageAttention (TRTLLM + quant_attention_config) self-attention.

    SageAttention quantizes Q/K/V with per-block scaling factors, so outputs
    are expected to differ from the naive SDPA reference. We verify:
    1. Forward pass completes without error
    2. Output shape matches naive
    3. Outputs are finite (no NaN/Inf)
    4. Approximate agreement with naive (cosine similarity > 0.99)
    """
    compute_capability = torch.cuda.get_device_capability()
    gpu_arch = f"sm_{compute_capability[0]}{compute_capability[1]}a"
    if qk_dtype == "int8" and gpu_arch not in ["sm_100a"]:
        pytest.skip("Int8 kernels are only available for SM100 devices.")
    print("\n" + "=" * 60)
    print(f"Testing SageAttention (qk_dtype={qk_dtype}, B={batch_size}, S={seq_len})")
    print("=" * 60)

    # The sm100 sage kernel only has cubins for head_dim=128,
    # so match the WAN model dimensions (12 heads, head_dim=128).
    num_heads = 12
    head_dim = 128
    hidden_size = num_heads * head_dim  # 1536
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Config: B={batch_size}, S={seq_len}, H={hidden_size}, heads={num_heads}, D={head_dim}")
    print(f"Device: {device}, dtype: {dtype}")

    quant_cfg = QuantAttentionConfig(
        qk_dtype=qk_dtype,
        q_block_size=1,
        k_block_size=16 if qk_dtype == "int8" else 1,
        v_block_size=1,
    )

    # Create models
    naive = NaiveWanSelfAttention(hidden_size, num_heads, head_dim, dtype=dtype).to(device)

    model_config = create_model_config(
        hidden_size,
        num_heads,
        head_dim,
        attn_backend="TRTLLM",
        quant_attention_config=quant_cfg,
    )
    integrated = Attention(
        hidden_size, num_heads, qkv_mode=QKVMode.FUSE_QKV, config=model_config
    ).to(device)

    # Copy weights
    copy_weights_self_attention(naive, integrated)

    naive.eval()
    integrated.eval()

    # Create inputs
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    freqs_cos_HSD, freqs_sin_HSD = generate_rope_embeddings(seq_len, head_dim, device, is_HSD=True)
    freqs_cos_SHD, freqs_sin_SHD = generate_rope_embeddings(seq_len, head_dim, device, is_HSD=False)

    # Forward pass
    with torch.no_grad():
        out_naive = naive(hidden_states, freqs_cos_HSD, freqs_sin_HSD)
        out_sage = integrated(hidden_states, freqs=(freqs_cos_SHD, freqs_sin_SHD))

    # --- Assertions ---

    # 1. Shape match
    assert out_sage.shape == out_naive.shape, (
        f"Shape mismatch: sage={out_sage.shape}, naive={out_naive.shape}"
    )

    # 2. All values finite (no NaN / Inf)
    assert torch.isfinite(out_sage).all(), (
        f"SageAttention output contains NaN or Inf (B={batch_size}, S={seq_len})"
    )

    # 3. Cosine similarity — sage quantization (FP8 per-block) introduces larger
    #    error than bf16 rounding, so elementwise allclose is too strict.
    #    Cosine similarity captures directional agreement robustly.
    max_diff = (out_naive - out_sage).abs().max().item()
    mean_diff = (out_naive - out_sage).abs().mean().item()
    cos_sim = F.cosine_similarity(
        out_naive.reshape(-1).float(), out_sage.reshape(-1).float(), dim=0
    ).item()

    print(f"\n  Output shape: {out_sage.shape}")
    print(f"  Max absolute diff:  {max_diff:.2e}")
    print(f"  Mean absolute diff: {mean_diff:.2e}")
    print(f"  Cosine similarity:  {cos_sim:.6f}")

    assert cos_sim > 0.99, (
        f"SageAttention cosine similarity too low: {cos_sim:.4f} < 0.99 "
        f"(B={batch_size}, S={seq_len}, qk_dtype={qk_dtype})"
    )
    return cos_sim > 0.99


@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize(
    ("attn_backend", "quant_attention_config"),
    [
        ("VANILLA", None),
        ("FA4", None),
        ("CUTEDSL", None),
        ("CUTEDSL", QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8")),
    ],
)
def test_cross_attention_equivalence(
    head_dim: int, attn_backend: str, quant_attention_config: "QuantAttentionConfig | None"
):
    """Test that integrated cross-attention produces same output as naive."""
    _require_attention_backend(attn_backend, head_dim)

    print("\n" + "=" * 60)
    print("Testing Cross-Attention Equivalence")
    print("=" * 60)

    # Config
    batch_size = 2
    seq_len = 16
    encoder_seq_len = 24  # Different from query seq_len
    num_heads = 4
    hidden_size = num_heads * head_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16  # Use bf16 since flashinfer doesn't support fp32

    print(
        f"Config: B={batch_size}, S_q={seq_len}, S_kv={encoder_seq_len}, H={hidden_size}, heads={num_heads}"
    )
    print(f"Device: {device}, dtype: {dtype}")

    # Create models
    naive = NaiveWanCrossAttention(hidden_size, num_heads, head_dim, dtype=dtype).to(device)

    model_config = create_model_config(
        hidden_size,
        num_heads,
        head_dim,
        attn_backend=attn_backend,
        quant_attention_config=quant_attention_config,
    )
    integrated = Attention(
        hidden_size, num_heads, qkv_mode=QKVMode.SEPARATE_QKV, config=model_config
    ).to(device)  # cross attention

    # Copy weights
    copy_weights_cross_attention(naive, integrated)

    # Set to eval mode
    naive.eval()
    integrated.eval()

    # Create inputs
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(
        batch_size, encoder_seq_len, hidden_size, device=device, dtype=dtype
    )

    # Forward pass
    with torch.no_grad():
        out_naive = naive(hidden_states, encoder_hidden_states)
        out_integrated = integrated(hidden_states, encoder_hidden_states)

    # Compare (using looser tolerance for bf16)
    max_diff = (out_naive - out_integrated).abs().max().item()
    mean_diff = (out_naive - out_integrated).abs().mean().item()
    tol = 1e-2 if quant_attention_config is None else 2e-2
    is_close = torch.allclose(out_naive, out_integrated, rtol=tol, atol=tol)

    print("\nResults:")
    print(f"  Output shape: naive={out_naive.shape}, integrated={out_integrated.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Outputs match (rtol={tol}, atol={tol}): {is_close}")

    if is_close:
        print("  ✅ PASS: Cross-attention outputs match!")
    else:
        print("  ❌ FAIL: Cross-attention outputs differ!")

    assert is_close, (
        f"Cross-attention outputs differ: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
    )
    return is_close


@pytest.mark.parametrize(
    "batch,seq_len_q,seq_len_kv,num_heads,head_dim",
    [
        (1, 1024, 512, 12, 128),
        (1, 2048, 512, 12, 128),
    ],
)
@pytest.mark.parametrize(
    ("attn_backend", "quant_attention_config"),
    [
        ("FA4", None),
        ("CUTEDSL", None),
        ("CUTEDSL", QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8")),
    ],
)
def test_fast_cross_attention_wan_shapes(
    batch: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    attn_backend: str,
    quant_attention_config: "QuantAttentionConfig | None",
):
    """Test fast cross-attention correctness at Wan-realistic shapes."""
    _require_attention_backend(attn_backend, head_dim)

    hidden_size = num_heads * head_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(
        f"\nConfig: B={batch}, S_q={seq_len_q}, S_kv={seq_len_kv}, heads={num_heads}, head_dim={head_dim}"
    )

    naive = NaiveWanCrossAttention(hidden_size, num_heads, head_dim, dtype=dtype).to(device)

    cfg_vanilla = create_model_config(hidden_size, num_heads, head_dim, attn_backend="VANILLA")
    ref = Attention(hidden_size, num_heads, qkv_mode=QKVMode.SEPARATE_QKV, config=cfg_vanilla).to(
        device
    )

    cfg_fast = create_model_config(
        hidden_size,
        num_heads,
        head_dim,
        attn_backend=attn_backend,
        quant_attention_config=quant_attention_config,
    )
    fast_model = Attention(
        hidden_size, num_heads, qkv_mode=QKVMode.SEPARATE_QKV, config=cfg_fast
    ).to(device)

    copy_weights_cross_attention(naive, ref)
    copy_weights_cross_attention(naive, fast_model)
    ref.eval()
    fast_model.eval()

    torch.manual_seed(42)
    hidden_states = torch.randn(batch, seq_len_q, hidden_size, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(batch, seq_len_kv, hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        out_ref = ref(hidden_states, encoder_hidden_states)
        out_fast = fast_model(hidden_states, encoder_hidden_states)

    max_diff = (out_ref - out_fast).abs().max().item()
    tol = 1e-2 if quant_attention_config is None else 2e-2
    is_close = torch.allclose(out_ref, out_fast, rtol=tol, atol=tol)
    print(f"  Max diff: {max_diff:.2e}, match: {is_close}")
    assert is_close, f"{attn_backend} cross-attn mismatch at Wan shapes: max_diff={max_diff:.2e}"


# ============================================================================
# VSA self-attention (CUTEDSL backend, sparse_attention_config.algorithm='vsa')
# ============================================================================


def _build_vsa_setup(sparsity: float, batch_size: int, seed: int):
    """Build naive + integrated models, VSA metadata, and inputs for a VSA test.

    latent (8,8,8) -> 512 tokens (divisible by block_size=64), head_dim=128.
    """
    from tensorrt_llm._torch.visual_gen.attention_backend import VSAMetadataBuilder

    latent_shape = (8, 8, 8)
    seq_len = latent_shape[0] * latent_shape[1] * latent_shape[2]
    num_heads = 4
    head_dim = 128
    hidden_size = num_heads * head_dim
    device = torch.device("cuda")
    dtype = torch.bfloat16

    naive = NaiveWanSelfAttention(hidden_size, num_heads, head_dim, dtype=dtype).to(device)
    cfg_vsa = create_model_config(
        hidden_size, num_heads, head_dim, attn_backend="CUTEDSL", vsa_sparsity=sparsity
    )
    integrated = Attention(hidden_size, num_heads, qkv_mode=QKVMode.FUSE_QKV, config=cfg_vsa).to(
        device
    )
    # Fail loudly if the VSA path silently fell back to dense (which would set
    # attn_backend to "VANILLA") instead of selecting the CUTEDSL/VSA backend.
    assert integrated.attn_backend == "CUTEDSL", (
        f"Expected CUTEDSL (VSA) backend, got {integrated.attn_backend!r}"
    )
    copy_weights_self_attention(naive, integrated)
    naive.eval()
    integrated.eval()

    metadata = VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=latent_shape,
        patch_size=(1, 1, 1),
        vsa_sparsity=sparsity,
        device=device,
    )
    torch.manual_seed(seed)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    return SimpleNamespace(
        naive=naive,
        integrated=integrated,
        metadata=metadata,
        hidden_states=hidden_states,
        gate_compress_zero=torch.zeros_like(hidden_states),
        freqs_HSD=generate_rope_embeddings(seq_len, head_dim, device, is_HSD=True),
        freqs_SHD=generate_rope_embeddings(seq_len, head_dim, device, is_HSD=False),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA needs CUDA")
def test_vsa_self_attention_equivalence_at_sparsity_zero():
    """VSA at sparsity=0 with G_c=0 reduces to dense attention (top_k=num_cubes,
    output=O_f); must match the naive SDPA reference modulo bf16 rounding."""
    from tensorrt_llm._torch.visual_gen.attention_backend import set_vsa_forward_context

    s = _build_vsa_setup(sparsity=0.0, batch_size=2, seed=42)

    with torch.no_grad():
        out_naive = s.naive(s.hidden_states, *s.freqs_HSD)
    with torch.no_grad(), set_vsa_forward_context(s.metadata):
        out_vsa = s.integrated(
            s.hidden_states, freqs=s.freqs_SHD, gate_compress=s.gate_compress_zero
        )

    assert out_naive.shape == out_vsa.shape, (
        f"shape mismatch: naive={out_naive.shape}, vsa={out_vsa.shape}"
    )
    max_diff = (out_naive - out_vsa).abs().max().item()
    mean_diff = (out_naive - out_vsa).abs().mean().item()
    assert torch.allclose(out_naive, out_vsa, rtol=1e-2, atol=1e-2), (
        f"VSA(sparsity=0, G_c=0) deviates from naive dense SDPA: "
        f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA needs CUDA")
@pytest.mark.parametrize("sparsity", [0.0, 0.5], ids=["s0", "s0p5"])
def test_vsa_self_attention_finite(sparsity: float):
    """VSA forward must produce finite output (no NaN/Inf) at any supported sparsity."""
    from tensorrt_llm._torch.visual_gen.attention_backend import set_vsa_forward_context

    s = _build_vsa_setup(sparsity=sparsity, batch_size=1, seed=0)

    with torch.no_grad(), set_vsa_forward_context(s.metadata):
        out = s.integrated(s.hidden_states, freqs=s.freqs_SHD, gate_compress=s.gate_compress_zero)

    assert out.shape == s.hidden_states.shape
    nan_count = torch.isnan(out).sum().item()
    inf_count = torch.isinf(out).sum().item()
    assert nan_count == 0 and inf_count == 0, (
        f"VSA produced non-finite output at sparsity={sparsity}: NaN={nan_count}, Inf={inf_count}"
    )


def test_trtllm_cached_prepare():
    """Test that TRTLLM attention cached prepare works correctly.

    This test verifies that when running multiple forward passes with same B/S
    but different q/k/v values, the cached prepare phase doesn't cause incorrect
    results (i.e., outputs should differ when inputs differ).
    """
    print("\n" + "=" * 60)
    print("Testing TRTLLM Cached Prepare Phase")
    print("=" * 60)

    # Config - same B, S for all iterations
    batch_size = 2
    seq_len = 16
    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads
    num_iterations = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Config: B={batch_size}, S={seq_len}, H={hidden_size}, heads={num_heads}")
    print(f"Running {num_iterations} iterations with same B/S but different inputs")

    # Create models - single instance to test caching
    naive = NaiveWanSelfAttention(hidden_size, num_heads, head_dim, dtype=dtype).to(device)
    model_config = create_model_config(hidden_size, num_heads, head_dim, attn_backend="TRTLLM")
    integrated = Attention(
        hidden_size, num_heads, qkv_mode=QKVMode.FUSE_QKV, config=model_config
    ).to(device)  # self attention

    # Copy weights
    copy_weights_self_attention(naive, integrated)

    naive.eval()
    integrated.eval()

    # Generate freqs (same for all iterations since S is same)
    freqs_cos_HSD, freqs_sin_HSD = generate_rope_embeddings(seq_len, head_dim, device, is_HSD=True)
    freqs_cos_SHD, freqs_sin_SHD = generate_rope_embeddings(seq_len, head_dim, device, is_HSD=False)

    all_passed = True
    outputs_integrated = []

    with torch.no_grad():
        for i in range(num_iterations):
            # Different random inputs for each iteration
            torch.manual_seed(42 + i)  # Different seed each time
            hidden_states = torch.randn(
                batch_size, seq_len, hidden_size, device=device, dtype=dtype
            )

            out_naive = naive(hidden_states, freqs_cos_HSD, freqs_sin_HSD)
            out_integrated = integrated(hidden_states, freqs=(freqs_cos_SHD, freqs_sin_SHD))

            # Check this iteration matches naive
            max_diff = (out_naive - out_integrated).abs().max().item()
            is_close = torch.allclose(out_naive, out_integrated, rtol=1e-2, atol=1e-2)

            status = "✅" if is_close else "❌"
            print(f"  Iteration {i + 1}: max_diff={max_diff:.2e} {status}")

            if not is_close:
                all_passed = False

            outputs_integrated.append(out_integrated.clone())

    # Additional check: outputs should be DIFFERENT across iterations
    # (since inputs were different)
    print("\n  Checking outputs differ across iterations (inputs were different):")
    outputs_differ = True
    for i in range(1, num_iterations):
        diff = (outputs_integrated[i] - outputs_integrated[0]).abs().max().item()
        if diff < 1e-6:
            print(
                f"    ⚠️  Iteration {i + 1} output same as iteration 1 (diff={diff:.2e}) - possible caching bug!"
            )
            outputs_differ = False
        else:
            print(f"    Iteration {i + 1} vs 1: diff={diff:.2e} ✅")

    if all_passed and outputs_differ:
        print("\n  ✅ PASS: Cached prepare works correctly!")
    else:
        print("\n  ❌ FAIL: Cached prepare may have issues!")
        all_passed = False

    assert all_passed, "Cached prepare: outputs did not match naive reference"
    assert outputs_differ, (
        "Cached prepare: outputs should differ across iterations with different inputs"
    )
    return all_passed


def test_trtllm_varying_seq_len():
    """Test TRTLLM attention with varying sequence lengths.

    This tests that the prepare phase correctly handles different seq_lens
    and doesn't incorrectly reuse cached metadata.
    """
    print("\n" + "=" * 60)
    print("Testing TRTLLM with Varying Sequence Lengths")
    print("=" * 60)

    batch_size = 2
    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads
    seq_lens = [8, 16, 32, 16, 8]  # Vary seq_len, including repeats
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Config: B={batch_size}, H={hidden_size}, heads={num_heads}")
    print(f"Testing seq_lens: {seq_lens}")

    # Create models - single instance to test caching across different seq_lens
    naive = NaiveWanSelfAttention(hidden_size, num_heads, head_dim, dtype=dtype).to(device)
    model_config = create_model_config(hidden_size, num_heads, head_dim, attn_backend="TRTLLM")
    integrated = Attention(
        hidden_size, num_heads, qkv_mode=QKVMode.FUSE_QKV, config=model_config
    ).to(device)  # self attention

    copy_weights_self_attention(naive, integrated)

    naive.eval()
    integrated.eval()

    all_passed = True

    with torch.no_grad():
        for i, seq_len in enumerate(seq_lens):
            torch.manual_seed(42 + i)
            hidden_states = torch.randn(
                batch_size, seq_len, hidden_size, device=device, dtype=dtype
            )

            freqs_cos_HSD, freqs_sin_HSD = generate_rope_embeddings(
                seq_len, head_dim, device, is_HSD=True
            )
            freqs_cos_SHD, freqs_sin_SHD = generate_rope_embeddings(
                seq_len, head_dim, device, is_HSD=False
            )

            out_naive = naive(hidden_states, freqs_cos_HSD, freqs_sin_HSD)
            out_integrated = integrated(hidden_states, freqs=(freqs_cos_SHD, freqs_sin_SHD))

            max_diff = (out_naive - out_integrated).abs().max().item()
            is_close = torch.allclose(out_naive, out_integrated, rtol=1e-2, atol=1e-2)

            status = "✅" if is_close else "❌"
            print(f"  seq_len={seq_len:3d}: max_diff={max_diff:.2e} {status}")

            if not is_close:
                all_passed = False

    if all_passed:
        print("\n  ✅ PASS: Varying seq_len handled correctly!")
    else:
        print("\n  ❌ FAIL: Issues with varying seq_len!")

    assert all_passed, "Varying seq_len: outputs did not match naive reference"
    return all_passed


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("WAN Attention Integration Tests")
    print("=" * 60)

    results = {}

    # Run self-attention tests with different backends
    fast_backends = ["FA4"] if _flash_attn4_available else []
    if _cute_dsl_available:
        fast_backends.append("CUTEDSL")
    for backend in ["VANILLA", "TRTLLM"] + fast_backends:
        results[f"self_attention_{backend}"] = test_self_attention_equivalence(backend, "NO_QUANT")

    # Run SageAttention self-attention tests (subset for manual runner)
    for batch_size in [1, 2]:
        for seq_len in [4096, 32760]:
            for qk_dtype in ["fp8", "int8"]:
                label = f"sage_B{batch_size}_S{seq_len}_QkDtype{qk_dtype}"
                results[label] = test_sage_attention_self_attention(
                    qk_dtype=qk_dtype, batch_size=batch_size, seq_len=seq_len
                )

    # Run cross-attention tests
    results["cross_attention_VANILLA"] = test_cross_attention_equivalence("VANILLA", "NO_QUANT")
    if _flash_attn4_available:
        results["cross_attention_FA4"] = test_cross_attention_equivalence("FA4", "NO_QUANT")
    if _cute_dsl_available:
        results["cross_attention_CUTEDSL"] = test_cross_attention_equivalence("CUTEDSL", "NO_QUANT")

    # Run TRTLLM-specific caching tests
    results["trtllm_cached_prepare"] = test_trtllm_cached_prepare()
    results["trtllm_varying_seq_len"] = test_trtllm_varying_seq_len()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = all(results.values())
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")

    print()
    if all_passed:
        print("All tests passed! ✅")
    else:
        print("Some tests failed! ❌")

    return all_passed


if __name__ == "__main__":
    run_all_tests()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test WAN Attention Integration.

Compares the new integrated attention (using TRT-LLM backend) with the original
naive implementation to ensure numerical equivalence.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.visual_gen.config import AttentionConfig, DiffusionModelConfig

# Import new integrated versions
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode, apply_rotary_emb

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
):
    """Create a mock DiffusionModelConfig for testing."""
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        eps=eps,
    )

    # Create a minimal config without quantization
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(backend=attn_backend),
        skip_create_weights_in_init=False,
    )
    return config


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
# Test functions
# ============================================================================
@pytest.mark.parametrize("attn_backend", ["VANILLA", "TRTLLM"])
def test_self_attention_equivalence(attn_backend: str):
    """Test that integrated self-attention produces same output as naive."""
    print("\n" + "=" * 60)
    print("Testing Self-Attention Equivalence")
    print("=" * 60)

    # Config
    batch_size = 2
    seq_len = 16
    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16  # Use bf16 since flashinfer doesn't support fp32

    print(f"Config: B={batch_size}, S={seq_len}, H={hidden_size}, heads={num_heads}")
    print(f"Device: {device}, dtype: {dtype}")

    # Create models
    naive = NaiveWanSelfAttention(hidden_size, num_heads, head_dim, dtype=dtype).to(device)

    model_config = create_model_config(hidden_size, num_heads, head_dim, attn_backend=attn_backend)
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
    is_close = torch.allclose(out_naive, out_integrated, rtol=1e-2, atol=1e-3)

    print("\nResults:")
    print(f"  Output shape: naive={out_naive.shape}, integrated={out_integrated.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Outputs match (rtol=1e-2, atol=1e-3): {is_close}")

    if is_close:
        print("  ✅ PASS: Self-attention outputs match!")
    else:
        print("  ❌ FAIL: Self-attention outputs differ!")

    return is_close


@pytest.mark.parametrize("attn_backend", ["VANILLA"])
def test_cross_attention_equivalence(attn_backend: str):
    """Test that integrated cross-attention produces same output as naive."""
    print("\n" + "=" * 60)
    print("Testing Cross-Attention Equivalence")
    print("=" * 60)

    # Config
    batch_size = 2
    seq_len = 16
    encoder_seq_len = 24  # Different from query seq_len
    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16  # Use bf16 since flashinfer doesn't support fp32

    print(
        f"Config: B={batch_size}, S_q={seq_len}, S_kv={encoder_seq_len}, H={hidden_size}, heads={num_heads}"
    )
    print(f"Device: {device}, dtype: {dtype}")

    # Create models
    naive = NaiveWanCrossAttention(hidden_size, num_heads, head_dim, dtype=dtype).to(device)

    model_config = create_model_config(hidden_size, num_heads, head_dim, attn_backend=attn_backend)
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
    is_close = torch.allclose(out_naive, out_integrated, rtol=1e-2, atol=1e-3)

    print("\nResults:")
    print(f"  Output shape: naive={out_naive.shape}, integrated={out_integrated.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Outputs match (rtol=1e-2, atol=1e-3): {is_close}")

    if is_close:
        print("  ✅ PASS: Cross-attention outputs match!")
    else:
        print("  ❌ FAIL: Cross-attention outputs differ!")

    return is_close


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
            is_close = torch.allclose(out_naive, out_integrated, rtol=1e-2, atol=1e-3)

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
            is_close = torch.allclose(out_naive, out_integrated, rtol=1e-2, atol=1e-3)

            status = "✅" if is_close else "❌"
            print(f"  seq_len={seq_len:3d}: max_diff={max_diff:.2e} {status}")

            if not is_close:
                all_passed = False

    if all_passed:
        print("\n  ✅ PASS: Varying seq_len handled correctly!")
    else:
        print("\n  ❌ FAIL: Issues with varying seq_len!")

    return all_passed


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("WAN Attention Integration Tests")
    print("=" * 60)

    results = {}

    # Run self-attention tests with different backends
    for backend in ["VANILLA", "TRTLLM"]:
        results[f"self_attention_{backend}"] = test_self_attention_equivalence(backend)

    # Run cross-attention test (VANILLA only)
    results["cross_attention_VANILLA"] = test_cross_attention_equivalence("VANILLA")

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

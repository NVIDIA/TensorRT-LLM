# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical parity for the FlashAttn4 attention backend.

FA4 is used for LTX-2 audio self-attn and a2v cross-attn when the user picks
``attention.backend: FA4``.

**key_padding_mask**: under Ulysses, audio is padded so T_a is divisible by
ulysses_size, and padded K columns must produce zero attention contribution.
FA4's cute interface accepts ``seqused_k`` per-batch valid lengths; these tests
verify that translating a True-prefix bool mask to ``seqused_k`` yields output
identical to running FA4 on the unpadded K/V (within bf16 tolerance).

**split-KV**: ``FlashAttn4Attention`` passes ``num_splits=0``, so FA4's heuristic
picks the split count, and any count above 1 selects a separate split-KV kernel
that the CuTe DSL compiles at first use. The mask tests above use K/V short enough
that the heuristic returns 1 (it short-circuits at ``num_n_blocks <= 4``), so the
split-KV kernel needs its own longer-K/V shape to be covered at all.

Requires CUDA (FA4 is GPU-only).
"""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
    FlashAttn4Attention,
    _flash_attn_fwd,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 requires CUDA")


@pytest.fixture(autouse=True)
def require_bundled_fa4():
    assert _flash_attn_fwd is not None, "The bundled FA4 backend failed to import"
    assert getattr(_flash_attn_fwd, "visual_gen_tuning_api", None) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_autotuned_tactics_output_lse_and_graph_replay(dtype, strided):
    """Every demo tactic must preserve both attention output and float32 LSE."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("FA4 tuning demo targets SM100/SM103")
    from tensorrt_llm._torch.autotuner import OptimizationProfile
    from tensorrt_llm._torch.visual_gen.attention_backend.fa4_autotuner import Fa4Runner

    torch.manual_seed(42)
    inputs = [
        torch.randn(1, 512, 4 if strided else 2, 128, device="cuda", dtype=dtype) for _ in range(3)
    ]
    if strided:
        inputs = [t[:, :, ::2] for t in inputs]
    q, k, v = (t.float().transpose(1, 2) for t in inputs)
    scale = 128**-0.5
    scores = q @ k.transpose(-2, -1) * scale
    expected_lse = torch.logsumexp(scores, dim=-1)
    expected = (scores.softmax(dim=-1) @ v).transpose(1, 2).to(dtype)
    runner = Fa4Runner(inputs, scale)
    for tactic in runner.get_valid_tactics(inputs, OptimizationProfile()):
        output, lse = runner(inputs, tactic=tactic)
        torch.testing.assert_close(output, expected, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(lse, expected_lse, atol=2e-3, rtol=2e-3)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output, graph_lse = runner(inputs, tactic=tactic)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, output, atol=0, rtol=0)
        torch.testing.assert_close(graph_lse, lse, atol=0, rtol=0)


def _run_self_attn(B, S_real, S_pad, H, d_h, dtype=torch.bfloat16):
    """Helper: self-attention case (Q=K=V, padded only on the seq dim)."""
    device = "cuda"
    torch.manual_seed(0)
    S_full = S_real + S_pad

    # Build a padded sequence with random valid prefix + arbitrary junk suffix.
    x_valid = torch.randn(B, S_real, H, d_h, dtype=dtype, device=device)
    x_pad = torch.randn(B, S_pad, H, d_h, dtype=dtype, device=device)
    x_full = torch.cat([x_valid, x_pad], dim=1)
    mask = torch.zeros(B, S_full, dtype=torch.bool, device=device)
    mask[:, :S_real] = True

    fa4 = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)

    # FA4 path with seqused_k (q_full = k_full = v_full = x_full padded; only valid prefix attends).
    out_padded = fa4.forward(q=x_full, k=x_full, v=x_full, key_padding_mask=mask)

    # Reference: FA4 on unpadded inputs only (Q seq = K seq = S_real).
    out_ref = fa4.forward(q=x_valid, k=x_valid, v=x_valid)

    # Only the valid Q rows must match; pad Q rows are stripped downstream by
    # the caller and are not part of the contract.
    return out_padded[:, :S_real], out_ref


def _run_cross_attn(B, S_q, S_real_kv, S_pad_kv, H, d_h, dtype=torch.bfloat16):
    """Helper: cross-attention case (Q full, K/V padded on seq dim)."""
    device = "cuda"
    torch.manual_seed(1)
    S_full_kv = S_real_kv + S_pad_kv

    q = torch.randn(B, S_q, H, d_h, dtype=dtype, device=device)
    k_valid = torch.randn(B, S_real_kv, H, d_h, dtype=dtype, device=device)
    v_valid = torch.randn(B, S_real_kv, H, d_h, dtype=dtype, device=device)
    k_pad = torch.randn(B, S_pad_kv, H, d_h, dtype=dtype, device=device)
    v_pad = torch.randn(B, S_pad_kv, H, d_h, dtype=dtype, device=device)
    k_full = torch.cat([k_valid, k_pad], dim=1)
    v_full = torch.cat([v_valid, v_pad], dim=1)
    mask = torch.zeros(B, S_full_kv, dtype=torch.bool, device=device)
    mask[:, :S_real_kv] = True

    fa4 = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)

    out_padded = fa4.forward(q=q, k=k_full, v=v_full, key_padding_mask=mask)
    out_ref = fa4.forward(q=q, k=k_valid, v=v_valid)
    return out_padded, out_ref


def test_self_attn_padded_kv_with_mask_matches_unpadded():
    """FA4 self-attn: masked padded K/V matches unpadded K/V on valid Q rows."""
    out, ref = _run_self_attn(B=2, S_real=126, S_pad=2, H=8, d_h=64)
    torch.testing.assert_close(
        out,
        ref,
        rtol=2e-3,
        atol=2e-3,
        msg="FA4 self-attn key_padding_mask diverges from unpadded SDPA",
    )


def test_cross_attn_padded_kv_with_mask_matches_unpadded():
    """FA4 cross-attn: padded K/V + mask matches unpadded K/V (matches a2v use)."""
    out, ref = _run_cross_attn(B=2, S_q=320, S_real_kv=126, S_pad_kv=2, H=8, d_h=64)
    torch.testing.assert_close(
        out,
        ref,
        rtol=2e-3,
        atol=2e-3,
        msg="FA4 cross-attn key_padding_mask diverges from unpadded SDPA",
    )


def test_self_attn_pad_junk_values_dont_affect_valid_output():
    """Two different junk pad fills with same mask produce same valid-row output."""
    B, S_real, S_pad, H, d_h = 1, 64, 4, 4, 64
    device = "cuda"
    S_full = S_real + S_pad
    dtype = torch.bfloat16

    torch.manual_seed(7)
    x_valid = torch.randn(B, S_real, H, d_h, dtype=dtype, device=device)
    mask = torch.zeros(B, S_full, dtype=torch.bool, device=device)
    mask[:, :S_real] = True
    fa4 = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)

    # Pad fill A: zeros. Pad fill B: random.
    pad_a = torch.zeros(B, S_pad, H, d_h, dtype=dtype, device=device)
    pad_b = torch.randn(B, S_pad, H, d_h, dtype=dtype, device=device)
    x_a = torch.cat([x_valid, pad_a], dim=1)
    x_b = torch.cat([x_valid, pad_b], dim=1)

    out_a = fa4.forward(q=x_a, k=x_a, v=x_a, key_padding_mask=mask)
    out_b = fa4.forward(q=x_b, k=x_b, v=x_b, key_padding_mask=mask)

    # Q[:S_real] sees the same K/V[:S_real]; mask zeros K/V[S_real:] contribution.
    # Only the valid Q rows are part of the contract.
    torch.testing.assert_close(
        out_a[:, :S_real],
        out_b[:, :S_real],
        rtol=1e-5,
        atol=1e-5,
        msg="FA4 pad fill leaked into valid-row output — mask is not fully suppressing pads",
    )


@pytest.mark.parametrize("num_splits", [0, 8], ids=["auto", "forced"])
def test_split_kv_matches_no_split(num_splits):
    """Splitting K/V must not change FA4's output. Fails to compile on CUTLASS DSL < 4.6.2."""
    torch.manual_seed(0)
    device = "cuda"
    # S_kv must stay well above FA4's `num_n_blocks <= 4` short-circuit for the
    # `auto` case to reach the split-KV kernel; the short S_q keeps occupancy low
    # enough that the heuristic prefers splitting, which is the LTX-2 cross-attn shape.
    B, S_q, S_kv, H, d_h = 1, 64, 4096, 8, 128
    q, k, v = (
        torch.randn(B, s, H, d_h, dtype=torch.bfloat16, device=device) for s in (S_q, S_kv, S_kv)
    )

    def fa4(splits):
        out, _, *_ = _flash_attn_fwd(
            q,
            k,
            v,
            seqused_k=None,
            softmax_scale=d_h**-0.5,
            causal=False,
            window_size_left=None,
            window_size_right=None,
            learnable_sink=None,
            softcap=0.0,
            pack_gqa=None,
            mask_mod=None,
            block_sparse_tensors=None,
            return_lse=True,
            num_splits=splits,
        )
        return out

    # num_splits=1 is the same kernel without the split-KV path, so it is the reference the
    # split result has to reproduce; an SDPA reference would instead tie this test to
    # whichever SDPA backend torch dispatches.
    # 1e-2 is what test_ring_attention uses for FA4 against a reference; measured gap here is
    # 1-2 bf16 ULP (9.8e-4 worst over 20 seeds x both split modes).
    torch.testing.assert_close(
        fa4(num_splits),
        fa4(1),
        rtol=1e-2,
        atol=1e-2,
        msg=f"FA4 num_splits={num_splits} diverges from the non-split result",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

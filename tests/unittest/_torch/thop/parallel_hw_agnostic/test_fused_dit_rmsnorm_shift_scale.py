# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the LTX-2 fused RMSNorm + AdaLN affine-modulation kernel.
#
# bf16 variant:   fused_dit_rmsnorm_shift_scale
# NVFP4 variant:  fused_dit_rmsnorm_shift_scale_quant
#
# Signature: x, scale_table (fp32 [D]), scale_ts (bf16 [B, T, D]),
#               shift_table (fp32 [D]), shift_ts (bf16 [B, T, D]), [sf_scale,] eps
# Kernel composes the modulator inline: scale[d] = scale_table[d] + scale_ts[b, d].

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401  -- triggers libth_common.so load (registers trtllm ops)


@torch.inference_mode()
def torch_ref(x_2d, scale_table, scale_ts, shift_table, shift_ts, tokens_per_batch, eps):
    """Reference matching production semantics (apply_fused_adaln_modulate
    eager fallback in utils_ltx2.py + _get_ada_values modulator combine):

    Phase 0b: scale_bf16 = scale_table.to(bf16) + scale_ts  (bf16 narrow, bf16 hw add)
    Phase 1:  normed     = rms_norm(x)                      (bf16 in / out)
    Phase 2:  out        = normed * (1 + scale_bf16) + shift_bf16   (all bf16)
    """
    B = scale_ts.shape[0]
    D = x_2d.shape[-1]
    x_3d = x_2d.view(B, tokens_per_batch, D)
    normed = torch.nn.functional.rms_norm(x_3d, (D,), weight=None, eps=eps)
    scale_bf16 = scale_table.to(torch.bfloat16).view(1, 1, D) + scale_ts.view(B, 1, D)
    shift_bf16 = shift_table.to(torch.bfloat16).view(1, 1, D) + shift_ts.view(B, 1, D)
    out = normed * (1 + scale_bf16) + shift_bf16
    return out.view(-1, D)


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_batch", [1, 16, 512])
def test_rmsnorm_shift_scale_shapes(hidden_dim, batch_size, tokens_per_batch):
    device = "cuda"
    torch.random.manual_seed(42)

    num_tokens = batch_size * tokens_per_batch
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
    scale_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.5
    scale_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.5
    shift_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.5
    shift_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.5
    eps = 1e-6

    out = torch.ops.trtllm.fused_dit_rmsnorm_shift_scale(
        x, scale_table, scale_ts, shift_table, shift_ts, eps
    )
    ref = torch_ref(x, scale_table, scale_ts, shift_table, shift_ts, tokens_per_batch, eps)
    # Kernel uses pure bf16 hw `__hadd2`/`__hmul2`; PyTorch eager uses bf16
    # with fp32-accumulator internally. Diff is ~1 bf16 ULP per element on
    # rare tail samples (max abs ~0.012 at value ~1). Cosine vs eager >0.999.
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_rmsnorm_shift_scale_ltx2_video_shape():
    """LTX-2 video: B=1, S=12288, D=4096."""
    device = "cuda"
    torch.random.manual_seed(0)
    B, S, D = 1, 12288, 4096
    x = torch.randn(B * S, D, dtype=torch.bfloat16, device=device)
    scale_table = torch.randn(D, dtype=torch.float32, device=device) * 0.5
    scale_ts = torch.randn(B, D, dtype=torch.bfloat16, device=device) * 0.5
    shift_table = torch.randn(D, dtype=torch.float32, device=device) * 0.5
    shift_ts = torch.randn(B, D, dtype=torch.bfloat16, device=device) * 0.5

    out = torch.ops.trtllm.fused_dit_rmsnorm_shift_scale(
        x, scale_table, scale_ts, shift_table, shift_ts, 1e-6
    )
    ref = torch_ref(x, scale_table, scale_ts, shift_table, shift_ts, S, 1e-6)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_rmsnorm_shift_scale_ltx2_audio_shape():
    """LTX-2 audio: B=1, S=504, D=2048."""
    device = "cuda"
    torch.random.manual_seed(0)
    B, S, D = 1, 504, 2048
    x = torch.randn(B * S, D, dtype=torch.bfloat16, device=device)
    scale_table = torch.randn(D, dtype=torch.float32, device=device) * 0.5
    scale_ts = torch.randn(B, D, dtype=torch.bfloat16, device=device) * 0.5
    shift_table = torch.randn(D, dtype=torch.float32, device=device) * 0.5
    shift_ts = torch.randn(B, D, dtype=torch.bfloat16, device=device) * 0.5

    out = torch.ops.trtllm.fused_dit_rmsnorm_shift_scale(
        x, scale_table, scale_ts, shift_table, shift_ts, 1e-6
    )
    ref = torch_ref(x, scale_table, scale_ts, shift_table, shift_ts, S, 1e-6)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_rmsnorm_shift_scale_rejects_non_contiguous():
    device = "cuda"
    B, S, D = 2, 16, 4096
    base = torch.randn(B * S, 2 * D, dtype=torch.bfloat16, device=device)
    x_view = base[:, :D]
    assert not x_view.is_contiguous()
    scale_table = torch.randn(D, dtype=torch.float32, device=device)
    scale_ts = torch.randn(B, D, dtype=torch.bfloat16, device=device)
    shift_table = torch.randn(D, dtype=torch.float32, device=device)
    shift_ts = torch.randn(B, D, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match=r"contiguous"):
        torch.ops.trtllm.fused_dit_rmsnorm_shift_scale(
            x_view, scale_table, scale_ts, shift_table, shift_ts, 1e-6
        )


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("K", [5, 6])
def test_rmsnorm_shift_scale_strided_ts(hidden_dim, batch_size, K):
    """scale_ts / shift_ts may be unbind views of a [B, T_t, K, D] timestep tensor --
    row stride is K*D, inner stride is 1. Kernel must produce identical output."""
    device = "cuda"
    torch.random.manual_seed(7)
    D = hidden_dim
    T_t = 1
    tokens_per_batch = 16
    num_tokens = batch_size * tokens_per_batch

    x = torch.randn(num_tokens, D, dtype=torch.bfloat16, device=device)
    ts = torch.randn(batch_size, T_t, K, D, dtype=torch.bfloat16, device=device) * 0.5
    scale_table = torch.randn(D, dtype=torch.float32, device=device) * 0.5
    shift_table = torch.randn(D, dtype=torch.float32, device=device) * 0.5

    chunks = ts.unbind(dim=2)
    scale_ts_view, shift_ts_view = chunks[0], chunks[1]
    # Layout check: unbind dim=2 of a [B, T_t, K, D] contig tensor yields views with
    # strides (T_t*K*D, K*D, 1). is_contiguous() is True for shapes with size-1 dims
    # (B=T_t=1) since PyTorch ignores their strides; check stride pattern directly.
    assert scale_ts_view.stride() == (T_t * K * D, K * D, 1)
    if batch_size > 1:
        assert not scale_ts_view.squeeze(1).is_contiguous()

    out_strided = torch.ops.trtllm.fused_dit_rmsnorm_shift_scale(
        x, scale_table, scale_ts_view.squeeze(1), shift_table, shift_ts_view.squeeze(1), 1e-6
    )
    out_contig = torch.ops.trtllm.fused_dit_rmsnorm_shift_scale(
        x,
        scale_table,
        scale_ts_view.squeeze(1).contiguous(),
        shift_table,
        shift_ts_view.squeeze(1).contiguous(),
        1e-6,
    )
    torch.testing.assert_close(out_strided, out_contig, rtol=0.0, atol=0.0)


def test_rmsnorm_shift_scale_rejects_unsupported_dim():
    device = "cuda"
    D = 1024  # not in {2048, 4096}
    x = torch.randn(8, D, dtype=torch.bfloat16, device=device)
    scale_table = torch.randn(D, dtype=torch.float32, device=device)
    scale_ts = torch.randn(1, D, dtype=torch.bfloat16, device=device)
    shift_table = torch.randn(D, dtype=torch.float32, device=device)
    shift_ts = torch.randn(1, D, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fused_dit_rmsnorm_shift_scale(
            x, scale_table, scale_ts, shift_table, shift_ts, 1e-6
        )


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_batch", [126, 128])
def test_rmsnorm_shift_scale_fp4_quant(hidden_dim, batch_size, tokens_per_batch):
    """NVFP4 variant: end-to-end GEMM quality (cosine > 0.98) vs the bf16
    F.linear reference, matching the canonical FP4 test pattern."""
    device = "cuda"
    torch.manual_seed(7)
    num_tokens = batch_size * tokens_per_batch
    out_dim = 1024
    eps = 1e-6

    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1
    scale_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.3
    scale_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.3
    shift_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.3
    shift_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.3
    W = torch.randn(out_dim, hidden_dim, dtype=torch.bfloat16, device=device) * 0.05

    # bf16 reference: fused-kernel bf16 output -> F.linear with random W.
    out_bf16 = torch.ops.trtllm.fused_dit_rmsnorm_shift_scale(
        x, scale_table, scale_ts, shift_table, shift_ts, eps
    )
    D_ref = F.linear(out_bf16, W)

    sf_scale_x = (448.0 * 6.0) / out_bf16.abs().max().float()
    sf_scale_w = (448.0 * 6.0) / W.abs().max().float()

    # FP4 path: packed FP4 + SWIZZLED 1x16 FP8 SF directly.
    out_fp4, out_sf = torch.ops.trtllm.fused_dit_rmsnorm_shift_scale_quant(
        x, scale_table, scale_ts, shift_table, shift_ts, sf_scale_x, eps
    )

    # Weight quantize: trtllm.fp4_quantize defaults (sfUseUE8M0=False,
    # isSfSwizzledLayout=True) match the quant kernel's SWIZZLED SF layout.
    W_fp4, W_sf = torch.ops.trtllm.fp4_quantize(W, sf_scale_w, 16)

    alpha = 1.0 / (sf_scale_x * sf_scale_w).float()
    C = torch.ops.trtllm.nvfp4_gemm(out_fp4, W_fp4, out_sf, W_sf, alpha, torch.bfloat16)

    assert F.cosine_similarity(C.flatten().float(), D_ref.flatten().float(), dim=0).item() > 0.98

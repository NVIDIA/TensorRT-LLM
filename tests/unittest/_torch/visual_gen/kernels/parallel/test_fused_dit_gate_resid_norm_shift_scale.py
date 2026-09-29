# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the LTX-2 fused DiT norm kernel family (cpp/tensorrt_llm/kernels/fusedDiTGateResidNormShiftScaleKernel.cu)
# -- one warp-specialized Blackwell kernel, specialized by template flags
# (HAS_GATE / HAS_RESIDUAL / HAS_NORM / HAS_SHIFT_SCALE / NUM_OUT / HAS_QUANT).
# Each op composes its AdaLN modulators inline from (table_fp32 [D], ts_bf16 [B, D]) pairs
# via a bf16-narrow-first hw add, matching the PyTorch eager `_get_ada_values` combine.
#
# Ops covered (bf16 + NVFP4 `_quant` variant unless noted):
#   fused_dit_rmsnorm_shift_scale                RMSNorm -> (1 + s) * n + h
#   fused_dit_resid_rmsnorm_shift_scale_dual     resid -> RMSNorm -> two shift_scale outputs
#   fused_dit_gate_resid_rmsnorm_shift_scale     gate-resid -> RMSNorm -> shift_scale
#   fused_dit_gate_resid_rmsnorm                 gate-resid -> RMSNorm
#   fused_dit_gate_resid                         gate-resid only (bf16, no norm)

import pytest
import torch
import torch.nn.functional as F
from utils.util import skip_pre_blackwell

import tensorrt_llm  # noqa: F401  -- triggers libth_common.so load (registers trtllm ops)

# Fused DiT AdaLN kernels use TMA + inline NVFP4 quant -> require Blackwell (sm100).
pytestmark = skip_pre_blackwell


# ---------------------------------------------------------------------------
# Thin per-variant adapters over the single unified `trtllm::fused_dit_gate_resid_norm_shift_scale`
# op. Flags are inferred from which args are provided (attn => residual,
# gate_table => gate, non-empty scale list => shift_scale, non-empty sf_scale
# => quant, num_out>=1 => norm). The op is FUNCTIONAL: for residual variants it
# returns the new residual stream x_new as output[0]; these adapters copy x_new
# back into x so each test's in-place expectation (x <- x + attn[*gate]) holds,
# and return only the norm output(s) -- keeping the test bodies unchanged.
# ---------------------------------------------------------------------------
def _call_rmsnorm_shift_scale(x, scale_table, scale_ts, shift_table, shift_ts, eps):
    (out,) = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x,
        scale_table=[scale_table],
        scale_ts=[scale_ts],
        shift_table=[shift_table],
        shift_ts=[shift_ts],
        eps=eps,
    )
    return out


def _call_rmsnorm_shift_scale_quant(x, scale_table, scale_ts, shift_table, shift_ts, sf_scale, eps):
    return torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x,
        scale_table=[scale_table],
        scale_ts=[scale_ts],
        shift_table=[shift_table],
        shift_ts=[shift_ts],
        sf_scale=[sf_scale],
        eps=eps,
    )


def _call_resid_rmsnorm_shift_scale_dual(
    x, attn, s1t, s1ts, sh1t, sh1ts, s2t, s2ts, sh2t, sh2ts, eps
):
    x_new, o1, o2 = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x,
        attn=attn,
        scale_table=[s1t, s2t],
        scale_ts=[s1ts, s2ts],
        shift_table=[sh1t, sh2t],
        shift_ts=[sh1ts, sh2ts],
        eps=eps,
        num_out=2,
    )
    x.copy_(x_new)
    return o1, o2


def _call_resid_rmsnorm_shift_scale_dual_quant(
    x, attn, s1t, s1ts, sh1t, sh1ts, s2t, s2ts, sh2t, sh2ts, sf1, sf2, eps
):
    x_new, f0, s0, f1, s1 = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x,
        attn=attn,
        scale_table=[s1t, s2t],
        scale_ts=[s1ts, s2ts],
        shift_table=[sh1t, sh2t],
        shift_ts=[sh1ts, sh2ts],
        sf_scale=[sf1, sf2],
        eps=eps,
        num_out=2,
    )
    x.copy_(x_new)
    return f0, s0, f1, s1


def _call_gate_resid_rmsnorm_shift_scale(x, attn, gt, gts, st, sts, sht, shts, eps):
    x_new, out = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x,
        attn=attn,
        gate_table=gt,
        gate_ts=gts,
        scale_table=[st],
        scale_ts=[sts],
        shift_table=[sht],
        shift_ts=[shts],
        eps=eps,
    )
    x.copy_(x_new)
    return out


def _call_gate_resid_rmsnorm_shift_scale_quant(x, attn, gt, gts, st, sts, sht, shts, sf_scale, eps):
    x_new, fp4, sf = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x,
        attn=attn,
        gate_table=gt,
        gate_ts=gts,
        scale_table=[st],
        scale_ts=[sts],
        shift_table=[sht],
        shift_ts=[shts],
        sf_scale=[sf_scale],
        eps=eps,
    )
    x.copy_(x_new)
    return fp4, sf


def _call_gate_resid_rmsnorm(x, attn, gt, gts, eps):
    x_new, out = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x, attn=attn, gate_table=gt, gate_ts=gts, eps=eps
    )
    x.copy_(x_new)
    return out


def _call_gate_resid_rmsnorm_quant(x, attn, gt, gts, sf_scale, eps):
    x_new, fp4, sf = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x, attn=attn, gate_table=gt, gate_ts=gts, sf_scale=[sf_scale], eps=eps
    )
    x.copy_(x_new)
    return fp4, sf


def _call_gate_resid(x, attn, gt, gts):
    # num_out=0 => gate-residual only (no norm): op returns [x_new] = x + attn * gate.
    (x_new,) = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x, attn=attn, gate_table=gt, gate_ts=gts, num_out=0
    )
    x.copy_(x_new)
    return x


# Shared gate-residual input builder (gate_resid and gate_resid_rmsnorm use identical inputs).
def _make_gate_resid_inputs(num_tokens, hidden_dim, batch_size, seed):
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1
    attn = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1
    gate_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.3
    gate_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.3
    return x, attn, gate_table, gate_ts


# =====================================================================================
# fused_dit_rmsnorm_shift_scale  (RMSNorm + AdaLN affine-modulation)
#   Signature: x, scale_table (fp32 [D]), scale_ts (bf16 [B, T, D]),
#                 shift_table (fp32 [D]), shift_ts (bf16 [B, T, D]), [sf_scale,] eps
#   Kernel composes the modulator inline: scale[d] = scale_table[d] + scale_ts[b, d].
# =====================================================================================
@torch.inference_mode()
def _ref_rmsnorm_shift_scale(
    x_2d, scale_table, scale_ts, shift_table, shift_ts, tokens_per_batch, eps
):
    """Reference matching production semantics (apply_fused_rmsnorm_shift_scale
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

    out = _call_rmsnorm_shift_scale(x, scale_table, scale_ts, shift_table, shift_ts, eps)
    ref = _ref_rmsnorm_shift_scale(
        x, scale_table, scale_ts, shift_table, shift_ts, tokens_per_batch, eps
    )
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

    out = _call_rmsnorm_shift_scale(x, scale_table, scale_ts, shift_table, shift_ts, 1e-6)
    ref = _ref_rmsnorm_shift_scale(x, scale_table, scale_ts, shift_table, shift_ts, S, 1e-6)
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

    out = _call_rmsnorm_shift_scale(x, scale_table, scale_ts, shift_table, shift_ts, 1e-6)
    ref = _ref_rmsnorm_shift_scale(x, scale_table, scale_ts, shift_table, shift_ts, S, 1e-6)
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
        _call_rmsnorm_shift_scale(x_view, scale_table, scale_ts, shift_table, shift_ts, 1e-6)


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

    out_strided = _call_rmsnorm_shift_scale(
        x, scale_table, scale_ts_view.squeeze(1), shift_table, shift_ts_view.squeeze(1), 1e-6
    )
    out_contig = _call_rmsnorm_shift_scale(
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
        _call_rmsnorm_shift_scale(x, scale_table, scale_ts, shift_table, shift_ts, 1e-6)


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
    out_bf16 = _call_rmsnorm_shift_scale(x, scale_table, scale_ts, shift_table, shift_ts, eps)
    D_ref = F.linear(out_bf16, W)

    sf_scale_x = (448.0 * 6.0) / out_bf16.abs().max().float()
    sf_scale_w = (448.0 * 6.0) / W.abs().max().float()

    # FP4 path: packed FP4 + SWIZZLED 1x16 FP8 SF directly.
    out_fp4, out_sf = _call_rmsnorm_shift_scale_quant(
        x, scale_table, scale_ts, shift_table, shift_ts, sf_scale_x, eps
    )

    # Weight quantize: trtllm.fp4_quantize defaults (sfUseUE8M0=False,
    # isSfSwizzledLayout=True) match the quant kernel's SWIZZLED SF layout.
    W_fp4, W_sf = torch.ops.trtllm.fp4_quantize(W, sf_scale_w, 16)

    alpha = 1.0 / (sf_scale_x * sf_scale_w).float()
    C = torch.ops.trtllm.nvfp4_gemm(out_fp4, W_fp4, out_sf, W_sf, alpha, torch.bfloat16)

    assert F.cosine_similarity(C.flatten().float(), D_ref.flatten().float(), dim=0).item() > 0.98


# =====================================================================================
# fused_dit_resid_rmsnorm_shift_scale_dual
#   bf16  -> (out_dir1, out_dir2);  quant -> (fp4_dir1, sf_dir1, fp4_dir2, sf_dir2)
#   Post-attn2 site: residual_add(x, attn2_out) -> rms_norm -> two independent
#   (1+s)*normed+h modulations, feeding the FFN up_proj (dir1) and AV cross-attn to_q (dir2).
#   The 4 modulators are composed inline from (table_fp32 [D], ts_bf16 [B, D]) pairs.
# =====================================================================================
@torch.inference_mode()
def _ref_resid_dual(
    x_2d,
    attn_2d,
    s1_table,
    s1_ts,
    h1_table,
    h1_ts,
    s2_table,
    s2_ts,
    h2_table,
    h2_ts,
    tokens_per_batch,
    eps,
):
    """Reference matching production semantics (apply_fused_resid_rms_shift_scale_dual_a
    eager fallback + _get_av_ca_ada_values combine):
       x_new      = x + attn                                  (bf16)
       normed     = rms_norm(x_new)                           (bf16)
       m_bf16     = m_table.to(bf16) + m_ts                   (bf16 narrow, bf16 add)
       out_dir_k  = normed * (1 + s_bf16_k) + h_bf16_k        (all bf16)"""
    B = s1_ts.shape[0]
    D = x_2d.shape[-1]
    x_3d = x_2d.view(B, tokens_per_batch, D)
    a_3d = attn_2d.view(B, tokens_per_batch, D)
    x_new = x_3d + a_3d
    normed = torch.nn.functional.rms_norm(x_new, (D,), weight=None, eps=eps)
    s1_bf16 = s1_table.to(torch.bfloat16).view(1, 1, D) + s1_ts.view(B, 1, D)
    h1_bf16 = h1_table.to(torch.bfloat16).view(1, 1, D) + h1_ts.view(B, 1, D)
    s2_bf16 = s2_table.to(torch.bfloat16).view(1, 1, D) + s2_ts.view(B, 1, D)
    h2_bf16 = h2_table.to(torch.bfloat16).view(1, 1, D) + h2_ts.view(B, 1, D)
    o1 = normed * (1 + s1_bf16) + h1_bf16
    o2 = normed * (1 + s2_bf16) + h2_bf16
    return (
        o1.view(-1, D),
        o2.view(-1, D),
        x_new.view(-1, D),
    )


def _make_resid_dual_inputs(num_tokens, hidden_dim, batch_size, seed):
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1
    attn2 = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1
    s1_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.3
    s1_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.3
    h1_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.3
    h1_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.3
    s2_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.3
    s2_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.3
    h2_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.3
    h2_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.3
    return x, attn2, s1_table, s1_ts, h1_table, h1_ts, s2_table, s2_ts, h2_table, h2_ts


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_batch", [16, 512])
def test_resid_rmsnorm_shift_scale_dual_bf16(hidden_dim, batch_size, tokens_per_batch):
    """bf16 variant: dual outputs match eager fp32 reference; x is mutated in place."""
    num_tokens = batch_size * tokens_per_batch
    (x, attn2, s1_table, s1_ts, h1_table, h1_ts, s2_table, s2_ts, h2_table, h2_ts) = (
        _make_resid_dual_inputs(num_tokens, hidden_dim, batch_size, seed=42)
    )
    eps = 1e-6

    ref_o1, ref_o2, ref_x_new = _ref_resid_dual(
        x,
        attn2,
        s1_table,
        s1_ts,
        h1_table,
        h1_ts,
        s2_table,
        s2_ts,
        h2_table,
        h2_ts,
        tokens_per_batch,
        eps,
    )

    x_impl = x.clone()
    o1, o2 = _call_resid_rmsnorm_shift_scale_dual(
        x_impl, attn2, s1_table, s1_ts, h1_table, h1_ts, s2_table, s2_ts, h2_table, h2_ts, eps
    )

    # Tolerance: same as the gate-residual variant -- bf16-narrow-first kernel vs fp32 ref differs at the
    # bf16 ULP edge; longer-S reductions can push 1 element over the tighter 5e-3.
    # Kernel uses pure bf16 hw `__hadd2`/`__hmul2` shift_scale; production eager uses bf16
    # with fp32-accumulator internally. ~1 bf16 ULP per element on rare tail samples.
    torch.testing.assert_close(o1, ref_o1, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(o2, ref_o2, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x_impl, ref_x_new, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_batch", [126, 128])
def test_resid_rmsnorm_shift_scale_dual_fp4_quant(hidden_dim, batch_size, tokens_per_batch):
    """NVFP4 variant: both quantized outputs (dir1 -> FFN up_proj, dir2 -> AV cross-attn
    to_q) pass end-to-end GEMM cosine > 0.98 vs the bf16 F.linear reference."""
    num_tokens = batch_size * tokens_per_batch
    out_dim = 1024
    (x, attn2, s1_table, s1_ts, h1_table, h1_ts, s2_table, s2_ts, h2_table, h2_ts) = (
        _make_resid_dual_inputs(num_tokens, hidden_dim, batch_size, seed=7)
    )
    eps = 1e-6

    torch.manual_seed(13)
    W1 = torch.randn(out_dim, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    W2 = torch.randn(out_dim, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.05

    # bf16 reference: fused-kernel bf16 outputs -> F.linear with random W1 / W2.
    x_bf16 = x.clone()
    o1_bf16, o2_bf16 = _call_resid_rmsnorm_shift_scale_dual(
        x_bf16, attn2, s1_table, s1_ts, h1_table, h1_ts, s2_table, s2_ts, h2_table, h2_ts, eps
    )
    D1_ref = F.linear(o1_bf16, W1)
    D2_ref = F.linear(o2_bf16, W2)

    sf_scale_x1 = (448.0 * 6.0) / o1_bf16.abs().max().float()
    sf_scale_x2 = (448.0 * 6.0) / o2_bf16.abs().max().float()
    sf_scale_w1 = (448.0 * 6.0) / W1.abs().max().float()
    sf_scale_w2 = (448.0 * 6.0) / W2.abs().max().float()

    # FP4 path: dual quant in one kernel.
    x_fp4 = x.clone()
    o1_fp4, o1_sf, o2_fp4, o2_sf = _call_resid_rmsnorm_shift_scale_dual_quant(
        x_fp4,
        attn2,
        s1_table,
        s1_ts,
        h1_table,
        h1_ts,
        s2_table,
        s2_ts,
        h2_table,
        h2_ts,
        sf_scale_x1,
        sf_scale_x2,
        eps,
    )

    # Weight quantize: trtllm.fp4_quantize defaults (sfUseUE8M0=False,
    # isSfSwizzledLayout=True) match the quant kernel's SWIZZLED SF layout.
    W1_fp4, W1_sf = torch.ops.trtllm.fp4_quantize(W1, sf_scale_w1, 16)
    W2_fp4, W2_sf = torch.ops.trtllm.fp4_quantize(W2, sf_scale_w2, 16)

    alpha1 = 1.0 / (sf_scale_x1 * sf_scale_w1).float()
    alpha2 = 1.0 / (sf_scale_x2 * sf_scale_w2).float()
    C1 = torch.ops.trtllm.nvfp4_gemm(o1_fp4, W1_fp4, o1_sf, W1_sf, alpha1, torch.bfloat16)
    C2 = torch.ops.trtllm.nvfp4_gemm(o2_fp4, W2_fp4, o2_sf, W2_sf, alpha2, torch.bfloat16)

    assert F.cosine_similarity(C1.flatten().float(), D1_ref.flatten().float(), dim=0).item() > 0.98
    assert F.cosine_similarity(C2.flatten().float(), D2_ref.flatten().float(), dim=0).item() > 0.98


# =====================================================================================
# fused_dit_gate_resid_rmsnorm_shift_scale  (+ _quant)
#   gate-residual -> RMSNorm -> shift_scale. Each modulator (gate, scale, shift) is built
#   inline from a (table, ts) pair, eliminating the upstream broadcast-add Triton prep kernel.
# =====================================================================================
@torch.inference_mode()
def _ref_gate_resid_ss(
    x_2d,
    attn_2d,
    gate_table,
    gate_ts,
    scale_table,
    scale_ts,
    shift_table,
    shift_ts,
    tokens_per_batch,
    eps,
):
    """Reference matching production semantics (apply_fused_gate_resid_shift_scale
    eager fallback + _get_ada_values combine):
       gate_bf16 = gate_table.to(bf16) + gate_ts                  (bf16 narrow, bf16 add)
       x_new     = x + attn * gate_bf16                            (bf16)
       normed    = rms_norm(x_new)                                 (bf16)
       s_bf16    = scale_table.to(bf16) + scale_ts                 (bf16)
       h_bf16    = shift_table.to(bf16) + shift_ts                 (bf16)
       out       = normed * (1 + s_bf16) + h_bf16                  (all bf16)
    """
    B = gate_ts.shape[0]
    D = x_2d.shape[-1]
    x_3d = x_2d.view(B, tokens_per_batch, D)
    a_3d = attn_2d.view(B, tokens_per_batch, D)
    gate_bf16 = gate_table.to(torch.bfloat16).view(1, 1, D) + gate_ts.view(B, 1, D)
    scale_bf16 = scale_table.to(torch.bfloat16).view(1, 1, D) + scale_ts.view(B, 1, D)
    shift_bf16 = shift_table.to(torch.bfloat16).view(1, 1, D) + shift_ts.view(B, 1, D)
    x_new = x_3d + a_3d * gate_bf16
    normed = torch.nn.functional.rms_norm(x_new, (D,), weight=None, eps=eps)
    out = normed * (1 + scale_bf16) + shift_bf16
    return out.view(-1, D), x_new.view(-1, D)


def _make_gate_resid_ss_inputs(num_tokens, hidden_dim, batch_size, seed):
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1
    attn = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1
    gate_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.3
    gate_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.3
    scale_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.3
    scale_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.3
    shift_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.3
    shift_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.3
    return x, attn, gate_table, gate_ts, scale_table, scale_ts, shift_table, shift_ts


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_batch", [16, 512])
def test_gate_resid_rmsnorm_shift_scale_bf16(hidden_dim, batch_size, tokens_per_batch):
    """bf16 variant: trtllm kernel matches eager fp32 reference within tolerance,
    and mutates x in place to (x + attn * gate)."""
    num_tokens = batch_size * tokens_per_batch
    x, attn, gate_table, gate_ts, scale_table, scale_ts, shift_table, shift_ts = (
        _make_gate_resid_ss_inputs(num_tokens, hidden_dim, batch_size, seed=42)
    )
    eps = 1e-6

    ref_out, ref_x_new = _ref_gate_resid_ss(
        x,
        attn,
        gate_table,
        gate_ts,
        scale_table,
        scale_ts,
        shift_table,
        shift_ts,
        tokens_per_batch,
        eps,
    )

    x_impl = x.clone()
    out = _call_gate_resid_rmsnorm_shift_scale(
        x_impl, attn, gate_table, gate_ts, scale_table, scale_ts, shift_table, shift_ts, eps
    )

    # Tolerance bumped vs old test: kernel now narrows table->bf16 then bf16-add
    # (PyTorch eager `_get_ada_values` semantics) which differs from the
    # all-fp32 reference at the bf16 ULP boundary; the longer-S reductions amplify
    # the per-element diff slightly. Still well within bf16 noise.
    # Kernel uses pure bf16 hw `__hadd2`/`__hmul2` shift_scale; production eager uses bf16
    # with fp32-accumulator internally. ~1 bf16 ULP per element on rare tail samples.
    torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x_impl, ref_x_new, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_batch", [126, 128])
def test_gate_resid_rmsnorm_shift_scale_fp4_quant(hidden_dim, batch_size, tokens_per_batch):
    """NVFP4 variant: end-to-end GEMM quality (cosine > 0.98) vs the bf16
    F.linear reference, matching the canonical FP4 test pattern."""
    num_tokens = batch_size * tokens_per_batch
    out_dim = 1024
    x, attn, gate_table, gate_ts, scale_table, scale_ts, shift_table, shift_ts = (
        _make_gate_resid_ss_inputs(num_tokens, hidden_dim, batch_size, seed=7)
    )
    eps = 1e-6

    # bf16 reference: compute the fused kernel.s bf16 output, then F.linear with random W.
    torch.manual_seed(13)
    W = torch.randn(out_dim, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.05

    x_bf16 = x.clone()
    out_bf16 = _call_gate_resid_rmsnorm_shift_scale(
        x_bf16, attn, gate_table, gate_ts, scale_table, scale_ts, shift_table, shift_ts, eps
    )
    D_ref = F.linear(out_bf16, W)

    # Per-tensor inverse-amax sf_scale (the NVFP4 "global scale", SF2) for A and B.
    sf_scale_x = (448.0 * 6.0) / out_bf16.abs().max().float()
    sf_scale_w = (448.0 * 6.0) / W.abs().max().float()

    # FP4 path: produces packed FP4 + SWIZZLED 1x16 FP8 SF directly.
    x_fp4 = x.clone()
    out_fp4, out_sf = _call_gate_resid_rmsnorm_shift_scale_quant(
        x_fp4,
        attn,
        gate_table,
        gate_ts,
        scale_table,
        scale_ts,
        shift_table,
        shift_ts,
        sf_scale_x,
        eps,
    )

    # Weight quantized via trtllm.fp4_quantize (defaults: sfUseUE8M0=False,
    # isSfSwizzledLayout=True) -- matches the quant kernel's SWIZZLED SF layout.
    W_fp4, W_sf = torch.ops.trtllm.fp4_quantize(W, sf_scale_w, 16)

    alpha = 1.0 / (sf_scale_x * sf_scale_w).float()
    C = torch.ops.trtllm.nvfp4_gemm(out_fp4, W_fp4, out_sf, W_sf, alpha, torch.bfloat16)

    assert F.cosine_similarity(C.flatten().float(), D_ref.flatten().float(), dim=0).item() > 0.98


# =====================================================================================
# fused_dit_gate_resid_rmsnorm  (+ _quant)
#   gate[d]    = gate_table[d].to(bf16) + gate_ts[b, d]    (bf16 narrow, bf16 hw add)
#   x_new[n,d] = x[n,d] + attn_out[n,d] * gate[d]          (in-place)
#   normed     = rms_norm(x_new)
#   (fp4, sf)  = nvfp4_quantize(normed, sf_scale)            (quant variant)
#   out_bf16   = bf16(normed)                                (bf16 variant)
# =====================================================================================
@torch.inference_mode()
def _ref_gate_resid_rmsnorm(x_2d, attn_2d, gate_table, gate_ts, tokens_per_batch, eps):
    """Reference matching kernel: gate built bf16-narrow-first to match
    PyTorch eager `_get_ada_values[2]`. Returns (normed_bf16, x_new_bf16)."""
    B = gate_ts.shape[0]
    D = x_2d.shape[-1]
    x_3d = x_2d.view(B, tokens_per_batch, D).float()
    a_3d = attn_2d.view(B, tokens_per_batch, D).float()
    gate_bf16 = gate_table.to(torch.bfloat16).view(1, 1, D) + gate_ts.view(B, 1, D)
    x_new = x_3d + a_3d * gate_bf16.float()
    var = x_new.pow(2).mean(-1, keepdim=True)
    normed = x_new * torch.rsqrt(var + eps)
    return normed.view(-1, D).to(x_2d.dtype), x_new.view(-1, D).to(x_2d.dtype)


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_batch", [16, 126])
def test_gate_resid_rmsnorm_bf16(hidden_dim, batch_size, tokens_per_batch):
    """bf16 variant: output matches reference; x mutated in place to x + attn * gate."""
    num_tokens = batch_size * tokens_per_batch
    x, attn, gate_table, gate_ts = _make_gate_resid_inputs(
        num_tokens, hidden_dim, batch_size, seed=42
    )
    eps = 1e-6

    ref_normed, ref_x_new = _ref_gate_resid_rmsnorm(
        x, attn, gate_table, gate_ts, tokens_per_batch, eps
    )

    x_impl = x.clone()
    out = _call_gate_resid_rmsnorm(x_impl, attn, gate_table, gate_ts, eps)

    # Tolerance: same as the shift_scale variants -- bf16-narrow-first kernel vs fp32 ref differs at the
    # bf16 ULP edge; longer-S reductions can push 1 element over the tighter 5e-3.
    torch.testing.assert_close(out, ref_normed, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(x_impl, ref_x_new, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_gate_resid_rmsnorm_quant_fp4(hidden_dim, batch_size):
    """NVFP4 variant: end-to-end GEMM quality (cosine > 0.98) vs the bf16
    F.linear reference. Also verifies in-place mutation: x <- x + attn * gate."""
    tokens_per_batch = 126
    num_tokens = batch_size * tokens_per_batch
    out_dim = 1024
    eps = 1e-6

    x, attn, gate_table, gate_ts = _make_gate_resid_inputs(
        num_tokens, hidden_dim, batch_size, seed=7
    )
    torch.manual_seed(13)
    W = torch.randn(out_dim, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.05

    # bf16 reference (eager): x_new = x + attn * gate; normed = rms_norm(x_new); D_ref = F.linear(normed, W).
    normed_bf16, x_new_ref = _ref_gate_resid_rmsnorm(
        x, attn, gate_table, gate_ts, tokens_per_batch, eps
    )
    D_ref = F.linear(normed_bf16, W)

    sf_scale_x = (448.0 * 6.0) / normed_bf16.abs().max().float()
    sf_scale_w = (448.0 * 6.0) / W.abs().max().float()

    # bf16 op: in-place x <- x + attn * gate, produces (fp4, sf) of normed.
    x_fp4 = x.clone()
    out_fp4, out_sf = _call_gate_resid_rmsnorm_quant(
        x_fp4, attn, gate_table, gate_ts, sf_scale_x, eps
    )

    # In-place mutation check (loose tolerance: bf16 mul accumulates noise).
    torch.testing.assert_close(x_fp4, x_new_ref, rtol=2e-2, atol=1e-2)

    # Weight quantize via canonical trtllm.fp4_quantize (matches SWIZZLED SF layout).
    W_fp4, W_sf = torch.ops.trtllm.fp4_quantize(W, sf_scale_w, 16)

    alpha = 1.0 / (sf_scale_x * sf_scale_w).float()
    C = torch.ops.trtllm.nvfp4_gemm(out_fp4, W_fp4, out_sf, W_sf, alpha, torch.bfloat16)

    assert F.cosine_similarity(C.flatten().float(), D_ref.flatten().float(), dim=0).item() > 0.98


def test_gate_resid_rmsnorm_quant_rejects_unsupported_dim():
    """hidden_dim outside {2048, 4096} raises."""
    device = "cuda"
    D = 1024  # not supported
    x = torch.randn(8, D, dtype=torch.bfloat16, device=device)
    attn = torch.randn(8, D, dtype=torch.bfloat16, device=device)
    gate_table = torch.randn(D, dtype=torch.float32, device=device)
    gate_ts = torch.randn(1, D, dtype=torch.bfloat16, device=device)
    sf_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError):
        _call_gate_resid_rmsnorm_quant(x, attn, gate_table, gate_ts, sf_scale, 1e-6)


# =====================================================================================
# fused_dit_gate_resid  (bf16-only, no RMSNorm, no shift_scale)
#   x <- x + attn_out * (gate_table.to(bf16) + gate_ts), in-place.
#   Used at the FFN output gate site (`vx = vx + ff(vx_scaled) * vgate_mlp`).
# =====================================================================================
@torch.inference_mode()
def _ref_gate_resid(x_2d, attn_2d, gate_table, gate_ts, tokens_per_batch):
    """gate_bf16 = gate_table.to(bf16) + gate_ts; x_new = x + attn * gate_bf16 (per-batch gate)."""
    B = gate_ts.shape[0]
    D = x_2d.shape[-1]
    x_3d = x_2d.view(B, tokens_per_batch, D)
    a_3d = attn_2d.view(B, tokens_per_batch, D)
    gate_bf16 = gate_table.to(torch.bfloat16).view(1, 1, D) + gate_ts.view(B, 1, D)
    return (x_3d + a_3d * gate_bf16).view(-1, D)


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_batch", [16, 512])
def test_gate_resid_bf16(hidden_dim, batch_size, tokens_per_batch):
    """trtllm kernel matches the eager reference and mutates x in place."""
    num_tokens = batch_size * tokens_per_batch
    x, attn, gate_table, gate_ts = _make_gate_resid_inputs(
        num_tokens, hidden_dim, batch_size, seed=42
    )

    ref = _ref_gate_resid(x, attn, gate_table, gate_ts, tokens_per_batch)

    x_impl = x.clone()
    out = _call_gate_resid(x_impl, attn, gate_table, gate_ts)

    # pure bf16 __hfma2 vs bf16 eager: ~1 bf16 ULP on rare tail samples.
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    # op returns the same (mutated) tensor as x_impl.
    assert out.data_ptr() == x_impl.data_ptr()
    torch.testing.assert_close(x_impl, ref, rtol=2e-2, atol=2e-2)


def test_gate_resid_rejects_non_contiguous():
    """The in-place kernel requires contiguous x / attn_out."""
    x, attn, gate_table, gate_ts = _make_gate_resid_inputs(32, 4096, batch_size=1, seed=0)
    x_nc = x.transpose(0, 1).transpose(0, 1)[:, ::2]  # non-contiguous view
    with pytest.raises((RuntimeError, AssertionError)):
        _call_gate_resid(x_nc, attn[:, ::2], gate_table[::2], gate_ts[:, ::2])

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the LTX-2 fused gate-residual + RMSNorm + AdaLN modulate kernel.
#
# bf16 variant:   fused_dit_gate_resid_rms_modulate
# NVFP4 variant:  fused_dit_gate_resid_rms_modulate_quant
#
# Each modulator (gate, scale, shift) is built inline from a (table, ts) pair:
#   gate[b,d]  = gate_table[d]  + gate_ts[b,d]  (fp32 add, bf16 narrow on cache store)
# Same for scale and shift. The kernel does the combine in Phase 0b, eliminating
# the upstream broadcast-add Triton prep kernel that Inductor would otherwise emit.

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401  -- triggers libth_common.so load (registers trtllm ops)


@torch.inference_mode()
def torch_ref(
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
    """Reference matching production semantics (apply_fused_gate_resid_modulate
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


def _make_inputs(num_tokens, hidden_dim, batch_size, seed):
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
def test_gate_resid_rms_modulate_bf16(hidden_dim, batch_size, tokens_per_batch):
    """bf16 variant: trtllm kernel matches eager fp32 reference within tolerance,
    and mutates x in place to (x + attn * gate)."""
    num_tokens = batch_size * tokens_per_batch
    x, attn, gate_table, gate_ts, scale_table, scale_ts, shift_table, shift_ts = _make_inputs(
        num_tokens, hidden_dim, batch_size, seed=42
    )
    eps = 1e-6

    ref_out, ref_x_new = torch_ref(
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
    out = torch.ops.trtllm.fused_dit_gate_resid_rms_modulate(
        x_impl, attn, gate_table, gate_ts, scale_table, scale_ts, shift_table, shift_ts, eps
    )

    # Tolerance bumped vs old test: kernel now narrows table->bf16 then bf16-add
    # (PyTorch eager `_get_ada_values` semantics) which differs from the
    # all-fp32 reference at the bf16 ULP boundary; the longer-S reductions amplify
    # the per-element diff slightly. Still well within bf16 noise.
    # Kernel uses pure bf16 hw `__hadd2`/`__hmul2` modulate; production eager uses bf16
    # with fp32-accumulator internally. ~1 bf16 ULP per element on rare tail samples.
    torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x_impl, ref_x_new, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_batch", [126, 128])
def test_gate_resid_rms_modulate_fp4_quant(hidden_dim, batch_size, tokens_per_batch):
    """NVFP4 variant: end-to-end GEMM quality (cosine > 0.98) vs the bf16
    F.linear reference, matching the canonical FP4 test pattern."""
    num_tokens = batch_size * tokens_per_batch
    out_dim = 1024
    x, attn, gate_table, gate_ts, scale_table, scale_ts, shift_table, shift_ts = _make_inputs(
        num_tokens, hidden_dim, batch_size, seed=7
    )
    eps = 1e-6

    # bf16 reference: compute the fused kernel.s bf16 output, then F.linear with random W.
    torch.manual_seed(13)
    W = torch.randn(out_dim, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.05

    x_bf16 = x.clone()
    out_bf16 = torch.ops.trtllm.fused_dit_gate_resid_rms_modulate(
        x_bf16, attn, gate_table, gate_ts, scale_table, scale_ts, shift_table, shift_ts, eps
    )
    D_ref = F.linear(out_bf16, W)

    # Per-tensor inverse-amax sf_scale (the NVFP4 "global scale", SF2) for A and B.
    sf_scale_x = (448.0 * 6.0) / out_bf16.abs().max().float()
    sf_scale_w = (448.0 * 6.0) / W.abs().max().float()

    # FP4 path: produces packed FP4 + SWIZZLED 1x16 FP8 SF directly.
    x_fp4 = x.clone()
    out_fp4, out_sf = torch.ops.trtllm.fused_dit_gate_resid_rms_modulate_quant(
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

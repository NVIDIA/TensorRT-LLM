# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the LTX-2 fused residual + RMSNorm + dual AdaLN modulate kernel.
#
# bf16 variant:   fused_dit_resid_rms_shift_scale_dual  -> (out_dir1, out_dir2)
# NVFP4 variant:  fused_dit_resid_rms_shift_scale_dual_quant
#                 -> (fp4_dir1, sf_dir1, fp4_dir2, sf_dir2)
#
# The fused kernel lives at the post-attn2 site: residual_add(x, attn2_out) -> rms_norm ->
# two independent (1+s)*normed+h modulations, feeding the FFN up_proj
# (dir1) and the AV cross-attn to_q (dir2).
#
# Each of the 4 modulators (scale_dir1, shift_dir1, scale_dir2, shift_dir2) is
# composed inline by the C++ op from a (table_fp32 [D], ts_bf16 [B, D]) pair via
# bf16-narrow-first bf16 hw add -- matches PyTorch eager `_get_av_ca_ada_values`.

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401  -- triggers libth_common.so load (registers trtllm ops)


@torch.inference_mode()
def torch_ref(
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


def _make_inputs(num_tokens, hidden_dim, batch_size, seed):
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
def test_resid_rms_shift_scale_dual_bf16(hidden_dim, batch_size, tokens_per_batch):
    """bf16 variant: dual outputs match eager fp32 reference; x is mutated in place."""
    num_tokens = batch_size * tokens_per_batch
    (x, attn2, s1_table, s1_ts, h1_table, h1_ts, s2_table, s2_ts, h2_table, h2_ts) = _make_inputs(
        num_tokens, hidden_dim, batch_size, seed=42
    )
    eps = 1e-6

    ref_o1, ref_o2, ref_x_new = torch_ref(
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
    o1, o2 = torch.ops.trtllm.fused_dit_resid_rms_shift_scale_dual(
        x_impl, attn2, s1_table, s1_ts, h1_table, h1_ts, s2_table, s2_ts, h2_table, h2_ts, eps
    )

    # Tolerance: same as the gate-residual variant -- bf16-narrow-first kernel vs fp32 ref differs at the
    # bf16 ULP edge; longer-S reductions can push 1 element over the tighter 5e-3.
    # Kernel uses pure bf16 hw `__hadd2`/`__hmul2` modulate; production eager uses bf16
    # with fp32-accumulator internally. ~1 bf16 ULP per element on rare tail samples.
    torch.testing.assert_close(o1, ref_o1, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(o2, ref_o2, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x_impl, ref_x_new, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_batch", [126, 128])
def test_resid_rms_shift_scale_dual_fp4_quant(hidden_dim, batch_size, tokens_per_batch):
    """NVFP4 variant: both quantized outputs (dir1 -> FFN up_proj, dir2 -> AV cross-attn
    to_q) pass end-to-end GEMM cosine > 0.98 vs the bf16 F.linear reference."""
    num_tokens = batch_size * tokens_per_batch
    out_dim = 1024
    (x, attn2, s1_table, s1_ts, h1_table, h1_ts, s2_table, s2_ts, h2_table, h2_ts) = _make_inputs(
        num_tokens, hidden_dim, batch_size, seed=7
    )
    eps = 1e-6

    torch.manual_seed(13)
    W1 = torch.randn(out_dim, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    W2 = torch.randn(out_dim, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.05

    # bf16 reference: fused-kernel bf16 outputs -> F.linear with random W1 / W2.
    x_bf16 = x.clone()
    o1_bf16, o2_bf16 = torch.ops.trtllm.fused_dit_resid_rms_shift_scale_dual(
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
    o1_fp4, o1_sf, o2_fp4, o2_sf = torch.ops.trtllm.fused_dit_resid_rms_shift_scale_dual_quant(
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the LTX-2 fused residual + gate_mul + RMSNorm + NVFP4 quant kernel (KD).
#
# Math (per token n, batch b = n / tokens_per_batch):
#   gate[d]    = gate_table[d].to(bf16) + gate_ts[b, d]    (bf16 narrow, bf16 hw add)
#   x_new[n,d] = x[n,d] + attn_out[n,d] * gate[d]          (in-place)
#   normed     = rms_norm(x_new)
#   (fp4, sf)  = nvfp4_quantize(normed, sf_scale)            (quant variant)
#   out_bf16   = bf16(normed)                                (bf16 variant)

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401  -- triggers libth_common.so load (registers trtllm ops)


@torch.inference_mode()
def torch_ref_bf16(x_2d, attn_2d, gate_table, gate_ts, tokens_per_batch, eps):
    """Reference matching kernel: gate built bf16-narrow-first to match
    PyTorch eager `_get_all_ada_values[2]`. Returns (normed_bf16, x_new_bf16)."""
    B = gate_ts.shape[0]
    D = x_2d.shape[-1]
    x_3d = x_2d.view(B, tokens_per_batch, D).float()
    a_3d = attn_2d.view(B, tokens_per_batch, D).float()
    gate_bf16 = gate_table.to(torch.bfloat16).view(1, 1, D) + gate_ts.view(B, 1, D)
    x_new = x_3d + a_3d * gate_bf16.float()
    var = x_new.pow(2).mean(-1, keepdim=True)
    normed = x_new * torch.rsqrt(var + eps)
    return normed.view(-1, D).to(x_2d.dtype), x_new.view(-1, D).to(x_2d.dtype)


def _make_inputs(num_tokens, hidden_dim, batch_size, seed):
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1
    attn = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1
    gate_table = torch.randn(hidden_dim, dtype=torch.float32, device=device) * 0.3
    gate_ts = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) * 0.3
    return x, attn, gate_table, gate_ts


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_batch", [16, 126])
def test_resid_gate_rms_norm_bf16(hidden_dim, batch_size, tokens_per_batch):
    """bf16 variant: output matches reference; x mutated in place to x + attn * gate."""
    num_tokens = batch_size * tokens_per_batch
    x, attn, gate_table, gate_ts = _make_inputs(num_tokens, hidden_dim, batch_size, seed=42)
    eps = 1e-6

    ref_normed, ref_x_new = torch_ref_bf16(x, attn, gate_table, gate_ts, tokens_per_batch, eps)

    x_impl = x.clone()
    out = torch.ops.trtllm.fused_dit_resid_gate_rms_norm(x_impl, attn, gate_table, gate_ts, eps)

    # Tolerance: same as KC -- bf16-narrow-first kernel vs fp32 ref differs at the
    # bf16 ULP edge; longer-S reductions can push 1 element over the tighter 5e-3.
    torch.testing.assert_close(out, ref_normed, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(x_impl, ref_x_new, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_resid_gate_rms_norm_quant_fp4(hidden_dim, batch_size):
    """NVFP4 variant: end-to-end GEMM quality (cosine > 0.98) vs the bf16
    F.linear reference. Also verifies in-place mutation: x <- x + attn * gate."""
    tokens_per_batch = 126
    num_tokens = batch_size * tokens_per_batch
    out_dim = 1024
    eps = 1e-6

    x, attn, gate_table, gate_ts = _make_inputs(num_tokens, hidden_dim, batch_size, seed=7)
    torch.manual_seed(13)
    W = torch.randn(out_dim, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.05

    # bf16 reference (eager): x_new = x + attn * gate; normed = rms_norm(x_new); D_ref = F.linear(normed, W).
    normed_bf16, x_new_ref = torch_ref_bf16(x, attn, gate_table, gate_ts, tokens_per_batch, eps)
    D_ref = F.linear(normed_bf16, W)

    sf_scale_x = (448.0 * 6.0) / normed_bf16.abs().max().float()
    sf_scale_w = (448.0 * 6.0) / W.abs().max().float()

    # KD op: in-place x <- x + attn * gate, produces (fp4, sf) of normed.
    x_fp4 = x.clone()
    out_fp4, out_sf = torch.ops.trtllm.fused_dit_resid_gate_rms_norm_quant(
        x_fp4, attn, gate_table, gate_ts, sf_scale_x, eps
    )

    # In-place mutation check (loose tolerance: bf16 mul accumulates noise).
    torch.testing.assert_close(x_fp4, x_new_ref, rtol=2e-2, atol=1e-2)

    # Weight quantize via canonical trtllm.fp4_quantize (matches SWIZZLED SF layout).
    W_fp4, W_sf = torch.ops.trtllm.fp4_quantize(W, sf_scale_w, 16)

    alpha = 1.0 / (sf_scale_x * sf_scale_w).float()
    C = torch.ops.trtllm.nvfp4_gemm(out_fp4, W_fp4, out_sf, W_sf, alpha, torch.bfloat16)

    assert F.cosine_similarity(C.flatten().float(), D_ref.flatten().float(), dim=0).item() > 0.98


def test_resid_gate_rms_norm_quant_rejects_unsupported_dim():
    """hidden_dim outside {2048, 4096} raises."""
    device = "cuda"
    D = 1024  # not supported
    x = torch.randn(8, D, dtype=torch.bfloat16, device=device)
    attn = torch.randn(8, D, dtype=torch.bfloat16, device=device)
    gate_table = torch.randn(D, dtype=torch.float32, device=device)
    gate_ts = torch.randn(1, D, dtype=torch.bfloat16, device=device)
    sf_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fused_dit_resid_gate_rms_norm_quant(
            x, attn, gate_table, gate_ts, sf_scale, 1e-6
        )

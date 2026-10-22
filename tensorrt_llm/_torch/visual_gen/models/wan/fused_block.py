# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused WanBlock forward for static-FP8 blocks on Rubin (SM107).

Same math as WanBlock.forward; GEMM epilogues and row kernels fuse the element-wise work.
"""

import torch

from ...modules.wan_fused_fp8 import ops as fused_ops


def _static_fp8(linear) -> bool:
    weight = getattr(linear, "weight", None)
    return (
        weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and getattr(linear, "input_scale", None) is not None
        and not getattr(linear, "force_dynamic_quantization", False)
    )


def eligible(block, temb: torch.Tensor) -> bool:
    """Per-forward checks the load-time route cannot make: static-FP8 linears loaded, out-proj
    and FFN-down biases, no I2V/sparse-gate projections, per-sample temb.

    SM107, TP=1 and async Ulysses are decided by fused_fp8_route.route(). Pure, traceable.
    """
    attn1, attn2, ffn = block.attn1, block.attn2, block.ffn
    if getattr(attn1, "qkv_proj", None) is None or getattr(block.norm2, "weight", None) is None:
        return False
    linears = (
        attn1.qkv_proj,
        attn1.to_out[0],
        attn2.to_q,
        attn2.to_out[0],
        ffn.up_proj,
        ffn.down_proj,
    )
    return bool(
        block._fused_ln_supported
        and block.add_k_proj is None
        and block.to_gate_compress is None
        and block.to_gate_fine is None
        and all(_static_fp8(m) for m in linears)
        # resid_ln_quant adds these biases with the residual.
        and all(m.bias is not None for m in (attn1.to_out[0], attn2.to_out[0], ffn.down_proj))
        and temb.ndim == 3
    )


def _gemm(a8, linear, epilogue=0, d_scale=None):
    bias = linear.bias if epilogue else None
    return torch.ops.wanfused.gemm_fp8(
        a8, linear.weight, linear.input_scale, linear.weight_scale, bias, epilogue, d_scale
    )


def _inv_scale(linear):
    return (1.0 / linear.input_scale.float()).reshape(1)


def _quant(x2d, linear):
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        x2d.contiguous(), linear.input_scale
    )[0]


def forward(block, x, encoder_hidden_states, temb, freqs_cos, freqs_sin, timestep=None):
    """Fused WanBlock forward; x is [B, S, D] with D == 5120 (Wan A14B), guaranteed by
    WanBlock._fused_ln_supported."""
    rlq = torch.ops.wanfused.resid_ln_quant
    batch, seq, dim = x.shape
    tokens = batch * seq
    modulation = (block.scale_shift_table.float() + temb.float()).chunk(6, dim=1)
    shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = (
        m.reshape(batch, dim).contiguous() for m in modulation
    )
    attn1, attn2, ffn = block.attn1, block.attn2, block.ffn
    x2d = x.reshape(tokens, dim)
    eps = float(block.norm1.variance_epsilon)

    # AdaLN1 + FP8 quant, then QKV GEMM with bias.
    _, x8 = rlq(
        x2d, None, None, None, 2, scale_msa, shift_msa, seq, eps, _inv_scale(attn1.qkv_proj)
    )
    qkv = _gemm(x8, attn1.qkv_proj, epilogue=1).view(batch, seq, -1)
    fused_attn = attn1.fused_self_attention
    if fused_attn is not None and fused_ops.sp_mode(attn1)[0] == "none":
        # Fused attention writes FP8 for to_out, skipping a separate quant.
        attn_out8 = torch.ops.wanfused.fp8_self_attention_fp8_out(
            qkv, *fused_ops.attn_args(attn1, freqs_cos, freqs_sin), attn1.to_out[0].input_scale
        )
        y1 = _gemm(attn_out8.reshape(tokens, dim), attn1.to_out[0])
    else:
        attn_out = (fused_attn or fused_ops.module_self_attention)(
            attn1, qkv, freqs_cos, freqs_sin, timestep
        )
        y1 = _gemm(_quant(attn_out.reshape(tokens, dim), attn1.to_out[0]), attn1.to_out[0])

    # Out-proj bias + gated residual + LayerNorm2 + FP8 quant.
    xa, xa8 = rlq(
        x2d,
        y1,
        attn1.to_out[0].bias,
        gate_msa,
        1,
        block.norm2.weight.float(),
        block.norm2.bias.float(),
        seq,
        float(block.norm2.variance_epsilon),
        _inv_scale(attn2.to_q),
    )

    # Cross-attention: fused Q GEMM; K/V, norm and SDPA from modules.
    q = _gemm(xa8, attn2.to_q, epilogue=1).view(batch, seq, -1)
    k, v = attn2.to_k(encoder_hidden_states), attn2.to_v(encoder_hidden_states)
    q, k = attn2.apply_qk_norm(q, k)
    attn2_out = attn2._attn_impl(
        q,
        k,
        v,
        batch_size=batch,
        seq_len=seq,
        kv_seq_len=encoder_hidden_states.shape[1],
        timestep=timestep,
    )
    y2 = _gemm(_quant(attn2_out.reshape(tokens, dim), attn2.to_out[0]), attn2.to_out[0])

    # Out-proj bias + residual + AdaLN3 + FP8 quant.
    xb, xb8 = rlq(
        xa,
        y2,
        attn2.to_out[0].bias,
        None,
        2,
        c_scale,
        c_shift,
        seq,
        float(block.norm3.variance_epsilon),
        _inv_scale(ffn.up_proj),
    )

    # FFN-up GEMM with bias + GELU + FP8 out, then FFN-down.
    h8 = _gemm(xb8, ffn.up_proj, epilogue=2, d_scale=_inv_scale(ffn.down_proj))
    y3 = _gemm(h8, ffn.down_proj)

    # FFN-down bias + gated residual.
    out, _ = rlq(xb, y3, ffn.down_proj.bias, c_gate, 0, None, None, seq, eps, None)
    return out.view(batch, seq, dim)

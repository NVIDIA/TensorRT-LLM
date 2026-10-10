# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused WanBlock forward for static-FP8 blocks on Rubin (TRTLLM_WAN_FUSED_BLOCK=1).

Same math as WanBlock.forward; GEMM epilogues and row kernels fuse the element-wise work.
"""

import os

import torch

from tensorrt_llm._utils import get_sm_version

from ...modules.wan_fused_fp8 import ops as fused_ops

# Attention writes FP8 for to_out, skipping the quant; "0" disables.
_FP8_ATTN_OUT = os.environ.get("TRTLLM_WAN_FUSED_FP8_ATTN_OUT", "1") == "1"


def _static_fp8(linear) -> bool:
    weight = getattr(linear, "weight", None)
    return (
        weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and getattr(linear, "input_scale", None) is not None
        and not getattr(linear, "force_dynamic_quantization", False)
    )


def eligible(block, x: torch.Tensor, temb: torch.Tensor) -> bool:
    """Whether block can take the fused path (pure, traceable)."""
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
        get_sm_version() == 107
        and attn1.tp_size == 1
        and block._fused_ln_supported
        and not block._use_async_ulysses
        and block.add_k_proj is None
        and block.to_gate_compress is None
        and block.to_gate_fine is None
        and all(_static_fp8(m) for m in linears)
        and temb.ndim == 3
        and x.shape[-1] == 5120
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


def _attn_args(attn, freqs_cos, freqs_sin):
    return (
        attn.norm_q.weight,
        attn.norm_k.weight,
        freqs_cos,
        freqs_sin,
        attn.local_num_attention_heads,
        float(attn.eps),
        bool(attn.interleave),
    )


def _fused_attn_mode(attn):
    """sp_mode of attn, or "unsupported" when fused attention is off."""
    if not attn._use_wan_fused_fp8_attn():
        return "unsupported", None
    return fused_ops.sp_mode(attn)


def _self_attention(attn, qkv, freqs_cos, freqs_sin, timestep):
    mode, pg = _fused_attn_mode(attn)
    if mode == "ulysses":
        return fused_ops.ulysses_self_attention(qkv, *_attn_args(attn, freqs_cos, freqs_sin), pg)
    if mode == "none":
        return torch.ops.wanfused.fp8_self_attention(qkv, *_attn_args(attn, freqs_cos, freqs_sin))
    attn.apply_packed_qk_norm_rope(qkv, freqs_cos, freqs_sin)
    q, k, v = qkv.split([attn.local_q_dim, attn.local_kv_dim, attn.local_kv_dim], dim=-1)
    return attn._attn_impl(q, k, v, timestep=timestep)


def forward(block, x, encoder_hidden_states, temb, freqs_cos, freqs_sin, timestep=None):
    """Fused WanBlock forward; x is [B, S, 5120]."""
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
    if _FP8_ATTN_OUT and _fused_attn_mode(attn1)[0] == "none":
        attn_out8 = torch.ops.wanfused.fp8_self_attention_fp8_out(
            qkv, *_attn_args(attn1, freqs_cos, freqs_sin), attn1.to_out[0].input_scale
        )
        y1 = _gemm(attn_out8.reshape(tokens, dim), attn1.to_out[0])
    else:
        attn_out = _self_attention(attn1, qkv, freqs_cos, freqs_sin, timestep)
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

"""Fused per-head RMSNorm + partial split-half RoPE + FP8 E4M3 per-tensor cast of the
MiniMax-H3 attention inputs, for the cuDNN FP8 attention path.

Per attention layer the unfused path runs, on the packed QKV projection output: two row
reductions and two pointwise kernels (QK-norm + RoPE for Q and for K, inductor-fused), then
``torch.ops.trtllm.quantize_e4m3_per_tensor`` three times (one amax kernel and one cast kernel
each, plus the clone that makes the input contiguous). Here the same work is two Triton
launches over the packed buffer:

  pass 1  per-tensor amax of the normalised, rotated Q and K and of V      (dynamic scales)
  pass 2  recompute, scale, cast to FP8; program 0 writes the three descales

With static scales (``H3_FP8_SCALE_FUSION=static``, scales from ``H3_FP8_STATIC_SCALES``)
pass 1 is skipped.

The scale arithmetic reproduces TensorRT-LLM's per-tensor quantizer
(``cpp/tensorrt_llm/common/cudaFp8Utils.cu``): ``scale = bf16(max(amax / 448, 1 / (448 * 512)))``,
``fp8 = cvt.rn.satfinite.e4m3(x * (1 / float(scale)))``, ``descale = float(scale)``. The norm and
RoPE arithmetic follows the compiled eager path: fp32 math, one rounding to BF16 after
``x * rsqrt(mean(x^2) + eps)``, the weight multiply and the rotation in fp32 with the BF16-rounded
``cos`` / ``sin``, one rounding to BF16 before the cast.

Control: ``H3_FP8_SCALE_FUSION`` = ``0`` (off, default) | ``dynamic`` | ``static``.
Calibration: ``H3_FP8_AMAX_LOG=<json>`` records the running maximum descale per layer and
writes it at interpreter exit (input for the static mode).
"""

from __future__ import annotations

import atexit
import json
import os
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

FP8_E4M3_MAX = 448.0
MIN_SCALING_FACTOR = 1.0 / (FP8_E4M3_MAX * 512.0)

if hasattr(tl, "div_rn"):

    @triton.jit
    def _div_rn(a, b):
        return tl.div_rn(a, b)

else:

    @triton.jit
    def _div_rn(a, b):
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )


@triton.jit
def _norm_rope_tile(
    x_ptrs,
    xp_ptrs,
    mask,
    w_d,
    w_p,
    cos_v,
    sin_v,
    sign,
    inv_d,
    eps,
):
    """Normalise one [BLOCK_H, D] tile per head and rotate the leading ROT columns.

    ``x_ptrs`` address the tile, ``xp_ptrs`` the same rows with the rotary partner column
    (d + ROT/2 for the first half, d - ROT/2 for the second, d itself past ROT). Returns the
    BF16-rounded result as fp32.
    """
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    xp = tl.load(xp_ptrs, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=1)
    inv = tl.rsqrt(ss * inv_d + eps)
    yn = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32) * w_d[None, :]
    yp = (xp * inv[:, None]).to(tl.bfloat16).to(tl.float32) * w_p[None, :]
    y = yn * cos_v[None, :] + (sign * sin_v)[None, :] * yp
    return y.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _rope_vectors(cos_ptr, sin_ptr, t, d, ROT: tl.constexpr):
    half: tl.constexpr = ROT // 2
    is_rot = d < ROT
    partner = tl.where(d < half, d + half, tl.where(is_rot, d - half, d))
    sign = tl.where(d < half, -1.0, 1.0).to(tl.float32)
    cos_v = tl.load(cos_ptr + t * ROT + d, mask=is_rot, other=1.0)
    sin_v = tl.load(sin_ptr + t * ROT + d, mask=is_rot, other=0.0)
    # The eager path casts cos / sin to the activation dtype before the rotation.
    cos_v = cos_v.to(tl.float32).to(tl.bfloat16).to(tl.float32)
    sin_v = sin_v.to(tl.float32).to(tl.bfloat16).to(tl.float32)
    return partner, sign, cos_v, sin_v


@triton.jit
def _qkv_amax_kernel(
    qkv_ptr,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    sin_ptr,
    amax_ptr,
    S,
    eps,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    r = tl.program_id(0)
    t = r % S
    d = tl.arange(0, D)
    hh = tl.arange(0, BLOCK_H)
    partner, sign, cos_v, sin_v = _rope_vectors(cos_ptr, sin_ptr, t, d, ROT)
    wq_d = tl.load(wq_ptr + d).to(tl.float32)
    wq_p = tl.load(wq_ptr + partner).to(tl.float32)
    wk_d = tl.load(wk_ptr + d).to(tl.float32)
    wk_p = tl.load(wk_ptr + partner).to(tl.float32)
    inv_d = 1.0 / D
    row = qkv_ptr + r.to(tl.int64) * (3 * H * D)
    m_q = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    m_k = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    m_v = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + hh
        mask = (h < H)[:, None] & (d < D)[None, :]
        offs = h[:, None] * D + d[None, :]
        offs_p = h[:, None] * D + partner[None, :]
        yq = _norm_rope_tile(
            row + offs, row + offs_p, mask, wq_d, wq_p, cos_v, sin_v, sign, inv_d, eps
        )
        yk = _norm_rope_tile(
            row + H * D + offs,
            row + H * D + offs_p,
            mask,
            wk_d,
            wk_p,
            cos_v,
            sin_v,
            sign,
            inv_d,
            eps,
        )
        v = tl.load(row + 2 * H * D + offs, mask=mask, other=0.0).to(tl.float32)
        m_q = tl.maximum(m_q, tl.abs(yq))
        m_k = tl.maximum(m_k, tl.abs(yk))
        m_v = tl.maximum(m_v, tl.abs(v))
    tl.atomic_max(amax_ptr + 0, tl.max(tl.max(m_q, axis=1), axis=0))
    tl.atomic_max(amax_ptr + 1, tl.max(tl.max(m_k, axis=1), axis=0))
    tl.atomic_max(amax_ptr + 2, tl.max(tl.max(m_v, axis=1), axis=0))


@triton.jit
def _scale_from_amax(amax):
    # cudaFp8Utils.cu: scale = (bf16) max(amax / FP8_E4M3_MAX, 1 / (FP8_E4M3_MAX * 512)); the
    # cast kernel multiplies by 1 / float(scale) and the backend dequantises with float(scale).
    s = tl.maximum(_div_rn(amax, 448.0), 1.0 / (448.0 * 512.0))
    return s.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _qkv_cast_kernel(
    qkv_ptr,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    sin_ptr,
    amax_ptr,
    descale_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    S,
    eps,
    STATIC: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    r = tl.program_id(0)
    t = r % S
    d = tl.arange(0, D)
    hh = tl.arange(0, BLOCK_H)
    partner, sign, cos_v, sin_v = _rope_vectors(cos_ptr, sin_ptr, t, d, ROT)
    wq_d = tl.load(wq_ptr + d).to(tl.float32)
    wq_p = tl.load(wq_ptr + partner).to(tl.float32)
    wk_d = tl.load(wk_ptr + d).to(tl.float32)
    wk_p = tl.load(wk_ptr + partner).to(tl.float32)
    inv_d = 1.0 / D
    if STATIC:
        s_q = tl.load(descale_ptr + 0)
        s_k = tl.load(descale_ptr + 1)
        s_v = tl.load(descale_ptr + 2)
    else:
        s_q = _scale_from_amax(tl.load(amax_ptr + 0))
        s_k = _scale_from_amax(tl.load(amax_ptr + 1))
        s_v = _scale_from_amax(tl.load(amax_ptr + 2))
        if r == 0:
            tl.store(descale_ptr + 0, s_q)
            tl.store(descale_ptr + 1, s_k)
            tl.store(descale_ptr + 2, s_v)
    f_q = _div_rn(1.0, s_q)
    f_k = _div_rn(1.0, s_k)
    f_v = _div_rn(1.0, s_v)
    row = qkv_ptr + r.to(tl.int64) * (3 * H * D)
    out_row = r.to(tl.int64) * (H * D)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + hh
        mask = (h < H)[:, None] & (d < D)[None, :]
        offs = h[:, None] * D + d[None, :]
        offs_p = h[:, None] * D + partner[None, :]
        yq = _norm_rope_tile(
            row + offs, row + offs_p, mask, wq_d, wq_p, cos_v, sin_v, sign, inv_d, eps
        )
        tl.store(q_out_ptr + out_row + offs, (yq * f_q).to(tl.float8e4nv), mask=mask)
        yk = _norm_rope_tile(
            row + H * D + offs,
            row + H * D + offs_p,
            mask,
            wk_d,
            wk_p,
            cos_v,
            sin_v,
            sign,
            inv_d,
            eps,
        )
        tl.store(k_out_ptr + out_row + offs, (yk * f_k).to(tl.float8e4nv), mask=mask)
        v = tl.load(row + 2 * H * D + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(v_out_ptr + out_row + offs, (v * f_v).to(tl.float8e4nv), mask=mask)


def _launch_config() -> Tuple[int, int]:
    return (
        int(os.environ.get("H3_FP8_FUSION_BLOCK_H", "8")),
        int(os.environ.get("H3_FP8_FUSION_WARPS", "4")),
    )


def _split_launch() -> bool:
    """``H3_FP8_FUSION_SPLIT=1``: one launch per tensor (three amax, three cast) through the
    per-tensor kernels with the packed row stride, instead of one launch over the packed row:
    the same arithmetic, three times the programs with a third of the work each."""
    return os.environ.get("H3_FP8_FUSION_SPLIT", "0") == "1"


def _qk_norm_rope_fp8_split(
    qkv, weight_q, weight_k, cos, sin, static_descale, eps, h, d, s, rows, outs, block_h, num_warps
):
    width = h * d
    grid = (rows,)
    static = static_descale is not None
    if static:
        # A custom op's output may not alias an input: copy the static table.
        descale = static_descale.to(torch.float32).contiguous().clone()
        amax = descale
    else:
        amax = torch.zeros(3, dtype=torch.float32, device=qkv.device)
        descale = torch.empty(3, dtype=torch.float32, device=qkv.device)
    weights = (weight_q, weight_k, weight_q)
    rot = cos.shape[-1]
    if not static:
        for i in range(3):
            _tensor_amax_kernel[grid](
                qkv[..., i * width : (i + 1) * width],
                weights[i],
                cos,
                sin,
                amax[i : i + 1],
                s,
                eps,
                NORM_ROPE=i < 2,
                H=h,
                D=d,
                ROT=rot,
                BLOCK_H=block_h,
                ROW_STRIDE=3 * width,
                num_warps=num_warps,
            )
    for i in range(3):
        _tensor_cast_kernel[grid](
            qkv[..., i * width : (i + 1) * width],
            weights[i],
            cos,
            sin,
            amax[i : i + 1],
            descale[i : i + 1],
            outs[i],
            s,
            eps,
            NORM_ROPE=i < 2,
            OUT_FP8=True,
            STATIC=static,
            H=h,
            D=d,
            ROT=rot,
            BLOCK_H=block_h,
            ROW_STRIDE=3 * width,
            num_warps=num_warps,
        )
    return descale, static


@torch.library.custom_op("h3_lever8::qk_norm_rope_fp8", mutates_args=())
def qk_norm_rope_fp8(
    qkv: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    static_descale: Optional[torch.Tensor],
    eps: float,
    num_heads: int,
    head_dim: int,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP8 ``q``, ``k``, ``v`` of shape ``[B, S, H * D]`` and ``descale`` ``[3]`` fp32.

    ``qkv`` is the packed projection output ``[B, S, 3 * H * D]`` (BF16, contiguous);
    ``cos`` / ``sin`` are ``[S, ROT]``; ``static_descale`` ``[3]`` fp32 skips the amax pass.
    """
    b, s, width = qkv.shape
    h, d = num_heads, head_dim
    if width != 3 * h * d:
        raise ValueError(f"qkv width {width} != 3 * {h} * {d}")
    rot = cos.shape[-1]
    qkv = qkv.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    rows = b * s
    q8 = torch.empty((b, s, h * d), dtype=torch.float8_e4m3fn, device=qkv.device)
    k8 = torch.empty_like(q8)
    v8 = torch.empty_like(q8)
    block_h, num_warps = _launch_config()
    grid = (rows,)
    if _split_launch():
        descale, static = _qk_norm_rope_fp8_split(
            qkv,
            weight_q,
            weight_k,
            cos,
            sin,
            static_descale,
            eps,
            h,
            d,
            s,
            rows,
            (q8, k8, v8),
            block_h,
            num_warps,
        )
        if _AMAX_LOG is not None and not static:
            _record_descale(layer_idx, descale)
        return q8, k8, v8, descale
    if static_descale is None:
        amax = torch.zeros(3, dtype=torch.float32, device=qkv.device)
        descale = torch.empty(3, dtype=torch.float32, device=qkv.device)
        _qkv_amax_kernel[grid](
            qkv,
            weight_q,
            weight_k,
            cos,
            sin,
            amax,
            s,
            eps,
            H=h,
            D=d,
            ROT=rot,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
        static = False
    else:
        amax = static_descale  # unused in the static branch; keeps the signature
        descale = static_descale.to(torch.float32).contiguous().clone()  # no aliasing of an input
        static = True
    _qkv_cast_kernel[grid](
        qkv,
        weight_q,
        weight_k,
        cos,
        sin,
        amax,
        descale,
        q8,
        k8,
        v8,
        s,
        eps,
        STATIC=static,
        H=h,
        D=d,
        ROT=rot,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
    if _AMAX_LOG is not None and not static:
        _record_descale(layer_idx, descale)
    return q8, k8, v8, descale


@qk_norm_rope_fp8.register_fake
def _qk_norm_rope_fp8_fake(
    qkv, weight_q, weight_k, cos, sin, static_descale, eps, num_heads, head_dim, layer_idx
):
    b, s, _ = qkv.shape
    out = qkv.new_empty((b, s, num_heads * head_dim), dtype=torch.float8_e4m3fn)
    return (
        out,
        qkv.new_empty(out.shape, dtype=out.dtype),
        qkv.new_empty(out.shape, dtype=out.dtype),
        qkv.new_empty((3,), dtype=torch.float32),
    )


# ----------------------------------------------------------------------------------------------
# Per-tensor variant (async Ulysses: Q, K, V arrive as three [B, S, H, D] tensors)
# ----------------------------------------------------------------------------------------------


@triton.jit
def _tensor_amax_kernel(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    amax_ptr,
    S,
    eps,
    NORM_ROPE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
):
    r = tl.program_id(0)
    t = r % S
    d = tl.arange(0, D)
    hh = tl.arange(0, BLOCK_H)
    row = x_ptr + r.to(tl.int64) * ROW_STRIDE
    m = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    if NORM_ROPE:
        partner, sign, cos_v, sin_v = _rope_vectors(cos_ptr, sin_ptr, t, d, ROT)
        w_d = tl.load(w_ptr + d).to(tl.float32)
        w_p = tl.load(w_ptr + partner).to(tl.float32)
        inv_d = 1.0 / D
    for h0 in range(0, H, BLOCK_H):
        h = h0 + hh
        mask = (h < H)[:, None] & (d < D)[None, :]
        offs = h[:, None] * D + d[None, :]
        if NORM_ROPE:
            offs_p = h[:, None] * D + partner[None, :]
            y = _norm_rope_tile(
                row + offs, row + offs_p, mask, w_d, w_p, cos_v, sin_v, sign, inv_d, eps
            )
        else:
            y = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.maximum(m, tl.abs(y))
    tl.atomic_max(amax_ptr, tl.max(tl.max(m, axis=1), axis=0))


@triton.jit
def _tensor_cast_kernel(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    amax_ptr,
    descale_ptr,
    out_ptr,
    S,
    eps,
    NORM_ROPE: tl.constexpr,
    OUT_FP8: tl.constexpr,
    STATIC: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
):
    r = tl.program_id(0)
    t = r % S
    d = tl.arange(0, D)
    hh = tl.arange(0, BLOCK_H)
    row = x_ptr + r.to(tl.int64) * ROW_STRIDE
    out_row = out_ptr + r.to(tl.int64) * (H * D)
    if NORM_ROPE:
        partner, sign, cos_v, sin_v = _rope_vectors(cos_ptr, sin_ptr, t, d, ROT)
        w_d = tl.load(w_ptr + d).to(tl.float32)
        w_p = tl.load(w_ptr + partner).to(tl.float32)
        inv_d = 1.0 / D
    if OUT_FP8:
        if STATIC:
            s = tl.load(descale_ptr)
        else:
            s = _scale_from_amax(tl.load(amax_ptr))
            if r == 0:
                tl.store(descale_ptr, s)
        f = _div_rn(1.0, s)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + hh
        mask = (h < H)[:, None] & (d < D)[None, :]
        offs = h[:, None] * D + d[None, :]
        if NORM_ROPE:
            offs_p = h[:, None] * D + partner[None, :]
            y = _norm_rope_tile(
                row + offs, row + offs_p, mask, w_d, w_p, cos_v, sin_v, sign, inv_d, eps
            )
        else:
            y = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
        if OUT_FP8:
            tl.store(out_row + offs, (y * f).to(tl.float8e4nv), mask=mask)
        else:
            tl.store(out_row + offs, y.to(tl.bfloat16), mask=mask)


@torch.library.custom_op("h3_lever8::norm_rope_fp8_tensor", mutates_args=())
def norm_rope_fp8_tensor(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    static_descale: Optional[torch.Tensor],
    eps: float,
    out_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One ``[B, S, H, D]`` tensor: per-head RMSNorm + RoPE when ``weight`` is given (Q or K),
    identity otherwise (V); output BF16, or FP8 E4M3 with the per-tensor ``descale`` ``[1]`` fp32
    when ``out_fp8``. The BF16 output is the same value the compiled eager path produces; the FP8
    output uses the same scale arithmetic as ``qk_norm_rope_fp8``. ``S`` is the local sequence
    (``cos`` / ``sin`` are ``[S, ROT]`` for these rows). Returns ``(out, descale)``; ``descale``
    is a ``[1]`` fp32 tensor holding 1.0 for the BF16 output.
    """
    b, s, h, d = x.shape
    x = x.contiguous()
    norm_rope = weight is not None
    if norm_rope:
        if cos is None or sin is None:
            raise ValueError("norm + RoPE needs cos and sin")
        cos = cos.contiguous()
        sin = sin.contiguous()
        rot = cos.shape[-1]
    else:
        rot = 2  # unused
        cos = sin = weight = x  # placeholders, never dereferenced
    rows = b * s
    out = torch.empty(
        (b, s, h, d), dtype=torch.float8_e4m3fn if out_fp8 else torch.bfloat16, device=x.device
    )
    block_h, num_warps = _launch_config()
    grid = (rows,)
    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    if out_fp8 and static_descale is None:
        descale = torch.empty(1, dtype=torch.float32, device=x.device)
        _tensor_amax_kernel[grid](
            x,
            weight,
            cos,
            sin,
            amax,
            s,
            eps,
            NORM_ROPE=norm_rope,
            H=h,
            D=d,
            ROT=rot,
            BLOCK_H=block_h,
            ROW_STRIDE=h * d,
            num_warps=num_warps,
        )
        static = False
    elif out_fp8:
        descale = static_descale.to(torch.float32).reshape(1).contiguous().clone()
        static = True
    else:
        descale = torch.ones(1, dtype=torch.float32, device=x.device)
        static = True
    _tensor_cast_kernel[grid](
        x,
        weight,
        cos,
        sin,
        amax,
        descale,
        out,
        s,
        eps,
        NORM_ROPE=norm_rope,
        OUT_FP8=out_fp8,
        STATIC=static,
        H=h,
        D=d,
        ROT=rot,
        BLOCK_H=block_h,
        ROW_STRIDE=h * d,
        num_warps=num_warps,
    )
    return out, descale


@norm_rope_fp8_tensor.register_fake
def _norm_rope_fp8_tensor_fake(x, weight, cos, sin, static_descale, eps, out_fp8):
    return (
        x.new_empty(x.shape, dtype=torch.float8_e4m3fn if out_fp8 else torch.bfloat16),
        x.new_empty((1,), dtype=torch.float32),
    )


def norm_rope_tensor(
    x: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    eps: float,
    fp8: bool = False,
    static_descale: Optional[torch.Tensor] = None,
):
    """Per-tensor entry point for the async Ulysses path (lever 13, ¶14969).

    ``norm_rope_tensor(q, norm_q.weight, cos, sin, eps)`` -> BF16 ``[B, S, H, D]``;
    ``norm_rope_tensor(q, norm_q.weight, cos, sin, eps, fp8=True)`` -> ``(q_fp8, descale[1])``;
    ``norm_rope_tensor(v, None, None, None, eps, fp8=True)`` casts V without norm or RoPE.
    The three descales concatenated (``torch.cat``) are the ``fp8_descales`` the cuDNN backend takes.
    """
    out, descale = torch.ops.h3_lever8.norm_rope_fp8_tensor(
        x, norm_weight, cos, sin, static_descale, float(eps), bool(fp8)
    )
    return (out, descale) if fp8 else out


# ----------------------------------------------------------------------------------------------
# Mode, static scales and calibration log
# ----------------------------------------------------------------------------------------------


def _read_fusion_mode() -> str:
    mode = os.environ.get("H3_FP8_SCALE_FUSION", "0").strip().lower()
    if mode in ("", "0", "off", "false", "no"):
        return "off"
    if mode in ("1", "dynamic", "dyn", "on"):
        return "dynamic"
    if mode == "static":
        return "static"
    raise ValueError(f"H3_FP8_SCALE_FUSION={mode!r}: expected 0, dynamic or static")


# Read once at import so the compiled transformer sees a constant (no environment access in
# the traced forward).
_MODE = _read_fusion_mode()


def fusion_mode() -> str:
    """``off`` | ``dynamic`` | ``static`` from ``H3_FP8_SCALE_FUSION`` (read at import)."""
    return _MODE


_STATIC_TABLE: Optional[Dict[str, list]] = None
_STATIC_CACHE: Dict[Tuple[int, torch.device], torch.Tensor] = {}


@torch.compiler.disable
def static_descale(layer_idx: int, device: torch.device) -> torch.Tensor:
    """Per-layer ``[3]`` fp32 descale (q, k, v) from the JSON named by ``H3_FP8_STATIC_SCALES``.

    The file maps ``str(layer_idx)`` to ``[descale_q, descale_k, descale_v]`` (the running maxima
    written by the calibration log, optionally scaled by ``H3_FP8_STATIC_MARGIN``, default 1.0).
    Each value is rounded to BF16 like the dynamic quantizer's scale.
    """
    global _STATIC_TABLE
    key = (layer_idx, device)
    cached = _STATIC_CACHE.get(key)
    if cached is not None:
        return cached
    if _STATIC_TABLE is None:
        path = os.environ.get("H3_FP8_STATIC_SCALES")
        if not path:
            raise RuntimeError("H3_FP8_SCALE_FUSION=static needs H3_FP8_STATIC_SCALES=<json>")
        with open(path) as f:
            table = json.load(f)
        # Either a bare {layer: [dq, dk, dv]} map or the calibration log's {"descale_max": {...}}.
        _STATIC_TABLE = table.get("descale_max", table)
    margin = float(os.environ.get("H3_FP8_STATIC_MARGIN", "1.0"))
    values = _STATIC_TABLE[str(layer_idx)]
    t = torch.tensor([float(v) * margin for v in values], dtype=torch.float32)
    t = t.clamp_min(MIN_SCALING_FACTOR).to(torch.bfloat16).to(torch.float32).to(device)
    _STATIC_CACHE[key] = t
    return t


_AMAX_LOG = os.environ.get("H3_FP8_AMAX_LOG") or None
_RUNNING: Dict[int, torch.Tensor] = {}
_CALLS: Dict[int, int] = {}


def _record_descale(layer_idx: int, descale: torch.Tensor) -> None:
    running = _RUNNING.get(layer_idx)
    if running is None:
        _RUNNING[layer_idx] = descale.clone()
    else:
        torch.maximum(running, descale, out=running)
    _CALLS[layer_idx] = _CALLS.get(layer_idx, 0) + 1


def _dump_amax_log() -> None:
    if _AMAX_LOG is None or not _RUNNING:
        return
    table = {str(k): [float(x) for x in v.cpu().tolist()] for k, v in sorted(_RUNNING.items())}
    payload = {
        "descale_max": table,
        "calls": {str(k): v for k, v in sorted(_CALLS.items())},
        "note": "descale = max(amax / 448, 1 / (448 * 512)) rounded to bf16; q, k, v per layer",
    }
    path = (
        f"{_AMAX_LOG}.pid{os.getpid()}" if os.environ.get("H3_FP8_AMAX_LOG_PER_PID") else _AMAX_LOG
    )
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)


atexit.register(_dump_amax_log)

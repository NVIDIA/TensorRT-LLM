# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton SwiGLU + NVFP4 quantize over a gate_up output in the permuted column order.

Replaces ``silu_and_mul`` (BF16 out) followed by ``fp4_quantize(h, global_scale, 16)``
with bitwise-identical FP4 bytes and swizzled (128x4) SF bytes, including the zeroed
SF padding rows up to a multiple of 128:

* SwiGLU is ``silu_and_mul_kernel``'s expression (``a * sigmoid(a) * b``) rounded to
  BF16, as the stored intermediate would be;
* the quantizer reproduces ``cvt_warp_fp16_to_fp4`` as built (no fast math), with the
  helpers of ``quantization/_nvfp4_triton``.

The input is gate_up's output with its columns in
``_nvfp4_recipes.gate_up_swiglu_row_permutation`` order (the weight rows were
permuted for the SwiGLU-epilogue GEMM): output ``j``'s gate and up sit at adjacent
columns of its 128-column tile, and only the load addresses differ from the
unpermuted form.

Two kernels, the same bytes:

* the base kernel -- one program per (row, 512 columns), plus one program row that
  zeroes the SF padding rows; launched with PDL when the whole grid fits one wave
  (it releases its dependents first and waits before any access);
* the decode kernel (``V2``, rows 5-32) -- one thread per 16-output SF block, whose
  16 gate/up pairs are two 32-byte runs of the permuted row read as four 16-byte
  vectors; no PDL.
"""

import functools

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ..quantization._nvfp4_triton import (
    _cvt_e2m1x8,
    _cvt_e4m3_bits,
    _gdc_wait_with_memory_clobber,
    _mul_rn,
    _rcp_approx_ftz,
    _sf_offset,
)

_SF_VEC = 16
#: Rows the decode kernel serves; the base kernel takes every other row count.
V2_MIN_M, V2_MAX_M = 5, 32
_V2_NB = 32
_V2_WARPS = 1
_BLOCK = 512


@triton.jit
def fused_silu_mul_nvfp4_permuted_kernel(
    fp4_ptr,
    sf_ptr,
    x_ptr,
    x_stride,
    scale_ptr,
    M,
    d,
    swiglu_alpha: tl.constexpr,
    swiglu_beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    USE_PDL: tl.constexpr = False,
) -> None:
    if USE_PDL:
        # Let the consumer GEMM launch (and prefetch its weights) now; every
        # access of this kernel stays behind the wait.
        tl.extra.cuda.gdc_launch_dependents()
        _gdc_wait_with_memory_clobber()
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)
    NSF: tl.constexpr = BLOCK_SIZE // 16
    sf_cols = j * NSF + tl.arange(0, NSF)
    num_sf_cols = d // 16

    if i < M:
        x_row_ptr = x_ptr + x_stride * i
        offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < d

        # gate_up rows permuted for the SwiGLU GEMM epilogue
        # (gate_up_swiglu_row_permutation): output j's gate/up sit at
        # adjacent columns of its 128-column tile. Only the load addresses
        # change; the arithmetic below is unchanged.
        jl = offsets % 64
        col = (
            (offsets // 64) * 128
            + 32 * (2 * ((jl % 32) // 16) + (jl % 4) // 2)
            + 16 * (jl // 32)
            + 8 * (jl % 2)
            + 2 * ((jl % 16) // 4)
        )
        a = tl.load(x_row_ptr + col, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_row_ptr + col + 1, mask=mask, other=0.0).to(tl.float32)
        # Exactly silu_and_mul_kernel's expression (plain SwiGLU).
        result = a * tl.sigmoid(swiglu_alpha * a) * (b + swiglu_beta)
        # The unfused path stores BF16 and the quantizer reloads it.
        h = result.to(tl.bfloat16).to(tl.float32)

        h2 = tl.reshape(h, [NSF, 16])
        vmax = tl.max(tl.abs(h2), axis=1)  # [NSF], exact (max of BF16 values)

        s = tl.load(scale_ptr)
        s_v = tl.full([NSF], 0.0, tl.float32) + s
        r6 = _rcp_approx_ftz(tl.full([NSF], 6.0, tl.float32))
        sf_val = _mul_rn(s_v, _mul_rn(vmax, r6))
        sf_bits = _cvt_e4m3_bits(sf_val)
        sf_u8 = sf_bits.to(tl.uint8)
        sf_f = sf_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        out_scale = tl.where(vmax != 0.0, _rcp_approx_ftz(_mul_rn(sf_f, _rcp_approx_ftz(s_v))), 0.0)

        y = _mul_rn(h2, tl.broadcast_to(out_scale[:, None], [NSF, 16]))
        y8 = tl.reshape(y, [BLOCK_SIZE // 8, 2, 2, 2])
        p, q = tl.split(y8)  # last index 0 / 1
        p0, p1 = tl.split(p)
        q0, q1 = tl.split(q)
        p00, p01 = tl.split(p0)
        p10, p11 = tl.split(p1)
        q00, q01 = tl.split(q0)
        q10, q11 = tl.split(q1)
        # element index = 4*i + 2*j + k for [.., i, j, k]
        packed = _cvt_e2m1x8(p00, q00, p10, q10, p01, q01, p11, q11)

        w_offsets = j * (BLOCK_SIZE // 8) + tl.arange(0, BLOCK_SIZE // 8)
        fp4_row_ptr = fp4_ptr + i * (d // 8)
        tl.store(fp4_row_ptr + w_offsets, packed, mask=w_offsets < d // 8)

        sf_off = _sf_offset(i, sf_cols, NUM_K_TILES)
        tl.store(sf_ptr + sf_off, sf_u8, mask=sf_cols < num_sf_cols)
    else:
        # One extra program row zeroes the SF padding rows [M, ceil128(M)) for this
        # column block (same as quantize_with_block_size).
        m0 = (M // 128) * 128
        rows = m0 + tl.arange(0, 128).to(tl.int64)
        rmask = rows >= M
        off = _sf_offset(rows[:, None], sf_cols[None, :], NUM_K_TILES)
        pmask = rmask[:, None] & (sf_cols < num_sf_cols)[None, :]
        tl.store(sf_ptr + off, tl.zeros([128, NSF], tl.uint8), mask=pmask)


@triton.jit
def _swiglu_bf16_pairs(xv, NB: tl.constexpr, swiglu_alpha: tl.constexpr, swiglu_beta: tl.constexpr):
    # xv [NB, 8] = (lq, p) interleaved gate (p=0) / up (p=1) -> h [NB, 4].
    x2 = tl.reshape(xv, [NB, 4, 2])
    a, b = tl.split(x2)
    a = a.to(tl.float32)
    b = b.to(tl.float32)
    # Exactly silu_and_mul_kernel's expression (plain SwiGLU).
    result = a * tl.sigmoid(swiglu_alpha * a) * (b + swiglu_beta)
    return result.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _lq_split(y, NB: tl.constexpr):
    # y [NB, 4] indexed by lq = 2 * lqh + lql -> (lql=0, lql=1), each [NB, 2]
    # indexed by lqh.
    return tl.split(tl.reshape(y, [NB, 2, 2]))


@triton.jit
def fused_silu_mul_nvfp4_permuted_v2_kernel(
    fp4_ptr,
    sf_ptr,
    sf32_ptr,
    x_ptr,
    x_stride,
    scale_ptr,
    M,
    d,
    swiglu_alpha: tl.constexpr,
    swiglu_beta: tl.constexpr,
    NB: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    EVEN: tl.constexpr,
) -> None:
    """Decode-row variant of fused_silu_mul_nvfp4_permuted_kernel.

    One thread owns one 16-output SF block: its 16 gate/up pairs sit in two
    32-byte runs of the permuted gate_up row (base + 32 * al + [0, 16)), read
    as four 16-B vectors. The amax, SF and FP4 packing stay in the thread.
    Per-element arithmetic is identical to the base kernel.
    """
    i = tl.program_id(axis=0).to(tl.int64)
    jb = tl.program_id(axis=1)
    num_sf = d // 16
    if i < M:
        g = jb * NB + tl.arange(0, NB)  # SF block index within the row
        t = g // 4
        gl = g % 4
        # Block gl of 128-column tile t: nw = gl // 2, ah = gl % 2; its
        # columns are t*128 + 64*ah + 16*nw + 32*al + 8*b + 2*lq + p.
        base = t * 128 + 64 * (gl % 2) + 16 * (gl // 2)
        base = tl.multiple_of(base, 16)
        e = tl.arange(0, 8)
        p = x_ptr + x_stride * i + base[:, None] + e[None, :]
        if EVEN:
            x00 = tl.load(p)
            x01 = tl.load(p + 8)
            x10 = tl.load(p + 32)
            x11 = tl.load(p + 40)
        else:
            m2 = (g < num_sf)[:, None]
            x00 = tl.load(p, mask=m2, other=0.0)
            x01 = tl.load(p + 8, mask=m2, other=0.0)
            x10 = tl.load(p + 32, mask=m2, other=0.0)
            x11 = tl.load(p + 40, mask=m2, other=0.0)
        # h_{al b}[lq]: output 4*lq + 2*al + b of the block.
        h00 = _swiglu_bf16_pairs(x00, NB, swiglu_alpha, swiglu_beta)
        h01 = _swiglu_bf16_pairs(x01, NB, swiglu_alpha, swiglu_beta)
        h10 = _swiglu_bf16_pairs(x10, NB, swiglu_alpha, swiglu_beta)
        h11 = _swiglu_bf16_pairs(x11, NB, swiglu_alpha, swiglu_beta)
        vmax = tl.maximum(
            tl.maximum(tl.max(tl.abs(h00), axis=1), tl.max(tl.abs(h01), axis=1)),
            tl.maximum(tl.max(tl.abs(h10), axis=1), tl.max(tl.abs(h11), axis=1)),
        )

        s = tl.load(scale_ptr)
        s_v = tl.full([NB], 0.0, tl.float32) + s
        r6 = _rcp_approx_ftz(tl.full([NB], 6.0, tl.float32))
        sf_val = _mul_rn(s_v, _mul_rn(vmax, r6))
        sf_bits = _cvt_e4m3_bits(sf_val)
        sf_u8 = sf_bits.to(tl.uint8)
        sf_f = sf_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        out_scale = tl.where(vmax != 0.0, _rcp_approx_ftz(_mul_rn(sf_f, _rcp_approx_ftz(s_v))), 0.0)
        osb = tl.broadcast_to(out_scale[:, None], [NB, 4])
        y00a, y00b = _lq_split(_mul_rn(h00, osb), NB)
        y01a, y01b = _lq_split(_mul_rn(h01, osb), NB)
        y10a, y10b = _lq_split(_mul_rn(h10, osb), NB)
        y11a, y11b = _lq_split(_mul_rn(h11, osb), NB)
        # Word lqh, byte 2*lql + al, nibble b (low = even element).
        packed = _cvt_e2m1x8(y00a, y01a, y10a, y11a, y00b, y01b, y10b, y11b)
        w = (i * (d // 8) + 2 * g)[:, None] + tl.arange(0, 2)[None, :]
        sf_off = _sf_offset(i, g, NUM_K_TILES)
        if EVEN:
            tl.store(fp4_ptr + w, packed)
            tl.store(sf_ptr + sf_off, sf_u8)
        else:
            tl.store(fp4_ptr + w, packed, mask=(g < num_sf)[:, None])
            tl.store(sf_ptr + sf_off, sf_u8, mask=g < num_sf)
    else:
        # Zero the SF padding rows [M, ceil128(M)) of this program's k-tiles
        # as int32 words: word (m % 32) * 4 + (m % 128) // 32 of a k-tile
        # holds row m's 4 SF bytes.
        KT: tl.constexpr = NB // 4
        kt = jb * KT + tl.arange(0, KT)
        wl = tl.arange(0, 128)
        m0 = (M // 128) * 128
        rows = m0 + 32 * (wl % 4) + wl // 4
        off = ((m0 // 128) * NUM_K_TILES + kt[:, None]) * 128 + wl[None, :]
        pmask = (kt < NUM_K_TILES)[:, None] & (rows >= M)[None, :]
        tl.store(sf32_ptr + off, tl.zeros([KT, 128], tl.int32), mask=pmask)


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def uses_v2(m: int, d: int) -> bool:
    """Whether ``m`` rows of a ``d``-wide output run the decode kernel."""
    return V2_MIN_M <= m <= V2_MAX_M and d % 64 == 0


def silu_mul_nvfp4_permuted(x: torch.Tensor, global_scale: torch.Tensor):
    """``(fp4 [m, d/2] uint8, swizzled sf [ceil128(m) * ceil4(d/16) * 4] uint8)`` for a
    permuted gate_up output ``x`` ``[m, 2d]``."""
    assert x.dim() == 2 and x.stride(1) == 1
    m, n = x.shape
    assert n % 2 == 0
    d = n // 2
    assert d % 64 == 0, "SF columns must be a multiple of 4"
    num_sf = d // _SF_VEC
    num_k_tiles = (num_sf + 3) // 4
    m_pad = (m + 127) // 128 * 128
    fp4 = torch.empty((m, d // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((m_pad * num_k_tiles * 4,), dtype=torch.uint8, device=x.device)
    grid_rows = m + (1 if m % 128 != 0 else 0)
    if grid_rows == 0:
        return fp4, sf
    if uses_v2(m, d):
        grid = (grid_rows, triton.cdiv(num_sf, _V2_NB))
        fused_silu_mul_nvfp4_permuted_v2_kernel[grid](
            fp4.view(torch.int32),
            sf,
            sf.view(torch.int32),
            x,
            x.stride(0),
            global_scale,
            m,
            d,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            NB=_V2_NB,
            NUM_K_TILES=num_k_tiles,
            EVEN=(num_sf % _V2_NB == 0 and num_k_tiles * 4 == num_sf),
            num_warps=_V2_WARPS,
        )
        return fp4, sf
    grid = (grid_rows, triton.cdiv(d, _BLOCK))
    # PDL only while the whole grid is resident at once (one wave), so early-launched
    # consumer CTAs never wait behind unscheduled ones.
    use_pdl = get_env_enable_pdl() and grid[0] * grid[1] < _num_sms(x.device.index)
    fused_silu_mul_nvfp4_permuted_kernel[grid](
        fp4.view(torch.int32),
        sf,
        x,
        x.stride(0),
        global_scale,
        m,
        d,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        BLOCK_SIZE=_BLOCK,
        NUM_K_TILES=num_k_tiles,
        USE_PDL=use_pdl,
        num_warps=4,
        launch_pdl=use_pdl,
    )
    return fp4, sf

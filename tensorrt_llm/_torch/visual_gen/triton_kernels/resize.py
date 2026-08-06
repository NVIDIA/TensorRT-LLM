# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-point uint8 image resizing: bilinear, area-average, and bicubic.

These follow the standard fixed-point formulations used by image-processing
libraries, reproduced exactly so control frames are reproducible across
backends.  The arithmetic details below are load-bearing -- each was traced
back from a 1-LSB mismatch -- so treat them as the contract, not as
implementation freedom:

- Source coordinates ``(dx + 0.5) * scale - 0.5`` are narrowed to float32
  BEFORE the floor, and coefficients are quantised to 11-bit fixed point with
  round-half-even on the float32 product.
- Horizontal passes accumulate exactly in int32.
- The vertical descale differs per mode and is NOT a single fused rounding;
  see each function's docstring.

Layout is ``[T, H, W, C]`` uint8 on CUDA throughout.  One fused kernel per
mode: every output element gathers its source taps and runs the whole
fixed-point pipeline in registers, rather than materialising int32
intermediates between a horizontal and a vertical pass.

Bit-exactness notes: bilinear and area are pure integer, so nothing can
drift.  Bicubic's float32 lane uses ``libdevice`` ``mul_rn``/``add_rn`` so
ptxas cannot contract mul+add into an FMA (which would round differently from
the unfused eager reference), and ``float2int_rn`` for round-half-even.
"""

import functools

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 256
_COEF_BITS = 11
_COEF_SCALE = 1 << _COEF_BITS

# Axis tables are cached per (src, dst, C, device): a clip reuses the same
# handful of sizes for every frame, and rebuilding them on the host dominated
# the resize wall time (~3x the kernel time when measured). The cache is
# bounded because the key carries caller-supplied source dimensions -- a
# long-lived worker serving many distinct resolutions would otherwise retain a
# GPU table for every one it had ever seen. A transfer chain touches a handful
# of sizes, so this evicts only across unrelated requests.
_TABLE_CACHE_ENTRIES = 32


def _linear_axis(dst_n: int, src_n: int, clamp_coeffs: bool):
    d = np.arange(dst_n, dtype=np.float64)
    scale = np.float64(src_n) / np.float64(dst_n)
    f32 = ((d + 0.5) * scale - 0.5).astype(np.float32)
    s = np.floor(f32).astype(np.int64)
    f = (f32 - s.astype(np.float32)).astype(np.float32)
    if clamp_coeffs:
        # Only the horizontal axis clamps coefficients at the borders (fx=0,
        # sx pinned); the vertical axis keeps raw (sy, fy) and clips just the
        # row indices.  Clamping both costs 1 LSB on border rows.
        f = np.where(s < 0, np.float32(0), f).astype(np.float32)
        s = np.where(s < 0, 0, s)
        f = np.where(s >= src_n - 1, np.float32(0), f).astype(np.float32)
        s = np.where(s >= src_n - 1, src_n - 1, s)
    a0 = np.rint((np.float32(1.0) - f).astype(np.float32) * np.float32(_COEF_SCALE))
    a1 = np.rint(f * np.float32(_COEF_SCALE))
    s0 = np.clip(s, 0, src_n - 1)
    s1 = np.clip(s + 1, 0, src_n - 1)
    return s0, s1, a0.astype(np.int64), a1.astype(np.int64)


def _cubic_axis(dst_n: int, src_n: int):
    d = np.arange(dst_n, dtype=np.float64)
    scale = np.float64(src_n) / np.float64(dst_n)
    f32 = ((d + 0.5) * scale - 0.5).astype(np.float32)
    s = np.floor(f32).astype(np.int64)
    x = (f32 - s.astype(np.float32)).astype(np.float32)
    # Catmull-Rom-style cubic convolution kernel with A = -0.75, evaluated
    # op-for-op in float32.
    A = np.float32(-0.75)
    one = np.float32(1.0)
    xp1 = (x + one).astype(np.float32)
    c0 = ((A * xp1 - np.float32(5) * A) * xp1 + np.float32(8) * A) * xp1 - np.float32(4) * A
    c1 = ((A + 2) * x - (A + 3)) * x * x + one
    y = (one - x).astype(np.float32)
    c2 = ((A + 2) * y - (A + 3)) * y * y + one
    c3 = one - c0 - c1 - c2
    coef = [
        np.rint(c.astype(np.float32) * np.float32(_COEF_SCALE)).astype(np.int64)
        for c in (c0, c1, c2, c3)
    ]
    # Bicubic never clamps coefficients at the borders, only the tap indices.
    taps = [np.clip(s - 1 + j, 0, src_n - 1) for j in range(4)]
    return taps, coef


def _check_frames(
    frames: torch.Tensor,
    op: str,
    *,
    layout: str = "[T, H, W, C]",
    allow_outer_stride: bool = False,
) -> None:
    """Reject anything the kernels cannot address.

    Explicit raises rather than ``assert``: assertions vanish under ``python -O``,
    and only ``ValueError`` is classified as a client error by the worker, so an
    ``AssertionError`` would surface as a server fault.

    Most kernels index raw storage as one dense block and never consult strides,
    so a strided view would be read as if it were dense -- wrong pixels, no
    error. ``allow_outer_stride`` relaxes that for kernels that take dim 0's
    stride as an argument: dim 0 may then sit anywhere, but everything inside it
    must still be dense. Either way we reject rather than call ``.contiguous()``,
    since a silent full-clip copy does not belong on an inference path.
    """
    if not frames.is_cuda:
        raise ValueError(f"{op} requires a CUDA tensor, got device={frames.device}")
    if frames.dtype != torch.uint8:
        raise TypeError(f"{op} requires uint8 frames, got dtype={frames.dtype}")
    if frames.ndim != 4:
        raise ValueError(f"{op} expects {layout}, got shape={tuple(frames.shape)}")
    if allow_outer_stride:
        if not frames[0].is_contiguous():
            raise ValueError(
                f"{op} takes dim 0's stride but addresses the rest densely, so every "
                f"slice along dim 0 must be contiguous. Got shape={tuple(frames.shape)} "
                f"strides={tuple(frames.stride())}."
            )
    elif not frames.is_contiguous():
        raise ValueError(
            f"{op} requires a contiguous tensor; the kernels address storage densely and "
            f"would misread a strided view. Got shape={tuple(frames.shape)} "
            f"strides={tuple(frames.stride())} -- call .contiguous() first."
        )


@functools.lru_cache(maxsize=_TABLE_CACHE_ENTRIES)
def _linear_tables_x(src_w, dst_w, C, dev):
    """Horizontal tables re-indexed per output *byte* (k = x*C + c), so the
    kernel loads them contiguously instead of gathering per-x through a
    runtime div/mod."""
    xs0, xs1, xa0, xa1 = _linear_axis(dst_w, src_w, clamp_coeffs=True)
    ch = np.tile(np.arange(C, dtype=np.int64), dst_w)
    tabs = (
        np.repeat(xs0.astype(np.int64) * C, C) + ch,
        np.repeat(xs1.astype(np.int64) * C, C) + ch,
        np.repeat(xa0, C),
        np.repeat(xa1, C),
    )
    return tuple(torch.from_numpy(a).to(device=dev, dtype=torch.int32) for a in tabs)


@functools.lru_cache(maxsize=_TABLE_CACHE_ENTRIES)
def _linear_tables_y(src_h, dst_h, dev):
    return tuple(
        torch.from_numpy(a).to(device=dev, dtype=torch.int32)
        for a in _linear_axis(dst_h, src_h, clamp_coeffs=False)
    )


# ---------------------------------------------------------------------------
# Bilinear
# ---------------------------------------------------------------------------
# Grid scheme (all kernels here): (x-block, y, t) 3D grid + constexpr C, so
# there is NO runtime integer division in the hot path -- profiling showed the
# flat-index variant spending ~80% SM on idiv at 1-15% DRAM.
@triton.jit
def _linear_kernel(
    src,
    dst,
    xo0,
    xo1,
    aa0,
    aa1,
    sy0,
    sy1,
    ay0,
    ay1,
    SH,
    SW,
    DH,
    DW,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Two output rows per program: the x-tables are loaded once for both, and
    # the second row's independent gathers give the scheduler work to issue
    # while the first row's loads are in flight (the kernel is issue-bound).
    yA = tl.program_id(1) * 2
    t = tl.program_id(2)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)  # x*C + c within the row
    m = offs < DW * C
    yB = yA + 1
    mB = yB < DH

    x0 = tl.load(xo0 + offs, mask=m, other=0)  # sx*C + c, contiguous
    x1 = tl.load(xo1 + offs, mask=m, other=0)
    a0 = tl.load(aa0 + offs, mask=m, other=0)
    a1 = tl.load(aa1 + offs, mask=m, other=0)

    base = t * SH * SW * C
    out = dst + (t * DH + yA) * DW * C + offs

    y0 = tl.load(sy0 + yA)
    y1 = tl.load(sy1 + yA)
    b0 = tl.load(ay0 + yA)
    b1 = tl.load(ay1 + yA)
    r0 = base + y0 * SW * C
    r1 = base + y1 * SW * C
    p00 = tl.load(src + r0 + x0, mask=m, other=0).to(tl.int32)
    p01 = tl.load(src + r0 + x1, mask=m, other=0).to(tl.int32)
    p10 = tl.load(src + r1 + x0, mask=m, other=0).to(tl.int32)
    p11 = tl.load(src + r1 + x1, mask=m, other=0).to(tl.int32)
    h0 = p00 * a0 + p01 * a1
    h1 = p10 * a0 + p11 * a1
    v = (((h0 >> 4) * b0) >> 16) + (((h1 >> 4) * b1) >> 16)
    r = (v + 2) >> 2
    r = tl.minimum(tl.maximum(r, 0), 255)
    tl.store(out, r.to(tl.uint8), mask=m)

    y0 = tl.load(sy0 + yB, mask=mB, other=0)
    y1 = tl.load(sy1 + yB, mask=mB, other=0)
    b0 = tl.load(ay0 + yB, mask=mB, other=0)
    b1 = tl.load(ay1 + yB, mask=mB, other=0)
    r0 = base + y0 * SW * C
    r1 = base + y1 * SW * C
    p00 = tl.load(src + r0 + x0, mask=m & mB, other=0).to(tl.int32)
    p01 = tl.load(src + r0 + x1, mask=m & mB, other=0).to(tl.int32)
    p10 = tl.load(src + r1 + x0, mask=m & mB, other=0).to(tl.int32)
    p11 = tl.load(src + r1 + x1, mask=m & mB, other=0).to(tl.int32)
    h0 = p00 * a0 + p01 * a1
    h1 = p10 * a0 + p11 * a1
    v = (((h0 >> 4) * b0) >> 16) + (((h1 >> 4) * b1) >> 16)
    r = (v + 2) >> 2
    r = tl.minimum(tl.maximum(r, 0), 255)
    tl.store(out + DW * C, r.to(tl.uint8), mask=m & mB)


def resize_linear_u8(frames: torch.Tensor, dst_w: int, dst_h: int) -> torch.Tensor:
    """Bilinear resize: uint8 ``[T, H, W, C]`` CUDA -> ``[T, dst_h, dst_w, C]``.

    The uint8 vertical pass truncates PER TERM rather than as one fused
    descale::

        dst = (((b0 * (h0 >> 4)) >> 16) + ((b1 * (h1 >> 4)) >> 16) + 2) >> 2
    """
    _check_frames(frames, "resize_linear_u8")
    T, H, W, C = frames.shape
    dev = frames.device
    xo0, xo1, aa0, aa1 = _linear_tables_x(W, dst_w, C, dev)
    ys0, ys1, ya0, ya1 = _linear_tables_y(H, dst_h, dev)
    dst = torch.empty(T, dst_h, dst_w, C, dtype=torch.uint8, device=dev)
    _linear_kernel[(triton.cdiv(dst_w * C, _BLOCK), triton.cdiv(dst_h, 2), T)](
        frames, dst, xo0, xo1, aa0, aa1, ys0, ys1, ya0, ya1, H, W, dst_h, dst_w, C=C, BLOCK=_BLOCK
    )
    return dst


# ---------------------------------------------------------------------------
# Area average (integer factors 2 / 4)
# ---------------------------------------------------------------------------
@triton.jit
def _area_kernel(
    src,
    dst,
    SH,
    SW,
    DH,
    DW,
    C: tl.constexpr,
    F: tl.constexpr,
    HALF_UP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    y = tl.program_id(1)
    t = tl.program_id(2)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < DW * C
    x = offs // C
    c = offs % C

    base = t * SH * SW * C + c
    s = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in tl.static_range(F):
        row = base + (y * F + i) * SW * C
        for j in tl.static_range(F):
            s += tl.load(src + row + (x * F + j) * C, mask=m, other=0).to(tl.int32)
    if HALF_UP:
        r = (s + 2) >> 2
    else:
        BITS: tl.constexpr = 2 if F == 2 else 4
        HALF: tl.constexpr = 1 << (BITS - 1)
        r = (s + HALF - 1 + ((s >> BITS) & 1)) >> BITS
    r = tl.minimum(tl.maximum(r, 0), 255)
    tl.store(dst + (t * DH + y) * DW * C + offs, r.to(tl.uint8), mask=m)


@triton.jit
def _area3_kernel(
    src,
    dst,
    NLANES,
    SW: tl.constexpr,
    DW: tl.constexpr,
    F: tl.constexpr,
    HALF_UP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # SW/DW constexpr: the lane->(row, p) split and row-stride math strength-
    # reduce to mul/shift, off the critical path in front of the loads.  One
    # compile per (source, dest) size, which a clip reuses for every frame.
    # C=3 fast path: a lane owns 24 contiguous, word-aligned source bytes per
    # row (6 int32 loads) -> 4 output pixels for F=2 or 2 for F=4, instead of
    # the generic kernel's per-byte gathers (whose load latency was 63% of
    # stall time at 24% DRAM).  F=2 stores 3 full aligned words per lane; an
    # earlier pixel-pair variant's stride-6 u16 stores wasted write sectors.
    # One flat 1D grid over all lanes: per-row blocks were too small and
    # short-lived to hide latency (32% occupancy, 74% long_scoreboard).
    # Since SH = DH*F, the source row is F*row + i with row the flattened
    # (t, y) index -- no per-lane t/y split needed.
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = lane < NLANES
    P: tl.constexpr = 8 // F  # output pixels per lane
    NPL = DW // P
    row = lane // NPL
    p = lane % NPL
    src32 = src.to(tl.pointer_type(tl.int32))

    if F == 2:
        # both rows' 12 loads issued before any extraction: interleaving the
        # ~40 byte-extraction ops between 6-load batches halved the memory-
        # level parallelism and left DRAM under 80%
        rw = (2 * row) * (SW * 3 // 4) + p * 6
        # single-touch streams: evict_first keeps L2 free for write coalescing
        ep: tl.constexpr = "evict_first"
        w0 = tl.load(src32 + rw, mask=m, other=0, eviction_policy=ep)
        w1 = tl.load(src32 + rw + 1, mask=m, other=0, eviction_policy=ep)
        w2 = tl.load(src32 + rw + 2, mask=m, other=0, eviction_policy=ep)
        w3 = tl.load(src32 + rw + 3, mask=m, other=0, eviction_policy=ep)
        w4 = tl.load(src32 + rw + 4, mask=m, other=0, eviction_policy=ep)
        w5 = tl.load(src32 + rw + 5, mask=m, other=0, eviction_policy=ep)
        rw2 = rw + SW * 3 // 4
        x0 = tl.load(src32 + rw2, mask=m, other=0, eviction_policy=ep)
        x1 = tl.load(src32 + rw2 + 1, mask=m, other=0, eviction_policy=ep)
        x2 = tl.load(src32 + rw2 + 2, mask=m, other=0, eviction_policy=ep)
        x3 = tl.load(src32 + rw2 + 3, mask=m, other=0, eviction_policy=ep)
        x4 = tl.load(src32 + rw2 + 4, mask=m, other=0, eviction_policy=ep)
        x5 = tl.load(src32 + rw2 + 5, mask=m, other=0, eviction_policy=ep)
        o0 = (w0 & 255) + ((w0 >> 24) & 255) + (x0 & 255) + ((x0 >> 24) & 255)
        o1 = ((w0 >> 8) & 255) + (w1 & 255) + ((x0 >> 8) & 255) + (x1 & 255)
        o2 = ((w0 >> 16) & 255) + ((w1 >> 8) & 255) + ((x0 >> 16) & 255) + ((x1 >> 8) & 255)
        o3 = ((w1 >> 16) & 255) + ((w2 >> 8) & 255) + ((x1 >> 16) & 255) + ((x2 >> 8) & 255)
        o4 = ((w1 >> 24) & 255) + ((w2 >> 16) & 255) + ((x1 >> 24) & 255) + ((x2 >> 16) & 255)
        o5 = (w2 & 255) + ((w2 >> 24) & 255) + (x2 & 255) + ((x2 >> 24) & 255)
        o6 = (w3 & 255) + ((w3 >> 24) & 255) + (x3 & 255) + ((x3 >> 24) & 255)
        o7 = ((w3 >> 8) & 255) + (w4 & 255) + ((x3 >> 8) & 255) + (x4 & 255)
        o8 = ((w3 >> 16) & 255) + ((w4 >> 8) & 255) + ((x3 >> 16) & 255) + ((x4 >> 8) & 255)
        o9 = ((w4 >> 16) & 255) + ((w5 >> 8) & 255) + ((x4 >> 16) & 255) + ((x5 >> 8) & 255)
        o10 = ((w4 >> 24) & 255) + ((w5 >> 16) & 255) + ((x4 >> 24) & 255) + ((x5 >> 16) & 255)
        o11 = (w5 & 255) + ((w5 >> 24) & 255) + (x5 & 255) + ((x5 >> 24) & 255)
    else:
        o0 = tl.zeros((BLOCK,), dtype=tl.int32)
        o1 = tl.zeros((BLOCK,), dtype=tl.int32)
        o2 = tl.zeros((BLOCK,), dtype=tl.int32)
        o3 = tl.zeros((BLOCK,), dtype=tl.int32)
        o4 = tl.zeros((BLOCK,), dtype=tl.int32)
        o5 = tl.zeros((BLOCK,), dtype=tl.int32)
        for i in tl.static_range(F):
            # word offset of this lane's 24 input bytes in source row F*row + i
            rw = (F * row + i) * (SW * 3 // 4) + p * 6
            w0 = tl.load(src32 + rw, mask=m, other=0)
            w1 = tl.load(src32 + rw + 1, mask=m, other=0)
            w2 = tl.load(src32 + rw + 2, mask=m, other=0)
            w3 = tl.load(src32 + rw + 3, mask=m, other=0)
            w4 = tl.load(src32 + rw + 4, mask=m, other=0)
            w5 = tl.load(src32 + rw + 5, mask=m, other=0)
            o0 += (w0 & 255) + ((w0 >> 24) & 255) + ((w1 >> 16) & 255) + ((w2 >> 8) & 255)
            o1 += ((w0 >> 8) & 255) + (w1 & 255) + ((w1 >> 24) & 255) + ((w2 >> 16) & 255)
            o2 += ((w0 >> 16) & 255) + ((w1 >> 8) & 255) + (w2 & 255) + ((w2 >> 24) & 255)
            o3 += (w3 & 255) + ((w3 >> 24) & 255) + ((w4 >> 16) & 255) + ((w5 >> 8) & 255)
            o4 += ((w3 >> 8) & 255) + (w4 & 255) + ((w4 >> 24) & 255) + ((w5 >> 16) & 255)
            o5 += ((w3 >> 16) & 255) + ((w4 >> 8) & 255) + (w5 & 255) + ((w5 >> 24) & 255)

    BITS: tl.constexpr = 2 if F == 2 else 4
    HALF: tl.constexpr = 1 << (BITS - 1)
    if HALF_UP:
        r0 = (o0 + 2) >> 2
        r1 = (o1 + 2) >> 2
        r2 = (o2 + 2) >> 2
        r3 = (o3 + 2) >> 2
        r4 = (o4 + 2) >> 2
        r5 = (o5 + 2) >> 2
    else:
        r0 = (o0 + HALF - 1 + ((o0 >> BITS) & 1)) >> BITS
        r1 = (o1 + HALF - 1 + ((o1 >> BITS) & 1)) >> BITS
        r2 = (o2 + HALF - 1 + ((o2 >> BITS) & 1)) >> BITS
        r3 = (o3 + HALF - 1 + ((o3 >> BITS) & 1)) >> BITS
        r4 = (o4 + HALF - 1 + ((o4 >> BITS) & 1)) >> BITS
        r5 = (o5 + HALF - 1 + ((o5 >> BITS) & 1)) >> BITS

    if F == 2:
        r6 = (o6 + 2) >> 2
        r7 = (o7 + 2) >> 2
        r8 = (o8 + 2) >> 2
        r9 = (o9 + 2) >> 2
        r10 = (o10 + 2) >> 2
        r11 = (o11 + 2) >> 2
        dst32 = dst.to(tl.pointer_type(tl.int32))
        ow = row * (DW * 3) // 4 + p * 3
        tl.store(dst32 + ow, r0 | (r1 << 8) | (r2 << 16) | (r3 << 24), mask=m)
        tl.store(dst32 + ow + 1, r4 | (r5 << 8) | (r6 << 16) | (r7 << 24), mask=m)
        tl.store(dst32 + ow + 2, r8 | (r9 << 8) | (r10 << 16) | (r11 << 24), mask=m)
    else:
        dst16 = dst.to(tl.pointer_type(tl.uint16))
        ob = row * (DW * 3) // 2 + p * 3
        tl.store(dst16 + ob, (r0 | (r1 << 8)).to(tl.uint16), mask=m)
        tl.store(dst16 + ob + 1, (r2 | (r3 << 8)).to(tl.uint16), mask=m)
        tl.store(dst16 + ob + 2, (r4 | (r5 << 8)).to(tl.uint16), mask=m)


def resize_area_u8(frames: torch.Tensor, factor: int) -> torch.Tensor:
    """Area-average downscale by an integer ``factor`` of 2 or 4.

    Rounds half-UP for factor 2 with 1/3/4 channels and half-EVEN otherwise,
    matching the integer fast path and the generic ``sum * 1/area`` path
    respectively.  Non-divisible dimensions would need fractional area
    weights, which are deliberately unimplemented: every supported output
    bucket is a multiple of 16.
    """
    _check_frames(frames, "resize_area_u8")
    T, H, W, C = frames.shape
    if factor not in (2, 4):
        raise ValueError(f"resize_area_u8 factor={factor}, expected one of (2, 4)")
    if H % factor or W % factor:
        raise ValueError(f"resize_area_u8 source {W}x{H} not divisible by factor={factor}")
    dh, dw = H // factor, W // factor
    dst = torch.empty(T, dh, dw, C, dtype=torch.uint8, device=frames.device)
    half_up = factor == 2 and C in (1, 3, 4)
    pix = 8 // factor  # output pixels per lane (24 source bytes either way)
    # data_ptr alignment matters: the C=3 path reinterprets the buffer as
    # int32, and a contiguous *view* can still start on an odd byte, which
    # faults the device rather than returning wrong data.
    word_aligned = frames.data_ptr() % 4 == 0 and (W * 3) % 4 == 0
    if C == 3 and dw % pix == 0 and word_aligned and frames.is_contiguous():
        nlanes = T * dh * (dw // pix)
        # one lane per thread: this kernel streams (DRAM-bound), so resident
        # warps matter more than per-thread ILP (2 lanes/thread measured
        # slower for both factors)
        _area3_kernel[(triton.cdiv(nlanes, 256),)](
            frames, dst, nlanes, SW=W, DW=dw, F=factor, HALF_UP=half_up, BLOCK=256, num_warps=8
        )
    else:
        _area_kernel[(triton.cdiv(dw * C, _BLOCK), dh, T)](
            frames, dst, H, W, dh, dw, C=C, F=factor, HALF_UP=half_up, BLOCK=_BLOCK
        )
    return dst


# ---------------------------------------------------------------------------
# Bicubic
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=_TABLE_CACHE_ENTRIES)
def _cubic_tables_x(src_w, dst_w, C, dev):
    """Horizontal tap/coeff tables re-indexed per output byte, like bilinear."""
    xtaps, xcoef = _cubic_axis(dst_w, src_w)
    ch = np.tile(np.arange(C, dtype=np.int64), dst_w)
    txb = np.repeat(np.stack(xtaps).astype(np.int64) * C, C, axis=1) + ch[None, :]
    cxb = np.repeat(np.stack(xcoef), C, axis=1)
    return (
        torch.from_numpy(txb).to(device=dev, dtype=torch.int32).contiguous(),
        torch.from_numpy(cxb).to(device=dev, dtype=torch.int32).contiguous(),
    )


@functools.lru_cache(maxsize=_TABLE_CACHE_ENTRIES)
def _cubic_tables_y(src_h, dst_h, dev):
    ytaps, ycoef = _cubic_axis(dst_h, src_h)
    return (
        torch.from_numpy(np.stack(ytaps)).to(device=dev, dtype=torch.int32),
        torch.from_numpy(np.stack(ycoef)).to(device=dev, dtype=torch.int32),
    )


@triton.jit
def _cubic_kernel(
    src,
    dst,
    xt,
    xc,
    yt,
    yc,
    SH,
    SW,
    DH,
    DW,
    C: tl.constexpr,
    covered,
    TAIL: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # TAIL=False when DW*C is a multiple of 8: the hybrid vertical pass then
    # takes the float lane for EVERY element and the integer accumulators are
    # dead work (~15% of the issue-bound instruction stream) -- skip them at
    # compile time.  All production bucket sizes are tail-free.
    # R=2: two output rows per program, sharing the 8 x-table loads per tap
    # column and doubling the independent gathers in flight -- pays off on
    # wide rows (upscale).  R=1 for narrow rows (deep downscale), where the
    # doubled accumulator set costs occupancy instead.
    yA = tl.program_id(1) * R
    t = tl.program_id(2)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < DW * C
    yB = yA + 1
    mB = yB < DH

    base = t * SH * SW * C
    vA_int = tl.zeros((BLOCK,), dtype=tl.int32)
    vA_flt = tl.zeros((BLOCK,), dtype=tl.float32)
    vB_int = tl.zeros((BLOCK,), dtype=tl.int32)
    vB_flt = tl.zeros((BLOCK,), dtype=tl.float32)
    INV: tl.constexpr = 1.0 / 4194304.0  # 2^-22, exact
    # k walks the vertical taps last-to-first so the float chain associates
    # b3-first, matching the reference.
    for k in tl.static_range(3, -1, -1):
        tyA = tl.load(yt + k * DH + yA)
        byA = tl.load(yc + k * DH + yA)
        rowA = base + tyA * SW * C
        hA = tl.zeros((BLOCK,), dtype=tl.int32)
        if R == 2:
            tyB = tl.load(yt + k * DH + yB, mask=mB, other=0)
            byB = tl.load(yc + k * DH + yB, mask=mB, other=0)
            rowB = base + tyB * SW * C
            hB = tl.zeros((BLOCK,), dtype=tl.int32)
        for j in tl.static_range(4):
            tx = tl.load(xt + j * DW * C + offs, mask=m, other=0)  # sx*C + c
            ax = tl.load(xc + j * DW * C + offs, mask=m, other=0)
            hA += tl.load(src + rowA + tx, mask=m, other=0).to(tl.int32) * ax
            if R == 2:
                hB += tl.load(src + rowB + tx, mask=m & mB, other=0).to(tl.int32) * ax
        if TAIL:
            vA_int += hA * byA
        vA_flt = libdevice.add_rn(
            libdevice.mul_rn(hA.to(tl.float32), byA.to(tl.float32) * INV), vA_flt
        )
        if R == 2:
            if TAIL:
                vB_int += hB * byB
            vB_flt = libdevice.add_rn(
                libdevice.mul_rn(hB.to(tl.float32), byB.to(tl.float32) * INV), vB_flt
            )

    out = dst + (t * DH + yA) * DW * C + offs
    rA = libdevice.float2int_rn(vA_flt)
    if TAIL:
        rA_int = (vA_int + 2097152) >> 22  # fixed-point cast, 22 fractional bits
        rA = tl.where(offs < covered, rA, rA_int)
    rA = tl.minimum(tl.maximum(rA, 0), 255)
    tl.store(out, rA.to(tl.uint8), mask=m)
    if R == 2:
        rB = libdevice.float2int_rn(vB_flt)
        if TAIL:
            rB_int = (vB_int + 2097152) >> 22
            rB = tl.where(offs < covered, rB, rB_int)
        rB = tl.minimum(tl.maximum(rB, 0), 255)
        tl.store(out + DW * C, rB.to(tl.uint8), mask=m & mB)


def resize_cubic_u8(frames: torch.Tensor, dst_w: int, dst_h: int) -> torch.Tensor:
    """Bicubic resize: uint8 ``[T, H, W, C]`` CUDA -> ``[T, dst_h, dst_w, C]``.

    The uint8 vertical pass is hybrid per output row of ``n = dst_w * C``
    elements: the vectorised lane covers ``[0, 8 * (n // 8))`` in float32
    (b3-first mul/add chain, round-half-even) and the row tail uses the
    scalar integer fixed-point cast ``(v + 2^21) >> 22``.
    """
    _check_frames(frames, "resize_cubic_u8")
    T, H, W, C = frames.shape
    dev = frames.device
    xt, xc = _cubic_tables_x(W, dst_w, C, dev)
    yt, yc = _cubic_tables_y(H, dst_h, dev)
    dst = torch.empty(T, dst_h, dst_w, C, dtype=torch.uint8, device=dev)
    covered = (dst_w * C // 8) * 8
    # R=2 on wide rows (ILP is what feeds this issue-bound kernel); narrow
    # deep-downscale rows keep R=1, where the doubled accumulators would cost
    # occupancy instead.
    R = 2 if dst_w * C >= 1024 else 1
    BLOCK = 256 if R == 2 else 128
    _cubic_kernel[(triton.cdiv(dst_w * C, BLOCK), triton.cdiv(dst_h, R), T)](
        frames,
        dst,
        xt,
        xc,
        yt,
        yc,
        H,
        W,
        dst_h,
        dst_w,
        C=C,
        covered=covered,
        TAIL=(dst_w * C) % 8 != 0,
        R=R,
        BLOCK=BLOCK,
    )
    return dst

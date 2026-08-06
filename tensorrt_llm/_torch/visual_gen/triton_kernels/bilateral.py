# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bilateral filter (Tomasi & Manduchi, 1998) for uint8 RGB clips.

Contract, in the widely-used formulation this reproduces:

- ``radius = max(d // 2, 1)``; the support is the CIRCLE
  ``sqrt(i^2 + j^2) <= radius``, not the enclosing square.
- Space weight ``exp(r^2 * -0.5 / sigma_space^2)``; colour weight is a lookup
  over the L1 channel distance ``|db| + |dg| + |dr|``, i.e.
  ``exp(k^2 * -0.5 / sigma_color^2)``.
- Borders reflect without repeating the edge pixel.
- Accumulation and the final divide are float32, with taps visited in
  row-major circular order so the sums associate identically to the reference.

Unlike the resize and Canny paths, this op cannot be made bitwise reproducible
against an arbitrary CPU SIMD implementation: float32 tap-summation order
decides pixels that land on a .5 rounding tie, so two vectorisations of the
same formula legitimately differ by 1 LSB on a small fraction of pixels. The
enforced contract is therefore bitwise equality with the torch reference in
``reference.py``, which fixes the summation order.
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .resize import _check_frames


def circle_offsets(radius: int, gauss_space_coeff: float):
    """Taps of the circular support in row-major order, with space weights.

    Shared with the reference implementation so both see identical tables and
    any mismatch is attributable to the arithmetic, not the setup.
    """
    dys, dxs, sws = [], [], []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            r = math.sqrt(float(i * i + j * j))
            if r > radius:
                continue
            dys.append(i)
            dxs.append(j)
            # evaluated in double, stored as float32
            sws.append(math.exp(r * r * gauss_space_coeff))
    return dys, dxs, sws


def color_lut(channels: int, gauss_color_coeff: float, device: torch.device) -> torch.Tensor:
    """Colour weight lookup over the L1 channel distance, float32."""
    return (
        (torch.arange(256 * channels, dtype=torch.float64, device=device) ** 2 * gauss_color_coeff)
        .exp()
        .to(torch.float32)
    )


def reflect_pad(frames: torch.Tensor, radius: int) -> torch.Tensor:
    """uint8 ``[T, H, W, C]`` -> float32 ``[T, H+2r, W+2r, C]``, reflected."""
    x = frames.permute(0, 3, 1, 2).to(torch.float32)
    return F.pad(x, (radius,) * 4, mode="reflect").permute(0, 2, 3, 1).contiguous()


@triton.jit
def _bilateral_kernel(
    src_ptr,
    dst_ptr,
    lut_ptr,
    dy_ptr,
    dx_ptr,
    sw_ptr,
    T,
    H,
    W,
    Hp,
    Wp,
    K,
    radius,
    BLOCK: tl.constexpr,
):
    # (x-block, y, t) grid: no runtime integer division in the hot path
    y = tl.program_id(1)
    t = tl.program_id(2)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = x < W
    offs = (t * H + y) * W + x

    cbase = ((t * Hp + y + radius) * Wp + x + radius) * 3
    cb = tl.load(src_ptr + cbase + 0, mask=m, other=0.0)
    cg = tl.load(src_ptr + cbase + 1, mask=m, other=0.0)
    cr = tl.load(src_ptr + cbase + 2, mask=m, other=0.0)

    sb = tl.zeros((BLOCK,), dtype=tl.float32)
    sg = tl.zeros((BLOCK,), dtype=tl.float32)
    sr = tl.zeros((BLOCK,), dtype=tl.float32)
    ws = tl.zeros((BLOCK,), dtype=tl.float32)

    for k in range(K):  # K is runtime, so this stays a serial loop in tap order
        dy = tl.load(dy_ptr + k)
        dx = tl.load(dx_ptr + k)
        sw = tl.load(sw_ptr + k)
        nbase = ((t * Hp + y + radius + dy) * Wp + x + radius + dx) * 3
        nb = tl.load(src_ptr + nbase + 0, mask=m, other=0.0)
        ng = tl.load(src_ptr + nbase + 1, mask=m, other=0.0)
        nr = tl.load(src_ptr + nbase + 2, mask=m, other=0.0)
        dist = tl.abs(nb - cb) + tl.abs(ng - cg) + tl.abs(nr - cr)
        w = tl.load(lut_ptr + dist.to(tl.int32), mask=m, other=0.0) * sw
        # mul_rn/add_rn keep the accumulation unfused: ptxas would contract
        # nb*w + sb into an FMA, which rounds differently from the reference's
        # separate mul and add on .5-tie pixels.
        sb = libdevice.add_rn(libdevice.mul_rn(nb, w), sb)
        sg = libdevice.add_rn(libdevice.mul_rn(ng, w), sg)
        sr = libdevice.add_rn(libdevice.mul_rn(nr, w), sr)
        ws = libdevice.add_rn(w, ws)  # += would let ptxas fuse the lut*sw mul into an fma

    obase = offs * 3
    # div_rn: Triton's `/` is not guaranteed IEEE correctly-rounded on f32,
    # and a 1-ulp quotient difference flips torch.round on .5-tie pixels.
    tl.store(dst_ptr + obase + 0, libdevice.div_rn(sb, ws), mask=m)
    tl.store(dst_ptr + obase + 1, libdevice.div_rn(sg, ws), mask=m)
    tl.store(dst_ptr + obase + 2, libdevice.div_rn(sr, ws), mask=m)


def bilateral_filter(
    frames: torch.Tensor, d: int, sigma_color: float, sigma_space: float
) -> torch.Tensor:
    """Bilateral filter over a clip: uint8 ``[T, H, W, 3]`` CUDA -> same shape."""
    _check_frames(frames, "bilateral_filter")
    if frames.shape[-1] != 3:
        raise ValueError(f"bilateral_filter expects [T, H, W, 3], got shape={tuple(frames.shape)}")
    T, H, W, C = frames.shape
    dev = frames.device
    radius = max(d // 2, 1)

    lut = color_lut(C, -0.5 / (sigma_color * sigma_color), dev)
    dys, dxs, sws = circle_offsets(radius, -0.5 / (sigma_space * sigma_space))
    dy = torch.tensor(dys, dtype=torch.int32, device=dev)
    dx = torch.tensor(dxs, dtype=torch.int32, device=dev)
    sw = torch.tensor(sws, dtype=torch.float32, device=dev)

    src = reflect_pad(frames, radius)
    dst = torch.empty(T * H * W * 3, dtype=torch.float32, device=dev)
    BLOCK = 256
    _bilateral_kernel[(triton.cdiv(W, BLOCK), H, T)](
        src.reshape(-1),
        dst,
        lut,
        dy,
        dx,
        sw,
        T,
        H,
        W,
        src.shape[1],
        src.shape[2],
        len(dys),
        radius,
        BLOCK=BLOCK,
    )
    return torch.round(dst.reshape(T, H, W, 3)).clamp(0, 255).to(torch.uint8)

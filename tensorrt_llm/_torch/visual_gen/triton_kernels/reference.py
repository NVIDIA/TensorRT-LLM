# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch-op reference implementations of the control-generation kernels.

These are the executable specification of the arithmetic the Triton kernels
implement: readable, obviously-correct-by-inspection, and slow.  Nothing in
the inference path calls them -- they exist so
``tests/unittest/_torch/visual_gen/test_control_kernels.py`` can assert the
kernels bitwise, and so a future change to a kernel has something precise to
be checked against.

The axis tables, tap offsets and lookup tables are imported from the kernel
modules rather than rebuilt here: a mismatch should point at the arithmetic,
not at two subtly different table setups.  For the same reason Canny's
hysteresis is shared, since it is a torch loop in both paths.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .bilateral import circle_offsets, color_lut, reflect_pad
from .canny import _hysteresis
from .resize import _COEF_BITS, _COEF_SCALE, _check_frames, _cubic_axis, _linear_axis

_SHIFT = 15
_TG22 = 13573  # round(tan(22.5deg) * 2**15)
_SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
_SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]


def _dev_idx(a: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(a).to(device=device, dtype=torch.int32)


def canny_edges(frames: torch.Tensor, low: int, high: int) -> torch.Tensor:
    """Canny over a clip: uint8 ``[C, T, H, W]`` CUDA -> uint8 ``[T, H, W]``."""
    C, T, H, W = frames.shape
    dev = frames.device

    # Sobel values fit float32 exactly (|dx| <= 1020), then integer thereafter.
    x = frames.reshape(C * T, 1, H, W).to(torch.float32)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    kx = torch.tensor(_SOBEL_X, device=dev)[None, None]
    ky = torch.tensor(_SOBEL_Y, device=dev)[None, None]
    dx = F.conv2d(xp, kx).to(torch.int32).reshape(C, T, H, W)
    dy = F.conv2d(xp, ky).to(torch.int32).reshape(C, T, H, W)

    bdx, bdy = dx[0], dy[0]
    if C > 1:
        best = bdx.abs() + bdy.abs()
        for c in range(1, C):
            mag_c = dx[c].abs() + dy[c].abs()
            take = mag_c > best  # strict: the first channel wins ties
            best = torch.where(take, mag_c, best)
            bdx = torch.where(take, dx[c], bdx)
            bdy = torch.where(take, dy[c], bdy)

    mag = bdx.abs() + bdy.abs()
    ax = bdx.abs()
    y15 = bdy.abs() << _SHIFT
    tg22x = ax * _TG22
    tg67x = tg22x + (ax << 16)
    horiz = y15 < tg22x
    vert = (~horiz) & (y15 > tg67x)
    diag = ~(horiz | vert)
    s_pos = (bdx ^ bdy) >= 0

    m = F.pad(mag, (1, 1, 1, 1))  # zeros outside, matching the map borders
    c = m[:, 1 : H + 1, 1 : W + 1]

    def nb(dr: int, dc: int) -> torch.Tensor:
        return m[:, 1 + dr : H + 1 + dr, 1 + dc : W + 1 + dc]

    keep = horiz & (c > nb(0, -1)) & (c >= nb(0, 1))
    keep |= vert & (c > nb(-1, 0)) & (c >= nb(1, 0))
    keep |= diag & s_pos & (c > nb(-1, -1)) & (c > nb(1, 1))
    keep |= diag & ~s_pos & (c > nb(-1, 1)) & (c > nb(1, -1))

    strong = (keep & (mag > high)).to(torch.float32)[:, None]
    weak = (keep & (mag > low)).to(torch.float32)[:, None]
    return (_hysteresis(strong, weak)[:, 0] > 0).to(torch.uint8) * 255


def bilateral_filter(
    frames: torch.Tensor, d: int, sigma_color: float, sigma_space: float
) -> torch.Tensor:
    """Bilateral filter over a clip: uint8 ``[T, H, W, 3]`` CUDA -> same shape.

    One full-size temporary per tap, so it is O(taps) in both time and
    bandwidth -- fine for test sizes, hopeless at production resolution.
    """
    T, H, W, C = frames.shape
    dev = frames.device
    radius = max(d // 2, 1)

    lut = color_lut(C, -0.5 / (sigma_color * sigma_color), dev)
    dys, dxs, sws = circle_offsets(radius, -0.5 / (sigma_space * sigma_space))
    sw = torch.tensor(sws, dtype=torch.float32, device=dev)

    src = reflect_pad(frames, radius)
    center = src[:, radius : radius + H, radius : radius + W, :]
    total = torch.zeros(T, H, W, C, dtype=torch.float32, device=dev)
    wsum = torch.zeros(T, H, W, 1, dtype=torch.float32, device=dev)
    for k, (i, j) in enumerate(zip(dys, dxs)):  # row-major circular tap order
        nb = src[:, radius + i : radius + i + H, radius + j : radius + j + W, :]
        dist = (nb - center).abs().sum(-1).long()
        w = (lut[dist] * sw[k]).unsqueeze(-1)
        total = nb * w + total  # unfused mul then add, as the kernel forces
        wsum = w + wsum
    return torch.round(total / wsum).clamp(0, 255).to(torch.uint8)


def resize_linear_u8(frames: torch.Tensor, dst_w: int, dst_h: int) -> torch.Tensor:
    """Bilinear resize: uint8 ``[T, H, W, C]`` CUDA -> ``[T, dst_h, dst_w, C]``."""
    _check_frames(frames, "resize_linear_u8")
    T, H, W, C = frames.shape
    dev = frames.device
    sx0, sx1, ax0, ax1 = _linear_axis(dst_w, W, clamp_coeffs=True)
    sy0, sy1, ay0, ay1 = _linear_axis(dst_h, H, clamp_coeffs=False)

    f = frames.to(torch.int32)
    tx0 = _dev_idx(ax0, dev)[None, None, :, None]
    tx1 = _dev_idx(ax1, dev)[None, None, :, None]
    h = f.index_select(2, _dev_idx(sx0, dev)) * tx0 + f.index_select(2, _dev_idx(sx1, dev)) * tx1
    ty0 = _dev_idx(ay0, dev)[None, :, None, None]
    ty1 = _dev_idx(ay1, dev)[None, :, None, None]
    v = ((h.index_select(1, _dev_idx(sy0, dev)) >> 4) * ty0 >> 16) + (
        (h.index_select(1, _dev_idx(sy1, dev)) >> 4) * ty1 >> 16
    )
    return ((v + 2) >> 2).clamp(0, 255).to(torch.uint8)


def resize_area_u8(frames: torch.Tensor, factor: int) -> torch.Tensor:
    """Area-average downscale by an integer ``factor`` of 2 or 4."""
    _check_frames(frames, "resize_area_u8")
    T, H, W, C = frames.shape
    dh, dw = H // factor, W // factor
    s = frames.to(torch.int32).reshape(T, dh, factor, dw, factor, C).sum(dim=(2, 4))
    if factor == 2 and C in (1, 3, 4):
        out = (s + 2) >> 2
    else:
        bits = 2 if factor == 2 else 4
        half = 1 << (bits - 1)
        # branch-free round-half-even for division by 2**bits
        out = (s + half - 1 + ((s >> bits) & 1)) >> bits
    return out.clamp(0, 255).to(torch.uint8)


def resize_cubic_u8(frames: torch.Tensor, dst_w: int, dst_h: int) -> torch.Tensor:
    """Bicubic resize: uint8 ``[T, H, W, C]`` CUDA -> ``[T, dst_h, dst_w, C]``."""
    _check_frames(frames, "resize_cubic_u8")
    T, H, W, C = frames.shape
    dev = frames.device
    xtaps, xcoef = _cubic_axis(dst_w, W)
    ytaps, ycoef = _cubic_axis(dst_h, H)

    f = frames.to(torch.int32)
    h = sum(
        f.index_select(2, _dev_idx(t, dev)) * _dev_idx(a, dev)[None, None, :, None]
        for t, a in zip(xtaps, xcoef)
    )
    rows = [h.index_select(1, _dev_idx(t, dev)) for t in ytaps]
    betas = [_dev_idx(a, dev)[None, :, None, None] for a in ycoef]

    v_int = sum(r * b for r, b in zip(rows, betas))
    out_int = (v_int + (1 << (2 * _COEF_BITS - 1))) >> (2 * _COEF_BITS)

    inv = np.float32(1.0 / (_COEF_SCALE * _COEF_SCALE))  # 2^-22, exact
    t = rows[3].to(torch.float32) * (betas[3].to(torch.float32) * inv)
    for k in (2, 1, 0):
        t = rows[k].to(torch.float32) * (betas[k].to(torch.float32) * inv) + t
    out_float = torch.round(t).to(torch.int32)

    n = dst_w * C
    covered = (n // 8) * 8
    elem = torch.arange(n, device=dev, dtype=torch.int32).reshape(dst_w, C)
    out = torch.where(elem[None, None] < covered, out_float, out_int)
    return out.clamp(0, 255).to(torch.uint8)

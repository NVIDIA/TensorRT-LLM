# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canny edge detection (Canny, 1986) over a clip of uint8 frames.

Two elementwise kernels plus a hysteresis fixpoint loop:

1. ``_grad_kernel``: Sobel 3x3 with replicate borders per channel, then
   per-pixel channel selection by max ``|dx| + |dy|`` (strict ``>``, so the
   lowest channel index wins ties); writes int32 dx/dy/magnitude maps.
2. ``_nms_kernel``: direction binning in Q15 fixed point against
   ``tan(22.5deg)``, non-maximum suppression with per-direction tie-breaks,
   then the double threshold; writes a uint8 map (2 = strong, 1 = weak).
3. Hysteresis grows strong seeds through weak pixels, 8-connected. This stays
   a torch max-pool fixpoint loop: it is data-dependent global propagation,
   and it dominates the op's cost, so it is the next optimization target.

The gradient magnitude is L1 (no L2 gradient), and every stage is integer
arithmetic, so bit-exactness here has no floating-point caveats.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .resize import _check_frames

_BLOCK = 256
# tl.constexpr, not a bare int: a @triton.jit kernel may only read globals that
# are declared this way, and the NMS kernel uses this directly.
_TG22 = tl.constexpr(13573)  # round(tan(22.5deg) * 2**15)
# Output rows per NMS program. The launch grid divides by this, so the two
# must not drift apart -- a mismatch silently mis-strides the row band.
_NMS_ROWS = 4


@triton.jit
def _grad_kernel(src, bdx, bdy, mag, SC, H, W, C: tl.constexpr, BLOCK: tl.constexpr):
    # (x-block, y, t) grid: no runtime integer division (idiv dominated the SM
    # in the flat-index variant)
    y = tl.program_id(1)
    t = tl.program_id(2)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = x < W
    offs = (t * H + y) * W + x

    ym1 = tl.maximum(y - 1, 0)
    yp1 = tl.minimum(y + 1, H - 1)
    xm1 = tl.maximum(x - 1, 0)
    xp1 = tl.minimum(x + 1, W - 1)

    best = tl.zeros((BLOCK,), dtype=tl.int32) - 1  # any real mag (>=0) beats it
    dx = tl.zeros((BLOCK,), dtype=tl.int32)
    dy = tl.zeros((BLOCK,), dtype=tl.int32)
    # SC is dim 0's stride rather than a frame count, so a clip windowed along T
    # is read in place. c is a static_range constant, so c*SC folds to 0/SC/2*SC
    # -- the same address arithmetic the dense form compiled to.
    for c in tl.static_range(C):
        p = src + c * SC + t * H * W
        v00 = tl.load(p + ym1 * W + xm1, mask=m, other=0).to(tl.int32)
        v01 = tl.load(p + ym1 * W + x, mask=m, other=0).to(tl.int32)
        v02 = tl.load(p + ym1 * W + xp1, mask=m, other=0).to(tl.int32)
        v10 = tl.load(p + y * W + xm1, mask=m, other=0).to(tl.int32)
        v12 = tl.load(p + y * W + xp1, mask=m, other=0).to(tl.int32)
        v20 = tl.load(p + yp1 * W + xm1, mask=m, other=0).to(tl.int32)
        v21 = tl.load(p + yp1 * W + x, mask=m, other=0).to(tl.int32)
        v22 = tl.load(p + yp1 * W + xp1, mask=m, other=0).to(tl.int32)
        dx_c = (v02 + 2 * v12 + v22) - (v00 + 2 * v10 + v20)
        dy_c = (v20 + 2 * v21 + v22) - (v00 + 2 * v01 + v02)
        mag_c = tl.abs(dx_c) + tl.abs(dy_c)
        take = mag_c > best
        best = tl.where(take, mag_c, best)
        dx = tl.where(take, dx_c, dx)
        dy = tl.where(take, dy_c, dy)

    # int32 intermediates on purpose: int16 halves the DRAM traffic but Triton
    # unpacks every sub-word load through PRMT, which costs more ALU time than
    # the saved bytes buys (91 -> 114 us measured) -- nms is issue-bound, not
    # bandwidth-bound.
    tl.store(bdx + offs, dx, mask=m)
    tl.store(bdy + offs, dy, mask=m)
    tl.store(mag + offs, best, mask=m)


@triton.jit
def _nms_kernel(bdx, bdy, mag, out, lo, hi, H, W, R: tl.constexpr, BLOCK: tl.constexpr):
    # R output rows per program with register rotation of the 3-row magnitude
    # band: 3*(R+1) mag loads instead of 9*R -- nms is issue-bound (ALU pipe ==
    # SM throughput), so fewer load issues is the lever that matters.
    y0 = tl.program_id(1) * R
    t = tl.program_id(2)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = x < W
    ok_l = m & (x > 0)
    ok_r = m & (x < W - 1)
    base = (t * H + y0) * W + x

    # masked-out taps read 0, which is exactly the zero-padded border
    up_ok = y0 > 0
    pC = tl.load(mag + base - W, mask=m & up_ok, other=0)
    pL = tl.load(mag + base - W - 1, mask=ok_l & up_ok, other=0)
    pR = tl.load(mag + base - W + 1, mask=ok_r & up_ok, other=0)
    cC = tl.load(mag + base, mask=m, other=0)
    cL = tl.load(mag + base - 1, mask=ok_l, other=0)
    cR = tl.load(mag + base + 1, mask=ok_r, other=0)

    # software pipeline: row r's down-band and gradients load one iteration
    # ahead, so the current iteration's compare chain covers their latency
    # (without this, load -> compute -> load serializes: 61% long_scoreboard)
    dn0 = y0 < H - 1
    nC = tl.load(mag + base + W, mask=m & dn0, other=0)
    nL = tl.load(mag + base + W - 1, mask=ok_l & dn0, other=0)
    nR = tl.load(mag + base + W + 1, mask=ok_r & dn0, other=0)
    dx = tl.load(bdx + base, mask=m, other=0)
    dy = tl.load(bdy + base, mask=m, other=0)

    for r in tl.static_range(R):
        yr = y0 + r
        offs = base + r * W
        mr = m & (yr < H)
        if r < R - 1:
            dn_ok = yr + 1 < H - 1
            mr1 = m & (yr + 1 < H)
            fC = tl.load(mag + offs + 2 * W, mask=m & dn_ok, other=0)
            fL = tl.load(mag + offs + 2 * W - 1, mask=ok_l & dn_ok, other=0)
            fR = tl.load(mag + offs + 2 * W + 1, mask=ok_r & dn_ok, other=0)
            fdx = tl.load(bdx + offs + W, mask=mr1, other=0)
            fdy = tl.load(bdy + offs + W, mask=mr1, other=0)
        ax = tl.abs(dx)
        y15 = tl.abs(dy) << 15
        tg22 = ax * _TG22
        tg67 = tg22 + (ax << 16)
        horiz = y15 < tg22
        vert = (~horiz) & (y15 > tg67)
        diag = ~(horiz | vert)
        s_pos = (dx ^ dy) >= 0

        # select the direction's two neighbours, then compare once:
        #   horiz: cC > cL  && cC >= cR      vert: cC > pC && cC >= nC
        #   diag+: cC > pL  && cC > nR       diag-: cC > pR && cC > nL
        # (c == nb only survives for the non-diagonal >= comparisons)
        na = tl.where(horiz, cL, tl.where(vert, pC, tl.where(s_pos, pL, pR)))
        nb = tl.where(horiz, cR, tl.where(vert, nC, tl.where(s_pos, nR, nL)))
        keep = (cC > na) & ((cC > nb) | ((~diag) & (cC == nb)))
        res = tl.where(keep & (cC > hi), 2, tl.where(keep & (cC > lo), 1, 0))
        tl.store(out + offs, res.to(tl.uint8), mask=mr)

        if r < R - 1:
            pL, pC, pR = cL, cC, cR
            cL, cC, cR = nL, nC, nR
            nL, nC, nR = fL, fC, fR
            dx, dy = fdx, fdy


def _hysteresis(strong: torch.Tensor, weak: torch.Tensor) -> torch.Tensor:
    """Grow strong seeds through weak pixels to a fixpoint, 8-connected.

    Several growth steps per convergence check keep the host syncs down while
    staying exact, since the growth is monotone.
    """
    out = strong
    while True:
        prev = out
        for _ in range(4):
            out = torch.minimum(F.max_pool2d(out, 3, 1, 1), weak)
        if torch.equal(out, prev):
            break
    return out


def canny_edges(frames: torch.Tensor, low: int, high: int) -> torch.Tensor:
    """Canny over a clip: uint8 ``[C, T, H, W]`` CUDA -> uint8 ``[T, H, W]``.

    Dim 0 may be strided, so a caller windowing a longer clip along T can pass
    ``frames[:, start:stop]`` directly instead of materializing it.
    """
    _check_frames(frames, "canny_edges", layout="[C, T, H, W]", allow_outer_stride=True)
    C, T, H, W = frames.shape
    dev = frames.device

    bdx = torch.empty(T, H, W, dtype=torch.int32, device=dev)
    bdy = torch.empty_like(bdx)
    mag = torch.empty_like(bdx)
    _grad_kernel[(triton.cdiv(W, _BLOCK), H, T)](
        frames, bdx, bdy, mag, frames.stride(0), H, W, C=C, BLOCK=_BLOCK
    )

    cmap = torch.empty(T, H, W, dtype=torch.uint8, device=dev)
    _nms_kernel[(triton.cdiv(W, _BLOCK), triton.cdiv(H, 4), T)](
        bdx, bdy, mag, cmap, low, high, H, W, R=4, BLOCK=_BLOCK
    )

    out = _hysteresis(
        (cmap == 2).to(torch.float32)[:, None], (cmap >= 1).to(torch.float32)[:, None]
    )
    return (out[:, 0] > 0).to(torch.uint8) * 255

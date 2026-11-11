# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Limited-workspace 3D neighborhood attention (NATTEN ``na3d`` semantics) in pure torch.
Vendored from comfy-kitchen ``backends/eager/na.py`` (Apache-2.0) for DiffVAE hosts
without natten or Triton. Queries are tiled; tiles that share window geometry stack
into batched ``scaled_dot_product_attention`` calls with one additive mask per group.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional

# Element budget for one tile's [Nq, Nk] attention mask (bounds the mask
# allocation and, on CPU, the math-backend score materialization).
NA_SCORE_BUDGET = 2**25
# Element budget for the stacked K/V copies of one batched SDPA call on CUDA.
NA_KV_STACK_BUDGET = 2**28


def _window_bounds(length: int, kernel: int, causal: bool) -> tuple[list[int], list[int]]:
    """Per-index (start, end) of the attended window along one axis."""
    starts: list[int] = []
    ends: list[int] = []
    if causal:
        for i in range(length):
            starts.append(max(0, i - kernel + 1))
            ends.append(i + 1)
    else:
        kernel = min(kernel, length)
        lo = length - kernel
        half = kernel // 2
        for i in range(length):
            start = min(max(i - half, 0), lo)
            starts.append(start)
            ends.append(start + kernel)
    return starts, ends


def _pick_tiles(dims: tuple[int, int, int], kernels: list[int]) -> list[int]:
    """Per-axis query-tile lengths keeping one tile's [Nq, Nk] under budget."""
    tiles = list(dims)

    def cost(ts: list[int]) -> int:
        nq = math.prod(ts)
        nk = math.prod(min(d, t + k - 1) for t, k, d in zip(ts, kernels, dims, strict=True))
        return nq * nk

    while cost(tiles) > NA_SCORE_BUDGET and max(tiles) > 1:
        i = max(range(3), key=lambda a: tiles[a] / kernels[a])
        if tiles[i] <= 1:
            break
        tiles[i] = max(1, (tiles[i] + 1) // 2)
    return tiles


def _group_mask(
    rel_bounds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive ``[1, 1, Nq, Nk]`` mask for one tile-geometry group."""
    bools = []
    for starts, ends in rel_bounds:
        st = torch.tensor(starts, device=device)
        en = torch.tensor(ends, device=device)
        kj = torch.arange(int(en.max()), device=device)
        bools.append((kj[None, :] >= st[:, None]) & (kj[None, :] < en[:, None]))
    visible = (
        bools[0][:, None, None, :, None, None]
        & bools[1][None, :, None, None, :, None]
        & bools[2][None, None, :, None, None, :]
    )
    nq = visible.shape[0] * visible.shape[1] * visible.shape[2]
    nk = visible.shape[3] * visible.shape[4] * visible.shape[5]
    mask = torch.zeros((nq, nk), dtype=dtype, device=device)
    mask.masked_fill_(~visible.reshape(nq, nk), torch.finfo(dtype).min)
    return mask.reshape(1, 1, nq, nk)


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int] | tuple[int, ...],
    is_causal: list[bool] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """3D neighborhood attention over ``(B, T, H, W, NH, HD)`` tensors.
    ``scale`` defaults to ``head_dim**-0.5``. Pass ``scale=1.0`` when Q is already scaled.
    """
    batch, t, h, w, nh, hd = q.shape
    dims = (t, h, w)
    causal = [False, False, False] if is_causal is None else list(is_causal)
    kernels = [k_ if c else min(k_, d) for k_, c, d in zip(kernel_size, causal, dims, strict=True)]
    if scale is None:
        scale = hd**-0.5
    device = q.device
    if scale != 1.0:
        q = q * scale

    bounds = [_window_bounds(d, k_, c) for d, k_, c in zip(dims, kernels, causal, strict=True)]
    tile_t, tile_h, tile_w = _pick_tiles(dims, [min(k_, d) for k_, d in zip(kernels, dims, strict=True)])

    groups: dict[
        tuple[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[int, ...], tuple[int, ...]],
        ],
        list[tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]],
    ] = {}
    for t0 in range(0, t, tile_t):
        t1 = min(t0 + tile_t, t)
        rt0, rt1 = bounds[0][0][t0], bounds[0][1][t1 - 1]
        rel_t = (
            tuple(s - rt0 for s in bounds[0][0][t0:t1]),
            tuple(e - rt0 for e in bounds[0][1][t0:t1]),
        )
        for h0 in range(0, h, tile_h):
            h1 = min(h0 + tile_h, h)
            rh0, rh1 = bounds[1][0][h0], bounds[1][1][h1 - 1]
            rel_h = (
                tuple(s - rh0 for s in bounds[1][0][h0:h1]),
                tuple(e - rh0 for e in bounds[1][1][h0:h1]),
            )
            for w0 in range(0, w, tile_w):
                w1 = min(w0 + tile_w, w)
                rw0, rw1 = bounds[2][0][w0], bounds[2][1][w1 - 1]
                rel_w = (
                    tuple(s - rw0 for s in bounds[2][0][w0:w1]),
                    tuple(e - rw0 for e in bounds[2][1][w0:w1]),
                )
                groups.setdefault((rel_t, rel_h, rel_w), []).append(
                    (
                        (slice(t0, t1), slice(h0, h1), slice(w0, w1)),
                        (slice(rt0, rt1), slice(rh0, rh1), slice(rw0, rw1)),
                    )
                )

    out = torch.empty((batch, t, h, w, nh, hd), device=device, dtype=v.dtype)
    for rel, tiles in groups.items():
        mask = _group_mask(rel, q.dtype, device)
        nq, nk = mask.shape[2], mask.shape[3]
        g_max = max(1, NA_KV_STACK_BUDGET // max(1, batch * nh * nk * hd * 2)) if device.type == "cuda" else 1
        qs0, _ = tiles[0]
        tq = qs0[0].stop - qs0[0].start
        th = qs0[1].stop - qs0[1].start
        tw = qs0[2].stop - qs0[2].start
        for c0 in range(0, len(tiles), g_max):
            chunk = tiles[c0 : c0 + g_max]
            g = len(chunk)
            q_s = torch.stack([q[:, qs[0], qs[1], qs[2]] for qs, _ in chunk])
            k_s = torch.stack([k[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            v_s = torch.stack([v[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            q_s = q_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nq, hd)
            k_s = k_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            v_s = v_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            o = functional.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=mask, scale=1.0)
            o = o.view(g, batch, nh, tq, th, tw, hd).permute(0, 1, 3, 4, 5, 2, 6)
            for i, (qs, _) in enumerate(chunk):
                out[:, qs[0], qs[1], qs[2]] = o[i]

    return out

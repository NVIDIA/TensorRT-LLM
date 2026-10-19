# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-row hyper-connection site, second half: the grouped RMS, the SiLU
bottleneck and the injection weights from `qwen4_exp_hc_decode_first`'s
per-stream partials, then the up projection, the sigmoid gate on the normed
stream and the mean over streams -- one Triton GEMV launch.

Tier 5 (Triton written for this target). Each program owns `BLOCK_H` hidden
columns across all streams: it requests its whole up-projection tile before
any arithmetic, rebuilds `rstd` and `low` from the four partials (a few KB of
L2 per row, recomputed by every program), and keeps the `[T, C*H]` gate
logits in registers. Program 0 also writes the block-injection weights.
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Largest row count this entry is certified for.
MAX_ROWS = 16


@functools.lru_cache(maxsize=None)
def pdl_enabled(device_index: int) -> bool:
    """Programmatic dependent launch: sm_90+ unless TRTLLM_ENABLE_PDL=0."""
    props = torch.cuda.get_device_properties(device_index)
    return props.major >= 9 and os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


@triton.jit
def _hc_decode_mix_kernel(
    part_ptr,
    ss_ptr,
    up_ptr,
    x_ptr,
    norm_w_ptr,
    mixed_ptr,
    inj_out_ptr,
    stride_ps,
    stride_pm,
    stride_ss,
    stride_up_r,
    stride_up_n,
    stride_x,
    stride_mixed,
    eps,
    M: tl.constexpr,
    HIDDEN: tl.constexpr,
    HC: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    WITH_INJECT_OUT: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
):
    pid = tl.program_id(0)
    hh = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    sidx = tl.arange(0, HC)
    col = sidx[None, :] * HIDDEN + hh[:, None]
    hmask = (hh < HIDDEN)[:, None] & (sidx >= 0)[None, :]
    # The whole [BLOCK_H, HC, R] weight tile is requested before any arithmetic.
    ra = tl.arange(0, R1)
    w_a = tl.load(
        up_ptr + col[:, :, None] * stride_up_n + ra[None, None, :] * stride_up_r,
        mask=hmask[:, :, None],
        other=0.0,
    )
    if R2 > 0:
        rb = R1 + tl.arange(0, R2)
        w_b = tl.load(
            up_ptr + col[:, :, None] * stride_up_n + rb[None, None, :] * stride_up_r,
            mask=hmask[:, :, None],
            other=0.0,
        )
    gam = 1.0 + tl.load(norm_w_ptr + col, mask=hmask, other=0.0).to(tl.float32)
    if LAUNCH_WITH_PDL:
        if EARLY_TRIGGER:
            tl.extra.cuda.gdc_launch_dependents()
        # Weights above stream while the producer runs; its outputs are read after this.
        tl.extra.cuda.gdc_wait()
    for m in tl.static_range(M):
        ssm = tl.load(ss_ptr + sidx * stride_ss + m)
        pa = tl.load(part_ptr + sidx[:, None] * stride_ps + m * stride_pm + ra[None, :])
        if R2 > 0:
            pb = tl.load(part_ptr + sidx[:, None] * stride_ps + m * stride_pm + rb[None, :])
        x = tl.load(x_ptr + m * stride_x + col, mask=hmask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(ssm / HIDDEN + eps)
        za = tl.sum(pa * rstd[:, None], axis=0) * (1.0 / HC)
        acc = tl.sum(w_a.to(tl.float32) * (za * tl.sigmoid(za))[None, None, :], axis=2)
        if R2 > 0:
            zb = tl.sum(pb * rstd[:, None], axis=0) * (1.0 / HC)
            acc += tl.sum(w_b.to(tl.float32) * (zb * tl.sigmoid(zb))[None, None, :], axis=2)
        v = tl.sigmoid(acc) * (x * rstd[None, :] * gam)
        tl.store(
            mixed_ptr + m * stride_mixed + hh,
            (tl.sum(v, axis=1) * (1.0 / HC)).to(mixed_ptr.dtype.element_ty),
            mask=hh < HIDDEN,
        )
        if WITH_INJECT_OUT:
            ci = tl.arange(0, HC)
            pi = tl.load(
                part_ptr + sidx[:, None] * stride_ps + m * stride_pm + (R1 + R2) + ci[None, :],
                mask=(sidx >= 0)[:, None] & (pid == 0),
                other=0.0,
            )
            zi = tl.sum(pi * rstd[:, None], axis=0) * (1.0 / HC)
            tl.store(
                inj_out_ptr + m * HC + ci,
                (2.0 * tl.sigmoid(zi)).to(inj_out_ptr.dtype.element_ty),
                mask=(ci >= 0) & (pid == 0),
            )


#: Launch configuration: (BLOCK_H, num_warps). Swept on GB300 (see the note).
DEFAULT_CONFIG = (8, 8)
#: Let the kernel after this one launch as soon as every program has requested its
#: weights: a programmatic dependent (the decode projections) then streams its own
#: weight while this kernel finishes. Its `gdc_wait` still waits for this whole grid.
TRIGGER_DEPENDENTS = True


def _rank_split(rank: int) -> Tuple[int, int]:
    """`rank = R1 + R2` with `R1` the largest power of two not above it; `R2` must be
    a power of two or zero (320 = 256 + 64)."""
    r1 = 1 << (rank.bit_length() - 1)
    return r1, rank - r1


def launch(
    part,
    ss,
    up,
    hc_out,
    norm_weight,
    eps,
    hc_count,
    with_inject,
    config=DEFAULT_CONFIG,
    trigger=TRIGGER_DEPENDENTS,
):
    """The launch behind `Qwen4ExpHcDecodeMix.__call__`, tile config exposed.
    Returns `(mixed, inject_out)`."""
    rows, width = hc_out.shape
    hidden = width // hc_count
    rank = up.shape[0]
    r1, r2 = _rank_split(rank)
    block_h, warps = config
    dev, dt = hc_out.device, hc_out.dtype
    mixed = torch.empty((rows, hidden), dtype=dt, device=dev)
    inject_out = torch.empty((rows, hc_count), dtype=dt, device=dev) if with_inject else None
    pdl = pdl_enabled(dev.index if dev.index is not None else torch.cuda.current_device())
    if rows > 0:
        _hc_decode_mix_kernel[(triton.cdiv(hidden, block_h),)](
            part,
            ss,
            up,
            hc_out,
            norm_weight,
            mixed,
            inject_out if with_inject else mixed,
            part.stride(0),
            part.stride(1),
            ss.stride(0),
            up.stride(0),
            up.stride(1),
            hc_out.stride(0),
            mixed.stride(0),
            eps,
            M=rows,
            HIDDEN=hidden,
            HC=hc_count,
            R1=r1,
            R2=r2,
            BLOCK_H=block_h,
            WITH_INJECT_OUT=with_inject,
            LAUNCH_WITH_PDL=pdl,
            EARLY_TRIGGER=trigger,
            num_warps=warps,
            launch_pdl=pdl,
        )
    return mixed, inject_out


class Qwen4ExpHcDecodeMix(OpWrapper):
    """With `C = hc_count`, `H = width // C`, `R = up.shape[0]`, `part` / `ss` the
    per-stream partial logits `[C, T, >= R (+ C)]` and sums of squares `[C, T]`
    from `qwen4_exp_hc_decode_first`, `up` the `[R, width]` view of the
    `[width, R]` up-projection weight::

        rstd[t, s]       = 1 / sqrt(ss[s, t] / H + eps)
        logits[t, n]     = sum_s rstd[t, s] * part[s, t, n]
        low              = silu(logits[:, :R] / C)
        gate             = sigmoid(low @ up)                                  # [T, C*H]
        normed[t, s*H+i] = hc_out[t, s*H+i] * rstd[t, s] * (1 + norm_weight[s*H+i])
        mixed[t, i]      = mean over s of gate[t, s*H+i] * normed[t, s*H+i]
        inject_out       = 2 * sigmoid(logits[:, R:R+C] / C)                  # with_inject only

    fp32 inside, one rounding per output. Returns fresh `(mixed, inject_out)`
    in `hc_out`'s dtype; `inject_out` is None without `with_inject`.

    Fusion boundary: the grouped RMS, the bottleneck SiLU, the injection
    weights, the up projection, the sigmoid gate, the normed stream and the
    stream mean. The partials are the producer's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, with_inject (True), hc_count (4), hidden (2560), rank (320), dtype (bf16).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"checkpoint 4x2560 r320 bf16, inject, {why}", spec=dict(rows=t))
        for t, why in (
            (1, "decode 1 row (con 1)"),
            (2, "decode 2 rows"),
            (3, "decode 3 rows"),
            (8, "decode 8 rows"),
            (16, "decode 16 rows (MAX_ROWS)"),
        )
    ) + (
        Cell(why="stack-final mixer (no inject), 1 row", spec=dict(rows=1, with_inject=False)),
        Cell(why="stack-final mixer (no inject), 16 rows", spec=dict(rows=16, with_inject=False)),
        Cell(why="hc_out a row slice buf[::2], 4 rows", spec=dict(rows=4, layout="row_slice")),
        Cell(why="hc_count 2 at hidden 2560, 2 rows", spec=dict(rows=2, hc_count=2)),
        Cell(
            why="hidden 128 rank 32 (not the checkpoint), 5 rows",
            spec=dict(rows=5, hidden=128, rank=32),
        ),
        Cell(why="float16 operands, 4 rows", spec=dict(rows=4, dtype=torch.float16)),
        Cell(why="zero rows", spec=dict(rows=0)),
    )

    note = """
    Metadata consumed: none. Call sites: `DecodeTarget`'s trunk, right after
    each `qwen4_exp_hc_decode_first` (97 per forward, 1-row decode batches).

    Preconditions: `part` fp32 `[C, T, >= R (+ C)]` and `ss` fp32 `[C, T]`,
    both with unit last stride (other strides free); `up` `[R, width]` (any
    strides; the `.t()` view of the stored weight reads rows of `R` contiguous
    elements); `R` a power of two or a sum of two (320 = 256 + 64); `hc_out`
    `[T, width]` unit column stride; `norm_weight` contiguous `[width]`. Row
    count is a compile-time constant, at most MAX_ROWS certified.

    Numerics: fp32 throughout (the composition it replaces rounded `low`, the
    gate logits and `normed` to bf16), one rounding per output. Every program
    recomputes `rstd` and `low` from the partials in the same order, so all
    programs agree bitwise. Graph-capturable. Launched with programmatic
    dependent launch on sm_90+ (`TRTLLM_ENABLE_PDL=0` turns it off): the
    weight tile streams before `gdc_wait`, the producer's outputs after it.

    Grid: cdiv(H, 8) programs of 8 warps, each one [8, C, R] weight load
    (20 KB). Measured on GB300 (graph replay, cold weights): ~4 us at 1 row;
    each further row costs ~3 us (the row loop serializes its L2 round trips).
    """

    def __call__(
        self,
        part: torch.Tensor,
        ss: torch.Tensor,
        up: torch.Tensor,
        hc_out: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        hc_count: int,
        with_inject: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return launch(part, ss, up, hc_out, norm_weight, eps, hc_count, with_inject)

    def reference(
        self,
        part: torch.Tensor,
        ss: torch.Tensor,
        up: torch.Tensor,
        hc_out: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        hc_count: int,
        with_inject: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """float64, one rounding per output."""
        dtype = hc_out.dtype
        rank = up.shape[0]
        hidden = hc_out.shape[1] // hc_count
        rstd = torch.rsqrt(ss.double().t() / hidden + eps)
        logits = torch.einsum("ts,stn->tn", rstd, part.double())
        low = F.silu(logits[:, :rank] / hc_count)
        gate = torch.sigmoid(low @ up.double())
        x = hc_out.double().unflatten(-1, (hc_count, hidden)) * rstd.unsqueeze(-1)
        normed = x.flatten(-2) * (1.0 + norm_weight.double())
        mixed = (gate * normed).unflatten(-1, (hc_count, hidden)).mean(dim=-2).to(dtype)
        inject_out = None
        if with_inject:
            inject_out = (2.0 * torch.sigmoid(logits[:, rank : rank + hc_count] / hc_count)).to(
                dtype
            )
        return mixed, inject_out

    def is_valid(
        self,
        part: torch.Tensor,
        ss: torch.Tensor,
        up: torch.Tensor,
        hc_out: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        hc_count: int,
        with_inject: bool,
    ) -> None:
        rows, width = hc_out.shape
        rank = up.shape[0]
        assert hc_out.stride(1) == 1, "hc_out unit column stride"
        assert part.dtype == torch.float32 and part.stride(2) == 1, "part fp32, unit last stride"
        assert part.shape[:2] == (hc_count, rows), "part [C, T, N]"
        assert part.shape[2] >= rank + (hc_count if with_inject else 0), "part too narrow"
        assert ss.dtype == torch.float32 and ss.shape == (hc_count, rows), "ss fp32 [C, T]"
        assert rows <= 1 or ss.stride(1) == 1, "ss unit stride along rows"
        assert up.shape == (rank, width), "up [R, width]"
        assert norm_weight.shape == (width,) and norm_weight.is_contiguous()
        r2 = _rank_split(rank)[1]
        assert r2 & (r2 - 1) == 0, "rank must be a power of two or a sum of two"


qwen4_exp_hc_decode_mix = Qwen4ExpHcDecodeMix()

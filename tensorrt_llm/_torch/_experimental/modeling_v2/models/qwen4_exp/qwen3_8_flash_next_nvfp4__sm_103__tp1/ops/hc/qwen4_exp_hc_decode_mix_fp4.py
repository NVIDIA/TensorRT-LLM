# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-row hyper-connection site at an MoE site, second half:
`qwen4_exp_hc_decode_mix` (grouped RMS, SiLU bottleneck, injection weights, up
projection, sigmoid gate, stream mean) plus the NVFP4 quantization of the mean
for the routed experts -- one Triton GEMV launch.

Tier 5 (Triton written for this target). Same structure as
`qwen4_exp_hc_decode_mix`, except that each program owns a whole number of
16-column NVFP4 blocks so it can quantize its own columns: the scale block is
the only cross-column dependency the quantization adds.

The quantization helpers here (`nvfp4_quantize_blocks` in Triton,
`nvfp4_reference` in torch) reproduce `torch.ops.trtllm.fp4_quantize` (NVFP4,
linear scales) step for step, including its two `rcp.approx.ftz.f32`
reciprocals, so the packed data and the scales are bitwise that op's output on
the rounded mix; `qwen4_exp_hc_gate_mean_fp4` imports them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .qwen4_exp_hc_decode_mix import _rank_split, pdl_enabled

#: Largest row count this entry is certified for.
MAX_ROWS = 16

#: NVFP4 block: one e4m3 scale per this many contiguous elements.
SF_VEC = 16
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)
#: Same window `catalog/quantization/fp4_quantize` states for the same two
#: approximate reciprocals.
TIE_WINDOW = 2.0**-20


@triton.jit
def rcp_approx_ftz(x):
    """`rcp.approx.ftz.f32`, the reciprocal the trtllm quantizer uses."""
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def nvfp4_quantize_blocks(v, gs):
    """NVFP4 of `v` `[NB, 8, 2]` fp32 (values already rounded to the activation
    dtype; element `16*b + 2*p + j` of the row chunk at `[b, p, j]`) with the
    global scale `gs` `[NB]`. Returns the packed bytes `[NB, 8]` (int32, element
    `2k` in the low nibble) and the e4m3 scale bytes `[NB]` (uint8)."""
    amax = tl.max(tl.max(tl.abs(v), axis=2), axis=1)
    sfv = gs * (amax * rcp_approx_ftz(tl.zeros_like(amax) + 6.0))
    sf8 = sfv.to(tl.float8e4nv)
    osc = rcp_approx_ftz(sf8.to(tl.float32) * rcp_approx_ftz(gs))
    osc = tl.where(amax != 0.0, osc, 0.0)
    s = v * osc[:, None, None]
    a = tl.abs(s)
    # Round to nearest, ties to the even e2m1 code; saturate at 6.
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    code = code | tl.where(s.to(tl.int32, bitcast=True) < 0, 8, 0)
    shift = 4 * tl.arange(0, 2)
    packed = tl.sum(code << shift[None, None, :], axis=2)
    return packed, sf8.to(tl.uint8, bitcast=True)


def nvfp4_reference(
    x: torch.Tensor, global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Native-torch NVFP4 of `x` `[T, K]` with linear scales: per 16-block,
    `sf = e4m3(gs * amax / 6)` and the data `x * gs / sf` rounded to e2m1
    (nearest, ties to even). Returns `(packed [T, K/2] uint8, sf [T*K/16] uint8)`."""
    packed, sf, _ = _nvfp4_torch(x, global_scale)
    return packed, sf


def nvfp4_near_tie(x: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """`[T, K]` bool: elements whose pre-rounding value sits within `TIE_WINDOW`
    of an e2m1 midpoint, where approximate reciprocals may pick either side."""
    scaled = _nvfp4_torch(x, global_scale)[2]
    mid = torch.tensor(_E2M1_MIDPOINTS, device=x.device, dtype=torch.float32)
    a = scaled.abs().unsqueeze(-1)
    return ((a - mid).abs() <= TIE_WINDOW * a).any(-1)


def _nvfp4_torch(x, global_scale):
    gs = global_scale.float().reshape(())
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // SF_VEC, SF_VEC)
    vecmax = blocks.abs().amax(-1)
    sf = (gs * (vecmax / _E2M1_MAX)).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
    out_scale = torch.where(vecmax != 0, gs / sf.float(), torch.zeros_like(vecmax))
    scaled = (blocks * out_scale.unsqueeze(-1)).reshape(rows, k)
    mid = torch.tensor(_E2M1_MIDPOINTS, device=x.device, dtype=torch.float32)
    tie_up = torch.tensor(_TIE_UP, device=x.device, dtype=torch.bool)
    a = scaled.abs().unsqueeze(-1)
    code = ((a > mid).sum(-1) + ((a == mid) & tie_up).any(-1)).to(torch.uint8)
    negative = (scaled < 0) | ((scaled == 0) & torch.signbit(scaled))
    code = code | (negative.to(torch.uint8) << 3)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, sf.view(torch.uint8).reshape(-1), scaled


@triton.jit
def _hc_decode_mix_fp4_kernel(
    part_ptr,
    ss_ptr,
    up_ptr,
    x_ptr,
    norm_w_ptr,
    gs_ptr,
    mixed_ptr,
    inj_out_ptr,
    q_ptr,
    sf_ptr,
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
    NBH: tl.constexpr = BLOCK_H // 16
    pid = tl.program_id(0)
    hh = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    sidx = tl.arange(0, HC)
    col = sidx[None, :] * HIDDEN + hh[:, None]
    hmask = (hh < HIDDEN)[:, None] & (sidx >= 0)[None, :]
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
    b = pid * NBH + tl.arange(0, NBH)
    p = tl.arange(0, 8)
    bmask = b < HIDDEN // 16
    gs = tl.load(gs_ptr + tl.zeros([NBH], dtype=tl.int32))
    if LAUNCH_WITH_PDL:
        if EARLY_TRIGGER:
            tl.extra.cuda.gdc_launch_dependents()
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
        mixed = (tl.sum(v, axis=1) * (1.0 / HC)).to(mixed_ptr.dtype.element_ty)
        tl.store(mixed_ptr + m * stride_mixed + hh, mixed, mask=hh < HIDDEN)
        packed, sf = nvfp4_quantize_blocks(tl.reshape(mixed.to(tl.float32), [NBH, 8, 2]), gs)
        q_off = m * (HIDDEN // 2) + b[:, None] * 8 + p[None, :]
        tl.store(q_ptr + q_off, packed.to(tl.uint8), mask=bmask[:, None])
        tl.store(sf_ptr + m * (HIDDEN // 16) + b, sf, mask=bmask)
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


#: Launch configuration: (BLOCK_H, num_warps). BLOCK_H a multiple of 16. Swept
#: on GB300 at 1/2/4 rows (graph, cold weights): (16, 4) fastest at every row
#: count; (16, 2), (16, 8) and (32, 4) 0.3..5 us slower.
DEFAULT_CONFIG = (16, 4)
#: Let the kernel after this one launch as soon as every program has requested its
#: weights: a programmatic dependent (the decode projections) then streams its own
#: weight while this kernel finishes. Its `gdc_wait` still waits for this whole grid.
TRIGGER_DEPENDENTS = True


def launch(
    part,
    ss,
    up,
    hc_out,
    norm_weight,
    eps,
    hc_count,
    with_inject,
    global_scale,
    config=DEFAULT_CONFIG,
    trigger=TRIGGER_DEPENDENTS,
):
    """The launch behind `Qwen4ExpHcDecodeMixFp4.__call__`, tile config exposed.
    Returns `(mixed, inject_out, xq, xsf)`."""
    rows, width = hc_out.shape
    hidden = width // hc_count
    rank = up.shape[0]
    r1, r2 = _rank_split(rank)
    block_h, warps = config
    dev, dt = hc_out.device, hc_out.dtype
    mixed = torch.empty((rows, hidden), dtype=dt, device=dev)
    inject_out = torch.empty((rows, hc_count), dtype=dt, device=dev) if with_inject else None
    xq = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=dev)
    xsf = torch.empty((rows * (hidden // SF_VEC),), dtype=torch.uint8, device=dev)
    pdl = pdl_enabled(dev.index if dev.index is not None else torch.cuda.current_device())
    if rows > 0:
        _hc_decode_mix_fp4_kernel[(triton.cdiv(hidden, block_h),)](
            part,
            ss,
            up,
            hc_out,
            norm_weight,
            global_scale,
            mixed,
            inject_out if with_inject else mixed,
            xq,
            xsf,
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
    return mixed, inject_out, xq, xsf


class Qwen4ExpHcDecodeMixFp4(OpWrapper):
    """`qwen4_exp_hc_decode_mix` plus the NVFP4 of its `mixed`. With `C =
    hc_count`, `H = width // C`, `R = up.shape[0]`, `part` / `ss` the per-stream
    partial logits `[C, T, >= R (+ C)]` and sums of squares `[C, T]` from
    `qwen4_exp_hc_decode_first`, `up` the `[R, width]` view of the up weight::

        rstd[t, s]       = 1 / sqrt(ss[s, t] / H + eps)
        logits[t, n]     = sum_s rstd[t, s] * part[s, t, n]
        low              = silu(logits[:, :R] / C)
        gate             = sigmoid(low @ up)                                  # [T, C*H]
        normed[t, s*H+i] = hc_out[t, s*H+i] * rstd[t, s] * (1 + norm_weight[s*H+i])
        mixed[t, i]      = mean over s of gate[t, s*H+i] * normed[t, s*H+i]
        inject_out       = 2 * sigmoid(logits[:, R:R+C] / C)                  # with_inject only
        xq, xsf          = NVFP4(mixed)   # e2m1 pairs, e4m3 scale per 16, linear layout

    fp32 inside, one rounding per output; the quantization reads the rounded
    `mixed`. Returns fresh `(mixed, inject_out, xq [T, H/2] uint8, xsf [T*H/16]
    uint8)`; `inject_out` is None without `with_inject`. `global_scale` is the
    1-element fp32 NVFP4 global scale.

    Fusion boundary: everything `qwen4_exp_hc_decode_mix` fuses, plus the NVFP4
    quantization of `mixed`. The partials are the producer's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, with_inject (True), hc_count (4), hidden (2560), rank (320), dtype (bf16).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"checkpoint 4x2560 r320 bf16, inject, {why}", spec=dict(rows=t))
        for t, why in (
            (1, "decode 1 row (con 1)"),
            (2, "decode 2 rows"),
            (3, "decode 3 rows"),
            (16, "decode 16 rows (MAX_ROWS)"),
        )
    ) + (
        Cell(why="hc_out a row slice buf[::2], 4 rows", spec=dict(rows=4, layout="row_slice")),
        Cell(why="hc_count 2 at hidden 2560, 2 rows", spec=dict(rows=2, hc_count=2)),
        Cell(
            why="hidden 144 rank 32 (9 scale blocks), 5 rows",
            spec=dict(rows=5, hidden=144, rank=32),
        ),
        Cell(why="no inject, 1 row", spec=dict(rows=1, with_inject=False)),
        Cell(why="float16 operands, 4 rows", spec=dict(rows=4, dtype=torch.float16)),
        Cell(why="zero rows", spec=dict(rows=0)),
    )

    note = """
    Metadata consumed: none. Call sites: `DecodeTarget`'s trunk, the MoE-site
    mixer of every layer (48 per forward) at batches of at most
    `_HC_DECODE_GEMV_MAX_ROWS` rows, right after `qwen4_exp_hc_decode_first`;
    `xq`/`xsf` feed `fp4_block_scale_moe_renorm_routed` (linear scales).

    Preconditions: those of `qwen4_exp_hc_decode_mix`, plus `H` a multiple of
    16 and `global_scale` one fp32 element on the device. Row count is a
    compile-time constant, at most MAX_ROWS certified.

    Numerics: `mixed` and `inject_out` follow `qwen4_exp_hc_decode_mix`'s
    formula (fp32, one rounding); `inject_out` is bitwise its result, `mixed`
    can differ in a few elements by one rounding because the 16-column program
    tile reduces the up projection in another order. The NVFP4 is
    the trtllm recipe (two `rcp.approx.ftz.f32`, `cvt.rn.satfinite` e4m3
    scale, e2m1 round-to-nearest-even saturating at 6) on the rounded `mixed`;
    the GPU test checks it bitwise against `torch.ops.trtllm.fp4_quantize`.
    Graph-capturable; programmatic dependent launch as `qwen4_exp_hc_decode_mix`.

    Grid: cdiv(H, BLOCK_H) programs (BLOCK_H a multiple of 16; 16 -> 160
    programs of 4 warps, each one [16, C, R] weight load, 40 KB). Measured on
    GB300 (graph replay, cold weights, 1 row): 4.6 us, against 6.0 us for
    `qwen4_exp_hc_decode_mix` + `fp4_quantize` and 3.9 us for
    `qwen4_exp_hc_decode_mix` alone (its 8-column tile has twice the programs).
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
        global_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        return launch(part, ss, up, hc_out, norm_weight, eps, hc_count, with_inject, global_scale)

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
        global_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """float64 with one rounding per output, then the NVFP4 of the rounded
        `mixed` in native torch."""
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
        return (mixed, inject_out) + nvfp4_reference(mixed, global_scale)

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
        global_scale: torch.Tensor,
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
        assert (width // hc_count) % SF_VEC == 0, "hidden must be whole 16-blocks"
        # The kernel loads element 0 only; a per-row scale would be applied as row 0's.
        assert global_scale.numel() == 1 and global_scale.dtype == torch.float32


qwen4_exp_hc_decode_mix_fp4 = Qwen4ExpHcDecodeMixFp4()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-row W4A16 projection: `y[M, N] = x[M, K] @ dequant(W)^T` with NVFP4 weights (e2m1
codes, one e4m3 scale per 16 consecutive k of a row, one fp32 global scale) and bf16
activations, M = 1 .. 16.

CuTe DSL (inline PTX), one PDL launch, CUDA-graph capturable; compiled once per (N, K, plan)
and row octet (1..8 or 9..16 rows): the row count is a launch argument.

Math: swap-AB warp MMA (`mma.sync.m16n8k16`, f16 in, fp32 accumulate). A is 16 weight rows,
each e2m1 pair widened by `cvt.rn.f16x2.e2m1x2` and multiplied by its block scale in f16
(`cvt.rn.f16x2.e4m3x2`, `mul.rn.f16x2`): an e2m1 value times an e4m3 scale has at most five
significant bits and lies in [2^-10, 2688], so the dequantized weight is exact. B is up to 8
activation rows per n8 tile, rounded once from bf16 to f16 (exact for magnitudes in
[2^-14, 65504]; smaller ones lose low bits, larger ones saturate). The MMA sums in fp32 over
all of K; the sum times the global scale is rounded once to bf16.

Weight layout (`pack_weight`): 1152-byte units of 16 rows x 128 k -- 1024 bytes of codes then
128 bytes of scales -- stored k-block-major `[K/128][ceil(N/16)][1152]`. K is permuted
identically for both operands inside every 128-block: lane quad t of k16 step s holds
k = 32 t + 4 s + 0..3, so a thread's activations of a k-block are 32 consecutive values and
its four k of a step share one 16-element scale block (2 t for steps 0..3, 2 t + 1 for 4..7).
Codes: `[half h (2)][lane (32)][4 words]`, word i of half h the step 4 h + i, its bytes
`(row g, k0 | k1 << 4), (row g + 8, same k), (row g, k2 | k3 << 4), (row g + 8, same k)` --
one byte per A register. Scales: `[lane (32)][4 bytes]`: `(row g, block 2t), (row g + 8, 2t),
(row g, 2t + 1), (row g + 8, 2t + 1)`.

Work split (`Plan`): persistent, one or two resident CTAs per multiprocessor, each a contiguous
range of 16-row tiles taken `rt` tiles (a chunk) at a time over all of K. A producer warp streams the units
with `cp.async.bulk` into a ring of shared-memory slots (one k-block of the chunk's tiles a
slot) completed on mbarriers, continuously across chunks and before the PDL wait (the weight
is static). After `griddepcontrol.wait` the consumer warps convert the activations once to
f16 into shared memory in MMA-fragment order; consumer warp w then takes tiles w, w + warps,
... of every slot, its accumulators in registers for the chunk, and stores its tiles through a
per-warp staging buffer.

Split entries (`nvfp4_split_gemv*`, `SplitNvfp4Weight`): the same NVFP4 weight packed in
`fp8_gemv`'s fragment order (`pack_split`) and streamed by that kernel, which splits K across
warps and carries the GDN conv epilogues; for projections with few 16-row tiles a
multiprocessor (the GDN in_proj), where one warp per tile over all of K is latency-bound.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from .fp8_gemv import (
    TILE_MAJOR_MAX_TILES,
    _asm,
    _bar_sync,
    _bf16x2_to_f16x2,
    _bulk_g2s,
    _bulk_prefetch_l2,
    _dq_step,
    _env_pdl,
    _ir,
    _ldg4p,
    _lds4,
    _mbar_arrive,
    _mbar_expect_tx,
    _mbar_init,
    _mbar_init_fence,
    _mbar_wait,
    _mma_f16,
    _scales4,
    _sms,
    fp8_gemv,
    fp8_gemv_conv1d,
    fp8_gemv_conv1d_verify,
    fp8_gemv_conv1d_verify_op,
    verify_reference,
)
from .fp8_gemv import _compiled as _split_compiled
from .fp8_gemv import default_plan as _split_plan

TILE_N = 16
BLOCK_K = 128
SF_VEC = 16  # k per e4m3 block scale
DATA = TILE_N * BLOCK_K // 2  # bytes of e2m1 codes of one (16-row, 128-k) unit
SCALES = TILE_N * BLOCK_K // SF_VEC  # bytes of its e4m3 block scales
UNIT = DATA + SCALES
E2M1_MAX = 6.0
E4M3_MAX = 448.0
MAX_ROWS = 16
#: The e2m1 magnitudes, by the three low bits of a code (bit 3 is the sign).
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
#: Rounding thresholds between consecutive e2m1 magnitudes (a tie rounds toward zero).
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
#: Shared memory one CTA may take (the SM100 opt-in maximum less a margin) and a
#: multiprocessor holds (less a margin per resident CTA).
SMEM_MAX = 225 << 10
SMEM_SM = 228 << 10
#: Activation loads a thread keeps in flight in the prologue (16 bytes each).
_X_BATCH = 10


# ------------------------------------------------------------------------------------------
# inline PTX helpers
# ------------------------------------------------------------------------------------------


@dsl_user_op
def _lds32(saddr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(saddr, cutlass.Int32, loc, ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _sts4(saddr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip)]
        + [_ir(v, cutlass.Int32, loc, ip) for v in (v0, v1, v2, v3)],
        "{ st.shared.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,r,r,r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _stg4(gaddr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip)]
        + [_ir(v, cutlass.Int32, loc, ip) for v in (v0, v1, v2, v3)],
        "{ st.global.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,l,r,r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _lds_u16(saddr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(saddr, cutlass.Int32, loc, ip)],
            "ld.shared.u16 $0, [$1];",
            "=r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _stg_b16(gaddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(v, cutlass.Int32, loc, ip)],
        "{ .reg .b16 h; cvt.u16.u32 h, $2; st.global.b16 [$1], h; mov.u32 $0, 0; }",
        "=r,l,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _sts_bf16_scaled(saddr, v, s, *, loc=None, ip=None):
    """bf16(v * s) to shared memory."""
    _asm(
        T.i32(),
        [
            _ir(saddr, cutlass.Int32, loc, ip),
            _ir(v, cutlass.Float32, loc, ip),
            _ir(s, cutlass.Float32, loc, ip),
        ],
        "{ .reg .f32 f; .reg .b16 h; mul.rn.f32 f, $2, $3; cvt.rn.bf16.f32 h, f; "
        "st.shared.b16 [$1], h; mov.u32 $0, 0; }",
        "=r,r,f,f",
        loc,
        ip,
        side=True,
    )


# ------------------------------------------------------------------------------------------
# kernel
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """Launch shape: `grid` persistent CTAs (all resident: one or two per multiprocessor), each
    taking its tiles `rt` at a time; `warps` consumer warps; `slots` ring slots (one k-block of
    a chunk each); with `l2_ahead` > 0, issuing stage s also prefetches stage
    s + slots + l2_ahead - 1 into L2."""

    grid: int
    rt: int
    warps: int
    slots: int
    l2_ahead: int = 0


class _Nvfp4Gemv:
    def __init__(self, *, n, k, nt8, plan: Plan, pdl):
        assert k % BLOCK_K == 0 and nt8 in (1, 2)
        self.n, self.k, self.nt8, self.pdl = n, k, nt8, pdl
        self.n_tiles = -(-n // TILE_N)
        self.nkb = k // BLOCK_K
        self.grid = plan.grid
        self.rt, self.cw, self.slots, self.l2_ahead = plan.rt, plan.warps, plan.slots, plan.l2_ahead
        self.rtw = -(-self.rt // self.cw)
        self.threads = (self.cw + 1) * 32  # + the weight producer warp
        self.np = nt8 * 8
        self.slot_bytes = -(-(self.rt * UNIT) // 128) * 128
        self.o_ring = -(-(self.nkb * nt8 * 4 * 512) // 128) * 128  # activation fragments first
        self.o_stg = self.o_ring + self.slots * self.slot_bytes
        self.o_bar = self.o_stg + self.cw * self.np * TILE_N * 2
        self.smem = self.o_bar + 2 * self.slots * 8 + 128
        assert self.smem <= SMEM_MAX, (self.smem, plan)

    @cute.jit
    def __call__(
        self,
        out_t: cute.Tensor,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        x_stride: cutlass.Int32,
        m: cutlass.Int32,
        gscale: cutlass.Float32,
        stream,
    ):
        self.kernel(
            out_t.iterator.toint(), x_ptr.toint(), w_ptr.toint(), x_stride, m, gscale
        ).launch(
            grid=[self.grid, 1, 1],
            block=[self.threads, 1, 1],
            smem=self.smem,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.jit
    def _compute(self, acc, xf, slot, wi, ntc, lane):
        """This warp's tiles of the k-block in ring slot `slot` times the activation fragments
        `xf` ([nt8][16] f16x2), accumulated into `acc`."""
        cw: cutlass.Constexpr = self.cw
        nt8: cutlass.Constexpr = self.nt8
        for i in cutlass.range_constexpr(self.rtw):
            tl = wi + i * cw
            if tl < ntc:
                ua = slot + tl * UNIT
                sc = _scales4(_lds32(ua + DATA + lane * 4))
                wv = [_lds4(ua + h * 512 + lane * 16) for h in range(2)]
                for s in cutlass.range_constexpr(8):
                    h = s // 4
                    a = _dq_step(wv[h][s % 4], sc[2 * h], sc[2 * h + 1])
                    for j in cutlass.range_constexpr(nt8):
                        r = _mma_f16(
                            a,
                            [xf[j][2 * s], xf[j][2 * s + 1]],
                            [acc[i, j, 0], acc[i, j, 1], acc[i, j, 2], acc[i, j, 3]],
                        )
                        for e in cutlass.range_constexpr(4):
                            acc[i, j, e] = r[e]

    @cute.kernel
    def kernel(
        self,
        out_base: cutlass.Int64,
        x_base: cutlass.Int64,
        w_base: cutlass.Int64,
        x_stride: cutlass.Int32,
        m: cutlass.Int32,
        gscale: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        g = lane // 4
        t = lane % 4
        n_tiles: cutlass.Constexpr = self.n_tiles
        nkb: cutlass.Constexpr = self.nkb
        rt: cutlass.Constexpr = self.rt
        cw: cutlass.Constexpr = self.cw
        slots: cutlass.Constexpr = self.slots
        slot_bytes: cutlass.Constexpr = self.slot_bytes
        nt8: cutlass.Constexpr = self.nt8
        np_: cutlass.Constexpr = self.np
        rtw: cutlass.Constexpr = self.rtw
        n_out: cutlass.Constexpr = self.n
        l2a: cutlass.Constexpr = self.l2_ahead
        grid: cutlass.Constexpr = self.grid

        smem = cutlass.utils.SmemAllocator()
        base_t = smem.allocate_tensor(
            cutlass.Int8, cute.make_layout((self.smem - 128,), stride=(1,)), byte_alignment=128
        )
        sbase = cutlass.Int32(base_t.iterator.toint())
        xs = sbase  # activation fragments [kb][j][c][lane][16 B], f16
        ring = sbase + self.o_ring
        bars = sbase + self.o_bar  # full[slots], empty[slots]

        tile0 = (bidx * n_tiles) // grid
        ntot = ((bidx + 1) * n_tiles) // grid - tile0
        nch = (ntot + rt - 1) // rt

        if warp == cw:
            if lane == 0:
                for sl in cutlass.range_constexpr(slots):
                    _mbar_init(bars + 8 * sl, 1)
                    _mbar_init(bars + 8 * (slots + sl), cw)
                _mbar_init_fence()
        cute.arch.sync_threads()
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if warp == cw:
            # ---- weight producer (static data: no PDL wait); stage = (chunk, k-block) ----
            if lane == 0:
                for st in cutlass.range(nch * nkb):
                    sl = st % slots
                    if st >= slots:
                        _mbar_wait(bars + 8 * (slots + sl), ((st // slots) - 1) & 1)
                    c = st // nkb
                    kb = st - c * nkb
                    tc0 = tile0 + c * rt
                    ntc = cutlass.min(rt, ntot - c * rt)
                    if const_expr(l2a > 0):
                        pst = st + slots + l2a - 1
                        if pst < nch * nkb:
                            pc = pst // nkb
                            _bulk_prefetch_l2(
                                w_base
                                + (cutlass.Int64(pst - pc * nkb) * n_tiles + tile0 + pc * rt)
                                * UNIT,
                                cutlass.min(rt, ntot - pc * rt) * UNIT,
                            )
                    _mbar_expect_tx(bars + 8 * sl, ntc * UNIT)
                    _bulk_g2s(
                        ring + sl * slot_bytes,
                        w_base + (cutlass.Int64(kb) * n_tiles + tc0) * UNIT,
                        ntc * UNIT,
                        bars + 8 * sl,
                    )
        else:
            # ---- consumers ----
            if const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            # activations -> f16 fragments: item q = ((kb nt8 + j) 4 + c) 32 + lane holds row
            # 8 j + g, k = 128 kb + 32 t + 8 c .. + 7 (rows past m zero)
            n_items: cutlass.Constexpr = nkb * nt8 * 4 * 32
            n_it: cutlass.Constexpr = -(-n_items // (cw * 32))
            # loads issued a batch at a time ahead of their conversions (each load asm is
            # ordered, so a load placed after a conversion would wait out its predecessor)
            for b0 in cutlass.range_constexpr(0, n_it, _X_BATCH):
                vals = []
                for it in cutlass.range_constexpr(b0, min(b0 + _X_BATCH, n_it)):
                    q = tidx + it * (cw * 32)
                    ql = q % 32
                    row = 8 * ((q // 128) % nt8) + ql // 4
                    kx = (q // (128 * nt8)) * BLOCK_K + 32 * (ql % 4) + 8 * ((q // 32) % 4)
                    vals.append(
                        _ldg4p(
                            x_base
                            + cutlass.Int64(row) * cutlass.Int64(x_stride) * 2
                            + cutlass.Int64(kx * 2),
                            cutlass.Int32((row < m) & (q < n_items)),
                        )
                    )
                for it in cutlass.range_constexpr(b0, min(b0 + _X_BATCH, n_it)):
                    q = tidx + it * (cw * 32)
                    if q < n_items:
                        v = vals[it - b0]
                        _sts4(
                            xs + q * 16,
                            _bf16x2_to_f16x2(v[0]),
                            _bf16x2_to_f16x2(v[1]),
                            _bf16x2_to_f16x2(v[2]),
                            _bf16x2_to_f16x2(v[3]),
                        )
            _bar_sync(1, cw * 32)

            stg = sbase + self.o_stg + warp * (np_ * TILE_N * 2)  # [np][16] bf16 per warp
            acc = cute.make_rmem_tensor(
                cute.make_layout((rtw, nt8, 4), stride=(nt8 * 4, 4, 1)), cutlass.Float32
            )
            for c in cutlass.range(nch):
                ntc = cutlass.min(rt, ntot - c * rt)
                acc.fill(0.0)
                for kb in cutlass.range_constexpr(nkb):
                    st = c * nkb + kb
                    sl = st % slots
                    xf = [
                        [
                            v
                            for cc in range(4)
                            for v in _lds4(xs + (((kb * nt8 + j) * 4 + cc) * 32 + lane) * 16)
                        ]
                        for j in range(nt8)
                    ]
                    _mbar_wait(bars + 8 * sl, (st // slots) & 1)
                    self._compute(acc, xf, ring + sl * slot_bytes, warp, ntc, lane)
                    cute.arch.sync_warp()
                    if lane == 0:
                        _mbar_arrive(bars + 8 * (slots + sl))
                # ---- store: through the warp's staging buffer, a row's 16 outputs per 2 lanes ----
                for i in cutlass.range_constexpr(rtw):
                    tl = warp + i * cw
                    if tl < ntc:
                        for j in cutlass.range_constexpr(nt8):
                            n0 = 8 * j + 2 * t
                            p = stg + (n0 * TILE_N + g) * 2
                            _sts_bf16_scaled(p, acc[i, j, 0], gscale)
                            _sts_bf16_scaled(p + TILE_N * 2, acc[i, j, 1], gscale)
                            _sts_bf16_scaled(p + 8 * 2, acc[i, j, 2], gscale)
                            _sts_bf16_scaled(p + TILE_N * 2 + 8 * 2, acc[i, j, 3], gscale)
                        cute.arch.sync_warp()
                        orow0 = (tile0 + c * rt + tl) * TILE_N
                        for rr in cutlass.range_constexpr(-(-(np_ * 2) // 32)):
                            li = lane + rr * 32
                            en = li // 2
                            hh = li % 2
                            if (li < np_ * 2) & (en < m):
                                if const_expr(n_out % 8 == 0):
                                    if orow0 + 8 * hh < n_out:
                                        v = _lds4(stg + (en * TILE_N + 8 * hh) * 2)
                                        _stg4(
                                            out_base
                                            + (cutlass.Int64(en) * n_out + orow0 + 8 * hh) * 2,
                                            v[0],
                                            v[1],
                                            v[2],
                                            v[3],
                                        )
                                else:
                                    for e in cutlass.range_constexpr(8):
                                        if orow0 + 8 * hh + e < n_out:
                                            _stg_b16(
                                                out_base
                                                + (cutlass.Int64(en) * n_out + orow0 + 8 * hh + e)
                                                * 2,
                                                _lds_u16(stg + (en * TILE_N + 8 * hh + e) * 2),
                                            )
                        cute.arch.sync_warp()


# ------------------------------------------------------------------------------------------
# host side
# ------------------------------------------------------------------------------------------


@dataclass
class PackedNvfp4Weight:
    """`data`: uint8 `[K/128, ceil(N/16), 1152]` (see `pack_weight`); `global_scale`: the fp32
    dequantization multiplier (`w = e2m1 code x e4m3 block scale x global_scale`); logical
    shape `[n, k]`."""

    data: torch.Tensor
    global_scale: float
    n: int
    k: int


def _global_scale(w: torch.Tensor) -> float:
    """amax(|w|) / (448 x 6): the largest block's scale lands on e4m3's largest value."""
    return max(w.abs().amax().float().item(), 1e-30) / (E4M3_MAX * E2M1_MAX)


def _quantize_rows(w: torch.Tensor, gscale: float):
    """(codes uint8 `[n, K]`, e4m3 block scales `[n, K/16]`) of `w` under `gscale`."""
    n, k = w.shape
    blk = w.float().reshape(n, k // SF_VEC, SF_VEC)
    sf = (blk.abs().amax(2) / (E2M1_MAX * gscale)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    step = sf.float()[..., None] * gscale
    v = torch.where(step > 0, blk / step.clamp_min(1e-38), 0.0)
    bounds = torch.tensor(_E2M1_BOUNDS, device=w.device)
    idx = torch.bucketize(v.abs(), bounds).to(torch.uint8)
    codes = idx | (((v < 0) & (idx > 0)).to(torch.uint8) << 3)
    return codes.reshape(n, k), sf


def quantize_nvfp4(w: torch.Tensor, rows_per_chunk: int = 16384):
    """bf16/fp32 `[N, K]` -> (codes uint8 `[N, K]` -- bit 3 the sign, bits 0..2 the index into
    `E2M1_VALUES` --, e4m3 block scales `[N, K/16]`, global scale): global = amax(|w|) /
    (448 x 6), a block's scale = e4m3(amax(block) / 6 / global), a code the nearest e2m1 value
    of w / (scale x global) (ties toward zero), saturated at 6."""
    n, k = w.shape
    assert k % SF_VEC == 0
    gscale = _global_scale(w)
    codes = torch.empty(n, k, dtype=torch.uint8, device=w.device)
    scales = torch.empty(n, k // SF_VEC, dtype=torch.float8_e4m3fn, device=w.device)
    for lo in range(0, n, rows_per_chunk):
        hi = min(lo + rows_per_chunk, n)
        codes[lo:hi], scales[lo:hi] = _quantize_rows(w[lo:hi], gscale)
    return codes, scales, gscale


def dequantize_nvfp4(codes: torch.Tensor, scales: torch.Tensor, gscale: float) -> torch.Tensor:
    """fp32 `[N, K]` = e2m1(code) x its block scale x the global scale."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    val = lut[(codes & 7).long()] * (1.0 - 2.0 * (codes >> 3).float())
    return val * scales.float().repeat_interleave(SF_VEC, 1) * gscale


def _pack_into(out: torch.Tensor, tile0: int, codes: torch.Tensor, scales: torch.Tensor) -> None:
    """Rows `16 tile0 ..` (codes `[n, K]`, scales `[n, K/16]`) into the packed buffer `out`
    `[K/128, tiles, 1152]`; rows past n up to the tile boundary are zero."""
    n, k = codes.shape
    nkb = k // BLOCK_K
    tiles = -(-n // TILE_N)
    c = torch.zeros(tiles * TILE_N, k, dtype=torch.uint8, device=codes.device)
    c[:n] = codes
    # (tile, h', g, kb, t, hh, s4, e2, e1): row 8 h' + g, k = 128 kb + 32 t + 16 hh + 4 s4 + 2 e2 + e1
    c = c.reshape(tiles, 2, 8, nkb, 4, 2, 4, 2, 2)
    byte = c[..., 0] | (c[..., 1] << 4)
    # unit byte order [hh][g][t][s4][e2][h']
    out[:, tile0 : tile0 + tiles, :DATA] = byte.permute(3, 0, 5, 2, 4, 6, 7, 1).reshape(
        nkb, tiles, DATA
    )
    s = torch.zeros(tiles * TILE_N, k // SF_VEC, dtype=torch.uint8, device=codes.device)
    s[:n] = scales.view(torch.uint8)
    # (tile, h', g, kb, t, hb): block 8 kb + 2 t + hb; unit order [g][t][hb][h']
    out[:, tile0 : tile0 + tiles, DATA:] = (
        s.reshape(tiles, 2, 8, nkb, 4, 2).permute(3, 0, 2, 4, 5, 1).reshape(nkb, tiles, SCALES)
    )


def pack_weight(codes: torch.Tensor, scales: torch.Tensor, gscale: float) -> PackedNvfp4Weight:
    """(codes `[N, K]`, e4m3 scales `[N, K/16]`, global scale) -> the kernel's layout (module
    docstring)."""
    n, k = codes.shape
    assert k % BLOCK_K == 0 and scales.shape == (n, k // SF_VEC)
    out = torch.empty(k // BLOCK_K, -(-n // TILE_N), UNIT, dtype=torch.uint8, device=codes.device)
    _pack_into(out, 0, codes, scales)
    return PackedNvfp4Weight(out, float(gscale), n, k)


def pack_from_bf16(w: torch.Tensor, rows_per_chunk: int = 16384) -> PackedNvfp4Weight:
    """Converter from a bf16 `[N, K]` weight, quantized and packed in row chunks (the transient
    memory stays a few hundred MB for the 248320 x 2560 vocabulary)."""
    n, k = w.shape
    assert k % BLOCK_K == 0 and rows_per_chunk % TILE_N == 0
    gscale = _global_scale(w)
    out = torch.empty(k // BLOCK_K, -(-n // TILE_N), UNIT, dtype=torch.uint8, device=w.device)
    for lo in range(0, n, rows_per_chunk):
        hi = min(lo + rows_per_chunk, n)
        _pack_into(out, lo // TILE_N, *_quantize_rows(w[lo:hi], gscale))
    return PackedNvfp4Weight(out, float(gscale), n, k)


@dataclass
class SplitNvfp4Weight:
    """NVFP4 units in `fp8_gemv`'s fragment order, which that kernel (k split across warps, the
    GDN conv epilogues) streams in place of FP8 ones: `data` uint8 `[K/128, ceil(N/16), 1152]`
    (k-block-major) or `[ceil(N/16), K/128, 1152]` (`tile_major`), see `pack_split`; `scale` the
    fp32 global scale as a one-element device tensor (the kernel reads it), also
    `global_scale`; logical shape `[n, k]`."""

    data: torch.Tensor
    scale: torch.Tensor
    global_scale: float
    n: int
    k: int
    tile_major: bool = False
    w4: bool = True


def pack_split(
    codes: torch.Tensor, scales: torch.Tensor, gscale: float, tile_major: bool | None = None
) -> SplitNvfp4Weight:
    """(codes `[N, K]`, e4m3 scales `[N, K/16]`, global scale) -> `fp8_gemv`'s NVFP4 units.

    A unit (k-block kb, 16-row tile T) is 1152 bytes. Codes, 1024 bytes `[h (2)][lane (32)][4
    words]`, lane = 4 g + t: word 2 cc + st of half h is k16 step st of chunk c = 2 h + cc,
    covering k = 128 kb + 32 c + 8 t + 4 st + 0..3 (`fp8_gemv`'s k order), its bytes `(row g,
    k0 | k1 << 4), (row g + 8, same k), (row g, k2 | k3 << 4), (row g + 8, same k)`. Scales,
    128 bytes `[g (8)][t // 2 (2)][c (4)][row g, row g + 8]`: the block 2 c + t // 2 of the
    k-block. Rows past N are zero. Default `tile_major` as `fp8_gemv.pack_weight`'s."""
    n, k = codes.shape
    assert k % BLOCK_K == 0 and scales.shape == (n, k // SF_VEC)
    n_tiles = -(-n // TILE_N)
    nkb = k // BLOCK_K
    if tile_major is None:
        tile_major = n_tiles <= TILE_MAJOR_MAX_TILES
    c = torch.zeros(n_tiles * TILE_N, k, dtype=torch.uint8, device=codes.device)
    c[:n] = codes
    # (tile, rh, g, kb, h, cc, t, st, e2, e1): row 8 rh + g, k = 128 kb + 32 (2 h + cc) + 8 t
    # + 4 st + 2 e2 + e1
    c = c.reshape(n_tiles, 2, 8, nkb, 2, 2, 4, 2, 2, 2)
    byte = c[..., 0] | (c[..., 1] << 4)
    # unit byte order [h][g][t][cc][st][e2][rh]
    perm = (4, 2, 6, 5, 7, 8, 1)
    body = (0, 3) if tile_major else (3, 0)
    data = torch.empty(*(byte.shape[i] for i in body), UNIT, dtype=torch.uint8, device=codes.device)
    data[..., :DATA] = byte.permute(*body, *perm).reshape(*data.shape[:2], DATA)
    s = torch.zeros(n_tiles * TILE_N, k // SF_VEC, dtype=torch.uint8, device=codes.device)
    s[:n] = scales.view(torch.uint8)
    # (tile, rh, g, kb, c, th): block 8 kb + 2 c + th; unit order [g][th][c][rh]
    s = s.reshape(n_tiles, 2, 8, nkb, 4, 2)
    data[..., DATA:] = s.permute(*body, 2, 5, 4, 1).reshape(*data.shape[:2], SCALES)
    scale = torch.tensor([gscale], dtype=torch.float32, device=codes.device)
    return SplitNvfp4Weight(data, scale, float(gscale), n, k, tile_major)


def pack_split_from_bf16(w: torch.Tensor, rows_per_chunk: int = 16384) -> SplitNvfp4Weight:
    """Converter from a bf16 `[N, K]` weight (`quantize_nvfp4`, then `pack_split`)."""
    return pack_split(*quantize_nvfp4(w, rows_per_chunk))


#: Measured plans (GB300, 152 SMs, CUDA graph of back-to-back calls over weights rotated past
#: L2), keyed (N, K, row octets): `ctas` resident CTAs per multiprocessor (two small ones let
#: a launch's ramp overlap its neighbour's tail), tiles a chunk, consumer warps, ring slots,
#: L2 look-ahead. The 16-row variant's activations (80 KB in shared memory) leave room for
#: one CTA.
_TABLE: dict[tuple[int, int, int], dict] = {
    (248320, 2560, 1): dict(ctas=2, rt=8, warps=8, slots=5, l2_ahead=1),
    (248320, 2560, 2): dict(ctas=1, rt=32, warps=16, slots=3, l2_ahead=1),
}
_DEFAULT = {
    1: dict(ctas=2, rt=8, warps=8, slots=6, l2_ahead=1),
    2: dict(ctas=1, rt=16, warps=16, slots=6, l2_ahead=1),
}


def default_plan(n: int, k: int, nt8: int, sms: int) -> Plan:
    """The plan `nvfp4_gemv` launches with: the measured table entry for (N, K, nt8), else the
    octet's default, its ring trimmed to the shared memory its CTAs per multiprocessor share."""
    n_tiles = -(-n // TILE_N)
    f = dict(_TABLE.get((n, k, nt8), _DEFAULT[nt8]))
    ctas = f.pop("ctas")
    f["rt"] = min(f["rt"], n_tiles)
    slot = -(-(f["rt"] * UNIT) // 128) * 128
    fixed = -(-((k // BLOCK_K) * nt8 * 2048) // 128) * 128 + f["warps"] * nt8 * 256 + 128
    budget = min(SMEM_MAX, SMEM_SM // ctas - (1 << 10)) - fixed - 2 * 8 * f["slots"]
    f["slots"] = max(2, min(f["slots"], budget // slot))
    return Plan(grid=min(ctas * sms, n_tiles), **f)


@functools.lru_cache(maxsize=None)
def _compiled(n, k, nt8, plan: Plan, pdl: bool):
    kern = _Nvfp4Gemv(n=n, k=k, nt8=nt8, plan=plan, pdl=pdl)
    buf = torch.empty(16, dtype=torch.int32, device="cuda")
    out = torch.empty(nt8 * 8, n, dtype=torch.bfloat16, device="cuda")
    return cute.compile(
        kern,
        from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        make_ptr(cutlass.BFloat16, buf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, buf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        cutlass.Int32(k),
        cutlass.Int32(1),
        cutlass.Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def _octet(m: int) -> int:
    return 1 if m <= 8 else 2


def precompile(w: PackedNvfp4Weight, pdl: bool = True) -> None:
    """Compile both row octets of `w`'s shape now: a first call mid-serving would otherwise
    stall on the compile."""
    for nt8 in (1, 2):
        _compiled(w.n, w.k, nt8, default_plan(w.n, w.k, nt8, _sms()), bool(pdl and _env_pdl()))


def nvfp4_gemv(
    x: torch.Tensor, w: PackedNvfp4Weight, pdl: bool = True, plan: Plan | None = None
) -> torch.Tensor:
    """`x[M, K]` bf16 (row stride a multiple of 8, 16-byte aligned base) @ dequant(w)^T ->
    fresh `[M, N]` bf16. M = 1 .. MAX_ROWS."""
    m, k = x.shape
    assert k == w.k and x.dtype == torch.bfloat16 and x.stride(1) == 1
    assert 1 <= m <= MAX_ROWS
    assert x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    out = torch.empty((m, w.n), dtype=torch.bfloat16, device=x.device)
    nt8 = _octet(m)
    if plan is None:
        plan = default_plan(w.n, k, nt8, _sms())
    fn = _compiled(w.n, k, nt8, plan, bool(pdl and _env_pdl()))
    fn(out, x.data_ptr(), w.data.data_ptr(), x.stride(0), m, w.global_scale)
    return out


class Nvfp4Gemv(OpWrapper):
    """`nvfp4_gemv(x, w)`: `y[M, N] = x[M, K] @ dequant(w)^T` in bf16 for 1..MAX_ROWS rows, the
    weight a `PackedNvfp4Weight` (e2m1 codes, one e4m3 scale per 16 k of a row, an fp32 global
    scale; `pack_from_bf16` / `pack_weight`). The dequantized weight is exact in f16; the
    activation is rounded once to f16 on chip (exact for |x| in [2^-14, 65504]); fp32
    accumulation over all of K, the global scale applied in fp32, one bf16 rounding.

    Fusion boundary: the projection."""

    ARCHS = frozenset({Arch.SM_103})

    # spec keys: n, k, m (rows). Every cell: x a column slice of a wider bf16 row (row stride
    # past K); the weight `quantize_nvfp4` of a random bf16 matrix with an outlier column and
    # a block of near-zero values.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP draft head, one request's draft step (1 row)", spec=dict(n=248320, k=2560, m=1)
        ),
        Cell(why="draft head, 3 rows: a partial row octet", spec=dict(n=248320, k=2560, m=3)),
        Cell(why="draft head, 8 rows: a full row octet", spec=dict(n=248320, k=2560, m=8)),
        Cell(
            why="draft head, 9 rows: the two-octet variant with one row in the second",
            spec=dict(n=248320, k=2560, m=9),
        ),
        Cell(why="draft head, 16 rows: the largest dispatch", spec=dict(n=248320, k=2560, m=16)),
        Cell(
            why="N not a multiple of 8 (scalar stores, a partial last tile), 3 k-blocks, "
            "fewer tiles than multiprocessors",
            spec=dict(n=1001, k=384, m=5),
        ),
    )

    note = """
    Metadata consumed: none. Call site: the MTP draft head's vocabulary projection at up to
    `_FP8_GEMV_HEAD_ROWS` rows under `TRTLLM_MODELING_V2_MTP_HEAD_NVFP4=1` (the verify head
    is untouched). Preconditions: x bf16 `[M, K]`, unit column stride, row stride a multiple
    of 8 elements and a 16-byte aligned base; K a multiple of 128. One PDL launch (the weight
    streamed before the wait); CUDA-graph capturable; compiled per (N, K, plan) for rows
    1..8 and 9..16 (`precompile` builds both), the row count a launch argument.
    """

    def __call__(self, x: torch.Tensor, w: PackedNvfp4Weight) -> torch.Tensor:
        return nvfp4_gemv(x, w)

    def reference(
        self, x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, global_scale: float
    ) -> torch.Tensor:
        """fp32 `[M, N]` over the dequantized weight (a true fp32 matmul); takes the unpacked
        NVFP4 weight (`quantize_nvfp4`'s outputs) in place of the packed `w`."""
        with true_fp32_matmul():
            return x.float() @ dequantize_nvfp4(codes, scales, global_scale).t()

    def is_valid(self, x: torch.Tensor, w: PackedNvfp4Weight, **kwargs) -> None:
        assert x.float().abs().amax().item() <= 65504.0, (
            "an activation beyond f16's range saturates at 65504 in the kernel"
        )


def precompile_split(
    w: SplitNvfp4Weight,
    max_rows: int,
    conv_dim: int,
    pad_slot_id: int,
    verify_p: tuple[int, ...] = (),
    pdl: bool = True,
) -> None:
    """Compile now every `fp8_gemv` variant over `w`'s shape a decode step can reach: the
    projection and the single-token conv entry at 1..max_rows rows, and the verify entry at
    each p of `verify_p` for every whole number of requests within max_rows. The kernel is
    compiled per row count; a first call mid-serving would stall on the compile."""
    pdl = bool(pdl and _env_pdl())
    sms = _sms()
    for m in range(1, max_rows + 1):
        plan = _split_plan(w.n, w.k, m, sms, True)
        _split_compiled(w.n, w.k, m, plan, pdl, w.tile_major, w4=True)
        _split_compiled(w.n, w.k, m, plan, pdl, w.tile_major, conv_dim, pad_slot_id, w4=True)
        for p in verify_p:
            if m % p == 0 and m >= 2:
                _split_compiled(
                    w.n, w.k, m, plan, pdl, w.tile_major, conv_dim, pad_slot_id, p, w4=True
                )


def _split_cell(why: str, n: int, k: int, m: int) -> Cell:
    return Cell(why=why, spec=dict(n=n, k=k, m=m))


class Nvfp4SplitGemv(OpWrapper):
    """`nvfp4_split_gemv(x, w)` (`fp8_gemv` over a `SplitNvfp4Weight`): `y[M, N] = x[M, K] @
    dequant(w)^T` in bf16 for 1..32 rows, the weight NVFP4 (e2m1 codes, one e4m3 scale per 16 k
    of a row, an fp32 global scale; `pack_split_from_bf16` / `pack_split`). The dequantized
    weight is exact in f16; the activation is rounded once to f16 on chip (exact for |x| in
    [2^-14, 65504]); fp32 accumulation, the global scale applied in fp32, one bf16 rounding.

    Fusion boundary: the projection; `Nvfp4SplitGemvConv1d` / `Nvfp4SplitGemvConv1dVerify`
    also fuse the GDN conv step."""

    ARCHS = frozenset({Arch.SM_103})

    # spec keys: n, k, m (rows). Every cell: x a column slice of a wider bf16 row; the weight
    # `quantize_nvfp4` of a random bf16 matrix with an outlier column and near-zero blocks.
    CELLS: tuple[Cell, ...] = (
        _split_cell("GDN in_proj, one decode row", 16480, 2560, 1),
        _split_cell("GDN in_proj, 3 rows", 16480, 2560, 3),
        _split_cell("GDN in_proj, 9 rows (two n8 tiles)", 16480, 2560, 9),
        _split_cell("GDN in_proj, 16 rows", 16480, 2560, 16),
        _split_cell("small N, tile-major units, 3 k-blocks", 1001, 384, 5),
    )

    note = """
    Metadata consumed: none. Call site: the decode target's GDN in_proj at up to
    `_INPROJ_NVFP4_MAX_ROWS` rows of an eager step under `TRTLLM_MODELING_V2_GDN_INPROJ_NVFP4=1`.
    Preconditions: x bf16 `[M, K]`, unit column stride, row stride a multiple of 8 elements and
    a 16-byte aligned base; K a multiple of 128. One PDL launch (the weight streamed before the
    wait); CUDA-graph capturable; compiled per (N, K, M, plan) (`precompile_split`).
    """

    def __call__(self, x: torch.Tensor, w: SplitNvfp4Weight) -> torch.Tensor:
        return fp8_gemv(x, w)

    def reference(
        self, x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, global_scale: float
    ) -> torch.Tensor:
        """fp32 `[M, N]` over the dequantized weight (a true fp32 matmul); takes the unpacked
        NVFP4 weight (`quantize_nvfp4`'s outputs) in place of the packed `w`."""
        with true_fp32_matmul():
            return x.float() @ dequantize_nvfp4(codes, scales, global_scale).t()

    def is_valid(self, x: torch.Tensor, w: SplitNvfp4Weight, **kwargs) -> None:
        assert x.float().abs().amax().item() <= 65504.0, (
            "an activation beyond f16's range saturates at 65504 in the kernel"
        )


class Nvfp4SplitGemvConv1d(OpWrapper):
    """`nvfp4_split_gemv_conv1d` (`fp8_gemv_conv1d` over a `SplitNvfp4Weight`): the GDN input
    projection of decode rows over the NVFP4 copy, then for the conv columns of a live row the
    causal conv step against its paged state (SiLU; the state advanced in place). Returns the
    `[M, N]` projection with the conv columns replaced by the conv output."""

    ARCHS = frozenset({Arch.SM_103})

    # spec keys: slots (one per row; -1 the pad sentinel). Every cell: the GDN in_proj (16480 x
    # 2560, conv_dim 10240), a strided 64-slot pool, width 4, SiLU.
    CELLS: tuple[Cell, ...] = (
        Cell(why="STP decode: one row, live slot", spec=dict(slots=(37,))),
        Cell(why="two rows, one padding slot", spec=dict(slots=(17, -1))),
        Cell(why="four rows", spec=dict(slots=(0, 5, 63, 9))),
        Cell(why="sixteen rows (two n8 tiles)", spec=dict(slots=tuple(range(2, 50, 3)))),
    )

    note = """
    Call site: the decode target's GDN in_proj + conv step of a captured single-token step at
    up to `_INPROJ_NVFP4_MAX_ROWS` rows under `TRTLLM_MODELING_V2_GDN_INPROJ_NVFP4=1`.
    `conv_state` `[slots, conv_dim, 3]` bf16 (any strides), `conv_weight` `[conv_dim, 4]` bf16
    contiguous, int32 slots; a `pad_slot_id` row gets the plain projection. Slots, taps and the
    window are read before the PDL wait (nothing in a captured step writes this layer's pool
    ahead of it).
    """

    def __call__(self, x, w, conv_state, conv_weight, conv_state_indices, conv_dim, pad_slot_id):
        return fp8_gemv_conv1d(
            x, w, conv_state, conv_weight, conv_state_indices, conv_dim, pad_slot_id
        )

    def reference(self, *args, **kwargs):
        raise NotImplementedError("the test builds the fp32 reference from the dequantized weight")

    def is_valid(self, x, w, **kwargs) -> None:
        assert x.float().abs().amax().item() <= 65504.0, (
            "an activation beyond f16's range saturates at 65504 in the kernel"
        )


class Nvfp4SplitGemvConv1dVerify(OpWrapper):
    """`nvfp4_split_gemv_conv1d_verify` (`fp8_gemv_conv1d_verify` over a `SplitNvfp4Weight`):
    the GDN input projection of a speculative verify step over the NVFP4 copy with
    `causal_conv1d_triton_update`'s multi-token conv step as its epilogue. Returns `(conv [M,
    conv_dim], projection columns conv_dim.. [M, N - conv_dim])`.

    Fusion boundary: projection, conv, SiLU, snapshots and the pool advance."""

    ARCHS = frozenset({Arch.SM_103})

    # spec keys as `Fp8GemvConv1dVerify`'s.
    CELLS: tuple[Cell, ...] = fp8_gemv_conv1d_verify_op.CELLS

    note = """
    Call site: the decode target's GDN in_proj of an MTP verify step in a captured graph, 2..16
    rows, under `TRTLLM_MODELING_V2_GDN_INPROJ_NVFP4=1`; otherwise as `Fp8GemvConv1dVerify`
    (bitwise the projection + `causal_conv1d_triton_update` over the same NVFP4 projection).
    """

    def __call__(
        self,
        x,
        w,
        conv_state,
        conv_weight,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        conv_dim,
        rows_per_seq,
        pad_slot_id,
    ):
        return fp8_gemv_conv1d_verify(
            x,
            w,
            conv_state,
            conv_weight,
            conv_state_indices,
            intermediate_conv_window,
            intermediate_state_indices,
            conv_dim,
            rows_per_seq,
            pad_slot_id,
        )

    def reference(
        self,
        x,
        codes,
        scales,
        global_scale,
        conv_state,
        conv_weight,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        conv_dim,
        rows_per_seq,
        pad_slot_id,
    ):
        """As `Fp8GemvConv1dVerify.reference`, over the dequantized NVFP4 weight."""
        return verify_reference(
            x,
            dequantize_nvfp4(codes, scales, global_scale),
            conv_state,
            conv_weight,
            conv_state_indices,
            intermediate_conv_window,
            intermediate_state_indices,
            conv_dim,
            rows_per_seq,
            pad_slot_id,
        )

    def is_valid(
        self,
        x,
        w,
        conv_state,
        conv_weight,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        conv_dim,
        rows_per_seq,
        pad_slot_id,
        **kwargs,
    ) -> None:
        assert x.float().abs().amax().item() <= 65504.0, (
            "an activation beyond f16's range saturates at 65504 in the kernel"
        )
        fp8_gemv_conv1d_verify_op.is_valid(
            x,
            w,
            conv_state,
            conv_weight,
            conv_state_indices,
            intermediate_conv_window,
            intermediate_state_indices,
            conv_dim,
            rows_per_seq,
            pad_slot_id,
        )


nvfp4_gemv_op = Nvfp4Gemv()
nvfp4_split_gemv_op = Nvfp4SplitGemv()
nvfp4_split_gemv_conv1d_op = Nvfp4SplitGemvConv1d()
nvfp4_split_gemv_conv1d_verify_op = Nvfp4SplitGemvConv1dVerify()

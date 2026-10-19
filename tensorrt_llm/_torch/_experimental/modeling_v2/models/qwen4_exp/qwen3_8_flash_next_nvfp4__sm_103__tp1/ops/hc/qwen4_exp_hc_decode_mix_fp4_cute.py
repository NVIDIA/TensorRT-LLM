# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-row hyper-connection site at an MoE site, second half, for one-row batches:
the grouped RMS, the SiLU bottleneck and the injection weights from the first half's
partials, the up projection, the sigmoid gate, the stream mean and the NVFP4 of the
mean for the routed experts -- one PDL launch.

Tier 5 (CuTe DSL written for this target). `qwen4_exp_hc_decode_mix_fp4` computes the
same for up to 16 rows in Triton, whose programs hold a 64-row x 320 tile of the up
weight in 128 threads. Here a CTA
owns one 16-column NVFP4 block of the hidden width across the 4 streams (64 rows of
the up weight), and 8 threads share a row, 40 weights each; 160 CTAs of 512 threads,
all resident at once.

Order of work inside a CTA:

1. its 64 weight rows and its norm-weight elements into registers (static), before
   `griddepcontrol.wait`;
2. after the wait: the four sums of squares (every thread), one partial column per
   thread for the first 324 threads -> the shared `low` vector and, in CTA 0, the
   injection weights; each thread's residual element;
3. each row's dot with `low` over its 8 threads, the sigmoid gate on the normed
   element, the mean over streams (16 threads), the store; with NVFP4, warp 0 also
   quantizes the block as `torch.ops.trtllm.fp4_quantize` does (linear scales,
   `rcp.approx.ftz` reciprocals).
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .qwen4_exp_hc_decode_first_cute import pdl_enabled
from .qwen4_exp_hc_decode_mix import qwen4_exp_hc_decode_mix as _triton_mix
from .qwen4_exp_hc_decode_mix_fp4 import qwen4_exp_hc_decode_mix_fp4 as _triton_mix_fp4

MAX_ROWS = 1
VEC = 8
#: Hidden columns per CTA: one NVFP4 scale block.
BLOCK_H = 16
#: Threads sharing one up-weight row.
GROUP = 8
_COMPILE_OPTIONS = "--ptxas-options -maxrregcount=64"


class _MixKernel:
    def __init__(self, *, hidden, streams, rank, inject_out, fp4, pdl):
        assert rank % (GROUP * VEC) == 0 and hidden % BLOCK_H == 0
        self.hidden, self.streams, self.rank = hidden, streams, rank
        self.inject_out, self.fp4, self.pdl = inject_out, fp4, pdl
        self.k_tiles = rank // (GROUP * VEC)
        self.threads = streams * BLOCK_H * GROUP
        assert self.threads >= rank + streams and self.threads % 32 == 0

    @cute.jit
    def __call__(
        self,
        mPart,
        mSs,
        mUp,
        mHc,
        mNorm,
        mGs,
        mMixed,
        mInj,
        mXq,
        mXsf,
        eps: cutlass.Float32,
        stream: cuda_drv.CUstream,
    ):
        load_static = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.BFloat16,
            num_bits_per_copy=16 * VEC,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(
            mPart, mSs, mUp, mHc, mNorm, mGs, mMixed, mInj, mXq, mXsf, eps, load_static
        ).launch(
            grid=[self.hidden // BLOCK_H, 1, 1],
            block=[self.threads, 1, 1],
            smem=((self.rank // VEC) * (VEC + 1) + self.streams * BLOCK_H) * 4 + 32,
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mPart,
        mSs,
        mUp,
        mHc,
        mNorm,
        mGs,
        mMixed,
        mInj,
        mXq,
        mXsf,
        eps: cutlass.Float32,
        load_static: cute.CopyAtom,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        hidden: cutlass.Constexpr = self.hidden
        streams: cutlass.Constexpr = self.streams
        rank: cutlass.Constexpr = self.rank
        k_tiles: cutlass.Constexpr = self.k_tiles
        i0 = bidx * BLOCK_H
        row = tidx // GROUP  # (stream, column) row of this CTA: stream-major
        q = tidx % GROUP
        s = row // BLOCK_H
        i = row % BLOCK_H
        j = s * hidden + i0 + i  # the up-weight row, a column of the residual width

        # 1. Static operands, before the wait.
        u_vec = cute.logical_divide(mUp, (None, VEC))
        u_reg = cute.make_rmem_tensor(
            cute.make_layout((k_tiles, VEC), stride=(VEC, 1)), cutlass.BFloat16
        )
        for kt in cutlass.range_constexpr(k_tiles):
            cute.copy(load_static, u_vec[j, (None, kt * GROUP + q)], u_reg[kt, None])
        gam = 1.0 + mNorm[0, j].to(cutlass.Float32)
        gs = cutlass.Float32(0.0)
        if const_expr(self.fp4):
            gs = mGs[0]
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # 2. Grouped RMS, logits, low (shared), injection weights. Every producer value this
        # thread needs is requested before the first is used (a partial column clamped into
        # range for the threads past it), so the CTA pays one L2 round trip.
        cols: cutlass.Constexpr = rank + (streams if self.inject_out else 0)  # partial columns read
        pcol = cutlass.min(tidx, cols - 1)
        pv = [mPart[t, pcol] for t in range(streams)]
        ssv = [mSs[t] for t in range(streams)]
        x = mHc[0, j].to(cutlass.Float32)
        rstd = cute.make_rmem_tensor(cute.make_layout((streams,), stride=(1,)), cutlass.Float32)
        for t in cutlass.range_constexpr(streams):
            rstd[t] = cute.math.rsqrt(ssv[t] / hidden + eps)
        smem = cutlass.utils.SmemAllocator()
        # `low` in rows of VEC with one pad word, so the 8 threads of a row (stride VEC + 1
        # words) read 8 different banks.
        low = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((rank // VEC, VEC), stride=(VEC + 1, 1)),
            byte_alignment=16,
        )
        vals = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((streams, BLOCK_H), stride=(BLOCK_H, 1)),
            byte_alignment=16,
        )
        if tidx < cols:
            logit = cutlass.Float32(0.0)
            for t in cutlass.range_constexpr(streams):
                logit = pv[t] * rstd[t] + logit
            z = logit * (1.0 / streams)
            sig = 1.0 / (1.0 + cute.math.exp(-z))
            if tidx < rank:
                low[tidx // VEC, tidx % VEC] = z * sig
            else:
                if const_expr(self.inject_out):
                    if bidx == 0:
                        mInj[0, tidx - rank] = (2.0 * sig).to(cutlass.BFloat16)
        cute.arch.sync_threads()

        # 3. Up projection over the row's 8 threads (two independent sums), gate, normed
        # element, stream mean.
        sums = [cutlass.Float32(0.0), cutlass.Float32(0.0)]
        for kt in cutlass.range_constexpr(k_tiles):
            for e in cutlass.range_constexpr(VEC):
                sums[e % 2] = (
                    u_reg[kt, e].to(cutlass.Float32) * low[kt * GROUP + q, e] + sums[e % 2]
                )
        dot = sums[0] + sums[1]
        dot = cute.arch.warp_reduction(dot, lambda a, b: a + b, threads_in_group=GROUP)
        if q == 0:
            gate = 1.0 / (1.0 + cute.math.exp(-dot))
            vals[s, i] = gate * (x * rstd[s] * gam)
        cute.arch.sync_threads()
        if warp == 0:
            m = cutlass.Float32(0.0)
            if lane < BLOCK_H:
                tot = cutlass.Float32(0.0)
                for t in cutlass.range_constexpr(streams):
                    tot = tot + vals[t, lane]
                mb = (tot * (1.0 / streams)).to(cutlass.BFloat16)
                mMixed[0, i0 + lane] = mb
                m = mb.to(cutlass.Float32)
            if const_expr(self.fp4):
                # NVFP4 of the 16 rounded values in lanes 0..15 (lanes 16..31 a second,
                # discarded group), as `fp4_quantize`: e4m3(gs * amax / 6), then
                # x * rcp(sf * rcp(gs)) rounded to e2m1, nearest, ties to even.
                amax = cute.arch.warp_reduction_max(cute.math.abs(m), threads_in_group=BLOCK_H)
                sf8 = (gs * (amax * cute.arch.rcp_approx(6.0))).to(cutlass.Float8E4M3FN)
                osc = cute.arch.rcp_approx(sf8.to(cutlass.Float32) * cute.arch.rcp_approx(gs))
                if amax == 0.0:
                    osc = cutlass.Float32(0.0)
                sv = m * osc
                a = cute.math.abs(sv)
                code = cutlass.Int32(0)
                if a > 0.25:
                    code = code + 1
                if a >= 0.75:
                    code = code + 1
                if a > 1.25:
                    code = code + 1
                if a >= 1.75:
                    code = code + 1
                if a > 2.5:
                    code = code + 1
                if a >= 3.5:
                    code = code + 1
                if a > 5.0:
                    code = code + 1
                if cute.math.copysign(1.0, sv) < 0.0:  # the sign bit, -0.0 included
                    code = code + 8
                hi = cute.arch.shuffle_sync_down(code, offset=1)
                if lane < BLOCK_H:
                    if lane % 2 == 0:
                        mXq[0, (i0 + lane) // 2] = (code + hi * 16).to(cutlass.Uint8)
                    if lane == 0:
                        mXsf[bidx] = sf8


def _dl(t: torch.Tensor, align: int = 16):
    return from_dlpack(t, assumed_align=align)


@functools.lru_cache(maxsize=None)
def _compiled(
    hidden, streams, rank, part_cols, part_stride, ss_stride, hc_stride, inject_out, fp4, pdl
):
    kern = _MixKernel(
        hidden=hidden, streams=streams, rank=rank, inject_out=inject_out, fp4=fp4, pdl=pdl
    )
    dev, bf = "cuda", torch.bfloat16
    width = hidden * streams
    part = torch.empty(streams, part_stride, dtype=torch.float32, device=dev)[:, :part_cols]
    ss = torch.empty(streams * ss_stride, dtype=torch.float32, device=dev)[::ss_stride]
    up = torch.empty(width, rank, dtype=bf, device=dev)
    hc = torch.empty(1, hc_stride, dtype=bf, device=dev)[:, :width]
    norm = torch.empty(1, width, dtype=bf, device=dev)
    gs = torch.empty(1, dtype=torch.float32, device=dev)
    mixed = torch.empty(1, hidden, dtype=bf, device=dev)
    inj = torch.empty(1, streams, dtype=bf, device=dev)
    xq = torch.empty(1, hidden // 2, dtype=torch.uint8, device=dev)
    xsf = torch.empty(hidden // 16, dtype=torch.float8_e4m3fn, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(
        kern,
        _dl(part, 4),
        _dl(ss, 4),
        _dl(up),
        _dl(hc, 2),
        _dl(norm, 2),
        _dl(gs, 4),
        _dl(mixed, 2),
        _dl(inj, 2),
        _dl(xq, 1),
        _dl(xsf, 1),
        cutlass.Float32(1e-6),
        stream,
        options=_COMPILE_OPTIONS,
    )


def launch(part, ss, up, hc_out, norm_weight, eps, hc_count, with_inject, global_scale=None):
    """The launch behind `Qwen4ExpHcDecodeMixFp4Cute.__call__`. Returns `(mixed,
    inject_out, xq, xsf)`; without `global_scale` (the kernel's plain build, kept for
    measurement) `xq` / `xsf` are None."""
    rows, width = hc_out.shape
    hidden = width // hc_count
    rank = up.shape[0]
    dev, dt = hc_out.device, hc_out.dtype
    fp4 = global_scale is not None
    mixed = torch.empty((rows, hidden), dtype=dt, device=dev)
    inject_out = torch.empty((rows, hc_count), dtype=dt, device=dev) if with_inject else None
    xq = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=dev) if fp4 else None
    xsf = torch.empty((rows * (hidden // 16),), dtype=torch.uint8, device=dev) if fp4 else None
    if rows == 0:
        return mixed, inject_out, xq, xsf
    pdl = pdl_enabled(dev.index if dev.index is not None else torch.cuda.current_device())
    part2 = part[:, 0, :]
    fn = _compiled(
        hidden,
        hc_count,
        rank,
        part2.shape[1],
        part2.stride(0),
        ss.stride(0),
        hc_out.stride(0),
        bool(with_inject),
        fp4,
        bool(pdl),
    )
    stream = cuda_drv.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    gs = global_scale.reshape(1) if fp4 else mixed.new_empty(1, dtype=torch.float32)
    inj = inject_out if with_inject else mixed.new_empty(1, hc_count)
    q = xq if fp4 else mixed.new_empty(1, hidden // 2, dtype=torch.uint8)
    sf = (
        xsf.view(torch.float8_e4m3fn)
        if fp4
        else mixed.new_empty(hidden // 16, dtype=torch.float8_e4m3fn)
    )
    fn(
        _dl(part2, 4),
        _dl(ss[:, 0], 4),
        _dl(up.t()),
        _dl(hc_out, 2),
        _dl(norm_weight.reshape(1, -1), 2),
        _dl(gs, 4),
        _dl(mixed, 2),
        _dl(inj, 2),
        _dl(q, 1),
        _dl(sf, 1),
        cutlass.Float32(eps),
        stream,
    )
    return mixed, inject_out, xq, xsf


def _check(part, ss, up, hc_out, norm_weight, hc_count) -> None:
    rows, width = hc_out.shape
    assert rows <= MAX_ROWS, "one row"
    assert width // hc_count == 2560 and up.shape[0] == 320 and hc_count == 4, (
        "the checkpoint's 4 x 2560, rank 320"
    )
    assert hc_out.dtype == torch.bfloat16 and up.dtype == torch.bfloat16, "bf16"
    assert up.t().is_contiguous(), "up the [R, width] view of a row-major [width, R] weight"


_CELLS = (
    Cell(why="checkpoint 4x2560 r320 bf16, inject, 1 row (con 1)", spec=dict(rows=1)),
    Cell(why="stack-final mixer (no inject), 1 row", spec=dict(rows=1, with_inject=False)),
    Cell(why="hc_out a row slice buf[::2], 1 row", spec=dict(rows=1, layout="row_slice")),
)


class Qwen4ExpHcDecodeMixFp4Cute(OpWrapper):
    """`qwen4_exp_hc_decode_mix_fp4` for a one-row batch: `(mixed, inject_out, xq,
    xsf)`, the NVFP4 of the rounded `mixed` with linear scales (see that entry).

    Fusion boundary: as `qwen4_exp_hc_decode_mix_fp4`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _CELLS

    note = """
    Metadata consumed: none. Call sites: `DecodeTarget`'s MoE sites at a one-row
    decode step, after `qwen4_exp_hc_decode_first_cute`; batches of more rows keep the
    Triton entries.

    Preconditions: the Triton entry's, and one row, the checkpoint's 4 x 2560 width,
    rank 320, bf16; `up` the `.t()` view of a contiguous `[width, R]` weight;
    `global_scale` fp32 with one element.

    Numerics: fp32 inside, one rounding per output, the Triton entry's formula and
    order up to fp32 reassociation of the up-projection dot (inside its band); the
    NVFP4 is `fp4_quantize`'s arithmetic on the rounded `mixed` (`rcp.approx.ftz`
    reciprocals, e4m3 scale rounded to nearest, e2m1 nearest-even with saturation at
    6). Deterministic; graph-capturable once compiled (the first call per variant).
    PDL: the weight rows and norm elements stream before `griddepcontrol.wait`.
    """

    def __call__(self, part, ss, up, hc_out, norm_weight, eps, hc_count, with_inject, global_scale):
        return launch(part, ss, up, hc_out, norm_weight, eps, hc_count, with_inject, global_scale)

    def reference(self, *args, **kwargs):
        """The Triton entry's reference: float64, then the NVFP4 of the rounded `mixed`."""
        return _triton_mix_fp4.reference(*args, **kwargs)

    def is_valid(
        self, part, ss, up, hc_out, norm_weight, eps, hc_count, with_inject, global_scale
    ) -> None:
        _triton_mix_fp4.is_valid(
            part, ss, up, hc_out, norm_weight, eps, hc_count, with_inject, global_scale
        )
        _check(part, ss, up, hc_out, norm_weight, hc_count)
        assert global_scale.numel() == 1 and global_scale.dtype == torch.float32, (
            "global_scale fp32 [1]"
        )


qwen4_exp_hc_decode_mix_fp4_cute = Qwen4ExpHcDecodeMixFp4Cute()


class Qwen4ExpHcDecodeMixCute(OpWrapper):
    """`qwen4_exp_hc_decode_mix` for a one-row batch: `(mixed, inject_out)` -- this
    kernel's plain build (no NVFP4 output).

    Fusion boundary: as `qwen4_exp_hc_decode_mix`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _CELLS

    note = """
    Metadata consumed: none. Call sites: `DecodeTarget`'s non-MoE sites at a one-row
    decode step, after `qwen4_exp_hc_decode_first_cute`; batches of more rows keep the
    Triton entry. Beside a first kernel capped at 96 registers, its 160 CTAs (512
    threads at 49 registers) are all resident while that kernel runs, so the up weight
    is in registers when the wait returns; the Triton entry's 320 programs are not.

    Preconditions: the Triton entry's, and one row, the checkpoint's 4 x 2560 width,
    rank 320, bf16; `up` the `.t()` view of a contiguous `[width, R]` weight.

    Numerics: fp32 inside, one rounding per output, the Triton entry's formula and
    order up to fp32 reassociation of the up-projection dot (inside its band).
    Deterministic; graph-capturable once compiled (the first call per variant).
    """

    def __call__(self, part, ss, up, hc_out, norm_weight, eps, hc_count, with_inject):
        mixed, inject_out, _, _ = launch(
            part, ss, up, hc_out, norm_weight, eps, hc_count, with_inject
        )
        return mixed, inject_out

    def reference(self, *args, **kwargs):
        """The Triton entry's float64 reference."""
        return _triton_mix.reference(*args, **kwargs)

    def is_valid(self, part, ss, up, hc_out, norm_weight, eps, hc_count, with_inject) -> None:
        _triton_mix.is_valid(part, ss, up, hc_out, norm_weight, eps, hc_count, with_inject)
        _check(part, ss, up, hc_out, norm_weight, hc_count)


qwen4_exp_hc_decode_mix_cute = Qwen4ExpHcDecodeMixCute()

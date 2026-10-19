# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-row hyper-connection site, first half, for one-row batches: the previous
block's injection (with the MoE block's combine and the routed experts' finalize
folded in, when given) and the first projection per stream, one PDL launch.

Tier 5 (CuTe DSL written for this target). `qwen4_exp_hc_decode_first` computes the
same thing for up to 16 rows in Triton, whose power-of-two tiles pad a
2560-wide stream to 8 x 512: three eighths of every program's weight, activation
and scale tiles are masked lanes holding registers. Here a stream is exactly
`THREADS x VEC` elements, one 16-byte vector per thread, and the grid is sized so
that every CTA is resident at once (one wave), so all of them stream their weight
before the producer finishes.

Order of work inside a CTA (`OUTS` weight rows x one stream):

1. its weight rows and its slice of the norm weight into registers -- static, so
   before `griddepcontrol.wait`, while the kernel ahead of this one runs;
2. `griddepcontrol.wait`; then the stream's vector of the residual and of the
   block output -- with the fold, the slot map and the weights first, then the
   slot rows -- and the injection weight;
3. the injected residual rounded once (stored by the stream's first CTA), its sum
   of squares and its `(1 + w)`-scaled products with the weight rows, in fp32;
4. a warp then a cross-warp reduction; thread `j < OUTS` stores partial `j`, thread
   `OUTS` of the stream's first CTA the sum of squares.
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .qwen4_exp_hc_decode_first import qwen4_exp_hc_decode_first as _triton_first

#: One decode row per launch (the 1-row batch is the one the Triton entry is slowest at
#: relative to its weight stream).
MAX_ROWS = 1
VEC = 8
#: Threads per CTA: one 16-byte vector of a 2560-wide stream each.
THREADS = 320
#: Streaming multiprocessors the one-wave grid must fit (GB300).
_SMS = 152
_COMPILE_OPTIONS = "--ptxas-options -maxrregcount=128"


@functools.lru_cache(maxsize=None)
def pdl_enabled(device_index: int) -> bool:
    """Programmatic dependent launch: sm_90+ unless TRTLLM_ENABLE_PDL=0."""
    props = torch.cuda.get_device_properties(device_index)
    return props.major >= 9 and os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def outs_for(n_cols: int, hc_count: int) -> int:
    """Weight rows per CTA: the fewest that divide `n_cols` and keep the grid in one wave."""
    for outs in range(1, n_cols + 1):
        if n_cols % outs == 0 and (n_cols // outs) * hc_count <= _SMS:
            return outs
    raise AssertionError(f"no one-wave tiling for {n_cols} columns")


class _FirstKernel:
    def __init__(self, *, hidden, outs, inject, combine, topk, pdl):
        assert hidden == THREADS * VEC, "one vector per thread"
        self.hidden, self.outs = hidden, outs
        self.inject, self.combine, self.topk, self.pdl = inject, combine, topk, pdl
        self.warps = THREADS // 32

    @cute.jit
    def __call__(
        self,
        mHc,
        mBlock,
        mInj,
        mShared,
        mGate,
        mTopW,
        mTopRows,
        mNorm,
        mW,
        mHcOut,
        mPart,
        mSs,
        stream: cuda_drv.CUstream,
    ):
        n_tiles = cute.size(mPart, mode=[1]) // self.outs
        streams = cute.size(mPart, mode=[0])
        load_static = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.BFloat16,
            num_bits_per_copy=16 * VEC,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        load_act = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.BFloat16,
            num_bits_per_copy=16 * VEC,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=16 * VEC
        )
        self.kernel(
            mHc,
            mBlock,
            mInj,
            mShared,
            mGate,
            mTopW,
            mTopRows,
            mNorm,
            mW,
            mHcOut,
            mPart,
            mSs,
            load_static,
            load_act,
            store,
        ).launch(
            grid=[n_tiles, streams, 1],
            block=[THREADS, 1, 1],
            smem=(self.outs + 1) * self.warps * 4,
            stream=stream,
            use_pdl=self.pdl,
            # At most 96 registers a thread: the mix kernel that follows (512 threads at
            # 56) then fits on the SM beside this CTA and streams its weights before
            # this kernel ends.
            min_blocks_per_mp=2,
        )

    @cute.kernel
    def kernel(
        self,
        mHc,
        mBlock,
        mInj,
        mShared,
        mGate,
        mTopW,
        mTopRows,
        mNorm,
        mW,
        mHcOut,
        mPart,
        mSs,
        load_static: cute.CopyAtom,
        load_act: cute.CopyAtom,
        store: cute.CopyAtom,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, s, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        outs: cutlass.Constexpr = self.outs
        warps: cutlass.Constexpr = self.warps
        topk: cutlass.Constexpr = self.topk
        n0 = bidx * outs
        vi = s * THREADS + tidx  # this thread's vector of the width; `tidx` of the stream

        w_vec = cute.logical_divide(mW, (None, VEC))
        hc_vec = cute.logical_divide(mHc, (None, VEC))
        nw_vec = cute.logical_divide(mNorm, (None, VEC))

        # 1. Static operands, before the wait.
        w_reg = cute.make_rmem_tensor(
            cute.make_layout((outs, VEC), stride=(VEC, 1)), cutlass.BFloat16
        )
        for j in cutlass.range_constexpr(outs):
            cute.copy(load_static, w_vec[n0 + j, (None, vi)], w_reg[j, None])
        nw_reg = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        cute.copy(load_static, nw_vec[0, (None, vi)], nw_reg)
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # 2. The producers' outputs.
        x_reg = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        cute.copy(load_act, hc_vec[0, (None, vi)], x_reg)
        xr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.Float32)
        for v in cutlass.range_constexpr(VEC):
            xr[v] = x_reg[v].to(cutlass.Float32)
        if const_expr(self.inject):
            b_vec = cute.logical_divide(mBlock, (None, VEC))
            g = mInj[0, s].to(cutlass.Float32)
            b = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.Float32)
            if const_expr(topk > 0):
                # The runner's finalize: fp32 FMAs in slot order from zero, a slot mapped
                # to -1 skipped, one rounding. Every slot row is requested before any FMA.
                rows = cute.make_rmem_tensor(cute.make_layout((topk,), stride=(1,)), cutlass.Int32)
                wts = cute.make_rmem_tensor(cute.make_layout((topk,), stride=(1,)), cutlass.Float32)
                for k in cutlass.range_constexpr(topk):
                    rows[k] = mTopRows[0, k]
                    wts[k] = mTopW[0, k].to(cutlass.Float32)
                fc2 = cute.make_rmem_tensor(
                    cute.make_layout((topk, VEC), stride=(VEC, 1)), cutlass.BFloat16
                )
                for k in cutlass.range_constexpr(topk):
                    cute.copy(
                        load_act,
                        b_vec[cutlass.max(rows[k], cutlass.Int32(0)), (None, tidx)],
                        fc2[k, None],
                    )
                for v in cutlass.range_constexpr(VEC):
                    acc = cutlass.Float32(0.0)
                    for k in cutlass.range_constexpr(topk):
                        if rows[k] >= 0:
                            acc = cute.math.fma(fc2[k, v].to(cutlass.Float32), wts[k], acc)
                    b[v] = acc.to(cutlass.BFloat16).to(cutlass.Float32)
            else:
                blk = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
                cute.copy(load_act, b_vec[0, (None, tidx)], blk)
                for v in cutlass.range_constexpr(VEC):
                    b[v] = blk[v].to(cutlass.Float32)
            if const_expr(self.combine):
                # The MoE block output, routed + sigmoid(gate) * shared in fp32, rounded once.
                sh_vec = cute.logical_divide(mShared, (None, VEC))
                sh = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
                cute.copy(load_act, sh_vec[0, (None, tidx)], sh)
                z = mGate[0].to(cutlass.Float32)
                sg = 1.0 / (1.0 + cute.math.exp(-z))
                for v in cutlass.range_constexpr(VEC):
                    b[v] = (
                        cute.math.fma(sg, sh[v].to(cutlass.Float32), b[v])
                        .to(cutlass.BFloat16)
                        .to(cutlass.Float32)
                    )
            out = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
            for v in cutlass.range_constexpr(VEC):
                # Rounded once, like `addcmul`: the norm and every later site read this value.
                out[v] = cute.math.fma(b[v], g, xr[v]).to(cutlass.BFloat16)
                xr[v] = out[v].to(cutlass.Float32)
            if bidx == 0:
                o_vec = cute.logical_divide(mHcOut, (None, VEC))
                cute.copy(store, out, o_vec[0, (None, vi)])

        # 3. Sum of squares and the scaled products, fp32.
        acc = cute.make_rmem_tensor(cute.make_layout((outs + 1,), stride=(1,)), cutlass.Float32)
        acc.fill(0.0)
        for v in cutlass.range_constexpr(VEC):
            y = xr[v] * (1.0 + nw_reg[v].to(cutlass.Float32))
            for j in cutlass.range_constexpr(outs):
                acc[j] = y * w_reg[j, v].to(cutlass.Float32) + acc[j]
            acc[outs] = xr[v] * xr[v] + acc[outs]

        # 4. Reductions; thread j < OUTS stores partial j, thread OUTS the sum of squares.
        for j in cutlass.range_constexpr(outs + 1):
            acc[j] = cute.arch.warp_reduction_sum(acc[j])
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((outs + 1, warps), stride=(warps, 1)),
            byte_alignment=16,
        )
        with cute.arch.elect_one():
            for j in cutlass.range_constexpr(outs + 1):
                red[j, warp] = acc[j]
        cute.arch.sync_threads()
        if tidx < outs:
            total = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(warps):
                total = total + red[tidx, w]
            mPart[s, n0 + tidx] = total
        if tidx == outs:
            if bidx == 0:
                total = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(warps):
                    total = total + red[outs, w]
                mSs[s] = total


def _dl(t: torch.Tensor, align: int = 16):
    return from_dlpack(t, assumed_align=align)


@functools.lru_cache(maxsize=None)
def _compiled(
    hidden, hc_count, n_cols, n_rows, inject, combine, topk, n_perm, pdl, hc_stride, block_stride
):
    """One compiled launch per variant and geometry (strides are part of a CuTe layout)."""
    outs = outs_for(n_cols, hc_count)
    kern = _FirstKernel(
        hidden=hidden, outs=outs, inject=inject, combine=combine, topk=topk, pdl=pdl
    )
    dev, bf = "cuda", torch.bfloat16
    width = hidden * hc_count
    hc = torch.empty(1, hc_stride, dtype=bf, device=dev)[:, :width]
    block = torch.empty(max(n_perm, 1), block_stride, dtype=bf, device=dev)[:, :hidden]
    inj = torch.empty(1, hc_count, dtype=bf, device=dev)
    shared = torch.empty(1, hidden, dtype=bf, device=dev)
    gate = torch.empty(1, dtype=bf, device=dev)
    topw = torch.empty(1, max(topk, 1), dtype=bf, device=dev)
    toprows = torch.empty(1, max(topk, 1), dtype=torch.int32, device=dev)
    norm = torch.empty(1, width, dtype=bf, device=dev)
    w = torch.empty(n_rows, width, dtype=bf, device=dev)
    hc_out = torch.empty(1, width, dtype=bf, device=dev)
    part = torch.empty(hc_count, n_cols, dtype=torch.float32, device=dev)
    ss = torch.empty(hc_count, dtype=torch.float32, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(
        kern,
        _dl(hc),
        _dl(block),
        _dl(inj, 2),
        _dl(shared),
        _dl(gate, 2),
        _dl(topw, 2),
        _dl(toprows, 4),
        _dl(norm),
        _dl(w),
        _dl(hc_out),
        _dl(part, 4),
        _dl(ss, 4),
        stream,
        options=_COMPILE_OPTIONS,
    )


def launch(
    hyper_input,
    block_out,
    inject,
    norm_weight,
    first,
    hc_count,
    n_cols,
    shared=None,
    gate_logits=None,
    topk_weights=None,
    topk_rows=None,
):
    """The launch behind `Qwen4ExpHcDecodeFirstCute.__call__`. Returns `(hc_out, part, ss)`,
    `part` fp32 `[C, 1, n_cols]` and `ss` fp32 `[C, 1]`."""
    rows, width = hyper_input.shape
    hidden = width // hc_count
    dev = hyper_input.device
    has_inject = block_out is not None
    has_shared = shared is not None
    top_k = topk_rows.shape[1] if topk_rows is not None else 0
    hc_out = (
        torch.empty((rows, width), dtype=hyper_input.dtype, device=dev)
        if has_inject
        else hyper_input
    )
    part = torch.empty((hc_count, rows, n_cols), dtype=torch.float32, device=dev)
    ss = torch.empty((hc_count, rows), dtype=torch.float32, device=dev)
    if rows == 0:
        return hc_out, part, ss
    pdl = pdl_enabled(dev.index if dev.index is not None else torch.cuda.current_device())
    weight = first.t()  # the row-major [n_rows, width] weight
    gate = gate_logits.reshape(1) if has_shared else hyper_input.new_empty(1)
    n_perm = block_out.shape[0] if (has_inject and top_k) else 1
    fn = _compiled(
        hidden,
        hc_count,
        n_cols,
        weight.shape[0],
        has_inject,
        has_shared,
        top_k,
        n_perm,
        bool(pdl),
        hyper_input.stride(0),
        block_out.stride(0) if (has_inject and top_k) else hidden,
    )
    blk = block_out if has_inject else hyper_input.new_empty(1, hidden)
    inj = inject if has_inject else hyper_input.new_empty(1, hc_count)
    sh = shared if has_shared else hyper_input.new_empty(1, hidden)
    topw = topk_weights if top_k else hyper_input.new_empty(1, 1)
    toprows = topk_rows if top_k else torch.empty(1, 1, dtype=torch.int32, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    fn(
        _dl(hyper_input),
        _dl(blk),
        _dl(inj, 2),
        _dl(sh),
        _dl(gate, 2),
        _dl(topw, 2),
        _dl(toprows, 4),
        _dl(norm_weight.reshape(1, -1)),
        _dl(weight),
        _dl(hc_out),
        _dl(part[:, 0, :], 4),
        _dl(ss[:, 0], 4),
        stream,
    )
    return hc_out, part, ss


class Qwen4ExpHcDecodeFirstCute(OpWrapper):
    """`qwen4_exp_hc_decode_first`'s computation and return values for a one-row batch
    (see that entry for the formulas, the MoE combine and the finalize fold)::

        hc_out[0, s*H+i] = hyper_input[0, s*H+i] + block_out[0, i] * inject[0, s]   (rounded once)
        part[s, 0, n]    = sum_i hc_out[0, s*H+i] * (1 + norm_weight[s*H+i]) * first[s*H+i, n]
        ss[s, 0]         = sum_i hc_out[0, s*H+i]^2

    Returns `(hc_out, part, ss)`: `hc_out` fresh (it *is* `hyper_input` at a norm-only
    site), `part` fp32 `[C, 1, N]`, `ss` fp32 `[C, 1]`. The combine (`shared`,
    `gate_logits`) and the fold (`topk_weights`, `topk_rows`) are the Triton entry's.

    Fusion boundary: as `qwen4_exp_hc_decode_first`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: the Triton entry's (rows is 1).
    CELLS: tuple[Cell, ...] = (
        Cell(why="checkpoint 4x2560, 324 cols of 384, bf16, inject, 1 row", spec=dict(rows=1)),
        Cell(why="MoE combine, [T, 1] gate, 1 row", spec=dict(rows=1, moe=True)),
        Cell(why="MoE combine, flat [T] gate, 1 row", spec=dict(rows=1, moe=True, gate="flat")),
        Cell(
            why="MoE combine into the stack-final mixer: 320 cols",
            spec=dict(rows=1, moe=True, n_cols=320),
        ),
        Cell(why="finalize fold, top-10, 1 row (con 1)", spec=dict(rows=1, moe=True, fold=10)),
        Cell(
            why="finalize fold into the stack-final mixer: 320 cols",
            spec=dict(rows=1, moe=True, fold=10, n_cols=320),
        ),
        Cell(
            why="finalize fold, last 3 slots mapped to -1",
            spec=dict(rows=1, moe=True, fold=10, drop=True),
        ),
        Cell(why="norm-only site (stack start / after PLE)", spec=dict(rows=1, inject=False)),
        Cell(why="stack-final mixer: 320 cols", spec=dict(rows=1, n_cols=320)),
        Cell(why="hyper_input a row slice buf[::2]", spec=dict(rows=1, layout="row_slice")),
        Cell(why="unpadded weight [width, 324]", spec=dict(rows=1, n_rows=324)),
    )

    note = """
    Metadata consumed: none. Call sites: `DecodeTarget`'s trunk, every
    hyper-connection site of a one-row decode step (97 per forward), followed by
    `qwen4_exp_hc_decode_mix`; batches of more rows keep `qwen4_exp_hc_decode_first`.

    Preconditions: as the Triton entry, and: one row; `hidden` 2560 (320 threads x
    8); bf16 operands; `n_cols` with a divisor that keeps `n_cols / outs x C <= 152`
    CTAs (324 -> 9 rows x 36 tiles x 4 streams = 144; 320 -> 10 rows, 128); the
    permuted rows' count is part of the compiled layout (one compile per value).

    Numerics: `hc_out` the Triton entry's arithmetic (fp32 FMAs, one rounding; the
    finalize and the combine likewise), so bitwise equal to it; `part` and `ss`
    reduce in another fp32 order (inside the same band of the float64 reference).
    Deterministic. Graph-capturable: compiled on the first call per variant, which
    must precede capture (the engine's warmup forward does it).

    PDL: with `TRTLLM_ENABLE_PDL` (default on, sm_90+) the weight rows and the norm
    slice stream before `griddepcontrol.wait`, every activation read and every
    write after it, and dependents are released before the wait.
    """

    def __call__(
        self,
        hyper_input: torch.Tensor,
        block_out: Optional[torch.Tensor],
        inject: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        first: torch.Tensor,
        hc_count: int,
        n_cols: int,
        shared: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        topk_rows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return launch(
            hyper_input,
            block_out,
            inject,
            norm_weight,
            first,
            hc_count,
            n_cols,
            shared,
            gate_logits,
            topk_weights,
            topk_rows,
        )

    def reference(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The Triton entry's float64 reference."""
        return _triton_first.reference(*args, **kwargs)

    def is_valid(
        self,
        hyper_input: torch.Tensor,
        block_out: Optional[torch.Tensor],
        inject: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        first: torch.Tensor,
        hc_count: int,
        n_cols: int,
        shared: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        topk_rows: Optional[torch.Tensor] = None,
    ) -> None:
        _triton_first.is_valid(
            hyper_input,
            block_out,
            inject,
            norm_weight,
            first,
            hc_count,
            n_cols,
            shared,
            gate_logits,
            topk_weights,
            topk_rows,
        )
        assert hyper_input.shape[0] <= MAX_ROWS, "one row"
        assert hyper_input.shape[1] // hc_count == THREADS * VEC, "hidden 2560"
        assert hyper_input.dtype == torch.bfloat16, "bf16"
        outs_for(n_cols, hc_count)


qwen4_exp_hc_decode_first_cute = Qwen4ExpHcDecodeFirstCute()

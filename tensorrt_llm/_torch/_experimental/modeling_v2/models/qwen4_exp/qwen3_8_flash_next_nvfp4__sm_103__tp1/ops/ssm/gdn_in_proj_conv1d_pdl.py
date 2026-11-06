# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Gated-DeltaNet input projection at decode rows with the causal conv1d step
as its epilogue, one PDL launch.

Tier 5 (CuTe DSL written for this target). The decode GDN block projects into one
`[T, q|k|v | z | a | b]` buffer and then runs a separate depthwise conv kernel
over the q|k|v columns. That conv touches 3 cached values and 4 taps per column:
far too little work for a launch of its own. Here every CTA computes `OUTS`
output columns over the whole K, and for a column below `conv_dim` it finishes
the conv step on the projected value before storing it.

Order of work inside a CTA:

1. its weight rows into registers, the conv taps of its columns, the row's slot
   and the cached window of each of its conv columns -- static, host-filled, or
   last written by this kernel one step ago -- so before `griddepcontrol.wait`
   (PDL), while the kernel ahead of this one is still running;
2. `griddepcontrol.wait`;
3. the activation, the products, a warp then a cross-warp reduction;
4. per column: the projection rounded to bf16 (the value the unfused path
   stores), and for a conv column of a non-pad row the conv in fp32 over
   `[window, projected]`, SiLU, rounded once; the window advances to
   `[window[1:], projected]`.

Element `k` of a row is held by thread `(k // VEC) % THREADS` as its vector
`(k // VEC) // THREADS`, so each k-tile of loads is contiguous across a warp.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

VEC = 8
#: Before its PDL wait a CTA also has the weight rows of CTAs `bidx + j * L2_DIST`
#: (j = 1 .. L2_AHEAD) brought into L2: the kernel ahead of this one is latency-bound and
#: only a few of these CTAs fit beside it, so the later waves' weights would otherwise
#: start streaming only after it.
L2_AHEAD = 2
L2_DIST = 1216


@dsl_user_op
def _bulk_prefetch_l2(src, nbytes, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(src.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
            cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


#: Largest decode row count this entry serves (every row re-reads the weight from registers).
MAX_ROWS = 2
#: (threads, outputs per CTA) per K; swept on GB300 at the checkpoint's 16480 x 2560.
TILING = {2560: (64, 2)}
WIDTH = 4  # conv taps
_COMPILE_OPTIONS = "--ptxas-options -maxrregcount=64"


class _InProjConvKernel:
    def __init__(self, *, rows, k, threads, outs, conv_dim, pad_slot, pdl):
        assert k % (threads * VEC) == 0 and threads % 32 == 0
        self.rows, self.threads, self.outs = rows, threads, outs
        self.k_tiles = k // (threads * VEC)
        self.warps = threads // 32
        self.conv_dim, self.pad_slot, self.pdl = conv_dim, pad_slot, pdl

    @cute.jit
    def __call__(self, mA, mW, mOut, mState, mConvW, mSlot, stream: cuda_drv.CUstream):
        n = cute.size(mW, mode=[0])
        load_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.BFloat16,
            num_bits_per_copy=16 * VEC,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        load_w = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.BFloat16,
            num_bits_per_copy=16 * VEC,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(mA, mW, mOut, mState, mConvW, mSlot, load_a, load_w).launch(
            grid=[cute.ceil_div(n, self.outs), 1, 1],
            block=[self.threads, 1, 1],
            smem=self.rows * self.outs * self.warps * 4,
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mA,
        mW,
        mOut,
        mState,
        mConvW,
        mSlot,
        load_a: cute.CopyAtom,
        load_w: cute.CopyAtom,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        rows: cutlass.Constexpr = self.rows
        outs: cutlass.Constexpr = self.outs
        threads: cutlass.Constexpr = self.threads
        k_tiles: cutlass.Constexpr = self.k_tiles
        warps: cutlass.Constexpr = self.warps
        n0 = bidx * outs
        my_col = n0 + tidx

        # (row, (VEC, (threads, k_tiles))): thread t's vectors of a row, k-tile by k-tile.
        w_vec = cute.logical_divide(cute.logical_divide(mW, (None, VEC)), (None, (None, threads)))
        a_vec = cute.logical_divide(cute.logical_divide(mA, (None, VEC)), (None, (None, threads)))

        # 1. Static operands, before the wait.
        w_reg = cute.make_rmem_tensor(
            cute.make_layout((outs, k_tiles, VEC), stride=(k_tiles * VEC, VEC, 1)), cutlass.BFloat16
        )
        for j in cutlass.range_constexpr(outs):
            w_row = w_vec[n0 + j, (None, (tidx, None))]
            for kt in cutlass.range_constexpr(k_tiles):
                cute.copy(load_w, w_row[None, kt], w_reg[j, kt, None])
        tap = cute.make_rmem_tensor(cute.make_layout((WIDTH,), stride=(1,)), cutlass.BFloat16)
        win = cute.make_rmem_tensor(
            cute.make_layout((rows, WIDTH - 1), stride=(WIDTH - 1, 1)), cutlass.BFloat16
        )
        if tidx < outs:
            if my_col < self.conv_dim:
                for j4 in cutlass.range_constexpr(WIDTH):
                    tap[j4] = mConvW[my_col, j4]
                for r in cutlass.range_constexpr(rows):
                    slot = mSlot[r]
                    if slot != self.pad_slot:
                        for j3 in cutlass.range_constexpr(WIDTH - 1):
                            win[r, j3] = mState[slot, my_col, j3]

        if const_expr(self.pdl):
            n_rows = cute.size(mW, mode=[0])
            k_len = cute.size(mW, mode=[1])
            if tidx < L2_AHEAD:
                tgt = bidx + (tidx + 1) * L2_DIST
                if (tgt + 1) * outs <= n_rows:
                    _bulk_prefetch_l2(mW.iterator + tgt * outs * k_len, outs * k_len * 2)

        # 2. The producer's output only after the wait.
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # 3. Products and reductions.
        acc = cute.make_rmem_tensor(
            cute.make_layout((rows, outs), stride=(outs, 1)), cutlass.Float32
        )
        acc.fill(0.0)
        a_reg = cute.make_rmem_tensor(
            cute.make_layout((k_tiles, VEC), stride=(VEC, 1)), cutlass.BFloat16
        )
        for r in cutlass.range_constexpr(rows):
            a_row = a_vec[r, (None, (tidx, None))]
            for kt in cutlass.range_constexpr(k_tiles):
                cute.copy(load_a, a_row[None, kt], a_reg[kt, None])
            for kt in cutlass.range_constexpr(k_tiles):
                for v in cutlass.range_constexpr(VEC):
                    av = a_reg[kt, v].to(cutlass.Float32)
                    for j in cutlass.range_constexpr(outs):
                        acc[r, j] = acc[r, j] + av * w_reg[j, kt, v].to(cutlass.Float32)
        for r in cutlass.range_constexpr(rows):
            for j in cutlass.range_constexpr(outs):
                acc[r, j] = cute.arch.warp_reduction_sum(acc[r, j])
        smem = cutlass.utils.SmemAllocator()
        part = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((rows, outs, warps), stride=(outs * warps, warps, 1)),
            byte_alignment=16,
        )
        with cute.arch.elect_one():
            for r in cutlass.range_constexpr(rows):
                for j in cutlass.range_constexpr(outs):
                    part[r, j, warp] = acc[r, j]
        cute.arch.sync_threads()

        # 4. Thread j < outs finishes column n0 + j.
        if tidx < outs:
            for r in cutlass.range_constexpr(rows):
                total = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(warps):
                    total = total + part[r, tidx, w]
                proj = total.to(cutlass.BFloat16)
                res = proj
                if my_col < self.conv_dim:
                    slot = mSlot[r]
                    if slot != self.pad_slot:
                        c = (
                            tap[0].to(cutlass.Float32) * win[r, 0].to(cutlass.Float32)
                            + tap[1].to(cutlass.Float32) * win[r, 1].to(cutlass.Float32)
                            + tap[2].to(cutlass.Float32) * win[r, 2].to(cutlass.Float32)
                            + tap[3].to(cutlass.Float32) * proj.to(cutlass.Float32)
                        )
                        res = (c / (1.0 + cute.math.exp(-c))).to(cutlass.BFloat16)
                        mState[slot, my_col, 0] = win[r, 1]
                        mState[slot, my_col, 1] = win[r, 2]
                        mState[slot, my_col, 2] = proj
                mOut[r, my_col] = res


def _dl(t: torch.Tensor, align: int):
    return from_dlpack(t, assumed_align=align)


_COMPILED: dict = {}


def _operands(x, mat_b, out, conv_state, conv_weight, conv_state_indices, stream):
    return (
        _dl(x, 32),
        _dl(mat_b.t(), 32),
        _dl(out, 32),
        _dl(conv_state, 2),
        _dl(conv_weight, 8),
        _dl(conv_state_indices, 4),
        stream,
    )


def launch(x, mat_b, conv_state, conv_weight, conv_state_indices, conv_dim, pad_slot_id, pdl):
    """The launch behind `GdnInProjConv1dPdl.__call__`. Returns the fresh `[M, N]` output."""
    rows, k = x.shape
    n = mat_b.shape[1]
    dev = x.device
    pdl = bool(pdl and get_env_enable_pdl())
    out = torch.empty(rows, n, dtype=x.dtype, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    operands = (x, mat_b, out, conv_state, conv_weight, conv_state_indices, stream)
    key = (
        rows,
        n,
        k,
        conv_dim,
        int(pad_slot_id),
        pdl,
        tuple(conv_state.shape),
        conv_state.stride(),
    )
    fn = _COMPILED.get(key)
    if fn is None:
        # Compiled from the call's own operands (a stand-in for the pool would span every
        # layer of the coalesced conv cache).
        threads, outs = TILING[k]
        kern = _InProjConvKernel(
            rows=rows,
            k=k,
            threads=threads,
            outs=outs,
            conv_dim=conv_dim,
            pad_slot=int(pad_slot_id),
            pdl=pdl,
        )
        fn = _COMPILED[key] = cute_ext.compile(
            kern, *_operands(*operands), options=_COMPILE_OPTIONS
        )
    fn(*_operands(*operands))
    return out


def _cell(why: str, rows: int, slots) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, slots=slots, pdl=True, pool=461))


class GdnInProjConv1dPdl(OpWrapper):
    """`out = x @ mat_b`, then for `c < conv_dim` and a row whose slot is not
    `pad_slot_id` the causal conv step on the projected value, one launch.

    `x` is the dense row-major `[M, K]` activation (`M <= MAX_ROWS`), `mat_b` the
    dense column-major `[K, N]` view of the stored `[N, K]` projection (its
    `.t()`). Row `r`, slot `s = conv_state_indices[r]`, column `c < conv_dim`:
    `p = round_bf16(sum_k x[r, k] * W[c, k])` (fp32 products and sums),
    `out[r, c] = round_bf16(silu(sum_j conv_weight[c, j] * [conv_state[s, c, :], p][j]))`
    in fp32, and `conv_state[s, c, :] = [conv_state[s, c, 1:], p]`. Every other
    column, and every column of a pad row, is `p`; a pad row's slot is not read or
    written. Returns the fresh `[M, N]` output; `conv_state` is advanced in place.

    Equivalent to `low_m_bf16_mm` followed by `causal_conv1d_update(x=out[:, :conv_dim],
    conv_state, conv_weight, bias=None, silu_activation=True, cache_seqlens=None,
    conv_state_indices, pad_slot_id)` up to the GEMV's summation order.

    Fusion boundary: the projection, the conv step, SiLU and the window's advance.
    The split into q|k|v, z, a, b and everything after are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, slots (tuple per row; -1 is the pad slot), pdl, pool (slots in the pool).
    CELLS: tuple[Cell, ...] = (
        _cell("decode 1 row, slot 37 of the 461-slot pool", 1, (37,)),
        _cell("decode 2 rows, two slots", 2, (5, 211)),
        _cell("decode 2 rows, one a pad row (-1)", 2, (17, -1)),
    )

    note = """
    Metadata consumed: the decode rows' conv slots (`conv_state_indices`, int32
    `[M]`). Preconditions: bf16 `x` dense `[M, K]`, `1 <= M <= MAX_ROWS`; `mat_b`
    the dense column-major `[K, N]` view of a `[N, K]` weight with K in `TILING`
    and `N` a multiple of the tile's outputs; `conv_state` `[slots, conv_dim, 3]`
    bf16 (any strides; they are compiled in), `conv_weight` `[conv_dim, 4]` bf16
    contiguous, `conv_dim <= N`, no bias, SiLU on, `cache_seqlens` None (the
    window shifts; not a ring).

    PDL: the weight rows, the conv taps, the slots and the pool's windows are read
    before `griddepcontrol.wait` (the pool row is this layer's own, written by this
    kernel one step ago; nothing between writes it); the activation after it. The
    launch triggers its dependents right before its own wait.

    Compilation: CuTe DSL, JIT on the first call per (M, N, K, conv_dim,
    pad_slot_id, pdl, pool shape and strides); that call must come before
    CUDA-graph capture -- the engine's warmup forward does it.

    Numerics: the projection is fp32 products and sums rounded once to bf16 (the
    value stored for non-conv columns and pushed into the window); the conv is
    fp32 over the bf16 window, taps and projection, rounded once. Only the
    GEMV's summation order differs from `low_m_bf16_mm` + `causal_conv1d_update`.

    Certified with PDL on, over the 461-slot pool: 1 and 2 rows, and 2 rows with a pad
    row; two consecutive steps under CUDA-graph replay (1 row). Not certified: `pdl=False`.

    Hard limits
      M     rows per launch                boundary MAX_ROWS=2     driven at 1, 2 (cells)
      K     a TILING key                   2560                    the checkpoint's hidden size
      pool  any slot count and strides     461 driven (64 under graph replay)
    """

    def __call__(
        self,
        x: torch.Tensor,
        mat_b: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        conv_dim: int,
        pad_slot_id: int,
        pdl: bool = True,
    ) -> torch.Tensor:
        return launch(
            x, mat_b, conv_state, conv_weight, conv_state_indices, conv_dim, pad_slot_id, pdl
        )

    def reference(
        self,
        x: torch.Tensor,
        mat_b: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        conv_dim: int,
        pad_slot_id: int,
        pdl: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, conv_state)` as the call leaves them, in fp32 then rounded; mutates nothing."""
        with true_fp32_matmul():
            proj = (x.float() @ mat_b.float()).to(x.dtype)
        out = proj.clone()
        state = conv_state.clone()
        for r, s in enumerate(conv_state_indices.tolist()):
            if s == pad_slot_id:
                continue
            p = proj[r, :conv_dim].float()
            window = torch.cat([conv_state[s].float(), p[:, None]], dim=1)
            c = (window * conv_weight.float()).sum(1)
            out[r, :conv_dim] = (c * torch.sigmoid(c)).to(x.dtype)
            state[s] = window[:, 1:].to(conv_state.dtype)
        return out, state

    def is_valid(
        self,
        x: torch.Tensor,
        mat_b: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        conv_dim: int,
        pad_slot_id: int,
        pdl: bool = True,
    ) -> None:
        assert x.dtype == mat_b.dtype == conv_state.dtype == conv_weight.dtype == torch.bfloat16
        assert x.dim() == 2 and x.is_contiguous() and 1 <= x.shape[0] <= MAX_ROWS, tuple(x.shape)
        assert mat_b.shape[0] == x.shape[1] and x.shape[1] in TILING, (
            tuple(mat_b.shape),
            tuple(x.shape),
        )
        assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
            "mat_b must be dense column-major [K, N] (e.g. weight.t())"
        )
        assert mat_b.shape[1] % TILING[x.shape[1]][1] == 0, (
            "N must be a multiple of the tile's outputs"
        )
        assert conv_state.dim() == 3 and tuple(conv_state.shape[1:]) == (conv_dim, WIDTH - 1)
        assert tuple(conv_weight.shape) == (conv_dim, WIDTH) and conv_weight.is_contiguous()
        assert conv_dim <= mat_b.shape[1]
        assert conv_state_indices.dtype == torch.int32 and tuple(conv_state_indices.shape) == (
            x.shape[0],
        )


gdn_in_proj_conv1d_pdl = GdnInProjConv1dPdl()

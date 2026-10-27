# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""LDGSTS split-KV combine for paged decode attention."""

import math
from functools import partial
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync

from src.common.cute_dsl_utils import assume_tensor_aligned, torch2cute_dtype_map


class SparseDecodeForwardCombine:
    """Combine split-KV decode partials with FA-style LDGSTS staging.

    ``mO_partial`` and ``mLSE_partial`` use the split-major padded layout:
    ``partial_row = o_indptr[b] + split_idx * q_stride + q_token`` where
    ``q_stride = ceil_div(seqlen_q, q_tokens_per_group) * q_tokens_per_group``.
    A CTA covers ``tile_m`` flattened ``(q_token, q_head)`` rows and one
    ``k_block_size`` slice of D.  O_partial and LSE_partial are loaded to SMEM
    via ``cpasync.CopyG2SOp`` before the split reduction.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        *,
        tile_m: int = 64,
        k_block_size: int = 128,
        max_splits: int = 4,
        num_threads: int = 256,
        stages: int = 2,
    ):
        if head_dim != 128:
            raise NotImplementedError(
                f"SparseDecodeForwardCombine currently supports only D=128, got D={head_dim}"
            )
        if dtype not in [cutlass.BFloat16, cutlass.Float16, cutlass.Float32]:
            raise TypeError(f"Unsupported output dtype: {dtype}")
        if dtype_partial is not Float32:
            raise TypeError("decode O_partial must be Float32")
        if k_block_size != head_dim:
            raise NotImplementedError("decode combine currently uses one D=128 k block")
        if tile_m % 8 != 0:
            raise ValueError("decode combine tile_m must be divisible by 8")
        if max_splits < 1 or max_splits > 256:
            raise ValueError("decode combine max_splits must be in [1, 256]")

        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.k_block_size = k_block_size
        self.max_splits = max_splits
        self.num_threads = num_threads
        self.stages = stages
        self.is_even_k = head_dim % k_block_size == 0

    def _setup_attributes(self) -> None:
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype_partial.width
        assert self.k_block_size % async_copy_elems == 0

        k_block_gmem = (
            128
            if self.k_block_size % 128 == 0
            else (64 if self.k_block_size % 64 == 0 else 32)
        )
        gmem_threads_per_row = k_block_gmem // async_copy_elems
        assert self.num_threads % gmem_threads_per_row == 0

        atom_async_copy_partial = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_partial,
            num_bits_per_copy=universal_copy_bits,
        )
        tOpartial_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        vOpartial_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_O_partial = cute.make_tiled_copy_tv(
            atom_async_copy_partial, tOpartial_layout, vOpartial_layout
        )

        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=async_copy_elems * self.dtype.width,
        )
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tOpartial_layout, vOpartial_layout
        )

        lse_copy_bits = Float32.width
        m_block_smem = (
            128
            if self.tile_m % 128 == 0
            else (
                64
                if self.tile_m % 64 == 0
                else (32 if self.tile_m % 32 == 0 else (16 if self.tile_m % 16 == 0 else 8))
            )
        )
        gmem_threads_per_row_lse = m_block_smem
        assert self.num_threads % gmem_threads_per_row_lse == 0

        atom_async_copy_lse = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            Float32,
            num_bits_per_copy=lse_copy_bits,
        )
        tLSE_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_lse, gmem_threads_per_row_lse),
            order=(1, 0),
        )
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_lse, tLSE_layout, cute.make_layout(1)
        )

        self.smem_threads_per_col_lse = self.num_threads // m_block_smem
        assert 32 % self.smem_threads_per_col_lse == 0
        s2r_layout_atom_lse = cute.make_ordered_layout(
            (self.smem_threads_per_col_lse, self.num_threads // self.smem_threads_per_col_lse),
            order=(0, 1),
        )
        self.s2r_tiled_copy_LSE = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_layout_atom_lse,
            cute.make_layout(1),
        )

        if const_expr(m_block_smem == 8):
            smem_lse_swizzle = cute.make_swizzle(5, 0, 5)
        elif const_expr(m_block_smem == 16):
            smem_lse_swizzle = cute.make_swizzle(4, 0, 4)
        else:
            smem_lse_swizzle = cute.make_swizzle(3, 2, 3)
        lse_atom_splits = min(self.max_splits, 8)
        smem_layout_atom_lse = cute.make_composed_layout(
            smem_lse_swizzle,
            0,
            cute.make_ordered_layout((lse_atom_splits, m_block_smem), order=(1, 0)),
        )
        self.smem_layout_lse = cute.tile_to_shape(
            smem_layout_atom_lse, (self.max_splits, self.tile_m), (0, 1)
        )
        self.smem_layout_o = cute.make_ordered_layout(
            (self.tile_m, self.k_block_size, self.stages), order=(1, 0, 2)
        )

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,      # [partial_rows, Hq, D] fp32
        mLSE_partial: cute.Tensor,    # [partial_rows, Hq] fp32
        mSplitCounts: cute.Tensor,    # [B] int32
        mOIndptr: cute.Tensor,        # [B + 1] int32
        mO: cute.Tensor,              # [total_q, Hq, D]
        mLSE: cute.Tensor,            # [total_q, Hq] fp32
        seqlen_q: Int32,
        q_tokens_per_group: Int32,
        stream: cuda.CUstream = None,
    ):
        if const_expr(mO_partial.element_type is not Float32):
            raise TypeError("decode O_partial tensor must be Float32")
        if const_expr(mLSE_partial.element_type is not Float32):
            raise TypeError("decode LSE_partial tensor must be Float32")
        if const_expr(mLSE.element_type is not Float32):
            raise TypeError("decode LSE tensor must be Float32")
        if const_expr(mO.element_type != self.dtype):
            raise TypeError("decode O tensor dtype must match kernel dtype")
        if const_expr(mSplitCounts.element_type is not Int32):
            raise TypeError("decode split_counts tensor must be Int32")
        if const_expr(mOIndptr.element_type is not Int32):
            raise TypeError("decode o_indptr tensor must be Int32")

        mO_partial, mLSE_partial, mSplitCounts, mOIndptr, mO, mLSE = [
            assume_tensor_aligned(t)
            for t in (mO_partial, mLSE_partial, mSplitCounts, mOIndptr, mO, mLSE)
        ]
        self._setup_attributes()

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sMaxValidSplit: cute.struct.Align[
                cute.struct.MemRange[Int32, self.tile_m], 128
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128
            ]

        total_q = mO.shape[0]
        head_q = mO.shape[1]
        batch = mSplitCounts.shape[0]
        head_divmod = FastDivmodDivisor(head_q)
        grid = (
            cute.ceil_div(seqlen_q * head_q, self.tile_m),
            cute.ceil_div(self.head_dim, self.k_block_size),
            batch,
        )

        self.kernel(
            mO_partial,
            mLSE_partial,
            mSplitCounts,
            mOIndptr,
            mO,
            mLSE,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_o,
            self.gmem_tiled_copy_O_partial,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_LSE,
            self.s2r_tiled_copy_LSE,
            head_divmod,
            Int32(total_q),
            Int32(head_q),
            seqlen_q,
            q_tokens_per_group,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mOIndptr: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        head_divmod: FastDivmodDivisor,
        total_q: Int32,
        head_q: Int32,
        seqlen_q: Int32,
        q_tokens_per_group: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, k_block, batch_idx = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sMaxValidSplit = storage.sMaxValidSplit.get_tensor((self.tile_m,))
        sO = storage.sO.get_tensor(smem_layout_o)

        split_count = mSplitCounts[batch_idx]
        q_stride = (
            (seqlen_q + q_tokens_per_group - Int32(1))
            // q_tokens_per_group
        ) * q_tokens_per_group
        max_idx = seqlen_q * head_q

        if m_block * Int32(self.tile_m) < max_idx:
            gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_slice(tidx)
            tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE)
            cLSE = cute.make_identity_tensor((self.max_splits, self.tile_m))
            tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE)

            for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
                mi = tLSEcLSE[0, 0, m][1]
                idx = m_block * Int32(self.tile_m) + mi
                if idx < max_idx:
                    q_idx, q_head = divmod(idx, head_divmod)
                    partial_base = mOIndptr[batch_idx] + q_idx
                    for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                        si = tLSEcLSE[0, s, 0][0]
                        if si < split_count:
                            partial_row = partial_base + si * q_stride
                            lse_ptr = (
                                mLSE_partial.iterator
                                + Int64(partial_row) * Int64(head_q)
                                + Int64(q_head)
                            )
                            lse_gmem_ptr = cute.make_ptr(
                                Float32,
                                lse_ptr.toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            )
                            lse_src = cute.make_tensor(lse_gmem_ptr, (1,))
                            cute.copy(
                                gmem_thr_copy_LSE,
                                lse_src,
                                tLSEsLSE[None, s, m],
                            )
                        else:
                            tLSEsLSE[None, s, m].fill(-Float32.inf)
                else:
                    for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                        tLSEsLSE[None, s, m].fill(-Float32.inf)
            cute.arch.cp_async_commit_group()

            gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_slice(tidx)
            cO = cute.make_identity_tensor((self.tile_m, self.k_block_size))
            tOcO = gmem_thr_copy_O_partial.partition_D(cO)
            tOsO_partial = gmem_thr_copy_O_partial.partition_D(sO)

            num_rows = const_expr(cute.size(tOcO, mode=[1]))
            tOqidx = cute.make_rmem_tensor(num_rows, Int32)
            tOhidx = cute.make_rmem_tensor(num_rows, Int32)
            for m in cutlass.range(num_rows, unroll_full=True):
                mi = tOcO[0, m, 0][0]
                idx = m_block * Int32(self.tile_m) + mi
                if idx >= max_idx:
                    tOqidx[m] = Int32(0)
                    tOhidx[m] = -Int32(1)
                else:
                    tOqidx[m], tOhidx[m] = divmod(idx, head_divmod)

            load_O_partial = partial(
                self.load_O_partial,
                mO_partial,
                mOIndptr,
                gmem_tiled_copy_O_partial,
                tOsO_partial,
                tOqidx,
                tOhidx,
                tOcO,
                batch_idx,
                q_stride,
                split_count,
                head_q,
                k_block,
            )

            for stage in cutlass.range(self.stages - 1, unroll_full=True):
                if stage < split_count:
                    load_O_partial(stage, stage)
                cute.arch.cp_async_commit_group()

            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()

            s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_slice(tidx)
            ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE)
            ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
            cute.copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE)

            lse_sum = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Float32)
            ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE)
            max_valid_split = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Int32)
            assert cute.size(ts2rrLSE, mode=[0]) == 1
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                threads_per_col = const_expr(self.smem_threads_per_col_lse)
                lse_max = cute.arch.warp_reduction_max(
                    ts2rrLSE[None, None, m]
                    .load()
                    .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                    threads_in_group=threads_per_col,
                )
                max_valid_idx = -Int32(1)
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    if ts2rrLSE[0, s, m] != -Float32.inf:
                        max_valid_idx = ts2rcLSE[0, s, 0][0]
                max_valid_split[m] = cute.arch.warp_reduction_max(
                    max_valid_idx, threads_in_group=threads_per_col
                )

                lse_max_cur = Float32(0.0) if lse_max == -Float32.inf else lse_max
                LOG2_E = Float32(math.log2(math.e))
                lse_sum_cur = Float32(0.0)
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    scale = cute.math.exp2(
                        (ts2rrLSE[0, s, m] - lse_max_cur) * LOG2_E,
                        fastmath=True,
                    )
                    lse_sum_cur += scale
                    ts2rrLSE[0, s, m] = scale
                lse_sum_cur = cute.arch.warp_reduction_sum(
                    lse_sum_cur, threads_in_group=threads_per_col
                )
                lse_sum[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                inv_sum = (
                    Float32(0.0)
                    if (lse_sum_cur == Float32(0.0) or lse_sum_cur != lse_sum_cur)
                    else cute.arch.rcp_approx(lse_sum_cur)
                )
                ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
            cute.copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE)

            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                if ts2rcLSE[0, 0, m][0] == Int32(0):
                    mi = ts2rcLSE[0, 0, m][1]
                    if mi < Int32(self.tile_m):
                        sMaxValidSplit[mi] = max_valid_split[m]

            if k_block == Int32(0):
                for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                    if ts2rcLSE[0, 0, m][0] == Int32(0):
                        mi = ts2rcLSE[0, 0, m][1]
                        idx = m_block * Int32(self.tile_m) + mi
                        if idx < max_idx:
                            q_idx, q_head = divmod(idx, head_divmod)
                            q_abs = batch_idx * seqlen_q + q_idx
                            mLSE[q_abs, q_head] = lse_sum[m]

            cute.arch.sync_threads()

            thr_max_valid_split = sMaxValidSplit[tOcO[0, 0, 0][0]]
            for m in cutlass.range(1, cute.size(tOcO, mode=[1]), unroll_full=True):
                thr_max_valid_split = max(
                    thr_max_valid_split,
                    sMaxValidSplit[tOcO[0, m, 0][0]],
                )

            tOrO_partial = cute.make_rmem_tensor_like(tOsO_partial[None, None, None, 0])
            tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
            tOrO.fill(Float32(0.0))

            stage_load = self.stages - 1
            stage_compute = 0
            for s in cutlass.range(thr_max_valid_split + Int32(1), unroll=4):
                scale = cute.make_rmem_tensor(num_rows, Float32)
                for m in cutlass.range(num_rows, unroll_full=True):
                    scale[m] = sLSE[s, tOcO[0, m, 0][0]]

                split_to_load = s + Int32(self.stages - 1)
                if split_to_load <= thr_max_valid_split:
                    load_O_partial(split_to_load, stage_load)
                cute.arch.cp_async_commit_group()
                stage_load = 0 if stage_load == self.stages - 1 else stage_load + 1

                cute.arch.cp_async_wait_group(self.stages - 1)
                cute.autovec_copy(tOsO_partial[None, None, None, stage_compute], tOrO_partial)
                stage_compute = 0 if stage_compute == self.stages - 1 else stage_compute + 1

                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= Int32(0) and scale[m] > Float32(0.0):
                        tOrO[None, m, None].store(
                            tOrO[None, m, None].load()
                            + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                        )

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            rO = cute.make_rmem_tensor_like(tOrO, self.dtype)
            rO.store(tOrO.load().to(self.dtype))
            elems_per_store = const_expr(cute.size(gmem_tiled_copy_O.layout_tv_tiled[1]))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            for m in cutlass.range(num_rows, unroll_full=True):
                if tOhidx[m] >= Int32(0):
                    q_abs = batch_idx * seqlen_q + tOqidx[m]
                    row_ptr = (
                        mO.iterator
                        + (
                            (Int64(q_abs) * Int64(head_q) + Int64(tOhidx[m]))
                            * Int64(self.head_dim)
                            + Int64(k_block * Int32(self.k_block_size))
                        )
                    )
                    row_gmem_ptr = cute.make_ptr(
                        mO.element_type,
                        row_ptr.toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    mO_row = cute.make_tensor(
                        row_gmem_ptr,
                        cute.make_layout((self.k_block_size,)),
                    )
                    mO_row_copy = cute.tiled_divide(mO_row, (elems_per_store,))
                    for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                        k_idx = tOcO[0, 0, k][1] // elems_per_store
                        cute.copy(gmem_thr_copy_O, rO[None, m, k], mO_row_copy[None, k_idx])

    @cute.jit
    def load_O_partial(
        self,
        mO_partial: cute.Tensor,
        mOIndptr: cute.Tensor,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        tOsO_partial: cute.Tensor,
        tOqidx: cute.Tensor,
        tOhidx: cute.Tensor,
        tOcO: cute.Tensor,
        batch_idx: Int32,
        q_stride: Int32,
        split_count: Int32,
        head_q: Int32,
        k_block: Int32,
        split: Int32,
        stage: Int32,
    ) -> None:
        elems_per_load = const_expr(cute.size(gmem_tiled_copy_O_partial.layout_tv_tiled[1]))
        tOsO_partial_cur = tOsO_partial[None, None, None, stage]
        for m in cutlass.range(cute.size(tOcO, [1]), unroll_full=True):
            if tOhidx[m] >= Int32(0):
                if split < split_count:
                    partial_row = mOIndptr[batch_idx] + split * q_stride + tOqidx[m]
                    row_ptr = (
                        mO_partial.iterator
                        + (
                            (Int64(partial_row) * Int64(head_q) + Int64(tOhidx[m]))
                            * Int64(self.head_dim)
                            + Int64(k_block * Int32(self.k_block_size))
                        )
                    )
                    row_gmem_ptr = cute.make_ptr(
                        mO_partial.element_type,
                        row_ptr.toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    mO_partial_row = cute.make_tensor(
                        row_gmem_ptr,
                        cute.make_layout((self.k_block_size,)),
                    )
                    mO_partial_row_copy = cute.tiled_divide(
                        mO_partial_row, (elems_per_load,))
                    for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                        k_idx = tOcO[0, 0, k][1] // elems_per_load
                        cute.copy(
                            gmem_tiled_copy_O_partial,
                            mO_partial_row_copy[None, k_idx],
                            tOsO_partial_cur[None, m, k],
                        )
                else:
                    tOsO_partial_cur[None, m, None].fill(Float32(0.0))


_combine_compile_cache: dict[tuple[object, ...], object] = {}


def _next_power_of_2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


def run_decode_combine(
    O_partial: torch.Tensor,
    LSE_partial: torch.Tensor,
    split_counts: torch.Tensor,
    o_indptr: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    seqlen_q: int,
    q_tokens_per_group: int,
    max_split_count: int,
) -> None:
    """Launch LDGSTS decode split-KV combine."""

    if O_partial.dtype != torch.float32:
        raise TypeError(f"O_partial must be torch.float32, got {O_partial.dtype}")
    if LSE_partial.dtype != torch.float32:
        raise TypeError(f"LSE_partial must be torch.float32, got {LSE_partial.dtype}")
    if lse.dtype != torch.float32:
        raise TypeError(f"lse must be torch.float32, got {lse.dtype}")
    if split_counts.dtype != torch.int32:
        raise TypeError(f"split_counts must be torch.int32, got {split_counts.dtype}")
    if o_indptr.dtype != torch.int32:
        raise TypeError(f"o_indptr must be torch.int32, got {o_indptr.dtype}")
    if out.ndim != 3 or O_partial.ndim != 3:
        raise ValueError("decode combine expects O tensors with shape [rows, heads, D]")
    if LSE_partial.ndim != 2 or lse.ndim != 2:
        raise ValueError("decode combine expects LSE tensors with shape [rows, heads]")
    if out.shape[1:] != O_partial.shape[1:]:
        raise ValueError(f"O shape mismatch: out={out.shape}, O_partial={O_partial.shape}")
    if lse.shape != out.shape[:2]:
        raise ValueError(f"lse shape {lse.shape} must match out[:2] {out.shape[:2]}")
    if LSE_partial.shape != O_partial.shape[:2]:
        raise ValueError(
            f"LSE_partial shape {LSE_partial.shape} must match O_partial[:2] {O_partial.shape[:2]}"
        )
    if split_counts.ndim != 1 or o_indptr.ndim != 1:
        raise ValueError("split_counts and o_indptr must be rank-1 tensors")
    if o_indptr.shape != (split_counts.shape[0] + 1,):
        raise ValueError(
            f"o_indptr shape {o_indptr.shape} must be ({split_counts.shape[0] + 1},)"
        )
    seqlen_q = int(seqlen_q)
    q_tokens_per_group = int(q_tokens_per_group)
    if seqlen_q <= 0:
        raise ValueError("seqlen_q must be positive")
    if q_tokens_per_group <= 0:
        raise ValueError("q_tokens_per_group must be positive")
    if out.shape[0] != split_counts.shape[0] * seqlen_q:
        raise ValueError(
            f"out rows {out.shape[0]} must equal batch*seqlen_q "
            f"{split_counts.shape[0]}*{seqlen_q}"
        )

    max_split_count = int(max_split_count)
    if max_split_count <= 0:
        raise ValueError("max_split_count must be positive")
    if max_split_count > 256:
        raise NotImplementedError(
            f"LDGSTS decode combine supports at most 256 splits, got {max_split_count}"
        )
    max_splits = max(4, _next_power_of_2(max_split_count))
    tile_m = 64
    k_block_size = int(out.shape[-1])
    stages = 2

    dtype = torch2cute_dtype_map[out.dtype]
    key = (
        "decode_combine_ldgsts",
        out.shape[-1],
        dtype,
        O_partial.dtype,
        seqlen_q,
        q_tokens_per_group,
        tile_m,
        k_block_size,
        max_splits,
        stages,
    )
    if key not in _combine_compile_cache:
        from quack.compile_utils import make_fake_tensor

        total_q = cute.sym_int64()
        batch = cute.sym_int64()
        batch_plus_one = cute.sym_int64()
        partial_rows = cute.sym_int64()
        head_q = cute.sym_int64()
        head_dim = int(out.shape[-1])
        kernel = SparseDecodeForwardCombine(
            dtype=dtype,
            dtype_partial=Float32,
            head_dim=head_dim,
            tile_m=tile_m,
            k_block_size=k_block_size,
            max_splits=max_splits,
            stages=stages,
        )
        _combine_compile_cache[key] = cute.compile(
            kernel,
            make_fake_tensor(Float32, (partial_rows, head_q, head_dim), divisibility=4),
            make_fake_tensor(Float32, (partial_rows, head_q), divisibility=1, leading_dim=1),
            make_fake_tensor(Int32, (batch,), divisibility=1, leading_dim=0),
            make_fake_tensor(Int32, (batch_plus_one,), divisibility=1, leading_dim=0),
            make_fake_tensor(dtype, (total_q, head_q, head_dim), divisibility=128 // dtype.width),
            make_fake_tensor(Float32, (total_q, head_q), divisibility=1, leading_dim=1),
            Int32(seqlen_q),
            Int32(q_tokens_per_group),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    with torch.cuda.nvtx.range("Decode_Combine_LDGSTS"):
        _combine_compile_cache[key](
            O_partial,
            LSE_partial,
            split_counts,
            o_indptr,
            out,
            lse,
            seqlen_q,
            q_tokens_per_group,
        )


__all__ = ["SparseDecodeForwardCombine", "run_decode_combine"]

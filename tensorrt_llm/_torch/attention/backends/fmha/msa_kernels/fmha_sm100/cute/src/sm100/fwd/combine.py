# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Sparse forward combine kernel and public launcher.

This keeps the local fake-layout -> real-layout epilogue needed by the lean
sparse forward path.
"""

# Modified Step 7: O_out write with SMEM fake->real column permutation.
# O_partial dim is in STG.128 fake layout; O_out dim is real layout.
import logging
import math
import os
from typing import Type, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass import Float32, Int32, Int64, Boolean, const_expr

from src.common import utils
from src.common.cute_dsl_utils import assume_tensor_aligned, torch2cute_dtype_map
from src.common.seqlen_info import SeqlenInfo
from cutlass.cute import FastDivmodDivisor

from src.common.pack_gqa import PackGQAComb
from src.common import mxfp8_epi
from src.common.tma_utils import (
    stg128_fake_col_to_real_col,
    stg128_fp8_fake_col_to_real_col,
    stg128_half_fake_col_to_real_col,
)


class SparseAttentionForwardCombine:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        tile_m: int = 8,
        k_block_size: int = 64,
        topk: int = 16,
        num_threads: int = 256,
        stages: int = 4,
        use_pdl: bool = False,
        min_blocks_per_mp: int = 0,
        write_bf16: bool = True,
    ):
        """
        Forward combine kernel for split attention computation.

        :param dtype: output data type
        :param dtype_partial: partial accumulation data type
        :param head_dim: head dimension
        :param tile_m: m block size
        :param k_block_size: k block size
        :param topk: exact number of split partials
        :param num_threads: number of threads
        :param varlen: whether using variable length sequences
        :param stages: number of pipeline stages
        """
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.k_block_size = k_block_size
        self.topk = topk
        self.num_threads = num_threads
        self.is_even_k = head_dim % k_block_size == 0
        self.stages = stages
        self.use_pdl = use_pdl
        self.min_blocks_per_mp = min_blocks_per_mp
        # MXFP8 epilogue (Host, CTX (b)): the final O also written as MXFP8 (E4M3 + K32 UE8M0 scales, swizzled
        # 128x4) for the o_proj GEMM, bitwise mxfp8_quantize(O_bf16.view(total_q, nheads * head_dim), True).
        # write_bf16=False skips the bf16 O store (only when nothing else reads it).
        self.write_bf16 = write_bf16
        self.use_stg128_half_layout = dtype_partial in (cutlass.BFloat16, cutlass.Float16)
        self.use_stg128_fp8_layout = dtype_partial is cutlass.Float8E4M3FN

    @staticmethod
    def can_implement(
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        topk,
        num_threads,
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if dtype_partial not in [
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            Float32,
        ]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if tile_m % 8 != 0:
            return False
        if topk > 256:
            return False
        if (tile_m * topk) % num_threads != 0:
            return False
        return True

    def _setup_attributes(self):
        # GMEM copy setup for O partial
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype_partial.width
        assert self.k_block_size % async_copy_elems == 0

        k_block_gmem = (
            128 if self.k_block_size % 128 == 0 else (64 if self.k_block_size % 64 == 0 else 32)
        )
        gmem_threads_per_row = k_block_gmem // async_copy_elems
        assert self.num_threads % gmem_threads_per_row == 0

        # Async copy atom for O partial load
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

        # GMEM copy setup for final O (use universal copy for store).
        # Keep this independent from O_partial: fp8 partial uses 16 elements
        # per 128b transaction, while bf16/fp16 O stores must remain 8-wide.
        output_copy_elems = universal_copy_bits // self.dtype.width
        assert self.k_block_size % output_copy_elems == 0
        gmem_threads_per_row_o = k_block_gmem // output_copy_elems
        assert self.num_threads % gmem_threads_per_row_o == 0
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tO_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_o, gmem_threads_per_row_o),
            order=(1, 0),
        )
        vO_layout = cute.make_layout((1, output_copy_elems))
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy,
            tO_layout,
            vO_layout,
        )
        # LSE copy setup with async copy (alignment = 1)
        lse_copy_bits = Float32.width  # 1 element per copy, width is in bits
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

        # Async copy atom for LSE load
        atom_async_copy_lse = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            Float32,
            num_bits_per_copy=lse_copy_bits,
        )
        tLSE_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_lse, gmem_threads_per_row_lse),
            order=(1, 0),
        )
        vLSE_layout = cute.make_layout(1)
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_lse, tLSE_layout, vLSE_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory
        # ///////////////////////////////////////////////////////////////////////////////

        # Shared memory to register copy for LSE
        self.smem_threads_per_col_lse = self.num_threads // m_block_smem
        assert 32 % self.smem_threads_per_col_lse == 0  # Must divide warp size

        s2r_layout_atom_lse = cute.make_ordered_layout(
            (self.smem_threads_per_col_lse, self.num_threads // self.smem_threads_per_col_lse),
            order=(0, 1),
        )
        self.s2r_tiled_copy_LSE = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_layout_atom_lse,
            cute.make_layout(1),
        )

        # LSE shared memory layout with swizzling to avoid bank conflicts
        # This works for kBlockMSmem = 8, 16, 32, 64, 128, no bank conflicts
        if const_expr(m_block_smem == 8):
            smem_lse_swizzle = cute.make_swizzle(5, 0, 5)
        elif const_expr(m_block_smem == 16):
            smem_lse_swizzle = cute.make_swizzle(4, 0, 4)
        else:
            smem_lse_swizzle = cute.make_swizzle(3, 2, 3)
        lse_atom_splits = min(self.topk, 8)
        smem_layout_atom_lse = cute.make_composed_layout(
            smem_lse_swizzle,
            0,
            cute.make_ordered_layout((lse_atom_splits, m_block_smem), order=(1, 0)),
        )
        self.smem_layout_lse = cute.tile_to_shape(
            smem_layout_atom_lse, (self.topk, self.tile_m), (0, 1)
        )

        # O_partial staging layout.
        if const_expr(
            self.dtype_partial
            in [cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN]
        ):
            smem_layout_atom_o = _get_cpasync_smem_layout_atom(
                self.dtype_partial, self.k_block_size
            )
            self.smem_layout_o = cute.tile_to_shape(
                smem_layout_atom_o,
                (self.tile_m, self.k_block_size, self.stages),
                (0, 1, 2),
            )
        else:
            self.smem_layout_o = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size, self.stages), order=(1, 0, 2)
            )

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor] = None,
        mLSE_temperature_partial: Optional[cute.Tensor] = None,
        mLSE_temperature: Optional[cute.Tensor] = None,
        cu_seqlens: Optional[cute.Tensor] = None,
        seqused: Optional[cute.Tensor] = None,
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        varlen_batch_idx: Optional[cute.Tensor] = None,
        semaphore_to_reset: Optional[cute.Tensor] = None,
        mSplitCounts: Optional[cute.Tensor] = None,
        mOutputScale: Optional[cute.Tensor] = None,
        qhead_per_kvhead: Int32 = Int32(1),
        mOq: Optional[cute.Tensor] = None,
        mSFq: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Type checking
        if const_expr(not (mO_partial.element_type == self.dtype_partial)):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(not (mO.element_type == self.dtype)):
            raise TypeError("O tensor must match dtype")
        if const_expr(mLSE_partial.element_type not in [Float32]):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(
            mLSE_temperature_partial is not None
            and mLSE_temperature_partial.element_type not in [Float32]
        ):
            raise TypeError("temperature LSE partial tensor must be Float32")
        if const_expr(mLSE_temperature is not None and mLSE_temperature.element_type not in [Float32]):
            raise TypeError("temperature LSE tensor must be Float32")
        if const_expr((mLSE_temperature_partial is None) != (mLSE_temperature is None)):
            raise ValueError(
                "temperature LSE partial and output tensors must either both be provided or both be None"
            )

        # Shape validation - input tensors are in user format, need to be converted to kernel format
        if const_expr(len(mO_partial.shape) not in [4, 5]):
            raise ValueError(
                "O partial tensor must have 4 or 5 dimensions: (num_splits, batch, seqlen, nheads, headdim) or (num_splits, total_q, nheads, headdim)"
            )
        if const_expr(len(mLSE_partial.shape) not in [3, 4]):
            raise ValueError(
                "LSE partial tensor must have 3 or 4 dimensions: (num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(len(mO.shape) not in [3, 4]):
            raise ValueError(
                "O tensor must have 3 or 4 dimensions: (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim)"
            )
        if const_expr(mLSE is not None and len(mLSE.shape) not in [2, 3]):
            raise ValueError(
                "LSE tensor must have 2 or 3 dimensions: (batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(mLSE_temperature_partial is not None and len(mLSE_temperature_partial.shape) not in [3, 4]):
            raise ValueError(
                "temperature LSE partial tensor must have 3 or 4 dimensions: "
                "(num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(mLSE_temperature is not None and len(mLSE_temperature.shape) not in [2, 3]):
            raise ValueError(
                "temperature LSE tensor must have 2 or 3 dimensions: "
                "(batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(mSplitCounts is not None):
            if const_expr(mSplitCounts.element_type not in [Int32]):
                raise TypeError("split_counts tensor must be Int32")
            if const_expr(cu_seqlens is not None):
                if const_expr(len(mSplitCounts.shape) != 2):
                    raise ValueError("varlen split_counts tensor must have shape (total_q, nheads_kv)")
            elif const_expr(len(mSplitCounts.shape) != 3):
                raise ValueError("batched split_counts tensor must have shape (batch, seqlen, nheads_kv)")
        if const_expr(mOutputScale is not None and mOutputScale.element_type not in [Float32]):
            raise TypeError("output_scale tensor must be Float32")

        mO_partial, mO = [assume_tensor_aligned(t) for t in (mO_partial, mO)]
        # (num_splits, b, seqlen, h, d) -> (seqlen, d, num_splits, h, b)
        # or (num_splits, total_q, h, d) -> (total_q, d, num_splits, h)
        O_partial_layout_transpose = (
            [2, 4, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 3, 0, 2]
        )
        # (b, seqlen, h, d) -> (seqlen, d, h, b) or (total_q, h, d) -> (total_q, d, h)
        mO_partial = cute.make_tensor(
            mO_partial.iterator, cute.select(mO_partial.layout, mode=O_partial_layout_transpose)
        )
        O_layout_transpose = [1, 3, 2, 0] if const_expr(cu_seqlens is None) else [0, 2, 1]
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        # (num_splits, b, h, seqlen) -> (seqlen, num_splits, h, b)
        # Input is pre-transposed: [topK, B, Hq, Sq] with Sq innermost for K2-friendly reads.
        # or (num_splits, total_q, h) -> (total_q, num_splits, h)
        LSE_partial_layout_transpose = [3, 0, 2, 1] if const_expr(cu_seqlens is None) else [1, 0, 2]
        mLSE_partial = cute.make_tensor(
            mLSE_partial.iterator,
            cute.select(mLSE_partial.layout, mode=LSE_partial_layout_transpose),
        )
        # (b, seqlen, h) -> (seqlen, h, b) or (total_q, h) -> (total_q, h)
        LSE_layout_transpose = [1, 2, 0] if const_expr(cu_seqlens is None) else [0, 1]
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if mLSE is not None
            else None
        )
        mLSE_temperature_partial = (
            cute.make_tensor(
                mLSE_temperature_partial.iterator,
                cute.select(mLSE_temperature_partial.layout, mode=LSE_partial_layout_transpose),
            )
            if mLSE_temperature_partial is not None
            else None
        )
        mLSE_temperature = (
            cute.make_tensor(
                mLSE_temperature.iterator,
                cute.select(mLSE_temperature.layout, mode=LSE_layout_transpose),
            )
            if mLSE_temperature is not None
            else None
        )

        # Determine if we have variable length sequences
        varlen = const_expr(cu_seqlens is not None or seqused is not None)

        self._setup_attributes()

        # Output-dtype permutation buffer for Step 7 (tile_m × k_block_size).
        # Accumulation stays fp32; the final dtype conversion happens before
        # the fake→real SMEM scatter to reduce half-output SMEM pressure.
        if const_expr(self.dtype in [cutlass.Float16, cutlass.BFloat16]):
            smem_layout_perm = cute.make_layout(
                (self.tile_m, self.k_block_size),
                stride=(self.k_block_size + 16, 1),
            )
        else:
            smem_layout_perm = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size), order=(1, 0)
            )

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sLSETemperature: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sMaxValidSplit: cute.struct.Align[cute.struct.MemRange[Int32, self.tile_m], 128]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128
            ]
            sO_perm: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_perm)], 128
            ]

        smem_size = SharedStorage.size_in_bytes()

        # Grid: (ceil(seqlen/tile_m), ceil(dim/k_block), num_head * batch)
        # Head separated from seqlen → enables future TMA (contiguous Sq tiles)
        seqlen = mO_partial.shape[0]
        num_head = mO_partial.shape[3]
        batch_size = (
            mO_partial.shape[4]
            if const_expr(cu_seqlens is None)
            else Int32(cu_seqlens.shape[0] - 1)
        )

        seqlen_divmod = FastDivmodDivisor(seqlen)
        head_divmod = FastDivmodDivisor(num_head)

        grid_dim = (
            cute.ceil_div(seqlen * num_head, self.tile_m),
            cute.ceil_div(self.head_dim, self.k_block_size),
            batch_size,
        )

        self.kernel(
            mO_partial,
            mLSE_partial,
            mO,
            mLSE,
            mLSE_temperature_partial,
            mLSE_temperature,
            cu_seqlens,
            seqused,
            num_splits_dynamic_ptr,
            varlen_batch_idx,
            semaphore_to_reset,
            mSplitCounts,
            mOutputScale,
            qhead_per_kvhead,
            mOq,
            mSFq,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_o,
            smem_layout_perm,
            self.gmem_tiled_copy_O_partial,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_LSE,
            self.s2r_tiled_copy_LSE,
            seqlen_divmod,
            head_divmod,
            self.use_pdl,
            varlen,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def decode_flat_row_idx(
        self,
        idx: Int32,
        head_divmod: FastDivmodDivisor,
    ):
        """Decode flattened tile rows under the H_q-innermost contract."""
        q_idx_local, head_idx = divmod(idx, head_divmod)
        return q_idx_local, head_idx

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSE_temperature_partial: Optional[cute.Tensor],
        mLSE_temperature: Optional[cute.Tensor],
        cu_seqlens: Optional[cute.Tensor],
        seqused: Optional[cute.Tensor],
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        varlen_batch_idx: Optional[cute.Tensor],
        semaphore_to_reset: Optional[cute.Tensor],
        mSplitCounts: Optional[cute.Tensor],
        mOutputScale: Optional[cute.Tensor],
        qhead_per_kvhead: Int32,
        mOq: Optional[cute.Tensor],
        mSFq: Optional[cute.Tensor],
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout | cute.ComposedLayout,
        smem_layout_perm: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        seqlen_divmod: FastDivmodDivisor,
        head_divmod: FastDivmodDivisor,
        use_pdl: cutlass.Constexpr[bool],
        varlen: cutlass.Constexpr[bool],
    ):
        # Thread and block indices
        tidx, _, _ = cute.arch.thread_idx()
        m_block, k_block, maybe_virtual_batch = cute.arch.block_idx()

        batch_idx = (
            varlen_batch_idx[maybe_virtual_batch]
            if const_expr(varlen_batch_idx is not None)
            else maybe_virtual_batch
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sLSE_temperature = storage.sLSETemperature.get_tensor(smem_layout_lse)
        sMaxValidSplit = storage.sMaxValidSplit.get_tensor((self.tile_m,))
        sO = storage.sO.get_tensor(smem_layout_o)
        sO_perm_buf = storage.sO_perm.get_tensor(smem_layout_perm)

        # Handle semaphore reset — wait for dependent grids first
        if const_expr(use_pdl and semaphore_to_reset is not None):
            if (
                tidx == 0
                and m_block == cute.arch.grid_dim()[0] - 1
                and k_block == cute.arch.grid_dim()[1] - 1
                and maybe_virtual_batch == cute.arch.grid_dim()[2] - 1
            ):
                cute.arch.griddepcontrol_wait()
                semaphore_to_reset[0] = 0

        if const_expr(num_splits_dynamic_ptr is not None):
            raise ValueError("K2 combine requires compile-time exact topK")
        num_splits = Int32(self.topk)
        # Handle variable length sequences using SeqlenInfo
        seqlen_info = SeqlenInfo.create(
            batch_idx=batch_idx,
            seqlen_static=mO_partial.shape[0],
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            # Don't need to pass in tile size since we won't use offset_padded
        )
        seqlen, offset = seqlen_info.seqlen, seqlen_info.offset

        num_head = mO_partial.shape[3]
        max_idx = seqlen * num_head
        output_scale = Float32(1.0)
        if const_expr(mOutputScale is not None):
            output_scale = mOutputScale[0]

        if const_expr(not varlen) or m_block * self.tile_m < max_idx:
            # Wait for dependent grids (e.g., the main attention kernel that produces O_partial/LSE_partial)
            if const_expr(use_pdl):
                cute.arch.griddepcontrol_wait()

            # ===============================
            # Step 1: Load LSE_partial from gmem to shared memory
            # ===============================
            # `cLSE` (identity tensor for row/split coord tracking) is reused
            # later in steps 4-5, so it must be defined on both branches.
            cLSE = cute.make_identity_tensor((self.topk, self.tile_m))
            # Reshape mLSE_partial to PackGQA packed layout and delegate the
            # tile load to PackGQAComb.load_LSE. The packed form folds (H_q, Sq)
            # into one compound dim with H_q innermost (stride 1), so thread
            # rows that vary along h_pos produce one-sector coalesced reads.
            # Non-varlen path only — varlen keeps the original inline loop.
            if const_expr(not varlen):
                mLSE_partial_cur = seqlen_info.offset_batch(mLSE_partial, batch_idx, dim=3)
                # mLSE_partial_cur: (H_q, topK, Sq) — after initial transpose
                # [3,0,2,1] on [topK,B,Sq,H_q] and dropping B.
                # Reorder to (H_q, Sq, topK) then group modes 0..1 for packed dim:
                mLSE_partial_reord = cute.make_tensor(
                    mLSE_partial_cur.iterator,
                    cute.select(mLSE_partial_cur.layout, mode=[0, 2, 1]),
                )
                mLSE_partial_packed = cute.group_modes(mLSE_partial_reord, 0, 2)
                # shape ((H_q, Sq), topK) with H_q innermost.
                packgqa = PackGQAComb(
                    m_block_size=self.tile_m,
                    head_dim_padded=0,       # unused for LSE load
                    check_hdim_oob=False,    # unused for LSE load
                    qhead_per_kvhead=1,      # unused; num_heads_divmod is passed explicitly
                )
                packgqa.load_LSE(
                    mLSE_partial_packed,
                    sLSE,
                    self.topk,
                    gmem_tiled_copy_LSE,
                    tidx,
                    m_block,
                    num_splits,
                    seqlen,
                    head_divmod,
                    mSplitCounts,
                    batch_idx,
                    qhead_per_kvhead,
                )
                if const_expr(mLSE_temperature_partial is not None):
                    mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                        mLSE_temperature_partial, batch_idx, dim=3)
                    mLSE_temperature_partial_reord = cute.make_tensor(
                        mLSE_temperature_partial_cur.iterator,
                        cute.select(mLSE_temperature_partial_cur.layout, mode=[0, 2, 1]),
                    )
                    mLSE_temperature_partial_packed = cute.group_modes(
                        mLSE_temperature_partial_reord, 0, 2)
                    packgqa.load_LSE(
                        mLSE_temperature_partial_packed,
                        sLSE_temperature,
                        self.topk,
                        gmem_tiled_copy_LSE,
                        tidx,
                        m_block,
                        num_splits,
                        seqlen,
                        head_divmod,
                        mSplitCounts,
                        batch_idx,
                        qhead_per_kvhead,
                    )
            else:
                # Varlen path keeps the same H_q-innermost flat-row contract:
                # after transpose [1, 0, 2], mLSE_partial_cur is
                # (q_local, split, head).
                # mSplitCounts is the authoritative valid-split count per
                # packed (q_abs, kv_head); masked splits stay at -inf and
                # therefore drop out of the final kernel LSE_out reduction.
                mLSE_partial_cur = seqlen_info.offset_batch(mLSE_partial, batch_idx, dim=3)
                mLSE_partial_copy = cute.tiled_divide(mLSE_partial_cur, (1,))
                gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_slice(tidx)
                tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE)
                tLSEsLSE_temperature = gmem_thr_copy_LSE.partition_D(sLSE_temperature)
                tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE)
                if const_expr(mLSE_temperature_partial is not None):
                    mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                        mLSE_temperature_partial, batch_idx, dim=3)
                    mLSE_temperature_partial_copy = cute.tiled_divide(
                        mLSE_temperature_partial_cur, (1,))

                for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
                    mi = tLSEcLSE[0, 0, m][1]
                    idx = m_block * self.tile_m + mi
                    if idx < max_idx:
                        m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                        row_count = (
                            mSplitCounts[offset + m_idx, head_idx // qhead_per_kvhead]
                            if const_expr(mSplitCounts is not None)
                            else num_splits
                        )
                        mLSE_partial_cur_copy = mLSE_partial_copy[None, m_idx, None, head_idx]
                        if const_expr(mLSE_temperature_partial is not None):
                            mLSE_temperature_partial_cur_copy = (
                                mLSE_temperature_partial_copy[None, m_idx, None, head_idx])
                        for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                            si = tLSEcLSE[0, s, 0][0]
                            if si < num_splits and si < row_count:
                                cute.copy(
                                    gmem_thr_copy_LSE,
                                    mLSE_partial_cur_copy[None, si],
                                    tLSEsLSE[None, s, m],
                                )
                                if const_expr(mLSE_temperature_partial is not None):
                                    cute.copy(
                                        gmem_thr_copy_LSE,
                                        mLSE_temperature_partial_cur_copy[None, si],
                                        tLSEsLSE_temperature[None, s, m],
                                    )
                            else:
                                tLSEsLSE[None, s, m].fill(-Float32.inf)
                                if const_expr(mLSE_temperature_partial is not None):
                                    tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
                    else:
                        for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                            tLSEsLSE[None, s, m].fill(-Float32.inf)
                            if const_expr(mLSE_temperature_partial is not None):
                                tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
            cute.arch.cp_async_commit_group()

            # ===============================
            # Step 2: Load O_partial for pipeline stages
            # ===============================

            gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_slice(tidx)
            cO = cute.make_identity_tensor((self.tile_m, self.k_block_size))
            tOcO = gmem_thr_copy_O_partial.partition_D(cO)
            tOsO_partial = gmem_thr_copy_O_partial.partition_D(sO)
            mO_partial_cur = seqlen_info.offset_batch(mO_partial, batch_idx, dim=4)

            # Precompute per-row values for flattened (q_local, head) tiles.
            num_rows = const_expr(cute.size(tOcO, mode=[1]))
            tOmidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOhidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOSplitCount = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOrOptr = cute.make_rmem_tensor(num_rows, cutlass.Int64)
            for m in cutlass.range(num_rows, unroll_full=True):
                mi = tOcO[0, m, 0][0]  # m coordinate in tile
                idx = m_block * self.tile_m + mi
                if idx >= max_idx:
                    tOhidx[m] = -1
                    tOmidx[m] = 0
                    tOSplitCount[m] = 0
                    tOrOptr[m] = cutlass.Int64(0)
                else:
                    tOmidx[m], tOhidx[m] = self.decode_flat_row_idx(idx, head_divmod)
                    if const_expr(mSplitCounts is None):
                        tOSplitCount[m] = num_splits
                    elif const_expr(cu_seqlens is None):
                        tOSplitCount[m] = mSplitCounts[
                            batch_idx, tOmidx[m], tOhidx[m] // qhead_per_kvhead
                        ]
                    else:
                        tOSplitCount[m] = mSplitCounts[
                            offset + tOmidx[m], tOhidx[m] // qhead_per_kvhead
                        ]
                    tOrOptr[m] = utils.elem_pointer(
                        mO_partial_cur,
                        (tOmidx[m], k_block * self.k_block_size, 0, tOhidx[m]),
                    ).toint()

            tOpO = None
            if const_expr(not self.is_even_k):
                tOpO = cute.make_rmem_tensor(cute.size(tOcO, mode=[2]), Boolean)
                for k in cutlass.range(cute.size(tOpO), unroll_full=True):
                    tOpO[k] = tOcO[0, 0, k][1] < mO_partial.shape[1] - k_block * self.k_block_size
                # if cute.arch.thread_idx()[0] == 0 and k_block == 1: cute.print_tensor(tOpO)

            load_O_partial = partial(
                self.load_O_partial,
                gmem_tiled_copy_O_partial,
                tOrOptr,
                tOsO_partial,
                tOhidx,
                tOSplitCount,
                tOpO,
                tOcO,
                mO_partial_cur.layout,
            )

            # Load first few stages of O_partial
            for stage in cutlass.range(self.stages - 1, unroll_full=True):
                if stage < num_splits:
                    load_O_partial(stage, stage)
                cute.arch.cp_async_commit_group()

            # ===============================
            # Step 3: Load and transpose LSE from smem to registers
            # ===============================

            # Wait for LSE and initial O partial stages to complete
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()
            # if cute.arch.thread_idx()[0] == 0:
            #     # cute.print_tensor(sLSE)
            #     for i in range(64):
            #         cute.printf("sLSE[%d, 0] = %f", i, sLSE[i, 0])
            # cute.arch.sync_threads()

            s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_slice(tidx)
            ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE)
            ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
            cute.copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE)
            if const_expr(mLSE_temperature_partial is not None):
                ts2rsLSE_temperature = s2r_thr_copy_LSE.partition_S(sLSE_temperature)
                ts2rrLSE_temperature = cute.make_rmem_tensor_like(ts2rsLSE_temperature)
                cute.copy(
                    s2r_tiled_copy_LSE,
                    ts2rsLSE_temperature,
                    ts2rrLSE_temperature,
                )

            # ===============================
            # Step 4: Compute final LSE along split dimension
            # ===============================

            final_lse = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Float32)
            ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE)
            # We compute the max valid split for each row to short-circuit the computation later
            max_valid_split = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Int32)
            assert cute.size(ts2rrLSE, mode=[0]) == 1
            # Compute max, scales, and final LSE for each row. Invalid splits
            # have already been filled with -inf, so Step 5 can write the
            # kernel-native LSE_out directly.
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                # Find max LSE value across splits
                threads_per_col = const_expr(self.smem_threads_per_col_lse)
                lse_max = cute.arch.warp_reduction_max(
                    ts2rrLSE[None, None, m]
                    .load()
                    .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                    threads_in_group=threads_per_col,
                )
                # if cute.arch.thread_idx()[0] == 0: cute.printf(lse_max)
                # Find max valid split index
                max_valid_idx = -1
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    if ts2rrLSE[0, s, m] != -Float32.inf:
                        max_valid_idx = ts2rcLSE[0, s, 0][0]  # Get split coordinate
                # if cute.arch.thread_idx()[0] < 32: cute.printf(max_valid_idx)
                max_valid_split[m] = cute.arch.warp_reduction_max(
                    max_valid_idx, threads_in_group=threads_per_col
                )
                # Compute exp scales and sum
                lse_max_cur = (
                    0.0 if lse_max == -Float32.inf else lse_max
                )  # In case all local LSEs are -inf
                LOG2_E = math.log2(math.e)
                lse_sum_cur = 0.0
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    scale = cute.math.exp2(
                        ts2rrLSE[0, s, m] * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True
                    )
                    lse_sum_cur += scale
                    ts2rrLSE[0, s, m] = scale  # Store scale for later use
                lse_sum_cur = cute.arch.warp_reduction_sum(
                    lse_sum_cur, threads_in_group=threads_per_col
                )
                # Normalize scales
                inv_sum = 0.0
                if max_valid_split[m] < 0 or lse_sum_cur == 0.0 or lse_sum_cur != lse_sum_cur:
                    final_lse[m] = -Float32.inf
                else:
                    final_lse[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                    inv_sum = 1.0 / lse_sum_cur
                ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
            # Store the scales exp(lse - lse_logsum) back to smem
            cute.copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE)

            if const_expr(mLSE_temperature_partial is not None):
                final_lse_temperature = cute.make_rmem_tensor(
                    cute.size(ts2rrLSE_temperature, mode=[2]), Float32)
                for m in cutlass.range(cute.size(ts2rrLSE_temperature, mode=[2]), unroll_full=True):
                    threads_per_col = const_expr(self.smem_threads_per_col_lse)
                    lse_temperature_max = cute.arch.warp_reduction_max(
                        ts2rrLSE_temperature[None, None, m]
                        .load()
                        .reduce(
                            cute.ReductionOp.MAX,
                            init_val=-Float32.inf,
                            reduction_profile=0,
                        ),
                        threads_in_group=threads_per_col,
                    )
                    lse_temperature_max_cur = (
                        0.0 if lse_temperature_max == -Float32.inf else lse_temperature_max
                    )
                    LOG2_E = math.log2(math.e)
                    lse_temperature_sum_cur = 0.0
                    for s in cutlass.range(
                        cute.size(ts2rrLSE_temperature, mode=[1]), unroll_full=True):
                        scale = cute.math.exp2(
                            ts2rrLSE_temperature[0, s, m] * LOG2_E
                            - (lse_temperature_max_cur * LOG2_E),
                            fastmath=True,
                        )
                        lse_temperature_sum_cur += scale
                    lse_temperature_sum_cur = cute.arch.warp_reduction_sum(
                        lse_temperature_sum_cur, threads_in_group=threads_per_col
                    )
                    if (
                        max_valid_split[m] < 0
                        or lse_temperature_sum_cur == 0.0
                        or lse_temperature_sum_cur != lse_temperature_sum_cur
                    ):
                        final_lse_temperature[m] = -Float32.inf
                    else:
                        final_lse_temperature[m] = (
                            cute.math.log(lse_temperature_sum_cur, fastmath=True)
                            + lse_temperature_max
                        )

            # Store max valid split to smem
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                if ts2rcLSE[0, 0, m][0] == 0:  # Only thread responsible for s=0 writes
                    mi = ts2rcLSE[0, 0, m][1]
                    if mi < self.tile_m:
                        sMaxValidSplit[mi] = max_valid_split[m]

            # ===============================
            # Step 5: Store final LSE to gmem
            # This writeback is the authoritative LSE_out returned by the
            # public Sparse Attention / Sparse Page Attention interface.
            # ===============================

            if const_expr(mLSE is not None):
                if const_expr(cu_seqlens is None):
                    mLSE_cur = mLSE[None, None, batch_idx]
                else:
                    mLSE_cur = cute.domain_offset((offset, 0), mLSE)
                if const_expr(mLSE_temperature is not None):
                    if const_expr(cu_seqlens is None):
                        mLSE_temperature_cur = mLSE_temperature[None, None, batch_idx]
                    else:
                        mLSE_temperature_cur = cute.domain_offset(
                            (offset, 0), mLSE_temperature)
                if k_block == 0:  # Only first k_block writes LSE when mLSE is provided
                    for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                        if ts2rcLSE[0, 0, m][0] == 0:  # Only thread responsible for s=0 writes
                            mi = ts2rcLSE[0, 0, m][1]
                            idx = m_block * self.tile_m + mi
                            if idx < max_idx:
                                m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                                mLSE_cur[m_idx, head_idx] = final_lse[m]
                                if const_expr(mLSE_temperature is not None):
                                    mLSE_temperature_cur[m_idx, head_idx] = (
                                        final_lse_temperature[m])

            # ===============================
            # Step 6: Read O_partial and accumulate final O
            # ===============================

            cute.arch.sync_threads()

            # Get max valid split for this thread
            thr_max_valid_split = sMaxValidSplit[tOcO[0, 0, 0][0]]
            for m in cutlass.range(1, cute.size(tOcO, mode=[1]), unroll_full=True):
                thr_max_valid_split = max(thr_max_valid_split, sMaxValidSplit[tOcO[0, m, 0][0]])

            tOrO_partial = cute.make_rmem_tensor_like(tOsO_partial[None, None, None, 0])
            tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
            tOrO.fill(0.0)

            stage_load = self.stages - 1
            stage_compute = 0

            # Main accumulation loop
            for s in cutlass.range(thr_max_valid_split + 1, unroll=4):
                # Get scales for this split
                scale = cute.make_rmem_tensor(num_rows, Float32)
                for m in cutlass.range(num_rows, unroll_full=True):
                    scale[m] = sLSE[s, tOcO[0, m, 0][0]]  # Get scale from smem

                # Load next stage if needed
                split_to_load = s + self.stages - 1
                if split_to_load <= thr_max_valid_split:
                    load_O_partial(split_to_load, stage_load)
                cute.arch.cp_async_commit_group()
                stage_load = 0 if stage_load == self.stages - 1 else stage_load + 1

                # Wait for the current stage to be ready
                cute.arch.cp_async_wait_group(self.stages - 1)
                # We don't need __syncthreads() because each thread is just reading its own data from smem
                # Copy from smem to registers
                cute.autovec_copy(tOsO_partial[None, None, None, stage_compute], tOrO_partial)
                stage_compute = 0 if stage_compute == self.stages - 1 else stage_compute + 1

                # Accumulate scaled partial results
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0 and scale[m] > 0.0:
                        tOrO[None, m, None].store(
                            tOrO[None, m, None].load()
                            + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                        )

            # Flush any outstanding async-copy groups before the local Step-7
            # permutation buffer is read on the tail of the kernel.
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ===============================
            # Step 7: Write final O to gmem (fake→real via SMEM)
            # ===============================

            mO_cur = seqlen_info.offset_batch(mO, batch_idx, dim=3)
            if const_expr(cu_seqlens is None):
                mO_cur = mO[None, None, None, batch_idx]
            else:
                mO_cur = cute.domain_offset((offset, 0, 0), mO)
            mO_cur = utils.domain_offset_aligned((0, k_block * self.k_block_size, 0), mO_cur)
            num_vals = const_expr(cute.size(tOcO, mode=[0]))
            if const_expr(not use_pdl):
                # Direct / standalone calls don't participate in the K1->K2
                # dependency chain. Use a simple per-element real-column store
                # path here to keep mixed-shape launches stable.
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0:
                        for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                            if const_expr(self.is_even_k) or tOpO[k]:
                                for v in cutlass.range(num_vals, unroll_full=True):
                                    fake_col = tOcO[v, 0, k][1]
                                    if const_expr(self.use_stg128_fp8_layout):
                                        real_col = stg128_fp8_fake_col_to_real_col(fake_col)
                                    elif const_expr(self.use_stg128_half_layout):
                                        real_col = stg128_half_fake_col_to_real_col(fake_col)
                                    else:
                                        real_col = stg128_fake_col_to_real_col(fake_col)
                                    o_val = tOrO[v, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o_val = o_val * output_scale
                                    mO_cur[tOmidx[m], real_col, tOhidx[m]] = o_val.to(self.dtype)
            else:
                # 7a: fp32 accumulator -> output dtype SMEM with fake→real
                # permutation. The dedicated permutation buffer stays separate
                # from the O_partial pipeline staging buffer.
                sO_perm = sO_perm_buf

                if const_expr(self.dtype in [cutlass.BFloat16, cutlass.Float16]):
                    # O_partial uses a dtype-specific STG.128 fake layout, but
                    # sO_perm is in the final O dtype. For all supported fake
                    # layouts, adjacent fake pairs map to adjacent real columns,
                    # so write the final BF16/F16 O pair as one 32-bit SMEM store.
                    assert num_vals % 2 == 0
                    r2s_o_pair_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        cutlass.Int32,
                        num_bits_per_copy=32,
                    )
                    rO_pair_word = cute.make_rmem_tensor((1,), cutlass.Int32)
                    sO_perm_i32_base = cute.make_ptr(
                        dtype=cutlass.Int32,
                        value=sO_perm.iterator.toint(),
                        mem_space=sO_perm.iterator.memspace,
                        assumed_align=4,
                    )
                    sO_perm_i32_row_stride = Int32((self.k_block_size + 16) // 2)
                    for m in cutlass.range(num_rows, unroll_full=True):
                        row_local = tOcO[0, m, 0][0]
                        if tOhidx[m] >= 0:
                            for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                                for v_pair in cutlass.range(num_vals // 2, unroll_full=True):
                                    v = v_pair * 2
                                    fake_col = tOcO[v, 0, k][1]
                                    if const_expr(self.use_stg128_fp8_layout):
                                        real_col = stg128_fp8_fake_col_to_real_col(fake_col)
                                    elif const_expr(self.use_stg128_half_layout):
                                        real_col = stg128_half_fake_col_to_real_col(fake_col)
                                    else:
                                        real_col = stg128_fake_col_to_real_col(fake_col)
                                    o0 = tOrO[v, m, k]
                                    o1 = tOrO[v + 1, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o0, o1 = cute.arch.mul_packed_f32x2(
                                            (o0, o1),
                                            (output_scale, output_scale),
                                        )
                                    rO_pair_word[0] = utils.cvt_f16x2_f32(o0, o1, self.dtype)
                                    smem_pair_ptr = cute.make_ptr(
                                        dtype=cutlass.Int32,
                                        value=(
                                            sO_perm_i32_base.toint()
                                            + Int64(
                                                row_local * sO_perm_i32_row_stride
                                                + real_col // Int32(2)
                                            )
                                            * Int64(4)
                                        ),
                                        mem_space=sO_perm.iterator.memspace,
                                        assumed_align=4,
                                    )
                                    sO_pair = cute.make_tensor(
                                        smem_pair_ptr,
                                        cute.make_layout((1,), stride=(1,)),
                                    )
                                    cute.copy(r2s_o_pair_atom, rO_pair_word, sO_pair)
                else:
                    # 7a: iterate over ALL val elements in mode[0].
                    # tOcO[v, m, k][1] gives different fake_col for each v.
                    r2s_o_scalar_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.dtype,
                        num_bits_per_copy=self.dtype.width,
                    )
                    rO_scalar = cute.make_rmem_tensor((1,), self.dtype)
                    for m in cutlass.range(num_rows, unroll_full=True):
                        row_local = tOcO[0, m, 0][0]
                        if tOhidx[m] >= 0:
                            for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                                for v in cutlass.range(num_vals, unroll_full=True):
                                    fake_col = tOcO[v, 0, k][1]
                                    if const_expr(self.use_stg128_fp8_layout):
                                        real_col = stg128_fp8_fake_col_to_real_col(fake_col)
                                    elif const_expr(self.use_stg128_half_layout):
                                        real_col = stg128_half_fake_col_to_real_col(fake_col)
                                    else:
                                        real_col = stg128_fake_col_to_real_col(fake_col)
                                    o_val = tOrO[v, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o_val = o_val * output_scale
                                    rO_scalar[0] = o_val.to(self.dtype)
                                    smem_ptr = utils.elem_pointer(sO_perm, (row_local, real_col))
                                    smem_scalar_ptr = cute.make_ptr(
                                        dtype=self.dtype,
                                        value=smem_ptr.toint(),
                                        mem_space=sO_perm.iterator.memspace,
                                        assumed_align=self.dtype.width // 8,
                                    )
                                    sO_scalar = cute.make_tensor(
                                        smem_scalar_ptr,
                                        cute.make_layout((1,), stride=(1,)),
                                    )
                                    cute.copy(r2s_o_scalar_atom, rO_scalar, sO_scalar)

                cute.arch.sync_threads()

                # 7b: SMEM (real order, output dtype) → GMEM
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                tOcO_store = gmem_thr_copy_O.partition_D(cO)
                tOsO_store = gmem_thr_copy_O.partition_D(sO_perm)
                rO = cute.make_rmem_tensor(tOcO_store.shape, self.dtype)
                elems_per_store = const_expr(cute.size(gmem_tiled_copy_O.layout_tv_tiled[1]))
                num_store_rows = const_expr(cute.size(tOcO_store, mode=[1]))
                num_store_vals = const_expr(cute.size(tOcO_store, mode=[0]))
                tOpO_store = None
                if const_expr(not self.is_even_k):
                    tOpO_store = cute.make_rmem_tensor(cute.size(tOcO_store, mode=[2]), Boolean)
                    for k in cutlass.range(cute.size(tOpO_store), unroll_full=True):
                        tOpO_store[k] = (
                            tOcO_store[0, 0, k][1]
                            < mO_partial.shape[1] - k_block * self.k_block_size
                        )

                # Read output dtype from SMEM (now in real column order).
                for m in cutlass.range(num_store_rows, unroll_full=True):
                    for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                        if const_expr(self.is_even_k) or tOpO_store[k]:
                            cute.autovec_copy(tOsO_store[None, m, k], rO[None, m, k])

                if const_expr(self.write_bf16):
                    # Write bf16 to GMEM using gmem_tiled_copy_O (same as original FA Step 7)
                    for m in cutlass.range(num_store_rows, unroll_full=True):
                        row_local = tOcO_store[0, m, 0][0]
                        idx = m_block * self.tile_m + row_local
                        if idx < max_idx:
                            m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                            mO_cur_copy = cute.tiled_divide(
                                mO_cur[m_idx, None, head_idx], (elems_per_store,)
                            )
                            for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                                k_idx = tOcO_store[0, 0, k][1] // elems_per_store
                                if const_expr(self.is_even_k) or tOpO_store[k]:
                                    cute.copy(gmem_thr_copy_O, rO[None, m, k], mO_cur_copy[None, k_idx])

                if const_expr(mOq is not None):
                    self.quant_rows_mxfp8(
                        rO, tOcO_store, mOq, mSFq, m_block, max_idx, offset, head_divmod, num_store_rows
                    )

    @cute.jit
    def quant_rows_mxfp8(
        self,
        rO: cute.Tensor,
        tOcO_store: cute.Tensor,
        mOq: cute.Tensor,
        mSFq: cute.Tensor,
        m_block: Int32,
        max_idx: Int32,
        offset: Int32,
        head_divmod: FastDivmodDivisor,
        num_store_rows: cutlass.Constexpr[int],
    ):
        """The tile's final O rows as MXFP8 for the o_proj GEMM, from the bf16 registers 7b read out of sO_perm (one
        SMEM pass). 7b's copy gives each thread 8 consecutive real columns of a row and 16 consecutive lanes per row
        (k_block 128), the stock quantize_mxfp8_rows mapping: 4 lanes per K32 block, so amax / scale / E4M3 are
        bitwise mxfp8_quantize of the bf16 output. mOq: uint8 [total_q, nheads * head_dim]; mSFq: uint8 swizzled
        128x4 scales (padding rows zeroed by the caller)."""
        assert cute.size(tOcO_store, mode=[0]) == 8 and cute.size(tOcO_store, mode=[2]) == 1
        num_head = Int32(mOq.shape[1] // self.head_dim)
        col0 = tOcO_store[0, 0, 0][1]
        lane_block = col0 // 8
        for m in cutlass.range_constexpr(num_store_rows):
            row_local = tOcO_store[0, m, 0][0]
            idx = m_block * self.tile_m + row_local
            vals = cute.make_rmem_tensor((8,), Float32)
            for i in cutlass.range_constexpr(8):
                vals[i] = rO[i, m, 0].to(Float32)
            vmax = mxfp8_epi.fabs_f32(vals[0])
            for i in cutlass.range_constexpr(1, 8):
                vmax = mxfp8_epi.fmax_f32(vmax, mxfp8_epi.fabs_f32(vals[i]))
            # amax over the 4 lanes of a K32 block (consecutive lanes of one warp, one row).
            vmax = mxfp8_epi.fmax_f32(vmax, cute.arch.shuffle_sync_bfly(vmax, offset=1))
            vmax = mxfp8_epi.fmax_f32(vmax, cute.arch.shuffle_sync_bfly(vmax, offset=2))
            sf_bits = mxfp8_epi.ue8m0_round_up(vmax * mxfp8_epi.rcp_approx_ftz(Float32(448.0)))
            out_scale = Float32(0.0)
            if vmax != Float32(0.0):
                out_scale = mxfp8_epi.rcp_approx_ftz(mxfp8_epi.ue8m0_to_f32(sf_bits))
            if idx < max_idx:
                m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                tok = offset + m_idx
                q0 = utils.cvt_fp8x4_e4m3_f32(
                    vals[0] * out_scale, vals[1] * out_scale, vals[2] * out_scale, vals[3] * out_scale
                )
                q1 = utils.cvt_fp8x4_e4m3_f32(
                    vals[4] * out_scale, vals[5] * out_scale, vals[6] * out_scale, vals[7] * out_scale
                )
                qp = cute.make_ptr(
                    dtype=cutlass.Int32,
                    value=mOq.iterator.toint()
                    + Int64(tok) * Int64(mOq.shape[1])
                    + Int64(head_idx * self.head_dim + col0),
                    mem_space=mOq.iterator.memspace,
                    assumed_align=8,
                )
                qt = cute.make_tensor(qp, cute.make_layout((2,), stride=(1,)))
                qt[0] = q0
                qt[1] = q1
                if lane_block % 4 == 0:
                    k_idx = head_idx * (self.head_dim // 32) + lane_block // 4
                    sf_off = mxfp8_epi.sf_offset_128x4(tok, k_idx, num_head * (self.head_dim // 128))
                    sp = cute.make_ptr(
                        dtype=cutlass.Uint8,
                        value=mSFq.iterator.toint() + Int64(sf_off),
                        mem_space=mSFq.iterator.memspace,
                        assumed_align=1,
                    )
                    cute.make_tensor(sp, cute.make_layout((1,), stride=(1,)))[0] = cutlass.Uint8(sf_bits)

    @cute.jit
    def load_O_partial(
        self,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        tOrOptr: cute.Tensor,
        tOsO_partial: cute.Tensor,
        tOhidx: cute.Tensor,
        tOSplitCount: cute.Tensor,
        tOpO: Optional[cute.Tensor],
        tOcO: cute.Tensor,
        mO_cur_partial_layout: cute.Layout,
        split: Int32,
        stage: Int32,
    ) -> None:
        elems_per_load = const_expr(cute.size(gmem_tiled_copy_O_partial.layout_tv_tiled[1]))
        tOsO_partial_cur = tOsO_partial[None, None, None, stage]
        for m in cutlass.range(cute.size(tOcO, [1]), unroll_full=True):
            if tOhidx[m] >= 0:
                o_gmem_ptr = cute.make_ptr(
                    tOsO_partial.element_type, tOrOptr[m], cute.AddressSpace.gmem, assumed_align=16
                )
                mO_partial_cur = cute.make_tensor(
                    o_gmem_ptr, cute.slice_(mO_cur_partial_layout, (0, None, None, 0))
                )
                mO_partial_cur_copy = cute.tiled_divide(mO_partial_cur, (elems_per_load,))
                for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                    k_idx = tOcO[0, 0, k][1] // elems_per_load
                    if split < tOSplitCount[m] and (const_expr(tOpO is None) or tOpO[k]):
                        cute.copy(
                            gmem_tiled_copy_O_partial,
                            mO_partial_cur_copy[None, k_idx, split],
                            tOsO_partial_cur[None, m, k],
                        )
                    else:
                        tOsO_partial_cur[None, m, k].fill(0)


def _get_cutlass_dtype(torch_dtype: torch.dtype):
    if torch_dtype not in torch2cute_dtype_map:
        raise TypeError(f"Unsupported dtype: {torch_dtype}")
    return torch2cute_dtype_map[torch_dtype]


_combine_compile_cache = {}


def _get_cpasync_smem_layout_atom(dtype: Type[cutlass.Numeric], k_dim: int) -> cute.ComposedLayout:
    dtype_byte = const_expr(dtype.width // 8)
    bytes_per_row = const_expr(k_dim * dtype_byte)
    smem_k_block_size = (
        const_expr(
            128
            if bytes_per_row % 128 == 0
            else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
        )
        // dtype_byte
    )
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if const_expr(k_dim % 32 == 0) else 16, smem_k_block_size),
            order=(1, 0),
        ),
    )


# -----------------------------------------------------------------------------
# TMA-ring combine (transplanted from MSA msa_dev 25c1447, combine.py:1536-2347; dev math unchanged): BF16/FP16
# partials, varlen, no output_scale / seqused. combine() routes eligible calls here (TRTLLM_MSA_DISABLE_COMBINE_DEVCLASS=1
# restores SparseAttentionForwardCombine for every call). Plus our MXFP8 o_proj epilogue and write_bf16 gate.
# -----------------------------------------------------------------------------

# One tile is 64 flat (query, head) rows of 256 B, a 16 KB chunk per split; four slots keep
# 64 KB in flight per CTA and leave room for three CTAs per SM.
TILE_M = 64
NUM_THREADS = 256
STAGES = 4
# Use a two-slot ring when the reads are not steered on SM107. Unsteered partials are
# interleaved across the memory locality domains, so half of each CTA's reads cross the
# inter-domain fabric, and on SM107 that fabric caps interleaved reads well below the local HBM
# rate. Keep in flight only what the fabric drains in one round trip: a deeper ring adds
# queueing that also delays the local half of the reads. Steered reads are domain-local and
# B200/B300 interleaved reads are HBM-bound, so both keep the four-slot ring.
STAGES_INTERLEAVED = 2
# The partial buffer is addressed as (dim, flat row, split); the TMA descriptor lowering is
# relied on for split strides below this many elements.
MAX_SPLIT_SLAB_ELEMENTS = 2**31



class SparseAttentionForwardCombineTma:
    """Merge the split partials of every (query, head) row through a TMA ring.

    The sparse forward kernels write one partial output and one partial log-sum-exp per
    (query, head, selected page) into ``[num_splits, total_q, heads, dim]`` and
    ``[num_splits, total_q, heads]`` buffers. This kernel merges the splits of every (query, head)
    row into the final output and LSE. Rows are independent, so the concatenated varlen rows form
    one flat scheduling domain: a CTA owns ``tile_m`` consecutive flat rows, which is one contiguous
    chunk of each split's slab, and streams the chunks of its splits through a ring of ``stages``
    shared-memory slots with one bulk-tensor copy per (tile, split).

    The partial output columns are in the STG.128 "fake" order the forward epilogue stores (a 4x4
    transpose of 32-bit pairs inside each 32-column group); the epilogue here undoes it with two
    XOR-shuffle stages before one coalesced 128-bit store per row.

    The split slabs of the partial buffer may be padded apart. On a GPU with several memory
    locality domains (Rubin uGPU) the caller stripes the buffer so page ``p`` of every slab lives in
    domain ``p % num_domains``, and hands the kernel the domain of every SM: a CTA then claims a tile
    whose pages live in the domain of its own SM, from per-domain counters, and takes a tile of
    another domain only once its own are gone.

    Transplant (LongAttn, from MSA msa_dev 25c1447): optionally also writes the final O as MXFP8 for the o_proj GEMM
    (``mOq`` / ``mSFq``, see ``quant_rows_mxfp8``), and ``write_bf16=False`` skips the BF16 O store. The merge math
    is dev's, unchanged.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        topk: int,
        tile_m: int = TILE_M,
        num_threads: int = NUM_THREADS,
        stages: int = STAGES,
        use_pdl: bool = False,
        min_blocks_per_mp: int = 0,
        num_domains: int = 0,
        lse_split_aligned: bool = True,
        write_bf16: bool = True,
    ):
        """Forward combine kernel for split attention computation.

        :param dtype: output data type (BF16 or F16)
        :param dtype_partial: partial output data type (BF16 or F16)
        :param head_dim: head dimension; a tile row is one head's ``head_dim`` elements
        :param topk: exact number of split partials per row
        :param tile_m: flat (query, head) rows per CTA
        :param num_threads: threads per CTA
        :param stages: ring slots of ``tile_m`` partial rows each
        :param use_pdl: wait on the producer grid before reading its outputs
        :param min_blocks_per_mp: CTAs per SM ptxas must leave registers for (0 lets it choose)
        :param lse_split_aligned: the LSE partial slabs start 16-byte aligned (the flat row count is
            a multiple of 4), so one 2-D bulk-tensor copy loads a tile's LSEs of every split;
            otherwise the threads bring the tile in with 4-byte cp.async copies, since TMA needs
            a 16-byte aligned split stride and box start
        :param num_domains: memory locality domains the tiles are claimed by (0: tile = CTA index)
        :param write_bf16: store the BF16/F16 O; False only with the MXFP8 output and no other reader of O
        """
        if dtype not in (cutlass.BFloat16, cutlass.Float16):
            raise TypeError("the combine writes BF16 or F16 outputs")
        if dtype_partial not in (cutlass.BFloat16, cutlass.Float16):
            raise TypeError("the combine reads BF16 or F16 partials")
        if not 1 <= topk <= 256:
            raise ValueError(f"topk must be in [1, 256], got {topk}")
        if num_threads % 32 != 0 or stages < 1:
            raise ValueError("num_threads must be a multiple of 32 and stages at least 1")
        if num_domains < 0 or num_domains == 1:
            raise ValueError("num_domains must be 0 (no steering) or at least 2")
        copy_bits = 128
        self.elems_per_copy = copy_bits // dtype_partial.width
        if head_dim % self.elems_per_copy != 0:
            raise ValueError("head_dim must be a multiple of one 16-byte copy")
        # Threads that share one tile row, 16 bytes of it each.
        self.threads_per_row = head_dim // self.elems_per_copy
        if num_threads % self.threads_per_row != 0:
            raise ValueError("num_threads must be a multiple of the threads per row")
        self.rows_per_pass = num_threads // self.threads_per_row
        if tile_m % self.rows_per_pass != 0:
            raise ValueError("tile_m must be a multiple of the rows one pass covers")
        # Threads that transpose one row of the LSE tile.
        if num_threads % tile_m != 0:
            raise ValueError("num_threads must be a multiple of tile_m")
        self.lse_threads_per_row = num_threads // tile_m
        if 32 % self.lse_threads_per_row != 0 or topk % self.lse_threads_per_row != 0:
            raise ValueError("the threads per LSE row must divide the warp and the split count")
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.topk = topk
        self.tile_m = tile_m
        self.num_threads = num_threads
        self.num_warps = num_threads // 32
        self.stages = stages
        self.use_pdl = use_pdl
        self.min_blocks_per_mp = min_blocks_per_mp
        self.num_domains = num_domains
        self.lse_split_aligned = lse_split_aligned
        self.write_bf16 = write_bf16

    def _setup_attributes(self):
        copy_bits = 128
        # One thread map serves the reads of a ring slot and the output stores: BF16 and F16
        # have the same width, so a thread's 16 bytes cover the same columns in both.
        thr_layout = cute.make_ordered_layout(
            (self.rows_per_pass, self.threads_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, self.elems_per_copy))
        self.tiled_copy_partial = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dtype_partial, num_bits_per_copy=copy_bits
            ),
            thr_layout,
            val_layout,
        )
        self.tiled_copy_out = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=copy_bits),
            thr_layout,
            val_layout,
        )
        # LSE tile (split, row): consecutive lanes transpose the splits of one row.
        s2r_thr_layout = cute.make_ordered_layout(
            (self.lse_threads_per_row, self.tile_m), order=(0, 1)
        )
        self.s2r_tiled_copy_lse = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_thr_layout,
            cute.make_layout(1),
        )
        # Unaligned LSE slabs: every thread brings topk / lse_threads_per_row splits of one row in
        # with 4-byte cp.async copies; consecutive threads take consecutive rows of a split.
        g2s_thr_layout = cute.make_ordered_layout(
            (self.tile_m, self.lse_threads_per_row), order=(0, 1)
        )
        self.g2s_tiled_copy_lse = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cpasync.CopyG2SOp(), Float32, num_bits_per_copy=Float32.width),
            g2s_thr_layout,
            cute.make_layout((1, self.topk // self.lse_threads_per_row)),
        )
        # The bulk copy writes the tile as a (row, split) box over the same bytes; the trailing
        # mode is the single stage the TMA partition idiom expects.
        self.smem_layout_lse = cute.make_ordered_layout((self.topk, self.tile_m), order=(1, 0))
        self.smem_layout_lse_tma = cute.make_ordered_layout(
            (self.tile_m, self.topk, 1), order=(0, 1, 2)
        )
        # Ring slot: row-major (row, dim); 16 threads read 16 B each of one 256 B row, which is
        # conflict-free without a swizzle. The TMA view is its (dim, row) transpose.
        self.smem_layout_o = cute.make_ordered_layout(
            (self.tile_m, self.head_dim, self.stages), order=(1, 0, 2)
        )
        self.smem_layout_o_tma = cute.make_ordered_layout(
            (self.head_dim, self.tile_m, self.stages), order=(0, 1, 2)
        )

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor] = None,
        mLSE_temperature_partial: Optional[cute.Tensor] = None,
        mLSE_temperature: Optional[cute.Tensor] = None,
        mSplitCounts: Optional[cute.Tensor] = None,
        mOutputScale: Optional[cute.Tensor] = None,
        qhead_per_kvhead: Int32 = Int32(1),
        mSmDomains: Optional[cute.Tensor] = None,
        mTileCounters: Optional[cute.Tensor] = None,
        tiles_per_page_log2: Int32 = Int32(0),
        mOq: Optional[cute.Tensor] = None,
        mSFq: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        if const_expr(mO_partial.element_type != self.dtype_partial):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(mO.element_type != self.dtype):
            raise TypeError("O tensor must match dtype")
        if const_expr(mLSE_partial.element_type != Float32):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type != Float32):
            raise TypeError("LSE tensor must be Float32")
        if const_expr((mLSE_temperature_partial is None) != (mLSE_temperature is None)):
            raise ValueError(
                "temperature LSE partial and output tensors must either both be provided or both be None"
            )
        if const_expr(
            mLSE_temperature_partial is not None
            and (
                mLSE_temperature_partial.element_type != Float32
                or mLSE_temperature.element_type != Float32
            )
        ):
            raise TypeError("temperature LSE tensors must be Float32")
        if const_expr(len(mO_partial.shape) != 4):
            raise ValueError("O partial tensor must be (num_splits, total_q, nheads, headdim)")
        if const_expr(len(mLSE_partial.shape) != 3):
            raise ValueError("LSE partial tensor must be (num_splits, total_q, nheads)")
        if const_expr(len(mO.shape) != 3):
            raise ValueError("O tensor must be (total_q, nheads, headdim)")
        if const_expr(mLSE is not None and len(mLSE.shape) != 2):
            raise ValueError("LSE tensor must be (total_q, nheads)")
        if const_expr(
            mLSE_temperature_partial is not None
            and (len(mLSE_temperature_partial.shape) != 3 or len(mLSE_temperature.shape) != 2)
        ):
            raise ValueError("temperature LSE tensors must match the LSE partial and output shapes")
        if const_expr(mSplitCounts is not None):
            if const_expr(mSplitCounts.element_type != Int32):
                raise TypeError("split_counts tensor must be Int32")
            if const_expr(len(mSplitCounts.shape) != 2):
                raise ValueError("split_counts tensor must be (total_q, nheads_kv)")
        if const_expr(mOutputScale is not None and mOutputScale.element_type != Float32):
            raise TypeError("output_scale tensor must be Float32")
        if const_expr((mSmDomains is None) != (mTileCounters is None)):
            raise ValueError("sm_domains and tile_counters must either both be provided or both be None")
        if const_expr((mTileCounters is not None) != (self.num_domains > 0)):
            raise ValueError("tile_counters are required exactly when num_domains is set")
        if const_expr(
            mTileCounters is not None
            and (mSmDomains.element_type != Int32 or mTileCounters.element_type != Int32)
        ):
            raise TypeError("sm_domains and tile_counters must be Int32")
        if const_expr((mOq is None) != (mSFq is None)):
            raise ValueError("the MXFP8 output and its scales must either both be provided or both be None")
        if const_expr(mOq is None and not self.write_bf16):
            raise ValueError("write_bf16=False requires the MXFP8 output")

        mO_partial, mO = [assume_tensor_aligned(t) for t in (mO_partial, mO)]
        num_head = mO_partial.shape[2]
        total_flat = mO_partial.shape[1] * num_head
        self._setup_attributes()

        # (num_splits, total_q, nheads, headdim) with contiguous slabs as (headdim, flat row,
        # split): a (headdim, tile_m) box is one tile of one split. The slabs keep their own
        # stride, which may be padded.
        mO_partial_tma = cute.make_tensor(
            mO_partial.iterator,
            cute.make_layout(
                (self.head_dim, total_flat, self.topk),
                stride=(1, self.head_dim, mO_partial.stride[0]),
            ),
        )
        tma_atom_O_partial, mO_partial_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mO_partial_tma,
            cute.select(self.smem_layout_o_tma, mode=[0, 1]),
            (self.head_dim, self.tile_m),
        )
        num_m_blocks = cute.ceil_div(total_flat, self.tile_m)

        def make_lse_tma(mLSE_src: cute.Tensor):
            """The contiguous (num_splits, total_q, nheads) LSE partials as (flat row, split). With
            aligned slabs, a TMA atom whose (tile_m, num_splits) box is a CTA's whole LSE tile and
            its coordinate tensor; otherwise no atom and the plain view, which the threads copy
            with cp.async (TMA rejects a split stride or a box start that is not 16-byte aligned)."""
            mLSE_2d = cute.make_tensor(
                mLSE_src.iterator,
                cute.make_layout((total_flat, self.topk), stride=(1, total_flat)),
            )
            if const_expr(not self.lse_split_aligned):
                return None, mLSE_2d
            return cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mLSE_2d,
                cute.select(self.smem_layout_lse_tma, mode=[0, 1]),
                (self.tile_m, self.topk),
            )

        tma_atom_LSE, mLSE_partial_tma = make_lse_tma(mLSE_partial)
        tma_atom_LSE_temperature = None
        mLSE_temperature_partial_tma = None
        if const_expr(mLSE_temperature_partial is not None):
            tma_atom_LSE_temperature, mLSE_temperature_partial_tma = make_lse_tma(
                mLSE_temperature_partial
            )

        lse_tile_elems = cute.cosize(self.smem_layout_lse)

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[cute.struct.MemRange[Float32, lse_tile_elems], 128]
            sLSETemperature: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, lse_tile_elems if mLSE_temperature_partial is not None else 1
                ],
                128,
            ]
            sBlockMaxValidSplit: cute.struct.Align[
                cute.struct.MemRange[Int32, self.num_warps], 128
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128
            ]
            sRingMbar: cute.struct.Align[cute.struct.MemRange[Int64, self.stages], 8]
            sLseMbar: cute.struct.Align[cute.struct.MemRange[Int64, 1], 8]
            sTile: cute.struct.Align[cute.struct.MemRange[Int32, 1], 8]

        grid_dim = (num_m_blocks, 1, 1)
        self.kernel(
            mO_partial_tma,
            mLSE_partial_tma,
            mLSE_temperature_partial_tma,
            mO,
            mLSE,
            mLSE_temperature,
            mSplitCounts,
            mOutputScale,
            qhead_per_kvhead,
            mSmDomains,
            mTileCounters,
            tiles_per_page_log2,
            mOq,
            mSFq,
            tma_atom_O_partial,
            tma_atom_LSE,
            tma_atom_LSE_temperature,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_lse_tma,
            self.smem_layout_o,
            self.smem_layout_o_tma,
            self.tiled_copy_partial,
            self.tiled_copy_out,
            self.s2r_tiled_copy_lse,
            self.g2s_tiled_copy_lse,
            FastDivmodDivisor(num_head),
            self.use_pdl,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial_tma: cute.Tensor,
        mLSE_partial_tma: cute.Tensor,
        mLSE_temperature_partial_tma: Optional[cute.Tensor],
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSE_temperature: Optional[cute.Tensor],
        mSplitCounts: Optional[cute.Tensor],
        mOutputScale: Optional[cute.Tensor],
        qhead_per_kvhead: Int32,
        mSmDomains: Optional[cute.Tensor],
        mTileCounters: Optional[cute.Tensor],
        tiles_per_page_log2: Int32,
        mOq: Optional[cute.Tensor],
        mSFq: Optional[cute.Tensor],
        tma_atom_O_partial: cute.CopyAtom,
        tma_atom_LSE: Optional[cute.CopyAtom],
        tma_atom_LSE_temperature: Optional[cute.CopyAtom],
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout,
        smem_layout_lse_tma: cute.Layout,
        smem_layout_o: cute.Layout,
        smem_layout_o_tma: cute.Layout,
        tiled_copy_partial: cute.TiledCopy,
        tiled_copy_out: cute.TiledCopy,
        s2r_tiled_copy_lse: cute.TiledCopy,
        g2s_tiled_copy_lse: cute.TiledCopy,
        head_divmod: FastDivmodDivisor,
        use_pdl: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        m_block, _, _ = cute.arch.block_idx()
        total_tiles, _, _ = cute.arch.grid_dim()
        total_flat = mO.shape[0] * mO.shape[1]
        has_temperature = const_expr(mLSE_temperature_partial_tma is not None)
        steer_tiles = const_expr(mTileCounters is not None)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sLSE_tma = storage.sLSE.get_tensor(smem_layout_lse_tma)
        sLSE_temperature = None
        sLSE_temperature_tma = None
        if const_expr(has_temperature):
            sLSE_temperature = storage.sLSETemperature.get_tensor(smem_layout_lse)
            sLSE_temperature_tma = storage.sLSETemperature.get_tensor(smem_layout_lse_tma)
        sBlockMaxValidSplit = storage.sBlockMaxValidSplit.get_tensor((self.num_warps,))
        sO = storage.sO.get_tensor(smem_layout_o)
        sO_tma = storage.sO.get_tensor(smem_layout_o_tma)
        ring_mbar = storage.sRingMbar.data_ptr()
        lse_mbar = storage.sLseMbar.data_ptr()
        sTile = storage.sTile.get_tensor((1,))

        # Barriers are CTA-local state, and the tile claim depends on nothing the producer grid
        # writes (it zeroed the counters at its start): both happen while that grid is still
        # draining.
        if tidx == 0:
            for stage in cutlass.range(self.stages, unroll_full=True):
                cute.arch.mbarrier_init(ring_mbar + stage, 1)
            cute.arch.mbarrier_init(lse_mbar, 1 if self.lse_split_aligned else self.num_threads)
            cute.arch.mbarrier_init_fence()
            if const_expr(steer_tiles):
                sTile[0] = self.claim_tile(
                    mSmDomains, mTileCounters, total_tiles, tiles_per_page_log2
                )
        cute.arch.sync_threads()
        if const_expr(steer_tiles):
            m_block = sTile[0]
        tile_begin = m_block * self.tile_m
        if const_expr(use_pdl):
            cute.arch.griddepcontrol_wait()

        # This CTA's tile of every split, and its LSE tile(s).
        gO_partial = cute.local_tile(
            mO_partial_tma, (self.head_dim, self.tile_m), (0, None, None)
        )
        tOsO_tma, tOgO_tma = cpasync.tma_partition(
            tma_atom_O_partial,
            0,
            cute.make_layout(1),
            cute.group_modes(sO_tma, 0, 2),
            cute.group_modes(gO_partial, 0, 2),
        )
        o_tma_bytes = const_expr(
            cute.size_in_bytes(self.dtype_partial, cute.select(smem_layout_o_tma, mode=[0, 1]))
        )
        # The LSE tile: with aligned slabs one (tile_m, num_splits) TMA box per tile; otherwise
        # 4-byte cp.async copies by every thread into the same smem bytes, completing the same
        # barrier with one arrive per thread.
        if const_expr(self.lse_split_aligned):
            gLSE = cute.local_tile(mLSE_partial_tma, (self.tile_m, self.topk), (None, 0))
            tLSEsLSE_tma, tLSEgLSE_tma = cpasync.tma_partition(
                tma_atom_LSE,
                0,
                cute.make_layout(1),
                cute.group_modes(sLSE_tma, 0, 2),
                cute.group_modes(gLSE, 0, 2),
            )
            if const_expr(has_temperature):
                gLSE_temperature = cute.local_tile(
                    mLSE_temperature_partial_tma, (self.tile_m, self.topk), (None, 0)
                )
                tLSEsLSE_temperature_tma, tLSEgLSE_temperature_tma = cpasync.tma_partition(
                    tma_atom_LSE_temperature,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sLSE_temperature_tma, 0, 2),
                    cute.group_modes(gLSE_temperature, 0, 2),
                )
            lse_tma_bytes = const_expr(
                cute.size_in_bytes(Float32, cute.select(smem_layout_lse_tma, mode=[0, 1]))
            )
        else:
            g2s_thr_copy_lse = g2s_tiled_copy_lse.get_slice(tidx)
            row_in_tile = g2s_thr_copy_lse.partition_S(
                cute.make_identity_tensor((self.tile_m, self.topk))
            )[0, 0, 0][0]
            if tile_begin + row_in_tile < total_flat:
                gLSE_tile = cute.local_tile(mLSE_partial_tma, (self.tile_m, self.topk), (m_block, 0))
                cute.copy(
                    g2s_tiled_copy_lse,
                    g2s_thr_copy_lse.partition_S(gLSE_tile),
                    g2s_thr_copy_lse.partition_D(sLSE_tma[None, None, 0]),
                )
                if const_expr(has_temperature):
                    gLSE_temperature_tile = cute.local_tile(
                        mLSE_temperature_partial_tma, (self.tile_m, self.topk), (m_block, 0)
                    )
                    cute.copy(
                        g2s_tiled_copy_lse,
                        g2s_thr_copy_lse.partition_S(gLSE_temperature_tile),
                        g2s_thr_copy_lse.partition_D(sLSE_temperature_tma[None, None, 0]),
                    )
            cute.arch.cp_async_mbarrier_arrive_noinc(lse_mbar)
        primed_stages = const_expr(min(self.stages, self.topk))
        # One warp puts the LSE tile and the first ring slots in flight; they land while the
        # CTA decodes its rows.
        if warp_idx == 0:
            if const_expr(self.lse_split_aligned):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        lse_mbar, lse_tma_bytes * (2 if has_temperature else 1)
                    )
                cute.copy(
                    tma_atom_LSE,
                    tLSEgLSE_tma[None, m_block],
                    tLSEsLSE_tma[None, 0],
                    tma_bar_ptr=lse_mbar,
                )
                if const_expr(has_temperature):
                    cute.copy(
                        tma_atom_LSE_temperature,
                        tLSEgLSE_temperature_tma[None, m_block],
                        tLSEsLSE_temperature_tma[None, 0],
                        tma_bar_ptr=lse_mbar,
                    )
            for stage in cutlass.range(primed_stages, unroll_full=True):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(ring_mbar + stage, o_tma_bytes)
                cute.copy(
                    tma_atom_O_partial,
                    tOgO_tma[None, m_block, Int32(stage)],
                    tOsO_tma[None, stage],
                    tma_bar_ptr=ring_mbar + stage,
                )

        cO = cute.make_identity_tensor((self.tile_m, self.head_dim))
        thr_copy_partial = tiled_copy_partial.get_slice(tidx)
        tOcO = thr_copy_partial.partition_D(cO)
        tOsO = thr_copy_partial.partition_D(sO)
        num_rows = const_expr(cute.size(tOcO, mode=[1]))

        # The LSE tile holds stale values past a row's valid split count and past the last row:
        # fetch each row's count before the wait and mask after the transpose.
        cLSE = cute.make_identity_tensor((self.topk, self.tile_m))
        s2r_thr_copy_lse = s2r_tiled_copy_lse.get_slice(tidx)
        ts2rsLSE = s2r_thr_copy_lse.partition_S(sLSE)
        ts2rcLSE = s2r_thr_copy_lse.partition_S(cLSE)
        ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
        lse_rows = const_expr(cute.size(ts2rrLSE, mode=[2]))
        lse_splits = const_expr(cute.size(ts2rrLSE, mode=[1]))
        lse_row_count = cute.make_rmem_tensor(lse_rows, Int32)
        for m in cutlass.range(lse_rows, unroll_full=True):
            row_idx = tile_begin + ts2rcLSE[0, 0, m][1]
            lse_row_count[m] = Int32(0)
            if row_idx < total_flat:
                if const_expr(mSplitCounts is not None):
                    row_q, row_h = divmod(row_idx, head_divmod)
                    lse_row_count[m] = mSplitCounts[row_q, row_h // qhead_per_kvhead]
                else:
                    lse_row_count[m] = Int32(self.topk)
        cute.arch.mbarrier_wait(lse_mbar, 0)
        cute.copy(s2r_tiled_copy_lse, ts2rsLSE, ts2rrLSE)
        if const_expr(has_temperature):
            ts2rsLSE_temperature = s2r_thr_copy_lse.partition_S(sLSE_temperature)
            ts2rrLSE_temperature = cute.make_rmem_tensor_like(ts2rsLSE_temperature)
            cute.copy(s2r_tiled_copy_lse, ts2rsLSE_temperature, ts2rrLSE_temperature)
        for m in cutlass.range(lse_rows, unroll_full=True):
            for sp in cutlass.range(lse_splits, unroll_full=True):
                if ts2rcLSE[0, sp, 0][0] >= lse_row_count[m]:
                    ts2rrLSE[0, sp, m] = -Float32.inf
                    if const_expr(has_temperature):
                        ts2rrLSE_temperature[0, sp, m] = -Float32.inf

        # Per row: the max LSE over the valid splits, the split weights exp(lse - max) normalized
        # by their sum (written back over the LSE tile), the final LSE, and the last valid split.
        final_lse = cute.make_rmem_tensor(lse_rows, Float32)
        max_valid_split = cute.make_rmem_tensor(lse_rows, Int32)
        threads_per_row = const_expr(self.lse_threads_per_row)
        LOG2_E = math.log2(math.e)
        for m in cutlass.range(lse_rows, unroll_full=True):
            lse_max = cute.arch.warp_reduction_max(
                ts2rrLSE[None, None, m]
                .load()
                .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                threads_in_group=threads_per_row,
            )
            max_valid_idx = -1
            for sp in cutlass.range(lse_splits, unroll_full=True):
                if ts2rrLSE[0, sp, m] != -Float32.inf:
                    max_valid_idx = ts2rcLSE[0, sp, 0][0]
            max_valid_split[m] = cute.arch.warp_reduction_max(
                max_valid_idx, threads_in_group=threads_per_row
            )
            # A row without valid splits has max -inf; its weights are exp2(-inf) = 0.
            lse_max_cur = 0.0 if lse_max == -Float32.inf else lse_max
            lse_sum_cur = 0.0
            for sp in cutlass.range(lse_splits, unroll_full=True):
                scale = cute.math.exp2(
                    ts2rrLSE[0, sp, m] * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True
                )
                lse_sum_cur += scale
                ts2rrLSE[0, sp, m] = scale
            lse_sum_cur = cute.arch.warp_reduction_sum(
                lse_sum_cur, threads_in_group=threads_per_row
            )
            inv_sum = 0.0
            if max_valid_split[m] < 0 or lse_sum_cur == 0.0 or lse_sum_cur != lse_sum_cur:
                final_lse[m] = -Float32.inf
            else:
                final_lse[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                inv_sum = 1.0 / lse_sum_cur
            ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
        cute.copy(s2r_tiled_copy_lse, ts2rrLSE, ts2rsLSE)

        if const_expr(has_temperature):
            final_lse_temperature = cute.make_rmem_tensor(lse_rows, Float32)
            for m in cutlass.range(lse_rows, unroll_full=True):
                lse_temperature_max = cute.arch.warp_reduction_max(
                    ts2rrLSE_temperature[None, None, m]
                    .load()
                    .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                    threads_in_group=threads_per_row,
                )
                lse_temperature_max_cur = (
                    0.0 if lse_temperature_max == -Float32.inf else lse_temperature_max
                )
                lse_temperature_sum_cur = 0.0
                for sp in cutlass.range(lse_splits, unroll_full=True):
                    lse_temperature_sum_cur += cute.math.exp2(
                        ts2rrLSE_temperature[0, sp, m] * LOG2_E
                        - (lse_temperature_max_cur * LOG2_E),
                        fastmath=True,
                    )
                lse_temperature_sum_cur = cute.arch.warp_reduction_sum(
                    lse_temperature_sum_cur, threads_in_group=threads_per_row
                )
                if (
                    max_valid_split[m] < 0
                    or lse_temperature_sum_cur == 0.0
                    or lse_temperature_sum_cur != lse_temperature_sum_cur
                ):
                    final_lse_temperature[m] = -Float32.inf
                else:
                    final_lse_temperature[m] = (
                        cute.math.log(lse_temperature_sum_cur, fastmath=True)
                        + lse_temperature_max
                    )

        # The ring is shared by the CTA, so the split loop runs to the tile's deepest valid
        # split: one slot per warp, reduced after the barrier below.
        thread_max_valid_split = max_valid_split[0]
        for m in cutlass.range(1, lse_rows, unroll_full=True):
            thread_max_valid_split = max(thread_max_valid_split, max_valid_split[m])
        warp_max_valid_split = cute.arch.warp_reduction_max(
            thread_max_valid_split, threads_in_group=32
        )
        if tidx % 32 == 0:
            sBlockMaxValidSplit[tidx // 32] = warp_max_valid_split

        if const_expr(mLSE is not None):
            for m in cutlass.range(lse_rows, unroll_full=True):
                if ts2rcLSE[0, 0, m][0] == 0:  # the lane holding split 0 writes the row
                    row_idx = tile_begin + ts2rcLSE[0, 0, m][1]
                    if row_idx < total_flat:
                        row_q, row_h = divmod(row_idx, head_divmod)
                        mLSE[row_q, row_h] = final_lse[m]
                        if const_expr(mLSE_temperature is not None):
                            mLSE_temperature[row_q, row_h] = final_lse_temperature[m]

        cute.arch.sync_threads()
        block_max_valid_split = sBlockMaxValidSplit[0]
        for wi in cutlass.range(1, self.num_warps, unroll_full=True):
            block_max_valid_split = max(block_max_valid_split, sBlockMaxValidSplit[wi])

        # Accumulate the weighted partials in FP32. A row's weight is zero for its invalid splits
        # and for every split of a row past the last one, so stale partial rows never reach the
        # sum.
        tOrO_partial = cute.make_rmem_tensor_like(tOsO[None, None, None, 0])
        tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
        tOrO.fill(0.0)
        for s in cutlass.range(block_max_valid_split + 1, unroll=2):
            stage = s % Int32(self.stages)
            phase = (s // Int32(self.stages)) & Int32(1)
            scale = cute.make_rmem_tensor(num_rows, Float32)
            for m in cutlass.range(num_rows, unroll_full=True):
                scale[m] = sLSE[s, tOcO[0, m, 0][0]]
            cute.arch.mbarrier_wait(ring_mbar + stage, phase)
            cute.autovec_copy(tOsO[None, None, None, stage], tOrO_partial)
            for m in cutlass.range(num_rows, unroll_full=True):
                if scale[m] > 0.0:
                    tOrO[None, m, None].store(
                        tOrO[None, m, None].load()
                        + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                    )
            # Every thread has copied this slot to registers before it is refilled.
            cute.arch.sync_threads()
            split_to_load = s + Int32(self.stages)
            if warp_idx == 0 and split_to_load <= block_max_valid_split:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(ring_mbar + stage, o_tma_bytes)
                cute.copy(
                    tma_atom_O_partial,
                    tOgO_tma[None, m_block, split_to_load],
                    tOsO_tma[None, stage],
                    tma_bar_ptr=ring_mbar + stage,
                )

        # Primed slots the split loop did not consume still have a bulk copy in flight; wait for
        # them so none lands after the CTA has exited.
        for stage in cutlass.range(primed_stages, unroll_full=True):
            if Int32(stage) > block_max_valid_split:
                cute.arch.mbarrier_wait(ring_mbar + stage, 0)

        # The partial columns are a 4x4 transpose of 32-bit pairs inside each 32-column group:
        # undo it with two XOR-shuffle stages across the four lanes that share the group, then
        # keep the coalesced 128-bit store. No shared memory, no barrier.
        output_scale = Float32(1.0)
        if const_expr(mOutputScale is not None):
            output_scale = mOutputScale[0]
        thr_copy_out = tiled_copy_out.get_slice(tidx)
        tOcO_store = thr_copy_out.partition_D(cO)
        rO = cute.make_rmem_tensor(tOcO_store.shape, self.dtype)
        elems_per_store = const_expr(cute.size(tiled_copy_out.layout_tv_tiled[1]))
        num_vals = const_expr(cute.size(tOcO, mode=[0]))
        assert num_vals == 8 and cute.size(tOcO_store, mode=[1]) == num_rows
        lane_in_pair_group = tidx % Int32(4)
        for m in cutlass.range(num_rows, unroll_full=True):
            for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                o0, o1 = tOrO[0, m, k], tOrO[1, m, k]
                o2, o3 = tOrO[2, m, k], tOrO[3, m, k]
                o4, o5 = tOrO[4, m, k], tOrO[5, m, k]
                o6, o7 = tOrO[6, m, k], tOrO[7, m, k]
                if const_expr(mOutputScale is not None):
                    o0, o1 = cute.arch.mul_packed_f32x2((o0, o1), (output_scale, output_scale))
                    o2, o3 = cute.arch.mul_packed_f32x2((o2, o3), (output_scale, output_scale))
                    o4, o5 = cute.arch.mul_packed_f32x2((o4, o5), (output_scale, output_scale))
                    o6, o7 = cute.arch.mul_packed_f32x2((o6, o7), (output_scale, output_scale))
                word0 = utils.cvt_f16x2_f32(o0, o1, self.dtype)
                word1 = utils.cvt_f16x2_f32(o2, o3, self.dtype)
                word2 = utils.cvt_f16x2_f32(o4, o5, self.dtype)
                word3 = utils.cvt_f16x2_f32(o6, o7, self.dtype)

                partner0 = cute.arch.shuffle_sync_bfly(word1, offset=1)
                partner1 = cute.arch.shuffle_sync_bfly(word0, offset=1)
                partner2 = cute.arch.shuffle_sync_bfly(word3, offset=1)
                partner3 = cute.arch.shuffle_sync_bfly(word2, offset=1)
                stage0, stage1 = word0, word1
                stage2, stage3 = word2, word3
                if lane_in_pair_group % Int32(2) != Int32(0):
                    stage0, stage2 = partner0, partner2
                if lane_in_pair_group % Int32(2) != Int32(1):
                    stage1, stage3 = partner1, partner3

                partner0 = cute.arch.shuffle_sync_bfly(stage2, offset=2)
                partner1 = cute.arch.shuffle_sync_bfly(stage3, offset=2)
                partner2 = cute.arch.shuffle_sync_bfly(stage0, offset=2)
                partner3 = cute.arch.shuffle_sync_bfly(stage1, offset=2)
                out0, out1 = stage0, stage1
                out2, out3 = stage2, stage3
                if lane_in_pair_group // Int32(2) != Int32(0):
                    out0, out1 = partner0, partner1
                if lane_in_pair_group // Int32(2) != Int32(1):
                    out2, out3 = partner2, partner3

                rO_pairs = cute.recast_tensor(rO[None, m, k], Int32)
                rO_pairs[0] = out0
                rO_pairs[1] = out1
                rO_pairs[2] = out2
                rO_pairs[3] = out3

        if const_expr(self.write_bf16):
            for m in cutlass.range(num_rows, unroll_full=True):
                idx = tile_begin + tOcO_store[0, m, 0][0]
                if idx < total_flat:
                    row_q, row_h = divmod(idx, head_divmod)
                    mO_row = cute.tiled_divide(mO[row_q, row_h, None], (elems_per_store,))
                    for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                        k_idx = tOcO_store[0, 0, k][1] // elems_per_store
                        cute.copy(thr_copy_out, rO[None, m, k], mO_row[None, k_idx])

        if const_expr(mOq is not None):
            self.quant_rows_mxfp8(rO, tOcO_store, mOq, mSFq, tile_begin, total_flat, head_divmod, num_rows)

    @cute.jit
    def quant_rows_mxfp8(
        self,
        rO: cute.Tensor,
        tOcO_store: cute.Tensor,
        mOq: cute.Tensor,
        mSFq: cute.Tensor,
        tile_begin: Int32,
        total_flat: Int32,
        head_divmod: FastDivmodDivisor,
        num_store_rows: cutlass.Constexpr[int],
    ):
        """The tile's final O rows as MXFP8 for the o_proj GEMM, from the BF16 registers the shuffle epilogue left in
        ``rO`` (real column order). Each thread holds 8 consecutive real columns of a row, 16 consecutive lanes per
        row (head_dim 128): 4 lanes per K32 block, the stock quantize_mxfp8_rows mapping, so amax / scale / E4M3 are
        bitwise mxfp8_quantize of the BF16 output. Same body as SparseAttentionForwardCombine.quant_rows_mxfp8; only
        the row index differs (flat (q, head) row, no cu_seqlens offset: tok = row_q). mOq: uint8
        [total_q, nheads * head_dim]; mSFq: uint8 swizzled 128x4 scales (padding rows zeroed by the caller)."""
        assert cute.size(tOcO_store, mode=[0]) == 8 and cute.size(tOcO_store, mode=[2]) == 1
        assert self.threads_per_row == 16, "the MXFP8 epilogue needs 16 lanes per row (head_dim 128, 16-bit O)"
        num_head = Int32(mOq.shape[1] // self.head_dim)
        col0 = tOcO_store[0, 0, 0][1]
        lane_block = col0 // 8
        for m in cutlass.range_constexpr(num_store_rows):
            idx = tile_begin + tOcO_store[0, m, 0][0]
            vals = cute.make_rmem_tensor((8,), Float32)
            for i in cutlass.range_constexpr(8):
                vals[i] = rO[i, m, 0].to(Float32)
            vmax = mxfp8_epi.fabs_f32(vals[0])
            for i in cutlass.range_constexpr(1, 8):
                vmax = mxfp8_epi.fmax_f32(vmax, mxfp8_epi.fabs_f32(vals[i]))
            # amax over the 4 lanes of a K32 block (consecutive lanes of one warp, one row); every lane shuffles.
            vmax = mxfp8_epi.fmax_f32(vmax, cute.arch.shuffle_sync_bfly(vmax, offset=1))
            vmax = mxfp8_epi.fmax_f32(vmax, cute.arch.shuffle_sync_bfly(vmax, offset=2))
            sf_bits = mxfp8_epi.ue8m0_round_up(vmax * mxfp8_epi.rcp_approx_ftz(Float32(448.0)))
            out_scale = Float32(0.0)
            if vmax != Float32(0.0):
                out_scale = mxfp8_epi.rcp_approx_ftz(mxfp8_epi.ue8m0_to_f32(sf_bits))
            if idx < total_flat:
                tok, head_idx = divmod(idx, head_divmod)
                q0 = utils.cvt_fp8x4_e4m3_f32(
                    vals[0] * out_scale, vals[1] * out_scale, vals[2] * out_scale, vals[3] * out_scale
                )
                q1 = utils.cvt_fp8x4_e4m3_f32(
                    vals[4] * out_scale, vals[5] * out_scale, vals[6] * out_scale, vals[7] * out_scale
                )
                qp = cute.make_ptr(
                    dtype=cutlass.Int32,
                    value=mOq.iterator.toint()
                    + Int64(tok) * Int64(mOq.shape[1])
                    + Int64(head_idx * self.head_dim + col0),
                    mem_space=mOq.iterator.memspace,
                    assumed_align=8,
                )
                qt = cute.make_tensor(qp, cute.make_layout((2,), stride=(1,)))
                qt[0] = q0
                qt[1] = q1
                if lane_block % 4 == 0:
                    k_idx = head_idx * (self.head_dim // 32) + lane_block // 4
                    sf_off = mxfp8_epi.sf_offset_128x4(tok, k_idx, num_head * (self.head_dim // 128))
                    sp = cute.make_ptr(
                        dtype=cutlass.Uint8,
                        value=mSFq.iterator.toint() + Int64(sf_off),
                        mem_space=mSFq.iterator.memspace,
                        assumed_align=1,
                    )
                    cute.make_tensor(sp, cute.make_layout((1,), stride=(1,)))[0] = cutlass.Uint8(sf_bits)

    @cute.jit
    def claim_tile(
        self,
        mSmDomains: cute.Tensor,
        mTileCounters: cute.Tensor,
        total_tiles: Int32,
        tiles_per_page_log2: Int32,
    ) -> Int32:
        """One tile for this CTA, from the domain of its SM while that domain has tiles left.

        Tiles are dealt to domains by page: page ``j`` (``2 ** tiles_per_page_log2`` tiles)
        belongs to domain ``j % num_domains``, so the ``k``-th tile of domain ``d`` is tile
        ``k % page`` of page ``(k // page) * num_domains + d``. Each counter hands out claims in
        order; a claim past the domain's tile count moves on to the next domain, and since the
        grid holds exactly one CTA per tile, the claims of all domains together always fit.
        """
        num_domains = const_expr(self.num_domains)
        sm_id = cute.arch.smid()
        domain = mSmDomains[sm_id]
        if domain < 0:
            domain = sm_id % Int32(num_domains)
        page_tiles = Int32(1) << tiles_per_page_log2
        full_pages = total_tiles >> tiles_per_page_log2
        tail_tiles = total_tiles & (page_tiles - 1)
        tile = Int32(-1)
        for _ in cutlass.range_constexpr(num_domains):
            if tile < 0:
                claim = cute.arch.atomic_add(mTileCounters.iterator + domain, Int32(1))
                domain_pages = full_pages // Int32(num_domains)
                if full_pages % Int32(num_domains) > domain:
                    domain_pages = domain_pages + 1
                domain_tiles = domain_pages << tiles_per_page_log2
                if full_pages % Int32(num_domains) == domain:
                    domain_tiles = domain_tiles + tail_tiles
                if claim < domain_tiles:
                    page = (claim >> tiles_per_page_log2) * Int32(num_domains) + domain
                    tile = (page << tiles_per_page_log2) + (claim & (page_tiles - 1))
                else:
                    domain = (domain + 1) % Int32(num_domains)
        return tile



# Tuned combine launches (LongAttn C2, COMBINE_MB1 10-24): (compute capability, emit_mxfp8, write_bf16) ->
# (tile_m, num_threads, stages, min_blocks_per_mp) for the varlen bf16-partial D=128 use_pdl path without an output
# scale (FP8 q). Bitwise the default launch: every listed pair keeps 4 threads per row (num_threads == 4 * tile_m, as
# 256 / 64), so each row's 16 slots are reduced in the same order and lanes. GB300 (10, 3): 347.1 vs
# 377.1 us at the C165 16K shape. min_blocks_per_mp 4 is never listed for (10, 7): it hit a launch failure on VR.
# TRTLLM_MSA_DISABLE_COMBINE_TUNE=1 restores the default launch.
_COMBINE_TUNED_LAUNCH = {
    ((10, 3), False, True): (32, 128, 3, 1),
    # With the MXFP8 o_proj epilogue (COMBINE_MXFP8): 364.3 vs 404.2 us (bf16 kept), 350.9 vs 385.0 us (no bf16).
    ((10, 3), True, True): (32, 128, 4, 0),
    ((10, 3), True, False): (32, 128, 3, 0),
    # VR (10, 7): 180.5 vs 221.0 us; with the MXFP8 epilogue 200.6 vs 241.9 (bf16 kept), 191.8 vs 232.5 (no bf16).
    # 6,000-launch repeats clean (836221); occupancy-heavy neighbours (s2 mb4, s4 mb3) fault on VR, so keep to these.
    ((10, 7), False, True): (64, 256, 3, 3),
    ((10, 7), True, True): (64, 256, 3, 3),
    ((10, 7), True, False): (64, 256, 3, 3),
}
_COMBINE_TUNE_OFF = os.environ.get("TRTLLM_MSA_DISABLE_COMBINE_TUNE", "0") == "1"
for _cfg in _COMBINE_TUNED_LAUNCH.values():
    assert _cfg[1] == 4 * _cfg[0], "a tuned combine launch must keep 4 threads per row (num_threads == 4 * tile_m)"


_COMBINE_TUNE_LOGGED: set = set()  # one engagement line per (cc, path) and process (stderr, via logging)


def _combine_tuned_launch(device, emit_mxfp8: bool, write_bf16: bool):
    if _COMBINE_TUNE_OFF:
        return None
    cc = torch.cuda.get_device_capability(device)
    key = (cc, bool(emit_mxfp8), bool(write_bf16))
    tuned = _COMBINE_TUNED_LAUNCH.get(key)
    assert tuned is None or cc != (10, 7) or tuned[3] != 4, "min_blocks_per_mp 4 fails to launch on sm_107"
    if tuned is not None and key not in _COMBINE_TUNE_LOGGED:
        _COMBINE_TUNE_LOGGED.add(key)
        logging.getLogger("fmha_sm100.combine").warning(
            "[MSA COMBINE_TUNE] taken: cc=%d.%d mxfp8=%s write_bf16=%s launch(tile_m, threads, stages, mbpm)=%s",
            cc[0], cc[1], key[1], key[2], tuned,
        )
    return tuned


# TMA-ring class routing (LongAttn MSADEV transplant). combine() sends a call to SparseAttentionForwardCombineTma when
# it is the varlen BF16/FP16-partial head_dim-128 path without output_scale or seqused, on a listed compute capability,
# with contiguous 16-byte-aligned tensors; every other call keeps SparseAttentionForwardCombine (and its tune table).
# Both classes merge each row in the same order (4 lanes per LSE row, slots ascending), so O / LSE_out / MXFP8 are
# bitwise the same (mbkit6 3440366: dev O and LSE bitwise ours on realistic + 2^+-12 stress).
# TRTLLM_MSA_DISABLE_COMBINE_DEVCLASS=1 routes every call to SparseAttentionForwardCombine.
_COMBINE_DEVCLASS_OFF = os.environ.get("TRTLLM_MSA_DISABLE_COMBINE_DEVCLASS", "0") == "1"
# GB only until a VR microbench: on SM107 dev runs the 2-slot interleaved ring, and our tuned class (180.5 us) is
# not yet compared with it there.
# (10, 7): VR mbkit8 836923, bitwise + robust, -6.8 us vs C2 tuned at the default launch.
_COMBINE_DEVCLASS_CCS = {(10, 0), (10, 3), (10, 7)}
# (compute capability, emit_mxfp8, write_bf16) -> (tile_m, num_threads, stages, min_blocks_per_mp) for the TMA-ring
# class. Empty: every call takes dev's default launch (TILE_M, NUM_THREADS, STAGES / STAGES_INTERLEAVED on SM107, 0).
# Every entry must keep 4 threads per LSE row (num_threads == 4 * tile_m), which keeps the merge order bitwise.
# TRTLLM_MSA_DISABLE_COMBINE_TUNE=1 ignores it too.
_COMBINE_DEVCLASS_TUNED_LAUNCH = {}
for _cfg in _COMBINE_DEVCLASS_TUNED_LAUNCH.values():
    assert _cfg[1] == 4 * _cfg[0], "a TMA-ring combine launch must keep 4 threads per row (num_threads == 4 * tile_m)"
_COMBINE_DEVCLASS_LOGGED: set = set()


def _log_devclass_once(key, msg, *args):
    if key not in _COMBINE_DEVCLASS_LOGGED:
        _COMBINE_DEVCLASS_LOGGED.add(key)
        logging.getLogger("fmha_sm100.combine").warning(msg, *args)


def _combine_devclass_launch(cc, emit_mxfp8: bool, write_bf16: bool):
    launch = (TILE_M, NUM_THREADS, STAGES_INTERLEAVED if cc == (10, 7) else STAGES, 0)
    if not _COMBINE_TUNE_OFF:
        tuned = _COMBINE_DEVCLASS_TUNED_LAUNCH.get((cc, bool(emit_mxfp8), bool(write_bf16)))
        if tuned is not None:
            assert tuned[1] == 4 * tuned[0], "a TMA-ring combine launch must keep 4 threads per row"
            launch = tuned
    return launch


def _combine_devclass_reject(
    o_partial_fake, lse_partial, o_out, tensors, cu_seqlens, seqused, output_scale, cc, launch
):
    """None when the TMA-ring class takes this call, else the reason it does not."""
    if _COMBINE_DEVCLASS_OFF:
        return "TRTLLM_MSA_DISABLE_COMBINE_DEVCLASS=1"
    if cc not in _COMBINE_DEVCLASS_CCS:
        return f"cc {cc} not in {sorted(_COMBINE_DEVCLASS_CCS)}"
    if cu_seqlens is None or o_partial_fake.ndim != 4 or o_out.ndim != 3:
        return "not the varlen layout"
    if seqused is not None:
        return "seqused"
    if output_scale is not None:
        return "output_scale"
    if o_partial_fake.dtype not in (torch.bfloat16, torch.float16) or o_out.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return "partial / output dtype"
    num_splits, total_q, head_q, dim = o_partial_fake.shape
    if dim != 128 or not 1 <= num_splits <= 256:
        return "head_dim / num_splits"
    if total_q == 0 or head_q == 0:
        return "empty"
    tile_m, num_threads = launch[0], launch[1]
    if num_splits % (num_threads // tile_m) or tile_m % (num_threads // (dim * o_partial_fake.element_size() // 16)):
        return "num_splits / tile shape"
    split_stride = o_partial_fake.stride(0)
    if (
        o_partial_fake.stride()[1:] != (head_q * dim, dim, 1)
        or split_stride < total_q * head_q * dim
        or (split_stride * o_partial_fake.element_size()) % 16
        or split_stride >= MAX_SPLIT_SLAB_ELEMENTS
    ):
        return "o_partial_fake strides"
    if tuple(o_out.shape) != (total_q, head_q, dim) or tuple(lse_partial.shape) != (num_splits, total_q, head_q):
        return "shapes"
    if o_partial_fake.data_ptr() % 16:
        return "o_partial_fake alignment"
    for t in tensors:
        if t is not None and (not t.is_contiguous() or t.data_ptr() % 16 or t.device != o_out.device):
            return "a tensor is not contiguous / 16-byte aligned / on o_out's device"
    return None


def _combine_devclass(
    o_partial_fake,
    lse_partial,
    o_out,
    lse_out,
    lse_temperature_partial,
    lse_temperature_out,
    split_counts,
    qhead_per_kvhead,
    use_pdl,
    o_fp8,
    sf_out,
    write_bf16,
    cc,
    launch,
):
    """The dev launcher (msa_dev combine.py:2490-2651) without locality / output_scale, plus the MXFP8 outputs."""
    num_splits, _, head_q, dim = o_partial_fake.shape
    partial_dtype = _get_cutlass_dtype(o_partial_fake.dtype)
    out_dtype = _get_cutlass_dtype(o_out.dtype)
    has_lse = lse_out is not None
    has_temperature = lse_temperature_out is not None
    has_split_counts = split_counts is not None
    emit_mxfp8 = o_fp8 is not None
    tile_m, num_threads, stages, min_blocks_per_mp = launch
    # One 2-D TMA box per tile loads the LSEs when every split's FP32 LSE slab starts 16-byte aligned (head_q % 4 == 0).
    lse_split_aligned = head_q % 4 == 0
    key = (
        "combine_tma",
        cc,
        dim,
        tile_m,
        num_threads,
        stages,
        min_blocks_per_mp,
        num_splits,
        partial_dtype,
        out_dtype,
        has_lse,
        has_temperature,
        has_split_counts,
        use_pdl,
        lse_split_aligned,
        emit_mxfp8,
        write_bf16,
    )
    if key not in _combine_compile_cache:
        from src.common.aot_cache import try_load_aot, save_aot

        loaded = try_load_aot(key)
        if loaded is not None:
            _combine_compile_cache[key] = loaded
        else:
            from quack.compile_utils import make_fake_tensor

            kernel = SparseAttentionForwardCombineTma(
                dtype=out_dtype,
                dtype_partial=partial_dtype,
                head_dim=dim,
                topk=num_splits,
                tile_m=tile_m,
                num_threads=num_threads,
                stages=stages,
                use_pdl=use_pdl,
                min_blocks_per_mp=min_blocks_per_mp,
                num_domains=0,
                lse_split_aligned=lse_split_aligned,
                write_bf16=write_bf16,
            )
            total_q_sym, nheads_sym = (cute.sym_int64() for _ in range(2))
            mO_partial = make_fake_tensor(
                partial_dtype,
                (num_splits, total_q_sym, nheads_sym, dim),
                divisibility=128 // partial_dtype.width,
            )
            mLSE_partial = make_fake_tensor(
                Float32, (num_splits, total_q_sym, nheads_sym), divisibility=1, leading_dim=2
            )
            mO = make_fake_tensor(
                out_dtype, (total_q_sym, nheads_sym, dim), divisibility=128 // out_dtype.width
            )
            mLSE = (
                make_fake_tensor(Float32, (total_q_sym, nheads_sym), divisibility=1, leading_dim=1)
                if has_lse
                else None
            )
            mLSE_temperature_partial = (
                make_fake_tensor(
                    Float32, (num_splits, total_q_sym, nheads_sym), divisibility=1, leading_dim=2
                )
                if has_temperature
                else None
            )
            mLSE_temperature = (
                make_fake_tensor(Float32, (total_q_sym, nheads_sym), divisibility=1, leading_dim=1)
                if has_temperature
                else None
            )
            mSplitCounts = (
                make_fake_tensor(
                    Int32, (cute.sym_int64(), cute.sym_int64()), divisibility=1, leading_dim=1
                )
                if has_split_counts
                else None
            )
            mOq = (
                make_fake_tensor(cutlass.Uint8, (cute.sym_int64(), cute.sym_int64()), divisibility=16, leading_dim=1)
                if emit_mxfp8
                else None
            )
            mSFq = (
                make_fake_tensor(cutlass.Uint8, (cute.sym_int64(),), divisibility=4, leading_dim=0)
                if emit_mxfp8
                else None
            )
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            _combine_compile_cache[key] = cute.compile(
                kernel,
                mO_partial,
                mLSE_partial,
                mO,
                mLSE,
                mLSE_temperature_partial,
                mLSE_temperature,
                mSplitCounts,
                None,
                Int32(qhead_per_kvhead),
                None,
                None,
                Int32(0),
                mOq,
                mSFq,
                stream,
                options="--enable-tvm-ffi",
            )
            save_aot(key, _combine_compile_cache[key])

    with torch.cuda.nvtx.range("K2_Combine"):
        _combine_compile_cache[key](
            o_partial_fake,
            lse_partial,
            o_out,
            lse_out,
            lse_temperature_partial,
            lse_temperature_out,
            split_counts,
            None,
            qhead_per_kvhead,
            None,
            None,
            0,
            o_fp8,
            sf_out,
        )


def combine(
    o_partial_fake,
    lse_partial,
    o_out,
    lse_out,
    *,
    lse_temperature_partial=None,
    lse_temperature_out=None,
    cu_seqlens=None,
    seqused=None,
    split_counts=None,
    output_scale=None,
    use_pdl=False,
    o_fp8=None,
    sf_out=None,
    write_bf16=True,
):
    """K2: merge sparse forward split partials into the final output.

    STG.128 fake-layout handling remains an internal implementation detail.
    When lse_out is provided, the kernel writes the final authoritative
    log-sum-exp for each query row/head directly into that tensor.

    Args:
        o_partial_fake:
            Batched: [num_splits, batch, Sq, head_q, dim]
            Varlen:  [num_splits, total_q, head_q, dim]
        lse_partial:
            Batched: [num_splits, batch, Sq, head_q]
            Varlen:  [num_splits, total_q, head_q]
        o_out:
            Batched: [batch, Sq, head_q, dim]
            Varlen:  [total_q, head_q, dim]
        lse_out:
            Batched: [batch, Sq, head_q]
            Varlen:  [total_q, head_q]
        lse_temperature_partial:
            Optional temperature-scaled LSE partial with the same shape as
            lse_partial.
        lse_temperature_out:
            Optional temperature-scaled final LSE with the same shape as
            lse_out.
        cu_seqlens: Optional [batch + 1] int32 for varlen-Q combine.
        seqused: Optional [batch] int32 effective lengths for combine.
        split_counts: Optional int32 rowwise valid split counts prepared from
            q2k metadata. Batched: [batch, seqlen, head_kv]. Varlen:
            [total_q, head_kv].
        output_scale: Optional fp32 tensor with at least one element. When
            provided, the final O accumulator is multiplied once before store.
        use_pdl: When True, wait on PDL dependencies from the producer K1
            kernel. When False, launch without PDL waits.
        o_fp8, sf_out: Optional MXFP8 copy of the final O for the o_proj GEMM (Host, CTX (b)):
            o_fp8 uint8/float8_e4m3fn [total_q, head_q * dim] and sf_out uint8 1-D swizzled 128x4 K32 UE8M0
            scales of [total_q, head_q * dim]; bitwise mxfp8_quantize(o_out.view(total_q, -1), True). The
            caller zeroes sf_out's padding rows (total_q .. pad128). Varlen + use_pdl only.
        write_bf16: False skips the bf16 o_out store (only valid with o_fp8 and no other reader of o_out).

    Kernel class: the varlen BF16/FP16-partial head_dim-128 call without output_scale or seqused (the in-model
    sparse-prefill call) runs SparseAttentionForwardCombineTma (msa_dev's TMA-ring combine) on the GPUs in
    _COMBINE_DEVCLASS_CCS; it treats the total_q rows as one flat batch, which is the cu_seqlens result when
    cu_seqlens[0] == 0 and cu_seqlens[-1] == total_q (rows outside that range are written too, not left untouched).
    Every other call, or TRTLLM_MSA_DISABLE_COMBINE_DEVCLASS=1, runs SparseAttentionForwardCombine.
    """
    emit_mxfp8 = o_fp8 is not None
    if (o_fp8 is None) != (sf_out is None):
        raise ValueError("o_fp8 and sf_out must be given together")
    if not write_bf16 and not emit_mxfp8:
        raise ValueError("write_bf16=False requires the MXFP8 output")
    if emit_mxfp8:
        if cu_seqlens is None or not use_pdl or o_out.ndim != 3:
            raise ValueError("the MXFP8 combine epilogue supports the varlen use_pdl path only")
        if o_out.dtype != torch.bfloat16 or o_partial_fake.shape[-1] != 128:
            raise ValueError("the MXFP8 combine epilogue needs a bf16 output and head_dim 128")
        o_fp8 = o_fp8.view(torch.uint8)
        if o_fp8.shape != (o_out.shape[0], o_out.shape[1] * o_out.shape[2]) or not o_fp8.is_contiguous():
            raise ValueError(f"o_fp8 must be contiguous [total_q, head_q * dim], got {tuple(o_fp8.shape)}")
        rows_padded = (o_out.shape[0] + 127) // 128 * 128
        if sf_out.dtype != torch.uint8 or sf_out.numel() < rows_padded * o_fp8.shape[1] // 32:
            raise ValueError("sf_out must be uint8 with the padded 128x4 swizzled size")
    D = o_partial_fake.shape[-1]
    num_splits = o_partial_fake.shape[0]
    return_temperature_lse = (
        lse_temperature_partial is not None or lse_temperature_out is not None
    )
    if (lse_temperature_partial is None) != (lse_temperature_out is None):
        raise ValueError(
            "lse_temperature_partial and lse_temperature_out must either both be provided or both be None"
        )
    if lse_temperature_partial is not None and lse_temperature_partial.shape != lse_partial.shape:
        raise ValueError(
            "lse_temperature_partial must have the same shape as lse_partial, "
            f"got {lse_temperature_partial.shape} vs {lse_partial.shape}"
        )
    if lse_temperature_out is not None:
        if lse_out is None:
            raise ValueError("lse_temperature_out requires lse_out")
        if lse_temperature_out.shape != lse_out.shape:
            raise ValueError(
                "lse_temperature_out must have the same shape as lse_out, "
                f"got {lse_temperature_out.shape} vs {lse_out.shape}"
            )
        if lse_temperature_out.dtype != torch.float32 or lse_temperature_partial.dtype != torch.float32:
            raise TypeError("temperature LSE tensors must be torch.float32")

    partial_dtype = _get_cutlass_dtype(o_partial_fake.dtype)
    out_dtype = _get_cutlass_dtype(o_out.dtype)
    if output_scale is not None:
        if output_scale.dtype != torch.float32:
            raise TypeError(f"output_scale must be torch.float32, got {output_scale.dtype}")
        if output_scale.numel() < 1:
            raise ValueError("output_scale must contain at least one element")
        if output_scale.device != o_out.device:
            raise ValueError("output_scale must be on the same device as o_out")
        output_scale = output_scale.contiguous()
    if split_counts is not None:
        if split_counts.dtype != torch.int32:
            raise TypeError(f"split_counts must be torch.int32, got {split_counts.dtype}")
        if o_out.ndim == 4:
            if split_counts.ndim != 3:
                raise ValueError(
                    f"batched split_counts must have shape [batch, seqlen, head_kv], got {split_counts.shape}"
                )
            if split_counts.shape[:2] != o_out.shape[:2]:
                raise ValueError(
                    f"split_counts shape {split_counts.shape} must match batch/seqlen of o_out {o_out.shape}"
                )
        else:
            if cu_seqlens is None:
                raise ValueError("split_counts with varlen output requires cu_seqlens")
            if split_counts.ndim != 2:
                raise ValueError(
                    f"varlen split_counts must have shape [total_q, head_kv], got {split_counts.shape}"
                )
            if split_counts.shape[0] != o_out.shape[0]:
                raise ValueError(
                    f"split_counts total_q ({split_counts.shape[0]}) must match o_out total_q "
                    f"({o_out.shape[0]})"
                )
        if o_out.shape[-2] % split_counts.shape[-1] != 0:
            raise ValueError(
                f"o_out heads ({o_out.shape[-2]}) must be divisible by split_counts heads ({split_counts.shape[-1]})"
            )
        qheadperkv = o_out.shape[-2] // split_counts.shape[-1]
    else:
        qheadperkv = 1
    if cu_seqlens is not None:
        if cu_seqlens.dtype != torch.int32:
            raise TypeError(f"cu_seqlens must be torch.int32, got {cu_seqlens.dtype}")
        if cu_seqlens.ndim != 1:
            raise ValueError(f"cu_seqlens must be rank-1, got {cu_seqlens.shape}")
        if not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be contiguous")
    if seqused is not None:
        if seqused.dtype != torch.int32:
            raise TypeError(f"seqused must be torch.int32, got {seqused.dtype}")
        if seqused.ndim != 1:
            raise ValueError(f"seqused must be rank-1, got {seqused.shape}")
        if not seqused.is_contiguous():
            raise ValueError("seqused must be contiguous")

    cc = torch.cuda.get_device_capability(o_out.device)
    devclass_launch = _combine_devclass_launch(cc, emit_mxfp8, write_bf16)
    reject = _combine_devclass_reject(
        o_partial_fake,
        lse_partial,
        o_out,
        (lse_partial, o_out, lse_out, lse_temperature_partial, lse_temperature_out, split_counts),
        cu_seqlens,
        seqused,
        output_scale,
        cc,
        devclass_launch,
    )
    if reject is None:
        _log_devclass_once(
            ("taken", cc, emit_mxfp8, write_bf16, devclass_launch),
            "[MSA COMBINE_DEVCLASS] taken: cc=%d.%d mxfp8=%s write_bf16=%s launch(tile_m, threads, stages, mbpm)=%s",
            cc[0], cc[1], emit_mxfp8, write_bf16, devclass_launch,
        )
        _combine_devclass(
            o_partial_fake,
            lse_partial,
            o_out,
            lse_out,
            lse_temperature_partial,
            lse_temperature_out,
            split_counts,
            qheadperkv,
            use_pdl,
            o_fp8,
            sf_out,
            write_bf16,
            cc,
            devclass_launch,
        )
        return
    _log_devclass_once(("not_taken", reject), "[MSA COMBINE_DEVCLASS] not taken: %s", reject)

    k_block_size = 128 if D > 64 else 64
    tile_m = 64
    num_threads = 256
    stages = 2
    has_cu_seqlens = cu_seqlens is not None
    has_seqused = seqused is not None
    has_lse = lse_out is not None
    has_split_counts = split_counts is not None
    has_output_scale = output_scale is not None
    min_blocks_per_mp = 3 if has_output_scale and use_pdl else 0
    if D == 128 and has_cu_seqlens and use_pdl and not has_output_scale and partial_dtype is cutlass.BFloat16:
        tuned = _combine_tuned_launch(o_out.device, emit_mxfp8, write_bf16)
        if tuned is not None:
            tile_m, num_threads, stages, min_blocks_per_mp = tuned

    key = (
        "combine",
        D,
        k_block_size,
        tile_m,
        num_threads,
        stages,
        num_splits,
        partial_dtype,
        out_dtype,
        has_cu_seqlens,
        has_seqused,
        has_lse,
        bool(return_temperature_lse),
        has_split_counts,
        has_output_scale,
        use_pdl,
        min_blocks_per_mp,
        emit_mxfp8,
        write_bf16,
    )
    if key not in _combine_compile_cache:
        from src.common.aot_cache import try_load_aot, save_aot

        loaded = try_load_aot(key)
        if loaded is not None:
            _combine_compile_cache[key] = loaded
        else:
            from quack.compile_utils import make_fake_tensor

            kernel = SparseAttentionForwardCombine(
                dtype=out_dtype,
                dtype_partial=partial_dtype,
                head_dim=D,
                tile_m=tile_m,
                k_block_size=k_block_size,
                topk=num_splits,
                num_threads=num_threads,
                use_pdl=use_pdl,
                min_blocks_per_mp=min_blocks_per_mp,
                write_bf16=write_bf16,
                # stages=2 halves per-block SMEM (168 KB -> 103 KB) -> 2 blocks/SM,
                # theoretical occupancy 12.5% -> 25%. NCU DRAM throughput 76.35%
                # -> 88.64%. The tuned launches (_COMBINE_TUNED_LAUNCH) override it.
                stages=stages,
            )
            div = 128 // partial_dtype.width
            if has_cu_seqlens:
                total_q, nheads = (cute.sym_int64() for _ in range(2))
                mO_partial = make_fake_tensor(
                    partial_dtype, (num_splits, total_q, nheads, D), divisibility=div
                )
                mLSE_partial = make_fake_tensor(
                    Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=2
                )
                mO = make_fake_tensor(
                    out_dtype, (total_q, nheads, D), divisibility=128 // out_dtype.width
                )
                mLSE = (
                    make_fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=1)
                    if has_lse
                    else None
                )
                mLSE_temperature_partial = (
                    make_fake_tensor(
                        Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=2
                    )
                    if return_temperature_lse
                    else None
                )
                mLSE_temperature = (
                    make_fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=1)
                    if return_temperature_lse
                    else None
                )
            else:
                batch, sq, nheads = (cute.sym_int64() for _ in range(3))
                mO_partial = make_fake_tensor(
                    partial_dtype, (num_splits, batch, sq, nheads, D), divisibility=div
                )
                mLSE_partial = make_fake_tensor(
                    Float32, (num_splits, batch, sq, nheads), divisibility=1, leading_dim=3
                )
                mO = make_fake_tensor(
                    out_dtype, (batch, sq, nheads, D), divisibility=128 // out_dtype.width
                )
                mLSE = (
                    make_fake_tensor(Float32, (batch, sq, nheads), divisibility=1, leading_dim=2)
                    if has_lse
                    else None
                )
                mLSE_temperature_partial = (
                    make_fake_tensor(
                        Float32, (num_splits, batch, sq, nheads), divisibility=1, leading_dim=3
                    )
                    if return_temperature_lse
                    else None
                )
                mLSE_temperature = (
                    make_fake_tensor(Float32, (batch, sq, nheads), divisibility=1, leading_dim=2)
                    if return_temperature_lse
                    else None
                )
            if not has_split_counts:
                mSplitCounts = None
            elif has_cu_seqlens:
                total_q_ctr, nheads_kv = (cute.sym_int64() for _ in range(2))
                mSplitCounts = make_fake_tensor(
                    Int32, (total_q_ctr, nheads_kv), divisibility=1, leading_dim=1
                )
            else:
                nheads_kv = cute.sym_int64()
                mSplitCounts = make_fake_tensor(
                    Int32, (batch, sq, nheads_kv), divisibility=1, leading_dim=2
                )
            mOutputScale = (
                make_fake_tensor(Float32, (cute.sym_int64(),), divisibility=1, leading_dim=0)
                if has_output_scale
                else None
            )
            mOq = (
                make_fake_tensor(cutlass.Uint8, (cute.sym_int64(), cute.sym_int64()), divisibility=16, leading_dim=1)
                if emit_mxfp8
                else None
            )
            mSFq = (
                make_fake_tensor(cutlass.Uint8, (cute.sym_int64(),), divisibility=4, leading_dim=0)
                if emit_mxfp8
                else None
            )
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            _combine_compile_cache[key] = cute.compile(
                kernel,
                mO_partial,
                mLSE_partial,
                mO,
                mLSE,
                mLSE_temperature_partial,
                mLSE_temperature,
                None
                if cu_seqlens is None
                else make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=0),
                None
                if seqused is None
                else make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=0),
                None,
                None,
                None,
                mSplitCounts,
                mOutputScale,
                Int32(qheadperkv),
                mOq,
                mSFq,
                stream,
                options="--enable-tvm-ffi",
            )
            save_aot(key, _combine_compile_cache[key])

    with torch.cuda.nvtx.range("K2_Combine"):
        _combine_compile_cache[key](
            o_partial_fake,
            lse_partial,
            o_out,
            lse_out,
            lse_temperature_partial,
            lse_temperature_out,
            cu_seqlens,
            seqused,
            None,
            None,
            None,
            split_counts,
            output_scale,
            qheadperkv,
            o_fp8,
            sf_out,
        )

# SPDX-FileCopyrightText: Copyright (c) 2025, Tri Dao.
# SPDX-FileCopyrightText: Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code derived from the Quack library:
# https://github.com/Dao-AILab/quack
#
# Argmax kernel using CuTE DSL for TensorRT-LLM speculative decoding.

from typing import Optional, Tuple, Type

import torch

from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.arch.nvvm_wrappers import FULL_MASK
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.typing import Float32, Int, Int32
    from cutlass.cutlass_dsl import T, dsl_user_op

    # ============================================================================
    # Torch to CuTE dtype mapping
    # ============================================================================
    torch2cute_dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }

    # ============================================================================
    # CUDA Graph compatibility wrapper
    # ============================================================================
    class CUDAGraphCompatibleWrapper:
        """Wrapper to make tensors compatible with CUDA graph capture for DLPack export."""

        def __init__(self, tensor):
            self._tensor = tensor

        def __dlpack__(self, stream=None):
            return self._tensor.__dlpack__(stream=-1)

        def __dlpack_device__(self):
            return self._tensor.__dlpack_device__()

    # ============================================================================
    # Utility functions from quack/utils.py
    # ============================================================================
    @dsl_user_op
    def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
        return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)

    @dsl_user_op
    def set_block_rank(
        smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: cute.Int32, *, loc=None, ip=None
    ) -> cutlass.Int32:
        smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
                "mapa.shared::cluster.u32 $0, $1, $2;",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def store_shared_remote(
        val: float | Float32 | cutlass.Int64,
        smem_ptr: cute.Pointer,
        mbar_ptr: cute.Pointer,
        peer_cta_rank_in_cluster: cute.typing.Int,
        *,
        loc=None,
        ip=None,
    ) -> None:
        remote_smem_ptr_i32 = set_block_rank(
            smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
        ).ir_value()
        remote_mbar_ptr_i32 = set_block_rank(
            mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
        ).ir_value()
        if cutlass.const_expr(isinstance(val, float)):
            val = Float32(val)
        assert isinstance(val, (Float32, Int32, cutlass.Int64))
        suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
        constraint = {Float32: "f", Int32: "r", cutlass.Int64: "l"}[type(val)]
        llvm.inline_asm(
            None,
            [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
            f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
            f"r,{constraint},r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @cute.jit
    def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
        tApA = cute.make_fragment(
            cute.make_layout(
                (
                    cute.size(tAcA, mode=[0, 1]),
                    cute.size(tAcA, mode=[1]),
                    cute.size(tAcA, mode=[2]),
                ),
                stride=(cute.size(tAcA, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tApA.shape[0]):
            for rest_k in cutlass.range_constexpr(tApA.shape[2]):
                tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
        return tApA

    @cute.jit
    def fill_oob(tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
        tXrX_fill = cute.make_fragment_like(tXsX[(None, 0), None, 0])
        tXrX_fill.fill(fill_value)
        for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
            for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
                if cutlass.const_expr(tXpX is not None):
                    if not tXpX[rest_v, 0, rest_k]:
                        cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
                else:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])

    @dsl_user_op
    def domain_offset_i64(
        coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
    ) -> cute.Tensor:
        flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
        flat_stride = cute.flatten_to_tuple(tensor.stride)
        offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
        new_ptr = cute.make_ptr(
            tensor.element_type,
            tensor.iterator.toint() + offset * tensor.element_type.width // 8,
            tensor.memspace,
            assumed_align=tensor.iterator.max_alignment,
        )
        return cute.make_tensor(new_ptr, tensor.layout)

    # ============================================================================
    # Inline PTX for redux.sync operations
    # ============================================================================
    @dsl_user_op
    def ptx_redux_sync_max_f32(
        value: Float32, mask: Int = FULL_MASK, *, loc=None, ip=None
    ) -> Float32:
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [Float32(value).ir_value(loc=loc, ip=ip), Int32(mask).ir_value(loc=loc, ip=ip)],
                """redux.sync.max.f32 $0, $1, $2;""",
                "=f,f,i",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def ptx_redux_sync_min_u32(value: Int32, mask: Int = FULL_MASK, *, loc=None, ip=None) -> Int32:
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [Int32(value).ir_value(loc=loc, ip=ip), Int32(mask).ir_value(loc=loc, ip=ip)],
                """redux.sync.min.u32 $0, $1, $2;""",
                "=r,r,i",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def ptx_select_argmax_candidate(
        current_max: Float32, warp_max: Float32, current_argmax: Int32, *, loc=None, ip=None
    ) -> Int32:
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [
                    Float32(current_max).ir_value(loc=loc, ip=ip),
                    Float32(warp_max).ir_value(loc=loc, ip=ip),
                    Int32(current_argmax).ir_value(loc=loc, ip=ip),
                ],
                """{
                    .reg .pred p;
                    setp.eq.f32 p, $1, $2;
                    selp.s32 $0, $3, 0xffffffff, p;
                }""",
                "=r,f,f,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.jit
    def warp_argmax_redux(current_max: Float32, current_argmax: Int32):
        """Redux-based warp argmax - only works on sm_100f (Blackwell)."""
        warp_max = ptx_redux_sync_max_f32(current_max)
        candidate_idx = ptx_select_argmax_candidate(current_max, warp_max, current_argmax)
        winning_idx = ptx_redux_sync_min_u32(candidate_idx)
        return warp_max, winning_idx

    @cute.jit
    def warp_reduce_argmax(current_max: Float32, current_argmax: Int32):
        """Shuffle-based warp argmax - works on all architectures (Hopper+)."""
        warp_max = current_max
        warp_argmax = current_argmax

        # Use butterfly shuffle pattern for warp reduction
        for i in cutlass.range_constexpr(int(5)):  # log2(32) = 5 iterations
            # Get values from other lanes using butterfly pattern
            other_max = cute.arch.shuffle_sync_bfly(warp_max, offset=1 << i)
            other_argmax = cute.arch.shuffle_sync_bfly(warp_argmax, offset=1 << i)

            # Inline argmax comparison
            if other_max > warp_max:
                warp_max = other_max
                warp_argmax = other_argmax

        return warp_max, warp_argmax

    # ============================================================================
    # Reduction Base class
    # ============================================================================
    class ReductionBase:
        def __init__(
            self, dtype: Type[cutlass.Numeric], N: int, stage: int, reduction_dtype=cutlass.Float32
        ):
            self.dtype = dtype
            self.N = N
            self.stage = stage
            self.reduction_dtype = reduction_dtype

        def _calculate_threads_per_row(self):
            raise NotImplementedError()

        def _set_cluster_n(self):
            self.cluster_n = 1

        def _get_num_threads(self):
            return 128 if self.N <= 16384 else 256

        def _get_tv_layout(self, num_copy_bits=128):
            vecsize = num_copy_bits // self.dtype.width
            num_threads = self._get_num_threads()
            threads_per_row = self._calculate_threads_per_row()
            num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
            cols_per_block = num_threads // threads_per_row
            tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
            tv_layout = cute.make_layout(
                ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
                stride=(
                    (vecsize * cols_per_block, 1),
                    (cols_per_block, cols_per_block * vecsize * threads_per_row),
                ),
            )
            return tiler_mn, tv_layout

        def _smem_size_in_bytes(self, tiler_mn, num_warps):
            return (
                cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
                + self.stage * num_warps * self.cluster_n * (self.reduction_dtype.width // 8)
                + self.stage * (cutlass.Int64.width // 8)
            )

        def _get_reduction_buffer_layout(self, tv_layout: cute.Layout, cluster_n: int):
            num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
            warps_per_row = max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
            return cute.make_ordered_layout(
                (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
                order=(1, 0, 2),
            )

        def _allocate_reduction_buffer_and_mbar(
            self, smem: cutlass.utils.SmemAllocator, tv_layout: cute.Layout
        ) -> Tuple[cute.Tensor, Optional[cute.Pointer]]:
            reduction_buffer = smem.allocate_tensor(
                self.reduction_dtype,
                self._get_reduction_buffer_layout(tv_layout, self.cluster_n),
                byte_alignment=4,
            )
            if cutlass.const_expr(self.cluster_n > 1):
                mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=self.stage)
            else:
                mbar_ptr = None
            return reduction_buffer, mbar_ptr

        @cute.jit
        def _initialize_cluster(self, tidx: cutlass.Int32, mbar_ptr: cute.Pointer, num_warps: int):
            if cutlass.const_expr(self.cluster_n > 1):
                if tidx < self.stage:
                    cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.cluster_arrive_relaxed()

    # ============================================================================
    # Argmax Kernel class
    # ============================================================================
    class ArgmaxKernel(ReductionBase):
        def __init__(self, dtype: Type[cutlass.Numeric], N: int, use_redux: bool = False):
            super().__init__(dtype, N, stage=1, reduction_dtype=cutlass.Float32)
            # use_redux=True for Blackwell (sm_100f), False for Hopper (sm_90)
            self.use_redux = use_redux

        def _calculate_threads_per_row(self):
            N = self.N
            return (
                8
                if N <= 64
                else (
                    16
                    if N <= 128
                    else (32 if N <= 3072 else (64 if N <= 6144 else (128 if N <= 16384 else 256)))
                )
            )

        def _set_cluster_n(self):
            N = self.N
            if cutlass.const_expr(self.dtype.width == 16):
                self.cluster_n = (
                    1
                    if N <= 16 * 1024
                    else (
                        2
                        if N <= 32 * 1024
                        else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                    )
                )
            else:
                self.cluster_n = (
                    1
                    if N <= 32 * 1024
                    else (
                        2
                        if N <= 64 * 1024
                        else (4 if N <= 128 * 1024 else (8 if N <= 256 * 1024 else 16))
                    )
                )

        def _get_reduction_buffer_layout(self, tv_layout: cute.Layout, cluster_n: int):
            num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
            warps_per_row = max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
            return cute.make_ordered_layout(
                (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage, 2),
                order=(1, 0, 2, 3),
            )

        def _smem_size_in_bytes(self, tiler_mn, num_warps):
            return (
                cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
                + 2 * self.stage * num_warps * self.cluster_n * (self.reduction_dtype.width // 8)
                + self.stage * (cutlass.Int64.width // 8)
            )

        @cute.jit
        def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
            self._set_cluster_n()
            tiler_mn, tv_layout = self._get_tv_layout()
            num_threads = cute.size(tv_layout, mode=[0])
            num_warps = num_threads // cute.arch.WARP_SIZE

            self.kernel(mX, mO, tv_layout, tiler_mn).launch(
                grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
                block=[num_threads, 1, 1],
                cluster=[1, self.cluster_n, 1] if cutlass.const_expr(self.cluster_n > 1) else None,
                smem=self._smem_size_in_bytes(tiler_mn, num_warps),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self, mX: cute.Tensor, mO: cute.Tensor, tv_layout: cute.Layout, tiler_mn: cute.Shape
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, bidz = cute.arch.block_idx()

            if cutlass.const_expr(self.cluster_n > 1):
                cluster_y = cute.arch.block_idx()[1]
            else:
                cluster_y = cutlass.const_expr(0)

            shape = mX.shape
            idX = cute.make_identity_tensor(shape)

            mX, mO = [domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mX, mO)]
            gX, gO = [cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mX, mO)]
            cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

            smem = cutlass.utils.SmemAllocator()
            sX = smem.allocate_tensor(
                mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
            )
            reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

            copy_atom_load_X = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128
            )
            thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)

            tXgX = thr_copy_X.partition_S(gX)
            tXsX = thr_copy_X.partition_D(sX)
            tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

            tvlayout_cX = cute.composition(cX, tv_layout)
            thr_coord = (tidx, (None, None))
            thr_cX = tvlayout_cX[thr_coord]

            tXrX = cute.make_fragment_like(tXgX)
            num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
            self._initialize_cluster(tidx, mbar_ptr, num_warps)

            is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
            tXpX = (
                predicate_k(thr_copy_X.partition_S(cX), limit=shape[1]) if not is_even_N else None
            )

            if tXcX[0][0] < shape[0]:
                cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)

            if cutlass.const_expr(not is_even_N):
                fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)

            current_max = -tXsX.element_type.inf
            current_argmax = Int32(0xFFFFFFFF)

            for i in cutlass.range_constexpr(thr_cX.shape[0]):
                for j in cutlass.range_constexpr(thr_cX.shape[1]):
                    col_idx = thr_cX[i, j][1]
                    linear_idx = i + j * thr_cX.shape[0]
                    element_value1 = x[linear_idx]
                    if element_value1 > current_max:
                        current_max = element_value1
                        current_argmax = Int32(col_idx)

            lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
            if cutlass.const_expr(self.use_redux):
                warp_max, warp_argmax = warp_argmax_redux(current_max, current_argmax)
            else:
                warp_max, warp_argmax = warp_reduce_argmax(current_max, current_argmax)

            if cutlass.const_expr(self.cluster_n == 1):
                warps_per_row = cute.size(reduction_buffer.shape[1])
                row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row

                if lane_idx == 0:
                    reduction_buffer[row_idx, col_idx, 0, 0] = warp_max
                    reduction_buffer[row_idx, col_idx, 0, 1] = warp_argmax.to(cutlass.Float32)

                cute.arch.barrier()
                block_reduce_max = -tXsX.element_type.inf
                block_reduce_argmax = Int32(0xFFFFFFFF)

                if lane_idx < warps_per_row:
                    block_reduce_max = reduction_buffer[row_idx, lane_idx, 0, 0]
                    block_reduce_argmax = reduction_buffer[row_idx, lane_idx, 0, 1].to(
                        cutlass.Int32
                    )

                if cutlass.const_expr(self.use_redux):
                    warp_max, warp_argmax = warp_argmax_redux(block_reduce_max, block_reduce_argmax)
                else:
                    warp_max, warp_argmax = warp_reduce_argmax(
                        block_reduce_max, block_reduce_argmax
                    )
            else:
                cute.arch.cluster_wait()
                warps_per_row, cluster_n = reduction_buffer.shape[1]
                cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
                rows_per_block, (warps_per_row, cluster_n), _, _ = reduction_buffer.shape
                row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row

                if warp_idx == 0:
                    with cute.arch.elect_one():
                        num_warps = rows_per_block * warps_per_row
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_ptr,
                            num_warps * cluster_n * 2 * reduction_buffer.element_type.width // 8,
                        )

                if lane_idx < cluster_n:
                    store_shared_remote(
                        warp_max,
                        elem_pointer(
                            reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster), 0, 0)
                        ),
                        mbar_ptr,
                        peer_cta_rank_in_cluster=lane_idx,
                    )
                    store_shared_remote(
                        warp_argmax.to(cutlass.Float32),
                        elem_pointer(
                            reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster), 0, 1)
                        ),
                        mbar_ptr,
                        peer_cta_rank_in_cluster=lane_idx,
                    )

                cute.arch.mbarrier_wait(mbar_ptr, phase=0)
                block_reduce_val = -tXsX.element_type.inf
                block_reduce_argmax = Int32(0xFFFFFFFF)
                num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)

                for i in cutlass.range_constexpr(num_iter):
                    idx = lane_idx + i * cute.arch.WARP_SIZE
                    if idx < cute.size(reduction_buffer, mode=[1]):
                        element_max = reduction_buffer[row_idx, idx, 0, 0]
                        element_argmax = reduction_buffer[row_idx, idx, 0, 1].to(cutlass.Int32)
                        if element_max > block_reduce_val:
                            block_reduce_val = element_max
                            block_reduce_argmax = element_argmax

                if cutlass.const_expr(self.use_redux):
                    warp_max, warp_argmax = warp_argmax_redux(block_reduce_val, block_reduce_argmax)
                else:
                    warp_max, warp_argmax = warp_reduce_argmax(
                        block_reduce_val, block_reduce_argmax
                    )

            row_idx = tXcX[0][0]
            warps_per_row = tv_layout.shape[0][0] // cute.arch.WARP_SIZE
            local_row_idx = row_idx - (bidx * tiler_mn[0])
            first_warp_for_row = local_row_idx * warps_per_row
            first_thread_for_row = first_warp_for_row * cute.arch.WARP_SIZE

            if (
                tidx == first_thread_for_row
                and row_idx < shape[0]
                and local_row_idx >= 0
                and local_row_idx < tiler_mn[0]
                and (self.cluster_n == 1 or bidy == 0)
            ):
                mO[local_row_idx, 0] = warp_max.to(mO.element_type)
                mO[local_row_idx, 1] = warp_argmax.to(mO.element_type)

    # ============================================================================
    # Compiled kernel cache and forward function
    # ============================================================================
    _argmax_compile_cache = {}

    # Minimum vocab size for the CuTE tiled kernel.
    _MIN_VOCAB_SIZE_FOR_CUTE_KERNEL = 256

    # The async copy requires 128-byte alignment:
    # Since we only support float32 currently, use 32.
    _VOCAB_SIZE_ALIGNMENT = 32

    def _should_use_torch_fallback(N: int, dtype: torch.dtype) -> bool:
        """Check if we should fall back to torch.max instead of CuTE kernel."""
        if dtype != torch.float32:
            return True
        if N < _MIN_VOCAB_SIZE_FOR_CUTE_KERNEL:
            return True
        if N % _VOCAB_SIZE_ALIGNMENT != 0:
            return True
        # Fall back on sm_120 - CUTLASS DSL JIT not well-supported for this setup
        from ..._utils import get_sm_version

        if get_sm_version() >= 120:
            return True
        return False

    def argmax(x: torch.Tensor) -> torch.Tensor:
        """
        Compute argmax along the last dimension of the input tensor.

        Args:
            x: Input tensor of shape (M, N)

        Returns:
            Output tensor of shape (M, 2) where:
            - Column 0: Maximum value in each row
            - Column 1: Index of maximum value in each row (argmax)
        """
        assert x.dim() == 2, "Input must be 2D"
        assert x.is_cuda, "Tensor must be on CUDA device"
        assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"

        M, N = x.shape

        if _should_use_torch_fallback(N, x.dtype):
            max_vals, max_indices = torch.max(x, dim=-1, keepdim=True)
            return torch.cat([max_vals, max_indices.to(x.dtype)], dim=-1)

        out = torch.empty((M, 2), dtype=x.dtype, device=x.device)
        dtype = torch2cute_dtype_map[x.dtype]

        def convert_from_dlpack(tensor):
            return from_dlpack(
                CUDAGraphCompatibleWrapper(tensor.detach()), assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        x_tensor = convert_from_dlpack(x)
        out_tensor = convert_from_dlpack(out)
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # Detect compute capability: use redux instructions only on Blackwell (sm_100f)
        # redux.sync.max.f32 is only supported on sm_100f
        from ..._utils import get_sm_version

        sm_version = get_sm_version()
        use_redux = sm_version >= 100 and sm_version < 120

        compile_key = (dtype, N, use_redux)
        if compile_key not in _argmax_compile_cache:
            argmax_kernel = ArgmaxKernel(dtype, N, use_redux=use_redux)
            _argmax_compile_cache[compile_key] = cute.compile(
                argmax_kernel, x_tensor, out_tensor, current_stream
            )

        _argmax_compile_cache[compile_key](x_tensor, out_tensor, current_stream)
        return out

else:
    # Fallback if CUTLASS DSL is not available
    def argmax(x: torch.Tensor) -> torch.Tensor:
        """Fallback argmax using PyTorch when CUTLASS DSL is not available."""
        max_vals, max_indices = torch.max(x, dim=-1, keepdim=True)
        return torch.cat([max_vals, max_indices.to(x.dtype)], dim=-1)

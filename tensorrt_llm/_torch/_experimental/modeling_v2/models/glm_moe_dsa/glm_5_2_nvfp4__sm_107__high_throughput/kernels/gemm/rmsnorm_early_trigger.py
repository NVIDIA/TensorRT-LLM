# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Adapted from FlashInfer 0.6.18.post1 (https://github.com/flashinfer-ai/flashinfer) under the
# Apache-2.0 license: RMSNormKernel, _get_compiled_rmsnorm_kernel and rmsnorm_cute from
# flashinfer/norm/kernels/rmsnorm.py, and the reduction and predicate helpers they use from
# flashinfer/norm/utils.py. The only change is where the kernel triggers its PDL dependents.
"""RMSNorm CuTe DSL kernel that triggers its PDL dependents at kernel entry.

FlashInfer's RMSNormKernel executes ``griddepcontrol.launch_dependents`` after its output
store, so a dependent grid launches only once the norm has finished. This copy executes it
first, before its own ``griddepcontrol.wait``: the dependent grid can launch, and run the work
it does before its wait, while this kernel still waits for its producer. Dependents must
execute ``griddepcontrol.wait`` before they read the output; that wait returns only after this
grid has completed and its stores are visible. The arithmetic is FlashInfer's, so outputs are
bit-identical to ``flashinfer.norm.rmsnorm`` where FlashInfer runs its CuTe DSL kernel.
"""

import functools
import math
import operator
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

COPY_BITS = 128  # 128-bit vectorized loads

_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


@functools.lru_cache(maxsize=16)
def get_sm_version(device=None) -> int:
    """Get the SM version of a CUDA device (e.g., 100 for SM100)."""
    if not torch.cuda.is_available():
        return 80
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map smem pointer to address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
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
    val: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store Float32 value to shared memory on a remote CTA in the cluster."""
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    """Get pointer to element at coordinate in tensor."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def warp_reduce(val, op, width: cutlass.Constexpr[int] = 32):
    """Reduce across threads in a warp using butterfly shuffle."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val


@cute.jit
def block_reduce_multirow(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    """Block reduction with 2D buffer (rows_per_block, warps_per_row).

    Each warp writes its partial sum to the row it belongs to, then
    lane 0..warps_per_row-1 read back and do a final warp reduction.
    """
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op)


@cute.jit
def cluster_reduce_multirow(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr,
    cluster_n: cutlass.Constexpr[int],
    init_val: Float32,
) -> Float32:
    """Cluster reduction across multiple CTAs using mbarrier.

    reduction_buffer has shape (rows_per_block, (warps_per_row, cluster_n)).
    Each warp sends its partial result to all CTAs in the cluster via
    st.async.shared::cluster, then every CTA reduces the collected values.
    """
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()

    rows_per_block = reduction_buffer.shape[0]
    warps_per_row = reduction_buffer.shape[1][0]

    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            expected_bytes = num_warps * cluster_n * 4
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expected_bytes)

    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )

    cute.arch.mbarrier_wait(mbar_ptr, phase=0)

    num_total = warps_per_row * cluster_n
    num_iter = cute.ceil_div(num_total, 32)

    block_reduce_val = init_val
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * 32
        if idx < num_total:
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])

    return warp_reduce(block_reduce_val, op)


@cute.jit
def row_reduce_sum_multirow(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
    mbar_ptr,
    cluster_n: cutlass.Constexpr[int],
) -> Float32:
    """Row reduction for sum with optional cluster support.

    When cluster_n == 1, uses block-level reduction with 2D buffer
    (rows_per_block, warps_per_row). When cluster_n > 1, uses cross-CTA
    cluster reduction with hierarchical buffer
    (rows_per_block, (warps_per_row, cluster_n)).
    """
    local_val = x.reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)

    warp_width = min(threads_per_row, 32)
    warp_val = warp_reduce(local_val, operator.add, width=warp_width)

    warps_per_row = max(threads_per_row // 32, 1)

    if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
        if cutlass.const_expr(cluster_n == 1):
            return block_reduce_multirow(warp_val, operator.add, reduction_buffer, Float32(0.0))
        else:
            return cluster_reduce_multirow(
                warp_val,
                operator.add,
                reduction_buffer,
                mbar_ptr,
                cluster_n,
                Float32(0.0),
            )
    else:
        return warp_val


@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking (2D tensors)."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(tXcX[(0, rest_v), 0, rest_k][1], limit)
    return tXpX


class RMSNormEarlyTriggerKernel:
    """
    RMSNorm Kernel using CuTe-DSL.

    Computes: output = input / sqrt(mean(input^2) + eps) * (weight + weight_bias)
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        weight_bias: float = 0.0,
        sm_version: int | None = None,
    ):
        self.dtype = dtype
        self.H = H
        self.weight_bias = weight_bias
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        self.cluster_n = self._compute_cluster_n(H, dtype, self.sm_version)
        self.H_per_cta = H // self.cluster_n

        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes

        h_align = self.H_per_cta & (-self.H_per_cta)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * dtype.width

        self.threads_per_row = self._compute_threads_per_row(self.H_per_cta)
        self.num_threads = self._compute_num_threads(self.H_per_cta)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1,
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1) // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        if self.copy_bits >= 32:
            tile_bytes = self.rows_per_block * self.cols_per_tile * elem_bytes
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.use_async_copy = tile_bytes <= props.shared_memory_per_block_optin // 2
        else:
            self.use_async_copy = False

    @staticmethod
    def _compute_cluster_n(H: int, dtype: cutlass.Numeric, sm_version: int) -> int:
        """Compute optimal cluster size based on H and device shared memory."""
        if sm_version < 90:
            return 1

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        max_smem_bytes = props.shared_memory_per_block_optin
        elem_size = dtype.width // 8

        for cluster_n in [1, 2, 4, 8, 16]:
            if H % cluster_n != 0:
                continue
            smem_needed = RMSNormEarlyTriggerKernel._estimate_smem_bytes(H, cluster_n, elem_size)
            if smem_needed <= max_smem_bytes:
                return cluster_n

        return 16

    @staticmethod
    def _estimate_smem_bytes(H: int, cluster_n: int, elem_size: int) -> int:
        """Estimate shared memory bytes for a given cluster configuration."""
        H_per_cta = H // cluster_n
        threads_per_row = RMSNormEarlyTriggerKernel._compute_threads_per_row(H_per_cta)
        num_threads = RMSNormEarlyTriggerKernel._compute_num_threads(H_per_cta)
        rows_per_block = num_threads // threads_per_row
        warps_per_row = max(threads_per_row // 32, 1)

        max_vec_size = COPY_BITS // 8 // elem_size
        h_align = H_per_cta & (-H_per_cta)
        vec_size = min(h_align, max_vec_size)
        num_vec_blocks = max(1, (H_per_cta // vec_size + threads_per_row - 1) // threads_per_row)
        cols_per_tile = vec_size * num_vec_blocks * threads_per_row

        tile_bytes = rows_per_block * cols_per_tile * elem_size

        if cluster_n == 1:
            return tile_bytes + rows_per_block * warps_per_row * 4
        else:
            return (
                tile_bytes + rows_per_block * warps_per_row * cluster_n * 4 + 8  # mbarrier
            )

    @staticmethod
    def _compute_threads_per_row(H: int) -> int:
        if H <= 64:
            return 8
        elif H <= 128:
            return 16
        elif H <= 3072:
            return 32
        elif H <= 6144:
            return 64
        elif H <= 16384:
            return 128
        else:
            return 256

    @staticmethod
    def _compute_num_threads(H: int) -> int:
        return 128 if H <= 16384 else 256

    @staticmethod
    def _make_tv_layout(threads_per_row, rows_per_block, vec_size, num_vec_blocks):
        """Create Thread-Value layout for multi-row coalesced vectorized access."""
        shape = (
            (threads_per_row, rows_per_block),
            (vec_size, num_vec_blocks),
        )
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    def _smem_size_in_bytes(self) -> int:
        if self.use_async_copy:
            tile_bytes = self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
        else:
            tile_bytes = 0

        if self.cluster_n == 1:
            reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        else:
            reduction_bytes = self.rows_per_block * self.warps_per_row * self.cluster_n * 4

        mbar_bytes = 8 if self.cluster_n > 1 else 0
        return tile_bytes + reduction_bytes + mbar_bytes

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = self._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        cluster_n = self.cluster_n

        self.kernel(mX, mW, mY, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL (SM90+ only): let the dependent grid launch now, then wait for the
        # previous kernel. The dependent reads mY only after its own wait, which
        # returns once this grid has completed.
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_rmem_tensor(tXgX.shape, tXgX.element_type)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_rmem_tensor(tWgW.shape, tWgW.element_type)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition output
        tXgO = thr_copy_O.partition_D(gY)
        tXrO = cute.make_rmem_tensor(tXgO.shape, tXgO.element_type)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + compute sum of squares =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        # Re-load x from shared memory to relieve register pressure.
        # Without this, x (up to 128 FP32 values/thread at large H) must
        # survive across the reduction + barrier, causing spills to local mem.
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mY.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)


@functools.cache
def _get_compiled_rmsnorm_early_trigger_kernel(
    dtype: torch.dtype,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    sm_version: int,
    contiguous: bool = True,
):
    """Get a compiled RMSNorm kernel using TVM-FFI.

    When contiguous=True, tensors are compiled with compact (dense) layouts for
    optimal codegen. When False, symbolic row strides are used to support
    arbitrary row strides at the cost of some performance.
    """
    cutlass_dtype = _TORCH_TO_CUTLASS_DTYPE[dtype]
    kernel_obj = RMSNormEarlyTriggerKernel(cutlass_dtype, H, weight_bias, sm_version=sm_version)

    # 64-bit M so row-index arithmetic (row * H) does not overflow when
    # M * H exceeds INT32_MAX.
    sym_m = cute.sym_int(64)

    if contiguous:
        elem_bytes = cutlass_dtype.width // 8
        tensor_align = math.gcd(128, H * elem_bytes)
        x_fake = cute.runtime.make_fake_compact_tensor(
            cutlass_dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
        )
        y_fake = cute.runtime.make_fake_compact_tensor(
            cutlass_dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
        )
    else:
        sym_row_stride_x = cute.sym_int64(divisibility=kernel_obj.vec_size)
        sym_row_stride_y = cute.sym_int64(divisibility=kernel_obj.vec_size)
        x_fake = cute.runtime.make_fake_tensor(
            cutlass_dtype, (sym_m, H), (sym_row_stride_x, 1), assumed_align=16
        )
        y_fake = cute.runtime.make_fake_tensor(
            cutlass_dtype, (sym_m, H), (sym_row_stride_y, 1), assumed_align=16
        )

    w_fake = cute.runtime.make_fake_compact_tensor(cutlass_dtype, (H,), assumed_align=16)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


def rmsnorm_early_trigger(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """RMSNorm of a 2D or 3D FP16/BF16 tensor into ``out`` (stride[-1] must be 1).

    Same contract and kernel selection as FlashInfer's ``rmsnorm_cute``: a compact
    kernel for contiguous tensors and a strided kernel otherwise. With ``enable_pdl``
    the kernel triggers its dependents at entry.
    """
    shape = input.shape
    H = shape[-1]

    if len(shape) == 3:
        M = shape[0] * shape[1]
        input_2d = input.reshape(M, H)
        out_2d = out.reshape(M, H)
    else:
        M = shape[0]
        input_2d = input
        out_2d = out

    is_contiguous = input_2d.is_contiguous() and out_2d.is_contiguous()
    # When M*H exceeds INT32_MAX, use the strided compile path: its row stride
    # is a dynamic int64 so the offset arithmetic widens to int64. The compact
    # path bakes the row stride in as a constexpr int and computes row*H in
    # int32, which overflows.
    if is_contiguous and M * H > 2**31 - 1:
        is_contiguous = False
    kernel = _get_compiled_rmsnorm_early_trigger_kernel(
        input.dtype,
        H,
        weight_bias,
        enable_pdl,
        get_sm_version(input.device),
        contiguous=is_contiguous,
    )
    kernel(input_2d, weight, out_2d, M, eps)


__all__ = [
    "RMSNormEarlyTriggerKernel",
    "rmsnorm_early_trigger",
]

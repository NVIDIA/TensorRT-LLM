# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Prepare scheduler for SM100 sparse attention.

The scheduler converts uneven CSR k2q row fanout into a flat worklist consumed
by sparse attention kernels. Each work item covers a contiguous q-index range
within one (head_kv, csr row) and carries the decoded batch/KV-block coordinate.
"""

from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, const_expr

from src.common import copy_utils, utils
from src.common.cute_dsl_utils import (
    assume_tensor_aligned,
    to_cute_tensor as to_cute_tensor_kvouter,
)


_PREPARE_COMPILE_CACHE: dict = {}


@dataclass
class SparseAttentionSchedule:
    enabled: bool
    scheduler_metadata: Optional[torch.Tensor]
    work_count: Optional[torch.Tensor]
    qsplit_indices: Optional[torch.Tensor] = None
    split_counts: Optional[torch.Tensor] = None
    target_q_per_cta: int = 0

    @property
    def work_capacity(self) -> int:
        return 0 if self.scheduler_metadata is None else int(self.scheduler_metadata.shape[0])


SparseSchedulePlan = SparseAttentionSchedule


class SparseAttentionScheduleModel:
    """Host-side helpers for sparse attention schedule sizing."""

    @staticmethod
    def _round_up(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    @staticmethod
    def _ceil_div(x: int, y: int) -> int:
        return (x + y - 1) // y

    def _target_q_per_cta(
        self,
        *,
        total_q: int,
        topk: int,
        head_kv: int,
        qhead_per_kv: int,
        device: torch.device,
        usable_SM_count: int = -1,
    ) -> int:
        num_sm = torch.cuda.get_device_properties(device).multi_processor_count
        if usable_SM_count > 0:
            num_sm = min(int(usable_SM_count), num_sm)
        q_tokens_per_group = 128 // qhead_per_kv
        total_refs_upper = total_q * topk * head_kv
        desired_work_items = max(num_sm * 2, 1)
        total_groups_upper = self._ceil_div(max(total_refs_upper, 1), q_tokens_per_group)
        target_groups_per_cta = min(
            512,
            max(1, self._ceil_div(total_groups_upper, desired_work_items)),
        )
        return target_groups_per_cta * q_tokens_per_group

    def balanced_target_q_per_cta(
        self,
        *,
        total_q: int,
        topk: int,
        blk_kv: int,
        head_kv: int,
        qhead_per_kv: int,
        device: torch.device,
        usable_SM_count: int = -1,
    ) -> int:
        q_tokens_per_group = 128 // qhead_per_kv
        occupancy_target = self._target_q_per_cta(
            total_q=total_q,
            topk=topk,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            device=device,
            usable_SM_count=usable_SM_count,
        )
        sink_balance_cap = max(q_tokens_per_group, int(topk) * int(blk_kv) * 2)
        target = min(max(occupancy_target, q_tokens_per_group), sink_balance_cap)
        return self._round_up(target, q_tokens_per_group)

    def flat_schedule_capacity(
        self,
        *,
        total_rows: int,
        total_q: int,
        topk: int,
        head_kv: int,
        target_q_per_cta: int,
    ) -> int:
        row_upper = max(total_rows, 0) * max(head_kv, 1)
        refs_upper = max(total_q, 0) * max(topk, 1) * max(head_kv, 1)
        split_upper = self._ceil_div(max(refs_upper, 1), max(target_q_per_cta, 1))
        return max(1, row_upper + split_upper)


SPARSE_SCHEDULE_MODEL = SparseAttentionScheduleModel()


class SparseAttentionPrepareFlatScheduleSm100:
    """Build a compact flat worklist by splitting each CSR row into chunks."""

    def __init__(
        self,
        *,
        num_threads: int = 128,
    ):
        if num_threads % 32 != 0:
            raise ValueError(f"num_threads must be a multiple of 32, got {num_threads}")
        self.num_threads = num_threads
        self.warps_per_cta = num_threads // 32

    @cute.jit
    def _emit_work(
        self,
        mSchedulerMetadata: cute.Tensor,
        work_idx: Int32,
        work_capacity: Int32,
        head_kv_idx: Int32,
        row_linear: Int32,
        q_begin: Int32,
        q_count: Int32,
        batch_idx: Int32,
        kv_block_idx: Int32,
    ):
        if work_idx < work_capacity:
            mSchedulerMetadata[work_idx, Int32(0)] = head_kv_idx
            mSchedulerMetadata[work_idx, Int32(1)] = row_linear
            mSchedulerMetadata[work_idx, Int32(2)] = q_begin
            mSchedulerMetadata[work_idx, Int32(3)] = q_count
            mSchedulerMetadata[work_idx, Int32(4)] = batch_idx
            mSchedulerMetadata[work_idx, Int32(5)] = kv_block_idx

    @cute.jit
    def _rows_in_batch(
        self,
        mCuSeqlensK: cute.Tensor,
        batch_idx: Int32,
        blk_kv: Int32,
    ) -> Int32:
        seqlen = mCuSeqlensK[batch_idx + Int32(1)] - mCuSeqlensK[batch_idx]
        return (seqlen + blk_kv - Int32(1)) // blk_kv

    @cute.jit
    def _rows_before_level(
        self,
        mCuSeqlensK: cute.Tensor,
        level: Int32,
        blk_kv: Int32,
    ) -> Int32:
        total = Int32(0)
        batch = mCuSeqlensK.shape[0] - Int32(1)
        for b in cutlass.range(batch, unroll=1):
            rows = self._rows_in_batch(mCuSeqlensK, b, blk_kv)
            total += cutlass.min(rows, level)
        return total

    @cute.jit
    def _max_rows_per_batch(
        self,
        mCuSeqlensK: cute.Tensor,
        blk_kv: Int32,
    ) -> Int32:
        max_rows = Int32(0)
        batch = mCuSeqlensK.shape[0] - Int32(1)
        for b in cutlass.range(batch, unroll=1):
            rows = self._rows_in_batch(mCuSeqlensK, b, blk_kv)
            max_rows = cutlass.max(max_rows, rows)
        return max_rows

    @cute.jit
    def _decode_sparse_row_linear(
        self,
        mCuSeqlensK: cute.Tensor,
        row_linear: Int32,
        blk_kv: Int32,
    ) -> tuple[Int32, Int32]:
        lo = Int32(0)
        hi = self._max_rows_per_batch(mCuSeqlensK, blk_kv)
        while lo < hi:
            mid = (lo + hi) // Int32(2)
            rows_before_next = self._rows_before_level(
                mCuSeqlensK,
                mid + Int32(1),
                blk_kv,
            )
            if rows_before_next <= row_linear:
                lo = mid + Int32(1)
            else:
                hi = mid

        level = lo
        offset = row_linear - self._rows_before_level(mCuSeqlensK, level, blk_kv)
        active_idx = Int32(0)
        batch_idx = Int32(0)
        found = Int32(0)
        batch = mCuSeqlensK.shape[0] - Int32(1)
        for b in cutlass.range(batch, unroll=1):
            if found == Int32(0):
                rows = self._rows_in_batch(mCuSeqlensK, b, blk_kv)
                if rows > level:
                    if active_idx == offset:
                        batch_idx = b
                        found = Int32(1)
                    active_idx += Int32(1)
        return batch_idx, level

    @cute.jit
    def __call__(
        self,
        mK2qCounts: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mSchedulerMetadata: cute.Tensor,
        mWorkCount: cute.Tensor,
        target_q_per_cta: Int32,
        work_capacity: Int32,
        num_heads_kv: Int32,
        blk_kv: Int32,
        stream: cuda.CUstream = None,
    ):
        if const_expr(mK2qCounts.element_type != Int32):
            raise TypeError("mK2qCounts must be Int32")
        if const_expr(mCuSeqlensK.element_type != Int32):
            raise TypeError("mCuSeqlensK must be Int32")
        if const_expr(mSchedulerMetadata.element_type != Int32):
            raise TypeError("mSchedulerMetadata must be Int32")
        if const_expr(mWorkCount.element_type != Int32):
            raise TypeError("mWorkCount must be Int32")
        mK2qCounts, mCuSeqlensK, mSchedulerMetadata, mWorkCount = [
            assume_tensor_aligned(t)
            for t in (mK2qCounts, mCuSeqlensK, mSchedulerMetadata, mWorkCount)
        ]
        total_rows = mK2qCounts.shape[1] - Int32(1)
        total_row_heads = total_rows * num_heads_kv
        grid_ctas = cute.ceil_div(total_row_heads, self.warps_per_cta)

        self.kernel(
            mK2qCounts,
            mCuSeqlensK,
            mSchedulerMetadata,
            mWorkCount,
            target_q_per_cta,
            work_capacity,
            num_heads_kv,
            total_rows,
            blk_kv,
        ).launch(
            grid=(grid_ctas,),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mK2qCounts: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mSchedulerMetadata: cute.Tensor,
        mWorkCount: cute.Tensor,
        target_q_per_cta: Int32,
        work_capacity: Int32,
        num_heads_kv: Int32,
        total_rows: Int32,
        blk_kv: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        block_idx = cute.arch.block_idx()[0]
        lane_idx = tidx % Int32(32)
        warp_idx = tidx // Int32(32)
        row_head_idx = block_idx * Int32(self.warps_per_cta) + warp_idx
        total_row_heads = total_rows * num_heads_kv

        head_kv_idx = Int32(0)
        row_linear = Int32(0)
        row_count = Int32(0)
        num_chunks = Int32(0)
        batch_idx = Int32(0)
        kv_block_idx = Int32(0)
        if row_head_idx < total_row_heads:
            row_linear = row_head_idx // num_heads_kv
            head_kv_idx = row_head_idx - row_linear * num_heads_kv
            if lane_idx == Int32(0):
                row_start = mK2qCounts[head_kv_idx, row_linear]
                row_end = mK2qCounts[head_kv_idx, row_linear + Int32(1)]
                row_count = row_end - row_start
                batch_idx, kv_block_idx = self._decode_sparse_row_linear(
                    mCuSeqlensK,
                    row_linear,
                    blk_kv,
                )
                if row_count > Int32(0):
                    num_chunks = (
                        row_count + target_q_per_cta - Int32(1)
                    ) // target_q_per_cta
        row_count = cute.arch.shuffle_sync(row_count, offset=0)
        num_chunks = cute.arch.shuffle_sync(num_chunks, offset=0)
        batch_idx = cute.arch.shuffle_sync(batch_idx, offset=0)
        kv_block_idx = cute.arch.shuffle_sync(kv_block_idx, offset=0)

        chunk_idx = lane_idx
        while chunk_idx < num_chunks:
            work_idx = cute.arch.atomic_add(
                mWorkCount.iterator.llvm_ptr,
                Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            q_begin = chunk_idx * target_q_per_cta
            q_count = cutlass.min(target_q_per_cta, row_count - q_begin)
            self._emit_work(
                mSchedulerMetadata,
                work_idx,
                work_capacity,
                head_kv_idx,
                row_linear,
                q_begin,
                q_count,
                batch_idx,
                kv_block_idx,
            )
            chunk_idx += Int32(32)


class SparseAttentionPrepareFwdSplitAtomicSm100:
    """Build packed q_idx/split_slot metadata for fwd K1 without K1 atomics."""

    def __init__(
        self,
        *,
        num_threads: int = 256,
    ):
        if num_threads % 32 != 0:
            raise ValueError(f"num_threads must be a multiple of 32, got {num_threads}")
        self.num_threads = num_threads

        @cute.struct
        class SharedStorage:
            sRow: cute.struct.MemRange[Int32, 3]

        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(
        self,
        mK2qCounts: cute.Tensor,
        mK2qIndices: cute.Tensor,
        mSchedulerMetadata: cute.Tensor,
        mWorkCount: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        work_capacity: Int32,
        max_seqlen_q: Int32,
        topk: Int32,
        stream: cuda.CUstream = None,
    ):
        if const_expr(mK2qCounts.element_type != Int32):
            raise TypeError("mK2qCounts must be Int32")
        if const_expr(mK2qIndices.element_type != Int32):
            raise TypeError("mK2qIndices must be Int32")
        if const_expr(mSchedulerMetadata.element_type != Int32):
            raise TypeError("mSchedulerMetadata must be Int32")
        if const_expr(mWorkCount.element_type != Int32):
            raise TypeError("mWorkCount must be Int32")
        if const_expr(mK2qQSplitIndices.element_type != Int32):
            raise TypeError("mK2qQSplitIndices must be Int32")
        if const_expr(mSplitCounts.element_type != Int32):
            raise TypeError("mSplitCounts must be Int32")
        if const_expr(mCuSeqlensQ.element_type != Int32):
            raise TypeError("mCuSeqlensQ must be Int32")
        (
            mK2qCounts,
            mK2qIndices,
            mSchedulerMetadata,
            mWorkCount,
            mK2qQSplitIndices,
            mSplitCounts,
            mCuSeqlensQ,
        ) = [
            assume_tensor_aligned(t)
            for t in (
                mK2qCounts,
                mK2qIndices,
                mSchedulerMetadata,
                mWorkCount,
                mK2qQSplitIndices,
                mSplitCounts,
                mCuSeqlensQ,
            )
        ]
        self.kernel(
            mK2qCounts,
            mK2qIndices,
            mSchedulerMetadata,
            mWorkCount,
            mK2qQSplitIndices,
            mSplitCounts,
            mCuSeqlensQ,
            max_seqlen_q,
            topk,
        ).launch(
            grid=(work_capacity,),
            block=[self.num_threads, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mK2qCounts: cute.Tensor,
        mK2qIndices: cute.Tensor,
        mSchedulerMetadata: cute.Tensor,
        mWorkCount: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        max_seqlen_q: Int32,
        topk: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        block_idx = cute.arch.block_idx()[0]
        if block_idx < mWorkCount[Int32(0)]:
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            sRow = storage.sRow.get_tensor(cute.make_layout((3,)))
            head_kv_idx = mSchedulerMetadata[block_idx, Int32(0)]
            row_linear = mSchedulerMetadata[block_idx, Int32(1)]
            q_begin = mSchedulerMetadata[block_idx, Int32(2)]
            q_count = mSchedulerMetadata[block_idx, Int32(3)]
            batch_idx_t0 = mSchedulerMetadata[block_idx, Int32(4)]

            if tidx == Int32(0):
                row_start_t0 = mK2qCounts[head_kv_idx, row_linear] + q_begin
                sRow[0] = row_start_t0
                sRow[1] = q_count
                sRow[2] = batch_idx_t0
            cute.arch.barrier()
            row_start = sRow[0]
            row_count = sRow[1]
            batch_idx = sRow[2]
            qi = tidx
            while qi < row_count:
                edge = row_start + qi
                q_idx = mK2qIndices[head_kv_idx, edge]
                if q_idx >= Int32(0) and q_idx < max_seqlen_q:
                    q_abs = mCuSeqlensQ[batch_idx] + q_idx
                    split_ptr = utils.elem_pointer(
                        mSplitCounts,
                        (q_abs, head_kv_idx),
                    )
                    split_slot = copy_utils.atomic_add_i32(split_ptr)
                    if split_slot < topk:
                        mK2qQSplitIndices[head_kv_idx, edge] = (
                            q_idx | ((split_slot & Int32(0xFF)) << Int32(24))
                        )
                qi += Int32(self.num_threads)


def _get_sparse_prepare_fwd_split_atomic(
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    k2q_qsplit_indices: torch.Tensor,
    split_counts: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    work_capacity: int,
    max_seqlen_q: int,
    topk: int,
):
    key = (
        "sparse_prepare_fwd_split_atomic_sm100_csr_varlen",
    )
    if key not in _PREPARE_COMPILE_CACHE:
        from src.common.aot_cache import try_load_aot, save_aot

        loaded = try_load_aot(key)
        if loaded is not None:
            _PREPARE_COMPILE_CACHE[key] = loaded
        else:
            kernel = SparseAttentionPrepareFwdSplitAtomicSm100()
            _PREPARE_COMPILE_CACHE[key] = cute.compile(
                kernel,
                to_cute_tensor_kvouter(k2q_row_ptr),
                to_cute_tensor_kvouter(k2q_q_indices),
                to_cute_tensor_kvouter(scheduler_metadata),
                to_cute_tensor_kvouter(work_count),
                to_cute_tensor_kvouter(k2q_qsplit_indices),
                to_cute_tensor_kvouter(split_counts),
                to_cute_tensor_kvouter(cu_seqlens_q),
                Int32(work_capacity),
                Int32(max_seqlen_q),
                Int32(topk),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
            save_aot(key, _PREPARE_COMPILE_CACHE[key])
    return _PREPARE_COMPILE_CACHE[key]


def _get_sparse_prepare_flat_schedule(
    k2q_row_ptr: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    target_q_per_cta: int,
    scheduler_metadata_capacity: int,
    head_kv: int,
    blk_kv: int,
):
    key = (
        "sparse_prepare_flat_schedule_sm100_csr_varlen",
    )
    if key not in _PREPARE_COMPILE_CACHE:
        from src.common.aot_cache import try_load_aot, save_aot

        loaded = try_load_aot(key)
        if loaded is not None:
            _PREPARE_COMPILE_CACHE[key] = loaded
        else:
            kernel = SparseAttentionPrepareFlatScheduleSm100()
            _PREPARE_COMPILE_CACHE[key] = cute.compile(
                kernel,
                to_cute_tensor_kvouter(k2q_row_ptr),
                to_cute_tensor_kvouter(cu_seqlens_k),
                to_cute_tensor_kvouter(scheduler_metadata),
                to_cute_tensor_kvouter(work_count),
                Int32(target_q_per_cta),
                Int32(scheduler_metadata_capacity),
                Int32(head_kv),
                Int32(blk_kv),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
            save_aot(key, _PREPARE_COMPILE_CACHE[key])
    return _PREPARE_COMPILE_CACHE[key]


def prepare_sparse_flat_schedule(
    *,
    k2q_row_ptr: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_q: int,
    topk: int,
    blk_kv: int,
    head_kv: int,
    qhead_per_kv: int,
    device: torch.device,
    enabled: bool,
    usable_SM_count: int = -1,
) -> SparseSchedulePlan:
    if not enabled:
        return SparseSchedulePlan(enabled=False, scheduler_metadata=None, work_count=None)

    total_rows = int(k2q_row_ptr.shape[1] - 1)
    if total_rows <= 0 or head_kv <= 0:
        return SparseSchedulePlan(enabled=False, scheduler_metadata=None, work_count=None)
    if cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_k must be torch.int32")

    target_q_per_cta = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta(
        total_q=total_q,
        topk=topk,
        blk_kv=blk_kv,
        head_kv=head_kv,
        qhead_per_kv=qhead_per_kv,
        device=device,
        usable_SM_count=usable_SM_count,
    )
    scheduler_metadata_capacity = SPARSE_SCHEDULE_MODEL.flat_schedule_capacity(
        total_rows=total_rows,
        total_q=total_q,
        topk=topk,
        head_kv=head_kv,
        target_q_per_cta=target_q_per_cta,
    )
    scheduler_metadata = torch.empty(
        (scheduler_metadata_capacity, 6),
        dtype=torch.int32,
        device=device,
    )
    work_count = torch.zeros((1,), dtype=torch.int32, device=device)
    scheduler_metadata.zero_()

    compiled_prepare = _get_sparse_prepare_flat_schedule(
        k2q_row_ptr,
        cu_seqlens_k,
        scheduler_metadata,
        work_count,
        target_q_per_cta,
        scheduler_metadata_capacity,
        head_kv,
        blk_kv,
    )
    with torch.cuda.nvtx.range("SparseAttention_PrepareFlatSchedule"):
        compiled_prepare(
            k2q_row_ptr,
            cu_seqlens_k,
            scheduler_metadata,
            work_count,
            target_q_per_cta,
            scheduler_metadata_capacity,
            head_kv,
            blk_kv,
        )

    return SparseSchedulePlan(
        enabled=True,
        scheduler_metadata=scheduler_metadata,
        work_count=work_count,
        target_q_per_cta=target_q_per_cta,
    )

def prepare_sparse_fwd_schedule_and_split(
    *,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    k2q_qsplit_indices: torch.Tensor,
    split_counts: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_q: int,
    max_seqlen_q: int,
    topk: int,
    head_kv: int,
    qhead_per_kv: int,
    blk_kv: int,
    device: torch.device,
    enabled: bool,
    usable_SM_count: int = -1,
) -> SparseSchedulePlan:
    plan = prepare_sparse_fwd_schedule(
        k2q_row_ptr=k2q_row_ptr,
        cu_seqlens_k=cu_seqlens_k,
        total_q=total_q,
        topk=topk,
        head_kv=head_kv,
        qhead_per_kv=qhead_per_kv,
        blk_kv=blk_kv,
        device=device,
        enabled=enabled,
        usable_SM_count=usable_SM_count,
    )
    if not plan.enabled:
        return plan
    if plan.scheduler_metadata is None or plan.work_count is None:
        raise RuntimeError("fwd GPU schedule requires metadata")
    if topk > 255:
        raise ValueError(f"packed qsplit metadata supports topK <= 255, got {topk}")
    if max_seqlen_q >= (1 << 24):
        raise ValueError(
            "packed qsplit metadata supports batch-local q_idx < 2^24, "
            f"got max_seqlen_q={max_seqlen_q}"
        )
    if k2q_qsplit_indices.shape != k2q_q_indices.shape:
        raise ValueError("k2q_qsplit_indices shape must match k2q_q_indices")
    if split_counts.dtype != torch.int32 or k2q_qsplit_indices.dtype != torch.int32:
        raise TypeError("split metadata tensors must be torch.int32")
    if split_counts.shape != (total_q, head_kv):
        raise ValueError(
            f"split_counts must have shape ({total_q}, {head_kv}), got {tuple(split_counts.shape)}"
        )
    if cu_seqlens_q.dtype != torch.int32:
        raise TypeError("cu_seqlens_q must be torch.int32")
    if cu_seqlens_q.ndim != 1 or not cu_seqlens_q.is_contiguous():
        raise ValueError("cu_seqlens_q must be a contiguous rank-1 tensor")
    if cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_k must be torch.int32")

    with torch.cuda.nvtx.range("SparseAttention_InitFwdSplitState"):
        split_counts.zero_()

    compiled_split = _get_sparse_prepare_fwd_split_atomic(
        k2q_row_ptr,
        k2q_q_indices,
        plan.scheduler_metadata,
        plan.work_count,
        k2q_qsplit_indices,
        split_counts,
        cu_seqlens_q,
        plan.work_capacity,
        max_seqlen_q,
        topk,
    )
    with torch.cuda.nvtx.range("SparseAttention_PrepareFwdSplit_Atomic"):
        compiled_split(
            k2q_row_ptr,
            k2q_q_indices,
            plan.scheduler_metadata,
            plan.work_count,
            k2q_qsplit_indices,
            split_counts,
            cu_seqlens_q,
            plan.work_capacity,
            max_seqlen_q,
            topk,
        )
    plan.qsplit_indices = k2q_qsplit_indices
    plan.split_counts = split_counts
    return plan


def prepare_sparse_fwd_schedule(
    *,
    k2q_row_ptr: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_q: int,
    topk: int,
    blk_kv: int,
    head_kv: int,
    qhead_per_kv: int,
    device: torch.device,
    enabled: bool,
    usable_SM_count: int = -1,
) -> SparseSchedulePlan:
    return prepare_sparse_flat_schedule(
        k2q_row_ptr=k2q_row_ptr,
        cu_seqlens_k=cu_seqlens_k,
        total_q=int(total_q),
        topk=int(topk),
        blk_kv=int(blk_kv),
        head_kv=int(head_kv),
        qhead_per_kv=int(qhead_per_kv),
        device=device,
        enabled=bool(enabled),
        usable_SM_count=int(usable_SM_count),
    )

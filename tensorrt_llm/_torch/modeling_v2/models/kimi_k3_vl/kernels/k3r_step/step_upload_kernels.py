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
"""Triton kernels for the small uploads and scatters around a decode step.

Between two decode CUDA-graph replays the execution stream runs a chain of small
operations that all wait for the previous replay: block-table copies, state-index
uploads and the speculative sampler's store scatters. Each kernel here does the work
of one or more of them with the same results, behind a switch that defaults to off:

* ``TRTLLM_KV_BLOCK_OFFSETS_TRITON=1`` (``2`` is accepted with the same meaning):
  ``KVCacheManagerV2.copy_batch_block_offsets`` uses
  :func:`copy_batch_block_offsets_from_host`, the C++
  ``copyBatchBlockOffsetsToDeviceKernel`` split over many CTAs (the C++ kernel runs one
  CTA per pool and sequence, bound by host-read latency).
* ``TRTLLM_SPEC_SAMPLER_FUSED_SCATTER=1``: where the one-model speculative sampler has
  no ``spec_step_copies`` slot scatter (that one is validated on SM100 only), it writes
  its four slot-indexed stores with :func:`scatter_spec_store` and keeps the device slot
  tensor while the batch's slots do not change.
* ``TRTLLM_STATE_INDEX_COPY_KERNEL=1``: when ``MambaHybridCacheManagerV2`` uploads its
  state indices or dummy-request mask, it writes both with
  :func:`copy_small_pinned_pair` (one SM kernel) instead of copy-engine copies, whose
  engine hand-offs delay the execution stream.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

BLOCK_OFFSETS_TRITON_ENV = "TRTLLM_KV_BLOCK_OFFSETS_TRITON"
SPEC_SAMPLER_FUSED_SCATTER_ENV = "TRTLLM_SPEC_SAMPLER_FUSED_SCATTER"
STATE_INDEX_COPY_KERNEL_ENV = "TRTLLM_STATE_INDEX_COPY_KERNEL"

_BLOCK_OFFSETS_TRITON = True  # TensorRT-LLM local change: 1007804's value (=2); the target reads no environment.
_SPEC_SAMPLER_FUSED_SCATTER = True  # TensorRT-LLM local change: 1007804's value (=1); the target reads no environment.
_STATE_INDEX_COPY_KERNEL = True  # TensorRT-LLM local change: 1007804's value (=1); the target reads no environment.

# Block-table entries per program of the block-offset copy.
_BLOCK_OFFSETS_BLOCK = 1024
# Above this many block-table entries per call the copy is bound by host-link
# bandwidth and the C++ kernel is as fast (GB300: 2 pools x 976 blocks or
# 1 pool x 15584 blocks at 2-32 sequences: 2.5-14.6 us vs 3.5-20.6 us; at
# 128-256 sequences of 15584 blocks both take 45-85 us).
_BLOCK_OFFSETS_TRITON_MAX_ENTRIES = 1 << 20


def _cdiv(a: int, b: int) -> int:
    # Plain integer ceil-div: triton.cdiv is a JIT function, slow to call from Python.
    return -(-a // b)


def block_offsets_triton_enabled() -> bool:
    """Whether ``TRTLLM_KV_BLOCK_OFFSETS_TRITON`` selects the Triton block-offset copy."""
    return _BLOCK_OFFSETS_TRITON


def spec_sampler_fused_scatter_enabled() -> bool:
    """Whether ``TRTLLM_SPEC_SAMPLER_FUSED_SCATTER=1`` selects the one-kernel store scatter."""
    return _SPEC_SAMPLER_FUSED_SCATTER


def state_index_copy_kernel_enabled() -> bool:
    """Whether ``TRTLLM_STATE_INDEX_COPY_KERNEL=1`` selects the one-kernel state-index upload."""
    return _STATE_INDEX_COPY_KERNEL


@triton.jit
def _block_offsets_rows(
    src_ptr,
    dst_ptr,
    copy_idx_ptr,
    scales_ptr,
    kv_offset_ptr,
    src_max_seqs,
    dst_max_seqs,
    num_blocks,
    pool,
    seq,
    chunk,
    BLOCK: tl.constexpr,
):
    # dst[pool, seq, 0, :] = scale * src[pool, copy_idx[seq], 0, :]
    # dst[pool, seq, 1, :] = scale * src[pool, copy_idx[seq], 0, :] + kv_offset
    # with BAD_PAGE_INDEX (-1) mapped to 0 in both halves.
    row = tl.load(copy_idx_ptr + seq).to(tl.int64)
    scale = tl.load(scales_ptr + pool)
    kv_offset = tl.load(kv_offset_ptr + pool)
    pool64 = pool.to(tl.int64)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_blocks
    src_row = src_ptr + (pool64 * src_max_seqs + row) * 2 * num_blocks
    val = tl.load(src_row + offs, mask=mask, other=0)
    bad = val == -1
    k = tl.where(bad, 0, scale * val)
    v = tl.where(bad, 0, scale * val + kv_offset)
    dst_row = dst_ptr + (pool64 * dst_max_seqs + seq) * 2 * num_blocks
    tl.store(dst_row + offs, k, mask=mask)
    tl.store(dst_row + num_blocks + offs, v, mask=mask)


@triton.jit(
    do_not_specialize=["src_max_seqs", "dst_max_seqs", "num_blocks", "num_seqs", "num_chunks"]
)
def _copy_block_offsets_kernel(
    src_ptr,
    dst_ptr,
    copy_idx_ptr,
    scales_ptr,
    kv_offset_ptr,
    src_max_seqs,
    dst_max_seqs,
    num_blocks,
    num_seqs,
    num_chunks,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = pid % num_chunks
    seq = (pid // num_chunks) % num_seqs
    pool = pid // (num_chunks * num_seqs)
    _block_offsets_rows(
        src_ptr,
        dst_ptr,
        copy_idx_ptr,
        scales_ptr,
        kv_offset_ptr,
        src_max_seqs,
        dst_max_seqs,
        num_blocks,
        pool,
        seq,
        chunk,
        BLOCK,
    )


def block_offsets_copy_job(
    host_offsets: torch.Tensor,
    dst: torch.Tensor,
    copy_idx: torch.Tensor,
    index_scales: torch.Tensor,
    kv_offset: torch.Tensor,
) -> Optional[tuple]:
    """Kernel arguments of one :func:`copy_batch_block_offsets_from_host` call, or None.

    None when the Triton copy does not apply (see
    :func:`copy_batch_block_offsets_from_host`); otherwise ``(src, dst, copy_idx, scales,
    kv_offset, src_max_seqs, dst_max_seqs, num_blocks, num_seqs, num_chunks,
    num_programs)`` with ``num_programs == 0`` when there is nothing to copy.
    """
    if host_offsets.dim() != 4 or dst.dim() != 4:
        return None
    num_pools, src_max_seqs, kv_factor, num_blocks = host_offsets.shape
    num_seqs = copy_idx.shape[0]
    host_tensors = (host_offsets, copy_idx, index_scales, kv_offset)
    if not (
        kv_factor == 2
        and dst.shape[2] == 2
        and dst.shape[3] == num_blocks
        and dst.shape[0] >= num_pools
        and dst.shape[1] >= num_seqs
        and dst.is_cuda
        and dst.is_contiguous()
        and dst.dtype == torch.int32
        and all(
            t.dtype == torch.int32 and t.is_pinned() and t.is_contiguous() for t in host_tensors
        )
        and index_scales.numel() >= num_pools
        and kv_offset.numel() >= num_pools
    ):
        return None
    if num_pools * num_seqs * num_blocks > _BLOCK_OFFSETS_TRITON_MAX_ENTRIES:
        return None
    num_chunks = _cdiv(num_blocks, _BLOCK_OFFSETS_BLOCK) if num_blocks else 0
    num_programs = num_pools * num_seqs * num_chunks
    return (
        host_offsets,
        dst,
        copy_idx,
        index_scales,
        kv_offset,
        src_max_seqs,
        dst.shape[1],
        num_blocks,
        num_seqs,
        num_chunks,
        num_programs,
    )


def copy_batch_block_offsets_from_host(
    host_offsets: torch.Tensor,
    dst: torch.Tensor,
    copy_idx: torch.Tensor,
    index_scales: torch.Tensor,
    kv_offset: torch.Tensor,
    stream: Optional[torch.cuda.Stream] = None,
) -> bool:
    """Triton version of ``copy_batch_block_offsets_to_device``.

    ``host_offsets`` is the pinned ``[num_pools, max_seqs, 2, max_blocks]`` host table and
    ``copy_idx`` / ``index_scales`` / ``kv_offset`` are pinned int32 host tensors; all are
    read when the kernel runs, as in the C++ kernel. Writes pools ``[0, num_pools)`` and
    sequences ``[0, len(copy_idx))`` of ``dst``, on ``stream`` (the current stream if
    None). Returns False (nothing launched) when the tensors do not have that layout or
    the copy is large enough for the C++ kernel to be as fast, so the caller uses the C++
    kernel instead.
    """
    job = block_offsets_copy_job(host_offsets, dst, copy_idx, index_scales, kv_offset)
    if job is None:
        return False
    if job[-1] > 0:
        if stream is None or stream == torch.cuda.current_stream():
            _copy_block_offsets_kernel[(job[-1],)](*job[:-1], BLOCK=_BLOCK_OFFSETS_BLOCK)
        else:
            with torch.cuda.stream(stream):
                _copy_block_offsets_kernel[(job[-1],)](*job[:-1], BLOCK=_BLOCK_OFFSETS_BLOCK)
    return True


@triton.jit(
    do_not_specialize=[
        "width_new",
        "width_next",
        "width_draft",
        "num_slots",
        "s_new_r",
        "s_new_c",
        "s_next_r",
        "s_next_c",
        "s_draft_r",
        "s_draft_c",
        "s_lens",
    ]
)
def _spec_store_scatter_kernel(
    slots_ptr,
    store_new,
    store_next,
    store_lens,
    store_draft,
    src_new,
    src_next,
    src_lens,
    src_draft,
    width_new,
    width_next,
    width_draft,
    num_slots,
    s_new_r,
    s_new_c,
    s_next_r,
    s_next_c,
    s_draft_r,
    s_draft_c,
    s_lens,
    BLOCK: tl.constexpr,
):
    # Request p (program p) with slot s = slots[p]:
    #   store_new[t, s] = src_new[p, t]      t < width_new   (store_new is [width_new, num_slots])
    #   store_next[t, s] = src_next[p, t]    t < width_next
    #   store_lens[s] = src_lens[p]
    #   store_draft[s, j] = src_draft[p, j]  j < width_draft (store_draft is [num_slots, width_draft])
    p = tl.program_id(0)
    slot = tl.load(slots_ptr + p).to(tl.int64)
    p64 = p.to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < width_new
    val = tl.load(src_new + p64 * s_new_r + offs * s_new_c, mask=m)
    tl.store(store_new + offs * num_slots + slot, val, mask=m)
    m = offs < width_next
    val = tl.load(src_next + p64 * s_next_r + offs * s_next_c, mask=m)
    tl.store(store_next + offs * num_slots + slot, val, mask=m)
    tl.store(store_lens + slot, tl.load(src_lens + p64 * s_lens))
    m = offs < width_draft
    val = tl.load(src_draft + p64 * s_draft_r + offs * s_draft_c, mask=m)
    tl.store(store_draft + slot * width_draft + offs, val, mask=m)


def scatter_spec_store(
    slots: torch.Tensor,
    store_new: torch.Tensor,
    store_next: torch.Tensor,
    store_lens: torch.Tensor,
    store_draft: torch.Tensor,
    src_new: torch.Tensor,
    src_next: torch.Tensor,
    src_lens: torch.Tensor,
    src_draft: torch.Tensor,
) -> bool:
    """The four slot scatters of the one-model speculative sampler's stores in one kernel.

    Same result as ``store_new.T.index_copy_(0, slots, src_new)``,
    ``store_next.T.index_copy_(0, slots, src_next)``, ``store_lens.index_copy_(0, slots,
    src_lens)`` and ``store_draft.index_copy_(0, slots, src_draft)`` for unique slots, with
    ``store_new`` / ``store_next`` the ``[width, num_slots]`` and ``store_draft`` the
    ``[num_slots, width]`` contiguous store buffers and the sources ``[n, width]`` / ``[n]``.
    Returns False (nothing launched) when the shapes or dtypes do not match that layout.
    """
    n = slots.shape[0]
    stores = (store_new, store_next, store_lens, store_draft)
    srcs = (src_new, src_next, src_lens, src_draft)
    num_slots = store_lens.shape[0]
    if not (
        slots.is_cuda
        and slots.dim() == 1
        and all(t.is_cuda and t.is_contiguous() for t in stores)
        and all(s.is_cuda and s.dtype == t.dtype for s, t in zip(srcs, stores))
        and store_new.dim() == 2
        and store_next.dim() == 2
        and store_draft.dim() == 2
        and store_lens.dim() == 1
        and store_new.shape[1] == num_slots
        and store_next.shape[1] == num_slots
        and store_draft.shape[0] == num_slots
        and src_new.dim() == 2
        and src_next.dim() == 2
        and src_draft.dim() == 2
        and src_lens.dim() == 1
        and tuple(src_new.shape) == (n, store_new.shape[0])
        and tuple(src_next.shape) == (n, store_next.shape[0])
        and tuple(src_draft.shape) == (n, store_draft.shape[1])
        and src_lens.shape[0] == n
    ):
        return False
    if n == 0:
        return True
    width = max(store_new.shape[0], store_next.shape[0], store_draft.shape[1], 1)
    _spec_store_scatter_kernel[(n,)](
        slots,
        store_new,
        store_next,
        store_lens,
        store_draft,
        src_new,
        src_next,
        src_lens,
        src_draft,
        store_new.shape[0],
        store_next.shape[0],
        store_draft.shape[1],
        num_slots,
        src_new.stride(0),
        src_new.stride(1),
        src_next.stride(0),
        src_next.stride(1),
        src_draft.stride(0),
        src_draft.stride(1),
        src_lens.stride(0),
        BLOCK=triton.next_power_of_2(width),
    )
    return True


@triton.jit(do_not_specialize=["n0", "n1"])
def _small_copy_pair_kernel(dst0, src0, n0, dst1, src1, n1, BLOCK: tl.constexpr):
    # Program 0 copies n0 bytes src0 -> dst0, program 1 copies n1 bytes src1 -> dst1.
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        mask = offs < n0
        tl.store(dst0 + offs, tl.load(src0 + offs, mask=mask), mask=mask)
    else:
        mask = offs < n1
        tl.store(dst1 + offs, tl.load(src1 + offs, mask=mask), mask=mask)


_SMALL_COPY_MAX_BYTES = 4096


def copy_small_pinned_pair(
    dst0: torch.Tensor, src0: torch.Tensor, dst1: torch.Tensor, src1: torch.Tensor
) -> bool:
    """``dst0.copy_(src0, non_blocking=True); dst1.copy_(src1, non_blocking=True)`` as one
    kernel on the current stream, for small contiguous pinned host sources.

    The kernel reads the host buffers over the host link when it runs, as the copy engine
    reads them when the DMA runs. Unlike a ``copy_``, the kernel's read is not recorded
    with the pinned-memory allocator: the caller keeps the sources alive until the kernel
    has completed. Returns False (nothing launched) when a pair does not qualify.
    """
    pairs = ((dst0, src0), (dst1, src1))
    for dst, src in pairs:
        if not (
            dst.is_cuda
            and src.is_pinned()
            and dst.is_contiguous()
            and src.is_contiguous()
            and dst.dtype == src.dtype
            and dst.numel() == src.numel()
            and dst.numel() * dst.element_size() <= _SMALL_COPY_MAX_BYTES
        ):
            return False
    nbytes = [dst.numel() * dst.element_size() for dst, _ in pairs]
    block = triton.next_power_of_2(max(max(nbytes), 16))
    _small_copy_pair_kernel[(2,)](
        dst0.view(torch.uint8),
        src0.view(torch.uint8),
        nbytes[0],
        dst1.view(torch.uint8),
        src1.view(torch.uint8),
        nbytes[1],
        BLOCK=block,
    )
    return True

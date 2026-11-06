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
"""One-launch upload of a decode step's input buffers (TRTLLM_GEN_HOST_BUNDLE=1).

Input preparation for a CUDA-graph decode step fills a few dozen small device buffers
from host data (positions, gather ids, KV lengths, helix offsets, sampling seeds, ...)
and, under the overlap scheduler, gathers the previous step's tokens and accepted
lengths on the device. Issued one by one, these copies and gathers form a chain of
~40 tiny operations that runs between two graph replays, because the buffers they
write are still read by the previous replay. ``InputCommit`` collects the host values
into a pinned arena while the step is being prepared and writes every destination
with one kernel (reading the arena directly over the host link) right before the
replay. The overlap-scheduler gathers run in the same kernel.

Contract for staged copies: between ``stage`` and ``commit`` no device operation may
read or write a staged destination. Call sites stage only buffers that the input
preparation itself does not read back on the device; the device-side gathers that do
read them (previous-step tokens) are part of the commit kernel.

The block-offset copy of ``KVCacheManagerV2`` has its own kernel here
(``copy_batch_block_offsets_from_host``): same semantics as the C++
``copyBatchBlockOffsetsToDeviceKernel`` (reads the pinned host table when it runs),
but split over many CTAs instead of one CTA per (pool, sequence).

Engagement is counted, not assumed: ``bundle_census`` records, on live decode steps only
(engine warm-up never counts), the commits taken and the steps that could not use the commit
(with the reason), and the launches and fallbacks of the other parts. It logs one
``InputCommit census`` line after 100, 1000, ... live steps and at engine cleanup.
"""

import os
import threading
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger


def _env_int(name: str, default: int) -> int:
    """An integer environment knob; a malformed value is reported and ignored."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not an integer; using {default}")
        return default


# One switch (default off) for everything that shortens the chain of tiny operations between
# two CUDA-graph decode replays: the one-launch input commit, the Triton block-offset copy
# (target and draft tables in one launch), the one-kernel upload of the mamba state indices
# and the one-kernel store scatter of the one-model spec sampler. All of them write the same
# bytes as the operations they replace. The bundle was measured on tensor-parallel GEN
# workers only, so the engine switches it off with attention DP (``disable_gen_host_bundle``);
# set the variable for the GEN role only. A configuration a part does not cover (for example
# the FP4 MLA pools for the input commit) keeps the original path of that part, and the census
# (``bundle_census``) logs, per part, how often it engaged on live decode steps.
_GEN_HOST_BUNDLE = os.environ.get("TRTLLM_GEN_HOST_BUNDLE", "0") == "1"
# Debug aid: after each of the first N real commits (-1: all; engine warm-up commits do not
# count), synchronize and compare every destination with the staged bytes / a torch reference
# of the overlap gathers (raises on a mismatch; logs once when the first N commits have matched).
_VERIFY_COMMITS = _env_int("TRTLLM_GEN_HOST_BUNDLE_VERIFY", 0)


class _ActiveCommit(threading.local):
    """The commit of the step being prepared on the calling thread."""

    commit: Optional["InputCommit"] = None


_ACTIVE = _ActiveCommit()

# torch dtype -> (numpy dtype, element size) of the staged values.
_TORCH_TO_NUMPY = {
    torch.int32: (np.int32, 4),
    torch.int64: (np.int64, 8),
    torch.bool: (np.bool_, 1),
    torch.float32: (np.float32, 4),
    torch.uint8: (np.uint8, 1),
    torch.int8: (np.int8, 1),
}

# Bytes copied by one program of the commit kernel.
_COPY_CHUNK = 512
# Elements per program of the overlap fixup.
_FIX_BLOCK = 256
# Block-table entries per program of the block-offset copy.
_BLOCK_OFFSETS_BLOCK = 1024
# Above this many block-table entries per call the copy is bound by host-link
# bandwidth, not by per-CTA read latency, and the C++ kernel is as fast.
_BLOCK_OFFSETS_TRITON_MAX_ENTRIES = 1 << 20


def _cdiv(a: int, b: int) -> int:
    # Plain integer ceil-div: triton.cdiv is a JIT function, slow to call from Python.
    return -(-a // b)


def gen_host_bundle_kernels_enabled() -> bool:
    return _GEN_HOST_BUNDLE


def disable_gen_host_bundle(reason: str) -> None:
    """Switch the bundle off in this process: a configuration it was not measured in.

    Call before the parts are created (engine construction); every part asks
    ``gen_host_bundle_kernels_enabled`` when it is used or created, so all of them then keep
    their original path.
    """
    global _GEN_HOST_BUNDLE
    if _GEN_HOST_BUNDLE:
        _GEN_HOST_BUNDLE = False
        logger.warning(f"TRTLLM_GEN_HOST_BUNDLE=1 is ignored: {reason}")


# Live decode steps after which the census line is logged first (then every factor of ten until
# _CENSUS_LINEAR_LOG_STEPS, then every _CENSUS_LINEAR_LOG_STEPS steps: a run that is killed without an engine
# cleanup still has a census line within that many steps of its end).
_CENSUS_FIRST_LOG_STEPS = 100
_CENSUS_LINEAR_LOG_STEPS = 10000


class BundleCensus:
    """What the bundle did on live decode steps; engine warm-up never counts.

    A kernel that ran during warm-up is no evidence that it engages when serving, so the
    parts report here only while the engine says that it executes a live step: the input
    commit its commits and staged copies, the engine the steps it did not hand to the commit
    (with the reason), and each of the other parts every launch of its kernel and every fallback
    to the original operations. The totals are logged after 100, 1000 and 10000 live steps, then
    every 10000 steps, and at engine cleanup as one ``InputCommit census`` line of ``key=value``
    words, so a run's log shows which parts ran on real steps.
    """

    def __init__(
        self,
        first_log_steps: int = _CENSUS_FIRST_LOG_STEPS,
        linear_log_steps: int = _CENSUS_LINEAR_LOG_STEPS,
    ) -> None:
        self._first_log_steps = first_log_steps
        self._linear_log_steps = linear_log_steps
        self.reset()

    def reset(self) -> None:
        self._counts: Dict[str, int] = {}
        self._skipped: Dict[str, int] = {}
        self._steps = 0
        self._next_log = self._first_log_steps
        self.live = False

    @property
    def steps(self) -> int:
        return self._steps

    def get(self, key: str) -> int:
        return self._counts.get(key, 0)

    def begin_step(self, live: bool, skipped: Optional[str] = None) -> None:
        """One forward step of the engine: ``live`` is False during engine warm-up;
        ``skipped`` is why the step does not use the input commit (None: it does)."""
        self.live = live
        if not live:
            return
        self._steps += 1
        if skipped is not None:
            self._skipped[skipped] = self._skipped.get(skipped, 0) + 1
        if self._steps >= self._next_log:
            if self._next_log < self._linear_log_steps:
                self._next_log *= 10
            else:
                self._next_log += self._linear_log_steps
            self.log()

    def count(self, key: str, n: int = 1) -> None:
        if self.live:
            self._counts[key] = self._counts.get(key, 0) + n

    @contextmanager
    def paused(self):
        """Nothing counts inside the block (warm-up calls of the parts)."""
        live = self.live
        self.live = False
        try:
            yield self
        finally:
            self.live = live

    def summary(self) -> str:
        counts = self._counts
        commits = counts.get("commits", 0)
        copies = counts.get("staged_copies", 0) / commits if commits else 0.0
        skipped = ",".join(f"{k}:{v}" for k, v in sorted(self._skipped.items())) or "none"
        words = [
            f"steps={self._steps}",
            f"commits={commits}",
            f"overlap_commits={counts.get('overlap_commits', 0)}",
            f"verified_commits={counts.get('verified_commits', 0)}",
            f"staged_copies_per_commit={copies:.1f}",
            f"skipped={skipped}",
        ]
        words += [
            f"{key}={counts.get(key, 0)}"
            for key in (
                "block_offsets_triton",
                "block_offsets_pair",
                "block_offsets_pair_fallback",
                "block_offsets_cpp",
                "state_upload_kernel",
                "state_upload_fallback",
                "spec_scatter_kernel",
                "spec_scatter_fallback",
            )
        ]
        return " ".join(words)

    def log(self, final: bool = False) -> None:
        if self._steps or final:
            logger.info(f"InputCommit census{' (final)' if final else ''}: {self.summary()}")


_CENSUS = BundleCensus()


def bundle_census() -> BundleCensus:
    return _CENSUS


def active_input_commit() -> Optional["InputCommit"]:
    """The commit collecting the current step's uploads on this thread, or None (copy directly)."""
    return _ACTIVE.commit


@contextmanager
def input_commit_scope(commit: Optional["InputCommit"]):
    previous = _ACTIVE.commit
    _ACTIVE.commit = commit
    try:
        yield commit
    finally:
        _ACTIVE.commit = previous


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


def copy_batch_block_offsets_from_host(
    host_offsets: torch.Tensor,
    dst: torch.Tensor,
    copy_idx: torch.Tensor,
    index_scales: torch.Tensor,
    kv_offset: torch.Tensor,
) -> bool:
    """Triton version of ``copy_batch_block_offsets_to_device`` on the current stream.

    ``host_offsets`` is the pinned ``[num_pools, max_seqs, 2, max_blocks]`` host table and
    ``copy_idx`` / ``index_scales`` / ``kv_offset`` are pinned int32 host tensors; all are
    read when the kernel runs, as in the C++ kernel. Writes pools ``[0, num_pools)`` and
    sequences ``[0, len(copy_idx))`` of ``dst``. Returns False (nothing launched) when the
    tensors do not have that layout or the copy is large enough for the C++ kernel to be
    as fast, so the caller uses the C++ kernel instead.
    """
    job = block_offsets_copy_job(host_offsets, dst, copy_idx, index_scales, kv_offset)
    if job is None:
        _CENSUS.count("block_offsets_cpp")
        return False
    if job[-1] > 0:
        _copy_block_offsets_kernel[(job[-1],)](*job[:-1], BLOCK=_BLOCK_OFFSETS_BLOCK)
    _CENSUS.count("block_offsets_triton")
    return True


def block_offsets_copy_job(
    host_offsets: torch.Tensor,
    dst: torch.Tensor,
    copy_idx: torch.Tensor,
    index_scales: torch.Tensor,
    kv_offset: torch.Tensor,
) -> Optional[tuple]:
    """Kernel arguments of one ``copy_batch_block_offsets_from_host`` call, or None.

    None when the Triton copy does not apply (see ``copy_batch_block_offsets_from_host``);
    otherwise ``(src, dst, copy_idx, scales, kv_offset, src_max_seqs, dst_max_seqs,
    num_blocks, num_seqs, num_chunks, num_programs)`` with ``num_programs == 0`` when there
    is nothing to copy.
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


@triton.jit(
    do_not_specialize=[
        "a_src_max_seqs",
        "a_dst_max_seqs",
        "a_num_blocks",
        "a_num_seqs",
        "a_num_chunks",
        "a_programs",
        "b_src_max_seqs",
        "b_dst_max_seqs",
        "b_num_blocks",
        "b_num_seqs",
        "b_num_chunks",
    ]
)
def _copy_block_offsets_pair_kernel(
    a_src,
    a_dst,
    a_copy_idx,
    a_scales,
    a_kv_offset,
    a_src_max_seqs,
    a_dst_max_seqs,
    a_num_blocks,
    a_num_seqs,
    a_num_chunks,
    a_programs,
    b_src,
    b_dst,
    b_copy_idx,
    b_scales,
    b_kv_offset,
    b_src_max_seqs,
    b_dst_max_seqs,
    b_num_blocks,
    b_num_seqs,
    b_num_chunks,
    BLOCK: tl.constexpr,
):
    # Programs [0, a_programs) do job a, the rest job b; each as in _copy_block_offsets_kernel.
    pid = tl.program_id(0)
    if pid < a_programs:
        chunk = pid % a_num_chunks
        seq = (pid // a_num_chunks) % a_num_seqs
        pool = pid // (a_num_chunks * a_num_seqs)
        _block_offsets_rows(
            a_src,
            a_dst,
            a_copy_idx,
            a_scales,
            a_kv_offset,
            a_src_max_seqs,
            a_dst_max_seqs,
            a_num_blocks,
            pool,
            seq,
            chunk,
            BLOCK,
        )
    else:
        pid_b = pid - a_programs
        chunk = pid_b % b_num_chunks
        seq = (pid_b // b_num_chunks) % b_num_seqs
        pool = pid_b // (b_num_chunks * b_num_seqs)
        _block_offsets_rows(
            b_src,
            b_dst,
            b_copy_idx,
            b_scales,
            b_kv_offset,
            b_src_max_seqs,
            b_dst_max_seqs,
            b_num_blocks,
            pool,
            seq,
            chunk,
            BLOCK,
        )


def copy_block_offsets_jobs(jobs: Sequence[tuple]) -> None:
    """Run ``block_offsets_copy_job`` results on the current stream, two jobs per launch."""
    jobs = [job for job in jobs if job[-1] > 0]
    for i in range(0, len(jobs) - 1, 2):
        a, b = jobs[i], jobs[i + 1]
        _copy_block_offsets_pair_kernel[(a[-1] + b[-1],)](*a, *b[:-1], BLOCK=_BLOCK_OFFSETS_BLOCK)
        _CENSUS.count("block_offsets_pair")
    if len(jobs) % 2:
        job = jobs[-1]
        _copy_block_offsets_kernel[(job[-1],)](*job[:-1], BLOCK=_BLOCK_OFFSETS_BLOCK)
        _CENSUS.count("block_offsets_triton")


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
    reads them when the DMA runs, so the host-side lifetime rules are those of the copies.
    Returns False (nothing launched) when a pair does not qualify.
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
            _CENSUS.count("state_upload_fallback")
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
    _CENSUS.count("state_upload_kernel")
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
    """The four slot scatters of the one-model spec sampler's store in one kernel.

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
        _CENSUS.count("spec_scatter_fallback")
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
    _CENSUS.count("spec_scatter_kernel")
    return True


@triton.jit(
    do_not_specialize=[
        "num_desc",
        "num_prev",
        "tokens_per_req",
        "draft_width",
        "new_tokens_stride_t",
        "new_tokens_stride_s",
        "next_draft_stride_s",
        "input_ids_start",
        "draft_tokens_start",
        "pos_off_len",
        "pos_off_start",
        "kv_off_len",
        "kv_off_start",
    ]
)
def _input_commit_kernel(
    desc_ptr,
    num_desc,
    slots_ptr,
    num_prev,
    tokens_per_req,
    draft_width,
    new_tokens_ptr,
    new_tokens_stride_t,
    new_tokens_stride_s,
    new_tokens_lens_ptr,
    next_draft_ptr,
    next_draft_stride_s,
    input_ids_ptr,
    input_ids_start,
    draft_tokens_ptr,
    draft_tokens_start,
    pos_off_ptr,
    pos_off_len,
    pos_off_start,
    kv_off_ptr,
    kv_off_len,
    kv_off_start,
    CHUNK: tl.constexpr,
    FIX_BLOCK: tl.constexpr,
    HAS_FIX: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < num_desc:
        # Staged host->device copy: (dst address, src address, bytes <= CHUNK).
        desc = desc_ptr + pid * 3
        dst = tl.load(desc).to(tl.pointer_type(tl.uint8))
        src = tl.load(desc + 1).to(tl.pointer_type(tl.uint8))
        nbytes = tl.load(desc + 2)
        byte_offs = tl.arange(0, CHUNK)
        byte_mask = byte_offs < nbytes
        tl.store(dst + byte_offs, tl.load(src + byte_offs, mask=byte_mask), mask=byte_mask)
    elif HAS_FIX:
        # Overlap-scheduler inputs of the requests that ran in the previous step
        # (slot = their sequence slot in the sampler store):
        #   input_ids[input_ids_start + p*T + t] = new_tokens[t, slot_p]
        #   draft_tokens[draft_tokens_start + p*D + j] = next_draft[slot_p, j]
        #   pos_off = 0 except pos_off[pos_off_start + p*T + t] = new_tokens_lens[slot_p]
        #   kv_off = 0 except kv_off[kv_off_start + p] = new_tokens_lens[slot_p] - T
        idx = (pid - num_desc) * FIX_BLOCK + tl.arange(0, FIX_BLOCK)
        num_tok = num_prev * tokens_per_req

        m_tok = idx < num_tok
        p_tok = idx // tokens_per_req
        t_tok = idx - p_tok * tokens_per_req
        slot_tok = tl.load(slots_ptr + p_tok, mask=m_tok, other=0).to(tl.int64)
        tok = tl.load(
            new_tokens_ptr + t_tok * new_tokens_stride_t + slot_tok * new_tokens_stride_s,
            mask=m_tok,
            other=0,
        )
        tl.store(input_ids_ptr + input_ids_start + idx, tok, mask=m_tok)

        m_pos = idx < pos_off_len
        j_pos = idx - pos_off_start
        in_pos = m_pos & (j_pos >= 0) & (j_pos < num_tok)
        p_pos = tl.where(in_pos, j_pos, 0) // tokens_per_req
        slot_pos = tl.load(slots_ptr + p_pos, mask=in_pos, other=0).to(tl.int64)
        lens_pos = tl.load(new_tokens_lens_ptr + slot_pos, mask=in_pos, other=0)
        tl.store(pos_off_ptr + idx, tl.where(in_pos, lens_pos, 0), mask=m_pos)

        num_draft = num_prev * draft_width
        m_dr = idx < num_draft
        p_dr = idx // tl.maximum(draft_width, 1)
        j_dr = idx - p_dr * draft_width
        slot_dr = tl.load(slots_ptr + p_dr, mask=m_dr, other=0).to(tl.int64)
        dr = tl.load(next_draft_ptr + slot_dr * next_draft_stride_s + j_dr, mask=m_dr, other=0)
        tl.store(draft_tokens_ptr + draft_tokens_start + idx, dr, mask=m_dr)

        m_kv = idx < kv_off_len
        j_kv = idx - kv_off_start
        in_kv = m_kv & (j_kv >= 0) & (j_kv < num_prev)
        slot_kv = tl.load(slots_ptr + tl.where(in_kv, j_kv, 0), mask=in_kv, other=0).to(tl.int64)
        lens_kv = tl.load(new_tokens_lens_ptr + slot_kv, mask=in_kv, other=0)
        tl.store(kv_off_ptr + idx, tl.where(in_kv, lens_kv - tokens_per_req, 0), mask=m_kv)


class _PinnedArena:
    """A pinned host byte buffer with typed numpy views for staging values."""

    def __init__(self, capacity: int):
        self.tensor = torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
        self.bytes = self.tensor.numpy()
        self.base = self.tensor.data_ptr()
        self.cursor = 0
        # Whole-arena typed views; staged values start at 16-byte aligned offsets.
        self.views = {
            np_dtype: self.bytes[: capacity - capacity % size].view(np_dtype)
            for np_dtype, size in _TORCH_TO_NUMPY.values()
        }

    @property
    def capacity(self) -> int:
        return self.bytes.shape[0]

    def alloc(self, nbytes: int) -> int:
        offset = self.cursor
        self.cursor = (offset + nbytes + 15) & ~15
        return offset


class InputCommit:
    """Collects one step's host->device input copies and writes them with one kernel.

    ``begin()`` opens a step, ``stage()`` records a copy (the host values are copied
    into a pinned arena immediately, so the caller may reuse its source right away),
    ``set_overlap_fixup()`` records the previous-step gathers, and ``commit()``
    launches the kernel on the current stream. Two arenas alternate between steps;
    an event recorded after each commit kernel guards an arena's reuse.
    """

    def __init__(self, capacity: int = 1 << 20):
        self._arenas = [_PinnedArena(capacity), _PinnedArena(capacity)]
        self._events: List[Optional[torch.cuda.Event]] = [None, None]
        self._index = 0
        self._arena = self._arenas[0]
        self._entries: List[Tuple[int, int, int]] = []
        # The destinations of the staged copies, held until the commit kernel is launched: an
        # entry records the destination's address only, which must not be freed or reused
        # between ``stage`` and ``commit``.
        self._dsts: List[torch.Tensor] = []
        self._rollbacks: List[Callable[[], None]] = []
        self._fixup: Optional[dict] = None
        self._open = False
        self._verify_dsts: List[Tuple[torch.Tensor, int, int]] = []
        # Real (non-warm-up) commits only: they drive the first-commit log lines and the
        # VERIFY budget.
        self._num_commits = 0
        self._logged_overlap = False
        self._warming_up = False

    @property
    def is_open(self) -> bool:
        return self._open

    def _verifying(self) -> bool:
        return _VERIFY_COMMITS < 0 or self._num_commits < _VERIFY_COMMITS

    @contextmanager
    def warmup(self):
        """Commits taken inside the block only compile and exercise the kernels.

        They are not counted, do not use up the VERIFY budget and log nothing, so the
        counters and the first-commit log lines describe real decode steps only (a line
        from a warm-up commit says nothing about the engine using the commit). The commit
        is idle and in its initial state again after the block.
        """
        assert not self._open, "InputCommit.warmup() inside an open step"
        self._warming_up = True
        try:
            yield self
        finally:
            self._warming_up = False
            if self._open:
                self.abort()
            torch.cuda.current_stream().synchronize()
            self._events = [None, None]
            self._index = 0
            self._arena = self._arenas[0]
            self._num_commits = 0
            self._logged_overlap = False

    def begin(self) -> None:
        assert not self._open, "InputCommit.begin() called twice without commit()"
        self._index ^= 1
        event = self._events[self._index]
        if event is not None:
            event.synchronize()
        self._arena = self._arenas[self._index]
        self._arena.cursor = 0
        self._entries.clear()
        self._dsts.clear()
        self._rollbacks.clear()
        self._fixup = None
        self._verify_dsts.clear()
        self._open = True

    def on_abort(self, rollback: Callable[[], None]) -> None:
        """Run ``rollback`` if this step is aborted instead of committed.

        For host state that a staging site updates on the assumption that its staged copy
        will happen (for example a record of what the device buffers hold): an aborted step
        drops the staged values, so that state has to be undone.
        """
        assert self._open
        self._rollbacks.append(rollback)

    def abort(self) -> None:
        """Drop the staged copies and run the registered rollbacks."""
        rollbacks, self._rollbacks = self._rollbacks, []
        self._entries.clear()
        self._dsts.clear()
        self._fixup = None
        self._open = False
        for rollback in rollbacks:
            rollback()

    def _reserve(self, nbytes: int) -> int:
        if self._arena.cursor + nbytes > self._arena.capacity:
            self._grow(self._arena.cursor + nbytes)
        return self._arena.alloc(nbytes)

    def _grow(self, needed: int) -> None:
        # Rare. Replaces the arena of the open step (nothing reads it yet) by a larger
        # one and rebases the host addresses recorded so far.
        old = self._arena
        new = _PinnedArena(max(2 * old.capacity, 2 * needed))
        new.bytes[: old.cursor] = old.bytes[: old.cursor]
        new.cursor = old.cursor
        delta = new.base - old.base
        self._entries = [(dst, src + delta, n) for dst, src, n in self._entries]
        if self._fixup is not None:
            self._fixup["slots_addr"] += delta
        self._arenas[self._index] = new
        self._arena = new

    def stage(self, dst: torch.Tensor, values: Union[Sequence, np.ndarray, torch.Tensor]) -> None:
        """Record ``dst.copy_(values)``; ``values`` has ``dst.numel()`` elements."""
        assert self._open
        assert dst.is_cuda and dst.is_contiguous(), "staged destinations must be contiguous"
        n = dst.numel()
        if n == 0:
            return
        np_dtype, itemsize = _TORCH_TO_NUMPY[dst.dtype]
        nbytes = n * itemsize
        offset = self._reserve(nbytes)
        arena = self._arena
        first = offset // itemsize
        view = arena.views[np_dtype][first : first + n]
        if isinstance(values, torch.Tensor):
            src = values.reshape(-1)
            if src.dtype != dst.dtype:
                src = src.to(dst.dtype)
            view[:] = src.numpy()
        else:
            view[:] = values
        self._entries.append((dst.data_ptr(), arena.base + offset, nbytes))
        self._dsts.append(dst)
        if self._verifying():
            self._verify_dsts.append((dst, offset, nbytes))

    def stage_host_values(self, values: Sequence[int], dtype=np.int32) -> Tuple[int, int]:
        """Copy ``values`` into the arena without a destination; returns (address, offset)."""
        assert self._open
        arr = np.asarray(values, dtype=dtype)
        offset = self._reserve(max(arr.nbytes, 1))
        self._arena.bytes[offset : offset + arr.nbytes].view(dtype)[:] = arr
        return self._arena.base + offset, offset

    def set_overlap_fixup(
        self,
        *,
        previous_slots: Sequence[int],
        tokens_per_req: int,
        draft_width: int,
        new_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_start: int,
        draft_tokens: torch.Tensor,
        draft_tokens_start: int,
        pos_offsets: torch.Tensor,
        pos_offsets_start: int,
        kv_offsets: torch.Tensor,
        kv_offsets_start: int,
    ) -> None:
        """Record the overlap-scheduler gathers of ``_prepare_tp_inputs``.

        ``new_tokens`` is ``[T_store, num_slots, 1]``, ``next_draft_tokens`` is
        ``[num_slots, D_store]``; ``pos_offsets`` / ``kv_offsets`` are zeroed except the
        ranges the gathers write.
        """
        assert self._open and self._fixup is None
        for t in (
            new_tokens,
            new_tokens_lens,
            next_draft_tokens,
            input_ids,
            draft_tokens,
            pos_offsets,
            kv_offsets,
        ):
            assert t.is_cuda and t.dtype == torch.int32
        assert input_ids.is_contiguous() and draft_tokens.is_contiguous()
        assert pos_offsets.is_contiguous() and kv_offsets.is_contiguous()
        assert new_tokens.dim() == 3 and next_draft_tokens.dim() == 2
        assert new_tokens.stride(2) == 1 or new_tokens.shape[2] == 1
        assert next_draft_tokens.stride(1) == 1
        num_prev = len(previous_slots)
        assert num_prev * tokens_per_req <= input_ids.numel() - input_ids_start
        assert num_prev * draft_width <= draft_tokens.numel() - draft_tokens_start
        slots_addr, _ = self.stage_host_values(previous_slots if num_prev else [0])
        self._fixup = dict(
            slots_addr=slots_addr,
            num_prev=num_prev,
            tokens_per_req=tokens_per_req,
            draft_width=draft_width,
            new_tokens=new_tokens,
            new_tokens_lens=new_tokens_lens,
            next_draft_tokens=next_draft_tokens,
            input_ids=input_ids,
            input_ids_start=input_ids_start,
            draft_tokens=draft_tokens,
            draft_tokens_start=draft_tokens_start,
            pos_offsets=pos_offsets,
            pos_offsets_start=pos_offsets_start,
            kv_offsets=kv_offsets,
            kv_offsets_start=kv_offsets_start,
        )

    def commit(self) -> None:
        """Launch the commit kernel on the current stream and close the step.

        A failure drops the step like ``abort`` (rollbacks included) and is re-raised.
        """
        assert self._open
        try:
            self._launch()
        except BaseException:
            self.abort()
            raise

    def _launch(self) -> None:
        entries = self._entries
        fixup = self._fixup
        if not entries and fixup is None:
            self._open = False
            self._rollbacks.clear()
            return
        num_desc = sum(_cdiv(nbytes, _COPY_CHUNK) for _, _, nbytes in entries)
        # Reserving may grow the arena and rebase the recorded host addresses, so the
        # entries and the fixup are read again afterwards.
        desc_offset = self._reserve(max(24 * num_desc, 8))
        self._open = False
        entries = self._entries
        fixup = self._fixup
        arena = self._arena
        descs: List[int] = []
        for dst, src, nbytes in entries:
            for start in range(0, nbytes, _COPY_CHUNK):
                descs.extend((dst + start, src + start, min(_COPY_CHUNK, nbytes - start)))
        if descs:
            arena.bytes[desc_offset : desc_offset + 8 * len(descs)].view(np.int64)[:] = descs
        desc_tensor = arena.tensor[desc_offset : desc_offset + max(8 * len(descs), 8)].view(
            torch.int64
        )
        if fixup is not None:
            slots = _host_view(arena, fixup["slots_addr"], max(fixup["num_prev"], 1))
            num_fix = max(
                fixup["num_prev"] * fixup["tokens_per_req"],
                fixup["pos_offsets"].numel(),
                fixup["num_prev"] * fixup["draft_width"],
                fixup["kv_offsets"].numel(),
            )
            num_fix_programs = _cdiv(max(num_fix, 1), _FIX_BLOCK)
            new_tokens = fixup["new_tokens"]
            args = (
                slots,
                fixup["num_prev"],
                fixup["tokens_per_req"],
                fixup["draft_width"],
                new_tokens,
                new_tokens.stride(0),
                new_tokens.stride(1),
                fixup["new_tokens_lens"],
                fixup["next_draft_tokens"],
                fixup["next_draft_tokens"].stride(0),
                fixup["input_ids"],
                fixup["input_ids_start"],
                fixup["draft_tokens"],
                fixup["draft_tokens_start"],
                fixup["pos_offsets"],
                fixup["pos_offsets"].numel(),
                fixup["pos_offsets_start"],
                fixup["kv_offsets"],
                fixup["kv_offsets"].numel(),
                fixup["kv_offsets_start"],
            )
        else:
            num_fix_programs = 0
            dummy = desc_tensor
            args = (
                dummy,
                0,
                1,
                0,
                dummy,
                0,
                0,
                dummy,
                dummy,
                0,
                dummy,
                0,
                dummy,
                0,
                dummy,
                0,
                0,
                dummy,
                0,
                0,
            )
        _input_commit_kernel[(num_desc + num_fix_programs,)](
            desc_tensor,
            num_desc,
            *args,
            CHUNK=_COPY_CHUNK,
            FIX_BLOCK=_FIX_BLOCK,
            HAS_FIX=fixup is not None,
        )
        event = self._events[self._index]
        if event is None:
            event = torch.cuda.Event()
            self._events[self._index] = event
        event.record()
        verified = self._verifying()
        if verified:
            self._verify(fixup)
        if not self._warming_up:
            self._note_real_commit(len(entries), fixup is not None, verified)
        self._entries = []
        self._dsts.clear()
        self._rollbacks.clear()
        self._fixup = None

    def _note_real_commit(self, num_copies: int, has_fixup: bool, verified: bool) -> None:
        # Bookkeeping of a commit on a real decode step (never called for warm-up commits).
        if verified and self._num_commits + 1 == _VERIFY_COMMITS:
            logger.info(f"InputCommit verify: the first {_VERIFY_COMMITS} commits matched")
        if self._num_commits == 0:
            logger.info(
                f"InputCommit: first batched input commit ({num_copies} staged copies, "
                f"overlap fixup: {has_fixup})"
            )
        if has_fixup and not self._logged_overlap:
            # The steady-state decode step: previous-step gathers included.
            self._logged_overlap = True
            logger.info(f"InputCommit: first overlap-scheduler commit ({num_copies} staged copies)")
        self._num_commits += 1
        census = bundle_census()
        census.count("commits")
        census.count("staged_copies", num_copies)
        if has_fixup:
            census.count("overlap_commits")
        if verified:
            census.count("verified_commits")

    def _verify(self, fixup: Optional[dict]) -> None:
        torch.cuda.current_stream().synchronize()
        arena = self._arena
        for dst, offset, nbytes in self._verify_dsts:
            expected = torch.from_numpy(arena.bytes[offset : offset + nbytes].copy())
            got = dst.reshape(-1).view(torch.uint8).cpu()
            if not torch.equal(got, expected):
                raise RuntimeError(
                    f"InputCommit verify: staged copy mismatch for a {dst.dtype} "
                    f"destination of {dst.numel()} elements"
                )
        if fixup is None:
            return
        num_prev = fixup["num_prev"]
        tokens = fixup["tokens_per_req"]
        width = fixup["draft_width"]
        slots_view = _host_view(arena, fixup["slots_addr"], max(num_prev, 1))[:num_prev]
        slots = slots_view.to(device="cuda", dtype=torch.long)
        lens = fixup["new_tokens_lens"]
        ref_ids = fixup["new_tokens"].transpose(0, 1)[slots, :tokens].flatten()
        ref_draft = fixup["next_draft_tokens"][slots, :width].flatten()
        ref_pos = torch.zeros_like(fixup["pos_offsets"])
        start = fixup["pos_offsets_start"]
        ref_pos[start : start + num_prev * tokens] = lens[slots.repeat_interleave(tokens)]
        ref_kv = torch.zeros_like(fixup["kv_offsets"])
        start = fixup["kv_offsets_start"]
        ref_kv[start : start + num_prev] = lens[slots] - tokens
        ids_start = fixup["input_ids_start"]
        draft_start = fixup["draft_tokens_start"]
        checks = (
            ("input_ids", fixup["input_ids"][ids_start : ids_start + num_prev * tokens], ref_ids),
            (
                "draft_tokens",
                fixup["draft_tokens"][draft_start : draft_start + num_prev * width],
                ref_draft,
            ),
            ("pos_offsets", fixup["pos_offsets"], ref_pos),
            ("kv_offsets", fixup["kv_offsets"], ref_kv),
        )
        for name, got, ref in checks:
            if not torch.equal(got, ref.to(got.dtype)):
                raise RuntimeError(f"InputCommit verify: overlap fixup mismatch in {name}")


def warmup_input_commit_kernels(commit: Optional[InputCommit]) -> None:
    """Compile the kernels of this module ahead of serving (the commit is not used
    during engine warmup, so its first use would otherwise pay the JIT compile).

    Nothing done here counts as engagement: the commit's warm-up commits are neither counted
    nor logged (``InputCommit.warmup``) and the census is paused.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    with bundle_census().paused():
        if commit is not None:
            buf = torch.zeros(64, dtype=torch.int32, device=device)
            new_tokens = torch.zeros((2, 2, 1), dtype=torch.int32, device=device)
            lens = torch.ones(2, dtype=torch.int32, device=device)
            draft = torch.zeros((2, 1), dtype=torch.int32, device=device)
            pos_off = torch.zeros(4, dtype=torch.int32, device=device)
            kv_off = torch.zeros(2, dtype=torch.int32, device=device)
            with commit.warmup():
                for with_fixup in (False, True):
                    commit.begin()
                    commit.stage(buf[:4], [1, 2, 3, 4])
                    if with_fixup:
                        commit.set_overlap_fixup(
                            previous_slots=[0, 1],
                            tokens_per_req=2,
                            draft_width=1,
                            new_tokens=new_tokens,
                            new_tokens_lens=lens,
                            next_draft_tokens=draft,
                            input_ids=buf[16:32],
                            input_ids_start=0,
                            draft_tokens=buf[32:48],
                            draft_tokens_start=0,
                            pos_offsets=pos_off,
                            pos_offsets_start=0,
                            kv_offsets=kv_off,
                            kv_offsets_start=0,
                        )
                    commit.commit()
        if _GEN_HOST_BUNDLE:
            host = torch.zeros((1, 2, 2, 4), dtype=torch.int32, pin_memory=True)
            dst = torch.zeros((1, 2, 2, 4), dtype=torch.int32, device=device)
            copy_idx = torch.tensor([1, 0], dtype=torch.int32).pin_memory()
            ones = torch.ones(1, dtype=torch.int32).pin_memory()
            copy_batch_block_offsets_from_host(host, dst, copy_idx, ones, ones)
            job = block_offsets_copy_job(host, dst, copy_idx, ones, ones)
            if job is not None:
                copy_block_offsets_jobs([job, job])
        torch.cuda.synchronize()


def _host_view(arena: _PinnedArena, address: int, count: int) -> torch.Tensor:
    offset = address - arena.base
    return arena.tensor[offset : offset + 4 * count].view(torch.int32)

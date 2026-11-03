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
"""One-launch upload of a decode step's input buffers (the batched input commit).

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
read them (previous-step tokens) are part of the commit kernel. Bytes staged more than
once in a commit keep the value staged last, as a sequence of direct copies leaves them.

The block-offset copy of ``KVCacheManagerV2`` has its own kernel here
(``copy_batch_block_offsets_from_host``): same semantics as the C++
``copyBatchBlockOffsetsToDeviceKernel`` (reads the pinned host table when it runs),
but split over many CTAs instead of one CTA per (pool, sequence).
"""

import os
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned
from tensorrt_llm.logger import logger

# Decode-step paths of a Kimi K3.1 engine, selected once per process from its model config
# (configure_k31_engine, called before the model is built). Every other model keeps the
# default paths.
_K31 = False
_K31_DRAFTS = False
_K31_ATTENTION_DP = False
_K31_GENERATION = False

_ACTIVE: Optional["InputCommit"] = None

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
# bandwidth and the C++ kernel is as fast (GB300: 2 pools x 976 blocks or
# 1 pool x 15584 blocks at 2-32 sequences: 2.5-14.6 us vs 3.5-20.6 us; at
# 128-256 sequences of 15584 blocks both take 45-85 us).
_BLOCK_OFFSETS_TRITON_MAX_ENTRIES = 1 << 20


def _cdiv(a: int, b: int) -> int:
    # Plain integer ceil-div: triton.cdiv is a JIT function, slow to call from Python.
    return -(-a // b)


def disagg_role() -> str:
    """This server's disaggregated-serving role: "context", "generation", or "" (aggregated).

    ``trtllm-serve`` exports it from ``--server_role`` to every worker.
    """
    return os.environ.get("TRTLLM_DISAGG_ROLE", "")


def configure_k31_engine(k31: bool, spec_config, mapping) -> None:
    """Record whether this process builds a Kimi K3.1 engine and how it drafts.

    ``k31``: the text config carries a ``dsa_config``. The engine drafts when its
    speculative decoding is DSpark / DFlash without ``context_kv_only`` (generation servers
    and aggregated serving; context servers only build the drafter's context KV).
    """
    global _K31, _K31_DRAFTS, _K31_ATTENTION_DP, _K31_GENERATION
    mode = getattr(spec_config, "spec_dec_mode", None)
    _K31 = k31
    _K31_DRAFTS = (
        k31
        and mode is not None
        and (mode.is_dspark() or mode.is_dflash())
        and not getattr(spec_config, "context_kv_only", False)
    )
    _K31_ATTENTION_DP = k31 and mapping.enable_attention_dp
    _K31_GENERATION = _K31_DRAFTS and disagg_role() == "generation"


def k31_engine() -> bool:
    """Whether this process builds a Kimi K3.1 engine."""
    return _K31


def k31_attention_dp() -> bool:
    """Whether this process builds a Kimi K3.1 engine with attention DP."""
    return _K31_ATTENTION_DP


def k31_draft_engine() -> bool:
    """Whether this process runs a drafting Kimi K3.1 DSpark / DFlash engine."""
    return _K31_DRAFTS


def k31_draft_tp_engine() -> bool:
    """A drafting Kimi K3.1 engine without attention DP. Selects the fused draft tail
    (``spec_tail_kernels``, ``dspark_glue_kernels``, ``dspark_draft3``, the one-launch FP8
    draft query of ``dspark_ull2``) and the fused spec-sampler scatter: the values of the
    paths they replace."""
    return _K31_DRAFTS and not _K31_ATTENTION_DP


def k31_generation_server() -> bool:
    """A drafting Kimi K3.1 engine on a disaggregated generation server: the paths that
    hold extra memory or change the drafter's rounding are scoped to it."""
    return _K31_GENERATION


def k31_tp_generation_server() -> bool:
    """:func:`k31_generation_server` without attention DP."""
    return k31_generation_server() and not _K31_ATTENTION_DP


def batched_input_commit_enabled() -> bool:
    """Whether eligible decode steps upload their inputs through an :class:`InputCommit`
    (pinned arenas, Kimi K3.1 generation servers)."""
    return k31_generation_server()


def block_offsets_triton_enabled() -> bool:
    """Whether ``KVCacheManagerV2`` copies block offsets with the Triton kernel below (same
    values as the C++ kernel)."""
    return _K31_DRAFTS


def spec_sampler_fused_scatter_enabled() -> bool:
    """Whether the one-model spec sampler's four store scatters run as one kernel, with the
    slot tensor kept on the device while the slots do not change."""
    return k31_draft_tp_engine()


def active_input_commit() -> Optional["InputCommit"]:
    """The commit collecting the current step's uploads, or None (copy directly)."""
    return _ACTIVE


def stage_or_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst.copy_(src, non_blocking=True)`` of a host ``src``: staged in the open input
    commit of the step when there is one (written by its next commit kernel), else
    copied directly."""
    commit = _ACTIVE
    if (
        commit is not None
        and commit.is_open
        and src.device.type == "cpu"
        and dst.is_cuda
        and dst.is_contiguous()
        and dst.numel() == src.numel()
        and dst.dtype in _TORCH_TO_NUMPY
    ):
        commit.stage(dst, src)
        return
    dst.copy_(src, non_blocking=True)


def stage_scatter(target: torch.Tensor, slots: Sequence[int], values: Sequence[int]) -> bool:
    """``target[slots[i]] = values[i]`` for a 1-D CUDA ``target`` and distinct ``slots``
    (``index_copy_`` along dim 0), staged in the open input commit of the step with one
    entry per slot; False (nothing staged) when there is no open commit or the arena cannot
    take ``target``."""
    commit = _ACTIVE
    if (
        commit is None
        or not commit.is_open
        or not target.is_cuda
        or target.dim() != 1
        or not target.is_contiguous()
        or target.dtype not in _TORCH_TO_NUMPY
    ):
        return False
    for slot, value in zip(slots, values):
        commit.stage(target[slot : slot + 1], [value])
    return True


def flush_input_commit() -> None:
    """Write every copy staged so far in this step (one kernel) and keep staging: for
    a device operation of the input preparation that reads a staged destination."""
    commit = _ACTIVE
    if commit is not None and commit.is_open:
        commit.commit()
        commit.begin()


@contextmanager
def input_commit_scope(commit: Optional["InputCommit"]):
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = commit
    try:
        yield commit
    finally:
        _ACTIVE = previous


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
        return False
    if job[-1] > 0:
        _copy_block_offsets_kernel[(job[-1],)](*job[:-1], BLOCK=_BLOCK_OFFSETS_BLOCK)
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


def _has_overlap(entries: Sequence[Tuple[int, int, int]]) -> bool:
    """Whether two staged (dst, src, nbytes) entries share a destination byte."""
    end = -1
    for dst, _, nbytes in sorted(entries):
        if dst < end:
            return True
        end = dst + nbytes
    return False


class _PinnedArena:
    """A pinned host byte buffer with typed numpy views for staging values."""

    def __init__(self, capacity: int):
        # The commit kernel reads the arena in place over the host link, so it must be pinned;
        # callers create an InputCommit only where prefer_pinned() holds.
        assert prefer_pinned(), "the input commit arena needs pinned host memory"
        self.tensor = torch.empty(capacity, dtype=torch.uint8, pin_memory=prefer_pinned())
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
    launches the kernel on the current stream. The arenas are used round robin, one
    per commit; an event recorded after each commit kernel guards an arena's reuse.
    ``num_arenas`` covers the commits of the steps the host runs ahead of the GPU
    (a step that flushes, see :func:`flush_input_commit`, commits more than once).
    """

    def __init__(self, capacity: int = 1 << 20, num_arenas: int = 2):
        self._arenas = [_PinnedArena(capacity) for _ in range(num_arenas)]
        self._events: List[Optional[torch.cuda.Event]] = [None] * num_arenas
        self._index = 0
        self._arena = self._arenas[0]
        self._entries: List[Tuple[int, int, int]] = []
        self._fixup: Optional[dict] = None
        self._open = False
        # Runs of overlapping staged entries merged so far (see _merge_overlapping_entries).
        self.num_merged_runs = 0
        self._merge_logged = False

    @property
    def is_open(self) -> bool:
        return self._open

    def begin(self) -> None:
        assert not self._open, "InputCommit.begin() called twice without commit()"
        self._index = (self._index + 1) % len(self._arenas)
        event = self._events[self._index]
        if event is not None:
            event.synchronize()
        self._arena = self._arenas[self._index]
        self._arena.cursor = 0
        self._entries.clear()
        self._fixup = None
        self._open = True

    def abort(self) -> None:
        self._entries.clear()
        self._fixup = None
        self._open = False

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

    def stage_host_values(self, values: Sequence[int], dtype=np.int32) -> Tuple[int, int]:
        """Copy ``values`` into the arena without a destination; returns (address, offset)."""
        assert self._open
        arr = np.asarray(values, dtype=dtype)
        offset = self._reserve(max(arr.nbytes, 1))
        self._arena.bytes[offset : offset + arr.nbytes].view(dtype)[:] = arr
        return self._arena.base + offset, offset

    def _merge_overlapping_entries(self) -> None:
        """Replace each run of overlapping staged entries by one entry holding the bytes
        they leave when applied in staging order.

        The commit kernel writes its entries in parallel, so two entries with a common
        destination byte would race. Each run's bytes are composed in a new arena range:
        the union of overlapping ranges is contiguous and fully written.
        """
        entries = self._entries
        order = sorted(range(len(entries)), key=lambda i: entries[i][0])
        runs: List[List[int]] = []
        run_end = -1
        for i in order:
            dst, _, nbytes = entries[i]
            if runs and dst < run_end:
                runs[-1].append(i)
                run_end = max(run_end, dst + nbytes)
            else:
                runs.append([i])
                run_end = dst + nbytes
        spans = []
        for run in runs:
            if len(run) > 1:
                start = min(entries[i][0] for i in run)
                stop = max(entries[i][0] + entries[i][2] for i in run)
                spans.append((sorted(run), start, stop - start))
        offsets = []
        for _, _, nbytes in spans:
            # Reserving may grow the arena and rebase the recorded source addresses.
            offsets.append(self._reserve(nbytes))
        entries = self._entries
        arena = self._arena
        replaced = {}
        for (run, start, nbytes), offset in zip(spans, offsets):
            for i in run:  # staging order: a later entry overwrites an earlier one
                dst, src, size = entries[i]
                at = offset + dst - start
                src_offset = src - arena.base
                arena.bytes[at : at + size] = arena.bytes[src_offset : src_offset + size]
                replaced[i] = None
            replaced[run[0]] = (start, arena.base + offset, nbytes)
        self._entries = [
            replaced[i] if i in replaced else entry
            for i, entry in enumerate(entries)
            if i not in replaced or replaced[i] is not None
        ]
        self.num_merged_runs += len(spans)
        if not self._merge_logged:
            self._merge_logged = True
            logger.debug(
                f"InputCommit: {len(spans)} destination range(s) staged more than once in one "
                "commit were merged in staging order"
            )

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
        """Launch the commit kernel on the current stream and close the step."""
        assert self._open
        entries = self._entries
        fixup = self._fixup
        if not entries and fixup is None:
            self._open = False
            return
        if len(entries) > 1 and _has_overlap(entries):
            self._merge_overlapping_entries()
            entries = self._entries
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
        self._entries = []
        self._fixup = None


def warmup_input_commit_kernels(commit: Optional[InputCommit]) -> None:
    """Compile the kernels of this module ahead of serving (the commit is not used
    during engine warmup, so its first use would otherwise pay the JIT compile)."""
    device = torch.device("cuda", torch.cuda.current_device())
    if commit is not None:
        buf = torch.zeros(64, dtype=torch.int32, device=device)
        new_tokens = torch.zeros((2, 2, 1), dtype=torch.int32, device=device)
        lens = torch.ones(2, dtype=torch.int32, device=device)
        draft = torch.zeros((2, 1), dtype=torch.int32, device=device)
        pos_off = torch.zeros(4, dtype=torch.int32, device=device)
        kv_off = torch.zeros(2, dtype=torch.int32, device=device)
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
    if block_offsets_triton_enabled():
        host = torch.zeros((1, 2, 2, 4), dtype=torch.int32, pin_memory=prefer_pinned())
        dst = torch.zeros((1, 2, 2, 4), dtype=torch.int32, device=device)
        copy_idx = maybe_pin_memory(torch.tensor([1, 0], dtype=torch.int32))
        ones = maybe_pin_memory(torch.ones(1, dtype=torch.int32))
        copy_batch_block_offsets_from_host(host, dst, copy_idx, ones, ones)
    torch.cuda.synchronize()


def _host_view(arena: _PinnedArena, address: int, count: int) -> torch.Tensor:
    offset = address - arena.base
    return arena.tensor[offset : offset + 4 * count].view(torch.int32)

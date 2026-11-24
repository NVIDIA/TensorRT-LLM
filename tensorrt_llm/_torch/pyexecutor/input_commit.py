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
"""One-launch upload of a decode step's input buffers (``TRTLLM_BATCHED_INPUT_COMMIT=1``).

Input preparation for a CUDA-graph decode step fills small device buffers from host data
(positions, KV and prompt lengths, helix offsets, sampling seeds, ...) and, under the
overlap scheduler, gathers the previous step's tokens and accepted lengths on the device.
Issued one by one, these copies and gathers form a chain of small operations between two
graph replays, each waiting for the previous replay because it writes buffers that replay
reads. ``InputCommit`` collects the host values into a pinned arena while the step is
prepared and writes every destination with one kernel right before the replay (the kernel
reads the arena directly over the host link). The overlap-scheduler gathers run in the same
kernel.

Contract for staged copies: between ``stage`` and ``commit`` no device operation may read or
write a staged destination. Call sites stage only buffers that the input preparation does
not read back on the device; the device-side gathers that do read them (the previous step's
tokens) are part of the commit kernel. A preparation step that must read a staged destination
commits what it staged so far first (``commit()`` then ``begin()``).

Switches (read once per process, default off):

* ``TRTLLM_BATCHED_INPUT_COMMIT=1``: the decoder runner opens a commit around the input
  preparation of an eligible step (see ``DecoderRunner._input_commit_for_step``).
  The decode-step shell's ``step_prep`` (``ull_shell.py``) turns it on as well.
* ``TRTLLM_BATCHED_INPUT_COMMIT_VERIFY=<n>`` (default: the step shell's ``verify_commits``):
  after each of the first ``n`` commits (``-1``: all), synchronize and compare every
  destination with the staged bytes and with a torch reference of the overlap gathers; raises
  on a mismatch. While such a commit is open, every torch operator and Triton launch on the
  preparing thread is checked against the contract above (:class:`_StagedHazardCheck`);
  violations are logged. Once the first ``n`` commits matched with no violation, logs
  ``InputCommit verify: the first <n> commits matched``.
"""

import bisect
import os
import threading
import traceback
from contextlib import contextmanager, nullcontext
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from .step_upload_kernels import (
    _BLOCK_OFFSETS_BLOCK,
    _block_offsets_rows,
    block_offsets_copy_job,
    copy_batch_block_offsets_from_host,
)
from .ull_shell import STEP_FOLD, default_verify_commits

BATCHED_INPUT_COMMIT_ENV = "TRTLLM_BATCHED_INPUT_COMMIT"
BATCHED_INPUT_COMMIT_VERIFY_ENV = "TRTLLM_BATCHED_INPUT_COMMIT_VERIFY"

_ENABLED = os.environ.get(BATCHED_INPUT_COMMIT_ENV, "0") == "1"
_VERIFY_COMMITS_ENV = (os.environ.get(BATCHED_INPUT_COMMIT_VERIFY_ENV) or "").strip()
# Commits to verify; None: the step shell's ``verify_commits`` (read when a commit is built).
_VERIFY_COMMITS: Optional[int] = int(_VERIFY_COMMITS_ENV) if _VERIFY_COMMITS_ENV else None

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

# The commit collecting the current step's uploads, per thread: only the thread that
# prepares the step may stage into it.
_SCOPE = threading.local()


def _cdiv(a: int, b: int) -> int:
    # Plain integer ceil-div: triton.cdiv is a JIT function, slow to call from Python.
    return -(-a // b)


def batched_input_commit_enabled() -> bool:
    """Whether ``TRTLLM_BATCHED_INPUT_COMMIT=1`` is set."""
    return _ENABLED


def active_input_commit() -> Optional["InputCommit"]:
    """The commit collecting the current step's uploads, or None (copy directly)."""
    return getattr(_SCOPE, "commit", None)


@contextmanager
def input_commit_scope(commit: Optional["InputCommit"]) -> Iterator[Optional["InputCommit"]]:
    """Make ``commit`` the active one on this thread for the duration of the block."""
    previous = active_input_commit()
    _SCOPE.commit = commit
    try:
        yield commit
    finally:
        _SCOPE.commit = previous


@contextmanager
def input_commit_step(commit: Optional["InputCommit"]) -> Iterator[Optional["InputCommit"]]:
    """Open ``commit`` for the step prepared in the block and launch it when the block ends.

    With ``commit`` None the block runs as is (every upload copies directly). On an
    exception the staged values are dropped and nothing is launched.
    """
    if commit is None:
        yield None
        return
    commit.begin()
    try:
        with input_commit_scope(commit), commit.hazard_check():
            yield commit
    except BaseException:
        commit.abort()
        raise
    commit.commit()


def flush_active_input_commit() -> bool:
    """Write every copy staged so far in this step (one kernel) and keep staging.

    For a preparation step that reads a staged destination on the device next. Returns
    whether a commit was open (False: every upload already copied directly).
    """
    commit = active_input_commit()
    if commit is None or not commit.is_open:
        return False
    commit.commit()
    commit.begin()
    return True


def overlap_fixup_supported(
    new_tokens: Optional[torch.Tensor],
    new_tokens_lens: Optional[torch.Tensor],
    next_draft_tokens: Optional[torch.Tensor],
) -> bool:
    """Whether :meth:`InputCommit.set_overlap_fixup` covers these sampler stores.

    ``new_tokens`` must be an int32 ``[T_store, num_slots, 1]`` CUDA tensor,
    ``new_tokens_lens`` an int32 ``[num_slots]`` and ``next_draft_tokens`` an int32
    ``[num_slots, D_store]`` with unit stride in its last dimension.
    """
    tensors = (new_tokens, new_tokens_lens, next_draft_tokens)
    return (
        all(isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == torch.int32 for t in tensors)
        and new_tokens.dim() == 3
        and (new_tokens.shape[2] == 1 or new_tokens.stride(2) == 1)
        and new_tokens_lens.dim() == 1
        and next_draft_tokens.dim() == 2
        and next_draft_tokens.stride(1) == 1
    )


@triton.jit
def _commit_staged_copy(desc_ptr, pid, CHUNK: tl.constexpr):
    # Staged host->device copy: (dst address, src address, bytes <= CHUNK).
    desc = desc_ptr + pid * 3
    dst = tl.load(desc).to(tl.pointer_type(tl.uint8))
    src = tl.load(desc + 1).to(tl.pointer_type(tl.uint8))
    nbytes = tl.load(desc + 2)
    byte_offs = tl.arange(0, CHUNK)
    byte_mask = byte_offs < nbytes
    tl.store(dst + byte_offs, tl.load(src + byte_offs, mask=byte_mask), mask=byte_mask)


@triton.jit
def _commit_overlap_fixup(
    fix_pid,
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
    FIX_BLOCK: tl.constexpr,
):
    # Overlap-scheduler inputs of the requests that ran in the previous step
    # (slot = their sequence slot in the sampler store):
    #   input_ids[input_ids_start + p*T + t] = new_tokens[t, slot_p]
    #   draft_tokens[draft_tokens_start + p*D + j] = next_draft[slot_p, j]
    #   pos_off = 0 except pos_off[pos_off_start + p*T + t] = new_tokens_lens[slot_p]
    #   kv_off = 0 except kv_off[kv_off_start + p] = new_tokens_lens[slot_p] - T
    idx = fix_pid * FIX_BLOCK + tl.arange(0, FIX_BLOCK)
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


_COMMIT_DO_NOT_SPECIALIZE = [
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


@triton.jit(do_not_specialize=_COMMIT_DO_NOT_SPECIALIZE)
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
        _commit_staged_copy(desc_ptr, pid, CHUNK)
    elif HAS_FIX:
        _commit_overlap_fixup(
            pid - num_desc,
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
            FIX_BLOCK,
        )


@triton.jit(
    do_not_specialize=_COMMIT_DO_NOT_SPECIALIZE
    + [
        "num_fix",
        "blk_src_max_seqs",
        "blk_dst_max_seqs",
        "blk_num_blocks",
        "blk_num_seqs",
        "blk_num_chunks",
    ]
)
def _input_commit_blk_kernel(
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
    num_fix,
    blk_src_ptr,
    blk_dst_ptr,
    blk_copy_idx_ptr,
    blk_scales_ptr,
    blk_kv_offset_ptr,
    blk_src_max_seqs,
    blk_dst_max_seqs,
    blk_num_blocks,
    blk_num_seqs,
    blk_num_chunks,
    CHUNK: tl.constexpr,
    FIX_BLOCK: tl.constexpr,
    HAS_FIX: tl.constexpr,
    BLK_BLOCK: tl.constexpr,
):
    # _input_commit_kernel's programs, then num_pools * num_seqs * num_chunks programs of
    # the KV block-offset copy (the step shell's step_fold; step_upload_kernels.py).
    pid = tl.program_id(0)
    if pid < num_desc:
        _commit_staged_copy(desc_ptr, pid, CHUNK)
    elif pid < num_desc + num_fix:
        if HAS_FIX:
            _commit_overlap_fixup(
                pid - num_desc,
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
                FIX_BLOCK,
            )
    else:
        bid = pid - num_desc - num_fix
        chunk = bid % blk_num_chunks
        seq = (bid // blk_num_chunks) % blk_num_seqs
        pool = bid // (blk_num_chunks * blk_num_seqs)
        _block_offsets_rows(
            blk_src_ptr,
            blk_dst_ptr,
            blk_copy_idx_ptr,
            blk_scales_ptr,
            blk_kv_offset_ptr,
            blk_src_max_seqs,
            blk_dst_max_seqs,
            blk_num_blocks,
            pool,
            seq,
            chunk,
            BLK_BLOCK,
        )


class _PinnedArena:
    """A pinned host byte buffer with typed numpy views for staging values."""

    def __init__(self, capacity: int):
        # The commit kernel reads the arena over the host link, so it must be pinned;
        # InputCommit is only created where prefer_pinned() holds.
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

    ``begin()`` opens a step, ``stage()`` records a copy (the host values are copied into a
    pinned arena immediately, so the caller may reuse its source right away),
    ``set_overlap_fixup()`` records the previous-step gathers, and ``commit()`` launches the
    kernel on the current stream. Two arenas alternate between steps; an event recorded
    after each commit kernel guards an arena's reuse.

    Args:
        capacity: Initial size in bytes of each of the two pinned arenas; an arena grows
            when a step stages more.
    """

    def __init__(self, capacity: int = 1 << 20):
        self._arenas = [_PinnedArena(capacity), _PinnedArena(capacity)]
        self._events: List[Optional[torch.cuda.Event]] = [None, None]
        self._index = 0
        self._arena = self._arenas[0]
        self._entries: List[Tuple[int, int, int]] = []
        # The staged destination byte ranges of the open step, sorted: (start, end, entry index).
        self._dst_ranges: List[Tuple[int, int, int]] = []
        # Verify only: the call site of each staged entry, and the restaging site pairs logged.
        self._entry_sites: dict = {}
        self._restage_sites: set = set()
        self._fixup: Optional[dict] = None
        # A KV block-offset copy run by the commit kernel (add_block_offsets).
        self._block_job: Optional[tuple] = None
        self._open = False
        self._verify_dsts: List[Tuple[torch.Tensor, int, int]] = []
        self._num_commits = 0
        # Off while warmup_input_commit_kernels runs: its commits are neither counted nor
        # verified.
        self._counting = True
        self._default_verify_commits = default_verify_commits()
        # Verify only: device byte ranges the open commit will write, and the contract
        # violations seen so far (_StagedHazardCheck).
        self._hazard_ranges: List[Tuple[int, int, str]] = []
        self._in_commit = False
        self._num_hazards = 0
        self._hazard_sites: set = set()
        # Device operations to run right after the step's commit kernel (after_commit).
        self._after_commit: List[Callable[[], None]] = []

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def num_commits(self) -> int:
        """How many step commits have launched their kernel (warmup excluded)."""
        return self._num_commits

    @property
    def _verify_commits(self) -> int:
        return _VERIFY_COMMITS if _VERIFY_COMMITS is not None else self._default_verify_commits

    def _verifying(self) -> bool:
        n = self._verify_commits
        return self._counting and (n < 0 or self._num_commits < n)

    def hazard_check(self):
        """While verifying: check the operators run in the block against the staged contract."""
        if not self._verifying():
            return nullcontext()
        _install_triton_hazard_hook()
        return _StagedHazardCheck(self)

    def _add_hazard_range(self, t: torch.Tensor, what: str) -> None:
        if self._verifying() and t.numel():
            start, end = _byte_range(t)
            self._hazard_ranges.append((start, end, f"{what} {t.dtype} x{t.numel()}"))

    def _hazard(self, op: str, overlaps: List[str]) -> None:
        """Record an operation that touched a destination the open commit writes later."""
        self._num_hazards += 1
        site = _caller_site()
        key = (op, site)
        if key in self._hazard_sites:
            return
        self._hazard_sites.add(key)
        logger.warning(
            f"InputCommit verify: {op} touches a destination before its commit "
            f"({', '.join(overlaps)}) at {site}"
        )

    def begin(self) -> None:
        """Open a step; waits until the kernel that last read the step's arena has run."""
        assert not self._open, "InputCommit.begin() called twice without commit()"
        self._index ^= 1
        event = self._events[self._index]
        if event is not None:
            event.synchronize()
        self._arena = self._arenas[self._index]
        self._arena.cursor = 0
        self._entries.clear()
        self._dst_ranges.clear()
        self._entry_sites.clear()
        self._fixup = None
        self._block_job = None
        self._verify_dsts.clear()
        self._hazard_ranges.clear()
        self._after_commit = []
        self._open = True

    def abort(self) -> None:
        """Close the step without launching anything."""
        self._entries.clear()
        self._dst_ranges.clear()
        self._entry_sites.clear()
        self._fixup = None
        self._block_job = None
        self._verify_dsts.clear()
        self._hazard_ranges.clear()
        self._after_commit = []
        self._open = False

    def _reserve(self, nbytes: int) -> int:
        if self._arena.cursor + nbytes > self._arena.capacity:
            self._grow(self._arena.cursor + nbytes)
        return self._arena.alloc(nbytes)

    def _grow(self, needed: int) -> None:
        # Rare. Replaces the arena of the open step (nothing reads it yet) by a larger one
        # and rebases the host addresses recorded so far.
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
        """Record ``dst.copy_(values)``.

        Args:
            dst: Contiguous CUDA tensor of one of the dtypes in ``_TORCH_TO_NUMPY``.
            values: ``dst.numel()`` host values: a sequence, a numpy array or a CPU tensor
                (converted to ``dst.dtype``). Copied into the arena before returning.
        """
        assert self._open
        assert dst.is_cuda and dst.is_contiguous(), "staged destinations must be contiguous"
        n = dst.numel()
        if n == 0:
            return
        np_dtype, itemsize = _TORCH_TO_NUMPY[dst.dtype]
        nbytes = n * itemsize
        start = dst.data_ptr()
        previous = self._staged_overlap(start, start + nbytes)
        if previous is not None:
            # Two copies of the same bytes in one commit kernel would race.
            prev_start, prev_end, index = previous
            if prev_start == start and prev_end == start + nbytes:
                # The same destination again: its staged bytes take the new values (the
                # copies' last-wins).
                if self._verifying():
                    self._log_restage(index, nbytes)
                offset = self._entries[index][1] - self._arena.base
                first = offset // itemsize
                self._write_values(self._arena.views[np_dtype][first : first + n], values, dst)
                return
            if self._verifying():
                raise RuntimeError(
                    f"InputCommit: a staged copy of {nbytes} bytes partly overlaps one staged "
                    f"earlier in the step ({prev_end - prev_start} bytes) at {_caller_site()}"
                )
            logger.warning_once(
                "InputCommit: partly overlapping staged copies in one step; committing the "
                f"earlier ones first ({_caller_site()})",
                key="input_commit_partial_overlap",
            )
            self.commit()
            self.begin()
        offset = self._reserve(nbytes)
        arena = self._arena
        first = offset // itemsize
        view = arena.views[np_dtype][first : first + n]
        self._write_values(view, values, dst)
        bisect.insort(self._dst_ranges, (start, start + nbytes, len(self._entries)))
        self._entries.append((start, arena.base + offset, nbytes))
        if self._verifying():
            self._entry_sites[len(self._entries) - 1] = _caller_sites()
            self._verify_dsts.append((dst, offset, nbytes))
            self._add_hazard_range(dst, "staged")

    def _log_restage(self, index: int, nbytes: int) -> None:
        """Verify only: log once per pair of call sites a destination staged twice in a step."""
        first = self._entry_sites.get(index, "?")
        second = _caller_sites()
        if (first, second) in self._restage_sites:
            return
        self._restage_sites.add((first, second))
        logger.info(
            f"InputCommit verify: a destination of {nbytes} bytes staged twice in one step, "
            f"the later values kept: first at {first}, again at {second}"
        )

    @staticmethod
    def _write_values(view: np.ndarray, values, dst: torch.Tensor) -> None:
        if isinstance(values, torch.Tensor):
            src = values.reshape(-1)
            if src.dtype != dst.dtype:
                src = src.to(dst.dtype)
            view[:] = src.numpy()
        else:
            view[:] = values

    def _staged_overlap(self, start: int, end: int) -> Optional[Tuple[int, int, int]]:
        """The staged range of the open step that overlaps ``[start, end)``, or None. The staged
        ranges never overlap one another, so their ends are sorted as their starts are: only the
        last range starting before ``end`` can reach past ``start``."""
        ranges = self._dst_ranges
        i = bisect.bisect_left(ranges, (end,))
        if i > 0 and ranges[i - 1][1] > start:
            return ranges[i - 1]
        return None

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
        """Record the overlap-scheduler gathers of the decoder input preparation.

        For previous-step request ``p`` with sampler slot ``s = previous_slots[p]``, writes
        ``input_ids[input_ids_start + p*T + t] = new_tokens[t, s, 0]``,
        ``draft_tokens[draft_tokens_start + p*D + j] = next_draft_tokens[s, j]``, and the
        whole ``pos_offsets`` / ``kv_offsets`` buffers: zero except
        ``pos_offsets[pos_offsets_start + p*T + t] = new_tokens_lens[s]`` and
        ``kv_offsets[kv_offsets_start + p] = new_tokens_lens[s] - T``
        (``T = tokens_per_req``, ``D = draft_width``).

        Args:
            previous_slots: Host sampler slots of the requests that ran in the previous step.
            tokens_per_req: Tokens per generation request ``T``.
            draft_width: Draft tokens per request ``D`` (``T - 1``).
            new_tokens: int32 ``[T_store, num_slots, 1]`` sampler store.
            new_tokens_lens: int32 ``[num_slots]`` accepted lengths.
            next_draft_tokens: int32 ``[num_slots, D_store]`` next drafts.
            input_ids: Contiguous int32 input-id buffer.
            input_ids_start: First row the gathered tokens go to.
            draft_tokens: Contiguous int32 draft-token buffer.
            draft_tokens_start: First row the gathered drafts go to.
            pos_offsets: Contiguous int32 per-token position corrections.
            pos_offsets_start: First token of the previous-step requests.
            kv_offsets: Contiguous int32 per-request KV-length corrections.
            kv_offsets_start: First previous-step request.
        """
        assert self._open and self._fixup is None
        assert overlap_fixup_supported(new_tokens, new_tokens_lens, next_draft_tokens)
        for t in (input_ids, draft_tokens, pos_offsets, kv_offsets):
            assert t.is_cuda and t.dtype == torch.int32 and t.is_contiguous()
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
        self._add_hazard_range(
            input_ids[input_ids_start : input_ids_start + num_prev * tokens_per_req],
            "fixup input_ids",
        )
        self._add_hazard_range(
            draft_tokens[draft_tokens_start : draft_tokens_start + num_prev * draft_width],
            "fixup draft_tokens",
        )
        self._add_hazard_range(pos_offsets, "fixup pos_offsets")
        self._add_hazard_range(kv_offsets, "fixup kv_offsets")

    def add_block_offsets(self, job: tuple) -> bool:
        """Record a KV block-offset copy for the commit kernel (the step shell's ``step_fold``).

        ``job`` is :func:`~.step_upload_kernels.block_offsets_copy_job`'s result. The kernel
        reads the pinned host table when it runs, like the copy it replaces, so the caller's
        host buffers must stay unchanged until the step's input copy has completed (the
        executor's ``wait_for_input_copy``). Returns False (nothing recorded) when the step
        already has one; the caller then copies directly.

        Contract: no device operation between this call and ``commit`` reads the
        destination rows.
        """
        assert self._open
        if self._block_job is not None:
            return False
        self._block_job = job
        host, dst, num_seqs = job[0], job[1], job[8]
        for pool in range(host.shape[0]):
            self._add_hazard_range(dst[pool, :num_seqs], "folded block offsets")
        return True

    def after_commit(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` in :meth:`commit`, right after the commit kernel on the same stream.

        For a device operation of the step's preparation that reads a staged destination: it
        then sees the committed bytes without a flush (an extra commit kernel). ``fn`` runs
        once, also when nothing was staged; an aborted step drops it.
        """
        assert self._open
        self._after_commit.append(fn)

    def commit(self) -> None:
        """Launch the commit kernel on the current stream, run the :meth:`after_commit`
        operations and close the step."""
        assert self._open
        self._in_commit = True
        try:
            self._commit()
            deferred, self._after_commit = self._after_commit, []
            for fn in deferred:
                fn()
        finally:
            self._in_commit = False
            self._hazard_ranges.clear()

    def _commit(self) -> None:
        block_job = self._block_job
        if block_job is not None and block_job[-1] == 0:
            block_job = None
        if not self._entries and self._fixup is None and block_job is None:
            self._open = False
            return
        num_desc = sum(_cdiv(nbytes, _COPY_CHUNK) for _, _, nbytes in self._entries)
        # Reserving may grow the arena and rebase the recorded host addresses, so the
        # entries and the fixup are read afterwards.
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
        if block_job is None:
            _input_commit_kernel[(num_desc + num_fix_programs,)](
                desc_tensor,
                num_desc,
                *args,
                CHUNK=_COPY_CHUNK,
                FIX_BLOCK=_FIX_BLOCK,
                HAS_FIX=fixup is not None,
            )
        else:
            _input_commit_blk_kernel[(num_desc + num_fix_programs + block_job[-1],)](
                desc_tensor,
                num_desc,
                *args,
                num_fix_programs,
                *block_job[:-1],
                CHUNK=_COPY_CHUNK,
                FIX_BLOCK=_FIX_BLOCK,
                HAS_FIX=fixup is not None,
                BLK_BLOCK=_BLOCK_OFFSETS_BLOCK,
            )
        event = self._events[self._index]
        if event is None:
            event = torch.cuda.Event()
            self._events[self._index] = event
        event.record()
        verifying = self._verifying()
        if block_job is not None and (verifying or STEP_FOLD.take_block_check()):
            self._verify_block_offsets(block_job)
        if verifying:
            self._verify(fixup)
            n = self._verify_commits
            if self._num_commits + 1 == n:
                if self._num_hazards == 0:
                    logger.info(f"InputCommit verify: the first {n} commits matched")
                else:
                    logger.warning(
                        f"InputCommit verify: the first {n} commits wrote the staged bytes, but "
                        f"{self._num_hazards} operations touched a destination before its commit"
                    )
        if self._counting:
            self._num_commits += 1
        self._entries = []
        self._dst_ranges = []
        self._entry_sites = {}
        self._fixup = None
        self._block_job = None

    @staticmethod
    def _verify_block_offsets(job: tuple) -> None:
        """Compare the committed block offsets with the standalone copy of the same job."""
        host, dst, copy_idx, scales, kv_offset = job[:5]
        torch.cuda.current_stream().synchronize()
        reference = dst.clone()
        assert copy_batch_block_offsets_from_host(
            host, reference, copy_idx, scales, kv_offset, note_launch=False
        )
        torch.cuda.current_stream().synchronize()
        if not torch.equal(dst, reference):
            raise RuntimeError("InputCommit verify: block offset mismatch")
        STEP_FOLD.count("block_offsets_checked")

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


def fold_block_offsets_copy(
    host_offsets: torch.Tensor,
    dst: torch.Tensor,
    copy_idx: torch.Tensor,
    index_scales: torch.Tensor,
    kv_offset: torch.Tensor,
    stream: Optional[torch.cuda.Stream] = None,
) -> bool:
    """Hand ``copy_batch_block_offsets_to_device``'s copy to the open input commit.

    Under the step shell's ``step_fold`` the commit kernel of the step does the copy (same
    values as the standalone copy, at the commit's position on the same stream). Returns
    False (the caller copies) when the switch is off for this step, no commit is open, the
    copy would run on another stream than the commit, the layout does not fit the Triton
    copy, or the commit already holds a copy.
    """
    if not STEP_FOLD.active:
        return False
    commit = active_input_commit()
    if commit is None:
        return False
    if stream is not None and stream != torch.cuda.current_stream():
        STEP_FOLD.count("block_offsets_other_stream")
        return False
    job = block_offsets_copy_job(host_offsets, dst, copy_idx, index_scales, kv_offset)
    if job is None or not commit.add_block_offsets(job):
        STEP_FOLD.count("block_offsets_not_folded")
        return False
    STEP_FOLD.count("block_offsets_folded")
    return True


def warmup_input_commit_kernels(commit: InputCommit) -> None:
    """Compile the commit kernel (with and without the overlap fixup) ahead of serving.

    Warmup and capture forwards do not use the commit, so its first use would otherwise
    pay the JIT compile. Does not count towards ``TRTLLM_BATCHED_INPUT_COMMIT_VERIFY``.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    buf = torch.zeros(64, dtype=torch.int32, device=device)
    new_tokens = torch.zeros((2, 2, 1), dtype=torch.int32, device=device)
    lens = torch.ones(2, dtype=torch.int32, device=device)
    draft = torch.zeros((2, 1), dtype=torch.int32, device=device)
    pos_off = torch.zeros(4, dtype=torch.int32, device=device)
    kv_off = torch.zeros(2, dtype=torch.int32, device=device)
    block_job = None
    if STEP_FOLD.enabled:
        # The step shell's step_fold: the variant with a KV block-offset copy too.
        pinned = dict(dtype=torch.int32, pin_memory=prefer_pinned())
        block_job = block_offsets_copy_job(
            torch.zeros((1, 1, 2, 4), **pinned),
            torch.zeros((1, 1, 2, 4), dtype=torch.int32, device=device),
            torch.zeros(1, **pinned),
            torch.ones(1, **pinned),
            torch.zeros(1, **pinned),
        )
    commit._counting = False
    try:
        for with_fixup, with_blocks in ((False, False), (True, False), (False, True), (True, True)):
            if with_blocks and block_job is None:
                continue
            with input_commit_step(commit):
                commit.stage(buf[:4], [1, 2, 3, 4])
                if with_blocks:
                    commit.add_block_offsets(block_job)
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
    finally:
        commit._counting = True
    torch.cuda.synchronize()


def _host_view(arena: _PinnedArena, address: int, count: int) -> torch.Tensor:
    offset = address - arena.base
    return arena.tensor[offset : offset + 4 * count].view(torch.int32)


def _byte_range(t: torch.Tensor) -> Tuple[int, int]:
    """Device bytes ``[start, end)`` that a tensor (view) spans."""
    span = 1 + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()) if size > 0)
    start = t.data_ptr()
    return start, start + span * t.element_size()


_THIS_FILE = os.path.abspath(__file__)


def _caller_site() -> str:
    """``file:line (function)`` of the innermost frame outside this module and torch."""
    for frame in reversed(traceback.extract_stack(limit=40)):
        name = os.path.abspath(frame.filename)
        if name == _THIS_FILE or "/torch/" in name or "/triton/" in name:
            continue
        return f"{frame.filename}:{frame.lineno} ({frame.name})"
    return "?"


def _caller_sites(depth: int = 3) -> str:
    """The innermost ``depth`` frames outside this module and torch, innermost first (staging
    goes through wrappers such as ``_stage_or_copy``, so one frame rarely names the caller)."""
    sites = []
    for frame in reversed(traceback.extract_stack(limit=40)):
        name = os.path.abspath(frame.filename)
        if name == _THIS_FILE or "/torch/" in name or "/triton/" in name:
            continue
        sites.append(f"{os.path.basename(frame.filename)}:{frame.lineno} ({frame.name})")
        if len(sites) == depth:
            break
    return " <- ".join(sites) or "?"


# Operators that only make views or read metadata: they touch no device memory.
_NO_DATA_OPS = frozenset(
    {
        "aten::detach",
        "aten::alias",
        "aten::lift_fresh",
        "aten::record_stream",
        "aten::size",
        "aten::stride",
        "aten::sym_size",
        "aten::sym_stride",
        "aten::sym_numel",
        "aten::sym_storage_offset",
        "aten::is_contiguous",
        "aten::_has_compatible_shallow_copy_type",
    }
)


class _StagedHazardCheck(TorchDispatchMode):
    """Verify only: flags operators and Triton launches of the preparing thread that touch a
    destination the open commit writes later (``stage`` / overlap fixup / folded block offsets).

    Such an operation sees (or writes) different bytes than with the copy-by-copy uploads, so
    the graph's inputs could differ. Covers torch operators (custom ops included) and Triton
    kernels launched through ``JITFunction.run``; device code launched by other means (CuTe DSL,
    direct bindings) is not seen.
    """

    def __init__(self, commit: "InputCommit"):
        super().__init__()
        self.commit = commit

    def __enter__(self):
        self._previous = getattr(_SCOPE, "hazard", None)
        _SCOPE.hazard = self
        return super().__enter__()

    def __exit__(self, *exc):
        _SCOPE.hazard = self._previous
        return super().__exit__(*exc)

    def overlaps(self, tensors) -> List[str]:
        commit = self.commit
        if commit._in_commit or not commit._hazard_ranges:
            return []
        found = []
        for t in tensors:
            if not isinstance(t, torch.Tensor) or not t.is_cuda or t.numel() == 0:
                continue
            start, end = _byte_range(t)
            for lo, hi, what in commit._hazard_ranges:
                if start < hi and lo < end:
                    found.append(what)
        return found

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name
        if not getattr(func, "is_view", False) and name not in _NO_DATA_OPS:
            tensors, _ = tree_flatten((args, kwargs))
            found = self.overlaps(tensors)
            if found:
                self.commit._hazard(str(func), found)
        return func(*args, **kwargs)

    def check_triton(self, kernel, args, kwargs) -> None:
        found = self.overlaps(list(args) + list(kwargs.values()))
        if found:
            self.commit._hazard(f"triton {getattr(kernel, '__name__', kernel)}", found)


def _install_triton_hazard_hook() -> None:
    """Route Triton launches through the open :class:`_StagedHazardCheck` (once per process)."""
    from triton.runtime.jit import JITFunction

    original = JITFunction.run
    if getattr(original, "_input_commit_hazard_hook", False):
        return

    def run(self, *args, **kwargs):
        check = getattr(_SCOPE, "hazard", None)
        if check is not None:
            check.check_triton(self, args, kwargs)
        return original(self, *args, **kwargs)

    run._input_commit_hazard_hook = True
    JITFunction.run = run

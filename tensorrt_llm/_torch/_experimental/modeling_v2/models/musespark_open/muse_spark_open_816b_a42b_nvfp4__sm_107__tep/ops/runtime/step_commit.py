# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One launch that performs a list of small 32-bit data movements recorded on the host.

A ``StepStaging`` is a pinned host block holding a descriptor table and the
bytes some descriptors read. The host records descriptors; ``launch`` issues
one Triton kernel that executes them, one program per (descriptor, 1024-element
chunk). The kernel reads the descriptor table and the recorded bytes directly
from pinned host memory (zero-copy over C2C), so a launch is a single stream
operation with no preceding host-to-device copy.

Descriptor kinds (every element is a 32-bit word):

* ``COPY``   ``dst[i] = staged[i]`` -- host bytes recorded with ``copy``.
* ``GATHER`` ``dst[i * dst_stride] = src[i * src_stride] + add`` -- device int32.
* ``FILL``   ``dst[i * dst_stride] = value``.
* ``BLOCKS`` the KV-cache block-offset transform over the first ``count``
  columns of one host page-index row: ``K[i] = scale * v`` and
  ``V[i] = scale * v + kv_offset`` in int32 arithmetic, and both are
  ``bad_page`` where the page index is ``BAD_PAGE_INDEX`` (-1). With
  ``bad_page`` 0 this is ``copyBatchBlockOffsetsToDeviceKernel``
  (``cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu``); with -1 it is
  ``KVCacheManagerV2``'s attention-op table conversion.

Programs run concurrently and in no particular order, so the descriptors of one
launch must write disjoint words and must not read a word another descriptor of
the same launch writes. Callers that need an order split it over launches.

Lifetime: the kernel reads the block after the host has moved on, so the block
must not be rewritten until the launch has executed. ``launch`` records an
event; ``reset`` waits for it before the block is reused.

Capacity: callers ask ``room`` before recording a group of descriptors and
write through their own path when it answers ``False``; recording past the end
of the block is a programming error and raises. A ``False`` answer also makes
the next ``reset`` double the block (up to ``MAX_DESCRIPTORS`` /
``MAX_DATA_BYTES``), so a batch larger than the initial size falls back for
one step and is staged from the next one on.

Descriptor read (``configure``'s ``v2_mode``, the ``step_commit_v2`` lever; ``""``
or ``"0"``, the default, keeps ``_step_commit_kernel``). Each of its programs loads the
descriptor's ``count`` and, when its chunk is in range, the six other fields:
two dependent uncached round trips to pinned host memory and seven requests per
warp before a COPY / BLOCKS payload read adds a third round trip, and seven of
the eight programs of a short descriptor in a launch with one long one exit
after the first. With ``"1"`` (or ``"split"``), ``_step_commit_kernel_v2`` loads
the eight fields as one 64-byte read and then performs ``_step_commit_kernel``'s
writes (the same words, the same values) on ``_V2_NUM_WARPS`` warps, and a
launch whose longest descriptor exceeds one chunk also cuts every descriptor
longer than a chunk into chunk-sized ones on the host (``_split_pieces``), so
every program has work and the grid is one-dimensional; the pieces go to a
pinned side table that lives like the block (released by ``reset``). With
``"nosplit"`` (A/B only) the new kernel runs on the previous two-dimensional grid.
Microbenchmark (Vera Rubin, ncu): commit 2 of TEP4 bs2 / bs3
11.4-12.5 / 10.4-12.3 us -> 5.3 us, commit 1 5.5-6.0 -> 4.0-4.3 us, bitwise equal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

KIND_COPY, KIND_GATHER, KIND_FILL, KIND_BLOCKS = 0, 1, 2, 3

#: int64 fields per descriptor: kind, dst, src, count, a4, a5, a6, a7.
DESC_FIELDS = 8
_DESC_BYTES = DESC_FIELDS * 8

#: Elements per program, and its warps: 8 int32 per thread.
_BLOCK = 1024
_NUM_WARPS = 4

#: Byte alignment of each COPY payload inside the data area.
_ALIGN = 16

#: Upper bounds for ``reset``'s growth of a block that ran out of room.
MAX_DESCRIPTORS = 1 << 16
MAX_DATA_BYTES = 1 << 24

#: The descriptor read (module docstring, ``configure``): "" / "0" off; "1" / "split" the 64-byte
#: read with host-split pieces; "nosplit" the 64-byte read on the previous grid.
_V2_MODE = ""
STEP_COMMIT_V2 = False
STEP_COMMIT_V2_SPLIT = False
#: Warps of a ``_step_commit_kernel_v2`` program (measured fastest of 1, 2 and 4 with the split).
_V2_NUM_WARPS = 4
#: Rows of the first pinned side table of pieces (``split``).
_PIECES_MIN_ROWS = 256


def configure(*, v2_mode: str = "") -> None:
    """Set the descriptor read of ``StepStaging.launch`` and ``warm_up`` (module docstring).

    ``lever_settings.apply`` sets it from the model's plan before any install.

    Args:
        v2_mode: ``""``, ``"0"``, ``"off"`` or ``"false"`` keep ``_step_commit_kernel``;
            ``"nosplit"`` runs ``_step_commit_kernel_v2`` on the previous two-dimensional grid;
            any other value (``"1"``, ``"split"``) runs it with the host-split pieces. Case and
            surrounding blanks are ignored.
    """
    global _V2_MODE, STEP_COMMIT_V2, STEP_COMMIT_V2_SPLIT
    mode = str(v2_mode).strip().lower()
    _V2_MODE = mode
    STEP_COMMIT_V2 = mode not in ("", "0", "off", "false")
    STEP_COMMIT_V2_SPLIT = STEP_COMMIT_V2 and mode != "nosplit"


@triton.jit(do_not_specialize=["desc_addr"])
def _step_commit_kernel(desc_addr, BLOCK: tl.constexpr):
    desc = desc_addr.to(tl.pointer_type(tl.int64)) + tl.program_id(0) * 8
    count = tl.load(desc + 3, cache_modifier=".cv")
    first = tl.program_id(1) * BLOCK
    if first < count:
        kind = tl.load(desc + 0, cache_modifier=".cv")
        dst = tl.load(desc + 1, cache_modifier=".cv").to(tl.pointer_type(tl.int32))
        src = tl.load(desc + 2, cache_modifier=".cv").to(tl.pointer_type(tl.int32))
        a4 = tl.load(desc + 4, cache_modifier=".cv")
        a5 = tl.load(desc + 5, cache_modifier=".cv")
        a6 = tl.load(desc + 6, cache_modifier=".cv")
        offs = first + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < count
        if kind == 0:
            values = tl.load(src + offs, mask=mask, cache_modifier=".cv")
            tl.store(dst + offs, values, mask=mask)
        elif kind == 1:
            values = tl.load(src + offs * a4, mask=mask)
            tl.store(dst + offs * a6, values + a5.to(tl.int32), mask=mask)
        elif kind == 2:
            values = tl.zeros([BLOCK], dtype=tl.int32) + a5.to(tl.int32)
            tl.store(dst + offs * a6, values, mask=mask)
        else:
            bad_page = tl.load(desc + 7, cache_modifier=".cv").to(tl.int32)
            pages = tl.load(src + offs, mask=mask, cache_modifier=".cv")
            scaled = pages * a4.to(tl.int32)
            bad = pages == -1
            tl.store(dst + offs, tl.where(bad, bad_page, scaled), mask=mask)
            dst_v = a6.to(tl.pointer_type(tl.int32))
            tl.store(dst_v + offs, tl.where(bad, bad_page, scaled + a5.to(tl.int32)), mask=mask)


@triton.jit(do_not_specialize=["desc_addr"])
def _step_commit_kernel_v2(desc_addr, BLOCK: tl.constexpr):
    """``_step_commit_kernel`` with the descriptor read as one 64-byte load (module docstring).

    The eight fields arrive together, so a program waits one round trip for its
    descriptor instead of two; the writes below are ``_step_commit_kernel``'s.
    """
    fields = tl.arange(0, 8)
    desc = desc_addr.to(tl.pointer_type(tl.int64)) + tl.program_id(0) * 8
    row = tl.load(desc + fields, cache_modifier=".cv")
    count = tl.sum(tl.where(fields == 3, row, 0), axis=0)
    first = tl.program_id(1) * BLOCK
    if first < count:
        kind = tl.sum(tl.where(fields == 0, row, 0), axis=0)
        dst = tl.sum(tl.where(fields == 1, row, 0), axis=0).to(tl.pointer_type(tl.int32))
        src = tl.sum(tl.where(fields == 2, row, 0), axis=0).to(tl.pointer_type(tl.int32))
        a4 = tl.sum(tl.where(fields == 4, row, 0), axis=0)
        a5 = tl.sum(tl.where(fields == 5, row, 0), axis=0)
        a6 = tl.sum(tl.where(fields == 6, row, 0), axis=0)
        offs = first + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < count
        if kind == 0:
            values = tl.load(src + offs, mask=mask, cache_modifier=".cv")
            tl.store(dst + offs, values, mask=mask)
        elif kind == 1:
            values = tl.load(src + offs * a4, mask=mask)
            tl.store(dst + offs * a6, values + a5.to(tl.int32), mask=mask)
        elif kind == 2:
            values = tl.zeros([BLOCK], dtype=tl.int32) + a5.to(tl.int32)
            tl.store(dst + offs * a6, values, mask=mask)
        else:
            pages = tl.load(src + offs, mask=mask, cache_modifier=".cv")
            scaled = pages * a4.to(tl.int32)
            bad = pages == -1
            tl.store(dst + offs, tl.where(bad, 0, scaled), mask=mask)
            dst_v = a6.to(tl.pointer_type(tl.int32))
            tl.store(dst_v + offs, tl.where(bad, 0, scaled + a5.to(tl.int32)), mask=mask)


def _split_pieces(rows: np.ndarray, block: int) -> np.ndarray:
    """``rows`` with every descriptor longer than ``block`` elements cut into descriptors of at most ``block``.

    Piece ``k`` of a descriptor covers its elements ``k * block`` to ``(k + 1) * block``:
    its destination (and a BLOCKS descriptor's V destination, ``a6``) and its source
    move by ``k * block`` elements times their strides (COPY and BLOCKS 1, GATHER
    ``a6`` / ``a4``, FILL ``a6`` and no source), and its count is what is left of the
    descriptor's, capped at ``block``. The pieces write the descriptor's words with
    its values. The rows of ``block`` elements or fewer come first, in their order,
    then the pieces; the input is not modified.

    Args:
        rows: int64 ``[k, DESC_FIELDS]``, every count positive.
        block: elements per program.

    Returns:
        int64 ``[k', DESC_FIELDS]``: ``rows`` itself when no count exceeds ``block``.
    """
    long = rows[:, 3] > block
    if not long.any():
        return rows
    big = rows[long]
    pieces = (big[:, 3] + block - 1) // block
    out = np.repeat(big, pieces, axis=0)
    first = np.repeat(np.cumsum(pieces) - pieces, pieces)
    offset = (np.arange(out.shape[0], dtype=np.int64) - first) * block
    kind = out[:, 0]
    strided = (kind == KIND_GATHER) | (kind == KIND_FILL)
    dst_step = np.where(strided, out[:, 6], 1)
    src_step = np.where(kind == KIND_GATHER, out[:, 4], np.where(kind == KIND_FILL, 0, 1))
    out[:, 3] = np.minimum(out[:, 3] - offset, block)
    out[:, 1] += offset * dst_step * 4
    out[:, 2] += offset * src_step * 4
    v_dst = kind == KIND_BLOCKS
    out[v_dst, 6] += offset[v_dst] * 4
    return np.concatenate((rows[~long], out))


# ---- The table in the kernel's parameter block ----------------------------------------------
# A ``StepStaging`` built with ``inline_commit`` (the ``step_loop_inline_commits`` fast path)
# runs a ``launch`` whose pending rows and payload words fit ``step_commit_inline``'s
# parameter block through that kernel instead (no pinned-memory reads; the same words); every
# other launch runs the kernel above.


def _inline_commit():
    """A new ``step_commit_inline.InlineCommit``, or None when the inline kernel is unavailable here."""
    from . import step_commit_inline

    return step_commit_inline.InlineCommit() if step_commit_inline.available() else None


# ---- End of the parameter-block launch ------------------------------------------------------


class StepStaging:
    """A pinned descriptor table plus a data area, executed by ``launch``.

    Descriptors accumulate between ``reset`` calls; each ``launch`` executes the
    ones recorded since the previous ``launch`` and leaves the block in use, so
    one step may launch more than once (to order writes across launches).

    Args:
        max_descriptors: descriptors the block holds initially.
        data_bytes: bytes of COPY payload the block holds initially.
        inline_commit: a launch whose pending rows fit a kernel parameter block
            passes its table there (``step_commit_inline``) instead of in pinned memory.
    """

    def __init__(
        self, max_descriptors: int = 1024, data_bytes: int = 1 << 18, inline_commit: bool = False
    ):
        self._allocate(max_descriptors, data_bytes)
        self.num = 0
        self.launched = 0
        self.data_used = 0
        self._max_count = 0
        self._event: Optional[torch.cuda.Event] = None
        #: Whether a launch since the last ``reset`` reads the block when it executes (``reset`` waits).
        self._fence = False
        #: Whether ``launch`` tries the parameter-block launch first (``_launch_inline``).
        self.inline_commit = inline_commit
        #: The parameter-block launcher, created on the first inline launch (False: unavailable).
        self._inline = None
        #: Set when ``room`` answered ``False``; ``reset`` then grows the block.
        self.short = False
        #: ``split``: the pinned side table of pieces, its rows in use since ``reset``, and
        #: tables a growth replaced while a launch could still read them (freed by ``reset``).
        self._pieces: Optional[torch.Tensor] = None
        self._pieces_np: Optional[np.ndarray] = None
        self._pieces_used = 0
        self._retired: list = []

    def _allocate(self, max_descriptors: int, data_bytes: int) -> None:
        # The kernel reads the block zero-copy from the device, so it must be pinned.
        if not prefer_pinned():
            raise RuntimeError(
                "StepStaging needs pinned host memory, which is unavailable here "
                "(no CUDA device, or confidential compute is enabled)"
            )
        self.desc = torch.zeros(
            (max_descriptors, DESC_FIELDS),
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.data = torch.zeros(
            data_bytes, dtype=torch.uint8, device="cpu", pin_memory=prefer_pinned()
        )
        self._desc_np = self.desc.numpy()
        self._data_np = self.data.numpy()
        self._desc_base = self.desc.data_ptr()
        self._data_base = self.data.data_ptr()

    @property
    def capacity(self) -> tuple[int, int]:
        """``(descriptors, data bytes)`` the block holds."""
        return self._desc_np.shape[0], self._data_np.shape[0]

    # Recording -------------------------------------------------------------

    def room(self, descriptors: int, data_bytes: int = 0, payloads: int = 0) -> bool:
        """Whether ``descriptors`` more descriptors fit, with ``payloads`` COPY payloads of ``data_bytes`` in all.

        Each payload may take up to ``_ALIGN - 1`` bytes of alignment padding.
        """
        fits = (
            self.num + descriptors <= self._desc_np.shape[0]
            and self.data_used + data_bytes + payloads * (_ALIGN - 1) <= self._data_np.shape[0]
        )
        if not fits:
            self.short = True
        return fits

    def _add(
        self, kind: int, dst: int, src: int, count: int, a4: int, a5: int, a6: int, a7: int = 0
    ) -> None:
        if count <= 0:
            return
        if self.num >= self._desc_np.shape[0]:
            raise RuntimeError(f"StepStaging: more than {self._desc_np.shape[0]} descriptors")
        self._desc_np[self.num] = (kind, dst, src, count, a4, a5, a6, a7)
        self.num += 1
        if count > self._max_count:
            self._max_count = count

    def extend(self, table: np.ndarray, max_count: Optional[int] = None) -> None:
        """Record a table of descriptors built elsewhere, in the layout ``_add`` writes.

        ``table`` is int64 ``[k, DESC_FIELDS]``; rows with a count of 0 or less
        are skipped, as ``_add`` skips them. A caller that passes ``max_count``
        states that every row's count is positive and at most ``max_count``, and
        the table is copied as it is. The caller asks ``room`` first.
        """
        if max_count is None:
            table = table[table[:, 3] > 0]
            if table.shape[0] == 0:
                return
            max_count = int(table[:, 3].max())
        k = table.shape[0]
        if k == 0:
            return
        if self.num + k > self._desc_np.shape[0]:
            raise RuntimeError(f"StepStaging: more than {self._desc_np.shape[0]} descriptors")
        self._desc_np[self.num : self.num + k] = table
        self.num += k
        if max_count > self._max_count:
            self._max_count = max_count

    def copy_bytes(self, dst_addr: int, payload: np.ndarray) -> None:
        """Record ``dst[i] = payload[i]`` for a uint8 payload whose size is a multiple of 4."""
        nbytes = payload.nbytes
        if nbytes % 4:
            raise ValueError(f"COPY payload of {nbytes} bytes is not whole 32-bit words")
        start = (self.data_used + _ALIGN - 1) // _ALIGN * _ALIGN
        if start + nbytes > self._data_np.shape[0]:
            raise RuntimeError(f"StepStaging: data area of {self._data_np.shape[0]} bytes is full")
        self._data_np[start : start + nbytes] = payload
        self.data_used = start + nbytes
        self._add(KIND_COPY, dst_addr, self._data_base + start, nbytes // 4, 0, 0, 0)

    def gather(
        self,
        dst_addr: int,
        dst_stride: int,
        src_addr: int,
        src_stride: int,
        count: int,
        add: int = 0,
    ) -> None:
        """Record ``dst[i * dst_stride] = src[i * src_stride] + add`` over int32 elements."""
        self._add(KIND_GATHER, dst_addr, src_addr, count, src_stride, add, dst_stride)

    def fill(self, dst_addr: int, dst_stride: int, count: int, value: int) -> None:
        """Record ``dst[i * dst_stride] = value`` over int32 elements."""
        self._add(KIND_FILL, dst_addr, 0, count, 0, value, dst_stride)

    def blocks(
        self,
        dst_k_addr: int,
        dst_v_addr: int,
        src_addr: int,
        count: int,
        scale: int,
        kv_offset: int,
        bad_page: int = 0,
    ) -> None:
        """Record the block-offset transform of one host page-index row's first ``count`` columns.

        ``bad_page`` is the word both lanes get for a ``BAD_PAGE_INDEX`` page.
        """
        self._add(KIND_BLOCKS, dst_k_addr, src_addr, count, scale, kv_offset, dst_v_addr, bad_page)

    # Execution -------------------------------------------------------------

    @property
    def pending(self) -> int:
        """Descriptors recorded since the last ``launch``."""
        return self.num - self.launched

    def launch(self) -> bool:
        """Execute the pending descriptors on the current stream; ``False`` if there were none.

        With ``inline_commit`` a launch whose rows fit the parameter block runs
        ``step_commit_inline``.
        """
        pending = self.num - self.launched
        if pending <= 0:
            return False
        if self.inline_commit and self._launch_inline():
            return True
        if STEP_COMMIT_V2_SPLIT and self._max_count > _BLOCK:
            self._launch_pieces(self._desc_np[self.launched : self.num])
        else:
            chunks = triton.cdiv(self._max_count, _BLOCK)
            if STEP_COMMIT_V2:
                _step_commit_kernel_v2[(pending, chunks)](
                    self._desc_base + self.launched * _DESC_BYTES,
                    BLOCK=_BLOCK,
                    num_warps=_V2_NUM_WARPS,
                )
            else:
                _step_commit_kernel[(pending, chunks)](
                    self._desc_base + self.launched * _DESC_BYTES,
                    BLOCK=_BLOCK,
                    num_warps=_NUM_WARPS,
                )
        self.launched = self.num
        self._max_count = 0
        if self._event is None:
            self._event = torch.cuda.Event()
        self._event.record()
        self._fence = True
        return True

    def _launch_inline(self) -> bool:
        """Launch the pending rows with their table in the parameter block; False if they do not fit.

        The parameter block is copied when the launch is issued, so this launch does not
        read the staging block afterwards and needs no fence in ``reset``.
        """
        if self._inline is None:
            self._inline = _inline_commit() or False
        if not self._inline or not self._inline.pack(
            self._desc_np[self.launched : self.num], self._data_np, self._data_base
        ):
            return False
        self._inline.launch()
        self.launched = self.num
        self._max_count = 0
        return True

    def _launch_pieces(self, rows: np.ndarray) -> None:
        """``split``: launch ``rows`` cut into chunk-sized pieces, one program each, from the side table."""
        pieces = _split_pieces(rows, _BLOCK)
        n = pieces.shape[0]
        if self._pieces_np is None or self._pieces_used + n > self._pieces_np.shape[0]:
            held = 0
            if self._pieces is not None:
                # A launch since the last reset may still read it; ``reset`` frees it.
                self._retired.append(self._pieces)
                held = self._pieces_np.shape[0]
            self._pieces = torch.zeros(
                (max(_PIECES_MIN_ROWS, 2 * n, 2 * held), DESC_FIELDS),
                dtype=torch.int64,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            self._pieces_np = self._pieces.numpy()
            self._pieces_used = 0
        start = self._pieces_used
        self._pieces_np[start : start + n] = pieces
        self._pieces_used = start + n
        _step_commit_kernel_v2[(n, 1)](
            self._pieces.data_ptr() + start * _DESC_BYTES, BLOCK=_BLOCK, num_warps=_V2_NUM_WARPS
        )

    def reset(self) -> None:
        """Make the block reusable: wait until every launch that reads it has executed.

        If ``room`` answered ``False`` since the last reset, the block is
        replaced by one twice as large (bounded by ``MAX_DESCRIPTORS`` /
        ``MAX_DATA_BYTES``); nothing reads the old one once the event has
        completed.
        """
        if self._event is not None and self._fence:
            self._event.synchronize()
        # Every launch since the last reset has executed, so the side table is free again.
        self._pieces_used = 0
        self._retired.clear()
        self._fence = False
        if self.short:
            self.short = False
            descriptors, data_bytes = self.capacity
            grown = (min(2 * descriptors, MAX_DESCRIPTORS), min(2 * data_bytes, MAX_DATA_BYTES))
            if grown != (descriptors, data_bytes):
                self._allocate(*grown)
        self.num = 0
        self.launched = 0
        self.data_used = 0
        self._max_count = 0


def warm_up(inline_commit: bool = False) -> None:
    """Build the step-commit launch now: one launch over a no-op descriptor list on the
    current device (through the parameter-block kernel when ``inline_commit`` and it is
    available, else the Triton kernel), checked against the expected words.

    Args:
        inline_commit: launch through a ``StepStaging`` built with ``inline_commit``, as the
            stagings it warms up are, so the parameter-block kernel is built (NVRTC) and checked.
    """
    staging = StepStaging(max_descriptors=4, data_bytes=64, inline_commit=inline_commit)
    scratch = torch.zeros(4, dtype=torch.int32, device="cuda")
    staging.fill(scratch.data_ptr(), 1, 4, 7)
    staging.launch()
    torch.cuda.current_stream().synchronize()
    if scratch.tolist() != [7, 7, 7, 7]:
        raise RuntimeError(f"step_commit warm-up wrote {scratch.tolist()}, expected [7, 7, 7, 7]")
    staging.reset()
    if STEP_COMMIT_V2:
        # One descriptor of more than two chunks: three programs (or, with ``split``, three pieces),
        # on a staging without the parameter-block launch, so that the v2 kernel itself is built.
        plain = StepStaging(max_descriptors=4, data_bytes=64) if inline_commit else staging
        long = torch.zeros(2 * _BLOCK + 3, dtype=torch.int32, device="cuda")
        plain.fill(long.data_ptr(), 1, long.numel(), 7)
        plain.launch()
        torch.cuda.current_stream().synchronize()
        if not bool((long == 7).all()):
            raise RuntimeError(
                "step_commit v2 warm-up: a fill of three chunks was not written whole"
            )
        plain.reset()
        logger.info(
            f"[musespark_open] step_commit descriptor read: v2_mode={_V2_MODE} "
            f"(split={STEP_COMMIT_V2_SPLIT}, warps={_V2_NUM_WARPS})"
        )
    if inline_commit:
        # Build the inline kernel now (NVRTC) and check one launch through it.
        staging.fill(scratch.data_ptr(), 1, 4, 9)
        staging.launch()
        torch.cuda.current_stream().synchronize()
        if scratch.tolist() != [9, 9, 9, 9]:
            raise RuntimeError(
                f"step_commit inline warm-up wrote {scratch.tolist()}, expected [9, 9, 9, 9]"
            )
        staging.reset()

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
* ``BLOCKS`` the KV-cache block-offset transform of
  ``copyBatchBlockOffsetsToDeviceKernel``
  (``cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu``) over the first
  ``count`` columns of one host page-index row: ``K[i] = 0`` if the page index
  is ``BAD_PAGE_INDEX`` (-1) else ``scale * v``, and ``V[i]`` the same plus
  ``kv_offset``, in int32 arithmetic as the C++ kernel.
* ``TABLE``  one lane of an attention-op page table row, as
  ``KVCacheManagerV2._copy_batch_block_offsets_by_op_pool`` converts it: ``(pool,
  scale, offset)`` read from the manager's device parameters, ``dst[i] = -1`` if
  ``i`` is past the snapshot's ``width`` or ``src[pool * pool_stride + i]`` is
  ``BAD_PAGE_INDEX``, else ``src[...] * scale + offset``, in int32 arithmetic.

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
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned

KIND_COPY, KIND_GATHER, KIND_FILL, KIND_BLOCKS, KIND_TABLE = 0, 1, 2, 3, 4

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
        elif kind == 3:
            pages = tl.load(src + offs, mask=mask, cache_modifier=".cv")
            scaled = pages * a4.to(tl.int32)
            bad = pages == -1
            tl.store(dst + offs, tl.where(bad, 0, scaled), mask=mask)
            dst_v = a6.to(tl.pointer_type(tl.int32))
            tl.store(dst_v + offs, tl.where(bad, 0, scaled + a5.to(tl.int32)), mask=mask)
        else:
            a7 = tl.load(desc + 7, cache_modifier=".cv")
            param = a4.to(tl.pointer_type(tl.int32))
            pool = tl.load(param).to(tl.int64)
            scale = tl.load(param + a5)
            offset = tl.load(param + 2 * a5)
            base = tl.load(
                src + pool * a7 + offs, mask=mask & (offs < a6), other=-1, cache_modifier=".cv"
            )
            tl.store(dst + offs, tl.where(base < 0, -1, base * scale + offset), mask=mask)


class StepStaging:
    """A pinned descriptor table plus a data area, executed by ``launch``.

    Descriptors accumulate between ``reset`` calls; each ``launch`` executes the
    ones recorded since the previous ``launch`` and leaves the block in use, so
    one step may launch more than once (to order writes across launches).
    """

    def __init__(self, max_descriptors: int = 1024, data_bytes: int = 1 << 18):
        self._allocate(max_descriptors, data_bytes)
        self.num = 0
        self.launched = 0
        self.data_used = 0
        self._max_count = 0
        self._event: Optional[torch.cuda.Event] = None
        #: Set when ``room`` answered ``False``; ``reset`` then grows the block.
        self.short = False

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

    def _add(self, kind: int, dst: int, src: int, count: int, a4: int, a5: int, a6: int) -> None:
        if count <= 0:
            return
        if self.num >= self._desc_np.shape[0]:
            raise RuntimeError(f"StepStaging: more than {self._desc_np.shape[0]} descriptors")
        self._desc_np[self.num] = (kind, dst, src, count, a4, a5, a6, 0)
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

    def stage(self, nbytes: int) -> tuple[int, np.ndarray]:
        """Reserve ``nbytes`` of the data area for descriptors to read: its address and a uint8 view."""
        start = (self.data_used + _ALIGN - 1) // _ALIGN * _ALIGN
        if start + nbytes > self._data_np.shape[0]:
            raise RuntimeError(f"StepStaging: data area of {self._data_np.shape[0]} bytes is full")
        self.data_used = start + nbytes
        return self._data_base + start, self._data_np[start : start + nbytes]

    def copy_bytes(self, dst_addr: int, payload: np.ndarray) -> None:
        """Record ``dst[i] = payload[i]`` for a uint8 payload whose size is a multiple of 4."""
        nbytes = payload.nbytes
        if nbytes % 4:
            raise ValueError(f"COPY payload of {nbytes} bytes is not whole 32-bit words")
        src_addr, staged = self.stage(nbytes)
        staged[:] = payload
        self._add(KIND_COPY, dst_addr, src_addr, nbytes // 4, 0, 0, 0)

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
    ) -> None:
        """Record the block-offset transform of one host page-index row's first ``count`` columns."""
        self._add(KIND_BLOCKS, dst_k_addr, src_addr, count, scale, kv_offset, dst_v_addr)

    def tables(
        self,
        dst_addrs: np.ndarray,
        src_addrs: np.ndarray,
        count: int,
        params_addrs: np.ndarray,
        params_stride: int,
        width: int,
        pool_stride: int,
    ) -> None:
        """Record attention-op table lanes from staged base page indices (``TABLE``), one per address.

        Each ``params_addrs`` entry points at its lane's int32 pool id, with the scale and
        offset ``params_stride`` and ``2 * params_stride`` elements further.
        """
        table = np.empty((dst_addrs.size, DESC_FIELDS), dtype=np.int64)
        table[:] = (KIND_TABLE, 0, 0, count, 0, params_stride, width, pool_stride)
        table[:, 1], table[:, 2], table[:, 4] = dst_addrs, src_addrs, params_addrs
        self.extend(table, max_count=count)

    # Execution -------------------------------------------------------------

    @property
    def pending(self) -> int:
        """Descriptors recorded since the last ``launch``."""
        return self.num - self.launched

    def launch(self) -> bool:
        """Execute the pending descriptors on the current stream; ``False`` if there were none."""
        pending = self.num - self.launched
        if pending <= 0:
            return False
        chunks = triton.cdiv(self._max_count, _BLOCK)
        _step_commit_kernel[(pending, chunks)](
            self._desc_base + self.launched * _DESC_BYTES, BLOCK=_BLOCK, num_warps=_NUM_WARPS
        )
        self.launched = self.num
        self._max_count = 0
        if self._event is None:
            self._event = torch.cuda.Event()
        self._event.record()
        return True

    def reset(self) -> None:
        """Make the block reusable: wait until every launch that reads it has executed.

        If ``room`` answered ``False`` since the last reset, the block is
        replaced by one twice as large (bounded by ``MAX_DESCRIPTORS`` /
        ``MAX_DATA_BYTES``); nothing reads the old one once the event has
        completed.
        """
        if self._event is not None and self.launched:
            self._event.synchronize()
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


def warm_up() -> None:
    """Compile the kernel now (one launch over a no-op descriptor list on the current device)."""
    staging = StepStaging(max_descriptors=4, data_bytes=64)
    scratch = torch.zeros(4, dtype=torch.int32, device="cuda")
    staging.fill(scratch.data_ptr(), 1, 4, 7)
    staging.launch()
    torch.cuda.current_stream().synchronize()
    if scratch.tolist() != [7, 7, 7, 7]:
        raise RuntimeError(f"step_commit warm-up wrote {scratch.tolist()}, expected [7, 7, 7, 7]")
    staging.reset()

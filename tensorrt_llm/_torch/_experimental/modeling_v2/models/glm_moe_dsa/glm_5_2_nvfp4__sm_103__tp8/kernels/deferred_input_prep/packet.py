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
"""One-launch application of many small device-buffer updates.

A ``PrepPacket`` collects the device writes of one input preparation as
descriptors (plus a byte payload for host values that are not already in
pinned memory) in pinned host memory. ``commit`` launches a single Triton
kernel, one CTA per descriptor chunk, that reads the packet and the pinned
host sources directly (zero-copy, no H2D memcpy op) and performs every write.
Descriptors of one packet must write disjoint bytes and must not read bytes
another descriptor of the same packet writes; the builder enforces the first
rule (``PacketOverlap``), the callers guarantee the second.

Descriptor kinds (32-bit words; a descriptor covers at most ``CHUNK`` words,
longer writes are split by the builder):

* ``COPY``: ``dst[i] = payload[i]``.
* ``GATHER``: ``dst[i] = src[idx[i]] + addend`` for ``i < n_idx`` and
  ``dst[i] = 0`` for ``n_idx <= i < n``; ``idx`` lives in the payload and
  ``src`` is a device buffer the packet does not write.
* ``BLOCK_OFFSETS``: the KV-cache-manager page-table expansion
  ``dst_k[i] = 0 if v == bad else scale * v`` and
  ``dst_v[i] = 0 if v == bad else scale * v + kv_offset`` with ``v`` read
  from a pinned host page-table row.
* ``CLAMP_MIN``: ``dst[i] = max(dst[i], lo)`` in place.
* ``COPY_CLAMP_MIN``: ``dst[i] = max(payload[i], lo)``.
* ``COPY_PTR`` / ``COPY_PTR_CLAMP_MIN``: as ``COPY`` / ``COPY_CLAMP_MIN`` but
  reading a pinned host source in place, at kernel execution time, exactly
  like the native asynchronous H2D copy from that source. The packet keeps
  the source tensor alive until its launch has completed.
"""

from typing import List, Optional, Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned

KIND_COPY = 0
KIND_GATHER = 1
KIND_BLOCK_OFFSETS = 2
KIND_CLAMP_MIN = 3
KIND_COPY_CLAMP_MIN = 4
KIND_COPY_PTR = 5
KIND_COPY_PTR_CLAMP_MIN = 6

# Words per descriptor: kind, dst, src, n, aux0..aux3.
DESC_WORDS = 8
CHUNK = 4096


class PacketFull(ValueError):
    """The packet has no room for this write; nothing of the write was added."""


class PacketOverlap(ValueError):
    """The write overlaps a destination of this packet; nothing was added.

    Descriptors of one launch run concurrently, so a later write to the same
    bytes must go into a later launch to keep last-write-wins order.
    """


@triton.jit
def _prep_commit_kernel(desc_addr, payload_addr, CHUNK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    desc = desc_addr.to(tl.pointer_type(tl.int64)) + pid * 8
    kind = tl.load(desc)
    dst = tl.load(desc + 1).to(tl.pointer_type(tl.int32))
    src_word = tl.load(desc + 2)
    n = tl.load(desc + 3).to(tl.int32)
    aux0 = tl.load(desc + 4)
    aux1 = tl.load(desc + 5)
    aux2 = tl.load(desc + 6)
    aux3 = tl.load(desc + 7)
    payload = payload_addr.to(tl.pointer_type(tl.int32))
    offs = tl.arange(0, CHUNK_SIZE)
    mask = offs < n
    if kind == 0:
        vals = tl.load(payload + src_word + offs, mask=mask)
        tl.store(dst + offs, vals, mask=mask)
    elif kind == 1:
        src = src_word.to(tl.pointer_type(tl.int32))
        n_idx = aux2.to(tl.int32)
        gmask = offs < n_idx
        idx = tl.load(payload + aux0 + offs, mask=gmask, other=0)
        vals = tl.load(src + idx, mask=gmask, other=0) + aux1.to(tl.int32)
        vals = tl.where(gmask, vals, 0)
        tl.store(dst + offs, vals, mask=mask)
    elif kind == 2:
        src = src_word.to(tl.pointer_type(tl.int32))
        dst_v = aux0.to(tl.pointer_type(tl.int32))
        scale = aux1.to(tl.int32)
        kv_off = aux2.to(tl.int32)
        bad = aux3.to(tl.int32)
        v = tl.load(src + offs, mask=mask, other=0)
        is_bad = v == bad
        k_val = tl.where(is_bad, 0, scale * v)
        v_val = tl.where(is_bad, 0, scale * v + kv_off)
        tl.store(dst + offs, k_val, mask=mask)
        tl.store(dst_v + offs, v_val, mask=mask)
    elif kind == 3:
        lo = aux0.to(tl.int32)
        vals = tl.load(dst + offs, mask=mask, other=0)
        tl.store(dst + offs, tl.maximum(vals, lo), mask=mask)
    elif kind == 4:
        lo = aux0.to(tl.int32)
        vals = tl.load(payload + src_word + offs, mask=mask, other=0)
        tl.store(dst + offs, tl.maximum(vals, lo), mask=mask)
    elif kind == 5:
        src = src_word.to(tl.pointer_type(tl.int32))
        vals = tl.load(src + offs, mask=mask)
        tl.store(dst + offs, vals, mask=mask)
    else:
        src = src_word.to(tl.pointer_type(tl.int32))
        lo = aux0.to(tl.int32)
        vals = tl.load(src + offs, mask=mask, other=0)
        tl.store(dst + offs, tl.maximum(vals, lo), mask=mask)


class _NeverDone:
    @staticmethod
    def query() -> bool:
        return False


_NEVER_DONE = _NeverDone()


class _PinnedSlot:
    """One packet's pinned storage and the event that marks its consumption."""

    # The commit kernel reads this storage in place, so it must be pinned. The stager
    # declines when prefer_pinned() is False, so a packet is only built when it is True.

    def __init__(self, max_desc: int, payload_words: int):
        self.desc = torch.zeros(
            (max_desc, DESC_WORDS), dtype=torch.int64, pin_memory=prefer_pinned()
        )
        self.payload = torch.zeros((payload_words,), dtype=torch.int32, pin_memory=prefer_pinned())
        self.desc_np = self.desc.numpy()
        self.payload_np = self.payload.numpy()
        self.event: Optional[torch.cuda.Event] = None
        # Pinned sources read in place by this slot's launch.
        self.keepalive: List[torch.Tensor] = []

    def free(self) -> bool:
        return self.event is None or self.event.query()


class PrepPacket:
    """Builder for one commit; reuses a ring of pinned slots without host syncs.

    A slot is reused only after the CUDA event recorded behind its commit has
    completed (polled with ``query``); otherwise a new slot is allocated, so
    building a packet never blocks the host.
    """

    def __init__(self, max_desc: int = 4096, payload_words: int = 1 << 18):
        self.max_desc = max_desc
        self.payload_words = payload_words
        self._slots: List[_PinnedSlot] = []
        self._cur: Optional[_PinnedSlot] = None
        self._descs: List[tuple] = []
        self._n_payload = 0
        self._ranges: List[tuple] = []

    # ---- building -------------------------------------------------------

    def begin(self) -> None:
        slot = next((s for s in self._slots if s.free()), None)
        if slot is None:
            slot = _PinnedSlot(self.max_desc, self.payload_words)
            self._slots.append(slot)
        slot.keepalive = []
        self._cur = slot
        self._descs = []
        self._n_payload = 0
        self._ranges = []

    def abort(self) -> None:
        if self._cur is not None:
            self._cur.keepalive = []
        self._cur = None
        self._descs = []

    @property
    def num_descriptors(self) -> int:
        return len(self._descs)

    def _reserve(self, n_desc: int, n_words: int) -> None:
        if len(self._descs) + n_desc > self.max_desc:
            raise PacketFull("packet descriptor table is full")
        if self._n_payload + n_words > self.payload_words:
            raise PacketFull("packet payload is full")

    @staticmethod
    def _dst_ptr(dst: torch.Tensor, nbytes: int) -> int:
        if not dst.is_cuda:
            raise ValueError("packet destinations must be CUDA tensors")
        if not dst.is_contiguous():
            raise ValueError("packet destinations must be contiguous")
        ptr = dst.data_ptr()
        if ptr % 4 or nbytes % 4:
            raise ValueError("packet destinations must be 4-byte aligned words")
        return ptr

    def _check_dsts(self, *dsts) -> List[int]:
        """Validate ``(tensor, nbytes)`` destinations and claim their bytes.

        All-or-nothing: on ``PacketOverlap`` no range is claimed.
        """
        ptrs = [self._dst_ptr(dst, nbytes) for dst, nbytes in dsts]
        new = [(p, p + nbytes) for p, (_, nbytes) in zip(ptrs, dsts)]
        for lo, hi in new:
            for a, b in self._ranges:
                if lo < b and a < hi:
                    raise PacketOverlap("overlapping destinations in one packet")
        self._ranges.extend(new)
        return ptrs

    def _check_dst(self, dst: torch.Tensor, nbytes: int) -> int:
        return self._check_dsts((dst, nbytes))[0]

    def _alloc_payload(self, nwords: int) -> int:
        off = self._n_payload
        self._n_payload = off + nwords
        return off

    def _emit_split(self, kind, dst_ptr, src, n, aux, src_step, aux0_step=0):
        """Append ``kind`` descriptors covering ``n`` words in ``CHUNK`` pieces.

        ``src_step`` / ``aux0_step``: per-word advance of ``src`` / ``aux0``.
        """
        a0, a1, a2, a3 = aux
        self._descs.extend(
            (
                kind,
                dst_ptr + 4 * s,
                src + src_step * s,
                min(CHUNK, n - s),
                a0 + aux0_step * s,
                a1,
                a2,
                a3,
            )
            for s in range(0, n, CHUNK)
        )

    @staticmethod
    def _num_chunks(n: int) -> int:
        return (n + CHUNK - 1) // CHUNK

    def add_copy(self, dst: torch.Tensor, src: torch.Tensor, clamp_min: Optional[int] = None):
        """Stage ``dst.copy_(src)`` for a CPU ``src`` with ``dst``'s dtype and shape.

        A contiguous pinned ``src`` is read in place by the kernel; any other
        ``src`` is copied into the payload now.
        """
        if src.dtype != dst.dtype or src.numel() != dst.numel():
            raise ValueError("add_copy needs matching dtype and element count")
        nbytes = dst.numel() * dst.element_size()
        nwords = nbytes // 4
        in_place = src.is_contiguous() and src.is_pinned() and src.data_ptr() % 4 == 0
        self._reserve(self._num_chunks(nwords), 0 if in_place else nwords)
        ptr = self._check_dst(dst, nbytes)
        lo = 0 if clamp_min is None else clamp_min
        if in_place:
            self._cur.keepalive.append(src)
            kind = KIND_COPY_PTR if clamp_min is None else KIND_COPY_PTR_CLAMP_MIN
            self._emit_split(kind, ptr, src.data_ptr(), nwords, (lo, 0, 0, 0), 4)
            return
        off = self._alloc_payload(nwords)
        src_words = src.contiguous().reshape(-1).view(torch.int32).numpy()
        self._cur.payload_np[off : off + nwords] = src_words
        kind = KIND_COPY if clamp_min is None else KIND_COPY_CLAMP_MIN
        self._emit_split(kind, ptr, off, nwords, (lo, 0, 0, 0), 1)

    def add_values(self, dst: torch.Tensor, values: Sequence[int]):
        """Stage ``dst.copy_(torch.tensor(values, dtype=dst.dtype))`` for int32 ``dst``."""
        if dst.dtype != torch.int32 or dst.numel() != len(values):
            raise ValueError("add_values needs an int32 destination of len(values)")
        n = len(values)
        self._reserve(self._num_chunks(n), n)
        ptr = self._check_dst(dst, 4 * n)
        off = self._alloc_payload(n)
        self._cur.payload_np[off : off + n] = values
        self._emit_split(KIND_COPY, ptr, off, n, (0, 0, 0, 0), 1)

    def add_gather(
        self, dst: torch.Tensor, src: torch.Tensor, indices: Sequence[int], addend: int = 0
    ):
        """Stage ``dst[:k] = src.flatten()[indices] + addend`` and ``dst[k:] = 0``.

        ``src`` is an int32 device buffer that this packet does not write;
        ``indices`` are flat element offsets into ``src``'s storage.
        """
        if dst.dtype != torch.int32 or src.dtype != torch.int32:
            raise ValueError("add_gather supports int32 only")
        n = dst.numel()
        if len(indices) > n or n > CHUNK:
            raise ValueError("add_gather: too many indices or destination too long")
        k = max(1, len(indices))
        self._reserve(1, k)
        ptr = self._check_dst(dst, 4 * n)
        off = self._alloc_payload(k)
        if indices:
            self._cur.payload_np[off : off + len(indices)] = indices
        self._descs.append((KIND_GATHER, ptr, src.data_ptr(), n, off, addend, len(indices), 0))

    def add_zero(self, dst: torch.Tensor):
        """Stage ``dst.zero_()`` for a 4-byte-word buffer."""
        nbytes = dst.numel() * dst.element_size()
        self._reserve(self._num_chunks(nbytes // 4), 0)
        ptr = self._check_dst(dst, nbytes)
        # A GATHER without indices writes zeros.
        self._emit_split(KIND_GATHER, ptr, 0, nbytes // 4, (0, 0, 0, 0), 0)

    def add_block_offsets(
        self,
        dst_k: torch.Tensor,
        dst_v: torch.Tensor,
        src_host_ptr: int,
        n: int,
        scale: int,
        kv_offset: int,
        bad: int,
    ):
        """Stage one (pool, sequence) page-table row expansion."""
        self._reserve(self._num_chunks(n), 0)
        ptr_k, ptr_v = self._check_dsts((dst_k, 4 * n), (dst_v, 4 * n))
        self._emit_split(
            KIND_BLOCK_OFFSETS,
            ptr_k,
            src_host_ptr,
            n,
            (ptr_v, scale, kv_offset, bad),
            4,
            aux0_step=4,
        )

    def add_clamp_min(self, dst: torch.Tensor, lo: int):
        """Stage ``dst.clamp_(min=lo)`` for an int32 buffer."""
        if dst.dtype != torch.int32:
            raise ValueError("add_clamp_min supports int32 only")
        n = dst.numel()
        self._reserve(self._num_chunks(n), 0)
        ptr = self._check_dst(dst, 4 * n)
        self._emit_split(KIND_CLAMP_MIN, ptr, 0, n, (lo, 0, 0, 0), 0)

    # ---- launch ---------------------------------------------------------

    def commit(self, stream: Optional[torch.cuda.Stream] = None, record_event: bool = True) -> int:
        """Launch the packet on ``stream`` (default: current).

        The slot is reused once an event recorded after the launch completes.
        With ``record_event=False`` the caller records one of its own events
        right after the launch and passes it to ``set_completion_event`` (an
        event record costs GPU time between tiny kernels).
        """
        slot = self._cur
        descs = self._descs
        self._cur = None
        self._descs = []
        n = len(descs)
        if n == 0:
            if slot is not None:
                slot.keepalive = []
            return 0
        slot.desc_np[:n] = np.array(descs, dtype=np.int64)
        stream = stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            _prep_commit_kernel[(n,)](
                slot.desc.data_ptr(), slot.payload.data_ptr(), CHUNK_SIZE=CHUNK, num_warps=4
            )
            if record_event:
                slot.event = torch.cuda.Event()
                slot.event.record(stream)
            else:
                # Not reusable until the caller provides the completion event.
                slot.event = _NEVER_DONE
                self._last_slot = slot
        return n

    def set_completion_event(self, event: torch.cuda.Event) -> None:
        """Mark the last launch consumed once ``event`` (recorded after it) completes."""
        slot = getattr(self, "_last_slot", None)
        if slot is not None:
            slot.event = event
            self._last_slot = None

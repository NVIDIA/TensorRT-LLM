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
"""One symmetric-memory allocation per process group, carved into slots and per-rank regions.

Why
---
A copy-engine push transport (the VisualGen token-sharded TP all-gather in
``tensorrt_llm/_torch/visual_gen/parallel/token_sharded_ce_gather.py``) writes each rank's rows
straight into every peer's memory, so the peers need a buffer that is (a) mapped into this
process's address space, (b) laid out identically on every rank and (c) allocated once, outside
CUDA-graph capture. ``torch.distributed._symmetric_memory`` gives exactly that: ``empty`` +
``rendezvous`` return a handle whose ``get_buffer(peer, ...)`` is a tensor view of the peer's copy
of the allocation and whose signal pad carries the ``put_signal`` / ``wait_signal`` flags that
order a consumer's reads after a producer's copies.

Layout
------
A :class:`PoolLayout` carves the allocation into ``slots`` slots of ``world_size`` regions of
``region_nbytes`` bytes each (a multiple of ``align``). Region ``r`` of a slot belongs to source
rank ``r``: it holds the bytes rank ``r`` pushed into that slot on every rank. A producer writes
its own region inside a peer's buffer (:meth:`SymmMemPool.peer_region`); a consumer reads source
``src``'s region of its local buffer (:meth:`SymmMemPool.region`). What the bytes mean (payload,
scales, ...) is the caller's layout, expressed as ``(offset, nbytes)`` inside a region. Several
slots let boundary ``b + 1`` be pushed while slot ``b`` is still being read.

Addressing
----------
``symm_mem.empty`` allocates out of an implicit memory pool, so the pool tensor may start at a
non-zero byte offset inside its symmetric *block* (``handle.offset``). ``handle.get_buffer``
addresses the peer's block base, not the tensor, so every peer view adds ``handle.offset`` (as
PyTorch's own ``symm_mem.get`` does). That frame of reference is an implementation detail of the
torch build, so every allocation ends with a collective marker check: each rank pushes a
rank-specific marker through :meth:`SymmMemPool.peer_region` and every rank reads all markers
back through :meth:`SymmMemPool.region`. A mismatch is reported at allocation time (the probe
declines, :meth:`SymmMemPool.reserve` raises) instead of pushing bytes next to a slot at run time.

Rules
-----
* :meth:`SymmMemPool.reserve` is collective (``all_gather_object`` of the request, every rank
  takes the maximum), grow-only and eager-only: it raises under CUDA-graph capture, so size the
  pool for the largest boundary in an eager warm-up before capturing.
* Signals: :meth:`SymmMemPool.signal` sets flag ``(channel, me -> peer)`` on the peer's pad and
  spins while the previous flag is still unconsumed; :meth:`SymmMemPool.wait` spins until flag
  ``(channel, src -> me)`` is set, then clears it. One put per wait. Channels
  ``0 .. channels - 1`` belong to the caller; the pad's last channel is reserved for
  :meth:`SymmMemPool.barrier`. Two protocols on one pool must use disjoint channels.
* Signal pad: the pool zeroes this rank's pad on every allocation, fenced before any peer can
  signal, so a protocol may start from "no flag set" after each :meth:`SymmMemPool.reserve`
  that reallocated. ``symm_mem.empty`` sub-allocates from torch's implicit symmetric memory
  pool, whose segments (and their pads) are cached and re-handed out, so a pad is only known
  zero because the pool clears it; it is shared with any other ``symm_mem`` tensor that lands
  in the same segment, so the pool's channels are exclusive only while it is the process's sole
  symmetric-memory user.
* Timeouts: a wait that exceeds ``timeout_ms`` is a device-side trap (PyTorch's signal kernels
  ``__trap()``: the CUDA context is lost and the process must exit). The spin starts when the GPU
  reaches the wait, so host-side skew between ranks (a recompile, a JIT, a collective) counts
  against it; the default of 10 minutes is a hang detector, not a latency bound. Triage: when 3
  of 4 ranks trap, the fourth rank raised between its push and its consumed signal; read that
  rank's traceback.
* Environment: nothing is required. The pool never binds a symmetric-memory backend
  (``set_backend``), reads no ``TORCH_SYMM_MEM_*`` variable and does not use multicast, so a
  failed NVLS multicast binding does not concern it.
* :meth:`SymmMemPool.release` before ``destroy_process_group``: peers may still map this rank's
  buffer, so it fences the group before unmapping.
"""

import gc
from dataclasses import dataclass
from types import ModuleType

import torch
import torch.distributed as dist

from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import pad_up

__all__ = [
    "DEFAULT_ALIGN",
    "DEFAULT_TIMEOUT_MS",
    "PoolLayout",
    "SymmMemPool",
    "probe_symmetric_memory",
]

DEFAULT_ALIGN = 256
"""int: Default region alignment in bytes (covers every dtype view a caller may take)."""

DEFAULT_TIMEOUT_MS = 600_000
"""int: Default device-side signal deadline, NCCL's 10-minute watchdog rather than the latency of a
healthy boundary (see the module docstring); ``0`` disables the deadline."""

_SIGNAL_BYTES = 4  # one uint32 flag per (channel, rank) on the signal pad
_MARKER_BYTES = 8  # one int64 marker per rank for the addressing check
_TIMEOUT_TRIAGE_HINT = (
    "a signal wait past the deadline is a device-side trap; when 3 of 4 ranks trap, the fourth "
    "rank raised between its push and its consumed signal: read that rank's traceback"
)


def _symm_mem() -> ModuleType:
    # Imported lazily so this module (and its probe) loads on a torch build without the symmetric
    # memory module; the probe reports the ImportError as the reason instead.
    import torch.distributed._symmetric_memory as symm_mem

    return symm_mem


def _is_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _check_align(align: int, who: str) -> None:
    if align < 16 or align & (align - 1):
        raise ValueError(f"{who}: align must be a power of two >= 16 (got {align}).")


# =============================================================================
# Layout (pure ints)
# =============================================================================


@dataclass(frozen=True)
class PoolLayout:
    """How a pool's bytes are carved: ``slots`` slots, each ``world_size`` regions of
    ``region_nbytes`` bytes, in rank order.

    Region ``r`` of every slot belongs to source rank ``r``. Python ints only (CUDA-graph safe).

    Attributes:
        world_size: Ranks of the group, hence regions per slot.
        slots: Slots carved out of the allocation.
        region_nbytes: Bytes per region, a positive multiple of ``align``.
        align: Region alignment in bytes (a power of two >= 16).
    """

    world_size: int
    slots: int
    region_nbytes: int
    align: int = DEFAULT_ALIGN

    def __post_init__(self) -> None:
        _check_align(self.align, "PoolLayout")
        if self.world_size < 1 or self.slots < 1:
            raise ValueError(
                f"PoolLayout needs world_size >= 1 and slots >= 1 (got {self.world_size}, "
                f"{self.slots})."
            )
        if self.region_nbytes < 1 or self.region_nbytes % self.align:
            raise ValueError(
                f"PoolLayout needs region_nbytes a positive multiple of align={self.align} (got "
                f"{self.region_nbytes})."
            )

    @classmethod
    def for_request(
        cls, slot_nbytes: int, slots: int, world_size: int, align: int = DEFAULT_ALIGN
    ) -> "PoolLayout":
        """The smallest layout whose slots hold ``slot_nbytes`` bytes as ``world_size`` aligned
        regions."""
        if slot_nbytes < 1 or world_size < 1:
            raise ValueError(
                f"PoolLayout.for_request needs slot_nbytes >= 1 and world_size >= 1 (got "
                f"{slot_nbytes}, {world_size})."
            )
        _check_align(align, "PoolLayout.for_request")
        region = pad_up(-(-slot_nbytes // world_size), align)
        return cls(world_size, slots, region, align)

    @property
    def slot_nbytes(self) -> int:
        return self.world_size * self.region_nbytes

    @property
    def nbytes(self) -> int:
        return self.slots * self.slot_nbytes

    def covers(self, other: "PoolLayout") -> bool:
        """Whether an allocation of this layout holds every region of ``other``."""
        return (
            self.world_size == other.world_size
            and self.align == other.align
            and self.slots >= other.slots
            and self.region_nbytes >= other.region_nbytes
        )

    def merged(self, other: "PoolLayout") -> "PoolLayout":
        """The grow-only union of two layouts of the same group: max slots, max region bytes."""
        if self.world_size != other.world_size or self.align != other.align:
            raise ValueError(f"PoolLayout.merged: {self} and {other} describe different pools.")
        return PoolLayout(
            self.world_size,
            max(self.slots, other.slots),
            max(self.region_nbytes, other.region_nbytes),
            self.align,
        )

    def region_start(self, slot: int, src_rank: int) -> int:
        """Byte offset of source ``src_rank``'s region in slot ``slot``."""
        if not 0 <= slot < self.slots:
            raise ValueError(f"slot {slot} out of range (layout has {self.slots} slots).")
        if not 0 <= src_rank < self.world_size:
            raise ValueError(
                f"source rank {src_rank} out of range (layout has {self.world_size} regions)."
            )
        return slot * self.slot_nbytes + src_rank * self.region_nbytes

    def region_span(self, slot: int, src_rank: int, offset: int, nbytes: int) -> tuple[int, int]:
        """``(start, stop)`` byte offsets of ``nbytes`` bytes at ``offset`` inside source
        ``src_rank``'s region of slot ``slot``; raises when the span leaves the region."""
        if nbytes < 1 or offset < 0 or offset + nbytes > self.region_nbytes:
            raise ValueError(
                f"span offset={offset} nbytes={nbytes} does not fit a region of "
                f"{self.region_nbytes} bytes."
            )
        start = self.region_start(slot, src_rank) + offset
        return start, start + nbytes


# =============================================================================
# Rank agreement
# =============================================================================


def _agree(group: dist.ProcessGroup, ok: bool, reason: str) -> tuple[bool, str]:
    """The MIN of ``ok`` over the group's ranks (collective), with the failing ranks' reasons.

    Carried by ``all_gather_object`` rather than an ``all_reduce`` so that every rank learns
    which rank declined and why (the one line the caller logs must name the reason).
    """
    verdicts: list[tuple[bool, str] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(verdicts, (bool(ok), reason), group=group)
    failures = []
    for rank, verdict in enumerate(verdicts):
        good, why = verdict if verdict is not None else (False, "no verdict received")
        if not good:
            failures.append(f"rank {rank}: {why}")
    return (True, "") if not failures else (False, "; ".join(failures))


def _nccl_backend_reasons(group: dist.ProcessGroup, device: torch.device) -> list[str]:
    pg_nccl = getattr(dist, "ProcessGroupNCCL", None)
    if pg_nccl is None:
        return ["this torch build has no NCCL process-group backend"]
    try:
        backend = group._get_backend(device)
    except RuntimeError as e:
        return [f"the group has no backend for {device} ({e})"]
    if not isinstance(backend, pg_nccl):
        return [
            f"{type(backend).__name__} serves {device} in the group; symmetric memory needs a "
            "ProcessGroupNCCL (a CUDA NCCL group)"
        ]
    return []


def _local_preconditions(group: dist.ProcessGroup, device: torch.device) -> list[str]:
    """Everything that must hold on this rank before a rendezvous is attempted; empty when OK."""
    reasons: list[str] = []
    try:
        _symm_mem()
    except ImportError as e:
        reasons.append(f"torch.distributed._symmetric_memory is not importable ({e})")
    world_size = dist.get_world_size(group)
    if world_size < 2:
        reasons.append(f"the group has {world_size} rank(s); symmetric memory needs >= 2 ranks")
    if device.type != "cuda":
        reasons.append(f"device {device} is not a CUDA device")
    elif not torch.cuda.is_available():
        reasons.append("CUDA is not available")
    else:
        current = torch.cuda.current_device()
        index = current if device.index is None else device.index
        if index != current:
            reasons.append(
                f"device {device} is not the current CUDA device (cuda:{current}); call "
                "torch.cuda.set_device first so the signal kernels run on the pool's device"
            )
    reasons.extend(_nccl_backend_reasons(group, device))
    return reasons


# =============================================================================
# Pool
# =============================================================================


class SymmMemPool:
    """One symmetric-memory allocation of a process group, carved by a :class:`PoolLayout`.

    Lifecycle: construct on every rank (cheap, no allocation), :meth:`reserve` eagerly and
    collectively, take views and signal per boundary, :meth:`release` before the group is
    destroyed. Decide eligibility beforehand with :func:`probe_symmetric_memory`; on an
    ineligible group :meth:`reserve` raises.

    Args:
        group: The process group whose ranks share the pool (a CUDA group served by NCCL).
        device: This rank's CUDA device; must be the current device.
        timeout_ms: Device-side deadline of :meth:`wait` / :meth:`signal` / :meth:`barrier` in
            milliseconds, ``0`` for none. See the module docstring on what a timeout means.
        align: Region alignment in bytes (a power of two >= 16).
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device | str,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        *,
        align: int = DEFAULT_ALIGN,
    ) -> None:
        if group is None:
            raise ValueError("SymmMemPool needs a torch.distributed process group; got None.")
        if timeout_ms < 0:
            raise ValueError(f"SymmMemPool needs timeout_ms >= 0 (got {timeout_ms}).")
        _check_align(align, "SymmMemPool")
        self.group = group
        self.group_name: str = group.group_name
        self.rank: int = dist.get_rank(group)
        self.world_size: int = dist.get_world_size(group)
        self.device = torch.device(device)
        self.timeout_ms: int = timeout_ms
        self.align: int = align
        self.generation: int = 0
        """int: Incremented per allocation. Each allocation zeroes this rank's signal pad and has
        new addresses, so protocols on the pool restart from their first use and drop cached
        views."""
        self._layout: PoolLayout | None = None
        self._buf: torch.Tensor | None = None
        self._handle = None  # torch._C._distributed_c10d._SymmetricMemory
        self._base_offset: int = 0  # bytes from the symmetric block base to the pool tensor
        self._peer_buffers: dict[int, torch.Tensor] = {}

    # --- state ------------------------------------------------------------------------------

    @property
    def reserved(self) -> bool:
        return self._handle is not None

    @property
    def layout(self) -> PoolLayout | None:
        """The current allocation's layout (``None`` before :meth:`reserve`)."""
        return self._layout

    @property
    def nbytes(self) -> int:
        """Bytes of the current allocation per rank (0 before :meth:`reserve`)."""
        return 0 if self._layout is None else self._layout.nbytes

    @property
    def slots(self) -> int:
        return 0 if self._layout is None else self._layout.slots

    @property
    def slot_nbytes(self) -> int:
        return 0 if self._layout is None else self._layout.slot_nbytes

    @property
    def region_nbytes(self) -> int:
        return 0 if self._layout is None else self._layout.region_nbytes

    @property
    def offset(self) -> int:
        """Byte offset of the pool tensor inside its symmetric block (``handle.offset``)."""
        return self._base_offset

    @property
    def signal_channels(self) -> int:
        """Channels the signal pad offers in total: ``signal_pad_size / (4 * world_size)`` (the
        kernels index the pad as ``world_size * channel + rank``; 0 before :meth:`reserve`)."""
        if self._handle is None:
            return 0
        return int(self._handle.signal_pad_size) // (_SIGNAL_BYTES * self.world_size)

    @property
    def channels(self) -> int:
        """Data channels available to callers: all but the last, which :meth:`barrier` owns."""
        return max(self.signal_channels - 1, 0)

    @property
    def barrier_channel(self) -> int:
        return self.signal_channels - 1

    def __repr__(self) -> str:
        state = (
            f"layout={self._layout} offset={self._base_offset} channels={self.channels} "
            f"generation={self.generation}"
            if self.reserved
            else "unreserved"
        )
        return (
            f"SymmMemPool(group={self.group_name!r}, rank={self.rank}/{self.world_size}, "
            f"device={self.device}, timeout_ms={self.timeout_ms}, {state})"
        )

    # --- allocation -------------------------------------------------------------------------

    def reserve(self, slot_nbytes: int, slots: int = 2, min_channels: int = 1) -> bool:
        """Collectively size the pool for ``slots`` slots of ``slot_nbytes`` bytes each.

        Every rank's request is all-gathered and the maximum taken, so the layout agrees on all
        ranks whatever each one asked for; the pool only grows (a request the current allocation
        already covers is a no-op and returns ``False``). A reallocation (returns ``True``) fences
        the group, frees the old allocation, allocates and rendezvous the new one, zeroes this
        rank's signal pad (fenced inside the addressing check before any peer can signal),
        checks that the pad offers ``min_channels`` data channels plus the barrier channel, runs
        the addressing check and logs the size. Protocols on the pool thus restart from their
        first use after every reallocation (:attr:`generation`).

        Args:
            slot_nbytes: Bytes one slot must hold (carved into ``world_size`` aligned regions).
            slots: Slots to carve (2 lets one boundary be pushed while the previous is read).
            min_channels: Data signal channels the caller needs.

        Returns:
            Whether a new allocation was made.

        Raises:
            RuntimeError: Under CUDA-graph capture, on an unsupported group or device, when the
                signal pad is too small or when the addressing check fails.
            ValueError: On a non-positive request or when the ranks disagree on the alignment.
        """
        if _is_capturing():
            raise RuntimeError(
                "SymmMemPool.reserve under CUDA-graph capture: reserve eagerly for the largest "
                "boundary (in the pipeline's eager warm-up) before capturing."
            )
        if slot_nbytes < 1 or slots < 1 or min_channels < 1:
            raise ValueError(
                f"SymmMemPool.reserve needs slot_nbytes >= 1, slots >= 1, min_channels >= 1 "
                f"(got {slot_nbytes}, {slots}, {min_channels})."
            )
        reasons = _local_preconditions(self.group, self.device)
        if reasons:
            raise RuntimeError(
                f"SymmMemPool.reserve on an unsupported group / device: {'; '.join(reasons)}."
            )
        want = PoolLayout.for_request(slot_nbytes, slots, self.world_size, self.align)
        requests: list[tuple[int, int, int, int] | None] = [None] * self.world_size
        dist.all_gather_object(
            requests, (want.region_nbytes, want.slots, min_channels, self.align), group=self.group
        )
        aligns = {req[3] for req in requests if req is not None}
        if aligns != {self.align}:
            raise ValueError(f"SymmMemPool.reserve: ranks disagree on align: {requests}.")
        agreed = PoolLayout(
            self.world_size,
            max(req[1] for req in requests if req is not None),
            max(req[0] for req in requests if req is not None),
            self.align,
        )
        min_channels = max(req[2] for req in requests if req is not None)
        if self._layout is not None:
            if self._layout.covers(agreed):
                self._require_channels(min_channels)
                return False
            agreed = self._layout.merged(agreed)
            self._free(fence=True)
        self._allocate(agreed)
        self._require_channels(min_channels)
        ok, reason = self._check_addressing()
        if not ok:
            raise RuntimeError(
                f"SymmMemPool: peer views do not address the peers' regions: {reason}"
            )
        logger.info(
            f"SymmMemPool[{self.group_name}]: reserved {agreed.slots} slot(s) x "
            f"{agreed.world_size} regions x {agreed.region_nbytes} B = "
            f"{agreed.nbytes / 2**30:.3f} GiB of symmetric memory per rank (backend "
            f"{_symm_mem().get_backend(self.device)}, block offset {self._base_offset}, "
            f"{self.channels} data signal channels, signal deadline {self.timeout_ms} ms; "
            f"{_TIMEOUT_TRIAGE_HINT})."
        )
        return True

    def _allocate(self, layout: PoolLayout) -> None:
        """Allocate and rendezvous ``layout`` (collective through ``rendezvous``); no fence."""
        symm_mem = _symm_mem()
        buf = symm_mem.empty(layout.nbytes, dtype=torch.uint8, device=self.device)
        handle = symm_mem.rendezvous(buf, self.group)
        if handle.world_size != self.world_size or handle.rank != self.rank:
            raise RuntimeError(
                f"SymmMemPool: rendezvous returned rank {handle.rank}/{handle.world_size}, "
                f"expected {self.rank}/{self.world_size} (wrong group?)."
            )
        base_offset = int(handle.offset)
        if base_offset < 0 or base_offset % layout.align:
            raise RuntimeError(
                f"SymmMemPool: the pool tensor sits at byte offset {base_offset} of its symmetric "
                f"block, which breaks the {layout.align}-byte alignment the layout assumes."
            )
        if base_offset + layout.nbytes > int(handle.buffer_size):
            raise RuntimeError(
                f"SymmMemPool: symmetric block holds {handle.buffer_size} bytes, the pool needs "
                f"{layout.nbytes} at offset {base_offset}."
            )
        self._buf, self._handle, self._base_offset, self._layout = buf, handle, base_offset, layout
        self._peer_buffers.clear()
        self.generation += 1
        self._zero_signal_pad()

    def _zero_signal_pad(self) -> None:
        """Clear every flag on this rank's signal pad (enqueued on the current stream).

        The pad belongs to the symmetric segment the pool tensor was sub-allocated from; a
        cached segment keeps the flags of its previous user, so the pad is cleared on every
        allocation. Fenced by :meth:`_check_addressing`'s first synchronize + barrier, which
        every allocation path runs before any peer can signal.
        """
        pad_nbytes = int(self._handle.signal_pad_size)
        self._handle.get_signal_pad(self.rank, (pad_nbytes,), torch.uint8).zero_()

    def _require_channels(self, min_channels: int) -> None:
        reason = self._channel_shortfall(min_channels)
        if reason:
            raise RuntimeError(f"SymmMemPool: {reason}")

    def _channel_shortfall(self, min_channels: int) -> str:
        if self.channels >= min_channels:
            return ""
        return (
            f"the signal pad offers {self.signal_channels} channels for {self.world_size} ranks "
            f"({self.channels} for data); need {min_channels} data channels plus the barrier. "
            "Call torch.distributed._symmetric_memory.set_signal_pad_size(bytes) before the "
            "first symmetric allocation of the process."
        )

    def _check_addressing(self) -> tuple[bool, str]:
        """Collective: prove that :meth:`peer_region` lands where the peer's :meth:`region` reads.

        Each rank pushes a rank- and generation-specific int64 marker into its own region of
        slot 0 on every rank (the production push path), the group fences (which also orders
        :meth:`_zero_signal_pad` before any peer's first signal), every rank reads all markers
        back from its local slot 0 and the verdict is agreed. Three host syncs at allocation
        time, never on the hot path; the markers are zeroed afterwards.
        """

        def marker_of(rank: int) -> int:
            return (rank + 1) * 1_000_003 + self.generation

        for peer in range(self.world_size):
            view = self.peer_region(peer, 0, 0, _MARKER_BYTES).view(torch.int64)
            view.fill_(marker_of(self.rank))
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        seen = {
            src: int(self.region(0, src, 0, _MARKER_BYTES).view(torch.int64).item())
            for src in range(self.world_size)
        }
        expected = {src: marker_of(src) for src in range(self.world_size)}
        detail = (
            ""
            if seen == expected
            else (
                f"read markers {seen}, expected {expected} (block offset {self._base_offset}, "
                f"block size {int(self._handle.buffer_size)}): get_buffer's frame of reference "
                "differs from the one this pool assumes"
            )
        )
        ok, reason = _agree(self.group, not detail, detail)
        dist.barrier(group=self.group)  # every peer has read our markers before anyone clears
        for src in range(self.world_size):
            self.region(0, src, 0, _MARKER_BYTES).zero_()
        torch.cuda.synchronize(self.device)
        return ok, reason

    def _require_reserved(self, op: str) -> None:
        if self._handle is None:
            raise RuntimeError(f"SymmMemPool.{op}: reserve() first.")

    def _check_peer(self, peer: int, who: str) -> None:
        if not 0 <= peer < self.world_size:
            raise ValueError(f"SymmMemPool.{who}: rank {peer} out of range for {self.world_size}.")

    def _check_channel(self, channel: int, who: str) -> None:
        if not 0 <= channel < self.channels:
            raise ValueError(
                f"SymmMemPool.{who}: channel {channel} out of range (pool offers {self.channels} "
                "data channels; the last pad channel is the barrier's)."
            )

    # --- views ------------------------------------------------------------------------------

    def region(self, slot: int, src_rank: int, offset: int, nbytes: int) -> torch.Tensor:
        """``nbytes`` bytes at ``offset`` of source ``src_rank``'s region in this rank's slot
        ``slot``: a 1-D ``uint8`` view of the LOCAL buffer, what a consumer reads."""
        self._require_reserved("region")
        start, stop = self._layout.region_span(slot, src_rank, offset, nbytes)
        return self._buf[start:stop]

    def peer_region(self, peer: int, slot: int, offset: int, nbytes: int) -> torch.Tensor:
        """``nbytes`` bytes at ``offset`` of THIS rank's region in ``peer``'s slot ``slot``: a 1-D
        ``uint8`` view of the peer's buffer mapped into this process, what a producer writes.

        ``get_buffer``'s ``storage_offset`` counts from the peer's symmetric block base, so the
        pool tensor's own offset inside the block is added (as ``symm_mem.get`` does). The
        whole-buffer peer view is cached per peer; slicing it is cheap but callers that need a
        view per boundary should cache the slices too.
        """
        self._require_reserved("peer_region")
        self._check_peer(peer, "peer_region")
        start, stop = self._layout.region_span(slot, self.rank, offset, nbytes)
        if peer == self.rank:
            return self._buf[start:stop]
        return self._peer_buffer(peer)[start:stop]

    def _peer_buffer(self, peer: int) -> torch.Tensor:
        view = self._peer_buffers.get(peer)
        if view is None:
            view = self._handle.get_buffer(
                peer, (self._layout.nbytes,), torch.uint8, self._base_offset
            )
            self._peer_buffers[peer] = view
        return view

    # --- signals (tiny kernels on the current CUDA stream) ------------------------------------

    def signal(self, peer: int, channel: int) -> None:
        """Set flag ``(channel, me -> peer)`` on ``peer``'s pad; spins (up to the deadline) while
        the previous flag on that channel is still unconsumed."""
        self._require_reserved("signal")
        self._check_peer(peer, "signal")
        self._check_channel(channel, "signal")
        self._handle.put_signal(peer, channel, self.timeout_ms)

    def wait(self, src: int, channel: int) -> None:
        """Spin (up to the deadline) until flag ``(channel, src -> me)`` is set, then clear it."""
        self._require_reserved("wait")
        self._check_peer(src, "wait")
        self._check_channel(channel, "wait")
        self._handle.wait_signal(src, channel, self.timeout_ms)

    def barrier(self) -> None:
        """Group-wide fence on the current stream, on the pad's reserved last channel."""
        self._require_reserved("barrier")
        self._handle.barrier(self.barrier_channel, self.timeout_ms)

    # --- teardown ---------------------------------------------------------------------------

    def release(self) -> None:
        """Free the allocation (collective fence first); call on every rank before
        ``destroy_process_group``. A no-op when nothing is reserved; safe to call twice."""
        if self.reserved:
            self._free(fence=True)

    def _free(self, *, fence: bool) -> None:
        # Peers may still read our buffer through their mapping: drain this device, then fence
        # the group before unmapping (both host-blocking; never on the hot path). `fence=False`
        # is for a rendezvous that failed on some rank: nothing was written, and a collective
        # here could not be matched by the rank that has no allocation.
        torch.cuda.synchronize(self.device)
        if fence and dist.is_initialized():
            dist.barrier(group=self.group)
        self._peer_buffers.clear()
        self._handle = None
        self._buf = None
        self._base_offset = 0
        self._layout = None
        gc.collect()


# =============================================================================
# Probe
# =============================================================================


def probe_symmetric_memory(
    group: dist.ProcessGroup, device: torch.device | str, *, min_channels: int = 1
) -> tuple[bool, str]:
    """Whether a :class:`SymmMemPool` can run on ``group`` at ``device``: collective, rank-agreed.

    Every rank of ``group`` must call it. Round one agrees on the local preconditions (the
    symmetric-memory module imports, the group has >= 2 ranks, ``device`` is the current CUDA
    device and NCCL serves it in the group); only when every rank passes does round two allocate
    and rendezvous one tiny region per rank and agree on the outcome; round three runs the
    addressing check (module docstring) and checks the signal pad offers ``min_channels`` data
    channels, then releases the probe allocation. Each verdict is the MIN over ranks, so a
    failure anywhere is a failure everywhere and the reason names the failing ranks. The probe
    never binds a backend, reads no ``TORCH_SYMM_MEM_*`` variable, tolerates a missing multicast
    binding and logs only at debug level: the caller decides how loudly to fall back.

    Args:
        group: The candidate process group.
        device: This rank's device.
        min_channels: Data signal channels the caller's protocol needs.

    Returns:
        ``(True, "")`` when every rank can use the pool, else ``(False, reason)``.
    """
    if group is None:
        raise ValueError("probe_symmetric_memory needs a torch.distributed process group.")
    device = torch.device(device)
    reasons = _local_preconditions(group, device)
    ok, reason = _agree(group, not reasons, "; ".join(reasons))
    if not ok:
        logger.debug(f"probe_symmetric_memory[{group.group_name}]: declined: {reason}")
        return False, reason
    pool = SymmMemPool(group, device)
    tiny = PoolLayout(pool.world_size, 1, pool.align, pool.align)
    detail = ""
    try:
        pool._allocate(tiny)
    except (RuntimeError, OSError, ValueError) as e:
        detail = f"tiny symmetric rendezvous failed: {type(e).__name__}: {e}"
    ok, reason = _agree(group, not detail, detail)
    if not ok:
        if pool.reserved:
            pool._free(fence=False)
        logger.debug(f"probe_symmetric_memory[{group.group_name}]: declined: {reason}")
        return False, reason
    ok, reason = pool._check_addressing()
    if ok:
        shortfall = pool._channel_shortfall(min_channels)
        ok, reason = _agree(group, not shortfall, shortfall)
    pool.release()
    if not ok:
        logger.debug(f"probe_symmetric_memory[{group.group_name}]: declined: {reason}")
    return ok, reason

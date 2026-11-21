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
"""A symmetric-memory allocation shared by the ranks of a process group, carved identically on
every rank into slots and per-rank regions.

Why
---
A copy-engine push transport writes each rank's rows straight into every peer's memory, so the
peers need a buffer that is (a) mapped into this process's address space, (b) laid out
identically on every rank and (c) allocated once, outside CUDA-graph capture.
``torch.distributed._symmetric_memory`` gives exactly that: ``empty`` + ``rendezvous`` return a
handle whose ``get_buffer(peer, ...)`` is a tensor view of the peer's copy of the allocation and
whose signal pad carries the ``put_signal`` / ``wait_signal`` flags that order a consumer's reads
after a producer's copies. Nothing in the tree uses the pool yet: the first follow-up is the
copy-engine all-gather of VisualGen's token-sharded tensor parallelism. Each
:class:`SymmMemPool` is one allocation; a group may carry several (one per protocol), nothing
ties a group to one.

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
A symmetric *block* is a signal pad followed by a data buffer. ``handle.get_buffer(peer, ...)``
counts its ``storage_offset`` from the start of the peer's data buffer, not from the pool
tensor, so every peer view adds ``handle.offset``, the tensor's byte offset inside the data
buffer (as PyTorch's own ``symm_mem.get`` does). On torch 2.14's ``CUDA`` backend
``symm_mem.empty`` makes one block per call and returns the tensor at the start of its data
buffer, so the offset is expected to be 0; the pool does not rely on that. This frame of
reference is an implementation detail of the torch build, so every allocation ends with a
collective marker check: each rank pushes a rank-specific marker through
:meth:`SymmMemPool.peer_region` and every rank reads all markers back through
:meth:`SymmMemPool.region`. A mismatch is reported at allocation time (the probe declines,
:meth:`SymmMemPool.reserve` raises) instead of pushing bytes next to a slot at run time.

Rules
-----
* :meth:`SymmMemPool.reserve` is collective (``all_gather_object`` of the request, every rank
  takes the maximum), grow-only and eager-only: it raises under CUDA-graph capture, so size the
  pool for the largest boundary in an eager warm-up before capturing.
* Allocation failures are agreed. ``symm_mem.empty`` is local to a rank and ``rendezvous``
  pairs the ranks through the group's store, so a rank that failed to allocate must not leave
  its peers to enter the rendezvous alone. Every local step of an allocation is caught and its
  verdict agreed before the next collective. On a failure anywhere, every rank drops what it
  holds, the pool is unreserved on every rank and :meth:`SymmMemPool.reserve` raises the same
  reason on every rank (the probe returns it). What the caller may do next depends on the
  reason. ``symm_mem.empty failed`` (typically out of device memory) was agreed before any
  rank entered the rendezvous: ``reserve`` may be retried, for a smaller size or after memory
  was freed. After any other reason, fall back to the path that does not need the pool. One
  case cannot be aligned from here: when torch's ``rendezvous`` fails on one rank only before
  its exchanges are complete (a driver error while exporting or mapping a block), the other
  ranks stay inside it until the group store's timeout (the ``timeout`` of
  ``init_process_group``; torch's default is 10 minutes for an ``nccl`` group and 30 minutes
  otherwise). The failing rank waits for them in the agreement under the group's collective
  timeout, by default the same value, so the agreement is not guaranteed to complete: a rank
  may end with a transport error instead of the agreed reason, and NCCL takes down a process
  whose collective timed out. Treat the processes as lost for symmetric memory: torch's
  pairing count (next rule) is left unequal.
* Groups with different members: torch pairs the store exchanges of a rendezvous by a count it
  keeps per process, across all groups. When a group allocates (a ``reserve`` that
  reallocates, or the probe), every member must have made the same number of rendezvous
  before, on whatever group and for whatever symmetric-memory user. Pools on sub-groups next
  to a pool on a larger group are therefore safe only while the ranks of the larger group
  allocate alike (each in one sub-group, the sub-groups growing together); otherwise the
  larger group's next allocation waits for the store's timeout and fails on every rank.
* Signals: :meth:`SymmMemPool.signal` sets flag ``(channel, me -> peer)`` on the peer's pad and
  spins while the previous flag is still unconsumed; :meth:`SymmMemPool.wait` spins until flag
  ``(channel, src -> me)`` is set, then clears it. One put per wait. Channels
  ``0 .. channels - 1`` all belong to the caller. Two protocols on one pool must use disjoint
  channels.
* Signal pad: every ``symm_mem.empty`` allocation is a symmetric block of its own with its own
  signal pad (torch 2.14's ``CUDA`` backend neither caches such blocks nor hands one out
  twice), so a pool's flags are private to it: other pools and the other symmetric-memory
  users of the process have their own pads. The pool clears this rank's pad on every
  allocation, before any peer can signal, so a protocol may start from "no flag set" after
  each :meth:`SymmMemPool.reserve` that reallocated. A pad has the size of torch's
  process-wide ``signal_pad_size`` at the time of its allocation, and the pool fixes
  :attr:`SymmMemPool.channels` from the value it reads then. ``set_signal_pad_size`` belongs
  before the first symmetric allocation of the process, with one value on every rank: torch's
  ``rendezvous`` fails on every rank when the sizes round to different multiples of 128 bytes,
  and the pool fails the allocation with the sizes by rank when they differ at all.
* Timeouts: a :meth:`SymmMemPool.wait` or :meth:`SymmMemPool.signal` that exceeds ``timeout_ms``
  is a device-side trap (PyTorch's signal kernels print a fatal message and ``__trap()``);
  treat the process as lost. The spin starts when the GPU reaches the kernel, so host-side skew
  between ranks (a recompile, a JIT, a collective) counts against it; the default of 10 minutes
  is a hang detector, not a latency bound.
* CUDA graphs: a graph that captured :meth:`SymmMemPool.region` / :meth:`SymmMemPool.peer_region`
  views or signals holds the allocation's addresses. Growing or releasing the pool while such a
  graph is alive invalidates the graph, so do both only once its runners are gone.
* Views: a view is valid for the allocation it was taken from. :meth:`SymmMemPool.reserve`
  returning ``True``, or a changed :attr:`SymmMemPool.generation`, tells that there is a new
  one. A ``reserve`` whose new allocation failed leaves the pool unreserved, whatever
  ``generation`` shows, also when it was growing an allocation; a ``reserve`` that raised for
  another reason keeps what it had (check :attr:`SymmMemPool.reserved`). A ``peer_region`` view
  does not own the peer's memory and dangles once the pool frees it; a ``region`` view keeps
  this rank's old block mapped until it is dropped. Drop cached views before a ``reserve`` that
  may grow and before ``release``.
* Environment: the pool sets nothing and never selects a symmetric-memory backend
  (``set_backend``); its channels need torch's default ``CUDA`` backend (``TORCH_SYMMMEM`` unset
  or ``CUDA``), whose signal pad holds one flag per channel and rank, and the probe and
  ``reserve`` decline another one. On a
  multicast-capable group torch's ``rendezvous`` also creates and binds an NVLS multicast
  object for each allocation, which the pool does not use;
  ``TORCH_SYMM_MEM_DISABLE_MULTICAST=1``, set on every rank, makes the group skip that.
* :meth:`SymmMemPool.release` before ``destroy_process_group``: peers may still map this rank's
  buffer, so it fences the group before unmapping.

Related code
------------
torch's per-group workspace (``get_symm_mem_workspace``, used by
``_torch/distributed/symm_mem_allgather.py``) is one tensor per group name that each rank regrows
on its own, with no fence and no size agreement across ranks. ``_torch/distributed/peer_buffers.py``
maps fixed-size CUDA IPC or MNNVL memory for the tensor-parallel group of a ``Mapping``; it has no
signal pad. ``cpp/tensorrt_llm/thop/asyncUlyssesOp.cpp`` keeps its own ring of three such blocks
per group, with a pad barrier, for the life of the process.
"""

from collections.abc import Callable
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
"""int: Default region alignment in bytes. It aligns offsets inside the allocation, which is what
a dtype view of a region needs, and 256 covers every dtype. The address of a region is torch's:
on torch 2.14's ``CUDA`` backend the data buffer starts at the block's mapping plus the signal pad
size rounded up to 128 bytes, so a region starts on a multiple of ``min(align, 128)`` bytes."""

DEFAULT_TIMEOUT_MS = 600_000
"""int: Default device-side signal deadline: 10 minutes, the default collective timeout of an NCCL
process group, rather than the latency of a healthy boundary (see the module docstring); ``0``
disables the deadline."""

_SIGNAL_BYTES = 4  # one uint32 flag per (channel, rank) on the signal pad
_MARKER_BYTES = 8  # one int64 marker per rank for the addressing check


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


def _agree(
    group: dist.ProcessGroup, ok: bool, reason: str, *, same: tuple[str, int] | None = None
) -> tuple[bool, str]:
    """The MIN of ``ok`` over the group's ranks (collective), with the failing ranks' reasons.

    Carried by ``all_gather_object`` rather than an ``all_reduce`` so that every rank learns
    which rank declined and why (the one line the caller logs must name the reason). ``same``
    is a ``(name, value)`` the ranks must also share: when every rank passed and the values
    differ, the verdict is a failure that lists them by rank.
    """
    # Every entry is filled by the gather.
    verdicts: list[tuple[bool, str, int | None]] = [None] * dist.get_world_size(group)
    dist.all_gather_object(verdicts, (bool(ok), reason, same and same[1]), group=group)
    failures = [f"rank {rank}: {why}" for rank, (good, why, _) in enumerate(verdicts) if not good]
    values = [value for _, _, value in verdicts]
    if not failures and same is not None and len(set(values)) > 1:
        failures = [f"the ranks differ on {same[0]}: {values} by rank"]
    return (True, "") if not failures else (False, "; ".join(failures))


def _is_registered(group: dist.ProcessGroup) -> bool:
    """Whether ``group`` still exists: ``destroy_process_group`` unregisters it, after which
    ``get_rank`` no longer resolves it and no collective can run on it."""
    try:
        dist.get_rank(group)
    except ValueError:
        return False
    return True


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
            f"{type(backend).__name__} serves {device} in the group; the pool supports only a "
            "ProcessGroupNCCL (a CUDA NCCL group)"
        ]
    return []


def _symm_backend_reasons(device: torch.device) -> list[str]:
    try:
        backend = _symm_mem().get_backend(device)
    except ImportError:
        return []  # reported by the caller
    if backend != "CUDA":
        return [
            f"torch's symmetric-memory backend for {device.type} devices is {backend!r}; the "
            "pool's signal channels need the default 'CUDA' backend (one flag per channel and "
            "rank on each block's own pad)"
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
        reasons.append(f"the group has {world_size} rank(s); the pool needs >= 2 ranks")
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
                "torch.cuda.set_device first: the capture guard and the group's object "
                "collectives act on the current device"
            )
        reasons.extend(_symm_backend_reasons(device))
    reasons.extend(_nccl_backend_reasons(group, device))
    return reasons


# =============================================================================
# Pool
# =============================================================================


class SymmMemPool:
    """A symmetric-memory allocation shared by the ranks of a process group, carved by a
    :class:`PoolLayout`.

    Lifecycle: construct on every rank (cheap, no allocation), :meth:`reserve` eagerly and
    collectively, take views and signal per boundary, :meth:`release` before the group is
    destroyed. Decide eligibility beforehand with :func:`probe_symmetric_memory`; on an
    ineligible group :meth:`reserve` raises.

    Args:
        group: The process group whose ranks share the pool (a CUDA group served by NCCL).
        device: This rank's CUDA device; must be the current device.
        timeout_ms: Device-side deadline of :meth:`wait` / :meth:`signal` in milliseconds, ``0``
            for none. See the module docstring on what a timeout means.
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
        """int: Incremented per allocation the ranks agreed on, so it is the same on every rank.
        Each allocation zeroes this rank's signal pad and has new addresses, so protocols on the
        pool restart from their first use and drop cached views. Not advanced by a
        :meth:`reserve` that fails before the ranks adopt the new allocation (``symm_mem.empty``,
        the rendezvous, the handle and pad checks); a failed addressing check has counted.
        Either way the pool is then unreserved (:attr:`reserved` is ``False``), also when it was
        growing an allocation."""
        self._layout: PoolLayout | None = None
        self._buf: torch.Tensor | None = None
        self._handle = None  # torch._C._distributed_c10d._SymmetricMemory
        self._base_offset: int = 0  # bytes from the block's data buffer start to the pool tensor
        self._pad_nbytes: int = 0  # the signal pad size the current allocation was made with
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
        """Byte offset of the pool tensor inside the data buffer of its symmetric block
        (``handle.offset``; 0 before :meth:`reserve`)."""
        return self._base_offset

    @property
    def channels(self) -> int:
        """Signal channels the pad offers, all of them the caller's: ``signal_pad_size / (4 *
        world_size)`` (the kernels index the pad as ``world_size * channel + rank``; 0 before
        :meth:`reserve`). Fixed per allocation: torch's ``signal_pad_size`` is one process-wide
        setting that a handle reports whatever size its own pad has, so the pool reads it once,
        when it allocates."""
        return self._pad_nbytes // (_SIGNAL_BYTES * self.world_size)

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
        ranks whatever sizes each one asked for; the pool only grows (a request the current
        allocation already covers is a no-op and returns ``False``). A reallocation (returns
        ``True``) frees the old allocation behind a group fence when the pool grows, allocates
        the new one and rendezvouses it, checks that its pad offers ``min_channels`` signal
        channels, clears this rank's pad, runs the addressing check and logs the size.
        Protocols on the pool thus restart from their first use after every reallocation
        (:attr:`generation`).

        A failure of the new allocation on any rank is agreed before any rank raises (module
        docstring, "Allocation failures"): every rank raises ``RuntimeError`` with the same
        reason and the pool is unreserved on every rank. That includes a pool that was growing:
        every rank drops the old allocation before any rank allocates the new one, so that the
        two never occupy device memory at once (provided the caller holds no ``region`` view of
        the old one).

        The checks that run before the requests are gathered are local to a rank: the capture
        guard, the argument check and the group / device preconditions. They must hold on every
        rank (reserve eagerly, with valid arguments, on a group that
        :func:`probe_symmetric_memory` accepted); a rank that raises there alone leaves its
        peers waiting in the gather. The pool must also be in the same state on every rank (the
        same history of ``reserve`` and ``release`` on the same pool object): the ranks compare
        their states with the requests and all raise when they differ.

        Args:
            slot_nbytes: Bytes one slot must hold (carved into ``world_size`` aligned regions).
            slots: Slots to carve (2 lets one boundary be pushed while the previous is read).
            min_channels: Signal channels the caller needs.

        Returns:
            Whether a new allocation was made.

        Raises:
            RuntimeError: Under CUDA-graph capture, on an unsupported group or device, when the
                ranks hold the pool in different states, and with the agreed reason when the
                allocation fails on any rank (``symm_mem.empty``, the rendezvous, a signal pad
                that is too small or whose size differs across ranks, the addressing check).
                Whatever a group collective raises
                after the rendezvous propagates as well, with the allocation dropped without a
                fence.
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
        # Every entry is filled by the gather: the request and the state this rank is in.
        requests: list[tuple[int, int, int, int, tuple]] = [None] * self.world_size
        state = (self._layout, self.generation)
        dist.all_gather_object(
            requests,
            (want.region_nbytes, want.slots, min_channels, self.align, state),
            group=self.group,
        )
        if {req[3] for req in requests} != {self.align}:
            raise ValueError(f"SymmMemPool.reserve: ranks disagree on align: {requests}.")
        # What follows (no-op, growth or first allocation) depends on the state, and so do the
        # collectives it issues: the ranks must be in the same one.
        if any(req[4] != state for req in requests):
            raise RuntimeError(
                "SymmMemPool.reserve: ranks disagree on the pool's state (layout, generation): "
                f"{[req[4] for req in requests]}; reserve and release the same pool on every rank."
            )
        agreed = PoolLayout(
            self.world_size,
            max(req[1] for req in requests),
            max(req[0] for req in requests),
            self.align,
        )
        min_channels = max(req[2] for req in requests)
        if self._layout is not None:
            if self._layout.covers(agreed):
                self._require_channels(min_channels)
                return False
            agreed = self._layout.merged(agreed)
            self._free(fence=True)
            # The device memory of a block is returned only once every rank has unmapped it,
            # which each rank does after the fence: wait for all before any rank allocates.
            dist.barrier(group=self.group)
        ok, reason = self._allocate(agreed, min_channels)
        if not ok:
            raise RuntimeError(f"SymmMemPool.reserve: {reason}")
        logger.info(
            f"SymmMemPool[{self.group_name}]: reserved {agreed.slots} slot(s) x "
            f"{agreed.world_size} regions x {agreed.region_nbytes} B = "
            f"{agreed.nbytes / 2**30:.3f} GiB of symmetric memory per rank (backend "
            f"{_symm_mem().get_backend(self.device)}, block offset {self._base_offset}, "
            f"{self.channels} signal channels, signal deadline {self.timeout_ms} ms)."
        )
        return True

    def _allocate(self, layout: PoolLayout, min_channels: int) -> tuple[bool, str]:
        """Collective: allocate, rendezvous and check ``layout``. The one allocation path of
        :meth:`reserve` and :func:`probe_symmetric_memory`.

        Returns ``(True, "")`` with the pool reserved on every rank, or ``(False, reason)`` with
        nothing reserved or mapped on any rank. Every rank issues the same collectives in the
        same order whatever fails locally, because each local step is caught and its verdict
        agreed before the next collective:

        1. ``symm_mem.empty`` (local), then an agreement. ``rendezvous`` pairs the ranks through
           the group's store and a rank that enters it alone waits for the store's timeout, so
           no rank enters it unless every rank has its block.
        2. ``rendezvous`` (collective inside torch), the checks of the handle it returns and
           the clearing of this rank's signal pad (local), then an agreement, which also
           requires one signal pad size on every rank. The pool's state is written only
           after it, so ``reserved`` and ``generation`` never differ across ranks, and a rank
           that passed while a peer failed just drops its block: no peer wrote to it.
        3. The addressing check (collective, phase-aligned), with a fenced free on an agreed
           failure: markers were pushed into the peers' buffers.

        What the pool cannot align is a failure inside ``rendezvous`` itself on one rank only
        (see "Allocation failures" in the module docstring).
        """
        buf = handle = None
        pad_nbytes = 0
        detail = ""
        try:
            buf = _symm_mem().empty(layout.nbytes, dtype=torch.uint8, device=self.device)
        except Exception as e:  # noqa: BLE001 - every local failure becomes an agreed verdict
            detail = f"symm_mem.empty failed: {type(e).__name__}: {e}"
        ok, reason = _agree(self.group, not detail, detail)
        if not ok:
            return False, reason
        try:
            handle = _symm_mem().rendezvous(buf, self.group)
            pad_nbytes = int(handle.signal_pad_size)  # the one read of this allocation
            detail = self._handle_defect(handle, layout, pad_nbytes, min_channels)
            if not detail:
                # Every flag of this rank's pad is clear, and the clearing drained, before any
                # rank can pass the agreement below, hence before any peer can signal.
                handle.get_signal_pad(self.rank, (pad_nbytes,), torch.uint8).zero_()
                torch.cuda.synchronize(self.device)
        except Exception as e:  # noqa: BLE001 - every local failure becomes an agreed verdict
            detail = f"rendezvous failed: {type(e).__name__}: {e}"
        # The pad size is a per-process torch setting: ranks with pads of different sizes would
        # count different channels, and a signal on one could have no matching wait on another.
        ok, reason = _agree(
            self.group, not detail, detail, same=("the signal pad size in bytes", pad_nbytes)
        )
        if not ok:
            return False, reason
        self._buf, self._handle, self._layout = buf, handle, layout
        self._base_offset, self._pad_nbytes = int(handle.offset), pad_nbytes
        self._peer_buffers.clear()
        self.generation += 1
        # The pool's attributes are now the only references to the block, so _free unmaps it.
        del buf, handle
        try:
            ok, reason = self._check_addressing()
        except BaseException:
            # A collective of the check itself failed, so no further collective can be matched:
            # unmap without the fence and let the error propagate.
            self._free(fence=False)
            raise
        if not ok:
            self._free(fence=True)
            return False, f"peer views do not address the peers' regions: {reason}"
        return True, ""

    def _handle_defect(
        self, handle: object, layout: PoolLayout, pad_nbytes: int, min_channels: int
    ) -> str:
        """Why the rendezvoused ``handle`` (a ``_SymmetricMemory``) cannot carry ``layout`` on
        this rank; empty when it can."""
        if handle.world_size != self.world_size or handle.rank != self.rank:
            return (
                f"rendezvous returned rank {handle.rank}/{handle.world_size}, expected "
                f"{self.rank}/{self.world_size} (wrong group?)"
            )
        base_offset = int(handle.offset)
        if base_offset < 0 or base_offset % layout.align:
            return (
                f"the pool tensor sits at byte offset {base_offset} of its symmetric block, "
                f"which breaks the {layout.align}-byte alignment the layout assumes"
            )
        if base_offset + layout.nbytes > int(handle.buffer_size):
            return (
                f"the symmetric block holds {handle.buffer_size} bytes, the pool needs "
                f"{layout.nbytes} at offset {base_offset}"
            )
        channels = pad_nbytes // (_SIGNAL_BYTES * self.world_size)
        return self._channel_shortfall(channels, min_channels)

    def _require_channels(self, min_channels: int) -> None:
        """Collective: raise on every rank when the pad is too small for ``min_channels``.

        Every rank computes the same verdict (one pad size per allocation, the agreed maximum
        of ``min_channels``); it is gathered like every other one so the reason has the same
        by-rank form.
        """
        shortfall = self._channel_shortfall(self.channels, min_channels)
        ok, reason = _agree(self.group, not shortfall, shortfall)
        if not ok:
            raise RuntimeError(f"SymmMemPool.reserve: {reason}")

    def _channel_shortfall(self, channels: int, min_channels: int) -> str:
        if channels >= min_channels:
            return ""
        return (
            f"the signal pad offers {channels} channels for {self.world_size} ranks; need "
            f"{min_channels}. Call torch.distributed._symmetric_memory.set_signal_pad_size(bytes) "
            "on every rank before the first symmetric allocation of the process."
        )

    def _check_addressing(self) -> tuple[bool, str]:
        """Collective: prove that :meth:`peer_region` lands where the peer's :meth:`region` reads.

        Each rank pushes a rank- and generation-specific int64 marker into its own region of
        slot 0 on every rank (the production push path), the group fences, every rank reads all
        markers back from its local slot 0 and the verdict is agreed. Phase-aligned: every local
        phase is caught and agreed on before the next collective, so a rank that fails locally
        never leaves its peers waiting at a barrier; a failed push ends the check with the
        agreed reason and no further collective. Host syncs at allocation time only, never on
        the hot path. The markers are zeroed before the last fence (best effort, the verdict is
        already agreed), so no rank returns, and pushes into a peer's slot 0, while that peer is
        still clearing it.
        """

        def marker_of(rank: int) -> int:
            return (rank + 1) * 1_000_003 + self.generation

        def local(phase: str, fn: Callable[[], None]) -> str:
            try:
                fn()
                return ""
            except Exception as e:  # noqa: BLE001 - every local failure becomes an agreed verdict
                return f"{phase} failed: {type(e).__name__}: {e}"

        def push_markers() -> None:
            for peer in range(self.world_size):
                view = self.peer_region(peer, 0, 0, _MARKER_BYTES).view(torch.int64)
                view.fill_(marker_of(self.rank))
            torch.cuda.synchronize(self.device)

        detail = local("marker push", push_markers)
        ok, reason = _agree(self.group, not detail, detail)
        if not ok:
            return False, reason
        dist.barrier(group=self.group)

        seen: dict[int, int] = {}

        def read_markers() -> None:
            for src in range(self.world_size):
                seen[src] = int(self.region(0, src, 0, _MARKER_BYTES).view(torch.int64).item())

        detail = local("marker read", read_markers)
        if not detail:
            expected = {src: marker_of(src) for src in range(self.world_size)}
            if seen != expected:
                detail = (
                    f"read markers {seen}, expected {expected} (block offset {self._base_offset}, "
                    f"pool size {self._layout.nbytes}): get_buffer's frame of "
                    "reference differs from the one this pool assumes"
                )
        ok, reason = _agree(self.group, not detail, detail)

        def clear_markers() -> None:
            for src in range(self.world_size):
                self.region(0, src, 0, _MARKER_BYTES).zero_()
            torch.cuda.synchronize(self.device)

        leftover = local("marker clear", clear_markers)
        if leftover:
            # The agreed verdict stands; a marker left behind stays until a push overwrites it.
            logger.warning(f"SymmMemPool[{self.group_name}]: {leftover}")
        # No rank returns, and so no rank pushes into a peer's slot 0, before every rank has
        # cleared its markers there.
        dist.barrier(group=self.group)
        return ok, reason

    def _require_reserved(self, op: str) -> None:
        if not self.reserved:
            raise RuntimeError(f"SymmMemPool.{op}: reserve() first.")

    def _check_peer(self, peer: int, who: str) -> None:
        if not 0 <= peer < self.world_size:
            raise ValueError(f"SymmMemPool.{who}: rank {peer} out of range for {self.world_size}.")

    def _check_channel(self, channel: int, who: str) -> None:
        if not 0 <= channel < self.channels:
            raise ValueError(
                f"SymmMemPool.{who}: channel {channel} out of range (the pool offers "
                f"{self.channels} channels)."
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

        ``get_buffer``'s ``storage_offset`` counts from the start of the peer's data buffer, so
        the pool tensor's own offset inside it is added (as ``symm_mem.get`` does). The
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

    # --- teardown ---------------------------------------------------------------------------

    def release(self) -> None:
        """Free the allocation (collective fence first); call on every rank before
        ``destroy_process_group``. A no-op when nothing is reserved; safe to call twice. Eager
        only, on every rank alike: it raises under CUDA-graph capture, where neither the device
        synchronize nor the fence can run. Once it got that far it always leaves the pool
        unreserved: a device error while draining is logged and the fence still runs, and a
        group that was already destroyed is released without the fence, with a warning."""
        if not self.reserved:
            return
        if _is_capturing():
            raise RuntimeError(
                "SymmMemPool.release under CUDA-graph capture: release eagerly, after the graphs "
                "that use the pool are gone."
            )
        self._free(fence=True)

    def _free(self, *, fence: bool) -> None:
        # Peers may still read our buffer through their mapping: drain this device, then fence
        # the group before unmapping (both host-blocking; never on the hot path). A device error
        # on this rank must not keep it out of the fence its peers are in, so it is logged, not
        # raised, and nothing may skip the clearing. `fence=False` is for a collective that
        # itself failed: no further collective can be matched. The block is unmapped when its
        # last reference goes, which is here unless a caller still holds a region() view.
        try:
            try:
                torch.cuda.synchronize(self.device)
            except Exception as e:  # noqa: BLE001 - no local failure may skip the fence
                logger.warning(
                    f"SymmMemPool[{self.group_name}]: device synchronize failed while freeing: {e}"
                )
            if fence:
                if _is_registered(self.group):
                    dist.barrier(group=self.group)
                else:
                    logger.warning(
                        f"SymmMemPool[{self.group_name}]: the group was destroyed before the "
                        "pool was released; freeing without the fence."
                    )
        finally:
            self._peer_buffers.clear()
            self._handle = None
            self._buf = None
            self._base_offset = self._pad_nbytes = 0
            self._layout = None


# =============================================================================
# Probe
# =============================================================================


def probe_symmetric_memory(
    group: dist.ProcessGroup, device: torch.device | str, *, min_channels: int = 1
) -> tuple[bool, str]:
    """Whether a :class:`SymmMemPool` can run on ``group`` at ``device``: collective, rank-agreed.

    Every rank of ``group`` must call it, eagerly: it raises under CUDA-graph capture. Round one
    agrees on the local preconditions (the symmetric-memory module imports, the group has >= 2
    ranks, ``device`` is the current CUDA device, torch's symmetric-memory backend for it is
    ``CUDA``, NCCL serves it in the group, and ``min_channels`` is positive). Only when
    every rank passes does it allocate one tiny region per rank (the allocation path of
    :meth:`SymmMemPool.reserve`, with the check that the signal pad offers ``min_channels``
    channels) and release it again. Each verdict is the MIN over ranks, so a failure anywhere is
    a failure everywhere and the reason names the failing ranks (or the values the ranks
    differ on); a declining probe leaves
    nothing reserved or mapped on any rank. The probe sets nothing in the environment and
    reports a decline at debug level only: the caller decides how loudly to fall back. Its
    allocation is a rendezvous like any other: see "Groups with different members" in the
    module docstring before probing groups that overlap.

    Args:
        group: The candidate process group.
        device: This rank's device.
        min_channels: Signal channels the caller's protocol needs.

    Returns:
        ``(True, "")`` when every rank can use the pool, else ``(False, reason)``.

    Raises:
        ValueError: Without a group.
        RuntimeError: Under CUDA-graph capture. Both are local to a rank and must hold on every
            rank, as for :meth:`SymmMemPool.reserve`. Whatever a group collective raises after
            the rendezvous propagates as well, with the allocation dropped without a fence.
    """
    if group is None:
        raise ValueError("probe_symmetric_memory needs a torch.distributed process group.")
    if _is_capturing():
        raise RuntimeError(
            "probe_symmetric_memory under CUDA-graph capture: probe eagerly (in the pipeline's "
            "eager warm-up) before capturing."
        )
    try:
        device = torch.device(device)
        reasons = _local_preconditions(group, device)
        if min_channels < 1:
            reasons.append(f"min_channels must be >= 1 (got {min_channels})")
    except Exception as e:  # noqa: BLE001 - every local failure becomes an agreed verdict
        reasons = [f"the local preconditions could not be evaluated: {type(e).__name__}: {e}"]
    ok, reason = _agree(group, not reasons, "; ".join(reasons))
    if not ok:
        logger.debug(f"probe_symmetric_memory[{group.group_name}]: declined: {reason}")
        return False, reason
    pool = SymmMemPool(group, device)
    tiny = PoolLayout(pool.world_size, 1, pool.align, pool.align)
    ok, reason = pool._allocate(tiny, min_channels)
    if ok:
        pool.release()
    else:
        logger.debug(f"probe_symmetric_memory[{group.group_name}]: declined: {reason}")
    return ok, reason

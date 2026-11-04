# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A step's small copies into graph-stable device buffers, merged as the model's copy plan says.

Between two decode CUDA graph replays, the next step's inputs and metadata reach their device buffers as a chain of tiny
copies (positions, KV lengths, page tables, the graph runner's static inputs). Each one is a separate operation on the
stream the next graph waits for, so the GPU idles for the whole chain.

Every such copy in the runtime names its call site with a ``CopySite`` and goes through ``copy`` (or ``fill_tail``).
Two questions decide what happens to it, and two different owners answer them:

* **When** a copy must land is the copy plan's business. It depends on the destination's readers, which the model
  chose, so the model declares a ``CopyPlan`` (``copy_plan`` on the model):

  - ``merge``: the sites whose copies are staged into the step's pack. A host-to-device copy snapshots its host bytes
    into pinned staging and a device-to-device copy records its (dst, src); a flush issues one upload of
    [descriptors | data] and one ``trtllm::device_copy_scatter`` launch, which writes every staged host copy and then
    every device copy. A merged copy lands before the next barrier, and at the latest at the graph replay, the step's
    final barrier. A pack is opened only on a step that replays an already captured graph, so it always reaches it.
  - ``flush_after`` (a subset of ``merge``): barrier sites. When a call through one of them stages its copy, the pack
    is flushed right after it, so that copy has landed before anything that runs next. Only the anchor's own write
    is guaranteed: a reader that needs other merged data anchors on those sites too.

  Listing a site in ``merge`` asserts that between the call and its barrier no device work reads or writes the
  destination, nor, for a device-to-device copy, writes the source (which is read at the flush). A site in neither
  set, and every site of an engine whose model declares no plan, is ``dst.copy_(src, non_blocking=True)``.
* **Whether** a copy is needed at all is the buffer owner's business. It depends on the values and on who else writes
  the buffer, which only the owner knows: the owner passes ``unchanged=True`` when the destination already holds what
  the call would write, and the call is not issued. Owners keep that state themselves, with main's
  ``DeviceCopyBuffer`` (``UploadRecords`` packages the usual pattern), record only copies that were issued, and pass
  ``on_drop`` so a merged copy that never lands (its step failed and the pack was discarded) invalidates the record.
  ``unchanged=True`` is refused during CUDA graph capture: a captured copy reruns on every replay.

Model-specific backends name their own sites with their own ``CopySiteEnum``. Adding a model: declare the plan from
the readers of each destination (merge what no device work touches before the replay, and anchor a barrier on each
copy that has an eager reader), and let each buffer owner decide on its own whether its uploads may be skipped.

``TLLM_DEVICE_COPY_CHECK=1`` (``CHECK_MODE``, read once at import) verifies the assertions at runtime, at the cost of a
device synchronization per check: a write of a merged destination before its flush, and an ``unchanged`` copy or fill
whose destination does not hold the values. A read of a merged destination before its barrier is caught only where
the reader asserts it with ``expect_landed`` under ``if device_copy.CHECK_MODE:``; elsewhere an early reader sees the
poison bytes instead and fails downstream.

Each engine with a plan owns one ``DeviceCopier`` and makes it the active one for its forward (``active``); the module
functions act on the calling context's copier, so engines on other threads or devices never share one. The attention
and speculative metadata that copy have no engine to ask, which is why the copier travels in the context.
"""

from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Optional

import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._utils import prefer_pinned

CHECK_ENV = "TLLM_DEVICE_COPY_CHECK"
#: Whether the plans' assertions are verified at runtime (``TLLM_DEVICE_COPY_CHECK=1``), read once at import.
CHECK_MODE: bool = os.environ.get(CHECK_ENV, "0") == "1"
# Byte written into a merged copy's destination in check mode, so an early reader sees garbage and an early writer
# is caught at the flush.
_POISON = 0xA5


class Direction(Enum):
    H2D = "h2d"
    D2D = "d2d"
    # A device-side fill of a buffer's unused tail.
    FILL = "fill"


class CopySiteEnum(Enum):
    """Base of the copy-site enums: each member is ``(key, direction)``, the key unique across all enums."""

    def __init__(self, key: str, direction: Direction) -> None:
        self.key = key
        self.direction = direction


class CopySite(CopySiteEnum):
    """Copy sites in shared runtime code."""

    # CUDAGraphRunner.replay: the step's input ids / position ids into the graph's static inputs.
    GRAPH_INPUT_IDS = ("graph.input_ids", Direction.D2D)
    GRAPH_POSITION_IDS = ("graph.position_ids", Direction.D2D)
    # Decoder inputs: host position ids. (The speculative gather and previous-batch tables, and the speculative
    # metadata's batch-index and slot uploads, skip unchanged values for every model through DeviceCopyBuffer or a
    # host-side comparison, and no plan merges them.)
    INPUT_POSITION_IDS = ("inputs.position_ids", Direction.H2D)
    # TrtllmAttentionMetadata.
    ATTN_KV_LENS = ("attn.kv_lens", Direction.H2D)
    ATTN_PROMPT_LENS = ("attn.prompt_lens", Direction.H2D)
    # SpecMetadata: the one-model speculative sampler's per-step Philox seeds and offsets (the forward reads them first).
    SPEC_RNG_REQUEST_SEEDS = ("spec.rng_request_seeds", Direction.H2D)
    SPEC_RNG_REQUEST_OFFSETS = ("spec.rng_request_offsets", Direction.H2D)
    SPEC_RNG_SEEDS = ("spec.rng_seeds", Direction.H2D)
    SPEC_RNG_OFFSETS = ("spec.rng_offsets", Direction.H2D)


@dataclass(frozen=True)
class CopyPlan:
    """When a model's per-step copies land. Empty by default: every copy is plain.

    ``merge``: sites whose copies are staged into the step's pack and land before the next barrier (see the module
    docstring). ``flush_after``: barrier sites, a subset of ``merge``. A call through one that stages its copy flushes
    the pack right after it; a call whose copy is unchanged, or runs eagerly, has nothing of its own to land and does
    not flush.
    """

    merge: frozenset[CopySiteEnum] = frozenset()
    flush_after: frozenset[CopySiteEnum] = frozenset()

    def __post_init__(self) -> None:
        for name in ("merge", "flush_after"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        for site in self.merge | self.flush_after:
            if not isinstance(site, CopySiteEnum):
                raise TypeError(f"CopyPlan sites must be CopySiteEnum members, got {site!r}")
        fills = [site for site in self.merge if site.direction is Direction.FILL]
        if fills:
            raise ValueError(f"fill sites run on the device and cannot be merged: {fills}")
        unmerged = self.flush_after - self.merge
        if unmerged:
            raise ValueError(
                "a flush_after barrier fires only when its own copy was merged, so it must be in merge: "
                f"{sorted(unmerged, key=str)}"
            )

    def __bool__(self) -> bool:
        return bool(self.merge)


def copy_plan_of(model: object) -> CopyPlan:
    """The plan ``model`` declares as ``copy_plan``, or the empty plan."""
    plan = getattr(model, "copy_plan", None)
    if plan is None:
        return CopyPlan()
    if not isinstance(plan, CopyPlan):
        raise TypeError(
            f"{type(model).__name__}.copy_plan must be a CopyPlan, got {type(plan).__name__}"
        )
    return plan


def _byte_range(t: torch.Tensor) -> tuple[int, int]:
    """[first, last + 1) device bytes ``t`` covers."""
    if t.numel() == 0:
        return (t.data_ptr(), t.data_ptr())
    if t.is_contiguous():
        return (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
    span = sum((size - 1) * stride for size, stride in zip(t.shape, t.stride())) + 1
    return (t.data_ptr(), t.data_ptr() + span * t.element_size())


def _overlaps(r: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    return any(r[0] < hi and lo < r[1] for lo, hi in ranges)


def _site_name(site: CopySiteEnum) -> str:
    return f"{type(site).__name__}.{site.name}"


class _Pack:
    """One engine's pack of merged copies (see the module docstring).

    A step's pack lives in one slot of a ring of pinned staging buffers and in one device staging buffer. Each barrier
    issues the copies staged since the previous one as a segment ([header | descriptors | data], aligned) appended
    after the previous segment in the same slot, with its own upload and scatter launch. The ring advances once per
    step, at ``begin``, which waits only for the upload that last used the slot RING steps ago.
    """

    CAPACITY = 1 << 16
    MAX_DESC = 256
    # Pinned staging buffers in rotation, one per step: a step stages without waiting on the previous steps' uploads.
    RING = 4
    HEADER_BYTES = 16
    DESC_BYTES = 24
    ALIGN = 16

    def __init__(self, device: torch.device, check: bool) -> None:
        self.host = [
            torch.empty(self.CAPACITY, dtype=torch.uint8, pin_memory=prefer_pinned())
            for _ in range(self.RING)
        ]
        self.events: list[Optional[torch.cuda.Event]] = [None] * self.RING
        self.dev = torch.empty(self.CAPACITY, dtype=torch.uint8, device=device)
        # A segment's data starts after its header and descriptor area.
        self.data_start = self.HEADER_BYTES + self.DESC_BYTES * self.MAX_DESC
        self.slot = self.RING - 1
        self.open = False
        self.check = check
        # First byte of the current segment in the slot.
        self.seg = 0
        self.h2d: list[tuple[int, int, int]] = []
        self.d2d: list[tuple[int, int, int]] = []
        # Bytes every staged device copy of the segment reads or writes: a later host copy into them would land out
        # of order (the scatter writes the host copies first).
        self.d2d_ranges: list[tuple[int, int]] = []
        # Check mode: bytes every staged copy of the segment writes.
        self.dst_ranges: list[tuple[int, int]] = []
        # Every tensor a staged copy reads or writes, kept alive until its segment is issued.
        self.staged: list[torch.Tensor] = []
        # The callers' on_drop callbacks of the segment's copies, called if the pack is discarded.
        self.drops: list[Callable[[], None]] = []
        # Check mode: (site, destination) of every staged copy, and (site, source, snapshot) of every device copy.
        self.poisoned: list[tuple[CopySiteEnum, torch.Tensor]] = []
        self.sources: list[tuple[CopySiteEnum, torch.Tensor, torch.Tensor]] = []
        self.used = self.seg + self.data_start

    def begin(self) -> None:
        if self.open:
            self.flush()
        self.slot = (self.slot + 1) % self.RING
        event = self.events[self.slot]
        if event is not None:
            # The uploads that last read this pinned buffer have run.
            event.synchronize()
        self.seg = 0
        self._clear()
        self.open = True

    def _clear(self) -> None:
        self.h2d.clear()
        self.d2d.clear()
        self.d2d_ranges.clear()
        self.dst_ranges.clear()
        self.staged.clear()
        self.drops.clear()
        self.poisoned.clear()
        self.sources.clear()
        self.used = self.seg + self.data_start

    def writes(self, t: torch.Tensor) -> bool:
        """Whether a staged copy writes any byte of ``t`` (check mode only)."""
        return _overlaps(_byte_range(t), self.dst_ranges)

    def pending_sites(self, t: torch.Tensor) -> list[CopySiteEnum]:
        """The sites of staged, not yet issued copies that write any byte of ``t`` (check mode only)."""
        r = _byte_range(t)
        return [site for site, dst in self.poisoned if _overlaps(r, [_byte_range(dst)])]

    def stage(
        self,
        site: CopySiteEnum,
        dst: torch.Tensor,
        src: torch.Tensor,
        on_drop: Optional[Callable[[], None]] = None,
    ) -> bool:
        if not dst.is_cuda or dst.numel() != src.numel():
            return False
        nbytes = dst.numel() * dst.element_size()
        # Byte ranges are needed only against staged device copies, and in check mode.
        dst_range = _byte_range(dst) if self.d2d_ranges or self.check or src.is_cuda else None
        if not src.is_cuda:
            if self.d2d_ranges and _overlaps(dst_range, self.d2d_ranges):
                return False
            # A contiguous destination is one descriptor; a 2-D slice with contiguous rows (a page table's leading
            # rows and columns) is one per row.
            if dst.is_contiguous():
                rows, row_bytes, row_stride = 1, nbytes, 0
            elif dst.dim() == 2 and dst.stride(1) == 1:
                rows = dst.shape[0]
                row_bytes = dst.shape[1] * dst.element_size()
                row_stride = dst.stride(0) * dst.element_size()
            else:
                return False
            offset = (self.used + self.ALIGN - 1) // self.ALIGN * self.ALIGN
            if (
                offset + nbytes > self.CAPACITY
                or len(self.h2d) + len(self.d2d) + rows > self.MAX_DESC
            ):
                return False
            # A dtype change converts on the host, as copy_ would on the device.
            src = src.to(dst.dtype).contiguous()
            self.host[self.slot][offset : offset + nbytes].copy_(src.view(-1).view(torch.uint8))
            base = dst.data_ptr()
            # The scatter reads a host copy's bytes at an offset from its segment's start.
            rel = offset - self.seg
            for row in range(rows):
                self.h2d.append((base + row * row_stride, rel + row * row_bytes, row_bytes))
            self.staged.append(dst)
            self.used = offset + nbytes
        else:
            if self.used > self.CAPACITY or len(self.h2d) + len(self.d2d) >= self.MAX_DESC:
                return False
            if not (
                src.is_cuda
                and src.dtype == dst.dtype
                and src.is_contiguous()
                and dst.is_contiguous()
            ):
                return False
            src_range = _byte_range(src)
            if self.check and not _overlaps(src_range, self.dst_ranges):
                # A source no staged copy fills must not change before the flush reads it.
                self.sources.append((site, src, src.clone()))
            self.d2d.append((dst.data_ptr(), src.data_ptr(), nbytes))
            self.d2d_ranges += (dst_range, src_range)
            self.staged += (dst, src)
        if on_drop is not None:
            self.drops.append(on_drop)
        if self.check:
            self.dst_ranges.append(dst_range)
            dst.view(torch.uint8).fill_(_POISON)
            self.poisoned.append((site, dst))
        return True

    def _verify(self) -> None:
        for site, dst in self.poisoned:
            if not bool(dst.view(torch.uint8).eq(_POISON).all()):
                raise RuntimeError(
                    f"{CHECK_ENV}: the destination of merged copy {_site_name(site)} was written between the "
                    "copy and its flush; the plan must not merge this site, or needs a barrier before that writer"
                )
        for site, src, snapshot in self.sources:
            if not torch.equal(src, snapshot):
                raise RuntimeError(
                    f"{CHECK_ENV}: the source of merged device copy {_site_name(site)} changed between the copy "
                    "and its flush; the plan must not merge this site, or needs a barrier before that writer"
                )

    def _issue(self) -> None:
        """Issue the segment staged since the last barrier and start the next one after it in the same slot."""
        if not self.h2d and not self.d2d:
            return
        if self.check:
            self._verify()
        host, seg = self.host[self.slot], self.seg
        desc = [len(self.h2d), len(self.d2d)] + [
            value for copy in self.h2d + self.d2d for value in copy
        ]
        host[seg : seg + 8 * len(desc)].view(torch.int64).copy_(
            torch.tensor(desc, dtype=torch.int64)
        )
        # Descriptors and data in one upload; the gap between them is unused padding.
        self.dev[seg : self.used].copy_(host[seg : self.used], non_blocking=True)
        torch.ops.trtllm.device_copy_scatter(self.dev[seg:])
        event = self.events[self.slot]
        if event is None:
            event = self.events[self.slot] = torch.cuda.Event()
        event.record()
        # The scatter is queued on the current stream: the caching allocator keeps the staged tensors' memory for it.
        self.seg = (self.used + self.ALIGN - 1) // self.ALIGN * self.ALIGN
        self._clear()

    def barrier(self) -> None:
        """Issue what is staged; the pack stays open for the rest of the step."""
        if self.open:
            self._issue()

    def flush(self) -> None:
        """Issue what is staged and close the pack (the step's final barrier)."""
        if not self.open:
            return
        self.open = False
        self._issue()

    def discard(self) -> None:
        """Close the pack without issuing it (its step failed); the unissued copies' on_drop callbacks run."""
        drops = list(self.drops)
        self.open = False
        self.seg = 0
        self._clear()
        for on_drop in drops:
            on_drop()


def _view_key(t: torch.Tensor) -> tuple:
    return (t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)


class DeviceCopier:
    """One engine's merge pack, applied per site as its ``CopyPlan`` says.

    In check mode (``CHECK_MODE`` when the copier is built), every merged copy's destination is poisoned at the call and
    verified untouched when its segment is issued, every merged device copy's source is verified unchanged then, and
    every ``unchanged`` copy or fill is verified against the device contents. A failed check raises and names the site.
    A device read of a merged destination before its barrier is detected only by a reader that calls
    ``expect_landed``; any other early reader reads the poison.
    """

    def __init__(self, plan: CopyPlan, device: torch.device) -> None:
        self.plan = plan
        self.check = CHECK_MODE
        self._pack = _Pack(device, self.check) if plan.merge else None
        # Site membership by identity: an Enum member hashes through a Python-level __hash__.
        self._merge_ids = frozenset(map(id, plan.merge))
        self._barrier_ids = frozenset(map(id, plan.flush_after))

    @property
    def merges(self) -> bool:
        return self._pack is not None

    def pending_sites(self, t: torch.Tensor) -> list[CopySiteEnum]:
        """Check mode: the merged sites whose staged, not yet issued copies write any byte of ``t``."""
        if self._pack is None or not self._pack.open:
            return []
        return self._pack.pending_sites(t)

    def writes_pending(self, t: torch.Tensor) -> bool:
        """Check mode: whether a staged, not yet issued copy writes any byte of ``t``."""
        return self._pack is not None and self._pack.open and self._pack.writes(t)

    @property
    def is_open(self) -> bool:
        return self._pack is not None and self._pack.open

    def begin(self) -> None:
        if self._pack is not None:
            self._pack.begin()

    def flush(self) -> None:
        if self._pack is not None:
            self._pack.flush()

    def discard(self) -> None:
        """Drop the open pack: the step failed, so its staged copies never land (their on_drop callbacks run)."""
        if self._pack is not None:
            self._pack.discard()

    def copy(
        self,
        site: CopySiteEnum,
        dst: torch.Tensor,
        src: torch.Tensor,
        unchanged: bool = False,
        on_drop: Optional[Callable[[], None]] = None,
    ) -> bool:
        if site.direction is Direction.FILL:
            raise ValueError(f"{_site_name(site)} is a fill site; use fill_tail")
        if src.is_cuda == (site.direction is Direction.H2D):
            raise ValueError(
                f"{_site_name(site)} is declared {site.direction.value}, but its source is on {src.device}"
            )
        if unchanged:
            _check_unchanged_copy(site, dst, src, self)
            return False
        pack = self._pack
        if (
            pack is not None
            and pack.open
            and id(site) in self._merge_ids
            and not torch.cuda.is_current_stream_capturing()
        ):
            if pack.stage(site, dst, src, on_drop):
                if id(site) in self._barrier_ids:
                    pack.barrier()
                return True
            # Cannot be staged: what is staged lands first, then this copy (which keeps the order).
            pack.barrier()
        dst.copy_(src, non_blocking=True)
        return True

    def fill_tail(
        self, site: CopySiteEnum, buf: torch.Tensor, start: int, value: int, unchanged: bool = False
    ) -> None:
        if site.direction is not Direction.FILL:
            raise ValueError(f"{_site_name(site)} is not a fill site")
        _fill_tail(site, buf, start, value, unchanged, self)


def _check_unchanged_copy(
    site: CopySiteEnum, dst: torch.Tensor, src: torch.Tensor, copier: Optional[DeviceCopier]
) -> None:
    """An ``unchanged`` copy is refused during capture and, in check mode, verified against the device."""
    if torch.cuda.is_current_stream_capturing():
        raise ValueError(
            f"{_site_name(site)}: unchanged=True during CUDA graph capture; a captured copy reruns on every "
            "replay, so it must be issued (and not recorded as skippable)"
        )
    if CHECK_MODE and not (copier is not None and copier.writes_pending(dst)):
        if not torch.equal(dst.cpu(), src.to(device="cpu", dtype=dst.dtype)):
            raise RuntimeError(
                f"{CHECK_ENV}: copy {_site_name(site)} was passed unchanged=True, but its destination holds other "
                "values; something else writes this destination, or the owner's record is stale"
            )


def _fill_tail(
    site: CopySiteEnum,
    buf: torch.Tensor,
    start: int,
    value: int,
    unchanged: bool,
    copier: Optional[DeviceCopier],
) -> None:
    if start >= buf.shape[0]:
        return
    if not unchanged:
        buf[start:].fill_(value)
        return
    if torch.cuda.is_current_stream_capturing():
        raise ValueError(
            f"{_site_name(site)}: unchanged=True during CUDA graph capture; a captured fill reruns on every replay"
        )
    tail = buf[start:]
    if CHECK_MODE and not (copier is not None and copier.writes_pending(tail)):
        if not bool(tail.eq(value).all()):
            raise RuntimeError(
                f"{CHECK_ENV}: tail fill {_site_name(site)} was passed unchanged=True, but the tail holds other "
                "values; something else writes this tail, or the owner's record is stale"
            )


class UploadRecords:
    """A buffer owner's record of what its uploads last wrote, to pass ``unchanged`` to ``copy`` and ``fill_tail``.

    The usual owner-side pattern around main's ``DeviceCopyBuffer``: compare, issue, record only an issued copy, drop
    the record before issuing (so a copy that fails leaves none) and on ``on_drop`` (so a merged copy that never lands
    leaves none), and never skip or record during capture. Keyed by device storage, and each entry keeps the storage
    alive so the address is not reused by a new buffer; the record key is the destination view, so a write through
    another view of the same storage replaces the record. The owner calls ``forget`` when it writes a recorded buffer
    by other means.
    """

    def __init__(self) -> None:
        # Storage address -> (storage, the host values last uploaded into it, keyed by the view).
        self._copies: dict[int, tuple[object, DeviceCopyBuffer]] = {}
        # Storage address -> (storage, (view, first row, value) of its last tail fill).
        self._tails: dict[int, tuple[object, tuple]] = {}

    def forget(self, t: torch.Tensor) -> None:
        """Drop the records of ``t``'s storage: something other than these uploads wrote it."""
        addr = t.untyped_storage().data_ptr()
        entry = self._copies.pop(addr, None)
        if entry is not None:
            entry[1].invalidate()
        self._tails.pop(addr, None)

    def clear(self) -> None:
        self._copies.clear()
        self._tails.clear()

    def copy(self, site: CopySiteEnum, dst: torch.Tensor, src: torch.Tensor) -> bool:
        """``device_copy.copy(site, dst, src)``, passing unchanged=True when the record says ``dst`` holds ``src``."""
        if torch.cuda.is_current_stream_capturing():
            self.forget(dst)
            return copy(site, dst, src)
        storage = dst.untyped_storage()
        addr = storage.data_ptr()
        key = _view_key(dst)
        entry = self._copies.get(addr)
        if entry is not None and entry[1].matches(src, key=key, exact=True):
            return copy(site, dst, src, unchanged=True)
        record = entry[1] if entry is not None else DeviceCopyBuffer(dst)
        record.invalidate()
        self._copies[addr] = (storage, record)
        issued = copy(site, dst, src, on_drop=record.invalidate)
        record.record(src, key=key)
        return issued

    def fill_tail(self, site: CopySiteEnum, buf: torch.Tensor, start: int, value: int) -> None:
        """``device_copy.fill_tail``, passing unchanged=True when the last fill of ``buf`` was the same.

        Call it after every write of the live rows ``buf[:start]``, including one that covers the whole of ``buf``
        (``start == len(buf)``, which fills nothing): a live-row write past the recorded start overwrites the tail.
        """
        storage = buf.untyped_storage()
        addr = storage.data_ptr()
        if start >= buf.shape[0] or torch.cuda.is_current_stream_capturing():
            self._tails.pop(addr, None)
            fill_tail(site, buf, start, value)
            return
        key = (_view_key(buf), start, value)
        entry = self._tails.get(addr)
        unchanged = entry is not None and entry[1] == key
        if not unchanged:
            self._tails.pop(addr, None)
        fill_tail(site, buf, start, value, unchanged=unchanged)
        self._tails[addr] = (storage, key)


_active: ContextVar[Optional[DeviceCopier]] = ContextVar("trtllm_device_copier", default=None)


@contextlib.contextmanager
def active(copier: Optional[DeviceCopier]) -> Iterator[None]:
    """Make ``copier`` the calling context's copier (None: plain copies) for the duration."""
    token = _active.set(copier)
    try:
        yield
    finally:
        _active.reset(token)


def current() -> Optional[DeviceCopier]:
    return _active.get()


def copy(
    site: CopySiteEnum,
    dst: torch.Tensor,
    src: torch.Tensor,
    *,
    unchanged: bool = False,
    on_drop: Optional[Callable[[], None]] = None,
) -> bool:
    """``dst.copy_(src, non_blocking=True)``, merged as the active copier's plan says; True if issued.

    ``unchanged``: the caller's assertion that ``dst`` already holds ``src``; the copy is not issued and False is
    returned. Refused during CUDA graph capture, and verified in check mode. ``on_drop``: called if the copy was merged
    and its pack is then discarded without issuing it (the caller's record of the copy is then stale).
    """
    copier = _active.get()
    if copier is not None:
        return copier.copy(site, dst, src, unchanged, on_drop)
    if unchanged:
        _check_unchanged_copy(site, dst, src, None)
        return False
    dst.copy_(src, non_blocking=True)
    return True


def fill_tail(
    site: CopySiteEnum, buf: torch.Tensor, start: int, value: int, *, unchanged: bool = False
) -> None:
    """``buf[start:].fill_(value)``; not issued when the caller asserts ``unchanged`` (the tail already holds it).

    ``unchanged`` is refused during CUDA graph capture and verified in check mode.
    """
    copier = _active.get()
    if copier is not None:
        copier.fill_tail(site, buf, start, value, unchanged)
        return
    _fill_tail(site, buf, start, value, unchanged, None)


def expect_landed(site: CopySiteEnum, t: torch.Tensor) -> None:
    """Check mode: assert that no merged copy into ``t`` is still staged, i.e. a barrier precedes this reader.

    ``site`` names the reader's anchor (the site whose ``flush_after`` barrier should have issued the copies) in the
    error. Call it only under ``if device_copy.CHECK_MODE:``; it raises when check mode is off, so an unguarded call
    cannot add work to the hot path unnoticed. Without an active copier nothing is staged and it passes.
    """
    if not CHECK_MODE:
        raise RuntimeError(
            f"expect_landed({_site_name(site)}) is a {CHECK_ENV} assertion; call it under `if device_copy.CHECK_MODE:`"
        )
    copier = _active.get()
    if copier is None:
        return
    pending = copier.pending_sites(t)
    if pending:
        raise RuntimeError(
            f"{CHECK_ENV}: a reader anchored at {_site_name(site)} reads bytes that merged copies "
            f"{sorted({_site_name(p) for p in pending})} have not written yet; the plan needs {_site_name(site)} "
            "in flush_after, and its call must precede this reader"
        )


def begin() -> None:
    """Open the active copier's pack. Call only when the step will reach ``flush`` (a captured graph's replay)."""
    copier = _active.get()
    if copier is not None:
        copier.begin()


def is_open() -> bool:
    copier = _active.get()
    return copier is not None and copier.is_open


def flush() -> None:
    """Issue the active open pack's last segment (one upload and one scatter launch on the current stream) and close it.

    The step's final barrier, at the graph replay.
    """
    copier = _active.get()
    if copier is not None:
        copier.flush()


def discard() -> None:
    """Drop the active copier's open pack (its step failed); the staged copies' on_drop callbacks run."""
    copier = _active.get()
    if copier is not None:
        copier.discard()

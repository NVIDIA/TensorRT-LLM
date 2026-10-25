# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A step's small copies into graph-stable device buffers, routed by the model's copy plan.

Between two decode CUDA graph replays, the next step's inputs and metadata reach their device buffers as a chain of tiny
copies (positions, KV lengths, page tables, the graph runner's static inputs). Each one is a separate operation on the
stream the next graph waits for, so the GPU idles for the whole chain.

Every such copy in the runtime names its call site with a ``CopySite`` and goes through ``copy``. A model declares a
``CopyPlan`` (``copy_plan`` on the model) listing the sites whose copies may be

* deferred (``defer``): a host-to-device copy snapshots its host bytes into pinned staging and a device-to-device copy
  records its (dst, src); right before the graph replay, ``flush`` issues one upload of [descriptors | data] and one
  ``trtllm::device_copy_scatter`` launch, which writes every staged host copy and then every device copy. A pack is
  opened only on a step that replays an already captured graph, so it always reaches the flush;
* skipped (``skip_unchanged``): a host-to-device copy, tail fill or block-offset gather whose destination already holds
  what it would write is not issued. Each engine keeps one host shadow per destination for this.

A site in neither set, and every site of an engine whose model declares no plan, is ``dst.copy_(src,
non_blocking=True)``. Model-specific backends name their own sites with their own ``CopySiteEnum``.

Listing a site in ``defer`` asserts that between the call and the replay no device work reads or writes the destination,
nor, for a device-to-device copy, writes the source (which is read at the flush). Listing it in ``skip_unchanged``
asserts that nothing but that site writes the destination. Whether a site may be deferred depends on the destination's
readers, not on the copy: the plan is where the model, which chose those readers, says so. ``TLLM_DEVICE_COPY_CHECK=1``
verifies the assertions at runtime, at the cost of a device synchronization per check (see ``DeviceCopier``). It
catches a write of a deferred destination, not a read: an eager reader sees the poison bytes instead and fails
downstream, so a site with an eager reader in some configuration says which one where it is declared.

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
from typing import Iterator, Optional

import torch

from tensorrt_llm._utils import prefer_pinned

CHECK_ENV = "TLLM_DEVICE_COPY_CHECK"
# Byte written into a deferred copy's destination in check mode, so an early reader sees garbage and an early writer
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
    # Decoder inputs: host position ids, and the speculative gather / previous-batch tables.
    INPUT_POSITION_IDS = ("inputs.position_ids", Direction.H2D)
    SPEC_GATHER_IDS = ("inputs.gather_ids", Direction.H2D)
    SPEC_PREVIOUS_BATCH_INDICES = ("inputs.previous_batch_indices", Direction.H2D)
    SPEC_NUM_ACCEPTED_DRAFT_TOKENS = ("inputs.num_accepted_draft_tokens", Direction.H2D)
    SPEC_PREVIOUS_POS_INDICES = ("inputs.previous_pos_indices", Direction.H2D)
    # TrtllmAttentionMetadata.
    ATTN_KV_LENS = ("attn.kv_lens", Direction.H2D)
    ATTN_PROMPT_LENS = ("attn.prompt_lens", Direction.H2D)
    # Eagle3SpecMetadata.
    EAGLE3_BATCH_INDICES = ("eagle3.batch_indices", Direction.H2D)
    EAGLE3_SLOT_IDS = ("eagle3.slot_ids", Direction.H2D)


@dataclass(frozen=True)
class CopyPlan:
    """The copy sites a model routes through deferral and skipping. Empty by default: every copy is plain."""

    defer: frozenset[CopySiteEnum] = frozenset()
    skip_unchanged: frozenset[CopySiteEnum] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "defer", frozenset(self.defer))
        object.__setattr__(self, "skip_unchanged", frozenset(self.skip_unchanged))
        for site in self.defer | self.skip_unchanged:
            if not isinstance(site, CopySiteEnum):
                raise TypeError(f"CopyPlan sites must be CopySiteEnum members, got {site!r}")
        fills = [site for site in self.defer if site.direction is Direction.FILL]
        if fills:
            raise ValueError(f"fill sites run on the device and cannot be deferred: {fills}")
        device_copies = [site for site in self.skip_unchanged if site.direction is Direction.D2D]
        if device_copies:
            raise ValueError(
                "device-to-device copies cannot be skipped (their source has no host shadow): "
                f"{device_copies}"
            )

    def __bool__(self) -> bool:
        return bool(self.defer or self.skip_unchanged)


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
    span = sum((size - 1) * stride for size, stride in zip(t.shape, t.stride())) + 1
    return (t.data_ptr(), t.data_ptr() + span * t.element_size())


def _overlaps(r: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    return any(r[0] < hi and lo < r[1] for lo, hi in ranges)


def _site_name(site: CopySiteEnum) -> str:
    return f"{type(site).__name__}.{site.name}"


class _Pack:
    """One engine's pack of deferred copies (see the module docstring)."""

    CAPACITY = 1 << 16
    MAX_DESC = 256
    # Pinned staging buffers in rotation: a forced flush reopens the pack without waiting on the upload just issued.
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
        self.data_start = self.HEADER_BYTES + self.DESC_BYTES * self.MAX_DESC
        self.parity = 0
        self.open = False
        self.check = check
        self.h2d: list[tuple[int, int, int]] = []
        self.d2d: list[tuple[int, int, int]] = []
        # Bytes every staged device copy reads or writes: a later host copy into them would land out of order.
        self.d2d_ranges: list[tuple[int, int]] = []
        # Bytes every staged copy writes.
        self.dst_ranges: list[tuple[int, int]] = []
        # Every tensor a staged copy reads or writes, kept alive until the pack is issued.
        self.staged: list[torch.Tensor] = []
        # Check mode: (site, destination) of every staged copy, and (site, source, snapshot) of every device copy.
        self.poisoned: list[tuple[CopySiteEnum, torch.Tensor]] = []
        self.sources: list[tuple[CopySiteEnum, torch.Tensor, torch.Tensor]] = []
        self.used = self.data_start

    def begin(self) -> None:
        if self.open:
            self.flush()
        event = self.events[self.parity]
        if event is not None:
            # The upload that last read this pinned buffer has run.
            event.synchronize()
        self._clear()
        self.open = True

    def _clear(self) -> None:
        self.h2d.clear()
        self.d2d.clear()
        self.d2d_ranges.clear()
        self.dst_ranges.clear()
        self.staged.clear()
        self.poisoned.clear()
        self.sources.clear()
        self.used = self.data_start

    def writes(self, t: torch.Tensor) -> bool:
        """Whether a staged copy writes any byte of ``t``."""
        return _overlaps(_byte_range(t), self.dst_ranges)

    def stage(self, site: CopySiteEnum, dst: torch.Tensor, src: torch.Tensor) -> bool:
        if not dst.is_cuda or dst.numel() != src.numel():
            return False
        nbytes = dst.numel() * dst.element_size()
        dst_range = _byte_range(dst)
        if src.device.type == "cpu":
            if _overlaps(dst_range, self.d2d_ranges):
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
            self.host[self.parity][offset : offset + nbytes].copy_(src.view(-1).view(torch.uint8))
            base = dst.data_ptr()
            for row in range(rows):
                self.h2d.append((base + row * row_stride, offset + row * row_bytes, row_bytes))
            self.staged.append(dst)
            self.used = offset + nbytes
        else:
            if len(self.h2d) + len(self.d2d) >= self.MAX_DESC:
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
        self.dst_ranges.append(dst_range)
        if self.check:
            dst.view(torch.uint8).fill_(_POISON)
            self.poisoned.append((site, dst))
        return True

    def _verify(self) -> None:
        for site, dst in self.poisoned:
            if not bool(dst.view(torch.uint8).eq(_POISON).all()):
                raise RuntimeError(
                    f"{CHECK_ENV}: the destination of deferred copy {_site_name(site)} was written between the "
                    "copy and the flush; the plan must not defer this site"
                )
        for site, src, snapshot in self.sources:
            if not torch.equal(src, snapshot):
                raise RuntimeError(
                    f"{CHECK_ENV}: the source of deferred device copy {_site_name(site)} changed between the copy "
                    "and the flush; the plan must not defer this site"
                )

    def flush(self) -> None:
        if not self.open:
            return
        self.open = False
        if not self.h2d and not self.d2d:
            return
        if self.check:
            self._verify()
        host = self.host[self.parity]
        desc = [len(self.h2d), len(self.d2d)] + [
            value for copy in self.h2d + self.d2d for value in copy
        ]
        host[: 8 * len(desc)].view(torch.int64).copy_(torch.tensor(desc, dtype=torch.int64))
        # Descriptors and data in one upload; the gap between them is unused padding.
        self.dev[: self.used].copy_(host[: self.used], non_blocking=True)
        torch.ops.trtllm.device_copy_scatter(self.dev)
        event = self.events[self.parity]
        if event is None:
            event = self.events[self.parity] = torch.cuda.Event()
        event.record()
        self.parity = (self.parity + 1) % self.RING
        # The scatter is queued on the current stream: the caching allocator keeps the staged tensors' memory for it.
        self._clear()

    def discard(self) -> None:
        """Close the pack without issuing it (its step failed)."""
        self.open = False
        self._clear()

    def copy(self, site: CopySiteEnum, dst: torch.Tensor, src: torch.Tensor) -> None:
        """Stage the copy, or flush and copy now when it cannot be staged (which keeps the order)."""
        if self.stage(site, dst, src):
            return
        self.flush()
        dst.copy_(src, non_blocking=True)
        self.begin()


def _view_key(t: torch.Tensor) -> tuple:
    return (t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)


class _SkipCache:
    """One engine's host shadows of its skip-routed destinations.

    Keyed by device storage, and each entry keeps the storage alive so the address is not reused by a new buffer.
    Keyed by storage rather than by view base: tensors made under inference mode do not record their base. A write
    through a different view of the same storage replaces the shadow, so overlapping views never skip on stale state.
    """

    def __init__(self) -> None:
        # Storage address -> (storage, view key, host values last copied into the view).
        self.shadow: dict[int, tuple[object, tuple, torch.Tensor]] = {}
        # Storage address -> (storage, view key, (first row, value) of the view's last tail fill).
        self.tails: dict[int, tuple[object, tuple, tuple[int, int]]] = {}
        # Destination address -> (its storage, inputs key, host table) of the last gather into it. Keyed by the view's
        # address: a gather writes one table of a storage that holds several.
        self.inputs: dict[int, tuple[object, tuple, torch.Tensor]] = {}

    def clear(self) -> None:
        self.shadow.clear()
        self.tails.clear()
        self.inputs.clear()

    def holds(self, dst: torch.Tensor, src: torch.Tensor) -> bool:
        entry = self.shadow.get(dst.untyped_storage().data_ptr())
        return (
            entry is not None
            and entry[1] == _view_key(dst)
            and entry[2].dtype == src.dtype
            and entry[2].shape == src.shape
            and torch.equal(entry[2], src)
        )

    def record(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        storage = dst.untyped_storage()
        self.shadow[storage.data_ptr()] = (storage, _view_key(dst), src.clone())

    def forget(self, dst: torch.Tensor) -> None:
        """Drop the value shadow of ``dst``'s storage (a copy the shadow does not describe wrote it)."""
        self.shadow.pop(dst.untyped_storage().data_ptr(), None)

    def tail_holds(self, buf: torch.Tensor, start: int, value: int) -> bool:
        entry = self.tails.get(buf.untyped_storage().data_ptr())
        return entry is not None and entry[1] == _view_key(buf) and entry[2] == (start, value)

    def record_tail(self, buf: torch.Tensor, start: int, value: int) -> None:
        storage = buf.untyped_storage()
        self.tails[storage.data_ptr()] = (storage, _view_key(buf), (start, value))

    def inputs_hold(self, dst: torch.Tensor, key: tuple, host: torch.Tensor) -> bool:
        entry = self.inputs.get(dst.data_ptr())
        return (
            entry is not None
            and entry[1] == key
            and entry[2].shape == host.shape
            and torch.equal(entry[2], host)
        )

    def record_inputs(self, dst: torch.Tensor, key: tuple, host: torch.Tensor) -> None:
        entry = self.inputs.get(dst.data_ptr())
        if entry is not None and entry[2].shape == host.shape and entry[2].dtype == host.dtype:
            # Host tables can be large: refill the previous snapshot instead of allocating one per step.
            snapshot = entry[2].copy_(host)
        else:
            snapshot = host.clone()
        self.inputs[dst.data_ptr()] = (dst.untyped_storage(), key, snapshot)


class DeviceCopier:
    """One engine's deferral pack and skip cache, applied per site as its ``CopyPlan`` says.

    In check mode (``TLLM_DEVICE_COPY_CHECK=1`` when the copier is built), every deferred copy's destination is
    poisoned at the call and verified untouched at the flush, every deferred device copy's source is verified unchanged
    at the flush, and every skipped copy or tail fill is verified against the device contents. A failed check raises
    and names the site. A device read of a deferred destination before the flush is not detected: it reads the poison.
    """

    def __init__(self, plan: CopyPlan, device: torch.device) -> None:
        self.plan = plan
        self.check = os.environ.get(CHECK_ENV, "0") == "1"
        self._pack = _Pack(device, self.check) if plan.defer else None
        self._skip = _SkipCache() if plan.skip_unchanged else None

    @property
    def defers(self) -> bool:
        return self._pack is not None

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
        """Drop the open pack and every shadow: the step failed, so none of its copies may have landed."""
        if self._pack is not None:
            self._pack.discard()
        if self._skip is not None:
            self._skip.clear()

    def _expect(self, site: CopySiteEnum, src: torch.Tensor) -> None:
        if site.direction is Direction.FILL:
            raise ValueError(f"{_site_name(site)} is a fill site; use fill_tail")
        on_host = src.device.type == "cpu"
        if on_host != (site.direction is Direction.H2D):
            raise ValueError(
                f"{_site_name(site)} is declared {site.direction.value}, but its source is on {src.device}"
            )

    def copy(self, site: CopySiteEnum, dst: torch.Tensor, src: torch.Tensor) -> bool:
        self._expect(site, src)
        if torch.cuda.is_current_stream_capturing():
            # A captured copy reruns on every replay, so no shadow can vouch for the buffer.
            if self._skip is not None:
                self._skip.forget(dst)
            dst.copy_(src, non_blocking=True)
            return True
        skip = self._skip is not None and site in self.plan.skip_unchanged and dst.is_cuda
        if skip and self._skip.holds(dst, src):
            if self.check and not (self._pack is not None and self._pack.writes(dst)):
                if not torch.equal(dst.cpu(), src.to(dst.dtype)):
                    raise RuntimeError(
                        f"{CHECK_ENV}: skipped copy {_site_name(site)} found other values on the device; something "
                        "else writes this destination, so the plan must not skip this site"
                    )
            return False
        if site in self.plan.defer and self.is_open:
            self._pack.copy(site, dst, src)
        else:
            dst.copy_(src, non_blocking=True)
        if skip:
            self._skip.record(dst, src)
        elif self._skip is not None:
            self._skip.forget(dst)
        return True

    def fill_tail(self, site: CopySiteEnum, buf: torch.Tensor, start: int, value: int) -> None:
        if site.direction is not Direction.FILL:
            raise ValueError(f"{_site_name(site)} is not a fill site")
        skip = self._skip is not None and site in self.plan.skip_unchanged
        if start >= buf.shape[0]:
            # The live rows cover the whole view: a previous tail fill was overwritten.
            if skip:
                self._skip.tails.pop(buf.untyped_storage().data_ptr(), None)
            return
        if skip and torch.cuda.is_current_stream_capturing():
            self._skip.tails.pop(buf.untyped_storage().data_ptr(), None)
            skip = False
        if skip and self._skip.tail_holds(buf, start, value):
            if self.check and not bool(buf[start:].eq(value).all()):
                raise RuntimeError(
                    f"{CHECK_ENV}: skipped tail fill {_site_name(site)} found other values on the device; "
                    "something else writes this tail, so the plan must not skip this site"
                )
            return
        buf[start:].fill_(value)
        if skip:
            self._skip.record_tail(buf, start, value)

    def inputs_unchanged(
        self, site: CopySiteEnum, dst: torch.Tensor, key: Optional[tuple], host: torch.Tensor
    ) -> bool:
        """Whether a gather into ``dst`` from ``key`` and ``host`` would rewrite what the last one wrote.

        For copies the runtime does not issue itself (a block-offset gather): the caller skips its copy on True and
        issues it on False, which records the inputs. ``key=None`` marks a gather whose inputs the caller does not
        compare: it is issued and the record is dropped.
        """
        if self._skip is None or site not in self.plan.skip_unchanged:
            return False
        if key is None or torch.cuda.is_current_stream_capturing():
            self._skip.inputs.pop(dst.data_ptr(), None)
            return False
        if self._skip.inputs_hold(dst, key, host):
            return True
        self._skip.record_inputs(dst, key, host)
        return False


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


def copy(site: CopySiteEnum, dst: torch.Tensor, src: torch.Tensor) -> bool:
    """``dst.copy_(src, non_blocking=True)``, deferred or skipped as the active copier's plan says; True if issued."""
    copier = _active.get()
    if copier is None:
        dst.copy_(src, non_blocking=True)
        return True
    return copier.copy(site, dst, src)


def fill_tail(site: CopySiteEnum, buf: torch.Tensor, start: int, value: int) -> None:
    """``buf[start:].fill_(value)``, skipped when the plan skips ``site`` and the last fill of ``buf`` was the same.

    Call it after every write of the live rows ``buf[:start]``, including one that covers the whole of ``buf``
    (``start == len(buf)``, which fills nothing): a live-row write past the recorded start overwrites part of the tail.
    """
    copier = _active.get()
    if copier is None:
        if start < buf.shape[0]:
            buf[start:].fill_(value)
        return
    copier.fill_tail(site, buf, start, value)


def skips(site: CopySiteEnum) -> bool:
    """Whether the active copier's plan skips ``site`` when unchanged."""
    copier = _active.get()
    return copier is not None and site in copier.plan.skip_unchanged


def inputs_unchanged(
    site: CopySiteEnum, dst: torch.Tensor, key: Optional[tuple], host: torch.Tensor
) -> bool:
    """See ``DeviceCopier.inputs_unchanged``; always False without an active copier."""
    copier = _active.get()
    return copier is not None and copier.inputs_unchanged(site, dst, key, host)


def begin() -> None:
    """Open the active copier's pack. Call only when the step will reach ``flush`` (a captured graph's replay)."""
    copier = _active.get()
    if copier is not None:
        copier.begin()


def is_open() -> bool:
    copier = _active.get()
    return copier is not None and copier.is_open


def flush() -> None:
    """Issue the active open pack (one upload and one scatter launch on the current stream) and close it."""
    copier = _active.get()
    if copier is not None:
        copier.flush()


def discard() -> None:
    """Drop the active copier's open pack and shadows (its step failed)."""
    copier = _active.get()
    if copier is not None:
        copier.discard()

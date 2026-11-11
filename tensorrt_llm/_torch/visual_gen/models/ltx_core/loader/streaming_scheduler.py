"""Hook-driven execution-unit delivery with persistent read-ahead.
The scheduler owns all policy: which unit is read when, which buffer it uses, which thread transfers it,
cancellation, and when a buffer may be reused. Per-device mechanism lives in a ``Transport``, and how a unit
becomes active tensors lives in ``streaming_delivery``. Control flows one way, from the scheduler to both.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.buffers import aligned_region_bytes
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.checkpoint import UnitSource
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.lora import FLOAT_FUSE_RULE, LoRAFuser
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.mapping import ExecutionCache, file_read_parts
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.state_entry import StateEntry
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.streaming_delivery import (
    Readiness,
    Staged,
    UnitViews,
    deliver,
    device_nbytes,
    host_buffer_nbytes,
    host_nbytes,
    owner_matches,
    read_unit,
    readiness_of,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.streaming_transport import Completion, Transfer, Transport, make_transport
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import PatchedUnitSpec, TensorsUnit, UnitSpec


class BufferPool:
    """Byte buffers of one kind, host or device, each with the completion of its last use."""

    def __init__(self, buffers: Sequence[torch.Tensor]) -> None:
        if any(buffer.dtype != torch.uint8 or buffer.ndim != 1 or not buffer.is_contiguous() for buffer in buffers):
            raise ValueError("pooled buffers must be contiguous one-dimensional uint8 tensors")
        self._buffers = tuple(buffers)
        self._free = list(range(len(self._buffers)))
        self._completions: dict[int, Completion] = {}
        self._cv = threading.Condition()
        self._closed = False

    def __len__(self) -> int:
        return len(self._buffers)

    @property
    def buffers(self) -> tuple[torch.Tensor, ...]:
        return self._buffers

    @property
    def available(self) -> int:
        """Number of buffers not leased."""
        with self._cv:
            return len(self._free)

    def buffer(self, index: int) -> torch.Tensor:
        return self._buffers[index]

    def lease(self, cancel: Event) -> tuple[int, Completion | None]:
        """Lease a free buffer and return the completion of its previous use, if any."""
        with self._cv:
            while not self._free and not (cancel.is_set() or self._closed):
                self._cv.wait()
            if cancel.is_set() or self._closed:
                raise RuntimeError("buffer lease was retired")
            index = self._free.pop()
            return index, self._completions.pop(index, None)

    def release(self, index: int, completion: Completion | None) -> None:
        """Return a leased buffer; ``completion`` covers the last work that used it."""
        with self._cv:
            if completion is not None:
                self._completions[index] = completion
            self._free.append(index)
            self._cv.notify_all()

    def wake(self) -> None:
        """Let waiting lessees recheck their cancel events."""
        with self._cv:
            self._cv.notify_all()

    def close(self) -> None:
        """Make waiting and later leases raise, and drop the buffer storage."""
        with self._cv:
            self._closed = True
            self._buffers = ()
            self._completions.clear()
            self._cv.notify_all()


@dataclass(frozen=True, slots=True)
class Delivered:
    """A unit's active views in a device buffer, plus the completion of the work that filled them."""

    uid: str
    views: Mapping[str, torch.Tensor]
    device_index: int
    completion: Completion


@dataclass(frozen=True, slots=True)
class _Pending:
    cancel: Event
    staged: Future[Staged]
    delivered: Future[Delivered] | None  # None when the transport delivers on the consumer thread


class StreamingScheduler:
    """Read streamed units ahead through host and device buffers and deliver them to forward hooks.
    ``retained`` maps unit IDs to the states the builder selected from its registry: an active entry is only
    copied, a clean entry is cast and patched with ``factors``, and every other unit is read from ``source``,
    which the caller keeps open until the scheduler is closed. ``transfer_at`` maps unit IDs to the readiness
    at which they cross to the device, ``CLEAN`` by default.
    """

    def __init__(
        self,
        units: Sequence[PatchedUnitSpec],
        source: UnitSource,
        transport: Transport,
        host: BufferPool,
        device: BufferPool,
        *,
        retained: Mapping[str, StateEntry] | None = None,
        transfer_at: Mapping[str, Readiness] | None = None,
        fuser: LoRAFuser | None = None,
        factors: Mapping[str, Sequence[TensorsUnit]] | None = None,
        prefetch_depth: int = 2,
    ) -> None:
        self.units = tuple(units)
        self.indices = {patched.clean.unit.unit_id: i for i, patched in enumerate(self.units)}
        if not self.units or len(self.indices) != len(self.units):
            raise ValueError("streaming needs uniquely identified units")
        if prefetch_depth < 0 or not len(device):
            raise ValueError("streaming needs device buffers and a nonnegative prefetch depth")
        if any(buffer.device.type != transport.device.type for buffer in device.buffers):
            raise ValueError(f"streaming transport needs {transport.device.type} device buffers")
        if any(buffer.device.type != "cpu" for buffer in host.buffers):
            raise ValueError("host buffers must be on CPU")
        retained = dict(retained or {})
        self._retained = retained
        self._readiness = {uid: readiness_of(retained.get(uid)) for uid in self.indices}
        self._transfer_at = {uid: (transfer_at or {}).get(uid, Readiness.CLEAN) for uid in self.indices}
        if Readiness.CHECKPOINT in self._transfer_at.values() and transport.device.type != "cuda":
            raise ValueError("device projection needs a CUDA streaming target")
        self._owners = {uid for uid, entry in retained.items() if owner_matches(entry, self.units[self.indices[uid]])}
        self._factors = {uid: tuple(values) for uid, values in (factors or {}).items()}
        for patched in self.units:
            uid = patched.clean.unit.unit_id
            if (
                patched.loras
                and self._readiness[uid] is not Readiness.ACTIVE
                and (fuser is None or len(self._factors.get(uid, ())) != len(patched.loras))
            ):
                raise ValueError("streaming a clean unit with LoRAs needs a fuser and one factor unit per LoRA")
        staged_bytes = [
            host_buffer_nbytes(patched, self._readiness[uid], self._transfer_at[uid]) for uid, patched in self._items()
        ]
        if any(staged_bytes) and (not len(host) or max(staged_bytes) > min(b.numel() for b in host.buffers)):
            raise ValueError("units staged on the host need host buffers that fit them")
        needed = max(
            device_nbytes(patched, self._readiness[uid], self._transfer_at[uid]) for uid, patched in self._items()
        )
        if needed > min(buffer.numel() for buffer in device.buffers):
            raise ValueError("device buffers are too small for the largest unit")
        self._transport = transport
        self._host = host
        self._device = device
        self.fuser = fuser
        self.prefetch_depth = prefetch_depth
        self.source: UnitSource | None = source
        self._views = UnitViews()
        cleaned = [
            (uid, patched)
            for uid, patched in self._items()
            if self._readiness[uid] is Readiness.UNREAD and self._transfer_at[uid] is not Readiness.CHECKPOINT
        ]
        self._shared = tuple(patched.clean for _, patched in cleaned)
        self._shared_set = frozenset(self._shared)
        # Units cleaned from their checkpoint each delivery are split once for file reads, not once per forward.
        self._file_reads = {uid: file_read_parts(patched) for uid, patched in cleaned}
        self._cache = ExecutionCache.if_shared(self._shared) if self._shared else None
        self._cache_uses: set[UnitSpec] = set()
        self._reads = ThreadPoolExecutor(1, "stream-read")
        self._copies = ThreadPoolExecutor(1, "stream-copy") if transport.transfer_thread == "any" else None
        self._pending: dict[str, _Pending] = {}
        self._entered: dict[str, Delivered] = {}
        self._last_index: int | None = None
        self._closed = False

    @classmethod
    def from_specs(
        cls,
        specs: Sequence[UnitSpec],
        source: UnitSource,
        host: Sequence[torch.Tensor],
        device: Sequence[torch.Tensor],
        *,
        prefetch_depth: int = 2,
    ) -> StreamingScheduler:
        """Stream clean units without LoRAs, projected on the host, through caller-owned buffers."""
        if not device:
            raise ValueError("streaming needs device buffers")
        units = tuple(PatchedUnitSpec(spec, FLOAT_FUSE_RULE) for spec in specs)
        transport = make_transport(device[0].device)
        return cls(units, source, transport, BufferPool(host), BufferPool(device), prefetch_depth=prefetch_depth)

    @property
    def host(self) -> BufferPool:
        return self._host

    @property
    def device(self) -> BufferPool:
        return self._device

    def readiness(self, uid: str) -> Readiness:
        """Where the unit starts before each delivery."""
        return self._readiness[uid]

    def transfer_at(self, uid: str) -> Readiness:
        """The readiness at which the unit crosses to the device."""
        return self._transfer_at[uid]

    def need(self, uid: str) -> Mapping[str, torch.Tensor]:
        """Deliver one unit and read ahead up to ``prefetch_depth`` following units.
        Units passed over since the previous request are retired even when the read-ahead wraps around to
        them: a skipped unit would otherwise hold a device buffer this unit may need.
        """
        if self._closed:
            raise RuntimeError("scheduler is closed")
        index = self.indices[uid]
        count = len(self.units)
        skipped = (
            set()
            if self._last_index is None
            else {
                self.units[(self._last_index + step) % count].clean.unit.unit_id
                for step in range(1, (index - self._last_index) % count)
            }
        )
        self._last_index = index
        wanted = [
            item
            for step in range(min(self.prefetch_depth + 1, count))
            if (item := self.units[(index + step) % count].clean.unit.unit_id) not in skipped
        ]
        self._retire(set(wanted))
        for item in wanted:
            self._queue(item)
        pending = self._pending[uid]
        try:
            delivered = (
                pending.delivered.result()
                if pending.delivered is not None
                else self._deliver(pending.staged.result(), pending.cancel)
            )
        except BaseException:
            self._retire(set())
            raise
        delivered.completion.wait_device()
        self._entered[uid] = delivered
        return delivered.views

    def done(self, uid: str) -> None:
        """Release a delivered unit's device buffer after its triggering module finishes."""
        delivered = self._entered.pop(uid, None)
        if delivered is None:
            raise ValueError(f"unit {uid!r} has not entered")
        self._pending.pop(uid, None)
        self._device.release(delivered.device_index, self._transport.compute_completion())

    def is_entered(self, uid: str) -> bool:
        """Whether a model hook has acquired this unit but not released it."""
        return uid in self._entered

    def close(self) -> None:
        """Stop workers, wait for device work, and drop buffer, source and retained references."""
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for uid in tuple(self._entered):
            try:
                self.done(uid)
            except BaseException as error:
                errors.append(error)
        self._entered.clear()
        try:
            self._retire(set())
        except BaseException as error:
            errors.append(error)
        for pending in self._pending.values():
            pending.cancel.set()
        self._host.wake()
        self._device.wake()
        executors = (self._reads,) if self._copies is None else (self._reads, self._copies)
        for executor in executors:
            try:
                executor.shutdown(wait=True)
            except BaseException as error:
                errors.append(error)
        try:
            self._transport.close()
        except BaseException as error:
            errors.append(error)
        self.source = None
        self._host.close()
        self._device.close()
        self._pending.clear()
        self._retained = {}
        self._factors = {}
        self._views.clear()
        self._cache = None
        self._cache_uses.clear()
        self.fuser = None
        if errors:
            raise errors[0]

    def _items(self) -> tuple[tuple[str, PatchedUnitSpec], ...]:
        return tuple((patched.clean.unit.unit_id, patched) for patched in self.units)

    def _queue(self, uid: str) -> None:
        if uid in self._pending:
            return
        cancel = Event()
        staged = self._reads.submit(self._stage, uid, cancel)
        delivered = self._copies.submit(self._deliver_staged, staged, cancel) if self._copies is not None else None
        self._pending[uid] = _Pending(cancel, staged, delivered)

    def _retire(self, keep: set[str]) -> None:
        for uid in [uid for uid in self._pending if uid not in keep and uid not in self._entered]:
            pending = self._pending.pop(uid)
            pending.cancel.set()
            (pending.delivered or pending.staged).add_done_callback(self._release_unconsumed)
        self._host.wake()
        self._device.wake()

    def _release_unconsumed(self, future: Future[Staged] | Future[Delivered]) -> None:
        """Free what a retired unit still holds once its last stage finishes."""
        try:
            result = future.result()
        except BaseException:
            return  # a failing stage releases what it held before raising
        if isinstance(result, Delivered):
            self._device.release(result.device_index, result.completion)
        elif result.host_index is not None:
            self._host.release(result.host_index, None)

    def _stage(self, uid: str, cancel: Event) -> Staged:
        """Runs on the read worker."""
        patched = self.units[self.indices[uid]]
        readiness, transfer_at = self._readiness[uid], self._transfer_at[uid]
        entry = self._retained.get(uid)
        if readiness >= transfer_at:
            assert entry is not None
            if uid in self._owners:
                return Staged(uid, readiness, payload=entry.buffer)
            return Staged(uid, readiness, tensors=entry.tensors)
        index, previous = self._host.lease(cancel)
        try:
            if previous is not None:
                previous.wait_host()  # the last transfer reading this buffer has finished
            buffer = self._host.buffer(index)
            region = buffer.narrow(0, 0, host_nbytes(patched, readiness, transfer_at))
            if entry is not None:
                # A clean state fused before its transfer is fused in a copy, never in the registry entry.
                for name, view in self._views.active(region, patched).items():
                    view.copy_(entry.tensors[name])
                return Staged(uid, Readiness.CLEAN, payload=region, host_index=index)
            cache = self._cache_for(patched.clean)
            assert self.source is not None
            with torch.inference_mode():
                staged = read_unit(
                    patched,
                    self.source,
                    region,
                    self._views,
                    transfer_at=transfer_at,
                    cache=cache,
                    parts=self._file_reads.get(uid),
                    # Raw inputs staged for value operations follow the transfer bytes (``host_buffer_nbytes``).
                    staging=buffer[aligned_region_bytes(region.numel()) :],
                )
            if cache is not None:
                self._cache_uses.add(patched.clean)
        except BaseException:
            self._reset_cache()
            self._host.release(index, None)
            raise
        return Staged(uid, staged, payload=region, host_index=index)

    def _deliver_staged(self, staged: Future[Staged], cancel: Event) -> Delivered:
        """Runs on the copy worker."""
        return self._deliver(staged.result(), cancel)

    def _deliver(self, staged: Staged, cancel: Event) -> Delivered:
        """Runs on the copy worker, or in ``need`` when the transport delivers on the consumer thread."""
        patched = self.units[self.indices[staged.uid]]
        try:
            index, after = self._device.lease(cancel)
        except BaseException:
            self._release_host(staged, None)
            raise
        handle: Transfer | None = None
        try:
            with torch.inference_mode(), self._transport.transfer(after) as handle:
                views = deliver(
                    patched,
                    staged,
                    self._transfer_at[staged.uid],
                    self._device.buffer(index),
                    handle,
                    self._views,
                    self.fuser,
                    self._factors.get(staged.uid, ()),
                )
        except BaseException:
            completion = handle.completion if handle is not None else after
            self._device.release(index, completion)
            self._release_host(staged, completion)
            raise
        self._release_host(staged, handle.completion)
        return Delivered(staged.uid, views, index, handle.completion)

    def _release_host(self, staged: Staged, completion: Completion | None) -> None:
        if staged.host_index is not None:
            self._host.release(staged.host_index, completion)

    def _cache_for(self, spec: UnitSpec) -> ExecutionCache | None:
        """Start a new shared-read pass when this unit was already consumed."""
        if self._cache is None or spec not in self._shared_set or not self._cache.tracks(spec):
            return None
        if spec in self._cache_uses:
            self._reset_cache()
        return self._cache

    def _reset_cache(self) -> None:
        if self._cache is not None and self.source is not None:
            self._cache = ExecutionCache(self._shared)
            self._cache_uses.clear()

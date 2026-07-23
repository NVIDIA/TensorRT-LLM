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
"""The two KV bounce transports (the real fabric-VMM one and the disabled null object) implementing
the contract in core.py. Holds the buffers, the gather and scatter kernels, and the scatter worker,
and runs the side effects that drive each region's state machine. Never imports transfer.py."""

import queue
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.agent import (
    MemoryDescs,
    MemoryType,
    TransferOp,
    TransferRequest,
)
from tensorrt_llm._utils import CUASSERT

from .buffer import SlotAllocator
from .config import SizingContext, fit_within_free
from .core import (
    BounceTransport,
    Disposition,
    GatherSourceInDoubtError,
    RecvBounceContext,
    ScatterState,
    Settlement,
)
from .gather_scatter import Plan, gather_contiguous, release_meta_buffers, scatter_contiguous

RidSlice = tuple  # the request id and slice id a region serves
_MIB = 1024 * 1024
_RESERVE_TIMEOUT_S = 0.2  # max wait for a bounce region before falling back to per-fragment
_CLOSE_JOIN_S = 2.0  # max wait for the scatter thread to drain on close
_WORKER_START_TIMEOUT_S = 2.0

# A partially constructed transport whose rollback could not finish must stay
# reachable until its registrations, streams, and mappings can be retried.
_INCOMPLETE_TRANSPORTS: set[object] = set()


class IncompleteBounceInitializationError(RuntimeError):
    """Construction failed and at least one transport resource remains owned."""

    def __init__(self, owner: "VmmBounceTransport", cause: Exception):
        super().__init__(f"{cause} (bounce initialization rollback remains incomplete)")
        self.owner = owner


@dataclass
class _PendingSettlement:
    settlement: Settlement
    physical_committed: bool = False
    physical_in_progress: bool = False
    callback_in_progress: bool = False


class VmmBounceTransport(BounceTransport):
    """The real transport: gather the request's cache into one fabric region, issue a single coalesced
    multi-rail write, and scatter it back on the receiver."""

    enabled = True

    @classmethod
    def from_config(
        cls, agent, cfg, *, device_id: int, block_bytes_per_group: List[int]
    ) -> Optional["VmmBounceTransport"]:
        """Build a transport sized from the config and clamped to free memory, or None if not even one
        chunk fits."""
        chunk = cfg.chunk_mb * _MIB
        free_b, total_b = CUASSERT(cudart.cudaMemGetInfo())
        want_capacity = cfg.sizing.resolve(
            SizingContext(
                free_bytes=free_b, total_bytes=total_b, chunk_bytes=chunk, device_id=device_id
            )
        )
        capacity_bytes = fit_within_free(want_capacity, free_bytes=free_b, chunk_bytes=chunk)
        if capacity_bytes is None:
            logger.warning(f"[kv-bounce] disabled: only {free_b // _MIB}MiB free")
            return None
        if capacity_bytes != want_capacity:
            logger.warning(
                f"[kv-bounce] each region clamped to {capacity_bytes // _MIB}MiB "
                f"(2x total) to fit {free_b // _MIB}MiB free"
            )
        return cls(
            agent,
            device_id=device_id,
            capacity_bytes=capacity_bytes,
            phys_chunk_size=chunk,
            block_bytes_per_group=block_bytes_per_group,
            min_blocks=cfg.min_blocks,
        )

    def __init__(
        self,
        agent,
        *,
        device_id: int,
        capacity_bytes: int,
        phys_chunk_size: int,
        block_bytes_per_group: List[int],
        min_blocks: int = 96,
        name: str = "kv_bounce",
    ):
        self._agent = agent
        self._device_id = device_id
        # The byte size of one cache block, listed for each attention layer group.
        self._block_bytes_per_group = list(block_bytes_per_group)
        # Below this many blocks, skip bounce: coalescing only pays off for long context (the default
        # is roughly twelve thousand tokens; a heuristic, and tunable).
        self._min_blocks = min_blocks
        self._send_alloc = None
        self._recv_alloc = None
        self._send_alloc_closed = False
        self._recv_alloc_closed = False
        self._reg_descs = []
        # Only descriptors still registered with the agent live here. Successful
        # teardown removes each entry immediately so a retry never deregisters it twice.
        self._registered_descs = []
        # CUDA events remain owned until destruction succeeds. A failed destroy
        # is retried by close rather than losing the handle.
        self._pending_events = []
        self._send_stream = None
        self._send_stream_healthy = True
        self._scatter_stream = None
        self._scatter_stream_healthy = True
        self._scatter_thread = None
        self._scatter_q = None
        self._scatter_ready = None
        self._scatter_start_error = None
        self._scatter_worker_error = None
        # gather_scatter metadata is reused per stream and remains live until
        # its async H2D copy completes, so one owner holds this through launch
        # and event completion.
        self._send_stream_lock = threading.RLock()
        self._init_recv_state()

        try:
            # One registered region each for sending and receiving. Construction
            # is transactional: every later failure rolls back the resources
            # that were successfully created before it.
            self._send_alloc = SlotAllocator(capacity_bytes, phys_chunk_size, name=f"{name}_send")
            self._recv_alloc = SlotAllocator(capacity_bytes, phys_chunk_size, name=f"{name}_recv")
            self._reg_descs = [self._send_alloc.reg_descs(), self._recv_alloc.reg_descs()]
            for desc in self._reg_descs:
                # Registration can fail after partially mutating the backend.
                # Track the attempt first so rollback conservatively requires
                # successful deregistration before unmapping its arena.
                self._registered_descs.append(desc)
                self._agent.register_memory(desc)

            self._send_stream = self._new_stream()
            self._start_scatter_worker(name)
        except Exception as e:
            if not self._rollback_initialization():
                _INCOMPLETE_TRANSPORTS.add(self)
                raise IncompleteBounceInitializationError(self, e) from e
            raise

        logger.info(
            f"[kv-bounce] Transport: send+recv regions of "
            f"{self._send_alloc.capacity / _MIB:.1f}MiB each"
        )

    def _stop_scatter_worker(self) -> None:
        """Stop an initialized scatter worker once; safe to retry after a timeout."""
        thread = getattr(self, "_scatter_thread", None)
        if thread is None:
            return
        if thread.is_alive():
            scatter_q = getattr(self, "_scatter_q", None)
            if scatter_q is not None:
                scatter_q.put(None)
            thread.join(timeout=_CLOSE_JOIN_S)
        if thread.is_alive():
            raise RuntimeError("[kv-bounce] scatter worker did not drain during close")
        self._scatter_thread = None

    def _destroy_stream(self, attr_name: str) -> None:
        """Evict quiesced metadata and destroy one CUDA stream.

        Callers establish that the stream has no outstanding work before this
        point.  Evict the cache before releasing the CUDA handle so another
        thread cannot create a new stream with the recycled handle and then
        have its metadata removed by this teardown.
        """
        stream = getattr(self, attr_name, None)
        if stream is None:
            return
        release_meta_buffers(stream, self._device_id)
        CUASSERT(cudart.cudaStreamDestroy(stream))
        setattr(self, attr_name, None)

    def _destroy_streams(self) -> list[Exception]:
        errors = []
        # The scatter worker must be stopped before its stream; send ownership
        # checks in close() prove no gather can still use the send stream.
        for attr_name in ("_scatter_stream", "_send_stream"):
            try:
                self._destroy_stream(attr_name)
            except Exception as e:
                errors.append(e)
        return errors

    def _destroy_event(self, event) -> None:
        CUASSERT(cudart.cudaEventDestroy(event))
        self._pending_events.remove(event)

    def _destroy_pending_events(self) -> list[Exception]:
        errors = []
        for event in list(getattr(self, "_pending_events", ())):
            try:
                self._destroy_event(event)
            except Exception as e:
                errors.append(e)
        return errors

    def _deregister_registered_descriptors(self) -> list[Exception]:
        errors = []
        remaining = []
        for desc in getattr(self, "_registered_descs", ()):
            try:
                self._agent.deregister_memory(desc)
            except Exception as e:
                errors.append(e)
                remaining.append(desc)
        self._registered_descs = remaining
        return errors

    def _close_allocators(self) -> list[Exception]:
        errors = []
        for attr_name, closed_attr in (
            ("_send_alloc", "_send_alloc_closed"),
            ("_recv_alloc", "_recv_alloc_closed"),
        ):
            allocator = getattr(self, attr_name, None)
            if allocator is None or getattr(self, closed_attr, False):
                continue
            try:
                allocator.close()
            except Exception as e:
                errors.append(e)
            else:
                setattr(self, closed_attr, True)
        return errors

    def _rollback_initialization(self) -> bool:
        """Retry fail-safe rollback; return whether every resource retired."""
        self._accepting_reservations = False
        worker_stopped = True
        try:
            self._stop_scatter_worker()
        except Exception as e:
            worker_stopped = False
            logger.error(f"[kv-bounce] constructor worker rollback failed: {e}")

        stream_errors = self._destroy_streams() if worker_stopped else []
        for error in stream_errors:
            logger.error(f"[kv-bounce] constructor stream rollback failed: {error}")
        event_errors = (
            self._destroy_pending_events() if worker_stopped and not stream_errors else []
        )
        for error in event_errors:
            logger.error(f"[kv-bounce] constructor event rollback failed: {error}")
        deregistration_errors = self._deregister_registered_descriptors()
        for error in deregistration_errors:
            logger.error(f"[kv-bounce] constructor deregistration rollback failed: {error}")

        # Never unmap an allocator while a registration, worker, or CUDA stream
        # may still refer to it. Buffer.__del__ intentionally retains such VMM.
        streams_destroyed = all(
            getattr(self, attr_name, None) is None
            for attr_name in ("_scatter_stream", "_send_stream")
        )
        if (
            worker_stopped
            and streams_destroyed
            and not self._pending_events
            and not self._registered_descs
        ):
            for error in self._close_allocators():
                logger.error(f"[kv-bounce] constructor allocator rollback failed: {error}")
        allocators_closed = all(
            getattr(self, allocator_attr, None) is None or getattr(self, closed_attr, False)
            for allocator_attr, closed_attr in (
                ("_send_alloc", "_send_alloc_closed"),
                ("_recv_alloc", "_recv_alloc_closed"),
            )
        )
        complete = (
            worker_stopped
            and streams_destroyed
            and not self._pending_events
            and not self._registered_descs
            and allocators_closed
        )
        if complete:
            self._closed = True
            _INCOMPLETE_TRANSPORTS.discard(self)
        return complete

    def retry_initialization_rollback(self) -> None:
        """Retry incomplete constructor cleanup without losing the owning object."""
        if not self._rollback_initialization():
            raise RuntimeError("[kv-bounce] initialization rollback is still incomplete")

    def _init_recv_state(self) -> None:
        # Live per-transfer state, guarded by a leaf lock: mutate and decide under it, then release
        # it before any CUDA sync, allocator call, or callback.
        self._reserved_map: Dict[RidSlice, RecvBounceContext] = {}
        self._reserved_map_lock = threading.Lock()
        self._accepting_reservations = True
        self._pending_reservations = 0
        self._pending_send_reservations = 0
        self._pending_settlements: Dict[RidSlice, _PendingSettlement] = {}
        self._closed = False
        self._close_lock = threading.Lock()

    def _start_scatter_worker(self, name: str) -> None:
        # Scatter runs on its own thread: it ends in a blocking sync, so keeping it off the
        # completion handler lets that handler keep draining other transfers.
        self._scatter_q: "queue.Queue" = queue.Queue()
        self._scatter_stream = self._new_stream()
        self._scatter_ready = threading.Event()
        self._scatter_start_error = None
        self._scatter_thread = threading.Thread(
            target=self._scatter_loop, name=f"{name}-scatter", daemon=True
        )
        self._scatter_thread.start()
        if not self._scatter_ready.wait(timeout=_WORKER_START_TIMEOUT_S):
            raise RuntimeError("[kv-bounce] scatter worker initialization timed out")
        if self._scatter_start_error is not None:
            raise RuntimeError(
                "[kv-bounce] scatter worker failed to initialize"
            ) from self._scatter_start_error

    def _new_stream(self):
        return CUASSERT(cudart.cudaStreamCreate())[0]

    def _launch_gather(self, src_addr: int, write_meta, total: int):
        """Launch the gather into the send region and return an event to wait on."""
        plan = Plan(write_meta.src_ptrs, write_meta.dst_ptrs, write_meta.sizes, total)
        with self._send_stream_lock:
            event = None
            try:
                gather_contiguous(
                    src_addr, plan.src_ptrs, plan.sizes, plan.offsets, stream=self._send_stream
                )
                event = CUASSERT(cudart.cudaEventCreate())[0]
                self._pending_events.append(event)
                CUASSERT(cudart.cudaEventRecord(event, self._send_stream))
            except Exception:
                if event is not None:
                    try:
                        self._destroy_event(event)
                    except Exception as cleanup_error:
                        logger.error(
                            f"[kv-bounce] failed to destroy gather event after launch error: "
                            f"{cleanup_error}"
                        )
                raise
        return event

    def _wait_gather(self, event) -> None:
        if event is not None:
            try:
                CUASSERT(cudart.cudaEventSynchronize(event))
            except Exception:
                try:
                    self._destroy_event(event)
                except Exception as cleanup_error:
                    logger.error(
                        f"[kv-bounce] failed to destroy gather event after wait error: "
                        f"{cleanup_error}"
                    )
                raise
            self._destroy_event(event)

    def _rollback_send_slot(self, slot_id: int) -> bool:
        """Retire a failed send slot; return whether source access is proven quiescent."""
        try:
            with self._send_stream_lock:
                CUASSERT(cudart.cudaStreamSynchronize(self._send_stream))
        except Exception as e:
            # A failed CUDA fence leaves the gather's physical-access state
            # ambiguous. Never hand the region to another transfer.
            self._send_stream_healthy = False
            try:
                self._send_alloc.quarantine(slot_id)
            except Exception as quarantine_error:
                # Poison the stream before this fallible bookkeeping step.  The
                # caller must still retain the transfer/source owner even when
                # the allocator cannot move the slot into its quarantine map.
                logger.critical(
                    f"[kv-bounce] failed to quarantine send slot {slot_id} after "
                    f"stream fence failure; arena ownership remains in doubt: "
                    f"{quarantine_error}"
                )
            logger.error(f"[kv-bounce] retained send slot {slot_id} after stream fence failed: {e}")
            return False
        else:
            self._send_alloc.release(slot_id)
            return True

    def _make_write(self, src_addr: int, write_meta, total: int):
        # one coalesced descriptor spanning the whole region
        sizes = np.array([total], dtype=np.int64)
        src = MemoryDescs.from_arrays_uniform_device(
            MemoryType.VRAM, np.array([src_addr], dtype=np.int64), sizes, self._device_id
        )
        dst = MemoryDescs.from_arrays_uniform_device(
            MemoryType.VRAM,
            np.array([write_meta.bounce_dst_base], dtype=np.int64),
            sizes,
            write_meta.dst_device_id,
        )
        return TransferRequest(TransferOp.WRITE, src, dst, write_meta.peer_name, None)

    def _reserve_send_slot(self, write_meta, *, timeout):
        """Reserve a send slot, or return None on backpressure/unhealthy transport."""
        total = int(write_meta.sizes.sum())
        with self._reserved_map_lock:
            if not self._accepting_reservations or not self._send_stream_healthy:
                return None
            self._pending_send_reservations += 1
        try:
            res = self._send_alloc.reserve(total, timeout=timeout)
        finally:
            with self._reserved_map_lock:
                self._pending_send_reservations -= 1
        if res is None:
            logger.debug(
                f"[kv-bounce] in-place: no send region space for {total // _MIB}MiB within {timeout}s "
                f"(sender backpressure); falling back"
            )
            return None
        slot_id, src_addr = res
        return slot_id, src_addr, total

    def build_request(self, write_meta):
        """Gather into a send slot and build the coalesced write, or None on backpressure. Frees the
        slot if the gather raises."""
        reserved = self._reserve_send_slot(write_meta, timeout=_RESERVE_TIMEOUT_S)
        if reserved is None:  # backpressure or unhealthy stream: fall back
            return None
        slot_id, src_addr, total = reserved
        with self._send_stream_lock:
            # Another gather may have poisoned the stream while this caller
            # reserved outside the stream-critical section.
            if not self._send_stream_healthy:
                self._send_alloc.release(slot_id)
                return None
            try:
                event = self._launch_gather(src_addr, write_meta, total)
                self._wait_gather(event)
                request = self._make_write(src_addr, write_meta, total)
            except Exception as e:
                if not self._rollback_send_slot(slot_id):
                    raise GatherSourceInDoubtError(
                        f"gather for send slot {slot_id} failed without a positive CUDA fence"
                    ) from e
                raise
        return request, slot_id

    def release_send(self, slot_id) -> None:
        """Release a send region after its write has completed."""
        self._send_alloc.release(slot_id)

    def quarantine_send(self, slot_id) -> None:
        """Retain a send slot until an external NIXL quiescence mechanism exists."""
        self._send_alloc.quarantine(slot_id)

    @staticmethod
    def _normalize_writer_ranks(writer_ranks: Sequence[int]) -> tuple[int, ...]:
        """Return an exact ordered writer plan, or empty for safe direct fallback.

        Integer counts are deliberately rejected: synthesizing ``range(count)``
        associates ownership with ranks that may not be the actual writers.
        """
        if isinstance(writer_ranks, (bool, int)):
            return ()
        try:
            ranks = tuple(writer_ranks)
        except TypeError:
            return ()
        if any(not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 for rank in ranks):
            return ()
        if len(set(ranks)) != len(ranks):
            return ()
        return ranks

    @staticmethod
    def _skip_bounce(reason: str, *, warn_key: Optional[str] = None) -> bool:
        """Log why a transfer falls back to the per-fragment path and return False, so the guards
        above stay one line each."""
        msg = f"[kv-bounce] in-place: {reason}"
        logger.warning_once(msg, key=warn_key) if warn_key else logger.debug(msg)
        return False

    def reserve(
        self,
        recv_req,
        writer_ranks: Sequence[int] = (),
        *,
        timeout: Optional[float] = _RESERVE_TIMEOUT_S,
        destination_intervals: Optional[Iterable[tuple[int, int]]] = None,
        destination_intervals_factory: Optional[Callable[[], Iterable[tuple[int, int]]]] = None,
    ) -> bool:
        """Reserve a region and create its state, recording the address for one sender.

        Multi-writer bounce is intentionally rejected until the caller can
        provide receiver-owned, byte-accurate offsets and extents for every
        writer.  Divisibility and uniform slot sizes do not constrain the size
        of the NIXL descriptor that a remote writer will submit.
        """
        ranks = self._normalize_writer_ranks(writer_ranks)
        if not ranks:
            return self._skip_bounce("writer plan is empty, invalid, or contains duplicate ranks")
        if destination_intervals is not None and destination_intervals_factory is not None:
            return self._skip_bounce(
                "trusted destination intervals and their lazy factory are mutually exclusive"
            )
        num_writers = len(ranks)
        if num_writers > 1:
            return self._skip_bounce(
                "multi-writer bounce requires byte-accurate receiver-owned writer extents",
                warn_key="kv-bounce-multiwriter-without-extents",
            )
        with self._reserved_map_lock:
            if not self._accepting_reservations or not self._scatter_stream_healthy:
                return self._skip_bounce("transport is closing")
        valid_block_counts = [
            int(np.count_nonzero(np.asarray(block_ids, dtype=np.int64) >= 0))
            for block_ids in recv_req.block_ids_per_layer_groups
        ]
        nblocks = sum(valid_block_counts)
        if nblocks < self._min_blocks:
            return self._skip_bounce(f"{nblocks} blocks < min {self._min_blocks} (too small)")
        total = 0
        for g, valid_blocks in enumerate(valid_block_counts):
            if g >= len(self._block_bytes_per_group):
                return self._skip_bounce(f"layer group {g} has no known slot size (e.g. mamba)")
            total += valid_blocks * self._block_bytes_per_group[g]
        if total <= 0:
            return self._skip_bounce(f"computed transfer size {total} <= 0")
        if total > self._recv_alloc.capacity:  # too big to ever fit, unlike transient backpressure
            return self._skip_bounce(
                f"transfer {total // _MIB}MiB exceeds the {self._recv_alloc.capacity // _MIB}MiB bounce "
                f"region; raise the bounce arena size to re-enable coalescing",
                warn_key="kv-bounce-oversize",
            )
        with self._reserved_map_lock:
            if not self._accepting_reservations or not self._scatter_stream_healthy:
                return self._skip_bounce("transport is closing")
            self._pending_reservations += 1
        try:
            res = self._recv_alloc.reserve(total, timeout=timeout)
            if res is None:
                return self._skip_bounce(
                    f"no recv region space for {total // _MIB}MiB within {timeout}s (backpressure)"
                )
            slot_id, addr = res
            try:
                if destination_intervals_factory is not None:
                    destination_intervals = destination_intervals_factory()
                    if destination_intervals is None:
                        raise ValueError(
                            "trusted destination interval factory returned no intervals"
                        )
                ctx = RecvBounceContext(
                    rid_slice=(recv_req.unique_rid, recv_req.slice_id),
                    slot_id=slot_id,
                    base_addr=addr,
                    per_writer_bytes=total // num_writers,
                    writer_ranks=ranks,
                    destination_intervals=destination_intervals,
                )
            except (TypeError, ValueError) as e:
                logger.error(f"[kv-bounce] invalid receive ownership plan: {e}")
                self._recv_alloc.release(slot_id)
                return self._skip_bounce("receive ownership plan is invalid")
            except Exception:
                self._recv_alloc.release(slot_id)
                raise

            duplicate = False
            with self._reserved_map_lock:
                if (
                    not self._accepting_reservations
                    or ctx.rid_slice in self._reserved_map
                    or ctx.rid_slice in self._pending_settlements
                ):
                    duplicate = True
                else:
                    self._reserved_map[ctx.rid_slice] = ctx
            if duplicate:
                self._recv_alloc.release(slot_id)
                return self._skip_bounce("transport is closing or transfer context already exists")
            recv_req.bounce_dst_base = addr
            # Positive marker: every fallback guard passed, so this transfer
            # provably uses the coalesced bounce path.
            logger.info_once(
                f"[kv-bounce] coalesced {nblocks} blocks / {total // _MIB}MiB into one region "
                f"across {num_writers} writer(s)",
                key="kv-bounce-coalesced",
            )
            return True
        finally:
            with self._reserved_map_lock:
                self._pending_reservations -= 1

    def writer_base(self, rid_slice: RidSlice, peer_rank: int) -> Optional[int]:
        """Where the exact planned fan-in writer writes in the region."""
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if ctx is None:
                return None
            try:
                return ctx.writer_base(peer_rank)
            except KeyError:
                return None

    def is_bounced(self, rid_slice: RidSlice) -> bool:
        with self._reserved_map_lock:
            return rid_slice in self._reserved_map or rid_slice in self._pending_settlements

    def release_idle_reservation(self, rid_slice: RidSlice) -> None:
        """Compatibility cancellation entry point.

        A genuinely idle context releases immediately. If any writer was exposed, this only closes
        future publication and retains the slot until those writers become terminal.
        """
        self.mark_logical_failure(rid_slice)

    def orphan_reservation(self, rid_slice: RidSlice) -> None:
        """Retain an ambiguous reservation until backend-wide quiescence."""
        self.mark_protocol_conflict(rid_slice)

    def _apply(self, rid_slice: RidSlice, mutate: Callable[[RecvBounceContext], None]) -> None:
        """Mutate and enqueue under the lock, then settle without holding it across allocator or
        callback work. No-op if the region is already gone."""
        retry_settlement = False
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if ctx is None:
                retry_settlement = rid_slice in self._pending_settlements
            else:
                mutate(ctx)
                if ctx.ready_to_scatter():
                    ctx.begin_scatter()
                    if self._scatter_stream_healthy:
                        # Queue insertion is inside the publication/close lock. This guarantees
                        # close's FIFO poison pill cannot overtake an accepted scatter.
                        self._enqueue_scatter(ctx, ctx.sorted_scatter_descs())
                    else:
                        # This context never reached CUDA because a prior job
                        # poisoned the shared stream.
                        ctx.suppress_scatter()
                if ctx.ready_to_settle():
                    settlement = ctx.settle()
                    if settlement is not None:
                        self._reserved_map.pop(rid_slice, None)
                        self._pending_settlements[rid_slice] = _PendingSettlement(settlement)
                        retry_settlement = True
        if retry_settlement:
            self._commit_pending_settlement(rid_slice)
        self._retry_logical_failure_notification(rid_slice)

    def _retry_logical_failure_notification(self, rid_slice: RidSlice) -> bool:
        """Deliver logical scatter failure without releasing its physical lease."""
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if ctx is None:
                return True
            callback = ctx.begin_logical_failure_notification()
            if callback is None:
                return not (
                    ctx.scatter_state is ScatterState.FAILED
                    and ctx.on_logical_failure is not None
                    and not ctx._logical_failure_notification_delivered
                )
        try:
            callback()
        except Exception as e:
            with self._reserved_map_lock:
                current = self._reserved_map.get(rid_slice)
                if current is ctx:
                    current.finish_logical_failure_notification(False)
            logger.error(
                f"[kv-bounce] logical scatter-failure notification failed "
                f"(slot={ctx.slot_id}); retaining it for retry: {e}"
            )
            return False
        with self._reserved_map_lock:
            current = self._reserved_map.get(rid_slice)
            if current is ctx:
                current.finish_logical_failure_notification(True)
        return True

    def _enqueue_scatter(self, ctx: RecvBounceContext, descs: List[tuple]) -> None:
        """Hand the per-writer fragments to the worker. Each is scattered from its own source, so a
        writer that fell back to the in-place path cannot shift where the others are read from."""
        self._scatter_q.put((ctx, descs))

    def _commit_pending_settlement(self, rid_slice: RidSlice) -> bool:
        """Commit physical release once and durably retry its ownership acknowledgement."""
        with self._reserved_map_lock:
            pending = self._pending_settlements.get(rid_slice)
            if pending is None:
                return True
            if pending.physical_in_progress or pending.callback_in_progress:
                return False
            settlement = pending.settlement
            commit_physical = not pending.physical_committed
            if commit_physical:
                pending.physical_in_progress = True

        if commit_physical:
            try:
                if settlement.disposition is Disposition.QUARANTINE:
                    self._recv_alloc.quarantine(settlement.slot_id)
                else:
                    self._recv_alloc.release(settlement.slot_id)
            except Exception as e:
                with self._reserved_map_lock:
                    pending.physical_in_progress = False
                logger.error(
                    f"[kv-bounce] physical settlement failed "
                    f"(slot={settlement.slot_id}, disposition={settlement.disposition.name}); "
                    f"retaining it for retry: {e}"
                )
                return False
            with self._reserved_map_lock:
                pending.physical_in_progress = False
                pending.physical_committed = True

        callback = settlement.on_done
        if callback is not None:
            with self._reserved_map_lock:
                current = self._pending_settlements.get(rid_slice)
                if current is not pending or pending.callback_in_progress:
                    return current is None
                pending.callback_in_progress = True
            try:
                callback(settlement.success)
            except Exception as e:
                with self._reserved_map_lock:
                    pending.callback_in_progress = False
                logger.error(
                    f"[kv-bounce] completion acknowledgement failed "
                    f"(slot={settlement.slot_id}); retaining it for retry: {e}"
                )
                return False
            else:
                # Acknowledge and remove atomically so no concurrent retry can
                # invoke an already successful callback a second time.
                with self._reserved_map_lock:
                    pending.callback_in_progress = False
                    if self._pending_settlements.get(rid_slice) is pending:
                        self._pending_settlements.pop(rid_slice, None)
                return True

        with self._reserved_map_lock:
            if self._pending_settlements.get(rid_slice) is pending:
                self._pending_settlements.pop(rid_slice, None)
        return True

    @staticmethod
    def _settlement_in_scope(rid_slice: RidSlice, scope) -> bool:
        """Whether an exact key belongs to a global, request, or exact-key retry."""
        if scope is None:
            return True
        if isinstance(scope, tuple):
            return rid_slice == scope
        return rid_slice[0] == scope

    def _retry_pending_settlements(self, scope=None) -> bool:
        with self._reserved_map_lock:
            keys = tuple(
                key for key in self._pending_settlements if self._settlement_in_scope(key, scope)
            )
        for key in keys:
            self._commit_pending_settlement(key)
        with self._reserved_map_lock:
            return not any(
                self._settlement_in_scope(key, scope) for key in self._pending_settlements
            )

    def retry_settlements(self, scope=None) -> bool:
        """Retry pending acknowledgements globally, for one request RID, or for one exact key."""
        with self._reserved_map_lock:
            keys = tuple(key for key in self._reserved_map if self._settlement_in_scope(key, scope))
        notifications_delivered = True
        for key in keys:
            notifications_delivered = (
                self._retry_logical_failure_notification(key) and notifications_delivered
            )
        return notifications_delivered and self._retry_pending_settlements(scope)

    def record_result(
        self,
        rid_slice: RidSlice,
        peer_rank: int,
        dst_ptrs=None,
        sizes=None,
        src_base=None,
        on_done: Optional[Callable[[bool], None]] = None,
        on_logical_failure: Optional[Callable[[], None]] = None,
    ) -> None:
        """A writer reported success. The completion callback fires only after the scatter lands, so
        the reader never sees completion before the cache is in place."""

        def mut(ctx: RecvBounceContext) -> None:
            accepted = ctx.record_writer_result(
                peer_rank, succeeded=True, src_base=src_base, dst_ptrs=dst_ptrs, sizes=sizes
            )
            if accepted:
                ctx.set_on_done(on_done)
                ctx.set_on_logical_failure(on_logical_failure)

        self._apply(rid_slice, mut)

    def record_failure(
        self,
        rid_slice: RidSlice,
        peer_rank: int,
        on_done: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """A writer reported failure (it has drained). The region is freed only once every writer has
        reported, not here."""

        def mut(ctx: RecvBounceContext) -> None:
            if ctx.record_writer_result(peer_rank, succeeded=False):
                ctx.set_on_done(on_done)

        self._apply(rid_slice, mut)

    def mark_writer_exposed(self, rid_slice: RidSlice, peer_rank: int) -> bool:
        """Atomically retain the slot for ``peer_rank`` before its address is published."""
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            return False if ctx is None else ctx.mark_writer_exposed(peer_rank)

    def record_no_access(
        self,
        rid_slice: RidSlice,
        peer_rank: int,
        *,
        succeeded: bool = True,
        on_done: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Settle a planned writer using proof that it cannot access the bounce slot."""

        def mut(ctx: RecvBounceContext) -> None:
            if ctx.mark_writer_no_access(peer_rank, succeeded=succeeded):
                ctx.set_on_done(on_done)

        self._apply(rid_slice, mut)

    def mark_logical_failure(
        self,
        rid_slice: RidSlice,
        on_done: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Suppress scatter/publication while retaining any possibly exposed writer."""

        def mut(ctx: RecvBounceContext) -> None:
            ctx.set_on_done(on_done)
            ctx.mark_logical_failure()

        self._apply(rid_slice, mut)

    def mark_protocol_conflict(
        self,
        rid_slice: RidSlice,
        on_done: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Suppress work and retain the slot until backend-wide quiescence."""

        def mut(ctx: RecvBounceContext) -> None:
            ctx.set_on_done(on_done)
            ctx.mark_protocol_conflict()

        self._apply(rid_slice, mut)

    def mark_backend_quiesced(
        self,
        rid_slice: Optional[RidSlice] = None,
        on_done: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Use backend-wide quiescence as terminal evidence for in-doubt remote access."""

        def mut(ctx: RecvBounceContext) -> None:
            ctx.set_on_done(on_done)
            ctx.mark_backend_quiesced()

        if rid_slice is not None:
            self._apply(rid_slice, mut)
            return
        with self._reserved_map_lock:
            keys = tuple(self._reserved_map)
        for key in keys:
            self._apply(key, mut)

    def _suppress_queued_scatters(self) -> None:
        """Fail queue entries that provably never launched after the stream was poisoned."""
        while True:
            try:
                item = self._scatter_q.get_nowait()
            except queue.Empty:
                return
            try:
                if item is None:
                    continue
                ctx, _descs = item
                self._apply(ctx.rid_slice, lambda current: current.suppress_scatter())
            finally:
                self._scatter_q.task_done()

    def _fail_scatter_worker(self, error: Exception) -> None:
        """Fail closed if an unexpected error escapes normal per-scatter handling."""
        with self._reserved_map_lock:
            self._scatter_stream_healthy = False
            self._accepting_reservations = False
            self._scatter_worker_error = error
        logger.error(
            f"[kv-bounce] scatter worker stopped unexpectedly; "
            f"disabling receive-bounce admission: {error}"
        )
        try:
            # Entries still in the queue never launched, so they can settle as
            # failures. The entry that escaped remains owned because its local
            # CUDA-access state is ambiguous.
            self._suppress_queued_scatters()
        except Exception as cleanup_error:
            logger.error(
                f"[kv-bounce] failed to suppress unlaunched scatters after worker failure; "
                f"retaining them for teardown retry: {cleanup_error}"
            )

    def _fail_current_scatter(self, ctx: RecvBounceContext) -> None:
        """Latch logical failure for the dequeued item without claiming a fence.

        This is the last-resort boundary for an exception that escapes the
        normal per-scatter path, including a state-update failure after CUDA
        work was launched. The context remains physically owned because the
        exception site does not itself prove local stream quiescence.
        """
        with self._reserved_map_lock:
            current = self._reserved_map.get(ctx.rid_slice)
            if current is ctx:
                current.finish_scatter(False)
        self._retry_logical_failure_notification(ctx.rid_slice)

    def _scatter_loop(self) -> None:
        """Run the worker with a fail-closed boundary around unexpected errors."""
        try:
            self._run_scatter_loop()
        except Exception as e:
            self._fail_scatter_worker(e)

    def _run_scatter_loop(self) -> None:
        try:
            CUASSERT(cudart.cudaSetDevice(self._device_id))
        except Exception as e:
            self._scatter_start_error = e
            self._scatter_ready.set()
            return
        self._scatter_ready.set()
        while True:
            try:
                item = self._scatter_q.get()
                if item is None:
                    return  # FIFO poison pill: every earlier scatter has finished
                ctx, descs = item
                try:
                    ok = True
                    try:
                        # Scatter each writer's fragments from its own source, never one global
                        # offset, so a missing or fallback writer cannot shift where the others
                        # are read from.
                        for src_base, dst_ptrs, sizes in descs:
                            p = Plan(dst_ptrs, dst_ptrs, sizes, int(sizes.sum()))
                            scatter_contiguous(
                                src_base,
                                p.dst_ptrs,
                                p.sizes,
                                p.offsets,
                                stream=self._scatter_stream,
                            )
                            # The metadata staging buffer is shared per stream, so it cannot be
                            # refilled for another writer until this writer's async metadata copy
                            # has completed.
                            CUASSERT(cudart.cudaStreamSynchronize(self._scatter_stream))
                    except Exception as e:
                        # A scatter failure must not kill the worker nor be reported as success.
                        ok = False
                        logger.error(f"[kv-bounce] scatter failed (slot={ctx.slot_id}): {e}")

                    # Success settles after the positive fence above. Failure is retained and
                    # poisons the shared stream: a CUDA error does not prove queued accesses
                    # stopped, so no later job may use it.
                    def finish(c, ok=ok):
                        if not ok:
                            self._scatter_stream_healthy = False
                        c.finish_scatter(ok)

                    self._apply(ctx.rid_slice, finish)
                    if not ok:
                        self._suppress_queued_scatters()
                        return
                except Exception:
                    self._fail_current_scatter(ctx)
                    raise
            finally:
                self._scatter_q.task_done()

    def close(self) -> None:
        """Drain the worker and destroy the arenas, but never underneath a live lease."""
        with self._close_lock:
            self._close_locked()

    def _close_locked(self) -> None:
        """Retryable close body, serialized so resources are retired at most once."""
        # Close reservation admission before inspecting drain state. If this
        # attempt finds live work and returns retryable failure, no new receive
        # or send lease may race the later retry.
        with self._reserved_map_lock:
            if self._closed:
                return
            self._accepting_reservations = False
        if not self._retry_pending_settlements():
            raise RuntimeError("[kv-bounce] cannot close with unacknowledged settlements")
        with self._reserved_map_lock:
            non_scatter_contexts = sum(
                ctx.scatter_state is not ScatterState.QUEUED for ctx in self._reserved_map.values()
            )
            if non_scatter_contexts:
                raise RuntimeError(
                    f"[kv-bounce] cannot close with {non_scatter_contexts} live receive contexts"
                )
            if self._pending_reservations or self._pending_send_reservations:
                raise RuntimeError("[kv-bounce] cannot close while a reservation is pending")
            if self._send_alloc.has_outstanding:
                raise RuntimeError("[kv-bounce] cannot close with outstanding arena slots")
            if not self._reserved_map and self._recv_alloc.has_outstanding:
                raise RuntimeError("[kv-bounce] cannot close with outstanding arena slots")

        # FIFO poison pill makes the worker complete every item already accepted before exiting.
        self._stop_scatter_worker()
        with self._reserved_map_lock:
            if self._reserved_map or self._recv_alloc.has_outstanding:
                raise RuntimeError("[kv-bounce] scatter drain left outstanding receive contexts")
        if not self._retry_pending_settlements():
            raise RuntimeError("[kv-bounce] cannot close with unacknowledged settlements")

        stream_errors = self._destroy_streams()
        if stream_errors:
            raise RuntimeError(
                f"[kv-bounce] failed to destroy {len(stream_errors)} CUDA stream(s)"
            ) from stream_errors[0]

        event_errors = self._destroy_pending_events()
        if event_errors:
            raise RuntimeError(
                f"[kv-bounce] failed to destroy {len(event_errors)} CUDA event(s)"
            ) from event_errors[0]

        deregistration_errors = self._deregister_registered_descriptors()
        if deregistration_errors:
            raise RuntimeError(
                f"[kv-bounce] failed to deregister "
                f"{len(deregistration_errors)} memory descriptor(s)"
            ) from deregistration_errors[0]

        allocator_errors = self._close_allocators()
        if allocator_errors:
            raise RuntimeError(
                f"[kv-bounce] failed to close {len(allocator_errors)} arena allocator(s)"
            ) from allocator_errors[0]
        with self._reserved_map_lock:
            self._closed = True


class NoBounceTransport(BounceTransport):
    """The disabled transport, used when bounce is off so callers need no None checks. Every method
    is a no-op or a negative answer."""

    enabled = False
    _reg_descs = ()

    def build_request(self, write_meta):
        return None

    def release_send(self, slot_id) -> None:
        pass

    def quarantine_send(self, slot_id) -> None:
        pass

    def reserve(
        self,
        recv_req,
        writer_ranks: Sequence[int] = (),
        *,
        timeout: Optional[float] = _RESERVE_TIMEOUT_S,
        destination_intervals: Optional[Iterable[tuple[int, int]]] = None,
        destination_intervals_factory: Optional[Callable[[], Iterable[tuple[int, int]]]] = None,
    ) -> bool:
        return False

    def writer_base(self, rid_slice, peer_rank: int):
        return None

    def is_bounced(self, rid_slice) -> bool:
        return False

    def release_idle_reservation(self, rid_slice) -> None:
        pass

    def orphan_reservation(self, rid_slice) -> None:
        pass

    def mark_writer_exposed(self, rid_slice, peer_rank: int) -> bool:
        return False

    def record_no_access(
        self, rid_slice, peer_rank: int, *, succeeded: bool = True, on_done=None
    ) -> None:
        pass

    def mark_logical_failure(self, rid_slice, on_done=None) -> None:
        pass

    def mark_protocol_conflict(self, rid_slice, on_done=None) -> None:
        pass

    def mark_backend_quiesced(self, rid_slice=None, on_done=None) -> None:
        pass

    def retry_settlements(self, scope=None) -> bool:
        return True

    def record_result(
        self,
        rid_slice,
        peer_rank,
        dst_ptrs=None,
        sizes=None,
        src_base=None,
        on_done=None,
        on_logical_failure=None,
    ):
        pass

    def record_failure(self, rid_slice, peer_rank, on_done=None) -> None:
        pass

    def close(self) -> None:
        pass


class CleanupPendingBounceTransport(NoBounceTransport):
    """Direct-path transport that retains a partially constructed VMM owner."""

    def __init__(self, owner: VmmBounceTransport):
        self._owner = owner

    def close(self) -> None:
        owner = self._owner
        if owner is None:
            return
        owner.retry_initialization_rollback()
        self._owner = None


def create_bounce(agent, cfg, *, device_id: int, page_table) -> BounceTransport:
    """Build the real transport from the config, or the disabled one when bounce is off, it cannot
    fit, or the fabric allocation races."""
    if cfg is None:
        return NoBounceTransport()
    try:
        transport = VmmBounceTransport.from_config(
            agent, cfg, device_id=device_id, block_bytes_per_group=block_bytes_per_group(page_table)
        )
        return transport if transport is not None else NoBounceTransport()
    except IncompleteBounceInitializationError as e:
        logger.error(
            f"[kv-bounce] disabled with incomplete initialization cleanup: {e}; "
            "retaining cleanup owner"
        )
        return CleanupPendingBounceTransport(e.owner)
    except Exception as e:  # rare race: memory taken between the query and allocation
        logger.warning(f"[kv-bounce] disabled (alloc failed: {e}); using in-place path")
        return NoBounceTransport()


def build_send_request(bounce, write_meta, fallback):
    """Build a coalesced bounce write when eligible (release the returned slot afterward), otherwise
    fall back to the per-fragment request."""
    if write_meta.bounce_dst_base is not None:
        built = bounce.build_request(write_meta)
        if built is not None:
            return built
    return fallback(), None


def scatter_write_result(
    bounce, rid_slice, peer_rank: int, dst_ptrs, sizes, src_base=None, on_done=None
) -> None:
    """Handle a success result: a bounced transfer records the writer and scatters once all arrive; a
    non-bounced transfer already landed in place, so fire the callback inline."""
    if bounce.is_bounced(rid_slice):
        bounce.record_result(rid_slice, peer_rank, dst_ptrs, sizes, src_base, on_done)
    elif on_done is not None:
        on_done(True)


def encode_result_tail(write_meta) -> list:
    """The binary tail appended to a bounced result: the destination fragment table and the source
    this writer wrote to, so the receiver can scatter and order the fan-in writers."""
    sb = write_meta.bounce_dst_base if write_meta.bounce_dst_base is not None else 0
    return [
        write_meta.dst_ptrs.tobytes(),
        write_meta.sizes.tobytes(),
        np.array([sb], dtype=np.int64).tobytes(),
    ]


def decode_result_tail(message):
    """Recover the destination fragments, sizes, and source from the optional trailing frames, or
    nothing if the tail is absent."""
    if len(message) >= 5:
        return (
            np.frombuffer(message[2], dtype=np.int64),
            np.frombuffer(message[3], dtype=np.int64),
            int(np.frombuffer(message[4], dtype=np.int64)[0]),
        )
    return None, None, None


def block_bytes_per_group(page_table) -> list:
    """Byte size of one cache block for each leading attention layer group.

    A layer group can expose multiple pool views (for example, the ordinary KV pool plus an indexer
    pool). The sender gathers fragments from every mapped view, so the receive reservation must
    cover every pool view. This intentionally mirrors the sender's per-view descriptor loop: even
    repeated views of one physical pool contribute repeated gathered extents. Stop at the first
    non-attention group because bounce does not size Mamba state.
    """
    from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup
    from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool

    assert page_table is not None
    out: list = []
    for lg_idx, lg in enumerate(page_table.layer_groups):
        if not isinstance(lg, AttentionLayerGroup):
            break
        out.append(
            sum(
                int(get_physical_pool(page_table, lg_idx, int(pool_view.pool_idx)).slot_bytes)
                for pool_view in lg.pool_views
            )
        )
    return out

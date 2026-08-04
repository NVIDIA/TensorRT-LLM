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
from .core import BounceTransport, Disposition, Settlement, TransferContext
from .gather_scatter import Plan, gather_contiguous, scatter_contiguous

RidSlice = tuple  # the request id and slice id a region serves
_MIB = 1024 * 1024
_SCATTER_POLL_S = 0.5  # how often the scatter worker wakes to re-check the stop flag and reclaim
_RESERVE_TIMEOUT_S = 0.2  # max wait for a bounce region before falling back to per-fragment
_CLOSE_JOIN_S = 2.0  # max wait for the scatter thread to drain on close
_QUARANTINE_GRACE_S = 60.0  # how long an orphaned region is held out of reuse


class VmmBounceTransport(BounceTransport):
    """The real transport: gather the request's cache into one fabric region, issue a single coalesced
    multi-rail write, and scatter it back on the receiver."""

    enabled = True

    @classmethod
    def from_config(
        cls,
        agent,
        cfg,
        *,
        device_id: int,
        block_bytes_per_group: List[int],
        destination_pool_layouts: Optional[List[List[tuple[int, int, int]]]] = None,
        valid_destination_ranges: Optional[List[tuple[int, int]]] = None,
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
            destination_pool_layouts=destination_pool_layouts,
            valid_destination_ranges=valid_destination_ranges,
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
        destination_pool_layouts: Optional[List[List[tuple[int, int, int]]]] = None,
        valid_destination_ranges: Optional[List[tuple[int, int]]] = None,
        min_blocks: int = 96,
        quarantine_grace_s: float = _QUARANTINE_GRACE_S,
        name: str = "kv_bounce",
    ):
        self._agent = agent
        self._device_id = device_id
        # The byte size of one cache block, listed for each attention layer group.
        self._block_bytes_per_group = list(block_bytes_per_group)
        # Receiver-owned physical slot layouts, indexed by layer group. These
        # produce a request-specific destination allowlist and validate the
        # exact per-rank plans before any bounce address is advertised.
        # ``valid_destination_ranges`` is retained only as a compatibility/testing
        # fallback for callers without page-table metadata; production
        # construction always supplies the layouts.
        self._destination_pool_layouts = tuple(
            tuple(group) for group in (destination_pool_layouts or ())
        )
        self._valid_destination_ranges = tuple(valid_destination_ranges or ())
        # Below this many blocks, skip bounce: coalescing only pays off for long context (the default
        # is roughly twelve thousand tokens; a heuristic, and tunable).
        self._min_blocks = min_blocks
        # how long an orphaned region is held out of reuse; must outlast the worst in-flight write
        self._quarantine_grace_s = quarantine_grace_s

        # one registered region each for sending and receiving
        self._send_alloc = SlotAllocator(capacity_bytes, phys_chunk_size, name=f"{name}_send")
        self._recv_alloc = SlotAllocator(capacity_bytes, phys_chunk_size, name=f"{name}_recv")
        self._reg_descs = [self._send_alloc.reg_descs(), self._recv_alloc.reg_descs()]
        for d in self._reg_descs:
            self._agent.register_memory(d)

        self._send_stream = self._new_stream()
        self._send_stream_lock = threading.Lock()

        self._init_recv_state()
        self._start_scatter_worker(name)

        logger.info(
            f"[kv-bounce] Transport: send+recv regions of "
            f"{self._send_alloc.capacity / _MIB:.1f}MiB each"
        )

    def _init_recv_state(self) -> None:
        # Live per-transfer state, guarded by a leaf lock: mutate and decide under it, then release
        # it before any CUDA sync, allocator call, or callback.
        self._reserved_map: Dict[RidSlice, TransferContext] = {}
        self._reserved_map_lock = threading.Lock()

    def _start_scatter_worker(self, name: str) -> None:
        # Scatter runs on its own thread: it ends in a blocking sync, so keeping it off the
        # completion handler lets that handler keep draining other transfers.
        self._scatter_q: "queue.Queue" = queue.Queue()
        self._scatter_stream = self._new_stream()
        self._stop = threading.Event()
        self._scatter_thread = threading.Thread(
            target=self._scatter_loop, name=f"{name}-scatter", daemon=True
        )
        self._scatter_thread.start()

    def _new_stream(self):
        return CUASSERT(cudart.cudaStreamCreate())[0]

    def _gather_blocking(self, src_addr: int, write_meta, total: int) -> None:
        """Gather the scattered fragments into the send region and block until done. The whole gather
        runs under the stream lock so a second sender thread can't overwrite the shared staging buffer
        mid-copy and corrupt this region; only the fast gather serializes, the writes stay parallel."""
        plan = Plan(write_meta.src_ptrs, write_meta.dst_ptrs, write_meta.sizes, total)
        with self._send_stream_lock:
            gather_contiguous(
                src_addr, plan.src_ptrs, plan.sizes, plan.offsets, stream=self._send_stream
            )
            event = CUASSERT(cudart.cudaEventCreate())[0]
            try:
                CUASSERT(cudart.cudaEventRecord(event, self._send_stream))
                CUASSERT(cudart.cudaEventSynchronize(event))
            finally:
                CUASSERT(cudart.cudaEventDestroy(event))

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

    def _reserve_and_gather(self, write_meta, *, timeout):
        """Reserve a send slot and gather into it, or None on send-region backpressure. Eligibility
        was already decided by the receiver, so the sender only falls back under backpressure."""
        total = int(write_meta.sizes.sum())
        res = self._send_alloc.reserve(total, timeout=timeout)
        if res is None:
            logger.debug(
                f"[kv-bounce] in-place: no send region space for {total // _MIB}MiB within {timeout}s "
                f"(sender backpressure); falling back"
            )
            return None
        slot_id, src_addr = res
        try:
            self._gather_blocking(src_addr, write_meta, total)
        except Exception:
            self._send_alloc.release(slot_id)  # free the slot if the gather raises
            raise
        return slot_id, src_addr, total

    @staticmethod
    def _canonicalize_write_fragments(write_meta) -> None:
        """Put bounce fragments in receiver-verifiable destination order.

        The contiguous bounce source preserves this order. Sorting all three
        arrays together therefore keeps source data paired with its destination
        while making the result tail independently checkable by the receiver.
        """
        if write_meta.dst_ptrs.size < 2:
            return
        order = sorted(
            range(write_meta.dst_ptrs.size),
            key=lambda index: int(write_meta.dst_ptrs[index]),
        )
        if order == list(range(write_meta.dst_ptrs.size)):
            return
        write_meta.src_ptrs = np.asarray(
            [write_meta.src_ptrs[index] for index in order], dtype=np.int64
        )
        write_meta.dst_ptrs = np.asarray(
            [write_meta.dst_ptrs[index] for index in order], dtype=np.int64
        )
        write_meta.sizes = np.asarray([write_meta.sizes[index] for index in order], dtype=np.int64)

    def build_request(self, write_meta):
        """Gather into a send slot and build the coalesced write, or None on backpressure. The gather
        blocks (and frees the slot on failure) inside _reserve_and_gather."""
        self._canonicalize_write_fragments(write_meta)
        gathered = self._reserve_and_gather(write_meta, timeout=_RESERVE_TIMEOUT_S)
        if gathered is None:  # backpressure: fall back
            return None
        slot_id, src_addr, total = gathered
        try:
            return self._make_write(src_addr, write_meta, total), slot_id
        except Exception:
            # The gather completed but submission has not started, so this
            # send slot is still locally quiescent and can be released.
            self._send_alloc.release(slot_id)
            raise

    def release_send(self, slot_id) -> None:
        """Release a send region after its write has completed."""
        self._send_alloc.release(slot_id)

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
        num_writers: int = 1,
        *,
        timeout: Optional[float] = _RESERVE_TIMEOUT_S,
        expected_destination_plans: Optional[dict[int, tuple[np.ndarray, np.ndarray]]] = None,
    ) -> bool:
        """Reserve a region and create its state, recording the address for the senders. Returns
        False to fall back to the per-fragment path. A fan-in splits the region evenly, so the total
        must divide across the writers."""
        if getattr(recv_req, "mamba_state_index", None) is not None:
            return self._skip_bounce("mamba state has no receiver-owned bounce scatter plan")

        valid_block_ids_per_group: list[np.ndarray] = []
        for block_ids in recv_req.block_ids_per_layer_groups:
            # Production requests carry int64 ndarrays. The size-only fallback
            # keeps lightweight unit fakes usable while production remains
            # request-bound through ``_destination_pool_layouts`` below.
            try:
                values = np.asarray(block_ids, dtype=np.int64)
            except (TypeError, ValueError):
                values = np.asarray(list(range(int(block_ids.size))), dtype=np.int64)
            if values.ndim != 1:
                return self._skip_bounce("receiver block IDs are not one-dimensional")
            values = np.asarray(
                [int(block_id) for block_id in values if int(block_id) >= 0],
                dtype=np.int64,
            )
            valid_block_ids_per_group.append(values)

        nblocks = sum(int(a.size) for a in valid_block_ids_per_group)
        if nblocks < self._min_blocks:
            return self._skip_bounce(f"{nblocks} blocks < min {self._min_blocks} (too small)")
        total = 0
        for g, block_ids in enumerate(valid_block_ids_per_group):
            if g >= len(self._block_bytes_per_group):
                return self._skip_bounce(f"layer group {g} has no known slot size (e.g. mamba)")
            total += int(block_ids.size) * self._block_bytes_per_group[g]
        if total <= 0:
            return self._skip_bounce(f"computed transfer size {total} <= 0")
        if num_writers > 1 and total % num_writers != 0:
            return self._skip_bounce(
                f"fan-in {total}B across {num_writers} senders is not an even split "
                f"({total % num_writers}B remainder); head-mismatch explosion NOT mitigated",
                warn_key="kv-bounce-uneven-fanin",
            )
        if num_writers > 1:
            # Fan-in gives each writer an equal share of the region, which only matches where it
            # writes when all writers send the same bytes. Equal layer count guarantees that only
            # when the per-block sizes match, so require that here, else fall back.
            present_slot_bytes = {
                self._block_bytes_per_group[g]
                for g, block_ids in enumerate(valid_block_ids_per_group)
                if int(block_ids.size) > 0
            }
            if len(present_slot_bytes) > 1:
                return self._skip_bounce(
                    f"fan-in across {num_writers} senders with non-uniform layer-group slot bytes "
                    f"{sorted(present_slot_bytes)}; the equal split would overrun a sub-region",
                    warn_key="kv-bounce-heterogeneous-fanin",
                )
        if total > self._recv_alloc.capacity:  # too big to ever fit, unlike transient backpressure
            return self._skip_bounce(
                f"transfer {total // _MIB}MiB exceeds the {self._recv_alloc.capacity // _MIB}MiB bounce "
                f"region; raise the bounce arena size to re-enable coalescing",
                warn_key="kv-bounce-oversize",
            )
        res = self._recv_alloc.reserve(total, timeout=timeout)
        if res is None:
            return self._skip_bounce(
                f"no recv region space for {total // _MIB}MiB within {timeout}s (backpressure)"
            )
        slot_id, addr = res
        recv_req.bounce_dst_base = addr
        try:
            allowed_destination_ranges = self._request_destination_ranges(valid_block_ids_per_group)
            normalized_destination_plans = self._normalize_destination_plans(
                expected_destination_plans,
                num_writers=num_writers,
                per_writer_bytes=total // num_writers,
                allowed_destination_ranges=allowed_destination_ranges,
            )
        except ValueError as error:
            self._recv_alloc.release(slot_id)
            recv_req.bounce_dst_base = None
            return self._skip_bounce(str(error))
        with self._reserved_map_lock:
            ctx = TransferContext(
                rid_slice=(recv_req.unique_rid, recv_req.slice_id),
                slot_id=slot_id,
                base_addr=addr,
                per_writer_bytes=total // num_writers,
                num_writers=num_writers,
                allowed_destination_ranges=allowed_destination_ranges,
                expected_destination_plans=normalized_destination_plans,
            )
            self._reserved_map[ctx.rid_slice] = ctx  # inactive until the first writer reports
        # Positive marker: all fall-back guards above passed, so this transfer provably takes the
        # coalesced-bounce WRITE path. Logged once (per process) so an e2e test can assert that
        # bounce actually engaged instead of silently falling back to the per-fragment path.
        logger.info_once(
            f"[kv-bounce] coalesced {nblocks} blocks / {total // _MIB}MiB into one region "
            f"across {num_writers} writer(s)",
            key="kv-bounce-coalesced",
        )
        return True

    def _request_destination_ranges(
        self, block_ids_per_group: List[np.ndarray]
    ) -> tuple[tuple[int, int], ...]:
        """Build the immutable receiver-owned destination allowlist.

        Result tails describe scatter fragments for performance, but they must
        never gain authority to write into another request's cache blocks. The
        receiver derives the only legal physical slots from its own page table
        and the block IDs already attached to this receive task.
        """
        if not self._destination_pool_layouts:
            if not self._valid_destination_ranges:
                raise ValueError("no receiver-owned KV destination ranges are available")
            return self._valid_destination_ranges
        if len(block_ids_per_group) > len(self._destination_pool_layouts):
            raise ValueError("receiver block groups exceed the local KV pool layout")

        ranges: set[tuple[int, int]] = set()
        for group_index, block_ids in enumerate(block_ids_per_group):
            layouts = self._destination_pool_layouts[group_index]
            if block_ids.size and not layouts:
                raise ValueError(f"layer group {group_index} has no local KV destination pool")
            for base, slot_bytes, num_slots in layouts:
                bad = next(
                    (int(block_id) for block_id in block_ids if int(block_id) >= num_slots),
                    None,
                )
                if bad is not None:
                    raise ValueError(
                        f"receiver block ID {bad} exceeds layer-group {group_index} "
                        f"pool capacity {num_slots}"
                    )
                ranges.update(
                    (base + int(block_id) * slot_bytes, base + (int(block_id) + 1) * slot_bytes)
                    for block_id in block_ids
                )
        if not ranges:
            raise ValueError("receiver request has no valid KV destination slots")
        return tuple(sorted(ranges))

    def _normalize_destination_plans(
        self,
        plans: Optional[dict[int, tuple[np.ndarray, np.ndarray]]],
        *,
        num_writers: int,
        per_writer_bytes: int,
        allowed_destination_ranges: tuple[tuple[int, int], ...],
    ) -> dict[int, tuple[tuple[int, int], ...]]:
        """Validate and freeze exact receiver-derived plans before advertisement."""
        if not self._destination_pool_layouts:
            return {}
        if plans is None or len(plans) != num_writers:
            raise ValueError(
                "receiver-derived bounce destination plans do not match the writer fan-in"
            )

        normalized: dict[int, tuple[tuple[int, int], ...]] = {}
        all_fragments: list[tuple[int, int]] = []
        for peer_rank, (dst_ptrs, sizes) in plans.items():
            dst_ptrs = np.asarray(dst_ptrs, dtype=np.int64)
            sizes = np.asarray(sizes, dtype=np.int64)
            if dst_ptrs.ndim != 1 or sizes.ndim != 1 or dst_ptrs.size != sizes.size:
                raise ValueError(f"invalid receiver-derived scatter plan for rank {peer_rank}")
            if np.any(sizes <= 0):
                raise ValueError(
                    f"receiver-derived scatter plan for rank {peer_rank} has non-positive sizes"
                )
            plan = tuple((int(ptr), int(size)) for ptr, size in zip(dst_ptrs, sizes, strict=True))
            if sum(size for _ptr, size in plan) != per_writer_bytes:
                raise ValueError(
                    f"receiver-derived scatter plan for rank {peer_rank} does not describe "
                    f"exactly {per_writer_bytes} bytes"
                )
            for index, (ptr, size) in enumerate(plan):
                end = ptr + size
                if index and ptr < plan[index - 1][0]:
                    raise ValueError(
                        f"receiver-derived scatter plan for rank {peer_rank} is not canonical"
                    )
                if index and ptr < plan[index - 1][0] + plan[index - 1][1]:
                    raise ValueError(f"receiver-derived scatter plan for rank {peer_rank} overlaps")
                if not any(
                    valid_start <= ptr and end <= valid_end
                    for valid_start, valid_end in allowed_destination_ranges
                ):
                    raise ValueError(
                        f"receiver-derived scatter plan for rank {peer_rank} is outside "
                        "the request's KV slots"
                    )
                all_fragments.append((ptr, end))
            normalized[int(peer_rank)] = plan

        try:
            actual = self._coalesce_destination_ranges(all_fragments, reject_overlap=True)
            expected = self._coalesce_destination_ranges(
                allowed_destination_ranges, reject_overlap=False
            )
        except RuntimeError as error:
            raise ValueError(str(error)) from error
        if actual != expected:
            raise ValueError(
                "receiver-derived writer plans do not exactly cover the request's KV slots"
            )
        return normalized

    def writer_base(self, rid_slice: RidSlice, writer_index: int) -> Optional[int]:
        """Where the given fan-in writer writes in the region."""
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            return None if ctx is None else ctx.writer_base(writer_index)

    def bind_writer(self, rid_slice: RidSlice, peer_rank: int, writer_index: int) -> Optional[int]:
        """Bind a rank before its sub-region address is advertised."""
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if ctx is None:
                return None
            src_base = ctx.writer_base(writer_index)
            ctx.bind_writer(peer_rank, src_base)
            return src_base

    def set_completion_callback(
        self, rid_slice: RidSlice, on_settled: Callable[[bool], None]
    ) -> None:
        """Install unconditional settlement accounting before any writer is advertised."""
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if ctx is None:
                raise RuntimeError(f"bounce callback for unknown reservation {rid_slice}")
            ctx.set_completion_callback(on_settled)

    def is_bounced(self, rid_slice: RidSlice) -> bool:
        with self._reserved_map_lock:
            return rid_slice in self._reserved_map

    def release_idle_reservation(self, rid_slice: RidSlice) -> None:
        """Immediately release a reservation cancelled before any address went out; no write can be
        in flight. Idempotent. Drained transfers finalize through the result path instead."""
        with self._reserved_map_lock:
            ctx = self._reserved_map.pop(rid_slice, None)
        if ctx is not None:
            self._recv_alloc.release(ctx.slot_id)

    def orphan_reservation(self, rid_slice: RidSlice) -> None:
        """Retain a reservation whose remote write may still be in flight.

        A time-based quarantine is not a safety proof: an RMA can outlive any
        chosen grace period. ``confirm_drained`` or all writer results must
        prove quiescence before this slot returns to the allocator.
        """
        self._apply(rid_slice, lambda ctx: ctx.mark_orphaned())

    def confirm_drained(self, rid_slice: RidSlice) -> None:
        """Release an orphan after sender drain ACK proves reuse is safe."""
        self._apply(rid_slice, lambda ctx: ctx.confirm_drained())

    def _apply(
        self,
        rid_slice: RidSlice,
        mutate: Callable[[TransferContext], None],
        *,
        require_present: bool = False,
    ) -> None:
        """Mutate the state under the lock, then do what it asks (scatter or settle) with the lock
        released, never holding it across a CUDA sync, a queue put, or a callback. No-op if the
        region is already gone."""
        scatter: Optional[tuple] = None
        settlement: Optional[Settlement] = None
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if ctx is None:
                if require_present:
                    raise RuntimeError(f"bounced result for unknown reservation {rid_slice}")
                return
            mutate(ctx)
            if ctx.ready_to_scatter():
                ctx.begin_scatter()
                scatter = (ctx, ctx.sorted_scatter_descs())
            elif ctx.ready_to_settle():
                settlement = ctx.settle()
                if settlement is not None:
                    self._reserved_map.pop(rid_slice, None)
        if scatter is not None:
            self._enqueue_scatter(*scatter)
        if settlement is not None:
            self._commit(settlement)

    def _enqueue_scatter(self, ctx: TransferContext, descs: List[tuple]) -> None:
        """Hand the per-writer fragments to the worker. Each is scattered from its own source, so a
        writer that fell back to the in-place path cannot shift where the others are read from."""
        self._scatter_q.put((ctx, descs))

    def _commit(self, settlement: Settlement) -> None:
        """Carry out the decision: release or quarantine the slot, then fire the callback once. No
        lock is held."""
        if settlement.disposition is Disposition.QUARANTINE:
            self._recv_alloc.quarantine(settlement.slot_id, self._quarantine_grace_s)
        else:
            self._recv_alloc.release(settlement.slot_id)
        if settlement.on_done is not None:
            try:
                settlement.on_done(settlement.success)
            except Exception as e:  # never let the callback strand the arena
                logger.error(
                    f"[kv-bounce] completion callback failed (slot={settlement.slot_id}): {e}"
                )

    def record_result(
        self,
        rid_slice: RidSlice,
        peer_rank: int,
        dst_ptrs=None,
        sizes=None,
        src_base=None,
        on_done: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """A writer reported success. The completion callback fires only after the scatter lands, so
        the reader never sees completion before the cache is in place."""

        def mut(ctx: TransferContext) -> None:
            validated_dst_ptrs, validated_sizes = self._validate_scatter_tail(
                ctx, peer_rank, dst_ptrs, sizes, src_base
            )
            if on_done is not None:
                ctx.on_done = on_done
            ctx.record_writer_result(
                peer_rank,
                succeeded=True,
                src_base=src_base,
                dst_ptrs=validated_dst_ptrs,
                sizes=validated_sizes,
            )

        self._apply(rid_slice, mut, require_present=True)

    def _validate_scatter_tail(
        self, ctx: TransferContext, peer_rank: int, dst_ptrs, sizes, src_base
    ) -> tuple[np.ndarray, np.ndarray]:
        if dst_ptrs is None or sizes is None or src_base is None:
            raise RuntimeError("incomplete bounced-result scatter tail")
        dst_ptrs = np.asarray(dst_ptrs, dtype=np.int64)
        sizes = np.asarray(sizes, dtype=np.int64)
        if dst_ptrs.ndim != 1 or sizes.ndim != 1 or dst_ptrs.size != sizes.size:
            raise RuntimeError("invalid bounced-result pointer/size arrays")
        if np.any(sizes <= 0):
            raise RuntimeError("bounced-result fragment sizes must be positive")
        expected_src_base = ctx._expected_writer_bases.get(peer_rank)
        if expected_src_base is None or src_base != expected_src_base:
            raise RuntimeError(f"bounced result source identity mismatch for rank {peer_rank}")
        described_bytes = sum(int(size) for size in sizes)
        if described_bytes != ctx.per_writer_bytes:
            raise RuntimeError(
                f"bounced result describes {described_bytes} bytes, expected {ctx.per_writer_bytes}"
            )
        fragments = [
            (int(ptr), int(ptr) + int(size)) for ptr, size in zip(dst_ptrs, sizes, strict=True)
        ]
        for index, (start, end) in enumerate(fragments):
            if index and start < fragments[index - 1][0]:
                raise RuntimeError(
                    "bounced-result destination fragments are not in canonical address order"
                )
            if index and start < fragments[index - 1][1]:
                raise RuntimeError("bounced-result destination fragments overlap or duplicate")
            if not any(
                valid_start <= start and end <= valid_end
                for valid_start, valid_end in ctx.allowed_destination_ranges
            ):
                raise RuntimeError(
                    f"bounced-result destination [{start}, {end}) is outside the "
                    "receiver-owned KV destination plan"
                )
        expected_plan = ctx.expected_destination_plans.get(peer_rank)
        actual_plan = tuple(
            (int(ptr), int(size)) for ptr, size in zip(dst_ptrs, sizes, strict=True)
        )
        if ctx.expected_destination_plans and actual_plan != expected_plan:
            raise RuntimeError(
                f"bounced-result fragments for rank {peer_rank} do not match the exact "
                "receiver-derived destination plan"
            )
        return dst_ptrs, sizes

    @staticmethod
    def _coalesce_destination_ranges(
        ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        *,
        reject_overlap: bool,
    ) -> tuple[tuple[int, int], ...]:
        coalesced: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if coalesced and start < coalesced[-1][1]:
                if reject_overlap:
                    raise RuntimeError(
                        "bounced-result destination fragments overlap or duplicate across writers"
                    )
                raise RuntimeError("receiver-owned KV destination ranges overlap")
            if coalesced and start == coalesced[-1][1]:
                coalesced[-1] = (coalesced[-1][0], end)
            else:
                coalesced.append((start, end))
        return tuple(coalesced)

    def record_failure(
        self,
        rid_slice: RidSlice,
        peer_rank: int,
        on_done: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """A writer reported failure (it has drained). The region is freed only once every writer has
        reported, not here."""

        def mut(ctx: TransferContext) -> None:
            if on_done is not None:
                ctx.on_done = on_done
            ctx.record_writer_result(peer_rank, succeeded=False)

        self._apply(rid_slice, mut)

    def _scatter_loop(self):
        CUASSERT(cudart.cudaSetDevice(self._device_id))
        while not self._stop.is_set():
            try:
                item = self._scatter_q.get(timeout=_SCATTER_POLL_S)
            except queue.Empty:
                # idle: reclaim quarantine past its grace period, independent of any reserve call
                self._recv_alloc.reclaim_expired()
                continue
            if item is None:
                break  # poison pill from close: wake and exit
            ctx, descs = item
            ok = True
            try:
                # Scatter each writer's fragments from its own source, never one global offset, so a
                # missing or fallback writer cannot shift where the others are read from.
                for src_base, dst_ptrs, sizes in descs:
                    p = Plan(dst_ptrs, dst_ptrs, sizes, int(sizes.sum()))
                    scatter_contiguous(
                        src_base, p.dst_ptrs, p.sizes, p.offsets, stream=self._scatter_stream
                    )
                CUASSERT(cudart.cudaStreamSynchronize(self._scatter_stream))
            except Exception as e:
                # a scatter failure must not kill the worker nor be reported as success
                ok = False
                logger.error(f"[kv-bounce] scatter failed (slot={ctx.slot_id}): {e}")
            # record the outcome and settle; completion fires only after the sync above
            self._apply(ctx.rid_slice, lambda c, ok=ok: c.finish_scatter(ok))

    def close(self) -> None:
        self._stop.set()
        # poison pill: wake the worker now instead of waiting out its poll
        self._scatter_q.put(None)
        if self._scatter_thread.is_alive():
            self._scatter_thread.join(timeout=_CLOSE_JOIN_S)
        if self._scatter_thread.is_alive():
            raise RuntimeError("KV bounce scatter thread did not exit; retaining registered memory")
        for d in self._reg_descs:
            try:
                self._agent.deregister_memory(d)
            except Exception:
                pass
        self._send_alloc.close()
        self._recv_alloc.close()


class NoBounceTransport(BounceTransport):
    """The disabled transport, used when bounce is off so callers need no None checks. Every method
    is a no-op or a negative answer."""

    enabled = False
    _reg_descs = ()

    def build_request(self, write_meta):
        return None

    def release_send(self, slot_id) -> None:
        pass

    def reserve(
        self,
        recv_req,
        num_writers: int = 1,
        *,
        timeout: Optional[float] = _RESERVE_TIMEOUT_S,
        expected_destination_plans: Optional[dict[int, tuple[np.ndarray, np.ndarray]]] = None,
    ) -> bool:
        return False

    def writer_base(self, rid_slice, writer_index: int):
        return None

    def bind_writer(self, rid_slice, peer_rank: int, writer_index: int):
        return None

    def set_completion_callback(self, rid_slice, on_settled) -> None:
        pass

    def is_bounced(self, rid_slice) -> bool:
        return False

    def release_idle_reservation(self, rid_slice) -> None:
        pass

    def orphan_reservation(self, rid_slice) -> None:
        pass

    def confirm_drained(self, rid_slice) -> None:
        pass

    def record_result(
        self, rid_slice, peer_rank, dst_ptrs=None, sizes=None, src_base=None, on_done=None
    ):
        pass

    def record_failure(self, rid_slice, peer_rank, on_done=None) -> None:
        if on_done is not None:
            on_done(False)

    def close(self) -> None:
        pass


def create_bounce(agent, cfg, *, device_id: int, page_table) -> BounceTransport:
    """Build the real transport from the config, or the disabled one when bounce is off, it cannot
    fit, or the fabric allocation races."""
    if cfg is None:
        return NoBounceTransport()
    try:
        destination_pool_layouts = _destination_pool_layouts(page_table)
        transport = VmmBounceTransport.from_config(
            agent,
            cfg,
            device_id=device_id,
            block_bytes_per_group=block_bytes_per_group(page_table),
            destination_pool_layouts=destination_pool_layouts,
            valid_destination_ranges=[],
        )
        return transport if transport is not None else NoBounceTransport()
    except (
        Exception
    ) as e:  # rare race: memory taken between the free-memory query and the allocation
        logger.warning(f"[kv-bounce] disabled (alloc failed: {e}); using in-place path")
        return NoBounceTransport()


def build_send_request(bounce, write_meta, fallback):
    """Build a coalesced bounce write when eligible (release the returned slot afterward), otherwise
    fall back to the per-fragment request."""
    if write_meta.bounce_dst_base is not None:
        built = bounce.build_request(write_meta)
        if built is not None:
            return built
        # The receiver already advertised a leased bounce destination. A
        # sender-side in-place fallback would make a tail-less SUCCESS
        # indistinguishable from a malformed or stale bounced result. Fail the
        # operation explicitly so the receiver retains/retires the reservation
        # through the normal terminal protocol.
        raise RuntimeError("receiver-advertised bounce request could not be built")
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


def decode_result_tail(message, *, tail_index: int = 2):
    """Recover the destination fragments, sizes, and source from the optional trailing frames, or
    nothing if the tail is absent."""
    if len(message) >= tail_index + 3:
        return (
            np.frombuffer(message[tail_index], dtype=np.int64),
            np.frombuffer(message[tail_index + 1], dtype=np.int64),
            int(np.frombuffer(message[tail_index + 2], dtype=np.int64)[0]),
        )
    return None, None, None


def block_bytes_per_group(page_table) -> list:
    """Byte size of one cache block for each leading attention layer group, stopping at the first
    non-attention group."""
    assert page_table is not None
    return [
        sum(slot_bytes for _base, slot_bytes, _slots in group)
        for group in _destination_pool_layouts(page_table)
    ]


def _destination_pool_layouts(page_table) -> list[list[tuple[int, int, int]]]:
    """Return deduplicated physical slot layouts for each leading attention group."""
    from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup
    from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool

    assert page_table is not None
    groups: list[list[tuple[int, int, int]]] = []
    for lg_idx, layer_group in enumerate(page_table.layer_groups):
        if not isinstance(layer_group, AttentionLayerGroup):
            break
        unique_layouts = {
            (
                int(pool.base_address),
                int(pool.slot_bytes),
                int(pool.num_slots),
            )
            for pool_view in layer_group.pool_views
            for pool in [get_physical_pool(page_table, lg_idx, pool_view.pool_idx)]
        }
        groups.append(sorted(unique_layouts))
    return groups

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

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

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
from .config import DEFAULT_MIN_BYTES, SizingContext, fit_within_free
from .core import BounceTransport, Disposition, Settlement, TransferContext
from .gather_scatter import Plan, gather_contiguous, scatter_contiguous

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable

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
        cls, agent, cfg, *, device_id: int, block_bytes_per_group: list[int | None]
    ) -> VmmBounceTransport | None:
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
            min_bytes=cfg.min_bytes,
            min_blocks=cfg.min_blocks,
        )

    def __init__(
        self,
        agent,
        *,
        device_id: int,
        capacity_bytes: int,
        phys_chunk_size: int,
        block_bytes_per_group: list[int | None],
        min_bytes: int = DEFAULT_MIN_BYTES,
        min_blocks: int = 96,
        quarantine_grace_s: float = _QUARANTINE_GRACE_S,
        name: str = "kv_bounce",
    ):
        self._agent = agent
        self._device_id = device_id
        # The byte size of one cache block, listed for each attention layer group.
        self._block_bytes_per_group = list(block_bytes_per_group)
        # Size gates below which bounce is skipped: coalescing only pays off once the transfer is
        # large enough to beat the gather+scatter overhead (see config.DEFAULT_MIN_BYTES for the
        # rationale). min_bytes applies to payloads carrying recurrent (mamba/KDA) state;
        # min_blocks applies to plain-KV payloads (see the gate in reserve()).
        self._min_bytes = min_bytes
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
            logger.warning_once(
                f"[kv-bounce] in-place: no send region space for {total // _MIB}MiB within {timeout}s "
                f"(sender backpressure); falling back",
                key="kv-bounce-send-backpressure",
            )
            return None
        slot_id, src_addr = res
        try:
            self._gather_blocking(src_addr, write_meta, total)
        except Exception:
            self._send_alloc.release(slot_id)  # free the slot if the gather raises
            raise
        return slot_id, src_addr, total

    def build_request(self, write_meta):
        """Gather into a send slot and build the coalesced write, or None on backpressure. The gather
        blocks (and frees the slot on failure) inside _reserve_and_gather."""
        gathered = self._reserve_and_gather(write_meta, timeout=_RESERVE_TIMEOUT_S)
        if gathered is None:  # backpressure: fall back
            return None
        slot_id, src_addr, total = gathered
        return self._make_write(src_addr, write_meta, total), slot_id

    def release_send(self, slot_id) -> None:
        """Release a send region after its write has completed."""
        self._send_alloc.release(slot_id)

    @staticmethod
    def _skip_bounce(reason: str, *, warn_key: str) -> bool:
        """Log why a transfer falls back to the per-fragment path and return False, so the guards
        above stay one line each. Every reason logs at warning once per key: silently skipping
        bounce can be a ~1000x bandwidth cliff (host-staged tcp vs cuda_ipc), so the first skip per
        distinct reason must be visible at the default log level."""
        logger.warning_once(f"[kv-bounce] in-place: {reason}", key=warn_key)
        return False

    def reserve(
        self,
        recv_req,
        num_writers: int = 1,
        *,
        timeout: Optional[float] = _RESERVE_TIMEOUT_S,
        extra_bytes: int = 0,
    ) -> bool:
        """Reserve a region and create its state, recording the address for the senders. Returns
        False to fall back to the per-fragment path. A fan-in splits the region evenly, so the total
        must divide across the writers. ``extra_bytes`` is the non-paged payload the sender appends
        to the same coalesced write (mamba/KDA recurrent state, sized by the receiver via
        ``MambaPolicy.payload_bytes``); the region must cover it or the write would overrun into the
        neighboring slot."""
        total = 0
        for g, block_ids in enumerate(recv_req.block_ids_per_layer_groups):
            if int(block_ids.size) == 0:
                continue
            if g >= len(self._block_bytes_per_group):
                return self._skip_bounce(
                    f"layer group {g} has blocks but no known slot size",
                    warn_key="kv-bounce-unknown-slot-size",
                )
            if not self._block_bytes_per_group[g]:
                # This group holds recurrent state (mamba/KDA), not paged KV blocks;
                # its size is accounted for by extra_bytes below, not per-block math.
                continue
            total += int(block_ids.size) * self._block_bytes_per_group[g]
        if extra_bytes > 0 and num_writers > 1:
            # Each fan-in writer appends its own recurrent-state fragments, whose sizes may differ
            # per writer (PP stages hold different mamba layers), breaking the equal region split.
            return self._skip_bounce(
                f"fan-in across {num_writers} senders with {extra_bytes}B of recurrent state; the "
                f"equal split cannot account for per-writer state fragments",
                warn_key="kv-bounce-mamba-fanin",
            )
        total += int(extra_bytes)
        if total <= 0:
            return self._skip_bounce(
                f"computed transfer size {total} <= 0", warn_key="kv-bounce-nonpositive-size"
            )
        # Which size gate applies depends on the payload. Payloads carrying recurrent (mamba/KDA)
        # state gate on BYTES: the cost the gate guards (falling back to the slow per-fragment
        # path) scales with bytes, and a block count is meaningless for the non-paged state (the
        # Kimi-K3 regression: a 433 MiB transfer of 67 small blocks plus KDA state failed a
        # 96-block gate and fell onto the ~0.4 GB/s host-staged path). Plain-KV payloads keep the
        # original block-count gate so pre-existing bounce deployments (opted in via
        # kv_cache_bounce_size_mb) see no change in which transfers use the arena.
        # TODO(TRTLLM-15194): investigate whether the byte-only gate is
        # safe (or better) for plain-KV payloads too, so this special case can be removed and
        # both payload kinds share one gate.
        nblocks = sum(int(a.size) for a in recv_req.block_ids_per_layer_groups)
        if extra_bytes > 0:
            if total < self._min_bytes:
                return self._skip_bounce(
                    f"{total}B ({nblocks} blocks + recurrent state) < min {self._min_bytes}B "
                    f"(too small; tune TRTLLM_KV_CACHE_BOUNCE_MIN_BYTES)",
                    warn_key="kv-bounce-below-min-bytes",
                )
        elif nblocks < self._min_blocks:
            return self._skip_bounce(
                f"{nblocks} blocks < min {self._min_blocks} (too small; tune "
                f"TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS)",
                warn_key="kv-bounce-below-min-blocks",
            )
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
                for g, block_ids in enumerate(recv_req.block_ids_per_layer_groups)
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
                f"no recv region space for {total // _MIB}MiB within {timeout}s (backpressure)",
                warn_key="kv-bounce-recv-backpressure",
            )
        slot_id, addr = res
        recv_req.bounce_dst_base = addr
        with self._reserved_map_lock:
            ctx = TransferContext(
                rid_slice=(recv_req.unique_rid, recv_req.slice_id),
                slot_id=slot_id,
                base_addr=addr,
                per_writer_bytes=total // num_writers,
                num_writers=num_writers,
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

    def writer_base(self, rid_slice: RidSlice, writer_index: int) -> Optional[int]:
        """Where the given fan-in writer writes in the region."""
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            return None if ctx is None else ctx.writer_base(writer_index)

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
        """Give up on a reservation whose write may still be in flight (cancel/timeout/lost result).
        The write can't be aborted, so quarantine the region (reclaimed later) rather than releasing
        or leaking it. Idempotent; a no-op once the transfer has settled."""
        self._apply(rid_slice, lambda ctx: ctx.mark_orphaned())

    def _apply(self, rid_slice: RidSlice, mutate: Callable[[TransferContext], None]) -> None:
        """Mutate the state under the lock, then do what it asks (scatter or settle) with the lock
        released, never holding it across a CUDA sync, a queue put, or a callback. No-op if the
        region is already gone."""
        scatter: Optional[tuple] = None
        settlement: Optional[Settlement] = None
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if ctx is None:
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
            if on_done is not None:
                ctx.on_done = on_done
            ctx.record_writer_result(
                peer_rank, succeeded=True, src_base=src_base, dst_ptrs=dst_ptrs, sizes=sizes
            )

        self._apply(rid_slice, mut)

    def record_failure(self, rid_slice: RidSlice, peer_rank: int) -> None:
        """A writer reported failure (it has drained). The region is freed only once every writer has
        reported, not here."""
        self._apply(rid_slice, lambda ctx: ctx.record_writer_result(peer_rank, succeeded=False))

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
        extra_bytes: int = 0,
    ) -> bool:
        return False

    def writer_base(self, rid_slice, writer_index: int):
        return None

    def is_bounced(self, rid_slice) -> bool:
        return False

    def release_idle_reservation(self, rid_slice) -> None:
        pass

    def orphan_reservation(self, rid_slice) -> None:
        pass

    def record_result(
        self, rid_slice, peer_rank, dst_ptrs=None, sizes=None, src_base=None, on_done=None
    ):
        pass

    def record_failure(self, rid_slice, peer_rank) -> None:
        pass

    def close(self) -> None:
        pass


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


def block_bytes_per_group(page_table: KVCachePageTable) -> list[int | None]:
    """Return transferred bytes per cache block for each layer group.

    All distinct physical pools exposed by an attention group contribute to its
    transfer size. Multiple logical views of the same physical pool contribute
    only once. Non-attention groups retain a ``None`` placeholder so the result
    remains aligned with receive-request layer-group indices.
    """
    from tensorrt_llm._torch.disaggregation.resource.page import CacheKind
    from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool

    assert page_table is not None
    out: list[int | None] = []
    for lg_idx, lg in enumerate(page_table.layer_groups):
        if lg.kind != CacheKind.PAGED:
            out.append(None)
            continue
        pool_indices = {pool_view.pool_idx for pool_view in lg.pool_views}
        out.append(
            sum(
                int(get_physical_pool(page_table, lg_idx, pool_idx).slot_bytes)
                for pool_idx in pool_indices
            )
        )
    return out

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
"""KV bounce transport: coalesce a request's scattered KV fragments into one contiguous WRITE and
scatter it back on arrival. Holds the send/recv buffers, gather/scatter kernels and scatter worker;
NoBounce is the disabled no-op. The wiring functions live here (deps injected) so this module never
imports transfer.py."""

import queue
import threading
from dataclasses import dataclass, field
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
from tensorrt_llm.runtime.generation import CUASSERT

from .buffer import SlotAllocator
from .config import SizingContext, fit_within_free
from .gather import Plan, gather_contiguous, scatter_contiguous

_Key = tuple  # (unique_rid, slice_id)
_MIB = 1024 * 1024
_SCATTER_POLL_S = 0.5  # scatter worker wakeup to re-check the stop flag
_RESERVE_TIMEOUT_S = 0.2  # max wait for a bounce region before falling back to per-fragment
_CLOSE_JOIN_S = 2.0  # max wait for the scatter thread to drain on close


@dataclass
class _RecvEntry:
    """Bookkeeping for one reserved recv region while its fan-in writers arrive. The base/per_writer
    layout and the acc tuple mirror writer_base() and the result tail (encode/decode_result_tail) —
    change them together."""

    slot_id: int
    base: int
    per_writer: int
    arrived: int = 0
    acc: List[tuple] = field(default_factory=list)  # (src_base, dst_ptrs, sizes) per writer
    on_done: Optional["Callable[[bool], None]"] = (
        None  # completion cb (success flag), fired after scatter
    )


class Transport:
    enabled = True

    @classmethod
    def from_config(
        cls, agent, cfg, *, device_id: int, recv_slot_bytes: List[int]
    ) -> Optional["Transport"]:
        """Build a Transport with send/recv regions sized from cfg.sizing and clamped to free memory; None if
        not even one chunk fits."""
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
            recv_slot_bytes=recv_slot_bytes,
            min_blocks=cfg.min_blocks,
        )

    def __init__(
        self,
        agent,
        *,
        device_id: int,
        capacity_bytes: int,
        phys_chunk_size: int,
        recv_slot_bytes: List[int],
        min_blocks: int = 96,
        name: str = "kv_bounce",
    ):
        self._agent = agent
        self._device_id = device_id
        # Per-group block byte size; recv reservation = sum_g count_g * recv_slot_bytes[g].
        self._recv_slot_bytes = list(recv_slot_bytes)
        # Skip bounce for short transfers — the gather/coordination overhead only pays off for long context.
        self._min_blocks = min_blocks

        # One registered region each for send and recv (multi-rail via the bump cursor).
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
        self._reserved: Dict[_Key, _RecvEntry] = {}
        self._reserved_lock = threading.Lock()

    def _start_scatter_worker(self, name: str) -> None:
        # Gen-side scatter worker: the scatter ends in a blocking stream sync, so run it off the
        # completion handler so that handler keeps draining other transfers instead of stalling.
        self._scatter_q: "queue.Queue" = queue.Queue()
        self._scatter_stream = self._new_stream()
        self._stop = threading.Event()
        self._scatter_thread = threading.Thread(
            target=self._scatter_loop, name=f"{name}-scatter", daemon=True
        )
        self._scatter_thread.start()

    def _new_stream(self):
        return CUASSERT(cudart.cudaStreamCreate())[0]

    def _launch_gather(self, src_addr: int, write_meta, total: int):
        """Launch (async) the gather of frags -> contiguous send slot on the send stream;
        return a CUDA event to _wait_gather on."""
        plan = Plan(write_meta.src_ptrs, write_meta.dst_ptrs, write_meta.sizes, total)
        with self._send_stream_lock:
            gather_contiguous(
                src_addr, plan.src_ptrs, plan.sizes, plan.offsets, stream=self._send_stream
            )
            event = CUASSERT(cudart.cudaEventCreate())[0]
            CUASSERT(cudart.cudaEventRecord(event, self._send_stream))
        return event

    def _wait_gather(self, event) -> None:
        if event is not None:
            CUASSERT(cudart.cudaEventSynchronize(event))
            CUASSERT(cudart.cudaEventDestroy(event))

    def _make_write(self, src_addr: int, write_meta, total: int):
        # One coalesced descriptor: a single src/dst pointer over the whole region.
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

    # ---------- SENDER hooks ----------
    def _reserve_and_gather(self, write_meta, *, timeout):
        """Reserve a send slot and gather src into it; None on send-region backpressure. Eligibility was
        already settled by the receiver's reserve(), so the sender only falls back on backpressure."""
        total = int(write_meta.sizes.sum())
        res = self._send_alloc.reserve(total, timeout=timeout)
        if res is None:
            logger.debug(
                f"[kv-bounce] in-place: no send region space for {total // _MIB}MiB within {timeout}s "
                f"(sender backpressure) -> fall back"
            )
            return None
        slot_id, src_addr = res
        return slot_id, src_addr, total, self._launch_gather(src_addr, write_meta, total)

    def build_request(self, write_meta, thread_idx: int):
        """Gather src into a send slot and build the single-descriptor WRITE; returns (request, send_slot_id) to release
        after, or None on send-region backpressure (fall back to the per-fragment path)."""
        gathered = self._reserve_and_gather(write_meta, timeout=_RESERVE_TIMEOUT_S)
        if gathered is None:  # send backpressure -> fall back
            return None
        slot_id, src_addr, total, event = gathered
        self._wait_gather(event)
        return self._make_write(src_addr, write_meta, total), slot_id

    def release_send(self, slot_id) -> None:
        """Release a send region after its WRITE has completed."""
        self._send_alloc.release(slot_id)

    # ---------- RECEIVER hooks ----------
    @staticmethod
    def _skip_bounce(reason: str, *, warn_key: Optional[str] = None) -> bool:
        """Log why a transfer falls back to the per-fragment path (warn-once if warn_key, else debug)
        and return False, so reserve()'s guards stay one line each."""
        msg = f"[kv-bounce] in-place: {reason}"
        logger.warning_once(msg, key=warn_key) if warn_key else logger.debug(msg)
        return False

    def reserve(
        self, recv_req, num_writers: int = 1, *, timeout: Optional[float] = _RESERVE_TIMEOUT_S
    ) -> bool:
        """Reserve a contiguous region for this incoming transfer and record its address on recv_req for
        the sender to write into. Returns True if reserved; False means the caller uses the normal
        per-fragment path instead.

        With TP fan-in (num_writers>1) several senders write into this one region, each into an equal
        slice placed one after another, so the total must divide evenly by num_writers; if it doesn't,
        return False."""
        nblocks = sum(int(a.size) for a in recv_req.block_ids_per_layer_groups)
        if nblocks < self._min_blocks:
            return self._skip_bounce(f"{nblocks} blocks < min {self._min_blocks} (too small)")
        total = 0
        for g, block_ids in enumerate(recv_req.block_ids_per_layer_groups):
            if g >= len(self._recv_slot_bytes):
                return self._skip_bounce(f"layer group {g} has no known slot size (e.g. mamba)")
            total += int(block_ids.size) * self._recv_slot_bytes[g]
        if total <= 0:
            return self._skip_bounce(f"computed transfer size {total} <= 0")
        if num_writers > 1 and total % num_writers != 0:
            return self._skip_bounce(
                f"fan-in {total}B across {num_writers} senders is not an even split "
                f"({total % num_writers}B remainder); head-mismatch explosion NOT mitigated",
                warn_key="kv-bounce-uneven-fanin",
            )
        if (
            total > self._recv_alloc.capacity
        ):  # never fits -> permanent, distinct from transient backpressure
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
        with self._reserved_lock:
            self._reserved[(recv_req.unique_rid, recv_req.slice_id)] = _RecvEntry(
                slot_id=slot_id, base=addr, per_writer=total // num_writers
            )
        return True

    def writer_base(self, key: _Key, writer_index: int) -> Optional[int]:
        """Per-sender destination base for TP fan-in: writer i targets region_base + i*per_writer."""
        with self._reserved_lock:
            entry = self._reserved.get(key)
            return None if entry is None else entry.base + writer_index * entry.per_writer

    def is_bounced(self, key: _Key) -> bool:
        with self._reserved_lock:
            return key in self._reserved

    def release_reservation(self, key: _Key) -> None:
        """Drop a reserved recv region whose transfer FAILED or was cancelled and free its slot.
        The SUCCESS path releases via accumulate_and_scatter instead; idempotent (no-op once the
        key has completed/released), so failure and cancel flows can both call it safely."""
        with self._reserved_lock:
            entry = self._reserved.pop(key, None)
        if entry is not None:
            self._recv_alloc.release(entry.slot_id)

    def accumulate_and_scatter(
        self,
        key: _Key,
        num_writers: int,
        dst_ptrs=None,
        sizes=None,
        src_base=None,
        on_done: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Record each writer's fragments; once all arrive, scatter the region back into the real KV
        blocks and release it. Sorting by src base makes the back-to-back fan-in layout one contiguous
        scatter, identical to the single-writer path. on_done (set only on the completing message) is
        the deferred task completion: it fires AFTER the scatter lands (on the worker), or inline here
        when there is nothing to scatter, so the gen never observes completion before the KV is in place."""
        with self._reserved_lock:
            entry = self._reserved.get(key)
            if entry is None:
                # Key already gone (should not happen for a live completing message): don't silently
                # drop the sole completion authority — surface it as a failure rather than hang.
                if on_done is not None:
                    on_done(False)
                return
            entry.arrived += 1
            if on_done is not None:
                entry.on_done = on_done
            if dst_ptrs is not None and int(dst_ptrs.size) > 0:
                entry.acc.append(
                    (src_base if src_base is not None else entry.base, dst_ptrs, sizes)
                )
            if entry.arrived < max(num_writers, 1):
                return
            self._reserved.pop(key)
            slot_id, addr, acc, on_done = entry.slot_id, entry.base, entry.acc, entry.on_done
        if not acc:
            # Bounced SUCCESS with no scatter tail: nothing to copy, but the task must still complete.
            self._recv_alloc.release(slot_id)
            if on_done is not None:
                on_done(True)
            return
        acc.sort(key=lambda t: t[0])
        all_dst_ptrs = acc[0][1] if len(acc) == 1 else np.concatenate([t[1] for t in acc])
        all_sizes = acc[0][2] if len(acc) == 1 else np.concatenate([t[2] for t in acc])
        self._scatter_q.put(
            (
                slot_id,
                addr,
                Plan(all_dst_ptrs, all_dst_ptrs, all_sizes, int(all_sizes.sum())),
                on_done,
            )
        )

    def _scatter_loop(self):
        CUASSERT(cudart.cudaSetDevice(self._device_id))
        while not self._stop.is_set():
            try:
                item = self._scatter_q.get(timeout=_SCATTER_POLL_S)
            except queue.Empty:
                continue  # idle poll — loop back to re-check the stop flag
            if item is None:
                break  # poison pill from close(): wake immediately and exit
            slot_id, addr, plan, on_done = item
            ok = True
            try:
                scatter_contiguous(
                    addr,
                    plan.dst_ptrs,
                    plan.sizes,
                    plan.offsets,
                    stream=self._scatter_stream,
                )
                CUASSERT(cudart.cudaStreamSynchronize(self._scatter_stream))
            except Exception as e:
                # Per-item guard: a scatter failure must NOT kill the single worker (that would
                # strand every later task) nor be reported as success. Mark failed and carry on.
                ok = False
                logger.error(f"[kv-bounce] scatter failed (slot={slot_id}): {e}")
            finally:
                self._recv_alloc.release(slot_id)
            # Completion fires only after the scatter has landed (sync done) and the slot is freed;
            # ok=False routes to task.fail so the gen never observes success for un-scattered KV.
            if on_done is not None:
                try:
                    on_done(ok)
                except Exception as e:  # never let the callback kill the worker
                    logger.error(f"[kv-bounce] completion callback failed (slot={slot_id}): {e}")

    def close(self) -> None:
        self._stop.set()
        self._scatter_q.put(
            None
        )  # poison pill: unblock the worker's get() now, don't wait out the poll
        if self._scatter_thread.is_alive():
            self._scatter_thread.join(timeout=_CLOSE_JOIN_S)
        for d in self._reg_descs:
            try:
                self._agent.deregister_memory(d)
            except Exception:
                pass
        self._send_alloc.close()
        self._recv_alloc.close()


class NoBounce:
    """No-op stand-in used when bounce is disabled, so transfer.py never needs `if bounce is not None` guards."""

    enabled = False
    _reg_descs = ()

    def build_request(self, write_meta, thread_idx: int = 0):
        return None

    def release_send(self, slot_id) -> None:
        pass

    def reserve(
        self, recv_req, num_writers: int = 1, *, timeout: Optional[float] = _RESERVE_TIMEOUT_S
    ) -> bool:
        return False

    def writer_base(self, key, writer_index: int):
        return None

    def is_bounced(self, key) -> bool:
        return False

    def release_reservation(self, key) -> None:
        pass

    def accumulate_and_scatter(
        self, key, num_writers: int, dst_ptrs=None, sizes=None, src_base=None, on_done=None
    ) -> None:
        pass

    def close(self) -> None:
        pass


def create_bounce(agent, cfg, *, device_id: int, page_table):
    """Build a Transport from cfg, or a NoBounce when cfg is None / it can't fit / alloc races."""
    if cfg is None:
        return NoBounce()
    try:
        transport = Transport.from_config(
            agent, cfg, device_id=device_id, recv_slot_bytes=recv_slot_bytes(page_table)
        )
        return transport if transport is not None else NoBounce()
    except Exception as e:  # rare race: mem grabbed between query and alloc
        logger.warning(f"[kv-bounce] disabled (alloc failed: {e}); using in-place path")
        return NoBounce()


def build_send_request(bounce, write_meta, fallback):
    """Return (request, send_slot_id): a coalesced bounce WRITE when eligible (release the slot_id
    after), else fallback() with send_slot_id None."""
    if write_meta.bounce_dst_base is not None:
        built = bounce.build_request(write_meta, 0)
        if built is not None:
            return built
    return fallback(), None


def scatter_write_result(
    bounce, key, num_writers: int, dst_ptrs, sizes, src_base=None, on_done=None
) -> None:
    """Scatter a bounced slice back into its KV blocks once all writers arrive; no-op for non-bounced
    transfers. on_done (the task-completion callback, set only on the completing message) is invoked
    exactly once with a success flag: the bounced path defers it until after the scatter lands (or
    inline on the empty-acc case); the non-bounced / NoBounce path fires it inline here, exactly as
    completion happened before (the in-place WRITE already landed the KV)."""
    if bounce.is_bounced(key):
        bounce.accumulate_and_scatter(key, num_writers, dst_ptrs, sizes, src_base, on_done)
    elif on_done is not None:
        on_done(True)


def encode_result_tail(write_meta) -> list:
    """Binary tail (dst fragment table + the src base this writer wrote to) appended to a bounced
    KV_AGENT_RESULT so the receiver can scatter — src base lets it order TP fan-in writers."""
    sb = write_meta.bounce_dst_base if write_meta.bounce_dst_base is not None else 0
    return [
        write_meta.dst_ptrs.tobytes(),
        write_meta.sizes.tobytes(),
        np.array([sb], dtype=np.int64).tobytes(),
    ]


def decode_result_tail(message):
    """Inverse of encode_result_tail: (dst_ptrs, sizes, src_base) from the optional binary tail at
    message[2:] (after [0]=msg type, [1]=packed prefix), or (None, None, None) if absent."""
    if len(message) >= 5:
        return (
            np.frombuffer(message[2], dtype=np.int64),
            np.frombuffer(message[3], dtype=np.int64),
            int(np.frombuffer(message[4], dtype=np.int64)[0]),
        )
    return None, None, None


def recv_slot_bytes(page_table) -> list:
    """Per-layer-group block byte size (primary KV pool slot_bytes) for the leading attention
    groups, stopping at the first non-attention group."""
    from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup
    from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool

    assert page_table is not None
    out: list = []
    for lg_idx, lg in enumerate(page_table.layer_groups):
        if not isinstance(lg, AttentionLayerGroup):
            break
        out.append(int(get_physical_pool(page_table, lg_idx, 0).slot_bytes))
    return out

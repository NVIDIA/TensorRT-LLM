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

import os
import queue
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
from .config import (
    _HEADROOM_FRACTION,
    DEFAULT_MIN_BYTES,
    DIRECTION_BOTH,
    DIRECTION_RECV,
    DIRECTION_SEND,
    SEND_STREAM_BLOCKING,
    SEND_STREAM_NONBLOCKING,
    SEND_STREAM_NONBLOCKING_HIGH_PRIORITY,
    SizingContext,
    _round_up,
    bounce_jit_from_env,
    direction_from_env,
    fit_within_free,
    max_free_fraction_from_env,
    recv_scatter_drain_from_env,
    recv_scatter_tuned_from_env,
    send_stream_mode_from_env,
)
from .core import (
    BounceTransport,
    Disposition,
    JitAction,
    JitGrantCheck,
    JitTransferContext,
    JitVerdict,
    Settlement,
    TransferContext,
)
from .gather_scatter import Plan, gather_contiguous, scatter_contiguous

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable

RidSlice = tuple  # the request id and slice id a region serves
_MIB = 1024 * 1024
_SCATTER_POLL_S = 0.5  # how often the scatter worker wakes to re-check the stop flag and reclaim
_RESERVE_TIMEOUT_S = 0.2  # max wait for a bounce region before falling back to per-fragment
_CLOSE_JOIN_S = 2.0  # max wait for the scatter thread to drain on close
_QUARANTINE_GRACE_S = 60.0  # how long an orphaned region is held out of reuse
_WRITER_ALIGN_BYTES = 256  # start alignment of each writer's sub-region in an exact fan-in layout
_JIT_STATS_PERIOD_S = 60.0  # at most one just-in-time grant counter summary per period
_DRAIN_STATS_PERIOD_S = 60.0  # at most one batched receive scatter counter summary per period
_DRAIN_WARN_PERIOD_S = 60.0  # at most one "batched scatter failed" warning per period
# consecutive drains whose batch failed while every item alone succeeded, before the worker gives up
# on batching and goes back to one scatter and one sync per item for the rest of the run
_DRAIN_BATCH_ONLY_FAILURES_MAX = 3

# Opt-in (default off): create the gather/scatter streams with cudaStreamNonBlocking.
# cudaStreamCreate makes a *blocking* stream, so every gather/scatter implicitly waits for
# all work already queued on the legacy default stream. The executor makes that stream wait
# for the whole in-flight forward (``current_stream().wait_stream(execution_stream)``), so a
# send handed off while the next step is queued waits for that step to finish on the GPU
# (~one prefill step, ~150 ms median on TEP4 8k chunks) once nothing blocks the host during
# the forward (e.g. TRTLLM_DFLASH_CTX_NO_HOST_SYNC=1). Ordering does not need that wait: the
# sender hands off a request only after the host synchronized on the sampler event of the
# step that wrote its KV, and the scatter syncs its own stream before it reports completion.
_NONBLOCKING_STREAMS_ENV = "TRTLLM_KV_BOUNCE_NONBLOCKING_STREAMS"


def nonblocking_streams_enabled() -> bool:
    """True when TRTLLM_KV_BOUNCE_NONBLOCKING_STREAMS=1 (default off)."""
    return os.environ.get(_NONBLOCKING_STREAMS_ENV, "0").strip() == "1"


class VmmBounceTransport(BounceTransport):
    """The real transport: gather the request's cache into one fabric region, issue a single coalesced
    multi-rail write, and scatter it back on the receiver."""

    enabled = True
    #: Whether the just-in-time grant is on (TRTLLM_KV_BOUNCE_JIT); set from the env at init.
    _jit = False

    @classmethod
    def from_config(
        cls, agent, cfg, *, device_id: int, block_bytes_per_group: list[int | None]
    ) -> VmmBounceTransport | None:
        """Build a transport sized from the config and clamped to free memory, or None if not even one
        chunk fits."""
        chunk = cfg.chunk_mb * _MIB
        # Opt-in (TRTLLM_KV_BOUNCE_DIRECTION): a role-scoped transport has one region, which gets
        # the budget both regions share by default.
        direction = direction_from_env()
        num_regions = 2 if direction == DIRECTION_BOTH else 1
        # Opt-in (TRTLLM_KV_BOUNCE_MAX_FREE_FRACTION): the share of free memory the regions may take.
        max_free_fraction = max_free_fraction_from_env()
        free_b, total_b = CUASSERT(cudart.cudaMemGetInfo())
        want_capacity = cfg.sizing.resolve(
            SizingContext(
                free_bytes=free_b, total_bytes=total_b, chunk_bytes=chunk, device_id=device_id
            )
        )
        capacity_bytes = fit_within_free(
            want_capacity,
            free_bytes=free_b,
            chunk_bytes=chunk,
            max_free_fraction=max_free_fraction,
            num_regions=num_regions,
        )
        if max_free_fraction != _HEADROOM_FRACTION:
            logger.info(
                f"[kv-bounce] regions may take {max_free_fraction} of {free_b // _MIB}MiB free "
                f"({num_regions} region(s))"
            )
        if capacity_bytes is None:
            logger.warning(f"[kv-bounce] disabled: only {free_b // _MIB}MiB free")
            return None
        if capacity_bytes != want_capacity:
            logger.warning(
                f"[kv-bounce] each region clamped to {capacity_bytes // _MIB}MiB "
                f"({num_regions}x total) to fit {free_b // _MIB}MiB free"
            )
        return cls(
            agent,
            device_id=device_id,
            capacity_bytes=capacity_bytes,
            phys_chunk_size=chunk,
            block_bytes_per_group=block_bytes_per_group,
            min_bytes=cfg.min_bytes,
            min_blocks=cfg.min_blocks,
            direction=direction,
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
        direction: str = DIRECTION_BOTH,
    ):
        if direction not in (DIRECTION_BOTH, DIRECTION_RECV, DIRECTION_SEND):
            raise ValueError(f"unsupported bounce direction {direction!r}")
        self._agent = agent
        self._direction = direction
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

        # Opt-in (TRTLLM_KV_BOUNCE_SEND_STREAM_NONBLOCKING): 0 keeps the blocking stream. Parsed
        # before any region is allocated or registered, so a bad value leaks nothing.
        self._send_stream_mode = send_stream_mode_from_env()
        # Opt-in (TRTLLM_KV_BOUNCE_RECV_SCATTER_DRAIN): off keeps one scatter and one stream sync
        # per queued item. Parsed here too, before any region is allocated.
        self._scatter_drain = recv_scatter_drain_from_env()
        # Opt-in (TRTLLM_KV_BOUNCE_RECV_SCATTER_TUNED): off keeps the batched copy's default block
        # for the receive scatter. The send-side gather never takes the small block.
        self._scatter_tuned = recv_scatter_tuned_from_env()
        # Opt-in (TRTLLM_KV_BOUNCE_JIT): off registers no just-in-time context, so the result
        # paths skip the extra lookup under the reservation lock.
        self._jit = bounce_jit_from_env()

        # One registered region each for sending and receiving, or only the one the role uses
        # (TRTLLM_KV_BOUNCE_DIRECTION).
        self._send_alloc: Optional[SlotAllocator] = None
        self._recv_alloc: Optional[SlotAllocator] = None
        if direction != DIRECTION_RECV:
            self._send_alloc = SlotAllocator(capacity_bytes, phys_chunk_size, name=f"{name}_send")
        if direction != DIRECTION_SEND:
            self._recv_alloc = SlotAllocator(capacity_bytes, phys_chunk_size, name=f"{name}_recv")
        self._reg_descs = [
            a.reg_descs() for a in (self._send_alloc, self._recv_alloc) if a is not None
        ]
        for d in self._reg_descs:
            self._agent.register_memory(d)

        self._send_stream = None
        self._send_stream_lock = threading.Lock()
        if self._send_alloc is not None:
            self._send_stream = self._new_stream(self._send_stream_mode)
            if self._send_stream_mode != SEND_STREAM_BLOCKING:
                logger.info(f"[kv-bounce] send stream mode={self._send_stream_mode}")

        self._init_recv_state()
        self._scatter_thread: Optional[threading.Thread] = None
        if self._recv_alloc is not None:
            self._start_scatter_worker(name)
            if self._scatter_drain:
                logger.info(
                    "[kv-bounce] recv scatter drain on: one batched scatter and one sync per drain"
                )
            if self._scatter_tuned:
                logger.info(
                    "[kv-bounce] recv scatter tuned on: batched copy block 8 KiB, 2 warps "
                    "(receive scatter only)"
                )

        nonblocking = (
            ", non-blocking gather/scatter streams" if nonblocking_streams_enabled() else ""
        )
        if direction == DIRECTION_BOTH:
            logger.info(
                f"[kv-bounce] Transport: send+recv regions of "
                f"{self._send_alloc.capacity / _MIB:.1f}MiB each" + nonblocking
            )
        else:
            logger.info(
                f"[kv-bounce] Transport: direction={direction}, "
                f"send={self._region_mib(self._send_alloc)}, "
                f"recv={self._region_mib(self._recv_alloc)}" + nonblocking
            )

    @staticmethod
    def _region_mib(alloc: Optional[SlotAllocator]) -> str:
        return "none" if alloc is None else f"{alloc.capacity / _MIB:.1f}MiB"

    @property
    def can_send(self) -> bool:
        """Whether this transport has a send region (it can gather a coalesced write)."""
        return self._send_alloc is not None

    @property
    def can_recv(self) -> bool:
        """Whether this transport has a receive region (it can reserve and scatter)."""
        return self._recv_alloc is not None

    def _init_recv_state(self) -> None:
        # Live per-transfer state, guarded by a leaf lock: mutate and decide under it, then release
        # it before any CUDA sync, allocator call, or callback.
        self._reserved_map: Dict[RidSlice, Union[TransferContext, JitTransferContext]] = {}
        self._reserved_map_lock = threading.Lock()
        # Just-in-time grant counters (TRTLLM_KV_BOUNCE_JIT), guarded by the same lock.
        self._jit_stats = {
            "grants": 0,
            "retries": 0,
            "no": 0,
            "scatters": 0,
            "inflight_bytes": 0,
            "inflight_bytes_max": 0,
        }
        self._jit_stats_next_log = 0.0
        # Batched receive scatter counters (TRTLLM_KV_BOUNCE_RECV_SCATTER_DRAIN). Only the scatter
        # worker touches them, so they need no lock. "syncs" counts every scatter-and-sync,
        # per-item retries included; "retried" counts the items scattered again alone.
        self._drain_stats = {"syncs": 0, "items": 0, "max_items": 0, "retried": 0}
        self._drain_stats_next_log = 0.0
        self._drain_warn_next_log = 0.0
        self._drain_batch_only_failures = 0

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

    def _new_stream(self, mode: int = SEND_STREAM_BLOCKING):
        """Create a stream. Mode 0 is the blocking stream every caller used before
        (non-blocking under TRTLLM_KV_BOUNCE_NONBLOCKING_STREAMS=1, for the send and the
        scatter stream alike); modes 1 and 2 (send stream only) skip the implicit wait on the
        legacy default stream, mode 2 also at the device's greatest priority.

        Dropping that wait is safe for the gather: a send starts only after the request's last
        context chunk is sampled, which on a PP stage causally follows this stage's KV writes on
        the forward stream, and the blocks stay pinned until the write completes. The per-fragment
        path (a copy-engine write with no stream ordering) relies on the same invariant. The
        scatter stream stays mode 0."""
        if mode == SEND_STREAM_BLOCKING:
            if nonblocking_streams_enabled():
                return CUASSERT(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))[0]
            return CUASSERT(cudart.cudaStreamCreate())[0]
        if mode == SEND_STREAM_NONBLOCKING:
            return CUASSERT(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))[0]
        if mode == SEND_STREAM_NONBLOCKING_HIGH_PRIORITY:
            # the greatest priority is the numerically lowest value of the range
            _least, greatest = CUASSERT(cudart.cudaDeviceGetStreamPriorityRange())
            return CUASSERT(
                cudart.cudaStreamCreateWithPriority(cudart.cudaStreamNonBlocking, greatest)
            )[0]
        raise ValueError(f"unsupported bounce stream mode {mode}")

    def _gather_blocking(self, src_addr: int, write_meta, total: int) -> None:
        """Gather the scattered fragments into the send region and block until done. The whole gather
        runs under the stream lock so a second sender thread can't overwrite the shared staging buffer
        mid-copy and corrupt this region; only the fast gather serializes, the writes stay parallel."""
        plan = Plan(write_meta.src_ptrs, write_meta.dst_ptrs, write_meta.sizes, total)
        with self._send_stream_lock:
            gather_contiguous(
                src_addr,
                plan.src_ptrs,
                plan.sizes,
                plan.offsets,
                stream=self._send_stream,
                # a non-blocking stream must own its metadata block in the caching allocator
                alloc_on_stream=(self._send_stream_mode != SEND_STREAM_BLOCKING),
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
        if self._send_alloc is None:
            logger.warning_once(
                "[kv-bounce] in-place: this transport has no send region "
                "(TRTLLM_KV_BOUNCE_DIRECTION=recv); falling back",
                key="kv-bounce-no-send-region",
            )
            return None
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
        if slot_id is None or self._send_alloc is None:
            return
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
        writers_split_layers: bool = True,
        writer_block_bytes: Optional[Sequence[Sequence[int]]] = None,
    ) -> bool:
        """Reserve a region and create its state, recording the address for the senders. Returns
        False to fall back to the per-fragment path. A fan-in splits the region evenly, so the total
        must divide across the writers. ``extra_bytes`` is the non-paged payload the sender appends
        to the same coalesced write (mamba/KDA recurrent state, sized by the receiver via
        ``mamba_receiver_payload_bytes``); the region must cover it or the write would overrun into the
        neighboring slot. ``writers_split_layers`` is False when every fan-in writer sends a head
        slice of every layer (TP fan-in), True when the writers hold different layers (PP fan-in).
        ``writer_block_bytes`` replaces the equal split with an exact layout: writer i gets
        ``sum_g valid_blocks_g * writer_block_bytes[i][g]`` bytes at the prefix sum of the writers
        before it, so an uneven PP layer split and groups with different slot sizes both fit."""
        if self._recv_alloc is None:
            return self._skip_bounce(
                "this transport has no receive region (TRTLLM_KV_BOUNCE_DIRECTION=send)",
                warn_key="kv-bounce-no-recv-region",
            )
        plan = self._plan_region(
            recv_req, num_writers, extra_bytes, writers_split_layers, writer_block_bytes
        )
        if not plan:
            return False
        total, exact_sizes, writer_offsets, nblocks = plan
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
                writer_offsets=None if exact_sizes is None else tuple(writer_offsets),
                writer_sizes=None if exact_sizes is None else tuple(exact_sizes),
            )
            self._reserved_map[ctx.rid_slice] = ctx  # inactive until the first writer reports
        if exact_sizes is not None:
            logger.info_once(
                f"[kv-bounce] coalesced {nblocks} blocks / {total // _MIB}MiB into one region "
                f"across {num_writers} writer(s) with exact per-writer sizes "
                f"{[round(b / _MIB, 1) for b in exact_sizes]}MiB (uneven PP fan-in)",
                key="kv-bounce-coalesced-pp-uneven",
            )
            return True
        # Positive marker: all fall-back guards above passed, so this transfer provably takes the
        # coalesced-bounce WRITE path. Logged once (per process) so an e2e test can assert that
        # bounce actually engaged instead of silently falling back to the per-fragment path.
        logger.info_once(
            f"[kv-bounce] coalesced {nblocks} blocks / {total // _MIB}MiB into one region "
            f"across {num_writers} writer(s)",
            key="kv-bounce-coalesced",
        )
        return True

    def _plan_region(
        self,
        recv_req,
        num_writers: int,
        extra_bytes: int,
        writers_split_layers: bool,
        writer_block_bytes: Optional[Sequence[Sequence[int]]],
    ):
        """Size a receive region and apply every eligibility gate except the capacity check, which
        differs between the dispatch-time reserve (the whole region) and the just-in-time grant
        (each writer). Returns ``(total, exact_sizes, writer_offsets, nblocks)``, or False (logged)
        to fall back to the per-fragment path. ``exact_sizes`` and ``writer_offsets`` are None
        unless an exact fan-in layout applies."""
        total = 0
        has_state_group = False
        # Block tables are positional with -1 holes (see Chunk); only the
        # slots the receiver actually accepts are written into the region.
        valid_counts = [
            int((block_ids >= 0).sum()) for block_ids in recv_req.block_ids_per_layer_groups
        ]
        for g, n_valid in enumerate(valid_counts):
            if n_valid == 0:
                continue
            if g >= len(self._block_bytes_per_group):
                return self._skip_bounce(
                    f"layer group {g} has blocks but no known slot size",
                    warn_key="kv-bounce-unknown-slot-size",
                )
            if not self._block_bytes_per_group[g]:
                # This group holds recurrent state (mamba/KDA), not paged KV blocks;
                # its size is accounted for by extra_bytes below, not per-block math.
                has_state_group = True
                continue
            total += n_valid * self._block_bytes_per_group[g]
        if has_state_group and num_writers > 1:
            # Fan-in with recurrent-state groups is unsafe: each writer may
            # append its own state fragments (different PP stages hold different
            # mamba layers), and the representative extra_bytes (computed from
            # one sender's page table) may not reflect all participating senders.
            # Even when extra_bytes == 0 (e.g. the representative rank has no
            # mamba overlap), another PP sender may still contribute state bytes
            # that were not reserved in the bounce region.
            return self._skip_bounce(
                f"fan-in across {num_writers} senders with STATE groups present; the "
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
        nblocks = sum(valid_counts)
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
        exact_sizes = None
        writer_offsets = None
        if writer_block_bytes is not None and num_writers > 1:
            exact_sizes = self._exact_writer_sizes(
                writer_block_bytes, valid_counts, num_writers, extra_bytes
            )
            if exact_sizes is None:
                return False
            # Writer i starts where the writers before it end, each start rounded up so every
            # sub-region keeps the copy kernels' alignment.
            writer_offsets = []
            total = 0
            for size in exact_sizes:
                start = _round_up(total, _WRITER_ALIGN_BYTES) if size else total
                writer_offsets.append(start)
                total = start + size
        if exact_sizes is None and num_writers > 1 and total % num_writers != 0:
            return self._skip_bounce(
                f"fan-in {total}B across {num_writers} senders is not an even split "
                f"({total % num_writers}B remainder); head-mismatch explosion NOT mitigated",
                warn_key="kv-bounce-uneven-fanin",
            )
        if exact_sizes is None and num_writers > 1 and writers_split_layers:
            # Fan-in gives each writer an equal share of the region, which only matches where it
            # writes when all writers send the same bytes. A TP fan-in writer sends 1/num_writers
            # of every block of every group, so its share is exact for any per-group block sizes.
            # PP fan-in writers send different layers, and an equal layer count gives equal bytes
            # only when the per-block sizes match, so require that here, else fall back.
            # Exclude None entries (STATE groups) — they are already guarded above.
            present_slot_bytes = {
                self._block_bytes_per_group[g]
                for g, n_valid in enumerate(valid_counts)
                if n_valid > 0 and self._block_bytes_per_group[g]
            }
            if len(present_slot_bytes) > 1:
                return self._skip_bounce(
                    f"fan-in across {num_writers} senders with non-uniform layer-group slot bytes "
                    f"{sorted(present_slot_bytes)}; the equal split would overrun a sub-region",
                    warn_key="kv-bounce-heterogeneous-fanin",
                )
        return total, exact_sizes, writer_offsets, nblocks

    def _exact_writer_sizes(
        self,
        writer_block_bytes: Sequence[Sequence[int]],
        valid_counts: List[int],
        num_writers: int,
        extra_bytes: int,
    ) -> Optional[List[int]]:
        """Each writer's exact bytes for an exact fan-in layout, or None (logged) to fall back."""
        if len(writer_block_bytes) != num_writers or extra_bytes:
            self._skip_bounce(
                f"exact fan-in layout has {len(writer_block_bytes)} writer sizes for "
                f"{num_writers} writers (extra_bytes={extra_bytes})",
                warn_key="kv-bounce-exact-fanin-shape",
            )
            return None
        sizes = [0] * num_writers
        for g, n_valid in enumerate(valid_counts):
            if n_valid == 0 or not self._block_bytes_per_group[g]:
                continue  # unknown / STATE groups were rejected above
            per_writer = [int(wb[g]) if g < len(wb) else 0 for wb in writer_block_bytes]
            if any(b < 0 for b in per_writer) or sum(per_writer) > self._block_bytes_per_group[g]:
                self._skip_bounce(
                    f"exact fan-in sizes {per_writer} for layer group {g} exceed its "
                    f"{self._block_bytes_per_group[g]}B slot",
                    warn_key="kv-bounce-exact-fanin-overflow",
                )
                return None
            for i, b in enumerate(per_writer):
                sizes[i] += n_valid * b
        if sum(sizes) <= 0:
            self._skip_bounce(
                "exact fan-in layout writes no bytes", warn_key="kv-bounce-exact-fanin-empty"
            )
            return None
        return sizes

    def writer_base(self, rid_slice: RidSlice, writer_index: int) -> Optional[int]:
        """Where the given fan-in writer writes in the region."""
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if ctx is None or isinstance(ctx, JitTransferContext):
                return None  # a just-in-time writer learns its base from its grant
            return ctx.writer_base(writer_index)

    def writer_capacity(self, rid_slice: RidSlice, writer_index: int) -> Optional[int]:
        """The size of the writer's sub-region under an exact layout, else None."""
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if ctx is None or isinstance(ctx, JitTransferContext):
                return None
            return ctx.writer_capacity(writer_index)

    def is_bounced(self, rid_slice: RidSlice) -> bool:
        with self._reserved_map_lock:
            return rid_slice in self._reserved_map

    def release_idle_reservation(self, rid_slice: RidSlice) -> None:
        """Immediately release a reservation cancelled before any address went out; no write can be
        in flight. Idempotent. Drained transfers finalize through the result path instead."""
        if self._is_jit(rid_slice):
            self._release_idle_jit(rid_slice)
            return
        with self._reserved_map_lock:
            ctx = self._reserved_map.pop(rid_slice, None)
        if ctx is not None:
            self._recv_alloc.release(ctx.slot_id)

    def orphan_reservation(self, rid_slice: RidSlice) -> None:
        """Give up on a reservation whose write may still be in flight (cancel/timeout/lost result).
        The write can't be aborted, so quarantine the region (reclaimed later) rather than releasing
        or leaking it. Idempotent; a no-op once the transfer has settled."""
        if self._is_jit(rid_slice):
            self._apply_jit(rid_slice, lambda ctx: ctx.mark_orphaned())
            return
        self._apply(rid_slice, lambda ctx: ctx.mark_orphaned())

    def abort_publication(self, rid_slice: RidSlice, published_writers: set[int]) -> None:
        """Retain a failed fan-out until every successfully published writer drains."""
        if self._is_jit(rid_slice):
            self._apply_jit(rid_slice, lambda ctx: ctx.abort_publication(published_writers))
            return
        self._apply(
            rid_slice,
            lambda ctx: ctx.abort_publication(published_writers),
        )

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
        if settlement.slot_id is None:
            pass  # a just-in-time context has already released its per-writer slots
        elif settlement.disposition is Disposition.QUARANTINE:
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
        if self._is_jit(rid_slice):

            def mut_jit(ctx: JitTransferContext) -> List[JitAction]:
                if on_done is not None:
                    ctx.on_done = on_done
                return ctx.record_report(
                    peer_rank, True, src_base=src_base, dst_ptrs=dst_ptrs, sizes=sizes
                )

            self._apply_jit(rid_slice, mut_jit)
            return

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
        if self._is_jit(rid_slice):
            self._apply_jit(rid_slice, lambda ctx: ctx.record_report(peer_rank, False))
            return
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
            if self._scatter_drain:
                if self._scatter_drained(item):
                    break  # the poison pill was queued behind the drained items
                continue
            if len(item) == 3:
                self._scatter_jit_writer(*item)
                continue
            ctx, descs = item
            ok = True
            try:
                self._scatter_blocking(descs)
            except Exception as e:
                # a scatter failure must not kill the worker nor be reported as success
                ok = False
                logger.error(f"[kv-bounce] scatter failed (slot={ctx.slot_id}): {e}")
            # record the outcome and settle; completion fires only after the sync above
            self._apply(ctx.rid_slice, lambda c, ok=ok: c.finish_scatter(ok))

    def _scatter_blocking(self, descs: List[tuple]) -> None:
        """Scatter each writer's fragments from its own source, never one global offset, so a
        missing or fallback writer cannot shift where the others are read from. All writers go into
        ONE batched copy: the copy helper reuses one pinned metadata buffer per stream and needs a
        sync between calls, and one call costs one host round trip instead of one per writer."""
        srcs, dsts, frag_sizes = [], [], []
        for src_base, dst_ptrs, sizes in descs:
            p = Plan(dst_ptrs, dst_ptrs, sizes, int(sizes.sum()))
            srcs.append(np.int64(src_base) + p.offsets)
            dsts.append(p.dst_ptrs)
            frag_sizes.append(p.sizes)
        # the small block is passed only when it is on, so the default call is unchanged
        tuned = {"tuned": True} if self._scatter_tuned else {}
        scatter_contiguous(
            0,
            np.concatenate(dsts),
            np.concatenate(frag_sizes),
            np.concatenate(srcs),
            stream=self._scatter_stream,
            **tuned,
        )
        CUASSERT(cudart.cudaStreamSynchronize(self._scatter_stream))

    def _scatter_jit_writer(self, ctx: JitTransferContext, descs: List[tuple], rank: int) -> None:
        """Scatter one just-in-time writer's fragments on the same stream as the batched scatter,
        then release that writer's slot (and settle the group after its last scatter)."""
        ok = True
        try:
            self._scatter_blocking(descs)
        except Exception as e:
            # a scatter failure must not kill the worker nor be reported as success
            ok = False
            logger.error(
                f"[kv-bounce] jit scatter failed (rid_slice={ctx.rid_slice}, writer={rank}): {e}"
            )
        self._apply_jit(
            ctx.rid_slice, lambda c, ok=ok: c.finish_writer_scatter(rank, ok), expect=ctx
        )

    # ------------------------------------------------------------------ #
    # Batched receive scatter (TRTLLM_KV_BOUNCE_RECV_SCATTER_DRAIN)
    # ------------------------------------------------------------------ #
    def _scatter_drained(self, first: tuple) -> bool:
        """Scatter ``first`` together with every item already queued behind it, then finish them.

        An item is ``(ctx, descs)`` for a dispatch-time region or ``(ctx, descs, rank)`` for one
        just-in-time writer; both kinds mix in one batch. Returns True when the close poison pill
        was taken off the queue, so the worker exits once the drained items are finished.
        """
        items = [first]
        stop = False
        while True:
            try:
                item = self._scatter_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                stop = True
                break
            items.append(item)
        oks = self._scatter_batch(items)
        # finish in queue order, after the one sync, so no completion fires before its data landed
        for item, ok in zip(items, oks):
            self._finish_scattered(item, ok)
        self._maybe_log_drain_stats()
        return stop

    def _scatter_batch(self, items: List[tuple]) -> List[bool]:
        """Scatter all items' fragments in ONE batched copy on the blocking scatter stream and sync
        once; returns each item's outcome in queue order.

        Each fragment is still read from its own writer's source (see _scatter_blocking), and the
        items write disjoint cache blocks, so the batch changes neither what lands nor where. If the
        batch raises, every item is scattered again alone, so one bad item cannot fail the others.
        Copying an item a second time is safe: it rewrites the same bytes from the same source,
        whose slot is released only when the item is finished.

        A failure that only the batch hits (every item alone succeeds), for example a failed growth
        of the copy's metadata buffer, would otherwise cost an extra failed attempt on every drain.
        After _DRAIN_BATCH_ONLY_FAILURES_MAX such drains in a row the worker turns batching off and
        runs one scatter and one sync per item, as with the knob unset.
        """
        st = self._drain_stats
        st["syncs"] += 1
        st["items"] += len(items)
        st["max_items"] = max(st["max_items"], len(items))
        try:
            self._scatter_blocking([desc for item in items for desc in item[1]])
            self._drain_batch_only_failures = 0
            return [True] * len(items)
        except Exception as e:
            # a scatter failure must not kill the worker nor be reported as success
            if len(items) == 1:
                self._log_scatter_failure(items[0], e)
                return [False]
            self._maybe_warn_batch_failure(len(items), e)
        oks = []
        for item in items:
            st["syncs"] += 1
            st["retried"] += 1
            try:
                self._scatter_blocking(item[1])
                oks.append(True)
            except Exception as e:
                self._log_scatter_failure(item, e)
                oks.append(False)
        if not all(oks):
            self._drain_batch_only_failures = 0  # a bad item explains the batch failure
            return oks
        self._drain_batch_only_failures += 1
        if self._drain_batch_only_failures >= _DRAIN_BATCH_ONLY_FAILURES_MAX:
            self._scatter_drain = False  # only this worker thread reads it after __init__
            logger.warning(
                f"[kv-bounce] drain: the batched scatter failed {self._drain_batch_only_failures} "
                f"drains in a row while every item alone succeeded; batching is off for the rest "
                f"of the run (one scatter and one sync per item)"
            )
        return oks

    def _maybe_warn_batch_failure(self, n_items: int, err: Exception) -> None:
        """Warn that a batch failed, at most once per period; the "retried" counter of the stat line
        counts every occurrence."""
        now = time.monotonic()
        if now < self._drain_warn_next_log:
            return
        self._drain_warn_next_log = now + _DRAIN_WARN_PERIOD_S
        logger.warning(
            f"[kv-bounce] drain: batched scatter of {n_items} items failed ({err}); "
            f"scattering each item alone (retried so far: {self._drain_stats['retried']})"
        )

    @staticmethod
    def _log_scatter_failure(item: tuple, err: Exception) -> None:
        """Log a failed scatter with the same message as the per-item worker."""
        ctx = item[0]
        if len(item) == 3:
            logger.error(
                f"[kv-bounce] jit scatter failed (rid_slice={ctx.rid_slice}, writer={item[2]}): "
                f"{err}"
            )
        else:
            logger.error(f"[kv-bounce] scatter failed (slot={ctx.slot_id}): {err}")

    def _finish_scattered(self, item: tuple, ok: bool) -> None:
        """Record one drained item's outcome: release a just-in-time writer's slot, or settle a
        dispatch-time region, exactly as the per-item worker does. A failure here is logged and does
        not stop the items queued after it from being finished."""
        ctx = item[0]
        try:
            if len(item) == 3:
                rank = item[2]
                self._apply_jit(
                    ctx.rid_slice, lambda c: c.finish_writer_scatter(rank, ok), expect=ctx
                )
            else:
                self._apply(ctx.rid_slice, lambda c: c.finish_scatter(ok))
        except Exception as e:
            logger.error(f"[kv-bounce] drain: finishing rid_slice={ctx.rid_slice} failed: {e}")

    def _maybe_log_drain_stats(self) -> None:
        """Log the cumulative drain counters at INFO at most once per period, never per drain."""
        now = time.monotonic()
        if now < self._drain_stats_next_log:
            return
        self._drain_stats_next_log = now + _DRAIN_STATS_PERIOD_S
        st = self._drain_stats
        logger.info(
            f"[kv-bounce] drain: syncs={st['syncs']} items={st['items']} "
            f"max_items={st['max_items']} retried={st['retried']}"
        )

    # ------------------------------------------------------------------ #
    # Just-in-time per-writer receive grant (TRTLLM_KV_BOUNCE_JIT)
    # ------------------------------------------------------------------ #
    def register_jit(
        self,
        recv_req,
        *,
        writer_ranks: Sequence[int],
        extra_bytes: int = 0,
        writers_split_layers: bool = True,
        writer_block_bytes: Optional[Sequence[Sequence[int]]] = None,
    ) -> bool:
        """Register a request slice for the just-in-time grant. Applies the same gates as reserve()
        but reserves nothing: each writer asks for its own slot later (grant). The capacity check is
        per writer. False falls back to the per-fragment path."""
        if not self._jit:
            return self._skip_bounce(
                "just-in-time grant is off (TRTLLM_KV_BOUNCE_JIT)",
                warn_key="kv-bounce-jit-off",
            )
        if self._recv_alloc is None:
            return self._skip_bounce(
                "this transport has no receive region (TRTLLM_KV_BOUNCE_DIRECTION=send)",
                warn_key="kv-bounce-no-recv-region",
            )
        writer_ranks = tuple(int(r) for r in writer_ranks)
        num_writers = len(writer_ranks)
        if num_writers == 0 or len(set(writer_ranks)) != num_writers:
            return self._skip_bounce(
                f"just-in-time grant needs distinct writers, got {list(writer_ranks)}",
                warn_key="kv-bounce-jit-writers",
            )
        plan = self._plan_region(
            recv_req, num_writers, extra_bytes, writers_split_layers, writer_block_bytes
        )
        if not plan:
            return False
        total, exact_sizes, _offsets, nblocks = plan
        caps = (
            [int(b) for b in exact_sizes]
            if exact_sizes is not None
            else [total // num_writers] * num_writers
        )
        if max(caps) > self._recv_alloc.capacity:
            return self._skip_bounce(
                f"writer transfer {max(caps) // _MIB}MiB exceeds the "
                f"{self._recv_alloc.capacity // _MIB}MiB bounce region; raise the bounce arena size "
                f"to re-enable coalescing",
                warn_key="kv-bounce-oversize",
            )
        rid_slice = (recv_req.unique_rid, recv_req.slice_id)
        with self._reserved_map_lock:
            if rid_slice in self._reserved_map:
                live = True
            else:
                live = False
                self._reserved_map[rid_slice] = JitTransferContext(
                    rid_slice=rid_slice,
                    writer_ranks=writer_ranks,
                    writer_caps=dict(zip(writer_ranks, caps)),
                )
        if live:
            return self._skip_bounce(
                f"request slice {rid_slice} already has a live bounce region",
                warn_key="kv-bounce-jit-duplicate",
            )
        logger.info_once(
            f"[kv-bounce] jit: registered {nblocks} blocks / {total // _MIB}MiB across "
            f"{num_writers} writer(s) with per-writer caps "
            f"{[round(c / _MIB, 1) for c in caps]}MiB; each writer is granted its slot on request",
            key="kv-bounce-jit",
        )
        return True

    def grant(self, rid_slice: RidSlice, rank: int, nbytes: int) -> Tuple[int, int, int]:
        """Answer one writer's grant request: (verdict, base, capacity). Never blocks: the slot is
        reserved with timeout 0 while the leaf lock is held (lock order leaf -> allocator; the
        allocator never calls out), so a cancel cannot race the allocation."""
        verdict, base, cap = JitVerdict.NO, 0, 0
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if isinstance(ctx, JitTransferContext) and self._recv_alloc is not None:
                decision = ctx.grant_check(rank, nbytes)
                if decision is JitGrantCheck.REGRANT:
                    lease = ctx.lease(rank)
                    verdict, base, cap = JitVerdict.GRANT, lease.base, lease.nbytes
                elif decision is JitGrantCheck.ALLOCATE:
                    res = self._recv_alloc.reserve(nbytes, timeout=0)
                    if res is None:
                        verdict = JitVerdict.RETRY
                    else:
                        slot_id, addr = res
                        ctx.on_granted(rank, slot_id, addr, nbytes)
                        verdict, base, cap = JitVerdict.GRANT, addr, nbytes
                        self._jit_stats["inflight_bytes"] += nbytes
                        self._jit_stats["inflight_bytes_max"] = max(
                            self._jit_stats["inflight_bytes_max"], self._jit_stats["inflight_bytes"]
                        )
            stat = {JitVerdict.GRANT: "grants", JitVerdict.RETRY: "retries"}.get(verdict, "no")
            self._jit_stats[stat] += 1
        self._maybe_log_jit_stats()
        return int(verdict), int(base), int(cap)

    def _is_jit(self, rid_slice: RidSlice) -> bool:
        if not self._jit:
            return False  # register_jit refuses, so no just-in-time context can exist
        with self._reserved_map_lock:
            return isinstance(self._reserved_map.get(rid_slice), JitTransferContext)

    def _release_idle_jit(self, rid_slice: RidSlice) -> None:
        """Drop a just-in-time context cancelled before publication. No grant can exist then; if
        one does, give it up like an orphan instead of releasing a slot a write may target."""
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if not isinstance(ctx, JitTransferContext):
                return
            if not ctx.held_slots():
                self._reserved_map.pop(rid_slice, None)
                return
        self._apply_jit(rid_slice, lambda c: c.mark_orphaned())

    def _apply_jit(
        self,
        rid_slice: RidSlice,
        mutate: Callable[[JitTransferContext], Optional[List[JitAction]]],
        *,
        expect: Optional[JitTransferContext] = None,
    ) -> None:
        """Mutate a just-in-time context under the lock and collect what it asks for, then carry it
        out with the lock released: queue scatters, release or quarantine slots, and settle. No-op
        if the context is gone (or is not ``expect``)."""
        settlement: Optional[Settlement] = None
        with self._reserved_map_lock:
            ctx = self._reserved_map.get(rid_slice)
            if not isinstance(ctx, JitTransferContext) or (
                expect is not None and ctx is not expect
            ):
                return
            actions = list(mutate(ctx) or [])
            settlement = ctx.maybe_settle()
            if settlement is not None:
                self._reserved_map.pop(rid_slice, None)
            for action in actions:
                if action[0] in ("release", "quarantine"):
                    self._jit_stats["inflight_bytes"] -= action[2]
                elif action[0] == "scatter":
                    self._jit_stats["scatters"] += 1
        for action in actions:
            kind = action[0]
            if kind == "scatter":
                _kind, rank, desc = action
                self._scatter_q.put((ctx, [desc], rank))
            elif kind == "release":
                self._recv_alloc.release(action[1])
            elif kind == "quarantine":
                self._recv_alloc.quarantine(action[1], self._quarantine_grace_s)
        if settlement is not None:
            self._commit(settlement)
        self._maybe_log_jit_stats()

    def _maybe_log_jit_stats(self) -> None:
        """Log the grant counters at INFO at most once per period, never per grant."""
        now = time.monotonic()
        with self._reserved_map_lock:
            if now < self._jit_stats_next_log:
                return
            self._jit_stats_next_log = now + _JIT_STATS_PERIOD_S
            st = dict(self._jit_stats)
        logger.info(
            f"[kv-bounce] jit: grants={st['grants']} retries={st['retries']} no={st['no']} "
            f"scatters={st['scatters']} inflight_bytes={st['inflight_bytes']} "
            f"inflight_bytes_max={st['inflight_bytes_max']}"
        )

    def close(self) -> None:
        if self._scatter_thread is not None:
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
        for alloc in (self._send_alloc, self._recv_alloc):
            if alloc is not None:
                alloc.close()


class NoBounceTransport(BounceTransport):
    """The disabled transport, used when bounce is off so callers need no None checks. Every method
    is a no-op or a negative answer."""

    enabled = False
    _reg_descs = ()

    @property
    def can_send(self) -> bool:
        return False

    @property
    def can_recv(self) -> bool:
        return False

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
        writers_split_layers: bool = True,
        writer_block_bytes: Optional[Sequence[Sequence[int]]] = None,
    ) -> bool:
        return False

    def register_jit(
        self,
        recv_req,
        *,
        writer_ranks: Sequence[int],
        extra_bytes: int = 0,
        writers_split_layers: bool = True,
        writer_block_bytes: Optional[Sequence[Sequence[int]]] = None,
    ) -> bool:
        return False

    def grant(self, rid_slice, rank: int, nbytes: int) -> Tuple[int, int, int]:
        return int(JitVerdict.NO), 0, 0

    def writer_base(self, rid_slice, writer_index: int):
        return None

    def is_bounced(self, rid_slice) -> bool:
        return False

    def release_idle_reservation(self, rid_slice) -> None:
        pass

    def orphan_reservation(self, rid_slice) -> None:
        pass

    def abort_publication(self, rid_slice, published_writers: set[int]) -> None:
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
    # Validate the opt-in send-stream mode, region scope, free-memory share, just-in-time grant,
    # batched receive scatter and small-block receive scatter switches outside the broad except
    # below: a bad value must fail loudly, not silently disable bounce and fall back to the
    # per-fragment path.
    send_stream_mode_from_env()
    direction_from_env()
    max_free_fraction_from_env()
    bounce_jit_from_env()
    recv_scatter_drain_from_env()
    recv_scatter_tuned_from_env()
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
    fall back to the per-fragment request. Under an exact fan-in layout the receiver also sent this
    writer's sub-region size; a write that would not fit takes the per-fragment path instead, so a
    wrong size can only lose coalescing, never overrun a neighbouring writer."""
    if write_meta.bounce_dst_base is not None:
        capacity = getattr(write_meta, "bounce_dst_capacity", None)
        if capacity is not None and int(write_meta.sizes.sum()) > capacity:
            logger.warning_once(
                f"[kv-bounce] in-place: writer sends {int(write_meta.sizes.sum())}B but its exact "
                f"fan-in sub-region holds {capacity}B; falling back to the per-fragment path",
                key="kv-bounce-writer-over-capacity",
            )
        else:
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


def layer_bytes_per_group(page_table: KVCachePageTable) -> list[dict[int, int] | None]:
    """Return, for each layer group, the bytes one cache block holds for each global layer.

    Built from the buffer entries of every distinct physical pool the group exposes (a buffer seen
    through several views counts once), so per-layer sizes need not be uniform across groups.
    Non-attention groups keep a ``None`` placeholder, aligned with ``block_bytes_per_group``.
    """
    from tensorrt_llm._torch.disaggregation.resource.page import CacheKind

    assert page_table is not None
    out: list[dict[int, int] | None] = []
    for lg in page_table.layer_groups:
        if lg.kind != CacheKind.PAGED:
            out.append(None)
            continue
        local_to_global = {ll.local_layer_id: ll.global_layer_id for ll in lg.local_layers}
        seen: set[tuple[int, int, int]] = set()
        per_layer: dict[int, int] = {}
        for pool_view in lg.pool_views:
            for entry in pool_view.buffer_entries:
                local_layer_id = int(entry["local_layer_id"])
                key = (int(pool_view.pool_idx), local_layer_id, int(entry["offset"]))
                if key in seen:
                    continue
                seen.add(key)
                global_layer_id = int(local_to_global.get(local_layer_id, local_layer_id))
                per_layer[global_layer_id] = per_layer.get(global_layer_id, 0) + int(entry["size"])
        out.append(per_layer)
    return out


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

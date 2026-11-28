# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sharded cold pages for the KV pools that attention TP replicates (on with block reuse).

Without attention DP every TP rank holds the same MLA latent page and the same DSA index-key page, so each rank's
host and disk tiers would hold W identical copies of every evicted page. With this KV-V2 cold-page provider, rank
``r`` of the W-rank attention-TP group keeps only slice ``r`` of every replicated page in its cold tiers, so the same
host / disk quota holds W times as many of these pages:

- cold page = ``[slice r of every region][16-byte trailer]``. A region is a run of hot buffers that are contiguous
  inside one pool slot (the KEY pool and the INDEX_KEY pool of the MLA lifecycle), cut into W equal slices. The
  trailer holds a 64-bit checksum of the full hot page, written at encode: the pages are bitwise identical on every
  rank, so the checksum is a rank-invariant page id.
- encode (hot -> cold): this rank's slices and the trailer, no collective.
- decode (cold -> hot), on the migration stream KV-V2 hands over:
  1. Every TP rank posts the LOGICAL batch (call sequence number, kind, lifecycle, page count) to an MPI all-gather
     on a TP sub-communicator. Slot ids are not part of it: KV-V2 hands out a recycled slot only once its last GPU use
     has completed and a fresh one otherwise, so the same page lands in different slots on different ranks.
  2. While the agreement travels, this rank stages its index pairs and copies its slices host -> GPU staging.
  3. The agreement is waited for (with a deadline) before the collective: a mismatch or a timeout is fatal.
  4. Per chunk, an in-place NCCL all-gather on a communicator of its own fills in the other ranks' slices; one launch
     matches each of this rank's pages to the same page id in every rank's block (another position of the chunk when
     the order differs); one launch scatters the W slices into the hot pages.
  5. The match adds pages no rank block could supply to cumulative totals in pinned host memory. The host reads them
     every few calls and a watchdog thread every 0.25 s; any such page is fatal.
  6. The watchdog runs from the transport's creation. A decode call still inside its collectives, or decode GPU work
     not complete, ``nccl_timeout_s`` after it started is fatal. ``configure`` runs the all-gathers once, so NCCL sets
     up the communicator's connections there and not inside a decode call.

Lifecycles holding none of the provider's layers (the KDA recurrent state, which is head-sharded) keep KV-V2's
lossless default codec. Failures abort the process after logging: a rank that restored a page it could not rebuild,
or a TP group that no longer agrees, cannot continue correctly.
"""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import logging
import os
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np
import torch

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned
from tensorrt_llm.logger import logger

from ...kernels.common.match_page_ids import match_page_ids
from ...kernels.common.page_checksums import page_checksums
from ...kernels.common.paged_segment_copy import SEG_COLS, paged_segment_copy

#: Roles whose bytes are identical on every attention-TP rank (MLA latent, DSA index keys, their block scales).
REPLICATED_ROLES = frozenset({"key", "index_key", "key_block_scale"})
TRAILER_BYTES = 16
MAGIC = 0x53484344
_RING = 8
_MAX_PAIRS = 1 << 16
_EVENTS = 32
_POLL_EVERY = 8
_WATCHDOG_TICK_S = 0.25
_STATS_EVERY_S = 300.0
_KIND_DECODE = 2
_NCCL_UINT8 = 1


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


class _Nccl:
    """The few NCCL entry points the codec uses, through the libnccl.so.2 already loaded in the process."""

    def __init__(self) -> None:
        lib = ctypes.CDLL("libnccl.so.2")
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
        lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            _NcclUniqueId,
            ctypes.c_int,
        ]
        lib.ncclAllGather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib = lib

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(
                f"{what}: NCCL error {rc} ({self.lib.ncclGetErrorString(rc).decode()})"
            )

    def version(self) -> int:
        v = ctypes.c_int()
        self._check(self.lib.ncclGetVersion(ctypes.byref(v)), "ncclGetVersion")
        return v.value

    def unique_id(self) -> bytes:
        uid = _NcclUniqueId()
        self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
        return bytes(bytearray(uid.internal))

    def init_rank(self, nranks: int, uid: bytes, rank: int) -> ctypes.c_void_p:
        comm = ctypes.c_void_p()
        u = _NcclUniqueId()
        ctypes.memmove(ctypes.addressof(u), uid, 128)
        self._check(
            self.lib.ncclCommInitRank(ctypes.byref(comm), nranks, u, rank), "ncclCommInitRank"
        )
        return comm

    def all_gather_inplace(self, recv: int, count: int, rank: int, comm, stream: int) -> None:
        send = recv + rank * count
        self._check(
            self.lib.ncclAllGather(send, recv, count, _NCCL_UINT8, comm, stream), "ncclAllGather"
        )


class _Transport:
    """The attention-TP group's MPI sub-communicator (batch agreement), its own NCCL communicator and a watchdog."""

    def __init__(
        self, mapping, timeout_s: float, nccl_timeout_s: float, abort_on_error: bool
    ) -> None:
        from mpi4py import MPI

        from tensorrt_llm._utils import mpi_comm

        group = list(mapping.tp_group)
        self.world_size = len(group)
        self.rank = int(mapping.tp_rank)
        self.timeout_s = timeout_s
        self.nccl_timeout_s = nccl_timeout_s
        self.abort_on_error = abort_on_error
        self.poisoned: Optional[str] = None
        # A plain MPI communicator (the session communicator pickles in Python) and fixed buffers: one agreement is
        # in flight at a time.
        self.comm = MPI.Intracomm.Split(mpi_comm(), min(group), self.rank)
        if self.comm.Get_size() != self.world_size:
            raise RuntimeError(
                f"TP sub-communicator has {self.comm.Get_size()} ranks, expected {group}"
            )
        self.device = torch.cuda.current_device()
        self.nccl = _Nccl()
        uid = self.comm.bcast(self.nccl.unique_id() if self.rank == 0 else None, root=0)
        t0 = time.monotonic()
        self.nccl_comm = self.nccl.init_rank(self.world_size, uid, self.rank)
        self._send = np.zeros(3, dtype=np.uint64)
        self._recv = np.zeros((self.world_size, 3), dtype=np.uint64)
        self._pending_decode: deque[tuple[torch.cuda.Event, float, tuple[int, int]]] = deque()
        self._in_call = None
        self.on_tick = None
        self._stop = threading.Event()
        if self.rank == 0:
            logger.info(
                f"Sharded cold pages: TP group {group}, NCCL {self.nccl.version()} communicator in "
                f"{time.monotonic() - t0:.2f} s"
            )
        # The watchdog runs from the start, so it also times a first call that never returns.
        self._watchdog: Optional[threading.Thread] = threading.Thread(
            target=self._watch, name="shard-cold-pages-watchdog", daemon=True
        )
        self._watchdog.start()

    def fail(self, message: str) -> None:
        self.poisoned = f"sharded cold pages (TP rank {self.rank}): {message}"
        logger.error(self.poisoned)
        if self.abort_on_error:
            os.abort()
        raise RuntimeError(self.poisoned)

    def agree_post(self, seq: int, kind: int, lifecycle: int, pairs: np.ndarray):
        """Start the batch agreement; finish it with ``agree_wait`` before any collective.

        Every TP rank must present the same logical batch: call sequence number, kind, lifecycle and page count (the
        collective sizes). The slot ids are physical placement and differ across ranks; their digest is gathered for
        statistics only."""
        if self.poisoned:
            raise RuntimeError(self.poisoned)
        logical = hashlib.blake2b(
            struct.pack("<qqq", kind, lifecycle, pairs.shape[0]), digest_size=8
        )
        slots = hashlib.blake2b(pairs.tobytes(), digest_size=8)
        send, recv = self._send, self._recv
        send[0] = seq
        send[1] = int.from_bytes(logical.digest(), "little")
        send[2] = int.from_bytes(slots.digest(), "little")
        request = self.comm.Iallgather(send, recv)
        return (request, recv, time.monotonic(), seq, kind, lifecycle, pairs.shape[0])

    def agree_wait(self, handle) -> tuple:
        """Returns (seconds spent waiting for the other ranks, whether the slot ids differ across ranks); a logical
        mismatch or a timeout is fatal."""
        request, recv, t0, seq, kind, lifecycle, num_pages = handle
        spins = 0
        while not request.Test():
            if time.monotonic() - t0 > self.timeout_s:
                self.fail(
                    f"timed out after {self.timeout_s:.0f} s waiting for the other TP ranks at batch seq {seq} "
                    f"(kind {kind}, lifecycle {lifecycle}, {num_pages} pages)"
                )
            spins += 1
            if spins > 20000:
                time.sleep(5e-5)
        if not (recv[:, :2] == recv[0, :2]).all():
            per_rank = ", ".join(
                f"r{q}: seq {int(s)} batch {int(h):016x}" for q, (s, h, _) in enumerate(recv)
            )
            self.fail(
                f"migration batch mismatch across TP ranks at seq {seq} (kind {kind}, lifecycle {lifecycle}, "
                f"{num_pages} pages here; the digest covers kind, lifecycle and page count): {per_rank}"
            )
        return time.monotonic() - t0, bool((recv[:, 2] != recv[0, 2]).any())

    def all_gather(self, staging: int, count: int, stream: int) -> None:
        if self.poisoned:
            raise RuntimeError(self.poisoned)
        self.nccl.all_gather_inplace(staging, count, self.rank, self.nccl_comm, stream)

    def watch(self, event: torch.cuda.Event, what: tuple[int, int]) -> None:
        """Enqueue the single recording that marks this decode's completion."""
        self._pending_decode.append((event, time.monotonic(), what))

    def _oldest_pending_decode(self) -> tuple[torch.cuda.Event, float, tuple[int, int]] | None:
        # Submissions only append. The watchdog is the sole consumer, so the head stays
        # valid across the CUDA query even if a decode is submitted on another thread.
        while self._pending_decode:
            oldest = self._pending_decode[0]
            if not oldest[0].query():
                return oldest
            self._pending_decode.popleft()
        return None

    def begin_call(self, what) -> None:
        """Mark a call's collectives as in flight: one still in flight after the NCCL timeout is fatal (a peer that
        never joins a collective can block the calling thread inside NCCL)."""
        self._in_call = (time.monotonic(), what)

    def end_call(self) -> None:
        self._in_call = None

    def close(self) -> None:
        """Stop the watchdog thread (the communicators live as long as the process)."""
        self._stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=2 * _WATCHDOG_TICK_S + 1.0)
            self._watchdog = None

    def _fatal(self, message: str) -> None:
        """From the watchdog thread: log and abort (with ``abort_on_error``, else poison: the next call raises)."""
        self.poisoned = f"sharded cold pages (TP rank {self.rank}): {message}"
        logger.error(self.poisoned)
        if self.abort_on_error:
            os.abort()

    def _watch(self) -> None:
        torch.cuda.set_device(self.device)
        while not self._stop.wait(_WATCHDOG_TICK_S):
            if self.poisoned is not None:
                continue
            in_call = self._in_call
            if in_call is not None and time.monotonic() - in_call[0] > self.nccl_timeout_s:
                self._fatal(
                    f"{in_call[1]} still inside its collectives after {self.nccl_timeout_s:.0f} s"
                )
                continue
            now = time.monotonic()
            oldest = self._oldest_pending_decode()
            if oldest is not None and now - oldest[1] > self.nccl_timeout_s:
                self._fatal(
                    f"decode (seq, pages) {oldest[2]} not complete after {self.nccl_timeout_s:.0f} s"
                )
                continue
            if self.on_tick is not None:
                try:
                    self.on_tick()
                except Exception as error:  # noqa: BLE001 - the watchdog must keep running
                    logger.error(f"sharded cold pages: watchdog tick failed: {error}")


@dataclass
class _Region:
    key: tuple
    base: int
    stride: int
    nbytes: int
    slice_bytes: int = 0
    cold_off: int = 0
    page_off: int = 0


@dataclass
class _Plan:
    """One provider lifecycle: its regions, cold layout and launch tables (device int64 [S, 8])."""

    regions: list
    page_bytes: int
    cold_page_bytes: int
    trailer_off: int
    max_slice: int = 0
    encode: Any = None
    pack: Any = None
    scatter: Any = None
    gather: Any = None
    trailer: Any = None
    weights: Any = None


@dataclass
class _CodecState:
    """State of one KV-V2 manager's codec; ``layer_ids`` is read by the native adapter."""

    layer_ids: tuple
    plans: tuple = ()
    seq: int = 0


@dataclass
class _Stats:
    enc_calls: int = 0
    enc_pages: int = 0
    enc_s: float = 0.0
    dec_calls: int = 0
    dec_pages: int = 0
    dec_s: float = 0.0
    agree_wait_s: float = 0.0
    slot_divergent_calls: int = 0
    reordered_pages: int = 0


def _segs(rows: list) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, SEG_COLS).cuda()


class ShardColdPageProvider:
    """KV-V2 cold-page provider (``native.create_python_cold_page_codec``) for attention-TP-replicated pools."""

    provides_cold_page_codec = True
    uses_iteration_lifecycle = False
    config = None

    def __init__(
        self,
        mapping,
        *,
        timeout_s: float = 120.0,
        nccl_timeout_s: float = 600.0,
        chunk_pages: int = 128,
        abort_on_error: bool = True,
    ) -> None:
        """``timeout_s``: deadline of a batch agreement; ``nccl_timeout_s``: of a decode's GPU work;
        ``chunk_pages``: pages per all-gather (sets the GPU staging size); ``abort_on_error``: fatal errors abort the
        process (raise instead, for tests)."""
        self.mapping = mapping
        self.world_size = int(mapping.tp_size)
        self.rank = int(mapping.tp_rank)
        self.timeout_s = timeout_s
        self.nccl_timeout_s = nccl_timeout_s
        self.chunk_pages = chunk_pages
        self.abort_on_error = abort_on_error
        self._transport: Optional[_Transport] = None
        self._streams: dict = {}
        self._staging: Optional[torch.Tensor] = None
        self._trailers: Optional[torch.Tensor] = None
        self._arange: Optional[torch.Tensor] = None
        self._source: Optional[torch.Tensor] = None
        self._match_host: Optional[torch.Tensor] = None
        self._match_pending = None
        self._match_seen = (0, 0)
        self._match_lock = threading.Lock()
        self._ring_host: list = []
        self._ring_dev: list = []
        self._ring_np: list = []
        self._ring_addr: list = []
        self._ring_events: list = []
        self._ring_next = 0
        self._ring_in_call: list = []
        self._last_done: Optional[torch.cuda.Event] = None
        self._last_stream: Optional[int] = None
        self._events = [torch.cuda.Event() for _ in range(_EVENTS)]
        self._event_next = 0
        self._connected = False
        self.stats = _Stats()
        self._stats_at = time.monotonic()
        self._stats_prev = _Stats()

    # ---- KV-V2 manager side -------------------------------------------------------------------------------------

    def create_cold_page_codec(
        self,
        cache_config: object,
        *,
        runtime_dtype=None,
        pp_layers: Sequence[int] = (),
        num_kv_heads_per_layer: Sequence[int] = (),
        head_dim_per_layer: Sequence[int] = (),
        is_draft: bool = False,
    ) -> Optional[object]:
        """A native codec owning the replicated attention layers; None (lossless default) for draft managers or
        when no layer qualifies. ``runtime_dtype`` and the per-layer geometry are not needed: the codec moves raw
        bytes of whatever dtype each buffer holds."""
        # A partially committed page may contain different unused bytes on each rank.
        # Its full-page checksum cannot identify a replicated page.
        if is_draft or cache_config.enable_partial_commit:
            return None
        from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

        layer_ids = []
        for layer in cache_config.layers:
            if not isinstance(layer, AttentionLayerConfig):
                continue
            # The target's replicated MLA layers have full attention; a windowed layer holds the
            # standalone drafter's context (``UnifiedDraftCacheMixin``) and keeps the default codec.
            if layer.sliding_window_size is not None:
                continue
            roles = {str(buffer.role) for buffer in layer.buffers}
            if roles and roles <= REPLICATED_ROLES and not any(b.is_sparse for b in layer.buffers):
                layer_ids.append(int(layer.layer_id))
        if not layer_ids:
            return None
        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        return native.create_python_cold_page_codec(
            self, _CodecState(layer_ids=tuple(sorted(layer_ids)))
        )

    def configure(self, codec_state: _CodecState, lifecycles: Sequence[object]) -> list:
        """Resolve each lifecycle's regions and cold layout; collective over the TP group (layout check)."""
        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        transport = self._get_transport()
        plans, signature = [], []
        for lifecycle in lifecycles:
            plan = self._plan(lifecycle)
            plans.append(plan)
            signature.append(
                [(r.key, r.stride, r.nbytes) for r in plan.regions] + [plan.cold_page_bytes]
            )
        everyone = transport.comm.allgather(signature)
        if any(other != signature for other in everyone):
            transport.fail(f"cold-page layouts differ across TP ranks: {everyone}")
        codec_state.plans = tuple(plans)
        self._ensure_buffers(plans)
        if not self._connected:
            self._connect(plans)
        properties = []
        for plan in plans:
            prop = native.ColdPageLifecycleProperties()
            prop.cold_page_bytes = plan.cold_page_bytes
            prop.page_index_location = native.ColdPageIndexLocation.HOST
            properties.append(prop)
            if self.rank == 0:
                logger.info(
                    f"Sharded cold pages: {len(plan.regions)} regions "
                    f"{[(r.key, r.nbytes, r.slice_bytes) for r in plan.regions]}, hot page {plan.page_bytes} B, "
                    f"cold page {plan.cold_page_bytes} B (1/{self.world_size} + {TRAILER_BYTES} B trailer)"
                )
        return properties

    def encode_cold_pages(
        self, codec_state, lifecycle_index, cold_base, page_indices, num_pages, stream
    ) -> None:
        t0 = time.perf_counter()
        plan = codec_state.plans[lifecycle_index]
        pairs = self._host_pairs(page_indices, num_pages)
        self._poll_match()
        with torch.cuda.stream(self._stream(stream)):
            self._begin()
            for b in range(0, num_pages, _MAX_PAIRS):
                dev = self._device_pairs(pairs[b : b + _MAX_PAIRS])
                dst, src = dev[:, 0], dev[:, 1]
                paged_segment_copy(
                    plan.encode, src, dst, dst_add=cold_base, max_segment_bytes=plan.max_slice
                )
                for c in range(0, dst.shape[0], self.chunk_pages):
                    self._write_trailers(
                        plan,
                        src[c : c + self.chunk_pages],
                        dst[c : c + self.chunk_pages],
                        cold_base,
                    )
            self._end()
        self.stats.enc_calls += 1
        self.stats.enc_pages += num_pages
        self.stats.enc_s += time.perf_counter() - t0

    def decode_cold_pages(
        self, codec_state, lifecycle_index, cold_base, page_indices, num_pages, stream
    ) -> None:
        t0 = time.perf_counter()
        transport = self._transport
        plan = codec_state.plans[lifecycle_index]
        slot = None
        if num_pages <= _MAX_PAIRS:
            # KV-V2's index array goes straight into a pinned ring slot (no extra copy).
            slot = self._acquire_slot()
            ctypes.memmove(self._ring_addr[slot], page_indices, 8 * num_pages)
            pairs = self._ring_np[slot][:num_pages]
        else:
            pairs = self._host_pairs(page_indices, num_pages)
        codec_state.seq += 1
        seq = codec_state.seq
        # The agreement travels while this rank stages its indices and packs its slices; the all-gather waits.
        agreement = transport.agree_post(seq, _KIND_DECODE, lifecycle_index, pairs)
        if seq % _POLL_EVERY == 0:
            self._poll_match()
        cpb = plan.cold_page_bytes
        staging = self._staging.data_ptr()
        previous = torch.cuda.current_stream()
        torch.cuda.set_stream(self._stream(stream))
        transport.begin_call(f"decode (seq, pages) {(seq, num_pages)}")
        try:
            self._begin(stream)
            for b in range(0, num_pages, _MAX_PAIRS):
                if slot is not None:
                    dev = self._slot_to_device(slot, num_pages)
                else:
                    dev = self._device_pairs(pairs[b : b + _MAX_PAIRS])
                for c in range(0, dev.shape[0], self.chunk_pages):
                    dst = dev[c : c + self.chunk_pages, 0]
                    src = dev[c : c + self.chunk_pages, 1]
                    n = dst.shape[0]
                    paged_segment_copy(
                        plan.pack,
                        src,
                        self._arange[:n],
                        src_add=cold_base,
                        dst_add=staging + self.rank * n * cpb,
                        max_segment_bytes=cpb,
                    )
                    if agreement is not None:
                        wait_s, slot_divergent = transport.agree_wait(agreement)
                        agreement = None
                        self.stats.agree_wait_s += wait_s
                        self.stats.slot_divergent_calls += int(slot_divergent)
                    transport.all_gather(staging, n * cpb, stream)
                    source = self._source[: self.world_size * n]
                    match_page_ids(
                        source,
                        staging=staging,
                        counters=self._match_host,
                        n=n,
                        world=self.world_size,
                        rank=self.rank,
                        cold_page_bytes=cpb,
                        trailer_off=plan.trailer_off,
                        magic=MAGIC,
                    )
                    paged_segment_copy(
                        plan.scatter,
                        source,
                        dst,
                        src_add=staging,
                        q_stride=n * cpb,
                        idx_q_stride=n,
                        max_segment_bytes=plan.max_slice,
                    )
            # The watchdog deadline belongs to this recording, even after the encode ring wraps.
            event = torch.cuda.Event()
            event.record()
            self._match_pending = (event, seq)
            transport.watch(event, (seq, num_pages))
            self._end(event)
        finally:
            transport.end_call()
            torch.cuda.set_stream(previous)
        self.stats.dec_calls += 1
        self.stats.dec_pages += num_pages
        self.stats.dec_s += time.perf_counter() - t0

    def close(self) -> None:
        """Stop the watchdog thread; the provider must not be used afterwards."""
        if self._transport is not None:
            self._transport.close()
            atexit.unregister(self._log_stats_at_exit)

    # ---- internals -----------------------------------------------------------------------------------------------

    def _get_transport(self) -> _Transport:
        if self._transport is None:
            self._transport = _Transport(
                self.mapping, self.timeout_s, self.nccl_timeout_s, self.abort_on_error
            )
            atexit.register(self._log_stats_at_exit)
        return self._transport

    def _plan(self, lifecycle) -> _Plan:
        W = self.world_size
        buffers = []
        for layer_id, roles in lifecycle.layers.items():
            for role, hot in roles.items():
                buffers.append(
                    (
                        (int(layer_id), str(role)),
                        int(hot.raw_base),
                        int(hot.raw_slot_bytes),
                        int(hot.raw_bytes),
                    )
                )
        buffers.sort(key=lambda b: (b[2], b[1]))
        regions: list = []
        for key, base, stride, nbytes in buffers:
            last = regions[-1] if regions else None
            if last is not None and last.stride == stride and last.base + last.nbytes == base:
                last.nbytes += nbytes
                last.key = min(last.key, key)
            else:
                regions.append(_Region(key=key, base=base, stride=stride, nbytes=nbytes))
        # Rank-invariant order (pool addresses differ across ranks; the layer / role structure does not).
        regions.sort(key=lambda r: r.key)
        cold_off = page_off = 0
        for r in regions:
            if r.base % 16 or r.stride % 16 or r.nbytes % (16 * W):
                raise ValueError(
                    f"region {r.key}: base {r.base:#x}, slot stride {r.stride}, {r.nbytes} B cannot be cut into "
                    f"{W} 16-byte-aligned slices"
                )
            r.slice_bytes = r.nbytes // W
            r.cold_off, r.page_off = cold_off, page_off
            cold_off += r.slice_bytes
            page_off += r.nbytes
        plan = _Plan(
            regions=regions,
            page_bytes=page_off,
            cold_page_bytes=cold_off + TRAILER_BYTES,
            trailer_off=cold_off,
            max_slice=max(r.slice_bytes for r in regions),
        )
        cpb, me = plan.cold_page_bytes, self.rank
        plan.encode = _segs(
            [
                [r.base, r.stride, me * r.slice_bytes, 0, cpb, r.cold_off, r.slice_bytes, 0]
                for r in regions
            ]
        )
        plan.pack = _segs([[0, cpb, 0, 0, cpb, 0, cpb, 0]])
        plan.scatter = _segs(
            [
                [0, cpb, r.cold_off, r.base, r.stride, q * r.slice_bytes, r.slice_bytes, q]
                for q in range(W)
                for r in regions
            ]
        )
        plan.gather = _segs(
            [[r.base, r.stride, 0, 0, plan.page_bytes, r.page_off, r.nbytes, 0] for r in regions]
        )
        plan.trailer = _segs([[0, TRAILER_BYTES, 0, 0, cpb, plan.trailer_off, TRAILER_BYTES, 0]])
        # The page id: a checksum with odd 64-bit weights per word, the same weights on every rank.
        g = torch.Generator().manual_seed(0x5EED)
        w = torch.randint(-(2**62), 2**62, (plan.page_bytes // 8,), generator=g, dtype=torch.int64)
        plan.weights = (w | 1).cuda()
        return plan

    def _connect(self, plans) -> None:
        """Run the all-gathers once while every TP rank is here (configure is collective): NCCL sets up a
        communicator's connections at the first collective that needs them, blocking the calling thread until every
        rank has joined, so that happens here and not inside a decode call."""
        cpb = max(p.cold_page_bytes for p in plans)
        side = torch.cuda.Stream()
        self._transport.begin_call("configure all-gathers")
        try:
            for n in sorted({1, self.chunk_pages}):
                self._transport.all_gather(self._staging.data_ptr(), n * cpb, side.cuda_stream)
            side.synchronize()
        finally:
            self._transport.end_call()
        self._connected = True

    def _ensure_buffers(self, plans) -> None:
        W, ch = self.world_size, self.chunk_pages
        cpb = max(p.cold_page_bytes for p in plans)
        if self._staging is None or self._staging.numel() < W * ch * cpb:
            self._staging = torch.empty(W * ch * cpb, dtype=torch.uint8, device="cuda")
        if self._arange is None:
            self._arange = torch.arange(max(ch, 1), dtype=torch.int32, device="cuda")
            self._trailers = torch.empty((ch, 2), dtype=torch.int64, device="cuda")
            self._source = torch.empty(W * ch, dtype=torch.int32, device="cuda")
            # Cumulative (no match, reordered) totals in pinned host memory: the match kernel adds to them with GPU
            # atomics, so the host and the watchdog read them without a copy or a sync.
            self._match_host = maybe_pin_memory(torch.zeros(2, dtype=torch.int64))
            if not self._match_host.is_pinned():
                raise NotImplementedError(
                    "sharded cold pages need device-accessible pinned host memory"
                )
            self._transport.on_tick = self._tick
        if not self._ring_host:
            self._ring_host = [
                torch.empty((_MAX_PAIRS, 2), dtype=torch.int32, pin_memory=prefer_pinned())
                for _ in range(_RING)
            ]
            self._ring_dev = [
                torch.empty((_MAX_PAIRS, 2), dtype=torch.int32, device="cuda") for _ in range(_RING)
            ]
            self._ring_np = [t.numpy() for t in self._ring_host]
            self._ring_addr = [t.data_ptr() for t in self._ring_host]
            self._ring_events = [None] * _RING

    @staticmethod
    def _host_pairs(page_indices: int, num_pages: int) -> np.ndarray:
        """Copy of KV-V2's host PageIndexPair array ({int32 dst, int32 src} per page; valid only during the call)."""
        out = np.empty((num_pages, 2), dtype=np.int32)
        ctypes.memmove(out.ctypes.data, page_indices, 8 * num_pages)
        return out

    def _acquire_slot(self) -> int:
        """The next pinned ring slot, once the call that last used it has completed on the GPU."""
        slot = self._ring_next
        self._ring_next = (slot + 1) % _RING
        event = self._ring_events[slot]
        if event is not None:
            event.synchronize()
        return slot

    def _slot_to_device(self, slot: int, n: int) -> torch.Tensor:
        dev = self._ring_dev[slot][:n]
        dev.copy_(self._ring_host[slot][:n], non_blocking=True)
        self._ring_in_call.append(slot)
        return dev

    def _device_pairs(self, pairs: np.ndarray) -> torch.Tensor:
        slot = self._acquire_slot()
        self._ring_np[slot][: pairs.shape[0]] = pairs
        return self._slot_to_device(slot, pairs.shape[0])

    def _begin(self, stream: Optional[int] = None) -> None:
        self._ring_in_call = []
        if self._last_done is not None and (stream is None or stream != self._last_stream):
            torch.cuda.current_stream().wait_event(self._last_done)
        self._last_stream = stream

    def _next_event(self) -> torch.cuda.Event:
        """Encode events are recycled round-robin: a re-recorded event marks a later point on the stream, a conservative
        stand-in wherever the earlier point was still referenced."""
        self._event_next = (self._event_next + 1) % len(self._events)
        return self._events[self._event_next]

    def _end(self, done: Optional[torch.cuda.Event] = None) -> None:
        if done is None:
            done = self._next_event()
            done.record()
        self._last_done = done
        for slot in self._ring_in_call:
            self._ring_events[slot] = done

    def _stream(self, stream: int):
        ext = self._streams.get(stream)
        if ext is None:
            ext = self._streams[stream] = torch.cuda.ExternalStream(stream)
        return ext

    def _write_trailers(self, plan: _Plan, hot_slots, cold_slots, cold_base: int) -> None:
        n = hot_slots.shape[0]
        trailers = self._trailers[:n]
        # The page id, read from the hot pages through the gather table; no copy of the pages, no allocation.
        page_checksums(
            plan.gather,
            hot_slots,
            plan.weights,
            trailers[:, 0],
            max_segment_bytes=max(r.nbytes for r in plan.regions),
        )
        trailers[:, 1] = (MAGIC << 32) | self.rank
        paged_segment_copy(
            plan.trailer,
            self._arange[:n],
            cold_slots,
            src_add=trailers.data_ptr(),
            dst_add=cold_base,
            max_segment_bytes=TRAILER_BYTES,
        )

    def _poll_match(self, from_watchdog: bool = False, blocking: bool = False) -> None:
        """Read the cumulative page-id match totals; a page no rank block could supply is fatal (from the watchdog
        thread: the transport is poisoned, so the next call raises, unless the process aborts first)."""
        if self._match_host is None:
            return
        pending = self._match_pending
        if blocking and pending is not None:
            pending[0].synchronize()
        with self._match_lock:
            bad, reordered = (int(v) for v in self._match_host.tolist())
            bad_seen, _ = self._match_seen
            self._match_seen = (bad, reordered)
            self.stats.reordered_pages = reordered
        if bad > bad_seen:
            seq = pending[1] if pending is not None else 0
            message = (
                f"content mismatch: {bad - bad_seen} (rank block, page) pairs of the decode batches up to seq {seq} "
                "have no matching page id on some TP rank"
            )
            if from_watchdog:
                self._transport.poisoned = f"sharded cold pages (TP rank {self.rank}): {message}"
                logger.error(self._transport.poisoned)
                if self.abort_on_error:
                    os.abort()
                return
            self._transport.fail(message)

    def _tick(self) -> None:
        """Watchdog tick: the page-id check, and on TP rank 0 a statistics line every ``_STATS_EVERY_S``."""
        self._poll_match(from_watchdog=True)
        now = time.monotonic()
        if self.rank == 0 and now - self._stats_at >= _STATS_EVERY_S:
            self._log_stats(now - self._stats_at, self._stats_prev)
            self._stats_at, self._stats_prev = now, replace(self.stats)

    def _log_stats_at_exit(self) -> None:
        """The totals at interpreter exit, while the log's streams are open: a test runner may have closed the
        stream the log writes to by then, and logging would print an error instead of the line."""
        handlers = logging.getLogger(logger.PREFIX).handlers
        if any(getattr(getattr(handler, "stream", None), "closed", False) for handler in handlers):
            return
        self._log_stats()

    def _log_stats(self, interval_s: Optional[float] = None, prev: Optional[_Stats] = None) -> None:
        """Totals since start; with ``interval_s``, also the share of that interval the executor spent in the
        codec's calls (encode + decode, agreement waits included)."""
        s = self.stats
        if s.enc_calls == 0 and s.dec_calls == 0:
            return
        share = ""
        if interval_s:
            busy = (s.enc_s - prev.enc_s) + (s.dec_s - prev.dec_s)
            wait = s.agree_wait_s - prev.agree_wait_s
            share = (
                f"; last {interval_s:.0f} s: {s.enc_calls - prev.enc_calls} encode / "
                f"{s.dec_calls - prev.dec_calls} decode calls, in codec calls {busy:.2f} s = "
                f"{100.0 * busy / interval_s:.2f}% of wall (agreement wait {wait:.2f} s)"
            )
        logger.info(
            f"Sharded cold pages (TP rank {self.rank}): encode {s.enc_calls} calls / {s.enc_pages} pages / "
            f"{s.enc_s:.2f} s; decode {s.dec_calls} calls / {s.dec_pages} pages / {s.dec_s:.2f} s, agreement wait "
            f"{s.agree_wait_s:.2f} s, slot-divergent calls {s.slot_divergent_calls}, reordered pages "
            f"{s.reordered_pages}{share}"
        )


def create_shard_cold_page_provider(
    mapping, kv_cache_config=None
) -> Optional[ShardColdPageProvider]:
    """The provider on a server with block reuse on, else None.

    Sharded cold pages are on when the server's block reuse is on (``kv_cache_config.enable_block_reuse``): the
    measured context servers ran them with reuse on, and the generation servers, which run without reuse, did not.
    A server without them gets None, sets up no transport
    (NCCL communicator, watchdog thread, staging buffer) and keeps KV-V2's lossless default codec. Every layout of
    this target replicates the attention pools (attention TP without attention DP, CP or PP).
    """
    if kv_cache_config is None or not kv_cache_config.enable_block_reuse:
        return None
    if mapping.tp_rank == 0:
        logger.info(
            f"Sharded cold pages on: TP {mapping.tp_size}, rank slices of 1/{mapping.tp_size}"
        )
    return ShardColdPageProvider(mapping)


__all__ = ["ShardColdPageProvider", "create_shard_cold_page_provider"]

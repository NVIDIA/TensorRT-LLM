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
"""KV fabric batch writer (FBW).

For KV writes whose fragments are large (average >= 64 KiB, e.g. attention-DP), the sender maps the
receiver's KV pool chunks once per peer through their FABRIC handles and writes each request with a
few ``cudaMemcpyBatchAsync`` calls instead of one transfer-agent operation per fragment.

- Receiver side: :func:`export_rank_pools` exports the FABRIC handle of every physical chunk backing
  the registered KV pools; it rides in ``RankInfo.fabric_export``.
- Sender side: :class:`FabricWriter` imports a peer's chunks in a background thread (VA-contiguous,
  same layout as the owner), then serves eligible writes through ONE dispatcher thread: FIFO over
  requests, at most ``window`` batches of at most ``ops_per_batch`` copies in flight per process, one
  blocking-sync event per batch.
- Only device-to-device writes of requests without physical-ownership enforcement are eligible; the
  ownership-enforcing paths always submit through the transfer agent.
- NIXL keeps the handshake, the completion/notify protocol and every ineligible request. Any import,
  map or copy error disables the peer (its requests go through NIXL again) and the failed request is
  re-issued through NIXL, which is idempotent because the request still owns its destination pages.

Knobs (environment):
  TRTLLM_KV_FABRIC_WRITE=0                 off (default): exact transfer-agent routing; 1 = enable
  TRTLLM_KV_FABRIC_WRITE_WINDOW=2          batches in flight per process
  TRTLLM_KV_FABRIC_WRITE_OPS_PER_BATCH=1024 copies per cudaMemcpyBatchAsync call
  TRTLLM_KV_FABRIC_WRITE_MIN_AVG_BYTES=65536  average fragment size gate (smaller -> transfer agent)
  TRTLLM_KV_FABRIC_WRITE_MAX_SPAN_BYTES=64MiB coalesced copy size cap
  TRTLLM_KV_FABRIC_WRITE_STATS_INTERVAL_S=60  periodic stats line (0 = off)
"""

from __future__ import annotations

import collections
import hashlib
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import msgpack
import numpy as np

from tensorrt_llm import logger

try:
    from cuda.bindings import driver as cu
    from cuda.bindings import runtime as cudart
except ImportError:  # older cuda-python layout
    from cuda import cuda as cu
    from cuda import cudart

_EXPORT_VERSION = 1
_FAB = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
_ENABLE_ENV = "TRTLLM_KV_FABRIC_WRITE"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    s = raw.strip().upper()
    mult = 1
    for suffix, m in (("KIB", 1 << 10), ("MIB", 1 << 20), ("GIB", 1 << 30)):
        if s.endswith(suffix):
            s, mult = s[: -len(suffix)], m
            break
    return int(float(s) * mult)


def fabric_write_enabled() -> bool:
    return os.environ.get(_ENABLE_ENV, "0").strip() not in ("0", "false", "False", "")


class FabricWriteConfig:
    def __init__(self) -> None:
        self.window = max(1, _env_int("TRTLLM_KV_FABRIC_WRITE_WINDOW", 2))
        self.ops_per_batch = max(1, _env_int("TRTLLM_KV_FABRIC_WRITE_OPS_PER_BATCH", 1024))
        self.min_avg_bytes = max(0, _env_int("TRTLLM_KV_FABRIC_WRITE_MIN_AVG_BYTES", 65536))
        self.max_span_bytes = max(
            1 << 20, _env_int("TRTLLM_KV_FABRIC_WRITE_MAX_SPAN_BYTES", 64 << 20)
        )
        self.stats_interval_s = float(
            os.environ.get("TRTLLM_KV_FABRIC_WRITE_STATS_INTERVAL_S", "60")
        )

    def __repr__(self) -> str:
        return (
            f"window={self.window} ops_per_batch={self.ops_per_batch} "
            f"min_avg_bytes={self.min_avg_bytes} max_span_bytes={self.max_span_bytes}"
        )


def _ck(ret, what: str):
    """Unpack a cuda-python result tuple and raise on any error."""
    if isinstance(ret, tuple):
        err, rest = ret[0], ret[1:]
    else:
        err, rest = ret, ()
    code = getattr(err, "value", err)
    if int(code) != 0:
        raise RuntimeError(f"{what} failed: {err}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else rest


def _set_device(device_id: int) -> None:
    _ck(cudart.cudaSetDevice(device_id), "cudaSetDevice")
    _ck(cudart.cudaFree(0), "cudaFree(0)")  # make the primary context current for driver calls


# ----------------------------------------------------------------------------------------------
# Process-level refusal (receiver side). Physical chunks of an exported pool must stay mapped for
# the life of the process: a pool shrink (KV pool rebalance) or a sleep release would leave the
# importer writing into memory the owner no longer reads.
# ----------------------------------------------------------------------------------------------
_refusal_lock = threading.Lock()
_refusal_reason: Optional[str] = None


def refuse(reason: str) -> None:
    """Disable the fabric export of this process (logged once)."""
    global _refusal_reason
    with _refusal_lock:
        if _refusal_reason is None:
            _refusal_reason = reason
            logger.warning(f"KV fabric write refused in this process: {reason}; using NIXL only")


def refusal_reason() -> Optional[str]:
    return _refusal_reason


def note_worker_init() -> None:
    """Called while the transfer worker is created (inside the executor's allocation scopes).

    Sleep mode allocates every executor stage inside a virtual-memory scope whose memory can be
    released later; an active scope here means sleep is enabled. KV pool rebalance is not checked
    here: the executor never pauses for a rebalance while a KV cache transceiver exists.
    """
    if not fabric_write_enabled():
        return
    try:
        from tensorrt_llm._torch import virtual_memory

        if getattr(virtual_memory, "_pool_stack", None):
            refuse("sleep mode is enabled (virtual-memory scope active at transfer setup)")
    except Exception as error:  # the module is optional for this check
        logger.debug(f"KV fabric write: sleep-mode check skipped: {error}")


# ----------------------------------------------------------------------------------------------
# Export (receiver side)
# ----------------------------------------------------------------------------------------------
def _fabric_handle_bytes(fh) -> bytes:
    data = bytes(fh.data) if hasattr(fh, "data") else bytes(fh)
    if len(data) != 64:
        raise RuntimeError(f"unexpected fabric handle size {len(data)}")
    return data


def _min_granularity(device_id: int) -> int:
    loc = cu.CUmemLocation()
    loc.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    loc.id = device_id
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = loc
    prop.requestedHandleTypes = _FAB
    try:
        gran = _ck(
            cu.cuMemGetAllocationGranularity(
                prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
            ),
            "cuMemGetAllocationGranularity",
        )
        return max(int(gran), 1 << 16)
    except RuntimeError:
        return 2 << 20


def export_pools(memory_descs, device_id: int) -> Optional[bytes]:
    """Export every physical chunk backing ``memory_descs`` [(ptr, size, ...)]; None when refused.

    Each chunk is found with cuMemGetAddressRange and checked to be one allocation end to end
    (same allocation handle at its first and last byte) and FABRIC-exportable. Unmapped holes
    are skipped (writes into them stay on the transfer agent).
    """
    if not fabric_write_enabled():
        return None
    if refusal_reason() is not None:
        return None
    t0 = time.perf_counter()
    try:
        _set_device(device_id)
        gran = _min_granularity(device_id)
        chunks: Dict[int, Tuple[int, bytes]] = {}
        holes = 0
        for desc in memory_descs:
            ptr, size = int(desc[0]), int(desc[1])
            addr, end = ptr, ptr + size
            while addr < end:
                try:
                    base, length = _ck(cu.cuMemGetAddressRange(addr), "cuMemGetAddressRange")
                except RuntimeError:
                    holes += 1
                    addr = (addr // gran + 1) * gran
                    continue
                base, length = int(base), int(length)
                if not (base <= addr < base + length):
                    raise RuntimeError(f"address range [{base:#x}, +{length}) misses {addr:#x}")
                if base not in chunks:
                    handle = _ck(
                        cu.cuMemRetainAllocationHandle(addr), "cuMemRetainAllocationHandle"
                    )
                    try:
                        prop = _ck(
                            cu.cuMemGetAllocationPropertiesFromHandle(handle),
                            "cuMemGetAllocationPropertiesFromHandle",
                        )
                        if not (int(prop.requestedHandleTypes) & int(_FAB)):
                            raise RuntimeError("pool chunk is not FABRIC-exportable")
                        last = _ck(
                            cu.cuMemRetainAllocationHandle(base + length - 1),
                            "cuMemRetainAllocationHandle(last)",
                        )
                        same = int(last) == int(handle)
                        _ck(cu.cuMemRelease(last), "cuMemRelease(last)")
                        if not same:
                            raise RuntimeError(
                                f"address range [{base:#x}, +{length}) spans more than one allocation"
                            )
                        fh = _ck(cu.cuMemExportToShareableHandle(handle, _FAB, 0), "export")
                        chunks[base] = (length, _fabric_handle_bytes(fh))
                    finally:
                        _ck(cu.cuMemRelease(handle), "cuMemRelease")
                addr = base + length
    except Exception as error:
        refuse(f"pool export failed: {error}")
        return None
    if not chunks:
        return None
    items = sorted((b, s, h) for b, (s, h) in chunks.items())
    blob = msgpack.packb(
        {"v": _EXPORT_VERSION, "dev": device_id, "chunks": [list(x) for x in items]}
    )
    total = sum(s for _, s, _ in items)
    logger.info(
        f"KV fabric export: device={device_id} chunks={len(items)} bytes={total} "
        f"holes={holes} blob={len(blob)} in {(time.perf_counter() - t0) * 1e3:.1f} ms"
    )
    return blob


def export_rank_pools(rank_info) -> Optional[bytes]:
    """Export the device-resident target (and draft, when shipped) KV pools of ``rank_info``."""
    if not fabric_write_enabled() or refusal_reason() is not None:
        return None
    from tensorrt_llm._torch.disaggregation.resource.utils import get_unique_pool_memory_descs

    descs = []
    for table in (rank_info.page_table, getattr(rank_info, "draft_page_table", None)):
        if table is not None:
            descs.extend(get_unique_pool_memory_descs(table, rank_info.device_id, "VRAM"))
    if not descs:
        return None
    return export_pools(descs, rank_info.device_id)


# ----------------------------------------------------------------------------------------------
# Import (sender side)
# ----------------------------------------------------------------------------------------------
def _import_handle(data: bytes):
    try:
        return _ck(cu.cuMemImportFromShareableHandle(data, _FAB), "cuMemImportFromShareableHandle")
    except Exception:
        fh = cu.CUmemFabricHandle()
        fh.data = data
        return _ck(cu.cuMemImportFromShareableHandle(fh, _FAB), "cuMemImportFromShareableHandle")


def _access_desc(device_id: int):
    loc = cu.CUmemLocation()
    loc.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    loc.id = device_id
    acc = cu.CUmemAccessDesc()
    acc.location = loc
    acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return acc


class _PeerMap:
    """A peer's imported chunks: VA-contiguous runs with the owner's layout."""

    def __init__(self, name: str, digest: str):
        self.name = name
        self.digest = digest
        self.ready = False
        self.disabled: Optional[str] = None
        self.retired = False
        self.inflight = 0
        self.unquiesced = 0  # requests whose copies toward this peer were never proven complete
        self.released = False
        self.reservations: List[Tuple[int, int]] = []  # (local va, size)
        self.mapped: List[Tuple[int, int]] = []  # (local va, size) of each mapped chunk
        self.rstart = np.zeros(0, dtype=np.uint64)
        self.rend = np.zeros(0, dtype=np.uint64)
        self.lbase = np.zeros(0, dtype=np.uint64)
        self.nchunks = 0
        self.nbytes = 0

    def build(self, device_id: int, chunks: List[Tuple[int, int, bytes]]) -> None:
        runs: List[List[Tuple[int, int, bytes]]] = []
        for base, size, handle in chunks:
            if runs and runs[-1][-1][0] + runs[-1][-1][1] == base:
                runs[-1].append((base, size, handle))
            else:
                runs.append([(base, size, handle)])
        acc = _access_desc(device_id)
        rstart, rend, lbase = [], [], []
        for run in runs:
            rsize = run[-1][0] + run[-1][1] - run[0][0]
            va = int(_ck(cu.cuMemAddressReserve(rsize, 0, 0, 0), "cuMemAddressReserve"))
            self.reservations.append((va, rsize))
            for base, size, handle in run:
                h = _import_handle(handle)
                try:
                    _ck(cu.cuMemMap(va + (base - run[0][0]), size, 0, h, 0), "cuMemMap")
                    self.mapped.append((va + (base - run[0][0]), size))
                finally:
                    _ck(cu.cuMemRelease(h), "cuMemRelease(import)")
            _ck(cu.cuMemSetAccess(va, rsize, [acc], 1), "cuMemSetAccess")
            rstart.append(run[0][0])
            rend.append(run[0][0] + rsize)
            lbase.append(va)
        self.rstart = np.asarray(rstart, dtype=np.uint64)
        self.rend = np.asarray(rend, dtype=np.uint64)
        self.lbase = np.asarray(lbase, dtype=np.uint64)
        self.nchunks = len(chunks)
        self.nbytes = int(sum(c[1] for c in chunks))

    def translate(self, dst: np.ndarray, sizes: np.ndarray) -> Optional[np.ndarray]:
        """Local addresses of [dst, dst + size) or None when any range is not inside one mapped run."""
        if self.rstart.size == 0:
            return None
        idx = np.searchsorted(self.rstart, dst, side="right").astype(np.int64) - 1
        if np.any(idx < 0):
            return None
        if np.any(dst + sizes > self.rend[idx]):
            return None
        return self.lbase[idx] + (dst - self.rstart[idx])

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        for va, size in self.mapped:
            try:
                _ck(cu.cuMemUnmap(va, size), "cuMemUnmap")
            except Exception as error:
                logger.warning(f"KV fabric write: unmap for peer {self.name} failed: {error}")
        for va, size in self.reservations:
            try:
                _ck(cu.cuMemAddressFree(va, size), "cuMemAddressFree")
            except Exception as error:
                logger.warning(f"KV fabric write: VA free for peer {self.name} failed: {error}")
        self.mapped.clear()
        self.reservations.clear()


# ----------------------------------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------------------------------
class _IssueError(RuntimeError):
    """A batch issue failed; ``enqueued`` is True when copies may already be on the stream."""

    def __init__(self, msg: str, enqueued: bool):
        super().__init__(msg)
        self.enqueued = enqueued


class FabricJob:
    """One eligible KV write, also handed to the sender's backend submission in place of a
    transfer-agent request (``op`` / ``remote_name`` keep its error messages readable)."""

    op = "WRITE(fabric)"

    def __init__(self, peer: _PeerMap, src, dst, sizes, nbytes: int, ndescs: int, fallback):
        self.peer = peer
        self.remote_name = peer.name
        self.src = src
        self.dst = dst
        self.sizes = sizes
        self.n = int(sizes.size)
        self.nbytes = nbytes
        self.ndescs = ndescs
        self.fallback = fallback  # builds the transfer-agent request for the same write
        self.next = 0
        self.outstanding = 0
        self.batches = 0
        self.error: Optional[str] = None
        # False when a copy of this job may still be in flight (its completion could not be
        # proven): the job then never falls back and keeps its peer mapping pinned.
        self.quiesced = True
        self.done = threading.Event()
        self.t_submit = 0.0
        self.t_done = 0.0


class FabricStatus:
    """Transfer status for a fabric write; a failed write is re-issued through the transfer agent."""

    def __init__(self, writer: "FabricWriter", job: FabricJob):
        self._writer = writer
        self._job = job
        self._fallback_status = None
        self._detail: Optional[str] = None
        self._result: Optional[bool] = None
        self._lock = threading.Lock()

    def _resolve(self, timeout_s: Optional[float]) -> Optional[bool]:
        with self._lock:
            if self._result is not None:
                return self._result
            if not self._job.done.wait(timeout_s):
                return None
            if self._job.error is None:
                self._result = True
                return True
            if not self._job.quiesced:
                self._detail = f"fabric write failed and could not be quiesced: {self._job.error}"
                self._result = False
                return False
            if self._fallback_status is None:
                self._writer._count_fallback(self._job)
                request = self._job.fallback()
                self._fallback_status = self._writer._agent.submit_transfer_requests(request)
            ok = self._fallback_status.wait()
            if not ok:
                self._detail = (
                    f"fabric write failed ({self._job.error}); NIXL fallback failed: "
                    f"{getattr(self._fallback_status, 'last_status_str', lambda: '?')()}"
                )
            self._result = bool(ok)
            return self._result

    def wait(self, timeout_ms: Optional[int] = None) -> bool:
        timeout_s = None if timeout_ms is None or timeout_ms < 0 else timeout_ms / 1000.0
        res = self._resolve(timeout_s)
        return bool(res)

    def is_completed(self) -> bool:
        if not self._job.done.is_set():
            return False
        if self._job.error is not None and not self._job.quiesced:
            return False  # completion of the failed copy was never proven, even after wait()
        if self._result is not None or self._job.error is None:
            return True
        status = self._fallback_status
        return status is not None and status.is_completed()

    def last_status_str(self) -> str:
        if self._detail is not None:
            return self._detail
        return "fabric write ok" if self._job.error is None else f"fabric error: {self._job.error}"

    @property
    def unquiesced(self) -> bool:
        """True when the write failed and its copies could not be proven complete."""
        return self._job.done.is_set() and self._job.error is not None and not self._job.quiesced

    @property
    def used_fallback(self) -> bool:
        return self._fallback_status is not None


def _abort_process(reason: str) -> None:
    os.abort()


# Replaced only by tests.
_fail_stop_hook: Callable[[str], None] = _abort_process


def fail_stop(reason: str) -> None:
    """Stop this process: a KV write may still be landing in a peer's pages and no caller can
    safely release the request. Never returns."""
    logger.critical(f"KV fabric write: fail-stop: {reason}")
    _fail_stop_hook(reason)
    threading.Event().wait()  # a hook that returns must still never release the request


def is_fabric_job(request) -> bool:
    return isinstance(request, FabricJob)


def coalesce(src: np.ndarray, dst: np.ndarray, sizes: np.ndarray, cap: int):
    """Sort by source and merge fragments contiguous on BOTH sides into copies of at most ``cap``."""
    if src.size <= 1:
        return src, dst, sizes
    order = np.argsort(src, kind="stable")
    s, d, z = src[order], dst[order], sizes[order]
    starts = [0]
    run_bytes = int(z[0])
    contiguous = (s[1:] == s[:-1] + z[:-1]) & (d[1:] == d[:-1] + z[:-1])
    zl = z.tolist()
    cl = contiguous.tolist()
    for i in range(1, s.size):
        if cl[i - 1] and run_bytes + zl[i] <= cap:
            run_bytes += zl[i]
        else:
            starts.append(i)
            run_bytes = zl[i]
    st = np.asarray(starts, dtype=np.int64)
    return s[st], d[st], np.add.reduceat(z, st)


# ----------------------------------------------------------------------------------------------
# Writer (sender side)
# ----------------------------------------------------------------------------------------------
class FabricWriter:
    @classmethod
    def create(cls, agent, device_id: int) -> Optional["FabricWriter"]:
        if not fabric_write_enabled():
            return None
        try:
            return cls(agent, device_id, FabricWriteConfig())
        except Exception as error:
            logger.warning(f"KV fabric write disabled: writer init failed: {error}")
            return None

    def __init__(self, agent, device_id: int, config: FabricWriteConfig):
        self._agent = agent
        self._device_id = device_id
        self.config = config
        self._peers: Dict[str, _PeerMap] = {}
        self._retired: List[_PeerMap] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue: collections.deque = collections.deque()
        self._stop = False
        self._dispatcher: Optional[threading.Thread] = None
        self._import_q: collections.deque = collections.deque()
        self._importer: Optional[threading.Thread] = None
        self._broken: Optional[str] = None
        self._engaged_logged = False
        self._stats = collections.Counter()
        self._last_stats = time.monotonic()
        self._logged_reasons: set = set()
        logger.info(f"KV fabric writer ready: device={device_id} {config}")

    # ---- peers ----
    def add_peer(self, name: str, export_blob: Optional[bytes]) -> None:
        if not export_blob:
            # A peer that re-registers without an export stops taking fabric writes: retire the
            # mapping of its earlier registration (kept pinned until its in-flight jobs end).
            with self._lock:
                old = self._peers.pop(name, None)
                if old is not None:
                    old.retired = True
                    self._retired.append(old)
            if old is not None:
                self._release_retired()
            self._note_once(
                f"nopeer:{name}", f"KV fabric write: peer {name} exports no fabric pools; NIXL"
            )
            return
        digest = hashlib.sha1(export_blob, usedforsecurity=False).hexdigest()
        with self._lock:
            old = self._peers.get(name)
            if old is not None and old.digest == digest:
                return
            peer = _PeerMap(name, digest)
            self._peers[name] = peer
            if old is not None:
                old.retired = True
                self._retired.append(old)
            self._import_q.append((peer, export_blob))
            if self._importer is None:
                self._importer = threading.Thread(
                    target=self._import_loop, name="kv-fabric-import", daemon=True
                )
                self._importer.start()
            self._cv.notify_all()
        self._release_retired()

    def _import_loop(self) -> None:
        try:
            _set_device(self._device_id)
        except Exception as error:
            self._broken = f"importer device setup failed: {error}"
            logger.warning(f"KV fabric write disabled: {self._broken}")
            return
        while True:
            with self._cv:
                while not self._import_q and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                peer, blob = self._import_q.popleft()
            t0 = time.perf_counter()
            try:
                meta = msgpack.unpackb(blob, strict_map_key=False)
                if meta.get("v") != _EXPORT_VERSION:
                    raise RuntimeError(f"unknown export version {meta.get('v')}")
                chunks = [(int(b), int(s), bytes(h)) for b, s, h in meta["chunks"]]
                peer.build(self._device_id, chunks)
            except Exception as error:
                peer.disabled = f"import failed: {error}"
                peer.release()
                self._stats["peers_failed"] += 1
                logger.warning(
                    f"KV fabric write: peer {peer.name} import failed, staying on NIXL: {error}"
                )
                continue
            with self._lock:
                if peer.retired:
                    peer.release()
                    continue
                peer.ready = True
                self._stats["peers_ready"] += 1
            logger.info(
                f"KV fabric write: peer {peer.name} mapped chunks={peer.nchunks} bytes={peer.nbytes} "
                f"runs={peer.rstart.size} in {(time.perf_counter() - t0) * 1e3:.1f} ms"
            )

    def _release_retired(self) -> None:
        with self._lock:
            done = [p for p in self._retired if p.inflight == 0 and (p.ready or p.disabled)]
            self._retired = [p for p in self._retired if p not in done]
            for p in done:
                p.ready = False  # never pinned again; prepare() also checks retired
        for p in done:
            p.release()

    def peer_ready(self, name: str) -> bool:
        p = self._peers.get(name)
        return p is not None and p.ready and p.disabled is None

    # ---- routing ----
    def _ineligible(self, reason: str) -> None:
        self._stats[f"inelig_{reason}"] += 1

    def prepare(self, write_meta, fallback: Callable) -> Optional[FabricJob]:
        """A FabricJob for ``write_meta`` or None (the caller then uses the transfer agent)."""
        if self._broken is not None:
            self._ineligible("broken")
            return None
        n = int(write_meta.sizes.size)
        if n == 0:
            return None
        if write_meta.dst_device_id is None:
            self._ineligible("no_dst_device")
            return None
        nbytes = int(write_meta.sizes.sum())
        if nbytes < self.config.min_avg_bytes * n:
            self._ineligible("small_fragments")
            return None
        # Eligibility and the pin are one step under the lock: a pinned mapping is never released,
        # and a retired or released mapping is never pinned.
        with self._lock:
            peer = self._peers.get(write_meta.peer_name)
            if peer is None:
                reason = "no_peer_export"
            elif peer.disabled is not None:
                reason = "peer_disabled"
            elif not peer.ready or peer.retired or peer.released:
                reason = "peer_mapping"
            else:
                reason = None
                peer.inflight += 1
        if reason is not None:
            self._ineligible(reason)
            return None
        try:
            sizes = write_meta.sizes.astype(np.uint64)
            dst = peer.translate(write_meta.dst_ptrs.astype(np.uint64), sizes)
            if dst is None:
                self._ineligible("dst_unmapped")
                self._unpin(peer)
                return None
            src, dst, sizes = coalesce(
                write_meta.src_ptrs.astype(np.uint64), dst, sizes, self.config.max_span_bytes
            )
        except BaseException:
            self._unpin(peer)
            raise
        return FabricJob(peer, src, dst, sizes, nbytes, n, fallback)

    def _unpin(self, peer: _PeerMap) -> None:
        with self._lock:
            peer.inflight -= 1
        self._release_retired()

    def submit(self, job: FabricJob) -> FabricStatus:
        job.t_submit = time.perf_counter()
        with self._cv:
            if self._dispatcher is None:
                self._dispatcher = threading.Thread(
                    target=self._dispatch_loop, name="kv-fabric-dispatch", daemon=True
                )
                self._dispatcher.start()
            if self._broken is not None or self._stop:
                job.error = self._broken or "writer stopped"
                self._finish_locked(job)
            else:
                self._queue.append(job)
                self._cv.notify_all()
        return FabricStatus(self, job)

    # ---- dispatcher ----
    def _issue(self, job: FabricJob, a: int, b: int, stream, event) -> None:
        try:
            self._enqueue_batch(job, a, b, stream)
        except Exception as error:
            raise _IssueError(str(error), enqueued=True) from error
        try:
            _ck(cudart.cudaEventRecord(event, stream), "cudaEventRecord")
        except Exception as error:
            raise _IssueError(str(error), enqueued=True) from error

    def _enqueue_batch(self, job: FabricJob, a: int, b: int, stream) -> None:
        _ck(
            cudart.cudaMemcpyBatchAsync(
                job.dst[a:b].tolist(),
                job.src[a:b].tolist(),
                job.sizes[a:b].tolist(),
                b - a,
                [self._attr],
                [0],
                1,
                stream,
            ),
            "cudaMemcpyBatchAsync",
        )

    def _finish_locked(self, job: FabricJob) -> None:
        job.t_done = time.perf_counter()
        peer = job.peer
        if job.quiesced:
            peer.inflight -= 1
        else:
            peer.unquiesced += 1
            logger.error(
                f"KV fabric write: request to peer {peer.name} could not be quiesced after "
                f"{job.error}; it fails without a NIXL retry and its mapping stays pinned"
            )
        if job.error is None:
            self._stats["requests"] += 1
            self._stats["bytes"] += job.nbytes
            self._stats["descs"] += job.ndescs
            self._stats["ops"] += job.n
            self._stats["batches"] += job.batches
            self._stats["lat_ms_sum"] += (job.t_done - job.t_submit) * 1e3
        else:
            self._stats["copy_errors"] += 1
            if peer.disabled is None:
                peer.disabled = f"copy error: {job.error}"
                logger.warning(
                    f"KV fabric write: copy error toward peer {peer.name}; this request and all "
                    f"later ones to the peer use NIXL: {job.error}"
                )
        job.done.set()

    def _dispatch_loop(self) -> None:
        cfg = self.config
        slots: List[Optional[FabricJob]] = [None] * cfg.window
        inflight: collections.deque = collections.deque()
        try:
            _set_device(self._device_id)
            flags = getattr(cudart, "cudaStreamNonBlocking", 1)
            streams = [
                _ck(cudart.cudaStreamCreateWithFlags(flags), "cudaStreamCreateWithFlags")
                for _ in range(cfg.window)
            ]
            eflags = int(getattr(cudart, "cudaEventBlockingSync", 1)) | int(
                getattr(cudart, "cudaEventDisableTiming", 2)
            )
            events = [
                _ck(cudart.cudaEventCreateWithFlags(eflags), "cudaEventCreateWithFlags")
                for _ in range(cfg.window)
            ]
            attr = cudart.cudaMemcpyAttributes()
            attr.srcAccessOrder = cudart.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
            try:
                attr.flags = cudart.cudaMemcpyFlags.cudaMemcpyFlagPreferOverlapWithCompute
            except Exception:
                attr.flags = int(cudart.cudaMemcpyFlags.cudaMemcpyFlagPreferOverlapWithCompute)
            self._attr = attr
        except Exception as error:
            self._fail_all(f"dispatcher setup failed: {error}", slots)
            return

        def complete(i: int, err: Optional[str], quiesced: bool = True) -> None:
            job = slots[i]
            slots[i] = None
            job.outstanding -= 1
            if not quiesced:
                job.quiesced = False
            if err is not None and job.error is None:
                job.error = err
            if job.outstanding == 0 and (job.next == job.n or job.error is not None):
                with self._lock:
                    if not job.done.is_set():
                        self._finish_locked(job)
                self._release_retired()

        def drain() -> None:
            """Wait for every tracked batch; a failed wait leaves its job unquiesced."""
            while inflight:
                k = inflight.popleft()
                e = sync(k)
                complete(k, e, quiesced=e is None)

        def sync(i: int) -> Optional[str]:
            try:
                _ck(cudart.cudaEventSynchronize(events[i]), "cudaEventSynchronize")
                return None
            except Exception as error:
                return str(error)

        while True:
            with self._cv:
                while not self._queue and not inflight and not self._stop:
                    self._cv.wait(timeout=1.0)
                    self._maybe_log_stats_locked()
                if self._stop and not self._queue and not inflight:
                    break
            # Issue: FIFO head first, at most `window` batches in flight process wide.
            while len(inflight) < cfg.window:
                with self._lock:
                    job = self._queue[0] if self._queue else None
                if job is None:
                    break
                i = slots.index(None)
                a, b = job.next, min(job.n, job.next + cfg.ops_per_batch)
                try:
                    self._issue(job, a, b, streams[i], events[i])
                except Exception as error:
                    # Own everything already issued before the request ends: the tracked batches
                    # (this job's earlier ones included) and, when the failing batch may have
                    # been enqueued, its stream. Only then may the NIXL fallback or an unmap run.
                    drain()
                    if getattr(error, "enqueued", True):
                        try:
                            _ck(cudart.cudaStreamSynchronize(streams[i]), "cudaStreamSynchronize")
                        except Exception as sync_error:
                            job.quiesced = False
                            error = RuntimeError(f"{error}; stream drain failed: {sync_error}")
                    with self._lock:
                        if self._queue and self._queue[0] is job:
                            self._queue.popleft()
                    job.error = str(error)
                    if job.outstanding == 0:
                        with self._lock:
                            if not job.done.is_set():
                                self._finish_locked(job)
                        self._release_retired()
                    continue
                job.next = b
                job.outstanding += 1
                job.batches += 1
                slots[i] = job
                inflight.append(i)
                if b == job.n:
                    with self._lock:
                        if self._queue and self._queue[0] is job:
                            self._queue.popleft()
            if inflight:
                i = inflight.popleft()
                err = sync(i)
                complete(i, err, quiesced=err is None)
                if err is not None:
                    # A failed event wait is a sticky context error: stop using the writer.
                    drain()
                    self._fail_all(f"event wait failed: {err}", slots)
                    return
            if not self._engaged_logged and self._stats["requests"] > 0:
                self._engaged_logged = True
                logger.info(
                    f"KV fabric write engaged: device={self._device_id} {cfg} "
                    f"first request descs={self._stats['descs']} ops={self._stats['ops']} "
                    f"bytes={self._stats['bytes']} batches={self._stats['batches']}"
                )
            with self._lock:
                self._maybe_log_stats_locked()
        self._release_retired()

    def _fail_all(self, reason: str, slots) -> None:
        with self._lock:
            self._broken = reason
            pending = list(self._queue)
            self._queue.clear()
            for job in pending + [s for s in slots if s is not None]:
                if not job.done.is_set():
                    job.error = job.error or reason
                    self._finish_locked(job)
        logger.warning(f"KV fabric write disabled for this process: {reason}")

    # ---- stats / lifecycle ----
    def _count_fallback(self, job: FabricJob) -> None:
        with self._lock:
            self._stats["fallbacks"] += 1

    def _note_once(self, key: str, msg: str) -> None:
        if key not in self._logged_reasons:
            self._logged_reasons.add(key)
            logger.info(msg)

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._stats)

    def _maybe_log_stats_locked(self) -> None:
        iv = self.config.stats_interval_s
        now = time.monotonic()
        if iv <= 0 or now - self._last_stats < iv:
            return
        self._last_stats = now
        s = self._stats
        if not s:
            return
        reqs = s["requests"]
        inelig = " ".join(f"{k[7:]}={v}" for k, v in sorted(s.items()) if k.startswith("inelig_"))
        logger.info(
            f"KV fabric write stats: device={self._device_id} requests={reqs} bytes={s['bytes']} "
            f"descs={s['descs']} ops={s['ops']} batches={s['batches']} "
            f"lat_ms_avg={(s['lat_ms_sum'] / reqs) if reqs else 0:.2f} fallbacks={s['fallbacks']} "
            f"copy_errors={s['copy_errors']} peers_ready={s['peers_ready']} "
            f"peers_failed={s['peers_failed']} ineligible[{inelig}]"
        )

    # Bound on how long shutdown waits for issued copies to land before it fail-stops.
    _shutdown_timeout_s = 300.0

    def shutdown(self) -> None:
        """Return only once no fabric copy can still touch local or peer memory.

        The caller deregisters the source pools and shuts the transfer agent down next, so every
        issued batch must have landed first. Queued requests that never issued end with an error
        (their callers fall back or fail). If the dispatcher does not drain in time, or a request
        ended without proof that its copies completed, the process fail-stops instead of returning.
        """
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        deadline = time.monotonic() + self._shutdown_timeout_s
        dispatcher = self._dispatcher
        if dispatcher is not None:
            while dispatcher.is_alive():
                left = deadline - time.monotonic()
                if left <= 0:
                    fail_stop("fabric dispatcher still has copies in flight at shutdown")
                    return
                dispatcher.join(timeout=min(left, 30.0))
                if dispatcher.is_alive():
                    logger.warning("KV fabric write: shutdown waiting for in-flight copies")
        with self._lock:
            unquiesced = [
                p.name for p in list(self._peers.values()) + self._retired if p.unquiesced
            ]
        if unquiesced:
            fail_stop(f"unquiesced fabric writes toward {unquiesced} at shutdown")
            return
        importer = self._importer
        if importer is not None and hasattr(importer, "is_alive"):
            importer.join(timeout=max(0.0, deadline - time.monotonic()))
            if importer.is_alive():
                # Mapping peer memory touches neither the source pools nor the agent; leave the
                # importer's mappings in place rather than unmapping under it.
                logger.warning("KV fabric write: importer still running at shutdown; mappings kept")
                return
        with self._lock:
            peers = list(self._peers.values()) + self._retired
            self._peers.clear()
            self._retired = []
        for p in peers:
            if p.inflight == 0:
                p.release()

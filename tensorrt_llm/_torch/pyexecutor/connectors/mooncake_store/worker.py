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
"""Worker side of the Mooncake store KV cache connector.

One worker per rank owns a `MooncakeDistributedStore` handle and moves pages
between that pool and its own GPU KV cache. It is also the only place that
knows how a page is addressed and how a key is spelled, which is why the
leader, colocated with rank 0's worker, asks it to run prefix lookups.

Loads must land before the request computes: the runtime has already counted
those tokens, so a failed load is a wrong answer rather than a slow one. A
confirmed prefix reservation loads on a loader thread while its request waits
outside the batch (`SYNC_LOAD_ENV` loads it in the executor's step instead).

Saves are asynchronous and gated on a CUDA event, since the pages are complete
only once the forward pass that wrote them has retired and blocking the
executor loop on an RDMA write is the cost the store exists to avoid. The
scheduler reports such a request as saving asynchronously, which keeps its
pages pinned until `get_finished` says the writes landed. A save that fails is
logged and dropped: a write that did not land costs a future cache miss and
nothing else, so the request retires either way. Only a save or loader thread
failing to start is fatal, because that would strand every request behind pages
nothing will release or loads nothing will run.

A capacity-only worker opens its handle and stops there, with no layout, no
buffer registration and no save or loader threads, so a node can lend host memory to the
pool without an HCA that can pin GPU pages.
"""

import dataclasses
import heapq
import math
import os
import struct
import threading
import time
import traceback
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from operator import attrgetter
from queue import Empty, Queue
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from tensorrt_llm._utils import mpi_rank, mpi_world_size
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttnLifeCycle

from ..kv_cache_connector import KvCacheConnectorWorker
from ..kv_cache_layout import KvCacheLayout
from .addressing import PageAddressing
from .config import CONFIG_PATH_ENV, MooncakeStoreConnectorConfig, local_address, pool_config
from .keys import BlockHashChain, KeyNamespace, chain_memo_stats
from .ledger import record_segment
from .digest import PageDigestLog
from .save_retry import put_with_retries
from .snapshot import SnapshotRing
from .softpin import SOFTPIN_ENV, SoftPinStore, maybe_softpin_store
from .stored_set import LocalStoredSet
from .metadata import MooncakeStoreMetadata, PageTransfer, PrefixLoadTransfers, RequestTransfers, WindowRetire
from .staging import (
    MAX_STAGING_BUFFER_BYTES,
    HostStagingPool,
    batch_copy_mode,
    describe_batch_for_get,
    plan_slot_geometry,
    stage_batch_for_put,
    unstage_batch_after_get,
)
from .staging import sync_stream as _sync_stream
from .validation import validate_layout, validate_llm_args, validate_node_budget

__all__ = ["MooncakeStoreConnectorWorker", "PrefixHit", "resolve_local_worker"]

#: Set by the worker's constructor so the leader, which the executor builds in
#: the same process on rank 0, can reach the store handle without a second
#: connection. `py_executor_creator` constructs the two concurrently, so the
#: leader waits on `_LOCAL_WORKER_READY`.
_LOCAL_WORKER: Optional["MooncakeStoreConnectorWorker"] = None
_LOCAL_WORKER_READY = threading.Event()

#: Seconds `shutdown` gives the save thread, and each loader thread, to drain its queue and return.
#: Sized for a backlog of pages over a congested fabric rather than a healthy
#: one, since what follows the wait is only safe once the thread has stopped.
SAVE_DRAIN_TIMEOUT = 30.0

#: Seconds between the loader, save and lookup statistics reports.
LOAD_STATS_INTERVAL = 30.0

#: Asynchronous prefix-load threads per loading rank, each with a staging pool
#: and stream of its own.
LOAD_THREADS_ENV = "TRTLLM_MOONCAKE_STORE_LOAD_THREADS"
DEFAULT_LOAD_THREADS = 2


#: Set to 1 for reserved prefixes to load synchronously, in the executor's step that computes the rest of the prompt,
#: instead of on loader threads; read by the scheduler, and here to start no loader threads. Asynchronous loads (the
#: default) park the request until its pages are on the device while the others keep running, so a load costs that
#: request time instead of the batch.
SYNC_LOAD_ENV = "TRTLLM_MOONCAKE_STORE_SYNC_LOAD"

#: Window-page save snapshots (mcgb2, GB300 MSO): `wait_for_save` copies every sliding-window page a pass saves
#: into a pinned, store-registered host slot with `cudaMemcpyAsync` on the execution stream, ahead of the pass's
#: event, and the save thread puts from that slot. Without it the save thread reads the live device page after the
#: event, but the commit that closes a prompt (and every decode step after it) releases window blocks as the window
#: slides, so a page can be recycled and rewritten before the thread reads it: the store then holds bytes that are
#: neither this prompt's KV nor detectable by the save/load digests (both see the same wrong bytes). Full-attention
#: pages stay held by the request until its saves land, so only window groups need it, which keeps the copy to the
#: ~2 windows a prompt end saves. `TRTLLM_MOONCAKE_STORE_WINDOW_SNAPSHOT=0` turns it off; the slot count is
#: `TRTLLM_MOONCAKE_STORE_WINDOW_SNAPSHOT_SLOTS` (default 2048 pages per rank).
WINDOW_SNAPSHOT_ENV = "TRTLLM_MOONCAKE_STORE_WINDOW_SNAPSHOT"
WINDOW_SNAPSHOT_SLOTS_ENV = "TRTLLM_MOONCAKE_STORE_WINDOW_SNAPSHOT_SLOTS"
DEFAULT_WINDOW_SNAPSHOT_SLOTS = 2048

#: Issue the window-page snapshot copies on a side stream ordered exactly where they would run on the executor's
#: stream (the side stream waits for the executor stream, then the executor stream waits for the side stream), when
#: the executor runs on the legacy default stream. `cudaMemcpyBatchAsync` refuses the default stream, so there the
#: snapshot falls back to one Python-issued `cudaMemcpy(2D)Async` per page-region run on the executor thread (about a
#: hundred calls per prompt end, on every rank); on the side stream it is one batched call. Off by default.
SNAPSHOT_STREAM_ENV = "TRTLLM_MOONCAKE_STORE_SNAPSHOT_STREAM"

#: Whether a lookup that finds no window within the rewind looks further back for an
#: earlier prompt end whose window is stored, instead of reporting a miss (on by default; 0 turns it off).
WINDOW_FALLBACK_ENV = "TRTLLM_MOONCAKE_STORE_WINDOW_FALLBACK"

#: Keys per existence-probe call, 0 for one call. A lookup over a long prefix probes one key per block per rank, which
#: the master answers on one RPC thread while the scheduler waits; split, the calls run concurrently on the master's
#: thread pool.
PROBE_CHUNK_ENV = "TRTLLM_MOONCAKE_STORE_PROBE_CHUNK"
DEFAULT_PROBE_CHUNK = 4096
#: Concurrent probe calls per lookup.
PROBE_THREADS = 4

#: Prompts that computed at least this many tokens list every full-attention block again at their last chunk, so the
#: save path refreshes the chain's head and rewrites what the pool evicted during the prefill. 0 turns it off. Read by
#: the scheduler.
RESAVE_MIN_TOKENS_ENV = "TRTLLM_MOONCAKE_STORE_RESAVE_MIN_TOKENS"

#: Commit-swap guard (on by default; 0 restores the old hand-off, for A/B and repro runs only). When a pass completes a
#: full block that KV cache manager V2's radix tree already holds (another request computed or loaded it), the commit
#: that follows the pass rebases the request onto the tree's page and frees the page the request computed
#: (kvCache.cpp, "Existing block: rebase"), so that slot can be handed out again while the save list still names it.
#: `wait_for_save` therefore keeps a pass's save list (its window snapshots are already taken) until the commit has run
#: (`get_finished`, every iteration), then resolves every listed page against this rank's pages again: a page the
#: commit swapped is saved from the surviving page, which the request now holds, and a block the request no longer
#: has a page for is dropped.
COMMIT_SWAP_GUARD_ENV = "TRTLLM_MOONCAKE_STORE_COMMIT_SWAP_GUARD"

#: Slot lookups by block ordinal, the fast way (off by default; lanes/mc/PGFIX_FAST.md). `_localize` (rank 0's slots ->
#: this rank's) and the commit-swap re-check normally build each listed request's whole unmasked page table (every
#: block of every layer group, walked in C++ and converted to Python lists) and resolve page by page in Python, every
#: iteration on every rank. 1 = ask the KV cache manager for just the ordinals the pass names, per layer group, from
#: the page index row it publishes for the pages a request holds locked (`KVCacheManagerV2.
#: lookup_local_page_indices_for_id`; a group with an ordinal the row does not publish is answered by the full query),
#: and compare in bulk. Same slots, so the same pages are read and written. 2 = verify: every lookup also runs the full
#: query and the full query's slots are used; differences are counted on the save statistics line.
PGFIX_FAST_ENV = "TRTLLM_MOONCAKE_STORE_PGFIX_FAST"
_BLOCK_ORDINAL = attrgetter("block_ordinal")
_LAYER_GROUP = attrgetter("layer_group_id")
_PAGE_INDEX = attrgetter("page_index")

#: Window retirement (port of Zheyu Fu's 7d1ff7a236 / 59f187af63 / a224b031b3; off by default). Keep only the newest
#: N prompt-end windows of a chain in the pool (0 or unset = keep all, no behavior change). A request that stores
#: its prompt-end window records a small link object in the pool, from that end to the prompt end it resumed from
#: plus the chain's floor; the save thread follows N - 1 links back and removes the sliding-window pages of the
#: window found there, from max(its window start, sinks, floor) up to where the next kept window starts, plus its
#: link. Full-attention pages are never removed. Read here and by the scheduler.
WINDOW_KEEP_ENV = "TRTLLM_MOONCAKE_STORE_WINDOW_KEEP"
#: Seconds to wait before removing a retired window (default 65). Removal is non-forced, so pages read or probed
#: within the master's read lease (60 s in our deployments) stay; keep this above the lease.
WINDOW_RETIRE_DELAY_ENV = "TRTLLM_MOONCAKE_STORE_WINDOW_RETIRE_DELAY_S"
#: Prompt ends that resumed from fewer tokens than this get no link (default 2048), so a chain's first window, which
#: may share its pages with other sessions' (a common preamble), is never retired; nothing below the chain's floor
#: (its first linked resume point) is removed.
WINDOW_LINK_MIN_TOKENS_ENV = "TRTLLM_MOONCAKE_STORE_WINDOW_LINK_MIN_TOKENS"


# MCB diagnostics (TRTLLM_MCB_DIAG=1, logging only, rank 0 of each TP group): per async load
# ([MCB] load), per synchronous in-pass load ([MCB] syncload), per save batch ([MCB] save) and per request
# the time its KV stayed held by outstanding saves after it closed ([MCB] savehold).
_MCB_DIAG = os.environ.get("TRTLLM_MCB_DIAG", "0") == "1"


#: ldfix F2 (off by default): an asynchronous load starts right after the executor has launched a forward, so it
#: overlaps a forward already queued on the GPU instead of starving the next forward's launch of the GIL
#: (lanes/mc/STORE_CTX_OVERHEAD.md section 3: a forward launched before a load is barely slowed, one launched
#: during it stalls until the load ends). A loader that takes a load waits for the next `wait_for_save` (called
#: right after the forward returns on the host) unless one happened in the last `_GRACE_MS` (default 30), the
#: executor looks idle (no launch for `_IDLE_MS`, default 1000), or `_TIMEOUT_MS` (default 400) passes. Only when
#: a load starts changes; what it loads, on which stream and behind which event does not.
LOAD_AFTER_LAUNCH_ENV = "TRTLLM_MOONCAKE_STORE_LOAD_AFTER_LAUNCH"

#: lkp L1 (off by default; 1 turns it on): the full-attention leading run is found in two probe rounds instead of one
#: probe of every block. Round 1 asks for a few sentinel blocks (the first candidate, every ~sqrt(n)-th block and the
#: last); round 2 asks only for the blocks below the first sentinel with a missing key. The answer is the first block
#: with a missing key, exactly as one probe of every key finds it, but the blocks past that miss are never asked
#: about: a cold prompt costs its sentinels instead of all of its keys, and a warm one stops within a stride of the
#: end of its run. Every rank and layer group is still asked about for every block below the miss.
#: (lanes/mc/STORE_LOOKUP_KEYS.md; the profile in lanes/mc/STORE_PROFILE.md section 4.)
LOOKUP_STAGED_ENV = "TRTLLM_MOONCAKE_STORE_LOOKUP_STAGED"
#: Lookups over at most this many blocks keep the one-round probe: a second round costs more than they would save.
STAGED_MIN_SPAN = 64
STAGED_MIN_STRIDE = 8
STAGED_MAX_STRIDE = 256

#: lkp L2 (off by default; milliseconds, 0 = off): a lookup reuses the leading run an earlier lookup confirmed, for
#: this long, and probes only past it. A request whose reservation is released when the batch does not take it looks
#: its prefix up again on the next pass; with this, that repeat asks only about the blocks past the run it already
#: confirmed (which may have landed since). No background probing: nothing is asked that a lookup did not need.
#: Runs are keyed by their first and last chain hashes, so a reused run is the same tokens at the same ordinals; a
#: run built partly from a reused one keeps the older confirmation time, so no answer outlives the TTL. Keep the TTL
#: well below the master's KV lease (`batch_is_exist` grants the default KV lease, 5 s by default), which is what
#: already keeps a confirmed page from eviction between a lookup and its load.
LOOKUP_REUSE_MS_ENV = "TRTLLM_MOONCAKE_STORE_LOOKUP_REUSE_MS"
#: Runs remembered per first block, and first blocks remembered.
LOOKUP_REUSE_RUNS_PER_START = 8
LOOKUP_REUSE_STARTS = 4096

#: lkp L3 (off by default; 1 turns it on, needs ldfix F3 `TRTLLM_MOONCAKE_STORE_LOCAL_STORED_SET=1`): a page this rank
#: loaded from the pool goes into its local-stored set, so the save path's RESAVE re-listing of a loaded prefix skips
#: the `batch_is_exist` that would only find it present again. Same contract as F3: the set is consulted by saves only.
STORED_FROM_LOADS_ENV = "TRTLLM_MOONCAKE_STORE_LOCAL_STORED_FROM_LOADS"


def _leading_run_staged(
    block_count: int, start: int, probe_blocks
) -> Optional[int]:
    """The first block in ``[start, block_count)`` that ``probe_blocks`` reports incomplete, else ``block_count``.

    ``probe_blocks(blocks)`` answers, per block ordinal in ``blocks``, whether every key of that block is present,
    or None when the probe failed (then so does this). Blocks below ``start`` are known complete. Equal to probing
    every block at once and taking the first incomplete one; see `LOOKUP_STAGED_ENV`.
    """
    if start >= block_count:
        return block_count
    span = block_count - start
    if span <= STAGED_MIN_SPAN:
        complete = probe_blocks(list(range(start, block_count)))
        if complete is None:
            return None
        return next((start + i for i, ok in enumerate(complete) if not ok), block_count)
    stride = min(STAGED_MAX_STRIDE, max(STAGED_MIN_STRIDE, math.isqrt(span)))
    sentinels = sorted({start, block_count - 1, *range(start + stride, block_count, stride)})
    complete = probe_blocks(sentinels)
    if complete is None:
        return None
    bound = next((block for block, ok in zip(sentinels, complete) if not ok), block_count)
    probed = set(sentinels)
    rest = [block for block in range(start, bound) if block not in probed]
    if rest:
        complete = probe_blocks(rest)
        if complete is None:
            return None
        miss = next((block for block, ok in zip(rest, complete) if not ok), None)
        if miss is not None:
            return miss
    return bound


class _RunReuse:
    """Leading runs earlier lookups confirmed, reusable for `LOOKUP_REUSE_MS_ENV` milliseconds (lkp L2).

    A run is the blocks ``block_hashes[:run]`` of one lookup over one set of layer groups, every rank's key present.
    Chain hashes name a block together with everything before it, so a later candidate list that starts with the
    same hash and has the same hash at ``run - 1`` covers the same blocks. Executor thread only.
    """

    def __init__(self, ttl: float, clock=time.monotonic):
        self._ttl = ttl
        self._clock = clock
        # (groups, first hash) -> [(run, last hash, confirmed at)], most recent last.
        self._runs: "OrderedDict[Tuple[Tuple[int, ...], bytes], List[Tuple[int, bytes, float]]]" = OrderedDict()
        self.reused_lookups = 0
        self.reused_blocks = 0

    @classmethod
    def from_env(cls) -> Optional["_RunReuse"]:
        ttl_ms = float(os.environ.get(LOOKUP_REUSE_MS_ENV, "0") or 0)
        return cls(ttl_ms / 1e3) if ttl_ms > 0 else None

    def known(self, groups: Tuple[int, ...], block_hashes: Sequence[bytes]) -> Tuple[int, float]:
        """The longest fresh run that is a prefix of ``block_hashes``, and when it was confirmed (0 and now if none)."""
        now = self._clock()
        if not block_hashes:
            return 0, now
        entries = self._runs.get((groups, block_hashes[0]))
        best, confirmed = 0, now
        if entries:
            fresh = [entry for entry in entries if now - entry[2] <= self._ttl]
            for run, last, at in fresh:
                if best < run <= len(block_hashes) and block_hashes[run - 1] == last:
                    best, confirmed = run, at
            if len(fresh) != len(entries):
                if fresh:
                    entries[:] = fresh
                else:
                    del self._runs[(groups, block_hashes[0])]
        if best:
            self.reused_lookups += 1
            self.reused_blocks += best
        return best, confirmed

    def record(self, groups: Tuple[int, ...], block_hashes: Sequence[bytes], run: int, confirmed: float) -> None:
        if run <= 0:
            return
        key = (groups, block_hashes[0])
        entries = self._runs.get(key)
        if entries is None:
            entries = self._runs[key] = []
        else:
            self._runs.move_to_end(key)
        last = block_hashes[run - 1]
        for entry in entries:
            if entry[0] == run and entry[1] == last:
                # The same run confirmed twice: the newer confirmation stands.
                confirmed = max(confirmed, entry[2])
        entries[:] = [entry for entry in entries if not (entry[0] == run and entry[1] == last)]
        entries.append((run, last, confirmed))
        del entries[:-LOOKUP_REUSE_RUNS_PER_START]
        while len(self._runs) > LOOKUP_REUSE_STARTS:
            self._runs.popitem(last=False)
#: Rank-local lookups (off by default; 1 turns them on, tensor parallelism without attention DP only). Without it the
#: leader probes every rank's keys of a prefix while the other ranks wait in the result broadcast. With it every rank
#: hashes the prompt and probes only its own shard's keys, concurrently, and the ranks agree through an allreduce(MIN)
#: of the leading hit (and of the per-block window flags), which is exactly the conjunction the leader's probe of all
#: shards computes. A rank whose probe fails contributes a miss, so the agreed hit never exceeds any rank's own.
#: The leader still decides and broadcasts the offer; only the probing moves (lanes/mc/TEP4_BCAST_SHARE.md, option 2).
RANK_LOCAL_LOOKUP_ENV = "TRTLLM_MOONCAKE_STORE_RANK_LOCAL_LOOKUP"
#: Requests whose hash chain a follower rank keeps between lookups (the leader keeps its own in the request state).
RANK_LOCAL_CHAIN_CACHE_SIZE = 1024


class _LaunchGate:
    """Forward-launch notifications from the executor thread, waited on by loader threads."""

    def __init__(self, grace: float, timeout: float, idle: float, clock=time.monotonic):
        self._grace = grace
        self._timeout = timeout
        self._idle = idle
        self._clock = clock
        self._cond = threading.Condition()
        self._seq = 0
        self._last: Optional[float] = None

    @classmethod
    def from_env(cls) -> Optional["_LaunchGate"]:
        if os.environ.get(LOAD_AFTER_LAUNCH_ENV, "0") != "1":
            return None

        def ms(suffix: str, default: float) -> float:
            return float(os.environ.get(f"{LOAD_AFTER_LAUNCH_ENV}_{suffix}_MS", default)) / 1000.0

        return cls(ms("GRACE", 30), ms("TIMEOUT", 400), ms("IDLE", 1000))

    def launched(self) -> None:
        """The executor has just launched a forward (its kernels are queued)."""
        with self._cond:
            self._seq += 1
            self._last = self._clock()
            self._cond.notify_all()

    def wait(self) -> str:
        """Block until a load may start; returns why: grace, idle, launch or timeout."""
        with self._cond:
            now = self._clock()
            if self._last is None or now - self._last > self._idle:
                return "idle"
            if now - self._last <= self._grace:
                return "grace"
            seq = self._seq
            if self._cond.wait_for(lambda: self._seq != seq, self._timeout):
                return "launch"
            return "timeout"


def _load_thread_count() -> int:
    count = int(os.environ.get(LOAD_THREADS_ENV, DEFAULT_LOAD_THREADS))
    if count < 1:
        raise ValueError(f"{LOAD_THREADS_ENV} must be at least 1, got {count}")
    return count


def resolve_local_worker(timeout: float = 60.0) -> "MooncakeStoreConnectorWorker":
    """The worker living in this process, once it has been constructed.

    Args:
        timeout: Seconds to wait. Exceeding it means construction failed.
    """
    if not _LOCAL_WORKER_READY.wait(timeout):
        raise RuntimeError(
            "The mooncake-store leader could not find a worker in its process. "
            "The leader only runs on rank 0, where the executor also builds a "
            "worker, so this means worker construction failed."
        )
    assert _LOCAL_WORKER is not None
    return _LOCAL_WORKER


def _open_store(config: MooncakeStoreConnectorConfig):
    """Connect to the Mooncake master and return a live store handle.

    Returns:
        The store handle, and the host the segment is registered under.
    """
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as exc:
        raise ImportError(
            "The mooncake-store connector needs the Mooncake Python bindings, "
            "which `tensorrt-llm` pulls in as `mooncake-transfer-engine-cuda13`, "
            "so reinstall that package if this environment dropped it. The C++ "
            "transfer engine built into the container is a different component "
            "and does not provide MooncakeDistributedStore."
        ) from exc

    store = MooncakeDistributedStore()
    hostname = config.local_hostname or local_address(config.master_server_address)
    setup_kwargs = {}
    if config.tenant_id:
        setup_kwargs["tenant_id"] = config.tenant_id
    status = store.setup(
        hostname,
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
        **setup_kwargs,
    )
    if status != 0:
        raise RuntimeError(
            f"MooncakeDistributedStore.setup failed with status {status} "
            f"(master={config.master_server_address!r}, "
            f"metadata={config.metadata_server!r}, protocol={config.protocol!r}, "
            f"global_segment_size={config.global_segment_size}). The master "
            "must already be accepting connections; the protocol and device "
            "must be usable from this host; and this node must have the "
            "segment's worth of memory to spare once every rank on it has "
            f"claimed one. Check the config named by {CONFIG_PATH_ENV}."
        )
    return store, hostname


def _interleaved_keys(
    namespaces: Sequence[KeyNamespace], block_hashes: Sequence[bytes]
) -> List[str]:
    """Every namespace's key of every block, block by block in namespace order.

    Built per namespace from its fixed prefix, then interleaved: a lookup over a long prompt builds one key per block
    per rank on the scheduler's path, and calling `KeyNamespace.key` per key costs several times as much.
    """
    if not namespaces or not block_hashes:
        return []
    hexes = [block_hash.hex() for block_hash in block_hashes]
    columns = []
    for namespace in namespaces:
        prefix = namespace.prefix + "/"
        columns.append([prefix + digest for digest in hexes])
    if len(columns) == 1:
        return columns[0]
    return [key for row in zip(*columns) for key in row]


def _mpi_min_reducer():
    """An elementwise allreduce(MIN) of small int lists over a duplicate of the executor's communicator."""
    import numpy as np
    from mpi4py import MPI

    from tensorrt_llm._utils import mpi_comm

    comm = mpi_comm().Dup()

    def reduce(values: List[int]) -> List[int]:
        buffer = np.asarray(values, dtype=np.int64)
        comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MIN)
        return buffer.tolist()

    return reduce


def _batched(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _stream_handle(stream: Optional[torch.cuda.Stream]) -> int:
    """The raw CUDA stream handle behind a torch stream.

    `None` maps to 0, the default stream, which is what the runtime passes when
    it has no stream of its own to offer.
    """
    if stream is None:
        return 0
    return int(stream.cuda_stream)



def _save_retry_disabled() -> bool:
    """Kill switch of the save retry (same names as our e930 port)."""
    return (
        os.environ.get("TRTLLM_M31_DISABLE_STORE_SAVE_RETRY") == "1"
        or os.environ.get("TRTLLM_DISABLE_STORE_SAVE_RETRY") == "1"
    )

class _LoadStats:
    """The loader thread's running totals, reported once per `LOAD_STATS_INTERVAL`."""

    def __init__(self):
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self.since = now
        self.loads = self.pages = self.loaded_bytes = 0
        self.busy = self.waited = self.max_wait = 0.0

    def add(self, pages: int, loaded_bytes: int, wait: float, busy: float) -> Optional[str]:
        """Count one load; past the interval, return the report and start a new one."""
        self.loads += 1
        self.pages += pages
        self.loaded_bytes += loaded_bytes
        self.busy += busy
        self.waited += wait
        self.max_wait = max(self.max_wait, wait)
        now = time.monotonic()
        elapsed = now - self.since
        if elapsed < LOAD_STATS_INTERVAL:
            return None
        report = (
            f"{self.loads} loads, {self.pages} pages, {self.loaded_bytes / (1 << 30):.2f} GiB "
            f"in {elapsed:.0f}s ({self.busy:.1f}s busy, "
            f"{self.loaded_bytes / (1 << 30) / max(self.busy, 1e-9):.1f} GiB/s while busy); "
            f"queue wait mean {self.waited / self.loads:.3f}s max {self.max_wait:.3f}s"
        )
        self._reset(now)
        return report


class _SaveStats:
    """The save thread's running totals, reported once per `LOAD_STATS_INTERVAL`."""

    def __init__(self):
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self.since = now
        self.batches = self.written = self.present = self.written_bytes = 0
        self.busy = self.waited = self.max_wait = 0.0

    def add(
        self, written: int, present: int, written_bytes: int, wait: float, busy: float
    ) -> Optional[str]:
        """Count one queued save; past the interval, return the report and start a new one."""
        self.batches += 1
        self.written += written
        self.present += present
        self.written_bytes += written_bytes
        self.busy += busy
        self.waited += wait
        self.max_wait = max(self.max_wait, wait)
        now = time.monotonic()
        elapsed = now - self.since
        if elapsed < LOAD_STATS_INTERVAL:
            return None
        report = (
            f"{self.batches} batches, {self.written} pages written ({self.written_bytes / (1 << 30):.2f} GiB), "
            f"{self.present} already stored, in {elapsed:.0f}s ({self.busy:.1f}s busy); "
            f"queue wait mean {self.waited / self.batches:.3f}s max {self.max_wait:.3f}s"
        )
        self._reset(now)
        return report


class _RetireStats:
    """Window retirement counters, reported by the save thread once per `LOAD_STATS_INTERVAL`."""

    def __init__(self):
        self._reset(time.monotonic())
        #: Totals since start, never reset (read by tests and by the shutdown line).
        self.total_retired = self.total_removed = 0

    def _reset(self, now: float) -> None:
        self.since = now
        self.ends = self.links_put = self.links_found = self.retired = 0
        self.removed = self.kept = 0
        self.busy = 0.0

    def report(self, pending: int, force: bool = False) -> Optional[str]:
        """The interval's report, or None before `LOAD_STATS_INTERVAL` (unless `force`: a burst just drained)."""
        now = time.monotonic()
        elapsed = now - self.since
        if elapsed < LOAD_STATS_INTERVAL and not force:
            return None
        report = (
            f"{self.ends} prompt ends, {self.links_put} links put, {self.links_found} links followed, "
            f"{self.retired} windows retired, {self.removed} pages removed, {self.kept} not removed "
            f"(absent or leased), {pending} pending, in {elapsed:.0f}s ({self.busy:.2f}s busy); "
            f"total retired {self.total_retired} removed {self.total_removed}"
        )
        self._reset(now)
        return report


class _LookupStats:
    """Lookup outcomes on the leader, reported once per `LOAD_STATS_INTERVAL`.

    full: the window at the full-attention hit's end is stored. rewound: one within the rewind is. recovered: an
    earlier end found by the fallback. window_miss: full-attention blocks but no usable window. fa_miss: not even
    the first full-attention block.
    """

    OUTCOMES = ("full", "rewound", "recovered", "window_miss", "fa_miss")

    def __init__(self):
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self.since = now
        self.counts = dict.fromkeys(self.OUTCOMES, 0)
        self.hit_blocks = self.lost_blocks = 0
        self.probe_seconds = 0.0
        self.max_probe = 0.0
        self.keys = self.calls = self.reused_blocks = 0

    def add(
        self,
        outcome: str,
        hit_blocks: int,
        lost_blocks: int,
        probe_seconds: float,
        keys: int = 0,
        calls: int = 0,
        reused_blocks: int = 0,
    ) -> Optional[str]:
        self.counts[outcome] += 1
        self.keys += keys
        self.calls += calls
        self.reused_blocks += reused_blocks
        self.hit_blocks += hit_blocks
        self.lost_blocks += lost_blocks
        self.probe_seconds += probe_seconds
        self.max_probe = max(self.max_probe, probe_seconds)
        now = time.monotonic()
        elapsed = now - self.since
        if elapsed < LOAD_STATS_INTERVAL:
            return None
        total = sum(self.counts.values())
        report = (
            f"{total} lookups ("
            + ", ".join(f"{name} {count}" for name, count in self.counts.items())
            + f"), {self.hit_blocks} blocks hit, {self.lost_blocks} blocks behind the full-attention hit lost, "
            f"probe mean {self.probe_seconds / max(1, total) * 1e3:.1f}ms max {self.max_probe * 1e3:.1f}ms "
            f"in {elapsed:.0f}s; keys probed {self.keys} ({self.keys / max(1, total):.0f}/lookup), "
            f"probe rounds {self.calls} ({self.calls / max(1, total):.2f}/lookup), reused blocks {self.reused_blocks}"
        )
        memo = chain_memo_stats()
        if memo is not None:
            report += f"; {memo}"
        self._reset(now)
        return report


@dataclass(frozen=True)
class PrefixHit:
    """How far a request can resume from the store, and what backs it."""

    #: Leading candidate blocks the store can replay.
    blocks: int
    #: Per sliding-window group, the first block from which the store holds
    #: every page up to the hit's end. A replay the runtime cuts short reads its
    #: window from further back, which is sound only down to this block.
    window_present_from: Dict[int, int] = field(default_factory=dict)
    #: Leading candidate blocks whose full-attention pages the store holds (at least `blocks`; more when the window
    #: search rewound or fell back to an earlier prompt end). Read by window retirement.
    full_blocks: int = field(default=0, compare=False)


def _live_blocks(sinks: int, stale_end: int, lo: int, hi: int) -> List[int]:
    """Block ordinals in ``[lo, hi)`` a window still reads: its sinks and its tail."""
    return [*range(lo, min(sinks, hi)), *range(max(lo, sinks, stale_end), hi)]


class MooncakeStoreConnectorWorker(KvCacheConnectorWorker):
    """Moves KV pages between this rank's GPU cache and the Mooncake pool."""

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        validate_llm_args(llm_args)
        self._config = MooncakeStoreConnectorConfig.resolve(llm_args)
        # Where this rank records the segment it mounts. `None` when the
        # deployment named no directory, which only makes the run
        # unreportable; see `ledger.record_segment`.
        pool = pool_config(llm_args)
        self._run_dir = pool.run_dir if pool is not None else None
        self._rank = mpi_rank()
        self._world_size = mpi_world_size()
        self._model_key = self._config.resolve_model_key(llm_args.model)

        self._addressing: Optional[PageAddressing] = None
        # Namespaces for this rank, used for both directions of transfer.
        self._namespaces: Dict[int, KeyNamespace] = {}
        # The same namespaces for every rank. A prefix is only reusable when all
        # shards of it are present, so a lookup has to ask about all of them.
        self._peer_namespaces: Dict[int, Tuple[KeyNamespace, ...]] = {}
        # The cache's own life cycle per sliding-window group, which decides
        # the blocks such a group still reads at a given history length.
        self._life_cycles: Dict[int, AttnLifeCycle] = {}
        #: Pinned slots window pages are copied into on the execution stream before a save reads them.
        self._window_snapshot: Optional[SnapshotRing] = None
        # Side stream for the window snapshot copies (SNAPSHOT_STREAM_ENV), created on first use.
        self._snapshot_stream_on = os.environ.get(SNAPSHOT_STREAM_ENV, "0") == "1"
        self._snapshot_stream: Optional[torch.cuda.Stream] = None
        self._window_snapshot_dropped = 0

        # Checked before the segment is mounted, since the kernel answers an
        # unaffordable total by killing the process rather than by failing the
        # allocation.
        # No loader threads, and no staging pools for them, when reserved prefixes load synchronously.
        self._num_load_threads = (
            _load_thread_count()
            if self._config.role.loads and os.environ.get(SYNC_LOAD_ENV, "0") != "1"
            else 0
        )
        # A sliding-window layout is served through prefix reservations only; see register_kv_cache_layout.
        self._prefix_aware_scheduling = bool(
            getattr(
                getattr(llm_args, "scheduler_config", None), "enable_prefix_aware_scheduling", True
            )
        )
        validate_node_budget(self._config, load_threads=self._num_load_threads)

        self._store, self._segment_host = _open_store(self._config)
        if os.environ.get(SOFTPIN_ENV, "0") == "1" and not self._config.capacity_only:
            # M3.1 soft-pin port (off by default): saving ranks put every page soft-pinned; see softpin.py.
            self._store = maybe_softpin_store(
                self._store,
                rank=self._rank,
                role=self._config.role.value,
                saves=bool(self._config.role.saves),
                namespace=self._config.namespace,
                model_key=self._model_key,
                world_size=self._world_size,
            )
        record_segment(
            self._run_dir,
            host=self._segment_host,
            rank=self._rank,
            segment_size=self._config.global_segment_size,
            role=self._config.role.value,
            model_key=self._model_key,
        )

        self._save_queue: "Queue[Optional[Tuple[torch.cuda.Event, List[RequestTransfers]]]]" = (
            Queue()
        )
        self._save_thread: Optional[threading.Thread] = None
        #: Set once the save thread has finished its device and stream setup,
        #: whether or not that succeeded.
        self._save_started = threading.Event()
        self._save_lock = threading.Lock()
        # Host staging, when the pool cannot register device memory.
        self._load_staging: Optional[HostStagingPool] = None
        self._save_staging: Optional[HostStagingPool] = None
        self._save_stream: Optional[torch.cuda.Stream] = None
        # This rank's device, captured on the executor thread; see _drain_saves.
        self._device_index: Optional[int] = None
        # Pages per store call. Staging narrows this to the slots it can afford.
        self._batch_size = self._config.transfer_batch_size
        # Save submissions still in flight, per request.
        self._outstanding_saves: Dict[int, int] = defaultdict(int)
        # Requests the runtime has told us are done producing KV. Their pages
        # stay pinned until we report them back through `get_finished`.
        self._closed_requests: Set[int] = set()
        self._mcb_closed_at: Dict[int, float] = {}  # MCB diagnostics: request id -> close time
        # Why the save thread never reached its transfer loop, read back by
        # `register_kv_cache_layout`. A failure inside the loop is dropped
        # there instead.
        self._save_startup_error: Optional[BaseException] = None
        # Asynchronous prefix loads run on threads of their own, each with its
        # own staging slots and stream, while their requests wait outside the
        # batch. The threads share one queue.
        self._load_queue: "Queue[Optional[Tuple[int, RequestTransfers, Optional[List[torch.cuda.Event]], float]]]" = Queue()
        self._load_threads: List[threading.Thread] = []
        self._load_startup_error: Optional[BaseException] = None
        self._async_load_stagings: List[HostStagingPool] = []
        self._window_fallback = os.environ.get(WINDOW_FALLBACK_ENV, "1") != "0"
        self._lookup_stats = _LookupStats()
        self._probe_chunk = int(os.environ.get(PROBE_CHUNK_ENV, DEFAULT_PROBE_CHUNK))
        # lkp L1/L2 (off by default): staged leading-run probes, and reuse of runs earlier lookups confirmed.
        self._lookup_staged = os.environ.get(LOOKUP_STAGED_ENV, "0") == "1"
        self._run_reuse: Optional[_RunReuse] = _RunReuse.from_env() if self._config.role.loads else None
        # Keys and `batch_is_exist` rounds the lookup probes issued, read into the lookup statistics.
        self._probe_keys = 0
        self._probe_calls = 0
        # ldfix F3 (off by default): keys this rank recently confirmed in the pool, so saves skip re-probing them.
        self._local_stored: Optional[LocalStoredSet] = (
            LocalStoredSet.from_env() if self._config.role.saves else None
        )
        # Lookups run on the leader's scheduler, so only a loading role probes.
        self._probe_pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(
                max_workers=PROBE_THREADS, thread_name_prefix=f"mooncake-store-probe-{self._rank}"
            )
            if self._config.role.loads and self._probe_chunk > 0
            else None
        )
        # lkp L3 (off by default): pages this rank loads join its local-stored set (needs F3).
        self._stored_from_loads = (
            os.environ.get(STORED_FROM_LOADS_ENV, "0") == "1" and self._local_stored is not None
        )
        self._load_lock = threading.Lock()
        # ldfix F2 (off by default): loader threads start a load right after a forward launch.
        self._launch_gate: Optional[_LaunchGate] = (
            _LaunchGate.from_env() if self._num_load_threads > 0 else None
        )
        # Reservations whose pages landed, until `get_finished_prefix_loads`.
        self._finished_prefix_loads: List[int] = []
        # The loader thread's first failure, raised on the executor thread.
        self._load_error: Optional[BaseException] = None
        # Debug only (port of our e01dcc39a5): per-page digests of saves, loads
        # and retirements when TRTLLM_MC_PAGE_DIGEST_DIR is set.
        self._page_digest: Optional[PageDigestLog] = PageDigestLog.from_env(self._rank)
        # Pages whose rank-0 slot index differed from this rank's (see `_localize`), and save pages dropped because
        # this rank no longer had a slot for their block.
        self._remapped_pages = 0
        self._unresolved_save_pages = 0
        self._localize_seconds = 0.0
        # Commit-swap guard (see `COMMIT_SWAP_GUARD_ENV`): save lists held until the commit after their pass has run,
        # and what the re-check after that commit found.
        self._swap_guard = os.environ.get(COMMIT_SWAP_GUARD_ENV, "1") != "0"
        self._deferred_saves: List[Tuple[Tuple, Optional[List]]] = []
        self._swap_checked = 0
        self._swap_retargeted = 0
        self._swap_dropped = 0
        self._recheck_seconds = 0.0
        if self._config.role.saves and not self._config.capacity_only:
            logger.warning(
                f"mooncake-store rank {self._rank}: COMMIT_SWAP_GUARD={'on' if self._swap_guard else 'off'}"
            )
        # PGFIX_FAST_ENV: 0 off, 1 fast slot lookups, 2 fast + verified against the full query. Counters: lookups
        # (request x layer group), those the full query answered, and verify differences.
        self._pgfix_fast = int(os.environ.get(PGFIX_FAST_ENV, "0") or 0)
        self._pgfix_lookups = 0
        self._pgfix_fallbacks = 0
        self._pgfix_mismatches = 0
        if self._pgfix_fast and not self._config.capacity_only:
            logger.warning(
                f"mooncake-store rank {self._rank}: PGFIX_FAST_ARMED mode={self._pgfix_fast}"
                f"{' (verify)' if self._pgfix_fast >= 2 else ''}"
            )
        # Rank-local lookups (off by default): each rank probes its own shard's keys and the ranks agree by allreduce.
        # Attention DP already looks up per owner, and a single rank has nobody to agree with.
        self._rank_local_lookup = (
            os.environ.get(RANK_LOCAL_LOOKUP_ENV, "0") == "1"
            and self._world_size > 1
            and not bool(getattr(llm_args, "enable_attention_dp", False))
            and self._config.role.loads
            and not self._config.capacity_only
        )
        if self._rank_local_lookup:
            kv_cache_config = llm_args.kv_cache_config
            self._lookup_tokens_per_block = int(kv_cache_config.tokens_per_block)
            reuse_config = getattr(kv_cache_config, "block_reuse_config", None)
            self._lookup_rewind_tokens = int(
                getattr(reuse_config, "swa_endpoint_rewind_tokens", 0) or 0
            )
            # The follower ranks' hash chains, by request id (the leader's live in its request state).
            self._follower_chains: "OrderedDict[int, Tuple[object, BlockHashChain]]" = OrderedDict()
            # Elementwise MIN over the ranks; a communicator of its own is created at the first lookup, which every
            # rank reaches in the same order. Tests replace it.
            self._lookup_reducer = None
        # Window retirement (TRTLLM_MOONCAKE_STORE_WINDOW_KEEP, off at 0): removals waiting out the read lease,
        # owned by the save thread.
        self._window_keep = int(os.environ.get(WINDOW_KEEP_ENV, "0") or 0)
        self._retire_delay = float(os.environ.get(WINDOW_RETIRE_DELAY_ENV, "65") or 0)
        self._link_min_tokens = int(os.environ.get(WINDOW_LINK_MIN_TOKENS_ENV, "2048") or 0)
        self._retire_due: List[Tuple[float, int, List[str]]] = []
        self._retire_seq = 0
        self._retire_stats = _RetireStats()
        if self._window_keep > 0 and self._config.role.saves:
            logger.warning(
                f"mooncake-store rank {self._rank}: WINRET_ARMED keep={self._window_keep} "
                f"delay_s={self._retire_delay:g} link_min_tokens={self._link_min_tokens}"
            )

        global _LOCAL_WORKER
        _LOCAL_WORKER = self
        _LOCAL_WORKER_READY.set()

        logger.warning(
            f"mooncake-store worker rank {self._rank}/{self._world_size} ready "
            f"(role={self._config.role.value}, model_key={self._model_key}, "
            f"master={self._config.master_server_address}, "
            f"segment={self._config.global_segment_size / (1 << 30):.1f} GiB "
            f"on {self._segment_host})"
        )
        if self._lookup_staged or self._run_reuse is not None or self._stored_from_loads:
            logger.warning(
                f"mooncake-store worker rank {self._rank}: lookup keys (lkp): "
                f"staged={int(self._lookup_staged)} ({LOOKUP_STAGED_ENV}), "
                f"reuse_ms={self._run_reuse._ttl * 1e3 if self._run_reuse is not None else 0:g} "
                f"({LOOKUP_REUSE_MS_ENV}), stored_from_loads={int(self._stored_from_loads)} ({STORED_FROM_LOADS_ENV})"
            )
        elif os.environ.get(STORED_FROM_LOADS_ENV, "0") == "1":
            logger.warning(
                f"mooncake-store worker rank {self._rank}: {STORED_FROM_LOADS_ENV}=1 has no effect without "
                "TRTLLM_MOONCAKE_STORE_LOCAL_STORED_SET=1 on a saving role"
            )
        if self._rank_local_lookup:
            logger.warning(
                f"mooncake-store worker rank {self._rank}: rank-local lookups "
                f"({RANK_LOCAL_LOOKUP_ENV}=1): probing this shard's keys only, agreed by allreduce(MIN)"
            )
        if self.capacity_only:
            # Said plainly, since every other sign of a working connector is
            # absent by design and otherwise looks like a broken deployment.
            logger.warning(
                f"mooncake-store worker rank {self._rank}: capacity-only. Its "
                "memory is in the pool and its KV cache is not: no lookups, no "
                "loads, no saves, and no staging buffers pinned for them."
            )

    # ---- registration ----

    @property
    def capacity_only(self) -> bool:
        """Whether this rank lends memory to the pool and transfers nothing."""
        return self._config.capacity_only

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        """Reject the V1 single-pool registration, unless capacity-only.

        Raises:
            NotImplementedError: Unless capacity-only. V1 supplies block hashes
                over a single flat block space, while this connector keys pages
                per layer group from a hash chain of its own, so running the V2
                addressing against V1 block ids would mislabel pages.
        """
        if self.capacity_only:
            # Nothing is addressed, so which manager describes the cache does
            # not matter.
            return
        raise NotImplementedError(
            "The mooncake-store connector requires KVCacheManagerV2. Set "
            "kv_cache_config.use_kv_cache_manager_v2=True."
        )

    def register_kv_cache_layout(self, layout: KvCacheLayout) -> None:
        """Register the KV pools with Mooncake and start the save thread."""
        if self._addressing is not None:
            raise RuntimeError("KV cache layout already registered")

        if self.capacity_only:
            # Nothing will ask the store to reach this rank's pages, so the
            # addressing, buffer registration, staging pool and save thread
            # are all skipped.
            logger.warning(
                f"mooncake-store worker rank {self._rank}: capacity-only, so "
                "the KV cache layout is not registered and no page of it is "
                "reachable from the pool."
            )
            return

        validate_layout(layout)
        addressing = PageAddressing(layout)
        # Torch's current device is thread-local, so read it here on the
        # executor thread; the save thread would otherwise see device 0.
        if torch.cuda.is_available():
            self._device_index = torch.cuda.current_device()
        # Host staging is the only transfer path, which is why
        # stage_through_host is forced True upstream rather than read here.
        self._open_staging(addressing)

        self._addressing = addressing
        for layer_group_id in addressing.layer_group_ids:
            bytes_per_page = addressing.bytes_per_page(layer_group_id)
            self._namespaces[layer_group_id] = self._namespace(
                self._rank, layer_group_id, bytes_per_page
            )
            self._peer_namespaces[layer_group_id] = tuple(
                self._namespace(rank, layer_group_id, bytes_per_page)
                for rank in range(self._world_size)
            )
        for group in layout.groups:
            if group.window_size is not None:
                self._life_cycles[group.layer_group_id] = AttnLifeCycle.make(
                    group.window_size, group.num_sink_tokens, layout.tokens_per_block
                )
        self._open_window_snapshot(addressing)
        if self._life_cycles and self._config.role.loads and not self._prefix_aware_scheduling:
            raise ValueError(
                "The mooncake-store connector serves sliding-window layer groups only through prefix "
                "reservations, which scheduler_config.enable_prefix_aware_scheduling=False turns off. Without "
                "them the runtime can honour a stored prefix only up to a point whose window the lookup never "
                "verified (its fallback when the full offer does not fit), and the load would then read a "
                "window page nothing wrote. Leave enable_prefix_aware_scheduling on."
            )

        if self._config.role.saves:
            self._save_thread = threading.Thread(
                target=self._drain_saves,
                name=f"mooncake-store-save-{self._rank}",
                daemon=True,
            )
            self._save_thread.start()
            # A thread that cannot bind this rank's device has to fail bringup
            # here, or its requests stay pinned on saves nothing will consume.
            self._save_started.wait()
            with self._save_lock:
                startup_error, self._save_startup_error = self._save_startup_error, None
            if startup_error is not None:
                raise RuntimeError(
                    f"mooncake-store: the save thread for rank {self._rank} "
                    "could not start, so this worker would accept pages it "
                    "could never write to the pool."
                ) from startup_error

        if self._config.role.loads:
            started = []
            for index in range(self._num_load_threads):
                # Set once the thread has finished its device and stream setup,
                # whether or not that succeeded.
                event = threading.Event()
                thread = threading.Thread(
                    target=self._drain_loads,
                    args=(index, event),
                    name=f"mooncake-store-load-{self._rank}-{index}",
                    daemon=True,
                )
                thread.start()
                self._load_threads.append(thread)
                started.append(event)
            # As for the save thread: a loader that cannot bind this rank's
            # device would leave every parked request waiting for good.
            for event in started:
                event.wait()
            with self._load_lock:
                startup_error, self._load_startup_error = self._load_startup_error, None
            if startup_error is not None:
                raise RuntimeError(
                    f"mooncake-store: the load thread for rank {self._rank} "
                    "could not start, so asynchronous prefix loads would never land."
                ) from startup_error

        logger.warning(
            f"mooncake-store worker rank {self._rank} registered layout: {addressing.describe()}"
        )

    def _open_window_snapshot(self, addressing: PageAddressing) -> None:
        """Pinned slots for window-page save snapshots (see `WINDOW_SNAPSHOT_ENV`)."""
        if not (self._config.role.saves and self._life_cycles and torch.cuda.is_available()):
            return
        if os.environ.get(WINDOW_SNAPSHOT_ENV, "1") == "0":
            logger.warning(f"mooncake-store rank {self._rank}: {WINDOW_SNAPSHOT_ENV}=0, window pages are saved live")
            return
        slot_bytes = max(addressing.bytes_per_page(group) for group in self._life_cycles)
        num_slots = int(os.environ.get(WINDOW_SNAPSHOT_SLOTS_ENV, DEFAULT_WINDOW_SNAPSHOT_SLOTS))
        self._window_snapshot = SnapshotRing(
            HostStagingPool(slot_bytes=slot_bytes, num_slots=num_slots, store=self._store, label="window-snapshot")
        )

    def _snapshot_window_pages(self, saves: List[RequestTransfers], stream) -> Tuple[List[RequestTransfers], Dict]:
        """Copy this pass's window pages to pinned slots on `stream`; returns (saves, overrides).

        overrides maps (layer group, page index) to (slot, slot address, bytes). A pass whose window pages
        do not fit the ring drops them from the save (a later lookup falls back to an earlier prompt end)
        rather than saving pages that may be recycled before the save thread reads them.
        """
        ring = self._window_snapshot
        windowed = set(self._life_cycles)
        wanted = []
        for entry in saves:
            for page in entry.pages:
                if page.layer_group_id in windowed:
                    wanted.append((page.layer_group_id, page.page_index))
        wanted = list(dict.fromkeys(wanted))
        if not wanted:
            return saves, {}
        slots = ring.try_reserve(len(wanted))
        if slots is None:
            slots = ring.reserve(len(wanted), alive=lambda: self._save_thread is not None and self._save_thread.is_alive())
        if slots is None:
            self._window_snapshot_dropped += len(wanted)
            logger.warning(
                f"mooncake-store rank {self._rank}: {len(wanted)} window pages do not fit the "
                f"{ring.num_slots}-slot window snapshot; not saving them ({self._window_snapshot_dropped} so far)"
            )
            kept = [
                RequestTransfers(entry.request_id, [p for p in entry.pages if p.layer_group_id not in windowed])
                for entry in saves
            ]
            return kept, {}
        handle = _stream_handle(stream)
        overrides = {}
        if batch_copy_mode() is not None:
            # ldfix F1: one batched issue for the pass's window pages instead of a call per page region.
            buffers = [self._addressing.buffers(group, page_index) for group, page_index in wanted]
            side = self._snapshot_side_stream(stream) if handle == 0 else None
            if side is not None:
                # Same place in the executor stream's order: after everything queued so far (the forward that
                # wrote these pages), before anything queued later (the next forward, a commit or a recycle, and
                # the save hand-off event recorded right after this).
                side.wait_stream(stream)
                placed = ring.gather_many(
                    slots, [b[0] for b in buffers], [b[1] for b in buffers], _stream_handle(side)
                )
                stream.wait_stream(side)
            else:
                placed = ring.gather_many(slots, [b[0] for b in buffers], [b[1] for b in buffers], handle)
            for slot, key, (slot_address, total) in zip(slots, wanted, placed):
                overrides[key] = (slot, slot_address, total)
            return saves, overrides
        for slot, (group, page_index) in zip(slots, wanted):
            addresses, sizes = self._addressing.buffers(group, page_index)
            slot_address, total = ring.gather(slot, addresses, sizes, handle)
            overrides[(group, page_index)] = (slot, slot_address, total)
        return saves, overrides

    def _snapshot_side_stream(self, stream) -> Optional[torch.cuda.Stream]:
        """The side stream the window snapshot is issued on (SNAPSHOT_STREAM_ENV), or None to use `stream` itself."""
        if not self._snapshot_stream_on or stream is None:
            return None
        if self._snapshot_stream is None:
            self._snapshot_stream = torch.cuda.Stream(device=stream.device)
            logger.warning(
                f"mooncake-store rank {self._rank}: SNAPSHOT_STREAM_ARMED window snapshots are batched on a side "
                f"stream ordered with the executor stream ({SNAPSHOT_STREAM_ENV}=1)"
            )
        return self._snapshot_stream

    def _open_staging(self, addressing: PageAddressing) -> None:
        """Allocate and register the pinned slots pages will pass through.

        Only the directions this role drives get a pool, since each one costs
        a pinned allocation of its own.
        """
        max_bytes_per_page = max(
            addressing.bytes_per_page(layer_group_id)
            for layer_group_id in addressing.layer_group_ids
        )
        slot_bytes, num_slots = plan_slot_geometry(
            max_bytes_per_page,
            self._config.transfer_batch_size,
            MAX_STAGING_BUFFER_BYTES,
        )
        if self._config.role.loads:
            self._load_staging = HostStagingPool(
                slot_bytes=slot_bytes,
                num_slots=num_slots,
                store=self._store,
                label="load",
            )
            # Each loader thread's own slots, so none waits on another's or the
            # executor's.
            self._async_load_stagings = [
                HostStagingPool(
                    slot_bytes=slot_bytes,
                    num_slots=num_slots,
                    store=self._store,
                    label=f"async-load-{index}",
                )
                for index in range(self._num_load_threads)
            ]
        if self._config.role.saves:
            self._save_staging = HostStagingPool(
                slot_bytes=slot_bytes,
                num_slots=num_slots,
                store=self._store,
                label="save",
            )
        self._batch_size = min(self._config.transfer_batch_size, num_slots)
        if self._batch_size < self._config.transfer_batch_size:
            logger.warning(
                f"mooncake-store rank {self._rank} reduced its transfer batch from "
                f"{self._config.transfer_batch_size} to {self._batch_size} pages: "
                f"staging {max_bytes_per_page} B pages within the "
                f"{MAX_STAGING_BUFFER_BYTES} B pinned-memory ceiling does not fit "
                f"more. Lower transfer_batch_size to make the reduction explicit."
            )

    def _namespace(self, rank: int, layer_group_id: int, bytes_per_page: int) -> KeyNamespace:
        return KeyNamespace(
            namespace=self._config.namespace,
            model_key=self._model_key,
            rank=rank,
            world_size=self._world_size,
            layer_group_id=layer_group_id,
            tokens_per_block=self._addressing.tokens_per_block,
            bytes_per_page=bytes_per_page,
        )

    # ---- leader-facing lookup ----

    @property
    def config(self) -> MooncakeStoreConnectorConfig:
        """The resolved connector configuration."""
        return self._config

    @property
    def is_registered(self) -> bool:
        """Whether a KV cache layout has been registered yet."""
        return self._addressing is not None

    @property
    def windowed_group_ids(self) -> Tuple[int, ...]:
        """Layer groups with a sliding window, in layout order.

        Such a group keeps only the tail of a sequence, so the scheduler saves
        its pages where prompts end and a lookup has to land on one of those.
        """
        return tuple(self._life_cycles)

    def sink_blocks(self, layer_group_id: int) -> int:
        """Leading blocks a sliding-window group keeps however far it moves."""
        life_cycle = self._life_cycles.get(layer_group_id)
        return 0 if life_cycle is None else int(life_cycle.num_sink_blocks)

    def stale_blocks(self, layer_group_id: int, history_length: int) -> Tuple[int, int]:
        """Block ordinals ``[start, end)`` a group no longer reads at ``history_length``.

        Answered by the cache's own life cycle, as
        `KVCacheManagerV2._stale_block_range` is, so the blocks a lookup asks
        for are the ones the runtime gives pages to. Empty for full attention.
        """
        life_cycle = self._life_cycles.get(layer_group_id)
        if life_cycle is None:
            return 0, 0
        stale = life_cycle.get_stale_range(history_length, self._addressing.tokens_per_block)
        # Read `beg`/`end` rather than unpacking: only the pure-Python
        # `HalfOpenRange` is a tuple subclass, the bound one is not.
        return int(stale.beg), int(stale.end)

    def count_prefix_hit(self, block_hashes: Sequence[bytes]) -> int:
        """How many leading blocks of `block_hashes` are fully present.

        A block counts only when every layer group and every rank has its
        page, since a prefix is replayed as a whole. The scan stops at the
        first incomplete block, a later hit being unusable on its own.

        Args:
            block_hashes: Candidate hashes in block order.

        Returns:
            Length of the usable prefix, in blocks.
        """
        if not block_hashes or self._addressing is None:
            return 0
        return self._count_leading(block_hashes, tuple(self._peer_namespaces))

    @property
    def rank_local_lookup(self) -> bool:
        """Whether every rank probes its own keys and the ranks agree by allreduce (`RANK_LOCAL_LOOKUP_ENV`).

        When true, `count_prefix_hit` and `find_prefix_hit` are collective: the leader's lookup and every other
        rank's `follow_prefix_lookup` have to reach them for the same requests in the same order.
        """
        return self._rank_local_lookup

    def follow_prefix_lookup(self, request, num_computed_tokens: int) -> None:
        """A non-leader rank's share of the leader's lookup, for `rank_local_lookup`.

        Mirrors `MooncakeStoreConnectorScheduler._lookup` up to the store probe, so this rank joins the probe's
        allreduce exactly when the leader does and with the same candidate blocks. The answer is the leader's to
        give: it arrives through the broadcast that follows, so nothing is returned here.
        """
        if not self._rank_local_lookup:
            return
        # Imported here: the scheduler module imports this one.
        from .scheduler import _reuse_scope

        scope = _reuse_scope(request)
        if scope is None:
            return
        tokens = request.get_tokens(0)
        chain = self._follower_chain(request.request_id, scope, tokens)
        tokens_per_block = self._lookup_tokens_per_block
        if num_computed_tokens % tokens_per_block:
            return
        first_block = num_computed_tokens // tokens_per_block
        last_block = (len(tokens) - 1) // tokens_per_block
        candidates = chain.hashes[first_block:last_block]
        if not candidates:
            return
        if self.windowed_group_ids:
            self.find_prefix_hit(candidates, first_block, self._lookup_rewind_tokens)
        else:
            self.count_prefix_hit(candidates)

    def _follower_chain(self, request_id: int, scope, tokens: Sequence[int]) -> BlockHashChain:
        """The hash chain of `tokens`, kept per request across a follower's lookups, as the leader keeps its own."""
        chains = self._follower_chains
        cached = chains.get(request_id)
        chain = None
        if cached is not None and cached[0] == scope:
            chain = cached[1]
            try:
                chain.extend(tokens)
            except ValueError:
                chain = None
        if chain is None:
            chain = BlockHashChain(self._lookup_tokens_per_block, scope=scope)
            chain.extend(tokens)
        chains[request_id] = (scope, chain)
        chains.move_to_end(request_id)
        while len(chains) > RANK_LOCAL_CHAIN_CACHE_SIZE:
            chains.popitem(last=False)
        return chain

    def _agree_min(self, values: List[int]) -> List[int]:
        """Elementwise minimum of `values` over every rank (collective)."""
        reducer = self._lookup_reducer
        if reducer is None:
            reducer = self._lookup_reducer = _mpi_min_reducer()
        return reducer(values)

    def find_prefix_hit(
        self, block_hashes: Sequence[bytes], first_block: int, rewind_tokens: int = 0
    ) -> PrefixHit:
        """The longest stored prefix a request can resume from, windows included.

        Full-attention groups need every block of the prefix. A sliding-window
        group needs only the blocks it still reads where the prefix ends, and
        the store holds those only near the ends of earlier prompts, so the
        search starts at the longest full-attention hit and walks back up to
        `rewind_tokens`, taking the first prompt end whose window is complete.
        With no window this is `count_prefix_hit`.

        When no end within the rewind has its window, the full-attention blocks
        can still reach past the last stored window (a save that has not landed
        yet, or a window evicted on its own). Rather than give up the whole
        prefix, the search then probes the window pages further back and
        resumes at the latest earlier end whose window is complete.

        Args:
            block_hashes: Candidate hashes in block order.
            first_block: Block ordinal of `block_hashes[0]`. Everything before it
                is on device already, window included.
            rewind_tokens: How far back from the full-attention hit to look for
                a prompt end whose window is stored.

        Returns:
            The usable prefix in blocks, and for each sliding-window group the
            first block from which the store holds every page up to its end.
        """
        started = time.monotonic()
        keys, calls = self._probe_keys, self._probe_calls
        reused = self._run_reuse.reused_blocks if self._run_reuse is not None else 0
        hit, outcome, fa_end = self._find_prefix_hit(block_hashes, first_block, rewind_tokens)
        if self._window_keep > 0:
            # Window retirement: tell the scheduler how far the store holds full attention (see `store_held`).
            hit = dataclasses.replace(hit, full_blocks=max(hit.blocks, fa_end - first_block))
        if self._life_cycles and block_hashes and self._addressing is not None:
            report = self._lookup_stats.add(
                outcome,
                hit.blocks,
                max(0, fa_end - first_block - hit.blocks),
                time.monotonic() - started,
                keys=self._probe_keys - keys,
                calls=self._probe_calls - calls,
                reused_blocks=(self._run_reuse.reused_blocks if self._run_reuse is not None else 0) - reused,
            )
            if report:
                logger.info(f"mooncake-store rank {self._rank} lookups: {report}")
        return hit

    def _find_prefix_hit(
        self, block_hashes: Sequence[bytes], first_block: int, rewind_tokens: int
    ) -> Tuple[PrefixHit, str, int]:
        """`find_prefix_hit`, which of `_LookupStats.OUTCOMES` it came to, and where the full-attention hit ends."""
        if not block_hashes or self._addressing is None:
            return PrefixHit(0), "fa_miss", first_block
        full_groups = tuple(g for g in self._peer_namespaces if g not in self._life_cycles)
        hit = self._count_leading(block_hashes, full_groups) if full_groups else len(block_hashes)
        if not self._life_cycles or hit == 0:
            return PrefixHit(hit), ("full" if hit else "fa_miss"), first_block + hit

        tokens_per_block = self._addressing.tokens_per_block
        end = first_block + hit
        # The search for a stored prompt end goes back at most `rewind_tokens`;
        # past that, only the fallback looks.
        lowest_end = max(first_block + 1, end - -(-rewind_tokens // tokens_per_block))

        # Every block some candidate end reads, probed in one call: the sinks,
        # and the tail from where the lowest end's window starts.
        probed: Dict[int, List[int]] = {}
        for layer_group_id in self._life_cycles:
            _, stale_end = self.stale_blocks(layer_group_id, lowest_end * tokens_per_block)
            probed[layer_group_id] = _live_blocks(
                self.sink_blocks(layer_group_id), stale_end, first_block, end
            )
        stored = self._probe_window_blocks(block_hashes, first_block, probed)
        if stored is None:
            return PrefixHit(0), "window_miss", end

        for candidate in range(end, lowest_end - 1, -1):
            if self._window_complete(stored, first_block, candidate):
                outcome = "full" if candidate == end else "rewound"
                hit_at = PrefixHit(
                    candidate - first_block, self._present_from(stored, first_block, candidate)
                )
                return hit_at, outcome, end

        if not self._window_fallback or lowest_end <= first_block + 1:
            return PrefixHit(0), "window_miss", end

        # The fallback: probe each window group's blocks below what the first
        # call covered, then take the latest earlier end with a complete window.
        below: Dict[int, List[int]] = {}
        for layer_group_id, blocks in probed.items():
            low = max(first_block, self.sink_blocks(layer_group_id))
            first_probed = next((b for b in blocks if b >= low), end)
            below[layer_group_id] = list(range(low, first_probed))
        more = self._probe_window_blocks(block_hashes, first_block, below)
        if more is None:
            return PrefixHit(0), "window_miss", end
        for layer_group_id, blocks in more.items():
            stored[layer_group_id] |= blocks
        # Missing-block prefix sums per group over [first_block, end), so each
        # candidate's window is checked in constant time.
        missing: Dict[int, List[int]] = {}
        for layer_group_id, blocks in stored.items():
            sums = [0]
            for block in range(first_block, end):
                sums.append(sums[-1] + (block not in blocks))
            missing[layer_group_id] = sums
        for candidate in range(lowest_end - 1, first_block, -1):
            history_length = candidate * tokens_per_block
            complete = True
            for layer_group_id, sums in missing.items():
                sinks = self.sink_blocks(layer_group_id)
                tail_from = min(
                    candidate,
                    max(first_block, sinks, self.stale_blocks(layer_group_id, history_length)[1]),
                )
                sink_hi = max(first_block, min(sinks, candidate))
                if (
                    sums[sink_hi - first_block]
                    or sums[candidate - first_block] - sums[tail_from - first_block]
                ):
                    complete = False
                    break
            if complete:
                return (
                    PrefixHit(
                        candidate - first_block, self._present_from(stored, first_block, candidate)
                    ),
                    "recovered",
                    end,
                )
        return PrefixHit(0), "window_miss", end

    def _probe_window_blocks(
        self, block_hashes: Sequence[bytes], first_block: int, blocks_by_group: Dict[int, List[int]]
    ) -> Optional[Dict[int, Set[int]]]:
        """Which of the given window-group blocks every rank holds, or None if the probe failed."""
        if self._rank_local_lookup:
            return self._probe_window_blocks_rank_local(block_hashes, first_block, blocks_by_group)
        keys: List[str] = []
        for layer_group_id, blocks in blocks_by_group.items():
            keys.extend(
                _interleaved_keys(
                    self._peer_namespaces[layer_group_id],
                    [block_hashes[block - first_block] for block in blocks],
                )
            )
        present = self._probe(keys) if keys else []
        if present is None:
            return None
        stored: Dict[int, Set[int]] = {}
        index = 0
        for layer_group_id, blocks in blocks_by_group.items():
            stored[layer_group_id] = set()
            for block in blocks:
                statuses = present[index : index + self._world_size]
                index += self._world_size
                if all(status == 1 for status in statuses):
                    stored[layer_group_id].add(block)
        return stored

    def _probe_window_blocks_rank_local(
        self, block_hashes: Sequence[bytes], first_block: int, blocks_by_group: Dict[int, List[int]]
    ) -> Optional[Dict[int, Set[int]]]:
        """`_probe_window_blocks` over this rank's keys, agreed with every rank by allreduce(MIN).

        Collective: every rank calls it with the same blocks. The agreed flags are 1 exactly where every rank's own
        probe said 1, which is the leader's all-shards answer; a failed probe on any rank fails it on all.
        """
        keys: List[str] = []
        for layer_group_id, blocks in blocks_by_group.items():
            keys.extend(
                _interleaved_keys(
                    (self._namespaces[layer_group_id],),
                    [block_hashes[block - first_block] for block in blocks],
                )
            )
        present = self._probe(keys) if keys else []
        # Slot 0 says whether the probe answered; then one flag per block, in `blocks_by_group` order.
        flags = [0] * (1 + len(keys))
        if present is not None:
            flags[0] = 1
            for index, status in enumerate(present):
                flags[1 + index] = 1 if status == 1 else 0
        agreed = self._agree_min(flags)
        if agreed[0] != 1:
            return None
        stored: Dict[int, Set[int]] = {}
        index = 1
        for layer_group_id, blocks in blocks_by_group.items():
            stored[layer_group_id] = set()
            for block in blocks:
                if agreed[index] == 1:
                    stored[layer_group_id].add(block)
                index += 1
        return stored

    def _window_complete(
        self, stored: Dict[int, Set[int]], first_block: int, candidate: int
    ) -> bool:
        """Whether every window group holds the blocks it reads at `candidate`."""
        history_length = candidate * self._addressing.tokens_per_block
        return all(
            set(
                _live_blocks(
                    self.sink_blocks(layer_group_id),
                    self.stale_blocks(layer_group_id, history_length)[1],
                    first_block,
                    candidate,
                )
            )
            <= stored[layer_group_id]
            for layer_group_id in self._life_cycles
        )

    @staticmethod
    def _present_from(
        stored: Dict[int, Set[int]], first_block: int, candidate: int
    ) -> Dict[int, int]:
        """Per window group, the first block of the stored run that ends at `candidate`."""
        present_from = {}
        for layer_group_id, blocks in stored.items():
            block = candidate
            while block - 1 >= first_block and block - 1 in blocks:
                block -= 1
            present_from[layer_group_id] = block
        return present_from

    def _count_leading(self, block_hashes: Sequence[bytes], group_ids: Sequence[int]) -> int:
        """Leading blocks whose pages every rank holds in each of `group_ids`."""
        if self._rank_local_lookup:
            # This rank's leading run over its own keys, then the shortest over the ranks: the first block any
            # shard misses, which is where the leader's probe of every shard stops. A failed probe counts 0.
            # The lkp L1/L2 probes (when on) find the same first miss of this rank's keys from fewer of them; they
            # are local to the rank and issue no collective, so every rank still reaches the one allreduce.
            namespaces = [self._namespaces[layer_group_id] for layer_group_id in group_ids]
            local = self._leading_run(namespaces, group_ids, block_hashes)
            return int(self._agree_min([local])[0])
        namespaces = [
            namespace
            for layer_group_id in group_ids
            for namespace in self._peer_namespaces[layer_group_id]
        ]
        return self._leading_run(namespaces, group_ids, block_hashes)

    def _leading_run(
        self,
        namespaces: Sequence[KeyNamespace],
        group_ids: Sequence[int],
        block_hashes: Sequence[bytes],
    ) -> int:
        """Leading blocks of `block_hashes` whose key is present in every one of `namespaces` (0 if a probe failed)."""
        reuse = self._run_reuse
        if not self._lookup_staged and reuse is None:
            keys = _interleaved_keys(namespaces, block_hashes)
            keys_per_block = len(namespaces)

            present = self._probe(keys)
            if present is None:
                return 0

            # Mooncake reports 1 for present, 0 for absent and a negative value for
            # a failed probe. Anything but a definite 1 is treated as a miss, and
            # the hit ends at the block holding the first one.
            first_miss = next(
                (index for index, status in enumerate(present) if status != 1), len(present)
            )
            return first_miss // keys_per_block

        # lkp L1/L2: the same answer, the first block any rank or group misses, from fewer keys.
        groups = tuple(group_ids)
        known, confirmed = reuse.known(groups, block_hashes) if reuse is not None else (0, 0.0)
        if self._lookup_staged:
            run = _leading_run_staged(
                len(block_hashes), known, self._block_prober(namespaces, block_hashes)
            )
        else:
            run = self._leading_run_from(namespaces, block_hashes, known)
        if run is None:
            return 0
        if reuse is not None:
            reuse.record(groups, block_hashes, run, confirmed)
        return run

    def _block_prober(self, namespaces: Sequence[KeyNamespace], block_hashes: Sequence[bytes]):
        """``probe(blocks)``: per block index of ``block_hashes``, whether every namespace's key is present."""
        keys_per_block = len(namespaces)

        def probe(blocks: List[int]) -> Optional[List[bool]]:
            present = self._probe(_interleaved_keys(namespaces, [block_hashes[block] for block in blocks]))
            if present is None:
                return None
            return [
                all(status == 1 for status in present[index : index + keys_per_block])
                for index in range(0, len(present), keys_per_block)
            ]

        return probe

    def _leading_run_from(
        self, namespaces: Sequence[KeyNamespace], block_hashes: Sequence[bytes], start: int
    ) -> Optional[int]:
        """The one-round probe of every block from ``start`` on (blocks below it known complete)."""
        if start >= len(block_hashes):
            return len(block_hashes)
        present = self._probe(_interleaved_keys(namespaces, block_hashes[start:]))
        if present is None:
            return None
        first_miss = next((index for index, status in enumerate(present) if status != 1), len(present))
        return start + first_miss // len(namespaces)

    def _probe(self, keys: List[str]) -> Optional[List[int]]:
        """`batch_is_exist` over `keys`, or None when the probe itself failed.

        Long key lists go out as concurrent chunks of `PROBE_CHUNK_ENV` keys, answered in order.
        """
        self._probe_keys += len(keys)
        self._probe_calls += 1
        chunk = self._probe_chunk
        if self._probe_pool is None or chunk <= 0 or len(keys) <= chunk:
            chunks = [keys]
        else:
            chunks = [keys[start : start + chunk] for start in range(0, len(keys), chunk)]
        try:
            if len(chunks) == 1:
                answers = [self._store.batch_is_exist(chunks[0])]
            else:
                futures = [
                    self._probe_pool.submit(self._store.batch_is_exist, part) for part in chunks
                ]
                answers = [future.result() for future in futures]
        except Exception as exc:
            logger.warning(
                f"mooncake-store lookup failed; treating as a miss: "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            return None

        present: List[int] = []
        for part, answer in zip(chunks, answers):
            if len(answer) != len(part):
                logger.warning(
                    f"mooncake-store batch_is_exist returned {len(answer)} results for "
                    f"{len(part)} keys; treating as a miss"
                )
                return None
            present.extend(answer)
        local = self._local_stored
        if local is not None:
            # The only eviction signal Mooncake gives: a key this rank confirmed is now reported absent.
            local.discard_many(key for key, status in zip(keys, present) if status == 0)
        return present

    # ---- load path ----

    def start_load_kv(self, stream: torch.cuda.Stream):
        """Pull this pass's pages into their GPU slots, or hand them to the loader thread.

        Synchronous loads land before the forward pass. An asynchronous prefix
        load's request is parked outside the batch, so its pages go to the
        loader thread and are reported through `get_finished_prefix_loads`.
        """
        metadata: Optional[MooncakeStoreMetadata] = self.get_connector_meta()
        if metadata is None:
            return
        self._raise_load_error()
        self._localize(metadata)

        now: List[RequestTransfers] = list(metadata.loads)
        finished: List[int] = []
        for prefix_load in getattr(metadata, "prefix_loads", ()):
            if prefix_load.is_async:
                self._queue_async_load(prefix_load, stream)
            else:
                now.append(prefix_load.transfers)
                finished.append(prefix_load.reservation_id)
        if now:
            staging = self._load_staging
            _mcb_t0 = time.monotonic()
            _mcb_bytes = self._load_pages(now, staging, _stream_handle(stream) if staging is not None else 0)
            if _MCB_DIAG and self._rank == 0:
                logger.info(
                    f"[MCB] syncload rank {self._rank} reqs={len(now)} "
                    f"pages={sum(len(t.pages) for t in now)} bytes={_mcb_bytes} "
                    f"load={time.monotonic() - _mcb_t0:.4f} done={time.time():.3f}"
                )
        if finished:
            with self._load_lock:
                self._finished_prefix_loads.extend(finished)

    def get_finished_prefix_loads(self) -> List[int]:
        """Reservations whose pages are on this rank's device since the last call."""
        self._raise_load_error()
        with self._load_lock:
            finished, self._finished_prefix_loads = self._finished_prefix_loads, []
        return finished

    def _queue_async_load(self, prefix_load: PrefixLoadTransfers, stream) -> None:
        """Hand a parked request's pages to the loader thread.

        The pages were just allocated, and their previous owner's last work
        may still be queued, so the loader waits for events recorded at the
        hand-off before it writes them: on the executor's current stream
        (`stream`, e.g. a window snapshot still reading a page) and on the KV
        cache manager's execution stream. The current stream joins the
        execution stream only around forwards, so it does not cover what the
        manager queued there since the last one, nor the connector side
        allocation stream, which the execution stream joins when the batch is
        reported.
        """
        if not prefix_load.transfers.pages:
            with self._load_lock:
                self._finished_prefix_loads.append(prefix_load.reservation_id)
            return
        ready = None
        if torch.cuda.is_available():
            ready = []
            for handoff_stream in self._handoff_streams(stream):
                event = torch.cuda.Event()
                event.record(handoff_stream)
                ready.append(event)
        self._load_queue.put(
            (prefix_load.reservation_id, prefix_load.transfers, ready, time.monotonic())
        )

    def _handoff_streams(self, stream) -> List:
        """The streams a hand-off records its events on: `stream` and the execution stream, once each."""
        streams = [stream if stream is not None else torch.cuda.current_stream()]
        execution = self.get_execution_stream()
        if execution is not None and int(execution.cuda_stream) != int(streams[0].cuda_stream):
            streams.append(execution)
        return streams

    def _localize(self, metadata: MooncakeStoreMetadata) -> None:
        """Replace rank 0's page slot indices in `metadata` with this rank's own, in place, once.

        The metadata is built on rank 0 and broadcast. Every rank allocates the
        same blocks for a request, but KV cache manager V2 hands out a recycled
        slot whose last GPU use has finished ahead of a never-used one, so the
        slot behind a block can differ from rank to rank. Using rank 0's index
        then reads or writes another request's page on this rank: a load lands
        in it (and leaves this request's page unwritten) and a save stores it.

        Each transfer carries its block ordinal, which is resolved against this
        rank's own page list. A load whose block has no slot here is a logic
        error and raises; a save page without one is dropped (a later miss).
        Transfers without an ordinal keep their index.
        """
        if getattr(metadata, "_localized", False):
            return
        started = time.perf_counter()
        remapped, unresolved = self._remapped_pages, self._unresolved_save_pages
        resolve = self.get_local_page_indices()
        lookup = self._pgfix_lookup_fn(resolve)
        if lookup is not None:
            self._localize_fast(metadata, lookup)
        elif resolve is not None:
            tables: Dict[int, List[List[int]]] = {}

            def localize(entry: RequestTransfers, required: bool) -> RequestTransfers:
                pages: List[PageTransfer] = []
                table = None
                for page in entry.pages:
                    if page.block_ordinal < 0:
                        pages.append(page)
                        continue
                    if table is None:
                        table = tables.get(entry.request_id)
                        if table is None:
                            table = tables[entry.request_id] = resolve(entry.request_id)
                    group = table[page.layer_group_id] if page.layer_group_id < len(table) else []
                    local = group[page.block_ordinal] if page.block_ordinal < len(group) else -1
                    if local < 0:
                        if required:
                            raise RuntimeError(
                                f"mooncake-store rank {self._rank}: request {entry.request_id} has no page "
                                f"for block {page.block_ordinal} of layer group {page.layer_group_id} "
                                f"(rank 0 slot {page.page_index}), so its load has no destination"
                            )
                        self._unresolved_save_pages += 1
                        continue
                    if local != page.page_index:
                        if self._remapped_pages == 0:
                            logger.warning(
                                f"mooncake-store rank {self._rank}: request {entry.request_id} block "
                                f"{page.block_ordinal} of layer group {page.layer_group_id} is on slot "
                                f"{local} here and {page.page_index} on rank 0; using this rank's slots "
                                "(counted in the save statistics as remapped)"
                            )
                        self._remapped_pages += 1
                        page = PageTransfer(page.block_hash, page.layer_group_id, local, page.block_ordinal)
                    pages.append(page)
                return RequestTransfers(entry.request_id, pages)

            metadata.loads = [localize(entry, True) for entry in metadata.loads]
            metadata.saves = [localize(entry, False) for entry in metadata.saves]
            metadata.prefix_loads = [
                PrefixLoadTransfers(item.reservation_id, item.is_async, localize(item.transfers, True))
                for item in getattr(metadata, "prefix_loads", ())
            ]
        metadata._localized = True
        self._localize_seconds += time.perf_counter() - started
        if self._remapped_pages != remapped or self._unresolved_save_pages != unresolved:
            # Rare (only when this rank's allocation diverged from rank 0's), so every occurrence is logged.
            logger.warning(
                f"mooncake-store rank {self._rank} localize: remapped {self._remapped_pages - remapped} "
                f"unresolved save pages {self._unresolved_save_pages - unresolved} this pass; totals remapped "
                f"{self._remapped_pages} unresolved {self._unresolved_save_pages}"
            )

    # ---- PGFIX_FAST_ENV: slot lookups by block ordinal, in bulk ----

    def _pgfix_lookup_fn(self, resolve):
        """The KV cache manager's by-ordinal slot lookup when PGFIX_FAST_ENV is on and the full query is bound too."""
        if not getattr(self, "_pgfix_fast", 0) or resolve is None:
            return None
        return self.get_local_page_lookup()

    @staticmethod
    def _page_columns(pages: List[PageTransfer]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Block ordinals, layer groups and listed slots of `pages`, as int64 arrays."""
        n = len(pages)
        return (
            np.fromiter(map(_BLOCK_ORDINAL, pages), np.int64, n),
            np.fromiter(map(_LAYER_GROUP, pages), np.int64, n),
            np.fromiter(map(_PAGE_INDEX, pages), np.int64, n),
        )

    def _lookup_local_slots(
        self, lookup, request_id: int, ordinals: np.ndarray, groups: np.ndarray, wanted: np.ndarray
    ) -> np.ndarray:
        """This rank's slot of every page selected by `wanted`, -1 where it has none (and for unselected pages).

        The same values as indexing `get_local_page_indices()(request_id)[group][ordinal]` (-1 past the end).
        """
        local = np.full(len(ordinals), -1, dtype=np.int64)
        if not wanted.any():
            return local
        positions = {
            group: np.flatnonzero(wanted & (groups == group)) for group in np.unique(groups[wanted]).tolist()
        }
        verify = self._pgfix_fast >= 2
        found, fallbacks, mismatches = lookup(
            request_id, {group: ordinals[at] for group, at in positions.items()}, verify
        )
        for group, at in positions.items():
            local[at] = found[group]
        before = self._pgfix_lookups
        self._pgfix_lookups += len(positions)
        self._pgfix_fallbacks += fallbacks
        if before == 0 or (before.bit_length() - 1) // 2 != (self._pgfix_lookups.bit_length() - 1) // 2:
            # At 1, 4, 16, 64, ... lookups: the save statistics line (every 30 s) may never print in a short run.
            logger.info(
                f"mooncake-store rank {self._rank}: PGFIX_FAST lookups {self._pgfix_lookups} full-query "
                f"{self._pgfix_fallbacks} verify mismatches {self._pgfix_mismatches + mismatches} "
                f"(mode {self._pgfix_fast})"
            )
        if mismatches:
            self._pgfix_mismatches += mismatches
            logger.error(
                f"mooncake-store rank {self._rank}: PGFIX_FAST verify: request {request_id}: {mismatches} of "
                f"{len(positions)} layer group lookups differ from the full query (the full query's slots are used); "
                f"total {self._pgfix_mismatches}"
            )
        return local

    def _localize_fast(self, metadata: MooncakeStoreMetadata, lookup) -> None:
        """`_localize`'s rewrite with the slots looked up by ordinal (PGFIX_FAST_ENV); same result, page for page."""

        def localize(entry: RequestTransfers, required: bool) -> RequestTransfers:
            pages = entry.pages
            if not pages:
                return RequestTransfers(entry.request_id, [])
            ordinals, groups, listed = self._page_columns(pages)
            tagged = ordinals >= 0
            local = self._lookup_local_slots(lookup, entry.request_id, ordinals, groups, tagged)
            missing = tagged & (local < 0)
            moved = tagged & (local >= 0) & (local != listed)
            flagged = np.flatnonzero(missing | moved).tolist()
            if not flagged:
                result = RequestTransfers(entry.request_id, list(pages))
                # The columns of exactly these pages, for the commit-swap re-check of the same entry.
                result._page_cols = (ordinals, groups, listed)
                return result
            # Rare: walk the flagged pages in list order, exactly as `_localize` does page by page.
            out: List[Optional[PageTransfer]] = list(pages)
            dropped = False
            for i in flagged:
                page = pages[i]
                if missing[i]:
                    if required:
                        raise RuntimeError(
                            f"mooncake-store rank {self._rank}: request {entry.request_id} has no page "
                            f"for block {page.block_ordinal} of layer group {page.layer_group_id} "
                            f"(rank 0 slot {page.page_index}), so its load has no destination"
                        )
                    self._unresolved_save_pages += 1
                    out[i] = None
                    dropped = True
                    continue
                slot = int(local[i])
                if self._remapped_pages == 0:
                    logger.warning(
                        f"mooncake-store rank {self._rank}: request {entry.request_id} block "
                        f"{page.block_ordinal} of layer group {page.layer_group_id} is on slot "
                        f"{slot} here and {page.page_index} on rank 0; using this rank's slots "
                        "(counted in the save statistics as remapped)"
                    )
                self._remapped_pages += 1
                out[i] = PageTransfer(page.block_hash, page.layer_group_id, slot, page.block_ordinal)
            result = RequestTransfers(
                entry.request_id, [page for page in out if page is not None] if dropped else out
            )
            keep = ~missing
            result._page_cols = (ordinals[keep], groups[keep], np.where(moved, local, listed)[keep])
            return result

        metadata.loads = [localize(entry, True) for entry in metadata.loads]
        metadata.saves = [localize(entry, False) for entry in metadata.saves]
        metadata.prefix_loads = [
            PrefixLoadTransfers(item.reservation_id, item.is_async, localize(item.transfers, True))
            for item in getattr(metadata, "prefix_loads", ())
        ]

    def _recheck_after_commit_fast(
        self, saves: List[RequestTransfers], overrides: Dict, lookup, retarget: bool
    ) -> List[RequestTransfers]:
        """`_recheck_after_commit` with the slots looked up by ordinal (PGFIX_FAST_ENV); same result, page for page."""
        checked = retargeted = dropped = 0
        out: List[RequestTransfers] = []
        # (layer group, slot) keys of the snapshot overrides, as one slot array per group.
        overridden: Dict[int, List[int]] = {}
        for layer_group_id, page_index in overrides or ():
            overridden.setdefault(layer_group_id, []).append(page_index)
        overridden_slots = {group: np.array(slots, dtype=np.int64) for group, slots in overridden.items()}
        for entry in saves:
            pages = entry.pages
            if not pages:
                out.append(RequestTransfers(entry.request_id, []))
                continue
            # `_localize_fast` left the columns of exactly these pages on the entry (absent when the window snapshot
            # rebuilt the entry, or for a list it did not produce).
            cols = getattr(entry, "_page_cols", None)
            if cols is not None and len(cols[0]) == len(pages):
                ordinals, groups, listed = cols
            else:
                ordinals, groups, listed = self._page_columns(pages)
            check = ordinals >= 0
            for group, slots in overridden_slots.items():
                in_group = groups == group
                if in_group.any():
                    check[in_group] &= ~np.isin(listed[in_group], slots)
            current = self._lookup_local_slots(lookup, entry.request_id, ordinals, groups, check)
            checked += int(check.sum())
            changed = (check & (current != listed)).tolist()
            if not any(changed):
                out.append(RequestTransfers(entry.request_id, list(pages)))
                continue
            kept: List[PageTransfer] = []
            for i, page in enumerate(pages):
                if not changed[i]:
                    kept.append(page)
                    continue
                slot = int(current[i])
                if slot < 0 or not retarget:
                    dropped += 1
                    continue
                if self._swap_retargeted + retargeted == 0:
                    logger.warning(
                        f"mooncake-store rank {self._rank}: request {entry.request_id} block "
                        f"{page.block_ordinal} of layer group {page.layer_group_id} moved from slot "
                        f"{page.page_index} to {slot} at commit (the radix tree already held it); saving it "
                        "from the surviving page (counted in the save statistics as commit swaps)"
                    )
                retargeted += 1
                kept.append(PageTransfer(page.block_hash, page.layer_group_id, slot, page.block_ordinal))
            out.append(RequestTransfers(entry.request_id, kept))
        self._swap_checked += checked
        self._swap_retargeted += retargeted
        self._swap_dropped += dropped
        return out

    def _raise_load_error(self) -> None:
        """Surface a loader-thread failure on the executor thread.

        A parked request whose pages did not all land can neither run nor be
        reported done, and under tensor parallelism the ranks cannot agree to
        restart it, so the failure is fatal, as on the synchronous path.
        """
        with self._load_lock:
            error = self._load_error
        if error is not None:
            raise RuntimeError(
                f"mooncake-store: an asynchronous prefix load failed on rank {self._rank}"
            ) from error

    def _drain_loads(self, index: int, started: threading.Event) -> None:
        """A loader thread: one parked request at a time, on its own stream and staging."""
        try:
            if self._device_index is not None:
                # A new thread starts on device 0; see `_drain_saves`.
                torch.cuda.set_device(self._device_index)
            staging = self._async_load_stagings[index] if self._async_load_stagings else None
            stream = (
                torch.cuda.Stream() if staging is not None and torch.cuda.is_available() else None
            )
            handle = _stream_handle(stream)
        except Exception as exc:
            logger.error(
                f"mooncake-store load thread failed to start on rank {self._rank}: "
                f"{type(exc).__name__}: {exc}"
            )
            with self._load_lock:
                if self._load_startup_error is None:
                    self._load_startup_error = exc
            return
        finally:
            started.set()
        stats = _LoadStats()
        while True:
            item = self._load_queue.get()
            if item is None:
                return
            reservation_id, transfers, ready, queued_at = item
            gate_note = ""
            if self._launch_gate is not None:
                _gate_t0 = time.monotonic()
                gate_note = (
                    f" gate={self._launch_gate.wait()}:{time.monotonic() - _gate_t0:.4f}"
                )
            began = time.monotonic()
            try:
                for event in ready or ():
                    if stream is not None:
                        # The store fills host slots and may run ahead; only
                        # the scatters onto the pages wait for their release.
                        stream.wait_event(event)
                    else:
                        event.synchronize()
                loaded_bytes = self._load_pages([transfers], staging, handle)
            except Exception as exc:
                # Thread boundary: the executor thread raises it on its next
                # call, rather than leaving the request parked forever.
                logger.error(
                    f"mooncake-store rank {self._rank} failed the asynchronous load "
                    f"of reservation {reservation_id}: {type(exc).__name__}: {exc}"
                )
                with self._load_lock:
                    if self._load_error is None:
                        self._load_error = exc
                continue
            with self._load_lock:
                self._finished_prefix_loads.append(reservation_id)
            if _MCB_DIAG and self._rank == 0:
                logger.info(
                    f"[MCB] load rank {self._rank} loader {index} rid={reservation_id} "
                    f"req={transfers.request_id} pages={len(transfers.pages)} bytes={loaded_bytes} "
                    f"wait={began - queued_at:.4f} load={time.monotonic() - began:.4f} done={time.time():.3f} "
                    f"queued={self._load_queue.qsize()}{gate_note}"
                )
            report = stats.add(
                len(transfers.pages), loaded_bytes, began - queued_at, time.monotonic() - began
            )
            if report:
                logger.info(
                    f"mooncake-store rank {self._rank} loader {index}: {report}, "
                    f"{self._load_queue.qsize()} queued"
                )

    def _load_pages(
        self, transfers: Sequence[RequestTransfers], staging: Optional[HostStagingPool], handle: int
    ) -> int:
        """Fetch pages batch by batch and land them on the device before returning.

        Shared by the executor thread (synchronous loads, on its stream) and the
        loader thread (on its own stream and staging pool).

        Returns:
            The bytes loaded.
        """
        keys, addresses, sizes, total_pages = self._resolve(transfers)
        if not keys:
            return 0

        for batch in zip(
            _batched(keys, self._batch_size),
            _batched(addresses, self._batch_size),
            _batched(sizes, self._batch_size),
        ):
            batch_keys, batch_addresses, batch_sizes = batch
            if staging is None:
                target_addresses, target_sizes = list(batch_addresses), list(batch_sizes)
            else:
                target_addresses, target_sizes = describe_batch_for_get(staging, batch_sizes)
            results = self._store.batch_get_into_multi_buffers(
                list(batch_keys), target_addresses, target_sizes
            )
            failed = [
                key
                for key, result in zip(batch_keys, results)
                if not isinstance(result, int) or result < 0
            ]
            if failed or len(results) != len(batch_keys):
                # The runtime already counted these tokens as computed, so a
                # partial load would leave the forward pass reading
                # uninitialized KV and silently producing wrong tokens.
                raise RuntimeError(
                    f"mooncake-store failed to load {len(failed) or len(batch_keys)} of "
                    f"{len(batch_keys)} pages; the affected KV slots were already "
                    f"reported as computed. First failure: {failed[:1]}"
                )
            if staging is not None:
                # Only reached once every page in the batch landed, so no slot
                # holding a failed read is copied over a device page.
                unstage_batch_after_get(staging, batch_addresses, batch_sizes, handle)
                # The next batch reuses the slots and the forward pass reads
                # these pages, so the scatter has to complete before either.
                _sync_stream(handle)

        if self._stored_from_loads:
            # lkp L3: every page landed, so each key was in the pool just now; RESAVE need not ask again.
            self._local_stored.add_many(keys)
        if self._page_digest is not None:
            # Every batch landed (and was scattered) above: the bytes a forward reads.
            for entry in transfers:
                self._page_digest.loaded(entry.request_id, self._page_records(entry))
        logger.debug(f"mooncake-store rank {self._rank} loaded {total_pages} pages")
        return sum(sum(page_sizes) for page_sizes in sizes)

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        """No-op: loads complete in `start_load_kv`.

        Transfers are whole pages, so every layer of a group lands in one
        store call and nothing is outstanding when the first layer runs.
        """

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        """No-op: saves are submitted once per pass in `wait_for_save`.

        A page is only complete when every layer of its group has written its
        slice, so there is no correct per-layer submission point.
        """

    # ---- save path ----

    def wait_for_save(self, stream: torch.cuda.Stream):
        """Hand this pass's saves to the background thread, gated on an event."""
        if self._launch_gate is not None:
            # Called right after the forward returned on the host: its kernels are queued, so a load may start.
            self._launch_gate.launched()
        if self._deferred_saves:
            # Normally handed over by `get_finished` after the previous pass's commit; never later than this.
            self._flush_deferred_saves(retarget=False)
        metadata: Optional[MooncakeStoreMetadata] = self.get_connector_meta()
        if metadata is None or not self._config.role.saves:
            return
        retires = getattr(metadata, "retires", None) if self._window_keep > 0 else None
        if not metadata.saves:
            if retires:
                self._save_queue.put(list(retires))
            return
        self._localize(metadata)

        # The pages are written by kernels still queued on this stream, so the
        # event is the handoff: the thread reads GPU memory only after the
        # pass retires, without the executor loop waiting for it.
        saves, overrides = list(metadata.saves), {}
        if self._window_snapshot is not None:
            # Queued on this stream ahead of the event, so the copies read the pages before any later
            # forward, commit or recycle can rewrite them.
            saves, overrides = self._snapshot_window_pages(saves, stream)
        event = torch.cuda.Event()
        event.record(stream)

        with self._save_lock:
            for transfers in saves:
                self._outstanding_saves[transfers.request_id] += 1
        item = (event, saves, time.monotonic(), overrides)
        if self._swap_guard:
            # The commit of this pass's blocks has not run yet; see `COMMIT_SWAP_GUARD_ENV`. Counted as outstanding
            # above, so the requests' pages stay pinned until the save thread is done with them.
            self._deferred_saves.append((item, list(retires) if retires else None))
            return
        self._save_queue.put(item)
        if retires:
            # Behind the saves, so the new prompt end's window is in the pool before anything older is retired.
            self._save_queue.put(list(retires))

    def _flush_deferred_saves(self, retarget: bool = True) -> None:
        """Executor thread: hand the held save lists to the save thread, re-checked against this rank's pages.

        Called from `get_finished`, which the executor runs every iteration after committing the pass's blocks.
        `retarget=False` (the fallback in `wait_for_save` and `shutdown`, when the commit and the next scheduling pass
        may both have run) drops a page whose slot changed instead of following it: the request may have been
        rescheduled onto fresh pages that hold nothing yet.
        """
        pending, self._deferred_saves = self._deferred_saves, []
        resolve = self.get_local_page_indices()
        lookup = self._pgfix_lookup_fn(resolve)
        for (event, saves, queued_at, overrides), retires in pending:
            started = time.perf_counter()
            if lookup is not None:
                saves = self._recheck_after_commit_fast(saves, overrides, lookup, retarget)
            elif resolve is not None:
                saves = self._recheck_after_commit(saves, overrides, resolve, retarget)
            self._recheck_seconds += time.perf_counter() - started
            self._save_queue.put((event, saves, queued_at, overrides))
            if retires:
                self._save_queue.put(retires)

    def _recheck_after_commit(
        self, saves: List[RequestTransfers], overrides: Dict, resolve, retarget: bool
    ) -> List[RequestTransfers]:
        """`saves` with every page the commit swapped moved to the surviving page, and pages now gone dropped.

        Window pages already copied to snapshot slots (`overrides`) are not read from the device again, and a page
        without a block ordinal cannot be re-checked; both are kept as they are. Entries are kept even when empty, so
        the save thread retires each request's outstanding count.
        """
        tables: Dict[int, List[List[int]]] = {}
        checked = retargeted = dropped = 0
        out: List[RequestTransfers] = []
        for entry in saves:
            pages: List[PageTransfer] = []
            table = None
            for page in entry.pages:
                if page.block_ordinal < 0 or (page.layer_group_id, page.page_index) in overrides:
                    pages.append(page)
                    continue
                if table is None:
                    table = tables.get(entry.request_id)
                    if table is None:
                        table = tables[entry.request_id] = resolve(entry.request_id)
                group = table[page.layer_group_id] if page.layer_group_id < len(table) else []
                current = group[page.block_ordinal] if page.block_ordinal < len(group) else -1
                checked += 1
                if current == page.page_index:
                    pages.append(page)
                elif current < 0 or not retarget:
                    dropped += 1
                else:
                    if self._swap_retargeted + retargeted == 0:
                        logger.warning(
                            f"mooncake-store rank {self._rank}: request {entry.request_id} block "
                            f"{page.block_ordinal} of layer group {page.layer_group_id} moved from slot "
                            f"{page.page_index} to {current} at commit (the radix tree already held it); saving it "
                            "from the surviving page (counted in the save statistics as commit swaps)"
                        )
                    retargeted += 1
                    pages.append(PageTransfer(page.block_hash, page.layer_group_id, current, page.block_ordinal))
            out.append(RequestTransfers(entry.request_id, pages))
        self._swap_checked += checked
        self._swap_retargeted += retargeted
        self._swap_dropped += dropped
        return out

    # ---- window retirement (TRTLLM_MOONCAKE_STORE_WINDOW_KEEP) ----

    def _link_key(self, layer_group_id: int, block_hash: bytes) -> str:
        """The pool key of the link stored for the prompt end whose last block hashes to `block_hash`."""
        return f"{self._namespaces[layer_group_id].prefix}/link/{block_hash.hex()}"

    def _read_link(self, key: str) -> Optional[Tuple[int, int]]:
        """(previous prompt end, chain floor) stored under `key`, or None when absent or malformed."""
        value = self._store.get(key)
        if not value or len(value) < 16:
            return None
        return struct.unpack("<qq", bytes(value[:16]))

    def _remove_keys(self, keys: List[str]) -> int:
        """Non-forced removal of `keys`: a key under a read lease (or absent) stays. Returns how many went."""
        batch_remove = getattr(self._store, "batch_remove", None)
        if batch_remove is not None:
            results = list(batch_remove(keys, False))
            removed_keys = [key for key, result in zip(keys, results) if isinstance(result, int) and result == 0]
        else:
            removed_keys = []
            for key in keys:
                try:
                    result = self._store.remove(key, False)
                except TypeError:
                    result = self._store.remove(key)
                if isinstance(result, int) and result == 0:
                    removed_keys.append(key)
        if removed_keys and self._local_stored is not None:
            # The save path must not skip re-saving a page it no longer holds in the pool.
            self._local_stored.discard_many(removed_keys)
        return len(removed_keys)

    def _retire_windows(self, retires: Sequence[WindowRetire]) -> None:
        """Save thread: record each prompt end's link, follow the chain back and schedule the old window's removal.

        A link stores the prompt end its request resumed from and the chain's floor: the resume point of the
        chain's first linked end. Pages below the floor (a preamble, a prefix shared with other sessions) are never
        removed; a chain's first window (resumed from fewer than the link threshold) gets no link. The retired
        window loses its sliding-window pages from the floor up to where the oldest kept window starts.
        """
        groups = tuple(self._life_cycles)
        if not groups or self._store is None or self._addressing is None:
            return
        began = time.monotonic()
        tokens_per_block = self._addressing.tokens_per_block
        min_blocks = -(-self._link_min_tokens // tokens_per_block)
        link_group = groups[0]
        stats = self._retire_stats

        def read_link(end: int, hashes) -> Optional[Tuple[int, int]]:
            if end <= 0 or end > len(hashes):
                return None
            link = self._read_link(self._link_key(link_group, hashes[end - 1]))
            if link is None:
                return None
            previous, floor = link
            return (previous, floor) if min_blocks <= previous < end and floor <= previous else None

        for retire in retires:
            stats.ends += 1
            try:
                if retire.resume_block < max(1, min_blocks):
                    continue
                link = read_link(retire.resume_block, retire.hashes)
                floor = link[1] if link is not None else retire.resume_block
                status = self._store.put(
                    self._link_key(link_group, retire.end_hash),
                    struct.pack("<qq", retire.resume_block, floor),
                )
                stats.links_put += int(isinstance(status, int) and status == 0)
                # Kept windows, newest first: the new end, then the ends reached by following links.
                kept = [retire.end_block, retire.resume_block]
                while len(kept) <= self._window_keep and link is not None:
                    stats.links_found += 1
                    kept.append(link[0])
                    if len(kept) <= self._window_keep:
                        link = read_link(kept[-1], retire.hashes)
                if len(kept) <= self._window_keep:
                    continue
                target, newer = kept[self._window_keep], kept[self._window_keep - 1]
                keys = []
                for layer_group_id in groups:
                    sinks = self.sink_blocks(layer_group_id)
                    _, low = self.stale_blocks(layer_group_id, target * tokens_per_block)
                    _, newer_from = self.stale_blocks(layer_group_id, newer * tokens_per_block)
                    namespace = self._namespaces[layer_group_id]
                    for block in range(max(low, sinks, floor), min(target, newer_from)):
                        keys.append(namespace.key(retire.hashes[block]))
                if not keys:
                    # Everything the target window reads is below the floor or inside a kept window.
                    continue
                keys.append(self._link_key(link_group, retire.hashes[target - 1]))
                stats.retired += 1
                stats.total_retired += 1
                self._retire_seq += 1
                heapq.heappush(
                    self._retire_due, (time.monotonic() + self._retire_delay, self._retire_seq, keys)
                )
            except Exception as exc:
                logger.warning(
                    f"mooncake-store rank {self._rank} could not retire a window: {type(exc).__name__}: {exc}"
                )
        stats.busy += time.monotonic() - began
        self._remove_due_windows()

    def _remove_due_windows(self) -> None:
        """Save thread: remove retired windows whose delay has passed (non-forced: leased pages stay)."""
        now = time.monotonic()
        stats = self._retire_stats
        drained = False
        while self._retire_due and self._retire_due[0][0] <= now:
            drained = True
            _, _, keys = heapq.heappop(self._retire_due)
            try:
                removed = self._remove_keys(keys)
            except Exception as exc:
                logger.warning(
                    f"mooncake-store rank {self._rank} window removal failed: {type(exc).__name__}: {exc}"
                )
                removed = 0
            stats.removed += removed
            stats.total_removed += removed
            stats.kept += len(keys) - removed
        stats.busy += time.monotonic() - now
        # Also report when the last pending removal just went, so a burst's totals reach the log before the save
        # thread goes back to blocking on its queue.
        report = stats.report(len(self._retire_due), force=drained and not self._retire_due)
        if report:
            logger.info(f"mooncake-store rank {self._rank} window retirement: {report}")

    def get_finished(
        self, finished_gen_req_ids: List[int], started_loading_req_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Report which requests' saves have landed.

        Args:
            finished_gen_req_ids: Requests that will produce no further KV.
            started_loading_req_ids: Requests loading asynchronously. Always
                empty here, since `get_num_new_matched_tokens` only offers
                synchronous loads; echoed back so the runtime does not wait on
                something that already happened.

        Returns:
            Requests that have finished saving, and requests that have finished
            loading.
        """
        if self._deferred_saves:
            # The executor committed this iteration's blocks before calling here; see `COMMIT_SWAP_GUARD_ENV`.
            self._flush_deferred_saves()
        with self._save_lock:
            if _MCB_DIAG:
                _mcb_now = time.time()
                for request_id in finished_gen_req_ids:
                    self._mcb_closed_at.setdefault(request_id, _mcb_now)
            self._closed_requests.update(finished_gen_req_ids)
            finished_saving = [
                request_id
                for request_id in self._closed_requests
                if self._outstanding_saves.get(request_id, 0) == 0
            ]
            for request_id in finished_saving:
                self._closed_requests.discard(request_id)
                self._outstanding_saves.pop(request_id, None)
            if _MCB_DIAG:
                _mcb_now = time.time()
                for request_id in finished_saving:
                    closed = self._mcb_closed_at.pop(request_id, None)
                    if closed is not None and self._rank == 0:
                        logger.info(
                            f"[MCB] savehold rank {self._rank} req={request_id} hold={_mcb_now - closed:.4f} "
                            f"done={_mcb_now:.3f} open={len(self._closed_requests)}"
                        )
        if self._page_digest is not None:
            # Still pinned: the runtime frees them only once this returns them.
            for request_id in finished_saving:
                self._page_digest.retired(request_id)
        return finished_saving, list(started_loading_req_ids)

    def _drain_saves(self) -> None:
        try:
            # A new thread starts on device 0, so adopt the device captured on
            # the executor thread. Otherwise the stream below belongs to
            # device 0 while the KV pointers belong to the rank's device, and
            # the copy fails with cudaErrorInvalidValue on every rank but 0.
            if self._device_index is not None:
                torch.cuda.set_device(self._device_index)
            if self._save_staging is not None and torch.cuda.is_available():
                # Owned by this thread so the gather never queues behind the
                # executor's work.
                self._save_stream = torch.cuda.Stream()
        except Exception as exc:
            # Fatal, unlike a failure in the transfer loop below: a thread that
            # died here leaves every later save outstanding against nothing, so
            # the requests holding those pages never retire and the worker
            # looks merely slow.
            logger.error(
                f"mooncake-store save thread failed to start on rank {self._rank}: "
                f"{type(exc).__name__}: {exc}"
            )
            with self._save_lock:
                if self._save_startup_error is None:
                    self._save_startup_error = exc
            return
        finally:
            self._save_started.set()
        stats = _SaveStats()
        while True:
            if self._retire_due:
                # Window retirement only: wake up to remove what is due even when no save arrives.
                self._remove_due_windows()
                try:
                    item = self._save_queue.get(timeout=1.0)
                except Empty:
                    continue
            else:
                item = self._save_queue.get()
            if item is None:
                return
            if isinstance(item, list):
                # Window retirement work (only queued when TRTLLM_MOONCAKE_STORE_WINDOW_KEEP > 0).
                try:
                    self._retire_windows(item)
                except Exception as exc:
                    logger.warning(
                        f"mooncake-store rank {self._rank} window retirement failed: {type(exc).__name__}: {exc}"
                    )
                continue
            event, transfers, queued_at, *rest = item
            overrides = rest[0] if rest else {}
            began = time.monotonic()
            written = present = written_bytes = 0
            try:
                event.synchronize()
                written, present, written_bytes = self._put(transfers, overrides)
            except Exception as exc:
                # Broad on purpose: this is the thread boundary. The batch is
                # dropped and the request retired by the `finally` below, the
                # same policy `_put` applies to a negative store status.
                logger.error(
                    f"mooncake-store dropped a save batch on rank {self._rank}: "
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                )
            finally:
                if overrides:
                    self._window_snapshot.release([slot for slot, _, _ in overrides.values()])
                with self._save_lock:
                    for entry in transfers:
                        remaining = self._outstanding_saves.get(entry.request_id, 0) - 1
                        if remaining <= 0:
                            self._outstanding_saves.pop(entry.request_id, None)
                        else:
                            self._outstanding_saves[entry.request_id] = remaining
            if _MCB_DIAG and self._rank == 0:
                logger.info(
                    f"[MCB] save rank {self._rank} reqs={len(transfers)} "
                    f"pages={sum(len(e.pages) for e in transfers)} written={written} present={present} "
                    f"bytes={written_bytes} wait={began - queued_at:.4f} put={time.monotonic() - began:.4f} "
                    f"done={time.time():.3f} queued={self._save_queue.qsize()}"
                )
            report = stats.add(
                written, present, written_bytes, began - queued_at, time.monotonic() - began
            )
            if report:
                local_note = (
                    f", {self._local_stored.describe()}" if self._local_stored is not None else ""
                )
                local_note += (
                    f", slots remapped from rank 0 {self._remapped_pages}, "
                    f"unresolved save pages {self._unresolved_save_pages}, "
                    f"localize {self._localize_seconds * 1e3:.1f} ms total, "
                    f"commit swaps retargeted {self._swap_retargeted} dropped {self._swap_dropped} "
                    f"of {self._swap_checked} checked, recheck {self._recheck_seconds * 1e3:.1f} ms total"
                )
                if self._pgfix_fast:
                    local_note += (
                        f", pgfix fast lookups {self._pgfix_lookups} full-query {self._pgfix_fallbacks} "
                        f"verify mismatches {self._pgfix_mismatches}"
                    )
                if isinstance(self._store, SoftPinStore):
                    local_note += f", {self._store.put_summary()}"
                logger.info(
                    f"mooncake-store rank {self._rank} saves: {report}, {self._save_queue.qsize()} queued"
                    f"{local_note}"
                )

    def _put(self, transfers: Sequence[RequestTransfers], overrides: Optional[Dict] = None) -> Tuple[int, int, int]:
        """Write the pages the pool does not hold yet.

        `overrides` (from `_snapshot_window_pages`) are window pages already copied to pinned slots: they
        are put from those slots, the rest from the device pages through the save staging pool.

        Returns:
            Pages written, pages already stored, and the bytes written.
        """
        overrides = overrides or {}
        live, snap = [], []
        for entry in transfers:
            live.append(RequestTransfers(entry.request_id, [p for p in entry.pages if (p.layer_group_id, p.page_index) not in overrides]))
            snap.extend((entry.request_id, p) for p in entry.pages if (p.layer_group_id, p.page_index) in overrides)
        if self._page_digest is not None:
            # After the pass's event, before the store reads: the bytes a put ships.
            for entry in live:
                self._page_digest.saved(entry.request_id, self._page_records(entry))
        written = present_count = written_bytes = 0
        if snap:
            keys, _, _, _ = self._resolve([RequestTransfers(rid, [p]) for rid, p in snap])
            addresses = [[overrides[(p.layer_group_id, p.page_index)][1]] for _, p in snap]
            sizes = [[overrides[(p.layer_group_id, p.page_index)][2]] for _, p in snap]
            if self._page_digest is not None:
                self._page_digest.saved_host([(rid, key) for (rid, _), key in zip(snap, keys)], addresses, sizes)
            w, pc, wb = self._put_resolved(keys, addresses, sizes, staged=True)
            written, present_count, written_bytes = w, pc, wb
        keys, addresses, sizes, _ = self._resolve(live)
        if not keys:
            return written, present_count, written_bytes
        w, pc, wb = self._put_resolved(keys, addresses, sizes, staged=False)
        return written + w, present_count + pc, written_bytes + wb

    def _put_resolved(self, keys, addresses, sizes, staged: bool) -> Tuple[int, int, int]:
        """`_put`'s transfer loop over resolved pages; `staged` pages already sit in registered host slots."""
        written = present_count = written_bytes = 0

        staging = None if staged else self._save_staging
        handle = _stream_handle(self._save_stream) if staging is not None else 0

        for batch in zip(
            _batched(keys, self._batch_size),
            _batched(addresses, self._batch_size),
            _batched(sizes, self._batch_size),
        ):
            batch_keys, batch_addresses, batch_sizes = batch
            # Skip pages another rank or another instance already wrote. The
            # scheduler holds no store handle, and the answer changes between
            # the time it builds metadata and now.
            local = self._local_stored
            if local is None:
                present = self._store.batch_is_exist(list(batch_keys))
            else:
                present = self._probe_unless_known(local, batch_keys)
            pending = [
                index
                for index, status in enumerate(present)
                if status != 1  # absent, or a failed probe we retry as a write
            ]
            present_count += len(batch_keys) - len(pending)
            if not pending:
                continue
            source_addresses = [batch_addresses[i] for i in pending]
            source_sizes = [batch_sizes[i] for i in pending]
            if staging is not None:
                # Gathered after the existence filter, so a page already in the
                # pool costs no copy.
                source_addresses, source_sizes = stage_batch_for_put(
                    staging, source_addresses, source_sizes, handle
                )
                # The store reads the slots on this thread, so fill them first.
                _sync_stream(handle)
            pending_keys = [batch_keys[i] for i in pending]
            if _save_retry_disabled():
                results = self._store.batch_put_from_multi_buffers(
                    pending_keys,
                    source_addresses,
                    source_sizes,
                )
                ok = [isinstance(result, int) and result >= 0 for result in results]
            else:
                # Port of our e930 save retry (7eb3371446): up to 3 attempts
                # started within 100 ms, re-probing between them. The pages and
                # staging slots stay owned and unchanged until it returns.
                outcome = put_with_retries(
                    self._store, pending_keys, source_addresses, source_sizes
                )
                failed = set(outcome.failed_indices)
                ok = [index not in failed for index in range(len(pending_keys))]
                if outcome.retried_keys:
                    totals = getattr(self, "_save_retry_totals", None)
                    if totals is None:
                        totals = self._save_retry_totals = defaultdict(int)
                    totals["retried_keys"] += outcome.retried_keys
                    totals["recovered_keys"] += outcome.recovered_keys + outcome.recovered_by_probe
                    totals["failed_keys"] += len(failed)
                    logger.info(
                        f"mooncake-store rank {self._rank} save retry: totals {dict(totals)}"
                    )
            if local is not None:
                local.add_many(key for key, good in zip(pending_keys, ok) if good)
            failures = len(ok) - sum(ok)
            written += len(pending) - failures
            written_bytes += sum(sum(source_sizes[i]) for i, good in enumerate(ok) if good)
            if failures:
                # A dropped write only costs a future cache miss, which is not
                # worth failing a request that already answered.
                logger.warning(
                    f"mooncake-store rank {self._rank} failed to save {failures} of "
                    f"{len(pending)} pages"
                )
        return written, present_count, written_bytes

    def _probe_unless_known(self, local: LocalStoredSet, batch_keys: Sequence[str]) -> List[int]:
        """`batch_is_exist` statuses for `batch_keys`, answering 1 without an RPC for keys `local` confirms.

        Keys the probe reports present are confirmed into `local`. A probe that answers the wrong number of
        statuses leaves the unanswered keys at 0, so they are written (as a failed probe is today).
        """
        known = local.known(batch_keys)
        unknown = [index for index, is_known in enumerate(known) if not is_known]
        present = [1] * len(batch_keys)
        if not unknown:
            return present
        probed = list(self._store.batch_is_exist([batch_keys[index] for index in unknown]))
        probed += [0] * (len(unknown) - len(probed))
        for index, status in zip(unknown, probed):
            present[index] = status
        local.add_many(batch_keys[index] for index, status in zip(unknown, probed) if status == 1)
        return present

    # ---- shared ----

    def _page_records(self, entry: RequestTransfers) -> list:
        """(key, group, block ordinal, addresses, sizes) per page, for the digest log."""
        records = []
        for page in entry.pages:
            keys, addresses, sizes, _ = self._resolve([RequestTransfers(entry.request_id, [page])])
            ordinal = self._page_digest.ordinal(entry.request_id, page.block_hash)
            records.append((keys[0], page.layer_group_id, ordinal, addresses[0], sizes[0]))
        return records

    def _resolve(
        self, transfers: Sequence[RequestTransfers]
    ) -> Tuple[List[str], List[List[int]], List[List[int]], int]:
        """Expand per-request page transfers into parallel store call arguments."""
        if self._addressing is None:
            raise RuntimeError("KV cache layout has not been registered")
        keys: List[str] = []
        addresses: List[List[int]] = []
        sizes: List[List[int]] = []
        pages = 0
        for entry in transfers:
            for page in entry.pages:
                namespace = self._namespaces.get(page.layer_group_id)
                if namespace is None:
                    raise KeyError(
                        f"layer group {page.layer_group_id} is not in the registered "
                        "layout; the scheduler and worker disagree about the model"
                    )
                page_addresses, page_sizes = self._addressing.buffers(
                    page.layer_group_id, page.page_index
                )
                keys.append(namespace.key(page.block_hash))
                addresses.append(page_addresses)
                sizes.append(page_sizes)
                pages += 1
        return keys, addresses, sizes, pages

    def shutdown(self) -> None:
        """Stop the save and load threads, then release what they used. Idempotent."""
        if self._deferred_saves and self._save_thread is not None:
            # Ahead of the sentinel, so the save thread retires these requests' outstanding counts.
            self._flush_deferred_saves(retarget=False)
        threads = (
            [("save thread", self._save_thread, self._save_queue)] if self._save_thread else []
        )
        # The window snapshot ring is closed with the other pinned pools once the save thread is gone.
        threads += [("load thread", thread, self._load_queue) for thread in self._load_threads]
        # One sentinel per thread: a loader stops at the first one it takes,
        # after the loads queued ahead of it.
        for _, _, queue in threads:
            queue.put(None)
        for label, thread, _ in threads:
            thread.join(timeout=SAVE_DRAIN_TIMEOUT)
            if thread.is_alive():
                # The thread reads the KV pools through the store handle and
                # the staging slots below, so releasing either takes the memory
                # out from under a transfer in flight. Leaking both is the safer
                # end: the process is going down anyway, the thread is a daemon
                # and does not hold it open, and a later call retries the join.
                logger.error(
                    f"mooncake-store rank {self._rank}: the {label} "
                    f"did not stop within {SAVE_DRAIN_TIMEOUT:g}s, so the store handle and its "
                    "staging buffers are left in place rather than freed under a "
                    "transfer still reading them."
                )
                return
        self._save_thread = None
        self._load_threads = []
        if self._probe_pool is not None:
            # Lookups run on the scheduler thread, which is done by now.
            self._probe_pool.shutdown(wait=True)
            self._probe_pool = None
        store, self._store = self._store, None
        if store is not None:
            try:
                store.close()
            except Exception as exc:
                logger.warning(
                    f"mooncake-store close failed: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
        # Released only after the store is closed, since it holds registrations
        # against this memory.
        self._load_staging = None
        self._async_load_stagings = []
        self._save_staging = None
        self._window_snapshot = None
        self._save_stream = None
        global _LOCAL_WORKER
        if _LOCAL_WORKER is self:
            _LOCAL_WORKER = None
            _LOCAL_WORKER_READY.clear()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:  # noqa: S110 - interpreter teardown, nothing left to report to
            pass

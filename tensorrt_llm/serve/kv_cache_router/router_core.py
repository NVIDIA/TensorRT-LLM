# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Centralized KV-cache-aware router state and query API.

:class:`CentralizedKVCacheRouter` is the in-process core shared by the ZMQ ingest
server and the HTTP shim. It maintains, per routing namespace, a
:class:`WorkerPrefixTrie` of block ownership and a per-worker load snapshot, both
fed by ``apply_event_report`` / ``apply_load_report``. The query side,
``select_worker``, returns the instance with the best
``KV-match - load`` score for a request's block hashes -- the same scoring used
by the existing ``KvCacheAwareRouter``.

The transport (ZMQ PUSH/PULL today, optionally PUB/SUB later) lives outside this
class so swapping it never touches routing logic.
"""

import threading
import time
from typing import Dict, List, Optional

from tensorrt_llm.bindings.internal.batch_manager import (BlockKey,
                                                          BlockKeyHasher)
from tensorrt_llm.logger import logger

from .messages import KvCacheEventReport, Selection, WorkerLoadReport
from .prefix_trie import WorkerPrefixTrie

__all__ = ["CentralizedKVCacheRouter", "block_key_hasher"]


def block_key_hasher(token_ids: List[int],
                     parent_hash: Optional[int] = None) -> int:
    """Hash one block of token IDs, chaining in *parent_hash*.

    Mirrors ``tensorrt_llm.serve.router.block_key_hasher`` so request-side hashes
    line up with the hashes workers emit in their events.
    """
    block_key = BlockKey(token_ids)
    return BlockKeyHasher.hash(block_key,
                               0 if parent_hash is None else parent_hash)


class _NamespaceState:
    """All routing state for a single namespace."""

    __slots__ = ("trie", "loads", "last_event_seq", "last_load_seq",
                 "last_load_ts", "last_seen_ts", "rr_counter")

    def __init__(self) -> None:
        self.trie = WorkerPrefixTrie()
        self.loads: Dict[str, WorkerLoadReport] = {}
        self.last_event_seq: Dict[str, int] = {}
        self.last_load_seq: Dict[str, int] = {}
        # Time of the last LOAD report specifically (separate from last_seen_ts,
        # which any report touches). Drives the load-staleness suspend gate.
        self.last_load_ts: Dict[str, float] = {}
        self.last_seen_ts: Dict[str, float] = {}
        self.rr_counter = 0


class CentralizedKVCacheRouter:
    """Thread-safe KV-cache-aware router state with a block-hash query API.

    Args:
        tokens_per_block: KV-cache block size, used to convert matched blocks to
            matched tokens in the score. Must match the workers' configuration.
        load_suspend_s: A worker whose most recent *load* report is older than
            this is **suspended** -- excluded from selection until it reports load
            again. Its KV-cache entries are retained (so it can resume instantly),
            but without a fresh load reading the router cannot trust its capacity.
            Should be a small multiple of the reporter's ``load_interval_s``.
        stale_timeout_s: A worker with no report of *any* kind newer than this is
            fully evicted (trie + load) by :meth:`evict_stale_workers`. This is a
            longer, harder timeout than ``load_suspend_s``.
        clock: Injectable time source (seconds); defaults to ``time.monotonic``.
    """

    def __init__(self,
                 tokens_per_block: int = 32,
                 load_suspend_s: float = 3.0,
                 stale_timeout_s: float = 30.0,
                 clock=time.monotonic) -> None:
        self._tokens_per_block = tokens_per_block
        self._load_suspend_s = load_suspend_s
        self._stale_timeout_s = stale_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._namespaces: Dict[str, _NamespaceState] = {}
        # worker_id (== llm_id) -> server address, learned from /server_info.
        # Fleet-unique, so it is NOT namespace-scoped. Kept consistent with the
        # server list by register/unregister calls from the adapter.
        self._worker_address: Dict[str, str] = {}
        # Reverse index so a re-registered address (worker restarted under a new
        # id at the same URL) cleanly supersedes the stale id.
        self._address_worker: Dict[str, str] = {}

    # ----------------------------------------------------- worker<->address map

    def register_worker_address(self, worker_id: str, address: str) -> None:
        """Record the ``worker_id -> address`` mapping from ``/server_info``.

        Idempotent. If *address* was previously bound to a different
        ``worker_id`` (the instance at that URL restarted with a new id), the old
        binding is dropped so the map never points two ids at one URL or returns
        a dead id for that URL.
        """
        if not worker_id or not address:
            return
        with self._lock:
            prev_id = self._address_worker.get(address)
            if prev_id is not None and prev_id != worker_id:
                self._worker_address.pop(prev_id, None)
            prev_addr = self._worker_address.get(worker_id)
            if prev_addr is not None and prev_addr != address:
                self._address_worker.pop(prev_addr, None)
            self._worker_address[worker_id] = address
            self._address_worker[address] = worker_id

    def unregister_worker_address(self,
                                  *,
                                  worker_id: Optional[str] = None,
                                  address: Optional[str] = None) -> None:
        """Drop a worker's address mapping AND its routing state.

        The adapter calls this with ``address`` when a server leaves the pool
        (it may not know the id), so the id is resolved from the reverse index.
        Routing state (trie + load) is forgotten across all namespaces too, so a
        removed server stops being a routing candidate immediately rather than
        waiting for stale-eviction.
        """
        with self._lock:
            if address is not None and worker_id is None:
                worker_id = self._address_worker.get(address)
            if worker_id is not None:
                addr = self._worker_address.pop(worker_id, None)
                if addr is not None:
                    self._address_worker.pop(addr, None)
                for ns in self._namespaces.values():
                    self._forget_locked(ns, worker_id)
            if address is not None:
                self._address_worker.pop(address, None)

    def address_of(self, worker_id: str) -> Optional[str]:
        """Resolve a ``worker_id`` to its address, or ``None`` if unknown."""
        with self._lock:
            return self._worker_address.get(worker_id)

    # ------------------------------------------------------------------ ingest

    def apply_event_report(self, report: KvCacheEventReport) -> None:
        """Apply a KV-cache event report. Stale (``seq``) reports are dropped."""
        with self._lock:
            ns = self._namespaces.setdefault(report.namespace, _NamespaceState())
            last = ns.last_event_seq.get(report.worker_id, -1)
            if report.seq <= last and not report.is_full_snapshot:
                logger.debug(
                    f"KVCacheRouter: drop stale event report from "
                    f"{report.worker_id} seq={report.seq} <= {last}")
                return
            ns.last_event_seq[report.worker_id] = report.seq
            ns.last_seen_ts[report.worker_id] = self._clock()
            if report.is_full_snapshot:
                ns.trie.remove_worker(report.worker_id)
            self._apply_events(ns.trie, report.worker_id, report.events)

    def apply_load_report(self, report: WorkerLoadReport) -> None:
        """Apply a worker load report. Stale (``seq``) reports are dropped."""
        with self._lock:
            ns = self._namespaces.setdefault(report.namespace, _NamespaceState())
            last = ns.last_load_seq.get(report.worker_id, -1)
            if report.seq <= last:
                logger.debug(
                    f"KVCacheRouter: drop stale load report from "
                    f"{report.worker_id} seq={report.seq} <= {last}")
                return
            now = self._clock()
            ns.last_load_seq[report.worker_id] = report.seq
            ns.last_load_ts[report.worker_id] = now
            ns.last_seen_ts[report.worker_id] = now
            ns.loads[report.worker_id] = report

    def drop_worker(self, namespace: str, worker_id: str) -> None:
        """Forget a worker entirely (e.g. on disconnect)."""
        with self._lock:
            ns = self._namespaces.get(namespace)
            if ns is None:
                return
            self._forget_locked(ns, worker_id)

    def evict_stale_workers(self, now: Optional[float] = None) -> None:
        """Drop workers whose last report is older than ``stale_timeout_s``."""
        now = self._clock() if now is None else now
        with self._lock:
            for ns in self._namespaces.values():
                stale = [
                    w for w, ts in ns.last_seen_ts.items()
                    if now - ts > self._stale_timeout_s
                ]
                for w in stale:
                    self._forget_locked(ns, w)
                    logger.info(f"KVCacheRouter: evicted stale worker {w}")

    # ------------------------------------------------------------------- query

    def select_worker(self, namespace: str,
                      block_hashes: List[int]) -> Optional[Selection]:
        """Return the best worker in *namespace* for *block_hashes*.

        Score per worker is ``matched_tokens / padded_tokens -
        workload / max_batch_size`` (matching ``KvCacheAwareRouter``), where
        ``workload = num_active_requests + num_queued_requests``. Ties break
        round-robin. Returns ``None`` if the namespace has no eligible workers.

        Workers whose most recent load report is older than ``load_suspend_s``
        are **suspended**: excluded from candidates until they report load again.
        """
        with self._lock:
            ns = self._namespaces.get(namespace)
            if ns is None or not ns.loads:
                return None

            matched = ns.trie.match(block_hashes) if block_hashes else {}
            padded = max(len(block_hashes), 1) * self._tokens_per_block

            now = self._clock()
            best_score = None
            best_workers: List[str] = []
            for w in ns.loads:
                # Suspend routing to a worker with a stale load reading.
                load_ts = ns.last_load_ts.get(w, 0.0)
                if now - load_ts > self._load_suspend_s:
                    continue
                load = ns.loads[w]
                m_tokens = matched.get(w, 0) * self._tokens_per_block
                workload = load.num_active_requests + load.num_queued_requests
                max_bs = max(load.max_batch_size, 1)
                score = m_tokens / padded - workload / max_bs
                if best_score is None or score > best_score:
                    best_score = score
                    best_workers = [w]
                elif score == best_score:
                    best_workers.append(w)

            if not best_workers:
                return None
            winner = best_workers[ns.rr_counter % len(best_workers)]
            ns.rr_counter += 1
            return Selection(worker_id=winner,
                             address=self._worker_address.get(winner),
                             matched_blocks=matched.get(winner, 0))

    def select_worker_from_tokens(self, namespace: str,
                                  token_ids: List[int]) -> Optional[Selection]:
        """Convenience: hash *token_ids* into block hashes, then select.

        Drops the final token before blocking, matching the KV-cache manager
        (the last token is not part of a block key) and
        ``BlockHashMixin._compute_block_hashes``.
        """
        block_hashes = self._hash_tokens(token_ids)
        return self.select_worker(namespace, block_hashes)

    # --------------------------------------------------------------- internals

    def _hash_tokens(self, token_ids: List[int]) -> List[int]:
        tpb = self._tokens_per_block
        hashes: List[int] = []
        # The last token is excluded from block keys (see KvCacheManager).
        end = len(token_ids) - 1
        t = 0
        while t < end:
            t_end = min(t + tpb, end)
            parent = hashes[-1] if hashes else None
            hashes.append(block_key_hasher(token_ids[t:t_end], parent))
            t += tpb
        return hashes

    @staticmethod
    def _apply_events(trie: WorkerPrefixTrie, worker_id: str,
                      events: List[dict]) -> None:
        for event_raw in events:
            data = event_raw.get("data", event_raw)
            etype = data.get("type")
            if etype == "stored":
                block_hashes = [
                    b["block_hash"] for b in data.get("blocks", [])
                ]
                trie.add(worker_id, block_hashes)
            elif etype == "removed":
                trie.remove(worker_id, data.get("block_hashes", []))
            # "created" / "updated" carry no membership change for matching.

    @staticmethod
    def _forget_locked(ns: _NamespaceState, worker_id: str) -> None:
        ns.trie.remove_worker(worker_id)
        ns.loads.pop(worker_id, None)
        ns.last_event_seq.pop(worker_id, None)
        ns.last_load_seq.pop(worker_id, None)
        ns.last_load_ts.pop(worker_id, None)
        ns.last_seen_ts.pop(worker_id, None)

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
server and the HTTP shim. It maintains, per routing namespace, a hierarchical
instance -> rank structure:

- Each instance has a **combined trie** (union of all its ranks' hashes) for
  instance-level selection.
- Each rank within an instance has its own trie for rank-level selection.

The query side, ``select_worker``, performs two-phase routing:
1. Select the best *instance* using the combined trie and total instance load.
2. Select the best *rank* within that instance using per-rank tries and loads.

For legacy (non-per-rank) reporters, the instance IS the worker -- a single
"rank 0" state is used transparently.

The transport (ZMQ PUSH/PULL today, optionally PUB/SUB later) lives outside this
class so swapping it never touches routing logic.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

from tensorrt_llm.bindings.internal.batch_manager import (BlockKey,
                                                          BlockKeyHasher)
from tensorrt_llm.logger import logger

from .messages import KvCacheEventReport, Selection, WorkerLoadReport
from .prefix_trie import WorkerPrefixTrie

__all__ = [
    "CentralizedKVCacheRouter", "block_key_hasher",
    "score_kv_aware_candidates"
]


def score_kv_aware_candidates(
    candidates: List[Tuple[int, int, float]],
    *,
    load_weight: float,
    fair_share_multiplier: float,
    match_rate_threshold: float,
    total_units: int,
    debug_out: Optional[dict] = None,
) -> Optional[List[int]]:
    """Shared KV-cache-aware selection scorer used by BOTH the worker
    ``KVCacheAwareADPRouter`` and the centralized router's phase-2 rank pick,
    so the two run identical selection logic.

    Each candidate is ``(id, matched_units, load)`` where *matched_units* is the
    cached prefix length (tokens or blocks) and *load* is the candidate's active
    work in the same units' scale. *total_units* is the request's full prefix
    length (tokens or blocks). Higher score = better; returns the list of
    top-scoring candidate ids (ties preserved, caller breaks them).

    Algorithm (mirrors KVCacheAwareADPRouter):
      1. Fair-share cap: drop candidates whose load exceeds
         ``fair_share_multiplier * mean_load`` (unless that empties the set).
      2. Cache-affinity gate: if the best match covers less than
         ``match_rate_threshold`` of the request, zero all matches (route by
         load only).
      3. Score = matched - load_weight * (load / total_load * total_units),
         i.e. load normalized to the same unit scale as the match term.
    """
    if not candidates:
        return None

    # (1) Fair-share cap.
    mean_load = sum(c[2] for c in candidates) / len(candidates)
    cap = fair_share_multiplier * max(mean_load, 1.0)
    eligible = [c for c in candidates if c[2] <= cap] or candidates

    # (2) Cache-affinity gate.
    denom_units = max(total_units, 1)
    max_match = max((c[1] for c in eligible), default=0)
    cache_active = (max_match / denom_units) > match_rate_threshold

    # (3) Normalized-load scoring.
    total_load = sum(c[2] for c in eligible)
    load_denom = max(total_load, 1.0)
    scale = float(total_units) if total_units > 0 else 1.0

    best_score = None
    best_ids: List[int] = []
    rows = [] if debug_out is not None else None
    for cid, matched, load in eligible:
        match_term = matched if cache_active else 0
        normalized_load = load / load_denom * scale
        load_term = load_weight * normalized_load
        score = match_term - load_term
        if rows is not None:
            rows.append((cid, matched, load, match_term, round(load_term, 1),
                         round(score, 1)))
        if best_score is None or score > best_score:
            best_score = score
            best_ids = [cid]
        elif score == best_score:
            best_ids.append(cid)
    if debug_out is not None:
        debug_out.update(
            cap=cap, mean_load=mean_load, cache_active=cache_active,
            max_match=max_match, total_units=total_units,
            n_eligible=len(eligible), n_cands=len(candidates),
            # rows: (rank, matched, load, match_term, load_term, score)
            rows=rows, winners=best_ids)
    return best_ids


def block_key_hasher(token_ids: List[int],
                     parent_hash: Optional[int] = None) -> int:
    """Hash one block of token IDs, chaining in *parent_hash*.

    Mirrors ``tensorrt_llm.serve.router.block_key_hasher`` so request-side hashes
    line up with the hashes workers emit in their events.
    """
    block_key = BlockKey(token_ids)
    return BlockKeyHasher.hash(block_key,
                               0 if parent_hash is None else parent_hash)


def _parse_worker_id(worker_id: str) -> Tuple[str, Optional[int]]:
    """Parse a composite worker_id into (instance_id, rank).

    Format: "instance_id:rankN" -> (instance_id, N)
    Legacy: "instance_id" -> (instance_id, None)
    """
    idx = worker_id.rfind(":rank")
    if idx == -1:
        return worker_id, None
    instance_id = worker_id[:idx]
    try:
        rank = int(worker_id[idx + 5:])
        return instance_id, rank
    except ValueError:
        return worker_id, None


class _RankState:
    """Per-rank state within an instance."""

    __slots__ = ("rank", "trie", "load", "last_event_seq", "last_load_seq",
                 "last_load_ts", "last_seen_ts", "layer_group_refcount",
                 "owned_hashes")

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.trie = WorkerPrefixTrie()
        self.load: Optional[WorkerLoadReport] = None
        self.last_event_seq: int = -1
        self.last_load_seq: int = -1
        self.last_load_ts: float = 0.0
        self.last_seen_ts: float = 0.0
        self.layer_group_refcount: Dict[int, int] = {}
        # Flat set of block hashes this rank holds. In "none" mode the per-rank
        # trie is not maintained (never queried), so this set is what
        # _remove_rank_from_combined uses to clean the combined trie on a
        # full-snapshot resync. Cheap to maintain (one set op per event) vs the
        # trie's dual-index inserts. When the trie IS maintained, this mirrors
        # trie._worker_blocks[worker_id].
        self.owned_hashes: set = set()


class _InstanceState:
    """All ranks of one worker instance."""

    __slots__ = ("instance_id", "ranks", "combined_trie",
                 "combined_refcount", "rr_counter", "lock")

    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        self.ranks: Dict[int, _RankState] = {}
        self.combined_trie = WorkerPrefixTrie()
        # Tracks how many ranks in this instance still hold each hash.
        # Only remove from combined_trie when count drops to 0.
        self.combined_refcount: Dict[int, int] = {}
        # Round-robin tie-break across ranks (mirrors _NamespaceState.rr_counter
        # used for instance selection in phase 1).
        self.rr_counter = 0
        # Per-instance ("per tree") lock guarding this instance's mutable
        # routing state: ranks, each rank's trie/load/seqs/owned_hashes/
        # layer_group_refcount, combined_trie, combined_refcount, rr_counter.
        # Splitting the formerly-global lock per instance lets event ingest for
        # one instance run concurrently with select_worker reads of (and ingest
        # to) other instances -- the ingest/query contention that dominated
        # adp-mode get_next_server latency. Lock ordering is strictly
        # struct-lock -> instance-lock; the hot paths never hold the struct lock
        # while taking an instance lock, so there is no cycle.
        self.lock = threading.Lock()

    def total_load(self) -> Tuple[int, int]:
        """Sum of (active_requests, queued_requests) across all ranks."""
        active = 0
        queued = 0
        for rs in self.ranks.values():
            if rs.load is not None:
                active += rs.load.num_active_requests
                queued += rs.load.num_queued_requests
        return active, queued

    def freshest_load_ts(self) -> float:
        """Most recent load timestamp across all ranks."""
        ts = 0.0
        for rs in self.ranks.values():
            if rs.last_load_ts > ts:
                ts = rs.last_load_ts
        return ts

    def freshest_seen_ts(self) -> float:
        """Most recent seen timestamp across all ranks."""
        ts = 0.0
        for rs in self.ranks.values():
            if rs.last_seen_ts > ts:
                ts = rs.last_seen_ts
        return ts


class _NamespaceState:
    """All routing state for a single namespace."""

    __slots__ = ("instances", "rr_counter",
                 # Legacy flat state for non-per-rank workers
                 "trie", "loads", "last_event_seq", "last_load_seq",
                 "last_load_ts", "last_seen_ts", "layer_group_refcount")

    def __init__(self) -> None:
        # Hierarchical per-rank state
        self.instances: Dict[str, _InstanceState] = {}
        self.rr_counter = 0
        # Legacy flat state (for workers without :rank suffix)
        self.trie = WorkerPrefixTrie()
        self.loads: Dict[str, WorkerLoadReport] = {}
        self.last_event_seq: Dict[str, int] = {}
        self.last_load_seq: Dict[str, int] = {}
        self.last_load_ts: Dict[str, float] = {}
        self.last_seen_ts: Dict[str, float] = {}
        self.layer_group_refcount: Dict[str, Dict[int, int]] = {}


class CentralizedKVCacheRouter:
    """Thread-safe KV-cache-aware router state with a block-hash query API.

    Supports both legacy single-worker reporting and per-rank hierarchical
    reporting. The format of ``worker_id`` determines the mode:
    - ``"instance_id:rankN"`` -> per-rank mode (hierarchical routing)
    - ``"instance_id"`` -> legacy mode (flat routing, backward-compatible)

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

    #: Selectable phase-2 (per-rank) routing algorithms.
    RANK_ALGO_INSTANCE = "instance"  # same scoring as phase-1 instance routing
    RANK_ALGO_ADP = "adp"            # same scoring as the kvcache ADP router
    RANK_ALGO_NONE = "none"          # instance-only: no rank pick, no route_hint;
                                     # the worker's own ADP router selects the rank
                                     # (per-rank events still feed instance scoring,
                                     # but the host-side allgather is eliminated).

    def __init__(self,
                 tokens_per_block: int = 32,
                 load_weight: float = 0.25,
                 rank_routing_algo: str = RANK_ALGO_INSTANCE,
                 fair_share_multiplier: float = 2.0,
                 match_rate_threshold: float = 0.1,
                 load_suspend_s: float = 3.0,
                 stale_timeout_s: float = 30.0,
                 clock=time.monotonic) -> None:
        self._tokens_per_block = tokens_per_block
        self._load_weight = load_weight
        # Phase-2 rank-selection algorithm (tunable):
        #   "instance" -> identical scoring to phase-1 instance routing
        #                 (matched - load_weight * workload, argmax, RR tie-break)
        #   "adp"      -> identical scoring to the worker KVCacheAwareADPRouter
        #                 (fair-share cap + cache-affinity gate + normalized load,
        #                  via the shared score_kv_aware_candidates).
        if rank_routing_algo not in (self.RANK_ALGO_INSTANCE,
                                     self.RANK_ALGO_ADP,
                                     self.RANK_ALGO_NONE):
            raise ValueError(
                f"rank_routing_algo must be one of "
                f"{self.RANK_ALGO_INSTANCE!r}, {self.RANK_ALGO_ADP!r}, "
                f"{self.RANK_ALGO_NONE!r}; got {rank_routing_algo!r}")
        self._rank_routing_algo = rank_routing_algo
        # In "none" mode phase-2 is skipped, so the per-rank tries (rs.trie) are
        # NEVER queried -- only the instance combined_trie is. Skip maintaining
        # them on every event to halve event-ingest write work (and shorten the
        # lock-hold that contends with select_worker queries). The combined_trie
        # and all refcounts are still maintained.
        self._skip_rank_trie = (rank_routing_algo == self.RANK_ALGO_NONE)
        # ADP-algo knobs (used only when rank_routing_algo == "adp").
        self._fair_share_multiplier = max(1.0, float(fair_share_multiplier))
        self._match_rate_threshold = match_rate_threshold
        # Diagnostic: when TLLM_LOG_ROUTE_SCORES=1, log the per-rank cache-term
        # vs load-term breakdown of every Nth phase-2 decision, to root-cause
        # why a balance metric moves. Counter-gated to avoid log spam.
        import os
        self._log_route_scores = (
            os.environ.get("TLLM_LOG_ROUTE_SCORES", "0") == "1")
        self._score_log_n = 0
        self._load_suspend_s = load_suspend_s
        self._stale_timeout_s = stale_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._namespaces: Dict[str, _NamespaceState] = {}
        # instance_id -> server address, learned from /server_info.
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
                    self._forget_instance_locked(ns, worker_id)
                    self._forget_legacy_locked(ns, worker_id)
            if address is not None:
                self._address_worker.pop(address, None)

    def address_of(self, worker_id: str) -> Optional[str]:
        """Resolve a ``worker_id`` to its address, or ``None`` if unknown."""
        with self._lock:
            return self._worker_address.get(worker_id)

    # ------------------------------------------------------------------ ingest

    def apply_event_report(self, report: KvCacheEventReport) -> None:
        """Apply a KV-cache event report. Stale (``seq``) reports are dropped."""
        instance_id, rank = _parse_worker_id(report.worker_id)
        if rank is None:
            # Legacy flat workers live in struct-locked namespace state.
            with self._lock:
                ns = self._namespaces.setdefault(report.namespace,
                                                  _NamespaceState())
                self._apply_legacy_event_locked(ns, report)
            return
        # Per-rank: take the struct lock only long enough to find/create the
        # instance (a dict op -- no trie work), then do the heavy trie/refcount
        # mutation under that instance's own lock so it doesn't block ingest of
        # or queries to other instances.
        inst = self._get_or_create_instance(report.namespace, instance_id)
        with inst.lock:
            self._apply_per_rank_event_inst_locked(inst, rank, report)

    def _get_or_create_instance(self, namespace: str,
                                instance_id: str) -> _InstanceState:
        with self._lock:
            ns = self._namespaces.setdefault(namespace, _NamespaceState())
            inst = ns.instances.get(instance_id)
            if inst is None:
                inst = _InstanceState(instance_id)
                ns.instances[instance_id] = inst
            return inst

    def _apply_per_rank_event_inst_locked(self, inst: _InstanceState,
                                          rank: int,
                                          report: KvCacheEventReport) -> None:
        """Apply a per-rank event report. Caller must hold ``inst.lock``."""
        rs = inst.ranks.setdefault(rank, _RankState(rank))

        if report.seq <= rs.last_event_seq and not report.is_full_snapshot:
            logger.debug(
                f"KVCacheRouter: drop stale event report from "
                f"{report.worker_id} seq={report.seq} <= {rs.last_event_seq}")
            return

        rs.last_event_seq = report.seq
        rs.last_seen_ts = self._clock()

        if report.is_full_snapshot:
            # Remove this rank's hashes from combined trie (uses owned_hashes).
            self._remove_rank_from_combined(inst, rank)
            rs.trie.remove_worker(report.worker_id)
            rs.owned_hashes.clear()
            rs.layer_group_refcount.clear()

        # Apply events to per-rank trie AND combined trie
        self._apply_events_hierarchical(
            inst, rs, report.worker_id, report.events)

    def _apply_legacy_event_locked(self, ns: _NamespaceState,
                                   report: KvCacheEventReport) -> None:
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
            ns.layer_group_refcount.pop(report.worker_id, None)
        self._apply_events(ns.trie, report.worker_id, report.events,
                           ns.layer_group_refcount)

    def apply_load_report(self, report: WorkerLoadReport) -> None:
        """Apply a worker load report. Stale (``seq``) reports are dropped."""
        instance_id, rank = _parse_worker_id(report.worker_id)
        if rank is None:
            with self._lock:
                ns = self._namespaces.setdefault(report.namespace,
                                                  _NamespaceState())
                self._apply_legacy_load_locked(ns, report)
            return
        inst = self._get_or_create_instance(report.namespace, instance_id)
        with inst.lock:
            self._apply_per_rank_load_inst_locked(inst, rank, report)

    def _apply_per_rank_load_inst_locked(self, inst: _InstanceState,
                                         rank: int,
                                         report: WorkerLoadReport) -> None:
        """Apply a per-rank load report. Caller must hold ``inst.lock``."""
        rs = inst.ranks.setdefault(rank, _RankState(rank))

        if report.seq <= rs.last_load_seq:
            logger.debug(
                f"KVCacheRouter: drop stale load report from "
                f"{report.worker_id} seq={report.seq} <= {rs.last_load_seq}")
            return

        now = self._clock()
        rs.last_load_seq = report.seq
        rs.last_load_ts = now
        rs.last_seen_ts = now
        rs.load = report

    def _apply_legacy_load_locked(self, ns: _NamespaceState,
                                  report: WorkerLoadReport) -> None:
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
            instance_id, rank = _parse_worker_id(worker_id)
            if rank is not None:
                self._forget_rank_locked(ns, instance_id, rank)
            else:
                self._forget_legacy_locked(ns, worker_id)

    def evict_stale_workers(self, now: Optional[float] = None) -> None:
        """Drop workers whose last report is older than ``stale_timeout_s``."""
        now = self._clock() if now is None else now
        with self._lock:
            for ns in self._namespaces.values():
                # Evict stale legacy workers
                stale = [
                    w for w, ts in ns.last_seen_ts.items()
                    if now - ts > self._stale_timeout_s
                ]
                for w in stale:
                    self._forget_legacy_locked(ns, w)
                    logger.info(f"KVCacheRouter: evicted stale worker {w}")

                # Evict stale per-rank instances
                stale_instances = []
                for inst_id, inst in ns.instances.items():
                    if now - inst.freshest_seen_ts() > self._stale_timeout_s:
                        stale_instances.append(inst_id)
                for inst_id in stale_instances:
                    del ns.instances[inst_id]
                    logger.info(
                        f"KVCacheRouter: evicted stale instance {inst_id}")

    # ------------------------------------------------------------------- query

    def select_worker(self, namespace: str,
                      block_hashes: List[int]) -> Optional[Selection]:
        """Return the best worker in *namespace* for *block_hashes*.

        Two-phase routing for per-rank workers:
        1. Select best instance using combined trie + total instance load.
        2. Select best rank within that instance using per-rank trie + rank load.

        Legacy (flat) workers are scored alongside instances in phase 1.
        Returns ``None`` if the namespace has no eligible workers.
        """
        # Phase 0: snapshot instance refs + score legacy workers under the
        # (now structural) lock, then release it. The expensive per-instance
        # trie matching happens below under each instance's own lock, so it
        # runs concurrently with event ingest to *other* instances -- the
        # ingest/query contention on a single global lock was what dominated
        # adp-mode get_next_server latency.
        with self._lock:
            ns = self._namespaces.get(namespace)
            if ns is None:
                return None
            now = self._clock()
            inst_items = list(ns.instances.items())
            # Legacy flat workers live in struct-locked namespace state; score
            # them here while we hold the struct lock.
            legacy_matched = (ns.trie.match(block_hashes)
                              if block_hashes else {})
            legacy_scored: List[Tuple[str, int, float]] = []
            for w in ns.loads:
                load_ts = ns.last_load_ts.get(w, 0.0)
                if now - load_ts > self._load_suspend_s:
                    continue
                load = ns.loads[w]
                mb = legacy_matched.get(w, 0)
                workload = (load.num_active_requests
                            + load.num_queued_requests)
                legacy_scored.append(
                    (w, mb, mb - self._load_weight * workload))

        # Phase 1: Score instances, each under its own lock. Candidate tuple is
        # (id, marker, matched_blocks, inst_ref); marker None=instance, -1=legacy.
        best_score = None
        best_candidates: List[Tuple[str, Optional[int], int,
                                    Optional[_InstanceState]]] = []
        for inst_id, inst in inst_items:
            with inst.lock:
                if now - inst.freshest_load_ts() > self._load_suspend_s:
                    continue
                if block_hashes:
                    matched_blocks = inst.combined_trie.match_one(
                        inst_id, block_hashes)
                else:
                    matched_blocks = 0
                active, queued = inst.total_load()
            workload = active + queued
            score = (matched_blocks - self._load_weight * workload
                     if block_hashes else -self._load_weight * workload)
            if best_score is None or score > best_score:
                best_score = score
                best_candidates = [(inst_id, None, matched_blocks, inst)]
            elif score == best_score:
                best_candidates.append((inst_id, None, matched_blocks, inst))

        # Merge legacy candidates into the same ranking.
        for w, mb, score in legacy_scored:
            if best_score is None or score > best_score:
                best_score = score
                best_candidates = [(w, -1, mb, None)]
            elif score == best_score:
                best_candidates.append((w, -1, mb, None))

        if not best_candidates:
            return None

        # Tie-break round-robin + address resolution under the struct lock
        # (rr_counter and the address map are struct state). No instance lock is
        # held here, preserving the struct -> instance ordering.
        with self._lock:
            ns = self._namespaces.get(namespace)
            rr = ns.rr_counter if ns is not None else 0
            winner_id, marker, inst_matched, inst_ref = best_candidates[
                rr % len(best_candidates)]
            if ns is not None:
                ns.rr_counter += 1
            address = self._worker_address.get(winner_id)

        # Legacy worker path
        if marker == -1:
            return Selection(
                worker_id=winner_id,
                address=address,
                matched_blocks=legacy_matched.get(winner_id, 0))

        # Phase 2: Select rank within the chosen instance, under its own lock.
        return self._select_rank_in_instance(
            inst_ref, block_hashes, now, inst_matched, address)

    def _select_rank_in_instance(
            self, inst: _InstanceState, block_hashes: List[int],
            now: float, inst_matched: int, address: Optional[str]) -> Selection:
        """Phase 2: pick the best rank within *inst*.

        Dispatches on the configured ``rank_routing_algo``:
          * ``"none"``     -- instance-only: return the chosen instance with
            ``dp_rank=None`` so NO route_hint is injected; the worker's own
            ADP router selects the rank. (Per-rank events still drive instance
            scoring, but the host-side allgather is eliminated.)
          * ``"instance"`` -- identical scoring to phase-1 instance routing.
          * ``"adp"``      -- identical scoring to the worker
            ``KVCacheAwareADPRouter`` (shared ``score_kv_aware_candidates``).

        Both rank-picking modes gather per-rank ``(rank, matched_blocks, load)``
        candidates (skipping stale-load ranks) and break ties round-robin.
        """
        # Instance-only mode: defer rank selection to the worker's ADP router.
        if self._rank_routing_algo == self.RANK_ALGO_NONE:
            return Selection(
                worker_id=inst.instance_id,
                address=address,
                matched_blocks=inst_matched)

        # Gather (rank, matched_blocks, load) for fresh ranks (common to both),
        # reading this instance's rank tries/loads under its own lock.
        tpb = self._tokens_per_block
        adp = self._rank_routing_algo == self.RANK_ALGO_ADP
        candidates: List[Tuple[int, int, float]] = []
        with inst.lock:
            for rank, rs in inst.ranks.items():
                if now - rs.last_load_ts > self._load_suspend_s:
                    continue
                if block_hashes:
                    matched_blocks = rs.trie.match_one(
                        f"{inst.instance_id}:rank{rank}", block_hashes)
                else:
                    matched_blocks = 0
                load = 0.0
                if rs.load is not None:
                    if adp:
                        # Token-scale load, mirroring KVCacheAwareADPRouter
                        # (compute-weighted). Prefer reported active_tokens; fall
                        # back to request-count * block size if unavailable.
                        nat = getattr(rs.load, "num_active_tokens", 0)
                        load = float(nat) if nat else float(
                            (rs.load.num_active_requests
                             + rs.load.num_queued_requests) * tpb)
                    else:
                        load = float(rs.load.num_active_requests
                                     + rs.load.num_queued_requests)
                # ADP scores in tokens (match in tokens, load in tokens);
                # instance algo scores in blocks (match in blocks, load in
                # request count) to stay identical to phase-1.
                matched = matched_blocks * tpb if adp else matched_blocks
                candidates.append((rank, matched, load))

        if adp:
            # ADP-router algorithm (cap + gate + normalized load), token units.
            dbg = {} if self._log_route_scores else None
            best_ranks = score_kv_aware_candidates(
                candidates,
                load_weight=self._load_weight,
                fair_share_multiplier=self._fair_share_multiplier,
                match_rate_threshold=self._match_rate_threshold,
                total_units=len(block_hashes) * tpb,
                debug_out=dbg,
            )
            if dbg is not None:
                self._score_log_n += 1
                if self._score_log_n % 100 == 0:
                    logger.info(
                        f"[route_score] inst={inst.instance_id[:16]} "
                        f"req_tok={len(block_hashes)*tpb} cache_active={dbg['cache_active']} "
                        f"cap={dbg['cap']:.0f} mean_load={dbg['mean_load']:.0f} "
                        f"winners={dbg['winners']} rows={dbg['rows']}")
        else:
            # Instance-routing algorithm: score = matched - load_weight*load,
            # argmax (identical to phase-1 select_worker scoring).
            best_score = None
            best_ranks = []
            for rank, matched, load in candidates:
                score = matched - self._load_weight * load
                if best_score is None or score > best_score:
                    best_score = score
                    best_ranks = [rank]
                elif score == best_score:
                    best_ranks.append(rank)

        if not best_ranks:
            # All ranks suspended; return instance-level selection without rank.
            return Selection(
                worker_id=inst.instance_id,
                address=address,
                matched_blocks=inst_matched)

        # Round-robin tie-break (mirrors phase-1 instance tie-break). rr_counter
        # is instance state -- bump it under the instance lock.
        with inst.lock:
            chosen_rank = best_ranks[inst.rr_counter % len(best_ranks)]
            inst.rr_counter += 1
        return Selection(
            worker_id=inst.instance_id,
            address=address,
            matched_blocks=inst_matched,
            dp_rank=chosen_rank)

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
        end = len(token_ids) - 1
        t = 0
        while t < end:
            t_end = min(t + tpb, end)
            parent = hashes[-1] if hashes else None
            hashes.append(block_key_hasher(token_ids[t:t_end], parent))
            t += tpb
        return hashes

    _event_debug_logged = False

    def _apply_events_hierarchical(self, inst: _InstanceState,
                                   rs: _RankState, worker_id: str,
                                   events: List[dict]) -> None:
        """Apply events to both the rank's trie and the instance's combined trie."""
        for event_raw in events:
            data = event_raw.get("data", event_raw)
            etype = data.get("type")
            layer_group_id = event_raw.get("layer_group_id")
            if etype == "stored":
                block_hashes = [
                    b["block_hash"] for b in data.get("blocks", [])
                ]
                if not CentralizedKVCacheRouter._event_debug_logged and block_hashes:
                    CentralizedKVCacheRouter._event_debug_logged = True
                    hash_algo = event_raw.get(
                        "hash_algo", data.get("hash_algo"))
                    logger.info(
                        f"KVCacheRouter EVENT_DEBUG: first stored event "
                        f"worker={worker_id} "
                        f"hash_algo={hash_algo} "
                        f"hashes[:5]={block_hashes[:5]} "
                        f"types={[type(h).__name__ for h in block_hashes[:3]]} "
                        f"raw_blocks={data.get('blocks', [])[:2]}")
                # Add to rank trie (skipped in "none" mode -- never queried).
                # Always track owned_hashes (cheap) for combined-trie cleanup.
                if not self._skip_rank_trie:
                    rs.trie.add(worker_id, block_hashes)
                # combined_refcount counts how many ranks of this instance hold
                # each block, so the combined trie drops a block only once the
                # LAST rank evicts it. It MUST be incremented per distinct rank
                # acquisition, NOT per stored event: a block is re-emitted once
                # per layer group (add_stored_life_cycle_event_from_block), so
                # counting events would over-count by the layer-group factor
                # while the removed path only ever decrements once per rank ->
                # the count never reaches 0 and evicted blocks stay stale in the
                # combined trie, mis-routing phase-1 to instances that no longer
                # hold the prefix. Gate on "hash newly owned by this rank".
                new_to_rank = [h for h in block_hashes
                               if h not in rs.owned_hashes]
                rs.owned_hashes.update(block_hashes)
                # Add to combined trie (instance_id as the "worker" key)
                inst.combined_trie.add(inst.instance_id, block_hashes)
                # Update combined refcount (once per rank that newly holds h)
                for h in new_to_rank:
                    inst.combined_refcount[h] = (
                        inst.combined_refcount.get(h, 0) + 1)
                # Per-rank layer group refcount
                if layer_group_id is not None:
                    for h in block_hashes:
                        rs.layer_group_refcount[h] = (
                            rs.layer_group_refcount.get(h, 0) + 1)
            elif etype == "removed":
                removed_hashes = data.get("block_hashes", [])
                # Per-rank layer group refcount handling
                if layer_group_id is not None:
                    actually_removed = []
                    for h in removed_hashes:
                        rc = rs.layer_group_refcount.get(h, 0)
                        if rc <= 1:
                            rs.layer_group_refcount.pop(h, None)
                            actually_removed.append(h)
                        else:
                            rs.layer_group_refcount[h] = rc - 1
                else:
                    actually_removed = removed_hashes
                    for h in removed_hashes:
                        rs.layer_group_refcount.pop(h, None)

                if actually_removed:
                    # Remove from rank trie (skipped in "none" mode).
                    if not self._skip_rank_trie:
                        rs.trie.remove(worker_id, actually_removed)
                    rs.owned_hashes.difference_update(actually_removed)
                    # Remove from combined trie only when no rank holds it
                    combined_removed = []
                    for h in actually_removed:
                        rc = inst.combined_refcount.get(h, 0)
                        if rc <= 1:
                            inst.combined_refcount.pop(h, None)
                            combined_removed.append(h)
                        else:
                            inst.combined_refcount[h] = rc - 1
                    if combined_removed:
                        inst.combined_trie.remove(
                            inst.instance_id, combined_removed)

    @classmethod
    def _apply_events(cls, trie: WorkerPrefixTrie, worker_id: str,
                      events: List[dict],
                      refcounts: Optional[Dict[str, Dict[int,
                                                         int]]] = None
                      ) -> None:
        for event_raw in events:
            data = event_raw.get("data", event_raw)
            etype = data.get("type")
            layer_group_id = event_raw.get("layer_group_id")
            if etype == "stored":
                block_hashes = [
                    b["block_hash"] for b in data.get("blocks", [])
                ]
                if not cls._event_debug_logged and block_hashes:
                    cls._event_debug_logged = True
                    hash_algo = event_raw.get(
                        "hash_algo", data.get("hash_algo"))
                    logger.info(
                        f"KVCacheRouter EVENT_DEBUG: first stored event "
                        f"worker={worker_id} "
                        f"hash_algo={hash_algo} "
                        f"hashes[:5]={block_hashes[:5]} "
                        f"types={[type(h).__name__ for h in block_hashes[:3]]} "
                        f"raw_blocks={data.get('blocks', [])[:2]}")
                trie.add(worker_id, block_hashes)
                if refcounts is not None and layer_group_id is not None:
                    worker_rc = refcounts.setdefault(worker_id, {})
                    for h in block_hashes:
                        worker_rc[h] = worker_rc.get(h, 0) + 1
            elif etype == "removed":
                removed_hashes = data.get("block_hashes", [])
                if refcounts is not None and layer_group_id is not None:
                    worker_rc = refcounts.get(worker_id)
                    actually_removed = []
                    for h in removed_hashes:
                        if worker_rc is None:
                            actually_removed.append(h)
                            continue
                        rc = worker_rc.get(h, 0)
                        if rc <= 1:
                            worker_rc.pop(h, None)
                            actually_removed.append(h)
                        else:
                            worker_rc[h] = rc - 1
                    if actually_removed:
                        trie.remove(worker_id, actually_removed)
                else:
                    trie.remove(worker_id, removed_hashes)
                    if refcounts is not None:
                        worker_rc = refcounts.get(worker_id)
                        if worker_rc:
                            for h in removed_hashes:
                                worker_rc.pop(h, None)

    def _remove_rank_from_combined(self, inst: _InstanceState,
                                   rank: int) -> None:
        """Remove a rank's contribution from the instance combined trie."""
        rs = inst.ranks.get(rank)
        if rs is None:
            return
        # All hashes this rank owns (works whether or not the per-rank trie is
        # maintained -- owned_hashes is always kept).
        rank_hashes = rs.owned_hashes
        if not rank_hashes:
            return
        combined_removed = []
        for h in rank_hashes:
            rc = inst.combined_refcount.get(h, 0)
            if rc <= 1:
                inst.combined_refcount.pop(h, None)
                combined_removed.append(h)
            else:
                inst.combined_refcount[h] = rc - 1
        if combined_removed:
            inst.combined_trie.remove(inst.instance_id, combined_removed)

    def _forget_rank_locked(self, ns: _NamespaceState,
                            instance_id: str, rank: int) -> None:
        """Remove a single rank from an instance. Caller holds the struct lock;
        this also takes the instance lock (ordering struct -> instance)."""
        inst = ns.instances.get(instance_id)
        if inst is None:
            return
        with inst.lock:
            self._remove_rank_from_combined(inst, rank)
            inst.ranks.pop(rank, None)
            empty = not inst.ranks
        if empty:
            del ns.instances[instance_id]

    def _forget_instance_locked(self, ns: _NamespaceState,
                                instance_id: str) -> None:
        """Remove an entire instance (all ranks)."""
        ns.instances.pop(instance_id, None)

    @staticmethod
    def _forget_legacy_locked(ns: _NamespaceState, worker_id: str) -> None:
        ns.trie.remove_worker(worker_id)
        ns.loads.pop(worker_id, None)
        ns.last_event_seq.pop(worker_id, None)
        ns.last_load_seq.pop(worker_id, None)
        ns.last_load_ts.pop(worker_id, None)
        ns.last_seen_ts.pop(worker_id, None)
        ns.layer_group_refcount.pop(worker_id, None)

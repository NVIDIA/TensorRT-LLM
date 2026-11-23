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
"""Opt-in evict-first drop of idle child chains on a context (prefill) worker.

A child chain is the committed KV of the requests that carry a parent-session id
(``scheduling_params.subagent_affinity_id``), keyed by their own conversation id. Once such a chain
has been idle for ``TLLM_KV_SUBAGENT_IDLE_DROP_S`` seconds, its full-attention pages past the
child's first-request match are marked evict-first, so LRU eviction takes them before the prefixes
of returning main conversations. Nothing is freed at that point: a marked page is dropped only when
the normal eviction picks it, and a later match or commit of the chain clears the mark.

Ownership is decided in C++ (``ChainDemoteHandle.demote()``): only blocks whose whole subtree is
this chain's planned blocks, and that nothing holds, are marked. Dropping a page only shortens a
later reuse match (recompute), it never yields wrong KV.

Under pipeline parallelism rank 0 decides, once every ``DECIDE_EVERY_PASSES`` schedule passes, and
every stage applies the same key list at the same pass (it rides the schedule message). The idle
length is measured in seconds on rank 0's monotonic clock and must also span ``MIN_IDLE_PASSES``
passes; followers never read a clock for decisions. Chains stay in commit order on every stage (the
busy touch on rank 0 only stamps times), so the tracking caps release the same chain everywhere.

A tracked chain plans each turn past its reuse depth, since its handle already holds the blocks
before it. A chain that is not tracked (new, or demoted, capped or cleared before it returned)
plans from its anchor, so the blocks a returning chain rescued are tracked again and its next idle
period demotes the whole chain.

Enabled by ``TLLM_KV_SUBAGENT_IDLE_DROP_S=<T>`` (unset, empty or 0: off, and nothing is built).
``TLLM_KV_SUBAGENT_IDLE_DROP_MAX_REFS`` caps the page references held per rank (default 4194304).
The conversation id is never parsed: subagent and flat children are not told apart.

Parent-resume trigger (opt-in, ``TLLM_KV_SUBAGENT_PARENT_RESUME_DROP_S=<G>``; unset, empty or 0:
off). A child chain is also demoted once a request of its parent session (a request whose
conversation id equals the child's parent-session id) was seen active or queued on rank 0 after the
child's last commit, and the child has been idle for ``G`` seconds (and ``MIN_IDLE_PASSES`` passes).
The parent's next request is the signal that the parent consumed the child's result. A new commit of
the child clears the mark. With this trigger on and ``TLLM_KV_SUBAGENT_IDLE_DROP_S`` unset, the idle
trigger still runs at ``DEFAULT_PARENT_RESUME_FALLBACK_S`` seconds, so a chain whose parent never
returns is not tracked (and its references held) forever.
"""

import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple, Type

from tensorrt_llm.logger import logger

IDLE_S_ENV = "TLLM_KV_SUBAGENT_IDLE_DROP_S"
MAX_REFS_ENV = "TLLM_KV_SUBAGENT_IDLE_DROP_MAX_REFS"
PARENT_RESUME_S_ENV = "TLLM_KV_SUBAGENT_PARENT_RESUME_DROP_S"
DEFAULT_MAX_REFS = 4194304
# Idle trigger used when only the parent-resume trigger is requested.
DEFAULT_PARENT_RESUME_FALLBACK_S = 900.0
# Rank 0 decides only on schedule passes that are a multiple of this.
DECIDE_EVERY_PASSES = 16
# A chain must also have been idle for this many schedule passes.
MIN_IDLE_PASSES = 64
# Per-decision caps; a decision always takes at least one chain.
MAX_CHAINS_PER_DECISION = 8
MAX_REFS_PER_DECISION = 131072
# Tracking caps. A chain beyond MAX_CHAINS (or the reference cap) is released, never demoted.
MAX_CHAINS = 16384
MAX_ANCHORS = 65536
MAX_TOMBSTONES = 65536
LOG_EVERY_DROPS = 64
LOG_EVERY_PLANS = 4096
LOG_PREFIX = "[subagent_idle_drop]"

_STAT_NAMES = (
    "plans",
    "plan_none",
    "drops",
    "plan_blocks",
    "blocks",
    "pages",
    "gpu_pages",
    "kept_shared",
    "kept_held",
    "missing",
    "capped",
    "plan_errors",
    "drop_errors",
    "returned",
    "returned_lost_tokens",
    "absorbs",
    "anchor0",
    "parent_marks",
    "parent_drops",
)


def _parse_seconds(env: Mapping[str, str], name: str) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return 0.0
    try:
        seconds = float(raw)
    except ValueError as e:
        raise ValueError(
            f"{name} must be a finite, non-negative number of seconds, got {raw!r}"
        ) from e
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be a finite, non-negative number of seconds, got {raw!r}")
    return seconds


def parse_idle_s(env: Optional[Mapping[str, str]] = None) -> float:
    """Return the idle threshold in seconds; 0 means the feature is off.

    Raises:
        ValueError: the value is not a finite, non-negative number.
    """
    return _parse_seconds(os.environ if env is None else env, IDLE_S_ENV)


def parse_parent_resume_s(env: Optional[Mapping[str, str]] = None) -> float:
    """Return the parent-resume grace in seconds; 0 means the parent-resume trigger is off.

    Raises:
        ValueError: the value is not a finite, non-negative number.
    """
    return _parse_seconds(os.environ if env is None else env, PARENT_RESUME_S_ENV)


def parse_triggers(env: Optional[Mapping[str, str]] = None) -> Tuple[float, float]:
    """Return ``(idle_s, parent_resume_s)``; ``(0, 0)`` means the tracker is off.

    With only the parent-resume trigger set, the idle trigger falls back to
    ``DEFAULT_PARENT_RESUME_FALLBACK_S``.
    """
    idle_s = parse_idle_s(env)
    parent_resume_s = parse_parent_resume_s(env)
    if parent_resume_s > 0 and idle_s == 0:
        idle_s = DEFAULT_PARENT_RESUME_FALLBACK_S
    return idle_s, parent_resume_s


def subagent_idle_drop_requested(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the idle child-chain tracker is on: ``TLLM_KV_SUBAGENT_IDLE_DROP_S`` or
    ``TLLM_KV_SUBAGENT_PARENT_RESUME_DROP_S`` above 0 (``parse_triggers``).

    Raises:
        ValueError: a value is not a finite, non-negative number.
    """
    idle_s, _ = parse_triggers(env)
    return idle_s > 0


def parse_max_refs(env: Optional[Mapping[str, str]] = None) -> int:
    """Return the cap on page references held per rank.

    Raises:
        ValueError: the value is not a positive integer.
    """
    env = os.environ if env is None else env
    raw = env.get(MAX_REFS_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_REFS
    try:
        max_refs = int(raw)
    except ValueError as e:
        raise ValueError(f"{MAX_REFS_ENV} must be a positive integer, got {raw!r}") from e
    if max_refs <= 0:
        raise ValueError(f"{MAX_REFS_ENV} must be a positive integer, got {raw!r}")
    return max_refs


def chain_key(req: Any) -> Optional[str]:
    """The chain key of a request: its conversation id when it carries a parent-session id.

    ``None`` for a dummy request, a multimodal request, a request without a parent-session id, or
    one without a conversation id. The id is used as is, never parsed.
    """
    if getattr(req, "is_dummy_request", False):
        return None
    if getattr(req, "multimodal_hashes", None) is not None:
        return None
    affinity = getattr(req, "py_subagent_affinity_id", None)
    if affinity is None or not str(affinity).strip():
        return None
    params = getattr(req, "py_conversation_params", None)
    conversation_id = getattr(params, "conversation_id", None)
    if conversation_id is None:
        return None
    conversation_id = str(conversation_id).strip()
    return conversation_id or None


def chain_parent(req: Any) -> Optional[str]:
    """The parent-session id of a child request (its ``py_subagent_affinity_id``), or ``None``."""
    affinity = getattr(req, "py_subagent_affinity_id", None)
    if affinity is None:
        return None
    affinity = str(affinity).strip()
    return affinity or None


def _rss_mb() -> int:
    """This process's resident set in MiB, or -1 where ``/proc`` cannot tell."""
    try:
        with open("/proc/self/statm") as statm:
            resident_pages = int(statm.read().split()[1])
        return (resident_pages * os.sysconf("SC_PAGE_SIZE")) >> 20
    except (OSError, ValueError, IndexError):
        return -1


class ChainHandle(Protocol):
    """The part of ``ChainDemoteHandle`` the tracker uses."""

    @property
    def num_refs(self) -> int: ...

    def absorb(self, other: "ChainHandle") -> None: ...

    def demote(self) -> Tuple[int, int, int, int, int, int]: ...

    def release(self) -> None: ...


@dataclass(slots=True)
class _Chain:
    handle: ChainHandle
    num_refs: int
    # Pass and time of the last commit. Commits set the chain's place in ``_chains``, which is the
    # same on every stage.
    commit_pass: int
    commit_time: float
    # Pass and time of the last commit or busy touch (rank 0 only touches).
    last_pass: int
    last_time: float
    committed_tokens: int
    # Parent-session id, and whether a parent request was seen after the last commit.
    parent: Optional[str] = None
    parent_resumed: bool = False


class SubagentIdleDropTracker:
    """Tracks child chains by key and demotes the idle ones (see the module docstring).

    ``commit()`` and ``apply()`` run on every pipeline stage; ``decide()`` only on rank 0. A
    failed plan, absorb, demote or release is counted and logged once, never raised: it costs only
    reuse. ``plan_errors`` counts failed plans and absorbs, ``drop_errors`` failed demotes and
    releases. ``absorbs`` counts turns added to a tracked chain (a stable child id across turns),
    and ``anchor0`` chains whose first request matched no block, whose plan can then cover blocks
    the parent shares (a recompute, never wrong KV).
    """

    def __init__(
        self,
        idle_s: float,
        tokens_per_block: int,
        rank: int,
        max_refs: int = DEFAULT_MAX_REFS,
        errors: Tuple[Type[BaseException], ...] = (RuntimeError,),
        stats_fn: Optional[Callable[[], Tuple[int, int]]] = None,
        parent_resume_s: float = 0.0,
    ) -> None:
        if not (math.isfinite(idle_s) and idle_s > 0):
            raise ValueError(f"idle_s must be a positive finite number, got {idle_s}")
        if not (math.isfinite(parent_resume_s) and parent_resume_s >= 0):
            raise ValueError(
                f"parent_resume_s must be a non-negative finite number, got {parent_resume_s}"
            )
        if tokens_per_block < 1 or max_refs < 1:
            raise ValueError(
                f"invalid subagent idle-drop tracker: tokens_per_block={tokens_per_block}, "
                f"max_refs={max_refs}"
            )
        self.idle_s = float(idle_s)
        self.parent_resume_s = float(parent_resume_s)
        self.tokens_per_block = tokens_per_block
        self.rank = rank
        self.max_refs = max_refs
        self._errors = errors
        self._stats_fn = stats_fn
        # Key -> chain, in commit order (oldest first). Busy touches do not reorder it.
        self._chains: "OrderedDict[str, _Chain]" = OrderedDict()
        # Key -> first block of the child's own KV: its first request's reuse depth. Survives a
        # demotion, so a returning chain never plans into the prefix it shares with its parent.
        self._anchors: "OrderedDict[str, int]" = OrderedDict()
        # Key -> committed tokens of a demoted chain, to measure what a returning chain lost.
        self._tombstones: "OrderedDict[str, int]" = OrderedDict()
        # Parent-resume trigger (rank 0 only): parent id -> keys of its tracked children, and the
        # children whose parent was seen after their last commit, in marking order.
        self._children: Dict[str, Dict[str, None]] = {}
        self._resumed: "OrderedDict[str, None]" = OrderedDict()
        # Parent id -> decision pass at which its current busy streak was first seen.
        self._parent_seen: Dict[str, int] = {}
        self._total_refs = 0
        self._pass = 0
        self._warned: Dict[str, bool] = {}
        self.stats: Dict[str, int] = {name: 0 for name in _STAT_NAMES}
        self._logged_drops = 0
        self._logged_plans = 0

    key = staticmethod(chain_key)

    @property
    def num_chains(self) -> int:
        return len(self._chains)

    @property
    def total_refs(self) -> int:
        return self._total_refs

    @property
    def parent_resume_on(self) -> bool:
        return self.parent_resume_s > 0

    def is_parent_resumed(self, key: str) -> bool:
        chain = self._chains.get(key)
        return chain is not None and chain.parent_resumed

    def anchor(self, key: str) -> Optional[int]:
        return self._anchors.get(key)

    def is_tracked(self, key: str) -> bool:
        return key in self._chains

    def commit(self, req: Any, plan_fn: Callable[[int], Optional[ChainHandle]], now: float) -> None:
        """Record a finished context of a child chain (every rank, at context end).

        ``plan_fn(from_block)`` is ``kv_cache.plan_committed_chain_demote``.
        """
        key = chain_key(req)
        if key is None:
            return
        reused = max(0, int(req.prepopulated_prompt_len))
        prompt_end = int(req.context_current_position + req.context_remaining_length)
        reused_blocks = reused // self.tokens_per_block
        anchor = self._anchors.get(key)
        if anchor is None:
            anchor = reused_blocks
            self._anchors[key] = anchor
            if anchor == 0:
                self.stats["anchor0"] += 1
            while len(self._anchors) > MAX_ANCHORS:
                self._anchors.popitem(last=False)
        else:
            self._anchors.move_to_end(key)
        chain = self._chains.get(key)
        # A tracked chain's handle already holds its blocks up to its last commit. An untracked one
        # (new, or demoted, capped or cleared) plans from its anchor: no handle holds the blocks a
        # returning chain rescued, so they are tracked again and never counted twice.
        from_block = anchor if chain is None else max(anchor, reused_blocks)

        self.stats["plans"] += 1
        handle: Optional[ChainHandle] = None
        try:
            handle = plan_fn(from_block)
        except self._errors as e:
            self._warn_once("plan", f"plan_committed_chain_demote failed: {e!r}")
            self.stats["plan_errors"] += 1
        else:
            if handle is None:
                self.stats["plan_none"] += 1

        if handle is not None:
            if chain is None:
                chain = _Chain(
                    handle=handle,
                    num_refs=0,
                    commit_pass=self._pass,
                    commit_time=now,
                    last_pass=self._pass,
                    last_time=now,
                    committed_tokens=prompt_end,
                )
                self._chains[key] = chain
            else:
                try:
                    chain.handle.absorb(handle)
                except self._errors as e:
                    self._warn_once("absorb", f"ChainDemoteHandle.absorb failed: {e!r}")
                    self.stats["plan_errors"] += 1
                    self._release(handle)
                else:
                    self.stats["absorbs"] += 1
        if chain is not None:
            num_refs = self._num_refs(chain.handle)
            self._total_refs += num_refs - chain.num_refs
            chain.num_refs = num_refs
            chain.commit_pass = chain.last_pass = self._pass
            chain.commit_time = chain.last_time = now
            chain.committed_tokens = prompt_end
            self._chains.move_to_end(key)
            if self.parent_resume_on:
                # A new commit opens a new idle period: the parent must be seen again.
                self._unmark(key, chain)
                parent = chain_parent(req)
                if parent != chain.parent:
                    self._unlink(key, chain.parent)
                    chain.parent = parent
                    if parent is not None:
                        self._children.setdefault(parent, {})[key] = None

        tomb = self._tombstones.pop(key, None)
        if tomb is not None:
            self.stats["returned"] += 1
            self.stats["returned_lost_tokens"] += max(0, tomb - reused)

        while self._chains and (self._total_refs > self.max_refs or len(self._chains) > MAX_CHAINS):
            oldest_key, oldest = self._chains.popitem(last=False)
            self._forget(oldest_key, oldest)
            self._total_refs -= oldest.num_refs
            self._release(oldest.handle)
            self.stats["capped"] += 1
        self.log_stats()

    def decide(
        self,
        pass_idx: int,
        now: float,
        busy_fn: Callable[[], Iterable[str]],
        parent_busy_fn: Optional[Callable[[], Iterable[str]]] = None,
    ) -> List[str]:
        """Pick idle chains and demote them locally (rank 0, once per schedule pass).

        ``busy_fn`` gives the chain keys of the active and queued requests; ``parent_busy_fn``
        (parent-resume trigger only) gives the conversation ids of all active and queued requests.
        Returns the picked keys, which every other stage applies at the same pass.
        """
        self._pass = pass_idx
        if pass_idx % DECIDE_EVERY_PASSES != 0:
            return []
        if self.parent_resume_on and parent_busy_fn is not None:
            self._note_parents(pass_idx, parent_busy_fn())
        if not self._chains:
            return []
        for key in busy_fn():
            chain = self._chains.get(key)
            if chain is not None:
                # Stamp only: the chain keeps its commit-order place, which followers share.
                chain.last_pass = pass_idx
                chain.last_time = now
        picked: List[str] = []
        picked_refs = 0
        if self.parent_resume_on and parent_busy_fn is not None:
            self._mark_resumed()
            for key in self._resumed:
                chain = self._chains[key]
                if self._recent(
                    chain.last_pass, chain.last_time, pass_idx, now, self.parent_resume_s
                ):
                    continue
                if picked and (
                    len(picked) >= MAX_CHAINS_PER_DECISION
                    or picked_refs + chain.num_refs > MAX_REFS_PER_DECISION
                ):
                    break
                picked.append(key)
                picked_refs += chain.num_refs
            self.stats["parent_drops"] += len(picked)
        for key, chain in self._chains.items():
            # Commit order: every later chain was committed later, so stop at the first recent one.
            if self._recent(chain.commit_pass, chain.commit_time, pass_idx, now):
                break
            # Busy since its commit: skip it, a later chain may still be idle.
            if self._recent(chain.last_pass, chain.last_time, pass_idx, now):
                continue
            if chain.parent_resumed and key in picked:
                continue
            if picked and (
                len(picked) >= MAX_CHAINS_PER_DECISION
                or picked_refs + chain.num_refs > MAX_REFS_PER_DECISION
            ):
                break
            picked.append(key)
            picked_refs += chain.num_refs
        if picked:
            self.apply(picked)
        return picked

    def _recent(
        self,
        last_pass: int,
        last_time: float,
        pass_idx: int,
        now: float,
        idle_s: Optional[float] = None,
    ) -> bool:
        idle_s = self.idle_s if idle_s is None else idle_s
        return now - last_time < idle_s or pass_idx - last_pass < MIN_IDLE_PASSES

    def _note_parents(self, pass_idx: int, conversation_ids: Iterable[str]) -> None:
        """Record, per active or queued conversation, the decision pass it was first seen at."""
        previous = self._parent_seen
        self._parent_seen = {cid: previous.get(cid, pass_idx) for cid in conversation_ids}

    def _mark_resumed(self) -> None:
        """Mark the children whose parent's current request was first seen after their last commit.

        A parent request already active or queued when the child committed (a child running in the
        background of its parent) does not mark it; the parent's next request does.
        """
        for parent, first_pass in self._parent_seen.items():
            children = self._children.get(parent)
            if not children:
                continue
            for key in children:
                chain = self._chains.get(key)
                if chain is None or chain.parent_resumed or chain.commit_pass >= first_pass:
                    continue
                chain.parent_resumed = True
                self._resumed[key] = None
                self.stats["parent_marks"] += 1

    def _unmark(self, key: str, chain: _Chain) -> None:
        if chain.parent_resumed:
            chain.parent_resumed = False
            self._resumed.pop(key, None)

    def _unlink(self, key: str, parent: Optional[str]) -> None:
        if parent is None:
            return
        children = self._children.get(parent)
        if children is not None:
            children.pop(key, None)
            if not children:
                del self._children[parent]

    def _forget(self, key: str, chain: _Chain) -> None:
        """Drop a chain that left ``_chains`` from the parent-resume indexes."""
        self._resumed.pop(key, None)
        self._unlink(key, chain.parent)

    def apply(self, keys: Iterable[str], pass_idx: Optional[int] = None) -> None:
        """Demote the chains of ``keys`` (every rank; rank 0 through ``decide()``)."""
        if pass_idx is not None:
            self._pass = pass_idx
        for key in keys:
            chain = self._chains.pop(key, None)
            if chain is None:
                self.stats["missing"] += 1
                continue
            self._forget(key, chain)
            self._total_refs -= chain.num_refs
            try:
                result = chain.handle.demote()
            except self._errors as e:
                self._warn_once("demote", f"ChainDemoteHandle.demote failed: {e!r}")
                self.stats["drop_errors"] += 1
                self._release(chain.handle)
                continue
            plan_blocks, blocks, pages, gpu_pages, kept_shared, kept_held = result
            self.stats["drops"] += 1
            self.stats["plan_blocks"] += plan_blocks
            self.stats["blocks"] += blocks
            self.stats["pages"] += pages
            self.stats["gpu_pages"] += gpu_pages
            self.stats["kept_shared"] += kept_shared
            self.stats["kept_held"] += kept_held
            self._tombstones[key] = chain.committed_tokens
            self._tombstones.move_to_end(key)
            while len(self._tombstones) > MAX_TOMBSTONES:
                self._tombstones.popitem(last=False)
        self.log_stats()

    def clear(self) -> None:
        """Log, then release every handle without demoting.

        Call before the KV cache manager shuts down and after its reusable blocks are cleared.
        Anchors are kept: they only bound where later plans start.
        """
        self.log_stats(force=True)
        chains = list(self._chains.values())
        self._chains.clear()
        self._children.clear()
        self._resumed.clear()
        self._parent_seen.clear()
        self._tombstones.clear()
        self._total_refs = 0
        for chain in chains:
            self._release(chain.handle)

    def format_stats(self) -> str:
        """The stats line.

        ``refs`` counts live and expired weak references: an expired one still pins its page's
        allocation until its chain absorbs a turn, is demoted or is released. ``rss_mb`` is the
        process's resident set, which carries them.
        """
        s = self.stats
        evicted, rescued = -1, -1
        if self._stats_fn is not None:
            try:
                evicted, rescued = self._stats_fn()
            except self._errors:
                pass
        return (
            f"{LOG_PREFIX} rank={self.rank} pass={self._pass} chains={len(self._chains)} "
            f"refs={self._total_refs} plans={s['plans']} plan_none={s['plan_none']} "
            f"drops={s['drops']} blocks={s['blocks']} pages={s['pages']} "
            f"gpu_pages={s['gpu_pages']} kept_shared={s['kept_shared']} "
            f"kept_held={s['kept_held']} missing={s['missing']} capped={s['capped']} "
            f"errors={s['plan_errors'] + s['drop_errors']} plan_errors={s['plan_errors']} "
            f"drop_errors={s['drop_errors']} evicted_pages={evicted} rescued_pages={rescued} "
            f"returned={s['returned']} returned_lost_tokens={s['returned_lost_tokens']} "
            f"plan_blocks={s['plan_blocks']} absorbs={s['absorbs']} anchor0={s['anchor0']} "
            f"rss_mb={_rss_mb()}"
            + (
                f" parent_resume_s={self.parent_resume_s:g} parent_marks={s['parent_marks']} "
                f"parent_drops={s['parent_drops']} resumed={len(self._resumed)}"
                if self.parent_resume_on
                else ""
            )
        )

    def log_stats(self, force: bool = False) -> None:
        """Log every ``LOG_EVERY_DROPS`` drops or ``LOG_EVERY_PLANS`` plans, or when forced."""
        drops, plans = self.stats["drops"], self.stats["plans"]
        if not (
            force
            or drops - self._logged_drops >= LOG_EVERY_DROPS
            or plans - self._logged_plans >= LOG_EVERY_PLANS
        ):
            return
        self._logged_drops, self._logged_plans = drops, plans
        logger.info(self.format_stats())

    @staticmethod
    def _num_refs(handle: ChainHandle) -> int:
        return int(handle.num_refs)

    def _release(self, handle: ChainHandle) -> None:
        try:
            handle.release()
        except self._errors as e:
            self._warn_once("release", f"ChainDemoteHandle.release failed: {e!r}")
            self.stats["drop_errors"] += 1

    def _warn_once(self, what: str, message: str) -> None:
        if not self._warned.get(what):
            self._warned[what] = True
            logger.warning(f"{LOG_PREFIX} rank={self.rank} {message}")

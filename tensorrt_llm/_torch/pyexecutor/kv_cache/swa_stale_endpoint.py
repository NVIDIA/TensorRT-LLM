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
"""Opt-in drop of stale sliding-window (SWA) endpoint pages under ``all_reusable`` block reuse.

With ``swa_endpoint_rewind_tokens > 0`` every context request gives the SWA pages of its prompt
endpoint (the last ``rewind + window`` tokens) a high eviction priority, so that the next turn of
the conversation can resume there. The priority is fixed when a page is created, so the window of
every earlier turn keeps it after the conversation has moved on. Those stale windows fill the SWA
pool, and LRU then evicts the useful endpoints of idle conversations.

``SwaStaleEndpointTracker`` follows endpoints by token prefix. A committed context links to the
deepest earlier endpoint ``A`` that it continues: it shares all of ``A``'s prompt except at most
the last ``rewind`` tokens (``MAX_LINK_BACKOFF_TOKENS`` when ``A`` starts a line), so it can
resume at ``A``'s endpoint. When an endpoint is ``keep`` or more links behind a newly committed one on a fork-free
line, its window pages are dropped through the ``PlannedDropHandle`` planned when it was committed,
unless a page of that window lies in the protected but unplanned part (the rewind region) of an
endpoint that is still tracked; then the drop is deferred. A dropped page only shortens a later
reuse match (the radix tree prunes a match at its first missing window page); it never yields
wrong KV. The decisions depend only on the committed token streams and their commit order, so
every pipeline stage makes the same ones.

Enabled by ``TLLM_KV_SWA_STALE_ENDPOINT_KEEP=<N>`` (unset or 0: off, and nothing is built).
``TLLM_KV_SWA_STALE_ENDPOINT_MAX_NODES`` caps the number of tracked endpoints (default 32768).
"""

import os
import time
from collections import OrderedDict
from typing import Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Type

import numpy as np

from tensorrt_llm.logger import logger

KEEP_ENV = "TLLM_KV_SWA_STALE_ENDPOINT_KEEP"
MAX_NODES_ENV = "TLLM_KV_SWA_STALE_ENDPOINT_MAX_NODES"
DEFAULT_MAX_NODES = 32768
DEFAULT_LOG_EVERY = 1000
LOG_PREFIX = "[swa-stale-endpoint]"
# A later prompt continues the first endpoint of a line only if it diverges from it at most this
# many tokens before its end (a re-rendered generation prompt or last message). A deeper
# divergence is what two conversations that share a system prompt look like, so it does not link:
# otherwise one conversation's turns could drop another, idle conversation's only endpoint. Later
# endpoints of a line (which already continue an earlier one) accept divergence up to the rewind.
MAX_LINK_BACKOFF_TOKENS = 128
# A guard check scans at most this many tracked endpoints per probed block, then defers the drop.
_GUARD_SCAN_LIMIT = 64

_MASK64 = (1 << 64) - 1
# Odd multiplier, so it is invertible modulo 2**64.
_CHAIN_RADIX = 0x9E3779B97F4A7C15
_HASH_SEED = 0x5A11E7E4D90C3B21


def parse_keep(env: Optional[Mapping[str, str]] = None) -> int:
    """Return the configured keep count; 0 means the feature is off.

    Raises:
        ValueError: the value is not a non-negative integer.
    """
    env = os.environ if env is None else env
    raw = env.get(KEEP_ENV, "").strip()
    if not raw:
        return 0
    try:
        keep = int(raw)
    except ValueError as e:
        raise ValueError(f"{KEEP_ENV} must be a non-negative integer, got {raw!r}") from e
    if keep < 0:
        raise ValueError(f"{KEEP_ENV} must be a non-negative integer, got {raw!r}")
    if keep == 1:
        logger.warning(
            f"{LOG_PREFIX} {KEEP_ENV}=1 keeps only the newest endpoint of each line; a turn "
            "that resumes from the previous endpoint then misses its sliding window. Use 2 or "
            "more unless this is intended."
        )
    return keep


def parse_max_nodes(env: Optional[Mapping[str, str]] = None) -> int:
    """Return the cap on tracked endpoints.

    Raises:
        ValueError: the value is not a positive integer.
    """
    env = os.environ if env is None else env
    raw = env.get(MAX_NODES_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_NODES
    try:
        max_nodes = int(raw)
    except ValueError as e:
        raise ValueError(f"{MAX_NODES_ENV} must be a positive integer, got {raw!r}") from e
    if max_nodes <= 0:
        raise ValueError(f"{MAX_NODES_ENV} must be a positive integer, got {raw!r}")
    return max_nodes


def scope_seed(lora_id: Optional[int], salt: Optional[int]) -> int:
    """A deterministic 64-bit seed for a reuse scope (LoRA id, cache salt)."""
    lora = -1 if lora_id is None else int(lora_id)
    salt_value = -1 if salt is None else int(salt)
    seed = _HASH_SEED
    for value in (lora, salt_value):
        seed = ((seed ^ (value & _MASK64)) * 0xBF58476D1CE4E5B9) & _MASK64
        seed ^= seed >> 31
    return seed


class DropHandle(Protocol):
    """The part of ``PlannedDropHandle`` the tracker uses."""

    def drop(self) -> None: ...


class _PrefixHasher:
    """Per-block chain keys of a token sequence.

    Block content hash ``c_b = sum_j t[b * tpb + j] * v_j`` (mod 2**64, ``v`` random odd), chained
    as ``k_b = s * R**(b + 1) + sum_{i <= b} c_i * R**(b + 1 - i)``. Equal keys at block ``b`` mean
    equal token prefixes ``[0, (b + 1) * tpb)`` up to a hash collision, which can only mislink an
    endpoint (a cache miss, never wrong KV). The key "before block 0" is the seed ``s``.
    """

    def __init__(self, tokens_per_block: int) -> None:
        self._tpb = tokens_per_block
        rng = np.random.default_rng(_HASH_SEED)
        vec = rng.integers(0, 1 << 62, size=tokens_per_block, dtype=np.uint64)
        self._vec = (vec << np.uint64(1)) | np.uint64(1)
        self._pow = np.ones(1, dtype=np.uint64)
        self._inv_pow = np.ones(1, dtype=np.uint64)

    def _ensure(self, num_blocks: int) -> None:
        have = len(self._pow)
        if have >= num_blocks + 1:
            return
        size = max(num_blocks + 1, 2 * have)
        inv = pow(_CHAIN_RADIX, -1, 1 << 64)
        # R**i for i < size, doubling the filled prefix each step (array products wrap mod 2**64).
        pows = np.ones(size, dtype=np.uint64)
        inv_pows = np.ones(size, dtype=np.uint64)
        filled = 1
        while filled < size:
            n = min(filled, size - filled)
            step = np.uint64(pow(_CHAIN_RADIX, filled, 1 << 64))
            inv_step = np.uint64(pow(inv, filled, 1 << 64))
            pows[filled : filled + n] = pows[:n] * step
            inv_pows[filled : filled + n] = inv_pows[:n] * inv_step
            filled += n
        self._pow = pows
        self._inv_pow = inv_pows

    def block_keys(self, tokens: Sequence[int], num_blocks: int, seed: int) -> np.ndarray:
        """Chain keys of the first ``num_blocks`` full blocks, shape ``[num_blocks]``, uint64."""
        if num_blocks <= 0:
            return np.zeros(0, dtype=np.uint64)
        self._ensure(num_blocks)
        flat = np.asarray(tokens[: num_blocks * self._tpb])
        blocks = flat.astype(np.int64).astype(np.uint64).reshape(num_blocks, self._tpb)
        content = (blocks * self._vec).sum(axis=1, dtype=np.uint64)
        # k_b = R**(b+1) * (s + sum_{i<=b} c_i * R**-i), all mod 2**64.
        acc = np.cumsum(content * self._inv_pow[:num_blocks], dtype=np.uint64)
        acc += np.uint64(seed)
        return self._pow[1 : num_blocks + 1] * acc

    def tail_key(self, tokens: Sequence[int], start: int, end: int) -> int:
        """Content hash of the partial block ``tokens[start:end]`` (0 when empty)."""
        if end <= start:
            return 0
        tail = np.asarray(tokens[start:end]).astype(np.int64).astype(np.uint64)
        return int((tail * self._vec[: end - start]).sum(dtype=np.uint64))


class _EndpointNode:
    """One committed context endpoint."""

    __slots__ = (
        "uid",
        "seed",
        "num_tokens",
        "full_blocks",
        "prot_block",
        "link_block",
        "need_exact",
        "last_block",
        "plan_start_block",
        "link_key",
        "link_keys",
        "keys_base",
        "keys",
        "full_key",
        "tail_key",
        "handle",
        "parent",
        "children",
        "dropped",
    )

    def __init__(self, uid: int) -> None:
        self.uid = uid
        self.seed = 0
        self.num_tokens = 0
        self.full_blocks = 0
        self.prot_block = 0
        self.link_block = 0
        self.need_exact = False
        self.last_block = 0
        self.plan_start_block = 0
        self.link_key = 0
        # Chain keys of blocks [link_block, full_blocks), for ranking parent candidates.
        self.link_keys: Tuple[int, ...] = ()
        # Chain keys of blocks [keys_base, full_blocks): the link region, the planned window, the
        # unplanned protected region, and the look-back the guard check probes.
        self.keys_base = 0
        self.keys: np.ndarray = np.zeros(0, dtype=np.uint64)
        self.full_key: Tuple[int, int, int] = (0, 0, 0)
        self.tail_key = 0
        self.handle: Optional[DropHandle] = None
        self.parent: Optional["_EndpointNode"] = None
        self.children = 0
        self.dropped = False

    def key_at(self, block: int) -> int:
        """Chain key of full block ``block`` (``-1``: the seed). ``block`` must be stored."""
        if block < 0:
            return self.seed
        return int(self.keys[block - self.keys_base])

    @property
    def unplanned_end(self) -> int:
        """End (exclusive) of the full blocks of the protected region outside the planned window."""
        return min(self.plan_start_block, self.full_blocks)


class SwaStaleEndpointTracker:
    """Drops the SWA window of endpoints ``keep`` or more links behind on a fork-free line.

    Linking: a new endpoint ``E`` (``e`` tokens) links to the tracked endpoint ``A`` with
    ``A.e < e`` that it continues: ``E`` shares ``A``'s token prefix through ``A.e - rewind``,
    or through ``A.e - min(rewind, MAX_LINK_BACKOFF_TOKENS)`` when ``A`` has no parent (it may be
    another conversation's first turn behind a shared system prompt); block-aligned up, and
    through all of ``A`` when that falls in ``A``'s partial last block. Among several, the one with
    the longest shared prefix wins, then exact extension, then depth and recency. A prompt that
    shares less (another conversation with the same system prompt, or a deep rewrite) starts a
    new line.

    Dropping: walking up from ``E`` through parents, the first ``keep - 1`` ancestors are kept.
    Each older ancestor is dropped unless (a) its window reaches the protected region of a kept
    endpoint on the walk, or (b) a full block of its window is shared with the protected but
    unplanned region (the rewind region before the planned window) of any other tracked endpoint,
    which only the dropped plan would release; then it is deferred to a later commit on the line.
    The walk stops at an endpoint with two or more children (a fork: a regenerate or parallel
    subagents); that endpoint and everything above it are never dropped by a walk.

    An identical prompt (same tokens and length as a live endpoint) refreshes that endpoint's
    handle instead of linking. When more than ``max_nodes`` endpoints are tracked, the least
    recently touched one is dropped. Dropped endpoints are kept as tombstones (up to
    ``max_nodes``) so that a later prompt that would have resumed from one is counted as damage.
    """

    def __init__(
        self,
        keep: int,
        tokens_per_block: int,
        window_size: int,
        rewind_tokens: int,
        max_nodes: int = DEFAULT_MAX_NODES,
        log_every: int = DEFAULT_LOG_EVERY,
        name: str = "",
        errors: Tuple[Type[BaseException], ...] = (RuntimeError,),
    ) -> None:
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")
        if tokens_per_block < 1 or window_size < 1 or rewind_tokens < 0 or max_nodes < 1:
            raise ValueError(
                "invalid SWA stale-endpoint tracker geometry: "
                f"tokens_per_block={tokens_per_block}, window_size={window_size}, "
                f"rewind_tokens={rewind_tokens}, max_nodes={max_nodes}"
            )
        self.keep = keep
        self.tokens_per_block = tokens_per_block
        self.window_size = window_size
        self.rewind_tokens = rewind_tokens
        self.max_nodes = max_nodes
        # How far before an endpoint a continuation may diverge: a line's first endpoint may be
        # another conversation's first turn behind the same system prompt, so it only takes
        # small re-renders; a later endpoint takes any divergence its rewind region can serve.
        self.link_backoff = min(rewind_tokens, MAX_LINK_BACKOFF_TOKENS)
        self.line_link_backoff = rewind_tokens
        # Upper bound on unplanned-region length in blocks (prot_block to plan_start_block).
        self._unplanned_span = -(-(rewind_tokens + 1) // tokens_per_block) + 1
        self._log_every = log_every
        self._name = name
        self._errors = errors
        self._hasher = _PrefixHasher(tokens_per_block)
        self._next_uid = 0
        # Link key -> endpoints (live and tombstones) by uid; full key -> live endpoint.
        self._link_index: Dict[int, Dict[int, _EndpointNode]] = {}
        self._full_index: Dict[Tuple[int, int, int], _EndpointNode] = {}
        # Key of a live endpoint's first protected block -> endpoints by uid (guard check), and
        # key of the block before it (guard check of a partial last block).
        self._prot_index: Dict[int, Dict[int, _EndpointNode]] = {}
        self._prot_prev_index: Dict[int, Dict[int, _EndpointNode]] = {}
        self._lru: "OrderedDict[int, _EndpointNode]" = OrderedDict()
        self._tombs: "OrderedDict[int, _EndpointNode]" = OrderedDict()
        self.stats: Dict[str, float] = {}
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats = {
            "commits": 0,
            "linked": 0,
            "dropped": 0,
            "dropped_blocks": 0,
            "dropped_no_handle": 0,
            "deferred": 0,
            "guard_deferred": 0,
            "fork_stops": 0,
            "refreshed": 0,
            "cap_dropped": 0,
            "plan_none": 0,
            "plan_skipped": 0,
            "plan_error": 0,
            "drop_error": 0,
            "skipped_multimodal": 0,
            "skipped_short": 0,
            "skipped_tokens": 0,
            "parent_miss": 0,
            "parent_miss_tokens": 0,
            "resume_dropped": 0,
            "resume_dropped_tokens": 0,
            "total_ms": 0.0,
            "max_ms": 0.0,
        }

    @property
    def num_nodes(self) -> int:
        return len(self._lru)

    def note_skipped_multimodal(self) -> None:
        self.stats["skipped_multimodal"] += 1

    def link_geometry(self, num_tokens: int, backoff: int) -> Tuple[int, bool]:
        """(link block, whether a continuation must extend the endpoint exactly) for an endpoint
        of ``num_tokens`` tokens whose continuations share at least ``num_tokens - backoff``."""
        tpb = self.tokens_per_block
        full_blocks = num_tokens // tpb
        need_blocks = -(-max(num_tokens - backoff, 1) // tpb)
        return max(min(need_blocks, full_blocks) - 1, 0), need_blocks > full_blocks

    def geometry(self, num_tokens: int) -> Tuple[int, int, int, int, int]:
        """(protected start block, link block of a line's first endpoint, last block, planned
        window start block, full blocks) of an endpoint of ``num_tokens`` tokens."""
        tpb = self.tokens_per_block
        full_blocks = num_tokens // tpb
        # Same protected region as the priority callback (_swa_endpoint_priority).
        first_protected = max(0, num_tokens - self.rewind_tokens - self.window_size)
        prot_block = first_protected // tpb
        link_block, _ = self.link_geometry(num_tokens, self.link_backoff)
        last_block = (num_tokens - 1) // tpb
        # Window planned by KvCache::planCommittedBlockDrop (the getStaleRange end).
        plan_start_block = max(0, num_tokens + 1 - self.window_size) // tpb
        return prot_block, link_block, last_block, plan_start_block, full_blocks

    def commit(
        self,
        tokens: Sequence[int],
        num_tokens: int,
        seed: int,
        num_committed_tokens: int,
        plan_drop: Callable[[], Optional[DropHandle]],
        reused_tokens: Optional[int] = None,
    ) -> None:
        """Record a context request whose prompt ``tokens[:num_tokens]`` was just committed.

        Args:
            tokens: the request's beam-0 tokens (list or int32 ndarray), at least ``num_tokens``.
            num_tokens: the prompt length committed for reuse.
            seed: the reuse-scope seed (see ``scope_seed``).
            num_committed_tokens: the KV cache's committed token count; a drop is planned only
                when it equals ``num_tokens``.
            plan_drop: plans the drop of this endpoint's window (``plan_committed_block_drop``).
            reused_tokens: the prompt tokens this request reused from the cache, if known; used
                only by the damage counters.
        """
        start = time.perf_counter()
        stats = self.stats
        stats["commits"] += 1
        try:
            self._commit(tokens, num_tokens, seed, num_committed_tokens, plan_drop, reused_tokens)
        finally:
            elapsed = (time.perf_counter() - start) * 1e3
            stats["total_ms"] += elapsed
            stats["max_ms"] = max(stats["max_ms"], elapsed)
            if self._log_every > 0 and stats["commits"] % self._log_every == 0:
                self.log_stats()

    def _plan(
        self,
        num_tokens: int,
        num_committed_tokens: int,
        plan_drop: Callable[[], Optional[DropHandle]],
    ) -> Optional[DropHandle]:
        if num_committed_tokens != num_tokens:
            self.stats["plan_skipped"] += 1
            return None
        try:
            handle = plan_drop()
        except self._errors as e:
            if self.stats["plan_error"] == 0:
                logger.warning(f"{LOG_PREFIX} plan_committed_block_drop failed: {e!r}")
            self.stats["plan_error"] += 1
            return None
        if handle is None:
            self.stats["plan_none"] += 1
        return handle

    def _commit(
        self,
        tokens: Sequence[int],
        num_tokens: int,
        seed: int,
        num_committed_tokens: int,
        plan_drop: Callable[[], Optional[DropHandle]],
        reused_tokens: Optional[int],
    ) -> None:
        tpb = self.tokens_per_block
        if len(tokens) < num_tokens:
            # Not expected; never plan from a token view that does not cover the prompt.
            self.stats["skipped_tokens"] += 1
            return
        prot_block, _, last_block, plan_start_block, full_blocks = self.geometry(num_tokens)
        if full_blocks == 0:
            # Too short to key. No handle is planned: a discarded handle applies its drop.
            self.stats["skipped_short"] += 1
            return
        keys = self._hasher.block_keys(tokens, full_blocks, seed)
        tail_key = self._hasher.tail_key(tokens, full_blocks * tpb, num_tokens)
        full_key = (num_tokens, int(keys[full_blocks - 1]), tail_key)

        same = self._full_index.get(full_key)
        if same is not None:
            # The same prompt again: keep one endpoint and move it to the new plan. Dropping the
            # old handle only releases its count on pages the new plan also covers.
            handle = self._plan(num_tokens, num_committed_tokens, plan_drop)
            old, same.handle = same.handle, handle
            if old is not None:
                self._apply_drop(old)
            self._lru.move_to_end(same.uid)
            self.stats["refreshed"] += 1
            return

        parent, parent_shared, tomb, tomb_shared = self._find_parent(tokens, keys, num_tokens)
        self._count_damage(parent, parent_shared, tomb, tomb_shared, reused_tokens)

        node = _EndpointNode(self._next_uid)
        self._next_uid += 1
        node.seed = seed
        node.num_tokens = num_tokens
        node.full_blocks = full_blocks
        node.prot_block = prot_block
        # The rewind-wide backoff only for an endpoint that continues its parent through all of
        # the parent's full blocks: a link that diverged earlier may be another conversation's
        # first turn behind the same system prompt, and must not widen the next link.
        continues_line = parent is not None and parent_shared >= parent.full_blocks * tpb
        link_block, node.need_exact = self.link_geometry(
            num_tokens, self.line_link_backoff if continues_line else self.link_backoff
        )
        node.link_block = link_block
        node.last_block = last_block
        node.plan_start_block = plan_start_block
        node.link_key = int(keys[link_block])
        node.link_keys = tuple(keys[link_block:full_blocks].tolist())
        node.keys_base = max(
            0, min(prot_block, link_block, plan_start_block - self._unplanned_span) - 1
        )
        node.keys = keys[node.keys_base : full_blocks].copy()
        node.full_key = full_key
        node.tail_key = tail_key
        node.handle = self._plan(num_tokens, num_committed_tokens, plan_drop)
        if parent is not None:
            node.parent = parent
            parent.children += 1
            self._lru.move_to_end(parent.uid)
            self.stats["linked"] += 1
        self._link_index.setdefault(node.link_key, {})[node.uid] = node
        self._full_index[full_key] = node
        if prot_block < node.unplanned_end:
            self._prot_index.setdefault(node.key_at(prot_block), {})[node.uid] = node
            if prot_block >= 1:
                self._prot_prev_index.setdefault(node.key_at(prot_block - 1), {})[node.uid] = node
        self._lru[node.uid] = node

        self._drop_stale_ancestors(node)
        while len(self._lru) > self.max_nodes:
            _, oldest = self._lru.popitem(last=False)
            self.stats["cap_dropped"] += 1
            self._drop_node(oldest)

    def _find_parent(
        self, tokens: Sequence[int], keys: np.ndarray, num_tokens: int
    ) -> Tuple[Optional[_EndpointNode], int, Optional[_EndpointNode], int]:
        """The best live parent and the best tombstone, each with its shared prefix length."""
        tpb = self.tokens_per_block
        index = self._link_index
        best = tomb = None
        best_rank = tomb_rank = None
        ours = keys.tolist()
        num_blocks = len(ours)
        for block, key in enumerate(ours):
            nodes = index.get(key)
            if nodes is None:
                continue
            for cand in nodes.values():
                if cand.link_block != block or cand.num_tokens > num_tokens:
                    continue
                if cand.num_tokens == num_tokens and not cand.dropped:
                    # A live endpoint of the same length but other tokens is not an ancestor.
                    continue
                stored = cand.link_keys
                limit = min(len(stored), num_blocks - block)
                matched = 1  # The key at the link block matched.
                while matched < limit and stored[matched] == ours[block + matched]:
                    matched += 1
                shared_blocks = block + matched
                extends = shared_blocks == cand.full_blocks and (
                    self._hasher.tail_key(tokens, cand.full_blocks * tpb, cand.num_tokens)
                    == cand.tail_key
                )
                if cand.need_exact and not extends:
                    continue
                shared = cand.num_tokens if extends else shared_blocks * tpb
                rank = (shared, extends, cand.num_tokens, cand.uid)
                if cand.dropped:
                    if tomb_rank is None or rank > tomb_rank:
                        tomb, tomb_rank = cand, rank
                elif best_rank is None or rank > best_rank:
                    best, best_rank = cand, rank
        return (
            best,
            best_rank[0] if best_rank is not None else 0,
            tomb,
            tomb_rank[0] if tomb_rank is not None else 0,
        )

    def _count_damage(
        self,
        parent: Optional[_EndpointNode],
        parent_shared: int,
        tomb: Optional[_EndpointNode],
        tomb_shared: int,
        reused_tokens: Optional[int],
    ) -> None:
        """Count a prompt that could have resumed from an endpoint but reused less.

        ``resume_dropped``: the deepest endpoint it continues was dropped (by a walk or the node
        cap) and the reuse ended before that endpoint's rewind region: damage done by this
        feature. ``parent_miss``: it continues a tracked endpoint whose window is kept, yet the
        reuse ended before that endpoint's rewind region (pool pressure evicted the window, or a
        reuse cap such as the pipeline stage agreement applied).
        """
        stats = self.stats
        reused = 0 if reused_tokens is None else int(reused_tokens)
        if tomb is not None and (parent is None or tomb_shared > parent_shared):
            if reused_tokens is None or reused < tomb.num_tokens - self.rewind_tokens:
                stats["resume_dropped"] += 1
                stats["resume_dropped_tokens"] += max(0, tomb_shared - reused)
        elif parent is not None and reused_tokens is not None:
            if reused < parent.num_tokens - self.rewind_tokens:
                stats["parent_miss"] += 1
                stats["parent_miss_tokens"] += max(0, parent_shared - reused)

    def _guarded(self, node: _EndpointNode, stale: Dict[int, _EndpointNode]) -> bool:
        """Whether dropping ``node``'s plan could free a page that another tracked endpoint
        protects outside its own planned window.

        Scans the endpoints whose first protected block lies in ``node``'s planned window or
        just before it and that share ``node``'s prefix through that block; any whose unplanned
        protected region shares a full block of ``node``'s window guards it. A partial last
        block is treated as shared with any such endpoint that continues past it. Endpoints in
        ``stale`` (stale on the line being walked) do not guard.
        """
        lo = max(node.keys_base, node.plan_start_block - self._unplanned_span)
        full = node.full_blocks
        for block in range(max(lo, 0), full):
            others = self._prot_index.get(node.key_at(block))
            if not others:
                continue
            for scanned, other in enumerate(others.values()):
                if scanned >= _GUARD_SCAN_LIMIT:
                    return True
                if other is node or other.prot_block != block or other.uid in stale:
                    continue
                first = max(block, node.plan_start_block)
                end = min(other.unplanned_end, full)
                if first < end and node.key_at(first) == other.key_at(first):
                    return True
                if self._partial_shared(node, other):
                    return True
        if node.num_tokens > full * self.tokens_per_block and full >= 1:
            # Endpoints whose unplanned region starts exactly at node's partial last block.
            others = self._prot_prev_index.get(node.key_at(full - 1))
            if others:
                for scanned, other in enumerate(others.values()):
                    if scanned >= _GUARD_SCAN_LIMIT:
                        return True
                    if other is not node and other.prot_block == full and other.uid not in stale:
                        if self._partial_shared(node, other):
                            return True
        return False

    def _partial_shared(self, node: _EndpointNode, other: _EndpointNode) -> bool:
        """Conservative: ``node``'s partial last block may be ``other``'s page at that ordinal."""
        full = node.full_blocks
        if node.num_tokens <= full * self.tokens_per_block or full < 1:
            return False
        if not (other.prot_block <= full < other.unplanned_end):
            return False
        return node.key_at(full - 1) == other.key_at(full - 1)

    def _drop_stale_ancestors(self, node: _EndpointNode) -> None:
        kept_min_prot = node.prot_block
        path = []
        # Stale endpoints of this line (deferred ones included): their protection is not kept.
        stale: Dict[int, _EndpointNode] = {}
        ancestor = node.parent
        depth = 1
        while ancestor is not None:
            if ancestor.children >= 2:
                self.stats["fork_stops"] += 1
                break
            path.append(ancestor)
            if not ancestor.dropped:
                if depth < self.keep:
                    kept_min_prot = min(kept_min_prot, ancestor.prot_block)
                elif ancestor.last_block >= kept_min_prot:
                    self.stats["deferred"] += 1
                    stale[ancestor.uid] = ancestor
                elif self._guarded(ancestor, stale):
                    self.stats["guard_deferred"] += 1
                    stale[ancestor.uid] = ancestor
                else:
                    self._drop_node(ancestor)
            ancestor = ancestor.parent
            depth += 1
        # Detach the all-dropped top of the path; no later walk needs it.
        i = len(path)
        while i > 0 and path[i - 1].dropped:
            i -= 1
        if i < len(path):
            path[i].parent = None

    def _apply_drop(self, handle: DropHandle) -> bool:
        try:
            handle.drop()
        except self._errors as e:
            if self.stats["drop_error"] == 0:
                logger.warning(f"{LOG_PREFIX} PlannedDropHandle.drop failed: {e!r}")
            self.stats["drop_error"] += 1
            return False
        return True

    @staticmethod
    def _unindex(index: Dict[int, Dict[int, _EndpointNode]], key: int, node: _EndpointNode):
        nodes = index.get(key)
        if nodes is not None and nodes.pop(node.uid, None) is not None and not nodes:
            del index[key]

    def _drop_node(self, node: _EndpointNode) -> None:
        node.dropped = True
        self._lru.pop(node.uid, None)
        if self._full_index.get(node.full_key) is node:
            del self._full_index[node.full_key]
        if node.prot_block < node.unplanned_end:
            self._unindex(self._prot_index, node.key_at(node.prot_block), node)
            if node.prot_block >= 1:
                self._unindex(self._prot_prev_index, node.key_at(node.prot_block - 1), node)
        # Keep it as a tombstone in the link index for the damage counter.
        self._tombs[node.uid] = node
        while len(self._tombs) > self.max_nodes:
            _, old = self._tombs.popitem(last=False)
            self._unindex(self._link_index, old.link_key, old)
        handle, node.handle = node.handle, None
        if handle is None:
            self.stats["dropped_no_handle"] += 1
            return
        if self._apply_drop(handle):
            self.stats["dropped"] += 1
            self.stats["dropped_blocks"] += node.last_block - node.plan_start_block + 1

    def clear(self) -> None:
        """Apply every outstanding plan and forget all endpoints.

        Call before the KV cache manager shuts down (a handle keeps the manager alive and applies
        its plan when discarded) and after reusable blocks are cleared.
        """
        nodes = list(self._lru.values())
        self._lru.clear()
        self._tombs.clear()
        self._link_index.clear()
        self._full_index.clear()
        self._prot_index.clear()
        self._prot_prev_index.clear()
        for node in nodes:
            node.parent = None
            handle, node.handle = node.handle, None
            if handle is not None:
                self._apply_drop(handle)

    def log_stats(self) -> None:
        s = self.stats
        commits = max(1, int(s["commits"]))
        logger.info(
            f"{LOG_PREFIX}{self._name} commits={int(s['commits'])} nodes={self.num_nodes} "
            f"tombs={len(self._tombs)} linked={int(s['linked'])} dropped={int(s['dropped'])} "
            f"dropped_blocks={int(s['dropped_blocks'])} deferred={int(s['deferred'])} "
            f"guard_deferred={int(s['guard_deferred'])} fork_stops={int(s['fork_stops'])} "
            f"refreshed={int(s['refreshed'])} cap_dropped={int(s['cap_dropped'])} "
            f"plan_none={int(s['plan_none'])} plan_skipped={int(s['plan_skipped'])} "
            f"plan_error={int(s['plan_error'])} drop_error={int(s['drop_error'])} "
            f"no_handle={int(s['dropped_no_handle'])} skipped_mm={int(s['skipped_multimodal'])} "
            f"skipped_short={int(s['skipped_short'])} skipped_tokens={int(s['skipped_tokens'])} "
            f"parent_miss={int(s['parent_miss'])} "
            f"parent_miss_tokens={int(s['parent_miss_tokens'])} "
            f"resume_dropped={int(s['resume_dropped'])} "
            f"resume_dropped_tokens={int(s['resume_dropped_tokens'])} "
            f"mean_ms={s['total_ms'] / commits:.3f} max_ms={s['max_ms']:.3f} "
            f"total_ms={s['total_ms']:.0f}"
        )

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
"""Opt-in protection of long idle conversation chains on a context (prefill) worker.

A conversation whose latest prompt is at least ``TLLM_KV_LONGCHAIN_PROTECT_ISL`` tokens long is
protected for a while after its context ends: its committed chain is kept in the KV cache's last
(host) tier, so a return within ``TLLM_KV_LONGCHAIN_PROTECT_IDLE_S`` seconds matches the whole
prompt instead of computing it again.

The protection is a pin: a KV cache created with ``create_kv_cache`` on the chain's tokens and
never resumed. Its reuse match holds every matched full-attention page and the sliding-window pages
at the matched endpoint. A held page still moves from GPU to host in the normal LRU order, but is
never dropped from the last tier. Each protected chain gets two pins, one at the prompt's block end
``n`` and one ``ceil(rewind / tokens_per_block)`` blocks earlier, so the sliding windows of a
return that backs off up to ``rewind`` tokens stay too. Pins never allocate and are excluded from
the manager's statistics.

Lifecycle of a chain, keyed by its conversation id:

1. Context end (every rank): the key's generation counter goes up, any pins or pending entry of
   the key are closed and dropped, and a chain of at least the threshold becomes pending with a
   copy of its first ``n`` tokens.
2. Rank 0, every ``DECIDE_EVERY_PASSES`` passes: a pending chain idle for ``PIN_AFTER_S`` seconds
   is pinned (at most ``MAX_PINS_PER_DECISION``, and only while the last tier has at least
   ``MIN_HOST_AVAIL`` of its slots free or evictable in every pool group and the caps allow). A
   pinned chain idle for ``IDLE_S`` seconds is released: its pins close and the chain is marked
   evict-first (``ChainDemoteHandle.demote()``). A pending chain idle that long is dropped.
3. Every other rank applies rank 0's operations at the same pass (they ride the schedule
   message). Each operation names the key and its generation. A pin whose generation does not
   match the local pending chain is counted and skipped. A release or a drop means rank 0 no
   longer tracks the key, so it removes whatever this rank keeps for the key; a different
   generation there (a context end one rank missed) is only counted, so no pin or token copy
   outlives rank 0's entry.
4. Every rank, every pass: if any pool group of the last tier has less than ``FLOOR`` of its slots
   free or evictable, the oldest local pins are released at once, before the held pages could
   exceed what the tier can take.

A failed binding call costs only reuse: it is counted and logged once, never raised, and the chain
falls back to the normal eviction order.

Enabled by ``TLLM_KV_LONGCHAIN_PROTECT_ISL=<tokens>`` (unset, empty or 0: off, and nothing is
built). ``TLLM_KV_LONGCHAIN_PROTECT_PIN_AFTER_S`` (default 180), ``..._IDLE_S`` (400),
``..._MIN_HOST_AVAIL`` (0.20) and ``..._FLOOR`` (0.05) tune it.
"""

import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
)

from tensorrt_llm.logger import logger

ISL_ENV = "TLLM_KV_LONGCHAIN_PROTECT_ISL"
PIN_AFTER_S_ENV = "TLLM_KV_LONGCHAIN_PROTECT_PIN_AFTER_S"
IDLE_S_ENV = "TLLM_KV_LONGCHAIN_PROTECT_IDLE_S"
MIN_HOST_AVAIL_ENV = "TLLM_KV_LONGCHAIN_PROTECT_MIN_HOST_AVAIL"
FLOOR_ENV = "TLLM_KV_LONGCHAIN_PROTECT_FLOOR"
DEFAULT_PIN_AFTER_S = 180.0
DEFAULT_IDLE_S = 400.0
DEFAULT_MIN_HOST_AVAIL = 0.20
DEFAULT_FLOOR = 0.05
# Rank 0 decides only on schedule passes that are a multiple of this.
DECIDE_EVERY_PASSES = 16
# A pinned chain must also have been idle for this many schedule passes before its release.
MIN_IDLE_PASSES = 64
# Per-decision caps.
MAX_PINS_PER_DECISION = 2
MAX_RELEASES_PER_DECISION = 8
# The floor releases at most this many chains per pass.
MAX_FLOOR_RELEASES = 8
# Global caps. Over the pending-token cap the oldest pending chain is dropped (same on every rank).
MAX_PINNED_CHAINS = 128
MAX_PINNED_TOKENS = 32 << 20
MAX_PENDING_TOKENS = 64 << 20
# Generation counters kept; the least recently committed key is forgotten first (same everywhere).
MAX_GENS = 1 << 18
# Pin ids live above every request id.
PIN_ID_BASE = 1 << 63
LOG_EVERY_EVENTS = 32
LOG_EVERY_COMMITS = 4096
LOG_PREFIX = "[longchain_protect]"

# Operation kinds carried in the schedule message as (kind, key, generation).
OP_PIN = "p"
OP_RELEASE = "r"
OP_DROP = "d"
Op = Tuple[str, str, int]

_STAT_NAMES = (
    "commits",
    "giants",
    "nokey",
    "replaced",
    "pins",
    "pins_rewind",
    "short",
    "short_rewind",
    "releases",
    "rel_floor",
    "aged_out",
    "pending_dropped",
    "refused",
    "capped",
    "missing",
    "gen_mismatch",
    "plan_none",
    "demoted_blocks",
    "demoted_pages",
    "kept_shared",
    "kept_held",
    "pin_errors",
    "close_errors",
    "plan_errors",
    "demote_errors",
)


@dataclass(frozen=True, slots=True)
class LongChainProtectConfig:
    """Parsed knobs; ``isl`` is the prompt-length threshold in tokens."""

    isl: int
    pin_after_s: float = DEFAULT_PIN_AFTER_S
    idle_s: float = DEFAULT_IDLE_S
    min_host_avail: float = DEFAULT_MIN_HOST_AVAIL
    floor: float = DEFAULT_FLOOR


def _parse_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a finite number, got {raw!r}") from e
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    return value


def parse_config(env: Optional[Mapping[str, str]] = None) -> Optional[LongChainProtectConfig]:
    """Return the configuration, or None when ``TLLM_KV_LONGCHAIN_PROTECT_ISL`` is unset or 0.

    Raises:
        ValueError: a knob is malformed or the knobs contradict each other.
    """
    env = os.environ if env is None else env
    raw = env.get(ISL_ENV, "").strip()
    if not raw:
        return None
    try:
        isl = int(raw)
    except ValueError as e:
        raise ValueError(f"{ISL_ENV} must be a non-negative integer, got {raw!r}") from e
    if isl < 0:
        raise ValueError(f"{ISL_ENV} must be a non-negative integer, got {raw!r}")
    if isl == 0:
        return None
    pin_after_s = _parse_float(env, PIN_AFTER_S_ENV, DEFAULT_PIN_AFTER_S)
    idle_s = _parse_float(env, IDLE_S_ENV, DEFAULT_IDLE_S)
    min_host_avail = _parse_float(env, MIN_HOST_AVAIL_ENV, DEFAULT_MIN_HOST_AVAIL)
    floor = _parse_float(env, FLOOR_ENV, DEFAULT_FLOOR)
    if pin_after_s < 0:
        raise ValueError(f"{PIN_AFTER_S_ENV} must be non-negative, got {pin_after_s:g}")
    if idle_s <= pin_after_s:
        raise ValueError(
            f"{IDLE_S_ENV} ({idle_s:g}) must be greater than {PIN_AFTER_S_ENV} ({pin_after_s:g})"
        )
    if not 0.0 <= min_host_avail <= 1.0:
        raise ValueError(f"{MIN_HOST_AVAIL_ENV} must be in [0, 1], got {min_host_avail:g}")
    if not 0.0 <= floor <= min_host_avail:
        raise ValueError(
            f"{FLOOR_ENV} must be in [0, {MIN_HOST_AVAIL_ENV}] = [0, {min_host_avail:g}], "
            f"got {floor:g}"
        )
    return LongChainProtectConfig(
        isl=isl,
        pin_after_s=pin_after_s,
        idle_s=idle_s,
        min_host_avail=min_host_avail,
        floor=floor,
    )


def conversation_key(req: Any) -> Optional[str]:
    """The chain key of a request: its conversation id.

    ``None`` for a dummy request, a multimodal request (its reuse tokens are not its prompt
    tokens), or a request without a conversation id. The id is used as is, never parsed.
    """
    if getattr(req, "is_dummy_request", False):
        return None
    if getattr(req, "multimodal_hashes", None) is not None:
        return None
    params = getattr(req, "py_conversation_params", None)
    conversation_id = getattr(params, "conversation_id", None)
    if conversation_id is None:
        return None
    conversation_id = str(conversation_id).strip()
    return conversation_id or None


def copy_tokens(tokens: Sequence[int], n: int) -> Sequence[int]:
    """An owned copy of the first ``n`` tokens (a numpy view is copied, a list is sliced)."""
    head = tokens[:n]
    copy = getattr(head, "copy", None)
    return copy() if callable(copy) else list(head)


def _rss_mb() -> int:
    """This process's resident set in MiB, or -1 where ``/proc`` cannot tell."""
    try:
        with open("/proc/self/statm") as statm:
            resident_pages = int(statm.read().split()[1])
        return (resident_pages * os.sysconf("SC_PAGE_SIZE")) >> 20
    except (OSError, ValueError, IndexError):
        return -1


class ChainHandle(Protocol):
    """The part of ``ChainDemoteHandle`` the protector uses."""

    def demote(self) -> Tuple[int, int, int, int, int, int]: ...

    def release(self) -> None: ...


class Pin(Protocol):
    """The part of a never-resumed ``_KVCache`` the protector uses."""

    @property
    def num_committed_tokens(self) -> int: ...

    def stop_committing(self) -> None: ...

    def plan_committed_chain_demote(self, from_block: int) -> Optional[ChainHandle]: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _Pending:
    gen: int
    n: int
    tokens: Optional[Sequence[int]]
    scope: Any
    commit_time: float
    last_time: float
    last_pass: int


@dataclass(slots=True)
class _Pinned:
    gen: int
    n: int
    # (pin id, pin); the first is the pin at ``n``. Empty once the floor released the chain, or
    # when every pin failed: the entry stays until rank 0 releases it, so every rank agrees.
    pins: List[Tuple[int, Pin]]
    commit_time: float
    last_time: float
    last_pass: int


class LongChainProtector:
    """Pins long idle chains and releases them (see the module docstring).

    ``commit()``, ``apply()`` and ``floor()`` run on every pipeline stage; ``decide()`` only on
    rank 0, which applies its own operations before returning them. The binding is injected:

    - ``create_pin(scope, tokens, pin_id)`` returns a suspended KV cache matched on ``tokens``;
    - ``stats_fn()`` returns ``(available, total)`` slots per pool group of the last tier;
    - ``mark_excluded(pin_id)`` / ``clear_excluded(pin_id)`` keep a pin out of the statistics;
    - ``id_in_use(pin_id)`` tells whether a request already uses that id.
    """

    def __init__(
        self,
        config: LongChainProtectConfig,
        tokens_per_block: int,
        rewind_tokens: int,
        rank: int,
        create_pin: Callable[[Any, Sequence[int], int], Pin],
        stats_fn: Callable[[], Sequence[Tuple[int, int]]],
        mark_excluded: Callable[[int], None],
        clear_excluded: Callable[[int], None],
        id_in_use: Callable[[int], bool] = lambda pin_id: False,
        errors: Tuple[Type[BaseException], ...] = (RuntimeError,),
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if tokens_per_block < 1 or rewind_tokens < 0:
            raise ValueError(
                f"invalid long-chain protector: tokens_per_block={tokens_per_block}, "
                f"rewind_tokens={rewind_tokens}"
            )
        self.config = config
        self.tokens_per_block = tokens_per_block
        self.rewind_blocks = -(-rewind_tokens // tokens_per_block)
        self.rank = rank
        self._create_pin = create_pin
        self._stats_fn = stats_fn
        self._mark_excluded = mark_excluded
        self._clear_excluded = clear_excluded
        self._id_in_use = id_in_use
        self._errors = errors
        self._clock = clock
        # Key -> generation: the number of context ends of the key, the same on every rank.
        self._gens: "OrderedDict[str, int]" = OrderedDict()
        # Key -> pending chain, in commit order (oldest first).
        self._pending: "OrderedDict[str, _Pending]" = OrderedDict()
        # Key -> pinned chain, in pin order (oldest first).
        self._pinned: "OrderedDict[str, _Pinned]" = OrderedDict()
        self._pending_tokens = 0
        self._pinned_tokens = 0
        self._next_pin = 0
        self._pass = 0
        self._last_avail = -1.0
        self._pin_s = 0.0
        self._warned: Dict[str, bool] = {}
        self.stats: Dict[str, int] = {name: 0 for name in _STAT_NAMES}
        self._logged_events = 0
        self._logged_commits = 0

    key = staticmethod(conversation_key)

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    @property
    def pending_tokens(self) -> int:
        return self._pending_tokens

    @property
    def num_pinned(self) -> int:
        """Chains whose pins are open (a chain the floor released is not counted)."""
        return sum(1 for chain in self._pinned.values() if chain.pins)

    @property
    def pinned_tokens(self) -> int:
        return self._pinned_tokens

    @property
    def pin_ms(self) -> float:
        return self._pin_s * 1e3

    def gen(self, key: str) -> int:
        return self._gens.get(key, 0)

    def is_pending(self, key: str) -> bool:
        return key in self._pending

    def is_pinned(self, key: str) -> bool:
        chain = self._pinned.get(key)
        return chain is not None and bool(chain.pins)

    def protected_keys(self) -> Set[str]:
        """Keys of the pending chains and of the chains whose pins are open."""
        return set(self._pending) | {key for key, chain in self._pinned.items() if chain.pins}

    def pin_lengths(self, n: int) -> List[int]:
        """Token lengths of a chain's pins: ``n`` and, with a rewind span, an earlier block end."""
        lengths = [n]
        rewound = n - self.rewind_blocks * self.tokens_per_block
        if self.rewind_blocks > 0 and rewound >= self.tokens_per_block:
            lengths.append(rewound)
        return lengths

    # ---- every rank ---------------------------------------------------------------------------

    def commit(self, req: Any, kv_cache: Any, tokens: Sequence[int], now: float) -> None:
        """Record a finished context (every rank, at context end, before the SWA endpoint drop).

        ``kv_cache`` is the request's cache (``num_committed_tokens``, ``reuse_scope``) and
        ``tokens`` its beam-0 reuse tokens.
        """
        self.stats["commits"] += 1
        prompt_end = int(req.context_current_position + req.context_remaining_length)
        giant = prompt_end >= self.config.isl
        key = conversation_key(req)
        if key is None:
            if giant:
                self.stats["nokey"] += 1
            self.log_stats()
            return
        gen = self._gens.pop(key, 0) + 1
        self._gens[key] = gen
        while len(self._gens) > MAX_GENS:
            self._gens.popitem(last=False)
        # The request now holds the live chain: the older pins and copy are no longer needed.
        chain = self._pinned.pop(key, None)
        if chain is not None:
            self.stats["replaced"] += 1
            self._close_pins(chain)
        self._drop_pending(key)
        if giant:
            self.stats["giants"] += 1
            n = (
                int(kv_cache.num_committed_tokens) // self.tokens_per_block
            ) * self.tokens_per_block
            if n >= self.tokens_per_block:
                self._pending[key] = _Pending(
                    gen=gen,
                    n=n,
                    tokens=copy_tokens(tokens, n),
                    scope=kv_cache.reuse_scope,
                    commit_time=now,
                    last_time=now,
                    last_pass=self._pass,
                )
                self._pending_tokens += n
                while self._pending_tokens > MAX_PENDING_TOKENS:
                    self._drop_pending(next(iter(self._pending)))
                    self.stats["pending_dropped"] += 1
            else:
                self.stats["short"] += 1
        self.log_stats()

    def apply(self, ops: Iterable[Op], pass_idx: Optional[int] = None) -> None:
        """Apply rank 0's operations of this pass (every rank; rank 0 through ``decide()``)."""
        if pass_idx is not None:
            self._pass = pass_idx
        for kind, key, gen in ops:
            if kind == OP_PIN:
                entry = self._pending.get(key)
                if entry is None:
                    self.stats["missing"] += 1
                elif entry.gen != gen:
                    self.stats["gen_mismatch"] += 1
                else:
                    self._pin(key, entry)
            elif kind in (OP_RELEASE, OP_DROP):
                self._forget(kind, key, gen)
            else:
                raise ValueError(f"unknown long-chain protection operation {kind!r}")
        self.log_stats()

    def floor(self) -> int:
        """Release the oldest local pins while the last tier is below the floor (every rank).

        Unilateral: the stage agreement caps any depth difference to reuse. The chain stays listed
        (without pins) until rank 0 releases it. Returns the number of chains released.
        """
        if self._pinned_tokens == 0:
            return 0
        released = 0
        while released < MAX_FLOOR_RELEASES:
            avail = self._min_avail()
            if avail is None or avail >= self.config.floor:
                break
            oldest = min(
                (chain for chain in self._pinned.values() if chain.pins),
                key=lambda chain: chain.commit_time,
                default=None,
            )
            if oldest is None:
                break
            self._release(oldest)
            self.stats["rel_floor"] += 1
            released += 1
        if released:
            self.log_stats(force=True)
        return released

    def clear(self) -> None:
        """Log, then close every pin without demoting and drop every pending copy.

        Call before the KV cache manager clears its reusable blocks or shuts down: both refuse
        while a KV cache is open. Generation counters are kept.
        """
        self.log_stats(force=True)
        chains = list(self._pinned.values())
        self._pinned.clear()
        for chain in chains:
            self._close_pins(chain)
        self._pinned_tokens = 0
        for entry in self._pending.values():
            entry.tokens = None
        self._pending.clear()
        self._pending_tokens = 0

    # ---- rank 0 -------------------------------------------------------------------------------

    def decide(self, pass_idx: int, now: float, busy_fn: Callable[[], Iterable[str]]) -> List[Op]:
        """Pick this pass's operations and apply them locally (rank 0, every schedule pass).

        Returns the operations, which every other stage applies at the same pass.
        """
        self._pass = pass_idx
        if pass_idx % DECIDE_EVERY_PASSES != 0 or not (self._pending or self._pinned):
            return []
        busy = set(busy_fn())
        for key in busy:
            # Stamp only: the order stays the commit order, which every stage shares.
            for entry in (self._pending.get(key), self._pinned.get(key)):
                if entry is not None:
                    entry.last_time = now
                    entry.last_pass = pass_idx
        ops: List[Op] = []
        idle_s = self.config.idle_s
        num_releases = 0
        for key, chain in self._pinned.items():
            if num_releases >= MAX_RELEASES_PER_DECISION:
                break
            if now - chain.last_time > idle_s and pass_idx - chain.last_pass >= MIN_IDLE_PASSES:
                ops.append((OP_RELEASE, key, chain.gen))
                num_releases += 1
        num_pins = 0
        admitted: Optional[bool] = None
        live = self.num_pinned
        pinned_tokens = self._pinned_tokens
        for key, entry in self._pending.items():
            age = now - entry.last_time
            if age > idle_s:
                ops.append((OP_DROP, key, entry.gen))
                continue
            if age < self.config.pin_after_s or key in busy or num_pins >= MAX_PINS_PER_DECISION:
                continue
            if admitted is None:
                # One storage read per decision, only when a chain is due.
                avail = self._min_avail()
                admitted = avail is not None and avail >= self.config.min_host_avail
                if not admitted:
                    self.stats["refused"] += 1
            if not admitted:
                continue
            if live >= MAX_PINNED_CHAINS or pinned_tokens + entry.n > MAX_PINNED_TOKENS:
                self.stats["capped"] += 1
                admitted = False
                continue
            ops.append((OP_PIN, key, entry.gen))
            num_pins += 1
            live += 1
            pinned_tokens += entry.n
        if ops:
            self.apply(ops)
        return ops

    # ---- internals ----------------------------------------------------------------------------

    def _min_avail(self) -> Optional[float]:
        """The smallest free-or-evictable fraction over the last tier's pool groups."""
        try:
            stats = self._stats_fn()
        except self._errors as e:
            self._warn_once("stats", f"storage statistics failed: {e!r}")
            return None
        fractions = [available / total for available, total in stats if total > 0]
        if not fractions:
            return None
        self._last_avail = min(fractions)
        return self._last_avail

    def _forget(self, kind: str, key: str, gen: int) -> None:
        """Apply a release or a drop: rank 0 no longer tracks ``key``.

        Whatever this rank keeps for the key goes, pinned or pending (a key has at most one of
        the two). The stages commit the same context ends, so the entry normally has rank 0's
        generation and the kind rank 0 expects. Anything else means a stage missed a context
        end: it is counted as ``gen_mismatch``, and the stale entry is still removed, since
        nothing would release it later.
        """
        chain = self._pinned.pop(key, None)
        entry = self._pending.get(key)
        if chain is None and entry is None:
            self.stats["missing"] += 1
            return
        local_gen = chain.gen if chain is not None else entry.gen
        expected = chain if kind == OP_RELEASE else entry
        if expected is None or local_gen != gen:
            self.stats["gen_mismatch"] += 1
        if chain is not None:
            self.stats["releases"] += 1
            self._release(chain)
        if entry is not None:
            self._drop_pending(key)
            self.stats["aged_out"] += 1

    def _drop_pending(self, key: str) -> None:
        entry = self._pending.pop(key, None)
        if entry is not None:
            self._pending_tokens -= entry.n
            entry.tokens = None

    def _new_pin_id(self) -> int:
        while True:
            pin_id = PIN_ID_BASE + self._next_pin
            self._next_pin += 1
            if not self._id_in_use(pin_id):
                return pin_id

    def _pin(self, key: str, entry: _Pending) -> None:
        start = self._clock()
        tokens = entry.tokens
        self._drop_pending(key)
        pins: List[Tuple[int, Pin]] = []
        for i, length in enumerate(self.pin_lengths(entry.n)):
            pin_id = self._new_pin_id()
            pin = None
            try:
                # Before the create: the reuse match must not count as reuse or feed the tuner.
                self._mark_excluded(pin_id)
                pin = self._create_pin(entry.scope, tokens[:length], pin_id)
                pin.stop_committing()
                matched = int(pin.num_committed_tokens)
            except self._errors as e:
                self._warn_once("pin", f"pin creation failed: {e!r}")
                self.stats["pin_errors"] += 1
                if pin is not None:
                    self._close_one(pin_id, pin)
                else:
                    self._clear_one(pin_id)
                continue
            pins.append((pin_id, pin))
            if i == 0:
                self.stats["pins"] += 1
                if matched < entry.n - self.tokens_per_block:
                    self.stats["short"] += 1
            else:
                self.stats["pins_rewind"] += 1
                if matched < length - self.tokens_per_block:
                    self.stats["short_rewind"] += 1
        self._pinned[key] = _Pinned(
            gen=entry.gen,
            n=entry.n,
            pins=pins,
            commit_time=entry.commit_time,
            last_time=entry.last_time,
            last_pass=entry.last_pass,
        )
        if pins:
            self._pinned_tokens += entry.n
        self._pin_s += self._clock() - start

    def _release(self, chain: _Pinned) -> None:
        """Close the chain's pins and mark the chain evict-first."""
        if not chain.pins:
            return
        handle = None
        try:
            handle = chain.pins[0][1].plan_committed_chain_demote(0)
        except self._errors as e:
            self._warn_once("plan", f"plan_committed_chain_demote failed: {e!r}")
            self.stats["plan_errors"] += 1
        else:
            if handle is None:
                self.stats["plan_none"] += 1
        # Close first: a block a pin still holds is kept, not demoted.
        self._close_pins(chain)
        if handle is None:
            return
        try:
            result = handle.demote()
        except self._errors as e:
            self._warn_once("demote", f"ChainDemoteHandle.demote failed: {e!r}")
            self.stats["demote_errors"] += 1
            try:
                handle.release()
            except self._errors:
                pass
            return
        _, blocks, pages, _, kept_shared, kept_held = result
        self.stats["demoted_blocks"] += blocks
        self.stats["demoted_pages"] += pages
        self.stats["kept_shared"] += kept_shared
        self.stats["kept_held"] += kept_held

    def _close_pins(self, chain: _Pinned) -> None:
        if chain.pins:
            self._pinned_tokens -= chain.n
        pins, chain.pins = chain.pins, []
        for pin_id, pin in pins:
            self._close_one(pin_id, pin)

    def _close_one(self, pin_id: int, pin: Pin) -> None:
        try:
            pin.close()
        except self._errors as e:
            self._warn_once("close", f"pin close failed: {e!r}")
            self.stats["close_errors"] += 1
        self._clear_one(pin_id)

    def _clear_one(self, pin_id: int) -> None:
        try:
            self._clear_excluded(pin_id)
        except self._errors as e:
            self._warn_once("clear", f"clear_stats_excluded failed: {e!r}")
            self.stats["close_errors"] += 1

    # ---- logging ------------------------------------------------------------------------------

    def format_stats(self) -> str:
        """The stats line. ``host_avail`` is the last minimum free-or-evictable fraction read."""
        s = self.stats
        errors = s["pin_errors"] + s["close_errors"] + s["plan_errors"] + s["demote_errors"]
        return (
            f"{LOG_PREFIX} rank={self.rank} pass={self._pass} pending={len(self._pending)} "
            f"pending_tokens={self._pending_tokens} pinned={self.num_pinned} "
            f"pinned_tokens={self._pinned_tokens} commits={s['commits']} giants={s['giants']} "
            f"nokey={s['nokey']} replaced={s['replaced']} pins={s['pins']} "
            f"pins_rewind={s['pins_rewind']} short={s['short']} "
            f"short_rewind={s['short_rewind']} releases={s['releases']} "
            f"rel_floor={s['rel_floor']} aged_out={s['aged_out']} "
            f"pending_dropped={s['pending_dropped']} refused={s['refused']} "
            f"capped={s['capped']} missing={s['missing']} gen_mismatch={s['gen_mismatch']} "
            f"plan_none={s['plan_none']} demoted_blocks={s['demoted_blocks']} "
            f"demoted_pages={s['demoted_pages']} kept_shared={s['kept_shared']} "
            f"kept_held={s['kept_held']} errors={errors} pin_errors={s['pin_errors']} "
            f"close_errors={s['close_errors']} plan_errors={s['plan_errors']} "
            f"demote_errors={s['demote_errors']} pin_ms={self.pin_ms:.1f} "
            f"host_avail={self._last_avail:.4f} rss_mb={_rss_mb()}"
        )

    def log_stats(self, force: bool = False) -> None:
        """Log every ``LOG_EVERY_EVENTS`` pins and releases or ``LOG_EVERY_COMMITS`` commits."""
        s = self.stats
        events = s["pins"] + s["releases"] + s["rel_floor"] + s["aged_out"]
        commits = s["commits"]
        if not (
            force
            or events - self._logged_events >= LOG_EVERY_EVENTS
            or commits - self._logged_commits >= LOG_EVERY_COMMITS
        ):
            return
        self._logged_events, self._logged_commits = events, commits
        logger.info(self.format_stats())

    def _warn_once(self, what: str, message: str) -> None:
        if not self._warned.get(what):
            self._warned[what] = True
            logger.warning(f"{LOG_PREFIX} rank={self.rank} {message}")

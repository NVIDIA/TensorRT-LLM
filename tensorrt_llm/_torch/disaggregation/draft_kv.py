# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash drafter's context window under disaggregated serving, on the unified KV cache.

A DFlash drafter cross-attends the prompt's drafter K/V (its "context window"). The context
worker writes it during prefill; the generation worker never prefills, so it drafts from the
prompt only if that K/V reaches it with the target's KV.

Port of the MiniMax-M3.1 disagg draft-window transfer (``disaggregation/draft_kv.py``
and its default-on switch) onto this branch's unified
target + draft KV cache (``0f6cff3c0f``):

* M3.1 keeps the window in a separate draft KV manager and ships it as a second block set,
  with its own readiness fence, a ``draft_ctx_state`` carrier and a TP agreement step.
* Here the draft layers are layers of the target's own ``KVCacheManagerV2`` (one block table,
  ``reuse_addressable=True``). The transceiver moves them with the target's layers in the same
  session and fence (``native/peer.py`` validates the draft layer count on both sides), and the
  pages are addressed by absolute token position. The generation worker only has to number the
  drafter's context from the prompt end: ``ctx_len = prompt_len`` on its first step
  (``gen_window_seed``). No second block set, fence or state carrier is needed.
* What this module adds is the contract and the evidence: strict switches, the seed rule, and
  per-side counters whose periodic log lines show, per run, that every generation request
  drafted from a shipped window (the "drafter gets its full context" check).

Default on (as for MiniMax-M3.1): ``llm_args.disagg_shares_draft_kv`` shares the draft KV
for a DFlash drafter on the AUTO / TRTLLM backend when ``TRTLLM_DISAGG_SHARED_DRAFT_KV`` is
unset; ``TRTLLM_DISAGG_SHARED_DRAFT_KV=0`` turns the window off (the generation worker then
drafts from an empty window and ``DFlash acceptance is degraded`` is logged).

This module imports neither torch nor the bindings, so its contract is testable alone.
"""

import os
import threading
import time
from collections import Counter
from typing import Callable, List, Mapping, Optional, Tuple

SHARED_DRAFT_KV_ENV = "TRTLLM_DISAGG_SHARED_DRAFT_KV"
# Seconds between the counters' summary lines (0 logs every event).
DRAFT_WINDOW_LOG_INTERVAL_ENV = "TRTLLM_DISAGG_DRAFT_WINDOW_LOG_INTERVAL_S"

_TRUE_VALUES = frozenset({"1", "true", "on", "yes"})
_FALSE_VALUES = frozenset({"", "0", "false", "off", "no"})


def parse_strict_bool_env(name: str, environ: Optional[Mapping[str, str]] = None) -> bool:
    """Read a boolean environment variable under the strict value contract.

    ``1`` / ``true`` / ``on`` / ``yes`` (any case) are True; unset, empty, ``0`` / ``false`` /
    ``off`` / ``no`` are False; anything else raises ``ValueError`` naming the variable, so a typo
    never silently keeps (or drops) a feature.
    """
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of 1/true/on/yes or 0/false/off/no (or empty), got {raw!r}"
    )


# ---------------------------------------------------------------------------------------------
# Window status and the generation-side seed
# ---------------------------------------------------------------------------------------------

# How a generation request's drafter window was seeded.
SEED_WINDOW = "window"  # unified cache: drafter context numbered to the prompt end
SEED_EMPTY = "empty"  # private / unpaired draft storage: the drafter starts from nothing
SEED_DUMMY = "dummy_slot"  # no free drafter slot: drafts from the shared dummy slot

# Why a context worker's drafter did not write the whole prompt window. The request itself never
# fails (spec decode is lossless); the generation worker then drafts from stale or partial K/V.
FALLBACK_NO_FREE_SLOT = "no_free_slot"
FALLBACK_CTX_OVERFLOW = "ctx_overflow"


def gen_window_seed(unified_cache: bool, position: int, max_ctx: int) -> Tuple[int, str]:
    """``(ctx_len, seed)`` for a generation request that reaches the drafter without a prefill.

    ``position`` is the position of the request's first generation-step token, i.e. the prompt
    length (plus any tokens already generated, e.g. after a re-admission). On the unified cache
    the drafter's K/V for every earlier position lives in the request's own pages, transferred
    with the target's, so the drafter context is ``[0, position)`` (clamped to the arena's
    logical capacity). Anywhere else the drafter context is worker-local and starts empty.
    """
    if position < 0:
        raise ValueError(f"position must be >= 0, got {position}")
    if not unified_cache:
        return 0, SEED_EMPTY
    return min(int(position), int(max_ctx)), SEED_WINDOW


def draft_window_mode(
    disagg: bool, unified_cache: bool, reuse_addressable: bool
) -> Optional[str]:
    """One word for the startup log: how a disaggregated worker's drafter context travels.

    None for an aggregated worker. ``shipped`` when the draft layers live in the unified
    manager (they move with the target's KV); ``worker_local`` otherwise (a generation worker
    drafts from an empty window).
    """
    if not disagg:
        return None
    if unified_cache and reuse_addressable:
        return "shipped"
    return "worker_local"


# ---------------------------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------------------------


def _default_log_interval(environ: Optional[Mapping[str, str]] = None) -> float:
    env = os.environ if environ is None else environ
    raw = env.get(DRAFT_WINDOW_LOG_INTERVAL_ENV)
    if raw is None or raw.strip() == "":
        return 60.0
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{DRAFT_WINDOW_LOG_INTERVAL_ENV} must be a number, got {raw!r}")
    if value < 0:
        raise ValueError(f"{DRAFT_WINDOW_LOG_INTERVAL_ENV} must be >= 0, got {raw!r}")
    return value


def _trtllm_info(message: str) -> None:
    from tensorrt_llm.logger import logger

    logger.info(message)


def _trtllm_warning(message: str) -> None:
    from tensorrt_llm.logger import logger

    logger.warning(message)


class DraftWindowCounters:
    """Per-worker evidence of the drafter window: seeds by kind and context-side fallbacks.

    The summary line (``DFlash draft window (<role>): ...``) is logged at info on the first event
    and then at most once per ``interval_s``; the first ``empty`` / ``dummy_slot`` seed and the
    first fallback of each reason are also logged as a warning, so a run that drafted any request
    without its prompt window says so in its log.
    """

    PREFIX = "DFlash draft window"

    def __init__(
        self,
        role: str = "worker",
        interval_s: Optional[float] = None,
        info: Optional[Callable[[str], None]] = None,
        warning: Optional[Callable[[str], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.role = role
        self.interval_s = _default_log_interval() if interval_s is None else float(interval_s)
        self._info = info or _trtllm_info
        self._warning = warning or _trtllm_warning
        self._clock = clock
        self.seeds: Counter = Counter()
        self.seeded_tokens = 0
        self.fallback: Counter = Counter()
        self._last_log: Optional[float] = None
        self._lock = threading.Lock()

    def record_seed(self, seed: str, ctx_len: int = 0) -> None:
        with self._lock:
            first = self.seeds[seed] == 0
            self.seeds[seed] += 1
            if seed == SEED_WINDOW:
                self.seeded_tokens += int(ctx_len)
        if first and seed != SEED_WINDOW:
            self._warning(
                f"{self.PREFIX} ({self.role}): a generation request drafts without its prompt "
                f"window (seed={seed}); acceptance length drops for such requests."
            )
        self._maybe_log()

    def record_fallback(self, reason: str, detail: str = "") -> None:
        with self._lock:
            first = self.fallback[reason] == 0
            self.fallback[reason] += 1
        if first:
            self._warning(
                f"{self.PREFIX} ({self.role}): the drafter did not store the whole prompt "
                f"window: reason={reason} {detail}".rstrip()
            )
        self._maybe_log()

    def summary(self) -> str:
        with self._lock:
            seeds = ",".join(f"{k}={v}" for k, v in sorted(self.seeds.items()))
            reasons = ",".join(f"{k}={v}" for k, v in sorted(self.fallback.items()))
            return (
                f"{self.PREFIX} ({self.role}): seeded={{{seeds}}} "
                f"seeded_tokens={self.seeded_tokens} "
                f"ctx_store_fallback={sum(self.fallback.values())} "
                f"ctx_store_fallback_by_reason={{{reasons}}}"
            )

    def _maybe_log(self) -> None:
        now = self._clock()
        if self._last_log is not None and now - self._last_log < self.interval_s:
            return
        self._last_log = now
        self._info(self.summary())


# ---------------------------------------------------------------------------------------------
# Context-side state (kept from the M3.1 contract for a future carrier; unused on the unified
# cache, where the generation side derives the window from the request's position)
# ---------------------------------------------------------------------------------------------


def encode_draft_ctx_state(ctx_len: int, ctx_base: int, valid: bool) -> List[int]:
    """The context side's drafter state as ``[ctx_len, ctx_base, valid]``."""
    return [int(ctx_len), int(ctx_base), 1 if valid else 0]


def decode_draft_ctx_state(state) -> Optional[Tuple[int, int, bool]]:
    """``(ctx_len, ctx_base, valid)`` from ``encode_draft_ctx_state``, or None for anything else."""
    if state is None or isinstance(state, (bytes, bytearray, str)):
        return None
    try:
        ctx_len, ctx_base, valid = (int(x) for x in state)
    except (TypeError, ValueError):
        return None
    if ctx_len < 0 or ctx_base < 0 or valid not in (0, 1):
        return None
    return ctx_len, ctx_base, bool(valid)

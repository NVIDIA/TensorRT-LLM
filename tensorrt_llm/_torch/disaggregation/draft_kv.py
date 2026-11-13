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
"""Whether the disagg transceiver ships the standalone drafter's context window.

A DFlash / DSpark drafter with a separate draft KV pool keeps the prompt's drafter K/V in that pool
(the context window its block decode attends to). Without this transfer a generation worker starts
every request from an empty window. The window travels as a second block set in the same session
and write as the target's KV.

The draft set ships only when all of these hold:

* the target model opts in through ``RuntimeHooks.disagg_draft_kv`` (off for every model by default);
* a separate draft KV cache manager exists (both sides: the peers refuse each other otherwise);
* ``TRTLLM_DISABLE_DISAGG_DRAFT_KV`` is unset or false.

This module reads no model or manager state at import, so the environment contract is testable alone.
"""

import os
import threading
import time
from collections import Counter
from typing import List, Mapping, Optional, Tuple

DISABLE_DISAGG_DRAFT_KV_ENV = "TRTLLM_DISABLE_DISAGG_DRAFT_KV"

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


def disagg_draft_kv_disabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether ``TRTLLM_DISABLE_DISAGG_DRAFT_KV`` turns the draft-window transfer off."""
    return parse_strict_bool_env(DISABLE_DISAGG_DRAFT_KV_ENV, environ)


def select_disagg_draft_kv_cache_manager(model: object, draft_kv_cache_manager: Optional[object]):
    """The draft KV cache manager the transceiver should ship, or None for the target-only path.

    ``model`` is the engine's model, whose ``runtime_hooks`` carry the opt-in. A model that opts in
    must run a drafter whose context lives in the managed pool, unpaired with the target's block
    reuse; anything else is refused here rather than mid-transfer.
    """
    if draft_kv_cache_manager is None:
        return None
    # Imported lazily: the hook module is light, but this module stays importable on its own.
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import runtime_hooks_of

    # getattr: an engine whose RuntimeHooks lacks the field never opts in.
    if not getattr(runtime_hooks_of(model), "disagg_draft_kv", False):
        return None
    if disagg_draft_kv_disabled():
        return None
    worker = getattr(model, "spec_worker", None)
    if worker is None or not hasattr(worker, "engage_disagg_draft_kv"):
        # A separate draft pool behind a worker without a paged drafter context (e.g. a one-model
        # Eagle3 layer): there is no context window to ship.
        return None
    if getattr(draft_kv_cache_manager, "enable_joint_kv_cache_reuse", False):
        raise ValueError(
            "the disagg draft-window transfer requires an unpaired draft KV pool "
            "(enable_joint_kv_cache_reuse=False); "
            f"set {DISABLE_DISAGG_DRAFT_KV_ENV}=1 to serve without it"
        )
    return draft_kv_cache_manager


# ---------------------------------------------------------------------------------------------
# Per-request draft-window state and the fallback to an empty drafter window
# ---------------------------------------------------------------------------------------------

# Why a request's window is not shipped. The request itself never fails: it runs as if the
# transfer did not exist (the generation side's drafter starts from an empty window).
FALLBACK_NO_STORE_EVENT = "no_store_event"  # no drafter context store was recorded for it
FALLBACK_CTX_LEN_MISMATCH = "ctx_len_mismatch"  # the drafter did not store the whole prompt
FALLBACK_REUSE_BASE = "reuse_base"  # ctx_base outside the prompt's pages
FALLBACK_PAGE_MISMATCH = "page_mismatch"  # the context side lacks a window page
FALLBACK_NO_STATE = "no_ctx_state"  # generation side: no state arrived for the request
# The context TP ranks disagreed about a request (one shipped its slice, another did not), seen
# by the sender as a draft table on one side only, and by the generation ranks as a short draft
# write or an unusable state on some rank; every generation rank then falls back together.
FALLBACK_CTX_RANK_DIVERGENCE = "ctx_rank_divergence"
FALLBACK_RECEIVER_NOT_ASKED = "receiver_not_asked"  # context side shipped, the receiver asked none
FALLBACK_UNVERIFIED_WRITE = "unverified_write"  # the receive's byte attestation did not cover it
# The generation side receives synchronously (TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1) with TP
# ranks that share requests: that path has no cross-rank agreement on a window's usability, so it
# receives none (target only) rather than let its ranks draft from different windows.
FALLBACK_SYNC_PATH_TP = "sync_path_tp"


def draft_receive_disabled_reason(mapping, synchronous_receive: bool) -> Optional[str]:
    """Why this rank must take no drafter window, or None.

    The synchronous receive decides a window's usability per rank with no consensus; TP ranks
    that share every request (TP > 1 without attention DP) could then seed different windows.
    """
    if synchronous_receive and mapping.tp_size > 1 and not mapping.enable_attention_dp:
        return FALLBACK_SYNC_PATH_TP
    return None


def encode_draft_ctx_state(ctx_len: int, ctx_base: int, valid: bool) -> List[int]:
    """The context side's drafter state for a context-first response's ``draft_ctx_state``.

    It rides its own DisaggregatedParams field, never ``opaque_state``: that one becomes the C++
    ContextPhaseParams' serialized DataTransceiverState, which parses and re-serializes the bytes.
    """
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


def empty_window_state(prompt_len: int, tokens_per_block: int) -> Tuple[int, int]:
    """Drafter ``(ctx_len, ctx_base)`` for a request whose window did not arrive.

    The generation side's draft pool holds only the pages behind the prompt end, so a drafter
    numbering from position 0 would address released pages. Basing the slot past every prompt
    page reads nothing and appends into the pages it holds: the drafter starts from an empty
    window, exactly as without the transfer.
    """
    return int(prompt_len), -(-int(prompt_len) // int(tokens_per_block))


class DraftKvCounters:
    """``draft_kv_sent`` / ``draft_kv_fallback{reason}`` for one side, logged periodically.

    One info line per ``interval_s`` (and on the first event) so the serving log carries the
    fallback rate; each fallback is also logged at debug level.
    """

    def __init__(self, role: str, interval_s: float = 60.0):
        self.role = role
        self.interval_s = interval_s
        self.sent = 0
        self.fallback: Counter = Counter()
        self._last_log: Optional[float] = None
        # The sender's worker threads count divergences too.
        self._lock = threading.Lock()

    def record_sent(self) -> None:
        with self._lock:
            self.sent += 1
        self._maybe_log()

    def record_fallback(self, reason: str, detail: str = "") -> None:
        with self._lock:
            first = self.fallback[reason] == 0
            self.fallback[reason] += 1
        from tensorrt_llm.logger import logger

        message = f"Drafter context window ({self.role}) falls back: reason={reason} {detail}"
        if first:
            # The first of each reason at info, with its detail; the rest at debug.
            logger.info(message)
        else:
            logger.debug(message)
        self._maybe_log()

    def summary(self) -> str:
        reasons = ",".join(f"{reason}={count}" for reason, count in sorted(self.fallback.items()))
        return (
            f"Drafter context window ({self.role}): draft_kv_sent={self.sent} "
            f"draft_kv_fallback={sum(self.fallback.values())} "
            f"draft_kv_fallback_by_reason={{{reasons}}}"
        )

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if self._last_log is not None and now - self._last_log < self.interval_s:
            return
        self._last_log = now
        from tensorrt_llm.logger import logger

        logger.info(self.summary())

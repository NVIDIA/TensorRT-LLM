# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 GEN request-admission instrumentation and switches (default off).

TRTLLM_Q35_SCHED_TIMERS=1: per-call timers of the executor main thread's request-admission path
(`_fetch_new_requests`, `_schedule`) with a breakdown into its parts (request broadcast / merge into LlmRequests,
scheduler loop, KV cache manager V2 cache creation, prompt-token copies, resize). A call slower than
TRTLLM_Q35_SCHED_TIMERS_SLOW_MS (default 2) logs its breakdown (at most 3000 lines); every 5000 calls of a scope log
the cumulative totals. Off: one module-level bool check per call site.

TRTLLM_Q35_GEN_ADMIT=1: KVCacheManagerV2 skips the whole-prompt `get_tokens(0)` copy (C++ vector -> Python list) when
creating a request's cache without block reuse, the disagg generation case, where the tokens are never used.
"""

import os
import time
from collections import defaultdict

from tensorrt_llm.logger import logger

ENABLED = os.environ.get("TRTLLM_Q35_SCHED_TIMERS", "0") == "1"
GEN_ADMIT_LEAN = os.environ.get("TRTLLM_Q35_GEN_ADMIT", "0") == "1"
_SLOW_MS = float(os.environ.get("TRTLLM_Q35_SCHED_TIMERS_SLOW_MS", "2") or 2)
_MAX_SLOW_LINES = 3000
_SUMMARY_EVERY = 5000

_scopes = []  # stack of active scope names
_cur = defaultdict(lambda: defaultdict(float))  # scope -> part -> seconds of the current call
_cur_n = defaultdict(lambda: defaultdict(int))  # scope -> part -> count of the current call
_tot = defaultdict(lambda: defaultdict(float))  # scope -> part -> cumulative seconds
_tot_n = defaultdict(lambda: defaultdict(int))
_calls = defaultdict(int)
_slow_calls = defaultdict(int)
_slow_lines = 0
_lean_fired = 0
_lean_banner = False


def begin(scope):
    """Open a timed call of `scope` (nests: parts are added to the innermost open scope)."""
    _scopes.append(scope)
    _cur[scope].clear()
    _cur_n[scope].clear()


def add(part, seconds, count=1):
    """Charge `seconds` (and `count` events) to `part` of the innermost open scope."""
    if not _scopes:
        return
    scope = _scopes[-1]
    _cur[scope][part] += seconds
    _cur_n[scope][part] += count


def end(scope, seconds, extra=""):
    """Close the call of `scope`; log its breakdown when slow and the cumulative totals periodically."""
    global _slow_lines
    if _scopes and _scopes[-1] == scope:
        _scopes.pop()
    _calls[scope] += 1
    parts = _cur[scope]
    for p, v in parts.items():
        _tot[scope][p] += v
        _tot_n[scope][p] += _cur_n[scope][p]
    _tot[scope]["total"] += seconds
    _tot_n[scope]["total"] += 1
    if seconds * 1e3 >= _SLOW_MS:
        _slow_calls[scope] += 1
        if _slow_lines < _MAX_SLOW_LINES:
            _slow_lines += 1
            detail = ", ".join(f"{p} {v * 1e3:.2f} ms/{_cur_n[scope][p]}"
                               for p, v in sorted(parts.items(), key=lambda x: -x[1]))
            logger.info(f"[Q35_SCHED_TIMERS] slow {scope} {seconds * 1e3:.2f} ms ({extra}): {detail}")
    if _calls[scope] % _SUMMARY_EVERY == 0:
        tot = _tot[scope]
        detail = ", ".join(f"{p} {v * 1e3:.1f} ms/{_tot_n[scope][p]}"
                           for p, v in sorted(tot.items(), key=lambda x: -x[1]))
        logger.info(f"[Q35_SCHED_TIMERS] {scope}: {_calls[scope]} calls, {_slow_calls[scope]} >= {_SLOW_MS:g} ms; "
                    f"cumulative {detail}; GEN_ADMIT lean skips {_lean_fired}")


def note_lean_skip():
    """Count one skipped whole-prompt token copy (TRTLLM_Q35_GEN_ADMIT)."""
    global _lean_fired, _lean_banner
    _lean_fired += 1
    if not _lean_banner:
        _lean_banner = True
        logger.info("TRTLLM_Q35_GEN_ADMIT=1: skipping the whole-prompt get_tokens(0) copy on cache creation "
                    "without block reuse (disagg generation admission)")


now = time.perf_counter

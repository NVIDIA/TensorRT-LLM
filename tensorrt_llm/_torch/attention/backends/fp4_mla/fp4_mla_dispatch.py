# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA decode path selection: ``2x`` packed (default) | ``4x`` four-CTA-cluster | ``twostream`` (4x arm + 2x arm).

The decision is host arithmetic on static shapes plus the KV length, so it is the same in the eager warm-up, at
CUDA-graph capture and (through the graph that was captured) at replay.  Measured basis (hecate0358/0369, Rubin
212 SMs, 128 heads, graph replays, 2026-11-02; see the two-stream planner for the item cost model):

======================================  ==========  ===========  ===================  ===================
shape                                   2x packed   4x one-shot  two-stream           best
======================================  ==========  ===========  ===================  ===================
B128/q4, KV 30K                         520 us      484 us       456 us (r2 = 13)     two-stream (-12 %)
B128/q4, KV 16K                         299         286          272 (r2 = 13)        two-stream (-9 %)
B96/q4, KV 30K                          392         360          349 (r2 = 4)         two-stream (-11 %)
B64/q4, KV 30K                          274         244          = 4x (r2 = 0)        4x (-10 %)
B32/q4, KV 30K                          163         131          = 4x                 4x (-19 %)
B128/q2, KV 16K                         195         159          = 4x                 4x (-18 %)
B128/q4, KV 8K                          172         176          171 (r2 = 13)        tie (within 2 %)
B128/q4, KV 4K                          101         102          102                  tie
======================================  ==========  ===========  ===================  ===================

Graph-replay medians throughout (the 2x kernel's first replays at 8K are faster, 158 us cold, which is the only
sense in which 2x "wins" there); on the main tree's slower 2x packed kernel (#20576) the 8K row reads 190 / 191 /
192 and 4K 106 / 107 / 107.

Below ``TRTLLM_FP4_MLA_4X_MIN_KV`` (default 12 K tokens) the 2x packed kernel keeps the call: the 4x kernel pays
~13 us of fixed cost per 256-row item (task identity, barrier init, exposed Q load, pipeline fill and drain) that
only amortizes over long KV.  Above it the static 4x envelope (``fp4_mla_cutedsl_4x_host.envelope_reason_static``:
small-batch rules, Helix / softmax-stats / packed-V requirements, split-mode bounds) decides whether the 4x kernel
may run, and the two-stream planner (``fp4_mla_twostream``) decides whether handing the last requests to the 2x
kernel on a concurrent stream drops a whole 4x wave.

KV length used for the decision: ``live_max_kv`` when the caller knows the step's real longest KV (eager mode
narrows the page-table capacity to it; a graph runner that keys its graphs by KV regime can pass the regime
ceiling through ``Fp4MlaState.kv_regime_max_kv``), otherwise the page-table capacity.  The kernels themselves are
always compiled and planned for the capacity.

Environment:
``TRTLLM_FP4_MLA_CUTEDSL_4X``       master switch (default off): 0 -> always ``2x``.
``TRTLLM_FP4_MLA_DECODE_PATH``      ``auto`` (default) | ``2x`` | ``4x`` | ``twostream``: force a path (a forced 4x or
                                    two-stream path still needs the static envelope; ``twostream`` falls back to ``4x``
                                    when the planner finds no useful split).
``TRTLLM_FP4_MLA_4X_MIN_KV``        KV length (tokens) below which the call stays on ``2x`` (default 12288).
"""

from __future__ import annotations

import os
from typing import Callable, NamedTuple, Optional

from .fp4_mla_twostream import TwoStreamPlan, plan_two_stream

DECODE_PATH_ENV = "TRTLLM_FP4_MLA_DECODE_PATH"
MIN_KV_4X_ENV = "TRTLLM_FP4_MLA_4X_MIN_KV"
DEFAULT_MIN_KV_4X = 12288

PATH_2X = "2x"
PATH_4X = "4x"
PATH_TWO_STREAM = "twostream"
PATHS = (PATH_2X, PATH_4X, PATH_TWO_STREAM)

# KV regime ceilings a graph runner can key its decode graphs by (tokens).  ``kv_regime_ceiling`` maps a step's
# longest live KV to the smallest ceiling that covers it; the capacity itself is the last regime.
KV_REGIME_CEILINGS = (4096, 8192, 12288, 16384, 32768, 65536)


class DecodeStrategy(NamedTuple):
    path: str
    r2: int  # requests on the 2x arm when path == twostream, else 0
    reason: str
    kv_len_decision: int  # KV length the decision was taken on
    kv_source: str  # "live" | "capacity" | "forced"
    plan: Optional[TwoStreamPlan]

    @property
    def num_requests_2x(self) -> int:
        return self.r2 if self.path == PATH_TWO_STREAM else 0

    @property
    def record_tag(self) -> Optional[str]:
        """Tag for the dispatch recorder (``ts<r2>`` for two-stream, ``kv<decision>`` when live KV was used)."""
        parts = []
        if self.path == PATH_TWO_STREAM:
            parts.append(f"ts{self.r2}")
        if self.kv_source != "capacity":
            parts.append(f"kv{self.kv_len_decision}")
        return "|".join(parts) if parts else None


def _env_path() -> str:
    value = (os.getenv(DECODE_PATH_ENV) or "auto").strip().lower()
    return value if value in PATHS else "auto"


def min_kv_4x() -> int:
    value = os.getenv(MIN_KV_4X_ENV)
    if value is None or value.strip() == "":
        return DEFAULT_MIN_KV_4X
    try:
        return max(0, int(value))
    except ValueError:
        return DEFAULT_MIN_KV_4X


def kv_regime_ceiling(max_kv_len: int, capacity: int) -> int:
    """Smallest regime ceiling covering ``max_kv_len`` (never above ``capacity``)."""
    for ceiling in KV_REGIME_CEILINGS:
        if max_kv_len <= ceiling and ceiling <= capacity:
            return ceiling
    return int(capacity)


def choose_decode_strategy(
    *,
    num_requests: int,
    query_len: int,
    num_heads: int,
    rows_per_seq: int,
    kv_capacity: int,
    sm_count: int,
    envelope_reason: Callable[..., Optional[str]],
    live_max_kv: Optional[int] = None,
    helix: bool = False,
    softmax_stats: bool = False,
    v_packed: bool = True,
    four_x_enabled: bool = True,
) -> DecodeStrategy:
    """Pick the decode path for one packed-Q decode call.

    ``envelope_reason`` is ``fp4_mla_cutedsl_4x_host.envelope_reason_static`` (injected so this module stays
    importable without the kernel host); it is always evaluated on the page-table capacity, the geometry the
    kernels are compiled for.
    """
    forced = _env_path()
    kv_source = "capacity"
    kv_decision = int(kv_capacity)
    if live_max_kv is not None and live_max_kv > 0:
        kv_decision = min(int(live_max_kv), int(kv_capacity))
        kv_source = "live"

    def two_x(reason: str) -> DecodeStrategy:
        return DecodeStrategy(PATH_2X, 0, reason, kv_decision, kv_source, None)

    if not four_x_enabled:
        return two_x("TRTLLM_FP4_MLA_CUTEDSL_4X=0")
    if forced == PATH_2X:
        return DecodeStrategy(
            PATH_2X, 0, f"forced by {DECODE_PATH_ENV}", kv_decision, "forced", None
        )
    if forced == "auto" and kv_decision < min_kv_4x():
        return two_x(
            f"KV {kv_decision} < {min_kv_4x()}: the 2x packed kernel has the lower fixed cost ({MIN_KV_4X_ENV})"
        )
    reason = envelope_reason(
        num_heads=num_heads,
        query_len=query_len,
        num_requests=num_requests,
        max_kv_len=kv_capacity,
        sm_count=sm_count,
        helix=helix,
        softmax_stats=softmax_stats,
        v_packed=v_packed,
    )
    if reason is not None:
        return two_x(reason)
    if forced == PATH_4X:
        return DecodeStrategy(
            PATH_4X, 0, f"forced by {DECODE_PATH_ENV}", kv_decision, "forced", None
        )
    plan = plan_two_stream(
        num_requests=num_requests,
        rows_per_request=rows_per_seq,
        max_kv_len=kv_decision,
        sm_count=sm_count,
    )
    if plan.enabled:
        prefix_reason = envelope_reason(
            num_heads=num_heads,
            query_len=query_len,
            num_requests=plan.num_requests_4x,
            max_kv_len=kv_capacity,
            sm_count=sm_count,
            helix=False,
            softmax_stats=False,
            v_packed=True,
        )
        if prefix_reason is not None:
            plan = plan._replace(
                r2=0,
                reason=f"4x prefix of {plan.num_requests_4x} requests outside the envelope: {prefix_reason}",
            )
    if plan.enabled:
        return DecodeStrategy(
            PATH_TWO_STREAM,
            plan.r2,
            plan.reason,
            kv_decision,
            kv_source if forced == "auto" else "forced",
            plan,
        )
    return DecodeStrategy(
        PATH_4X,
        0,
        plan.reason if forced == "auto" else f"forced {forced}; {plan.reason}",
        kv_decision,
        kv_source if forced == "auto" else "forced",
        plan,
    )


def describe(strategy: DecodeStrategy) -> str:
    extra = f" r2={strategy.r2}" if strategy.path == PATH_TWO_STREAM else ""
    return f"{strategy.path}{extra} (kv {strategy.kv_len_decision} from {strategy.kv_source}): {strategy.reason}"

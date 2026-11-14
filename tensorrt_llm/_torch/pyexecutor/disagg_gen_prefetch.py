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
"""Generation-worker KV prefetch for disaggregated serving.

A disaggregated generation worker normally admits at most ``max_batch_size``
requests, so the next request's KV transfer only starts after a running
request finishes. With ``TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS=N`` (opt-in; unset
or 0 keeps the default behavior) the worker admits up to ``N`` more
generation-only requests. They receive their KV while the running batch
decodes; the forward batch is still capped at ``max_batch_size`` by the
scheduler's request budget (a request that is still receiving its KV is not
counted in that budget and is not part of the forward batch, and a request
whose KV has landed is ordered after the running decodes).

The extra requests need room in three pools, all widened by ``N``:

* the request admission cap (``PyExecutor._get_request_admission_capacity``);
* the sequence-slot pool (``model_engine.max_num_seq_slots``). A prefetched
  request is scheduled in the iteration where the request it replaces is
  ``GENERATION_TO_COMPLETE`` and still owns its slot;
* the KV-cache index pool (``KVCacheManagerV2.max_admissible_sequences``) and
  the standalone draft pool's per-request mirrors sized from the same count.

``N = 0`` leaves every pool and the admission cap unchanged.

Prefetch is turned off when ``TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN=1``.
That scheduler option orders generation-only requests that have not emitted
their first token ahead of running decodes, so a prefetched request whose KV
has landed would take a running request's place in the forward batch.
"""

import os
import time
from typing import Dict, Iterable, Optional

from tensorrt_llm.logger import logger

DISAGG_GEN_PREFETCH_REQUESTS_ENV = "TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS"
# Off unless requested. One request is the useful setting for a small decode
# batch: a generation worker of one to three requests then has the next
# request's KV arriving while its batch decodes (pair it with the gen router's
# max_inflight_per_server at max_batch_size + 1).
DEFAULT_DISAGG_GEN_PREFETCH_REQUESTS = 0
# Read with the same rule as ``scheduler_v2`` so both agree on whether it is on.
PRIORITIZE_FIRST_TOKEN_ENV = "TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN"  # nosec B105 - env var name

LOG_TAG = "[disagg_gen_prefetch]"


def _prioritize_first_token_enabled() -> bool:
    return os.environ.get(PRIORITIZE_FIRST_TOKEN_ENV, "0") == "1"


def _explicitly_requested() -> bool:
    return os.environ.get(DISAGG_GEN_PREFETCH_REQUESTS_ENV, "").strip() != ""


def get_disagg_gen_prefetch_requests() -> int:
    """Return the requested number of prefetch requests (the default when unset)."""
    raw = os.environ.get(DISAGG_GEN_PREFETCH_REQUESTS_ENV, "").strip()
    if not raw:
        return DEFAULT_DISAGG_GEN_PREFETCH_REQUESTS
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"{DISAGG_GEN_PREFETCH_REQUESTS_ENV} must be a non-negative integer, got {raw!r}"
        ) from e
    if value < 0:
        raise ValueError(
            f"{DISAGG_GEN_PREFETCH_REQUESTS_ENV} must be a non-negative integer, got {value}"
        )
    return value


def resolve_disagg_gen_prefetch_requests(
    *,
    is_disagg: bool,
    enable_attention_dp: bool,
    has_pp: bool,
    is_hybrid: bool = False,
    has_mrope_delta_cache: bool = False,
    kv_cache_manager_is_v2: bool = True,
    warn: bool = True,
) -> int:
    """Return the effective number of prefetch requests for this executor.

    Prefetch is applied only where widening the sequence-slot pool is known
    to be safe: a disaggregated executor with KV cache manager V2, no
    attention DP, no pipeline parallelism, no pool that is sized from
    ``max_batch_size`` instead of the sequence-slot pool (hybrid linear
    attention, MRoPE delta cache), and a scheduler that never lets a request
    that has not started decoding go ahead of a running one
    (``TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN`` unset). Elsewhere it returns 0
    and warns only when the setting was given explicitly. Every pool that
    prefetch widens is sized from this function, so they agree on the value.
    """
    requested = get_disagg_gen_prefetch_requests()
    if requested == 0:
        return 0
    reasons = []
    if not is_disagg:
        reasons.append("no KV cache transceiver")
    if enable_attention_dp:
        reasons.append("attention DP")
    if has_pp:
        reasons.append("pipeline parallelism")
    if is_hybrid:
        reasons.append("hybrid linear attention")
    if has_mrope_delta_cache:
        reasons.append("MRoPE delta cache")
    if not kv_cache_manager_is_v2:
        reasons.append("KV cache manager V1")
    if _prioritize_first_token_enabled():
        reasons.append(f"{PRIORITIZE_FIRST_TOKEN_ENV}=1")
    if reasons:
        if warn and _explicitly_requested():
            logger.warning_once(
                f"{DISAGG_GEN_PREFETCH_REQUESTS_ENV}={requested} is ignored "
                f"({', '.join(reasons)}).",
                key="disagg_gen_prefetch_ignored",
            )
        return 0
    return requested


def _is_landed(request) -> bool:
    """KV received: transmission complete, or already handed to generation."""
    if getattr(request, "is_disagg_generation_transmission_complete", False):
        return True
    return not (
        getattr(request, "is_disagg_generation_init_state", False)
        or getattr(request, "is_disagg_generation_transmission_in_progress", False)
    )


def _is_decoding(request) -> bool:
    return (
        not getattr(request, "is_disagg_generation_init_state", False)
        and not getattr(request, "is_disagg_generation_transmission_in_progress", False)
        and not getattr(request, "is_disagg_generation_transmission_complete", False)
    )


class PrefetchEngagement:
    """Engagement log for the prefetch slots (rank 0 only).

    ``note_admitted`` records a request admitted beyond ``max_batch_size``;
    ``observe`` (once per executor iteration, a no-op while nothing is
    tracked) logs when that request's KV has landed and whether other requests
    were decoding at that moment, i.e. whether the transfer overlapped decode.
    The first ``detail_lines`` events of each kind are logged one line each,
    then every ``every``-th; a cumulative summary line follows every
    ``summary_s`` seconds.
    """

    def __init__(
        self,
        enabled: bool,
        prefetch: int,
        max_batch_size: int,
        detail_lines: int = 20,
        every: int = 200,
        summary_s: float = 60.0,
    ) -> None:
        self.enabled = enabled
        self.prefetch = prefetch
        self.max_batch_size = max_batch_size
        self.detail_lines = detail_lines
        self.every = max(int(every), 1)
        self.summary_s = summary_s
        self._pending: Dict[int, tuple] = {}
        self.admitted = 0
        self.landed_during_decode = 0
        self.landed_idle = 0
        self.gone = 0
        self._next_summary = time.monotonic() + summary_s

    def _detail(self, count: int) -> bool:
        return count <= self.detail_lines or count % self.every == 0

    def note_admitted(self, request_id: int, num_active: int) -> None:
        if not self.enabled:
            return
        self.admitted += 1
        self._pending[request_id] = (time.monotonic(), num_active)
        if self._detail(self.admitted):
            logger.debug(
                f"{LOG_TAG} admit rid={request_id} active_before={num_active} "
                f"cap={self.max_batch_size}+{self.prefetch} n_admitted={self.admitted}"
            )

    def observe(self, active_requests: Iterable) -> None:
        if not self.enabled:
            return
        if self._pending:
            by_id = {}
            decoding = 0
            for req in active_requests:
                rid = getattr(req, "py_request_id", None)
                if rid in self._pending:
                    by_id[rid] = req
                elif _is_decoding(req):
                    decoding += 1
            now = time.monotonic()
            for rid in list(self._pending):
                req = by_id.get(rid)
                if req is None:
                    self._pending.pop(rid)
                    self.gone += 1
                    continue
                if not _is_landed(req):
                    continue
                start, _ = self._pending.pop(rid)
                if decoding > 0:
                    self.landed_during_decode += 1
                    count = self.landed_during_decode
                else:
                    self.landed_idle += 1
                    count = self.landed_idle
                if self._detail(count):
                    logger.debug(
                        f"{LOG_TAG} landed rid={rid} during_decode={int(decoding > 0)} "
                        f"decoding={decoding} wait_ms={(now - start) * 1e3:.0f} "
                        f"n_during_decode={self.landed_during_decode} n_idle={self.landed_idle}"
                    )
        if self.admitted and time.monotonic() >= self._next_summary:
            self._next_summary = time.monotonic() + self.summary_s
            logger.debug(self.summary())

    def summary(self) -> str:
        return (
            f"{LOG_TAG} summary admitted={self.admitted} "
            f"landed_during_decode={self.landed_during_decode} "
            f"landed_idle={self.landed_idle} gone={self.gone} pending={len(self._pending)}"
        )


def make_engagement(prefetch: int, max_batch_size: int, rank: Optional[int]) -> PrefetchEngagement:
    return PrefetchEngagement(
        enabled=prefetch > 0 and (rank or 0) == 0, prefetch=prefetch, max_batch_size=max_batch_size
    )


def disagg_gen_prefetch_leases(*, is_disagg: bool, mapping) -> int:
    """KV leases to add for prefetch in a KV cache manager or a pool sized per resident request.

    The model-type gates (hybrid linear attention, MRoPE delta cache) are not
    known there; on those models the engine adds no seats, so the extra leases
    are unused headroom.
    """
    return resolve_disagg_gen_prefetch_requests(
        is_disagg=is_disagg,
        enable_attention_dp=mapping.enable_attention_dp,
        has_pp=mapping.has_pp(),
        warn=False,
    )

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
request finishes. Setting ``TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS=N`` lets the
worker admit up to ``N`` more generation-only requests. They receive their KV
while the running batch decodes; the forward batch is still capped at
``max_batch_size`` by the scheduler budget.

The extra requests need room in three pools, all widened by ``N``:

* the request admission cap (``PyExecutor._get_request_admission_capacity``);
* the sequence-slot pool (``model_engine.max_num_seq_slots``). A prefetched
  request is scheduled in the iteration where the request it replaces is
  ``GENERATION_TO_COMPLETE`` and still owns its slot;
* the KV-cache index pool (``KVCacheManagerV2.max_admissible_sequences``).

``N = 0`` (the default) leaves every pool and the admission cap unchanged.

Prefetch is turned off when ``TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN=1``.
That scheduler option orders generation-only requests that have not emitted
their first token ahead of running decodes, so a prefetched request whose KV
has landed would take a running request's place in the forward batch, and with
it that request's speculative-decoding state, for an iteration.
"""

import os

from tensorrt_llm.logger import logger

DISAGG_GEN_PREFETCH_REQUESTS_ENV = "TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS"
# Read with the same rule as ``scheduler_v2`` so both agree on whether it is on.
PRIORITIZE_FIRST_TOKEN_ENV = "TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN"


def _prioritize_first_token_enabled() -> bool:
    return os.environ.get(PRIORITIZE_FIRST_TOKEN_ENV, "0") == "1"


def get_disagg_gen_prefetch_requests() -> int:
    """Return the requested number of prefetch requests (0 when unset)."""
    raw = os.environ.get(DISAGG_GEN_PREFETCH_REQUESTS_ENV, "").strip()
    if not raw:
        return 0
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
    (``TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN`` unset). Elsewhere the setting
    is ignored. Every pool that prefetch widens is sized from this function,
    so they agree on the value.
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
        if warn:
            logger.warning_once(
                f"{DISAGG_GEN_PREFETCH_REQUESTS_ENV}={requested} is ignored "
                f"({', '.join(reasons)}).",
                key="disagg_gen_prefetch_ignored",
            )
        return 0
    return requested

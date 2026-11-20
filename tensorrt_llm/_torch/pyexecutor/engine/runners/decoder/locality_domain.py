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
"""Locality domain request ordering and split arithmetic for the decoder runner."""

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


def locality_domain_forkjoin_enabled(kv_cache_manager) -> bool:
    """Whether to drive two-stream attention for *kv_cache_manager*.

    True iff the manager is :class:`KVCacheManagerV2` with
    ``num_locality_domains > 1`` and ``fork_join_attn`` set.
    """
    if not isinstance(kv_cache_manager, KVCacheManagerV2):
        return False
    if kv_cache_manager.num_locality_domains <= 1:
        return False
    return kv_cache_manager.fork_join_attn


def sort_generation_requests_by_locality_domain(
    requests: list[LlmRequest], promoted_context_request_ids: frozenset[int] = frozenset()
) -> list[LlmRequest]:
    """Group by locality domain inside the overlap scheduler's packing order.

    Input tokens are packed as [rows whose token comes from the host][rows that
    read it from the previous step's output], while positions follow request
    order, so host-token rows must stay first. CUDA graph padding dummies stay
    at the tail so ``pad_batch``'s ``generation_requests[:-padding_size]`` trims
    only them.
    """

    def reads_previous_step(request: LlmRequest) -> bool:
        return not (
            request.py_request_id in promoted_context_request_ids
            or request.is_dummy
            or request.py_batch_idx is None
        )

    return sorted(
        requests,
        key=lambda r: (r.is_cuda_graph_dummy, reads_previous_step(r), r.py_locality_domain_id),
    )


def balanced_generation_dispatch(num_rows: int, num_locality_domains: int) -> list[int]:
    """Domain for each packed generation row, by position only.

    CUDA graph capture bakes the per-domain request counts in as launch scalars,
    so the split must depend on the padded row count alone, never on which
    domain owns a row: capturing (half-2, half+2) and replaying (half+1, half-1)
    crashes. Rows that are repeated padding dummies count once per occurrence.
    """
    base, extra = divmod(num_rows, num_locality_domains)
    dispatch: list[int] = []
    for domain in range(num_locality_domains):
        dispatch.extend([domain] * (base + (1 if domain < extra else 0)))
    return dispatch


def locality_domain_row_splits(
    packed_requests: list[LlmRequest],
    sequence_lengths: list[int],
    num_locality_domains: int,
    *,
    positional: bool,
) -> tuple[list[int], list[int]]:
    """Per-domain request and token counts over one packed row range.

    The attention side turns the counts into cumulative ranges, so ownership
    can only be followed when the packed order is sorted by domain. Mixed
    generation categories (extend, generation) and unsorted context rows under
    chunked prefill break that, and then the rows are split by position
    instead, which is still correct and only costs locality.
    """
    if not positional:
        dispatch = [request.py_locality_domain_id for request in packed_requests]
        positional = dispatch != sorted(dispatch)
    if positional:
        dispatch = balanced_generation_dispatch(len(packed_requests), num_locality_domains)
    per_domain_reqs = [0] * num_locality_domains
    per_domain_tokens = [0] * num_locality_domains
    for i, domain in enumerate(dispatch):
        per_domain_reqs[domain] += 1
        per_domain_tokens[domain] += sequence_lengths[i]
    return per_domain_reqs, per_domain_tokens

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
"""KV-cache prefetch admission for a disaggregated generation worker.

A generation worker admits at most ``max_num_active_requests`` (the batch size)
requests, and a request in KV-cache transfer counts as active. When the batch is
full the next request therefore waits in the queue until a running request
finishes; only then is it admitted, its KV cache received (tens of ms for long
prompts) and its first step scheduled, so every hand-over runs the remaining
request alone for several steps.

``TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS=<k>`` lets the worker admit up to ``k``
generation-only requests beyond the batch size. Their KV-cache transfer runs
while the batch is full; the scheduler still schedules at most the batch size,
so an admitted request whose transfer has completed waits until a slot frees and
then joins without a transfer in between. ``k`` is clamped to the batch size:
the KV cache manager reserves index slots for one batch of requests in transfer
on top of the active batch (see ``KVCacheManagerV2``). Default 0 keeps the
current admission.
"""

import os
from typing import Callable, List

PREFETCH_REQUESTS_ENV = "TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS"


def prefetch_request_limit(max_num_active_requests: int) -> int:
    """Number of extra generation-only requests the worker may admit.

    Args:
        max_num_active_requests: The worker's admission cap (batch size).

    Returns:
        ``TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS`` clamped to
        ``[0, max_num_active_requests]``; 0 when unset or not an integer.
    """
    value = os.environ.get(PREFETCH_REQUESTS_ENV, "0")
    try:
        limit = int(value)
    except ValueError:
        return 0
    return max(0, min(limit, max_num_active_requests))


def num_prefetch_admissions(
    limit: int, max_num_active_requests: int, num_active_requests: int, num_pending_init: int
) -> int:
    """How many more requests may be admitted for KV prefetch right now.

    Args:
        limit: Result of :func:`prefetch_request_limit`.
        max_num_active_requests: The worker's admission cap (batch size).
        num_active_requests: Requests already active, including the ones
            admitted in this iteration.
        num_pending_init: Admitted requests whose KV cache is not prepared
            yet (``DISAGG_GENERATION_INIT``), including the ones admitted in
            this iteration. One scheduling pass prepares all of them at once,
            and the KV cache manager's per-pass state-index buffer holds one
            batch, so they stay within the batch size.

    Returns:
        ``max_num_active_requests + limit - num_active_requests``, at most
        ``max_num_active_requests - num_pending_init``, and not below 0.
    """
    if limit <= 0:
        return 0
    return max(
        0,
        min(
            max_num_active_requests + limit - num_active_requests,
            max_num_active_requests - num_pending_init,
        ),
    )


def pop_prefetch_requests(
    waiting_queue, max_count: int, is_prefetchable: Callable[[object], bool]
) -> List:
    """Pop up to ``max_count`` requests from the head of ``waiting_queue``.

    Stops at the first request ``is_prefetchable`` rejects, so the queue order
    is kept and nothing but prefetchable requests is admitted past the cap.

    Args:
        waiting_queue: The executor's waiting queue (``peek_request`` /
            ``pop_request`` / truthiness).
        max_count: Maximum number of requests to pop.
        is_prefetchable: Predicate on a queue item, e.g. "generation-only
            request without child requests".

    Returns:
        The popped queue items, in queue order.
    """
    items = []
    while len(items) < max_count and waiting_queue:
        if not is_prefetchable(waiting_queue.peek_request()):
            break
        items.append(waiting_queue.pop_request())
    return items

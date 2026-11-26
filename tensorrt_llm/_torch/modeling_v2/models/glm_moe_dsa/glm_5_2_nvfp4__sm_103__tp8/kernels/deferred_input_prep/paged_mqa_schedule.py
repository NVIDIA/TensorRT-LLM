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
"""Host replica of DeepGEMM's SM100 paged-MQA-logits schedule metadata.

``deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms)``
launches ``sm100_paged_mqa_logits_metadata`` (DeepGEMM scheduler header
``sm100_paged_mqa_logits.cuh``, non-varlen branch), whose output is a pure
function of the per-request context lengths:

* ``work[r] = ceil(len[r] / SPLIT_KV) * next_n`` and its inclusive prefix sum;
* for every ``sm`` in ``[0, num_sms]``: ``w = sm * q + min(sm, rem)`` with
  ``q, rem = divmod(total_work, num_sms)``; the first request ``r`` with
  ``prefix[r] > w`` gives ``(r * next_n, (w - prefix[r - 1]) // next_n)``,
  and ``(num_requests * next_n, 0)`` past the end.

``len[r]`` is ``context_lens[r, -1]`` for a 2-D ``[requests, next_n]``
input. deferred_input_prep stages the result as a packet copy because the lengths are known on
the host when the decode inputs are prepared; ``test_input_prep_kernels.py`` checks it
against the native kernel.
"""

import bisect
from typing import Sequence

import numpy as np

# DeepGEMM's KV split width for TRT-LLM's schedule block size (64 KV tokens
# per page, 4 pages per split); characterized against the native kernel by
# ``test_paged_mqa_schedule_host_replica_matches_deepgemm``.
SPLIT_KV_FOR_BLOCK_KV = {64: 256}


def paged_mqa_logits_schedule(
    context_lens: Sequence[int], next_n: int, split_kv: int, num_sms: int
) -> np.ndarray:
    """Return the ``[num_sms + 1, 2]`` int32 schedule for ``len(context_lens)`` requests."""
    n = len(context_lens)
    out = np.zeros((num_sms + 1, 2), dtype=np.int32)
    total_tokens = n * next_n
    if total_tokens == 0:
        return out
    prefix = []
    acc = 0
    for length in context_lens:
        acc += -(-int(length) // split_kv) * next_n
        prefix.append(acc)
    q, rem = divmod(acc, num_sms)
    for sm in range(num_sms + 1):
        w = sm * q + min(sm, rem)
        r = bisect.bisect_right(prefix, w)
        if r < n:
            before = prefix[r - 1] if r > 0 else 0
            out[sm, 0] = r * next_n
            out[sm, 1] = (w - before) // next_n
        else:
            out[sm, 0] = total_tokens
            out[sm, 1] = 0
    return out

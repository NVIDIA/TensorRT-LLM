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
"""Host-side pieces of the token-sharded (ShardQ) DSA sparse MLA prefill."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as dsa_indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa import shardq


@pytest.mark.parametrize("count,tp", [(16384, 8), (15367, 8), (5, 8), (3001, 4)])
def test_bounds_match_indexer_query_split(count, tp):
    begins, ends = shardq._shardq_bounds(count, tp)
    assert begins[0] == 0 and ends[-1] == count
    assert all(e == b for e, b in zip(ends[:-1], begins[1:]))
    for rank in range(tp):
        mapping = SimpleNamespace(enable_attention_dp=False, tp_size=tp, tp_rank=rank)
        b, e, sizes = dsa_indexer._split_prefill_queries(count, mapping, 0)
        assert (b, e) == (begins[rank], ends[rank])
        assert sizes == [hi - lo for lo, hi in zip(begins, ends)]


def _metadata(q_lens, cached):
    kv = [q + c for q, c in zip(q_lens, cached)]
    indptr = torch.zeros(len(kv) + 1, dtype=torch.int64)
    indptr[1:] = torch.cumsum(torch.tensor(kv), 0)
    return SimpleNamespace(
        num_contexts=len(q_lens),
        seq_lens=torch.tensor(q_lens, dtype=torch.int32),
        host_ctx_kv_indptr=indptr,
    )


def test_compact_rows_cover_only_the_owning_requests():
    # Three requests of 5000 / 6000 / 4000 new tokens on 100k / 50k / 10k cached tokens.
    meta = _metadata([5000, 6000, 4000], [100000, 50000, 10000])
    kv0, kv1, kv2 = 105000, 56000, 14000
    assert shardq._compact_rows_of_tokens(meta, 0, 1875) == (0, kv0)
    assert shardq._compact_rows_of_tokens(meta, 3750, 5625) == (0, kv0 + kv1)
    assert shardq._compact_rows_of_tokens(meta, 5000, 11000) == (kv0, kv0 + kv1)
    assert shardq._compact_rows_of_tokens(meta, 13125, 15000) == (kv0 + kv1, kv0 + kv1 + kv2)
    assert shardq._compact_rows_of_tokens(meta, 7, 7) == (0, 0)

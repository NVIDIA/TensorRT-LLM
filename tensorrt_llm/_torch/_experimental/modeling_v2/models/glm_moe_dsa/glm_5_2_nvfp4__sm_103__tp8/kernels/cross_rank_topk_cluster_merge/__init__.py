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
"""cross_rank_topk_cluster_merge: cross_rank_topk's context-partitioned top-k with a cluster merge
(selected with ``cross_rank_topk_cluster_merge``).

cross_rank_topk_cluster_merge extends cross_rank_topk (``cross_rank_topk``): the region, its policy
and its CP logits kernel are cross_rank_topk's (``indexer_topk/cross_rank_*.py``); the select / exchange /
merge op is ``glm_moe_dsa_cross_rank_topk_cluster_merge`` (``crossRankTopkClusterMerge.h``), which
merges each verify row on the row's eight CTAs as one thread-block cluster and keeps per-CTA epoch
words. Its output rows are byte-identical to cross_rank_topk's (set and order) on every rank.

Registered entry points (``dispatch.get_impl("cross_rank_topk_cluster_merge", name)``):

- ``cp_indexer_decode``: :func:`cp_indexer_decode.try_cp_indexer_decode`, the
  adapter hook of the GEN region (tried before cross_rank_topk's when
  ``cross_rank_topk_cluster_merge`` is selected).
- ``cp_decode_topk``: :func:`cp_indexer_decode.cp_decode_topk`, its
  tensor-level entry.
"""

from ...dispatch import register
from .cp_indexer_decode import cp_decode_topk, cp_topk_from_local_logits, try_cp_indexer_decode

register("cross_rank_topk_cluster_merge", "cp_indexer_decode", try_cp_indexer_decode)
register("cross_rank_topk_cluster_merge", "cp_decode_topk", cp_decode_topk)

__all__ = ["cp_decode_topk", "cp_topk_from_local_logits", "try_cp_indexer_decode"]

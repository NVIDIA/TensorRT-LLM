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
"""indexer_layer_front: the full-layer query front.

Requires ``absorbed_query``, ``indexer_topk``, ``draft_absorbed_query``, ``rope_kv_append`` and
``cross_rank_topk`` (the selector closure). Three forks, each byte-identical to the package it
forks:

* ``query_proj.dsa_proj``: absorbed_query's / draft_absorbed_query's full-layer query kernel
  (``kernels.front_qproj_indexer_kernel``) triggering its dependents after
  each program's last global load and reading positions after its wait.
  By default (``IndexerFrontConfig.norm="native"``) it takes the native q_a RMSNorm
  output as absorbed_query does; the options ``inline`` / ``programs`` normalize q_a
  inside the kernel instead (no native RMSNorm launch);
* ``rope.mla_rope_generation``: rope_kv_append's generation rope in which each thread
  triggers the dependents right after the global reads of its work item
  (``native_ops.indexer_layer_rope_kv_append``);
* ``cp_indexer_decode.try_cp_indexer_decode``: the cross_rank_topk / cross_rank_topk_cluster_merge
  GEN indexer region with cross_rank_topk's CP logits kernel waiting for its programmatic primary at
  entry and launched with PDL (``fp8_paged_mqa_logits_cp``).

Registered entry points (``dispatch.get_impl("indexer_layer_front", name)``): ``dsa_proj``,
``mla_rope_generation``, ``cp_indexer_decode``.
"""

from ... import native_ops
from ...dispatch import register
from .cp_indexer_decode import try_cp_indexer_decode
from .query_proj import IndexerFrontConfig, IndexerFrontQueryProjection, dsa_proj
from .rope import mla_rope_generation

register("indexer_layer_front", "dsa_proj", dsa_proj)
register("indexer_layer_front", "mla_rope_generation", mla_rope_generation)
register("indexer_layer_front", "cp_indexer_decode", try_cp_indexer_decode)

__all__ = [
    "IndexerFrontConfig",
    "IndexerFrontQueryProjection",
    "dsa_proj",
    "mla_rope_generation",
    "try_cp_indexer_decode",
]

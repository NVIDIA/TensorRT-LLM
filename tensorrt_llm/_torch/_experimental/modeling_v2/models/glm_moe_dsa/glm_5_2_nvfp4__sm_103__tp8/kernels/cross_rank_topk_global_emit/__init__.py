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
"""cross_rank_topk_global_emit: rope_kv_append's top-k bridge folded into
cross_rank_topk_cluster_merge's CP top-k emission (selected with ``cross_rank_topk_global_emit``).

cross_rank_topk_global_emit extends cross_rank_topk_cluster_merge
(``cross_rank_topk_cluster_merge``), cross_rank_topk (``cross_rank_topk``) and rope_kv_append
(``rope_kv_append``): the GEN indexer region is cross_rank_topk_cluster_merge's (cross_rank_topk's
policy and CP logits kernel), and its top-k op ``cross_rank_topk_global_emit``
(``crossRankTopkGlobalEmit.h``) writes, besides the id rows, the outputs of rope_kv_append's
``topkBridgeKernel`` for the call: the global KV-pool rows of the layer's remap group and the native
top-k copy rows. The backend's remap then returns those rows instead of launching the bridge, and
sparse_mla_decode runs directly behind the top-k. rope_kv_append's bridge still serves every call
cross_rank_topk_global_emit does not (MTP draft steps 1-4, which reuse the shared rows, and every
decline).

Registered entry points (``dispatch.get_impl("cross_rank_topk_global_emit", name)``):

- ``new_bridge_packet``: :func:`bridge.new_bridge_packet` (rope_kv_append's packet with a plan
  slot).
- ``cp_indexer_decode``: :func:`cp_indexer_decode.try_cp_indexer_decode`, the
  adapter hook of the GEN region (tried before cross_rank_topk_cluster_merge's when
  ``cross_rank_topk_global_emit`` is selected).
- ``remap_topk_to_global``: :func:`bridge.remap_topk_to_global`, the backend's
  remap (rope_kv_append's when the call carries no plan).
"""

from ...dispatch import register
from .bridge import (
    GlobalEmitBridgePacket,
    GlobalEmitContractError,
    new_bridge_packet,
    remap_topk_to_global,
)
from .cp_indexer_decode import cp_topk_from_local_logits, try_cp_indexer_decode

register("cross_rank_topk_global_emit", "new_bridge_packet", new_bridge_packet)
register("cross_rank_topk_global_emit", "cp_indexer_decode", try_cp_indexer_decode)
register("cross_rank_topk_global_emit", "remap_topk_to_global", remap_topk_to_global)

__all__ = [
    "GlobalEmitBridgePacket",
    "GlobalEmitContractError",
    "cp_topk_from_local_logits",
    "new_bridge_packet",
    "remap_topk_to_global",
    "try_cp_indexer_decode",
]

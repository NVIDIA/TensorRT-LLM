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
"""rope_kv_append: fused per-layer RoPE / FP8 query / KV append, indexer K-cache scatter
and top-k -> global KV index bridge for GLM-5.x DSA decode."""

from ... import dispatch
from .ops import (
    BridgeCopy,
    BridgePacket,
    IndexerKScatter,
    RopeKvAppendContractError,
    bridge_eligible,
    check_indexer_k_scatter_consumed,
    defer_indexer_k_scatter,
    defer_mtp_row_stash,
    defer_reuse_copy,
    defer_shared_copy,
    mla_rope_generation,
    new_bridge_packet,
    remap_topk_to_global,
    rope_eligible,
    run_indexer_k_scatter_native,
    topk_bridge,
)

_ENTRY_POINTS = {
    "rope_eligible": rope_eligible,
    "defer_indexer_k_scatter": defer_indexer_k_scatter,
    "check_indexer_k_scatter_consumed": check_indexer_k_scatter_consumed,
    "mla_rope_generation": mla_rope_generation,
    "run_indexer_k_scatter_native": run_indexer_k_scatter_native,
    "new_bridge_packet": new_bridge_packet,
    "defer_shared_copy": defer_shared_copy,
    "defer_reuse_copy": defer_reuse_copy,
    "defer_mtp_row_stash": defer_mtp_row_stash,
    "remap_topk_to_global": remap_topk_to_global,
}
for _name, _fn in _ENTRY_POINTS.items():
    dispatch.register("rope_kv_append", _name, _fn)

__all__ = [
    "BridgeCopy",
    "BridgePacket",
    "IndexerKScatter",
    "RopeKvAppendContractError",
    "bridge_eligible",
    "check_indexer_k_scatter_consumed",
    "defer_indexer_k_scatter",
    "defer_mtp_row_stash",
    "defer_reuse_copy",
    "defer_shared_copy",
    "mla_rope_generation",
    "new_bridge_packet",
    "remap_topk_to_global",
    "rope_eligible",
    "run_indexer_k_scatter_native",
    "topk_bridge",
]

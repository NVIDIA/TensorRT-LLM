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
"""How two ranks' cache pages correspond, for the native P/D transfer.

A rank's pages are its cache manager's page layout plus where its pools are (:class:`RankPages`).
One algorithm (``correspondence``) maps any two ranks' buffers onto each other from the layouts'
segments and their layer groups' grids alone; which units of a request move depends only on the
unit of their layer group (``units``). Nothing here is specific to a model: a cache describes itself
through its manager (``KVCacheManagerV2.get_page_layout`` and the per-request
``get_transfer_length`` family).
"""

from .correspondence import (
    MatchedPair,
    RunPlan,
    TransferGroup,
    buffer_runs,
    check_peer_compatible,
    match_buffers,
    notify_peer_registered,
    owns_transfer,
    plan_transfers,
    replicated,
    validate_peer_compatible,
)
from .ownership import requires_physical_ownership
from .rank_pages import BufferRef, PoolKey, RankPages
from .requests import (
    bounce_unit_bytes_per_layer_group,
    chunk_transfer_bytes,
    excluded_buffers,
    has_replicated_buffers,
    has_whole_request_groups,
    memory_registration_descs,
    receiver_extra_bytes,
)
from .topology import PeerOverlap, ShardGrid, rank_overlap, shard_grid
from .units import (
    ChunkContext,
    LocalUnitsContext,
    PairContext,
    RegistrationRegion,
    chunk_units,
    local_units,
    pair_units,
)

__all__ = [
    "BufferRef",
    "ChunkContext",
    "LocalUnitsContext",
    "MatchedPair",
    "PairContext",
    "PeerOverlap",
    "PoolKey",
    "RankPages",
    "RegistrationRegion",
    "RunPlan",
    "ShardGrid",
    "TransferGroup",
    "bounce_unit_bytes_per_layer_group",
    "buffer_runs",
    "check_peer_compatible",
    "chunk_transfer_bytes",
    "chunk_units",
    "excluded_buffers",
    "has_replicated_buffers",
    "has_whole_request_groups",
    "local_units",
    "match_buffers",
    "memory_registration_descs",
    "notify_peer_registered",
    "owns_transfer",
    "pair_units",
    "plan_transfers",
    "rank_overlap",
    "receiver_extra_bytes",
    "replicated",
    "requires_physical_ownership",
    "shard_grid",
    "validate_peer_compatible",
]

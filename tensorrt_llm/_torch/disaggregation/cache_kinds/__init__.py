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
"""Cache kinds for the native P/D transfer: what each cache is, and how its bytes map.

A rank's cache is a :class:`PageTable` of unit spaces and regions, each region carrying its own
layout declaration (:class:`BufferDecl`). One class per cache variant (``kinds/variants.py``)
claims the manager classes it understands and describes them (:class:`CacheKindSpec`); one algorithm
(``correspondence``) maps any two sides' regions onto each other. The rest of the transfer only
asks the spec of a unit space. See ``CACHE_KINDS_DEVELOPER_GUIDE.md`` for adding a kind.
"""

from . import kinds  # noqa: F401  (registers the built-in kinds)
from .build import build_page_table, requires_physical_ownership, role_layout
from .correspondence import (
    RegionRef,
    RunPlan,
    TransferGroup,
    check_peer_compatible,
    match_regions,
    notify_peer_registered,
    owns_transfer,
    plan_transfers,
    validate_peer_compatible,
)
from .page_table import BufferDecl, PageTable, Pool, Region, UnitSpace
from .queries import (
    bounce_unit_bytes_per_space,
    chunk_transfer_bytes,
    excluded_regions,
    has_replicated_regions,
    memory_registration_descs,
    receiver_extra_bytes,
    resume_after_receive,
    sends_first_tokens,
    transfer_length,
    whole_request_kinds,
)
from .registry import claim, register, registered_specs, spec_for, spec_named, specs_in
from .spec import CacheKindSpec, ChunkContext, LocalUnitsContext, PairContext, RegistrationRegion
from .topology import (
    ATTENTION_TP,
    STATE_TP,
    PeerOverlap,
    ShardGrid,
    attention_grid,
    rank_overlap,
    shard_grid,
    state_grid,
)

__all__ = [
    "ATTENTION_TP",
    "BufferDecl",
    "CacheKindSpec",
    "ChunkContext",
    "LocalUnitsContext",
    "PageTable",
    "PairContext",
    "PeerOverlap",
    "Pool",
    "Region",
    "RegionRef",
    "RegistrationRegion",
    "RunPlan",
    "STATE_TP",
    "ShardGrid",
    "TransferGroup",
    "UnitSpace",
    "attention_grid",
    "bounce_unit_bytes_per_space",
    "build_page_table",
    "check_peer_compatible",
    "chunk_transfer_bytes",
    "claim",
    "excluded_regions",
    "has_replicated_regions",
    "match_regions",
    "memory_registration_descs",
    "notify_peer_registered",
    "owns_transfer",
    "plan_transfers",
    "rank_overlap",
    "receiver_extra_bytes",
    "register",
    "registered_specs",
    "requires_physical_ownership",
    "resume_after_receive",
    "role_layout",
    "sends_first_tokens",
    "shard_grid",
    "spec_for",
    "spec_named",
    "specs_in",
    "state_grid",
    "transfer_length",
    "validate_peer_compatible",
    "whole_request_kinds",
]

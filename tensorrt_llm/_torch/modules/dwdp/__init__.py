# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""DWDP (Distributed Weight Data Parallelism) VA-based infrastructure.

MNNVL-based physical memory allocation, composite VA layout, page-aligned
mapping, and prefetch slot lifecycle management for DWDP expert weights.

Key design principles:
- No D2D copy: local experts are always zero-copy via cuMemMap
- Per-layer weight specs: different MoE layers may have different weight
  shapes/dtypes
- Page pool: remote buffer physical memory is a pool of pages assigned
  layer-by-layer
- Separation of concerns: WeightBuffer owns VA layout; Transport owns handle
  exchange; WeightManager owns scheduling
- One group API (allgather + Barrier) served by an MPI sub-communicator, or by
  a TCPStore rendezvous for independently launched workers -- see rendezvous.py

Dense DWDP (``dense/``) shards the replicated dense weights of the MoE layers the
same way.  Its exports (``DenseDWDP``, ``setup_dense_dwdp``) are resolved by the
module-level ``__getattr__`` on first use, so importing this package, which
every import of a ``dense`` module does first, loads no dense module.  Code
inside ``dense`` imports from the defining submodule, never from this package.
"""

import importlib
from typing import TYPE_CHECKING

from .page_pool import PagePool, compute_slot_sizes
from .rendezvous import StoreDwdpGroup, dwdp_global_rank, store_rendezvous_enabled
from .setup import setup_dwdp, teardown_dwdp
from .specs import (
    EdgeInfo,
    ExpertPartition,
    LayerWeightSpecs,
    MnnvlHandleSet,
    PageAlignedLayout,
    PeerRange,
    PeerRanges,
    WeightSpec,
    handle_geometry,
    lookup_owner,
)
from .transport import DWDPTransport
from .vmm import (
    VARegion,
    VMMHandle,
    align_down,
    align_up,
    get_allocation_granularity,
    get_allocation_prop,
    tensor_from_ptr,
)
from .weight_buffer import WeightBuffer
from .weight_manager import DWDPWeightManager

if TYPE_CHECKING:
    from .dense.runtime import DenseDWDP
    from .dense.setup import setup_dense_dwdp

_DENSE_EXPORTS = {
    "DenseDWDP": ".dense.runtime",
    "setup_dense_dwdp": ".dense.setup",
}

__all__ = [
    # Data classes
    "WeightSpec",
    "LayerWeightSpecs",
    "MnnvlHandleSet",
    "EdgeInfo",
    "PageAlignedLayout",
    "PeerRange",
    "PeerRanges",
    "ExpertPartition",
    "handle_geometry",
    "lookup_owner",
    # VMM utilities
    "align_up",
    "align_down",
    "get_allocation_prop",
    "get_allocation_granularity",
    "tensor_from_ptr",
    "VMMHandle",
    "VARegion",
    # Page pool
    "PagePool",
    "compute_slot_sizes",
    # Main classes
    "DWDPTransport",
    "WeightBuffer",
    "DWDPWeightManager",
    # Rendezvous
    "StoreDwdpGroup",
    "dwdp_global_rank",
    "store_rendezvous_enabled",
    # Setup orchestration
    "setup_dwdp",
    "teardown_dwdp",
    # Dense DWDP
    "DenseDWDP",
    "setup_dense_dwdp",
]


def __getattr__(name: str) -> object:
    """Resolve a dense DWDP export on first use and cache it on this module."""
    module_name = _DENSE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value

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
mapping, and double buffer lifecycle management for DWDP expert weights.

Key design principles:
- No D2D copy: local experts are always zero-copy via cuMemMap
- Per-layer weight specs: different MoE layers may have different weight
  shapes/dtypes
- Page pool: remote buffer physical memory is a pool of pages assigned
  layer-by-layer
- Separation of concerns: WeightBuffer owns VA layout; Transport owns handle
  exchange; WeightManager owns scheduling
- MPI everywhere: single communication backend (replaces tekit's TCPDWDPStore)
"""

from .page_pool import PagePool, compute_slot_sizes
from .setup import setup_dwdp
from .specs import EdgeInfo, LayerWeightSpecs, MnnvlHandleSet, PageAlignedLayout, WeightSpec
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

__all__ = [
    # Data classes
    "WeightSpec",
    "LayerWeightSpecs",
    "MnnvlHandleSet",
    "EdgeInfo",
    "PageAlignedLayout",
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
    # Setup orchestration
    "setup_dwdp",
]

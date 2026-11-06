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

"""Attention-owned descriptors and persistent sparse offload metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferId, LayerGroupId


@dataclass(frozen=True, slots=True)
class SparseOffloadLayerDescriptor:
    """One local model layer's main compressed buffer in KVCM's namespace.

    ``page_scale`` converts resident raw GPU slots to SHARED physical pages.
    ``fetched_page_scale`` must be bound separately once KVCM specifies fetch
    output units: 1 for SHARED physical pages, or the appropriate converter
    scale for raw slots in the same pool. None keeps decode disabled rather
    than guessing the unfinished API's units. Buffer pointers supply the
    layer offset in either case.
    """

    buffer_id: BufferId
    group_id: LayerGroupId
    page_scale: int
    # Exclusive physical-page bound relative to this layer's SHARED pointer,
    # as returned by KVCM. Its runtime implementation must include scratch.
    page_index_upper_bound: int
    fetched_page_scale: int | None = None


@dataclass(kw_only=True)
class SparseOffloadState:
    """Fixed-capacity storage owned by one serial attention metadata instance.

    All device tensors are int32. Tables are [B, M], selections are [B, S],
    history counts are [B], and the active count is [1]. History staging is
    pinned CPU memory. Layer workspaces are reused only after the previous
    layer's attention has consumed them on the same stream. Concurrent
    forwards require separate metadata and KVCM scratch ownership.

    ``prepared`` means preparation was enqueued successfully; stream ordering
    establishes device readiness. ``is_prefill`` is set by host batch validation
    before execution, never inferred from a device count inside a layer.

    ``read_table_valid`` is a private int32 [1] diagnostic flag, refreshed by
    GPU validation after each decode fetch. Only the device assertion requires
    ``TLLM_DSV4_OFFLOAD_DEBUG_ASSERT=1``, read at initialization before graph
    capture. A failed assertion makes the CUDA context unusable. By default,
    the flag does not trigger an error or request recovery; production trusts
    KVCM to provide valid mappings for every required page.
    """

    layers: dict[int, SparseOffloadLayerDescriptor]
    base_page_tables: dict[LayerGroupId, torch.Tensor]
    history_blocks: torch.Tensor
    history_blocks_host: torch.Tensor
    active_request_count: torch.Tensor
    selected_history_pages: torch.Tensor
    fetched_page_table: torch.Tensor
    compress_read_table: torch.Tensor
    read_table_valid: torch.Tensor
    debug_assert: bool = field(
        default_factory=lambda: os.environ.get("TLLM_DSV4_OFFLOAD_DEBUG_ASSERT", "0") == "1"
    )
    history_upload_done: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    history_upload_pending: bool = False
    prepared: bool = False
    is_prefill: bool = False

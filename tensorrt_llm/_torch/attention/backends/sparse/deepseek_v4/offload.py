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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferId, LayerGroupId


@dataclass(frozen=True, slots=True)
class SparseOffloadLayerDescriptor:
    """One local model layer's main compressed buffer in KVCM's namespace.

    ``page_scale`` converts resident raw GPU slots to SHARED physical pages.
    It does not specify the units of the unfinished sparse fetch API.
    """

    buffer_id: BufferId
    group_id: LayerGroupId
    page_scale: int


@dataclass(kw_only=True)
class SparseOffloadState:
    """Fixed-capacity storage owned by one serial attention metadata instance.

    All device tensors are int32. Tables are [B, M], selections are [B, S],
    history counts are [B], and the active count is [1]. History staging is
    pinned CPU memory. Layer workspaces are reused only after the previous
    layer's attention has consumed them on the same stream. Concurrent
    forwards require separate metadata and KVCM scratch ownership.
    """

    layers: dict[int, SparseOffloadLayerDescriptor]
    base_page_tables: dict[LayerGroupId, torch.Tensor]
    history_blocks: torch.Tensor
    history_blocks_host: torch.Tensor
    active_request_count: torch.Tensor
    selected_history_pages: torch.Tensor
    fetched_page_table: torch.Tensor
    compress_read_table: torch.Tensor
    history_upload_done: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    history_upload_pending: bool = False

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
"""Draft set of MTP-Eagle over the target's KV cache."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from ..base import _Registry
from ..core import TrtllmCoreMetadata
from .base import DraftMetadata, build_draft_set

if TYPE_CHECKING:
    from ..batch_inputs import BatchInputs


class SharedKVCacheLinearDraftMetadata(DraftMetadata):
    """The metadata of a linear draft chain of the target's own layers over the target's KV cache (MTP-Eagle).

    The draft layers are extra layers of the target's KV cache. They run the
    attention backend on a second set of metadata, draft_set: the target's types,
    built over that same cache with lengths of their own, so the draft steps
    leave the target's metadata untouched. The engine prepares it with each
    target step that drafts; the other draft metadata update it in place for
    each draft step.
    """

    # Constructed after the other metadata, whose types the draft set copies.
    builds_metadata = True

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        draft_manager = config.draft_kv_cache_manager
        if draft_manager is not None and draft_manager is not config.kv_cache_manager:
            raise ValueError("SharedKVCacheLinearDraftMetadata requires a shared draft KV cache")
        # The core of the metadata the draft layers receive as attn_metadata.
        self.draft_set = build_draft_set(
            self,
            dataclasses.replace(config, draft_kv_cache_manager=None, buffer_name_prefix="draft_"),
        )
        # The target and the draft layers never run at the same time, so the
        # draft set runs its attention kernels in the target's workspace.
        core = self.get_metadata(TrtllmCoreMetadata)
        self.draft_set.workspace = core.workspace
        self.draft_set.cuda_graph_workspace = core.cuda_graph_workspace

    def init_cuda_graph_metadata(self, graph_metadata: "SharedKVCacheLinearDraftMetadata") -> None:
        graph_metadata.draft_set.workspace = self.draft_set.workspace
        graph_metadata.draft_set.cuda_graph_workspace = self.draft_set.cuda_graph_workspace

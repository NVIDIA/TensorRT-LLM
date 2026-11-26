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
"""A DFlash drafter over a draft KV cache of its own."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..base import _Registry
from .base import DraftMetadata

if TYPE_CHECKING:
    from ..batch_inputs import BatchInputs


class DFlashDraftMetadata(DraftMetadata):
    """A DFlash drafter over a draft KV cache of its own.

    The drafter runs once per target step, on the generation rows, with its
    own attention over pages of the draft KV cache. It addresses those pages
    through the block offsets held here, in the target step's row order, and
    takes positions and lengths from the spec worker's own state.
    """

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        draft_manager = config.draft_kv_cache_manager
        num_pools = getattr(draft_manager, "num_attention_op_pools", draft_manager.num_pools)
        # Block offsets of the step's rows in the draft KV cache,
        # [num_pools, num_rows, 2, max_blocks_per_seq].
        self.draft_kv_cache_block_offsets = self.get_buffer(
            (num_pools, config.max_num_sequences, 2, draft_manager.max_blocks_per_seq),
            torch.int32,
            "draft_kv_cache_block_offsets",
        )

    def prepare(self, inputs: "BatchInputs") -> None:
        if not inputs.stage_host:
            return
        self.config.draft_kv_cache_manager.copy_batch_block_offsets(
            self.draft_kv_cache_block_offsets,
            inputs.request_ids,
            inputs.beam_width,
            inputs.num_contexts,
            inputs.num_seqs,
        )

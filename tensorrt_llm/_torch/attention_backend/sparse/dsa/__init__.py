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

# Re-export names that were module-level in the old flat dsa.py, needed for
# mock.patch targets in tests (e.g., 'sparse.dsa.RotaryEmbedding').
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding

from .backend import DSATrtllmAttention
from .cache_manager import DSACacheManager
from .indexer import (
    Indexer,
    IndexerPrefillChunkMetadata,
    _compute_slot_mappings,
    compute_cu_seqlen_kv_bounds_with_cache,
    rotate_activation,
    split_prefill_chunks,
    transform_local_topk_and_prepare_pool_view,
)
from .metadata import DSATrtllmAttentionMetadata, DSAtrtllmAttentionMetadata

__all__ = [
    "DSACacheManager",
    "DSATrtllmAttention",
    "DSATrtllmAttentionMetadata",
    "DSAtrtllmAttentionMetadata",
    "Indexer",
    "IndexerPrefillChunkMetadata",
    "RotaryEmbedding",
    "_compute_slot_mappings",
    "compute_cu_seqlen_kv_bounds_with_cache",
    "rotate_activation",
    "split_prefill_chunks",
    "transform_local_topk_and_prepare_pool_view",
]

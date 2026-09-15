# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""glm_kpool: the GLM-5.3-Flash pool-compressed sparse-MLA backend."""

from .backend import (
    INDEX_SENTINEL,
    GlmKpoolSparseAttention,
    latent_pool_rows,
    paged_slot_indices,
    positions_to_pool_rows,
)
from .cache_manager import Glm5NextCacheManager, Glm5NextMamba2Metadata
from .params import GlmKpoolBackendForwardArgs, GlmKpoolSparseParams

__all__ = [
    "INDEX_SENTINEL",
    "Glm5NextCacheManager",
    "Glm5NextMamba2Metadata",
    "GlmKpoolBackendForwardArgs",
    "GlmKpoolSparseAttention",
    "GlmKpoolSparseParams",
    "latent_pool_rows",
    "paged_slot_indices",
    "positions_to_pool_rows",
]

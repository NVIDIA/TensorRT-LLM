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
"""``QSAMambaHybridCacheManagerV2``'s layout: a hybrid manager's caches and QSA's index data."""

from __future__ import annotations

from tensorrt_llm._torch.attention.backends.sparse.qsa.cache_manager import (
    QSA_INDEX_POSITION,
    QSA_SIDE_PLANES,
)

from ...kv_cache_manager_v2 import Role
from ..page_layout import Tensor, replicated
from .mamba import MambaLayoutAdapter


class QSALayoutAdapter(MambaLayoutAdapter):
    def buffer_tensors(
        self, layer_id: int, role: str, size: int, tokens: int, state: bool
    ) -> tuple[Tensor, ...]:
        """Index positions are request-wide coordinates, so every rank holds the same bytes; so do
        the packed side planes and the index keys, which project one head alike on every rank.
        They carry no head axis, which splitting them by the rank's KV heads would assume."""
        if role in (QSA_INDEX_POSITION, QSA_SIDE_PLANES, Role.INDEX_KEY):
            return replicated(size)
        return super().buffer_tensors(layer_id, role, size, tokens, state)

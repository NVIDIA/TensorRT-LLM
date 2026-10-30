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
"""``Glm5NextCacheManager``'s layout: a hybrid manager's caches and indexer rows."""

from __future__ import annotations

from ...kv_cache_manager_v2 import Role
from ..page_layout import Tensor, replicated
from .mamba import MambaLayoutAdapter


class Glm5NextLayoutAdapter(MambaLayoutAdapter):
    def buffer_tensors(
        self, layer_id: int, role: str, size: int, tokens: int, state: bool
    ) -> tuple[Tensor, ...]:
        """The indexer rows are as wide on every tensor rank and hold the same bytes."""
        if role == Role.INDEX_KEY:
            return replicated(size)
        return super().buffer_tensors(layer_id, role, size, tokens, state)

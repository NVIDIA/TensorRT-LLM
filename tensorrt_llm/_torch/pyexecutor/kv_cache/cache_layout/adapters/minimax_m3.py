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
"""``MiniMaxM3KVCacheManagerV2``'s layout: K/V as its sparse backend writes them, and index-K."""

from __future__ import annotations

from ...kv_cache_manager_v2 import Role
from ..adapter import CacheLayoutAdapter
from ..page_layout import Tensor, replicated


class MiniMaxM3LayoutAdapter(CacheLayoutAdapter):
    def buffer_tensors(
        self, layer_id: int, role: str, size: int, tokens: int, state: bool
    ) -> tuple[Tensor, ...]:
        """K/V and their block scales split by head as the backend writes them, head-major (MSA)
        or token-major (Triton); index-K alike on every rank."""
        if role == Role.INDEX_KEY:
            return replicated(size)
        return self.head_split(
            layer_id,
            size,
            tokens,
            state,
            token_major_=self.manager._main_kv_storage_layout == "NHD",
        )

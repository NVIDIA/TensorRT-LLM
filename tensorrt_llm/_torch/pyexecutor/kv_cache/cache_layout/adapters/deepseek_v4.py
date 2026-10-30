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
"""``DeepseekV4CacheManager``'s layout: one MLA cache per attention type, compressed-cache block
scales and storage-only alignment pages."""

from __future__ import annotations

from typing import Optional

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    _ATTENTION_TYPE_ROLES,
    COMPRESS_BLOCK_SCALE_ROLE,
)

from ..adapter import CacheLayoutAdapter
from ..page_layout import Tensor


class DeepseekV4LayoutAdapter(CacheLayoutAdapter):
    def buffer_tensors(
        self, layer_id: int, role: str, size: int, tokens: int, state: bool
    ) -> tuple[Tensor, ...]:
        """The cache of each attention type and the compressed cache's block scales hold MLA
        latents of one KV head, which every tensor rank holds: split by head as the base adapter
        splits K/V. The padding pages never move (``untransferred_roles``)."""
        if role == COMPRESS_BLOCK_SCALE_ROLE or role in _ATTENTION_TYPE_ROLES:
            return self.head_split(layer_id, size, tokens, state)
        return super().buffer_tensors(layer_id, role, size, tokens, state)

    def untransferred_roles(self) -> frozenset[str]:
        """Storage-only NVFP4 alignment pages, which hold no state a peer needs."""
        return frozenset(str(role) for role in self.manager._disagg_ignored_roles)

    def quant_mode(self) -> Optional[str]:
        """``"fp8_ds_mla"`` for FP8 pages carrying their own scales."""
        return "fp8_ds_mla" if self.manager.use_fp8_ds_mla else super().quant_mode()

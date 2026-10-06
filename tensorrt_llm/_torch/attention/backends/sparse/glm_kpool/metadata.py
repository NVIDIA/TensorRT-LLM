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
"""Persistent GLM slot tables and host prefill schedules alongside KDA metadata."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from tensorrt_llm._utils import prefer_pinned

from .cache_manager import Glm5NextCacheManager


class Glm5NextMamba2Metadata(KimiK3MambaMetadata):
    def __init__(self, max_batch_size: int, chunk_size: int, max_num_tokens: int) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens)
        self.glm_block_tables: torch.Tensor | None = None
        self._glm_block_tables_cpu: torch.Tensor | None = None
        self.glm_cached_lens_host: list[int] = []
        self.glm_ctx_cu_seqlens: list[int] = [0]

    def _glm_ensure_tables(self, width: int) -> None:
        width = max(1, int(width))
        if self.glm_block_tables is None:
            self.glm_block_tables = torch.zeros(
                self.max_batch_size, width, dtype=torch.long, device="cuda"
            )
            self._glm_block_tables_cpu = torch.zeros(
                self.max_batch_size, width, dtype=torch.long, pin_memory=prefer_pinned()
            )
        elif self.glm_block_tables.shape[1] < width:
            raise RuntimeError(
                "glm5_next block-table buffer would need to grow from "
                f"{self.glm_block_tables.shape[1]} to {width} pages mid-run; "
                "captured CUDA graphs would keep reading the old buffer"
            )

    def prepare(self, attn_metadata) -> None:
        manager = attn_metadata.kv_cache_manager
        assert manager is None or isinstance(manager, Glm5NextCacheManager), (
            "glm5_next metadata requires Glm5NextCacheManager"
        )
        super().prepare(attn_metadata)
        kv_params = attn_metadata.kv_cache_params
        request_ids = attn_metadata.request_ids
        if (
            manager is None
            or kv_params is None
            or kv_params.num_cached_tokens_per_seq is None
            or request_ids is None
        ):
            return

        batch = attn_metadata.seq_lens.shape[0]
        num_contexts = int(attn_metadata.num_contexts)
        lens = [int(x) for x in attn_metadata.seq_lens[:batch]]
        cached_src = kv_params.num_cached_tokens_per_seq
        if isinstance(cached_src, torch.Tensor):
            cached = [int(x) for x in cached_src[:batch]]
        else:
            cached = [int(cached_src[i]) for i in range(batch)]

        self.glm_cached_lens_host = cached
        cu = [0]
        for length in lens[:num_contexts]:
            cu.append(cu[-1] + length)
        self.glm_ctx_cu_seqlens = cu

        width = manager.max_blocks_per_seq
        self._glm_ensure_tables(width)
        pages = manager.get_batch_slot_tables(list(request_ids)[:batch])
        staging = self._glm_block_tables_cpu
        staging[:batch].zero_()
        for row, page_ids in enumerate(pages):
            if page_ids:
                staging[row, : len(page_ids)].copy_(torch.as_tensor(page_ids, dtype=torch.long))
        self.glm_block_tables[:batch].copy_(staging[:batch], non_blocking=True)

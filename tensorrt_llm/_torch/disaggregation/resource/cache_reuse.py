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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, KVCacheManagerV2

from .page import AttentionLayerGroup
from .utils import get_global_layer_ids


class CacheReuseAdapter(ABC):
    """Uniform prefix-reuse API over KVCacheManager V1/V2."""

    @property
    @abstractmethod
    def enable_block_reuse(self) -> bool: ...

    @property
    @abstractmethod
    def tokens_per_block(self) -> int: ...

    @abstractmethod
    def get_cached_token_count(self, req: LlmRequest) -> int:
        """Block-aligned count of prefix tokens already cached for *req*."""

    @abstractmethod
    def get_block_ids(
        self,
        req: LlmRequest,
        group_idx: int,
        lg: AttentionLayerGroup,
    ) -> np.ndarray:
        """All block IDs for *req* in layer group *lg* (dtype ``int64``)."""

    @abstractmethod
    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        """Commit KV blocks to radix tree for future prefix reuse.

        Must be called after ``req.context_current_position = req.prompt_len``.
        """


class _CacheReuseAdapterV1(CacheReuseAdapter):
    """C++-backed KVCacheManager."""

    def __init__(self, mgr: KVCacheManager) -> None:
        self._mgr = mgr

    @property
    def enable_block_reuse(self) -> bool:
        return self._mgr.enable_block_reuse

    @property
    def tokens_per_block(self) -> int:
        return self._mgr.tokens_per_block

    def get_cached_token_count(self, req: LlmRequest) -> int:
        if not self.enable_block_reuse:
            return 0
        cached = req.prepopulated_prompt_len
        tpb = self.tokens_per_block
        return (cached // tpb) * tpb

    def get_block_ids(self, req, group_idx, lg):  # noqa: ARG002
        first_layer = get_global_layer_ids(lg)[0]
        return np.asarray(
            self._mgr.get_batch_cache_indices([req.py_request_id], layer_idx=first_layer)[0],
            dtype=np.int64,
        )

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        if not self.enable_block_reuse:
            return
        self._mgr.store_blocks_for_reuse(req, pin_blocks=False)


class _CacheReuseAdapterV2(CacheReuseAdapter):
    """Python-based KVCacheManagerV2."""

    def __init__(self, mgr: KVCacheManagerV2) -> None:
        self._mgr = mgr

    @property
    def enable_block_reuse(self) -> bool:
        return self._mgr.enable_block_reuse

    @property
    def tokens_per_block(self) -> int:
        return self._mgr.tokens_per_block

    def get_cached_token_count(self, req: LlmRequest) -> int:
        if not self.enable_block_reuse:
            return 0
        kv_cache = self._mgr.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return 0
        cached = kv_cache.num_committed_tokens
        tpb = self.tokens_per_block
        return (cached // tpb) * tpb

    def get_block_ids(self, req, group_idx, lg):  # noqa: ARG002
        return np.fromiter(
            self._mgr.kv_cache_map[req.py_request_id].get_aggregated_page_indices(
                group_idx, valid_only=True
            ),
            dtype=np.int64,
        )

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        if not self.enable_block_reuse:
            return
        kv_cache = self._mgr.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return
        self._mgr.try_commit_blocks_for_reuse(req, kv_cache)


def create_cache_reuse_adapter(
    mgr: Union[KVCacheManager, KVCacheManagerV2],
) -> CacheReuseAdapter:
    """Factory — pick the right adapter for the concrete manager type."""
    if isinstance(mgr, KVCacheManagerV2):
        return _CacheReuseAdapterV2(mgr)
    return _CacheReuseAdapterV1(mgr)

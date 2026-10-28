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
from typing import List, Sequence

import numpy as np

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

from .page import AttentionLayerGroup


class CacheReuseAdapter(ABC):
    """Prefix-reuse API for KVCacheManagerV2."""

    @property
    @abstractmethod
    def enable_block_reuse(self) -> bool: ...

    @property
    @abstractmethod
    def tokens_per_block(self) -> int: ...

    @abstractmethod
    def _global_cached_token_count(self, req: LlmRequest) -> int:
        """Block-aligned cached prefix length reported by the cache manager."""

    def get_cached_token_count_per_layer_group(
        self,
        req: LlmRequest,
        layer_groups: Sequence[AttentionLayerGroup],
    ) -> List[int]:
        """Per-layer-group cached prefix in tokens (block-aligned).

        Returns the reuse-hit prefix only; SWA stale-region handling lives at
        the transfer call site (it is a transport concern, not a cache one).
        """
        if not self.enable_block_reuse:
            return [0] * len(layer_groups)
        scalar = max(0, self._global_cached_token_count(req))
        return [scalar] * len(layer_groups)

    @abstractmethod
    def get_block_ids(
        self,
        req: LlmRequest,
        group_idx: int,
        lg: AttentionLayerGroup,
    ) -> np.ndarray:
        """Per-layer-group block identifiers for *req* (dtype ``int64``).

        Returned values are **primary memory-pool slot indices**, not raw block IDs:
        ``KVRegionExtractorV1.extract`` and downstream transfer code do
        ``base_ptr + slot_idx * slot_bytes`` and require the value to be a current
        primary-pool offset. V2 reports these pool slot indices directly.
        """

    @abstractmethod
    def get_block_ordinals(
        self,
        req: LlmRequest,
        group_idx: int,
        lg: AttentionLayerGroup,
    ) -> np.ndarray:
        """Positional block table for *req* (dtype ``int64``, beam 0).

        Entry ``i`` is the primary-pool slot for tokens ``[i * tpb, ...)``, or
        ``-1`` where the block is out-of-window (SWA-evicted) or unbound. Unlike
        :meth:`get_block_ids`, position is carried by the index rather than by
        the list length, so the caller can select a token range by slicing
        instead of counting. Block lists carry beam 0 only, so this is the
        positional table for every beam width.
        """

    @abstractmethod
    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        """Commit KV blocks to radix tree for future prefix reuse.

        Must be called after ``req.context_current_position = req.prompt_len``.
        """


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

    def _global_cached_token_count(self, req: LlmRequest) -> int:
        if not self.enable_block_reuse:
            return 0
        kv_cache = self._mgr.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return 0
        tpb = self.tokens_per_block
        return (kv_cache.num_committed_tokens // tpb) * tpb

    def get_cached_token_count_per_layer_group(
        self,
        req: LlmRequest,
        layer_groups: Sequence[AttentionLayerGroup],
    ) -> List[int]:
        """Skip only groups actually restored by admission and normal resume.

        V2 transfer groups are indexed by the native layer-group ID. OPTIONAL
        groups not selected at resume have incomplete status; PRIVATE groups
        never supply a reusable prefix, even when REQUIRED groups hit.
        """
        scalar = self._global_cached_token_count(req)
        if scalar == 0:
            return [0] * len(layer_groups)
        statuses = self._mgr.kv_cache_map[req.py_request_id].reuse_status
        tpb = self.tokens_per_block
        return [
            min(scalar, statuses[group].endpoint // tpb * tpb) if statuses[group].complete else 0
            for group in range(len(layer_groups))
        ]

    def get_block_ids(self, req, group_idx, lg):  # noqa: ARG002
        # V2 already returns per-cache-level pool slot indices (not logical block
        # IDs), and active sequences GPU-lock their pages (_UniqPageLock enforces
        # cache_level==GPU), so the slot_ids yielded here are already the right
        # offsets for primary-pool pointer arithmetic. No translation is needed.
        if getattr(lg, "cp_as_tp", False):
            ordinals = self.get_block_ordinals(req, group_idx, lg)
            return ordinals[ordinals >= 0]
        return np.fromiter(
            self._mgr.kv_cache_map[req.py_request_id].get_aggregated_page_indices(
                group_idx, valid_only=True
            ),
            dtype=np.int64,
        )

    def get_block_ordinals(self, req, group_idx, lg):  # noqa: ARG002
        # valid_only=False yields one entry per block ordinal, with -1
        # (BAD_PAGE_INDEX) for out-of-window (SWA-evicted) and unbound blocks.
        # Position is the index; no length arithmetic needed.
        ordinals = np.fromiter(
            self._mgr.kv_cache_map[req.py_request_id].get_aggregated_page_indices(
                group_idx, valid_only=False
            ),
            dtype=np.int64,
        )
        if lg.cp_as_tp:
            converter = self._mgr.impl.get_page_index_converter(
                lg.local_layers[0].local_layer_id, Role.KEY
            )
            return np.asarray(
                converter(ordinals.tolist(), index_mode=PageIndexMode.SHARED), dtype=np.int64
            )
        return ordinals

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        self._mgr.try_commit_blocks(req)


def create_cache_reuse_adapter(
    mgr: KVCacheManagerV2,
) -> CacheReuseAdapter:
    """Create a prefix-reuse adapter for a V2 cache manager."""
    if not isinstance(mgr, KVCacheManagerV2):
        raise ValueError("Python KV cache transfer requires KVCacheManagerV2; V1 is unsupported.")
    return _CacheReuseAdapterV2(mgr)

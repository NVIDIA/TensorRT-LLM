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
from typing import List, Sequence, Union

import numpy as np

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

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
        primary-pool offset. With host offload enabled, a block's logical ID can
        diverge from its primary slot index after offload/onboard, so each backend
        must translate before returning.
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

    def _global_cached_token_count(self, req: LlmRequest) -> int:
        if not self.enable_block_reuse:
            return 0
        tpb = self.tokens_per_block
        return (req.prepopulated_prompt_len // tpb) * tpb

    def get_block_ids(self, req, group_idx, lg):  # noqa: ARG002
        first_layer = get_global_layer_ids(lg)[0]
        raw_ids = self._mgr.get_batch_cache_indices([req.py_request_id], layer_idx=first_layer)[0]
        if not raw_ids:
            return np.array([], dtype=np.int64)
        # block_id != primary-pool slot index once host offload kicks in; translate
        # so the cache transceiver's pointer arithmetic is correct. The manager aborts
        # if any referenced block is currently offloaded — disagg transfer cannot read
        # from the secondary pool, and a held block can never be offloaded.
        window_size = lg.sliding_window_size
        # V1 layer groups carry the manager's window key (full-attention layers get the
        # max window), so this is always set; see kv_extractor.build_page_table.
        assert window_size is not None
        pool_indices = self._mgr.get_memory_pool_block_indices(
            list(raw_ids), window_size=window_size
        )
        return np.asarray(pool_indices, dtype=np.int64)

    def _translate_chain(self, req, chain, window_size) -> np.ndarray:
        """Positional pool slots for one beam chain, SWA-stale front masked to -1.

        V1 keeps the whole pre-eviction chain in order (index == ordinal), and
        only detaches front blocks during generation (``adjustBlocksIfNeeded``),
        so ``get_num_front_blocks_removed`` can still read 0 here. Mask the
        larger of that physical count and the logically stale count derived
        from window/prompt length, then translate only what's left -- evicted
        ids are back in the free pool and may be offloaded, which the
        primary-pool translation rejects.
        """
        ordinals = np.full(len(chain), -1, dtype=np.int64)
        if not chain:
            return ordinals
        tpb = self.tokens_per_block
        physically_removed = self._mgr.get_num_front_blocks_removed(
            req.py_request_id, window_size=window_size
        )
        logically_stale = max(0, (req.prompt_len + 1 - window_size) // tpb)
        stale = min(max(physically_removed, logically_stale), len(chain))
        live = list(chain[stale:])
        if live:
            ordinals[stale:] = self._mgr.get_memory_pool_block_indices(
                live, window_size=window_size
            )
        return ordinals

    def get_block_ordinals(self, req, group_idx, lg):  # noqa: ARG002
        window_size = lg.sliding_window_size
        # V1 layer groups carry the manager's window key (full-attention layers get the
        # max window), so this is always set; see kv_extractor.build_page_table.
        assert window_size is not None
        # Block lists carry beam 0 only (beam-search attention reads prompt
        # positions through beam 0's block table), so take the first chain.
        beams = self._mgr.impl.get_batch_cache_block_ids([req.py_request_id], window_size)[0]
        if not beams:
            return np.array([], dtype=np.int64)
        return self._translate_chain(req, list(beams[0]), window_size)

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

    def _global_cached_token_count(self, req: LlmRequest) -> int:
        if not self.enable_block_reuse:
            return 0
        kv_cache = self._mgr.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return 0
        tpb = self.tokens_per_block
        return (kv_cache.num_committed_tokens // tpb) * tpb

    def get_block_ids(self, req, group_idx, lg):  # noqa: ARG002
        # V2 already returns per-cache-level pool slot indices (not logical block
        # IDs), and active sequences GPU-lock their pages (_UniqPageLock enforces
        # cache_level==GPU), so the slot_ids yielded here are already the right
        # offsets for primary-pool pointer arithmetic. No translation is needed,
        # unlike V1 (see _CacheReuseAdapterV1.get_block_ids).
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
        return np.fromiter(
            self._mgr.kv_cache_map[req.py_request_id].get_aggregated_page_indices(
                group_idx, valid_only=False
            ),
            dtype=np.int64,
        )

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        self._mgr.try_commit_blocks(req)


def create_cache_reuse_adapter(
    mgr: Union[KVCacheManager, KVCacheManagerV2],
) -> CacheReuseAdapter:
    """Factory — pick the right adapter for the concrete manager type."""
    if isinstance(mgr, KVCacheManagerV2):
        return _CacheReuseAdapterV2(mgr)
    return _CacheReuseAdapterV1(mgr)

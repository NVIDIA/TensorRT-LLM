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

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

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
    def get_block_ids_range(
        self,
        req: LlmRequest,
        group_idx: int,
        lg: AttentionLayerGroup,
        block_begin: int,
        block_end: int,
    ) -> np.ndarray:
        """Resident block IDs for absolute request ordinals ``[block_begin, block_end)``.

        The result is the *contiguous* run of resident blocks ending at
        ``block_end``, i.e. ordinals ``[block_end - len(result), block_end)``.
        It can be shorter than the requested range when leading blocks have been
        evicted, as under sliding-window attention; blocks at or before any gap
        are dropped rather than compacted.

        Callers must not reorder or reinterpret the result positionally beyond
        that rule: the receiver reconstructs each layer group's starting token
        from ``block_end`` and the result length, so a result that is not a
        contiguous run ending at ``block_end`` would be written to the wrong
        offsets.

        Empty ranges are valid. Negative or reversed bounds raise
        ``ValueError``. A ``block_end`` past the request's allocated blocks also
        raises, though the exception type is backend-specific (``ValueError``
        from V2, ``RuntimeError`` out of the C++ manager for V1).
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
        beam_width = req.py_beam_width
        raw_ids = self._mgr.get_batch_cache_indices(
            [req.py_request_id], layer_idx=first_layer, beam_width=beam_width
        )[0]
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

    def get_block_ids_range(self, req, group_idx, lg, block_begin, block_end):  # noqa: ARG002
        window_size = lg.sliding_window_size
        # V1 layer groups always carry the manager's window key; see get_block_ids.
        assert window_size is not None
        raw_ids = self._mgr.get_cache_indices_range(
            req.py_request_id,
            block_begin=block_begin,
            block_end=block_end,
            window_size=window_size,
        )
        if not raw_ids:
            return np.array([], dtype=np.int64)
        # Same block_id -> primary-pool slot translation get_block_ids does; the
        # two diverge once host offload is enabled.
        pool_indices = self._mgr.get_memory_pool_block_indices(raw_ids, window_size=window_size)
        return np.asarray(pool_indices, dtype=np.int64)

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

    def get_block_ids_range(self, req, group_idx, lg, block_begin, block_end):  # noqa: ARG002
        if block_begin < 0 or block_end < 0:
            raise ValueError("block range bounds must be non-negative")
        if block_begin > block_end:
            raise ValueError("block_begin must not exceed block_end")
        # Neither V2 backend has a bounded query, so read the whole aggregated
        # list -- keeping the placeholders, which is what makes an entry's index
        # its block ordinal -- and cut the range out of it here.
        all_block_ids = np.fromiter(
            self._mgr.kv_cache_map[req.py_request_id].get_aggregated_page_indices(
                group_idx, valid_only=False
            ),
            dtype=np.int64,
        )
        if block_end > all_block_ids.size:
            raise ValueError(
                f"block_end={block_end} exceeds the {all_block_ids.size} allocated blocks; "
                "the result would not end at block_end, and callers recover block "
                "ordinals from its length"
            )
        block_ids = all_block_ids[block_begin:block_end]
        # Drop everything up to the last gap rather than compacting around it:
        # a life cycle that keeps sink tokens resident has a sink prefix plus a
        # window suffix, and returning both would misreport where the suffix
        # starts.
        gaps = np.flatnonzero(block_ids == BAD_PAGE_INDEX)
        return block_ids[gaps[-1] + 1 :] if gaps.size else block_ids

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        self._mgr.try_commit_blocks(req)


def create_cache_reuse_adapter(
    mgr: Union[KVCacheManager, KVCacheManagerV2],
) -> CacheReuseAdapter:
    """Factory — pick the right adapter for the concrete manager type."""
    if isinstance(mgr, KVCacheManagerV2):
        return _CacheReuseAdapterV2(mgr)
    return _CacheReuseAdapterV1(mgr)

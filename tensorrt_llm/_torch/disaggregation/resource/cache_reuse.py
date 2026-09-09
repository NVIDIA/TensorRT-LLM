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
        """Positional block table for *req* (dtype ``int64``, single beam only).

        Entry ``i`` is the primary-pool slot for tokens ``[i * tpb, ...)``, or
        ``-1`` where the block is out-of-window (SWA-evicted) or unbound. Unlike
        :meth:`get_block_ids`, position is carried by the index rather than by
        the list length, so the caller can select a token range by slicing
        instead of counting. Only meaningful for ``beam_width == 1`` (the packed
        multi-beam layout is not positional).
        """

    def get_beam0_ordinals_and_tails(
        self,
        req: LlmRequest,
        group_idx: int,
        lg: AttentionLayerGroup,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Positional beam-0 ordinals plus the packed divergent beam tails.

        ``beam0_ordinals`` follows the :meth:`get_block_ordinals` contract
        (index==ordinal, ``-1`` for stale/unbound), so the caller selects beam-0's
        window by slicing -- the same positional path as ``beam_width == 1``.
        ``tails`` are the per-beam final blocks that differ from beam-0's, carried
        verbatim after the window (they are not positional). Only V1 supports beam
        search over disagg; V2 raises.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support beam_width > 1 disagg transfer"
        )

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

    def get_block_ordinals(self, req, group_idx, lg):
        # V1 keeps the whole pre-eviction chain in order (removeFrontBlock only
        # bumps a counter, never drops the id), so the translated list is already
        # positional. SWA-evicted front blocks are still real ids here; mask them
        # to -1 using the manager's authoritative front-removed count so the
        # positional contract matches V2 (which reports holes directly).
        ids = self.get_block_ids(req, group_idx, lg)
        if ids.size == 0:
            return ids
        window_size = lg.sliding_window_size
        assert window_size is not None
        stale = self._mgr.get_num_front_blocks_removed(req.py_request_id, window_size=window_size)
        if stale > 0:
            ids = ids.copy()
            ids[: min(stale, ids.size)] = -1
        return ids

    def get_beam0_ordinals_and_tails(self, req, group_idx, lg):  # noqa: ARG002
        window_size = lg.sliding_window_size
        assert window_size is not None
        # Raw per-beam chains (before _pack_beam_cache_indices flattens them).
        # result[0] is this request's list of beam chains; beam 0 owns the shared
        # prefix, the others contribute only their final block.
        beams = self._mgr.impl.get_batch_cache_block_ids([req.py_request_id], window_size)[0]
        if not beams or not beams[0]:
            empty = np.array([], dtype=np.int64)
            return empty, empty
        beam0 = list(beams[0])
        beam0_last = beam0[-1]
        tail_ids = [beam[-1] for beam in beams[1:] if beam and beam[-1] != beam0_last]
        # Translate logical block ids to primary-pool slots (see get_block_ids).
        beam0_pool = np.asarray(
            self._mgr.get_memory_pool_block_indices(beam0, window_size=window_size),
            dtype=np.int64,
        )
        # Mask the SWA-evicted front to -1 (authoritative count), matching the
        # get_block_ordinals contract.
        stale = self._mgr.get_num_front_blocks_removed(req.py_request_id, window_size=window_size)
        if stale > 0:
            beam0_pool[: min(stale, beam0_pool.size)] = -1
        tails = (
            np.asarray(
                self._mgr.get_memory_pool_block_indices(tail_ids, window_size=window_size),
                dtype=np.int64,
            )
            if tail_ids
            else np.array([], dtype=np.int64)
        )
        return beam0_pool, tails

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

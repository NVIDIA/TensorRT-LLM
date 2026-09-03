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


def split_packed_beam_block_ids(
    block_ids: np.ndarray,
    beam_width: int,
    beam0_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split 1-D block IDs into beam-0 prefix and appended beam-tail blocks.

    ``beam0_len`` is the *known* beam-0 span length — with explicit anchors it
    is span_end − first_ordinal (e.g. prompt_blocks − first_ordinal for a
    whole-prompt span). The packer appends only the UNSHARED final block of
    each non-zero beam (``resource_manager._pack_beam_cache_indices``), so a
    list carries up to ``beam_width − 1`` tails but possibly fewer; the tail
    count must therefore be derived as ``size − beam0_len``, never guessed
    from ``beam_width``. ``beam_width <= 1`` lists never carry tails.
    """
    if beam_width <= 1 or block_ids.size <= beam0_len:
        return block_ids, block_ids[:0]
    return block_ids[:beam0_len], block_ids[beam0_len:]


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
    def get_transfer_span(
        self,
        req: LlmRequest,
        group_idx: int,
        lg: AttentionLayerGroup,
    ) -> tuple[np.ndarray, int]:
        """Anchored transferable block span for one attention layer group.

        Returns ``(pages, first_block_ordinal)``: ``pages`` is the beam-0 block
        list (dtype ``int64``) with manager-evicted/stale prefixes and
        speculative scratch handled, followed by packed beam-tail blocks when
        ``beam_width > 1``; ``first_block_ordinal`` is the 0-based sequence
        block ordinal of ``pages[0]`` (beam tails stay outside the anchored
        region). Empty spans use anchor 0. The anchor reflects manager facts
        only; transfer-policy trims (SWA bandwidth trim, gen-side reuse skip)
        belong to the caller, which must advance the anchor with any head-slice.

        Returned values are **primary memory-pool slot indices**, not raw block
        IDs: ``KVRegionExtractorV1.extract`` and downstream transfer code do
        ``base_ptr + slot_idx * slot_bytes`` and require the value to be a
        current primary-pool offset. With host offload enabled, a block's
        logical ID can diverge from its primary slot index after
        offload/onboard, so each backend must translate before returning.
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

    def _to_pool_indices(self, block_ids: np.ndarray, window_size: int) -> np.ndarray:
        """Translate logical block IDs to primary-pool slot indices.

        block_id != primary-pool slot index once host offload kicks in; translate
        so the cache transceiver's pointer arithmetic is correct. The manager aborts
        if any referenced block is currently offloaded — disagg transfer cannot read
        from the secondary pool, and a held block can never be offloaded.
        """
        if block_ids.size == 0:
            return np.array([], dtype=np.int64)
        pool_indices = self._mgr.get_memory_pool_block_indices(
            block_ids.tolist(), window_size=window_size
        )
        return np.asarray(pool_indices, dtype=np.int64)

    def get_transfer_span(self, req, group_idx, lg):  # noqa: ARG002
        window_size = lg.sliding_window_size
        # V1 layer groups carry the manager's window key (full-attention layers get the
        # max window), so this is always set; see kv_extractor.build_page_table.
        assert window_size is not None
        first_layer = get_global_layer_ids(lg)[0]
        beam_width = req.py_beam_width
        raw_ids = self._mgr.get_batch_cache_indices(
            [req.py_request_id], layer_idx=first_layer, beam_width=beam_width
        )[0]
        if not raw_ids:
            return np.array([], dtype=np.int64), 0
        block_ids = np.asarray(raw_ids, dtype=np.int64)
        if self._mgr.mapping.cp_size > 1:
            # Helix CP: the list holds this rank's strided local blocks, so
            # ordinals are local and scratch/eviction bookkeeping (which is
            # global) does not apply.
            return self._to_pool_indices(block_ids, window_size), 0

        tpb = self.tokens_per_block
        prompt_blocks = (req.prompt_len + tpb - 1) // tpb
        allocated_blocks = (req.prompt_len + self._mgr.num_extra_kv_tokens + tpb - 1) // tpb
        beam0, tails = split_packed_beam_block_ids(block_ids, beam_width, allocated_blocks)
        # Draft-token allocation can extend past the speculative scratch bound.
        if beam0.size > allocated_blocks:
            beam0 = beam0[:allocated_blocks]
        # Only prompt KV is transferred; drop the speculative scratch tail.
        scratch_blocks = allocated_blocks - prompt_blocks
        if scratch_blocks > 0:
            if beam_width != 1:
                raise ValueError("speculative scratch blocks require beam_width == 1")
            beam0 = beam0[:-scratch_blocks] if scratch_blocks < beam0.size else beam0[:0]

        # The anchor reflects what the manager actually evicted. Detached front
        # blocks remain in the C++ cache-block-id list (detachFrontBlock only
        # advances a counter), so the leading `anchor` entries are dangling ids
        # and must be dropped before pool translation — a detached block may
        # have been reused or offloaded by now.
        anchor = self._mgr.get_num_front_blocks_removed(req.py_request_id, window_size)
        if anchor > 0:
            # detachFrontBlock asserts beam_width == 1 in C++.
            assert beam_width == 1, "front-block eviction requires beam_width == 1"
            beam0 = beam0[anchor:] if anchor < beam0.size else beam0[:0]

        expected = max(0, prompt_blocks - anchor)
        if beam0.size != expected:
            raise RuntimeError(
                f"request {req.py_request_id} window={window_size}: beam-0 block list "
                f"holds {beam0.size} blocks after stripping {anchor} evicted front "
                f"block(s) and {max(0, scratch_blocks)} scratch tail block(s), expected "
                f"{expected} (= ceil(prompt_len={req.prompt_len} / tokens_per_block="
                f"{tpb}) - {anchor}); refusing to transfer misaligned KV blocks"
            )
        if beam0.size == 0:
            return np.array([], dtype=np.int64), 0
        pages = np.concatenate([beam0, tails]) if tails.size > 0 else beam0
        return self._to_pool_indices(pages, window_size), anchor

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

    def get_transfer_span(self, req, group_idx, lg):
        kv_cache = self._mgr.kv_cache_map[req.py_request_id]
        # V2 already returns per-cache-level pool slot indices (not logical block
        # IDs), and active sequences GPU-lock their pages (_UniqPageLock enforces
        # cache_level==GPU), so the slot_ids yielded here are already the right
        # offsets for primary-pool pointer arithmetic. No translation is needed,
        # unlike V1 (see _CacheReuseAdapterV1._to_pool_indices).
        #
        # valid_only=False keeps one entry per block ordinal (BAD_PAGE_INDEX for
        # unbacked ordinals), so index == ordinal and the anchor can be derived
        # instead of guessed from the list length.
        pages = np.fromiter(
            kv_cache.get_aggregated_page_indices(group_idx, valid_only=False),
            dtype=np.int64,
        )
        if self._mgr.mapping.cp_size > 1:
            # Helix CP: the list holds this rank's strided local blocks, so the
            # global ordinal math below (prompt cap, stale/scratch ranges) does
            # not apply. Mirrors _CacheReuseAdapterV1's helix pass-through.
            if bool((pages == BAD_PAGE_INDEX).any()):
                raise RuntimeError(
                    f"request {req.py_request_id} layer group {group_idx}: unbacked "
                    "block ordinals under helix CP cannot be anchored (local block "
                    "lists have no global stale/scratch bookkeeping)"
                )
            return pages, 0
        tpb = self.tokens_per_block
        prompt_blocks = (req.prompt_len + tpb - 1) // tpb
        # Blocks past the prompt (speculative scratch tail) are never transferred.
        pages = pages[: min(pages.size, prompt_blocks)]
        invalid = np.flatnonzero(pages == BAD_PAGE_INDEX)
        # The transferable span is the contiguous backed run ending at the last
        # prompt block; ordinals at or before an unbacked one hold no readable KV
        # (stale-unlocked, or written to rotating shared scratch slots).
        anchor = int(invalid[-1]) + 1 if invalid.size > 0 else 0
        span = pages[anchor:]
        if span.size == 0:
            return np.array([], dtype=np.int64), 0
        if invalid.size > 0:
            # A single anchor cannot represent a backed sink prefix followed by
            # an unbacked hole — it would silently drop the sink blocks, which
            # are required KV (unlike a held-for-commit stale prefix, which is
            # droppable). PyExecutor never enables sinks (its only
            # AttentionLayerConfig construction passes num_sink_tokens=None),
            # so fail loud if a sink-configured life cycle ever gets here.
            # Best-effort guard: only the Python reference backend exposes
            # _life_cycles; the default C++ backend
            # (TLLM_KV_CACHE_MANAGER_V2_BACKEND=cpp) has no public life-cycle
            # accessor, so the check is skipped there.
            life_cycles = getattr(kv_cache.manager, "_life_cycles", None)
            if life_cycles is not None:
                life_cycle = life_cycles[group_idx]
                if getattr(life_cycle, "num_sink_blocks", 0):
                    raise RuntimeError(
                        f"request {req.py_request_id} layer group {group_idx}: KV "
                        "transfer of anchored spans does not support token sinks "
                        f"(num_sink_blocks={life_cycle.num_sink_blocks})"
                    )
            # Consistency check against manager truths: every unbacked ordinal
            # must be explained by SWA staleness or scratch placement.
            window_size = lg.sliding_window_size
            stale_end = (
                max(0, (kv_cache.history_length + 1 - window_size) // tpb)
                if window_size is not None
                else 0
            )
            explained = invalid < stale_end
            scratch = kv_cache.get_scratch_desc(group_idx)
            if scratch is not None:
                beg, end = scratch.range
                explained |= (invalid >= beg) & (invalid < end)
            if not bool(explained.all()):
                raise RuntimeError(
                    f"request {req.py_request_id} layer group {group_idx}: unbacked "
                    f"block ordinals {invalid[~explained].tolist()} are neither SWA-"
                    f"stale (stale_end={stale_end}) nor scratch "
                    f"({scratch.range if scratch is not None else None}); refusing "
                    "to transfer misaligned KV blocks"
                )
        return span, anchor

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        self._mgr.try_commit_blocks(req)


def create_cache_reuse_adapter(
    mgr: Union[KVCacheManager, KVCacheManagerV2],
) -> CacheReuseAdapter:
    """Factory — pick the right adapter for the concrete manager type."""
    if isinstance(mgr, KVCacheManagerV2):
        return _CacheReuseAdapterV2(mgr)
    return _CacheReuseAdapterV1(mgr)

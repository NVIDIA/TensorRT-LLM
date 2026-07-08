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

import array
import enum
import math
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterator, NamedTuple, Type, cast

from .. import rawref
from .._block_radix_tree import Block, ReuseMatch, ReuseScope, RootBlock, UselessBlockError
from .._common import (
    BAD_BLOCK_ORDINAL,
    BAD_PAGE_INDEX,
    DEFAULT_BEAM_INDEX,
    GPU_LEVEL,
    NDEBUG,
    BeamIndex,
    BlockOrdinal,
    BlockOrdinalT,
    CacheLevel,
    CudaStream,
    PageIndex,
    PageIndexMode,
    Priority,
    TokenIdExt,
)
from .._config import WriteThroughPolicy
from .._copy_engine import CopyTask, batched_copy
from .._exceptions import LogicError, OutOfPagesError
from .._life_cycle_registry import (
    AttnLifeCycle,
    LayerGroupId,
    LifeCycle,
    LifeCycleId,
    SsmLifeCycle,
    compute_scratch_range,
)
from .._page import (
    BatchedLockTarget,
    BlockPage,
    CommittedPage,
    Page,
    PrivateCommittedPage,
    ScratchSlotLock,
    UncommittedPage,
    _PageHolder,
    _SharedPageLock,
    batched_lock_to_gpu,
)
from .._stats import KVCacheIterationStatsDelta, KVCacheStatsDelta
from .._storage._core import PoolGroupIndex, SequenceRange, Slot, SlotId
from .._storage_manager import StorageManager
from .._utils import (
    CachedCudaEvent,
    HalfOpenRange,
    TypedIndexList,
    div_up,
    expect_type,
    filled_list,
    intersect,
    make_typed,
    map_optional,
    stream_wait_events,
    to_typed,
    typed_enumerate,
    typed_len,
    typed_map,
    typed_range,
    unwrap_optional,
    unwrap_rawref,
    value_or,
)
from ._moving_average import Average
from ._pending_stats import _PendingStats

if TYPE_CHECKING:
    from ._kv_cache_manager import KVCacheManager, ScratchDesc


@dataclass(slots=True)
class SeqBlock:
    pages: TypedIndexList[BeamIndex, TypedIndexList[LifeCycleId, BlockPage]]
    # In rare cases, this may be the only strong reference to this block. Assume it's the last block we
    # committed on stop_committing(), and it's partial. At the same time, we have another _KVCache
    # generating same tokens plus some additional tokens. The block committed by the other _KVCache will
    # fully cover tokens of this block. In that case, we will remove this block from the radix tree.
    # Which means `tree_block not in tree_block.prev.next` will be True.
    tree_block: Block | None

    @property
    def is_committed(self) -> bool:
        ret = self.tree_block is not None
        assert NDEBUG or not ret or len(self.pages) == 1
        assert (
            NDEBUG
            or not ret
            or all(
                p is None or isinstance(p.page, CommittedPage)
                for p in chain.from_iterable(self.pages)
            )
        )
        assert (
            NDEBUG
            or ret
            or all(
                p is None or isinstance(p.page, UncommittedPage)
                for p in chain.from_iterable(self.pages)
            )
        )
        return ret

    def __del__(self) -> None:
        self.tree_block = None
        self.pages.clear()


class _Status(enum.Enum):
    ACTIVE = enum.auto()
    SUSPENDED = enum.auto()
    CLOSED = enum.auto()


class _CommitState(enum.Enum):
    ALLOWED = enum.auto()
    # user did not stop but we can't commit any more due to conflict with other blocks
    VIRTUAL_STOP = enum.auto()
    # user called stop_committing() or close()
    USER_STOP = enum.auto()


IndexSeq = array.array | memoryview


# The _KVCache holds unique/shared ownership of memory blocks. On deletion, the ownership if destroys
# and KVCacheManager takes control of them. A KV cache maintains three lengths:
#  1.	num_committed_tokens: the number of tokens that are finalized, immutable and ready for reuse.
#  2.	history_length: a cursor separating history and the space for next input tokens. History tokens
#     are defined as tokens without query data for the next inference step. For SWA layers, it decides
#     which blocks are out-of-window and can be evicted/dropped. In most cases, you don't need to touch
#     history_length as it's automatically bumped by the increase of num_committed_tokens, except a few
#     cases:
#     a.	Beam search where we can't commit tokens generated by the last step. But it still makes sense
#         to evict uncommitted pages for SWA layers to save memory.
#     b.	Disaggregated serving with SWA and the reusable tokens are in the other server. We need to
#         reserve space for history. Knowing history_length helps us accurately decide which blocks
#         needs to be allocated. Then users only transfer data for what is needed.
#     c.	Multi-round conversation with chain of thoughts (CoT) and excluding CoT tokens for the next
#         round. In this case, users should not commit tokens starting from CoT. Then history_length
#         needs to be explicitly bumped.
#  3.	capacity: the number of tokens that can be stored in the KV cache. It should include the number
#     of both historical tokens and input tokens for the next inference step, no matter if it's prefill,
#     chunked prefill or generation without/without speculative decoding. For tree-based speculative
#     decoding, the number of input tokens here should be the flatten draft length. For beam search,
#     multiple candidate tokens at the same position are counted as one.
# num_committed_tokens <= history_length <= capacity always holds. A newly created KV cache has all
# three lengths equal to the number of reused tokens.
# TODO: in __del__, we should check if committed pages are usable for SWA cases. e.g. all pages are
# dropped except the last one. The last one is not usable.
class _ArenaClosingPages(NamedTuple):
    """Partition of a sequence's committed GPU pages at close() (arena mode).

    A module-level class: mypyc cannot compile class definitions nested in a
    class body."""

    # canonical committed pages without a write-through copy: copy-on-free
    offload: "TypedIndexList[PoolGroupIndex, list[CommittedPage]]"
    offload_ordinals: "TypedIndexList[PoolGroupIndex, list[int]]"
    # canonical committed pages with a valid write-through host copy, and
    # those copies: adopted copy-free (§4.3)
    adopt_pages: "TypedIndexList[PoolGroupIndex, list[CommittedPage]]"
    adopt_slots: "TypedIndexList[PoolGroupIndex, list[Slot]]"
    adopt_ordinals: "TypedIndexList[PoolGroupIndex, list[int]]"
    # sequence-private reuse copies (§4.4): dropped -- but under lazy
    # retention their bytes re-register as fresh ghosts for the canonical
    # entry, keeping hot prefixes' D2D sources LRU-recent
    private: list[CommittedPage]
    private_meta: list[tuple[int, LifeCycleId]]  # (ordinal, lc) per entry


class _KVCache:
    __slots__ = (
        "id",
        "_manager",
        "_reuse_scope",
        "_get_priority",
        "_cuda_stream",
        "_status",
        "_beam_width",
        "_expected_prompt_length",
        "_generation_alloc_ready",
        "_capacity",
        "_history_length",
        "_commit_state",
        "_blocks",
        "_base_page_indices",
        "_committed_tokens",
        "_num_committed_blocks",
        "_finish_event",
        "_tokens_per_block",
        "_avg_history_length",
        "_avg_capacity",
        "_ssm_blocks",
        "_never_resumed",
        "_enable_swa_scratch_reuse",
        "_scratch_slots",
        "_pending_stats",
        "_max_capacity",
        "_arena_ranges",
        "_write_through_slots",
        "_alias_spans",
        "__rawref__",
    )

    Status: ClassVar[Type[_Status]] = _Status
    CommitState: ClassVar[Type[_CommitState]] = _CommitState

    id: int | None
    _manager: "KVCacheManager"
    _reuse_scope: ReuseScope
    _get_priority: Callable[[BlockOrdinal, LifeCycle], Priority]
    _cuda_stream: CudaStream | None
    _status: _Status
    _beam_width: BeamIndex
    _expected_prompt_length: int | None
    _generation_alloc_ready: bool
    _capacity: int
    _history_length: int
    _commit_state: _CommitState

    _blocks: TypedIndexList[BlockOrdinal, SeqBlock]
    # we maintain _base_page_indices to accelerate the get_base_page_indices() API. In principle it can
    # be computed on the fly, but that would be slow due to python.
    _base_page_indices: TypedIndexList[BeamIndex, TypedIndexList[LifeCycleId, IndexSeq]]
    _committed_tokens: list[TokenIdExt]
    # Sometimes we can't commit a block because all its tokens are already covered by another block in
    # the radix tree. But it's unsafe to just use the other block because: 1. the data may have numeric
    # difference, 2. if our block is a partial block, we can't write to memory of the other blocks.
    # Internally, we stop committing from such a block, but still give user an illusion that the block is
    # committed. In such cases, _committed_tokens contains what users have fed with commit(), while
    # _num_committed_blocks contains the number of blocks that are actually committed.
    _num_committed_blocks: BlockOrdinal
    # set when switch away from ACTIVE, cleared when switching to ACTIVE.
    _finish_event: CachedCudaEvent | None

    _tokens_per_block: int
    _avg_history_length: Average
    _avg_capacity: Average

    _ssm_blocks: TypedIndexList[BeamIndex, TypedIndexList[LifeCycleId, BlockPage]]
    _never_resumed: bool
    _enable_swa_scratch_reuse: bool
    # Scratch slots for SWA prefill memory reuse, per life cycle. These hold coalesced slots
    # whose sub-pages are reinterpreted as per-block storage for the currently executing layer.
    # Number of scratch blocks depends on diff between history_length and capacity.
    # Managed via delta in resize(): existing slots are reused across resize calls,
    # only the additional needed slots are allocated. Freed on teardown/suspend.
    _scratch_slots: TypedIndexList[LifeCycleId, list[ScratchSlotLock]]
    _pending_stats: _PendingStats
    # Contiguous-arena mode (DESIGN.md §4.1): the request's capacity ceiling in
    # tokens (sizes the VA reservation) and, while active, one contiguous
    # block-index range per pool group. None outside arena mode / while no
    # ranges are held.
    _max_capacity: int | None
    _arena_ranges: TypedIndexList[PoolGroupIndex, SequenceRange] | None
    # Write-through host copies (§4.3, WriteThroughPolicy.ON_COMMIT), keyed by
    # (ordinal, life cycle). A slot's ready event completes when its copy is
    # valid; close()/suspend() adopt these instead of copying.
    _write_through_slots: dict[tuple[int, int], Slot]
    # P3 prefix aliasing: per pool group, the canonical-span registry hit for
    # this admission's reuse match -- (key, span) or None. Set at
    # _setup_for_reuse, consumed (and cleared) by the FIRST resume's
    # reserve+onboard; never set on suspend/resume round-trips.
    _alias_spans: "list[tuple[int, Any] | None] | None"

    def __init__(
        self,
        manager: "KVCacheManager",
        reuse_scope: ReuseScope,
        reuse_match: ReuseMatch | None,
        id: int | None,
        custom_priority_callback: Callable[[BlockOrdinal, LifeCycle], Priority],
        expected_prompt_length: int | None = None,
        max_capacity: int | None = None,
    ):
        self.id = id
        self._manager = manager
        self._reuse_scope = reuse_scope
        self._get_priority = custom_priority_callback
        self._cuda_stream = None
        self._status = self.Status.SUSPENDED
        self._beam_width = BeamIndex(1)
        self._expected_prompt_length = (
            max(expected_prompt_length, 0) if expected_prompt_length is not None else None
        )
        self._generation_alloc_ready = False
        self._capacity = 0
        self._history_length = 0
        self._commit_state = self.CommitState.ALLOWED
        self._blocks = cast(TypedIndexList, [])
        self._base_page_indices = make_typed(
            lambda _: make_typed(lambda _: array.array("i"), self.manager._storage.num_life_cycles),
            self.beam_width,
        )
        self._committed_tokens = []
        self._num_committed_blocks = BlockOrdinal(0)
        self._finish_event = None
        self._tokens_per_block = manager.tokens_per_block
        self._ssm_blocks = make_typed(
            lambda _: filled_list(cast(BlockPage, None), manager._storage.num_life_cycles),
            self.beam_width,
        )
        self._never_resumed = True
        self._enable_swa_scratch_reuse = manager.enable_swa_scratch_reuse
        self._scratch_slots = make_typed(
            lambda _: list[ScratchSlotLock](), manager._storage.num_life_cycles
        )
        self._pending_stats = _PendingStats()
        self._max_capacity = max_capacity
        self._arena_ranges = None
        self._write_through_slots = {}
        self._alias_spans = None
        # Arena-mode preconditions are validated by KVCacheManager.create_kv_cache
        # (raising here would leave a partially constructed object for __del__).
        assert not manager._storage.is_arena_mode or (max_capacity is not None and max_capacity > 0)
        self.__rawref__ = rawref.NULL
        if reuse_match is not None:
            self._setup_for_reuse(reuse_match)
        self._refresh_generation_alloc_ready()
        self._avg_history_length = Average()
        self._avg_capacity = Average()
        self._avg_history_length.update(self.history_length)
        manager._living_kv_caches.add(rawref.ref(self))
        manager._avg_reused_length.update(self.history_length)
        manager._num_created_kv_caches += 1
        assert NDEBUG or self._check_sanity()

    def set_base_page_index_buf(
        self, beam_idx: BeamIndex, layer_group_id: LayerGroupId, buf: memoryview | None
    ) -> None:
        """
        Set the buffer for base page indices, so we directly update indices in user buffer to
        avoid user-side copy. This is the zero-copy alternative of get_base_page_indices().

        Note that base page indices are not meant for direct use in the kernels. They need to
        be scaled by kv_cache_manager.page_index_scale().
        """
        length = self.num_blocks
        old_indices = self._base_page_indices[beam_idx][layer_group_id]
        new_indices: IndexSeq
        if buf is None:
            new_indices = array.array("i", old_indices[:length])
        else:
            assert buf.ndim == 1 and buf.format == "i" and len(buf) >= length
            buf[:length] = old_indices[:length]
            buf[length:] = array.array("i", [BAD_PAGE_INDEX]) * (len(buf) - length)
            new_indices = buf
        self._base_page_indices[beam_idx][layer_group_id] = new_indices

    @property
    def manager(self) -> "KVCacheManager":
        return self._manager

    @property
    def cuda_stream(self) -> CudaStream:
        return unwrap_optional(self._cuda_stream)

    @cuda_stream.setter
    def cuda_stream(self, cuda_stream: CudaStream) -> None:
        if self._cuda_stream is not None:
            if self.is_active:
                CachedCudaEvent(self._cuda_stream).wait_in_stream(cuda_stream)
        else:
            assert self.status == self.Status.SUSPENDED and self._finish_event is None
        self._cuda_stream = cuda_stream

    @property
    def finish_event(self) -> CachedCudaEvent:
        "Event recorded when switching from active to suspended/closed state. Unavailable when active."
        return unwrap_optional(self._finish_event)

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    def _should_record_stats(self) -> bool:
        return self.manager._stats_enabled and not self.manager.is_stats_excluded(self.id)

    def commit_pending_stats(self) -> KVCacheStatsDelta:
        if not self._should_record_stats():
            self.discard_pending_stats()
            return KVCacheStatsDelta()
        self.manager.commit_stats(
            self._pending_stats.global_stats, self._pending_stats.iteration_stats_by_life_cycle
        )
        request_stats = self._pending_stats.request_stats.copy()
        self._pending_stats.clear()
        self.manager.clear_stats_dirty(self.id)
        return request_stats

    def discard_pending_stats(self) -> None:
        self._pending_stats.clear()
        self.manager.clear_stats_dirty(self.id)

    def _refresh_stats_dirty_state(self) -> None:
        if not self._pending_stats.empty:
            self.manager.mark_stats_dirty(self.id)
        else:
            self.manager.clear_stats_dirty(self.id)

    def _stats_life_cycle_key(self, life_cycle: LifeCycleId) -> LifeCycleId | None:
        life_cycle_obj = self.manager._life_cycles.get_life_cycle(life_cycle)
        if isinstance(life_cycle_obj, AttnLifeCycle):
            return life_cycle
        return None

    def _refresh_generation_alloc_ready(self) -> None:
        expected_prompt_length = self._expected_prompt_length
        if expected_prompt_length is not None and self._history_length >= expected_prompt_length:
            self._generation_alloc_ready = True

    def _should_record_generation_alloc_stats(self, capacity: int) -> bool:
        return self._generation_alloc_ready and capacity > self._capacity

    @staticmethod
    def _block_ranges_excluding(
        block_begin: BlockOrdinal,
        block_end: BlockOrdinal,
        excluded: HalfOpenRange[BlockOrdinal],
    ) -> Iterator[HalfOpenRange[BlockOrdinal]]:
        first_end = min(block_end, excluded.beg)
        if block_begin < first_end:
            yield HalfOpenRange(block_begin, first_end)
        second_begin = max(block_begin, excluded.end)
        if second_begin < block_end:
            yield HalfOpenRange(second_begin, block_end)

    def _record_resize_pending_allocations(
        self,
        block_begin: BlockOrdinal,
        block_end: BlockOrdinal,
        beam_width: BeamIndex,
        excluded_ranges: TypedIndexList[LifeCycleId, HalfOpenRange[BlockOrdinal]],
        count_as_generation: bool,
    ) -> None:
        if not self._should_record_stats() or block_begin >= block_end:
            return
        # V2 includes generation allocations in per-request alloc_total/new
        # metrics. This intentionally differs from the legacy V1 C++ manager,
        # where addToken() only updates manager-level generation counters.
        changed = False
        for lc_idx, _ in self.manager._life_cycles.attention_life_cycles():
            for block_range in self._block_ranges_excluding(
                block_begin, block_end, excluded_ranges[lc_idx]
            ):
                changed |= self._pending_stats.record_allocation_range(
                    lc_idx,
                    block_range.beg,
                    block_range.end,
                    beam_width=int(beam_width),
                    count_as_missed=not count_as_generation,
                    count_as_generation=count_as_generation,
                )
        if changed:
            self.manager.mark_stats_dirty(self.id)

    @staticmethod
    def _has_reuse_source(page: BlockPage) -> bool:
        if page is None or not isinstance(page.page, CommittedPage):
            return False
        return page.page.block() is not None

    def _subtract_pending_allocation_range(
        self, block_begin: BlockOrdinal, block_end: BlockOrdinal
    ) -> None:
        if self._pending_stats.subtract_allocation_range(block_begin, block_end):
            self._refresh_stats_dirty_state()

    def _record_direct_iteration_stats(
        self, life_cycle: LifeCycleId, iteration_stats: KVCacheIterationStatsDelta
    ) -> None:
        life_cycle_key = self._stats_life_cycle_key(life_cycle)
        if life_cycle_key is None or iteration_stats.empty or not self._should_record_stats():
            return
        self.manager.commit_stats(KVCacheStatsDelta(), {life_cycle_key: iteration_stats})

    def _record_migrated_slots(
        self,
        pages: Sequence[Page],
        slots: Sequence[Slot],
        src_level: CacheLevel,
        dst_level: CacheLevel,
    ) -> None:
        if not self._should_record_stats():
            return
        assert len(pages) == len(slots)
        for page in pages:
            life_cycle_key = self._stats_life_cycle_key(page.life_cycle)
            if life_cycle_key is None:
                continue
            pg_idx = self.manager._storage.get_pool_group_index(page.life_cycle)
            page_size = sum(self.manager._storage.slot_size(pg_idx))
            stats = KVCacheStatsDelta()
            iteration_stats = KVCacheIterationStatsDelta()
            if src_level == GPU_LEVEL and dst_level > GPU_LEVEL:
                iteration_stats.iter_offload_blocks = 1
                iteration_stats.iter_offload_bytes = page_size
            elif dst_level == GPU_LEVEL:
                stats.alloc_total_blocks = 1
                stats.alloc_new_blocks = 1
                iteration_stats.iter_alloc_total_blocks = 1
                iteration_stats.iter_alloc_new_blocks = 1
                if src_level > GPU_LEVEL:
                    iteration_stats.iter_onboard_blocks = 1
                    iteration_stats.iter_onboard_bytes = page_size
                elif src_level == GPU_LEVEL:
                    iteration_stats.iter_intra_device_copy_blocks = 1
                    iteration_stats.iter_intra_device_copy_bytes = page_size
            if not stats.empty or not iteration_stats.empty:
                self.manager.commit_stats(stats, {life_cycle_key: iteration_stats})

    # destroy ownership of memory blocks, so KV cache manager can decide to evict or drop them. After
    # close, uncommitted data in blocks for (beam_index >= beam_width) will be lost.
    def close(self, prior_event: CachedCudaEvent = CachedCudaEvent.NULL) -> None:
        """Close the sequence. ``prior_event`` should cover the sequence's
        last KV writes on the *forward* stream (see
        ``StorageManager.offload_arena_pages``): under the overlap scheduler a
        speculatively enqueued step may still be appending to the tail block
        when the request is freed, and the arena write-out below would
        otherwise copy it stale."""
        assert NDEBUG or self._check_sanity()
        if self.status == self.Status.CLOSED:
            return
        self.discard_pending_stats()
        self.stop_committing()
        assert NDEBUG or self._check_sanity()
        manager = self.manager
        if self.capacity > 0:
            self._avg_capacity.update(self.capacity)
            manager._avg_sqr_capacity.update(self._avg_capacity.value**2)
            manager._avg_sqr_history_length.update(self._avg_history_length.value**2)
            manager._num_sampled_kv_caches += 1
            manager._try_update_target_ratios()
        arena_ranges = self._arena_ranges
        arena_closing: _ArenaClosingPages | None = None
        arena_last_consumer = cast(CachedCudaEvent, CachedCudaEvent.NULL)
        if arena_ranges is not None:
            arena_cfg = manager._storage.arena_config
            if arena_cfg is not None and arena_cfg.batched_map_sweep:
                # The write-out below reads this sequence's pages; make sure
                # any still-deferred growth maps have executed.
                manager._storage.flush_gpu_mappings()
            arena_closing = self._arena_collect_committed_gpu_pages()
        with self._record_event():
            self._clear_blocks()
            # _record_event clears _finish_event on exit; capture it as the
            # range-reclaim gate (§4.2) while it is live.
            finish_event: CachedCudaEvent | None = self._finish_event
            if finish_event is not None:
                arena_last_consumer = finish_event
        if arena_ranges is not None:
            storage = manager._storage
            assert arena_closing is not None
            arena_lazy = (
                storage.arena_config is not None and storage.arena_config.lazy_gpu_retention
            )
            for pg_idx in typed_range(storage.num_pool_groups):
                # Written-through blocks move copy-free (§4.3).
                storage.adopt_stale_copies(
                    pg_idx, arena_closing.adopt_pages[pg_idx], arena_closing.adopt_slots[pg_idx]
                )
                if arena_lazy:
                    # The bytes stay resident in the (soon-retained) range:
                    # register them as D2D reuse sources (§4.4 phase 2).
                    storage.register_arena_ghosts(
                        pg_idx,
                        arena_ranges[pg_idx],
                        list(
                            zip(
                                arena_closing.adopt_ordinals[pg_idx],
                                arena_closing.adopt_pages[pg_idx],
                            )
                        ),
                    )
                try:
                    storage.offload_arena_pages(pg_idx, arena_closing.offload[pg_idx], prior_event)
                    if arena_lazy:
                        storage.register_arena_ghosts(
                            pg_idx,
                            arena_ranges[pg_idx],
                            list(
                                zip(
                                    arena_closing.offload_ordinals[pg_idx],
                                    arena_closing.offload[pg_idx],
                                )
                            ),
                        )
                    # P3 prefix aliasing: pin the committed prefix's resident
                    # pages so same-prefix admissions alias them (zero-copy).
                    # The prefix must be a contiguous ordinal run from 0 with
                    # a unique page per ordinal (v1: single-lc pool groups).
                    ordinals = arena_closing.offload_ordinals[pg_idx]
                    if ordinals and len(set(ordinals)) == len(ordinals):
                        by_ordinal = dict(zip(ordinals, arena_closing.offload[pg_idx]))
                        prefix_pages: list[CommittedPage] = []
                        o = 0
                        while o in by_ordinal:
                            prefix_pages.append(by_ordinal[o])
                            o += 1
                        if prefix_pages:
                            storage.register_arena_canonical_span(
                                pg_idx, arena_ranges[pg_idx], prefix_pages, prior_event
                            )
                except OutOfPagesError:
                    # Host tier cannot take the write-out: drop the blocks
                    # instead of keeping them (reuse loss only, never an
                    # error on the free path).
                    for page in arena_closing.offload[pg_idx]:
                        if page.scheduled_for_eviction:
                            storage.exclude_from_eviction(page)
            if arena_lazy:
                # Re-register each private copy's bytes as a fresh ghost for
                # its canonical entry: hot prefixes get their D2D source
                # refreshed by every closing user, so an LRU spill of the
                # oldest copy does not lose the (newer) resident one.
                for page, (ordinal, lc_idx) in zip(
                    arena_closing.private, arena_closing.private_meta, strict=True
                ):
                    canonical = self._find_canonical(page, lc_idx)
                    if canonical is not None and canonical is not page:
                        pg_idx = storage.get_pool_group_index(lc_idx)
                        storage.register_arena_ghosts(
                            pg_idx, arena_ranges[pg_idx], [(ordinal, canonical)]
                        )
            # Private copies are never evictable (nothing queues them); they
            # die with our refs, freeing their slots into their range.
            arena_closing = None  # drop our strong refs before range release
            self._arena_release_unused_write_through()
            for pg_idx, rng in typed_enumerate(arena_ranges):
                storage.free_gpu_sequence(pg_idx, rng, arena_last_consumer)
            self._arena_ranges = None
        self._status = self.Status.CLOSED
        manager._living_kv_caches.remove(self.__rawref__)

    def __del__(self) -> None:
        self.close()
        self.__rawref__.invalidate()

    @property
    def beam_width(self) -> BeamIndex:
        return self._beam_width

    # beam_width > 1 is only for generation. If decreasing beam_width, uncommitted data in blocks for
    # (beam_index >= beam_width) will be lost.
    @beam_width.setter
    def beam_width(self, beam_width: BeamIndex) -> None:
        raise NotImplementedError("Not implemented yet for beam search")

    # Get the indices of memory blocks for each beam.
    def get_base_page_indices(
        self, layer_group_id: LayerGroupId, beam_id: BeamIndex = DEFAULT_BEAM_INDEX
    ) -> IndexSeq:
        indices = self._base_page_indices[beam_id][layer_group_id]
        assert NDEBUG or all(
            v == value_or(r, BAD_PAGE_INDEX)
            for v, r in zip(indices, self._get_base_page_indices_ref(layer_group_id, beam_id))
        )
        return indices

    def get_ssm_block_base_index(
        self, layer_group_id: LayerGroupId, beam_id: BeamIndex = DEFAULT_BEAM_INDEX
    ) -> int:
        entry = self._ssm_blocks[beam_id][layer_group_id]
        if entry is None:
            return BAD_PAGE_INDEX
        return expect_type(_SharedPageLock, entry).page.slot_id

    def get_aggregated_page_indices(
        self,
        layer_group_id: LayerGroupId,
        beam_id: BeamIndex = DEFAULT_BEAM_INDEX,
        valid_only: bool = False,
    ) -> Iterator[int]:
        """
        Get the internal slot indices for the given layer group and beam.
        Each slot is a group of coalesced buffers in one memory pool group.
        This API exposes internal slot indices, mainly for efficient data transfer.
        For computation, use get_page_indices() instead.

        Args:
            layer_group_id: Layer group to inspect.
            beam_id: Beam index to read. Defaults to DEFAULT_BEAM_INDEX.

        Returns:
            Aggregated page index for each block, or BAD_PAGE_INDEX for invalid blocks.
        """
        for b in self._blocks:
            if (holder := b.pages[beam_id][layer_group_id]) is None:
                if not valid_only:
                    yield BAD_PAGE_INDEX
            else:
                yield holder.page.slot_id

    def get_scratch_desc(self, layer_group_id: LayerGroupId) -> "ScratchDesc | None":
        """
        Get scratch metadata for the given layer group, or None if scratch is not active.

        The returned ScratchDesc contains the scratch block ordinal range and the
        slot IDs for the scratch coalesced slots. Pass this to PageIndexConverter
        together with get_base_page_indices() to produce per-layer page indices.

        The returned ScratchDesc is invalidated by the next capacity/history_length update.
        """
        lc = self.manager._life_cycles[layer_group_id]
        sr = self._get_scratch_range(lc)
        if not sr:
            return None
        from ._kv_cache_manager import ScratchDesc

        return ScratchDesc(
            range=sr,
            slot_ids=[s.slot.slot_id for s in self._scratch_slots[layer_group_id]],
        )

    @property
    def has_scratch_slots(self) -> bool:
        """True if this KV cache currently has scratch slots allocated."""
        return any(len(s) > 0 for s in self._scratch_slots)

    @property
    def enable_swa_scratch_reuse(self) -> bool:
        return self._enable_swa_scratch_reuse

    @enable_swa_scratch_reuse.setter
    def enable_swa_scratch_reuse(self, enable: bool) -> None:
        if enable == self._enable_swa_scratch_reuse:
            return
        if enable:
            if not self.manager.enable_swa_scratch_reuse:
                raise ValueError(
                    "Cannot enable SWA scratch reuse for a request when it is disabled in "
                    "KV cache manager config"
                )
            if self._would_use_swa_scratch_blocks():
                raise ValueError(
                    "Cannot enable SWA scratch reuse while the current request state would "
                    "need scratch blocks"
                )
            self._enable_swa_scratch_reuse = True
            return

        if self._would_use_swa_scratch_blocks():
            raise ValueError("Cannot disable SWA scratch reuse while scratch blocks are needed")
        assert not self.has_scratch_slots
        self._enable_swa_scratch_reuse = False

    def supports_index_mode(self, mode: PageIndexMode) -> bool:
        match mode:
            case PageIndexMode.PER_LAYER:
                return True
            case PageIndexMode.SHARED:
                return not self.has_scratch_slots

    def _arena_collect_committed_gpu_pages(self) -> "_ArenaClosingPages":
        """Collect this sequence's committed GPU pages before the block chain
        is cleared at close(), partitioned by how they leave the arena:
        written-through pages adopt their host copy (copy-free), the rest are
        copied out on free (§4.3), and private reuse copies are dropped --
        their canonical stale copy already exists (§4.4). In arena mode GPU
        pages are never shared across sequences, so they are ours to move or
        drop.

        A separate method so its loop locals cannot outlive the collection: a
        lingering reference to a block's lock chain would delay the holder's
        death past close()'s eviction-exclusion pass, leaking the page into
        the (otherwise unused) GPU eviction queue.
        """
        storage = self.manager._storage
        ret = _ArenaClosingPages(
            make_typed(lambda _: list[CommittedPage](), storage.num_pool_groups),
            make_typed(lambda _: list[int](), storage.num_pool_groups),
            make_typed(lambda _: list[CommittedPage](), storage.num_pool_groups),
            make_typed(lambda _: list[Slot](), storage.num_pool_groups),
            make_typed(lambda _: list[int](), storage.num_pool_groups),
            [],
            [],
        )
        for ordinal, block in enumerate(self._blocks):
            if not block.is_committed:
                continue
            for beam_pages in block.pages:
                for lc_idx, p in typed_enumerate(beam_pages):
                    if p is None:
                        continue
                    page = p.page
                    if isinstance(page, CommittedPage) and page.cache_level == GPU_LEVEL:
                        if type(page) is not CommittedPage:
                            ret.private.append(page)
                            ret.private_meta.append((ordinal, lc_idx))
                            continue
                        pg_idx = storage.get_pool_group_index(lc_idx)
                        wt_slot = self._arena_take_write_through_slot(ordinal, lc_idx)
                        if wt_slot is not None:
                            ret.adopt_pages[pg_idx].append(page)
                            ret.adopt_slots[pg_idx].append(wt_slot)
                            ret.adopt_ordinals[pg_idx].append(ordinal)
                        else:
                            ret.offload[pg_idx].append(page)
                            ret.offload_ordinals[pg_idx].append(ordinal)
        return ret

    def _arena_evacuation_requirements(self) -> TypedIndexList[PoolGroupIndex, int]:
        """Per-pool-group host slots the §4.5 evacuation write-out will need.

        Mirrors :meth:`_arena_evacuate`'s partition WITHOUT side effects (in
        particular the write-through stash is inspected, not popped): private
        copies with a live canonical are dropped (0 slots), written-through
        pages adopt their pre-copied slot (0 slots), everything else on GPU
        migrates (1 slot). May overcount pages that later paths free before
        the migration (e.g. scratch) -- overcounting only reserves extra free
        slots, which the next consumer reuses; undercounting would break the
        pre-flight atomicity gate in :meth:`suspend`.
        """
        storage = self.manager._storage
        ssm_lc_id = self.manager._life_cycles.ssm_life_cycle_id
        requirements = make_typed(lambda _: 0, storage.num_pool_groups)
        for ordinal, beam_idx, lc_idx in self._active_pages():
            beam_block = (
                self._block(ordinal, beam_idx)
                if lc_idx != ssm_lc_id
                else self._ssm_blocks[beam_idx]
            )
            page = expect_type(_SharedPageLock, beam_block[lc_idx]).page
            if page.cache_level != GPU_LEVEL:
                continue
            if type(page) is PrivateCommittedPage:
                canonical = self._find_canonical(page, lc_idx)
                if canonical is not None and canonical is not page:
                    continue
            if (
                type(page) is CommittedPage
                and (int(ordinal), int(lc_idx)) in self._write_through_slots
            ):
                continue
            pg_idx = storage.get_pool_group_index(lc_idx)
            requirements[pg_idx] += 1
        return requirements

    @staticmethod
    def _find_canonical(page: CommittedPage, lc_idx: LifeCycleId) -> "CommittedPage | None":
        """The canonical radix-tree page for a private copy's block, or None
        if the tree entry is gone (orphaned block / dropped page)."""
        blk = page.block()
        if blk is None or blk.is_orphan:
            return None
        ref = blk.storage[lc_idx]
        if ref is None:
            return None
        return ref()

    def _arena_evacuate(self, prior_event: CachedCudaEvent = CachedCudaEvent.NULL) -> None:
        """Suspend-side write-out (DESIGN.md §4.5): move this sequence's GPU
        pages out of its arena ranges so the ranges can be unmapped.

        ``prior_event`` orders the evacuation copies after the sequence's last
        KV writes on the forward stream (see
        ``StorageManager.offload_arena_pages``) -- a scheduler eviction can
        suspend a sequence whose current step is still in flight.

        Written-through committed pages adopt their host copy (copy-free,
        §4.3); other canonical committed pages and uncommitted pages migrate
        to the host tier -- the block chain's holders follow their pages, so
        the logical chain survives suspension unchanged. Sequence-private
        reuse copies (§4.4) are swapped back to holding their canonical entry
        and dropped; if the canonical entry is gone, the private copy is the
        sole copy and migrates like a canonical page.

        Raises ``OutOfPagesError`` if the host tier cannot absorb the
        write-out even after eviction -- host-quota sizing (§4.6) must
        guarantee preemption headroom.

        A separate method so loop locals cannot outlive it (see
        :meth:`_arena_collect_committed_gpu_pages`).
        """
        storage = self.manager._storage
        assert storage.num_cache_levels > 1, "suspend in arena mode requires a host tier"
        beam_idx = DEFAULT_BEAM_INDEX
        move: TypedIndexList[PoolGroupIndex, list[Page]] = make_typed(
            lambda _: list[Page](), storage.num_pool_groups
        )
        adopt_pages: TypedIndexList[PoolGroupIndex, list[Page]] = make_typed(
            lambda _: list[Page](), storage.num_pool_groups
        )
        adopt_slots: TypedIndexList[PoolGroupIndex, list[Slot]] = make_typed(
            lambda _: list[Slot](), storage.num_pool_groups
        )
        swaps = list[tuple[BlockOrdinal, LifeCycleId, CommittedPage]]()
        dropped = list[CommittedPage]()
        for ordinal, entry_beam, lc_idx in self._active_pages():
            assert entry_beam == beam_idx
            holder = expect_type(_PageHolder, self._block(ordinal, beam_idx)[lc_idx])
            page = holder.page
            if page.cache_level != GPU_LEVEL:
                continue
            if type(page) is PrivateCommittedPage:
                canonical = self._find_canonical(page, lc_idx)
                if canonical is not None and canonical is not page:
                    swaps.append((ordinal, lc_idx, canonical))
                    dropped.append(page)
                    continue
            pg_idx = storage.get_pool_group_index(lc_idx)
            wt_slot = (
                self._arena_take_write_through_slot(int(ordinal), lc_idx)
                if type(page) is CommittedPage
                else None
            )
            if wt_slot is not None:
                adopt_pages[pg_idx].append(page)
                adopt_slots[pg_idx].append(wt_slot)
            else:
                move[pg_idx].append(page)
        for ordinal, lc_idx, canonical in swaps:
            # Holding the canonical page keeps it recallable for resume; the
            # private copy's holder dies with this assignment. Private copies
            # are never evictable, so they die with the last reference
            # (returning their arena slots) rather than lingering in an
            # eviction queue.
            self._block(ordinal, beam_idx)[lc_idx] = canonical.hold()
        dropped.clear()
        for pg_idx in typed_range(storage.num_pool_groups):
            storage.adopt_stale_copies(pg_idx, adopt_pages[pg_idx], adopt_slots[pg_idx])
            storage.offload_arena_pages(pg_idx, move[pg_idx], prior_event)
        self._arena_release_unused_write_through()

    def _arena_write_through(self, ordinal: BlockOrdinal) -> None:
        """Write-through on commit (§4.3, ``WriteThroughPolicy.ON_COMMIT``):
        copy the just-committed block's pages to fresh host slots now, so the
        free path (close/suspend) adopts them instead of copying. Skipped
        silently if the host tier cannot take the copy -- the copy-on-free
        fallback covers the block."""
        storage = self._storage
        beam_idx = DEFAULT_BEAM_INDEX
        beam_block = self._block(ordinal, beam_idx)
        per_pg_pages: dict[PoolGroupIndex, list[Page]] = {}
        per_pg_lcs: dict[PoolGroupIndex, list[LifeCycleId]] = {}
        for lc_idx, p in typed_enumerate(beam_block):
            if p is None:
                continue
            page = p.page
            if type(page) is not CommittedPage or page.cache_level != GPU_LEVEL:
                continue
            pg_idx = storage.get_pool_group_index(lc_idx)
            per_pg_pages.setdefault(pg_idx, []).append(page)
            per_pg_lcs.setdefault(pg_idx, []).append(lc_idx)
        if not per_pg_pages:
            return
        # The sources are still locked; order the copies after all work
        # enqueued so far on the sequence's stream (the block's writes).
        prior = CachedCudaEvent(self.cuda_stream)
        for pg_idx, pages in per_pg_pages.items():
            try:
                slots = storage.write_through_pages(pg_idx, pages, prior_event=prior)
            except OutOfPagesError:
                continue  # fall back to copy-on-free for this block
            for lc_idx, slot in zip(per_pg_lcs[pg_idx], slots):
                self._write_through_slots[(int(ordinal), int(lc_idx))] = slot

    def _arena_take_write_through_slot(self, ordinal: int, lc_idx: LifeCycleId) -> Slot | None:
        return self._write_through_slots.pop((int(ordinal), int(lc_idx)), None)

    def _arena_release_unused_write_through(self) -> None:
        """Return any unconsumed write-through host slots to their pool."""
        storage = self.manager._storage
        for (_, lc_int), slot in self._write_through_slots.items():
            storage.release_slot(LifeCycleId(lc_int), CacheLevel(GPU_LEVEL + 1), slot)
        self._write_through_slots.clear()

    # Pages to free per spill step under pressure; small enough to preserve
    # most of the retained cache, large enough to bound retry loops.
    _ARENA_SPILL_CHUNK_PAGES: ClassVar[int] = 64

    def _arena_ensure_mapped(self, num_valid_blocks: int, sync: bool = False) -> bool:
        """Make physical pages covering the first ``num_valid_blocks`` of this
        sequence's range available in every pool group (DESIGN.md §4.2).

        With ``batched_map_sweep`` (and ``sync=False``) the budget is charged
        now but the driver calls are deferred to the owner's per-iteration
        ``flush_gpu_mappings`` sweep; callers that touch the pages immediately
        (e.g. reuse onboarding copies) pass ``sync=True``. On page exhaustion,
        drains deferred reclaim, then spills lazily retained ranges
        (LRU-first, §4.4 phase 2), retrying while either makes progress;
        returns False if pages are still unavailable — the caller backs off
        exactly like a failed slot allocation (§4.6)."""
        storage = self._storage
        ranges = self._arena_ranges
        assert ranges is not None
        arena_cfg = storage.arena_config
        defer = not sync and arena_cfg is not None and arena_cfg.batched_map_sweep
        for pg_idx, rng in typed_enumerate(ranges):
            while True:
                try:
                    if defer:
                        storage.queue_gpu_mapping(pg_idx, rng, num_valid_blocks)
                    else:
                        storage.ensure_gpu_mapped(pg_idx, rng, num_valid_blocks)
                    break
                except OutOfPagesError:
                    if storage.drain_gpu_reclaim() > 0:
                        continue
                    if storage.spill_gpu_retained(self._ARENA_SPILL_CHUNK_PAGES) > 0:
                        continue
                    return False
        return True

    def _arena_onboard_matched(self) -> bool:
        """Bring this sequence's resident pages into its arena ranges
        (stale->active, DESIGN.md §4.4/§4.5). Serves both a fresh sequence
        with a reuse match and a post-suspend resume.

        Committed sources (canonical radix entries, or a private copy that
        outlived its entry) are *copied*: each becomes a sequence-private
        :class:`PrivateCommittedPage` at ``range base + ordinal`` and the
        source is left untouched wherever it lives (host, disk, or another
        sequence's arena -- committed blocks are immutable, so reading them
        concurrently is safe). Uncommitted pages (a suspended sequence's
        tail) are sequence-private already, so they are *moved*: the page
        object relocates into the arena slot and its host slot is freed.
        Consecutive ordinals give the explicit-destination migrate path
        contiguous targets to coalesce. Returns False (onboarding nothing)
        if physical pages are unavailable.
        """
        storage = self._storage
        ranges = self._arena_ranges
        assert ranges is not None
        beam_idx = DEFAULT_BEAM_INDEX
        entries = list[tuple[BlockOrdinal, LifeCycleId, _PageHolder]]()
        for ordinal, entry_beam, lc_idx in self._active_pages():
            assert entry_beam == beam_idx
            holder = expect_type(_PageHolder, self._block(ordinal, beam_idx)[lc_idx])
            entries.append((ordinal, lc_idx, holder))
        if not entries:
            self._alias_spans = None
            return True
        # P3 prefix aliasing: map the registry-pinned canonical span into any
        # FRESH range before growth maps run (frontier still at the range
        # start); a signature-matched adopted range already carries the
        # mappings. Consumed once -- suspend/resume round-trips never alias.
        alias_spans = self._alias_spans
        self._alias_spans = None
        if alias_spans is not None:
            for pg_idx, rng in typed_enumerate(ranges):
                hit = alias_spans[pg_idx]
                if hit is None:
                    continue
                if rng._alias_span_key is None:
                    storage.alias_arena_span(pg_idx, rng, hit[0], hit[1])
                else:
                    assert rng._alias_span_key == hit[0], (
                        "adopted range's alias signature must match the admission's span"
                    )
        if not self._arena_ensure_mapped(self.num_blocks, sync=True):
            return False
        # One explicit-destination migrate per (pool group, source level,
        # copy-vs-move); committed sources whose bytes are still resident in a
        # retained range short-circuit to D2D copies (§4.4 phase 2).
        dst_slots = list[Slot]()
        groups: dict[tuple[PoolGroupIndex, CacheLevel, bool], list[int]] = {}
        ghost_groups: dict[PoolGroupIndex, list[int]] = {}
        ghost_srcs: dict[PoolGroupIndex, list[SlotId]] = {}
        ghost_rngs: dict[PoolGroupIndex, list[SequenceRange]] = {}
        aliased_idx = list[int]()
        for i, (ordinal, lc_idx, holder) in enumerate(entries):
            pg_idx = storage.get_pool_group_index(lc_idx)
            dst_slots.append(storage.take_gpu_sequence_slot(pg_idx, ranges[pg_idx], int(ordinal)))
            page = holder.page
            if page.is_committed():
                if alias_spans is not None:
                    hit = alias_spans[pg_idx]
                    if (
                        hit is not None
                        and int(ordinal) < hit[1].num_blocks
                        and hit[1].page_refs[int(ordinal)]() is page
                    ):
                        # P3: the block's bytes are already visible at this
                        # exact slot through the alias mapping -- no copy.
                        # Consumers wait the span's ready event.
                        dst_slots[-1].ready_event = hit[1].ready_event
                        aliased_idx.append(i)
                        continue
                ghost = storage.lookup_arena_ghost(pg_idx, page)
                if ghost is not None:
                    src_id, src_rng = ghost
                    ghost_groups.setdefault(pg_idx, []).append(i)
                    ghost_srcs.setdefault(pg_idx, []).append(src_id)
                    ghost_rngs.setdefault(pg_idx, []).append(src_rng)
                    continue
            if page.scheduled_for_eviction:
                # Migration requires unscheduled sources; being held again on
                # the other side of onboarding re-schedules them as usual.
                storage.exclude_from_eviction(page)
            is_move = not page.is_committed()
            groups.setdefault((pg_idx, page.cache_level, is_move), []).append(i)
        copied: list[Slot | None] = [None] * len(entries)
        for (pg_idx, src_level, is_move), idx_list in groups.items():
            ret = storage._batched_migrate(
                pg_idx,
                GPU_LEVEL,
                src_level,
                [entries[i][2].page for i in idx_list],
                update_src=is_move,
                migration_recorder=self._record_migrated_slots,
                dst_slots=[dst_slots[i] for i in idx_list],
            )
            if not is_move:
                assert ret is not None
                for i, slot in zip(idx_list, ret):
                    copied[i] = slot
        for pg_idx, idx_list in ghost_groups.items():
            ghost_dsts = [dst_slots[i] for i in idx_list]
            storage.onboard_from_retained(
                pg_idx, ghost_srcs[pg_idx], ghost_dsts, ghost_rngs[pg_idx]
            )
            for i, slot in zip(idx_list, ghost_dsts):
                copied[i] = slot
        # P3 aliased blocks: the slot IS the data (no copy ran); the lock
        # loop below builds the same sequence-private page over it.
        for i in aliased_idx:
            copied[i] = dst_slots[i]
        stream_wait_events(
            self.cuda_stream,
            (
                slot.ready_event if slot is not None else entries[i][2].page.ready_event
                for i, slot in enumerate(copied)
            ),
        )
        # Lock everything in place. For copies, the private page replaces the
        # holder; dropping it releases the source back to eviction control at
        # its current level. Moved pages keep their holder and simply lock.
        for (ordinal, lc_idx, holder), copied_slot in zip(entries, copied):
            src_page = holder.page
            if copied_slot is None:  # moved uncommitted page, now living in the arena
                assert src_page.cache_level == GPU_LEVEL
                lock = holder.lock(self, beam_idx, ordinal, lc_idx, skip_wait=True)
            else:
                assert isinstance(src_page, CommittedPage), (
                    "arena mode matches full committed blocks only"
                )
                tree_block = src_page.block()
                assert tree_block is not None, "committed page lost its tree block while held"
                private = PrivateCommittedPage(
                    storage, tree_block, lc_idx, GPU_LEVEL, copied_slot, src_page.priority
                )
                lock = private.lock(self, beam_idx, ordinal, lc_idx, skip_wait=True)
            self._block(ordinal, beam_idx)[lc_idx] = lock
        return True

    # reserve space for next inference. Request new blocks from KVCacheManager if necessary.
    # if capacity is increased and beam_width > 1, blocks containing new tokens should be allocated for each beam.
    # Decrease of capacity may destroy stale blocks (if not used by other requests).
    # Decrease of capacity cannot remove historical or committed tokens.
    # History length cannot be decreased.
    # Increase of history length may trigger out-of-window block eviction/dropping for SWA layers.
    # If we use two separate APIs for capacity and history length, sometimes we will need to increase
    # capacity first to maintain capacity >= history_length. But then we may have a middle state (between
    # two APIs) where we use more pages than necessary for SWA layers. So we use a single API to avoid
    # this. Usually this is a concern only for prefill phase where we create many tokens in one step. For
    # other cases, we can just set the capacity and history_length properties instead.
    def resize(self, capacity: int | None, history_length: int | None = None) -> bool:
        assert self.status == self.Status.ACTIVE
        tokens_per_block = self.tokens_per_block
        assert div_up(self._capacity, tokens_per_block) == len(self._blocks)
        if capacity is None:
            capacity = self._capacity
        else:
            self._avg_capacity.update(capacity)
        if history_length is None:
            history_length = self._history_length
        else:
            self._avg_history_length.update(history_length)
        if history_length < self._history_length:
            raise ValueError("History length cannot be decreased")
        if capacity < history_length:
            raise ValueError("History length cannot be greater than capacity")
        if self._arena_ranges is not None and capacity > unwrap_optional(self._max_capacity):
            raise ValueError(
                f"capacity ({capacity}) exceeds the max_capacity "
                f"({self._max_capacity}) declared at creation (contiguous-arena mode, "
                f"DESIGN.md §4.1)"
            )
        manager = self.manager
        # Scratch reuse: compute scratch ranges and slot delta
        enable_scratch = self.enable_swa_scratch_reuse
        if enable_scratch and capacity != self._capacity:
            max_rewind_len = self._swa_scratch_max_rewind_len()
            min_history_length = max(0, self._capacity - max_rewind_len)
            assert min_history_length <= history_length <= self._capacity, (
                "SWA scratch requires "
                f"old_capacity - max_rewind_len ({min_history_length}) <= "
                f"history_length ({history_length}) <= "
                f"old_capacity ({self._capacity})"
            )
        record_generation_alloc_stats = self._should_record_generation_alloc_stats(capacity)
        if (
            not enable_scratch
            and self._shortcut_set_capacity(capacity)
            and self._shortcut_set_history_length(history_length)
        ):
            self._refresh_generation_alloc_ready()
            return True
        ssm_lc_id = manager._life_cycles.ssm_life_cycle_id
        beam_width = self.beam_width
        backup_holders = self._unlock_stale_blocks(history_length)
        old_num_blocks = BlockOrdinal(div_up(self._capacity, tokens_per_block))
        new_num_blocks = BlockOrdinal(div_up(capacity, tokens_per_block))
        num_life_cycles = manager._life_cycles.size
        if new_num_blocks < old_num_blocks:
            assert not self.has_scratch_slots, "Cannot shrink while scratch slots exist"
            self._subtract_pending_allocation_range(new_num_blocks, old_num_blocks)
            with self._record_event():
                del self._blocks[new_num_blocks:]
            for beam_indices in self._base_page_indices:
                for indices in beam_indices:
                    assert all(i == BAD_PAGE_INDEX for i in indices[new_num_blocks:])
                    if type(indices) is array.array:
                        del indices[new_num_blocks:]
                    else:
                        indices[new_num_blocks:] = array.array("i", [BAD_PAGE_INDEX]) * (
                            len(indices) - new_num_blocks
                        )

        excess_scratch_slots, delta_scratch_slots, scratch_ranges = self._take_excess_scratch_slots(
            capacity, history_length
        )

        if new_num_blocks >= old_num_blocks:
            num_new_slots = filled_list(0, num_life_cycles)
            stale_ranges = [
                _KVCache._get_stale_range(tokens_per_block, history_length, lc)
                for _, lc in manager._life_cycles.items()
            ]
            for lc in typed_range(num_life_cycles):
                if lc == ssm_lc_id:
                    continue
                stale_beg, stale_end = stale_ranges[lc]
                if enable_scratch:
                    # Only newly added blocks consume slots below; scratch range may
                    # extend before old_num_blocks when history_length < old_capacity.
                    new_block_range = HalfOpenRange(old_num_blocks, new_num_blocks)
                    num_new_blocks_using_scratch = len(
                        intersect(scratch_ranges[lc], new_block_range)
                    )
                    num_new_normal_blocks = len(new_block_range) - num_new_blocks_using_scratch
                    num_new_slots[lc] = num_new_normal_blocks * beam_width
                else:
                    if old_num_blocks < stale_beg:
                        assert new_num_blocks >= stale_end
                        num_new_blocks_to_add = (stale_beg - old_num_blocks) + (
                            new_num_blocks - stale_end
                        )
                    else:
                        num_new_blocks_to_add = new_num_blocks - max(stale_end, old_num_blocks)
                    num_new_slots[lc] = num_new_blocks_to_add * beam_width

            net_alloc_counts = make_typed(
                lambda lc: num_new_slots[lc] + delta_scratch_slots[lc], num_life_cycles
            )
            storage = self._storage
            arena_ranges = self._arena_ranges
            if arena_ranges is not None:
                # Arena growth (DESIGN.md §4.2): no scattered slot allocation.
                # Demand-map physical pages covering the new block frontier in
                # every pool group; block-index slots are issued per ordinal in
                # the construction loop below.
                assert self.beam_width == 1
                if new_num_blocks > old_num_blocks and not self._arena_ensure_mapped(
                    int(new_num_blocks)
                ):
                    self._recover_excess_scratch_slots(excess_scratch_slots)
                    self._lock_held_blocks(backup_holders)
                    return False
                new_slots = make_typed(lambda _: list[Slot](), num_life_cycles)
            elif any(c > 0 for c in net_alloc_counts):
                try:
                    new_slots = storage.new_gpu_slots(
                        make_typed(lambda lc: max(0, net_alloc_counts[lc]), num_life_cycles),
                        self._record_migrated_slots,
                    )
                except OutOfPagesError:
                    self._recover_excess_scratch_slots(excess_scratch_slots)
                    self._lock_held_blocks(backup_holders)
                    return False
            else:
                new_slots = make_typed(lambda _: list[Slot](), num_life_cycles)

            # Wait on newly allocated slots
            stream_wait_events(
                self.cuda_stream, (s.ready_event for s in chain.from_iterable(new_slots))
            )

            # Combine slots and distribute
            slots = make_typed(lambda _: list[Slot](), num_life_cycles)
            for lc in typed_range(num_life_cycles):
                slots[lc] = new_slots[lc] + [
                    lock.detach_slot() for lock in excess_scratch_slots[lc]
                ]
                new_slots[lc].clear()
                excess_scratch_slots[lc].clear()

            if any(cnt < 0 for cnt in net_alloc_counts):
                with self._record_event():
                    for lc in typed_range(num_life_cycles):
                        for _ in range(-net_alloc_counts[lc]):
                            slot = slots[lc].pop()
                            slot.ready_event = self.finish_event
                            storage.release_slot(lc, GPU_LEVEL, slot)

            assert arena_ranges is not None or all(
                len(slots[lc]) == num_new_slots[lc] + max(0, delta_scratch_slots[lc])
                for lc in typed_range(num_life_cycles)
            )

            # Fulfill additional scratch slots
            for lc in typed_range(num_life_cycles):
                for _ in range(delta_scratch_slots[lc]):
                    slot = slots[lc].pop()
                    self._scratch_slots[lc].append(ScratchSlotLock(slot, self, lc, skip_wait=True))

            for beam_indices in self._base_page_indices:
                for indices in beam_indices:
                    if type(indices) is array.array:
                        assert len(indices) == old_num_blocks
                        indices.extend([BAD_PAGE_INDEX] * (new_num_blocks - old_num_blocks))
                    else:
                        if len(indices) < new_num_blocks:
                            raise ValueError("User-provided base page indices is too short")

            stream_wait_events(
                self.cuda_stream, (s.ready_event for s in chain.from_iterable(slots))
            )

            # Scratch blocks use temporary shared SWA slots instead of normal
            # per-request KV pages, so they are excluded from alloc/miss stats.
            excluded_ranges = (
                scratch_ranges if enable_scratch else to_typed(LifeCycleId, stale_ranges)
            )
            self._record_resize_pending_allocations(
                old_num_blocks,
                new_num_blocks,
                beam_width,
                excluded_ranges,
                record_generation_alloc_stats,
            )
            for ordinal in typed_range(old_num_blocks, new_num_blocks):
                block = make_typed(
                    lambda _: filled_list(cast(BlockPage, None), num_life_cycles), beam_width
                )
                for beam_index in typed_range(beam_width):
                    for lc in typed_range(num_life_cycles):
                        if lc == ssm_lc_id:
                            continue  # SSM pages live in _ssm_blocks, not in _blocks
                        if enable_scratch:
                            # Assertion guarantees no new block is stale.
                            if ordinal in scratch_ranges[lc]:
                                continue  # Scratch block — no per-block page allocation
                        else:
                            stale_beg, stale_end = stale_ranges[lc]
                            if stale_beg <= ordinal < stale_end:
                                continue
                        if arena_ranges is not None:
                            # slot_id = range base + ordinal: the sequence's
                            # blocks are consecutive in VA (DESIGN.md §4.1).
                            pg_idx = storage.get_pool_group_index(lc)
                            slot = storage.take_gpu_sequence_slot(
                                pg_idx, arena_ranges[pg_idx], int(ordinal)
                            )
                        else:
                            slot = slots[lc].pop()
                        # We have already waited for ready_event of the slots.
                        block[beam_index][lc] = UncommittedPage(
                            self, ordinal, lc, GPU_LEVEL, slot, beam_index
                        ).lock(self, beam_index, ordinal, lc, skip_wait=True)
                self._blocks.append(SeqBlock(block, None))
            assert all(len(slots[lc]) == 0 for lc in typed_range(num_life_cycles))
        self._capacity = capacity
        self._history_length = history_length
        self._refresh_generation_alloc_ready()
        assert NDEBUG or self._check_sanity()
        return True

    @property
    def capacity(self) -> int:
        "Get the current capacity in number of tokens."
        return self._capacity

    @capacity.setter
    def capacity(self, capacity: int) -> None:
        """
        Reserve space for next inference. Capacity cannot be smaller than history length.
        Use resize() instead if you need to change both capacity and history length. If you use two
        separate APIs, you may have a middle state (between two APIs) where we use more pages than
        necessary for SWA layers.
        Expect OutOfPagesError exception if there are not enough pages in GPU memory.
        """
        if self.enable_swa_scratch_reuse:
            raise ValueError(
                "Cannot use capacity setter when SWA scratch reuse is enabled. "
                "Use resize(capacity, history_length) instead."
            )
        success = self.resize(capacity, None)
        if not success:
            raise OutOfPagesError("Not enough pages in GPU memory")

    @property
    def history_length(self) -> int:
        """
        Get the current history length in number of tokens. history_length decides how many blocks
        needs to be in GPU memory for SWA layers.
        """
        return self._history_length

    @history_length.setter
    def history_length(self, history_length: int) -> None:
        "History length cannot be decreased. Increase may trigger out-of-window block eviction/dropping for SWA layers."
        success = self.resize(None, history_length)
        assert success

    # notify KV cache manager that we have some finalized/accepted tokens. If a block becomes full,
    # also commit the block for reuse.
    # In case of beam search, this should be called only with finalized (converged) tokens, and the
    # token data must be in the 0th beam.
    # We'll destroy memory blocks for other beams if the whole block is full and committed.
    # Committed tokens are always history, so history_length will be automatically updated to maintain
    # (num_committed_tokens <= history_length). Note that history_length increase may trigger out-of-window
    # block eviction/dropping for SWA layers.
    # beam_search_indices: indices indicating which candidate to choose for each token. A block with all
    # tokens committed will be unified to one memory page and the other memory pages are dropped. Only for
    # beam search.
    def commit(
        self,
        accepted_input_tokens: Sequence[TokenIdExt],
        beam_search_indices: Sequence[int] | None = None,
    ):
        if self.beam_width != 1:
            raise NotImplementedError("Not implemented yet for beam search")
        if not accepted_input_tokens:
            return
        assert beam_search_indices is None
        assert self.status == self.Status.ACTIVE
        if self._commit_state == self.CommitState.USER_STOP:
            raise LogicError("Cannot commit tokens after stop_committing()")
        self._committed_tokens.extend(accepted_input_tokens)
        if self._commit_state == self.CommitState.VIRTUAL_STOP:
            return
        num_committed_blocks = self._num_committed_blocks
        new_num_full_blocks = BlockOrdinal(self.num_committed_tokens // self.tokens_per_block)
        if new_num_full_blocks > num_committed_blocks:
            with self._record_event():
                for ordinal in typed_range(num_committed_blocks, new_num_full_blocks):
                    if self._commit_state != self.CommitState.ALLOWED:
                        # A block in this batch hit a commit conflict
                        # (virtual stop); the remaining tokens stay virtually
                        # committed, same as the entry guard above.
                        break
                    self._commit_block(ordinal, False)
        if self.history_length < self.num_committed_tokens:
            self.history_length = self.num_committed_tokens

    # Note that the tokens may not be ready yet, if the event passed to the past commit() calls are not yet signaled.
    @property
    def num_committed_tokens(self) -> int:
        return len(self._committed_tokens)

    # Users promise to not commit any more tokens. For cases where we shouldn't reuse generated tokens
    # (eg. CoT), this helps us drop (instead of evict) out-of-window blocks for SWA layers.
    # If there is a uncommitted block containing committed tokens, we will commit the block immediately.
    def stop_committing(self) -> None:
        assert self.status != self.Status.CLOSED
        if self._commit_state == self.CommitState.USER_STOP:
            return
        assert NDEBUG or self._check_sanity()
        if self._commit_state == self.CommitState.VIRTUAL_STOP:
            self._commit_state = self.CommitState.USER_STOP
            return
        assert self._commit_state == self.CommitState.ALLOWED
        if self.num_committed_tokens % self.tokens_per_block != 0:
            ordinal = _KVCache._to_block_ordinal(self.tokens_per_block, self.num_committed_tokens)
            with self._record_event():
                self._commit_block(ordinal, True)
        else:
            self._commit_state = self.CommitState.USER_STOP
            self._on_stop_committing()
        # TODO: check if the last committed pages are usable, in case some prior pages are already
        # dropped. For SWA, this can be done only when we stop committing. (TRTLLM-8802)
        assert self._commit_state == self.CommitState.USER_STOP

    # Suspend, allow the KV cache manager to evict buffers from GPU, but don't drop them.
    # suspend+resume allows us to implement dynamic batch size. May also be used to support HSTU model.
    def suspend(self, prior_event: CachedCudaEvent = CachedCudaEvent.NULL) -> None:
        """Suspend the sequence, evacuating its GPU state to the host tier.

        ``prior_event`` should cover the sequence's last KV writes on the
        *forward* stream (see ``StorageManager.offload_arena_pages``): the
        scheduler evicts mid-pass, when the victim's current step may still be
        in flight, and the evacuation copies would otherwise read its tail
        block before the step's writes land.

        Raises ``OutOfPagesError`` -- with the sequence left fully intact and
        ACTIVE -- if the host tier cannot absorb the evacuation write-out even
        after LRU eviction (its capacity can be pinned by already-suspended
        sequences' HELD pages, which are not droppable). The whole write-out
        is pre-reserved below BEFORE any state mutation, so the error can
        never leave a half-evacuated sequence; callers degrade to
        drop-and-recompute (v1-style preemption) instead of crashing."""
        assert self.status == self.Status.ACTIVE
        assert self._check_sanity()
        if self._arena_ranges is not None and self.manager._storage.num_cache_levels > 1:
            # Pre-flight atomicity gate (§4.5 hardening): reserve host slots
            # for the entire evacuation while nothing has been mutated yet.
            # The manager is single-threaded, so slots freed here stay free
            # until _arena_evacuate's per-pool-group offloads consume them
            # (their own prepare_free_slots calls become no-ops).
            requirements = self._arena_evacuation_requirements()
            if any(requirements):
                self.manager._storage.prepare_free_slots(CacheLevel(GPU_LEVEL + 1), requirements)
        # Assert on a local, not the attribute: `assert self._finish_event is
        # None` narrows the member to None for the rest of the function under
        # mypy(c) -- it cannot see _record_event's side effect -- and compiled
        # code would then unbox the later read as literal None.
        entry_finish_event = self._finish_event
        assert entry_finish_event is None
        for beam_idx, beam_indices in typed_enumerate(self._base_page_indices):
            for lc, indices in typed_enumerate(beam_indices):
                if type(indices) is memoryview:
                    self.set_base_page_index_buf(beam_idx, lc, None)
        ssm_lc_id = self.manager._life_cycles.ssm_life_cycle_id
        arena_last_consumer = cast(CachedCudaEvent, CachedCudaEvent.NULL)
        with self._record_event():  # used by _SharedPageLock.__del__
            for ordinal, beam_idx, lc_idx in self._active_pages():
                beam_block = (
                    self._block(ordinal, beam_idx)
                    if lc_idx != ssm_lc_id
                    else self._ssm_blocks[beam_idx]
                )
                holder = expect_type(_SharedPageLock, beam_block[lc_idx]).holder
                # after this assignment, __del__ of the original _SharedPageLock will use self.finish_event
                # to indicate end of usage for the page.
                beam_block[lc_idx] = holder
            # Free scratch slots on suspend since the data is ephemeral
            self._free_scratch_slots()
            # _record_event clears _finish_event on exit; capture it as the
            # range-reclaim gate (§4.2) while it is live.
            finish_event: CachedCudaEvent | None = self._finish_event
            if finish_event is not None:
                arena_last_consumer = finish_event
        if self._arena_ranges is not None:
            # §4.5: move this sequence's pages out of its arena ranges, then
            # release the whole VA reservation (event-gated unmap). Resume
            # reserves fresh ranges and copies the resident state back in.
            arena_cfg = self.manager._storage.arena_config
            if arena_cfg is not None and arena_cfg.batched_map_sweep:
                # Evacuation reads this sequence's pages; execute any
                # still-deferred growth maps first.
                self.manager._storage.flush_gpu_mappings()
            self._arena_evacuate(prior_event)
            storage = self.manager._storage
            for pg_idx, rng in typed_enumerate(self._arena_ranges):
                storage.free_gpu_sequence(pg_idx, rng, arena_last_consumer)
            self._arena_ranges = None
        self._status = self.Status.SUSPENDED

    # Resume, migrate buffers to GPU memory.
    def resume(self, cuda_stream: CudaStream | None = None) -> bool:
        assert self.status == self.Status.SUSPENDED
        if cuda_stream is not None:
            self.cuda_stream = cuda_stream
        if self._storage.is_arena_mode:
            # Arena frees are deferred (event-gated unmap, §4.2), so page
            # utilization can stay pinned above the resume gate after other
            # sequences finished. Classic mode returns slots synchronously and
            # never needs this. Without the drain here, a caller that retries
            # resume() in a loop (e.g. the scheduler) can livelock: nothing is
            # schedulable, so no other path drains either.
            self._storage.drain_gpu_reclaim()
        utilization = max(self._storage.get_utilization(GPU_LEVEL))
        if utilization > self.manager._init_config.max_util_for_resume:
            return False
        assert self._cuda_stream is not None, "cuda_stream is never set"
        assert self._finish_event is None
        storage = self._storage
        if storage.is_arena_mode and self._arena_ranges is None:
            # Reserve this sequence's contiguous block-index range in every
            # pool group, sized for max_capacity (DESIGN.md §4.1). Pure VA
            # bookkeeping; pages are mapped by _arena_onboard_matched below
            # (resident prefix) and on demand by resize() (growth). Fresh and
            # post-suspend resumes take the same path; the base block index
            # may change across suspend/resume -- harmless, offset tables are
            # rebuilt from the locks (§4.5).
            max_capacity = self._max_capacity
            assert max_capacity is not None
            max_blocks = div_up(max_capacity, self.tokens_per_block)
            ranges = list[SequenceRange]()
            alias_spans = self._alias_spans
            try:
                pg_idx = PoolGroupIndex(0)
                while pg_idx < storage.num_pool_groups:
                    try:
                        # P3: a registry hit makes adoption prefix-affine --
                        # a parked range already alias-mapping this span is
                        # handed over with its mappings (and content) intact.
                        alias_key = None
                        if alias_spans is not None and alias_spans[pg_idx] is not None:
                            alias_key = alias_spans[pg_idx][0]
                        ranges.append(storage.reserve_gpu_sequence(pg_idx, max_blocks, alias_key))
                        pg_idx = PoolGroupIndex(pg_idx + 1)
                    except MemoryError:
                        # VA exhaustion: retained ranges hold VA too (§4.4
                        # phase 2); spill everything spillable and retry once.
                        if storage.spill_gpu_retained(1 << 62) == 0:
                            raise
            except MemoryError:
                # VA exhaustion / fragmentation: back off like any other
                # resource shortage. Nothing was mapped; undo the bookkeeping.
                for pg_idx, rng in typed_enumerate(
                    cast("TypedIndexList[PoolGroupIndex, SequenceRange]", ranges)
                ):
                    storage.free_gpu_sequence(pg_idx, rng, CachedCudaEvent.NULL)
                storage.drain_gpu_reclaim()
                return False
            self._arena_ranges = cast("TypedIndexList[PoolGroupIndex, SequenceRange]", ranges)
        ssm_lc_id = self.manager._life_cycles.ssm_life_cycle_id
        life_cycles = self.manager._life_cycles
        num_life_cycles = life_cycles.size

        # Pre-allocate GPU slots for deferred copies (partial blocks + SSM) before locking,
        # so we never end up in a state where pages are locked but we can't allocate for the copy.
        deferred_slots: TypedIndexList[LifeCycleId, Slot | None] = filled_list(
            None, storage.num_life_cycles
        )

        excess_scratch_slots, delta_scratch_slots, _ = self._take_excess_scratch_slots(
            self.capacity, self.history_length
        )
        assert all(len(s) == 0 for s in excess_scratch_slots)

        num_slots = filled_list(0, num_life_cycles)
        has_partial = False
        if self._never_resumed:
            assert self.beam_width == 1
            has_partial = self.num_committed_tokens % self.tokens_per_block != 0
            for lc_idx, lc in life_cycles.items():
                if type(lc) is SsmLifeCycle or has_partial:
                    num_slots[lc_idx] += 1

        for lc_idx in typed_range(num_life_cycles):
            num_slots[lc_idx] += delta_scratch_slots[lc_idx]

        if any(c > 0 for c in num_slots):
            try:
                tmp_slots = storage.new_gpu_slots(num_slots, self._record_migrated_slots)
            except OutOfPagesError:
                return False

            # Wait for scratch slots to be ready
            scratch_slots_to_add = make_typed(lambda _: list[Slot](), num_life_cycles)
            for lc_idx, slot_lst in zip(typed_range(num_life_cycles), tmp_slots, strict=True):
                if self._never_resumed and (
                    type(life_cycles[lc_idx]) is SsmLifeCycle or has_partial
                ):
                    deferred_slots[lc_idx] = slot_lst.pop(0)
                scratch_slots_to_add[lc_idx] = slot_lst

            stream_wait_events(
                self.cuda_stream, (s.ready_event for s in chain.from_iterable(scratch_slots_to_add))
            )

            for lc_idx in typed_range(num_life_cycles):
                for slot in scratch_slots_to_add[lc_idx]:
                    self._scratch_slots[lc_idx].append(
                        ScratchSlotLock(slot, self, lc_idx, skip_wait=True)
                    )

        if self._arena_ranges is not None:
            # Stale->active onboarding into this sequence's arena ranges
            # (DESIGN.md §4.4) instead of locking radix pages in place. No
            # deferred slots exist in arena mode (no SSM / scratch / partial
            # matches).
            assert all(slot is None for slot in deferred_slots)
            if not self._arena_onboard_matched():
                return False
        else:
            tasks = list[BatchedLockTarget]()
            for ordinal, beam_idx, lc_idx in self._active_pages():
                beam_block = (
                    self._block(ordinal, beam_idx)
                    if lc_idx != ssm_lc_id
                    else self._ssm_blocks[beam_idx]
                )
                page = expect_type(_PageHolder, beam_block[lc_idx]).page
                tasks.append(BatchedLockTarget(page, beam_idx, ordinal, lc_idx))
            try:
                locks = batched_lock_to_gpu(self, tasks, self._record_migrated_slots)
            except OutOfPagesError:
                for lc_idx, deferred_slot in typed_enumerate(deferred_slots):
                    if deferred_slot is not None:
                        storage.release_slot(lc_idx, GPU_LEVEL, deferred_slot)
                return False

            # Replace all holders with locks.
            for (ordinal, beam_idx, lc_idx), lock in zip(self._active_pages(), locks):
                beam_block = (
                    self._block(ordinal, beam_idx)
                    if lc_idx != ssm_lc_id
                    else self._ssm_blocks[beam_idx]
                )
                page = expect_type(_PageHolder, beam_block[lc_idx]).page
                assert page is lock.page
                beam_block[lc_idx] = lock

        # Deferred copy: for partial blocks and SSM, copy from now-locked source pages
        # to pre-allocated GPU slots, then unlock sources and replace with new pages.
        if self._never_resumed:
            beam_idx = DEFAULT_BEAM_INDEX
            last_ordinal = self._to_block_ordinal(
                self.tokens_per_block, self.num_committed_tokens - 1
            )
            # Phase 1: Copy GPU→GPU from locked source pages to pre-allocated slots.
            src_locks: list[_SharedPageLock] = []
            gpu_tier = storage.cache_tiers[GPU_LEVEL]
            # wait for all new slots to be ready
            stream_wait_events(
                self.cuda_stream, (slot.ready_event for slot in deferred_slots if slot is not None)
            )
            for lc_idx, new_slot in typed_enumerate(deferred_slots):
                if new_slot is None:
                    continue
                if lc_idx == ssm_lc_id:
                    if self.num_committed_tokens == 0:
                        continue  # fresh SSM — no source to copy from
                    lock = self._ssm_blocks[beam_idx][lc_idx]
                else:
                    lock = self._block(last_ordinal, beam_idx)[lc_idx]
                assert type(lock) is _SharedPageLock
                # V2 still copies a partial reuse into a private slot before writing to it.
                # The copy allocates a block, but it is a miss only without a reusable source.
                has_partial_reuse_source = self._has_reuse_source(lock)
                src_locks.append(lock)
                pg_idx = storage._life_cycle_grouping[lc_idx]
                slot_size = storage.slot_size(pg_idx)
                for p in typed_range(storage.num_pools(pg_idx)):
                    dst = storage.slot_address(GPU_LEVEL, pg_idx, new_slot.slot_id, p)
                    src = storage.slot_address(GPU_LEVEL, pg_idx, lock.page.slot_id, p)
                    # todo: add another batched copy supporting non-uniform size.
                    batched_copy(
                        gpu_tier,
                        gpu_tier,
                        slot_size[p],
                        [CopyTask(dst, src)],
                        self.cuda_stream,
                    )
                if lc_idx != ssm_lc_id:
                    life_cycle_key = self._stats_life_cycle_key(lc_idx)
                    if life_cycle_key is not None and self._should_record_stats():
                        changed = self._pending_stats.record_allocation_range(
                            life_cycle_key,
                            last_ordinal,
                            BlockOrdinal(last_ordinal + 1),
                            beam_width=1,
                            count_as_missed=not has_partial_reuse_source,
                        )
                        if changed:
                            self.manager.mark_stats_dirty(self.id)
                    self._record_direct_iteration_stats(
                        lc_idx,
                        KVCacheIterationStatsDelta(
                            iter_intra_device_copy_blocks=1,
                            iter_intra_device_copy_bytes=sum(storage.slot_size(pg_idx)),
                        ),
                    )
            # Unlock source pages — _record_event captures all prior cuda work
            # so the original pages know when we're done reading from them.
            if src_locks:
                with self._record_event():
                    for lock in src_locks:
                        lock.unlock()
            # Phase 2: Replace with new UncommittedPages (both copied and fresh SSM).
            for lc_idx, new_slot in typed_enumerate(deferred_slots):
                if new_slot is None:
                    continue
                if lc_idx == ssm_lc_id:
                    beam_block = self._ssm_blocks[beam_idx]
                    block_ordinal = BAD_BLOCK_ORDINAL
                else:
                    beam_block = self._block(last_ordinal, beam_idx)
                    block_ordinal = last_ordinal
                new_page = UncommittedPage(
                    self, block_ordinal, lc_idx, GPU_LEVEL, new_slot, beam_idx
                )
                new_lock = new_page.lock(self, beam_idx, block_ordinal, lc_idx, skip_wait=True)
                beam_block[lc_idx] = new_lock
            # Clear tree_block for the partial block — it's now uncommitted.
            if self.num_committed_tokens % self.tokens_per_block != 0:
                self._blocks[last_ordinal].tree_block = None
        self._never_resumed = False
        self._status = self.Status.ACTIVE
        return True

    def prefetch(self, target: CacheLevel) -> bool:
        """Best-effort prefetch active pages to the target cache level.

        The cache must be suspended. Prefetch is only a performance hint: a False
        return value means the requested pages could not be recalled due to cache
        pressure, but the cache remains functionally valid.

        Args:
            target: Destination cache level for active pages in lower tiers.

        Returns:
            True if the prefetch was dispatched, False if storage could not reserve enough pages.
        """
        assert self.status == self.Status.SUSPENDED
        manager = self.manager
        storage = manager._storage
        num_tiers = storage.num_cache_levels
        assert CacheLevel(0) <= target < num_tiers

        num_pool_groups = storage.num_pool_groups
        lc2pg = storage.get_pool_group_index

        all_pages = make_typed(
            lambda _: make_typed(lambda _: list[Page](), num_tiers), num_pool_groups
        )

        for ordinal, beam_idx, lc_idx in self._active_pages():
            holder = self._page(ordinal, beam_idx, lc_idx)
            if holder is None:
                continue
            page = expect_type(_PageHolder, holder).page
            lvl = page.cache_level
            if lvl < target:
                continue
            pg_idx = lc2pg(lc_idx)
            all_pages[pg_idx][lvl].append(page)

        try:
            storage.prefetch(target, all_pages)
        except OutOfPagesError:
            return False
        return True

    def _active_pages(self) -> Iterator[tuple[BlockOrdinal, BeamIndex, LifeCycleId]]:
        """Yields (ordinal, beam_idx, lc_idx) for all active pages.

        For attention life cycles, yields non-stale blocks from _blocks (excluding scratch blocks).
        For SSM, yields entries from _ssm_blocks with ordinal=BAD_BLOCK_ORDINAL.
        """
        ssm_lc_id = self.manager._life_cycles.ssm_life_cycle_id
        for lc_idx, lc in self.manager._life_cycles.items():
            if lc_idx == ssm_lc_id:
                assert ssm_lc_id is not None
                for beam_idx, beam_block in typed_enumerate(self._ssm_blocks):
                    if beam_block[ssm_lc_id] is not None:
                        yield BAD_BLOCK_ORDINAL, beam_idx, lc_idx
                continue
            stale_start, stale_end = _KVCache._get_stale_range(
                self.tokens_per_block, self.history_length, lc
            )
            scratch_range = self._get_scratch_range(lc)
            sink_blocks = typed_range(stale_start)
            window_blocks = typed_range(stale_end, typed_len(self._blocks))
            for ordinal in chain(sink_blocks, window_blocks):
                block = self._blocks[ordinal]
                for beam_idx, beam_block in typed_enumerate(block.pages):
                    is_scratch = ordinal in scratch_range
                    assert is_scratch == (beam_block[lc_idx] is None)
                    if not is_scratch:
                        yield ordinal, beam_idx, lc_idx

    @property
    def status(self) -> _Status:
        return self._status

    @property
    def is_active(self) -> bool:
        return self.status == self.Status.ACTIVE

    @property
    def tokens_per_block(self) -> int:
        return self._tokens_per_block

    def _page(
        self, block_ordinal: BlockOrdinal, beam_index: BeamIndex, life_cycle: LifeCycleId
    ) -> BlockPage:
        """Return the page holder for an attention block or the SSM block."""
        is_ssm = block_ordinal == BAD_BLOCK_ORDINAL
        assert (life_cycle == self.manager._life_cycles.ssm_life_cycle_id) == is_ssm
        return (
            self._ssm_blocks[beam_index][life_cycle]
            if is_ssm
            else self._blocks[block_ordinal].pages[beam_index][life_cycle]
        )

    def _block(
        self, block_ordinal: BlockOrdinal, beam_index: BeamIndex
    ) -> TypedIndexList[LifeCycleId, BlockPage]:
        """Return the life-cycle page list for an attention block or the SSM block."""
        is_ssm = block_ordinal == BAD_BLOCK_ORDINAL
        return (
            self._ssm_blocks[beam_index]
            if is_ssm
            else self._blocks[block_ordinal].pages[beam_index]
        )

    def _snapshot_ssm_to_tree_block(
        self, tree_block: Block, ssm_lc_id: LifeCycleId, beam_idx: BeamIndex
    ) -> None:
        """Copy live SSM state to a new page and attach it to the radix tree block."""
        storage = self.manager._storage
        ssm_lock = expect_type(_SharedPageLock, self._ssm_blocks[beam_idx][ssm_lc_id])
        src_page = ssm_lock.page
        pg_idx = storage.get_pool_group_index(ssm_lc_id)
        # Try to find a slot in any cache level, starting from the source page's level
        for i in range(storage.num_cache_levels):
            lvl = CacheLevel(i + src_page.cache_level)
            try:
                new_slot = storage.new_slots_for_pool_group(lvl, pg_idx, 1)[0]
            except OutOfPagesError:
                continue
            except Exception:
                raise
            cuda_stream = self.cuda_stream
            new_slot.ready_event.wait_in_stream(cuda_stream)
            slot_size = storage.slot_size(pg_idx)
            for p in typed_range(storage.num_pools(pg_idx)):
                dst = storage.slot_address(lvl, pg_idx, new_slot.slot_id, p)
                src = storage.slot_address(src_page.cache_level, pg_idx, src_page.slot_id, p)
                batched_copy(
                    storage.cache_tiers[lvl],
                    storage.cache_tiers[src_page.cache_level],
                    slot_size[p],
                    [CopyTask(dst, src)],
                    cuda_stream,
                )
            ready_event = CachedCudaEvent(cuda_stream)
            assert self.tokens_per_block * (tree_block.ordinal + 1) == self.num_committed_tokens
            temp_page = UncommittedPage(
                self, tree_block.ordinal, ssm_lc_id, lvl, new_slot, beam_idx
            )
            committed = temp_page.convert_to_committed(tree_block, ready_event)
            # The tree only holds a weak rawref to the page. Schedule for eviction so the
            # eviction controller keeps a strong reference, preventing the page from being GC'd.
            storage.schedule_for_eviction(committed)
            break  # success
        else:
            return  # No pages available in any level, silently skip snapshot

    def _commit_block(self, ordinal: BlockOrdinal, is_last: bool) -> None:
        "Commit the block for reuse. Block must be full of tokens except for the last block."
        assert self._commit_state == self.CommitState.ALLOWED
        assert (
            ordinal == self._num_committed_blocks or self._commit_state != self.CommitState.ALLOWED
        )
        seq_block = self._blocks[ordinal]
        assert typed_len(seq_block.pages) == 1, "Must have 1 beam only"
        beam_idx = DEFAULT_BEAM_INDEX
        beam_block = seq_block.pages[beam_idx]
        tokens_per_block = self.tokens_per_block
        start = ordinal * tokens_per_block
        tokens = self._committed_tokens[start : start + tokens_per_block]
        num_tokens = len(tokens)
        is_full = num_tokens == tokens_per_block
        if not is_last and not is_full:
            raise LogicError("Cannot commit block that is not full except last block")
        prev: RootBlock | Block
        if ordinal == 0:
            prev = self.manager._radix_tree.add_or_get_existing(self._reuse_scope)
        else:
            prev = self._get_tree_block(BlockOrdinal(ordinal - 1))
        try:
            tree_block = Block(tokens, prev)
            is_new = True
        except UselessBlockError as e:
            tree_block = e.block
            assert tree_block.tokens[:num_tokens] == tokens
            is_new = False

        assert tree_block.tokens_per_block == tokens_per_block
        ssm_lc_id = self.manager._life_cycles.ssm_life_cycle_id
        if is_new:
            # We are the only writer to padding. Other _KVCache reusing it should make copies.
            skip_lcs = {ssm_lc_id} if ssm_lc_id is not None else None
            uncommitted_pages = self._take_uncommitted_page(ordinal, beam_idx, skip_lcs)
            # convert uncommitted pages to committed pages and create a new block in the radix tree.
            for lc, (page, locked) in typed_enumerate(uncommitted_pages):
                if page is None:
                    continue
                p = page.convert_to_committed(tree_block, self.finish_event)
                tree_block.storage[lc] = rawref.ref(p)
                # The page comes from uncommitted page of self, so safe to skip wait.
                beam_block[lc] = (
                    p.lock(self, beam_idx, ordinal, lc, skip_wait=True) if locked else p.hold()
                )
            # SSM snapshot: copy live SSM state at interval boundaries.
            # The live SSM state corresponds to num_committed_tokens (updated
            # before _commit_block is called), so snapshot only on the block
            # whose end equals num_committed_tokens and that count is a
            # non-zero multiple of the reuse interval.
            if ssm_lc_id is not None:
                num_committed = self.num_committed_tokens
                block_end = (ordinal + 1) * tokens_per_block
                if (
                    block_end == num_committed
                    and num_committed % self.manager.ssm_reuse_interval == 0
                ):
                    self._snapshot_ssm_to_tree_block(tree_block, ssm_lc_id, beam_idx)
                else:
                    tree_block.storage[ssm_lc_id] = None
            seq_block.tree_block = tree_block
            assert self._get_tree_block(ordinal) is tree_block
            self._num_committed_blocks = BlockOrdinal(ordinal + 1)
            event_manager = self.manager.event_manager
            if event_manager is not None:
                event_manager.add_stored_block_event_from_block(tree_block)
            arena_cfg = self.manager._storage.arena_config
            if (
                self._arena_ranges is not None
                and arena_cfg is not None
                and arena_cfg.write_through == WriteThroughPolicy.ON_COMMIT
                and self.manager._storage.num_cache_levels > 1
            ):
                self._arena_write_through(ordinal)
        elif tree_block.is_full and self.manager.allow_seq_rebasing and is_full:
            # Happens when a concurrent request committed the same tokens before us.
            # Try to replace our pages with pages from the existing block to save memory.
            reuse_list = list[tuple[LifeCycleId, CommittedPage]]()
            for lc in typed_range(typed_len(beam_block)):
                if lc == ssm_lc_id:
                    continue  # SSM pages are not rebased
                if beam_block[lc] is None:
                    continue
                existing_page = map_optional(tree_block.storage[lc], lambda p: p())
                locked = isinstance(beam_block[lc], _SharedPageLock)
                if existing_page is None:
                    # The reusable page is gone. We put our own page into the tree block.
                    page = cast(UncommittedPage, cast(_SharedPageLock, beam_block[lc]).page)
                    beam_block[lc] = None
                    p = page.convert_to_committed(tree_block, self.finish_event)
                    event_manager = self.manager.event_manager
                    if event_manager is not None:
                        event_manager.add_stored_life_cycle_event_from_block(tree_block, int(lc))
                    # The page comes from uncommitted page of self, so safe to skip wait.
                    beam_block[lc] = (
                        p.lock(self, beam_idx, ordinal, lc, skip_wait=True) if locked else p.hold()
                    )
                else:
                    if locked:
                        beam_block[lc] = cast(_SharedPageLock, beam_block[lc]).holder
                    reuse_list.append((lc, existing_page))
            locks = batched_lock_to_gpu(
                self,
                [BatchedLockTarget(p, beam_idx, ordinal, lc) for lc, p in reuse_list],
                self._record_migrated_slots,
            )
            for (lc, _), lock in zip(reuse_list, locks):
                beam_block[lc] = lock
            seq_block.tree_block = tree_block
            assert self._get_tree_block(ordinal) is tree_block
            self._num_committed_blocks = BlockOrdinal(ordinal + 1)
        elif tree_block.is_full and is_full and self._arena_ranges is not None:
            # Arena mode: another sequence committed the same tokens first
            # (common for shared system prompts under concurrency). Rebasing
            # onto its pages would share GPU pages across arenas (§4.4
            # invariant), so keep our own pages as sequence-private committed
            # copies referencing the existing tree block, and keep committing.
            skip_lcs = {ssm_lc_id} if ssm_lc_id is not None else None
            uncommitted_pages = self._take_uncommitted_page(ordinal, beam_idx, skip_lcs)
            for lc, (page, locked) in typed_enumerate(uncommitted_pages):
                if page is None:
                    continue
                p = page.convert_to_private_committed(tree_block, self.finish_event)
                # The page comes from an uncommitted page of self, so safe to skip wait.
                beam_block[lc] = (
                    p.lock(self, beam_idx, ordinal, lc, skip_wait=True) if locked else p.hold()
                )
            seq_block.tree_block = tree_block
            assert self._get_tree_block(ordinal) is tree_block
            self._num_committed_blocks = BlockOrdinal(ordinal + 1)
        else:
            # We can't commit and can't reuse existing block. Just stop committing.
            self._commit_state = self.CommitState.VIRTUAL_STOP

        if seq_block.is_committed:
            for lc_idx, lc in self.manager._life_cycles.attention_life_cycles():
                stale_range = _KVCache._get_stale_range(tokens_per_block, self.history_length, lc)
                if ordinal in stale_range:
                    for beam_block in seq_block.pages:
                        beam_block[lc_idx] = None

        if is_last or self._commit_state == self.CommitState.VIRTUAL_STOP:
            self._commit_state = self.CommitState.USER_STOP
            self._on_stop_committing()

    def _on_stop_committing(self) -> None:
        # If there are stale held uncommitted pages, release them.
        # @TODO: add test for this.
        ssm_lc_id = self.manager._life_cycles.ssm_life_cycle_id
        for lc_idx, lc in self.manager._life_cycles.items():
            if lc_idx == ssm_lc_id:
                continue  # SSM pages live in _ssm_blocks, not in _blocks
            start, end = _KVCache._get_stale_range(self.tokens_per_block, self.history_length, lc)
            start = max(start, self._num_committed_blocks)
            for ordinal in typed_range(start, end):
                block = self._blocks[ordinal]
                assert not block.is_committed
                for beam_block in block.pages:
                    if beam_block[lc_idx] is None:
                        assert self.enable_swa_scratch_reuse
                        continue  # Scratch block — already handled
                    assert isinstance(beam_block[lc_idx], _PageHolder)
                    beam_block[lc_idx] = None
        assert NDEBUG or self._check_sanity()

    def _unlock_stale_blocks(
        self, new_history_length: int
    ) -> list[tuple[BlockOrdinal, BeamIndex, LifeCycleId, _PageHolder]]:
        "For SWA layers, unlock out-of-window blocks."
        if new_history_length == self.history_length:
            return []
        with self._record_event():
            ret = list[tuple[BlockOrdinal, BeamIndex, LifeCycleId, _PageHolder]]()
            ssm_lc_id = self.manager._life_cycles.ssm_life_cycle_id
            for lc_idx, lc in self.manager._life_cycles.items():
                if lc_idx == ssm_lc_id:
                    continue  # SSM pages live in _ssm_blocks, not in _blocks
                if isinstance(lc, AttnLifeCycle) and lc.window_size is None:
                    continue
                _, old_end = _KVCache._get_stale_range(
                    self.tokens_per_block, self.history_length, lc
                )
                new_beg, new_end = _KVCache._get_stale_range(
                    self.tokens_per_block, new_history_length, lc
                )
                for ordinal in typed_range(
                    max(old_end, new_beg), min(typed_len(self._blocks), new_end)
                ):
                    block = self._blocks[ordinal]
                    is_committed = block.is_committed
                    hold_for_commit = (
                        not is_committed and self._commit_state == self.CommitState.ALLOWED
                    )
                    for beam_idx, beam_block in typed_enumerate(block.pages):
                        if beam_block[lc_idx] is None:
                            assert self.enable_swa_scratch_reuse
                            continue  # Scratch block — no page to unlock
                        holder = expect_type(_SharedPageLock, beam_block[lc_idx]).holder
                        ret.append((ordinal, beam_idx, lc_idx, holder))
                        beam_block[lc_idx] = holder if hold_for_commit else None
            # Scratch slot lifetime is handled by resize() after target scratch ranges are recomputed.
        return ret

    def _lock_held_blocks(
        self, backup_holders: list[tuple[BlockOrdinal, BeamIndex, LifeCycleId, _PageHolder]]
    ):
        "Revert _unlock_unused_blocks() by locking the held blocks."
        locks = batched_lock_to_gpu(
            self,
            [
                BatchedLockTarget(holder.page, beam_idx, ordinal, lc)
                for ordinal, beam_idx, lc, holder in backup_holders
            ],
            self._record_migrated_slots,
        )
        for lock in locks:
            user = lock._user
            self._block(user.ordinal, user.beam_index)[user.life_cycle] = lock

    class DeltaScratchSlots(NamedTuple):
        excess: TypedIndexList[LifeCycleId, list[ScratchSlotLock]]
        delta_cnt: TypedIndexList[LifeCycleId, int]
        scratch_ranges: TypedIndexList[LifeCycleId, HalfOpenRange[BlockOrdinal]]

    def _take_excess_scratch_slots(self, capacity: int, history_length: int) -> DeltaScratchSlots:
        """
        Calculate scratch slot requirements and extract excess scratch slots.

        Returns:
            excess_scratch_slots: List of ScratchSlotLocks taken from `self._scratch_slots`.
            additional_scratch_slots: Number of extra slots needed per lifecycle (we have deficit).
            scratch_ranges: The scratch ranges per lifecycle for the new capacity/history_length.
        """
        num_life_cycles = self.manager._life_cycles.size
        excess = make_typed(lambda _: list[ScratchSlotLock](), num_life_cycles)
        delta_cnt = filled_list(0, num_life_cycles)
        scratch_ranges = make_typed(
            lambda _: HalfOpenRange[BlockOrdinal](BlockOrdinal(0), BlockOrdinal(0)), num_life_cycles
        )

        for lc_idx, lc in self.manager._life_cycles.items():
            scratch_range = self._get_scratch_range(lc, history_length, capacity)
            scratch_ranges[lc_idx] = scratch_range
            num_scratch_blocks = len(scratch_ranges[lc_idx])
            frac_max = self._storage._slot_util_frac_max[lc_idx]
            needed_slots = math.ceil(num_scratch_blocks * frac_max)
            existing_slots = len(self._scratch_slots[lc_idx])
            delta = needed_slots - existing_slots
            delta_cnt[lc_idx] = delta

            if delta < 0:
                for _ in range(-delta):
                    lock = self._scratch_slots[lc_idx].pop()
                    excess[lc_idx].append(lock)

        return self.DeltaScratchSlots(excess, delta_cnt, scratch_ranges)

    def _recover_excess_scratch_slots(
        self, excess_scratch_slots: TypedIndexList[LifeCycleId, list[ScratchSlotLock]]
    ) -> None:
        for lc_idx, locks in typed_enumerate(excess_scratch_slots):
            self._scratch_slots[lc_idx].extend(locks)
            locks.clear()

    @property
    def _storage(self) -> StorageManager:
        return self.manager._storage

    @staticmethod
    def _to_block_ordinal(tokens_per_block: int, token_ordinal: int) -> BlockOrdinal:
        return BlockOrdinal(token_ordinal // tokens_per_block)

    def _get_tree_block(self, ordinal: BlockOrdinal) -> Block:
        assert self._blocks[ordinal].is_committed
        ret = unwrap_optional(self._blocks[ordinal].tree_block)
        if not NDEBUG:
            ssm_lc_id = self.manager._life_cycles.ssm_life_cycle_id
            for lc, b in typed_enumerate(self._block(ordinal, DEFAULT_BEAM_INDEX)):
                if lc == ssm_lc_id:
                    assert b is None  # SSM pages live in _ssm_blocks
                elif b is not None:
                    assert isinstance(b.page, CommittedPage) and b.page.block() is ret
        return ret

    def _take_uncommitted_page(
        self,
        ordinal: BlockOrdinal,
        beam_idx: BeamIndex,
        skip_lcs: set[LifeCycleId] | None = None,
    ) -> TypedIndexList[LifeCycleId, tuple[UncommittedPage | None, bool]]:
        """
        Take ownership of the uncommitted pages, together with bool flag indicating if it was locked.
        And reset holders to None. SSM life cycles in skip_lcs are left in place.
        """
        holders = self._block(ordinal, beam_idx)
        num_life_cycles = self.manager._life_cycles.size
        ret: TypedIndexList[LifeCycleId, tuple[UncommittedPage | None, bool]] = filled_list(
            (None, False), num_life_cycles
        )
        for lc, holder in typed_enumerate(holders):
            if holder is None:
                continue
            if skip_lcs and lc in skip_lcs:
                continue
            assert isinstance(holder.page, UncommittedPage)
            locked = isinstance(holder, _SharedPageLock)
            ret[lc] = (holder.page, locked)
            # When using debugpy with breakpoints on exceptions enabled, the lock/holder is not GC'ed even
            # after return from this function. That will likely lead to assertion failures later.
            holders[lc] = None
        return ret

    def _check_sanity(self) -> bool:
        is_closed = self.status == self.Status.CLOSED
        if is_closed:
            return self.num_blocks == 0
        assert self.num_committed_tokens <= self.history_length <= self.capacity
        assert self.num_blocks == div_up(self.capacity, self.tokens_per_block)

        def get_range(lc: LifeCycle):
            return _KVCache._get_stale_range(self.tokens_per_block, self.history_length, lc)

        stale_ranges = typed_map(self.manager._life_cycles.get(), get_range)
        num_life_cycles = self.manager._life_cycles.size
        ssm_lc_id = self.manager._life_cycles.ssm_life_cycle_id
        for ordinal, block in typed_enumerate(self._blocks):
            is_committed = self._never_resumed or ordinal < self._num_committed_blocks
            assert is_committed == block.is_committed
            for beam_block in block.pages:
                assert typed_len(beam_block) == num_life_cycles
                for lc in typed_range(num_life_cycles):
                    holder = beam_block[lc]
                    if lc == ssm_lc_id:
                        # SSM pages live in _ssm_blocks, not in _blocks
                        assert holder is None
                        continue
                    start, end = stale_ranges[lc]
                    if start <= ordinal < end:
                        if is_committed or self._commit_state != self.CommitState.ALLOWED:
                            assert holder is None
                        else:
                            # For the decoder-side disagg case, for the first step, we will skip the
                            # out-of-window blocks.
                            assert isinstance(holder, _PageHolder) or (
                                holder is None and not self._committed_tokens
                            )
                    else:
                        # Scratch blocks have None pages but valid base_page_indices
                        lc_obj = self.manager._life_cycles[lc]
                        sr = self._get_scratch_range(lc_obj)
                        is_scratch = ordinal in sr
                        if is_scratch:
                            assert holder is None
                        else:
                            assert isinstance(
                                holder, (_SharedPageLock if self.is_active else _PageHolder)
                            )
                    if holder is not None:
                        assert is_committed == isinstance(holder.page, CommittedPage)
        # Check SSM blocks
        if ssm_lc_id is not None:
            for beam_block in self._ssm_blocks:
                holder = beam_block[ssm_lc_id]
                if holder is not None:
                    if self._never_resumed:
                        # Deferred copy: SSM holds CommittedPage from matched snapshot
                        assert isinstance(holder, _PageHolder)
                        assert isinstance(holder.page, CommittedPage)
                    else:
                        assert isinstance(holder, _SharedPageLock)
                        assert isinstance(holder.page, UncommittedPage)
        return True

    def _get_scratch_range(
        self,
        life_cycle: LifeCycle,
        history_length_override: int | None = None,
        capacity_override: int | None = None,
    ) -> HalfOpenRange[BlockOrdinal]:
        """
        Range of blocks that should use scratch (shared) slots during SWA prefill.

        Scratch = stale_at_capacity ∩ input_blocks, where:
        - stale_at_capacity: blocks out-of-window when all non-rewindable capacity tokens
          become history.
        - input_blocks: [div_up(history_length, tpb), div_up(capacity, tpb)) — new blocks
          for the current chunk. Blocks before this range already contain real KV data
          from previous chunks and must not be overwritten.

        The configured max_rewind_len excludes a speculative tail from scratch reuse.
        """
        if not self.enable_swa_scratch_reuse:
            return HalfOpenRange(BlockOrdinal(0), BlockOrdinal(0))
        history_length = value_or(history_length_override, self.history_length)
        capacity = value_or(capacity_override, self.capacity)
        max_rewind_len = self._swa_scratch_max_rewind_len()
        return compute_scratch_range(
            life_cycle,
            history_length,
            capacity,
            self.tokens_per_block,
            max_rewind_len,
        )

    def _would_use_swa_scratch_blocks(self) -> bool:
        max_rewind_len = self._swa_scratch_max_rewind_len()
        return any(
            compute_scratch_range(
                lc,
                self.history_length,
                self.capacity,
                self.tokens_per_block,
                max_rewind_len,
            )
            for lc in self.manager._life_cycles
        )

    def _swa_scratch_max_rewind_len(self) -> int:
        return unwrap_optional(self.manager.init_config.swa_scratch_reuse).max_rewind_len

    @staticmethod
    def _get_stale_range(
        tokens_per_block: int,
        history_length: int,
        life_cycle: LifeCycle,
    ) -> HalfOpenRange[BlockOrdinal]:
        """
        Range of the stale blocks. Stale blocks are no longer needed for inference. Stale pages should be
        held if we may commit them later, or droppable otherwise.
        """
        beg, end = life_cycle.get_stale_range(history_length, tokens_per_block)
        return HalfOpenRange(BlockOrdinal(beg), BlockOrdinal(end))

    def _get_matched_tokens(self, match: ReuseMatch) -> list[TokenIdExt]:
        ret: list[TokenIdExt] = []
        remaining = match.num_tokens
        for block in match.blocks:
            assert remaining > 0
            num_block_tokens = min(remaining, len(block.tokens))
            ret.extend(block.tokens[:num_block_tokens])
            remaining -= num_block_tokens
        assert remaining == 0
        return ret

    def _setup_for_reuse(self, match: ReuseMatch) -> None:
        manager = self.manager
        matched = match.blocks
        tokens_per_block = manager.tokens_per_block
        num_tokens = match.num_tokens
        life_cycles = manager._life_cycles
        ssm_lc_id = life_cycles.ssm_life_cycle_id
        self._committed_tokens = self._get_matched_tokens(match)
        self._history_length = num_tokens
        self._capacity = num_tokens
        full_reused_end = BlockOrdinal(num_tokens // tokens_per_block)
        has_partial_match = num_tokens % tokens_per_block != 0
        # fill self._blocks
        self._blocks = to_typed(
            BlockOrdinalT,
            [
                SeqBlock(
                    make_typed(
                        lambda _: filled_list(cast(BlockPage, None), life_cycles.size),
                        self.beam_width,
                    ),
                    block,
                )
                for block in matched
            ],
        )

        beam_idx = DEFAULT_BEAM_INDEX

        should_record_stats = self._should_record_stats()
        for lc_idx, lc in life_cycles.items():
            if lc_idx == ssm_lc_id:
                continue  # SSM is handled separately below
            stale_start, stale_end = _KVCache._get_stale_range(tokens_per_block, num_tokens, lc)
            full_reused_blocks = 0
            partial_reused_blocks = 0
            for ordinal in chain(
                typed_range(stale_start), typed_range(stale_end, BlockOrdinal(len(matched)))
            ):
                block = self._block(ordinal, beam_idx)
                holder = unwrap_rawref(unwrap_optional(matched[ordinal].storage[lc_idx])).hold()
                # For partial blocks (last block, not full), we defer the copy to first resume().
                # Just store the holder of the original committed page for now.
                block[lc_idx] = holder
                if should_record_stats and isinstance(lc, AttnLifeCycle):
                    if ordinal < full_reused_end:
                        full_reused_blocks += 1
                    elif (
                        has_partial_match
                        and ordinal == full_reused_end
                        and self._has_reuse_source(holder)
                    ):
                        partial_reused_blocks = 1
            if should_record_stats and isinstance(lc, AttnLifeCycle):
                changed = self._pending_stats.record_reuse(
                    lc_idx,
                    full_reused_blocks=full_reused_blocks,
                    partial_reused_blocks=partial_reused_blocks,
                )
                if changed:
                    self.manager.mark_stats_dirty(self.id)
        # SSM reuse: hold the snapshot from the last matched block. Copy is deferred to first resume().
        if ssm_lc_id is not None and matched:
            snapshot_block = matched[-1]
            snapshot_ref = snapshot_block.storage[ssm_lc_id]
            assert snapshot_ref is not None, (
                "Last matched block must have SSM snapshot after truncation"
            )
            snapshot_holder = unwrap_rawref(snapshot_ref).hold()
            self._ssm_blocks[DEFAULT_BEAM_INDEX][ssm_lc_id] = snapshot_holder
        self._num_committed_blocks = BlockOrdinal(len(self._committed_tokens) // tokens_per_block)
        for beam_indices in self._base_page_indices:
            for indices in beam_indices:
                if type(indices) is array.array:
                    indices.extend([BAD_PAGE_INDEX] * (self.num_blocks - len(indices)))
                else:
                    assert len(indices) >= self.num_blocks
        if self.manager._storage.is_arena_mode:
            self._arena_lookup_alias_spans(matched, int(full_reused_end))

    def _arena_lookup_alias_spans(self, matched: "Sequence[Block]", full_blocks: int) -> None:
        """Resolve this admission's reuse match against the canonical-span
        registry (P3 prefix aliasing): a hit means the matched prefix's bytes
        are still GPU-resident and refcount-pinned, so the first resume
        aliases them into the fresh ranges (or adopts a signature-matched
        parked range) instead of copying. V1 scope: pool groups with exactly
        one non-SSM life cycle; any staleness simply falls back to the copy
        paths."""
        storage = self._storage
        if not storage.arena_prefix_aliasing or full_blocks <= 0:
            return
        life_cycles = self.manager._life_cycles
        ssm_lc_id = life_cycles.ssm_life_cycle_id
        lc_of_pg: dict[int, "LifeCycleId | None"] = {}
        for lc_idx, _lc in life_cycles.items():
            if lc_idx == ssm_lc_id:
                continue
            pg = int(storage.get_pool_group_index(lc_idx))
            lc_of_pg[pg] = lc_idx if pg not in lc_of_pg else None  # None: multi-lc, skip
        spans: "list[tuple[int, Any] | None]" = [None] * int(storage.num_pool_groups)
        found = False
        for pg, lc_idx in lc_of_pg.items():
            if lc_idx is None:
                continue
            pages: list[CommittedPage] = []
            for ordinal in range(full_blocks):
                ref = matched[ordinal].storage[lc_idx]
                page = None if ref is None else ref()
                if page is None:
                    pages.clear()
                    break
                pages.append(page)
            if not pages:
                continue
            hit = storage.lookup_arena_canonical_span(PoolGroupIndex(pg), pages[0], pages)
            if hit is not None:
                spans[pg] = hit
                found = True
        if found:
            self._alias_spans = spans

    def _free_scratch_slots(self) -> None:
        """Free all scratch slots back to the storage manager."""
        for lc in typed_range(self.manager._life_cycles.size):
            for lock in self._scratch_slots[lc]:
                lock.unlock()
            self._scratch_slots[lc].clear()

    def _clear_blocks(self) -> None:
        # drop the last block first
        while self._blocks:
            self._blocks.pop()
        self._free_scratch_slots()
        ssm_lc_id = self.manager._life_cycles.ssm_life_cycle_id
        if ssm_lc_id is not None:
            for beam_block in self._ssm_blocks:
                beam_block[ssm_lc_id] = None

    @contextmanager
    def _record_event(self) -> Iterator[None]:
        assert self._finish_event is None
        if self._cuda_stream is None:
            # Cache was never resumed — no GPU work was performed,
            # so no CUDA event synchronization is needed.  Blocks
            # only contain _PageHolders (not _SharedPageLocks) and
            # their destructors do not read finish_event.
            yield
            return
        self._finish_event = CachedCudaEvent(self.cuda_stream)
        try:
            yield
        finally:
            self._finish_event = None

    def _update_base_page_index(
        self, beam_idx: BeamIndex, ordinal: BlockOrdinal, lc: LifeCycleId, page_index: PageIndex
    ) -> PageIndex:
        if ordinal == BAD_BLOCK_ORDINAL:
            return PageIndex(BAD_PAGE_INDEX)
        indices = self._base_page_indices[beam_idx][lc]
        old = PageIndex(indices[ordinal])
        indices[ordinal] = page_index
        return old

    def _get_base_page_indices_ref(
        self, lc: LifeCycleId, beam_id: BeamIndex = DEFAULT_BEAM_INDEX
    ) -> Iterator[int | None]:
        assert beam_id < self.beam_width
        assert self.is_active
        return self.get_aggregated_page_indices(lc, beam_id)

    def _shortcut_set_capacity(self, capacity: int) -> bool:
        "Shortcut for cases without side effects. Just for better performance."
        tokens_per_block = self.tokens_per_block
        if div_up(capacity, tokens_per_block) == div_up(self._capacity, tokens_per_block):
            self._capacity = capacity
            return True
        return False

    def _shortcut_set_history_length(self, history_length: int) -> bool:
        "Shortcut for cases without side effects. Just for better performance."
        tokens_per_block = self.tokens_per_block

        def no_side_effect(lc: LifeCycle) -> bool:
            if type(lc) is SsmLifeCycle:
                # history_length change does not impact blocks at all.
                return True
            assert type(lc) is AttnLifeCycle
            window = lc.window_size
            return window is None or lc.get_stale_range(
                history_length, tokens_per_block
            ) == lc.get_stale_range(self.history_length, tokens_per_block)

        if all(no_side_effect(lc) for lc in self.manager._life_cycles):
            self._history_length = history_length
            return True
        return False

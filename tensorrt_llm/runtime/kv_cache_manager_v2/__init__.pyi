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
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Iterable,
    Iterator,
    NamedTuple,
    NewType,
    Protocol,
    Sequence,
    Type,
    TypeAlias,
    Union,
)

# From _common.py
NDEBUG: Final[int]
DEFAULT_BEAM_INDEX: Final[BeamIndex]
BAD_PAGE_INDEX: Final[int]
GPU_LEVEL: Final[CacheLevel]
CACHE_LEVEL1: Final[CacheLevel]

class CorruptedError(Exception):
    """Raised by every public entry point once a broken invariant has been recorded."""

class CuError(Exception):
    """A CUDA driver call failed; carries the driver's own status code."""

    error_code: Any

class OutOfMemoryError(Exception): ...
class OutOfPagesError(OutOfMemoryError): ...

def poison_reason() -> str | None:
    """First recorded invariant violation, or None. Never clears, so it is safe to poll."""

def take_poison() -> str | None:
    """Report the recorded violation and clear it, but only once no manager is alive."""

def num_live_managers() -> int:
    """Number of constructed, not-yet-destroyed managers."""

class CacheTier(enum.IntEnum):
    GPU_MEM = 0
    HOST_MEM = 1
    DISK = 2

class PageStatus(enum.Enum):
    LOCKED = enum.auto()
    HELD = enum.auto()
    DROPPABLE = enum.auto()

class PageIndexMode(enum.IntEnum):
    SHARED = 0
    PER_LAYER = 1

LifeCycleId = NewType("LifeCycleId", int)
LayerGroupId: TypeAlias = LifeCycleId

class AttnLifeCycle:
    """The attention life cycle, keyed by its sliding-window and sink-token shape."""

    @staticmethod
    def make(
        window_size: int | None, num_sink_tokens: int | None, tokens_per_block: int
    ) -> "AttnLifeCycle": ...
    @property
    def window_size(self) -> int | None: ...
    @property
    def num_sink_blocks(self) -> int: ...
    def get_stale_range(self, history_length: int, tokens_per_block: int) -> HalfOpenRange: ...

CacheLevel = NewType("CacheLevel", int)
TokenId = NewType("TokenId", int)
TokenIdExt = Union[TokenId, bytes]

class PlannedDropHandle:
    def drop(self) -> None: ...

class ReuseScope(NamedTuple):
    lora_id: int | None = None
    salt: int | None = None

SlidingWindowSize: TypeAlias = int | None
LayerId = NewType("LayerId", int)
CudaStream = NewType("CudaStream", int)
BeamIndex = NewType("BeamIndex", int)
MemAddress = NewType("MemAddress", int)
Priority = NewType("Priority", int)
PoolGroupIndex = NewType("PoolGroupIndex", int)
PoolIndex = NewType("PoolIndex", int)

# From _storage_manager.py
class StorageStatistics:
    """Independent per-pool storage counts returned by the manager."""

    @property
    def slot_sizes(self) -> list[int]: ...
    @property
    def total(self) -> int: ...
    @property
    def free(self) -> int: ...
    @property
    def evictable(self) -> int: ...
    @property
    def available(self) -> int: ...
    @property
    def unavailable(self) -> int: ...

# From _stats.py
@dataclass(slots=True)
class KVCacheStatsDelta:
    alloc_total_blocks: int = 0
    alloc_new_blocks: int = 0
    reused_blocks: int = 0
    missed_blocks: int = 0

@dataclass(slots=True)
class KVCacheIterationStatsDelta:
    iter_alloc_total_blocks: int = 0
    iter_alloc_new_blocks: int = 0
    iter_reused_blocks: int = 0
    iter_full_reused_blocks: int = 0
    iter_partial_reused_blocks: int = 0
    iter_missed_blocks: int = 0
    iter_gen_alloc_blocks: int = 0
    iter_onboard_blocks: int = 0
    iter_onboard_bytes: int = 0
    iter_offload_blocks: int = 0
    iter_offload_bytes: int = 0
    iter_intra_device_copy_blocks: int = 0
    iter_intra_device_copy_bytes: int = 0
    iter_host_dropped_blocks: int = 0
    iter_host_dropped_bytes: int = 0

@dataclass(slots=True)
class ReusedBlocksByLevel:
    """Reuse block counts split by the cache level the reused pages were resident on.

    Indices are CacheLevel values, so entry i is the i-th configured tier.
    """

    full: list[int]
    partial: list[int]

class SsmSnapshotIterationStatsDelta:
    iter_snapshot_lookups: int = 0
    iter_snapshot_hits: int = 0
    iter_snapshot_misses: int = 0
    iter_reused_tokens: int = 0
    iter_unreused_tokens: int = 0
    iter_aligned_snapshot_hits: int = 0
    iter_unaligned_snapshot_hits: int = 0
    @property
    def iter_snapshot_hit_rate(self) -> float: ...

@dataclass(slots=True, frozen=True)
class PoolGroupPeakBlockStats:
    available: int
    unavailable: int
    evictable: int

class DeviceArray(Protocol):
    """A dense array in GPU memory, borrowed for the duration of a call.

    Structural, not nominal: anything exporting the DLPack protocol over a CUDA buffer
    satisfies it, so a ``torch.Tensor`` is accepted without this package depending on
    torch. Callers may equally pass a CuPy, JAX or numba device array.

    Shape and dtype are stated per parameter rather than in the type, and are checked at
    the boundary. Contents are read or written on the caller's stream; the array must
    stay alive and unmodified until that work completes.
    """

    def __dlpack__(self, *, stream: int | None = ...) -> Any: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

# From _config.py
DataRole = NewType("DataRole", str)

class CacheTierConfig(Protocol):
    quota: int
    @property
    def tier(self) -> CacheTier: ...
    def assert_valid(self) -> None: ...

@dataclass(slots=True)
class GpuCacheTierConfig:
    quota: int
    @property
    def tier(self) -> CacheTier: ...
    def assert_valid(self) -> None: ...

@dataclass(slots=True)
class HostCacheTierConfig:
    quota: int
    @property
    def tier(self) -> CacheTier: ...
    def assert_valid(self) -> None: ...

@dataclass(slots=True)
class DiskCacheTierConfig:
    quota: int
    path: str
    @property
    def tier(self) -> CacheTier: ...
    def assert_valid(self) -> None: ...

@dataclass(slots=True)
class BufferConfig:
    """One buffer of a layer's KV cache.

    ``is_sparse`` marks a buffer whose history is held on the host and disk tiers and
    read by sparse-attention block selection (see :meth:`KVCacheManager.is_sparse`).

    It changes where a page locks, and that is what distinguishes it. An ordinary buffer
    locks every page to ``GPU_LEVEL``. A sparse buffer locks a block holding input tokens
    to ``GPU_LEVEL``, but a pure-history block to ``CACHE_LEVEL1`` -- which must be
    configured with tier ``HOST_MEM``. Configuring level 1 with another tier, or declaring
    fewer than two cache tiers, raises from the ``KVCacheManager`` constructor.

    Because lock location is a per-buffer rule and a page is one slot shared by every
    buffer of its lifecycle, a sparse buffer requires a lifecycle of its own; sharing one
    with a non-sparse buffer raises from the ``KVCacheManager`` constructor.

    The buffer still has a GPU-tier pool, but for a sparse buffer it also carries scratch,
    in the manner of SWA scratch reuse: the per-layer pure-history pages a step actually
    selects are copied up by :meth:`KVCacheManager.fetch_sparse_pages`. That region is
    sized by the selection width, not by sequence length.

    A block leaves ``GPU_LEVEL`` for ``CACHE_LEVEL1`` once it becomes pure history -- once
    its ordinal falls below the watermark set by :meth:`_KVCache.set_history_length` --
    and is re-locked there rather than unlocked. Advancing ``history_length`` is therefore
    what pays the device-to-host copy; it is issued on the owning :class:`_KVCache`'s
    stream. Rewinding ``history_length`` below a block already demoted raises; a demoted
    block is not promoted back.

    A sparse buffer's page-index table is consequently mixed: entries below the history
    watermark are ``CACHE_LEVEL1`` slot indices, entries above it are ``GPU_LEVEL`` ones.
    Only the former may be resolved against the host pool, which is what makes
    "selections name only pure-history ordinals" a precondition of
    :meth:`KVCacheManager.fetch_sparse_pages` rather than a property it can check.

    Fixed at construction: the storage layout derived from it is built once.
    """

    role: DataRole
    size: int
    tokens_per_block_override: int | None = None
    is_sparse: bool = False

@dataclass(slots=True)
class AttentionLayerConfig:
    layer_id: LayerId
    buffers: list[BufferConfig]
    sliding_window_size: int | None = None
    num_sink_tokens: int | None = None
    @property
    def window_size(self) -> int | None: ...

@dataclass(slots=True)
class SsmLayerConfig:
    layer_id: LayerId
    buffers: list[BufferConfig]

LayerConfig = AttentionLayerConfig | SsmLayerConfig

@dataclass(slots=True)
class KVCacheDesc:
    capacity: int
    history_length: int

@dataclass(slots=True)
class BatchDesc:
    kv_caches: list[KVCacheDesc]
    system_prompt_length: int = 0

@dataclass(slots=True)
class SwaScratchReuseConfig:
    max_rewind_len: int = 0

@dataclass(slots=True)
class KVCacheManagerConfig:
    tokens_per_block: int
    cache_tiers: list[CacheTierConfig]
    layers: list[LayerConfig]
    max_util_for_resume: float = ...
    enable_partial_reuse: bool = True
    # Tokens trimmed off the tail of every prefix match; nonzero only for a pool
    # that also holds state reading that many tokens ahead of each position.
    reuse_match_backoff: int = 0
    constraints: list[BatchDesc] = ...
    typical_step: BatchDesc | None = None
    # One positive, normalized hot-tier byte-quota weight per layer group. Cold initialization preserves the implied
    # layer-group slot-count proportions.
    initial_pool_ratio: list[float] | None = None
    swa_scratch_reuse: SwaScratchReuseConfig | None = None
    commit_min_snapshot: bool = False
    enable_stats: bool = True
    text_only: bool = False
    @property
    def enable_swa_scratch_reuse(self) -> bool: ...

# From _event_manager.py
EventBlockHash: TypeAlias = int | str
BlockHashLike: TypeAlias = bytes | EventBlockHash
BlockHashesLike: TypeAlias = BlockHashLike | Iterable[BlockHashLike]
EventTokenId: TypeAlias = int | str
MmKey: TypeAlias = tuple[bytes, int] | tuple[bytes, int, str | None]
AttentionDpGatherFn: TypeAlias = Callable[[list["KVCacheEvent"]], list[list["KVCacheEvent"]]]

@dataclass(slots=True, frozen=True)
class UniqueToken:
    token_id: EventTokenId
    token_extra_id: int = ...

@dataclass(slots=True, frozen=True)
class KVCacheCreatedData:
    num_blocks_per_cache_level: list[int]

@dataclass(slots=True, frozen=True)
class KVCacheStoredBlockData:
    block_hash: EventBlockHash
    tokens: list[UniqueToken]
    cache_level: int
    priority: int
    mm_keys: list[MmKey] = ...
    cache_salt: str | None = ...

@dataclass(slots=True, frozen=True)
class KVCacheStoredData:
    parent_hash: EventBlockHash | None
    blocks: list[KVCacheStoredBlockData]

@dataclass(slots=True, frozen=True)
class KVCacheRemovedData:
    block_hashes: list[EventBlockHash]

@dataclass(slots=True, frozen=True)
class KVCacheEventDiff:
    old_value: int
    new_value: int

@dataclass(slots=True, frozen=True)
class KVCacheUpdatedData:
    block_hash: EventBlockHash
    cache_level: KVCacheEventDiff | None
    priority: KVCacheEventDiff | None

@dataclass(slots=True, frozen=True)
class KVCacheEvent:
    event_id: int
    data: KVCacheCreatedData | KVCacheStoredData | KVCacheRemovedData | KVCacheUpdatedData
    window_size: int
    hash_algo: str | None = None
    attention_dp_rank: int | None = None
    layer_group_id: int | None = None

class KVCacheEventManager:
    def __init__(
        self,
        max_kv_event_entries: int,
        *,
        window_size: int = ...,
        attention_dp_rank: int | None = None,
        attention_dp_gather: AttentionDpGatherFn | None = None,
        hash_algo: str = ...,
        window_size_by_layer_group: dict[int, int] | None = None,
    ) -> None: ...
    def add_created_event(
        self,
        num_blocks_per_cache_level: Sequence[int],
        layer_group_ids: Sequence[int] | None = None,
    ) -> None: ...
    def set_layer_group_window_sizes(self, window_sizes: dict[int, int]) -> None: ...
    def add_stored_event(
        self,
        parent_hash: EventBlockHash | None,
        blocks: Sequence[KVCacheStoredBlockData],
        layer_group_id: int | None = None,
    ) -> None: ...
    def add_stored_block_event_from_block(self, block: Any) -> None: ...
    def add_stored_life_cycle_event_from_block(self, block: Any, life_cycle_id: int) -> None: ...
    def add_removed_event(self, block_hashes: BlockHashesLike) -> None: ...
    def add_removed_life_cycle_event(self, block_hash: bytes, life_cycle_id: int) -> None: ...
    def add_updated_event(
        self,
        block_hash: BlockHashLike,
        *,
        cache_level: KVCacheEventDiff | None = None,
        priority: KVCacheEventDiff | None = None,
        layer_group_id: int | None = None,
    ) -> None: ...
    def flush_iteration_events(self) -> None: ...
    def get_latest_events(self, timeout_ms: float | None = None) -> list[KVCacheEvent]: ...

# Native key builders, shared with the radix tree so routing hashes match the engine's.
def gen_multimodal_cache_key_tokens(
    id_offset: int,
    multi_modal_data_digest: bytes,
    num_tokens: int,
    token_offset: int = 0,
) -> list[TokenIdExt]: ...
def sequence_to_blockchain_keys(
    tokens_per_block: int,
    reuse_scope: ReuseScope,
    tokens: Sequence[TokenIdExt],
) -> Iterator[tuple[list[TokenIdExt], bytes]]: ...

# From _core/_kv_cache.py
class _Status(enum.Enum):
    ACTIVE = enum.auto()
    SUSPENDED = enum.auto()
    CLOSED = enum.auto()

KvCacheStatus: TypeAlias = _Status

IndexSeq = array.array[int] | memoryview[int]

class _KVCache:
    Status: ClassVar[Type[_Status]]
    id: Any
    def __init__(
        self,
        manager: "KVCacheManager",
        reuse_scope: ReuseScope,
        reuse_match: Any | None,
        id: Any,
        custom_priority_callback: Callable[[int, Any], Priority],
        expected_prompt_length: int | None = None,
        text_only: bool | None = None,
        enable_request_stats: bool = False,
    ) -> None: ...
    def set_base_page_index_buf(
        self, beam_idx: BeamIndex, layer_group_id: LayerGroupId, buf: memoryview | None
    ) -> None: ...
    @property
    def manager(self) -> "KVCacheManager": ...
    @property
    def cuda_stream(self) -> CudaStream: ...
    @cuda_stream.setter
    def cuda_stream(self, cuda_stream: CudaStream) -> None: ...
    @property
    def num_blocks(self) -> int: ...
    def commit_pending_stats(self) -> KVCacheStatsDelta: ...
    def discard_pending_stats(self) -> None: ...
    def close(self) -> None: ...
    @property
    def beam_width(self) -> BeamIndex: ...
    @beam_width.setter
    def beam_width(self, beam_width: BeamIndex) -> None: ...
    def get_base_page_indices(
        self, layer_group_id: LayerGroupId, beam_id: BeamIndex = DEFAULT_BEAM_INDEX
    ) -> IndexSeq: ...
    def get_aggregated_page_indices(
        self,
        layer_group_id: LayerGroupId,
        beam_id: BeamIndex = DEFAULT_BEAM_INDEX,
        valid_only: bool = False,
    ) -> Iterator[int]: ...
    def resize(self, capacity: int | None, history_length: int | None = None) -> bool: ...
    @property
    def capacity(self) -> int: ...
    @capacity.setter
    def capacity(self, capacity: int) -> None: ...
    @property
    def history_length(self) -> int: ...
    @history_length.setter
    def history_length(self, history_length: int) -> None: ...
    def commit(
        self,
        accepted_input_tokens: Sequence[TokenIdExt],
        beam_search_indices: Sequence[int] | None = None,
        is_end: bool = False,
    ) -> None: ...
    @property
    def num_committed_tokens(self) -> int: ...
    @property
    def cached_tokens_by_level(self) -> list[int]: ...
    def _get_last_cached_token_level(self) -> int | None: ...
    @property
    def committed_tokens(self) -> list[TokenIdExt]: ...
    @property
    def reuse_scope(self) -> ReuseScope: ...
    def plan_committed_block_drop(self) -> PlannedDropHandle | None: ...
    def stop_committing(self) -> None: ...
    def suspend(self) -> None: ...
    def resume(self, cuda_stream: CudaStream | None = None) -> bool: ...
    def prefetch(self, target: CacheLevel) -> bool: ...
    def get_scratch_desc(self, layer_group_id: LayerGroupId) -> ScratchDesc | None: ...
    @property
    def has_scratch_slots(self) -> bool: ...
    @property
    def enable_swa_scratch_reuse(self) -> bool: ...
    @enable_swa_scratch_reuse.setter
    def enable_swa_scratch_reuse(self, enable: bool) -> None: ...
    @property
    def text_only(self) -> bool: ...
    @text_only.setter
    def text_only(self, text_only: bool) -> None: ...
    def supports_index_mode(self, mode: PageIndexMode) -> bool: ...
    @property
    def status(self) -> _Status: ...
    @property
    def is_active(self) -> bool: ...
    @property
    def tokens_per_block(self) -> int: ...

@dataclass(slots=True, frozen=True)
class PoolDesc:
    pool_index: PoolIndex
    base_address: MemAddress
    slot_bytes: int

class BufferId(NamedTuple):
    layer_id: LayerId
    role: DataRole

@dataclass(slots=True, frozen=True)
class ExpandedBuffer:
    id: BufferId
    expansion: int  # expansion factor of page due to heterogeneous tokens_per_block

@dataclass(slots=True, frozen=True)
class AggregatedPageDesc:
    """The data you need would be in the following byte ranges.

    (base + stride * i + Range(0, size) for i in aggregated_page_indices)
    """

    base: MemAddress
    size: int
    stride: int
    layer_group_id: LayerGroupId
    buffers: Sequence[ExpandedBuffer]

@dataclass(slots=True, frozen=True)
class CoalescedBuffer:
    single_buffer_size: int
    buffer_ids: Sequence[BufferId]
    @property
    def size(self) -> int: ...
    @property
    def num_buffers(self) -> int: ...

@dataclass(slots=True, frozen=True)
class SlotDescVariant:
    coalesced_buffers: Sequence[CoalescedBuffer]
    @property
    def layer_group_id(self) -> LayerGroupId: ...
    @property
    def slot_size_list(self) -> Sequence[int]: ...

@dataclass(slots=True, frozen=True)
class SlotDesc:
    variants: Sequence[SlotDescVariant]
    @property
    def slot_size_list(self) -> Sequence[int]: ...

@dataclass(slots=True, frozen=True)
class PoolGroupDesc:
    pool_group_index: PoolGroupIndex
    num_slots: int
    slot_desc: SlotDesc
    pools: Sequence[PoolDesc]

# From _core/_kv_cache_manager.py
class HalfOpenRange:
    def __init__(self, beg: int, end: int) -> None: ...
    @property
    def beg(self) -> int: ...
    @property
    def end(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

@dataclass(slots=True, frozen=True)
class ScratchDesc:
    range: HalfOpenRange
    slot_ids: Sequence[int]
    def __bool__(self) -> bool: ...

@dataclass(slots=True, frozen=True)
class PageIndexConverter:
    scale: int
    expansion: int
    layer_offset: int

    def __call__(
        self,
        base_indices: Sequence[int],
        index_mode: PageIndexMode | None = None,
        scratch: ScratchDesc | None = None,
    ) -> list[int]: ...

class IKvCacheColdPageCodec: ...

def create_default_kv_cache_cold_page_codec() -> IKvCacheColdPageCodec:
    """Create the default lossless cold-page codec.

    Passing ``cold_page_codec=None`` to ``KVCacheManager`` already selects this codec, so normal users do not need to
    call this factory. It is primarily provided to demonstrate how a native codec factory exposes an owning
    ``IKvCacheColdPageCodec`` object for transfer into ``KVCacheManager``. Any ``KVCacheManager`` construction attempt
    consumes an explicitly supplied codec, including an attempt that fails.
    """

class KVCacheManager:
    def __init__(
        self,
        config: KVCacheManagerConfig,
        event_manager: KVCacheEventManager | None = None,
        cold_page_codec: IKvCacheColdPageCodec | None = None,
    ) -> None: ...
    def __del__(self) -> None: ...
    def shutdown(self) -> None: ...
    def clear_reusable_blocks(self) -> None: ...
    def get_mem_pool_base_address(
        self, layer_id: LayerId, data_role: DataRole, index_mode: PageIndexMode | None = None
    ) -> MemAddress: ...
    def is_sparse(self, layer_id: LayerId, data_role: DataRole) -> bool:
        """Whether this buffer was declared ``BufferConfig.is_sparse``.

        A sparse buffer's pages do not all lock in one place. A block still holding input
        tokens locks to ``GPU_LEVEL``; once it falls below the history watermark it is
        re-locked to ``CACHE_LEVEL1``. Its pages are therefore split across two levels at
        any moment, and :meth:`_KVCache.get_base_page_indices` reports each in its own
        level's numbering -- so the answer is a property of the block, not of the buffer,
        and this reports only which rule applies. An ordinary buffer locks every page to
        ``GPU_LEVEL``.

        Neither level is what a caller addresses to read a page staged by
        :meth:`fetch_sparse_pages`. Those live in the ``GPU_LEVEL`` pool's scratch region,
        reached through the unchanged :meth:`get_mem_pool_base_address`.
        """

    def get_page_stride(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def get_page_index_upper_bound(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def copy_base_page_indices_to_device(
        self,
        kv_caches: Sequence[_KVCache],
        layer_group_id: LayerGroupId,
        out: DeviceArray,  # int32[batch, max_blocks_per_seq]
        stream: CudaStream,
        beam_id: BeamIndex = DEFAULT_BEAM_INDEX,
    ) -> None:
        """Gather a batch's base page indices for one layer group into a device array.

        Row b of ``out`` is the table :meth:`_KVCache.get_base_page_indices` reports for
        ``kv_caches[b]``, padded to the array's width with ``BAD_PAGE_INDEX``. Rows are
        positional, so the caller fixes the batch order here and every later call --
        :meth:`fetch_sparse_pages` above all -- inherits it.

        Indices are reported as stored: level-relative, unscaled, and with
        ``BAD_PAGE_INDEX`` preserved rather than folded to a safe slot. For a sparse
        buffer that is what makes the table usable, since a pure-history entry and an
        input-token entry are numbered in different levels' pools and only the sentinel
        distinguishes an absent block from slot zero.

        The source is host memory, so this enqueues a transfer on ``stream`` and returns.
        The table is only valid until the batch's locks next change -- a committed block,
        a beam fork, an SWA recycle, a ``history_length`` advance -- so it belongs in the
        same per-step preparation as the dense path's block offsets.
        """

    def fetch_sparse_pages(
        self,
        buffers: Sequence[BufferId],
        page_table: DeviceArray,  # int32[batch, max_blocks_per_seq]
        selected: DeviceArray,  # int32[batch, topk]
        num_blocks: DeviceArray,  # int32[batch]
        out: DeviceArray,  # int32[batch, max_blocks_per_seq]
        stream: CudaStream,
    ) -> None:
        """Stage a batch's selected pure-history pages into each buffer's GPU scratch pool.

        Device-driven and stream-ordered throughout: the selection is read from device
        memory, so the host neither inspects it nor decides anything, and the call
        enqueues onto ``stream`` and returns. Order it against the selection kernel by
        sharing a stream, or across streams with an event. Every shape is static across
        steps, so the call is capturable in a CUDA graph.

        ``buffers`` must share one layer group, so that a single ``page_table`` addresses
        them all; each must be ``BufferConfig.is_sparse``.

        ``page_table`` is that group's base page indices as filled by
        :meth:`copy_base_page_indices_to_device`, whose batch order it inherits.
        ``selected`` holds block ordinals, ``topk`` uniform across requests and ``-1``
        past a request's valid count. ``num_blocks`` is how many blocks of each request
        are eligible to be selected, and bounds the walk; it counts blocks rather than
        tokens because selection, staging and ``out`` are all in block units, so no
        request here needs token granularity.

        Eligible means below the history watermark: a block at or above it holds input
        tokens and is locked to ``GPU_LEVEL``, so resolving it against the host pool
        would address unrelated memory. The call cannot detect an out-of-range ordinal,
        so honouring ``num_blocks`` is the selector's responsibility.

        Pages are copied into the scratch region of each buffer's ``GPU_LEVEL`` pool,
        which this manager owns and sizes from the configured selection width. The
        scratch slot for the j-th selection of request b is positional, so no allocation
        happens per call and the mapping is stable under graph replay.

        ``out`` is shaped like ``page_table`` rather than compacted, and is written in
        full: entry ``[b][ord]`` is the ``GPU_LEVEL`` scratch index holding block ``ord``
        when ``ord`` was selected for request b, and ``BAD_PAGE_INDEX`` otherwise. A
        kernel that indexes a page table by block ordinal therefore consumes ``out`` in
        place of ``page_table``, against :meth:`get_mem_pool_base_address` as usual, with
        its selection array unchanged.

        The staged pages are a snapshot. That is sound only because a block below the
        watermark is committed history and immutable.
        """
    def get_page_index_scale(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def get_page_index_converter(
        self, layer_id: LayerId, data_role: DataRole
    ) -> PageIndexConverter: ...
    def create_kv_cache(
        self,
        reuse_scope: ReuseScope | None = None,
        input_tokens: Sequence[TokenIdExt] | None = None,
        id: Any = None,
        custom_priority_callback: Callable[[int, Any], Priority] = ...,
        expected_prompt_length: int | None = None,
        text_only: bool | None = None,
        enable_request_stats: bool = False,
    ) -> _KVCache: ...
    def probe_reuse(
        self,
        reuse_scope: ReuseScope | None = None,
        input_tokens: Sequence[TokenIdExt] | None = None,
    ) -> int: ...
    def resize(self, cache_level: CacheLevel, quota: int, best_efforts: bool = False) -> bool: ...
    def get_quota(self, cache_level: CacheLevel) -> int: ...
    def get_storage_statistics(self, cache_level: CacheLevel = ...) -> list[StorageStatistics]: ...
    def get_life_cycle_pool_group_indices(
        self, cache_level: CacheLevel = ...
    ) -> list[PoolGroupIndex]: ...
    def get_committed_stats(self) -> KVCacheStatsDelta: ...
    def get_and_reset_iteration_stats(self) -> dict[LifeCycleId, KVCacheIterationStatsDelta]: ...
    def get_and_reset_ssm_snapshot_iteration_stats(
        self,
    ) -> dict[LifeCycleId, SsmSnapshotIterationStatsDelta]: ...
    def record_request_suspended(self) -> None: ...
    def record_request_resumed(self) -> None: ...
    def get_and_reset_iteration_suspend_resume_stats(self) -> tuple[int, int]: ...
    def get_and_reset_iteration_disk_prefetch_blocks(self) -> int: ...
    def get_and_reset_iteration_cached_tokens_by_level(self) -> list[int]: ...
    def get_and_reset_iteration_reused_blocks_by_level(
        self,
    ) -> dict[LifeCycleId, ReusedBlocksByLevel]: ...
    def get_and_reset_iteration_peak_block_stats(
        self, cache_level: CacheLevel
    ) -> Sequence[PoolGroupPeakBlockStats]: ...
    def get_and_reset_iteration_peak_block_stats_by_level(
        self,
    ) -> Sequence[Sequence[PoolGroupPeakBlockStats]]: ...
    def mark_stats_dirty(self, kv_cache_id: int | None) -> None: ...
    def clear_stats_dirty(self, kv_cache_id: int | None) -> None: ...
    def get_dirty_stats_kv_cache_ids(self) -> set[int]: ...
    def mark_stats_excluded(self, kv_cache_id: int | None) -> None: ...
    def clear_stats_excluded(self, kv_cache_id: int | None) -> None: ...
    def is_stats_excluded(self, kv_cache_id: int | None) -> bool: ...
    @property
    def cache_tier_list(self) -> Sequence[CacheTier]: ...
    @property
    def tokens_per_block(self) -> int: ...
    @property
    def event_manager(self) -> Any | None: ...
    @property
    def init_config(self) -> KVCacheManagerConfig: ...
    @property
    def allow_seq_rebasing(self) -> bool: ...
    @property
    def enable_partial_match(self) -> bool: ...
    def supports_index_mode(self, mode: PageIndexMode) -> bool | None: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def layer_ids(self) -> Iterator[LayerId]: ...
    def get_layer_group_id(self, layer_id: LayerId) -> LayerGroupId: ...
    @property
    def layer_grouping(self) -> Sequence[Sequence[LayerId]]: ...
    @property
    def all_buffer_ids(self) -> Iterator[BufferId]: ...
    def get_aggregated_pages(self, buffers: Iterable[BufferId]) -> Iterator[AggregatedPageDesc]: ...
    @property
    def pool_group_descs(self) -> Sequence[PoolGroupDesc]: ...
    def clamp_max_seq_len_for_mem(self, batch_size: int, token_num_upper_bound: int) -> int: ...
    def adjust(self) -> None: ...
    @property
    def need_adjustment(self) -> bool: ...
    @property
    def commit_min_snapshot(self) -> bool: ...

def exact_div(x: int, y: int) -> int: ...
def typed_range(*args: int) -> range: ...

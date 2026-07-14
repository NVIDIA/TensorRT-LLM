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

class CacheTier(enum.IntEnum):
    GPU_MEM = 0
    HOST_MEM = 1
    DISK = 2

class PageIndexMode(enum.IntEnum):
    SHARED = 0
    PER_LAYER = 1

LifeCycleId = NewType("LifeCycleId", int)
LayerGroupId: TypeAlias = LifeCycleId
CacheLevel = NewType("CacheLevel", int)
TokenId = NewType("TokenId", int)
TokenIdExt = Union[TokenId, bytes]

class ReuseScope(NamedTuple):
    lora_id: int | None = None
    salt: int | None = None
    def to_bytes(self) -> bytes: ...

LayerId = NewType("LayerId", int)
CudaStream = NewType("CudaStream", int)
BeamIndex = NewType("BeamIndex", int)
MemAddress = NewType("MemAddress", int)
Priority = NewType("Priority", int)
PoolGroupIndex = NewType("PoolGroupIndex", int)

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

@dataclass(slots=True, frozen=True)
class PoolGroupPeakBlockStats:
    available: int
    unavailable: int
    evictable: int

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
    role: DataRole
    size: int
    tokens_per_block_override: int | None = None

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
    vocab_size: int
    cache_tiers: list[CacheTierConfig]
    layers: list[LayerConfig]
    max_util_for_resume: float = ...
    enable_partial_reuse: bool = True
    constraints: list[BatchDesc] = ...
    typical_step: BatchDesc | None = None
    initial_pool_ratio: list[float] | None = None
    swa_scratch_reuse: SwaScratchReuseConfig | None = None
    commit_min_snapshot: bool = False
    enable_stats: bool = True
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

# From _block_radix_tree.py
def gen_multimodal_cache_key_tokens(
    id_offset: int,
    multi_modal_data_digest: bytes,
    num_tokens: int,
    token_offset: int = 0,
) -> list[TokenIdExt]: ...

# From _core/_kv_cache.py
class _Status(enum.Enum):
    ACTIVE = enum.auto()
    SUSPENDED = enum.auto()
    CLOSED = enum.auto()

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
    def finish_event(self) -> Any: ...
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
    def supports_index_mode(self, mode: PageIndexMode) -> bool: ...
    @property
    def status(self) -> _Status: ...
    @property
    def is_active(self) -> bool: ...
    @property
    def tokens_per_block(self) -> int: ...

@dataclass(slots=True, frozen=True)
class MemoryPoolDesc:
    base: MemAddress
    page_size: int

@dataclass(slots=True, frozen=True)
class MemoryPoolGroupDesc:
    num_pages: int
    pools: Sequence[MemoryPoolDesc]

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

# From _core/_kv_cache_manager.py
@dataclass(slots=True, frozen=True)
class ScratchDesc:
    range: tuple[int, int]
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

class KVCacheManager:
    def __init__(
        self,
        config: KVCacheManagerConfig,
        event_manager: KVCacheEventManager | None = None,
    ) -> None: ...
    def __del__(self) -> None: ...
    def shutdown(self) -> None: ...
    def clear_reusable_blocks(self) -> None: ...
    def get_mem_pool_base_address(
        self, layer_id: LayerId, data_role: DataRole, index_mode: PageIndexMode | None = None
    ) -> MemAddress: ...
    def get_page_stride(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def get_page_index_upper_bound(self, layer_id: LayerId, data_role: DataRole) -> int: ...
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
    ) -> _KVCache: ...
    def probe_reuse(
        self,
        reuse_scope: ReuseScope | None = None,
        input_tokens: Sequence[TokenIdExt] | None = None,
    ) -> int: ...
    def resize(self, cache_level: CacheLevel, quota: int, best_efforts: bool = False) -> bool: ...
    def get_quota(self, cache_level: CacheLevel) -> int: ...
    def get_committed_stats(self) -> KVCacheStatsDelta: ...
    def get_and_reset_iteration_stats(self) -> dict[LifeCycleId, KVCacheIterationStatsDelta]: ...
    def get_and_reset_iteration_peak_block_stats(
        self, cache_level: CacheLevel
    ) -> Sequence[PoolGroupPeakBlockStats]: ...
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
    def clamp_max_seq_len_for_mem(self, batch_size: int, token_num_upper_bound: int) -> int: ...
    def adjust(self) -> None: ...
    @property
    def need_adjustment(self) -> bool: ...
    @property
    def commit_min_snapshot(self) -> bool: ...

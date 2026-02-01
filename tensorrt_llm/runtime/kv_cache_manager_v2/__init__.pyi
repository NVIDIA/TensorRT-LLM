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

LifeCycleId = NewType("LifeCycleId", int)
LayerGroupId: TypeAlias = LifeCycleId
CacheLevel = NewType("CacheLevel", int)
TokenId = NewType("TokenId", int)
TokenIdExt = Union[TokenId, bytes]
LayerId = NewType("LayerId", int)
CudaStream = NewType("CudaStream", int)
BeamIndex = NewType("BeamIndex", int)
MemAddress = NewType("MemAddress", int)
Priority = NewType("Priority", int)
PoolGroupIndex = NewType("PoolGroupIndex", int)

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

@dataclass(slots=True)
class HelixConfig:
    helix_group_size: int
    helix_gpu_rank: int
    helix_shard_size: int
    shared_comm_port: int

@dataclass(slots=True)
class AttentionLayerConfig:
    layer_id: LayerId
    buffers: list[BufferConfig]
    sliding_window_size: int | None = None
    num_sink_tokens: int | None = None
    @property
    def window_size(self) -> int | None: ...

@dataclass(slots=True)
class KVCacheManagerConfig:
    tokens_per_block: int
    vocab_size: int
    cache_tiers: list[CacheTierConfig]
    layers: list[AttentionLayerConfig]
    max_util_for_resume: float = ...
    helix_config: HelixConfig | None = None

# From _block_radix_tree.py
def gen_multi_modal_tokens(
    id_offset: int, multi_modal_data_digest: bytes, num_tokens: int
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
        lora_task_id: int | None,
        input_tokens: Sequence[TokenIdExt] | None,
        id: Any,
        custom_priority_callback: Callable[[int, Any], Priority],
    ) -> None: ...
    def set_page_index_buf(
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
    def close(self) -> None: ...
    @property
    def beam_width(self) -> BeamIndex: ...
    @beam_width.setter
    def beam_width(self, beam_width: BeamIndex) -> None: ...
    def get_page_indices(self, layer_group_id: int, beam_id: BeamIndex = ...) -> IndexSeq: ...
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
    ) -> None: ...
    @property
    def num_committed_tokens(self) -> int: ...
    def stop_committing(self) -> None: ...
    def suspend(self) -> None: ...
    def resume(self, cuda_stream: CudaStream | None = None) -> bool: ...
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
class BufferSlice:
    buffer_id: BufferId
    num_slices: int = 1
    slice_index: int = 1

@dataclass(slots=True, frozen=True)
class AggregatedPageDesc:
    """The data you need would be in the following byte ranges.

    (base + stride * i + Range(0, size) for i in aggregated_page_indices)
    """

    base: MemAddress
    size: int
    stride: int
    layer_group_id: LayerGroupId
    buffers: Sequence[BufferSlice]

# From _core/_kv_cache_manager.py
class KVCacheManager:
    def __init__(self, config: KVCacheManagerConfig) -> None: ...
    def clear_reusable_blocks(self) -> None: ...
    def get_mem_pool_base_address(self, layer_id: LayerId, data_role: DataRole) -> MemAddress: ...
    def get_page_stride(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def get_page_index_upper_bound(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def create_kv_cache(
        self,
        lora_task_id: int | None = None,
        input_tokens: Sequence[TokenIdExt] | None = None,
        id: Any = None,
        custom_priority_callback: Callable[[int, Any], Priority] = ...,
    ) -> _KVCache: ...
    def resize(self, cache_level: CacheLevel, quota: int, best_efforts: bool = False) -> bool: ...
    def get_quota(self, cache_level: CacheLevel) -> int: ...
    @property
    def cache_tier_list(self) -> Sequence[CacheTier]: ...
    @property
    def tokens_per_block(self) -> int: ...
    @property
    def allow_seq_rebasing(self) -> bool: ...
    @property
    def enable_partial_match(self) -> bool: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def layer_ids(self) -> Iterator[LayerId]: ...
    def get_layer_group_id(self, layer_id: LayerId) -> LayerGroupId: ...
    @property
    def layer_grouping(self) -> Sequence[Sequence[LayerId]]: ...
    @property
    def all_buffer_ids(self) -> Iterator[BufferId]: ...
    def get_aggregated_pages(
        self, buffers: Iterable[BufferSlice]
    ) -> Iterator[AggregatedPageDesc]: ...
    def clamp_max_seq_len_for_mem(self, batch_size: int) -> int: ...

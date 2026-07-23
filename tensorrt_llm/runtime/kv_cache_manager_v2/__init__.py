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

from . import rawref  # noqa: F401
from ._block_radix_tree import ReuseScope, gen_multimodal_cache_key_tokens  # noqa: F401
from ._common import (  # noqa: F401
    BAD_PAGE_INDEX,
    CACHE_LEVEL1,
    GPU_LEVEL,
    NDEBUG,
    CacheLevel,
    CacheTier,
    CudaStream,
    LayerId,
    MemAddress,
    PageIndexMode,
    PageStatus,
    Priority,
    SlidingWindowSize,
    TokenId,
    TokenIdExt,
)
from ._config import (  # noqa: F401
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    CacheTierConfig,
    DataRole,
    DiskCacheTierConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheDesc,
    KVCacheManagerConfig,
    SsmLayerConfig,
    SwaScratchReuseConfig,
)
from ._core import (  # noqa: F401
    DEFAULT_BEAM_INDEX,
    AggregatedPageDesc,
    BeamIndex,
    ExpandedBuffer,
    KVCacheManager,
    PageIndexConverter,
    PlannedDropHandle,
    PoolDesc,
    PoolGroupDesc,
    PoolGroupPeakBlockStats,
    ScratchDesc,
    _KVCache,
)
from ._core._kv_cache import _Status as KvCacheStatus  # noqa: F401
from ._event_manager import (  # noqa: F401
    KVCacheCreatedData,
    KVCacheEvent,
    KVCacheEventDiff,
    KVCacheEventManager,
    KVCacheRemovedData,
    KVCacheStoredBlockData,
    KVCacheStoredData,
    KVCacheUpdatedData,
    UniqueToken,
)
from ._exceptions import CuError, OutOfMemoryError, OutOfPagesError  # noqa: F401
from ._life_cycle_registry import AttnLifeCycle, LayerGroupId, LifeCycleId  # noqa: F401
from ._stats import (  # noqa: F401
    _KV_CACHE_ITERATION_STATS_DELTA_FIELDS,
    KVCacheIterationStatsDelta,
    KVCacheStatsDelta,
    SsmSnapshotIterationStatsDelta,
)
from ._storage import BufferId  # noqa: F401
from ._storage._config import CoalescedBuffer, SlotDesc, SlotDescVariant  # noqa: F401
from ._storage._core import PoolGroupIndex, PoolIndex  # noqa: F401
from ._utils import HalfOpenRange, exact_div, typed_range  # noqa: F401

__all__ = [
    "AggregatedPageDesc",
    "AttentionLayerConfig",
    "BAD_PAGE_INDEX",
    "CACHE_LEVEL1",
    "BatchDesc",
    "BeamIndex",
    "BufferConfig",
    "BufferId",
    "CoalescedBuffer",
    "CacheLevel",
    "CacheTier",
    "CacheTierConfig",
    "CudaStream",
    "DEFAULT_BEAM_INDEX",
    "DataRole",
    "DiskCacheTierConfig",
    "ExpandedBuffer",
    "GPU_LEVEL",
    "GpuCacheTierConfig",
    "HalfOpenRange",
    "HostCacheTierConfig",
    "KVCacheDesc",
    "KVCacheCreatedData",
    "KVCacheEvent",
    "KVCacheEventDiff",
    "KVCacheEventManager",
    "KVCacheManager",
    "KVCacheManagerConfig",
    "KVCacheRemovedData",
    "KVCacheStoredBlockData",
    "KVCacheStoredData",
    "KVCacheUpdatedData",
    "KvCacheStatus",
    "LayerGroupId",
    "PlannedDropHandle",
    "LayerId",
    "LifeCycleId",
    "MemAddress",
    "NDEBUG",
    "OutOfPagesError",
    "PageIndexConverter",
    "PoolGroupPeakBlockStats",
    "PageIndexMode",
    "PageStatus",
    "PoolDesc",
    "PoolGroupDesc",
    "PoolGroupIndex",
    "PoolIndex",
    "Priority",
    "ReuseScope",
    "ScratchDesc",
    "KVCacheIterationStatsDelta",
    "KVCacheStatsDelta",
    "SlidingWindowSize",
    "SlotDesc",
    "SlotDescVariant",
    "SsmLayerConfig",
    "SsmSnapshotIterationStatsDelta",
    "SwaScratchReuseConfig",
    "TokenId",
    "TokenIdExt",
    "UniqueToken",
    "AttnLifeCycle",
    "CuError",
    "OutOfMemoryError",
    "_KVCache",
    "exact_div",
    "gen_multimodal_cache_key_tokens",
    "rawref",
    "typed_range",
]

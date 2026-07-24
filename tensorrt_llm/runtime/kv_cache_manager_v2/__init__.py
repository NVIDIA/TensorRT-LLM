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

import os
import sys
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import NamedTuple, Optional, Union

_BACKEND = os.environ.get("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp").lower()

if _BACKEND == "python":
    from . import rawref  # noqa: F401
    from ._block_radix_tree import ReuseScope  # noqa: F401
    from ._cache_key import (  # noqa: F401
        gen_multimodal_cache_key_tokens,
        sequence_to_blockchain_keys,
    )
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
    )
    from ._storage import BufferId  # noqa: F401
    from ._storage._config import CoalescedBuffer, SlotDesc, SlotDescVariant  # noqa: F401
    from ._storage._core import PoolGroupIndex, PoolIndex  # noqa: F401
    from ._utils import HalfOpenRange, exact_div, typed_range  # noqa: F401

    _cpp_introspection = None
else:

    class ReuseScope(NamedTuple):
        lora_id: int | None = None
        salt: int | None = None

        def to_bytes(self) -> bytes:
            ret = sum((value is not None) << i for i, value in enumerate(self)).to_bytes(
                1, "little", signed=False
            )
            for value in self:
                if value is not None:
                    ret += value.to_bytes(8, "little", signed=False)
            return ret

    def _load_cpp_module():
        if "tensorrt_llm" in sys.modules:
            from tensorrt_llm.bindings.internal.batch_manager import kv_cache_manager_v2

            return kv_cache_manager_v2

        spec = find_spec("kv_cache_manager_v2")
        assert spec is not None and spec.origin is not None
        trtllm_root = str(Path(spec.origin).parent.parent.parent)
        sys.path.insert(0, trtllm_root)
        try:
            from bindings.internal.batch_manager import kv_cache_manager_v2

            return kv_cache_manager_v2
        finally:
            sys.path.remove(trtllm_root)

    _cpp = _load_cpp_module()

    AggregatedPageDesc = _cpp.AggregatedPageDesc
    AttentionLayerConfig = _cpp.AttentionLayerConfig
    BatchDesc = _cpp.BatchDesc
    BufferConfig = _cpp.BufferConfig
    BufferId = _cpp.BufferId
    CoalescedBuffer = _cpp.CoalescedBuffer
    CacheTier = _cpp.CacheTier
    DiskCacheTierConfig = _cpp.DiskCacheTierConfig
    GpuCacheTierConfig = _cpp.GpuCacheTierConfig
    ExpandedBuffer = _cpp.ExpandedBuffer
    HostCacheTierConfig = _cpp.HostCacheTierConfig
    KVCacheDesc = _cpp.KVCacheDesc
    KVCacheCreatedData = _cpp.KVCacheCreatedData
    KVCacheEvent = _cpp.KVCacheEvent
    KVCacheEventDiff = _cpp.KVCacheEventDiff
    KVCacheEventManager = _cpp.KVCacheEventManager
    KVCacheIterationStatsDelta = _cpp.KVCacheIterationStatsDelta
    KVCacheManager = _cpp.KVCacheManager
    KVCacheManagerConfig = _cpp.KVCacheManagerConfig
    # The C++ KVCacheManagerConfig binding replaces the Python @dataclass, but
    # callers (the DeepSeek-V4 cache manager's _build_cache_config and our own
    # host-tier fallback) use dataclasses.replace() on it. dataclasses.replace()
    # is a free function keyed on __dataclass_fields__: it reads each field via
    # getattr and rebuilds via cls(**fields). The binding already has a full
    # keyword __init__ and readable fields, so we only need to advertise the
    # dataclass field set. Field defaults/types are irrelevant here — replace()
    # only uses the field names + init flag. The read-only
    # enable_swa_scratch_reuse property is intentionally excluded (not a ctor
    # field), matching the Python dataclass.
    import dataclasses as _dataclasses

    @_dataclasses.dataclass
    class _KVCacheManagerConfigFieldSpec:
        tokens_per_block: int = 0
        cache_tiers: object = None
        layers: object = None
        max_util_for_resume: float = 0.97
        enable_partial_reuse: bool = True
        constraints: object = None
        typical_step: object = None
        initial_pool_ratio: object = None
        swa_scratch_reuse: object = None
        commit_min_snapshot: bool = False
        enable_stats: bool = True

    KVCacheManagerConfig.__dataclass_fields__ = _KVCacheManagerConfigFieldSpec.__dataclass_fields__
    del _KVCacheManagerConfigFieldSpec, _dataclasses
    KVCacheRemovedData = _cpp.KVCacheRemovedData
    KVCacheStatsDelta = _cpp.KVCacheStatsDelta
    KVCacheStoredBlockData = _cpp.KVCacheStoredBlockData
    KVCacheStoredData = _cpp.KVCacheStoredData
    KVCacheUpdatedData = _cpp.KVCacheUpdatedData
    KvCacheStatus = _cpp.KvCacheStatus
    OutOfPagesError = _cpp.OutOfPagesError
    PageStatus = _cpp.PageStatus
    PoolDesc = _cpp.PoolDesc
    PoolGroupDesc = _cpp.PoolGroupDesc
    PoolGroupPeakBlockStats = _cpp.PoolGroupPeakBlockStats
    SlotDesc = _cpp.SlotDesc
    SlotDescVariant = _cpp.SlotDescVariant
    SsmLayerConfig = _cpp.SsmLayerConfig
    _KVCache = _cpp._KVCache
    _cpp_introspection = getattr(_cpp, "_introspection", None)
    _KV_CACHE_ITERATION_STATS_DELTA_FIELDS = tuple(KVCacheIterationStatsDelta._field_names)
    PlannedDropHandle = _cpp.PlannedDropHandle

    # Symbols added on main that are not yet ported to the C++ backend.
    # TODO(kvCacheManagerV2-cpp): port these and replace the fallbacks.
    AttnLifeCycle = getattr(_cpp, "AttnLifeCycle", None)
    CuError = getattr(_cpp, "CuError", RuntimeError)
    OutOfMemoryError = getattr(_cpp, "OutOfMemoryError", MemoryError)
    PageIndexConverter = getattr(_cpp, "PageIndexConverter", None)
    ReuseScope = getattr(_cpp, "ReuseScope", ReuseScope)
    ScratchDesc = getattr(_cpp, "ScratchDesc", None)
    SwaScratchReuseConfig = getattr(_cpp, "SwaScratchReuseConfig", None)
    UniqueToken = _cpp.UniqueToken

    BeamIndex = int
    CacheLevel = int
    CacheTierConfig = Union[GpuCacheTierConfig, HostCacheTierConfig, DiskCacheTierConfig]
    CudaStream = int
    DataRole = str
    HalfOpenRange = getattr(_cpp, "HalfOpenRange", tuple)
    LayerGroupId = int
    LayerId = int
    LifeCycleId = int
    MemAddress = int
    PoolGroupIndex = int
    PoolIndex = int
    Priority = int
    SlidingWindowSize = Optional[int]
    TokenId = int
    TokenIdExt = Union[int, bytes]

    BAD_PAGE_INDEX = -1
    DEFAULT_BEAM_INDEX = 0
    GPU_LEVEL = 0
    CACHE_LEVEL1 = 1
    NDEBUG = os.environ.get("TLLM_DEBUG_MODE", "")[0:1] != "1"

    class _RawRef:
        def __init__(self, obj=None):
            self._obj = obj

        def __call__(self):
            return self._obj

        def invalidate(self) -> None:
            self._obj = None

        @classmethod
        def __class_getitem__(cls, _item):
            return cls

    rawref = ModuleType(f"{__name__}.rawref")
    rawref.ReferenceType = _RawRef
    rawref.ref = _RawRef
    rawref.NULL = _RawRef()
    sys.modules.setdefault(f"{__name__}.rawref", rawref)

    class PageIndexMode(int):
        SHARED = 0
        PER_LAYER = 1

    from ._cache_key import (  # noqa: F401
        gen_multimodal_cache_key_tokens,
        sequence_to_blockchain_keys,
    )

    def exact_div(x: int, y: int) -> int:
        assert x % y == 0
        return x // y

    def typed_range(*args: int) -> range:
        return range(*args)


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
    "LayerId",
    "LifeCycleId",
    "MemAddress",
    "NDEBUG",
    "OutOfPagesError",
    "PageIndexConverter",
    "PlannedDropHandle",
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
    "sequence_to_blockchain_keys",
    "rawref",
    "typed_range",
]

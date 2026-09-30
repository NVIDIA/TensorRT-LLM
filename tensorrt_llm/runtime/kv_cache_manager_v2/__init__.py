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

"""Python surface of KVCacheManagerV2.

The implementation lives in C++ under ``cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2``
and is reached through the nanobind module
``tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2``. This package only
re-exports that surface, adds the few plain-Python aliases and constants the bindings do
not carry, and hosts the backend-agnostic ``_introspection`` helpers.
"""

import os
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Optional, Union


def _load_cpp_module():
    if "tensorrt_llm" in sys.modules:
        from tensorrt_llm.bindings.internal.batch_manager import kv_cache_manager_v2

        return kv_cache_manager_v2

    # Dev mode: the package is importable as a top-level ``kv_cache_manager_v2`` (via
    # PYTHONPATH=.../tensorrt_llm/runtime/), so the bindings are reached by walking up to
    # the tensorrt_llm root rather than importing the whole package.
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
AttnLifeCycle = _cpp.AttnLifeCycle
BatchDesc = _cpp.BatchDesc
# BatchDesc is also consumed via dataclasses.replace(): MambaCacheManager's
# _build_cache_config appends dummy KVCacheDesc slots to each constraint with
# replace(batch, kv_caches=[...]). Like KVCacheManagerConfig below, the C++
# binding replaces the Python @dataclass, so advertise the dataclass field
# set (replace() is keyed on __dataclass_fields__: reads fields via getattr,
# rebuilds via BatchDesc(**fields)). The binding already has a keyword
# __init__ and readable kv_caches / system_prompt_length fields.
import dataclasses as _dataclasses_bd  # noqa: E402


@_dataclasses_bd.dataclass
class _BatchDescFieldSpec:
    kv_caches: object = None
    system_prompt_length: int = 0


BatchDesc.__dataclass_fields__ = _BatchDescFieldSpec.__dataclass_fields__
del _BatchDescFieldSpec, _dataclasses_bd
BufferConfig = _cpp.BufferConfig
BufferId = _cpp.BufferId
CoalescedBuffer = _cpp.CoalescedBuffer
CacheTier = _cpp.CacheTier
CorruptedError = _cpp.CorruptedError
CuError = _cpp.CuError
DiskCacheTierConfig = _cpp.DiskCacheTierConfig
GpuCacheTierConfig = _cpp.GpuCacheTierConfig
ExpandedBuffer = _cpp.ExpandedBuffer
HalfOpenRange = _cpp.HalfOpenRange
HostCacheTierConfig = _cpp.HostCacheTierConfig
KVCacheDesc = _cpp.KVCacheDesc
KVCacheCreatedData = _cpp.KVCacheCreatedData
KVCacheEvent = _cpp.KVCacheEvent
KVCacheEventDiff = _cpp.KVCacheEventDiff
KVCacheEventManager = _cpp.KVCacheEventManager
KVCacheIterationStatsDelta = _cpp.KVCacheIterationStatsDelta
KVCacheManager = _cpp.KVCacheManager
KVCacheManagerConfig = _cpp.KVCacheManagerConfig
IKvCacheColdPageCodec = _cpp.IKvCacheColdPageCodec
create_default_kv_cache_cold_page_codec = _cpp.create_default_kv_cache_cold_page_codec
# The C++ KVCacheManagerConfig binding replaces the Python @dataclass, but
# callers (the DeepSeek-V4 cache manager's _build_cache_config and our own
# host-tier fallback) use dataclasses.replace() on it. dataclasses.replace()
# is a free function keyed on __dataclass_fields__: it reads each field via
# getattr and rebuilds via cls(**fields). The binding already has a full
# keyword __init__ and readable fields, so we only need to advertise the
# dataclass field set. Field defaults/types are irrelevant here — replace()
# only uses the field names + init flag. The read-only
# enable_swa_scratch_reuse property is intentionally excluded (not a ctor
# field), matching the binding's constructor.
import dataclasses as _dataclasses  # noqa: E402


@_dataclasses.dataclass
class _KVCacheManagerConfigFieldSpec:
    tokens_per_block: int = 0
    cache_tiers: object = None
    layers: object = None
    max_util_for_resume: float = 0.97
    enable_partial_reuse: bool = True
    reuse_match_backoff: int = 0
    constraints: object = None
    typical_step: object = None
    initial_pool_ratio: object = None
    swa_scratch_reuse: object = None
    commit_min_snapshot: bool = False
    enable_stats: bool = True
    text_only: bool = False


KVCacheManagerConfig.__dataclass_fields__ = _KVCacheManagerConfigFieldSpec.__dataclass_fields__
del _KVCacheManagerConfigFieldSpec, _dataclasses
KVCacheRemovedData = _cpp.KVCacheRemovedData
KVCacheStatsDelta = _cpp.KVCacheStatsDelta
KVCacheStoredBlockData = _cpp.KVCacheStoredBlockData
KVCacheStoredData = _cpp.KVCacheStoredData
KVCacheUpdatedData = _cpp.KVCacheUpdatedData
KvCacheStatus = _cpp.KvCacheStatus
OutOfMemoryError = _cpp.OutOfMemoryError
OutOfPagesError = _cpp.OutOfPagesError
PageIndexConverter = _cpp.PageIndexConverter
PageIndexMode = _cpp.PageIndexMode
PageStatus = _cpp.PageStatus
PlannedDropHandle = _cpp.PlannedDropHandle
PoolDesc = _cpp.PoolDesc
PoolGroupDesc = _cpp.PoolGroupDesc
PoolGroupPeakBlockStats = _cpp.PoolGroupPeakBlockStats
ReuseScope = _cpp.ReuseScope
ReusedBlocksByLevel = _cpp.ReusedBlocksByLevel
ScratchDesc = _cpp.ScratchDesc
SlotDesc = _cpp.SlotDesc
SlotDescVariant = _cpp.SlotDescVariant
SsmLayerConfig = _cpp.SsmLayerConfig
StorageStatistics = _cpp.StorageStatistics
SsmSnapshotIterationStatsDelta = _cpp.SsmSnapshotIterationStatsDelta
SwaScratchReuseConfig = _cpp.SwaScratchReuseConfig
UniqueToken = _cpp.UniqueToken
_KVCache = _cpp._KVCache
_cpp_introspection = getattr(_cpp, "_introspection", None)
_KV_CACHE_ITERATION_STATS_DELTA_FIELDS = tuple(KVCacheIterationStatsDelta._field_names)

gen_multimodal_cache_key_tokens = _cpp.gen_multimodal_cache_key_tokens
num_live_managers = _cpp.num_live_managers
poison_reason = _cpp.poison_reason
sequence_to_blockchain_keys = _cpp.sequence_to_blockchain_keys
take_poison = _cpp.take_poison

BeamIndex = int
CacheLevel = int
CacheTierConfig = Union[GpuCacheTierConfig, HostCacheTierConfig, DiskCacheTierConfig]
CudaStream = int
DataRole = str
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
    "IKvCacheColdPageCodec",
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
    "ReusedBlocksByLevel",
    "KVCacheStatsDelta",
    "SsmSnapshotIterationStatsDelta",
    "SlidingWindowSize",
    "SlotDesc",
    "SlotDescVariant",
    "SsmLayerConfig",
    "SwaScratchReuseConfig",
    "TokenId",
    "TokenIdExt",
    "UniqueToken",
    "AttnLifeCycle",
    "CorruptedError",
    "CuError",
    "OutOfMemoryError",
    "_KVCache",
    "create_default_kv_cache_cold_page_codec",
    "exact_div",
    "gen_multimodal_cache_key_tokens",
    "num_live_managers",
    "poison_reason",
    "sequence_to_blockchain_keys",
    "take_poison",
    "typed_range",
]

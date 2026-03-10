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
from typing import NamedTuple, Optional, Union

_BACKEND = os.environ.get("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp").lower()

if _BACKEND == "python":
    from . import rawref  # noqa: F401
    from ._block_radix_tree import (  # noqa: F401
        BlockRadixTree,
        ReuseScope,
        gen_multimodal_cache_key_tokens,
    )
    from ._common import (  # noqa: F401
        BAD_PAGE_INDEX,
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
        HelixConfig,
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
        ScratchDesc,
        _KVCache,
    )
    from ._core._kv_cache import _Status as KvCacheStatus  # noqa: F401
    from ._exceptions import OutOfPagesError  # noqa: F401
    from ._life_cycle_registry import LayerGroupId, LifeCycleId  # noqa: F401
    from ._storage import BufferId  # noqa: F401
    from ._utils import HalfOpenRange  # noqa: F401
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
    BlockRadixTree = _cpp.BlockRadixTree
    BufferConfig = _cpp.BufferConfig
    BufferId = _cpp.BufferId
    CacheTier = _cpp.CacheTier
    DiskCacheTierConfig = _cpp.DiskCacheTierConfig
    GpuCacheTierConfig = _cpp.GpuCacheTierConfig
    HostCacheTierConfig = _cpp.HostCacheTierConfig
    KVCacheDesc = _cpp.KVCacheDesc
    KVCacheManager = _cpp.KVCacheManager
    KVCacheManagerConfig = _cpp.KVCacheManagerConfig
    KvCacheStatus = _cpp.KvCacheStatus
    OutOfPagesError = _cpp.OutOfPagesError
    PageStatus = _cpp.PageStatus
    SsmLayerConfig = _cpp.SsmLayerConfig
    _KVCache = _cpp._KVCache

    ExpandedBuffer = getattr(_cpp, "ExpandedBuffer", None)
    HelixConfig = getattr(_cpp, "HelixConfig", None)
    PageIndexConverter = getattr(_cpp, "PageIndexConverter", None)
    ReuseScope = getattr(_cpp, "ReuseScope", ReuseScope)
    ScratchDesc = getattr(_cpp, "ScratchDesc", None)
    SwaScratchReuseConfig = getattr(_cpp, "SwaScratchReuseConfig", None)

    BeamIndex = int
    CacheLevel = int
    CacheTierConfig = Union[GpuCacheTierConfig, HostCacheTierConfig, DiskCacheTierConfig]
    CudaStream = int
    DataRole = str
    HalfOpenRange = tuple
    LayerGroupId = int
    LayerId = int
    LifeCycleId = int
    MemAddress = int
    Priority = int
    SlidingWindowSize = Optional[int]
    TokenId = int
    TokenIdExt = Union[int, bytes]

    BAD_PAGE_INDEX = -1
    DEFAULT_BEAM_INDEX = 0
    GPU_LEVEL = 0
    NDEBUG = os.environ.get("TLLM_KV_CACHE_MANAGER_V2_DEBUG", "") == ""
    rawref = None

    class PageIndexMode(int):
        SHARED = 0
        PER_LAYER = 1

    def gen_multimodal_cache_key_tokens(
        id_offset: int, multi_modal_data_digest: bytes, num_tokens: int, token_offset: int = 0
    ) -> list[TokenIdExt]:
        assert num_tokens > 0
        assert token_offset >= 0
        return [
            multi_modal_data_digest
            if token_offset + i == 0
            else TokenId(id_offset + token_offset + i)
            for i in range(num_tokens)
        ]


__all__ = [
    "AggregatedPageDesc",
    "AttentionLayerConfig",
    "BAD_PAGE_INDEX",
    "BatchDesc",
    "BeamIndex",
    "BlockRadixTree",
    "BufferConfig",
    "BufferId",
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
    "HelixConfig",
    "HostCacheTierConfig",
    "KVCacheDesc",
    "KVCacheManager",
    "KVCacheManagerConfig",
    "KvCacheStatus",
    "LayerGroupId",
    "LayerId",
    "LifeCycleId",
    "MemAddress",
    "NDEBUG",
    "OutOfPagesError",
    "PageIndexConverter",
    "PageIndexMode",
    "PageStatus",
    "Priority",
    "ReuseScope",
    "ScratchDesc",
    "SlidingWindowSize",
    "SsmLayerConfig",
    "SwaScratchReuseConfig",
    "TokenId",
    "TokenIdExt",
    "_KVCache",
    "gen_multimodal_cache_key_tokens",
    "rawref",
]

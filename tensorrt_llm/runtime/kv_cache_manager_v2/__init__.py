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

# KVCacheManagerV2 package with backend switching.
#
# Backend selection (env var):
#   TLLM_KV_CACHE_MANAGER_V2_BACKEND=python  -> pure-Python implementation (in this package)
#   TLLM_KV_CACHE_MANAGER_V2_BACKEND=cpp     -> C++ nanobind (default)
#
# Dual-PYTHONPATH trick (cpp backend only):
#   Production:  tensorrt_llm already loaded -> use tensorrt_llm.bindings...
#   Fast dev:    PYTHONPATH=.../runtime/ -> resolve bindings.so via spec.origin

import os
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Optional, Union

_BACKEND = os.environ.get("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp").lower()

if _BACKEND == "python":
    # Pure-Python implementation — relative imports from this package
    from . import rawref  # noqa: F401
    from ._block_radix_tree import BlockRadixTree, gen_multi_modal_tokens  # noqa: F401
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
else:
    from . import rawref  # noqa: F401
    from ._block_radix_tree import gen_multi_modal_tokens  # noqa: F401
    from ._common import PageIndexMode  # noqa: F401
    from ._core import ScratchDesc  # noqa: F401

    # C++ nanobind backend — import heavy classes from bindings
    if "tensorrt_llm" in sys.modules:
        from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2 import (  # noqa: F401
            NDEBUG,
            AggregatedPageDesc,
            AttentionLayerConfig,
            BatchDesc,
            BlockRadixTree,
            BufferConfig,
            BufferId,
            CacheTier,
            DiskCacheTierConfig,
            ExpandedBuffer,
            GpuCacheTierConfig,
            HalfOpenRange,
            HelixConfig,
            HostCacheTierConfig,
            KVCacheDesc,
            KVCacheManager,
            KVCacheManagerConfig,
            KvCacheStatus,
            OutOfPagesError,
            PageIndexConverter,
            PageIndexMode,
            PageStatus,
            ScratchDesc,
            SsmLayerConfig,
            _KVCache,
        )
    else:
        spec = find_spec("kv_cache_manager_v2")
        assert spec is not None and spec.origin is not None
        _trtllm_root = str(Path(spec.origin).parent.parent.parent)
        sys.path.insert(0, _trtllm_root)
        try:
            from bindings.internal.batch_manager.kv_cache_manager_v2 import (  # noqa: F401
                NDEBUG,
                AggregatedPageDesc,
                AttentionLayerConfig,
                BatchDesc,
                BlockRadixTree,
                BufferConfig,
                BufferId,
                CacheTier,
                DiskCacheTierConfig,
                ExpandedBuffer,
                GpuCacheTierConfig,
                HalfOpenRange,
                HelixConfig,
                HostCacheTierConfig,
                KVCacheDesc,
                KVCacheManager,
                KVCacheManagerConfig,
                KvCacheStatus,
                OutOfPagesError,
                PageIndexConverter,
                PageIndexMode,
                PageStatus,
                ScratchDesc,
                SsmLayerConfig,
                _KVCache,
            )
        finally:
            sys.path.remove(_trtllm_root)

    # Type aliases & constants (not provided by C++ bindings)
    BeamIndex = int
    CacheLevel = int
    CacheTierConfig = Union[GpuCacheTierConfig, HostCacheTierConfig, DiskCacheTierConfig]
    CudaStream = int
    DataRole = str
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
    "ScratchDesc",
    "SlidingWindowSize",
    "TokenId",
    "TokenIdExt",
    "_KVCache",
    "gen_multi_modal_tokens",
    "rawref",
]

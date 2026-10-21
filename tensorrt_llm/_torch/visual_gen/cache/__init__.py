# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VisualGen caches: feature-cache acceleration (TeaCache, Cache-DiT) and the paged K/V cache
for block-causal rollouts."""

from .base import CacheAccelerator
from .block_causal_kv_cache import BlockCausalKVCacheManager, ForwardPlan, RolloutSpec
from .cache_dit_accelerator import CacheDiTAccelerator
from .teacache_accelerator import TeaCacheAccelerator

__all__ = [
    "CacheAccelerator",
    "CacheDiTAccelerator",
    "BlockCausalKVCacheManager",
    "ForwardPlan",
    "RolloutSpec",
    "TeaCacheAccelerator",
]

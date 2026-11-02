# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VisualGen caches: feature-cache acceleration (TeaCache, Cache-DiT) and the paged K/V cache
for block-causal rollouts."""

from .base import CacheAccelerator
from .block_causal_kv_cache import BlockCausalKVCacheManager
from .cache_dit_accelerator import CacheDiTAccelerator
from .rollout_kv_cache_interface import ForwardPlan, RolloutKVCacheInterface, RolloutSpec
from .teacache_accelerator import TeaCacheAccelerator

__all__ = [
    "CacheAccelerator",
    "CacheDiTAccelerator",
    "BlockCausalKVCacheManager",
    "ForwardPlan",
    "RolloutKVCacheInterface",
    "RolloutSpec",
    "TeaCacheAccelerator",
]

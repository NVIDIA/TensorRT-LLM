# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-5 sliding-window and sparse-indexer KV cache."""

from .cache_manager import Step5CacheManager
from .params import (
    STEP5_INDEX_KEY_PLANE,
    STEP5_INDEX_KZ_DTYPE,
    STEP5_INDEX_SUMMARY_DTYPE,
    STEP5_INDEX_Z_PLANE,
    Step5IndexerCacheGeometry,
    step5_kv_head_dim,
    step5_text_config,
)

__all__ = [
    "STEP5_INDEX_KEY_PLANE",
    "STEP5_INDEX_KZ_DTYPE",
    "STEP5_INDEX_SUMMARY_DTYPE",
    "STEP5_INDEX_Z_PLANE",
    "Step5CacheManager",
    "Step5IndexerCacheGeometry",
    "step5_kv_head_dim",
    "step5_text_config",
]

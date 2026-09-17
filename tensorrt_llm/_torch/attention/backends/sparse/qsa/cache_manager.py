# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for the Qwen4-owned QSA cache manager."""

from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import QSA_INDEX_POSITION
from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import (
    Qwen4ExpHybridCacheManagerV2 as QSAMambaHybridCacheManagerV2,
)

__all__ = ["QSA_INDEX_POSITION", "QSAMambaHybridCacheManagerV2"]

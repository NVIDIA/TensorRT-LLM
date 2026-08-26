# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Base class for KVCM V2 cold-page quantization."""

from abc import ABC, abstractmethod
from typing import Sequence

from ...pyexecutor.resource_manager import DataType, KVCacheCompressionManager


class ColdPageQuantizationCompression(KVCacheCompressionManager, ABC):
    """Base for storage-bound cold-page quantization implementations."""

    uses_iteration_lifecycle = False
    provides_cold_page_codec = True

    @abstractmethod
    def create_cold_page_codec(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> object:
        """Create a codec using the implementation's immutable calibration."""

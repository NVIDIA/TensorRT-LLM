# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage properties of an individual attention cache layer."""

from dataclasses import dataclass

from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType


@dataclass(frozen=True)
class KVCacheLayer:
    num_kv_heads: int
    head_dim: int
    dtype: DataType
    kv_factor: int = 2
    total_num_kv_heads: int | None = None
    cp_as_tp: bool = False

    def __post_init__(self) -> None:
        if self.num_kv_heads < 0 or self.head_dim < 0:
            raise ValueError("Cache layer dimensions must be nonnegative")
        if self.kv_factor not in (1, 2):
            raise ValueError("KV storage requires one or two planes")
        if self.total_num_kv_heads is not None and self.total_num_kv_heads < self.num_kv_heads:
            raise ValueError("Total KV heads must cover the rank-local heads")

    @property
    def bytes_per_token(self) -> int:
        elements = self.kv_factor * self.num_kv_heads * self.head_dim
        size = get_size_in_bytes(elements, self.dtype)
        if self.dtype == DataType.NVFP4:
            assert elements % 16 == 0, "NVFP4 cache size must be divisible by quant vector size"
            size += elements // 16
        return size

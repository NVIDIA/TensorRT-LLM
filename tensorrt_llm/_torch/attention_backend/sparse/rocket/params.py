# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

import tensorrt_llm
import tensorrt_llm.bindings

if TYPE_CHECKING:
    pass

from ..params import SparseMetadataParams, SparseParams

ModelConfig = tensorrt_llm.bindings.ModelConfig


@dataclass(frozen=True)
class RocketKVMetadataParams(SparseMetadataParams):
    """RocketKV metadata settings derived from user config."""

    prompt_budget: int
    window_size: int
    page_size: int
    topk: int


@dataclass(frozen=True)
class RocketKVParams(SparseParams):
    """RocketKV sparse attention backend parameters."""

    algorithm: Literal["rocket"] = field(init=False, default="rocket")
    window_size: int = 32
    kernel_size: int = 63
    topr: Union[int, float] = 128
    topk: int = 64
    prompt_budget: int = 2048
    page_size: int = 4
    kt_cache_dtype: str = "float8_e5m2"

    @property
    def indices_block_size(self) -> int:
        return self.page_size

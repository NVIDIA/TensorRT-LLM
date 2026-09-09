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
"""Shared sparse attention parameter types."""

from dataclasses import dataclass
from typing import Optional

import torch


class SparseParams:
    """Base parameters for a sparse attention backend."""

    algorithm: str


class SparseMetadataParams:
    """Base parameters for sparse attention metadata."""


@dataclass(kw_only=True, slots=True)
class SparseBackendForwardArgs:
    """Sparse inputs passed from an attention module to its backend."""

    # Shared by algorithms that accept precomputed top-k indices.
    topk_indices: Optional[torch.Tensor] = None


@dataclass(kw_only=True, slots=True)
class SparseRuntimeParams:
    """Flat optional sparse inputs passed from a backend to ``AttentionOp``."""

    # Sparse index inputs shared by multiple algorithms.
    sparse_kv_indices: Optional[torch.Tensor] = None
    sparse_kv_offsets: Optional[torch.Tensor] = None
    sparse_attn_indices: Optional[torch.Tensor] = None
    # Per-query offsets, or backend-specific secondary sparse indices
    # (DeepSeek-V4 fp8_ds_mla compressed-pool indices).
    sparse_attn_offsets: Optional[torch.Tensor] = None
    sparse_attn_indices_block_size: int = 0
    sparse_attn_kv_lens: Optional[torch.Tensor] = None
    aux_kv_cache_pool_ptr: Optional[int] = None

    # SkipSoftmax prefill threshold; kernels divide it by context length.
    threshold_scale_factor_prefill: float = 0.0
    # SkipSoftmax decode threshold; diffusion models leave it at zero.
    threshold_scale_factor_decode: float = 0.0

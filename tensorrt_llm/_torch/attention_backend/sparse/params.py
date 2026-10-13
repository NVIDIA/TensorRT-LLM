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

import os
from dataclasses import dataclass
from typing import Optional

import torch

_INDEXER_MQA_LOGITS_DEFAULT_ELEM_BUDGET = 1 << 31
_INDEXER_MQA_LOGITS_BYTES_PER_ELEMENT = 4


def get_indexer_mqa_logits_elem_budget() -> int:
    """Return the per-call Indexer MQA-logits cap used by the runtime."""
    return int(
        os.environ.get(
            "TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", _INDEXER_MQA_LOGITS_DEFAULT_ELEM_BUDGET
        )
    )


def get_indexer_mqa_logits_workspace_bytes(
    max_num_tokens: Optional[int] = None, max_seq_len: Optional[int] = None
) -> int:
    """Return the reachable maximum bytes for one FP32 MQA-logits tile."""
    elem_budget = get_indexer_mqa_logits_elem_budget()
    if max_num_tokens is not None and max_seq_len is not None:
        elem_budget = min(elem_budget, max_num_tokens * max_seq_len)
    return elem_budget * _INDEXER_MQA_LOGITS_BYTES_PER_ELEMENT


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
    sparse_attn_offsets: Optional[torch.Tensor] = None
    sparse_attn_indices_block_size: int = 0
    sparse_attn_kv_lens: Optional[torch.Tensor] = None
    aux_kv_cache_pool_ptr: Optional[int] = None

    # SkipSoftmax prefill threshold; kernels divide it by context length.
    threshold_scale_factor_prefill: float = 0.0
    # SkipSoftmax decode threshold; diffusion models leave it at zero.
    threshold_scale_factor_decode: float = 0.0
    # Context-phase spcompress (SM107 only).
    uses_spcompress: bool = False

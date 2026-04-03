# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Sparse attention parameters for the C++ attention kernel.

This dataclass holds all sparse-related parameters that are passed to
wrapper.plan() in the trtllm attention backend.  Each sparse attention
method (DSA, RocketKV, SkipSoftmax) populates the relevant fields via
the prepare_sparse_params() interface on TrtllmAttention subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SparseParams:
    """All sparse attention parameters expected by wrapper.plan()."""

    sparse_kv_indices: Optional[torch.Tensor] = None
    sparse_kv_offsets: Optional[torch.Tensor] = None
    sparse_attn_indices: Optional[torch.Tensor] = None
    sparse_attn_offsets: Optional[torch.Tensor] = None
    sparse_attn_indices_block_size: int = 1
    sparse_mla_topk: int = 0
    skip_softmax_threshold_scale_factor_prefill: Optional[float] = None
    skip_softmax_threshold_scale_factor_decode: Optional[float] = None

    def as_dict(self) -> dict:
        """Convert to dict for unpacking into wrapper.plan() kwargs."""
        return {
            "sparse_kv_indices": self.sparse_kv_indices,
            "sparse_kv_offsets": self.sparse_kv_offsets,
            "sparse_attn_indices": self.sparse_attn_indices,
            "sparse_attn_offsets": self.sparse_attn_offsets,
            "sparse_attn_indices_block_size": self.sparse_attn_indices_block_size,
            "sparse_mla_topk": self.sparse_mla_topk,
            "skip_softmax_threshold_scale_factor_prefill": self.skip_softmax_threshold_scale_factor_prefill,
            "skip_softmax_threshold_scale_factor_decode": self.skip_softmax_threshold_scale_factor_decode,
        }

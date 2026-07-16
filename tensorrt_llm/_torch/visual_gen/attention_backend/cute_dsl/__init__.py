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
"""
CuTe DSL attention backend family for visual generation models.

  fmha.py  — CuTeDSLAttention  (dense cubin path, head_dim=128)
  vsa.py   — VSAAttention       (Video Sparse Attention, CuTe JIT + SDPA fallback)
"""

from .fmha import CuTeDSLAttention, _cute_dsl_import_error
from .vsa import (
    VSA_KERNEL_MAX_CUBES,
    VSA_TILE_SIZE,
    VSAAttention,
    VSAMetadata,
    VSAMetadataBuilder,
    VSAPreprocessor,
    get_vsa_forward_context,
    set_vsa_forward_context,
)

__all__ = [
    "CuTeDSLAttention",
    "VSAAttention",
    "VSAMetadata",
    "VSAMetadataBuilder",
    "VSAPreprocessor",
    "VSA_TILE_SIZE",
    "VSA_KERNEL_MAX_CUBES",
    "set_vsa_forward_context",
    "get_vsa_forward_context",
    "_cute_dsl_import_error",
]

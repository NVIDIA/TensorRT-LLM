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
"""Inkling attention: Triton kernels, per-step metadata, backend, cache manager.

Laid out like ``sparse/minimax_m3`` (one module per concern), but deliberately
not under ``sparse/``: Inkling's attention is dense -- full causal on global
layers, a sliding window on local ones -- with a learned relative-bias
``score_mod``.
"""

from .backend import InklingTritonAttention
from .cache_manager import InklingHybridCacheManager
from .kernels import (
    build_page_table,
    inkling_decode_attention,
    inkling_prefill_attention,
    write_kv_cache_hnd,
)
from .metadata import InklingAttentionMetadata

__all__ = [
    "InklingAttentionMetadata",
    "InklingHybridCacheManager",
    "InklingTritonAttention",
    "build_page_table",
    "inkling_decode_attention",
    "inkling_prefill_attention",
    "write_kv_cache_hnd",
]

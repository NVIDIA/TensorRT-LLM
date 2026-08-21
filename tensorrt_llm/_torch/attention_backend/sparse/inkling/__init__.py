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
"""Inkling attention: Triton kernels, backend, cache manager, conv-state pool.

Inkling's attention is dense (full causal, sliding window on local layers, plus a
learned relative-bias ``score_mod``), not sparse. It lives under ``sparse/``
because that is where every model-specific attention backend already lives, and
selection goes through ``sparse/registry.py`` like its neighbours.

The attention metadata adds one thing to the base: the conv pool's per-step slot
write, which has to run outside the CUDA-graph capture region. Everything else the
decode kernel needs is a view of the base's own buffers (``page_table.py``).
"""

from .backend import InklingTritonAttention
from .cache_manager import InklingHybridCacheManager
from .conv_state import (
    CONV_ROLES,
    InklingConvRuntime,
    InklingConvState,
    InklingConvStateCache,
    InklingRole,
    apply_short_conv,
)
from .kernels import (
    build_page_table,
    inkling_decode_attention,
    inkling_prefill_attention,
    write_kv_cache_hnd,
)
from .metadata import InklingAttentionMetadata
from .page_table import gen_page_table, gen_seq_lens, page_div
from .params import (
    InklingBackendForwardArgs,
    InklingSparseAttentionConfig,
    InklingSparseParams,
    inkling_forward_args,
)

__all__ = [
    "InklingAttentionMetadata",
    "InklingBackendForwardArgs",
    "InklingSparseAttentionConfig",
    "InklingSparseParams",
    "InklingConvRuntime",
    "InklingConvState",
    "InklingConvStateCache",
    "InklingRole",
    "InklingHybridCacheManager",
    "InklingTritonAttention",
    "apply_short_conv",
    "build_page_table",
    "gen_page_table",
    "gen_seq_lens",
    "inkling_forward_args",
    "inkling_decode_attention",
    "inkling_prefill_attention",
    "page_div",
    "write_kv_cache_hnd",
]

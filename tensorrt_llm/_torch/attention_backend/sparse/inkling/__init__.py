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

**Inkling is not a sparse-attention algorithm.** Its attention is dense: full
causal on global layers, a sliding window on local ones, plus a learned
relative-bias ``score_mod``. It lives under ``sparse/`` anyway, and that is a
deliberate structural choice rather than a claim about the math:

``sparse/`` is where every *model-specific* attention backend already lives
(``dsa``, ``deepseek_v4``, ``minimax_m3``, ``rocket``). The package name is
historical -- what actually unifies its members is that each needs its own
``AttentionBackend`` + metadata + cache manager, which is exactly Inkling's
shape. Keeping it beside them means one place to look for "models with a
private attention stack" instead of one such package plus a lone top-level
exception.

Inkling differs from its neighbours in how the backend is *selected*: they are
dispatched from ``sparse/registry.py`` on ``SparseParams``, Inkling from its
backend name in ``attention_backend/utils.py``. That is not an oversight; see
``params.py`` for why the registry route was tried and reverted. The one sparse
interface Inkling does reuse is ``SparseBackendForwardArgs``, as the carrier for
``rel_logits`` -- also documented there.

If these mechanisms are ever renamed to something algorithm-neutral (say
``AttentionModuleHooks`` / ``model_backend_args``), Inkling should follow the
rename; nothing here depends on sparsity semantics.
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
from .params import InklingBackendForwardArgs, inkling_forward_args

__all__ = [
    "InklingAttentionMetadata",
    "InklingBackendForwardArgs",
    "InklingConvRuntime",
    "InklingConvState",
    "InklingConvStateCache",
    "InklingRole",
    "InklingHybridCacheManager",
    "InklingTritonAttention",
    "apply_short_conv",
    "build_page_table",
    "inkling_forward_args",
    "inkling_decode_attention",
    "inkling_prefill_attention",
    "write_kv_cache_hnd",
]

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
"""Compatibility shim for ``tensorrt_llm._torch.attention_backend``.

Will be removed once all usages are migrated to
``tensorrt_llm._torch.attention.backends``.

DO NOT ADD ANYTHING TO THIS FILE.
"""

import warnings

from tensorrt_llm._torch.attention.backends import (  # noqa: F401
    AttentionBackend,
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
    TrtllmAttention,
    TrtllmAttentionMetadata,
    VanillaAttention,
    VanillaAttentionMetadata,
    get_sparse_attn_kv_cache_manager,
)
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE

warnings.warn(
    "tensorrt_llm._torch.attention_backend has moved to "
    "tensorrt_llm._torch.attention.backends. The old path still works for "
    "now and will be removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "AttentionMetadata",
    "AttentionBackend",
    "AttentionForwardArgs",
    "AttentionInputType",
    "TrtllmAttention",
    "TrtllmAttentionMetadata",
    "VanillaAttention",
    "VanillaAttentionMetadata",
    "get_sparse_attn_kv_cache_manager",
]

if IS_FLASHINFER_AVAILABLE:
    from tensorrt_llm._torch.attention.backends.flashinfer import (  # noqa: F401
        FlashInferAttention,
        FlashInferAttentionMetadata,
    )

    __all__ += [
        "FlashInferAttention",
        "FlashInferAttentionMetadata",
    ]

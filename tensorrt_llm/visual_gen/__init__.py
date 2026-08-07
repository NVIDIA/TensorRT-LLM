# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Public VisualGen engine API.

Entry-point classes (``VisualGen``, ``VisualGenArgs``, ``VisualGenParams``,
``VisualGenResult``, ``VisualGenOutput``, ``VisualGenMetrics``, ``ExtraParamSchema``)
are also re-exported from ``tensorrt_llm`` at top-level.

Cross-cutting sub-configs live in this sub-package only.
``QuantConfig`` is re-exported for convenience.
"""

from tensorrt_llm.models.modeling_utils import QuantConfig

from .args import (
    AttentionConfig,
    CacheConfig,
    CacheDiTConfig,
    CompilationConfig,
    CudaGraphConfig,
    ParallelConfig,
    QuantAttentionConfig,
    SkipSoftmaxAttentionConfig,
    SparseAttentionConfig,
    TeaCacheConfig,
    TorchCompileConfig,
    VideoSparseAttentionConfig,
    VisualGenArgs,
)
from .output import VisualGenMetrics, VisualGenOutput
from .params import VisualGenParams
from .visual_gen import ExtraParamSchema, VisualGen, VisualGenResult

__all__ = [
    # Entry-point classes (also top-level re-exports)
    "VisualGen",
    "VisualGenArgs",
    "VisualGenParams",
    "VisualGenResult",
    "VisualGenOutput",
    "VisualGenMetrics",
    "ExtraParamSchema",
    # Cross-cutting sub-configs
    "CompilationConfig",
    "CudaGraphConfig",
    "TorchCompileConfig",
    "ParallelConfig",
    "AttentionConfig",
    "QuantAttentionConfig",
    "SparseAttentionConfig",
    "SkipSoftmaxAttentionConfig",
    "VideoSparseAttentionConfig",
    "CacheConfig",
    "TeaCacheConfig",
    "CacheDiTConfig",
    # Re-exported for convenience
    "QuantConfig",
]

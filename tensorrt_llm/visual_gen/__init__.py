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

The surface is loaded lazily (PEP 562). This is not only an import-cost
optimization: the runtime tree under ``tensorrt_llm._torch.visual_gen``
imports config leaf modules from this package (``.args``,
``.sparse_attention``, ``.output``), and importing a leaf always executes
this ``__init__`` first. An eager ``from .visual_gen import VisualGen`` here
would re-enter the still-initializing runtime tree (``visual_gen.py`` needs
``DiffusionRequest`` from it) and fail as a circular import whenever the
runtime tree is imported first. Resolving names on attribute access keeps
this ``__init__`` side-effect free, so both entry orders work.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

# Public name -> providing module.
_LAZY_ATTRS = {
    "VisualGen": "tensorrt_llm.visual_gen.visual_gen",
    "VisualGenResult": "tensorrt_llm.visual_gen.visual_gen",
    "ExtraParamSchema": "tensorrt_llm.visual_gen.visual_gen",
    "VisualGenArgs": "tensorrt_llm.visual_gen.args",
    "AttentionConfig": "tensorrt_llm.visual_gen.args",
    "CacheConfig": "tensorrt_llm.visual_gen.args",
    "CacheDiTConfig": "tensorrt_llm.visual_gen.args",
    "CompilationConfig": "tensorrt_llm.visual_gen.args",
    "CudaGraphConfig": "tensorrt_llm.visual_gen.args",
    "ParallelConfig": "tensorrt_llm.visual_gen.args",
    "QuantAttentionConfig": "tensorrt_llm.visual_gen.args",
    "SkipSoftmaxAttentionConfig": "tensorrt_llm.visual_gen.args",
    "SparseAttentionConfig": "tensorrt_llm.visual_gen.args",
    "TeaCacheConfig": "tensorrt_llm.visual_gen.args",
    "TorchCompileConfig": "tensorrt_llm.visual_gen.args",
    "VideoSparseAttentionConfig": "tensorrt_llm.visual_gen.args",
    "VisualGenMetrics": "tensorrt_llm.visual_gen.output",
    "VisualGenOutput": "tensorrt_llm.visual_gen.output",
    "VisualGenParams": "tensorrt_llm.visual_gen.params",
    "QuantConfig": "tensorrt_llm.models.modeling_utils",
}


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name), name)
        globals()[name] = value  # cache: subsequent access skips __getattr__
        return value
    # Fall back to plain submodules (tensorrt_llm.visual_gen.sparse_attention,
    # ...) that used to be reachable as attributes via the eager import chain.
    try:
        return importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError as e:
        # Only translate "no such submodule" into AttributeError. A
        # ModuleNotFoundError raised *inside* an existing submodule (missing
        # dependency) is a real error and must propagate unchanged.
        if e.name != f"{__name__}.{name}":
            raise
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(__all__) | set(globals()))


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

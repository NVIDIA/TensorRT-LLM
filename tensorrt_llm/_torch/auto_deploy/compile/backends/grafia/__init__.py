# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoDeploy Grafia compile backend."""

from ...compiler import CompileBackendRegistry
from .adapter import GrafiaBackendContext, GrafiaCTMBackendAdapter
from .compiler import GrafiaCompiler as _GrafiaCompiler
from .constants import BACKEND_NAME, GRAFIA_MODES, RMSNORM_OP_KIND
from .deps import (
    _import_ctm_spec_deps,
    _import_grafia_runtime_deps,
    _infer_compile_device,
    _require_cuda_device,
    importlib,
)
from .errors import GrafiaCompileError, GrafiaUnsupportedError
from .metadata import (
    _ctm_dtype_to_torch,
    _dtype_from_meta,
    _get_attr,
    _is_contiguous_meta,
    _node_meta_val,
    _shape_from_meta,
    _torch_dtype_to_ctm,
)
from .plugin import GrafiaBackendPlugin
from .runtime import (
    GrafiaCompiledGraph,
    GrafiaEagerDispatchPolicy,
    GrafiaExecutor,
    GrafiaLoweredArtifact,
    GrafiaRegionModule,
)

if CompileBackendRegistry.has("grafia"):
    GrafiaCompiler = CompileBackendRegistry.get("grafia")
else:
    GrafiaCompiler = CompileBackendRegistry.register("grafia")(_GrafiaCompiler)

__all__ = [
    "BACKEND_NAME",
    "GRAFIA_MODES",
    "RMSNORM_OP_KIND",
    "GrafiaBackendContext",
    "GrafiaCTMBackendAdapter",
    "GrafiaCompileError",
    "GrafiaCompiledGraph",
    "GrafiaCompiler",
    "GrafiaEagerDispatchPolicy",
    "GrafiaExecutor",
    "GrafiaLoweredArtifact",
    "GrafiaBackendPlugin",
    "GrafiaRegionModule",
    "GrafiaUnsupportedError",
    "_ctm_dtype_to_torch",
    "_dtype_from_meta",
    "_get_attr",
    "_import_ctm_spec_deps",
    "_import_grafia_runtime_deps",
    "_infer_compile_device",
    "_is_contiguous_meta",
    "_node_meta_val",
    "_require_cuda_device",
    "_shape_from_meta",
    "_torch_dtype_to_ctm",
    "importlib",
]

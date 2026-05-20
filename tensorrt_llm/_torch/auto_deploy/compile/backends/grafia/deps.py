# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional dependency import helpers for the Grafia backend."""

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch.fx import GraphModule

from .errors import GrafiaCompileError


def _import_ctm_spec_deps():
    try:
        spec_mod = importlib.import_module("backends.ctm.spec")
        types_mod = importlib.import_module("graph.types")
    except ImportError as exc:
        raise GrafiaCompileError(
            "compile_backend='grafia' requires CTM spec modules to be "
            "importable. Add $GRAFIA_ARM/thin_ir to PYTHONPATH."
        ) from exc
    return spec_mod, types_mod


def _import_grafia_runtime_deps():
    try:
        importlib.import_module("grafia_runtime")
        ctm = importlib.import_module("backends.ctm")
    except ImportError as exc:
        raise GrafiaCompileError(
            "compile_backend='grafia' requires Grafia runtime and CTM/ThinIR "
            "modules to be importable. Set GRAFIA_ARM, add "
            "$GRAFIA_ARM/grafia/python and $GRAFIA_ARM/thin_ir to PYTHONPATH, "
            "and set LD_LIBRARY_PATH for the Grafia CUDA toolkit."
        ) from exc
    return ctm


def _require_cuda_device(device: torch.device | str) -> str:
    if not torch.cuda.is_available():
        raise GrafiaCompileError("compile_backend='grafia' requires CUDA to be available")
    device = torch.device(device)
    if device.type != "cuda":
        raise GrafiaCompileError(
            f"compile_backend='grafia' requires a CUDA compile device, got {device}"
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    major, minor = torch.cuda.get_device_capability(index)
    if major < 10:
        raise GrafiaCompileError(
            "compile_backend='grafia' RMSNorm MVP requires a Blackwell-class "
            f"GPU for the sm100 cubin; got sm{major}{minor} on cuda:{index}"
        )
    return f"cuda:{index}"


def _infer_compile_device(gm: GraphModule, compiler_kwargs: dict[str, Any]) -> str:
    for arg in compiler_kwargs.get("args", ()):
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            return _require_cuda_device(arg.device)
    for tensor in list(gm.parameters()) + list(gm.buffers()):
        if tensor.is_cuda:
            return _require_cuda_device(tensor.device)
    return _require_cuda_device(torch.device("cuda"))

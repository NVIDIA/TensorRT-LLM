# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel dispatch for the in-tree Kimi K3 Attention Residual fused op.

The optimized ``attn_res_fwd`` kernel (SM100/B200 CuTe TMA warp-specialised
online-softmax + RMSNorm) lives in the NVIDIA+Moonshot internal
collaboration collection at
``exisiting_optimization_work/Attention_residual`` and is Blackwell sm_100
only. Like the KDA decode extension it is not open-sourced; this dispatch
resolves the kernel via env-configurable paths so the production tree
carries no source copies.

Environment
-----------
``KIMI_K3_ATTN_RES_OPTIMIZED_KERNEL_DIR`` — root of the
``exisiting_optimization_work`` collection (contains an ``Attention_residual``
sub-directory).

``KIMI_K3_ATTN_RES_CUTLASS_DIR`` — CUTLASS root providing ``include/cute``
and ``include/cutlass`` (defaults to the CUTLASS shipped inside
TensorRT-LLM's ``tensorrt_llm/include/trtllm_gen_kernels/fmha/cutlass``).

``KIMI_K3_ATTN_RES_EXT_SO`` — optional path to a pre-built extension
``.so``; when set the loader skips JIT compilation entirely.

Dispatch
--------
On sm_100 with the optimization tree available, ``load_extension``
JIT-builds (or loads the pre-built) fused kernel and returns the raw
``attn_res_fwd_cuda`` PyBind entry point. On any other arch — or when the
optimization tree is unreachable — the dispatch falls back to the pure-
torch chunked reference implemented in
:mod:`kimi_k3_attn_res.kimi_k3_attn_res`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch

try:
    from tensorrt_llm._utils import get_sm_version as _tllm_get_sm_version
except Exception:  # pragma: no cover — source-loader stub path
    _tllm_get_sm_version = None


KIMI_K3_ATTN_RES_OPTIMIZED_KERNEL_ENV = "KIMI_K3_ATTN_RES_OPTIMIZED_KERNEL_DIR"
"""Env variable pointing at the ``exisiting_optimization_work`` root."""

KIMI_K3_ATTN_RES_CUTLASS_ENV = "KIMI_K3_ATTN_RES_CUTLASS_DIR"
"""Env variable pointing at a CUTLASS root with ``include/cute`` and ``include/cutlass``."""

KIMI_K3_ATTN_RES_EXT_SO_ENV = "KIMI_K3_ATTN_RES_EXT_SO"
"""Env variable pointing at a pre-built extension ``.so`` to load directly."""


def _resolve_optimization_root() -> Optional[str]:
    root = os.environ.get(KIMI_K3_ATTN_RES_OPTIMIZED_KERNEL_ENV)
    if not root:
        return None
    p = Path(root)
    if not p.is_dir():
        return None
    if not (p / "Attention_residual" / "csrc" / "pybind.cu").is_file():
        return None
    return str(p)


def _resolve_cutlass_root() -> Optional[str]:
    override = os.environ.get(KIMI_K3_ATTN_RES_CUTLASS_ENV)
    candidates = []
    if override:
        candidates.append(Path(override))
    # Default: CUTLASS shipped inside TensorRT-LLM at
    # tensorrt_llm/include/trtllm_gen_kernels/fmha/cutlass. This module
    # file lives at TRT/tensorrt_llm/_torch/modules/kimi_k3_attn_res/, so
    # walk up three levels to reach tensorrt_llm/, then into include/...
    trt_include = (
        Path(__file__).resolve().parents[3] / "include" / "trtllm_gen_kernels" / "fmha" / "cutlass"
    )
    candidates.append(trt_include)
    for cand in candidates:
        if (cand / "include" / "cute" / "tensor.hpp").is_file() and (
            cand / "include" / "cutlass" / "arch" / "barrier.h"
        ).is_file():
            return str(cand)
    return None


def _default_get_sm_version() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return -1
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor


def get_attn_res_sm_version() -> int:
    """Return the runtime SM version used for kernel selection.

    Prefers ``tensorrt_llm._utils.get_sm_version`` when the real package is
    importable so environment-side overrides propagate. Falls back to a
    plain CUDA-property probe otherwise.
    """
    if _tllm_get_sm_version is not None:
        try:
            return int(_tllm_get_sm_version())
        except Exception:
            return _default_get_sm_version()
    return _default_get_sm_version()


def is_attn_res_optimized_supported() -> bool:
    """The optimized ``attn_res_fwd`` kernel is Blackwell sm_100 only."""
    return get_attn_res_sm_version() in (100, 103)


# ---------------------------------------------------------------------------
# Extension load / JIT build.
# ---------------------------------------------------------------------------

_EXTENSION: Optional[ModuleType] = None


def _nvcc_flags() -> list[str]:
    return [
        "-O3",
        "-std=c++17",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
        "-gencode",
        "arch=compute_100a,code=sm_100a",
    ]


def load_extension(verbose: bool = False) -> ModuleType:
    """Load or JIT-build the ``attn_res_fwd`` extension.

    Preference order:

    1. ``KIMI_K3_ATTN_RES_EXT_SO`` — load a pre-built ``.so`` directly.
    2. JIT build from
       ``$KIMI_K3_ATTN_RES_OPTIMIZED_KERNEL_DIR/Attention_residual/csrc/*.cu``
       using ``torch.utils.cpp_extension.load``, with CUTLASS resolved from
       ``KIMI_K3_ATTN_RES_CUTLASS_DIR`` or the tree-shipped default.

    Raises
    ------
    RuntimeError
        If the current device is not sm_100 or the optimization root is
        not reachable. Callers who want a fallback must check
        :func:`is_attn_res_optimized_supported` and resolve the optimization
        root themselves before calling this.
    """
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    prebuilt = os.environ.get(KIMI_K3_ATTN_RES_EXT_SO_ENV)
    if prebuilt:
        so_path = Path(prebuilt)
        if not so_path.is_file():
            raise FileNotFoundError(f"prebuilt attn_res extension not found at {so_path}")
        spec = importlib.util.spec_from_file_location("kimi_k3_attn_res_ext", str(so_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"could not build import spec for {so_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXTENSION = module
        return _EXTENSION

    root = _resolve_optimization_root()
    if root is None:
        raise RuntimeError(
            f"{KIMI_K3_ATTN_RES_OPTIMIZED_KERNEL_ENV} is unset or does not point at a "
            "directory containing Attention_residual/csrc/pybind.cu."
        )
    if not is_attn_res_optimized_supported():
        raise RuntimeError(
            f"attn_res_fwd kernel is sm_100 only; current SM is {get_attn_res_sm_version()}."
        )
    cutlass_root = _resolve_cutlass_root()
    if cutlass_root is None:
        raise RuntimeError(
            f"CUTLASS root not found. Set {KIMI_K3_ATTN_RES_CUTLASS_ENV} or ensure "
            "the TensorRT-LLM tree ships CUTLASS at "
            "tensorrt_llm/include/trtllm_gen_kernels/fmha/cutlass."
        )

    csrc = Path(root) / "Attention_residual" / "csrc"
    cutlass_include = Path(cutlass_root) / "include"
    cutlass_tools_include = Path(cutlass_root) / "tools" / "util" / "include"

    # Ensure TORCH_CUDA_ARCH_LIST includes sm_100a so torch.utils.cpp_extension
    # does not append conflicting -gencode flags. The nvcc flag list also
    # embeds -gencode arch=compute_100a,code=sm_100a explicitly.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

    # Default TORCH_EXTENSIONS_DIR to a workspace-local path if the caller
    # did not set one, mirroring the KDA_decode dispatch behaviour.
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(Path.cwd() / ".torch_extensions_kimi_k3_attn_res"),
    )

    from torch.utils.cpp_extension import load

    module = load(
        name="kimi_k3_attn_res_ext",
        sources=[
            str(csrc / "pybind.cu"),
            str(csrc / "attn_res_api.cu"),
            str(csrc / "attn_res" / "attn_res_fwd_tma.cu"),
        ],
        extra_include_paths=[
            str(csrc),
            str(csrc / "attn_res"),
            str(cutlass_include),
            str(cutlass_tools_include),
        ],
        extra_cflags=["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"],
        extra_cuda_cflags=_nvcc_flags(),
        verbose=verbose,
    )
    _EXTENSION = module

    # Also register the extension inside a synthetic ``attention_residual``
    # package so the source-shipped ``interface.py`` wrapper (which does
    # ``from . import _C``) can be reused verbatim when callers want it.
    _register_attention_residual_package(module)
    return module


def _register_attention_residual_package(ext: ModuleType) -> None:
    """Expose the loaded extension as ``attention_residual._C`` in sys.modules.

    The source-shipped Python wrapper at
    ``exisiting_optimization_work/Attention_residual/attention_residual/``
    does ``from . import _C`` and then dispatches to ``_C.attn_res_fwd_cuda``.
    Registering our JIT-built extension under that name lets callers reuse
    the shipped ``attn_res_fwd`` wrapper without pointing PYTHONPATH at the
    optimization tree explicitly.

    This is idempotent and skipped if ``attention_residual._C`` is already
    installed by another loader.
    """
    import types

    if "attention_residual._C" in sys.modules:
        return
    # Only create synthetic parent when it does not already exist so that
    # an explicit ``import attention_residual`` (from a PYTHONPATH pointing
    # at the optimization tree) still binds to the source-shipped package.
    if "attention_residual" not in sys.modules:
        pkg = types.ModuleType("attention_residual")
        pkg.__path__ = []
        sys.modules["attention_residual"] = pkg
    sys.modules["attention_residual._C"] = ext
    setattr(sys.modules["attention_residual"], "_C", ext)


def optimized_extension_source() -> Optional[str]:
    """Return the loaded extension's ``__file__`` (if any) for logging."""
    if _EXTENSION is None:
        return None
    return getattr(_EXTENSION, "__file__", None)


def resolve_optimization_root() -> Optional[str]:
    """Public alias for :func:`_resolve_optimization_root`."""
    return _resolve_optimization_root()

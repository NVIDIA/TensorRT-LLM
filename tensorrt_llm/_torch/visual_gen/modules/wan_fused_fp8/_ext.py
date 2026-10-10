# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT-built extensions for the Wan fused FP8 ops, built on first use."""

import functools
import os
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME, load

import tensorrt_llm

_CSRC = Path(__file__).resolve().parent / "csrc"
_CUDA_FLAGS = ["-O3", "-gencode=arch=compute_107a,code=sm_107a", "-std=c++17"]
_TRTLLM_ROOT = Path(tensorrt_llm.__file__).resolve().parent.parent
_DEFAULT_BUILD_DIR = Path.home() / ".cache" / "tensorrt_llm" / "wan_fused_fp8"


def _build_dir(name: str) -> str:
    path = Path(os.environ.get("TRTLLM_WAN_FUSED_BUILD_DIR", _DEFAULT_BUILD_DIR)) / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@functools.cache
def block():
    """cuBLASLt FP8 GEMM epilogues and the residual/norm/quant row kernel."""
    return load(
        name="wan_fused_block_ext",
        sources=[str(_CSRC / "wan_block_fused.cu")],
        extra_cuda_cflags=_CUDA_FLAGS,
        extra_ldflags=["-lcublasLt"],
        build_directory=_build_dir("block"),
    )


@functools.cache
def prep():
    """Fused QK RMSNorm + RoPE + FP8 quant of packed QKV."""
    return load(
        name="wan_fused_qkv_prep_ext",
        sources=[str(_CSRC / "fused_qkv_fp8.cu")],
        extra_cuda_cflags=_CUDA_FLAGS,
        build_directory=_build_dir("prep"),
    )


@functools.cache
def peer():
    """Copy-engine copies and stream flags for the Ulysses overlap."""
    return load(
        name="wan_fused_peer_copy_ext",
        sources=[str(_CSRC / "peer_copy.cpp")],
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        with_cuda=True,
        build_directory=_build_dir("peer"),
    )


@functools.cache
def fmha():
    """Binding over TllmGenFmhaRunner; needs a TensorRT-LLM source build."""
    cpp = _TRTLLM_ROOT / "cpp"
    deps = cpp / "build" / "_deps"
    if not (deps / "cutlass-src" / "include").is_dir():
        raise RuntimeError(f"Wan fused FP8 attention needs a source build: {deps} missing.")
    cuda = Path(CUDA_HOME or "/usr/local/cuda")
    kernels = cpp / "tensorrt_llm" / "kernels"
    include_dirs = [
        cpp,
        cpp / "include",
        cpp / "tensorrt_llm" / "cutlass_extensions" / "include",
        kernels / "contextFusedMultiHeadAttention",
        kernels / "internal_cutlass_kernels" / "include",
        kernels / "trtllmGenKernels" / "fmha" / "trtllmGen_fmha_export",
        deps / "cutlass-src" / "include",
        deps / "cutlass-src" / "tools" / "util" / "include",
        deps / "json-src" / "include",
        cuda / "include",
    ]
    for target in ("sbsa-linux", "x86_64-linux"):
        target_inc = cuda / "targets" / target / "include"
        include_dirs += [p for p in (target_inc, target_inc / "cccl") if p.is_dir()]
    defines = [
        "-DENABLE_BF16",
        "-DENABLE_FP4",
        "-DENABLE_FP8",
        "-DENABLE_MULTI_DEVICE=1",
        "-DTLLM_ENABLE_CUDA",
        "-DTLLM_FMHA_TRTLLM_COMPAT",
        "-DTLLM_GEN_EXPORT_INTERFACE",
        "-DTORCH_CUDA=1",
        "-DTRTLLM_ABI_NAMESPACE=_v1",
        "-DNDEBUG",
        "-DOMPI_SKIP_MPICXX",
    ]
    libs = _TRTLLM_ROOT / "tensorrt_llm" / "libs"
    return load(
        name="wan_trtllmgen_fmha_ext",
        sources=[str(_CSRC / "trtllmgen_fmha.cpp")],
        extra_include_paths=[str(p) for p in include_dirs],
        extra_cflags=["-O2", "-std=c++17"] + defines,
        extra_ldflags=[f"-L{libs}", "-ltensorrt_llm", f"-Wl,-rpath,{libs}", "-lcuda"],
        build_directory=_build_dir("fmha"),
    )

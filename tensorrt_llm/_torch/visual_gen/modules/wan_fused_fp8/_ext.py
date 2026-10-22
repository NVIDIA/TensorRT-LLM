# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT-built extensions for the Wan fused FP8 ops, built on first use."""

import functools
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME, load

import tensorrt_llm

_CSRC = Path(__file__).resolve().parent / "csrc"
_CUDA_FLAGS = ["-O3", "-gencode=arch=compute_107a,code=sm_107a", "-std=c++17"]
_TRTLLM_ROOT = Path(tensorrt_llm.__file__).resolve().parent.parent
_CUTLASS_INCLUDE = _TRTLLM_ROOT / "cpp" / "build" / "_deps" / "cutlass-src" / "include"


@functools.cache
def jit_available() -> bool:
    """Whether nvcc is present to build the fused kernels on first use."""
    return CUDA_HOME is not None and (Path(CUDA_HOME) / "bin" / "nvcc").is_file()


@functools.cache
def fmha_available() -> bool:
    """Whether the FMHA binding can be built (needs a TensorRT-LLM source build)."""
    return jit_available() and _CUTLASS_INCLUDE.is_dir()


@functools.cache
def block():
    """cuBLASLt FP8 GEMM epilogues and the residual/norm/quant row kernel."""
    return load(
        name="wan_fused_block_ext",
        sources=[str(_CSRC / "wan_block_fused.cu")],
        extra_cuda_cflags=_CUDA_FLAGS,
        extra_ldflags=["-lcublasLt"],
    )


@functools.cache
def prep():
    """QK RMSNorm + RoPE + FP8 quant of packed QKV, V amax/scale, and LSE combine."""
    return load(
        name="wan_fused_qkv_prep_ext",
        sources=[str(_CSRC / "fused_qkv_fp8.cu")],
        extra_cuda_cflags=_CUDA_FLAGS,
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
    )


@functools.cache
def fmha():
    """Binding over TllmGenFmhaRunner; needs a TensorRT-LLM source build."""
    cpp = _TRTLLM_ROOT / "cpp"
    deps = cpp / "build" / "_deps"
    if not fmha_available():
        raise RuntimeError(f"Wan fused FP8 attention needs a source build: {deps} missing.")
    cuda = Path(CUDA_HOME)
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
    )

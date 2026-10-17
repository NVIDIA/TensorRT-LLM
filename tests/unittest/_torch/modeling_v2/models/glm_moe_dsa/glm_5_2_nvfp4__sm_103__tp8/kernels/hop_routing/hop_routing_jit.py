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
"""Load ``trtllm::glm_moe_dsa_post_attention_hop_routing`` (and the test occupier) for tests.

A TensorRT-LLM build of this branch registers the routing hop in
``th_common``. On an older image (development overlay) it is compiled from
this checkout's C++ sources with ``torch.utils.cpp_extension``, unmodified and
with the same op schema. ``GLM_MOE_DSA_HOP_ROUTING_JIT_DEFINES`` adds compile definitions for
diagnostic builds (``-DTLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS``); such a build is always
compiled here and registers ``trtllm::glm_moe_dsa_post_attention_hop_routing_diag``.

``ensure_occupier`` builds ``z2test::occupy(int blocks, int ns)``, a test-only
kernel of ``blocks`` one-warp CTAs that each spin for ``ns`` nanoseconds; the
progress tests run it on another stream to hold SMs while the hop waits.
"""

import hashlib
import os
from pathlib import Path
from typing import Callable

import torch

_REPO = Path(__file__).resolve().parents[9]
_SOURCES = (
    _REPO / "cpp/tensorrt_llm/thop/glmMoeDsa/postAttentionHopRoutingOp.cpp",
    _REPO / "cpp/tensorrt_llm/kernels/glmMoeDsa/postAttentionHopRouting.cu",
)
_HEADERS = (
    _REPO / "cpp/tensorrt_llm/kernels/glmMoeDsa/postAttentionHopRouting.h",
    _REPO / "cpp/tensorrt_llm/kernels/glmMoeDsa/postAttentionHop.h",
)
_EXTRA = [f for f in os.environ.get("GLM_MOE_DSA_HOP_ROUTING_JIT_DEFINES", "").split() if f]
DIAG = "-DTLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS" in _EXTRA
# The diagnostic build registers ``glm_moe_dsa_post_attention_hop_routing_diag``.
OP_NAME = (
    "glm_moe_dsa_post_attention_hop_routing_diag"
    if DIAG
    else "glm_moe_dsa_post_attention_hop_routing"
)
_NAME = "hop_routing_diag_jit" if DIAG else "hop_routing_jit"

_OCCUPIER_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void z2OccupyKernel(long long ns)
{
    unsigned long long start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    unsigned long long now = start;
    while (now - start < static_cast<unsigned long long>(ns))
    {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    }
}

void occupy(int64_t blocks, int64_t ns)
{
    // One CTA per SM: all of an SM's shared memory, so no other CTA fits next to it.
    int device = 0;
    cudaGetDevice(&device);
    int smem = 0;
    cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaFuncSetAttribute(z2OccupyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    z2OccupyKernel<<<static_cast<unsigned>(blocks), 32, smem, at::cuda::getCurrentCUDAStream()>>>(ns);
}

// No tensor arguments: registered as a catch-all kernel.
TORCH_LIBRARY(z2test, m)
{
    m.def("occupy(int blocks, int ns) -> ()", &occupy);
}
"""


def _digest() -> str:
    h = hashlib.sha256(" ".join(_EXTRA).encode())
    for path in (*_SOURCES, *_HEADERS):
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def ensure_op(build_root: str, is_builder: bool, barrier: Callable[[], None]) -> str:
    """Make the routing hop available on every rank; returns where it came from.

    ``is_builder`` must be true on exactly one rank per shared build root;
    ``barrier`` synchronizes the ranks after that rank has built it.
    """
    import tensorrt_llm  # noqa: F401  (loads th_common and the trtllm namespace)

    if hasattr(torch.ops.trtllm, OP_NAME) and not _EXTRA:
        barrier()
        return "builtin"
    build_dir = Path(build_root) / f"hop_routing-{_digest()}"
    library = build_dir / f"{_NAME}.so"
    if is_builder and not library.exists():
        from torch.utils.cpp_extension import load

        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.3"
        build_dir.mkdir(parents=True, exist_ok=True)
        load(
            name=_NAME,
            sources=[str(s) for s in _SOURCES],
            extra_include_paths=[str(_REPO / "cpp/include"), str(_REPO / "cpp")],
            extra_cflags=["-O3", "-std=c++17", *_EXTRA],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-Xptxas=-v",
                *_EXTRA,
            ],
            build_directory=str(build_dir),
            is_python_module=False,
            verbose=True,
        )
    barrier()
    if not hasattr(torch.ops.trtllm, OP_NAME):
        torch.ops.load_library(str(library))
    return str(library)


def ensure_occupier(build_root: str, is_builder: bool, barrier: Callable[[], None]) -> None:
    """Make ``torch.ops.z2test.occupy`` available on every rank."""
    if hasattr(torch.ops, "z2test") and hasattr(torch.ops.z2test, "occupy"):
        barrier()
        return
    digest = hashlib.sha256(_OCCUPIER_SRC.encode()).hexdigest()[:16]
    build_dir = Path(build_root) / f"hop_routing_occupier-{digest}"
    library = build_dir / "z2test_occupy.so"
    if is_builder and not library.exists():
        from torch.utils.cpp_extension import load

        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.3"
        build_dir.mkdir(parents=True, exist_ok=True)
        src = build_dir / "z2test_occupy.cu"
        src.write_text(_OCCUPIER_SRC)
        load(
            name="z2test_occupy",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            build_directory=str(build_dir),
            is_python_module=False,
            verbose=False,
        )
    barrier()
    if not (hasattr(torch.ops, "z2test") and hasattr(torch.ops.z2test, "occupy")):
        torch.ops.load_library(str(library))

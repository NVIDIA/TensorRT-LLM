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
"""Test-only PDL producer and consumer kernels for the sparse_mla_decode stress test.

``delayed_scatter`` is a producer: it allows its dependents to launch at once
(``griddepcontrol.launch_dependents``), waits for its own predecessor, spins for
a given time and only then copies a list of segments into the buffers the next
kernel reads. ``pdl_consumer`` is a consumer: it also allows its dependents to
launch at once, waits for its predecessor and copies a buffer out, optionally
after a spin. Both are launched with programmatic stream serialization, so a
chain producer -> sparse_mla_decode -> consumer -> producer -> ... has no plain kernel
boundary: every kernel may become resident while its predecessor still runs,
and only ``griddepcontrol.wait`` orders the memory accesses.

Both kernels record ``%globaltimer`` stamps so that the test can show that the
early launches actually happened.
"""

import os

import torch

_CPP_DECLS = r"""
#include <torch/extension.h>
void delayed_scatter(torch::Tensor dsts, torch::Tensor srcs, torch::Tensor counts, int64_t spin_ns,
    torch::Tensor stamps, bool use_pdl, int64_t num_blocks);
void pdl_consumer(torch::Tensor dst, torch::Tensor src, int64_t spin_ns, torch::Tensor stamps, bool use_pdl);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace
{

__device__ __forceinline__ unsigned long long globalTimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void spinFor(long long spinNs)
{
    if (spinNs <= 0)
    {
        return;
    }
    if (threadIdx.x == 0)
    {
        unsigned long long const start = globalTimer();
        while (globalTimer() - start < static_cast<unsigned long long>(spinNs))
        {
        }
    }
    __syncthreads();
}

// stamps: [0] producer start (before waiting), [1] producer after its wait, [2] producer end.
__global__ void delayedScatterKernel(long long const* dsts, long long const* srcs, long long const* counts,
    int numSegments, long long spinNs, unsigned long long* stamps)
{
    unsigned long long const t0 = globalTimer();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned long long const t1 = globalTimer();
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        stamps[0] = t0;
        stamps[1] = t1;
    }
    spinFor(spinNs);
    for (int s = 0; s < numSegments; ++s)
    {
        uint4* dst = reinterpret_cast<uint4*>(dsts[s]);
        uint4 const* src = reinterpret_cast<uint4 const*>(srcs[s]);
        long long const n = counts[s];
        for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
             i += static_cast<long long>(gridDim.x) * blockDim.x)
        {
            dst[i] = src[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicMax(stamps + 2, globalTimer());
    }
}

// stamps: [0] consumer start (before waiting), [1] consumer after its wait.
__global__ void pdlConsumerKernel(
    uint4 const* src, uint4* dst, long long n, long long spinNs, unsigned long long* stamps)
{
    unsigned long long const t0 = globalTimer();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned long long const t1 = globalTimer();
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        stamps[0] = t0;
        stamps[1] = t1;
    }
    spinFor(spinNs);
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
    {
        dst[i] = src[i];
    }
}

cudaLaunchAttribute pdlAttribute(bool usePdl)
{
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = usePdl ? 1 : 0;
    return attr;
}

} // namespace

void delayed_scatter(torch::Tensor dsts, torch::Tensor srcs, torch::Tensor counts, int64_t spin_ns,
    torch::Tensor stamps, bool use_pdl, int64_t num_blocks)
{
    TORCH_CHECK(dsts.is_cuda() && srcs.is_cuda() && counts.is_cuda() && stamps.is_cuda());
    TORCH_CHECK(dsts.scalar_type() == torch::kInt64 && srcs.scalar_type() == torch::kInt64
        && counts.scalar_type() == torch::kInt64 && stamps.scalar_type() == torch::kInt64);
    TORCH_CHECK(dsts.numel() == srcs.numel() && dsts.numel() == counts.numel() && stamps.numel() >= 3);
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned>(num_blocks));
    config.blockDim = dim3(256);
    config.stream = at::cuda::getCurrentCUDAStream(dsts.get_device());
    cudaLaunchAttribute attr = pdlAttribute(use_pdl);
    config.attrs = &attr;
    config.numAttrs = 1;
    TORCH_CHECK(cudaLaunchKernelEx(&config, delayedScatterKernel, reinterpret_cast<long long const*>(dsts.data_ptr()),
                    reinterpret_cast<long long const*>(srcs.data_ptr()),
                    reinterpret_cast<long long const*>(counts.data_ptr()), static_cast<int>(dsts.numel()),
                    static_cast<long long>(spin_ns), reinterpret_cast<unsigned long long*>(stamps.data_ptr()))
        == cudaSuccess);
}

void pdl_consumer(torch::Tensor dst, torch::Tensor src, int64_t spin_ns, torch::Tensor stamps, bool use_pdl)
{
    TORCH_CHECK(dst.is_cuda() && src.is_cuda() && stamps.is_cuda() && stamps.scalar_type() == torch::kInt64);
    TORCH_CHECK(dst.is_contiguous() && src.is_contiguous() && dst.nbytes() == src.nbytes() && src.nbytes() % 16 == 0);
    TORCH_CHECK(stamps.numel() >= 2);
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(16);
    config.blockDim = dim3(256);
    config.stream = at::cuda::getCurrentCUDAStream(dst.get_device());
    cudaLaunchAttribute attr = pdlAttribute(use_pdl);
    config.attrs = &attr;
    config.numAttrs = 1;
    TORCH_CHECK(cudaLaunchKernelEx(&config, pdlConsumerKernel, reinterpret_cast<uint4 const*>(src.data_ptr()),
                    reinterpret_cast<uint4*>(dst.data_ptr()), static_cast<long long>(src.nbytes() / 16),
                    static_cast<long long>(spin_ns), reinterpret_cast<unsigned long long*>(stamps.data_ptr()))
        == cudaSuccess);
}
"""

_MODULE = None


def load_pdl_helpers():
    """Build (once per process) and return the helper extension."""
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline

        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
        build_dir = os.path.join(
            os.environ.get(
                "GLM_MOE_DSA_SPARSE_MLA_DECODE_HELPER_BUILD_DIR",
                "/tmp/glm_moe_dsa_sparse_mla_decode_helpers",
            ),
            arch,
        )
        os.makedirs(build_dir, exist_ok=True)
        _MODULE = load_inline(
            name=f"glm_moe_dsa_sparse_mla_decode_pdl_helpers_{arch}",
            cpp_sources=_CPP_DECLS,
            cuda_sources=_CUDA_SRC,
            functions=["delayed_scatter", "pdl_consumer"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
            build_directory=build_dir,
            verbose=False,
        )
    return _MODULE

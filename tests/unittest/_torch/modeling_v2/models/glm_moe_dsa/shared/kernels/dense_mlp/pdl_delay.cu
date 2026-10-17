/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Test-only delayed producer for programmatic dependent launch: releases its dependents immediately,
// spins for a given number of nanoseconds, then copies src into dst. A PDL consumer that reads dst
// before griddepcontrol.wait would observe stale data.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>

namespace
{

__global__ void delayedCopyKernel(uint4 const* __restrict__ src, uint4* __restrict__ dst, int64_t numVec, int64_t delayNs)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    uint64_t start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    uint64_t now = start;
    while (now - start < static_cast<uint64_t>(delayNs))
    {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    }
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < numVec;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        dst[i] = src[i];
    }
}

} // namespace

void delayedCopy(torch::Tensor src, torch::Tensor dst, int64_t delayNs)
{
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous() && src.nbytes() == dst.nbytes() && src.nbytes() % 16 == 0);
    auto const stream = at::cuda::getCurrentCUDAStream(src.get_device());
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(4);
    config.blockDim = dim3(128);
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    TORCH_CHECK(cudaLaunchKernelEx(&config, delayedCopyKernel, reinterpret_cast<uint4 const*>(src.data_ptr()),
                    reinterpret_cast<uint4*>(dst.data_ptr()), static_cast<int64_t>(src.nbytes() / 16), delayNs)
        == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("delayed_copy", &delayedCopy);
}

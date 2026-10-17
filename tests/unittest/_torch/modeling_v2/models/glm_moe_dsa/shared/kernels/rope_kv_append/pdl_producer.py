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
"""Test-only delayed metadata producer launched with programmatic dependent launch.

``delayed_write(srcs, dsts, delay_cycles)`` enqueues one kernel on the current
stream (capturable into a CUDA graph) that

1. executes ``griddepcontrol.launch_dependents`` first, so a following kernel
   launched with programmatic stream serialization may start right away;
2. waits for its own predecessor (``griddepcontrol.wait``);
3. spins for ``delay_cycles`` SM cycles;
4. copies every ``srcs[i]`` into ``dsts[i]``.

It stands for an in-graph metadata writer (for example a fused DSA metadata
kernel) that is PDL-chained to the next kernel: a consumer is correct only if
it reads ``dsts`` after its own ``griddepcontrol.wait``.

``early_read(srcs, dsts)`` is the sensitivity control of that stress: a
PDL-launched kernel that copies ``srcs`` into ``dsts`` *before* its own
``griddepcontrol.wait`` (then waits and releases its dependents), i.e. a
consumer that violates the rule. Placed right after ``delayed_write`` it must
observe the stale values, which shows the stress opens the window.
"""

import functools
from typing import List

import torch

_CPP_DECL = (
    "void delayed_write(std::vector<torch::Tensor> srcs, std::vector<torch::Tensor> dsts, "
    "int64_t delay_cycles);\n"
    "void early_read(std::vector<torch::Tensor> srcs, std::vector<torch::Tensor> dsts);"
)

_CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace
{
constexpr int kMaxCopies = 8;

struct Copies
{
    uint32_t const* src[kMaxCopies];
    uint32_t* dst[kMaxCopies];
    int64_t words[kMaxCopies];
    int32_t n;
};

__global__ void delayedWriterKernel(Copies c, int64_t delayCycles)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    if (threadIdx.x == 0)
    {
        long long const start = clock64();
        while (clock64() - start < delayCycles)
        {
        }
    }
    __syncthreads();
    for (int32_t i = 0; i < c.n; ++i)
    {
        for (int64_t w = threadIdx.x; w < c.words[i]; w += blockDim.x)
        {
            c.dst[i][w] = c.src[i][w];
        }
    }
}

__global__ void earlyReaderKernel(Copies c)
{
    for (int32_t i = 0; i < c.n; ++i)
    {
        for (int64_t w = threadIdx.x; w < c.words[i]; w += blockDim.x)
        {
            c.dst[i][w] = c.src[i][w];
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

Copies makeCopies(std::vector<torch::Tensor> const& srcs, std::vector<torch::Tensor> const& dsts)
{
    TORCH_CHECK(srcs.size() == dsts.size() && srcs.size() <= kMaxCopies, "up to 8 (src, dst) pairs");
    Copies c{};
    c.n = static_cast<int32_t>(srcs.size());
    for (size_t i = 0; i < srcs.size(); ++i)
    {
        TORCH_CHECK(srcs[i].is_cuda() && dsts[i].is_cuda() && srcs[i].is_contiguous() && dsts[i].is_contiguous(),
            "contiguous CUDA tensors expected");
        TORCH_CHECK(srcs[i].nbytes() == dsts[i].nbytes() && srcs[i].nbytes() % 4 == 0, "equal sizes, 4-byte words");
        c.src[i] = static_cast<uint32_t const*>(srcs[i].data_ptr());
        c.dst[i] = static_cast<uint32_t*>(dsts[i].data_ptr());
        c.words[i] = static_cast<int64_t>(srcs[i].nbytes() / 4);
    }
    return c;
}

template <typename Kernel, typename... Args>
void launchPdl(Kernel kernel, Args... args)
{
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}
} // namespace

void delayed_write(std::vector<torch::Tensor> srcs, std::vector<torch::Tensor> dsts, int64_t delay_cycles)
{
    launchPdl(delayedWriterKernel, makeCopies(srcs, dsts), delay_cycles);
}

void early_read(std::vector<torch::Tensor> srcs, std::vector<torch::Tensor> dsts)
{
    launchPdl(earlyReaderKernel, makeCopies(srcs, dsts));
}
"""


@functools.lru_cache(maxsize=1)
def _module():
    import os

    from torch.utils.cpp_extension import load_inline

    # Build for the visible GPU only (the image's TORCH_CUDA_ARCH_LIST includes pre-sm_90 targets).
    major, minor = torch.cuda.get_device_capability()
    saved = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        return load_inline(
            name="rope_append_pdl_delayed_producer",
            cpp_sources=[_CPP_DECL],
            cuda_sources=[_CUDA_SRC],
            functions=["delayed_write", "early_read"],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        )
    finally:
        if saved is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = saved


def available() -> bool:
    """Whether the producer can be built (needs nvcc and a compiler at test time)."""
    try:
        _module()
    except (OSError, RuntimeError, ImportError) as err:  # no toolchain in this environment
        print(f"pdl_producer unavailable: {err}")
        return False
    return True


def delayed_write(srcs: List[torch.Tensor], dsts: List[torch.Tensor], delay_cycles: int) -> None:
    _module().delayed_write(list(srcs), list(dsts), int(delay_cycles))


def early_read(srcs: List[torch.Tensor], dsts: List[torch.Tensor]) -> None:
    _module().early_read(list(srcs), list(dsts))

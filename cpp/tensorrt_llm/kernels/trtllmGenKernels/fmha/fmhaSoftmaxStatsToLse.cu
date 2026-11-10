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

#include "fmhaSoftmaxStatsToLse.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr float kLog2e = 1.4426950408889634f;

__global__ void fmhaSoftmaxStatsToLseKernel(float2 const* __restrict__ stats, float* __restrict__ lse, int32_t numHeads,
    int64_t lseStrideTokens, int64_t lseStrideHeads, int32_t numRows)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // The stats are written by the FMHA kernel (or its reduction kernel) right before this one.
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    if (row < numRows)
    {
        float2 const md = stats[row];
        int32_t const token = row / numHeads;
        int32_t const head = row - token * numHeads;
        lse[static_cast<int64_t>(token) * lseStrideTokens + static_cast<int64_t>(head) * lseStrideHeads]
            = md.x * kLog2e + __log2f(md.y);
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

} // namespace

void invokeFmhaSoftmaxStatsToLse(float2 const* stats, float* lse, int32_t numTokens, int32_t numHeads,
    int64_t lseStrideTokens, int64_t lseStrideHeads, cudaStream_t stream)
{
    int32_t const numRows = numTokens * numHeads;
    if (numRows <= 0)
    {
        return;
    }
    constexpr int32_t kThreadsPerCta = 256;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3((numRows + kThreadsPerCta - 1) / kThreadsPerCta);
    config.blockDim = dim3(kThreadsPerCta);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(
        &config, fmhaSoftmaxStatsToLseKernel, stats, lse, numHeads, lseStrideTokens, lseStrideHeads, numRows));
}

} // namespace kernels

TRTLLM_NAMESPACE_END

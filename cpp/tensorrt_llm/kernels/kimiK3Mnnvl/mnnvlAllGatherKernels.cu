/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/kimiK3Mnnvl/mnnvlAllGatherKernels.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/lamportUtils.cuh"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimiK3Mnnvl
{

using tensorrt_llm::common::isLamportDirty;
using tensorrt_llm::common::LamportFlags;
using tensorrt_llm::common::loadPackedVolatile;
using tensorrt_llm::common::VolatilePackedLoad;

namespace
{

constexpr int kThreads = 256;

// Payload vectors (16 bytes) of one rank's slot for one token: the bf16 part, then the fp32 part.
__host__ __device__ inline int slotVectors(int bf16Columns, int fp32Columns)
{
    return bf16Columns / 8 + fp32Columns / 4;
}

// No payload word may equal the Lamport sentinel (the fp32 -0.0 word): -0.0 halves of a bf16
// pair and -0.0 fp32 words become +0.0.
__device__ inline uint32_t sanitizeBf16Pair(uint32_t word)
{
    if ((word & 0xffffu) == 0x8000u)
    {
        word &= 0xffff0000u;
    }
    if ((word >> 16) == 0x8000u)
    {
        word &= 0x0000ffffu;
    }
    return word;
}

__device__ inline uint32_t sanitizeFp32(uint32_t word)
{
    return word == 0x80000000u ? 0u : word;
}

__device__ inline uint32_t packBf16Pair(float lo, float hi)
{
    __nv_bfloat162 const pair = __floats2bfloat162_rn(lo, hi);
    return sanitizeBf16Pair(*reinterpret_cast<uint32_t const*>(&pair));
}

// One CTA per token: broadcast this rank's slot of the token, then poll every rank's slot and
// scatter it to the outputs.
__global__ void __launch_bounds__(kThreads) mnnvlAllGatherSplitKernel(AllGatherSplitParams const p)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
    // Dependents read the outputs (and the Lamport flags) only after their own wait for this grid.
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    int const token = blockIdx.x;
    int const bf16Vectors = p.bf16Columns / 8;
    int const vectors = slotVectors(p.bf16Columns, p.fp32Columns);
    int const inputColumns = p.bf16Columns + p.fp32Columns;

    LamportFlags<float4> flag(p.bufferFlags, 1);
    auto* lamportMcast = reinterpret_cast<uint4*>(flag.getCurLamportBuf(p.multicastPtr, 0));
    auto* lamportLocal = reinterpret_cast<uint4*>(flag.getCurLamportBuf(p.bufferPtrsDev[p.rank], 0));

    float const* row = p.input + static_cast<int64_t>(token) * inputColumns;
    for (int v = threadIdx.x; v < vectors; v += blockDim.x)
    {
        uint4 packed;
        if (v < bf16Vectors)
        {
            float4 const a = reinterpret_cast<float4 const*>(row)[2 * v];
            float4 const b = reinterpret_cast<float4 const*>(row)[2 * v + 1];
            packed = make_uint4(
                packBf16Pair(a.x, a.y), packBf16Pair(a.z, a.w), packBf16Pair(b.x, b.y), packBf16Pair(b.z, b.w));
        }
        else
        {
            uint4 const words = reinterpret_cast<uint4 const*>(row + p.bf16Columns)[v - bf16Vectors];
            packed = make_uint4(
                sanitizeFp32(words.x), sanitizeFp32(words.y), sanitizeFp32(words.z), sanitizeFp32(words.w));
        }
        lamportMcast[(static_cast<int64_t>(token) * p.nRanks + p.rank) * vectors + v] = packed;
    }

    flag.ctaArrive();
    flag.clearDirtyLamportBuf(p.bufferPtrsDev[p.rank], -1);

    for (int i = threadIdx.x; i < p.nRanks * vectors; i += blockDim.x)
    {
        int const r = i / vectors;
        int const v = i % vectors;
        VolatilePackedLoad<float4> value;
        do
        {
            value
                = loadPackedVolatile<float4>(&lamportLocal[(static_cast<int64_t>(token) * p.nRanks + r) * vectors + v]);
        } while (isLamportDirty(value));
        uint4 const words = make_uint4(value.words[0], value.words[1], value.words[2], value.words[3]);
        if (v < bf16Vectors)
        {
            reinterpret_cast<uint4*>(
                p.bf16Output + static_cast<int64_t>(token) * p.nRanks * p.bf16Columns + r * p.bf16Columns)[v]
                = words;
        }
        else
        {
            reinterpret_cast<uint4*>(p.fp32Output + static_cast<int64_t>(token) * p.nRanks * p.fp32Columns
                + r * p.fp32Columns)[v - bf16Vectors]
                = words;
        }
    }

    flag.waitAndUpdate({static_cast<uint32_t>(p.numTokens * p.nRanks * vectors * sizeof(uint4)), 0, 0, 0});
}

} // namespace

int64_t mnnvlAllGatherSplitFootprint(int numTokens, int bf16Columns, int fp32Columns, int nRanks)
{
    return static_cast<int64_t>(numTokens) * nRanks * slotVectors(bf16Columns, fp32Columns) * sizeof(uint4);
}

void mnnvlAllGatherSplitOp(AllGatherSplitParams const& params)
{
    TLLM_CHECK_WITH_INFO(params.bf16Columns % 8 == 0 && params.fp32Columns % 4 == 0,
        "[mnnvlAllGatherSplit] needs bf16 columns in multiples of 8 and fp32 columns in multiples of 4");
    TLLM_CHECK_WITH_INFO(params.numTokens > 0, "[mnnvlAllGatherSplit] needs at least one token");
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(params.numTokens);
    config.blockDim = dim3(kThreads);
    config.stream = params.stream;
    config.attrs = attrs;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, mnnvlAllGatherSplitKernel, params));
}

} // namespace kernels::kimiK3Mnnvl

TRTLLM_NAMESPACE_END

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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmMnnvlAllgather.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/lamportUtils.cuh"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmLamport.cuh"

#include <algorithm>
#include <cuda_bf16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
namespace
{

constexpr int kAllgatherBlockSize = 512;
// Gathered packets each thread polls together.
constexpr int kAllgatherPacketsInFlight = 2;
// A payload word equal to the empty word is sent as this word: still a pair of BF16 NaNs.
constexpr uint32_t kAllgatherSubstituteWord = kMnnvlExchangeEmptyWord - 1U;
constexpr int kPacketBytes = static_cast<int>(sizeof(float4));

union AllgatherPacket
{
    float4 packed;
    uint32_t words[4];
};

struct AllgatherKernelParams
{
    float4 const* input; // [numRows, shardPackets] packets of BF16
    float* output;       // [numRows, nRanks * shardPackets] packets of BF16, written as FP32
    void* bufferLocal;
    void* bufferMcast;
    uint32_t* bufferFlags;
    int numRows;
    int shardPackets;
    int nRanks;
    int rank;
};

__device__ __forceinline__ bool isEmptyPacket(AllgatherPacket const& packet)
{
    bool empty = false;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        empty |= packet.words[i] == kMnnvlExchangeEmptyWord;
    }
    return empty;
}

// Writes gathered packet packetIdx (eight BF16 values) to the output as FP32. A BF16 value is the upper half of the
// FP32 value it widens to: shift the bits, NaN payloads included.
__device__ __forceinline__ void storeWidened(float* output, int packetIdx, AllgatherPacket const& packet)
{
    float4* out = reinterpret_cast<float4*>(output) + 2 * packetIdx;
    out[0] = make_float4(__uint_as_float(packet.words[0] << 16), __uint_as_float(packet.words[0] & 0xFFFF0000U),
        __uint_as_float(packet.words[1] << 16), __uint_as_float(packet.words[1] & 0xFFFF0000U));
    out[1] = make_float4(__uint_as_float(packet.words[2] << 16), __uint_as_float(packet.words[2] & 0xFFFF0000U),
        __uint_as_float(packet.words[3] << 16), __uint_as_float(packet.words[3] & 0xFFFF0000U));
}

__global__ void __launch_bounds__(kAllgatherBlockSize) mnnvlAllgatherKernel(AllgatherKernelParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // The shard is written by the preceding kernel. Dependents read the output only after their own wait.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();

    GlmLamportFlags<float4> flag(params.bufferFlags, /*numStages=*/1, /*volatileRead=*/false);
    auto* stageMcast = static_cast<float4*>(flag.getCurLamportBuf(params.bufferMcast));
    auto const* stageLocal = static_cast<float4 const*>(flag.getCurLamportBuf(params.bufferLocal));

    int const tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int const numThreads = static_cast<int>(gridDim.x * blockDim.x);
    int const rowPackets = params.nRanks * params.shardPackets;
    int const shardTotal = params.numRows * params.shardPackets;
    int const gatherTotal = params.numRows * rowPackets;

    // Store the shard once through the multicast address, at [row][rank][packet] of the gathered layout.
    for (int idx = tid; idx < shardTotal; idx += numThreads)
    {
        int const row = idx / params.shardPackets;
        int const col = idx - row * params.shardPackets;
        AllgatherPacket packet;
        packet.packed = params.input[idx];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (packet.words[i] == kMnnvlExchangeEmptyWord)
            {
                packet.words[i] = kAllgatherSubstituteWord;
            }
        }
        stageMcast[row * rowPackets + params.rank * params.shardPackets + col] = packet.packed;
    }

    flag.ctaArrive();
    // Empty the buffer of the previous call; it is written again two calls later.
    AllgatherPacket empty;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        empty.words[i] = kMnnvlExchangeEmptyWord;
    }
    flag.fillDirtyLamportBuf(params.bufferLocal, empty.packed);

    // Copy the gathered rows out of the local buffer as the packets of every rank land.
    for (int base = tid; base < gatherTotal; base += numThreads * kAllgatherPacketsInFlight)
    {
        AllgatherPacket packets[kAllgatherPacketsInFlight];
        bool pending[kAllgatherPacketsInFlight];
#pragma unroll
        for (int j = 0; j < kAllgatherPacketsInFlight; ++j)
        {
            int const idx = base + j * numThreads;
            pending[j] = idx < gatherTotal;
            if (pending[j])
            {
                packets[j].packed = common::loadPackedVolatile<float4>(&stageLocal[idx]).packed;
            }
        }
        bool anyPending = true;
        while (anyPending)
        {
            anyPending = false;
#pragma unroll
            for (int j = 0; j < kAllgatherPacketsInFlight; ++j)
            {
                if (!pending[j])
                {
                    continue;
                }
                int const idx = base + j * numThreads;
                if (isEmptyPacket(packets[j]))
                {
                    packets[j].packed = common::loadPackedVolatile<float4>(&stageLocal[idx]).packed;
                    anyPending = true;
                }
                else
                {
                    storeWidened(params.output, idx, packets[j]);
                    pending[j] = false;
                }
            }
        }
    }

    flag.waitAndUpdate({static_cast<uint32_t>(gatherTotal * kPacketBytes), 0, 0, 0});
#endif
}

} // namespace

void mnnvlAllgather(MnnvlAllgatherParams const& params)
{
    TLLM_CHECK_WITH_INFO(common::getSMVersion() >= 90, "[GLM MNNVL all-gather] requires SM 90 or newer.");
    constexpr int kEltsPerPacket = kPacketBytes / static_cast<int>(sizeof(__nv_bfloat16));
    TLLM_CHECK_WITH_INFO(params.shardDim % kEltsPerPacket == 0,
        "[GLM MNNVL all-gather] a shard row must be a multiple of %d BF16 elements, got %d", kEltsPerPacket,
        params.shardDim);
    if (params.numRows == 0 || params.shardDim == 0)
    {
        return;
    }
    int const shardPackets = params.shardDim / kEltsPerPacket;
    int const gatherTotal = params.numRows * params.nRanks * shardPackets;
    static int const kSmCount = common::getMultiProcessorCount();
    int constexpr kPacketsPerBlock = kAllgatherBlockSize * kAllgatherPacketsInFlight;
    // At most one CTA per SM: every CTA is resident and stores its part of the shard, so the packets the CTAs of other
    // ranks poll for are always on their way.
    int const numBlocks = std::max(1, std::min((gatherTotal + kPacketsPerBlock - 1) / kPacketsPerBlock, kSmCount));
    AllgatherKernelParams const kernelParams{static_cast<float4 const*>(params.input), params.output,
        params.bufferPtrLocal, params.multicastPtr, params.bufferFlags, params.numRows, shardPackets, params.nRanks,
        params.rank};
    common::launchWithPdlWhenEnabled("glmMnnvlAllgather", mnnvlAllgatherKernel, dim3(numBlocks),
        dim3(kAllgatherBlockSize), 0, params.stream, kernelParams);
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END

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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/highThroughput/glmMtpHeadGather.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/lamportUtils.cuh"

#include <algorithm>
#include <cuda_bf16.h>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_rubin_mtp
{
namespace
{

using tensorrt_llm::common::LamportFlags;
using tensorrt_llm::common::loadPackedVolatile;

constexpr int kAllGatherBlockSize = 512;
// Gathered packets each thread polls together.
constexpr int kAllGatherPacketsInFlight = 2;
// A payload word equal to the empty word is sent as this word: still a NaN pair (BF16) or a NaN (FP32).
constexpr uint32_t kAllGatherSubstituteWord = kAllGatherEmptyWord - 1U;

union AllGatherPacket
{
    float4 packed;
    uint32_t words[4];
};

struct AllGatherKernelParams
{
    float4 const* input; // [numRows, shardPackets]
    void* output;        // [numRows, nRanks * shardPackets] packets of the input type, written as the output type
    void* bufferLocal;
    void* bufferMcast;
    uint32_t* bufferFlags;
    int numRows;
    int shardPackets;
    int nRanks;
    int rank;
};

__device__ __forceinline__ bool isEmptyPacket(AllGatherPacket const& packet)
{
    bool empty = false;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        empty |= packet.words[i] == kAllGatherEmptyWord;
    }
    return empty;
}

// Writes gathered packet packetIdx (16 bytes of TIn) to the output as TOut.
template <typename TIn, typename TOut>
__device__ __forceinline__ void storeGatheredPacket(void* output, int packetIdx, AllGatherPacket const& packet)
{
    if constexpr (std::is_same_v<TIn, TOut>)
    {
        reinterpret_cast<float4*>(output)[packetIdx] = packet.packed;
    }
    else
    {
        static_assert(std::is_same_v<TIn, __nv_bfloat16> && std::is_same_v<TOut, float>, "unsupported conversion");
        // A BF16 value is the upper half of the FP32 value it widens to: shift the bits, NaN payloads included.
        float4* out = reinterpret_cast<float4*>(output) + 2 * packetIdx;
        out[0] = make_float4(__uint_as_float(packet.words[0] << 16), __uint_as_float(packet.words[0] & 0xFFFF0000U),
            __uint_as_float(packet.words[1] << 16), __uint_as_float(packet.words[1] & 0xFFFF0000U));
        out[1] = make_float4(__uint_as_float(packet.words[2] << 16), __uint_as_float(packet.words[2] & 0xFFFF0000U),
            __uint_as_float(packet.words[3] << 16), __uint_as_float(packet.words[3] & 0xFFFF0000U));
    }
}

template <typename TIn, typename TOut>
__global__ void __launch_bounds__(kAllGatherBlockSize) oneshotAllGatherKernel(AllGatherKernelParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // The shard is written by the preceding kernel. Dependents read the output only after their own wait.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    LamportFlags<float4> flag(params.bufferFlags, 1);
    // This workspace has one stage and uses all-ones words as its empty
    // sentinel. Snapshot the dirty range before ctaArrive permits flag rotation.
    uint4 const dirtyState = reinterpret_cast<uint4 const*>(params.bufferFlags)[0];
    uint32_t const dirtyBytes = params.bufferFlags[4];
    auto* stageMcast = reinterpret_cast<float4*>(flag.getCurLamportBuf(params.bufferMcast, 0));
    auto const* stageLocal = reinterpret_cast<float4 const*>(flag.getCurLamportBuf(params.bufferLocal, 0));

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
        AllGatherPacket packet;
        packet.packed = params.input[idx];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (packet.words[i] == kAllGatherEmptyWord)
            {
                packet.words[i] = kAllGatherSubstituteWord;
            }
        }
        stageMcast[row * rowPackets + params.rank * params.shardPackets + col] = packet.packed;
    }

    flag.ctaArrive();
    // Empty the buffer of the previous call; it is written again two calls later.
    if (dirtyState.w > 0)
    {
        auto* dirty = reinterpret_cast<uint4*>(
            static_cast<char*>(params.bufferLocal) + static_cast<size_t>(dirtyState.y) * dirtyState.z);
        uint32_t const dirtyPackets = (dirtyBytes + sizeof(uint4) - 1) / sizeof(uint4);
        uint4 const empty
            = make_uint4(kAllGatherEmptyWord, kAllGatherEmptyWord, kAllGatherEmptyWord, kAllGatherEmptyWord);
        for (uint32_t idx = static_cast<uint32_t>(tid); idx < dirtyPackets; idx += numThreads)
        {
            dirty[idx] = empty;
        }
    }

    // Copy the gathered rows out of the local buffer as the packets of every rank land.
    for (int base = tid; base < gatherTotal; base += numThreads * kAllGatherPacketsInFlight)
    {
        AllGatherPacket packets[kAllGatherPacketsInFlight];
        bool pending[kAllGatherPacketsInFlight];
#pragma unroll
        for (int j = 0; j < kAllGatherPacketsInFlight; ++j)
        {
            int const idx = base + j * numThreads;
            pending[j] = idx < gatherTotal;
            if (pending[j])
            {
                packets[j].packed = loadPackedVolatile<float4>(&stageLocal[idx]).packed;
            }
        }
        bool anyPending = true;
        while (anyPending)
        {
            anyPending = false;
#pragma unroll
            for (int j = 0; j < kAllGatherPacketsInFlight; ++j)
            {
                if (!pending[j])
                {
                    continue;
                }
                int const idx = base + j * numThreads;
                if (isEmptyPacket(packets[j]))
                {
                    packets[j].packed = loadPackedVolatile<float4>(&stageLocal[idx]).packed;
                    anyPending = true;
                }
                else
                {
                    storeGatheredPacket<TIn, TOut>(params.output, idx, packets[j]);
                    pending[j] = false;
                }
            }
        }
    }

    flag.waitAndUpdate({static_cast<uint32_t>(gatherTotal * sizeof(float4)), 0, 0, 0});
}

template <typename TIn, typename TOut>
void launchOneshotAllGather(AllGatherParams const& params, int shardPackets)
{
    static int const kSmCount = tensorrt_llm::common::getMultiProcessorCount();
    int const gatherTotal = params.numRows * params.nRanks * shardPackets;
    int constexpr kPacketsPerBlock = kAllGatherBlockSize * kAllGatherPacketsInFlight;
    // At most one CTA per SM: every CTA is resident and stores its part of the shard, so the packets the CTAs of
    // other ranks poll for are always on their way.
    int const numBlocks = std::max(1, std::min((gatherTotal + kPacketsPerBlock - 1) / kPacketsPerBlock, kSmCount));
    AllGatherKernelParams kernelParams{static_cast<float4 const*>(params.input), params.output, params.bufferPtrLocal,
        params.multicastPtr, params.bufferFlags, params.numRows, shardPackets, params.nRanks, params.rank};
    tensorrt_llm::common::launchWithPdlWhenEnabled("mnnvlOneshotAllGather", oneshotAllGatherKernel<TIn, TOut>,
        dim3(numBlocks), dim3(kAllGatherBlockSize), 0, params.stream, kernelParams);
}

} // namespace

void oneshotAllGatherOp(AllGatherParams const& params)
{
    static int const kSmVersion = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(kSmVersion >= 90, "[MNNVL AllGather] requires SM 90 or newer.");
    bool const isBf16 = params.dType == tensorrt_llm::DataType::kBF16;
    TLLM_CHECK_WITH_INFO(
        isBf16 || params.dType == tensorrt_llm::DataType::kFLOAT, "[MNNVL AllGather] supports BF16 and FP32 input.");
    int const eltSize = isBf16 ? 2 : 4;
    int constexpr kPacketBytes = static_cast<int>(sizeof(float4));
    TLLM_CHECK_WITH_INFO((params.shardDim * eltSize) % kPacketBytes == 0,
        "[MNNVL AllGather] a shard row must be a multiple of 16 bytes, got %d elements", params.shardDim);
    if (params.numRows == 0 || params.shardDim == 0)
    {
        return;
    }
    int const shardPackets = params.shardDim * eltSize / kPacketBytes;
    if (!isBf16)
    {
        launchOneshotAllGather<float, float>(params, shardPackets);
    }
    else if (params.floatOutput)
    {
        launchOneshotAllGather<__nv_bfloat16, float>(params, shardPackets);
    }
    else
    {
        launchOneshotAllGather<__nv_bfloat16, __nv_bfloat16>(params, shardPackets);
    }
}

} // namespace kernels::glm_rubin_mtp

TRTLLM_NAMESPACE_END

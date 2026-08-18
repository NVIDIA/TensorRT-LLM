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

#include "nvfp4MlaKvCacheGather.h"

#include "tensorrt_llm/common/assert.h"

#include <algorithm>
#include <cub/device/device_scan.cuh>
#include <cuda_fp16.h>
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int32_t kWarpSize = 32;
constexpr int32_t kWarpsPerBlock = 8;
constexpr int32_t kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int32_t kBlocksPerSm = 6;
constexpr size_t kWorkspaceAlignment = 256;

constexpr size_t alignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__device__ __forceinline__ uint2 convertE2m1x4ToFp16x4(uint16_t packed)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t fp16Low;
    uint32_t fp16High;
    asm volatile(
        "{\n"
        ".reg .b8 low, high;\n"
        "mov.b16 {low, high}, %2;\n"
        "cvt.rn.f16x2.e2m1x2 %0, low;\n"
        "cvt.rn.f16x2.e2m1x2 %1, high;\n"
        "}\n"
        : "=r"(fp16Low), "=r"(fp16High)
        : "h"(packed));
    return make_uint2(fp16Low, fp16High);
#else
    return make_uint2(0, 0);
#endif
}

__device__ __forceinline__ float4 convertE2m1x4ToFloat4(uint16_t packed)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint2 const fp16 = convertE2m1x4ToFp16x4(packed);
    float2 const low = __half22float2(reinterpret_cast<__half2 const&>(fp16.x));
    float2 const high = __half22float2(reinterpret_cast<__half2 const&>(fp16.y));
    return make_float4(low.x, low.y, high.x, high.y);
#else
    constexpr float kMagnitude[8] = {0.F, 0.5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
    float4 output;
    uint8_t const value0 = static_cast<uint8_t>(packed & 0xFU);
    uint8_t const value1 = static_cast<uint8_t>((packed >> 4) & 0xFU);
    uint8_t const value2 = static_cast<uint8_t>((packed >> 8) & 0xFU);
    uint8_t const value3 = static_cast<uint8_t>((packed >> 12) & 0xFU);
    output.x = (value0 & 0x8U) == 0 ? kMagnitude[value0 & 0x7U] : -kMagnitude[value0 & 0x7U];
    output.y = (value1 & 0x8U) == 0 ? kMagnitude[value1 & 0x7U] : -kMagnitude[value1 & 0x7U];
    output.z = (value2 & 0x8U) == 0 ? kMagnitude[value2 & 0x7U] : -kMagnitude[value2 & 0x7U];
    output.w = (value3 & 0x8U) == 0 ? kMagnitude[value3 & 0x7U] : -kMagnitude[value3 & 0x7U];
    return output;
#endif
}

__device__ __forceinline__ uint32_t convertFloat4ToE4m3(float4 values)
{
    uint32_t output;
    reinterpret_cast<__nv_fp8x4_e4m3&>(output) = __nv_fp8x4_e4m3(values);
    return output;
}

__device__ __forceinline__ uint32_t convertUnitScaledE2m1x4ToE4m3(uint16_t packed, __nv_fp8_e4m3 scale)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint2 const fp16 = convertE2m1x4ToFp16x4(packed);
    __half2 const scale2 = __half2half2(static_cast<__half>(scale));
    __half2 const low = __hmul2(reinterpret_cast<__half2 const&>(fp16.x), scale2);
    __half2 const high = __hmul2(reinterpret_cast<__half2 const&>(fp16.y), scale2);
    return __nv_fp8x4_e4m3(low, high).__x;
#else
    float4 values = convertE2m1x4ToFloat4(packed);
    float const scaleFloat = static_cast<float>(scale);
    values.x *= scaleFloat;
    values.y *= scaleFloat;
    values.z *= scaleFloat;
    values.w *= scaleFloat;
    return convertFloat4ToE4m3(values);
#endif
}

template <bool kUnitGlobalScale>
__device__ __forceinline__ void dequantizeRow(uint8_t const* __restrict__ dataPool,
    __nv_fp8_e4m3 const* __restrict__ scalePool, __nv_fp8_e4m3* __restrict__ output, int32_t globalIdx,
    int32_t outputIdx, int32_t headDim, float dequantScale, int32_t lane)
{
    int32_t const packedHeadDim = headDim / 2;
    int32_t const scalesPerToken = headDim / 16;
    auto const* packedRow = dataPool + static_cast<int64_t>(globalIdx) * packedHeadDim;
    auto const* scaleRow = scalePool + static_cast<int64_t>(globalIdx) * scalesPerToken;
    auto* outputRow = output + static_cast<int64_t>(outputIdx) * headDim;

    // Four neighboring lanes cooperatively convert one 16-value scaling group.
    // Each lane handles four packed values and emits one 4-byte FP8 store. For
    // a unit global scale, E2M1 times E4M3 is exactly representable in FP16, so
    // the half2 path preserves the FP8 result. Other scales retain FP32 math.
    int32_t const groupLane = lane % 4;
    for (int32_t groupBase = 0; groupBase < scalesPerToken; groupBase += kWarpSize / 4)
    {
        int32_t const group = groupBase + lane / 4;
        bool const active = group < scalesPerToken;
        uint32_t const activeMask = __ballot_sync(0xFFFFFFFFU, active);
        if (!active)
        {
            continue;
        }
        uint16_t const packed
            = *reinterpret_cast<uint16_t const*>(packedRow + static_cast<int64_t>(group) * 8 + groupLane * 2);
        uint32_t converted;
        if constexpr (kUnitGlobalScale)
        {
            converted = convertUnitScaledE2m1x4ToE4m3(packed, scaleRow[group]);
        }
        else
        {
            float scale = groupLane == 0 ? static_cast<float>(scaleRow[group]) * dequantScale : 0.F;
            scale = __shfl_sync(activeMask, scale, lane - groupLane);
            float4 values = convertE2m1x4ToFloat4(packed);
            values.x *= scale;
            values.y *= scale;
            values.z *= scale;
            values.w *= scale;
            converted = convertFloat4ToE4m3(values);
        }
        *reinterpret_cast<uint32_t*>(outputRow + static_cast<int64_t>(group) * 16 + groupLane * 4) = converted;
    }
}

__global__ void nvFp4MlaKvCacheGatherKernel(uint8_t const* __restrict__ dataPool,
    __nv_fp8_e4m3 const* __restrict__ scalePool, int32_t const* __restrict__ globalIndices,
    __nv_fp8_e4m3* __restrict__ output, int32_t* __restrict__ compactIndices,
    float const* __restrict__ globalDequantScale, int64_t numPairs, int32_t headDim, int64_t numPoolTokens)
{
    int32_t const warp = threadIdx.x / kWarpSize;
    int32_t const lane = threadIdx.x % kWarpSize;
    float const dequantScale = globalDequantScale == nullptr ? 1.F : globalDequantScale[0];
    int64_t pair = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    int64_t const pairStride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
    for (; pair < numPairs; pair += pairStride)
    {
        int32_t globalIdx = 0;
        if (lane == 0)
        {
            globalIdx = globalIndices[pair];
            compactIndices[pair]
                = globalIdx >= 0 && static_cast<int64_t>(globalIdx) < numPoolTokens ? static_cast<int32_t>(pair) : -1;
        }
        globalIdx = __shfl_sync(0xFFFFFFFFU, globalIdx, 0);
        if (globalIdx >= 0 && static_cast<int64_t>(globalIdx) < numPoolTokens)
        {
            if (dequantScale == 1.F)
            {
                dequantizeRow<true>(
                    dataPool, scalePool, output, globalIdx, static_cast<int32_t>(pair), headDim, dequantScale, lane);
            }
            else
            {
                dequantizeRow<false>(
                    dataPool, scalePool, output, globalIdx, static_cast<int32_t>(pair), headDim, dequantScale, lane);
            }
        }
    }
}

__global__ void markContextTopKKernel(int32_t const* __restrict__ localTopKIndices,
    int32_t const* __restrict__ queryReqIndices, int64_t const* __restrict__ cuKvLengths,
    int32_t const* __restrict__ blockTable, int32_t* __restrict__ selectedFlags,
    int32_t* __restrict__ selectedGlobalIndices, int32_t numQueryRows, int32_t topK, int32_t numRequests,
    int32_t maxBlocksPerRequest, int32_t totalKvTokens, int32_t tokensPerBlock, int32_t pageStride, int32_t layerId,
    int64_t numPoolTokens)
{
    int32_t const row = blockIdx.x;
    if (row >= numQueryRows)
    {
        return;
    }
    int32_t const request = queryReqIndices[row];
    if (request < 0 || request >= numRequests)
    {
        return;
    }
    int64_t const requestStart = cuKvLengths[request];
    int64_t const requestLength = cuKvLengths[request + 1] - requestStart;
    for (int32_t col = threadIdx.x; col < topK; col += blockDim.x)
    {
        int32_t const token = localTopKIndices[static_cast<int64_t>(row) * topK + col];
        int64_t const packedToken = requestStart + token;
        if (token >= 0 && static_cast<int64_t>(token) < requestLength && packedToken >= 0
            && packedToken < totalKvTokens)
        {
            int32_t const block = token / tokensPerBlock;
            if (block >= 0 && block < maxBlocksPerRequest)
            {
                int32_t const physicalBlock = blockTable[static_cast<int64_t>(request) * maxBlocksPerRequest + block];
                int64_t const globalIdx = static_cast<int64_t>(physicalBlock) * pageStride + token % tokensPerBlock
                    + static_cast<int64_t>(layerId) * tokensPerBlock;
                if (physicalBlock >= 0 && globalIdx >= 0 && globalIdx < numPoolTokens
                    && globalIdx <= std::numeric_limits<int32_t>::max())
                {
                    selectedGlobalIndices[packedToken] = static_cast<int32_t>(globalIdx);
                    atomicExch(selectedFlags + packedToken, 1);
                }
            }
        }
    }
}

__global__ void nvFp4MlaContextKvCacheGatherKernel(uint8_t const* __restrict__ dataPool,
    __nv_fp8_e4m3 const* __restrict__ scalePool, int32_t const* __restrict__ selectedFlags,
    int32_t const* __restrict__ selectedOffsets, int32_t const* __restrict__ selectedGlobalIndices,
    __nv_fp8_e4m3* __restrict__ output, float const* __restrict__ globalDequantScale, int32_t totalKvTokens,
    int32_t headDim, int64_t numPoolTokens)
{
    int32_t const warp = threadIdx.x / kWarpSize;
    int32_t const lane = threadIdx.x % kWarpSize;
    float const dequantScale = globalDequantScale == nullptr ? 1.F : globalDequantScale[0];
    int64_t packedToken = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    int64_t const tokenStride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
    for (; packedToken < totalKvTokens; packedToken += tokenStride)
    {
        int32_t selected = 0;
        if (lane == 0)
        {
            selected = selectedFlags[packedToken];
        }
        selected = __shfl_sync(0xFFFFFFFFU, selected, 0);
        if (selected == 0)
        {
            continue;
        }

        int32_t globalIdx = -1;
        int32_t outputIdx = 0;
        if (lane == 0)
        {
            globalIdx = selectedGlobalIndices[packedToken];
            outputIdx = selectedOffsets[packedToken];
        }
        globalIdx = __shfl_sync(0xFFFFFFFFU, globalIdx, 0);
        outputIdx = __shfl_sync(0xFFFFFFFFU, outputIdx, 0);
        if (globalIdx >= 0 && static_cast<int64_t>(globalIdx) < numPoolTokens && outputIdx >= 0
            && outputIdx < totalKvTokens)
        {
            if (dequantScale == 1.F)
            {
                dequantizeRow<true>(dataPool, scalePool, output, globalIdx, outputIdx, headDim, dequantScale, lane);
            }
            else
            {
                dequantizeRow<false>(dataPool, scalePool, output, globalIdx, outputIdx, headDim, dequantScale, lane);
            }
        }
    }
}

__global__ void remapContextTopKKernel(int32_t const* __restrict__ localTopKIndices,
    int32_t const* __restrict__ queryReqIndices, int64_t const* __restrict__ cuKvLengths,
    int32_t const* __restrict__ selectedOffsets, int32_t* __restrict__ compactIndices, int64_t numPairs, int32_t topK,
    int32_t numRequests, int32_t totalKvTokens)
{
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair < numPairs;
         pair += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int32_t const row = static_cast<int32_t>(pair / topK);
        int32_t const request = queryReqIndices[row];
        int32_t const token = localTopKIndices[pair];
        int32_t compact = -1;
        if (request >= 0 && request < numRequests && token >= 0)
        {
            int64_t const requestStart = cuKvLengths[request];
            int64_t const requestLength = cuKvLengths[request + 1] - requestStart;
            int64_t const packedToken = requestStart + token;
            if (static_cast<int64_t>(token) < requestLength && packedToken >= 0 && packedToken < totalKvTokens)
            {
                compact = selectedOffsets[packedToken];
            }
        }
        compactIndices[pair] = compact;
    }
}

int32_t getPersistentBlockCount(int64_t workItems)
{
    int32_t const workBlocks = static_cast<int32_t>((workItems + kWarpsPerBlock - 1) / kWarpsPerBlock);
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    return std::max(1, std::min(workBlocks, smCount * kBlocksPerSm));
}

} // namespace

void invokeNvFp4MlaKvCacheGather(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool, int32_t const* globalIndices,
    __nv_fp8_e4m3* output, int32_t* compactIndices, float const* globalDequantScale, int32_t numRows, int32_t topK,
    int32_t headDim, int64_t numPoolTokens, cudaStream_t stream)
{
    if (numRows == 0 || topK == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(headDim > 0 && headDim % 16 == 0,
        "NVFP4 MLA gather requires head_dim to be a positive multiple of 16, got %d", headDim);

    int64_t const numPairs = static_cast<int64_t>(numRows) * topK;
    TLLM_CHECK_WITH_INFO(
        numPairs <= std::numeric_limits<int32_t>::max(), "NVFP4 MLA gather compact indices exceed int32 capacity");
    int32_t const blocks = getPersistentBlockCount(numPairs);
    nvFp4MlaKvCacheGatherKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(dataPool, scalePool, globalIndices, output,
        compactIndices, globalDequantScale, numPairs, headDim, numPoolTokens);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

size_t getNvFp4MlaContextKvCacheGatherWorkspaceSize(int32_t totalKvTokens, cudaStream_t stream)
{
    if (totalKvTokens <= 0)
    {
        return 0;
    }
    size_t scanWorkspaceSize = 0;
    TLLM_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanWorkspaceSize, static_cast<int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr), totalKvTokens, stream));
    size_t const tensorWorkspaceSize
        = alignUp(static_cast<size_t>(totalKvTokens) * sizeof(int32_t) * 3, kWorkspaceAlignment);
    return tensorWorkspaceSize + scanWorkspaceSize;
}

void invokeNvFp4MlaContextKvCacheGather(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool,
    int32_t const* localTopKIndices, int32_t const* queryReqIndices, int32_t const* blockTable,
    int64_t const* cuKvLengths, __nv_fp8_e4m3* output, int32_t* compactIndices, float const* globalDequantScale,
    void* workspace, size_t workspaceSize, int32_t numQueryRows, int32_t topK, int32_t numRequests,
    int32_t maxBlocksPerRequest, int32_t totalKvTokens, int32_t tokensPerBlock, int32_t pageStride, int32_t layerId,
    int32_t headDim, int64_t numPoolTokens, cudaStream_t stream)
{
    if (numQueryRows == 0 || topK == 0 || totalKvTokens == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(headDim > 0 && headDim % 16 == 0,
        "NVFP4 MLA context gather requires head_dim to be a positive multiple of 16, got %d", headDim);
    TLLM_CHECK_WITH_INFO(tokensPerBlock > 0, "tokens_per_block must be positive");
    TLLM_CHECK_WITH_INFO(numRequests > 0, "num_requests must be positive");

    size_t scanWorkspaceSize = 0;
    TLLM_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanWorkspaceSize, static_cast<int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr), totalKvTokens, stream));
    size_t const tensorWorkspaceSize
        = alignUp(static_cast<size_t>(totalKvTokens) * sizeof(int32_t) * 3, kWorkspaceAlignment);
    TLLM_CHECK_WITH_INFO(
        workspaceSize >= tensorWorkspaceSize + scanWorkspaceSize, "NVFP4 MLA context gather workspace is too small");
    auto* selectedFlags = static_cast<int32_t*>(workspace);
    auto* selectedOffsets = selectedFlags + totalKvTokens;
    auto* selectedGlobalIndices = selectedOffsets + totalKvTokens;
    auto* scanWorkspace = reinterpret_cast<uint8_t*>(workspace) + tensorWorkspaceSize;

    TLLM_CUDA_CHECK(cudaMemsetAsync(selectedFlags, 0, static_cast<size_t>(totalKvTokens) * sizeof(int32_t), stream));
    markContextTopKKernel<<<numQueryRows, kThreadsPerBlock, 0, stream>>>(localTopKIndices, queryReqIndices, cuKvLengths,
        blockTable, selectedFlags, selectedGlobalIndices, numQueryRows, topK, numRequests, maxBlocksPerRequest,
        totalKvTokens, tokensPerBlock, pageStride, layerId, numPoolTokens);
    TLLM_CUDA_CHECK(cudaGetLastError());
    TLLM_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scanWorkspace, scanWorkspaceSize, selectedFlags, selectedOffsets, totalKvTokens, stream));

    int32_t const gatherBlocks = getPersistentBlockCount(totalKvTokens);
    nvFp4MlaContextKvCacheGatherKernel<<<gatherBlocks, kThreadsPerBlock, 0, stream>>>(dataPool, scalePool,
        selectedFlags, selectedOffsets, selectedGlobalIndices, output, globalDequantScale, totalKvTokens, headDim,
        numPoolTokens);
    TLLM_CUDA_CHECK(cudaGetLastError());

    int64_t const numPairs = static_cast<int64_t>(numQueryRows) * topK;
    int32_t const remapBlocks = std::min(static_cast<int32_t>((numPairs + kThreadsPerBlock - 1) / kThreadsPerBlock),
        tensorrt_llm::common::getMultiProcessorCount() * kBlocksPerSm);
    remapContextTopKKernel<<<remapBlocks, kThreadsPerBlock, 0, stream>>>(localTopKIndices, queryReqIndices, cuKvLengths,
        selectedOffsets, compactIndices, numPairs, topK, numRequests, totalKvTokens);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END

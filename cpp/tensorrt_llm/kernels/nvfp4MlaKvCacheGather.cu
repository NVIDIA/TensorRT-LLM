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
constexpr int32_t kGenerationBlocksPerSm = 8;
constexpr int32_t kAsyncCopyBytes = 16;
constexpr int32_t kScaleCopyBytes = 4;
constexpr int32_t kMlaHeadDim = 576;
constexpr int32_t kMlaResidualDim = 64;
constexpr int32_t kMlaPackedHeadDim = kMlaHeadDim / 2;
constexpr int32_t kMlaScalesPerToken = kMlaHeadDim / 16;
constexpr int32_t kMlaResidualPackedHeadDim = (kMlaHeadDim + kMlaResidualDim) / 2;
constexpr int32_t kMlaResidualScalesPerToken = (kMlaHeadDim + kMlaResidualDim) / 16;
constexpr int32_t kMlaStagingRowBytes = (kMlaResidualPackedHeadDim + kMlaResidualScalesPerToken + kAsyncCopyBytes - 1)
    / kAsyncCopyBytes * kAsyncCopyBytes;
constexpr int32_t kMlaStagingBuffers = 3;
constexpr size_t kWorkspaceAlignment = 256;

constexpr size_t alignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__device__ __forceinline__ void copyAsync16(void* destination, void const* source, uint32_t sourceBytes)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    uint32_t const sharedAddress = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(sharedAddress), "l"(source), "r"(sourceBytes));
#else
    if (sourceBytes == kAsyncCopyBytes)
    {
        *static_cast<uint4*>(destination) = *static_cast<uint4 const*>(source);
    }
    else
    {
        *static_cast<uint4*>(destination) = make_uint4(0, 0, 0, 0);
    }
#endif
}

__device__ __forceinline__ void copyAsync4(void* destination, void const* source, uint32_t sourceBytes)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    uint32_t const sharedAddress = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :
                 : "r"(sharedAddress), "l"(source), "r"(sourceBytes));
#else
    *static_cast<uint32_t*>(destination) = sourceBytes == kScaleCopyBytes ? *static_cast<uint32_t const*>(source) : 0;
#endif
}

__device__ __forceinline__ void commitAsyncCopies()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__ void waitAsyncCopies()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n");
#endif
}

__device__ __forceinline__ void waitAsyncCopiesKeepOne()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 1;\n");
#endif
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

__device__ __forceinline__ uint2 convertUnitScaledE2m1x8ToE4m3(uint32_t packed, __nv_fp8_e4m3 scale)
{
    uint32_t const low = convertUnitScaledE2m1x4ToE4m3(static_cast<uint16_t>(packed), scale);
    uint32_t const high = convertUnitScaledE2m1x4ToE4m3(static_cast<uint16_t>(packed >> 16), scale);
    return make_uint2(low, high);
}

__device__ __forceinline__ uint4 convertUnitScaledE2m1x16ToE4m3(uint2 packed, __nv_fp8_e4m3 scale)
{
    uint2 const low = convertUnitScaledE2m1x8ToE4m3(packed.x, scale);
    uint2 const high = convertUnitScaledE2m1x8ToE4m3(packed.y, scale);
    return make_uint4(low.x, low.y, high.x, high.y);
}

__device__ __forceinline__ uint32_t convertResidualE2m1x4ToE4m3(
    uint16_t mainPacked, uint16_t residualPacked, __nv_fp8_e4m3 mainScale, __nv_fp8_e4m3 residualScale)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint2 const mainFp16 = convertE2m1x4ToFp16x4(mainPacked);
    uint2 const residualFp16 = convertE2m1x4ToFp16x4(residualPacked);
    __half2 const mainScale2 = __half2half2(static_cast<__half>(mainScale));
    __half2 const residualScale2 = __half2half2(static_cast<__half>(residualScale));
    __half2 const low = __hadd2(__hmul2(reinterpret_cast<__half2 const&>(mainFp16.x), mainScale2),
        __hmul2(reinterpret_cast<__half2 const&>(residualFp16.x), residualScale2));
    __half2 const high = __hadd2(__hmul2(reinterpret_cast<__half2 const&>(mainFp16.y), mainScale2),
        __hmul2(reinterpret_cast<__half2 const&>(residualFp16.y), residualScale2));
    return __nv_fp8x4_e4m3(low, high).__x;
#else
    float4 mainValues = convertE2m1x4ToFloat4(mainPacked);
    float4 residualValues = convertE2m1x4ToFloat4(residualPacked);
    float const mainScaleFloat = static_cast<float>(mainScale);
    float const residualScaleFloat = static_cast<float>(residualScale);
    mainValues.x = mainValues.x * mainScaleFloat + residualValues.x * residualScaleFloat;
    mainValues.y = mainValues.y * mainScaleFloat + residualValues.y * residualScaleFloat;
    mainValues.z = mainValues.z * mainScaleFloat + residualValues.z * residualScaleFloat;
    mainValues.w = mainValues.w * mainScaleFloat + residualValues.w * residualScaleFloat;
    return convertFloat4ToE4m3(mainValues);
#endif
}

__device__ __forceinline__ uint4 convertResidualE2m1x16ToE4m3(
    uint2 mainPacked, uint2 residualPacked, __nv_fp8_e4m3 mainScale, __nv_fp8_e4m3 residualScale)
{
    uint4 output;
    output.x = convertResidualE2m1x4ToE4m3(
        static_cast<uint16_t>(mainPacked.x), static_cast<uint16_t>(residualPacked.x), mainScale, residualScale);
    output.y = convertResidualE2m1x4ToE4m3(static_cast<uint16_t>(mainPacked.x >> 16),
        static_cast<uint16_t>(residualPacked.x >> 16), mainScale, residualScale);
    output.z = convertResidualE2m1x4ToE4m3(
        static_cast<uint16_t>(mainPacked.y), static_cast<uint16_t>(residualPacked.y), mainScale, residualScale);
    output.w = convertResidualE2m1x4ToE4m3(static_cast<uint16_t>(mainPacked.y >> 16),
        static_cast<uint16_t>(residualPacked.y >> 16), mainScale, residualScale);
    return output;
}

__device__ __forceinline__ void prefetchMlaRow(uint8_t* stagingRow, uint8_t const* dataPool,
    __nv_fp8_e4m3 const* scalePool, int32_t globalIdx, int32_t lane, bool compactLayout, int32_t residualDim)
{
    bool const valid = globalIdx >= 0;
    int32_t const packedHeadDim = kMlaPackedHeadDim + residualDim / 2;
    int32_t const scalesPerToken = kMlaScalesPerToken + residualDim / 16;
    if (compactLayout)
    {
        int32_t const compactRowBytes
            = (packedHeadDim + scalesPerToken + kAsyncCopyBytes - 1) / kAsyncCopyBytes * kAsyncCopyBytes;
        auto const* compactRow = valid ? dataPool + static_cast<int64_t>(globalIdx) * compactRowBytes : dataPool;
        if (lane < compactRowBytes / kAsyncCopyBytes)
        {
            copyAsync16(
                stagingRow + lane * kAsyncCopyBytes, compactRow + lane * kAsyncCopyBytes, valid ? kAsyncCopyBytes : 0U);
        }
        commitAsyncCopies();
        return;
    }

    auto const* packedRow = valid ? dataPool + static_cast<int64_t>(globalIdx) * packedHeadDim : dataPool;
    auto const* scaleRow = valid ? scalePool + static_cast<int64_t>(globalIdx) * scalesPerToken : scalePool;
    uint32_t const dataBytes = valid ? kAsyncCopyBytes : 0U;
    uint32_t const scaleBytes = valid ? kScaleCopyBytes : 0U;
    int32_t const dataCopyLanes = packedHeadDim / kAsyncCopyBytes;
    int32_t const scaleCopyLanes = scalesPerToken / kScaleCopyBytes;
    if (lane < dataCopyLanes)
    {
        copyAsync16(stagingRow + lane * kAsyncCopyBytes, packedRow + lane * kAsyncCopyBytes, dataBytes);
    }
    else if (lane < dataCopyLanes + scaleCopyLanes)
    {
        int32_t const scaleLane = lane - dataCopyLanes;
        copyAsync4(stagingRow + packedHeadDim + scaleLane * kScaleCopyBytes, scaleRow + scaleLane * kScaleCopyBytes,
            scaleBytes);
    }
    commitAsyncCopies();
}

__device__ __forceinline__ int32_t fetchGlobalIndex(
    int32_t const* globalIndices, int32_t* compactIndices, int64_t pair, int64_t numPoolTokens, int32_t lane)
{
    int32_t globalIdx = -1;
    if (lane == 0)
    {
        globalIdx = globalIndices[pair];
        bool const valid = globalIdx >= 0 && static_cast<int64_t>(globalIdx) < numPoolTokens;
        compactIndices[pair] = valid ? static_cast<int32_t>(pair) : -1;
        globalIdx = valid ? globalIdx : -1;
    }
    return __shfl_sync(0xFFFFFFFFU, globalIdx, 0);
}

template <bool kUnitGlobalScale>
__device__ __forceinline__ void dequantizeRow(uint8_t const* __restrict__ dataPool,
    __nv_fp8_e4m3 const* __restrict__ scalePool, __nv_fp8_e4m3* __restrict__ output, int32_t globalIdx,
    int32_t outputIdx, int32_t headDim, int32_t residualDim, float dequantScale, int32_t lane)
{
    int32_t const packedHeadDim = (headDim + residualDim) / 2;
    int32_t const scalesPerToken = (headDim + residualDim) / 16;
    int32_t const residualStartGroup = (headDim - residualDim) / 16;
    auto const* packedRow = dataPool + static_cast<int64_t>(globalIdx) * packedHeadDim;
    auto const* scaleRow = scalePool + static_cast<int64_t>(globalIdx) * scalesPerToken;
    auto* outputRow = output + static_cast<int64_t>(outputIdx) * headDim;

    if constexpr (kUnitGlobalScale)
    {
        // E2M1 times E4M3 is exactly representable in FP16, so each lane can
        // convert a complete 16-value scaling group and emit one 16-byte store.
        int32_t const outputGroups = headDim / 16;
        for (int32_t groupBase = 0; groupBase < outputGroups; groupBase += kWarpSize)
        {
            int32_t const group = groupBase + lane;
            if (group < outputGroups)
            {
                uint4 values;
                if (group < residualStartGroup)
                {
                    uint2 const packed = *reinterpret_cast<uint2 const*>(packedRow + static_cast<int64_t>(group) * 8);
                    values = convertUnitScaledE2m1x16ToE4m3(packed, scaleRow[group]);
                }
                else
                {
                    int32_t const residualGroup = group - residualStartGroup;
                    auto const* residualData = packedRow + static_cast<int64_t>(residualStartGroup) * 8
                        + static_cast<int64_t>(residualGroup) * 16;
                    uint2 const mainPacked = *reinterpret_cast<uint2 const*>(residualData);
                    uint2 const residualPacked = *reinterpret_cast<uint2 const*>(residualData + 8);
                    int32_t const scaleIdx = residualStartGroup + residualGroup * 2;
                    values = convertResidualE2m1x16ToE4m3(
                        mainPacked, residualPacked, scaleRow[scaleIdx], scaleRow[scaleIdx + 1]);
                }
                *reinterpret_cast<uint4*>(outputRow + static_cast<int64_t>(group) * 16) = values;
            }
        }
    }
    else
    {
        // Four neighboring lanes cooperatively convert one scaling group. FP32
        // math preserves the result for arbitrary global dequantization scales.
        int32_t const groupLane = lane % 4;
        int32_t const outputGroups = headDim / 16;
        for (int32_t groupBase = 0; groupBase < outputGroups; groupBase += kWarpSize / 4)
        {
            int32_t const group = groupBase + lane / 4;
            bool const active = group < outputGroups;
            uint32_t const activeMask = __ballot_sync(0xFFFFFFFFU, active);
            if (!active)
            {
                continue;
            }
            uint16_t mainPacked;
            uint16_t residualPacked = 0;
            int32_t mainScaleIdx;
            int32_t residualScaleIdx = -1;
            if (group < residualStartGroup)
            {
                mainPacked
                    = *reinterpret_cast<uint16_t const*>(packedRow + static_cast<int64_t>(group) * 8 + groupLane * 2);
                mainScaleIdx = group;
            }
            else
            {
                int32_t const residualGroup = group - residualStartGroup;
                auto const* residualData = packedRow + static_cast<int64_t>(residualStartGroup) * 8
                    + static_cast<int64_t>(residualGroup) * 16;
                mainPacked = *reinterpret_cast<uint16_t const*>(residualData + groupLane * 2);
                residualPacked = *reinterpret_cast<uint16_t const*>(residualData + 8 + groupLane * 2);
                mainScaleIdx = residualStartGroup + residualGroup * 2;
                residualScaleIdx = mainScaleIdx + 1;
            }
            float mainScale = groupLane == 0 ? static_cast<float>(scaleRow[mainScaleIdx]) * dequantScale : 0.F;
            mainScale = __shfl_sync(activeMask, mainScale, lane - groupLane);
            float4 values = convertE2m1x4ToFloat4(mainPacked);
            values.x *= mainScale;
            values.y *= mainScale;
            values.z *= mainScale;
            values.w *= mainScale;
            if (residualScaleIdx >= 0)
            {
                float residualScale
                    = groupLane == 0 ? static_cast<float>(scaleRow[residualScaleIdx]) * dequantScale : 0.F;
                residualScale = __shfl_sync(activeMask, residualScale, lane - groupLane);
                float4 residualValues = convertE2m1x4ToFloat4(residualPacked);
                values.x += residualValues.x * residualScale;
                values.y += residualValues.y * residualScale;
                values.z += residualValues.z * residualScale;
                values.w += residualValues.w * residualScale;
            }
            *reinterpret_cast<uint32_t*>(outputRow + static_cast<int64_t>(group) * 16 + groupLane * 4)
                = convertFloat4ToE4m3(values);
        }
    }
}

__global__ __launch_bounds__(kThreadsPerBlock, kGenerationBlocksPerSm) void nvFp4MlaKvCacheGatherKernel(
    uint8_t const* __restrict__ dataPool, __nv_fp8_e4m3 const* __restrict__ scalePool,
    int32_t const* __restrict__ globalIndices, __nv_fp8_e4m3* __restrict__ output, int32_t* __restrict__ compactIndices,
    float const* __restrict__ globalDequantScale, int64_t numPairs, int32_t headDim, int32_t residualDim,
    int64_t numPoolTokens)
{
    int32_t const warp = threadIdx.x / kWarpSize;
    int32_t const lane = threadIdx.x % kWarpSize;
    float const dequantScale = globalDequantScale == nullptr ? 1.F : globalDequantScale[0];
    int32_t const packedHeadDim = kMlaPackedHeadDim + residualDim / 2;
    bool const compactLayout = reinterpret_cast<uint8_t const*>(scalePool) == dataPool + packedHeadDim;
    __shared__ __align__(16) uint8_t staging[kMlaStagingBuffers][kWarpsPerBlock][kMlaStagingRowBytes];
    int64_t pair = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    int64_t const pairStride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
    if (headDim == kMlaHeadDim && (residualDim == 0 || residualDim == kMlaResidualDim))
    {
        if (pair >= numPairs)
        {
            return;
        }

        int32_t globalIdx = fetchGlobalIndex(globalIndices, compactIndices, pair, numPoolTokens, lane);
        int32_t stage = 0;
        prefetchMlaRow(staging[stage][warp], dataPool, scalePool, globalIdx, lane, compactLayout, residualDim);
        int64_t nextPair = pair + pairStride;
        bool hasNext = nextPair < numPairs;
        int32_t nextGlobalIdx = -1;
        if (hasNext)
        {
            nextGlobalIdx = fetchGlobalIndex(globalIndices, compactIndices, nextPair, numPoolTokens, lane);
            prefetchMlaRow(staging[1][warp], dataPool, scalePool, nextGlobalIdx, lane, compactLayout, residualDim);
        }
        while (true)
        {
            if (hasNext)
            {
                waitAsyncCopiesKeepOne();
            }
            else
            {
                waitAsyncCopies();
            }
            __syncwarp();

            int64_t const followingPair = nextPair + pairStride;
            bool const hasFollowing = hasNext && followingPair < numPairs;
            int32_t followingGlobalIdx = -1;
            if (hasFollowing)
            {
                followingGlobalIdx
                    = fetchGlobalIndex(globalIndices, compactIndices, followingPair, numPoolTokens, lane);
                int32_t const prefetchStage = stage == 0 ? 2 : stage - 1;
                prefetchMlaRow(staging[prefetchStage][warp], dataPool, scalePool, followingGlobalIdx, lane,
                    compactLayout, residualDim);
            }

            if (globalIdx >= 0)
            {
                auto const* stagedScales = reinterpret_cast<__nv_fp8_e4m3 const*>(
                    staging[stage][warp] + kMlaPackedHeadDim + residualDim / 2);
                if (dequantScale == 1.F)
                {
                    dequantizeRow<true>(staging[stage][warp], stagedScales, output, 0, static_cast<int32_t>(pair),
                        kMlaHeadDim, residualDim, 1.F, lane);
                }
                else
                {
                    dequantizeRow<false>(staging[stage][warp], stagedScales, output, 0, static_cast<int32_t>(pair),
                        kMlaHeadDim, residualDim, dequantScale, lane);
                }
            }
            if (!hasNext)
            {
                break;
            }
            pair = nextPair;
            globalIdx = nextGlobalIdx;
            nextPair = followingPair;
            nextGlobalIdx = followingGlobalIdx;
            hasNext = hasFollowing;
            stage = stage == kMlaStagingBuffers - 1 ? 0 : stage + 1;
        }
        return;
    }

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
                dequantizeRow<true>(dataPool, scalePool, output, globalIdx, static_cast<int32_t>(pair), headDim,
                    residualDim, dequantScale, lane);
            }
            else
            {
                dequantizeRow<false>(dataPool, scalePool, output, globalIdx, static_cast<int32_t>(pair), headDim,
                    residualDim, dequantScale, lane);
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
    int32_t headDim, int32_t residualDim, int64_t numPoolTokens)
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
                dequantizeRow<true>(
                    dataPool, scalePool, output, globalIdx, outputIdx, headDim, residualDim, dequantScale, lane);
            }
            else
            {
                dequantizeRow<false>(
                    dataPool, scalePool, output, globalIdx, outputIdx, headDim, residualDim, dequantScale, lane);
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

int32_t getPersistentBlockCount(int64_t workItems, int32_t blocksPerSm = kBlocksPerSm)
{
    int32_t const workBlocks = static_cast<int32_t>((workItems + kWarpsPerBlock - 1) / kWarpsPerBlock);
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    return std::max(1, std::min(workBlocks, smCount * blocksPerSm));
}

} // namespace

void invokeNvFp4MlaKvCacheGather(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool, int32_t const* globalIndices,
    __nv_fp8_e4m3* output, int32_t* compactIndices, float const* globalDequantScale, int32_t numRows, int32_t topK,
    int32_t headDim, int32_t residualDim, int64_t numPoolTokens, cudaStream_t stream)
{
    if (numRows == 0 || topK == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(headDim > 0 && headDim % 16 == 0,
        "NVFP4 MLA gather requires head_dim to be a positive multiple of 16, got %d", headDim);
    TLLM_CHECK_WITH_INFO(residualDim >= 0 && residualDim <= headDim && residualDim % 16 == 0,
        "NVFP4 MLA gather residual_dim must be a multiple of 16 in [0, head_dim], got %d", residualDim);

    int64_t const numPairs = static_cast<int64_t>(numRows) * topK;
    TLLM_CHECK_WITH_INFO(
        numPairs <= std::numeric_limits<int32_t>::max(), "NVFP4 MLA gather compact indices exceed int32 capacity");
    int32_t const blocks = getPersistentBlockCount(numPairs, kGenerationBlocksPerSm);
    nvFp4MlaKvCacheGatherKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(dataPool, scalePool, globalIndices, output,
        compactIndices, globalDequantScale, numPairs, headDim, residualDim, numPoolTokens);
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
    int32_t headDim, int32_t residualDim, int64_t numPoolTokens, cudaStream_t stream)
{
    if (numQueryRows == 0 || topK == 0 || totalKvTokens == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(headDim > 0 && headDim % 16 == 0,
        "NVFP4 MLA context gather requires head_dim to be a positive multiple of 16, got %d", headDim);
    TLLM_CHECK_WITH_INFO(residualDim >= 0 && residualDim <= headDim && residualDim % 16 == 0,
        "NVFP4 MLA context gather residual_dim must be a multiple of 16 in [0, head_dim], got %d", residualDim);
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
        residualDim, numPoolTokens);
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

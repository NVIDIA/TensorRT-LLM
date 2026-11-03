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

#include "glmNvfp4OwnerWorklist.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"

#include <algorithm>
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
constexpr int32_t kAsyncCopyBytes = 16;
constexpr int32_t kScaleCopyBytes = 8;
constexpr int32_t kMlaHeadDim = 576;
constexpr int32_t kMlaResidualDim = 64;
constexpr int32_t kMlaPackedHeadDim = kMlaHeadDim / 2;
constexpr int32_t kMlaScalesPerToken = kMlaHeadDim / 16;

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

__device__ __forceinline__ void copyAsync8(void* destination, void const* source, uint32_t sourceBytes)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    uint32_t const sharedAddress = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                 :
                 : "r"(sharedAddress), "l"(source), "r"(sourceBytes));
#else
    uint2 value = make_uint2(0, 0);
    if (sourceBytes == kScaleCopyBytes)
    {
        value = *static_cast<uint2 const*>(source);
    }
    else if (sourceBytes == sizeof(uint32_t))
    {
        value.x = *static_cast<uint32_t const*>(source);
    }
    *static_cast<uint2*>(destination) = value;
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

__device__ __forceinline__ uint32_t convertGloballyScaledE2m1x4ToE4m3(
    uint16_t packed, __nv_fp8_e4m3 scale, float globalScale)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint2 const fp16 = convertE2m1x4ToFp16x4(packed);
    __half2 const scale2 = __float2half2_rn(static_cast<float>(scale) * globalScale);
    __half2 const low = __hmul2(reinterpret_cast<__half2 const&>(fp16.x), scale2);
    __half2 const high = __hmul2(reinterpret_cast<__half2 const&>(fp16.y), scale2);
    return __nv_fp8x4_e4m3(low, high).__x;
#else
    float4 values = convertE2m1x4ToFloat4(packed);
    float const combinedScale = static_cast<float>(scale) * globalScale;
    values.x *= combinedScale;
    values.y *= combinedScale;
    values.z *= combinedScale;
    values.w *= combinedScale;
    return convertFloat4ToE4m3(values);
#endif
}

__device__ __forceinline__ uint4 convertGloballyScaledE2m1x16ToE4m3(
    uint2 packed, __nv_fp8_e4m3 scale, float globalScale)
{
    uint4 output;
    output.x = convertGloballyScaledE2m1x4ToE4m3(static_cast<uint16_t>(packed.x), scale, globalScale);
    output.y = convertGloballyScaledE2m1x4ToE4m3(static_cast<uint16_t>(packed.x >> 16), scale, globalScale);
    output.z = convertGloballyScaledE2m1x4ToE4m3(static_cast<uint16_t>(packed.y), scale, globalScale);
    output.w = convertGloballyScaledE2m1x4ToE4m3(static_cast<uint16_t>(packed.y >> 16), scale, globalScale);
    return output;
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

__device__ __forceinline__ uint32_t convertGloballyScaledResidualE2m1x4ToE4m3(uint16_t mainPacked,
    uint16_t residualPacked, __nv_fp8_e4m3 mainScale, __nv_fp8_e4m3 residualScale, float globalScale)
{
    // Keep the second-level sum in FP32. Folding both scales into FP16 can
    // move values across an E4M3 rounding boundary even though the main-only
    // product is bit-exact after the final FP8 conversion.
    float4 mainValues = convertE2m1x4ToFloat4(mainPacked);
    float4 residualValues = convertE2m1x4ToFloat4(residualPacked);
    float const mainCombinedScale = static_cast<float>(mainScale) * globalScale;
    float const residualCombinedScale = static_cast<float>(residualScale) * globalScale;
    mainValues.x = mainValues.x * mainCombinedScale + residualValues.x * residualCombinedScale;
    mainValues.y = mainValues.y * mainCombinedScale + residualValues.y * residualCombinedScale;
    mainValues.z = mainValues.z * mainCombinedScale + residualValues.z * residualCombinedScale;
    mainValues.w = mainValues.w * mainCombinedScale + residualValues.w * residualCombinedScale;
    return convertFloat4ToE4m3(mainValues);
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
    int32_t const dataCopyLanes = packedHeadDim / kAsyncCopyBytes;
    int32_t const scaleCopyLanes = (scalesPerToken + kScaleCopyBytes - 1) / kScaleCopyBytes;
    if (lane < dataCopyLanes)
    {
        copyAsync16(stagingRow + lane * kAsyncCopyBytes, packedRow + lane * kAsyncCopyBytes, dataBytes);
    }
    else if (lane < dataCopyLanes + scaleCopyLanes)
    {
        int32_t const scaleLane = lane - dataCopyLanes;
        int32_t const remainingScales = scalesPerToken - scaleLane * kScaleCopyBytes;
        uint32_t const scaleBytes = valid ? static_cast<uint32_t>(min(kScaleCopyBytes, remainingScales)) : 0U;
        copyAsync8(stagingRow + packedHeadDim + scaleLane * kScaleCopyBytes, scaleRow + scaleLane * kScaleCopyBytes,
            scaleBytes);
    }
    commitAsyncCopies();
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
        // convert a complete main scaling group and emit one 16-byte store.
        for (int32_t groupBase = 0; groupBase < residualStartGroup; groupBase += kWarpSize)
        {
            int32_t const group = groupBase + lane;
            if (group < residualStartGroup)
            {
                uint2 const packed = *reinterpret_cast<uint2 const*>(packedRow + static_cast<int64_t>(group) * 8);
                uint4 const values = convertUnitScaledE2m1x16ToE4m3(packed, scaleRow[group]);
                *reinterpret_cast<uint4*>(outputRow + static_cast<int64_t>(group) * 16) = values;
            }
        }

        // Four neighboring lanes cooperate on one residual group. This keeps
        // 16 lanes active for GLM's four RoPE groups instead of serializing the
        // second-level dequantization in only four lanes.
        int32_t const residualGroup = lane / 4;
        int32_t const groupLane = lane % 4;
        if (residualGroup < residualDim / 16)
        {
            auto const* residualData
                = packedRow + static_cast<int64_t>(residualStartGroup) * 8 + static_cast<int64_t>(residualGroup) * 16;
            uint16_t const mainPacked
                = *reinterpret_cast<uint16_t const*>(residualData + static_cast<int64_t>(groupLane) * 2);
            uint16_t const residualPacked
                = *reinterpret_cast<uint16_t const*>(residualData + 8 + static_cast<int64_t>(groupLane) * 2);
            int32_t const scaleIdx = residualStartGroup + residualGroup * 2;
            uint32_t const values
                = convertResidualE2m1x4ToE4m3(mainPacked, residualPacked, scaleRow[scaleIdx], scaleRow[scaleIdx + 1]);
            int32_t const outputGroup = residualStartGroup + residualGroup;
            *reinterpret_cast<uint32_t*>(
                outputRow + static_cast<int64_t>(outputGroup) * 16 + static_cast<int64_t>(groupLane) * 4)
                = values;
        }
    }
    else
    {
        // Fold the global scale into each E4M3 block scale. One lane then
        // converts a complete main group and emits one coalesced 16B store,
        // matching the unit-scale path's work distribution.
        for (int32_t groupBase = 0; groupBase < residualStartGroup; groupBase += kWarpSize)
        {
            int32_t const group = groupBase + lane;
            if (group < residualStartGroup)
            {
                uint2 const packed = *reinterpret_cast<uint2 const*>(packedRow + static_cast<int64_t>(group) * 8);
                uint4 const values = convertGloballyScaledE2m1x16ToE4m3(packed, scaleRow[group], dequantScale);
                *reinterpret_cast<uint4*>(outputRow + static_cast<int64_t>(group) * 16) = values;
            }
        }

        int32_t const residualGroup = lane / 4;
        int32_t const groupLane = lane % 4;
        if (residualGroup < residualDim / 16)
        {
            auto const* residualData
                = packedRow + static_cast<int64_t>(residualStartGroup) * 8 + static_cast<int64_t>(residualGroup) * 16;
            uint16_t const mainPacked
                = *reinterpret_cast<uint16_t const*>(residualData + static_cast<int64_t>(groupLane) * 2);
            uint16_t const residualPacked
                = *reinterpret_cast<uint16_t const*>(residualData + 8 + static_cast<int64_t>(groupLane) * 2);
            int32_t const scaleIdx = residualStartGroup + residualGroup * 2;
            uint32_t const values = convertGloballyScaledResidualE2m1x4ToE4m3(
                mainPacked, residualPacked, scaleRow[scaleIdx], scaleRow[scaleIdx + 1], dequantScale);
            int32_t const outputGroup = residualStartGroup + residualGroup;
            *reinterpret_cast<uint32_t*>(
                outputRow + static_cast<int64_t>(outputGroup) * 16 + static_cast<int64_t>(groupLane) * 4)
                = values;
        }
    }
}

__device__ __forceinline__ uint32_t ownerWorklistHash(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

// Pipelined owner gather. A leader call publishes, for every layer of its top-k group, an owner list of
// {output pair, physical index} (publishOwnerMembersKernel) and gathers its own layer from it; a layer that
// reuses the top-k gathers from its own list (nvFp4MlaMemberOwnerGatherKernel). Rows are dequantized with the
// separate data/scale-pool staging layout and NVFP4-to-FP8 conversions.
constexpr int32_t kMlaRowStagingBytes
    = (kMlaPackedHeadDim + kMlaResidualDim / 2 + kMlaScalesPerToken + kMlaResidualDim / 16 + kAsyncCopyBytes - 1)
    / kAsyncCopyBytes * kAsyncCopyBytes;
// Member indices the publication loads with its own; the rest are loaded after the block's prefix sum.
constexpr int32_t kPublishEarlyMembers = 3;
// Owner rows per warp and round in the member gather, all in flight at once.
constexpr int32_t kMemberGatherChunkRows = 4;
constexpr int32_t kMemberGatherBlocksPerSm = 6;

template <int32_t kPendingGroups>
__device__ __forceinline__ void waitAsyncCopiesAllBut()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kPendingGroups));
#endif
}

__device__ __forceinline__ void prefetchL2(void const* address)
{
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(address));
}

__device__ __forceinline__ void waitPrerequisiteGrids()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

__device__ __forceinline__ int32_t loadAcquire(int32_t const* address)
{
    int32_t value;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void launchDependentGrids()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;\n" :::);
#endif
}

__device__ __forceinline__ void dequantizeStagedMlaRow(
    uint8_t const* stagingRow, __nv_fp8_e4m3* output, int32_t outputIdx, float dequantScale, int32_t lane)
{
    constexpr int32_t kPackedHeadDim = kMlaPackedHeadDim + kMlaResidualDim / 2;
    auto const* stagedScales = reinterpret_cast<__nv_fp8_e4m3 const*>(stagingRow + kPackedHeadDim);
    if (dequantScale == 1.F)
    {
        dequantizeRow<true>(stagingRow, stagedScales, output, 0, outputIdx, kMlaHeadDim, kMlaResidualDim, 1.F, lane);
    }
    else
    {
        dequantizeRow<false>(
            stagingRow, stagedScales, output, 0, outputIdx, kMlaHeadDim, kMlaResidualDim, dequantScale, lane);
    }
}

__device__ __forceinline__ int32_t validPhysicalOrInvalid(int32_t physical, int64_t numPoolTokens)
{
    return physical >= 0 && static_cast<int64_t>(physical) < numPoolTokens ? physical : -1;
}

// The publication of a pipelined leader call deduplicates physical rows. The blocks
// adding their owner counts to ownerState[0] (zero on entry) and writing {pair, member physical index or
// -1} of each owner at its owner position to the list of every member of the top-k group (memberOwners +
// member * memberStride; member 0 is this layer). Each new owner's row of member 0 is prefetched into L2
// for the gather that follows, which this grid lets launch at its start; that gather waits for this grid
// to complete before it reads anything, then stores the owner count and zeroes ownerState.
__global__ __launch_bounds__(kThreadsPerBlock) void publishOwnerMembersKernel(uint8_t const* __restrict__ dataPool,
    __nv_fp8_e4m3 const* __restrict__ scalePool, int32_t const* __restrict__ memberIndices, int32_t numMembers,
    int32_t const* __restrict__ queryReqIndices, int32_t* __restrict__ compactIndices,
    unsigned long long* __restrict__ ownerTable, int2* __restrict__ memberOwners, int64_t memberStride,
    int32_t* __restrict__ ownerState, int32_t numPairs, int32_t topK, int32_t topKShift, int32_t numRequests,
    int32_t tableSlots, int64_t numPoolTokens)
{
    launchDependentGrids();
    int64_t const pair = static_cast<int64_t>(blockIdx.x) * kThreadsPerBlock + threadIdx.x;
    bool const active = pair < numPairs;
    int32_t physical = -1;
    int32_t request = -1;
    int32_t earlyMemberPhysical[kPublishEarlyMembers];
#pragma unroll
    for (int32_t member = 0; member < kPublishEarlyMembers; ++member)
    {
        earlyMemberPhysical[member] = -1;
    }
    if (active)
    {
        int32_t const pair32 = static_cast<int32_t>(pair);
        int32_t const row = topKShift >= 0 ? pair32 >> topKShift : pair32 / topK;
        physical = memberIndices[pair];
        request = queryReqIndices[row];
#pragma unroll
        for (int32_t member = 0; member < kPublishEarlyMembers; ++member)
        {
            if (member + 1 < numMembers)
            {
                earlyMemberPhysical[member] = memberIndices[static_cast<int64_t>(member + 1) * numPairs + pair];
            }
        }
    }
    bool winner = false;
    if (active)
    {
        int32_t compact = -1;
        if (physical >= 0 && static_cast<int64_t>(physical) < numPoolTokens)
        {
            // A full table or unexpected request ID may lose deduplication, never a valid value.
            compact = static_cast<int32_t>(pair);
            winner = true;
            if (request >= 0 && request < numRequests)
            {
                auto* table = ownerTable + static_cast<int64_t>(request) * tableSlots;
                uint32_t slot = ownerWorklistHash(static_cast<uint32_t>(physical)) & (tableSlots - 1);
                unsigned long long const entry
                    = (static_cast<unsigned long long>(static_cast<uint32_t>(physical)) << 32)
                    | static_cast<uint32_t>(pair);
                for (int32_t probe = 0; probe < tableSlots; ++probe, slot = (slot + 1) & (tableSlots - 1))
                {
                    unsigned long long const old = atomicCAS(table + slot, ~0ULL, entry);
                    if (old == ~0ULL || static_cast<uint32_t>(old >> 32) == static_cast<uint32_t>(physical))
                    {
                        int32_t const owner = old == ~0ULL ? static_cast<int32_t>(pair)
                                                           : static_cast<int32_t>(static_cast<uint32_t>(old));
                        compact = owner;
                        winner = owner == pair;
                        break;
                    }
                }
            }
        }
        compactIndices[pair] = compact;
    }
    if (winner)
    {
        uint8_t const* row = dataPool + static_cast<int64_t>(physical) * (kMlaPackedHeadDim + kMlaResidualDim / 2);
        auto const* scales = reinterpret_cast<uint8_t const*>(
            scalePool + static_cast<int64_t>(physical) * (kMlaScalesPerToken + kMlaResidualDim / 16));
        prefetchL2(row);
        prefetchL2(row + 128);
        prefetchL2(row + 256);
        prefetchL2(row + kMlaPackedHeadDim + kMlaResidualDim / 2 - 1);
        prefetchL2(scales);
        prefetchL2(scales + kMlaScalesPerToken + kMlaResidualDim / 16 - 1);
    }

    // Invalid IDs and partial blocks still participate in every collective.
    int32_t const lane = threadIdx.x % kWarpSize;
    int32_t const warp = threadIdx.x / kWarpSize;
    unsigned int const mask = __ballot_sync(0xFFFFFFFFU, winner);
    int32_t const rank = __popc(mask & ((1U << lane) - 1));
    __shared__ int32_t counts[kWarpsPerBlock], offsets[kWarpsPerBlock], base;
    if (lane == 0)
    {
        counts[warp] = __popc(mask);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int32_t total = 0;
        for (int32_t i = 0; i < kWarpsPerBlock; ++i)
        {
            offsets[i] = total;
            total += counts[i];
        }
        base = total != 0 ? atomicAdd(ownerState, total) : 0;
    }
    __syncthreads();
    if (winner)
    {
        int64_t const owner = static_cast<int64_t>(base) + offsets[warp] + rank;
        int32_t const pair32 = static_cast<int32_t>(pair);
        memberOwners[owner] = make_int2(pair32, physical);
#pragma unroll
        for (int32_t member = 0; member < kPublishEarlyMembers; ++member)
        {
            if (member + 1 < numMembers)
            {
                memberOwners[(member + 1) * memberStride + owner]
                    = make_int2(pair32, validPhysicalOrInvalid(earlyMemberPhysical[member], numPoolTokens));
            }
        }
        for (int32_t member = kPublishEarlyMembers + 1; member < numMembers; ++member)
        {
            int32_t const value = memberIndices[static_cast<int64_t>(member) * numPairs + pair];
            memberOwners[member * memberStride + owner]
                = make_int2(pair32, validPhysicalOrInvalid(value, numPoolTokens));
        }
    }
}

// The rows of one layer from its owner list: entries [0, owner count) of memberOwners hold {output pair,
// physical index} (publishOwnerMembersKernel). Warp w of block b takes chunks of kChunkRows consecutive
// entries, chunk b * warps + w first, then every gridDim.x * warps chunks, with all rows of a chunk in
// flight. Nothing is read before griddepcontrol.wait (programmatic dependent launch); before it, the
// warp's first entries and the count are prefetched into L2. A layer that reuses a leader's top-k reads
// the owner count from ownerCount. In the leader call (kLeader) thread 0 of each block reads it from
// ownerState[0], where the publication that just completed accumulated it, and counts the block in
// ownerState[1]; the block that counts last stores the count to ownerCount and zeroes ownerState when it
// finishes, after every block has read the count. The grid also refills ownerTableVectors 16-byte vectors
// of the owner table with the empty key. A pair whose physical index is invalid gets compact index -1.
template <int32_t kChunkRows, bool kLeader>
__global__ __launch_bounds__(kThreadsPerBlock, kMemberGatherBlocksPerSm) void nvFp4MlaMemberOwnerGatherKernel(
    uint8_t const* __restrict__ dataPool, __nv_fp8_e4m3 const* __restrict__ scalePool,
    int2 const* __restrict__ memberOwners, int32_t* __restrict__ ownerCount, __nv_fp8_e4m3* __restrict__ output,
    int32_t* __restrict__ compactIndices, float const* __restrict__ globalDequantScale, int32_t numPairs,
    int32_t* __restrict__ ownerState, uint4* __restrict__ ownerTable, int64_t ownerTableVectors)
{
    static_assert(kChunkRows <= kWarpSize);
    __shared__ __align__(16) uint8_t staging[kWarpsPerBlock][kChunkRows][kMlaRowStagingBytes];
    __shared__ int32_t leaderOwners;
    int32_t const lane = threadIdx.x % kWarpSize;
    int32_t const warp = threadIdx.x / kWarpSize;
    int32_t const* const countSource = kLeader ? ownerState : ownerCount;
    int64_t const chunkStride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
    int64_t chunk = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    int64_t first = chunk * kChunkRows;
    if (lane == 0 && first < numPairs)
    {
        prefetchL2(memberOwners + first);
        prefetchL2(countSource);
    }
    waitPrerequisiteGrids();
    int2 entry = make_int2(-1, -1);
    if (lane < kChunkRows && first + lane < numPairs)
    {
        entry = memberOwners[first + lane];
    }
    int32_t selectedOwners = 0;
    int32_t countedBefore = 0;
    if constexpr (kLeader)
    {
        if (threadIdx.x == 0)
        {
            // The acquire orders the count read before the block counts itself; the result is checked when
            // the block finishes.
            leaderOwners = loadAcquire(ownerState);
            countedBefore = atomicAdd(ownerState + 1, 1);
        }
        __syncthreads();
        selectedOwners = leaderOwners;
    }
    else
    {
        selectedOwners = *ownerCount;
    }
    float const dequantScale = globalDequantScale == nullptr ? 1.F : globalDequantScale[0];
    if constexpr (kLeader)
    {
        uint4 const emptyKeys = make_uint4(~0U, ~0U, ~0U, ~0U);
        int64_t const stride = static_cast<int64_t>(gridDim.x) * kThreadsPerBlock;
        for (int64_t vector = static_cast<int64_t>(blockIdx.x) * kThreadsPerBlock + threadIdx.x;
             vector < ownerTableVectors; vector += stride)
        {
            ownerTable[vector] = emptyKeys;
        }
    }
    bool const validCount = selectedOwners >= 0 && selectedOwners <= numPairs;
    while (validCount && first < selectedOwners)
    {
        int32_t const rows = static_cast<int32_t>(min(static_cast<int64_t>(kChunkRows), selectedOwners - first));
        bool const inRange = entry.x >= 0 && entry.x < numPairs;
        if (lane < rows && inRange && entry.y < 0)
        {
            compactIndices[entry.x] = -1;
        }
        int32_t const pairSource = inRange ? entry.x : -1;
        int32_t const physicalSource = inRange ? entry.y : -1;
#pragma unroll
        for (int32_t row = 0; row < kChunkRows; ++row)
        {
            int32_t const physical = __shfl_sync(0xFFFFFFFFU, physicalSource, row);
            if (row < rows)
            {
                prefetchMlaRow(staging[warp][row], dataPool, scalePool, physical, lane, false, kMlaResidualDim);
            }
            else
            {
                commitAsyncCopies();
            }
        }
        // One cp.async group per row: waiting for all but the newest kChunkRows - 1 groups completes the
        // oldest outstanding row; each step commits an empty group to keep that count.
#pragma unroll
        for (int32_t row = 0; row < kChunkRows; ++row)
        {
            waitAsyncCopiesAllBut<kChunkRows - 1>();
            __syncwarp();
            int32_t const physical = __shfl_sync(0xFFFFFFFFU, physicalSource, row);
            int32_t const outputPair = __shfl_sync(0xFFFFFFFFU, pairSource, row);
            if (row < rows && physical >= 0)
            {
                dequantizeStagedMlaRow(staging[warp][row], output, outputPair, dequantScale, lane);
            }
            __syncwarp();
            commitAsyncCopies();
        }
        waitAsyncCopies();
        chunk += chunkStride;
        first = chunk * kChunkRows;
        entry = make_int2(-1, -1);
        if (lane < kChunkRows && first + lane < numPairs && first < selectedOwners)
        {
            entry = memberOwners[first + lane];
        }
    }
    if (kLeader && threadIdx.x == 0 && countedBefore == static_cast<int32_t>(gridDim.x) - 1)
    {
        // Every block has read the count: store it for the member calls, zero the state for the next
        // publication.
        *ownerCount = selectedOwners;
        ownerState[0] = 0;
        ownerState[1] = 0;
    }
}

template <bool kLeader>
void launchMemberOwnerGather(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool, int2 const* memberOwners,
    int32_t* ownerCount, __nv_fp8_e4m3* output, int32_t* compactIndices, float const* globalDequantScale,
    int64_t numPairs, int32_t* ownerState, uint4* ownerTable, int64_t ownerTableVectors, cudaStream_t stream)
{
    int64_t const chunks = (numPairs + kMemberGatherChunkRows - 1) / kMemberGatherChunkRows;
    int32_t const workBlocks = static_cast<int32_t>((chunks + kWarpsPerBlock - 1) / kWarpsPerBlock);
    int32_t const blocks
        = std::min(workBlocks, tensorrt_llm::common::getMultiProcessorCount() * kMemberGatherBlocksPerSm);
    tensorrt_llm::common::launchWithPdlWhenEnabled("nvFp4MlaMemberOwnerGather",
        nvFp4MlaMemberOwnerGatherKernel<kMemberGatherChunkRows, kLeader>, dim3(blocks), dim3(kThreadsPerBlock), 0,
        stream, dataPool, scalePool, memberOwners, ownerCount, output, compactIndices, globalDequantScale,
        static_cast<int32_t>(numPairs), ownerState, ownerTable, ownerTableVectors);
}

} // namespace

void invokeGlmRubinNvfp4WorklistPipelined(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool,
    int32_t const* memberIndices, int32_t numMembers, int32_t const* queryReqIndices, __nv_fp8_e4m3* output,
    int32_t* compactIndices, int64_t* ownerTable, int32_t* memberOwners, int64_t memberStride, int32_t* ownerCount,
    int32_t* ownerState, float const* globalDequantScale, int32_t numRows, int32_t topK, int32_t numRequests,
    int32_t tableSlots, int64_t numPoolTokens, cudaStream_t stream)
{
    int64_t const numPairs = static_cast<int64_t>(numRows) * topK;
    TLLM_CHECK(numRows > 0 && topK > 0 && numPairs <= std::numeric_limits<int32_t>::max() && numMembers > 0
        && numRequests > 0 && numPoolTokens > 0 && tableSlots > 1 && (tableSlots & (tableSlots - 1)) == 0
        && memberOwners != nullptr && memberStride >= numPairs);
    TLLM_CHECK(static_cast<size_t>(numRequests)
        <= std::numeric_limits<size_t>::max() / sizeof(int64_t) / static_cast<size_t>(tableSlots));
    // Keep the common power-of-two top-K path division-free.
    int32_t topKShift = -1;
    if ((topK & (topK - 1)) == 0)
    {
        topKShift = 0;
        for (int32_t value = topK; value > 1; value >>= 1)
        {
            ++topKShift;
        }
    }
    int32_t const blocks = static_cast<int32_t>((numPairs + kThreadsPerBlock - 1) / kThreadsPerBlock);
    publishOwnerMembersKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(dataPool, scalePool, memberIndices, numMembers,
        queryReqIndices, compactIndices, reinterpret_cast<unsigned long long*>(ownerTable),
        reinterpret_cast<int2*>(memberOwners), memberStride, ownerState, static_cast<int32_t>(numPairs), topK,
        topKShift, numRequests, tableSlots, numPoolTokens);
    TLLM_CUDA_CHECK(cudaGetLastError());
    int64_t const ownerTableVectors = static_cast<int64_t>(numRequests) * tableSlots
        * static_cast<int64_t>(sizeof(int64_t)) / static_cast<int64_t>(sizeof(uint4));
    launchMemberOwnerGather<true>(dataPool, scalePool, reinterpret_cast<int2 const*>(memberOwners), ownerCount, output,
        compactIndices, globalDequantScale, numPairs, ownerState, reinterpret_cast<uint4*>(ownerTable),
        ownerTableVectors, stream);
}

void invokeGlmRubinNvfp4WorklistMember(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool,
    int32_t const* memberOwners, int32_t const* ownerCount, __nv_fp8_e4m3* output, int32_t* compactIndices,
    float const* globalDequantScale, int32_t numRows, int32_t topK, cudaStream_t stream)
{
    int64_t const numPairs = static_cast<int64_t>(numRows) * topK;
    TLLM_CHECK(numRows > 0 && topK > 0 && numPairs <= std::numeric_limits<int32_t>::max());
    // The kernel only reads the count of a member call.
    launchMemberOwnerGather<false>(dataPool, scalePool, reinterpret_cast<int2 const*>(memberOwners),
        const_cast<int32_t*>(ownerCount), output, compactIndices, globalDequantScale, numPairs, nullptr, nullptr, 0,
        stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END

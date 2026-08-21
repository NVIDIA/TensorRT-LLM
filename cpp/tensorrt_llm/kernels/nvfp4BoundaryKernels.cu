/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

#include "tensorrt_llm/kernels/nvfp4BoundaryKernels.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/cudaAsyncOps.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <exception>
#include <limits>
#include <type_traits>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

// Match KVCM V2's mapped-Host copy CTA and one-split policy.
constexpr std::uint32_t kThreadsPerBlock = 128;
constexpr std::uint32_t kAsyncStages = 4;
// Mapped-Host reads use eight cp.async stages; GPU-resident input uses four.
constexpr std::uint32_t kHostLoadAsyncStages = 8;
constexpr std::uint32_t kHostMemorySplits = 1;
constexpr std::uint32_t kMaxTasksPerLaunch = 256;
// Bound by-value buffer metadata to CUDA's kernel-parameter limit.
constexpr std::uint32_t kMaxBuffersPerLaunch = kNvfp4BoundaryMaxBuffersPerLaunch;
constexpr std::uint32_t kElementsPerLane = 8;
constexpr std::uint32_t kElementsPerBlockScale = 16;
// Bound per-tile shared scale staging to 1 KiB.
constexpr std::uint32_t kTargetScaleTransferBytes = 1024;
constexpr std::uint32_t kHalfGroupsPerTransfer = 2U * kTargetScaleTransferBytes;
constexpr std::size_t kModernKernelParameterLimit = 32764;

static_assert(kThreadsPerBlock % 2 == 0, "An NVFP4 scale group is shared by two lanes");
static_assert(kTargetScaleTransferBytes > 0 && kTargetScaleTransferBytes % sizeof(uint4) == 0,
    "Compact scale transfer must remain a positive 16-byte multiple");
static_assert(kHalfGroupsPerTransfer % 2U == 0, "A transfer tile must not split an NVFP4 scale group");
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryOffloadPageTask>);
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryOnboardPageTask>);
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryBufferPlan>);
static_assert(
    kMaxBuffersPerLaunch <= std::numeric_limits<unsigned short>::max(), "Boundary buffer count must fit CUDA grid.y");
static_assert(sizeof(std::array<Nvfp4BoundaryOffloadPageTask, kMaxTasksPerLaunch>)
        + sizeof(std::array<Nvfp4BoundaryBufferPlan, kMaxBuffersPerLaunch>) + 2U * sizeof(std::uintptr_t)
        + 3U * sizeof(std::uint32_t)
    <= kModernKernelParameterLimit);
static_assert(sizeof(std::array<Nvfp4BoundaryOnboardPageTask, kMaxTasksPerLaunch>)
        + sizeof(std::array<Nvfp4BoundaryBufferPlan, kMaxBuffersPerLaunch>) + 2U * sizeof(std::uintptr_t)
        + 3U * sizeof(std::uint32_t)
    <= kModernKernelParameterLimit);

// Issue one predicated 16-byte cp.async load.
template <typename T>
__device__ __forceinline__ void copyAsyncGlobalToShared(T* shared, T const* global, bool valid)
{
    static_assert(sizeof(T) == 16, "Boundary transfer grains must match batchedCopy's 16-byte width");
    if (valid)
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "l"(__cvta_generic_to_shared(shared)), "l"(global)
                     : "memory");
    }
}

struct OffloadBufferTask
{
    std::uint8_t const* raw;
    std::uint8_t* coldData;
    std::uint8_t* coldScale;
    std::uint8_t* coldPadding;
};

struct OnboardBufferTask
{
    std::uint8_t const* coldData;
    std::uint8_t const* coldScale;
    std::uint8_t* raw;
};

__device__ OffloadBufferTask resolveTask(Nvfp4BoundaryOffloadPageTask const& page,
    Nvfp4BoundaryBufferPlan const& buffer, std::uint8_t* coldBase, std::size_t coldPageBytes)
{
    std::size_t const gpuPage = static_cast<std::size_t>(page.gpuPageIndex);
    auto* coldPage = coldBase + static_cast<std::size_t>(page.coldPageIndex) * coldPageBytes;
    return {reinterpret_cast<std::uint8_t const*>(buffer.rawBase + gpuPage * buffer.rawSlotBytes),
        coldPage + buffer.coldDataOffset, coldPage + buffer.coldScaleOffset, coldPage + buffer.coldPaddingOffset};
}

__device__ OnboardBufferTask resolveTask(Nvfp4BoundaryOnboardPageTask const& page,
    Nvfp4BoundaryBufferPlan const& buffer, std::uint8_t const* coldBase, std::size_t coldPageBytes)
{
    std::size_t const gpuPage = static_cast<std::size_t>(page.gpuPageIndex);
    auto const* coldPage = coldBase + static_cast<std::size_t>(page.coldPageIndex) * coldPageBytes;
    return {coldPage + buffer.coldDataOffset, coldPage + buffer.coldScaleOffset,
        reinterpret_cast<std::uint8_t*>(buffer.rawBase + gpuPage * buffer.rawSlotBytes)};
}

__host__ __device__ constexpr std::uint32_t packedBytesPerBuffer(std::uint32_t halfGroups)
{
    return halfGroups * sizeof(std::uint32_t);
}

__host__ __device__ constexpr std::uint32_t scaleBytesPerBuffer(std::uint32_t halfGroups)
{
    return halfGroups / 2U;
}

// Shared staging padding is not stored in the cold record.
__host__ __device__ constexpr std::uint32_t packedStagingBytesPerBuffer(std::uint32_t halfGroups)
{
    return (packedBytesPerBuffer(halfGroups) + sizeof(uint4) - 1U) / sizeof(uint4) * sizeof(uint4);
}

__host__ __device__ constexpr std::uint32_t compactStagingBytesPerBuffer(std::uint32_t halfGroups)
{
    return packedStagingBytesPerBuffer(halfGroups) + scaleBytesPerBuffer(halfGroups);
}

__host__ __device__ constexpr std::uint32_t totalHalfGroupsPerBuffer(Nvfp4BoundaryKernelParams const& params)
{
    return static_cast<std::uint32_t>(params.numKvHeads) * static_cast<std::uint32_t>(params.tokensPerPage)
        * (static_cast<std::uint32_t>(params.headDim) / kElementsPerLane);
}

// Tile at most 2,048 half-groups; headDim % 16 prevents splitting a scale group.
__host__ __device__ constexpr std::uint32_t compressedTransferHalfGroups(Nvfp4BoundaryKernelParams const& params)
{
    // Avoid std::min: its reference return ODR-uses this host/device constexpr.
    auto const halfGroups = totalHalfGroupsPerBuffer(params);
    return halfGroups < kHalfGroupsPerTransfer ? halfGroups : kHalfGroupsPerTransfer;
}

// Flush vectorized packed values and scales, then copy remaining tails bytewise.
__device__ void flushCompactRangeToHost(std::uint8_t const* compactStages, OffloadBufferTask const& task,
    std::uint32_t packedStageCapacityBytes, std::uint32_t packedDestinationOffset, std::uint32_t packedBytes,
    std::uint32_t scaleDestinationOffset, std::uint32_t scaleBytes)
{
    auto const* packedSource = compactStages;
    auto* packedDestination = task.coldData + packedDestinationOffset;
    bool const alignedPacked = reinterpret_cast<std::uintptr_t>(packedSource) % sizeof(uint4) == 0
        && reinterpret_cast<std::uintptr_t>(packedDestination) % sizeof(uint4) == 0;
    bool const alignedPackedPair = reinterpret_cast<std::uintptr_t>(packedSource) % sizeof(uint2) == 0
        && reinterpret_cast<std::uintptr_t>(packedDestination) % sizeof(uint2) == 0;
    std::uint32_t const packedVectorBytes = alignedPacked ? packedBytes - packedBytes % sizeof(uint4) : 0;
    std::uint32_t const packedPairBytes = alignedPackedPair ? packedBytes - packedBytes % sizeof(uint2) : 0;
    for (std::uint32_t grain = threadIdx.x; grain < packedVectorBytes / sizeof(uint4); grain += blockDim.x)
    {
        reinterpret_cast<uint4*>(packedDestination)[grain] = reinterpret_cast<uint4 const*>(packedSource)[grain];
    }
    for (std::uint32_t pair = packedVectorBytes / sizeof(uint2) + threadIdx.x; pair < packedPairBytes / sizeof(uint2);
         pair += blockDim.x)
    {
        reinterpret_cast<uint2*>(packedDestination)[pair] = reinterpret_cast<uint2 const*>(packedSource)[pair];
    }
    for (std::uint32_t byte = packedPairBytes + threadIdx.x; byte < packedBytes; byte += blockDim.x)
    {
        packedDestination[byte] = packedSource[byte];
    }

    auto const* scaleSource = compactStages + packedStageCapacityBytes;
    auto* scaleDestination = task.coldScale + scaleDestinationOffset;
    bool const alignedScale = reinterpret_cast<std::uintptr_t>(scaleSource) % sizeof(uint4) == 0
        && reinterpret_cast<std::uintptr_t>(scaleDestination) % sizeof(uint4) == 0;
    std::uint32_t const scaleVectorBytes = alignedScale ? scaleBytes - scaleBytes % sizeof(uint4) : 0;
    for (std::uint32_t grain = threadIdx.x; grain < scaleVectorBytes / sizeof(uint4); grain += blockDim.x)
    {
        reinterpret_cast<uint4*>(scaleDestination)[grain] = reinterpret_cast<uint4 const*>(scaleSource)[grain];
    }
    for (std::uint32_t byte = scaleVectorBytes + threadIdx.x; byte < scaleBytes; byte += blockDim.x)
    {
        scaleDestination[byte] = scaleSource[byte];
    }
}

// Zero codec-specified record padding so persisted cold Slots are deterministic.
__device__ void clearColdPadding(OffloadBufferTask const& task, Nvfp4BoundaryBufferPlan const& buffer)
{
    if (blockIdx.x != 0U)
    {
        return;
    }
    for (std::uint32_t byte = threadIdx.x; byte < buffer.coldPaddingBytes; byte += blockDim.x)
    {
        task.coldPadding[byte] = 0U;
    }
}

// CTA-uniform byte-exact copy used by lossless side buffers. Vectorize only when both
// endpoints permit it; arbitrary descriptor offsets and byte tails remain supported.
__device__ void copyLosslessBytes(std::uint8_t const* source, std::uint8_t* destination, std::size_t bytes)
{
    bool const aligned = reinterpret_cast<std::uintptr_t>(source) % sizeof(uint4) == 0
        && reinterpret_cast<std::uintptr_t>(destination) % sizeof(uint4) == 0;
    std::size_t const vectorBytes = aligned ? bytes - bytes % sizeof(uint4) : 0U;
    for (std::size_t grain = threadIdx.x; grain < vectorBytes / sizeof(uint4); grain += blockDim.x)
    {
        reinterpret_cast<uint4*>(destination)[grain] = reinterpret_cast<uint4 const*>(source)[grain];
    }
    for (std::size_t byte = vectorBytes + threadIdx.x; byte < bytes; byte += blockDim.x)
    {
        destination[byte] = source[byte];
    }
}

__device__ __forceinline__ uint4 collectTwoFp8Words(std::uint64_t first, std::uint64_t second)
{
    return make_uint4(static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(first >> 32U),
        static_cast<std::uint32_t>(second), static_cast<std::uint32_t>(second >> 32U));
}

// E2M1-to-FP16x2 PTX is adapted from arcquantFP4.cu and fusedMoeCommKernels.cu.
__device__ void unpackE2m1ToFloat(std::uint32_t packed, float2 (&values)[4])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    std::uint32_t fp16Pairs[4];
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(fp16Pairs[0]), "=r"(fp16Pairs[1]), "=r"(fp16Pairs[2]), "=r"(fp16Pairs[3])
        : "r"(packed));

#pragma unroll
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        values[i] = __half22float2(reinterpret_cast<__half2&>(fp16Pairs[i]));
    }
#endif
}

template <typename T>
__device__ void store16BitValues(T* output, std::uint32_t elementOffset, float2 const (&values)[4], float scale)
{
    // Store through uint4 to preserve STG.128; nvcc may scalarize PackedVec<T>.
    std::uint32_t outputWords[4];
#pragma unroll
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        float2 const scaled = make_float2(values[i].x * scale, values[i].y * scale);
        if constexpr (std::is_same_v<T, half>)
        {
            half2 const pair = __float22half2_rn(scaled);
            outputWords[i] = reinterpret_cast<std::uint32_t const&>(pair);
        }
        else
        {
            __nv_bfloat162 const pair = __float22bfloat162_rn(scaled);
            outputWords[i] = reinterpret_cast<std::uint32_t const&>(pair);
        }
    }
    uint4 const outputGrain = make_uint4(outputWords[0], outputWords[1], outputWords[2], outputWords[3]);
    reinterpret_cast<uint4*>(output + elementOffset)[0] = outputGrain;
}

// Preserve independent source-FP8 and destination-NVFP4 global scales.
// Restore in FP32, round pairs to FP16, then reduce each 16-value grain.
__device__ uint2 quantizeFp8GrainToNvfp4(PackedVec<__nv_fp8_e4m3> const& grain, float fp8ScaleQuantOrig,
    float nvfp4ScaleOrigQuant, std::uint8_t* scaleOutput)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Use the production packed FP8x2 conversion surface.
    PackedVec<half> restored[2];
#pragma unroll
    for (std::uint32_t pair = 0; pair < 8; ++pair)
    {
        float2 values = static_cast<float2>(grain.elts[pair]);
        values.x *= fp8ScaleQuantOrig;
        values.y *= fp8ScaleQuantOrig;
        restored[pair / 4U].elts[pair % 4U] = __float22half2_rn(values);
    }

    auto firstHalfMax = cuda_abs(restored[0].elts[0]);
    auto secondHalfMax = cuda_abs(restored[1].elts[0]);
#pragma unroll
    for (std::uint32_t i = 1; i < 4; ++i)
    {
        firstHalfMax = cuda_max(firstHalfMax, cuda_abs(restored[0].elts[i]));
        secondHalfMax = cuda_max(secondHalfMax, cuda_abs(restored[1].elts[i]));
    }
    auto const localMax = cuda_max(firstHalfMax, secondHalfMax);
    float const vecMax = static_cast<float>(cuda_max(localMax.x, localMax.y));

    float scaleValue = nvfp4ScaleOrigQuant * (vecMax * reciprocal_approximate_ftz(6.0F));
    __nv_fp8_e4m3 const roundedScale(scaleValue);
    *scaleOutput = roundedScale.__x;
    scaleValue = static_cast<float>(roundedScale);
    float const outputScale = vecMax != 0.0F
        ? reciprocal_approximate_ftz(scaleValue * reciprocal_approximate_ftz(nvfp4ScaleOrigQuant))
        : 0.0F;

    std::uint32_t packed[2];
#pragma unroll
    for (std::uint32_t halfGroup = 0; halfGroup < 2; ++halfGroup)
    {
        float2 values[4];
#pragma unroll
        for (std::uint32_t i = 0; i < 4; ++i)
        {
            values[i] = __half22float2(restored[halfGroup].elts[i]);
            values[i].x *= outputScale;
            values[i].y *= outputScale;
        }
        packed[halfGroup] = fp32_vec_to_e2m1(values);
    }
    return make_uint2(packed[0], packed[1]);
#else
    static_cast<void>(grain);
    static_cast<void>(fp8ScaleQuantOrig);
    static_cast<void>(nvfp4ScaleOrigQuant);
    static_cast<void>(scaleOutput);
    return make_uint2(0U, 0U);
#endif
}

// Restore one natural 16-value NVFP4 scale group.
template <typename T>
__device__ float onboardDequantScale(std::uint8_t encodedScale, Nvfp4BoundaryKernelParams const& params)
{
    __nv_fp8_e4m3 blockScale;
    blockScale.__x = encodedScale;
    float scale = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig;
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
    {
        scale *= params.fp8ScaleOrigQuant;
    }
    return scale;
}

template <typename T>
__device__ void restoreNvfp4Pair(uint2 packedPair, T* output, std::uint32_t firstHalfGroup, float dequantScale)
{
    std::uint32_t const packedWords[2] = {packedPair.x, packedPair.y};
    if constexpr (!std::is_same_v<T, __nv_fp8_e4m3>)
    {
#pragma unroll
        for (std::uint32_t laneInScale = 0; laneInScale < 2; ++laneInScale)
        {
            float2 values[4];
            unpackE2m1ToFloat(packedWords[laneInScale], values);
            store16BitValues(output, (firstHalfGroup + laneInScale) * kElementsPerLane, values, dequantScale);
        }
    }
    else
    {
        std::uint64_t packedFp8[2];
#pragma unroll
        for (std::uint32_t laneInScale = 0; laneInScale < 2; ++laneInScale)
        {
            float2 values[4];
            unpackE2m1ToFloat(packedWords[laneInScale], values);
#pragma unroll
            for (std::uint32_t i = 0; i < 4; ++i)
            {
                values[i].x *= dequantScale;
                values[i].y *= dequantScale;
            }
            packedFp8[laneInScale] = fp32_vec_to_e4m3(values);
        }
        reinterpret_cast<uint4*>(output)[firstHalfGroup / 2U] = collectTwoFp8Words(packedFp8[0], packedFp8[1]);
    }
}

// FP16/BF16 GPU Page -> mapped-Host NVFP4 in bounded tiles.
template <typename T>
__global__ void offloadFrom16BitTiledKernel(
    std::array<Nvfp4BoundaryOffloadPageTask, kMaxTasksPerLaunch> const __grid_constant__ pages,
    std::array<Nvfp4BoundaryBufferPlan, kMaxBuffersPerLaunch> const __grid_constant__ buffers, std::uint8_t* coldBase,
    std::size_t coldPageBytes, std::uint32_t numBuffers)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("griddepcontrol.launch_dependents;\n");

    std::uint32_t const bufferIndex = blockIdx.y;
    assert(bufferIndex < numBuffers);
    auto const& buffer = buffers[bufferIndex];
    auto const task = resolveTask(pages[blockIdx.z], buffer, coldBase, coldPageBytes);

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    if (buffer.transform == Nvfp4BoundaryTransform::kLossless)
    {
        copyLosslessBytes(task.raw, task.coldData, buffer.rawBytes);
        clearColdPadding(task, buffer);
        return;
    }

    auto const params = buffer.params;
    std::uint32_t const halfGroupsPerBuffer = totalHalfGroupsPerBuffer(params);
    std::uint32_t const tileHalfGroups = compressedTransferHalfGroups(params);
    std::uint32_t const packedStageCapacityBytes = packedStagingBytesPerBuffer(tileHalfGroups);

    // Use a four-stage 16-byte cp.async ring and tile-bounded compact staging.
    __shared__ __align__(16) PackedVec<T> rawStages[kAsyncStages][kThreadsPerBlock];
    extern __shared__ __align__(16) std::uint8_t compactStages[];
    auto* packedStages = reinterpret_cast<std::uint32_t*>(compactStages);
    auto* scaleStages = compactStages + packedStageCapacityBytes;

    for (std::uint32_t firstHalfGroup = blockIdx.x * tileHalfGroups; firstHalfGroup < halfGroupsPerBuffer;
         firstHalfGroup += gridDim.x * tileHalfGroups)
    {
        std::uint32_t const halfGroups = std::min(tileHalfGroups, halfGroupsPerBuffer - firstHalfGroup);
        std::uint32_t const iterations = (halfGroups + kThreadsPerBlock - 1U) / kThreadsPerBlock;

        for (std::uint32_t iteration = 0; iteration < iterations + kAsyncStages; ++iteration)
        {
            std::uint32_t const stage = iteration % kAsyncStages;
            if (iteration >= kAsyncStages)
            {
                std::uint32_t const transformIteration = iteration - kAsyncStages;
                std::uint32_t const localHalfGroup = kThreadsPerBlock * transformIteration + threadIdx.x;
                cp_async_wait_group<kAsyncStages - 1>();
                if (localHalfGroup < halfGroups)
                {
                    std::uint32_t const globalHalfGroup = firstHalfGroup + localHalfGroup;
                    std::uint32_t const laneInScale = globalHalfGroup & 1U;

                    // Even tile boundaries preserve the 16-value scale groups.
                    std::uint32_t const localScaleOffset = localHalfGroup >> 1U;
                    std::uint8_t* scale = laneInScale == 0 ? scaleStages + localScaleOffset : nullptr;
                    PackedVec<T> input = rawStages[stage][threadIdx.x];
                    packedStages[localHalfGroup] = cvt_warp_fp16_to_fp4<T, kElementsPerBlockScale, false>(
                        input, params.nvfp4ScaleOrigQuant, scale);
                }
            }

            std::uint32_t const localLoadHalfGroup = kThreadsPerBlock * iteration + threadIdx.x;
            bool const valid = localLoadHalfGroup < halfGroups;
            auto const* rawInput = reinterpret_cast<PackedVec<T> const*>(task.raw);
            auto const* source = valid ? rawInput + firstHalfGroup + localLoadHalfGroup : rawInput;
            copyAsyncGlobalToShared(&rawStages[stage][threadIdx.x], source, valid);
            cp_async_commit_group();
        }

        // Publish quant results and finish the flush before reusing staging.
        cp_async_wait_group<0>();
        __syncthreads();
        flushCompactRangeToHost(compactStages, task, packedStageCapacityBytes, packedBytesPerBuffer(firstHalfGroup),
            packedBytesPerBuffer(halfGroups), firstHalfGroup / 2U, halfGroups / 2U);
        __syncthreads();
    }
    clearColdPadding(task, buffer);
#endif
}

// FP8 E4M3 GPU Page -> mapped-Host NVFP4 in bounded tiles.
__global__ void offloadFromFp8TiledKernel(
    std::array<Nvfp4BoundaryOffloadPageTask, kMaxTasksPerLaunch> const __grid_constant__ pages,
    std::array<Nvfp4BoundaryBufferPlan, kMaxBuffersPerLaunch> const __grid_constant__ buffers, std::uint8_t* coldBase,
    std::size_t coldPageBytes, std::uint32_t numBuffers)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("griddepcontrol.launch_dependents;\n");

    std::uint32_t const bufferIndex = blockIdx.y;
    assert(bufferIndex < numBuffers);
    auto const& buffer = buffers[bufferIndex];
    auto const task = resolveTask(pages[blockIdx.z], buffer, coldBase, coldPageBytes);

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    if (buffer.transform == Nvfp4BoundaryTransform::kLossless)
    {
        copyLosslessBytes(task.raw, task.coldData, buffer.rawBytes);
        clearColdPadding(task, buffer);
        return;
    }

    auto const params = buffer.params;
    std::uint32_t const halfGroupsPerBuffer = totalHalfGroupsPerBuffer(params);
    std::uint32_t const tileHalfGroups = compressedTransferHalfGroups(params);
    std::uint32_t const packedStageCapacityBytes = packedStagingBytesPerBuffer(tileHalfGroups);

    // Each cp.async moves one 16-byte grain in the production PackedVec layout.
    __shared__ __align__(16) PackedVec<__nv_fp8_e4m3> rawStages[kAsyncStages][kThreadsPerBlock];
    extern __shared__ __align__(16) std::uint8_t compactStages[];
    auto* packedStages = reinterpret_cast<std::uint32_t*>(compactStages);
    auto* scaleStages = compactStages + packedStageCapacityBytes;

    for (std::uint32_t firstHalfGroup = blockIdx.x * tileHalfGroups; firstHalfGroup < halfGroupsPerBuffer;
         firstHalfGroup += gridDim.x * tileHalfGroups)
    {
        std::uint32_t const halfGroups = std::min(tileHalfGroups, halfGroupsPerBuffer - firstHalfGroup);
        std::uint32_t const firstGrain = firstHalfGroup / 2U;
        std::uint32_t const grains = halfGroups / 2U;
        std::uint32_t const iterations = (grains + kThreadsPerBlock - 1U) / kThreadsPerBlock;

        for (std::uint32_t iteration = 0; iteration < iterations + kAsyncStages; ++iteration)
        {
            std::uint32_t const stage = iteration % kAsyncStages;
            if (iteration >= kAsyncStages)
            {
                std::uint32_t const transformIteration = iteration - kAsyncStages;
                cp_async_wait_group<kAsyncStages - 1>();
                // One lane owns the complete 16-value scale group.
                std::uint32_t const localGrain = kThreadsPerBlock * transformIteration + threadIdx.x;
                if (localGrain < grains)
                {
                    uint2 const packed = quantizeFp8GrainToNvfp4(rawStages[stage][threadIdx.x],
                        params.fp8ScaleQuantOrig, params.nvfp4ScaleOrigQuant, scaleStages + localGrain);
                    reinterpret_cast<uint2*>(packedStages)[localGrain] = packed;
                }
            }

            std::uint32_t const localLoadGrain = kThreadsPerBlock * iteration + threadIdx.x;
            bool const valid = localLoadGrain < grains;
            auto const* rawInput = reinterpret_cast<PackedVec<__nv_fp8_e4m3> const*>(task.raw);
            auto const* source = valid ? rawInput + firstGrain + localLoadGrain : rawInput;
            copyAsyncGlobalToShared(&rawStages[stage][threadIdx.x], source, valid);
            cp_async_commit_group();
        }

        cp_async_wait_group<0>();
        __syncthreads();
        flushCompactRangeToHost(compactStages, task, packedStageCapacityBytes, packedBytesPerBuffer(firstHalfGroup),
            packedBytesPerBuffer(halfGroups), firstHalfGroup / 2U, halfGroups / 2U);
        __syncthreads();
    }
    clearColdPadding(task, buffer);
#endif
}

// Load packed values and one-byte scales from mapped Host memory into shared.
// Use uint4/uint2 when aligned and byte tails otherwise.
__device__ void loadCompactRangeFromHost(std::uint8_t* compactStages, OnboardBufferTask const& task,
    std::uint32_t packedStageCapacityBytes, std::uint32_t packedSourceOffset, std::uint32_t packedBytes,
    std::uint32_t scaleSourceOffset, std::uint32_t scaleBytes)
{
    auto const* packedSource = task.coldData + packedSourceOffset;
    auto* packedDestination = compactStages;
    auto const* scaleSource = task.coldScale + scaleSourceOffset;
    auto* scaleDestination = compactStages + packedStageCapacityBytes;
    bool const alignedPacked = reinterpret_cast<std::uintptr_t>(packedSource) % sizeof(uint4) == 0
        && reinterpret_cast<std::uintptr_t>(packedDestination) % sizeof(uint4) == 0;
    bool const alignedPackedPair = reinterpret_cast<std::uintptr_t>(packedSource) % sizeof(uint2) == 0
        && reinterpret_cast<std::uintptr_t>(packedDestination) % sizeof(uint2) == 0;
    std::uint32_t const packedVectorBytes = alignedPacked ? packedBytes - packedBytes % sizeof(uint4) : 0;
    std::uint32_t const packedPairBytes = alignedPackedPair ? packedBytes - packedBytes % sizeof(uint2) : 0;
    bool const alignedScale = reinterpret_cast<std::uintptr_t>(scaleSource) % sizeof(uint4) == 0
        && reinterpret_cast<std::uintptr_t>(scaleDestination) % sizeof(uint4) == 0;
    std::uint32_t const scaleVectorBytes = alignedScale ? scaleBytes - scaleBytes % sizeof(uint4) : 0;
    std::uint32_t const packedGrains = packedVectorBytes / sizeof(uint4);
    std::uint32_t const scaleGrains = scaleVectorBytes / sizeof(uint4);
    std::uint32_t const totalGrains = packedGrains + scaleGrains;
    std::uint32_t const iterations = (totalGrains + blockDim.x - 1U) / blockDim.x;

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        if (iteration >= kHostLoadAsyncStages)
        {
            cp_async_wait_group<kHostLoadAsyncStages - 1>();
        }
        std::uint32_t const grain = blockDim.x * iteration + threadIdx.x;
        bool const valid = grain < totalGrains;
        auto const* source = reinterpret_cast<uint4 const*>(packedSource);
        auto* destination = reinterpret_cast<uint4*>(packedDestination);
        if (valid && grain < packedGrains)
        {
            source += grain;
            destination += grain;
        }
        else if (valid)
        {
            source = reinterpret_cast<uint4 const*>(scaleSource) + grain - packedGrains;
            destination = reinterpret_cast<uint4*>(scaleDestination) + grain - packedGrains;
        }
        copyAsyncGlobalToShared(destination, source, valid);
        cp_async_commit_group();
    }
    cp_async_wait_group<0>();

    // headDim % 16 makes packed intervals exact uint2 scale groups.
    for (std::uint32_t pair = packedVectorBytes / sizeof(uint2) + threadIdx.x; pair < packedPairBytes / sizeof(uint2);
         pair += blockDim.x)
    {
        reinterpret_cast<uint2*>(packedDestination)[pair] = reinterpret_cast<uint2 const*>(packedSource)[pair];
    }
    for (std::uint32_t byte = packedPairBytes + threadIdx.x; byte < packedBytes; byte += blockDim.x)
    {
        packedDestination[byte] = packedSource[byte];
    }
    for (std::uint32_t byte = scaleVectorBytes + threadIdx.x; byte < scaleBytes; byte += blockDim.x)
    {
        scaleDestination[byte] = scaleSource[byte];
    }
    __syncthreads();
}

// Mapped-Host NVFP4 -> runtime GPU Page in bounded tiles.
template <typename T>
__global__ void onboardTiledKernel(
    std::array<Nvfp4BoundaryOnboardPageTask, kMaxTasksPerLaunch> const __grid_constant__ pages,
    std::array<Nvfp4BoundaryBufferPlan, kMaxBuffersPerLaunch> const __grid_constant__ buffers,
    std::uint8_t const* coldBase, std::size_t coldPageBytes, std::uint32_t numBuffers)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("griddepcontrol.launch_dependents;\n");

    std::uint32_t const bufferIndex = blockIdx.y;
    assert(bufferIndex < numBuffers);
    auto const& buffer = buffers[bufferIndex];
    auto const task = resolveTask(pages[blockIdx.z], buffer, coldBase, coldPageBytes);

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    if (buffer.transform == Nvfp4BoundaryTransform::kLossless)
    {
        copyLosslessBytes(task.coldData, task.raw, buffer.rawBytes);
        return;
    }

    auto const params = buffer.params;
    std::uint32_t const halfGroupsPerBuffer = totalHalfGroupsPerBuffer(params);
    std::uint32_t const tileHalfGroups = compressedTransferHalfGroups(params);
    std::uint32_t const packedStageCapacityBytes = packedStagingBytesPerBuffer(tileHalfGroups);
    extern __shared__ __align__(16) std::uint8_t compactStages[];
    auto* rawOutput = reinterpret_cast<T*>(task.raw);

    for (std::uint32_t firstHalfGroup = blockIdx.x * tileHalfGroups; firstHalfGroup < halfGroupsPerBuffer;
         firstHalfGroup += gridDim.x * tileHalfGroups)
    {
        std::uint32_t const halfGroups = std::min(tileHalfGroups, halfGroupsPerBuffer - firstHalfGroup);
        std::uint32_t const packedBytes = packedBytesPerBuffer(halfGroups);
        std::uint32_t const scaleBytes = halfGroups / 2U;

        // Stage packed data and scales before dequantization.
        loadCompactRangeFromHost(compactStages, task, packedStageCapacityBytes, packedBytesPerBuffer(firstHalfGroup),
            packedBytes, firstHalfGroup / 2U, scaleBytes);

        auto const* packedStages = reinterpret_cast<uint4 const*>(compactStages);
        auto const* scaleStages = compactStages + packedStageCapacityBytes;
        std::uint32_t const packedGrains = packedBytes / sizeof(uint4);
        for (std::uint32_t localGrain = threadIdx.x; localGrain < packedGrains; localGrain += blockDim.x)
        {
            uint4 const packedGrain = packedStages[localGrain];
            std::uint32_t const packedWords[4] = {packedGrain.x, packedGrain.y, packedGrain.z, packedGrain.w};
#pragma unroll
            for (std::uint32_t pair = 0; pair < 2; ++pair)
            {
                std::uint32_t const localScaleGroup = localGrain * 2U + pair;
                std::uint32_t const firstPairHalfGroup = firstHalfGroup + localScaleGroup * 2U;

                restoreNvfp4Pair(make_uint2(packedWords[pair * 2U], packedWords[pair * 2U + 1U]), rawOutput,
                    firstPairHalfGroup, onboardDequantScale<T>(scaleStages[localScaleGroup], params));
            }
        }
        if (packedBytes % sizeof(uint4) != 0U && threadIdx.x == 0)
        {
            std::uint32_t const localScaleGroup = packedGrains * 2U;
            std::uint32_t const firstPairHalfGroup = firstHalfGroup + localScaleGroup * 2U;
            restoreNvfp4Pair(reinterpret_cast<uint2 const*>(compactStages)[localScaleGroup], rawOutput,
                firstPairHalfGroup, onboardDequantScale<T>(scaleStages[localScaleGroup], params));
        }

        // Finish consumers before reusing shared memory.
        __syncthreads();
    }
#endif
}

void validateParams(Nvfp4BoundaryKernelParams const& params, bool useFp8)
{
    TLLM_CHECK_WITH_INFO(params.numKvHeads > 0, "numKvHeads must be positive");
    TLLM_CHECK_WITH_INFO(params.tokensPerPage > 0, "tokensPerPage must be positive");
    TLLM_CHECK_WITH_INFO(params.headDim > 0 && params.headDim % kElementsPerBlockScale == 0,
        "headDim must be positive and divisible by 16, got %d", params.headDim);

    std::uint64_t const rows
        = static_cast<std::uint64_t>(params.numKvHeads) * static_cast<std::uint64_t>(params.tokensPerPage);
    constexpr std::uint64_t maxHalfGroups = std::numeric_limits<std::uint32_t>::max() / kElementsPerLane;
    std::uint64_t const halfGroupsPerRow = static_cast<std::uint64_t>(params.headDim / kElementsPerLane);
    TLLM_CHECK_WITH_INFO(rows <= maxHalfGroups / halfGroupsPerRow,
        "Page geometry exceeds the 32-bit compact-offset range: "
        "heads=%d, tokens=%d, headDim=%d",
        params.numKvHeads, params.tokensPerPage, params.headDim);

    TLLM_CHECK_WITH_INFO(std::isfinite(params.nvfp4ScaleOrigQuant) && params.nvfp4ScaleOrigQuant > 0.0F,
        "NVFP4 original-to-quantized scale must be finite and positive");
    TLLM_CHECK_WITH_INFO(std::isfinite(params.nvfp4ScaleQuantOrig) && params.nvfp4ScaleQuantOrig > 0.0F,
        "NVFP4 quantized-to-original scale must be finite and positive");
    if (useFp8)
    {
        TLLM_CHECK_WITH_INFO(std::isfinite(params.fp8ScaleOrigQuant) && params.fp8ScaleOrigQuant > 0.0F,
            "FP8 original-to-quantized scale must be finite and positive");
        TLLM_CHECK_WITH_INFO(std::isfinite(params.fp8ScaleQuantOrig) && params.fp8ScaleQuantOrig > 0.0F,
            "FP8 quantized-to-original scale must be finite and positive");
    }
}

struct ColdInterval
{
    std::size_t begin;
    std::size_t end;
};

void addColdInterval(std::vector<ColdInterval>& intervals, std::size_t offset, std::size_t bytes,
    std::size_t coldPageBytes, char const* label)
{
    if (bytes == 0U)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(
        offset <= coldPageBytes && bytes <= coldPageBytes - offset, "%s exceeds the cold Page stride", label);
    intervals.push_back({offset, offset + bytes});
}

void validateBufferPlan(
    Nvfp4BoundaryBufferPlan const& buffer, std::size_t coldPageBytes, bool useFp8, std::vector<ColdInterval>& intervals)
{
    TLLM_CHECK_WITH_INFO(buffer.rawBase != 0U, "rawBase must not be null");
    TLLM_CHECK_WITH_INFO(buffer.rawBytes > 0 && buffer.rawBytes <= buffer.rawSlotBytes,
        "Raw buffer bytes must be positive and fit within the GPU Slot stride");

    switch (buffer.transform)
    {
    case Nvfp4BoundaryTransform::kNvfp4:
    {
        validateParams(buffer.params, useFp8);
        TLLM_CHECK_WITH_INFO(
            buffer.rawBase % alignof(uint4) == 0, "rawBase must be aligned to %zu bytes", alignof(uint4));
        TLLM_CHECK_WITH_INFO(buffer.rawSlotBytes % alignof(uint4) == 0,
            "GPU raw Slot stride must be aligned to %zu bytes for NVFP4", alignof(uint4));
        std::uint64_t const elements = static_cast<std::uint64_t>(buffer.params.numKvHeads)
            * static_cast<std::uint64_t>(buffer.params.tokensPerPage)
            * static_cast<std::uint64_t>(buffer.params.headDim);
        std::uint64_t const expectedRawBytes = elements * (useFp8 ? 1U : 2U);
        TLLM_CHECK_WITH_INFO(buffer.rawBytes == static_cast<std::size_t>(expectedRawBytes),
            "Raw buffer size does not match NVFP4 geometry and runtime type");
        addColdInterval(intervals, buffer.coldDataOffset, static_cast<std::size_t>(elements / 2U), coldPageBytes,
            "NVFP4 packed-data interval");
        addColdInterval(intervals, buffer.coldScaleOffset, static_cast<std::size_t>(elements / 16U), coldPageBytes,
            "NVFP4 scale interval");
        break;
    }
    case Nvfp4BoundaryTransform::kLossless:
        addColdInterval(intervals, buffer.coldDataOffset, buffer.rawBytes, coldPageBytes, "Lossless-data interval");
        break;
    default: TLLM_THROW("Unsupported NVFP4 boundary buffer transform");
    }
    addColdInterval(
        intervals, buffer.coldPaddingOffset, buffer.coldPaddingBytes, coldPageBytes, "Cold-record padding interval");
}

// Launch the 256-descriptor raw-argument ABI with optional PDL.
// Only a partial chunk needs a zero-padded stack argument.
template <typename Task, typename Kernel, typename ColdPointer>
void launchBoundaryBatch(Kernel kernel, Task const* tasks, std::uint32_t count, dim3 grid, dim3 block,
    std::uint32_t dynamicSmemBytes, std::array<Nvfp4BoundaryBufferPlan, kMaxBuffersPerLaunch> const& buffers,
    ColdPointer coldBase, std::size_t coldPageBytes, std::uint32_t numBuffers, cudaStream_t stream)
{
    static_assert(std::is_trivially_copyable_v<std::array<Task, kMaxTasksPerLaunch>>);
    static_assert(sizeof(std::array<Task, kMaxTasksPerLaunch>) == sizeof(Task) * kMaxTasksPerLaunch);
    TLLM_CHECK(count > 0 && count <= kMaxTasksPerLaunch);

    // Zero initialization keeps partial argument padding deterministic.
    std::array<Task, kMaxTasksPerLaunch> tail{};
    void const* taskArgument = tasks;
    if (count < kMaxTasksPerLaunch)
    {
        std::copy_n(tasks, count, tail.begin());
        taskArgument = tail.data();
    }

    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = common::getEnvEnablePDL() ? 1 : 0;

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = dynamicSmemBytes;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;

    void* arguments[] = {const_cast<void*>(taskArgument), const_cast<Nvfp4BoundaryBufferPlan*>(buffers.data()),
        &coldBase, &coldPageBytes, &numBuffers};
    TLLM_CUDA_CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<void const*>(kernel), arguments));
}

// Submit Page descriptors in fixed-capacity chunks.
template <typename Task, typename LaunchBatch>
void launchTaskBatches(std::vector<Task> const& tasks, LaunchBatch const& launchBatch)
{
    std::size_t offset = 0;
    while (offset < tasks.size())
    {
        std::uint32_t const count
            = static_cast<std::uint32_t>(std::min<std::size_t>(tasks.size() - offset, kMaxTasksPerLaunch));
        launchBatch(tasks.data() + offset, count);
        offset += count;
    }
}

// One tiled SM100 kernel family covers every runtime dtype, transform, and Page geometry.
// grid.y selects one independent buffer; geometry only changes the tile loop length.

template <typename T>
void launchOffloadFrom16Bit(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, std::uint8_t* coldBase, cudaStream_t stream)
{
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(pages,
        [&](Nvfp4BoundaryOffloadPageTask const* taskData, std::uint32_t count)
        {
            dim3 const grid(kHostMemorySplits, plan.numBuffers, count);
            launchBoundaryBatch(offloadFrom16BitTiledKernel<T>, taskData, count, grid, block,
                compactStagingBytesPerBuffer(plan.maxTileHalfGroups), plan.buffers, coldBase, plan.coldPageBytes,
                plan.numBuffers, stream);
        });
}

void launchOffloadFromFp8(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages, Nvfp4BoundaryPreparedPlan const& plan,
    std::uint8_t* coldBase, cudaStream_t stream)
{
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(pages,
        [&](Nvfp4BoundaryOffloadPageTask const* taskData, std::uint32_t count)
        {
            dim3 const grid(kHostMemorySplits, plan.numBuffers, count);
            launchBoundaryBatch(offloadFromFp8TiledKernel, taskData, count, grid, block,
                compactStagingBytesPerBuffer(plan.maxTileHalfGroups), plan.buffers, coldBase, plan.coldPageBytes,
                plan.numBuffers, stream);
        });
}

template <typename T>
void launchOnboard(std::vector<Nvfp4BoundaryOnboardPageTask> const& pages, Nvfp4BoundaryPreparedPlan const& plan,
    std::uint8_t const* coldBase, cudaStream_t stream)
{
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(pages,
        [&](Nvfp4BoundaryOnboardPageTask const* taskData, std::uint32_t count)
        {
            dim3 const grid(kHostMemorySplits, plan.numBuffers, count);
            launchBoundaryBatch(onboardTiledKernel<T>, taskData, count, grid, block,
                compactStagingBytesPerBuffer(plan.maxTileHalfGroups), plan.buffers, coldBase, plan.coldPageBytes,
                plan.numBuffers, stream);
        });
}

// Drain earlier chunks after synchronous launch failure before Slots are recycled.
template <typename Launch>
void launchAndDrainOnFailure(cudaStream_t stream, Launch const& launch)
{
    try
    {
        launch();
    }
    catch (...)
    {
        cudaError_t const drainStatus = cudaStreamSynchronize(stream);
        if (drainStatus != cudaSuccess)
        {
            // An asynchronous drain failure leaves Slot ownership unknown; fail-stop.
            TLLM_LOG_ERROR("NVFP4 boundary rollback drain failed: %s", cudaGetErrorString(drainStatus));
            std::terminate();
        }
        throw;
    }
}

} // namespace

Nvfp4BoundaryPreparedPlan prepareNvfp4BoundaryPlan(std::vector<Nvfp4BoundaryBufferPlan> const& buffers,
    std::size_t coldPageBytes, Nvfp4BoundaryRuntimeType runtimeType)
{
    // TODO: Make this codec-private, then remove caller-guaranteed admission checks.
    TLLM_CHECK_WITH_INFO(common::isSM100Family(), "NVFP4 boundary kernels require an SM100-family GPU");
    TLLM_CHECK_WITH_INFO(!buffers.empty(), "NVFP4 boundary launch requires at least one buffer");
    TLLM_CHECK_WITH_INFO(buffers.size() <= kMaxBuffersPerLaunch,
        "NVFP4 boundary launch supports at most %u local buffers, got %zu", kMaxBuffersPerLaunch, buffers.size());
    TLLM_CHECK_WITH_INFO(coldPageBytes > 0, "Cold Page stride must be positive");
    TLLM_CHECK_WITH_INFO(
        coldPageBytes % alignof(uint4) == 0, "Cold Page stride must be aligned to %zu bytes", alignof(uint4));

    bool useFp8 = false;
    switch (runtimeType)
    {
    case Nvfp4BoundaryRuntimeType::kFloat16:
    case Nvfp4BoundaryRuntimeType::kBfloat16: break;
    case Nvfp4BoundaryRuntimeType::kFp8E4m3: useFp8 = true; break;
    default: TLLM_THROW("Unsupported NVFP4 boundary runtime type");
    }

    Nvfp4BoundaryPreparedPlan plan;
    plan.numBuffers = static_cast<std::uint32_t>(buffers.size());
    plan.coldPageBytes = coldPageBytes;
    plan.runtimeType = runtimeType;
    std::copy(buffers.begin(), buffers.end(), plan.buffers.begin());
    std::vector<ColdInterval> intervals;
    intervals.reserve(3U * buffers.size());
    for (auto const& buffer : buffers)
    {
        validateBufferPlan(buffer, coldPageBytes, useFp8, intervals);
        if (buffer.transform == Nvfp4BoundaryTransform::kNvfp4)
        {
            plan.maxTileHalfGroups = std::max(plan.maxTileHalfGroups, compressedTransferHalfGroups(buffer.params));
        }
    }
    std::sort(intervals.begin(), intervals.end(),
        [](ColdInterval const& lhs, ColdInterval const& rhs)
        { return lhs.begin < rhs.begin || (lhs.begin == rhs.begin && lhs.end < rhs.end); });
    for (std::size_t index = 1; index < intervals.size(); ++index)
    {
        TLLM_CHECK_WITH_INFO(
            intervals[index - 1U].end <= intervals[index].begin, "Cold record intervals must not overlap");
    }
    return plan;
}

void invokeNvfp4BoundaryOffloadCompress(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void* coldBase, cudaStream_t stream)
{
    if (pages.empty())
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(coldBase != nullptr, "coldBase must not be null");
    switch (plan.runtimeType)
    {
    case Nvfp4BoundaryRuntimeType::kFloat16:
        launchAndDrainOnFailure(
            stream, [&] { launchOffloadFrom16Bit<half>(pages, plan, static_cast<std::uint8_t*>(coldBase), stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kBfloat16:
        launchAndDrainOnFailure(stream,
            [&] { launchOffloadFrom16Bit<__nv_bfloat16>(pages, plan, static_cast<std::uint8_t*>(coldBase), stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kFp8E4m3:
        launchAndDrainOnFailure(
            stream, [&] { launchOffloadFromFp8(pages, plan, static_cast<std::uint8_t*>(coldBase), stream); });
        break;
    default: TLLM_THROW("Unsupported NVFP4 boundary runtime type");
    }
}

void invokeNvfp4BoundaryOnboardDecompress(std::vector<Nvfp4BoundaryOnboardPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void const* coldBase, cudaStream_t stream)
{
    if (pages.empty())
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(coldBase != nullptr, "coldBase must not be null");
    switch (plan.runtimeType)
    {
    case Nvfp4BoundaryRuntimeType::kFloat16:
        launchAndDrainOnFailure(
            stream, [&] { launchOnboard<half>(pages, plan, static_cast<std::uint8_t const*>(coldBase), stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kBfloat16:
        launchAndDrainOnFailure(stream,
            [&] { launchOnboard<__nv_bfloat16>(pages, plan, static_cast<std::uint8_t const*>(coldBase), stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kFp8E4m3:
        launchAndDrainOnFailure(stream,
            [&] { launchOnboard<__nv_fp8_e4m3>(pages, plan, static_cast<std::uint8_t const*>(coldBase), stream); });
        break;
    default: TLLM_THROW("Unsupported NVFP4 boundary runtime type");
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END

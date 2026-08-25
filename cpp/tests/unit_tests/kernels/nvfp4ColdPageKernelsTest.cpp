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

#include "tensorrt_llm/kernels/nvfp4ColdPageKernels.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{

using tensorrt_llm::batch_manager::kv_cache_manager_v2::HostMem;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::MemAddress;
using tensorrt_llm::kernels::Nvfp4ColdPageBufferPlan;
using tensorrt_llm::kernels::Nvfp4ColdPageKernelParams;
using tensorrt_llm::kernels::Nvfp4ColdPageOffloadPageTask;
using tensorrt_llm::kernels::Nvfp4ColdPageOnboardPageTask;
using tensorrt_llm::kernels::Nvfp4ColdPagePreparedPlan;
using tensorrt_llm::kernels::Nvfp4ColdPageRuntimeType;
using tensorrt_llm::kernels::Nvfp4ColdPageTransform;

constexpr std::size_t kGuardBytes = 64;
constexpr std::uint8_t kCanary = 0xA5;
constexpr std::size_t kDefaultNumPages = 3;
constexpr std::size_t kCrossLaunchNumPages = 257;

struct PageGeometry
{
    std::int32_t numHeads;
    std::int32_t tokensPerPage;
    std::int32_t headDim;
};

constexpr PageGeometry kDefaultGeometry{2, 8, 32};
constexpr PageGeometry kMinimumCompactGeometry{1, 1, 16};
constexpr PageGeometry kPackedBodyAndTailGeometry{1, 3, 16};
constexpr PageGeometry kSmallVectorGeometry{1, 4, 16};
constexpr PageGeometry kLinearScaleTailGeometry{1, 5, 32};
constexpr PageGeometry kTiledLinearScaleTailGeometry{1, 4097, 16};
constexpr PageGeometry kCrossRowTileGeometry{1, 343, 48};
constexpr PageGeometry kLargeHeadDimTailGeometry{1, 1, 65552};
constexpr PageGeometry kModelLikeGeometry{8, 64, 128};
constexpr std::array<std::int32_t, 5> kValidTokenCounts{1, 16, 17, 63, 64};

enum class RawKind
{
    kFloat16,
    kBfloat16,
    kFp8,
};

enum class InputPattern
{
    kDense,
    kAllZero,
    kSparseOutlier,
    kRoundingMargins,
};

std::size_t roundUp(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

class CudaStream
{
public:
    CudaStream()
    {
        TLLM_CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }

    ~CudaStream()
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    operator cudaStream_t() const
    {
        return mStream;
    }

    CudaStream(CudaStream const&) = delete;
    CudaStream& operator=(CudaStream const&) = delete;

private:
    cudaStream_t mStream{};
};

//! Device allocation guarded by canaries to catch descriptor or vector-tail out-of-bounds writes.
class DeviceRegion
{
public:
    explicit DeviceRegion(std::size_t payloadBytes)
        : mPayloadBytes(payloadBytes)
        , mTotalBytes(payloadBytes + 2 * kGuardBytes)
    {
        TLLM_CUDA_CHECK(cudaMalloc(&mBase, mTotalBytes));
        TLLM_CUDA_CHECK(cudaMemset(mBase, kCanary, mTotalBytes));
    }

    ~DeviceRegion()
    {
        if (mBase != nullptr)
        {
            cudaFree(mBase);
        }
    }

    DeviceRegion(DeviceRegion const&) = delete;
    DeviceRegion& operator=(DeviceRegion const&) = delete;

    void* data() const
    {
        return static_cast<std::uint8_t*>(mBase) + kGuardBytes;
    }

    void copyFrom(std::vector<std::uint8_t> const& bytes)
    {
        ASSERT_EQ(bytes.size(), mPayloadBytes);
        ASSERT_EQ(cudaMemcpy(data(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice), cudaSuccess);
    }

    void copyFrom(std::size_t offset, std::vector<std::uint8_t> const& bytes)
    {
        ASSERT_LE(offset + bytes.size(), mPayloadBytes);
        ASSERT_EQ(
            cudaMemcpy(static_cast<std::uint8_t*>(data()) + offset, bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
    }

    std::vector<std::uint8_t> copyToHost() const
    {
        std::vector<std::uint8_t> bytes(mPayloadBytes);
        EXPECT_EQ(cudaMemcpy(bytes.data(), data(), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
        return bytes;
    }

    std::vector<std::uint8_t> copyToHost(std::size_t offset, std::size_t bytes) const
    {
        EXPECT_LE(offset + bytes, mPayloadBytes);
        std::vector<std::uint8_t> result(bytes);
        EXPECT_EQ(
            cudaMemcpy(result.data(), static_cast<std::uint8_t const*>(data()) + offset, bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        return result;
    }

    void expectCanaries() const
    {
        std::vector<std::uint8_t> bytes(mTotalBytes);
        ASSERT_EQ(cudaMemcpy(bytes.data(), mBase, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_TRUE(std::all_of(
            bytes.begin(), bytes.begin() + kGuardBytes, [](std::uint8_t value) { return value == kCanary; }));
        EXPECT_TRUE(
            std::all_of(bytes.end() - kGuardBytes, bytes.end(), [](std::uint8_t value) { return value == kCanary; }));
    }

private:
    void* mBase{};
    std::size_t mPayloadBytes{};
    std::size_t mTotalBytes{};
};

//! CUDA-mapped HostMem matching KVCM V2's Host carrier.
class MappedHostRegion
{
public:
    explicit MappedHostRegion(std::size_t payloadBytes)
        : mMemory(roundUp(kGuardBytes + payloadBytes + kGuardBytes, HostMem::kAlignment))
        , mPayloadBytes(payloadBytes)
    {
        TLLM_CHECK_WITH_INFO(kGuardBytes + payloadBytes + kGuardBytes <= mMemory.size(),
            "Mapped Host test allocation is too small for payload and canaries");
        std::memset(reinterpret_cast<void*>(mMemory.address()), kCanary, mMemory.size());
    }

    void* data() const
    {
        return reinterpret_cast<void*>(mMemory.address() + kGuardBytes);
    }

    std::uint8_t* bytes() const
    {
        return static_cast<std::uint8_t*>(data());
    }

    std::vector<std::uint8_t> payload() const
    {
        return {bytes(), bytes() + mPayloadBytes};
    }

    void expectCanaries() const
    {
        auto const* base = reinterpret_cast<std::uint8_t const*>(mMemory.address());
        EXPECT_TRUE(std::all_of(base, base + kGuardBytes, [](std::uint8_t value) { return value == kCanary; }));
        EXPECT_TRUE(std::all_of(base + kGuardBytes + mPayloadBytes, base + mMemory.size(),
            [](std::uint8_t value) { return value == kCanary; }));
    }

private:
    HostMem mMemory;
    std::size_t mPayloadBytes{};
};

struct LayerBuffers
{
    explicit LayerBuffers(std::size_t rawBytes)
        : rawK(rawBytes)
        , rawV(rawBytes)
    {
    }

    DeviceRegion rawK;
    DeviceRegion rawV;
};

std::size_t numElements(PageGeometry const& geometry)
{
    return static_cast<std::size_t>(geometry.numHeads) * geometry.tokensPerPage * geometry.headDim;
}

std::size_t rawBytes(RawKind kind, PageGeometry const& geometry)
{
    return numElements(geometry) * (kind == RawKind::kFp8 ? 1 : 2);
}

std::size_t rawElementBytes(RawKind kind)
{
    return kind == RawKind::kFp8 ? 1U : 2U;
}

std::size_t packedBytes(PageGeometry const& geometry)
{
    return numElements(geometry) / 2;
}

std::size_t scaleBytes(PageGeometry const& geometry)
{
    return numElements(geometry) / 16;
}

Nvfp4ColdPageKernelParams makeParams(PageGeometry const& geometry = kDefaultGeometry, std::uint32_t role = 0U)
{
    Nvfp4ColdPageKernelParams params{};
    params.numKvHeads = geometry.numHeads;
    params.tokensPerPage = geometry.tokensPerPage;
    params.headDim = geometry.headDim;
    params.nvfp4ScaleOrigQuant = role == 0U ? 1.0F : 2.0F;
    params.nvfp4ScaleQuantOrig = role == 0U ? 1.0F : 0.5F;
    params.fp8ScaleOrigQuant = role == 0U ? 2.0F : 4.0F;
    params.fp8ScaleQuantOrig = role == 0U ? 0.5F : 0.25F;
    return params;
}

Nvfp4ColdPageRuntimeType runtimeType(RawKind kind)
{
    switch (kind)
    {
    case RawKind::kFloat16: return Nvfp4ColdPageRuntimeType::kFloat16;
    case RawKind::kBfloat16: return Nvfp4ColdPageRuntimeType::kBfloat16;
    case RawKind::kFp8: return Nvfp4ColdPageRuntimeType::kFp8E4m3;
    }
    return Nvfp4ColdPageRuntimeType::kFloat16;
}

template <typename T>
void storeScalar(std::vector<std::uint8_t>& bytes, std::size_t index, T value)
{
    std::memcpy(bytes.data() + index * sizeof(T), &value, sizeof(T));
}

template <typename T>
T loadScalar(std::vector<std::uint8_t> const& bytes, std::size_t index)
{
    T value;
    std::memcpy(&value, bytes.data() + index * sizeof(T), sizeof(T));
    return value;
}

void storeRawValue(std::vector<std::uint8_t>& bytes, RawKind kind, std::size_t index, float value,
    Nvfp4ColdPageKernelParams const& params)
{
    switch (kind)
    {
    case RawKind::kFloat16: storeScalar(bytes, index, __float2half(value)); break;
    case RawKind::kBfloat16: storeScalar(bytes, index, __float2bfloat16(value)); break;
    case RawKind::kFp8: storeScalar(bytes, index, __nv_fp8_e4m3(value * params.fp8ScaleOrigQuant)); break;
    }
}

float loadRawValue(
    std::vector<std::uint8_t> const& bytes, RawKind kind, std::size_t index, Nvfp4ColdPageKernelParams const& params)
{
    switch (kind)
    {
    case RawKind::kFloat16: return __half2float(loadScalar<half>(bytes, index));
    case RawKind::kBfloat16: return __bfloat162float(loadScalar<__nv_bfloat16>(bytes, index));
    case RawKind::kFp8: return static_cast<float>(loadScalar<__nv_fp8_e4m3>(bytes, index)) * params.fp8ScaleQuantOrig;
    }
    return 0.0F;
}

std::uint32_t linearScaleOffset(std::uint32_t row, std::uint32_t scaleInRow, PageGeometry const& geometry)
{
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(geometry.headDim) / 16;
    return row * scalesPerRow + scaleInRow;
}

constexpr std::array<float, 8> kE2m1Levels{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

float e2m1Value(std::uint8_t nibble)
{
    float const value = kE2m1Levels[nibble & 0x7U];
    return (nibble & 0x8U) != 0 ? -value : value;
}

//! Independent nearest-level oracle; fixtures avoid ties instead of duplicating production tie rules.
std::uint8_t quantizeE2m1(float value)
{
    bool const negative = std::signbit(value);
    float const magnitude = std::abs(value);
    std::uint8_t best = 0;
    float bestDistance = std::abs(magnitude - kE2m1Levels[0]);
    for (std::uint8_t index = 1; index < kE2m1Levels.size(); ++index)
    {
        float const distance = std::abs(magnitude - kE2m1Levels[index]);
        if (distance < bestDistance)
        {
            best = index;
            bestDistance = distance;
        }
    }
    return static_cast<std::uint8_t>(best | (negative ? 0x8U : 0U));
}

//! Exactly representable E2M1 values and E4M3 scales keep byte comparisons deterministic.
std::vector<std::uint8_t> makeRawPage(RawKind kind, std::size_t page, std::uint32_t role,
    Nvfp4ColdPageKernelParams const& params, PageGeometry const& geometry, InputPattern inputPattern)
{
    constexpr std::array<float, 16> densePattern{
        0.0F, 0.5F, -1.0F, 1.5F, -2.0F, 3.0F, -4.0F, 6.0F, -0.5F, 1.0F, -1.5F, 2.0F, -3.0F, 4.0F, -6.0F, 0.5F};
    constexpr std::array<float, 16> firstLaneOutlierPattern{
        6.0F, -0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    constexpr std::array<float, 16> secondLaneOutlierPattern{
        0.0F, -0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, -6.0F};
    constexpr std::array<float, 16> roundingMarginsPattern{
        6.0F, 0.20F, 0.30F, 0.65F, 0.85F, 1.10F, 1.40F, 1.60F, 1.90F, 2.30F, 2.70F, 3.20F, 3.80F, 4.50F, 5.20F, -0.30F};
    constexpr std::array<float, 4> blockScales{0.25F, 0.5F, 1.0F, 2.0F};

    std::vector<std::uint8_t> bytes(rawBytes(kind, geometry));
    for (std::size_t index = 0; index < numElements(geometry); ++index)
    {
        std::size_t const scaleGroup = index / 16;
        float const blockScale = blockScales[(scaleGroup + page + role) % blockScales.size()];
        float normalizedValue = 0.0F;
        if (inputPattern == InputPattern::kDense)
        {
            normalizedValue = densePattern[index % densePattern.size()];
        }
        else if (inputPattern == InputPattern::kSparseOutlier)
        {
            auto const& pattern = (scaleGroup & 1U) == 0 ? firstLaneOutlierPattern : secondLaneOutlierPattern;
            normalizedValue = pattern[index % pattern.size()];
        }
        else if (inputPattern == InputPattern::kRoundingMargins)
        {
            normalizedValue = roundingMarginsPattern[index % roundingMarginsPattern.size()];
        }
        // kAllZero intentionally keeps the zero initializer.
        if (((page / 32) & 1U) != 0)
        {
            normalizedValue = -normalizedValue;
        }
        float const value = normalizedValue * blockScale / params.nvfp4ScaleOrigQuant;
        storeRawValue(bytes, kind, index, value, params);
    }
    return bytes;
}

struct ReferenceNvfp4
{
    std::vector<std::uint8_t> packed;
    std::vector<std::uint8_t> scales;
};

ReferenceNvfp4 compressReference(std::vector<std::uint8_t> const& raw, RawKind kind,
    Nvfp4ColdPageKernelParams const& params, PageGeometry const& geometry)
{
    ReferenceNvfp4 result{{}, {}};
    result.packed.resize(packedBytes(geometry));
    result.scales.resize(scaleBytes(geometry));
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(geometry.headDim) / 16;
    std::uint32_t const rows = static_cast<std::uint32_t>(geometry.numHeads * geometry.tokensPerPage);

    for (std::uint32_t row = 0; row < rows; ++row)
    {
        for (std::uint32_t scaleInRow = 0; scaleInRow < scalesPerRow; ++scaleInRow)
        {
            std::size_t const blockStart = static_cast<std::size_t>(row) * geometry.headDim + scaleInRow * 16;
            float amax = 0.0F;
            for (std::uint32_t i = 0; i < 16; ++i)
            {
                amax = std::max(amax, std::abs(loadRawValue(raw, kind, blockStart + i, params)));
            }

            __nv_fp8_e4m3 blockScale(params.nvfp4ScaleOrigQuant * amax / 6.0F);
            result.scales[linearScaleOffset(row, scaleInRow, geometry)] = blockScale.__x;
            float const blockScaleFloat = static_cast<float>(blockScale);
            float const outputScale = blockScaleFloat == 0.0F ? 0.0F : params.nvfp4ScaleOrigQuant / blockScaleFloat;
            for (std::uint32_t i = 0; i < 16; i += 2)
            {
                std::uint8_t const lo = quantizeE2m1(loadRawValue(raw, kind, blockStart + i, params) * outputScale);
                std::uint8_t const hi = quantizeE2m1(loadRawValue(raw, kind, blockStart + i + 1, params) * outputScale);
                result.packed[(blockStart + i) / 2] = static_cast<std::uint8_t>(lo | (hi << 4));
            }
        }
    }
    return result;
}

std::vector<std::uint8_t> decompressReference(ReferenceNvfp4 const& compressed, RawKind kind,
    Nvfp4ColdPageKernelParams const& params, PageGeometry const& geometry)
{
    std::vector<std::uint8_t> raw(rawBytes(kind, geometry));
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(geometry.headDim) / 16;
    std::uint32_t const rows = static_cast<std::uint32_t>(geometry.numHeads * geometry.tokensPerPage);
    for (std::uint32_t row = 0; row < rows; ++row)
    {
        for (std::uint32_t scaleInRow = 0; scaleInRow < scalesPerRow; ++scaleInRow)
        {
            __nv_fp8_e4m3 blockScale;
            blockScale.__x = compressed.scales[linearScaleOffset(row, scaleInRow, geometry)];
            float const dequantScale = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig;
            std::size_t const blockStart = static_cast<std::size_t>(row) * geometry.headDim + scaleInRow * 16;
            for (std::uint32_t i = 0; i < 16; ++i)
            {
                std::uint8_t const byte = compressed.packed[(blockStart + i) / 2];
                std::uint8_t const nibble = (i & 1U) == 0 ? byte & 0xFU : byte >> 4;
                storeRawValue(raw, kind, blockStart + i, e2m1Value(nibble) * dequantScale, params);
            }
        }
    }
    return raw;
}

void runColdPageRoundTrip(RawKind kind, PageGeometry const& geometry = kDefaultGeometry,
    std::size_t numPages = kDefaultNumPages, InputPattern inputPattern = InputPattern::kDense,
    bool synchronizeBetweenDirections = true, bool repeatRoundTrip = false, std::size_t coldBaseOffset = 0)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 cold-page kernels require an SM100-family GPU";
    }
    std::array<Nvfp4ColdPageKernelParams, 2> const params{makeParams(geometry, 0U), makeParams(geometry, 1U)};
    CudaStream stream;
    std::size_t const rawSlotBytes = rawBytes(kind, geometry);
    // Align compact Slot strides while independently testing an arbitrary staging-base offset.
    std::size_t const compactSlotBytes = roundUp(2U * (packedBytes(geometry) + scaleBytes(geometry)), alignof(uint4));
    // Use alternate Slots to cover non-contiguous KVCM Page indices.
    std::size_t const slotCapacity = 2U * numPages;
    DeviceRegion rawInputK(slotCapacity * rawSlotBytes);
    DeviceRegion rawInputV(slotCapacity * rawSlotBytes);
    DeviceRegion rawOutputK(slotCapacity * rawSlotBytes);
    DeviceRegion rawOutputV(slotCapacity * rawSlotBytes);
    MappedHostRegion compactPages(coldBaseOffset + slotCapacity * compactSlotBytes);
    auto* compactBase = compactPages.bytes() + coldBaseOffset;
    std::vector<std::array<std::vector<std::uint8_t>, 2>> rawHost(numPages);
    std::vector<Nvfp4ColdPageOffloadPageTask> offloadTasks;
    offloadTasks.reserve(numPages);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        std::size_t const slot = 2U * page;
        rawHost[page][0] = makeRawPage(kind, page, 0, params[0], geometry, inputPattern);
        rawHost[page][1] = makeRawPage(kind, page, 1, params[1], geometry, inputPattern);
        rawInputK.copyFrom(slot * rawSlotBytes, rawHost[page][0]);
        rawInputV.copyFrom(slot * rawSlotBytes, rawHost[page][1]);
        offloadTasks.push_back({static_cast<std::int32_t>(slot), static_cast<std::int32_t>(slot)});
    }

    std::size_t const packed = packedBytes(geometry);
    std::size_t const scale = scaleBytes(geometry);
    std::size_t const payloadBytes = 2U * (packed + scale);
    std::uint32_t const paddingBytes = static_cast<std::uint32_t>(compactSlotBytes - payloadBytes);
    std::vector<Nvfp4ColdPageBufferPlan> const inputBuffers{
        {reinterpret_cast<std::uintptr_t>(rawInputK.data()), rawSlotBytes, rawSlotBytes, 0U, 2U * packed, 0U, 0U,
            Nvfp4ColdPageTransform::kNvfp4, params[0]},
        {reinterpret_cast<std::uintptr_t>(rawInputV.data()), rawSlotBytes, rawSlotBytes, packed, 2U * packed + scale,
            payloadBytes, paddingBytes, Nvfp4ColdPageTransform::kNvfp4, params[1]}};

    auto const inputPlan
        = tensorrt_llm::kernels::prepareNvfp4ColdPagePlan(inputBuffers, compactSlotBytes, runtimeType(kind));
    std::vector<std::array<ReferenceNvfp4, 2>> references(numPages);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        references[page][0] = compressReference(rawHost[page][0], kind, params[0], geometry);
        references[page][1] = compressReference(rawHost[page][1], kind, params[1], geometry);
    }

    auto const verifyCompressedPages = [&]
    {
        auto const payload = compactPages.payload();
        EXPECT_TRUE(std::all_of(payload.begin(), payload.begin() + static_cast<std::ptrdiff_t>(coldBaseOffset),
            [](std::uint8_t value) { return value == kCanary; }));
        for (std::size_t page = 0; page < numPages; ++page)
        {
            std::size_t const base = coldBaseOffset + 2U * page * compactSlotBytes;
            auto const region = [&](std::size_t offset, std::size_t bytes)
            {
                return std::vector<std::uint8_t>(payload.begin() + static_cast<std::ptrdiff_t>(base + offset),
                    payload.begin() + static_cast<std::ptrdiff_t>(base + offset + bytes));
            };
            EXPECT_EQ(region(0, packed), references[page][0].packed);
            EXPECT_EQ(region(packed, packed), references[page][1].packed);
            EXPECT_EQ(region(2 * packed, scale), references[page][0].scales);
            EXPECT_EQ(region(2 * packed + scale, scale), references[page][1].scales);
            auto const padding = region(2U * (packed + scale), compactSlotBytes - 2U * (packed + scale));
            EXPECT_TRUE(std::all_of(padding.begin(), padding.end(), [](std::uint8_t value) { return value == 0U; }));

            std::size_t const unusedBase = coldBaseOffset + (2U * page + 1U) * compactSlotBytes;
            EXPECT_TRUE(std::all_of(payload.begin() + static_cast<std::ptrdiff_t>(unusedBase),
                payload.begin() + static_cast<std::ptrdiff_t>(unusedBase + compactSlotBytes),
                [](std::uint8_t value) { return value == kCanary; }));
        }
    };

    tensorrt_llm::kernels::invokeNvfp4ColdPageEncode(offloadTasks, inputPlan, compactBase, stream);

    if (synchronizeBetweenDirections)
    {
        // Read the Host Slot only after StorageManager-style event fencing.
        cudaEvent_t offloadComplete{};
        ASSERT_EQ(cudaEventCreateWithFlags(&offloadComplete, cudaEventDisableTiming), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(offloadComplete, stream), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(offloadComplete), cudaSuccess);
        ASSERT_EQ(cudaEventDestroy(offloadComplete), cudaSuccess);
        verifyCompressedPages();

        std::size_t const compactPayloadBytes = 2U * (packedBytes(geometry) + scaleBytes(geometry));
        if (compactPayloadBytes != compactSlotBytes)
        {
            // Re-encode poisoned recycled Slots to verify deterministic payload and padding bytes.
            auto const firstSerialization = compactPages.payload();
            for (std::size_t page = 0; page < numPages; ++page)
            {
                std::memset(compactBase + 2U * page * compactSlotBytes, 0x5A, compactSlotBytes);
            }
            tensorrt_llm::kernels::invokeNvfp4ColdPageEncode(offloadTasks, inputPlan, compactBase, stream);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            EXPECT_EQ(compactPages.payload(), firstSerialization);
            verifyCompressedPages();
        }
    }

    std::vector<Nvfp4ColdPageOnboardPageTask> onboardTasks;
    onboardTasks.reserve(numPages);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        std::size_t const slot = 2U * page;
        onboardTasks.push_back({static_cast<std::int32_t>(slot), static_cast<std::int32_t>(slot)});
    }
    std::vector<Nvfp4ColdPageBufferPlan> const outputBuffers{
        {reinterpret_cast<std::uintptr_t>(rawOutputK.data()), rawSlotBytes, rawSlotBytes, 0U, 2U * packed, 0U, 0U,
            Nvfp4ColdPageTransform::kNvfp4, params[0]},
        {reinterpret_cast<std::uintptr_t>(rawOutputV.data()), rawSlotBytes, rawSlotBytes, packed, 2U * packed + scale,
            payloadBytes, paddingBytes, Nvfp4ColdPageTransform::kNvfp4, params[1]}};
    auto const outputPlan
        = tensorrt_llm::kernels::prepareNvfp4ColdPagePlan(outputBuffers, compactSlotBytes, runtimeType(kind));
    tensorrt_llm::kernels::invokeNvfp4ColdPageDecode(onboardTasks, outputPlan, compactBase, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    if (!synchronizeBetweenDirections)
    {
        // Verify back-to-back offload/onboard without an intervening Host fence.
        verifyCompressedPages();
    }

    if (repeatRoundTrip)
    {
        // A second lossy round trip catches stale descriptors and validates Q(D(Q(D(Q(x))))).
        for (std::size_t page = 0; page < numPages; ++page)
        {
            for (std::uint32_t role = 0; role < 2; ++role)
            {
                auto const restored = decompressReference(references[page][role], kind, params[role], geometry);
                references[page][role] = compressReference(restored, kind, params[role], geometry);
            }
        }
        tensorrt_llm::kernels::invokeNvfp4ColdPageEncode(offloadTasks, outputPlan, compactBase, stream);
        tensorrt_llm::kernels::invokeNvfp4ColdPageDecode(onboardTasks, inputPlan, compactBase, stream);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        verifyCompressedPages();
    }

    for (std::size_t page = 0; page < numPages; ++page)
    {
        std::size_t const slotOffset = 2U * page * rawSlotBytes;
        auto const& finalK = repeatRoundTrip ? rawInputK : rawOutputK;
        auto const& finalV = repeatRoundTrip ? rawInputV : rawOutputV;
        EXPECT_EQ(finalK.copyToHost(slotOffset, rawSlotBytes),
            decompressReference(references[page][0], kind, params[0], geometry));
        EXPECT_EQ(finalV.copyToHost(slotOffset, rawSlotBytes),
            decompressReference(references[page][1], kind, params[1], geometry));
    }
    rawInputK.expectCanaries();
    rawInputV.expectCanaries();
    rawOutputK.expectCanaries();
    rawOutputV.expectCanaries();
    compactPages.expectCanaries();
}

std::vector<std::uint8_t> makePartialRawPage(RawKind kind, std::int32_t validTokens, bool zeroTail, std::uint32_t role,
    Nvfp4ColdPageKernelParams const& params, PageGeometry const& geometry)
{
    std::vector<std::uint8_t> bytes(rawBytes(kind, geometry));
    for (std::int32_t head = 0; head < geometry.numHeads; ++head)
    {
        for (std::int32_t token = 0; token < geometry.tokensPerPage; ++token)
        {
            for (std::int32_t dim = 0; dim < geometry.headDim; ++dim)
            {
                std::size_t const index
                    = (static_cast<std::size_t>(head) * geometry.tokensPerPage + token) * geometry.headDim + dim;
                float value = 0.0F;
                if (token < validTokens)
                {
                    // Zero-tail and stale-tail fixtures share valid rows with distinct K/V values and scales.
                    value = static_cast<float>((dim % 13) - 6) * 0.125F + static_cast<float>(head) * 0.03125F
                        + static_cast<float>(token) * 0.0078125F + static_cast<float>(role) * 0.0625F;
                }
                else if (!zeroTail)
                {
                    // Poison inactive rows; 16-value groups stay within a token row and cannot affect the prefix.
                    if (dim % 31 == 0)
                    {
                        value = std::numeric_limits<float>::quiet_NaN();
                    }
                    else if (dim % 29 == 0)
                    {
                        value = std::numeric_limits<float>::infinity();
                    }
                    else
                    {
                        value = static_cast<float>((dim % 9) - 4) * 0.25F + static_cast<float>(token) * 0.015625F
                            + static_cast<float>(head + 3 * role) * 0.046875F;
                    }
                }
                storeRawValue(bytes, kind, index, value, params);
            }
        }
    }
    return bytes;
}

void expectSameValidPrefix(std::vector<std::uint8_t> const& lhs, std::vector<std::uint8_t> const& rhs, RawKind kind,
    std::int32_t validTokens, PageGeometry const& geometry)
{
    std::size_t const rowBytes = static_cast<std::size_t>(geometry.headDim) * rawElementBytes(kind);
    for (std::int32_t head = 0; head < geometry.numHeads; ++head)
    {
        for (std::int32_t token = 0; token < validTokens; ++token)
        {
            std::size_t const offset = (static_cast<std::size_t>(head) * geometry.tokensPerPage + token) * rowBytes;
            EXPECT_EQ(std::memcmp(lhs.data() + offset, rhs.data() + offset, rowBytes), 0)
                << "valid prefix differs at head=" << head << " token=" << token;
        }
    }
}

void expectZeroTail(
    std::vector<std::uint8_t> const& bytes, RawKind kind, std::int32_t validTokens, PageGeometry const& geometry)
{
    std::size_t const rowBytes = static_cast<std::size_t>(geometry.headDim) * rawElementBytes(kind);
    std::vector<std::uint8_t> const zero(rowBytes, 0);
    for (std::int32_t head = 0; head < geometry.numHeads; ++head)
    {
        for (std::int32_t token = validTokens; token < geometry.tokensPerPage; ++token)
        {
            std::size_t const offset = (static_cast<std::size_t>(head) * geometry.tokensPerPage + token) * rowBytes;
            EXPECT_EQ(std::memcmp(bytes.data() + offset, zero.data(), rowBytes), 0)
                << "zero tail changed at head=" << head << " token=" << token;
        }
    }
}

void runPartialPageTailIsolation(RawKind kind)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 cold-page kernels require an SM100-family GPU";
    }

    PageGeometry constexpr geometry = kModelLikeGeometry;
    std::size_t constexpr pageVariants = 2U;
    std::size_t const numPages = pageVariants * kValidTokenCounts.size();
    std::array<Nvfp4ColdPageKernelParams, 2> const params{makeParams(geometry, 0U), makeParams(geometry, 1U)};
    std::size_t const rawSlotBytes = rawBytes(kind, geometry);
    std::size_t const compactSlotBytes = roundUp(2U * (packedBytes(geometry) + scaleBytes(geometry)), alignof(uint4));

    DeviceRegion rawInputK(numPages * rawSlotBytes);
    DeviceRegion rawInputV(numPages * rawSlotBytes);
    DeviceRegion rawOutputK(numPages * rawSlotBytes);
    DeviceRegion rawOutputV(numPages * rawSlotBytes);
    MappedHostRegion compactPages(numPages * compactSlotBytes);
    std::vector<Nvfp4ColdPageOffloadPageTask> offloadTasks;
    std::vector<Nvfp4ColdPageOnboardPageTask> onboardTasks;
    offloadTasks.reserve(numPages);
    onboardTasks.reserve(numPages);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        std::int32_t const validTokens = kValidTokenCounts[page / pageVariants];
        bool const zeroTail = page % pageVariants == 0U;
        rawInputK.copyFrom(
            page * rawSlotBytes, makePartialRawPage(kind, validTokens, zeroTail, 0, params[0], geometry));
        rawInputV.copyFrom(
            page * rawSlotBytes, makePartialRawPage(kind, validTokens, zeroTail, 1, params[1], geometry));
        auto const pageIndex = static_cast<std::int32_t>(page);
        offloadTasks.push_back({pageIndex, pageIndex});
        onboardTasks.push_back({pageIndex, pageIndex});
    }

    std::size_t const packed = packedBytes(geometry);
    std::size_t const scale = scaleBytes(geometry);
    std::size_t const payloadBytes = 2U * (packed + scale);
    auto const makeBuffers = [&](DeviceRegion const& rawK, DeviceRegion const& rawV)
    {
        return std::vector<Nvfp4ColdPageBufferPlan>{
            {reinterpret_cast<std::uintptr_t>(rawK.data()), rawSlotBytes, rawSlotBytes, 0U, 2U * packed, 0U, 0U,
                Nvfp4ColdPageTransform::kNvfp4, params[0]},
            {reinterpret_cast<std::uintptr_t>(rawV.data()), rawSlotBytes, rawSlotBytes, packed, 2U * packed + scale,
                payloadBytes, static_cast<std::uint32_t>(compactSlotBytes - payloadBytes),
                Nvfp4ColdPageTransform::kNvfp4, params[1]}};
    };
    auto const inputPlan = tensorrt_llm::kernels::prepareNvfp4ColdPagePlan(
        makeBuffers(rawInputK, rawInputV), compactSlotBytes, runtimeType(kind));
    auto const outputPlan = tensorrt_llm::kernels::prepareNvfp4ColdPagePlan(
        makeBuffers(rawOutputK, rawOutputV), compactSlotBytes, runtimeType(kind));
    CudaStream stream;
    tensorrt_llm::kernels::invokeNvfp4ColdPageEncode(offloadTasks, inputPlan, compactPages.data(), stream);
    tensorrt_llm::kernels::invokeNvfp4ColdPageDecode(onboardTasks, outputPlan, compactPages.data(), stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (std::size_t pair = 0; pair < kValidTokenCounts.size(); ++pair)
    {
        std::int32_t const validTokens = kValidTokenCounts[pair];
        for (std::uint32_t role = 0; role < 2U; ++role)
        {
            auto const& output = role == 0U ? rawOutputK : rawOutputV;
            auto const zeroOutput = output.copyToHost(pageVariants * pair * rawSlotBytes, rawSlotBytes);
            auto const staleOutput = output.copyToHost((pageVariants * pair + 1U) * rawSlotBytes, rawSlotBytes);
            expectSameValidPrefix(zeroOutput, staleOutput, kind, validTokens, geometry);
            expectZeroTail(zeroOutput, kind, validTokens, geometry);
        }
    }

    rawInputK.expectCanaries();
    rawInputV.expectCanaries();
    rawOutputK.expectCanaries();
    rawOutputV.expectCanaries();
    compactPages.expectCanaries();
}

struct RoundTripCase
{
    char const* name;
    RawKind kind;
    PageGeometry geometry{kDefaultGeometry};
    std::size_t numPages{kDefaultNumPages};
    InputPattern inputPattern{InputPattern::kDense};
    bool synchronizeBetweenDirections{true};
    bool repeatRoundTrip{false};
    std::size_t coldBaseOffset{0};
};

RoundTripCase constexpr kRoundTripCases[]{
    {"DefaultFloat16", RawKind::kFloat16},
    {"DefaultBfloat16", RawKind::kBfloat16},
    {"DefaultFp8IndependentScales", RawKind::kFp8},
    {"SmallVectorFloat16", RawKind::kFloat16, kSmallVectorGeometry, 1},
    {"MinimumFloat16", RawKind::kFloat16, kMinimumCompactGeometry, 1},
    {"MinimumBfloat16", RawKind::kBfloat16, kMinimumCompactGeometry, 1},
    {"MinimumFp8", RawKind::kFp8, kMinimumCompactGeometry, 1},
    {"PackedScaleTailFloat16", RawKind::kFloat16, kPackedBodyAndTailGeometry, 1},
    {"PackedScaleTailFp8", RawKind::kFp8, kPackedBodyAndTailGeometry, 1},
    {"ByteAlignedColdBaseBfloat16", RawKind::kBfloat16, kPackedBodyAndTailGeometry, 1, InputPattern::kDense, true,
        false, 1},
    {"ByteAlignedColdBaseFp8", RawKind::kFp8, kPackedBodyAndTailGeometry, 1, InputPattern::kDense, true, false, 1},
    {"OddTokenFloat16", RawKind::kFloat16, kLinearScaleTailGeometry, 1},
    {"OddTokenBfloat16", RawKind::kBfloat16, kLinearScaleTailGeometry, 1},
    {"OddTokenFp8", RawKind::kFp8, kLinearScaleTailGeometry, 1},
    {"TiledTailsFloat16", RawKind::kFloat16, kTiledLinearScaleTailGeometry, 1},
    {"TiledTailsFp8", RawKind::kFp8, kTiledLinearScaleTailGeometry, 1},
    {"CrossRowTileBfloat16", RawKind::kBfloat16, kCrossRowTileGeometry, 1, InputPattern::kDense, true, false, 1},
    {"LargeHeadDimFp8", RawKind::kFp8, kLargeHeadDimTailGeometry, 1},
    {"ModelLikeBfloat16", RawKind::kBfloat16, kModelLikeGeometry, 1},
    {"ModelLikeFp8", RawKind::kFp8, kModelLikeGeometry, 1},
    {"ZeroGroupsFloat16", RawKind::kFloat16, kDefaultGeometry, 1, InputPattern::kAllZero},
    {"ZeroGroupsBfloat16", RawKind::kBfloat16, kDefaultGeometry, 1, InputPattern::kAllZero},
    {"ZeroGroupsFp8", RawKind::kFp8, kDefaultGeometry, 1, InputPattern::kAllZero},
    {"WarpLaneAmaxFloat16", RawKind::kFloat16, kDefaultGeometry, 1, InputPattern::kSparseOutlier},
    {"WarpLaneAmaxBfloat16", RawKind::kBfloat16, kDefaultGeometry, 1, InputPattern::kSparseOutlier},
    {"WarpLaneAmaxFp8", RawKind::kFp8, kDefaultGeometry, 1, InputPattern::kSparseOutlier},
    {"ReuseDefaultFloat16", RawKind::kFloat16, kDefaultGeometry, 3, InputPattern::kDense, true, true},
    {"ReuseDefaultBfloat16", RawKind::kBfloat16, kDefaultGeometry, 3, InputPattern::kDense, true, true},
    {"ReuseDefaultFp8", RawKind::kFp8, kDefaultGeometry, 3, InputPattern::kDense, true, true},
    {"ReuseModelLikeFloat16", RawKind::kFloat16, kModelLikeGeometry, 2, InputPattern::kDense, true, true},
    {"ReuseModelLikeBfloat16", RawKind::kBfloat16, kModelLikeGeometry, 2, InputPattern::kDense, true, true},
    {"ReuseModelLikeFp8", RawKind::kFp8, kModelLikeGeometry, 2, InputPattern::kDense, true, true},
    {"RoundingMarginsFloat16", RawKind::kFloat16, kDefaultGeometry, 1, InputPattern::kRoundingMargins},
    {"CrossLaunchBfloat16", RawKind::kBfloat16, kSmallVectorGeometry, kCrossLaunchNumPages},
    {"CrossLaunchFp8", RawKind::kFp8, kSmallVectorGeometry, kCrossLaunchNumPages},
    {"PdlBfloat16", RawKind::kBfloat16, kSmallVectorGeometry, 65, InputPattern::kDense, false},
    {"PdlFp8", RawKind::kFp8, kSmallVectorGeometry, 65, InputPattern::kDense, false},
};

class Nvfp4ColdPageRoundTripTest : public testing::TestWithParam<RoundTripCase>
{
};

TEST_P(Nvfp4ColdPageRoundTripTest, MatchesReference)
{
    auto const& test = GetParam();
    runColdPageRoundTrip(test.kind, test.geometry, test.numPages, test.inputPattern, test.synchronizeBetweenDirections,
        test.repeatRoundTrip, test.coldBaseOffset);
}

std::string roundTripCaseName(testing::TestParamInfo<RoundTripCase> const& info)
{
    return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(Scenarios, Nvfp4ColdPageRoundTripTest, testing::ValuesIn(kRoundTripCases), roundTripCaseName);

void runUnaryMlaWithLosslessSideRoundTrip(RawKind kind)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 cold-page kernels require an SM100-family GPU";
    }

    PageGeometry constexpr geometry{1, 64, 576};
    std::size_t constexpr numPages = 2;
    std::size_t constexpr sideRawBytes = 64U * (128U + 4U);
    std::size_t constexpr sideSlotBytes = sideRawBytes + 13U;
    std::size_t constexpr coldBaseOffset = 1;
    auto const params = makeParams(geometry);
    std::size_t const mlaRawBytes = rawBytes(kind, geometry);
    std::size_t const mlaPackedBytes = packedBytes(geometry);
    std::size_t const mlaScaleBytes = scaleBytes(geometry);
    std::size_t const mlaPayloadBytes = mlaPackedBytes + mlaScaleBytes;
    std::size_t constexpr gapBeforeSide = 3;
    std::size_t const sideColdOffset = mlaPayloadBytes + gapBeforeSide;
    std::size_t const sideColdEnd = sideColdOffset + sideRawBytes;
    std::size_t const coldPageBytes = roundUp(sideColdEnd, alignof(uint4));

    DeviceRegion mlaInput(numPages * mlaRawBytes);
    DeviceRegion mlaOutput(numPages * mlaRawBytes);
    DeviceRegion sideInput(numPages * sideSlotBytes);
    DeviceRegion sideOutput(numPages * sideSlotBytes);
    MappedHostRegion coldStorage(coldBaseOffset + numPages * coldPageBytes);
    auto* coldBase = coldStorage.bytes() + coldBaseOffset;

    std::array<std::vector<std::uint8_t>, numPages> mlaHost;
    std::array<std::vector<std::uint8_t>, numPages> sideHost;
    std::array<ReferenceNvfp4, numPages> references;
    std::vector<Nvfp4ColdPageOffloadPageTask> offloadTasks;
    std::vector<Nvfp4ColdPageOnboardPageTask> onboardTasks;
    for (std::size_t page = 0; page < numPages; ++page)
    {
        mlaHost[page] = makeRawPage(kind, page, 0U, params, geometry, InputPattern::kDense);
        references[page] = compressReference(mlaHost[page], kind, params, geometry);
        sideHost[page].resize(sideRawBytes);
        for (std::size_t byte = 0; byte < sideRawBytes; ++byte)
        {
            sideHost[page][byte] = static_cast<std::uint8_t>((17U * byte + 53U * page + 11U) & 0xFFU);
        }
        mlaInput.copyFrom(page * mlaRawBytes, mlaHost[page]);
        sideInput.copyFrom(page * sideSlotBytes, sideHost[page]);
        auto const pageIndex = static_cast<std::int32_t>(page);
        offloadTasks.push_back({pageIndex, pageIndex});
        onboardTasks.push_back({pageIndex, pageIndex});
    }

    auto const makePlans = [&](DeviceRegion const& mla, DeviceRegion const& side)
    {
        return std::vector<Nvfp4ColdPageBufferPlan>{
            {reinterpret_cast<std::uintptr_t>(mla.data()), mlaRawBytes, mlaRawBytes, 0U, mlaPackedBytes,
                mlaPayloadBytes, static_cast<std::uint32_t>(gapBeforeSide), Nvfp4ColdPageTransform::kNvfp4, params},
            {reinterpret_cast<std::uintptr_t>(side.data()), sideSlotBytes, sideRawBytes, sideColdOffset, 0U,
                sideColdEnd, static_cast<std::uint32_t>(coldPageBytes - sideColdEnd),
                Nvfp4ColdPageTransform::kLosslessCopy, {}}};
    };
    auto const inputPlan = tensorrt_llm::kernels::prepareNvfp4ColdPagePlan(
        makePlans(mlaInput, sideInput), coldPageBytes, runtimeType(kind));
    auto const outputPlan = tensorrt_llm::kernels::prepareNvfp4ColdPagePlan(
        makePlans(mlaOutput, sideOutput), coldPageBytes, runtimeType(kind));

    CudaStream stream;
    tensorrt_llm::kernels::invokeNvfp4ColdPageEncode(offloadTasks, inputPlan, coldBase, stream);
    tensorrt_llm::kernels::invokeNvfp4ColdPageDecode(onboardTasks, outputPlan, coldBase, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    auto const cold = coldStorage.payload();
    EXPECT_EQ(cold.front(), kCanary);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        std::size_t const coldPage = coldBaseOffset + page * coldPageBytes;
        auto const coldRegion = [&](std::size_t offset, std::size_t bytes)
        {
            return std::vector<std::uint8_t>(cold.begin() + static_cast<std::ptrdiff_t>(coldPage + offset),
                cold.begin() + static_cast<std::ptrdiff_t>(coldPage + offset + bytes));
        };
        EXPECT_EQ(coldRegion(0U, mlaPackedBytes), references[page].packed);
        EXPECT_EQ(coldRegion(mlaPackedBytes, mlaScaleBytes), references[page].scales);
        EXPECT_TRUE(std::all_of(cold.begin() + static_cast<std::ptrdiff_t>(coldPage + mlaPayloadBytes),
            cold.begin() + static_cast<std::ptrdiff_t>(coldPage + sideColdOffset),
            [](std::uint8_t value) { return value == 0U; }));
        EXPECT_EQ(coldRegion(sideColdOffset, sideRawBytes), sideHost[page]);
        EXPECT_TRUE(std::all_of(cold.begin() + static_cast<std::ptrdiff_t>(coldPage + sideColdEnd),
            cold.begin() + static_cast<std::ptrdiff_t>(coldPage + coldPageBytes),
            [](std::uint8_t value) { return value == 0U; }));

        EXPECT_EQ(mlaOutput.copyToHost(page * mlaRawBytes, mlaRawBytes),
            decompressReference(references[page], kind, params, geometry));
        EXPECT_EQ(sideOutput.copyToHost(page * sideSlotBytes, sideRawBytes), sideHost[page]);
        for (auto const* side : {&sideInput, &sideOutput})
        {
            auto const slotTail = side->copyToHost(page * sideSlotBytes + sideRawBytes, sideSlotBytes - sideRawBytes);
            EXPECT_TRUE(
                std::all_of(slotTail.begin(), slotTail.end(), [](std::uint8_t value) { return value == kCanary; }));
        }
    }
    mlaInput.expectCanaries();
    mlaOutput.expectCanaries();
    sideInput.expectCanaries();
    sideOutput.expectCanaries();
    coldStorage.expectCanaries();
}

class Nvfp4ColdPageMlaSideTest : public testing::TestWithParam<RawKind>
{
};

TEST_P(Nvfp4ColdPageMlaSideTest, MlaPageAndDefaultDsaIndexKeyRoundTripExactly)
{
    runUnaryMlaWithLosslessSideRoundTrip(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    AllRuntimeTypes, Nvfp4ColdPageMlaSideTest, testing::Values(RawKind::kFloat16, RawKind::kBfloat16, RawKind::kFp8));

TEST(Nvfp4ColdPageWholePageTest, DifferentLayerScalesRemainInOneCompletePageBatch)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 cold-page kernels require an SM100-family GPU";
    }

    constexpr std::size_t numLayers = 2;
    RawKind constexpr kind = RawKind::kBfloat16;
    PageGeometry constexpr geometry = kMinimumCompactGeometry;
    std::size_t const rawSlotBytes = rawBytes(kind, geometry);
    std::size_t const layerRecordBytes = 2U * (packedBytes(geometry) + scaleBytes(geometry));
    std::size_t const layerRecordStride = roundUp(layerRecordBytes, alignof(uint4));
    std::size_t const coldPageBytes = numLayers * layerRecordStride;

    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawInputK;
    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawInputV;
    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawOutputK;
    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawOutputV;
    std::array<std::array<Nvfp4ColdPageKernelParams, 2>, numLayers> params{
        std::array{makeParams(geometry, 0U), makeParams(geometry, 1U)},
        std::array{makeParams(geometry, 0U), makeParams(geometry, 1U)}};
    // Distinct per-layer K/V scales verify blockIdx.y selects immutable launch metadata.
    params[1][0].nvfp4ScaleOrigQuant = 0.5F;
    params[1][0].nvfp4ScaleQuantOrig = 2.0F;
    params[1][1].nvfp4ScaleOrigQuant = 4.0F;
    params[1][1].nvfp4ScaleQuantOrig = 0.25F;

    std::array<std::array<std::vector<std::uint8_t>, 2>, numLayers> rawHost;
    std::array<std::array<ReferenceNvfp4, 2>, numLayers> references;
    std::vector<Nvfp4ColdPageBufferPlan> inputPlans;
    std::vector<Nvfp4ColdPageBufferPlan> outputPlans;
    inputPlans.reserve(2U * numLayers);
    outputPlans.reserve(2U * numLayers);
    for (std::size_t layer = 0; layer < numLayers; ++layer)
    {
        rawInputK[layer] = std::make_unique<DeviceRegion>(rawSlotBytes);
        rawInputV[layer] = std::make_unique<DeviceRegion>(rawSlotBytes);
        rawOutputK[layer] = std::make_unique<DeviceRegion>(rawSlotBytes);
        rawOutputV[layer] = std::make_unique<DeviceRegion>(rawSlotBytes);
        for (std::uint32_t role = 0; role < 2; ++role)
        {
            rawHost[layer][role] = makeRawPage(kind, layer, role, params[layer][role], geometry, InputPattern::kDense);
            references[layer][role] = compressReference(rawHost[layer][role], kind, params[layer][role], geometry);
        }
        rawInputK[layer]->copyFrom(rawHost[layer][0]);
        rawInputV[layer]->copyFrom(rawHost[layer][1]);
        std::size_t const base = layer * layerRecordStride;
        std::size_t const packed = packedBytes(geometry);
        std::size_t const scale = scaleBytes(geometry);
        auto const appendPlans = [&](auto& plans, DeviceRegion const& rawK, DeviceRegion const& rawV)
        {
            plans.push_back({reinterpret_cast<std::uintptr_t>(rawK.data()), rawSlotBytes, rawSlotBytes, base,
                base + 2U * packed, 0U, 0U, Nvfp4ColdPageTransform::kNvfp4, params[layer][0]});
            plans.push_back({reinterpret_cast<std::uintptr_t>(rawV.data()), rawSlotBytes, rawSlotBytes, base + packed,
                base + 2U * packed + scale, base + layerRecordBytes,
                static_cast<std::uint32_t>(layerRecordStride - layerRecordBytes), Nvfp4ColdPageTransform::kNvfp4,
                params[layer][1]});
        };
        appendPlans(inputPlans, *rawInputK[layer], *rawInputV[layer]);
        appendPlans(outputPlans, *rawOutputK[layer], *rawOutputV[layer]);
    }

    MappedHostRegion compactPage(coldPageBytes);
    CudaStream stream;
    auto const inputPlan = tensorrt_llm::kernels::prepareNvfp4ColdPagePlan(
        inputPlans, coldPageBytes, Nvfp4ColdPageRuntimeType::kBfloat16);
    auto const outputPlan = tensorrt_llm::kernels::prepareNvfp4ColdPagePlan(
        outputPlans, coldPageBytes, Nvfp4ColdPageRuntimeType::kBfloat16);
    tensorrt_llm::kernels::invokeNvfp4ColdPageEncode({{0, 0}}, inputPlan, compactPage.data(), stream);
    tensorrt_llm::kernels::invokeNvfp4ColdPageDecode({{0, 0}}, outputPlan, compactPage.data(), stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    auto const compact = compactPage.payload();
    std::size_t const packed = packedBytes(geometry);
    std::size_t const scale = scaleBytes(geometry);
    auto const compactRegion = [&](std::size_t offset, std::size_t bytes)
    {
        return std::vector<std::uint8_t>(compact.begin() + static_cast<std::ptrdiff_t>(offset),
            compact.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    };
    for (std::size_t layer = 0; layer < numLayers; ++layer)
    {
        std::size_t const base = layer * layerRecordStride;
        EXPECT_EQ(compactRegion(base, packed), references[layer][0].packed);
        EXPECT_EQ(compactRegion(base + packed, packed), references[layer][1].packed);
        EXPECT_EQ(compactRegion(base + 2U * packed, scale), references[layer][0].scales);
        EXPECT_EQ(compactRegion(base + 2U * packed + scale, scale), references[layer][1].scales);
        auto const padding = compactRegion(base + layerRecordBytes, layerRecordStride - layerRecordBytes);
        EXPECT_TRUE(std::all_of(padding.begin(), padding.end(), [](std::uint8_t value) { return value == 0U; }));
        EXPECT_EQ(rawOutputK[layer]->copyToHost(),
            decompressReference(references[layer][0], kind, params[layer][0], geometry));
        EXPECT_EQ(rawOutputV[layer]->copyToHost(),
            decompressReference(references[layer][1], kind, params[layer][1], geometry));
    }
    compactPage.expectCanaries();
}

void expectWholePageLaunchTopology(std::size_t numPages, std::vector<std::uint32_t> expectedGridZ)
{
    constexpr std::size_t numLayers = 2;
    RawKind constexpr kind = RawKind::kBfloat16;
    std::size_t const rawSlotBytes = rawBytes(kind, kSmallVectorGeometry);
    std::size_t const recordBytes = 2U * (packedBytes(kSmallVectorGeometry) + scaleBytes(kSmallVectorGeometry));
    std::size_t const recordStride = roundUp(recordBytes, alignof(uint4));
    std::size_t const coldPageBytes = numLayers * recordStride;

    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawK;
    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawV;
    std::vector<Nvfp4ColdPageBufferPlan> buffers;
    buffers.reserve(2U * numLayers);
    for (std::size_t layer = 0; layer < numLayers; ++layer)
    {
        rawK[layer] = std::make_unique<DeviceRegion>(numPages * rawSlotBytes);
        rawV[layer] = std::make_unique<DeviceRegion>(numPages * rawSlotBytes);
        auto kParams = makeParams(kSmallVectorGeometry, 0U);
        auto const vParams = makeParams(kSmallVectorGeometry, 1U);
        kParams.nvfp4ScaleOrigQuant *= static_cast<float>(layer + 1U);
        kParams.nvfp4ScaleQuantOrig /= static_cast<float>(layer + 1U);
        std::size_t const base = layer * recordStride;
        std::size_t const packed = packedBytes(kSmallVectorGeometry);
        std::size_t const scale = scaleBytes(kSmallVectorGeometry);
        buffers.push_back({reinterpret_cast<std::uintptr_t>(rawK[layer]->data()), rawSlotBytes, rawSlotBytes, base,
            base + 2U * packed, 0U, 0U, Nvfp4ColdPageTransform::kNvfp4, kParams});
        buffers.push_back({reinterpret_cast<std::uintptr_t>(rawV[layer]->data()), rawSlotBytes, rawSlotBytes,
            base + packed, base + 2U * packed + scale, base + recordBytes,
            static_cast<std::uint32_t>(recordStride - recordBytes), Nvfp4ColdPageTransform::kNvfp4, vParams});
    }

    MappedHostRegion coldPages(numPages * coldPageBytes);
    std::vector<Nvfp4ColdPageOffloadPageTask> offloadPages;
    std::vector<Nvfp4ColdPageOnboardPageTask> onboardPages;
    offloadPages.reserve(numPages);
    onboardPages.reserve(numPages);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        auto const pageIndex = static_cast<std::int32_t>(page);
        offloadPages.push_back({pageIndex, pageIndex});
        onboardPages.push_back({pageIndex, pageIndex});
    }

    CudaStream stream;
    auto const plan
        = tensorrt_llm::kernels::prepareNvfp4ColdPagePlan(buffers, coldPageBytes, Nvfp4ColdPageRuntimeType::kBfloat16);
    auto const expectWholePageKernels = [&](auto const& enqueue)
    {
        cudaGraph_t graph{};
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        enqueue();
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);

        std::size_t numNodes = 0;
        ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &numNodes), cudaSuccess);
        std::vector<cudaGraphNode_t> nodes(numNodes);
        ASSERT_EQ(cudaGraphGetNodes(graph, nodes.data(), &numNodes), cudaSuccess);
        std::size_t kernelNodes = 0;
        std::vector<std::uint32_t> actualGridZ;
        for (auto const node : nodes)
        {
            cudaGraphNodeType nodeType{};
            ASSERT_EQ(cudaGraphNodeGetType(node, &nodeType), cudaSuccess);
            if (nodeType != cudaGraphNodeTypeKernel)
            {
                continue;
            }
            ++kernelNodes;
            cudaKernelNodeParams nodeParams{};
            ASSERT_EQ(cudaGraphKernelNodeGetParams(node, &nodeParams), cudaSuccess);
            EXPECT_EQ(nodeParams.gridDim.y, 2U * numLayers);
            actualGridZ.push_back(nodeParams.gridDim.z);
        }
        std::sort(actualGridZ.begin(), actualGridZ.end());
        std::sort(expectedGridZ.begin(), expectedGridZ.end());
        EXPECT_EQ(kernelNodes, expectedGridZ.size());
        EXPECT_EQ(actualGridZ, expectedGridZ);
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    };

    expectWholePageKernels(
        [&] { tensorrt_llm::kernels::invokeNvfp4ColdPageEncode(offloadPages, plan, coldPages.data(), stream); });
    expectWholePageKernels(
        [&] { tensorrt_llm::kernels::invokeNvfp4ColdPageDecode(onboardPages, plan, coldPages.data(), stream); });
}

TEST(Nvfp4ColdPageWholePageTest, TwoHundredFiftySevenPagesUseExactlyTwoWholePageKernelsPerDirection)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 cold-page kernels require an SM100-family GPU";
    }
    expectWholePageLaunchTopology(257, {1, 256});
}

class Nvfp4ColdPageTailTest : public testing::TestWithParam<RawKind>
{
};

TEST_P(Nvfp4ColdPageTailTest, InactiveRowsDoNotAffectTheValidPrefix)
{
    runPartialPageTailIsolation(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    AllRuntimeTypes, Nvfp4ColdPageTailTest, testing::Values(RawKind::kFloat16, RawKind::kBfloat16, RawKind::kFp8));

TEST(Nvfp4ColdPageValidationTest, EmptyBatchIsAnAsyncNoOp)
{
    tensorrt_llm::kernels::invokeNvfp4ColdPageEncode({}, Nvfp4ColdPagePreparedPlan{}, nullptr, nullptr);
    tensorrt_llm::kernels::invokeNvfp4ColdPageDecode({}, Nvfp4ColdPagePreparedPlan{}, nullptr, nullptr);
}

TEST(Nvfp4ColdPageValidationTest, RejectsInvalidGeometryAndScalesBeforeLaunch)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 cold-page kernels require an SM100-family GPU";
    }

    std::size_t const rawSlotBytes = rawBytes(RawKind::kFloat16, kDefaultGeometry);
    LayerBuffers buffers(rawSlotBytes);
    std::size_t const coldPageBytes = 2U * (packedBytes(kDefaultGeometry) + scaleBytes(kDefaultGeometry));
    auto const prepare =
        [&](Nvfp4ColdPageKernelParams const& params, Nvfp4ColdPageRuntimeType type = Nvfp4ColdPageRuntimeType::kFloat16)
    {
        std::size_t activeRawBytes = rawSlotBytes;
        std::size_t dataBytes = packedBytes(kDefaultGeometry);
        std::size_t scales = scaleBytes(kDefaultGeometry);
        if (params.numKvHeads > 0 && params.tokensPerPage > 0 && params.headDim > 0)
        {
            std::uint64_t const elements = static_cast<std::uint64_t>(params.numKvHeads)
                * static_cast<std::uint64_t>(params.tokensPerPage) * static_cast<std::uint64_t>(params.headDim);
            std::uint64_t const candidateRawBytes = elements * (type == Nvfp4ColdPageRuntimeType::kFp8E4m3 ? 1U : 2U);
            if (candidateRawBytes > 0U && candidateRawBytes <= rawSlotBytes)
            {
                activeRawBytes = static_cast<std::size_t>(candidateRawBytes);
                dataBytes = static_cast<std::size_t>(elements / 2U);
                scales = static_cast<std::size_t>(elements / 16U);
            }
        }
        Nvfp4ColdPageBufferPlan const buffer{reinterpret_cast<std::uintptr_t>(buffers.rawK.data()), rawSlotBytes,
            activeRawBytes, 0U, dataBytes, dataBytes + scales,
            static_cast<std::uint32_t>(coldPageBytes - dataBytes - scales), Nvfp4ColdPageTransform::kNvfp4, params};
        static_cast<void>(tensorrt_llm::kernels::prepareNvfp4ColdPagePlan({buffer}, coldPageBytes, type));
    };
    auto const expectInvalid
        = [&](char const* name, auto const& mutate, Nvfp4ColdPageRuntimeType type = Nvfp4ColdPageRuntimeType::kFloat16)
    {
        SCOPED_TRACE(name);
        auto params = makeParams();
        mutate(params);
        EXPECT_ANY_THROW(prepare(params, type));
    };

    auto valid = makeParams();
    valid.tokensPerPage = 6;
    EXPECT_NO_THROW(prepare(valid));
    EXPECT_NO_THROW(prepare(makeParams(PageGeometry{1, 1, 16})));

    expectInvalid("zero heads", [](auto& params) { params.numKvHeads = 0; });
    expectInvalid("zero tokens", [](auto& params) { params.tokensPerPage = 0; });
    expectInvalid("zero head dimension", [](auto& params) { params.headDim = 0; });
    expectInvalid("unaligned head dimension", [](auto& params) { params.headDim = 24; });
    expectInvalid(
        "element count overflow", [](auto& params) { params.numKvHeads = std::numeric_limits<std::int32_t>::max(); });
    expectInvalid("zero NVFP4 quant scale", [](auto& params) { params.nvfp4ScaleOrigQuant = 0.0F; });
    expectInvalid("negative NVFP4 dequant scale", [](auto& params) { params.nvfp4ScaleQuantOrig = -1.0F; });
    expectInvalid("NaN NVFP4 quant scale",
        [](auto& params) { params.nvfp4ScaleOrigQuant = std::numeric_limits<float>::quiet_NaN(); });
    expectInvalid("infinite NVFP4 dequant scale",
        [](auto& params) { params.nvfp4ScaleQuantOrig = std::numeric_limits<float>::infinity(); });
    expectInvalid(
        "zero FP8 quant scale", [](auto& params) { params.fp8ScaleOrigQuant = 0.0F; },
        Nvfp4ColdPageRuntimeType::kFp8E4m3);
    expectInvalid(
        "infinite FP8 dequant scale",
        [](auto& params) { params.fp8ScaleQuantOrig = std::numeric_limits<float>::infinity(); },
        Nvfp4ColdPageRuntimeType::kFp8E4m3);
}

TEST(Nvfp4ColdPageValidationTest, RejectsInvalidLaunchDescriptors)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 cold-page kernels require an SM100-family GPU";
    }

    std::size_t const rawSlotBytes = rawBytes(RawKind::kFloat16, kDefaultGeometry);
    LayerBuffers buffers(rawSlotBytes);
    Nvfp4ColdPageOffloadPageTask const validOffload{0, 0};
    std::size_t const coldPageBytes = 2U * (packedBytes(kDefaultGeometry) + scaleBytes(kDefaultGeometry));
    std::size_t const packed = packedBytes(kDefaultGeometry);
    std::size_t const scale = scaleBytes(kDefaultGeometry);
    Nvfp4ColdPageBufferPlan const validBuffer{reinterpret_cast<std::uintptr_t>(buffers.rawK.data()), rawSlotBytes,
        rawSlotBytes, 0U, packed, packed + scale, static_cast<std::uint32_t>(coldPageBytes - packed - scale),
        Nvfp4ColdPageTransform::kNvfp4, makeParams()};
    auto const prepare = [&](Nvfp4ColdPageBufferPlan const& buffer, std::size_t pageBytes,
                             Nvfp4ColdPageRuntimeType type = Nvfp4ColdPageRuntimeType::kFloat16)
    { return tensorrt_llm::kernels::prepareNvfp4ColdPagePlan({buffer}, pageBytes, type); };
    auto const expectInvalid
        = [&](char const* name, auto const& mutate, Nvfp4ColdPageRuntimeType type = Nvfp4ColdPageRuntimeType::kFloat16)
    {
        SCOPED_TRACE(name);
        auto buffer = validBuffer;
        mutate(buffer);
        EXPECT_ANY_THROW(static_cast<void>(prepare(buffer, coldPageBytes, type)));
    };
    auto const validPlan = prepare(validBuffer, coldPageBytes);

    expectInvalid("unaligned raw base", [](auto& buffer) { buffer.rawBase += 1U; });
    EXPECT_ANY_THROW(tensorrt_llm::kernels::invokeNvfp4ColdPageEncode({validOffload}, validPlan, nullptr, nullptr));
    expectInvalid("unaligned raw stride", [](auto& buffer) { buffer.rawSlotBytes += alignof(uint4) / 2U; });
    expectInvalid("raw bytes exceed stride", [](auto& buffer) { buffer.rawBytes = buffer.rawSlotBytes + 1U; });
    expectInvalid("raw bytes mismatch geometry", [](auto& buffer) { buffer.rawBytes -= alignof(uint4); });
    expectInvalid("cold data interval exceeds page", [&](auto& buffer) { buffer.coldDataOffset = coldPageBytes; });
    expectInvalid("cold intervals overlap", [](auto& buffer) { buffer.coldScaleOffset = buffer.coldDataOffset; });
    EXPECT_ANY_THROW(static_cast<void>(prepare(validBuffer, coldPageBytes + alignof(uint4) / 2U)));
    expectInvalid(
        "unaligned FP8 raw base", [](auto& buffer) { buffer.rawBase += 1U; }, Nvfp4ColdPageRuntimeType::kFp8E4m3);

    auto const unsupportedType = static_cast<Nvfp4ColdPageRuntimeType>(255);
    EXPECT_ANY_THROW(static_cast<void>(prepare(validBuffer, coldPageBytes, unsupportedType)));
}

} // namespace

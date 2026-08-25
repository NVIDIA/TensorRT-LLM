/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kv_cache_compression/coldPageCodec.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tensorrt_llm::kv_cache_compression
{
namespace
{

namespace kv = batch_manager::kv_cache_manager_v2;

static_assert(std::is_base_of_v<kv::IKvCacheColdPageCodec, PlannedColdPageCodec>);

struct RecordedLaunch
{
    int encodeCalls = 0;
    int decodeCalls = 0;
    std::vector<kernels::Nvfp4ColdPageOffloadPageTask> encodePages;
    std::vector<kernels::Nvfp4ColdPageOnboardPageTask> decodePages;
    kernels::Nvfp4ColdPagePreparedPlan plan;
    void const* coldBase = nullptr;
    cudaStream_t stream{};
};

RecordedLaunch gLaunch;

constexpr std::uintptr_t kGpuKBase = 0x100000;
constexpr std::uintptr_t kGpuVBase = 0x200000;
constexpr std::uintptr_t kColdBase = 0x300000;
constexpr std::uintptr_t kStreamValue = 0x7000;
constexpr std::size_t kRawBytes = 320;
constexpr std::size_t kColdBytes = 192;

void resetLaunch()
{
    gLaunch = {};
}

Nvfp4ColdPageParams makeParams(float scale = 1.0F)
{
    Nvfp4ColdPageParams params;
    params.runtimeType = kernels::Nvfp4ColdPageRuntimeType::kFloat16;
    params.numKvHeads = 1;
    params.tokensPerPage = 5;
    params.headDim = 32;
    params.nvfp4ScaleOrigQuant = scale;
    params.nvfp4ScaleQuantOrig = 1.0F / scale;
    return params;
}

ColdPageLayerPlan makeAttentionPlan(int layerId, float keyScale = 1.0F, float valueScale = 2.0F)
{
    return ColdPageLayerPlan{layerId, kColdBytes, 180U, 12U,
        {ColdPageBufferPlan{"key", ColdPageTransformKind::kNvfp4, kRawBytes, 0U, 160U, makeParams(keyScale)},
            ColdPageBufferPlan{"value", ColdPageTransformKind::kNvfp4, kRawBytes, 80U, 170U, makeParams(valueScale)}}};
}

ColdPageLayerPlan makeMlaPlan(int layerId, bool hasIndex)
{
    std::vector<ColdPageBufferPlan> buffers;
    buffers.push_back(ColdPageBufferPlan{"key", ColdPageTransformKind::kNvfp4, kRawBytes, 0U, 80U, makeParams()});
    if (hasIndex)
    {
        buffers.push_back(
            ColdPageBufferPlan{"index_key", ColdPageTransformKind::kLosslessCopy, 68U, 90U, 0U, std::nullopt});
        return ColdPageLayerPlan{layerId, 160U, 158U, 2U, std::move(buffers)};
    }
    return ColdPageLayerPlan{layerId, 96U, 90U, 6U, std::move(buffers)};
}

kv::PoolGroupDesc makeAttentionDesc(kv::PoolGroupIndex poolGroupIndex = kv::PoolGroupIndex{0},
    kv::LayerGroupId lifeCycle = kv::LayerGroupId{0}, std::size_t count = 1U, int firstLayer = 0,
    std::uintptr_t keyBase = kGpuKBase, std::uintptr_t valueBase = kGpuVBase)
{
    kv::CoalescedBuffer keys{kRawBytes, {}};
    kv::CoalescedBuffer values{kRawBytes, {}};
    for (std::size_t index = 0; index < count; ++index)
    {
        auto const layerId = firstLayer + static_cast<int>(index);
        keys.bufferIds.push_back({layerId, "key"});
        values.bufferIds.push_back({layerId, "value"});
    }
    auto const slotBytes = count * kRawBytes;
    kv::SlotDescVariant variant{
        lifeCycle, kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer>{std::move(keys), std::move(values)}};
    return kv::PoolGroupDesc{poolGroupIndex, kv::SlotCount{512}, kv::SlotDesc{{std::move(variant)}},
        kv::TypedVec<kv::PoolIndex, kv::PoolDesc>{
            {kv::PoolIndex{0}, keyBase, slotBytes}, {kv::PoolIndex{1}, valueBase, slotBytes}}};
}

kv::PoolGroupDesc makeMlaDesc(bool hasIndex)
{
    kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer> buffers;
    buffers.push_back(kv::CoalescedBuffer{kRawBytes, {{0, "key"}}});
    if (hasIndex)
    {
        buffers.push_back(kv::CoalescedBuffer{68U, {{0, "index_key"}}});
    }
    kv::SlotDescVariant variant{kv::LayerGroupId{0}, std::move(buffers)};
    kv::TypedVec<kv::PoolIndex, kv::PoolDesc> pools;
    pools.push_back({kv::PoolIndex{0}, kGpuKBase, kRawBytes});
    if (hasIndex)
    {
        pools.push_back({kv::PoolIndex{1}, kGpuVBase, 68U});
    }
    return {kv::PoolGroupIndex{0}, kv::SlotCount{512}, kv::SlotDesc{{std::move(variant)}}, std::move(pools)};
}

kv::PoolGroupDesc makeLosslessDesc(kv::PoolGroupIndex poolGroupIndex, kv::LayerGroupId lifeCycle)
{
    kv::SlotDescVariant variant;
    variant.lifeCycleId = lifeCycle;
    variant.coalescedBuffers = kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer>{
        kv::CoalescedBuffer{64U, {{10, "ssm_state"}}}, kv::CoalescedBuffer{32U, {{10, "conv_state"}}}};
    return {poolGroupIndex, kv::SlotCount{8}, kv::SlotDesc{{std::move(variant)}},
        kv::TypedVec<kv::PoolIndex, kv::PoolDesc>{
            {kv::PoolIndex{0}, 0x400000, 64U}, {kv::PoolIndex{1}, 0x500000, 32U}}};
}

bool configureOne(PlannedColdPageCodec& codec, kv::PoolGroupDesc const& desc)
{
    return codec.configure(&desc, kv::PoolGroupIndex{1});
}

TEST(PlannedColdPageCodecTest, ResolvesPythonAuthoredLayoutAndScales)
{
    resetLaunch();
    PlannedColdPageCodec codec{{makeAttentionPlan(0, 2.0F, 3.0F), makeAttentionPlan(1, 4.0F, 5.0F)}};
    ASSERT_TRUE(configureOne(codec, makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{3}, 2U)));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{3}), 2U * kColdBytes);

    kv::PageIndexPair const indices[]{{2, 1}, {5, 3}};
    auto const stream = reinterpret_cast<cudaStream_t>(kStreamValue);
    ASSERT_TRUE(
        codec.encode(kv::LayerGroupId{3}, reinterpret_cast<void*>(kColdBase), indices, std::size(indices), stream));
    ASSERT_EQ(gLaunch.encodeCalls, 1);
    ASSERT_EQ(gLaunch.encodePages.size(), 2U);
    EXPECT_EQ(gLaunch.encodePages[0].gpuPageIndex, 1);
    EXPECT_EQ(gLaunch.encodePages[0].coldPageIndex, 2);
    ASSERT_EQ(gLaunch.plan.numBuffers, 4U);
    EXPECT_EQ(gLaunch.plan.buffers[0].rawBase, kGpuKBase);
    EXPECT_EQ(gLaunch.plan.buffers[1].rawBase, kGpuVBase);
    EXPECT_EQ(gLaunch.plan.buffers[2].rawBase, kGpuKBase + kRawBytes);
    EXPECT_EQ(gLaunch.plan.buffers[2].coldDataOffset, kColdBytes);
    EXPECT_EQ(gLaunch.plan.buffers[3].coldScaleOffset, kColdBytes + 170U);
    EXPECT_EQ(gLaunch.plan.buffers[3].coldPaddingOffset, kColdBytes + 180U);
    EXPECT_EQ(gLaunch.plan.buffers[3].coldPaddingBytes, 12U);
    EXPECT_FLOAT_EQ(gLaunch.plan.buffers[0].params.nvfp4ScaleOrigQuant, 2.0F);
    EXPECT_FLOAT_EQ(gLaunch.plan.buffers[3].params.nvfp4ScaleOrigQuant, 5.0F);

    ASSERT_TRUE(codec.decode(
        kv::LayerGroupId{3}, reinterpret_cast<void const*>(kColdBase), indices, std::size(indices), stream));
    ASSERT_EQ(gLaunch.decodeCalls, 1);
    EXPECT_EQ(gLaunch.decodePages[0].gpuPageIndex, 2);
    EXPECT_EQ(gLaunch.decodePages[0].coldPageIndex, 1);
}

TEST(PlannedColdPageCodecTest, PreservesExplicitMlaLosslessSideBuffer)
{
    resetLaunch();
    PlannedColdPageCodec codec{{makeMlaPlan(0, true)}};
    ASSERT_TRUE(configureOne(codec, makeMlaDesc(true)));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 160U);

    kv::PageIndexPair const indices[]{{0, 0}};
    ASSERT_TRUE(codec.encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, nullptr));
    ASSERT_EQ(gLaunch.plan.numBuffers, 2U);
    EXPECT_EQ(gLaunch.plan.buffers[0].transform, kernels::Nvfp4ColdPageTransform::kNvfp4);
    EXPECT_EQ(gLaunch.plan.buffers[1].transform, kernels::Nvfp4ColdPageTransform::kLosslessCopy);
    EXPECT_EQ(gLaunch.plan.buffers[1].rawBase, kGpuVBase);
    EXPECT_EQ(gLaunch.plan.buffers[1].rawBytes, 68U);
    EXPECT_EQ(gLaunch.plan.buffers[1].coldDataOffset, 90U);
    EXPECT_EQ(gLaunch.plan.buffers[1].coldPaddingOffset, 158U);
}

TEST(PlannedColdPageCodecTest, UnplannedLifecycleDelegatesToDefaultCodec)
{
    PlannedColdPageCodec codec{{makeAttentionPlan(0)}};
    std::array descs{makeAttentionDesc(), makeLosslessDesc(kv::PoolGroupIndex{1}, kv::LayerGroupId{1})};
    ASSERT_TRUE(codec.configure(descs.data(), kv::PoolGroupIndex{2}));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), kColdBytes);
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{1}), 96U);
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{1}), kv::LayerGroupId{1});
}

TEST(PlannedColdPageCodecTest, RejectsDuplicateLayerAndRolePlans)
{
    std::vector<ColdPageLayerPlan> duplicateLayers{makeAttentionPlan(0), makeAttentionPlan(0)};
    EXPECT_THROW({ PlannedColdPageCodec codec{duplicateLayers}; }, std::invalid_argument);

    auto plan = makeAttentionPlan(0);
    plan.buffers.push_back(plan.buffers.front());
    EXPECT_THROW({ PlannedColdPageCodec codec{{plan}}; }, std::invalid_argument);
}

TEST(PlannedColdPageCodecTest, RejectsMissingNvfp4Parameters)
{
    auto plan = makeAttentionPlan(0);
    plan.buffers.front().nvfp4Params.reset();
    PlannedColdPageCodec codec{{plan}};
    EXPECT_FALSE(configureOne(codec, makeAttentionDesc()));
}

TEST(PlannedColdPageCodecTest, RejectsRawSizeAndRoleMismatches)
{
    auto wrongSize = makeAttentionPlan(0);
    wrongSize.buffers.front().rawBytes += 1U;
    PlannedColdPageCodec sizeCodec{{wrongSize}};
    EXPECT_FALSE(configureOne(sizeCodec, makeAttentionDesc()));

    auto missingRole = makeAttentionPlan(0);
    missingRole.buffers.pop_back();
    PlannedColdPageCodec roleCodec{{missingRole}};
    EXPECT_FALSE(configureOne(roleCodec, makeAttentionDesc()));
}

TEST(PlannedColdPageCodecTest, RejectsPlannedAndUnplannedLayersInOneLifecycle)
{
    PlannedColdPageCodec codec{{makeAttentionPlan(0)}};
    EXPECT_FALSE(configureOne(codec, makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 2U)));
}

TEST(PlannedColdPageCodecTest, RejectsPlanAbsentFromGpuDescriptors)
{
    PlannedColdPageCodec codec{{makeAttentionPlan(0), makeAttentionPlan(1)}};
    EXPECT_FALSE(configureOne(codec, makeAttentionDesc()));
}

TEST(PlannedColdPageCodecTest, RejectsDuplicateLifecycleAcrossPoolGroups)
{
    PlannedColdPageCodec codec{{makeAttentionPlan(0), makeAttentionPlan(1)}};
    std::array descs{
        makeAttentionDesc(), makeAttentionDesc(kv::PoolGroupIndex{1}, kv::LayerGroupId{0}, 1U, 1, 0x600000, 0x700000)};
    EXPECT_FALSE(codec.configure(descs.data(), kv::PoolGroupIndex{2}));
}

TEST(PlannedColdPageCodecTest, RejectsInvalidLayerRelativeIntervals)
{
    auto plan = makeAttentionPlan(0);
    plan.buffers.back().coldDataOffset = plan.coldPageBytes;
    PlannedColdPageCodec codec{{plan}};
    EXPECT_FALSE(configureOne(codec, makeAttentionDesc()));
}

TEST(PlannedColdPageCodecTest, EmptyBatchIsValidAndInvalidBatchesFailBeforeLaunch)
{
    resetLaunch();
    PlannedColdPageCodec codec{{makeAttentionPlan(0)}};
    ASSERT_TRUE(configureOne(codec, makeAttentionDesc()));
    EXPECT_TRUE(codec.encode(kv::LayerGroupId{0}, nullptr, nullptr, 0U, nullptr));
    EXPECT_TRUE(codec.decode(kv::LayerGroupId{0}, nullptr, nullptr, 0U, nullptr));

    kv::PageIndexPair const valid[]{{0, 0}};
    EXPECT_FALSE(codec.encode(kv::LayerGroupId{0}, nullptr, valid, 1U, nullptr));
    EXPECT_FALSE(codec.decode(kv::LayerGroupId{0}, nullptr, valid, 1U, nullptr));
    EXPECT_EQ(gLaunch.encodeCalls, 0);
    EXPECT_EQ(gLaunch.decodeCalls, 0);
}

TEST(PlannedColdPageCodecTest, UnknownLifecycleUsesFailureSentinels)
{
    PlannedColdPageCodec codec{{makeAttentionPlan(0)}};
    ASSERT_TRUE(configureOne(codec, makeAttentionDesc()));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{99}), 0U);
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{99}), kv::LayerGroupId{-1});
    EXPECT_EQ(codec.queryPageIndexLocation(kv::LayerGroupId{99}), kv::PageIndexLocation::kBadLocation);
}

} // namespace
} // namespace tensorrt_llm::kv_cache_compression

namespace tensorrt_llm::kernels
{
namespace
{

struct Interval
{
    std::size_t begin;
    std::size_t end;
};

void appendInterval(std::vector<Interval>& intervals, std::size_t begin, std::size_t bytes, std::size_t pageBytes)
{
    if (bytes == 0U)
    {
        return;
    }
    if (begin > pageBytes || bytes > pageBytes - begin)
    {
        throw std::invalid_argument("test interval exceeds cold Page");
    }
    intervals.push_back({begin, begin + bytes});
}

} // namespace

Nvfp4ColdPagePreparedPlan prepareNvfp4ColdPagePlan(std::vector<Nvfp4ColdPageBufferPlan> const& buffers,
    std::size_t coldPageBytes, Nvfp4ColdPageRuntimeType runtimeType)
{
    if (buffers.empty() || buffers.size() > kNvfp4ColdPageMaxBuffersPerLaunch || coldPageBytes == 0U)
    {
        throw std::invalid_argument("invalid test launch plan");
    }
    std::vector<Interval> intervals;
    for (auto const& buffer : buffers)
    {
        if (buffer.rawBytes == 0U || buffer.rawBytes > buffer.rawSlotBytes)
        {
            throw std::invalid_argument("invalid raw buffer");
        }
        if (buffer.transform == Nvfp4ColdPageTransform::kNvfp4)
        {
            auto const& params = buffer.params;
            if (params.numKvHeads <= 0 || params.tokensPerPage <= 0 || params.headDim <= 0 || params.headDim % 16 != 0)
            {
                throw std::invalid_argument("invalid NVFP4 geometry");
            }
            std::size_t const elements = static_cast<std::size_t>(params.numKvHeads)
                * static_cast<std::size_t>(params.tokensPerPage) * static_cast<std::size_t>(params.headDim);
            auto const elementBytes = runtimeType == Nvfp4ColdPageRuntimeType::kFp8E4m3 ? 1U : 2U;
            if (buffer.rawBytes != elements * elementBytes)
            {
                throw std::invalid_argument("raw size mismatch");
            }
            appendInterval(intervals, buffer.coldDataOffset, elements / 2U, coldPageBytes);
            appendInterval(intervals, buffer.coldScaleOffset, elements / 16U, coldPageBytes);
        }
        else
        {
            appendInterval(intervals, buffer.coldDataOffset, buffer.rawBytes, coldPageBytes);
        }
        appendInterval(intervals, buffer.coldPaddingOffset, buffer.coldPaddingBytes, coldPageBytes);
    }
    std::sort(
        intervals.begin(), intervals.end(), [](auto const& lhs, auto const& rhs) { return lhs.begin < rhs.begin; });
    for (std::size_t index = 1; index < intervals.size(); ++index)
    {
        if (intervals[index - 1U].end > intervals[index].begin)
        {
            throw std::invalid_argument("test intervals overlap");
        }
    }

    Nvfp4ColdPagePreparedPlan plan;
    std::copy(buffers.begin(), buffers.end(), plan.buffers.begin());
    plan.numBuffers = static_cast<std::uint32_t>(buffers.size());
    plan.coldPageBytes = coldPageBytes;
    plan.runtimeType = runtimeType;
    return plan;
}

void invokeNvfp4ColdPageEncode(std::vector<Nvfp4ColdPageOffloadPageTask> const& pages,
    Nvfp4ColdPagePreparedPlan const& plan, void* coldBase, cudaStream_t stream)
{
    auto& launch = kv_cache_compression::gLaunch;
    ++launch.encodeCalls;
    launch.encodePages = pages;
    launch.plan = plan;
    launch.coldBase = coldBase;
    launch.stream = stream;
}

void invokeNvfp4ColdPageDecode(std::vector<Nvfp4ColdPageOnboardPageTask> const& pages,
    Nvfp4ColdPagePreparedPlan const& plan, void const* coldBase, cudaStream_t stream)
{
    auto& launch = kv_cache_compression::gLaunch;
    ++launch.decodeCalls;
    launch.decodePages = pages;
    launch.plan = plan;
    launch.coldBase = coldBase;
    launch.stream = stream;
}

} // namespace tensorrt_llm::kernels

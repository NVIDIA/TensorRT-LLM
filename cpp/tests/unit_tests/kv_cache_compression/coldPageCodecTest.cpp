/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kv_cache_compression/nativeColdPageCodec.h"
#include "tensorrt_llm/kv_cache_compression/nvfp4ColdPageCodec.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorrt_llm::kv_cache_compression
{
namespace
{

namespace kv = batch_manager::kv_cache_manager_v2;

static_assert(std::is_abstract_v<NativeColdPageCodec>);
static_assert(std::is_base_of_v<kv::IKvCacheColdPageCodec, NativeColdPageCodec>);

constexpr std::uintptr_t kGpuKBase = 0x100000;
constexpr std::uintptr_t kGpuVBase = 0x200000;
constexpr std::uintptr_t kColdBase = 0x300000;
constexpr std::uintptr_t kStreamValue = 0x7000;
constexpr std::size_t kRawBytes = 320;
constexpr std::size_t kColdBytes = 192;

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

kv::PoolGroupDesc makeMlaDesc()
{
    kv::SlotDescVariant variant{kv::LayerGroupId{0},
        kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer>{
            kv::CoalescedBuffer{kRawBytes, {{0, "key"}}}, kv::CoalescedBuffer{68U, {{0, "index_key"}}}}};
    return {kv::PoolGroupIndex{0}, kv::SlotCount{512}, kv::SlotDesc{{std::move(variant)}},
        kv::TypedVec<kv::PoolIndex, kv::PoolDesc>{
            {kv::PoolIndex{0}, kGpuKBase, kRawBytes}, {kv::PoolIndex{1}, kGpuVBase, 68U}}};
}

kv::PoolGroupDesc makeLosslessDesc(kv::PoolGroupIndex poolGroupIndex, kv::LayerGroupId lifeCycle)
{
    kv::SlotDescVariant variant{lifeCycle,
        kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer>{
            kv::CoalescedBuffer{64U, {{10, "ssm_state"}}}, kv::CoalescedBuffer{32U, {{10, "conv_state"}}}}};
    return {poolGroupIndex, kv::SlotCount{8}, kv::SlotDesc{{std::move(variant)}},
        kv::TypedVec<kv::PoolIndex, kv::PoolDesc>{
            {kv::PoolIndex{0}, 0x400000, 64U}, {kv::PoolIndex{1}, 0x500000, 32U}}};
}

bool configureOne(kv::IKvCacheColdPageCodec& codec, kv::PoolGroupDesc const& desc)
{
    return codec.configure(&desc, kv::PoolGroupIndex{1});
}

class RecordingCodec final : public NativeColdPageCodec
{
public:
    explicit RecordingCodec(std::set<kv::LayerId> layerIds)
        : mLayerIds(std::move(layerIds))
    {
    }

    [[nodiscard]] std::set<kv::LayerId> const& getLayerIds() const noexcept override
    {
        return mLayerIds;
    }

    std::vector<ColdPageLifecycleProperties> configureAlgorithm(
        std::vector<ResolvedHotLifecycle> const& lifecycles) override
    {
        resolved = lifecycles;
        if (failConfigure)
        {
            throw std::runtime_error("requested configure failure");
        }
        return std::vector<ColdPageLifecycleProperties>(
            lifecycles.size(), ColdPageLifecycleProperties{777U, kv::PageIndexLocation::kHost});
    }

    void encodeAlgorithm(std::size_t planIndex, void* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream) override
    {
        if (failBatches)
        {
            throw std::runtime_error("requested batch failure");
        }
        ++encodeCalls;
        lastPlanIndex = planIndex;
        lastColdBase = coldBase;
        lastIndices.assign(pageIndices, pageIndices + numPages);
        lastStream = stream;
    }

    void decodeAlgorithm(std::size_t planIndex, void const* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream) override
    {
        if (failBatches)
        {
            throw std::runtime_error("requested batch failure");
        }
        ++decodeCalls;
        lastPlanIndex = planIndex;
        lastColdBase = coldBase;
        lastIndices.assign(pageIndices, pageIndices + numPages);
        lastStream = stream;
    }

    bool failConfigure = false;
    bool failBatches = false;
    int encodeCalls = 0;
    int decodeCalls = 0;
    std::size_t lastPlanIndex = 0;
    void const* lastColdBase = nullptr;
    cudaStream_t lastStream{};
    std::vector<kv::PageIndexPair> lastIndices;
    std::vector<ResolvedHotLifecycle> resolved;

private:
    std::set<kv::LayerId> mLayerIds;
};

TEST(NativeColdPageCodecTest, ResolvesKvcManagerLayoutAndForwardsBatches)
{
    RecordingCodec codec{{0, 1}};

    ASSERT_TRUE(configureOne(codec, makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{3}, 2U)));
    ASSERT_EQ(codec.resolved.size(), 1U);
    EXPECT_EQ(codec.resolved.front().lifeCycleId, kv::LifeCycleId{3});
    auto const& layers = codec.resolved.front().layers;
    EXPECT_EQ(layers.at(0).at("key").rawBase, kGpuKBase);
    EXPECT_EQ(layers.at(0).at("value").rawBase, kGpuVBase);
    EXPECT_EQ(layers.at(1).at("key").rawBase, kGpuKBase + kRawBytes);
    EXPECT_EQ(layers.at(1).at("key").rawSlotBytes, 2U * kRawBytes);
    EXPECT_EQ(layers.at(1).at("key").rawBytes, kRawBytes);
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{3}), 777U);

    kv::PageIndexPair const indices[]{{2, 1}, {5, 3}};
    auto const stream = reinterpret_cast<cudaStream_t>(kStreamValue);
    ASSERT_TRUE(
        codec.encode(kv::LayerGroupId{3}, reinterpret_cast<void*>(kColdBase), indices, std::size(indices), stream));
    EXPECT_EQ(codec.encodeCalls, 1);
    EXPECT_EQ(codec.lastPlanIndex, 0U);
    EXPECT_EQ(codec.lastIndices[1].src, 3);
    EXPECT_EQ(codec.lastColdBase, reinterpret_cast<void*>(kColdBase));
    EXPECT_EQ(codec.lastStream, stream);

    ASSERT_TRUE(codec.decode(
        kv::LayerGroupId{3}, reinterpret_cast<void const*>(kColdBase), indices, std::size(indices), stream));
    EXPECT_EQ(codec.decodeCalls, 1);
}

TEST(NativeColdPageCodecTest, UnownedLifecycleUsesLosslessFallback)
{
    RecordingCodec codec{{0}};
    std::array descs{makeAttentionDesc(), makeLosslessDesc(kv::PoolGroupIndex{1}, kv::LayerGroupId{1})};

    ASSERT_TRUE(codec.configure(descs.data(), kv::PoolGroupIndex{2}));
    EXPECT_EQ(codec.resolved.size(), 1U);
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 777U);
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{1}), 96U);
    EXPECT_EQ(codec.queryPageIndexLocation(kv::LayerGroupId{1}), kv::PageIndexLocation::kHost);
}

TEST(NativeColdPageCodecTest, RejectsMixedMissingAndDuplicateLifecycleMappings)
{
    {
        RecordingCodec codec{{0}};
        EXPECT_FALSE(configureOne(codec, makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 2U)));
    }
    {
        RecordingCodec codec{{0, 1}};
        EXPECT_FALSE(configureOne(codec, makeAttentionDesc()));
    }
    {
        RecordingCodec codec{{0, 1}};
        std::array descs{makeAttentionDesc(),
            makeAttentionDesc(kv::PoolGroupIndex{1}, kv::LayerGroupId{0}, 1U, 1, 0x600000, 0x700000)};
        EXPECT_FALSE(codec.configure(descs.data(), kv::PoolGroupIndex{2}));
    }
}

TEST(NativeColdPageCodecTest, CatchesAlgorithmConfigureAndBatchFailures)
{
    RecordingCodec codec{{0}};
    codec.failConfigure = true;
    EXPECT_FALSE(configureOne(codec, makeAttentionDesc()));

    RecordingCodec validCodec{{0}};
    ASSERT_TRUE(configureOne(validCodec, makeAttentionDesc()));
    EXPECT_TRUE(validCodec.encode(kv::LayerGroupId{0}, nullptr, nullptr, 0U, nullptr));
    EXPECT_TRUE(validCodec.decode(kv::LayerGroupId{0}, nullptr, nullptr, 0U, nullptr));
    kv::PageIndexPair const indices[]{{0, 0}};
    EXPECT_FALSE(validCodec.encode(kv::LayerGroupId{0}, nullptr, indices, 1U, nullptr));
    EXPECT_FALSE(validCodec.decode(kv::LayerGroupId{0}, nullptr, indices, 1U, nullptr));
    validCodec.failBatches = true;
    EXPECT_FALSE(validCodec.encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, nullptr));
    EXPECT_FALSE(
        validCodec.decode(kv::LayerGroupId{0}, reinterpret_cast<void const*>(kColdBase), indices, 1U, nullptr));
    EXPECT_EQ(validCodec.encodeCalls, 0);
    EXPECT_EQ(validCodec.decodeCalls, 0);
}

TEST(NativeColdPageCodecTest, UnknownLifecycleUsesFailureSentinels)
{
    RecordingCodec codec{{0}};
    ASSERT_TRUE(configureOne(codec, makeAttentionDesc()));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{99}), 0U);
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{99}), kv::LayerGroupId{-1});
    EXPECT_EQ(codec.queryPageIndexLocation(kv::LayerGroupId{99}), kv::PageIndexLocation::kBadLocation);
}

Nvfp4ColdPageScales makeScales(float scale)
{
    Nvfp4ColdPageScales scales;
    scales.nvfp4ScaleOrigQuant = scale;
    scales.nvfp4ScaleQuantOrig = 1.0F / scale;
    return scales;
}

Nvfp4ColdPageLayerLayout makeAttentionLayout(int layerId, float keyScale = 1.0F, float valueScale = 2.0F)
{
    return Nvfp4ColdPageLayerLayout{layerId, kernels::Nvfp4ColdPageRuntimeType::kFloat16, 1, 5, 32, kColdBytes, 180U,
        {Nvfp4ColdPageBufferLayout{"key", 0U, 160U, makeScales(keyScale)},
            Nvfp4ColdPageBufferLayout{"value", 80U, 170U, makeScales(valueScale)}}};
}

Nvfp4ColdPageLayerLayout makeMlaLayout()
{
    return Nvfp4ColdPageLayerLayout{0, kernels::Nvfp4ColdPageRuntimeType::kFloat16, 1, 5, 32, 160U, 158U,
        {Nvfp4ColdPageBufferLayout{"key", 0U, 80U, makeScales(1.0F)},
            Nvfp4ColdPageBufferLayout{"index_key", 90U, 0U, std::nullopt}}};
}

struct RecordedLaunch
{
    int prepareCalls = 0;
    int encodeCalls = 0;
    int decodeCalls = 0;
    std::vector<kernels::Nvfp4ColdPageOffloadPageTask> encodePages;
    std::vector<kernels::Nvfp4ColdPageOnboardPageTask> decodePages;
    kernels::Nvfp4ColdPagePreparedPlan plan;
};

RecordedLaunch gLaunch;

TEST(Nvfp4ColdPageCodecTest, LowersMhaLayoutOnceAndDispatchesEncodeDecode)
{
    gLaunch = {};
    auto codec = createNvfp4ColdPageCodec({makeAttentionLayout(0, 2.0F, 3.0F), makeAttentionLayout(1, 4.0F, 5.0F)});
    ASSERT_TRUE(configureOne(*codec, makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{3}, 2U)));
    EXPECT_EQ(gLaunch.prepareCalls, 1);
    EXPECT_EQ(codec->queryColdPageBytes(kv::LayerGroupId{3}), 2U * kColdBytes);

    kv::PageIndexPair const indices[]{{2, 1}, {5, 3}};
    auto const stream = reinterpret_cast<cudaStream_t>(kStreamValue);
    ASSERT_TRUE(
        codec->encode(kv::LayerGroupId{3}, reinterpret_cast<void*>(kColdBase), indices, std::size(indices), stream));
    ASSERT_EQ(gLaunch.plan.numBuffers, 4U);
    EXPECT_EQ(gLaunch.plan.buffers[0].rawBase, kGpuKBase);
    EXPECT_EQ(gLaunch.plan.buffers[2].rawBase, kGpuKBase + kRawBytes);
    EXPECT_EQ(gLaunch.plan.buffers[2].coldDataOffset, kColdBytes);
    EXPECT_EQ(gLaunch.plan.buffers[3].coldScaleOffset, kColdBytes + 170U);
    EXPECT_EQ(gLaunch.plan.buffers[3].coldPaddingOffset, kColdBytes + 180U);
    EXPECT_EQ(gLaunch.plan.buffers[3].coldPaddingBytes, 12U);
    EXPECT_FLOAT_EQ(gLaunch.plan.buffers[0].params.nvfp4ScaleOrigQuant, 2.0F);
    EXPECT_FLOAT_EQ(gLaunch.plan.buffers[3].params.nvfp4ScaleOrigQuant, 5.0F);
    ASSERT_EQ(gLaunch.encodePages.size(), 2U);
    EXPECT_EQ(gLaunch.encodePages[0].gpuPageIndex, 1);
    EXPECT_EQ(gLaunch.encodePages[0].coldPageIndex, 2);

    ASSERT_TRUE(codec->decode(
        kv::LayerGroupId{3}, reinterpret_cast<void const*>(kColdBase), indices, std::size(indices), stream));
    ASSERT_EQ(gLaunch.decodePages.size(), 2U);
    EXPECT_EQ(gLaunch.decodePages[0].gpuPageIndex, 2);
    EXPECT_EQ(gLaunch.decodePages[0].coldPageIndex, 1);
}

TEST(Nvfp4ColdPageCodecTest, PreservesMlaSideBufferLosslessly)
{
    gLaunch = {};
    auto codec = createNvfp4ColdPageCodec({makeMlaLayout()});
    ASSERT_TRUE(configureOne(*codec, makeMlaDesc()));

    kv::PageIndexPair const indices[]{{0, 0}};
    ASSERT_TRUE(codec->encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, nullptr));
    ASSERT_EQ(gLaunch.plan.numBuffers, 2U);
    EXPECT_EQ(gLaunch.plan.buffers[0].transform, kernels::Nvfp4ColdPageTransform::kNvfp4);
    EXPECT_EQ(gLaunch.plan.buffers[1].transform, kernels::Nvfp4ColdPageTransform::kLosslessCopy);
    EXPECT_EQ(gLaunch.plan.buffers[1].rawBase, kGpuVBase);
    EXPECT_EQ(gLaunch.plan.buffers[1].rawBytes, 68U);
    EXPECT_EQ(gLaunch.plan.buffers[1].coldDataOffset, 90U);
    EXPECT_EQ(gLaunch.plan.buffers[1].coldPaddingOffset, 158U);
    EXPECT_EQ(gLaunch.plan.buffers[1].coldPaddingBytes, 2U);
}

TEST(Nvfp4ColdPageCodecTest, RejectsDuplicateLayoutsAndMissingRoles)
{
    EXPECT_THROW(
        {
            auto codec = createNvfp4ColdPageCodec({makeAttentionLayout(0), makeAttentionLayout(0)});
        },
        std::invalid_argument);

    auto duplicateRole = makeAttentionLayout(0);
    duplicateRole.buffers.push_back(duplicateRole.buffers.front());
    EXPECT_THROW({ auto codec = createNvfp4ColdPageCodec({duplicateRole}); }, std::invalid_argument);

    auto missingRole = makeAttentionLayout(0);
    missingRole.buffers.pop_back();
    auto codec = createNvfp4ColdPageCodec({missingRole});
    EXPECT_FALSE(configureOne(*codec, makeAttentionDesc()));
}

} // namespace
} // namespace tensorrt_llm::kv_cache_compression

namespace tensorrt_llm::kernels
{

Nvfp4ColdPagePreparedPlan prepareNvfp4ColdPagePlan(std::vector<Nvfp4ColdPageBufferPlan> const& buffers,
    std::size_t coldPageBytes, Nvfp4ColdPageRuntimeType runtimeType)
{
    auto& launch = kv_cache_compression::gLaunch;
    ++launch.prepareCalls;
    if (buffers.empty() || buffers.size() > kNvfp4ColdPageMaxBuffersPerLaunch)
    {
        throw std::invalid_argument("invalid test launch plan");
    }
    Nvfp4ColdPagePreparedPlan plan;
    std::copy(buffers.begin(), buffers.end(), plan.buffers.begin());
    plan.numBuffers = static_cast<std::uint32_t>(buffers.size());
    plan.coldPageBytes = coldPageBytes;
    plan.runtimeType = runtimeType;
    return plan;
}

void invokeNvfp4ColdPageEncode(
    std::vector<Nvfp4ColdPageOffloadPageTask> const& pages, Nvfp4ColdPagePreparedPlan const& plan, void*, cudaStream_t)
{
    auto& launch = kv_cache_compression::gLaunch;
    ++launch.encodeCalls;
    launch.encodePages = pages;
    launch.plan = plan;
}

void invokeNvfp4ColdPageDecode(std::vector<Nvfp4ColdPageOnboardPageTask> const& pages,
    Nvfp4ColdPagePreparedPlan const& plan, void const*, cudaStream_t)
{
    auto& launch = kv_cache_compression::gLaunch;
    ++launch.decodeCalls;
    launch.decodePages = pages;
    launch.plan = plan;
}

} // namespace tensorrt_llm::kernels

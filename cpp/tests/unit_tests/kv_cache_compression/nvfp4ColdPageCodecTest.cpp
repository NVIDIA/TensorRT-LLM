/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kv_cache_compression/nvfp4ColdPageCodec.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tensorrt_llm::kv_cache_compression
{
namespace
{

namespace kv = batch_manager::kv_cache_manager_v2;

static_assert(std::is_base_of_v<kv::IKvCacheColdPageCodec, Nvfp4ColdPageCodec>);

struct RecordedLaunch
{
    int offloadCalls = 0;
    int onboardCalls = 0;
    std::vector<kernels::Nvfp4BoundaryOffloadPageTask> offloadPages;
    std::vector<kernels::Nvfp4BoundaryOnboardPageTask> onboardPages;
    kernels::Nvfp4BoundaryPreparedPlan plan;
    void const* coldBase = nullptr;
    cudaStream_t stream{};
};

RecordedLaunch gLaunch;

constexpr std::uintptr_t kGpuKBase = 0x100000;
constexpr std::uintptr_t kGpuVBase = 0x200000;
constexpr std::uintptr_t kColdBase = 0x300000;
constexpr std::size_t kLayerRawBytes = 320;
constexpr std::size_t kLayerColdBytesAligned = 192;
constexpr std::size_t kNumAttentionLayers = 8;
constexpr std::size_t kGpuSlotBytes = kNumAttentionLayers * kLayerRawBytes;
constexpr std::size_t kColdSlotBytes = kNumAttentionLayers * kLayerColdBytesAligned;
constexpr std::size_t kMlaRawBytes = 64U * 576U * 2U;
constexpr std::size_t kMlaColdBytes = 64U * 576U / 2U + 64U * 576U / 16U;
constexpr std::size_t kDsaIndexKeyBytes = 64U * (128U + 4U);
constexpr std::uintptr_t kStreamValue = 0x7000;

void resetLaunch()
{
    gLaunch = {};
}

std::vector<Nvfp4ColdPageLayerConfig> makeLayers(std::size_t count = kNumAttentionLayers, int firstLayer = 0)
{
    std::vector<Nvfp4ColdPageLayerConfig> layers;
    layers.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        auto const scale = static_cast<float>(index + 2U);
        layers.push_back({firstLayer + static_cast<int>(index), kernels::Nvfp4BoundaryRuntimeType::kFloat16, 1, 5, 32,
            {scale, scale + 0.5F}, {1.0F / scale, 1.0F / (scale + 0.5F)}});
    }
    return layers;
}

std::vector<Nvfp4ColdPageLayerConfig> makeMlaLayers(std::size_t count)
{
    std::vector<Nvfp4ColdPageLayerConfig> layers;
    layers.reserve(count);
    for (std::size_t layer = 0; layer < count; ++layer)
    {
        layers.push_back({static_cast<int>(layer), kernels::Nvfp4BoundaryRuntimeType::kFloat16, 1, 64, 576,
            {1.0F, 1.0F}, {1.0F, 1.0F}});
    }
    return layers;
}

kv::PoolGroupDesc makeAttentionDesc(kv::PoolGroupIndex poolGroupIndex = kv::PoolGroupIndex{0},
    kv::LayerGroupId lifeCycle = kv::LayerGroupId{3}, std::size_t count = kNumAttentionLayers, int firstLayer = 0,
    std::uintptr_t keyBase = kGpuKBase, std::uintptr_t valueBase = kGpuVBase)
{
    kv::CoalescedBuffer keys{kLayerRawBytes, {}};
    kv::CoalescedBuffer values{kLayerRawBytes, {}};
    for (std::size_t index = 0; index < count; ++index)
    {
        auto const layerId = firstLayer + static_cast<int>(index);
        keys.bufferIds.push_back({layerId, "key"});
        values.bufferIds.push_back({layerId, "value"});
    }

    kv::SlotDescVariant variant{
        lifeCycle, kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer>{std::move(keys), std::move(values)}};
    auto const slotBytes = count * kLayerRawBytes;
    return kv::PoolGroupDesc{poolGroupIndex, kv::SlotCount{512}, kv::SlotDesc{{std::move(variant)}},
        kv::TypedVec<kv::PoolIndex, kv::PoolDesc>{
            kv::PoolDesc{kv::PoolIndex{0}, keyBase, slotBytes}, kv::PoolDesc{kv::PoolIndex{1}, valueBase, slotBytes}}};
}

kv::PoolGroupDesc makeMlaDesc(std::vector<bool> const& ownsIndexer, kv::LayerGroupId lifeCycle = kv::LayerGroupId{0},
    int firstLayer = 0, std::size_t keyBytes = kLayerRawBytes, std::size_t indexBytes = 68U,
    std::uintptr_t keyBase = kGpuKBase, std::uintptr_t indexBase = kGpuVBase)
{
    kv::CoalescedBuffer keys{keyBytes, {}};
    kv::CoalescedBuffer indexes{indexBytes, {}};
    for (std::size_t index = 0; index < ownsIndexer.size(); ++index)
    {
        auto const layerId = firstLayer + static_cast<int>(index);
        keys.bufferIds.push_back({layerId, "key"});
        if (ownsIndexer[index])
        {
            indexes.bufferIds.push_back({layerId, "index_key"});
        }
    }

    kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer> buffers;
    buffers.push_back(std::move(keys));
    if (!indexes.bufferIds.empty())
    {
        buffers.push_back(std::move(indexes));
    }
    kv::SlotDescVariant variant{lifeCycle, std::move(buffers)};

    kv::TypedVec<kv::PoolIndex, kv::PoolDesc> pools;
    pools.push_back(kv::PoolDesc{kv::PoolIndex{0}, keyBase, ownsIndexer.size() * keyBytes});
    auto const indexCount = static_cast<std::size_t>(std::count(ownsIndexer.begin(), ownsIndexer.end(), true));
    if (indexCount != 0U)
    {
        pools.push_back(kv::PoolDesc{kv::PoolIndex{1}, indexBase, indexCount * indexBytes});
    }
    return kv::PoolGroupDesc{
        kv::PoolGroupIndex{0}, kv::SlotCount{512}, kv::SlotDesc{{std::move(variant)}}, std::move(pools)};
}

bool configureOne(Nvfp4ColdPageCodec& codec, kv::PoolGroupDesc const& desc)
{
    return codec.configure(&desc, kv::PoolGroupIndex{1});
}

std::unique_ptr<Nvfp4ColdPageCodec> makeConfiguredAttentionCodec(
    std::size_t count = kNumAttentionLayers, kv::LayerGroupId lifeCycle = kv::LayerGroupId{3})
{
    auto codec = std::make_unique<Nvfp4ColdPageCodec>(makeLayers(count));
    EXPECT_TRUE(configureOne(*codec, makeAttentionDesc(kv::PoolGroupIndex{0}, lifeCycle, count)));
    return codec;
}

TEST(Nvfp4ColdPageCodecTest, OneCompletePageTaskCoversAllLayersWithDistinctScales)
{
    resetLaunch();
    auto codec = makeConfiguredAttentionCodec();
    EXPECT_EQ(codec->queryColdPageBytes(kv::LayerGroupId{3}), kColdSlotBytes);
    EXPECT_EQ(codec->queryPageIndexLocation(kv::LayerGroupId{3}), kv::PageIndexLocation::kHost);

    kv::PageIndexPair const indices[]{{2, 1}, {5, 3}};
    auto const stream = reinterpret_cast<cudaStream_t>(kStreamValue);
    ASSERT_TRUE(
        codec->encode(kv::LayerGroupId{3}, reinterpret_cast<void*>(kColdBase), indices, std::size(indices), stream));

    EXPECT_EQ(gLaunch.offloadCalls, 1);
    ASSERT_EQ(gLaunch.offloadPages.size(), 2U);
    EXPECT_EQ(gLaunch.offloadPages[0].gpuPageIndex, 1);
    EXPECT_EQ(gLaunch.offloadPages[0].coldPageIndex, 2);
    EXPECT_EQ(gLaunch.offloadPages[1].gpuPageIndex, 3);
    EXPECT_EQ(gLaunch.offloadPages[1].coldPageIndex, 5);
    ASSERT_EQ(gLaunch.plan.numBuffers, 2U * kNumAttentionLayers);
    for (std::size_t layer = 0; layer < kNumAttentionLayers; ++layer)
    {
        auto const& key = gLaunch.plan.buffers[2U * layer];
        auto const& value = gLaunch.plan.buffers[2U * layer + 1U];
        auto const layerOffset = layer * kLayerColdBytesAligned;
        EXPECT_EQ(key.rawBase, kGpuKBase + layer * kLayerRawBytes);
        EXPECT_EQ(value.rawBase, kGpuVBase + layer * kLayerRawBytes);
        EXPECT_EQ(key.rawSlotBytes, kGpuSlotBytes);
        EXPECT_EQ(value.rawSlotBytes, kGpuSlotBytes);
        EXPECT_EQ(key.rawBytes, kLayerRawBytes);
        EXPECT_EQ(value.rawBytes, kLayerRawBytes);
        EXPECT_EQ(key.coldDataOffset, layerOffset);
        EXPECT_EQ(value.coldDataOffset, layerOffset + 80U);
        EXPECT_EQ(key.coldScaleOffset, layerOffset + 160U);
        EXPECT_EQ(value.coldScaleOffset, layerOffset + 170U);
        EXPECT_EQ(value.coldPaddingOffset, layerOffset + 180U);
        EXPECT_EQ(value.coldPaddingBytes, 12U);
        EXPECT_EQ(key.transform, kernels::Nvfp4BoundaryTransform::kNvfp4);
        EXPECT_EQ(value.transform, kernels::Nvfp4BoundaryTransform::kNvfp4);
        EXPECT_EQ(key.params.tokensPerPage, 5);
        EXPECT_EQ(key.params.headDim, 32);
        EXPECT_FLOAT_EQ(key.params.nvfp4ScaleOrigQuant, static_cast<float>(layer + 2U));
        EXPECT_FLOAT_EQ(value.params.nvfp4ScaleOrigQuant, static_cast<float>(layer + 2U) + 0.5F);
    }
    EXPECT_EQ(gLaunch.coldBase, reinterpret_cast<void*>(kColdBase));
    EXPECT_EQ(gLaunch.plan.coldPageBytes, kColdSlotBytes);
    EXPECT_EQ(gLaunch.stream, stream);

    ASSERT_TRUE(codec->decode(
        kv::LayerGroupId{3}, reinterpret_cast<void const*>(kColdBase), indices, std::size(indices), stream));
    EXPECT_EQ(gLaunch.onboardCalls, 1);
    ASSERT_EQ(gLaunch.onboardPages.size(), 2U);
    EXPECT_EQ(gLaunch.onboardPages[0].gpuPageIndex, 2);
    EXPECT_EQ(gLaunch.onboardPages[0].coldPageIndex, 1);
}

TEST(Nvfp4ColdPageCodecTest, KeyOnlyMlaUsesLatentPackedThenScaleLayout)
{
    resetLaunch();
    Nvfp4ColdPageCodec codec{makeLayers(1)};
    ASSERT_TRUE(configureOne(codec, makeMlaDesc({false})));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 96U);

    kv::PageIndexPair const indices[]{{0, 0}};
    ASSERT_TRUE(codec.encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, nullptr));
    ASSERT_EQ(gLaunch.plan.numBuffers, 1U);
    auto const& latent = gLaunch.plan.buffers[0];
    EXPECT_EQ(latent.rawBase, kGpuKBase);
    EXPECT_EQ(latent.rawSlotBytes, kLayerRawBytes);
    EXPECT_EQ(latent.rawBytes, kLayerRawBytes);
    EXPECT_EQ(latent.coldDataOffset, 0U);
    EXPECT_EQ(latent.coldScaleOffset, 80U);
    EXPECT_EQ(latent.coldPaddingOffset, 90U);
    EXPECT_EQ(latent.coldPaddingBytes, 6U);
    EXPECT_EQ(latent.transform, kernels::Nvfp4BoundaryTransform::kNvfp4);
}

TEST(Nvfp4ColdPageCodecTest, KeyAndIndexAppendsLosslessIndexWithinTheLayerRecord)
{
    resetLaunch();
    Nvfp4ColdPageCodec codec{makeLayers(1)};
    ASSERT_TRUE(configureOne(codec, makeMlaDesc({true})));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 160U);

    kv::PageIndexPair const indices[]{{0, 0}};
    ASSERT_TRUE(codec.encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, nullptr));
    ASSERT_EQ(gLaunch.plan.numBuffers, 2U);
    auto const& latent = gLaunch.plan.buffers[0];
    auto const& index = gLaunch.plan.buffers[1];
    EXPECT_EQ(latent.coldDataOffset, 0U);
    EXPECT_EQ(latent.coldScaleOffset, 80U);
    EXPECT_EQ(index.rawBase, kGpuVBase);
    EXPECT_EQ(index.rawSlotBytes, 68U);
    EXPECT_EQ(index.rawBytes, 68U);
    EXPECT_EQ(index.coldDataOffset, 90U);
    EXPECT_EQ(index.coldPaddingOffset, 158U);
    EXPECT_EQ(index.coldPaddingBytes, 2U);
    EXPECT_EQ(index.transform, kernels::Nvfp4BoundaryTransform::kLossless);
}

TEST(Nvfp4ColdPageCodecTest, FullAndSharedIndexerLayersHaveDistinctPerLayerRecords)
{
    resetLaunch();
    Nvfp4ColdPageCodec codec{makeLayers(3)};
    ASSERT_TRUE(configureOne(codec, makeMlaDesc({true, false, true})));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 416U);

    kv::PageIndexPair const indices[]{{0, 0}};
    ASSERT_TRUE(codec.encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, nullptr));
    ASSERT_EQ(gLaunch.plan.numBuffers, 5U);
    EXPECT_EQ(gLaunch.plan.buffers[0].rawBase, kGpuKBase);
    EXPECT_EQ(gLaunch.plan.buffers[1].rawBase, kGpuVBase);
    EXPECT_EQ(gLaunch.plan.buffers[2].rawBase, kGpuKBase + kLayerRawBytes);
    EXPECT_EQ(gLaunch.plan.buffers[3].rawBase, kGpuKBase + 2U * kLayerRawBytes);
    EXPECT_EQ(gLaunch.plan.buffers[4].rawBase, kGpuVBase + 68U);
    EXPECT_EQ(gLaunch.plan.buffers[0].rawSlotBytes, 3U * kLayerRawBytes);
    EXPECT_EQ(gLaunch.plan.buffers[1].rawSlotBytes, 2U * 68U);
    EXPECT_EQ(gLaunch.plan.buffers[0].coldDataOffset, 0U);
    EXPECT_EQ(gLaunch.plan.buffers[1].coldDataOffset, 90U);
    EXPECT_EQ(gLaunch.plan.buffers[2].coldDataOffset, 160U);
    EXPECT_EQ(gLaunch.plan.buffers[3].coldDataOffset, 256U);
    EXPECT_EQ(gLaunch.plan.buffers[4].coldDataOffset, 346U);
    EXPECT_EQ(gLaunch.plan.buffers[4].coldPaddingOffset, 414U);
    EXPECT_EQ(gLaunch.plan.buffers[4].coldPaddingBytes, 2U);
}

TEST(Nvfp4ColdPageCodecTest, DeepSeekV32AllIndexerLayoutFitsOneColdPagePlan)
{
    resetLaunch();
    std::vector<bool> const ownsIndexer(61, true);
    Nvfp4ColdPageCodec codec{makeMlaLayers(ownsIndexer.size())};
    ASSERT_TRUE(configureOne(codec, makeMlaDesc(ownsIndexer, kv::LayerGroupId{0}, 0, kMlaRawBytes, kDsaIndexKeyBytes)));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 61U * (kMlaColdBytes + kDsaIndexKeyBytes));

    kv::PageIndexPair const indices[]{{0, 0}};
    ASSERT_TRUE(codec.encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, nullptr));
    EXPECT_EQ(gLaunch.plan.numBuffers, 122U);
    EXPECT_EQ(gLaunch.plan.coldPageBytes, 1780224U);
}

TEST(Nvfp4ColdPageCodecTest, Glm52MixedIndexerLayoutFitsOneColdPagePlan)
{
    resetLaunch();
    std::vector<bool> ownsIndexer(78, false);
    for (std::size_t layer = 0; layer < ownsIndexer.size(); ++layer)
    {
        ownsIndexer[layer] = layer < 3U || (layer >= 6U && layer % 4U == 2U);
    }
    ASSERT_EQ(std::count(ownsIndexer.begin(), ownsIndexer.end(), true), 21);

    Nvfp4ColdPageCodec codec{makeMlaLayers(ownsIndexer.size())};
    ASSERT_TRUE(configureOne(codec, makeMlaDesc(ownsIndexer, kv::LayerGroupId{0}, 0, kMlaRawBytes, kDsaIndexKeyBytes)));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 78U * kMlaColdBytes + 21U * kDsaIndexKeyBytes);

    kv::PageIndexPair const indices[]{{0, 0}};
    ASSERT_TRUE(codec.encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, nullptr));
    EXPECT_EQ(gLaunch.plan.numBuffers, 99U);
    EXPECT_EQ(gLaunch.plan.coldPageBytes, 1794816U);
}

TEST(Nvfp4ColdPageCodecTest, PreservesOneCodecSubmissionAcrossThe256PageKernelBoundary)
{
    resetLaunch();
    auto codec = makeConfiguredAttentionCodec();

    std::vector<kv::PageIndexPair> indices(257);
    for (std::size_t page = 0; page < indices.size(); ++page)
    {
        indices[page] = {static_cast<std::int32_t>(500U - page), static_cast<std::int32_t>(page * 2U)};
    }
    ASSERT_TRUE(codec->encode(kv::LayerGroupId{3}, reinterpret_cast<void*>(kColdBase), indices.data(), indices.size(),
        reinterpret_cast<cudaStream_t>(kStreamValue)));
    EXPECT_EQ(gLaunch.offloadCalls, 1);
    EXPECT_EQ(gLaunch.offloadPages.size(), 257U);
    EXPECT_EQ(gLaunch.plan.numBuffers, 2U * kNumAttentionLayers);
}

TEST(Nvfp4ColdPageCodecTest, EmptyAttentionBatchIsValidAndDoesNotLaunch)
{
    resetLaunch();
    auto codec = makeConfiguredAttentionCodec(1, kv::LayerGroupId{0});

    EXPECT_TRUE(codec->encode(kv::LayerGroupId{0}, nullptr, nullptr, 0U, nullptr));
    EXPECT_TRUE(codec->decode(kv::LayerGroupId{0}, nullptr, nullptr, 0U, nullptr));
    EXPECT_EQ(gLaunch.offloadCalls, 0);
    EXPECT_EQ(gLaunch.onboardCalls, 0);
}

TEST(Nvfp4ColdPageCodecTest, DefaultStreamIsAccepted)
{
    resetLaunch();
    auto codec = makeConfiguredAttentionCodec(1, kv::LayerGroupId{0});

    kv::PageIndexPair const indices[]{{0, 0}};
    EXPECT_TRUE(codec->encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, nullptr));
    EXPECT_TRUE(codec->decode(kv::LayerGroupId{0}, reinterpret_cast<void const*>(kColdBase), indices, 1U, nullptr));
    EXPECT_EQ(gLaunch.offloadCalls, 1);
    EXPECT_EQ(gLaunch.onboardCalls, 1);
}

TEST(Nvfp4ColdPageCodecTest, NonEmptyAttentionBatchRequiresPageIndices)
{
    auto codec = makeConfiguredAttentionCodec(1, kv::LayerGroupId{0});

    auto const stream = reinterpret_cast<cudaStream_t>(kStreamValue);
    EXPECT_FALSE(codec->encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), nullptr, 1U, stream));
    EXPECT_FALSE(codec->decode(kv::LayerGroupId{0}, reinterpret_cast<void const*>(kColdBase), nullptr, 1U, stream));
}

TEST(Nvfp4ColdPageCodecTest, OnlyFp8RuntimeRequiresFp8Scales)
{
    auto layers = makeLayers(1);
    layers.front().fp8ScaleOrigQuant = {0.0F, 0.0F};
    layers.front().fp8ScaleQuantOrig = {0.0F, 0.0F};
    EXPECT_NO_THROW({ Nvfp4ColdPageCodec codec{layers}; });

    layers.front().runtimeType = kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3;
    EXPECT_THROW({ Nvfp4ColdPageCodec codec{layers}; }, std::invalid_argument);
}

TEST(Nvfp4ColdPageCodecTest, DiscoversLifecycleMembershipAcrossPoolGroups)
{
    auto layers = makeLayers(2);
    auto secondGroupLayers = makeLayers(2, 2);
    layers.insert(layers.end(), secondGroupLayers.begin(), secondGroupLayers.end());
    Nvfp4ColdPageCodec codec{layers};
    std::array descs{makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 2),
        makeAttentionDesc(kv::PoolGroupIndex{1}, kv::LayerGroupId{1}, 2, 2, 0x400000, 0x500000)};
    ASSERT_TRUE(codec.configure(descs.data(), kv::PoolGroupIndex{2}));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 2U * kLayerColdBytesAligned);
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{1}), 2U * kLayerColdBytesAligned);
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{0}), kv::LayerGroupId{0});
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{1}), kv::LayerGroupId{1});
}

TEST(Nvfp4ColdPageCodecTest, RejectsConfiguredAttentionLayerAbsentFromAllGpuDescriptors)
{
    Nvfp4ColdPageCodec codec{makeLayers(2)};
    EXPECT_FALSE(configureOne(codec, makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 1)));
}

TEST(Nvfp4ColdPageCodecTest, RejectsAttentionBufferWithMismatchedGeometry)
{
    Nvfp4ColdPageCodec codec{makeLayers()};
    auto desc = makeAttentionDesc();
    desc.slotDesc.variants.front().coalescedBuffers[kv::PoolIndex{0}].singleBufferSize += 16U;
    desc.pools[kv::PoolIndex{0}].slotBytes += 16U * kNumAttentionLayers;

    EXPECT_FALSE(configureOne(codec, desc));
}

TEST(Nvfp4ColdPageCodecTest, CoalescedAttentionSideBufferUsesItsOwnBaseOffsetAndSlotStride)
{
    resetLaunch();
    Nvfp4ColdPageCodec codec{makeLayers(1)};
    auto desc = makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 1);
    auto& keys = desc.slotDesc.variants.front().coalescedBuffers[kv::PoolIndex{0}];
    keys.bufferIds.push_back({0, "index_key"});
    desc.pools[kv::PoolIndex{0}].slotBytes += keys.singleBufferSize;

    ASSERT_TRUE(configureOne(codec, desc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 512U);
    kv::PageIndexPair const indices[]{{0, 0}};
    ASSERT_TRUE(codec.encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, nullptr));
    ASSERT_EQ(gLaunch.plan.numBuffers, 3U);
    auto const& side = gLaunch.plan.buffers[2];
    EXPECT_EQ(side.rawBase, kGpuKBase + kLayerRawBytes);
    EXPECT_EQ(side.rawSlotBytes, 2U * kLayerRawBytes);
    EXPECT_EQ(side.rawBytes, kLayerRawBytes);
    EXPECT_EQ(side.coldDataOffset, 180U);
    EXPECT_EQ(side.coldPaddingOffset, 500U);
    EXPECT_EQ(side.coldPaddingBytes, 12U);
    EXPECT_EQ(side.transform, kernels::Nvfp4BoundaryTransform::kLossless);
}

TEST(Nvfp4ColdPageCodecTest, UnknownLifecycleUsesFailureSentinels)
{
    auto codec = makeConfiguredAttentionCodec();
    EXPECT_EQ(codec->queryColdPageBytes(kv::LayerGroupId{99}), 0U);
    EXPECT_EQ(codec->getBatchingLayerGroupId(kv::LayerGroupId{99}), kv::LayerGroupId{-1});
    EXPECT_EQ(codec->queryPageIndexLocation(kv::LayerGroupId{99}), kv::PageIndexLocation::kBadLocation);
}

TEST(Nvfp4ColdPageCodecTest, NonAttentionLifecycleUsesLosslessSingleBlob)
{
    resetLaunch();
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "CUDA device is required for the lossless copy data-plane test";
    }

    constexpr std::size_t kPoolBytes = 64;
    constexpr std::size_t kSlots = 2;
    std::byte* statePool = nullptr;
    std::byte* convPool = nullptr;
    std::byte* coldPool = nullptr;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&statePool), kPoolBytes * kSlots), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&convPool), kPoolBytes * kSlots), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&coldPool), 2U * kPoolBytes * kSlots), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    kv::SlotDescVariant variant;
    variant.lifeCycleId = kv::LayerGroupId{1};
    variant.coalescedBuffers = kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer>{
        kv::CoalescedBuffer{kPoolBytes, {{10, "ssm_state"}}},
        kv::CoalescedBuffer{kPoolBytes, {{10, "conv_state"}}},
    };
    kv::PoolGroupDesc desc;
    desc.poolGroupIndex = kv::PoolGroupIndex{1};
    desc.numSlots = kSlots;
    desc.slotDesc.variants = {variant};
    desc.pools = kv::TypedVec<kv::PoolIndex, kv::PoolDesc>{
        kv::PoolDesc{kv::PoolIndex{0}, reinterpret_cast<std::uintptr_t>(statePool), kPoolBytes},
        kv::PoolDesc{kv::PoolIndex{1}, reinterpret_cast<std::uintptr_t>(convPool), kPoolBytes},
    };

    Nvfp4ColdPageCodec codec{makeLayers(1)};
    std::array descs{makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 1), desc};
    ASSERT_TRUE(codec.configure(descs.data(), kv::PoolGroupIndex{2}));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{1}), 2U * kPoolBytes);

    std::vector<std::byte> state(kPoolBytes);
    std::vector<std::byte> conv(kPoolBytes);
    for (std::size_t index = 0; index < kPoolBytes; ++index)
    {
        state[index] = static_cast<std::byte>(index + 1U);
        conv[index] = static_cast<std::byte>(index + 65U);
    }
    ASSERT_EQ(cudaMemcpy(statePool + kPoolBytes, state.data(), kPoolBytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(convPool + kPoolBytes, conv.data(), kPoolBytes, cudaMemcpyHostToDevice), cudaSuccess);

    kv::PageIndexPair const encodePair{0, 1};
    ASSERT_TRUE(codec.encode(kv::LayerGroupId{1}, coldPool, &encodePair, 1U, stream));
    ASSERT_EQ(cudaMemsetAsync(statePool, 0, kPoolBytes, stream), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(convPool, 0, kPoolBytes, stream), cudaSuccess);
    kv::PageIndexPair const decodePair{0, 0};
    ASSERT_TRUE(codec.decode(kv::LayerGroupId{1}, coldPool, &decodePair, 1U, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<std::byte> restoredState(kPoolBytes);
    std::vector<std::byte> restoredConv(kPoolBytes);
    ASSERT_EQ(cudaMemcpy(restoredState.data(), statePool, kPoolBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(restoredConv.data(), convPool, kPoolBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(restoredState, state);
    EXPECT_EQ(restoredConv, conv);
    EXPECT_EQ(gLaunch.offloadCalls, 0);

    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    EXPECT_EQ(cudaFree(coldPool), cudaSuccess);
    EXPECT_EQ(cudaFree(convPool), cudaSuccess);
    EXPECT_EQ(cudaFree(statePool), cudaSuccess);
}

TEST(Nvfp4ColdPageCodecTest, AttentionAndSsmSharingOneHotPoolGroupUseDifferentTransforms)
{
    Nvfp4ColdPageCodec codec{makeLayers(1)};
    auto desc = makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 1);

    kv::SlotDescVariant ssm;
    ssm.lifeCycleId = kv::LayerGroupId{1};
    ssm.coalescedBuffers = kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer>{
        kv::CoalescedBuffer{kLayerRawBytes, {{10, "ssm_state"}}},
        kv::CoalescedBuffer{kLayerRawBytes, {{10, "conv_state"}}},
    };
    desc.slotDesc.variants.push_back(std::move(ssm));

    ASSERT_TRUE(configureOne(codec, desc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), kLayerColdBytesAligned);
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{1}), 2U * kLayerRawBytes);
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{0}), kv::LayerGroupId{0});
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{1}), kv::LayerGroupId{1});
}

} // namespace
} // namespace tensorrt_llm::kv_cache_compression

namespace tensorrt_llm::kernels
{

Nvfp4BoundaryPreparedPlan prepareNvfp4BoundaryPlan(std::vector<Nvfp4BoundaryBufferPlan> const& buffers,
    std::size_t coldPageBytes, Nvfp4BoundaryRuntimeType runtimeType)
{
    if (buffers.empty() || buffers.size() > kNvfp4BoundaryMaxBuffersPerLaunch || coldPageBytes == 0U)
    {
        throw std::invalid_argument("invalid test launch plan");
    }
    Nvfp4BoundaryPreparedPlan plan;
    std::copy(buffers.begin(), buffers.end(), plan.buffers.begin());
    plan.numBuffers = static_cast<std::uint32_t>(buffers.size());
    plan.coldPageBytes = coldPageBytes;
    plan.runtimeType = runtimeType;
    return plan;
}

void invokeNvfp4BoundaryOffloadCompress(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void* coldBase, cudaStream_t stream)
{
    auto& launch = kv_cache_compression::gLaunch;
    ++launch.offloadCalls;
    launch.offloadPages = pages;
    launch.plan = plan;
    launch.coldBase = coldBase;
    launch.stream = stream;
}

void invokeNvfp4BoundaryOnboardDecompress(std::vector<Nvfp4BoundaryOnboardPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void const* coldBase, cudaStream_t stream)
{
    auto& launch = kv_cache_compression::gLaunch;
    ++launch.onboardCalls;
    launch.onboardPages = pages;
    launch.plan = plan;
    launch.coldBase = coldBase;
    launch.stream = stream;
}

} // namespace tensorrt_llm::kernels

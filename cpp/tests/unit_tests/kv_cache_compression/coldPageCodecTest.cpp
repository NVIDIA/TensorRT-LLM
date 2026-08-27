/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kv_cache_compression/nativeColdPageCodec.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
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
        : NativeColdPageCodec(std::move(layerIds))
    {
    }

    std::vector<ColdPageLifecycleProperties> configureProvider(
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

    void encodeProvider(std::size_t lifecycleIndex, void* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream) override
    {
        if (failBatches)
        {
            enqueueFailureMarker(stream);
            throw std::runtime_error("requested batch failure");
        }
        ++encodeCalls;
        lastLifecycleIndex = lifecycleIndex;
        lastColdBase = coldBase;
        lastIndices.assign(pageIndices, pageIndices + numPages);
        lastStream = stream;
    }

    void decodeProvider(std::size_t lifecycleIndex, void const* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream) override
    {
        if (failBatches)
        {
            enqueueFailureMarker(stream);
            throw std::runtime_error("requested batch failure");
        }
        ++decodeCalls;
        lastLifecycleIndex = lifecycleIndex;
        lastColdBase = coldBase;
        lastIndices.assign(pageIndices, pageIndices + numPages);
        lastStream = stream;
    }

    bool failConfigure = false;
    bool failBatches = false;
    std::atomic_bool* failureMarker = nullptr;
    int encodeCalls = 0;
    int decodeCalls = 0;
    std::size_t lastLifecycleIndex = 0;
    void const* lastColdBase = nullptr;
    cudaStream_t lastStream{};
    std::vector<kv::PageIndexPair> lastIndices;
    std::vector<ResolvedHotLifecycle> resolved;

private:
    void enqueueFailureMarker(cudaStream_t stream)
    {
        if (failureMarker != nullptr
            && cudaLaunchHostFunc(
                   stream, [](void* marker) { static_cast<std::atomic_bool*>(marker)->store(true); }, failureMarker)
                != cudaSuccess)
        {
            throw std::runtime_error("failed to enqueue the requested batch failure marker");
        }
    }
};

TEST(NativeColdPageCodecTest, ResolvesKvcManagerLayoutAndForwardsWholeBatchOnce)
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

    std::vector<kv::PageIndexPair> indices(4096);
    for (std::size_t index = 0; index < indices.size(); ++index)
    {
        indices[index] = {static_cast<std::int32_t>(index + 1U), static_cast<std::int32_t>(index)};
    }
    auto const stream = reinterpret_cast<cudaStream_t>(kStreamValue);
    ASSERT_TRUE(
        codec.encode(kv::LayerGroupId{3}, reinterpret_cast<void*>(kColdBase), indices.data(), indices.size(), stream));
    EXPECT_EQ(codec.encodeCalls, 1);
    EXPECT_EQ(codec.lastLifecycleIndex, 0U);
    EXPECT_EQ(codec.lastIndices.size(), 4096U);
    EXPECT_EQ(codec.lastIndices.back().src, 4095);
    EXPECT_EQ(codec.lastColdBase, reinterpret_cast<void*>(kColdBase));
    EXPECT_EQ(codec.lastStream, stream);

    ASSERT_TRUE(codec.decode(
        kv::LayerGroupId{3}, reinterpret_cast<void const*>(kColdBase), indices.data(), indices.size(), stream));
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

TEST(NativeColdPageCodecTest, CatchesProviderConfigureFailuresAndInvalidBatches)
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
    EXPECT_EQ(validCodec.encodeCalls, 0);
    EXPECT_EQ(validCodec.decodeCalls, 0);
}

TEST(NativeColdPageCodecTest, ProviderFailureUsesTheSuppliedCudaStreamForRollback)
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "Failure draining requires a CUDA device";
    }

    RecordingCodec codec{{0}};
    ASSERT_TRUE(configureOne(codec, makeAttentionDesc()));
    codec.failBatches = true;
    std::atomic_bool completed = false;
    codec.failureMarker = &completed;
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    kv::PageIndexPair const indices[]{{0, 0}};
    EXPECT_FALSE(codec.encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, stream));
    EXPECT_TRUE(completed.exchange(false));
    EXPECT_FALSE(codec.decode(kv::LayerGroupId{0}, reinterpret_cast<void const*>(kColdBase), indices, 1U, stream));
    EXPECT_TRUE(completed.load());
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(NativeColdPageCodecTest, UnknownLifecycleUsesFailureSentinels)
{
    RecordingCodec codec{{0}};
    ASSERT_TRUE(configureOne(codec, makeAttentionDesc()));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{99}), 0U);
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{99}), kv::LayerGroupId{-1});
    EXPECT_EQ(codec.queryPageIndexLocation(kv::LayerGroupId{99}), kv::PageIndexLocation::kBadLocation);
}

} // namespace
} // namespace tensorrt_llm::kv_cache_compression

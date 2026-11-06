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

#include "kvCacheManagerV2TestUtils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/funcGuard.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeConfig;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeTieredConfig;
using tensorrt_llm::common::TllmException;

KVCacheManagerConfig makeOptionalDiskConfig(bool hostTier)
{
    auto config = makeConfig();
    if (hostTier)
    {
        config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    }
    config.cacheTiers.emplace_back(DiskCacheTierConfig{4 << 20, "/tmp"});
    for (int i = 1; i <= 2; ++i)
    {
        auto layer = std::get<AttentionLayerConfig>(config.layers.front());
        layer.layerId = i;
        layer.slidingWindowSize = i == 1 ? 4 : 2;
        layer.reusePolicy = ReusePolicy::OPTIONAL;
        // Unequal buffer sizes exercise multiple physical pools per page.
        layer.buffers.push_back(BufferConfig{"score", 1024, std::nullopt});
        config.layers.emplace_back(std::move(layer));
    }
    return config;
}

// A custom lossless codec using host indices and ordinary CUDA copies. Its
// padded representation deliberately differs from the default cold layout.
class OptionalTestColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const* groups, PoolGroupIndex count) noexcept override
    {
        for (int i = 0; i < count.value(); ++i)
        {
            for (auto const& variant : groups[i].slotDesc.variants)
            {
                mLayouts.emplace(variant.lifeCycleId.value(), groups[i]);
            }
        }
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId id) const noexcept override
    {
        auto const it = mLayouts.find(id.value());
        if (it == mLayouts.end())
        {
            return 0;
        }
        size_t bytes = 64;
        for (auto const& pool : it->second.pools)
        {
            bytes += pool.slotBytes;
        }
        return bytes;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(
        LayerGroupId id, void* cold, PageIndexPair const* indices, size_t count, cudaStream_t stream) noexcept override
    {
        ++encodeCalls;
        return copy(id, cold, indices, count, stream, true);
    }

    bool decode(LayerGroupId id, void const* cold, PageIndexPair const* indices, size_t count,
        cudaStream_t stream) noexcept override
    {
        ++decodeCalls;
        return copy(id, const_cast<void*>(cold), indices, count, stream, false);
    }

    int encodeCalls = 0;
    int decodeCalls = 0;

private:
    bool copy(LayerGroupId id, void* cold, PageIndexPair const* indices, size_t count, cudaStream_t stream,
        bool encode) noexcept
    {
        auto const it = mLayouts.find(id.value());
        if (it == mLayouts.end())
        {
            return false;
        }
        size_t const coldBytes = queryColdPageBytes(id);
        for (size_t i = 0; i < count; ++i)
        {
            auto const hotIndex = encode ? indices[i].src : indices[i].dst;
            auto const coldIndex = encode ? indices[i].dst : indices[i].src;
            size_t offset = 64;
            for (auto const& pool : it->second.pools)
            {
                auto* hotPtr = reinterpret_cast<char*>(pool.baseAddress) + hotIndex * pool.slotBytes;
                auto* coldPtr = static_cast<char*>(cold) + coldIndex * coldBytes + offset;
                auto const status = cudaMemcpyAsync(
                    encode ? coldPtr : hotPtr, encode ? hotPtr : coldPtr, pool.slotBytes, cudaMemcpyDefault, stream);
                if (status != cudaSuccess)
                {
                    return false;
                }
                offset += pool.slotBytes;
            }
        }
        return true;
    }

    std::unordered_map<int, PoolGroupDesc> mLayouts;
};

class OptionalDiskCheckpointTest : public ::testing::TestWithParam<std::tuple<bool, bool>>
{
};

TEST_P(OptionalDiskCheckpointTest, PayloadRestoreRetryAndIndependentEviction)
{
    auto const [hostTier, customCodec] = GetParam();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    FuncGuard streamGuard([&] { cudaStreamDestroy(stream); });
    auto codec = customCodec ? std::make_unique<OptionalTestColdPageCodec>() : nullptr;
    auto* codecObserver = codec.get();
    auto manager = std::make_shared<KvCacheManager>(makeOptionalDiskConfig(hostTier), nullptr, std::move(codec));
    auto& storage = manager->storage();
    auto const global = manager->getLayerGroupId(LayerId{0});
    auto const encoder = manager->getLayerGroupId(LayerId{1});
    auto const peer = manager->getLayerGroupId(LayerId{2});
    auto const hotGroup = storage.getPoolGroupIndex(kHotLevel, encoder);
    auto const sizes = storage.slotSize(hotGroup);
    ASSERT_EQ(sizes.size(), PoolIndex{2});
    std::vector<TokenIdExt> tokens;
    for (int i = 0; i < 8; ++i)
    {
        tokens.emplace_back(TokenId{i});
    }
    TokenSpan const span{tokens.data(), static_cast<int>(tokens.size())};
    auto source = manager->createKvCache({}, {}, 1);
    ASSERT_TRUE(source->resume(stream));
    ASSERT_TRUE(source->resize(8));
    for (auto const id : {encoder, peer})
    {
        auto const slot = SlotId{source->getBasePageIndices(id)[1]};
        for (int p = 0; p < sizes.size().value(); ++p)
        {
            auto const address = std::get<MemAddress>(storage.slotAddress(kHotLevel, hotGroup, slot, PoolIndex{p}));
            ASSERT_EQ(
                cudaMemsetAsync(reinterpret_cast<void*>(address), 31 + id.value() * 7 + p, sizes[PoolIndex{p}], stream),
                cudaSuccess);
        }
    }
    source->commit(span);
    auto block = source->blocks()[BlockOrdinal{1}].treeBlock;
    auto* globalPage = block->getPage(global);
    auto const* encoderPage = block->getPage(encoder);
    auto const* peerPage = block->getPage(peer);
    source->close();

    auto pressure = [&](CacheLevel level)
    {
        auto const group = storage.getPoolGroupIndex(level, encoder);
        TypedVec<PoolGroupIndex, SlotCount> goals(storage.numPoolGroups(level), SlotCount{0});
        goals[group] = storage.getStatistics(level, group).total;
        storage.prepareFreeSlots(level, goals);
    };
    auto const diskLevel = CacheLevel{hostTier ? 2 : 1};
    for (int level = 0; level < diskLevel.value(); ++level)
    {
        pressure(CacheLevel{level});
        for (auto const id : {encoder, peer})
        {
            ASSERT_NE(block->getPage(id), nullptr);
            EXPECT_EQ(block->getPage(id)->cacheLevel, CacheLevel{level + 1});
        }
        EXPECT_EQ(block->getPage(global), globalPage);
        EXPECT_EQ(manager->radixTree().match({}, span).numTokens, 8);
    }

    auto reader = manager->createKvCache({}, span, 2);
    ASSERT_TRUE(reader->reuseStatus()[encoder.value()].complete);
    auto reservations
        = storage.newSlotsForPoolGroup(kHotLevel, hotGroup, storage.getStatistics(kHotLevel, hotGroup).free);
    for (auto const& slot : reservations)
    {
        // Offload can still be reading the recycled GPU slots on another stream.
        slot.readyEvent.waitInStream(reinterpret_cast<CudaStream>(stream));
    }
    // Poison the entire hot Encoder group: a false restore cannot pass using
    // stale bytes from the original slot after it was offloaded to disk.
    for (int p = 0; p < sizes.size().value(); ++p)
    {
        auto const address = std::get<MemAddress>(storage.slotAddress(kHotLevel, hotGroup, SlotId{0}, PoolIndex{p}));
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(address), 0xEE,
                      sizes[PoolIndex{p}] * storage.getStatistics(kHotLevel, hotGroup).total, stream),
            cudaSuccess);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_FALSE(reader->resume(stream, std::vector<LifeCycleId>{encoder, peer}));
    EXPECT_FALSE(reader->isActive());
    EXPECT_EQ(reader->numCommittedTokens(), 8);
    for (auto& slot : reservations)
    {
        storage.releaseSlot(encoder, kHotLevel, std::move(slot));
    }
    ASSERT_TRUE(reader->resume(stream, std::vector<LifeCycleId>{encoder, peer}));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(block->getPage(encoder), encoderPage);
    EXPECT_EQ(block->getPage(peer), peerPage);
    for (auto const id : {encoder, peer})
    {
        auto const slot = SlotId{reader->getBasePageIndices(id)[1]};
        EXPECT_EQ(slot, block->getPage(id)->slotId());
        for (int p = 0; p < sizes.size().value(); ++p)
        {
            auto const address = std::get<MemAddress>(storage.slotAddress(kHotLevel, hotGroup, slot, PoolIndex{p}));
            std::vector<uint8_t> bytes(sizes[PoolIndex{p}]);
            ASSERT_EQ(
                cudaMemcpy(bytes.data(), reinterpret_cast<void const*>(address), bytes.size(), cudaMemcpyDeviceToHost),
                cudaSuccess);
            EXPECT_TRUE(std::all_of(
                bytes.begin(), bytes.end(), [&](uint8_t value) { return value == 31 + id.value() * 7 + p; }));
        }
    }
    if (codecObserver)
    {
        EXPECT_GT(codecObserver->encodeCalls, 0);
        EXPECT_GT(codecObserver->decodeCalls, 0);
    }
    reader->close();
    for (int level = 0; level <= diskLevel.value(); ++level)
    {
        pressure(CacheLevel{level});
    }
    EXPECT_EQ(block->getPage(encoder), nullptr);
    EXPECT_EQ(block->getPage(peer), nullptr);
    ASSERT_EQ(block->getPage(global), globalPage);
    EXPECT_EQ(manager->radixTree().match({}, span).numTokens, 8);
    auto miss = manager->createKvCache({}, span, 3);
    ASSERT_TRUE(miss->resume(stream));
    EXPECT_EQ(miss->numCommittedTokens(), 8);
    EXPECT_FALSE(miss->reuseStatus()[encoder.value()].complete);
    miss->close();
    block.reset();
    manager->shutdown();
}

INSTANTIATE_TEST_SUITE_P(
    DefaultAndCustomCodec, OptionalDiskCheckpointTest, ::testing::Combine(::testing::Bool(), ::testing::Bool()));

KVCacheManagerConfig makeDiskTieredConfig()
{
    auto config = makeConfig();
    config.cacheTiers.emplace_back(DiskCacheTierConfig{4 << 20, "/tmp"});
    return config;
}

KVCacheManagerConfig makeGpuTieredConfig()
{
    auto config = makeTieredConfig();
    config.cacheTiers[1] = GpuCacheTierConfig{4 << 20};
    return config;
}

//! One sparse attention layer, over a hot tier and a cold tier.
//!
//! The hot pool group is host-backed because the layer is sparse; the cold tier is an
//! ordinary codec target, which is the whole point of the arrangement.
KVCacheManagerConfig makeSparseTieredConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    AttentionLayerConfig layer;
    layer.layerId = 0;
    BufferConfig buffer{"key", 2 << 20, std::nullopt};
    buffer.isSparse = true;
    layer.buffers.push_back(std::move(buffer));
    layer.sparseCache = SparseCacheConfig{/*numSelectedBlocks=*/1, /*maxCachedUnits=*/2, /*unitTokens=*/4};
    // A pool group sizes its slots from its device-backed pools, so a sparse layer carries
    // one -- which is also the shape a model uses.
    layer.buffers.push_back(BufferConfig{"index", 256, std::nullopt});
    config.layers.emplace_back(std::move(layer));
    BatchDesc constraint;
    constraint.kvCaches.assign(2, KVCacheDesc{/*capacity=*/16, /*historyLength=*/0});
    config.constraints.push_back(std::move(constraint));
    return config;
}

KVCacheManagerConfig makeSplitColdGroupingConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    // The host tier is distributed in units of HostMem::sizeGranularity(), so a quota
    // of only a few multiples of it cannot express a pool ratio: every group
    // rounds to the same single unit. Sized here to leave enough units for the
    // ratios these tests assert on.
    config.cacheTiers.emplace_back(HostCacheTierConfig{256 << 20});

    AttentionLayerConfig first;
    first.layerId = 0;
    first.slidingWindowSize = 128;
    first.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(first));

    AttentionLayerConfig second;
    second.layerId = 1;
    second.slidingWindowSize = 256;
    second.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(second));
    return config;
}

//! A config whose cold tier holds exactly one page.
//!
//! The host tier is allocated in whole multiples of HostMem::sizeGranularity(), so a
//! pool contains a single slot only when one page fills that unit. Tests that
//! need two pages to contend for one cold slot size the page accordingly rather
//! than shrinking the quota, which cannot go below one unit.
KVCacheManagerConfig makeSingleColdSlotConfig()
{
    KVCacheManagerConfig config = makeSplitColdGroupingConfig();
    for (auto& layer : config.layers)
    {
        std::get<AttentionLayerConfig>(layer).buffers[0].size = HostMem::sizeGranularity();
    }
    // The hot tier holds pages of the same size, so it is sized in the same
    // unit to leave room for the slots these tests allocate up front.
    config.cacheTiers[0] = GpuCacheTierConfig{8 * HostMem::sizeGranularity()};
    config.cacheTiers[1] = HostCacheTierConfig{HostMem::sizeGranularity()};
    return config;
}

class RejectingColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    explicit RejectingColdPageCodec(int& destructionCount)
        : mDestructionCount(destructionCount)
    {
    }

    ~RejectingColdPageCodec() override
    {
        ++mDestructionCount;
    }

    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return false;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 1;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

private:
    int& mDestructionCount;
};

class SplitColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    explicit SplitColdPageCodec(bool batchTogether = false)
        : mBatchTogether(batchTogether)
    {
    }

    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId layerGroupId) const noexcept override
    {
        if (layerGroupId == LayerGroupId{0})
            return 1024;
        if (layerGroupId == LayerGroupId{1})
            return 2048;
        return 0;
    }

    LayerGroupId getBatchingLayerGroupId(LayerGroupId layerGroupId) const noexcept override
    {
        return mBatchTogether ? LayerGroupId{0} : layerGroupId;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return true;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return true;
    }

private:
    bool mBatchTogether;
};

class OversizedColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return std::numeric_limits<size_t>::max() / 3 + 1;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }
};

class MixedIndexLocationColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 1024;
    }

    LayerGroupId getBatchingLayerGroupId(LayerGroupId) const noexcept override
    {
        return LayerGroupId{0};
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId layerGroupId) const noexcept override
    {
        return layerGroupId == LayerGroupId{0} ? PageIndexLocation::kHost : PageIndexLocation::kDevice;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }
};

//! Records which life cycles were encoded and decoded, without interfering.
//!
//! Compression lives behind this interface, so "can sparse history be compressed" reduces
//! to "does a sparse life cycle's page reach the codec at all". Before sparse history moved
//! to the hot tier it could not: the codec runs on the hot/cold boundary, and that history
//! was locked on the cold side and never crossed.
class RecordingColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 2 << 20;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId lg, void*, PageIndexPair const*, size_t pages, cudaStream_t) noexcept override
    {
        mEncoded[lg.value()] += pages;
        return true;
    }

    bool decode(LayerGroupId lg, void const*, PageIndexPair const*, size_t pages, cudaStream_t) noexcept override
    {
        mDecoded[lg.value()] += pages;
        return true;
    }

    size_t encoded(LayerGroupId lg) const
    {
        auto const it = mEncoded.find(lg.value());
        return it == mEncoded.end() ? 0 : it->second;
    }

    size_t decoded(LayerGroupId lg) const
    {
        auto const it = mDecoded.find(lg.value());
        return it == mDecoded.end() ? 0 : it->second;
    }

private:
    std::map<int, size_t> mEncoded;
    std::map<int, size_t> mDecoded;
};

class AsyncRejectingColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    enum class Operation
    {
        kEncode,
        kDecode,
    };

    explicit AsyncRejectingColdPageCodec(Operation operation)
        : mOperation(operation)
    {
    }

    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 2 << 20;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t stream) noexcept override
    {
        return mOperation == Operation::kEncode ? reject(stream) : true;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t stream) noexcept override
    {
        return mOperation == Operation::kDecode ? reject(stream) : true;
    }

    bool launched() const noexcept
    {
        return mLaunched.load(std::memory_order_acquire);
    }

    void release() noexcept
    {
        mRelease.store(true, std::memory_order_release);
    }

private:
    static void CUDART_CB waitForRelease(void* data)
    {
        auto& codec = *static_cast<AsyncRejectingColdPageCodec*>(data);
        while (!codec.mRelease.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
    }

    bool reject(cudaStream_t stream) noexcept
    {
        bool const launched = cudaLaunchHostFunc(stream, waitForRelease, this) == cudaSuccess;
        mLaunched.store(launched, std::memory_order_release);
        return false;
    }

    Operation mOperation;
    std::atomic<bool> mLaunched{false};
    std::atomic<bool> mRelease{false};
};

SharedPtr<CommittedPage> makeCommittedPage(KvCacheManager& manager, StorageManager& storage, CacheLevel level,
    Slot& slot, LifeCycleId lifeCycle = LifeCycleId{0}, Priority priority = kPriorityDefault, int tokenBase = 0)
{
    RootBlock& root = manager.radixTree().addOrGetExisting({});
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < manager.tokensPerBlock(); ++token)
    {
        tokens.emplace_back(TokenId{tokenBase + token});
    }
    auto block = addOrGetExistingBlock(&root, std::move(tokens), /*knownNoDigest=*/true);
    auto page = makeShared<CommittedPage>(
        &storage, block, lifeCycle, level, static_cast<int>(block->tokens.size()), priority);
    page->setSlot(slot);
    block->storage[lifeCycle] = page.get();
    storage.scheduleForEviction(*page);
    return page;
}

TEST(KvCacheManagerV2ColdPageTest, ConstructionFailureDestroysCodec)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    int destructionCount = 0;
    std::unique_ptr<IKvCacheColdPageCodec> codec = std::make_unique<RejectingColdPageCodec>(destructionCount);
    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(makeConfig(), nullptr, std::move(codec));
            (void) manager;
        },
        TllmException);

    EXPECT_EQ(destructionCount, 1);
}

TEST(KvCacheManagerV2ColdPageTest, RejectsColdPageStagingSizeOverflow)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(
                makeDiskTieredConfig(), nullptr, std::make_unique<OversizedColdPageCodec>());
            (void) manager;
        },
        TllmException);
}

TEST(KvCacheManagerV2ColdPageTest, DoesNotSizePageStagingWithoutColdTier)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_NO_THROW({
        auto manager
            = std::make_shared<KvCacheManager>(makeConfig(), nullptr, std::make_unique<OversizedColdPageCodec>());
        (void) manager;
    });
}

TEST(KvCacheManagerV2ColdPageTest, ColdGpuTierSupportsSingleSlotRoundTrip)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeGpuTieredConfig());
    auto& storage = manager->storage();
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto streamGuard = FuncGuard([stream]() { cudaStreamDestroy(stream); });
    LifeCycleId const lifeCycle{0};
    CacheLevel const coldLevel{1};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    auto hotSlots = storage.newSlots(kHotLevel, oneSlot);
    auto coldSlots = storage.newSlots(coldLevel, oneSlot);
    ASSERT_EQ(hotSlots[lifeCycle].size(), 1);
    ASSERT_EQ(coldSlots[lifeCycle].size(), 1);

    Slot& hotSlot = hotSlots[lifeCycle].front();
    Slot& coldSlot = coldSlots[lifeCycle].front();
    PoolGroupIndex const hotPoolGroup = storage.getPoolGroupIndex(kHotLevel, lifeCycle);
    size_t const hotPageBytes = storage.slotSize(hotPoolGroup).at(PoolIndex{0});
    MemAddress const hotAddress
        = std::get<MemAddress>(storage.slotAddress(kHotLevel, hotPoolGroup, hotSlot.slotId(), PoolIndex{0}));
    constexpr uint8_t kPattern = 0xA7;
    // Every operation on the hot page must be ordered against the migrations, so issue the memsets
    // on `stream` too. cudaMemset() goes to the legacy NULL stream and is asynchronous with respect
    // to the host for device memory; because `stream` is cudaStreamNonBlocking it does not
    // implicitly synchronize with the legacy stream, so a plain cudaMemset() here would be free to
    // land after the migration that is supposed to overwrite it.
    ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(hotAddress), kPattern, hotPageBytes, stream), cudaSuccess);

    storage.copySlotData(
        lifeCycle, coldLevel, kHotLevel, coldSlot.slotId(), hotSlot.slotId(), reinterpret_cast<CUstream>(stream));
    ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(hotAddress), 0, hotPageBytes, stream), cudaSuccess);
    storage.copySlotData(
        lifeCycle, kHotLevel, coldLevel, hotSlot.slotId(), coldSlot.slotId(), reinterpret_cast<CUstream>(stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<uint8_t> restoredPage(hotPageBytes);
    ASSERT_EQ(cudaMemcpy(
                  restoredPage.data(), reinterpret_cast<void const*>(hotAddress), hotPageBytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_TRUE(std::all_of(restoredPage.begin(), restoredPage.end(), [](uint8_t byte) { return byte == kPattern; }));

    storage.releaseSlot(lifeCycle, coldLevel, std::move(coldSlot));
    storage.releaseSlot(lifeCycle, kHotLevel, std::move(hotSlot));
}

TEST(KvCacheManagerV2ColdPageTest, AsyncEncodeRejectionFencesRecycledColdSlotAndReschedulesSource)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<AsyncRejectingColdPageCodec>(AsyncRejectingColdPageCodec::Operation::kEncode);
    auto* codecPtr = codec.get();
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig(), nullptr, std::move(codec));
    auto releaseCodecGuard = FuncGuard([codecPtr]() { codecPtr->release(); });
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const coldLevel{1};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);

    auto coldSlots = storage.newSlots(coldLevel, oneSlot);
    Slot coldBlocker = std::move(coldSlots[lifeCycle].front());
    auto hotSlots = storage.newGpuSlots(twoSlots);
    Slot hotBlocker = std::move(hotSlots[lifeCycle].back());
    auto sourcePage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[lifeCycle].front());
    SlotId const sourceSlotId = sourcePage->slotId();

    EXPECT_THROW(storage.newGpuSlots(oneSlot), TllmException);
    ASSERT_TRUE(codecPtr->launched());
    EXPECT_EQ(sourcePage->cacheLevel, kHotLevel);
    EXPECT_EQ(sourcePage->slotId(), sourceSlotId);
    EXPECT_TRUE(sourcePage->scheduledForEviction());

    auto recycledSlots = storage.newSlots(coldLevel, oneSlot);
    Slot recycledColdSlot = std::move(recycledSlots[lifeCycle].front());
    EXPECT_FALSE(recycledColdSlot.queryReady());
    EXPECT_FALSE(sourcePage->queryReady());

    codecPtr->release();
    recycledColdSlot.readyEvent.synchronize();
    storage.releaseSlot(lifeCycle, coldLevel, std::move(recycledColdSlot));
    storage.releaseSlot(lifeCycle, coldLevel, std::move(coldBlocker));
    storage.releaseSlot(lifeCycle, kHotLevel, std::move(hotBlocker));
}

TEST(KvCacheManagerV2ColdPageTest, ForceEvictFailureReschedulesFallenPage)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<AsyncRejectingColdPageCodec>(AsyncRejectingColdPageCodec::Operation::kEncode);
    auto* codecPtr = codec.get();
    auto config = makeSplitColdGroupingConfig();
    std::get<AttentionLayerConfig>(config.layers[1]).buffers.front().size *= 2;
    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
    auto releaseCodecGuard = FuncGuard([codecPtr]() { codecPtr->release(); });
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{1};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{2}, 1);

    auto hotSlots = storage.newGpuSlots(oneSlot);
    auto sourcePage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[lifeCycle].front(), lifeCycle, 80);
    auto lowPage
        = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[LifeCycleId{0}].front(), LifeCycleId{0}, 10);
    SlotId const lowSlotId = lowPage->slotId();
    WeakPtr<CommittedPage> lowPageRef = lowPage;
    lowPage.reset();
    SlotId const sourceSlotId = sourcePage->slotId();
    TypedVec<PoolGroupIndex, SlotCount> evictOne(storage.numPoolGroups(kHotLevel), 0);
    evictOne[storage.getPoolGroupIndex(kHotLevel, lifeCycle)] = 1;
    evictOne[storage.getPoolGroupIndex(kHotLevel, LifeCycleId{0})] = 1;
    size_t dropped = 0;
    auto recordDrop = [&](std::vector<SharedPtr<Page>> const& pages, CacheLevel) { dropped += pages.size(); };

    EXPECT_THROW(storage.forceEvict(kHotLevel, evictOne, recordDrop), TllmException);
    ASSERT_TRUE(codecPtr->launched());
    EXPECT_EQ(sourcePage->cacheLevel, kHotLevel);
    EXPECT_EQ(sourcePage->slotId(), sourceSlotId);
    EXPECT_TRUE(sourcePage->scheduledForEviction());
    EXPECT_FALSE(sourcePage->queryReady());
    EXPECT_EQ(sourcePage->priority(), 80);
    EXPECT_EQ(dropped, 0);
    lowPage = lowPageRef.lock();
    ASSERT_TRUE(lowPage);
    EXPECT_EQ(lowPage->cacheLevel, kHotLevel);
    EXPECT_EQ(lowPage->slotId(), lowSlotId);
    EXPECT_TRUE(lowPage->scheduledForEviction());
    EXPECT_EQ(lowPage->priority(), 10);

    codecPtr->release();
    sourcePage->readyEvent.synchronize();
}

TEST(KvCacheManagerV2ColdPageTest, AsyncDecodeRejectionFencesRecycledGpuSlot)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<AsyncRejectingColdPageCodec>(AsyncRejectingColdPageCodec::Operation::kDecode);
    auto* codecPtr = codec.get();
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig(), nullptr, std::move(codec));
    auto releaseCodecGuard = FuncGuard([codecPtr]() { codecPtr->release(); });
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const coldLevel{1};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);

    auto hotSlots = storage.newGpuSlots(oneSlot);
    Slot hotBlocker = std::move(hotSlots[lifeCycle].front());
    auto coldSlots = storage.newSlots(coldLevel, oneSlot);
    auto sourcePage = makeCommittedPage(*manager, storage, coldLevel, coldSlots[lifeCycle].front());
    SlotId const sourceSlotId = sourcePage->slotId();
    storage.excludeFromEviction(*sourcePage);
    auto cache = manager->createKvCache();
    EXPECT_THROW(storage.batchedMigrate(kHotLevel, {sourcePage}, {}), TllmException);
    ASSERT_TRUE(codecPtr->launched());
    EXPECT_EQ(sourcePage->cacheLevel, coldLevel);
    EXPECT_EQ(sourcePage->slotId(), sourceSlotId);

    auto recycledSlots = storage.newGpuSlots(oneSlot);
    Slot recycledGpuSlot = std::move(recycledSlots[lifeCycle].front());
    EXPECT_FALSE(recycledGpuSlot.queryReady());
    EXPECT_FALSE(sourcePage->queryReady());

    codecPtr->release();
    recycledGpuSlot.readyEvent.synchronize();
    storage.releaseSlot(lifeCycle, kHotLevel, std::move(recycledGpuSlot));
    storage.releaseSlot(lifeCycle, kHotLevel, std::move(hotBlocker));
    storage.scheduleForEviction(*sourcePage);
    cache->close();
}

TEST(KvCacheManagerV2ColdPageTest, RejectsBatchingClassWithDifferentColdPageSizes)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(
                makeSplitColdGroupingConfig(), nullptr, std::make_unique<SplitColdPageCodec>(true));
            (void) manager;
        },
        TllmException);
}

TEST(KvCacheManagerV2ColdPageTest, RejectsBatchingClassWithDifferentIndexLocations)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(
                makeSplitColdGroupingConfig(), nullptr, std::make_unique<MixedIndexLocationColdPageCodec>());
            (void) manager;
        },
        TllmException);
}

TEST(KvCacheManagerV2ColdPageTest, ColdGroupingIsIndependentOfHotGrouping)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeSplitColdGroupingConfig();
    config.initialPoolRatio = std::vector<float>{0.25F, 0.75F};
    auto codec = std::make_unique<SplitColdPageCodec>();
    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
    EXPECT_FALSE(codec);

    StorageManager const& storage = manager->storage();
    EXPECT_EQ(storage.numLifeCycles(), LifeCycleId{2});
    EXPECT_EQ(storage.numPoolGroups(kHotLevel), PoolGroupIndex{1});
    EXPECT_EQ(storage.numPoolGroups(CacheLevel{1}), PoolGroupIndex{2});

    PoolGroupIndex const hotGroup0 = storage.getPoolGroupIndex(kHotLevel, LifeCycleId{0});
    PoolGroupIndex const hotGroup1 = storage.getPoolGroupIndex(kHotLevel, LifeCycleId{1});
    EXPECT_EQ(hotGroup0, hotGroup1);
    EXPECT_EQ(storage.getRatioList(kHotLevel), (TypedVec<PoolGroupIndex, float>{1.0F}));

    auto const coldRatio = storage.getRatioList(CacheLevel{1});
    ASSERT_EQ(coldRatio.size(), PoolGroupIndex{2});
    constexpr float kFirstColdByteRatio = (0.25F * 1024) / (0.25F * 1024 + 0.75F * 2048);
    EXPECT_NEAR(coldRatio[PoolGroupIndex{0}], kFirstColdByteRatio, 0.01F);
    EXPECT_NEAR(coldRatio[PoolGroupIndex{1}], 1.0F - kFirstColdByteRatio, 0.01F);

    auto const firstColdSlots = storage.numSlots(PoolGroupIndex{0}, CacheLevel{1});
    auto const secondColdSlots = storage.numSlots(PoolGroupIndex{1}, CacheLevel{1});
    EXPECT_NEAR(
        static_cast<float>(firstColdSlots) / static_cast<float>(firstColdSlots + secondColdSlots), 0.25F, 0.01F);

    for (LifeCycleId lifeCycle{0}; lifeCycle < LifeCycleId{2}; ++lifeCycle)
    {
        PoolGroupIndex const coldGroup = storage.getPoolGroupIndex(CacheLevel{1}, lifeCycle);
        EXPECT_NE(coldGroup, storage.getPoolGroupIndex(CacheLevel{1}, LifeCycleId{1 - lifeCycle.value()}));
        EXPECT_EQ(storage.numPools(CacheLevel{1}, coldGroup), PoolIndex{1});
        auto const coldSlotSizes = storage.slotSize(CacheLevel{1}, coldGroup);
        ASSERT_EQ(coldSlotSizes.size(), PoolIndex{1});
        size_t const expectedBytes = lifeCycle == LifeCycleId{0} ? 1024 : 2048;
        EXPECT_EQ(coldSlotSizes.at(PoolIndex{0}), expectedBytes);
    }
}

TEST(KvCacheManagerV2ColdPageTest, MigrationStatsUseColdPageBytes)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeSplitColdGroupingConfig();
    constexpr size_t kHotPageBytes = 2U << 20U;
    constexpr size_t kGpuQuota = 12U << 20U;
    auto& firstLayer = std::get<AttentionLayerConfig>(config.layers[0]);
    auto& secondLayer = std::get<AttentionLayerConfig>(config.layers[1]);
    firstLayer.buffers[0].size = kHotPageBytes;
    firstLayer.slidingWindowSize = std::nullopt;
    secondLayer.buffers[0].size = kHotPageBytes;
    secondLayer.slidingWindowSize = 8;
    config.cacheTiers[0] = GpuCacheTierConfig{kGpuQuota};
    config.initialPoolRatio = std::vector<float>{0.5F, 0.5F};

    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::make_unique<SplitColdPageCodec>());
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto streamGuard = FuncGuard([stream]() { cudaStreamDestroy(stream); });

    constexpr int kNumPages = 3;
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < kNumPages * manager->tokensPerBlock(); ++token)
    {
        tokens.emplace_back(TokenId{token});
    }

    auto first = manager->createKvCache();
    ASSERT_TRUE(first->resume(reinterpret_cast<CUstream>(stream)));
    EXPECT_TRUE(first->resize(static_cast<int>(tokens.size())));
    first->commit(toSpan(tokens));
    first->suspend();
    manager->getAndResetIterationStats();

    auto second = manager->createKvCache();
    ASSERT_TRUE(second->resume(reinterpret_cast<CUstream>(stream)));
    EXPECT_TRUE(second->resize(static_cast<int>(tokens.size())));

    auto offloadStats = manager->getAndResetIterationStats();
    ASSERT_EQ(offloadStats.size(), 2);
    EXPECT_EQ(offloadStats.at(LifeCycleId{0}).iterOffloadBlocks, kNumPages);
    EXPECT_EQ(offloadStats.at(LifeCycleId{0}).iterOffloadBytes, kNumPages * 1024);
    EXPECT_EQ(offloadStats.at(LifeCycleId{1}).iterOffloadBlocks, kNumPages);
    EXPECT_EQ(offloadStats.at(LifeCycleId{1}).iterOffloadBytes, kNumPages * 2048);

    second->close();
    ASSERT_TRUE(first->resume(reinterpret_cast<CUstream>(stream)));
    auto onboardStats = manager->getAndResetIterationStats();
    ASSERT_EQ(onboardStats.size(), 2);
    EXPECT_EQ(onboardStats.at(LifeCycleId{0}).iterOnboardBlocks, kNumPages);
    EXPECT_EQ(onboardStats.at(LifeCycleId{0}).iterOnboardBytes, kNumPages * 1024);
    constexpr int kSwaOnboardPages = 2;
    EXPECT_EQ(onboardStats.at(LifeCycleId{1}).iterOnboardBlocks, kSwaOnboardPages);
    EXPECT_EQ(onboardStats.at(LifeCycleId{1}).iterOnboardBytes, kSwaOnboardPages * 2048);
    first->close();
}

TEST(KvCacheManagerV2ColdPageTest, EvictionRoutesLifecycleQueuesToDifferentColdPoolGroups)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(
        makeSplitColdGroupingConfig(), nullptr, std::make_unique<SplitColdPageCodec>());
    auto& storage = manager->storage();
    TypedVec<LifeCycleId, SlotCount> oneSlotPerLifeCycle(LifeCycleId{2}, 1);
    auto hotSlots = storage.newGpuSlots(oneSlotPerLifeCycle);
    auto firstPage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[LifeCycleId{0}].front(), LifeCycleId{0});
    auto secondPage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[LifeCycleId{1}].front(), LifeCycleId{1});

    TypedVec<PoolGroupIndex, SlotCount> evictBoth(storage.numPoolGroups(kHotLevel), 0);
    evictBoth[storage.getPoolGroupIndex(kHotLevel, LifeCycleId{0})] = 2;
    storage.forceEvict(kHotLevel, evictBoth);

    EXPECT_EQ(firstPage->cacheLevel, CacheLevel{1});
    EXPECT_EQ(secondPage->cacheLevel, CacheLevel{1});
    EXPECT_NE(storage.getPoolGroupIndex(CacheLevel{1}, firstPage->lifeCycle),
        storage.getPoolGroupIndex(CacheLevel{1}, secondPage->lifeCycle));
    EXPECT_TRUE(firstPage->scheduledForEviction());
    EXPECT_TRUE(secondPage->scheduledForEviction());
}

//! A max_quota below quota is a config that can never be satisfied: the initial
//! size would not fit the reservation. Rejecting it here names the field,
//! whereas letting it through fails later inside HostMem.
TEST(KvCacheManagerV2ColdPageTest, HostTierConfigRejectsMaxQuotaBelowQuota)
{
    HostCacheTierConfig config{size_t{256} << 20, size_t{128} << 20};
    EXPECT_THROW(config.validate(), std::invalid_argument);

    config.maxQuota = config.quota; // equal is the tightest valid bound
    EXPECT_NO_THROW(config.validate());

    config.maxQuota = std::nullopt; // unset means the host memory the OS reports
    EXPECT_NO_THROW(config.validate());
}

//! A resize past the tier's reserved maximum must be refused up front. The
//! address space was reserved once so the base address never moves, so growing
//! past it is not serviceable -- and discovering that inside a pool would leave
//! earlier pool groups already resized.
TEST(KvCacheManagerV2ColdPageTest, ResizeBeyondMaxQuotaIsRefused)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr size_t kQuota = size_t{128} << 20;
    constexpr size_t kMaxQuota = size_t{256} << 20;

    auto config = makeSplitColdGroupingConfig();
    config.cacheTiers[1] = HostCacheTierConfig{kQuota, kMaxQuota};
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    CacheLevel const coldLevel{1};

    EXPECT_TRUE(manager->resize(coldLevel, kMaxQuota, /*bestEfforts=*/false)) << "growing to the bound must work";
    auto const quotaBefore = manager->getQuota(coldLevel);
    EXPECT_FALSE(manager->resize(coldLevel, kMaxQuota * 2, /*bestEfforts=*/false)) << "growing past it must not";
    // Equality, not a bound: a refusal that had already resized some pool groups
    // would still satisfy <= kMaxQuota and go unnoticed.
    EXPECT_EQ(manager->getQuota(coldLevel), quotaBefore) << "the refused resize must not have taken effect";
}

//! A max_quota that is not a multiple of the tier's commit unit must still be
//! reachable. Every requested quota is rounded up to that unit before it is
//! compared, so a bound left unrounded would sit below the smallest quota that
//! can represent it and refuse a resize the pools have the address space for.
TEST(KvCacheManagerV2ColdPageTest, ResizeToUnalignedMaxQuotaIsAllowed)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr size_t kQuota = size_t{128} << 20;
    // One MiB past a multiple of the 2 MiB commit unit this quota resolves to.
    constexpr size_t kMaxQuota = kQuota + (size_t{1} << 20);

    auto config = makeSplitColdGroupingConfig();
    config.cacheTiers[1] = HostCacheTierConfig{kQuota, kMaxQuota};
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    CacheLevel const coldLevel{1};

    EXPECT_TRUE(manager->resize(coldLevel, kMaxQuota, /*bestEfforts=*/false))
        << "growing to an unaligned bound must work";
    EXPECT_GE(manager->getQuota(coldLevel), kMaxQuota) << "the resize must have taken effect";
}

//! Each pool group needs an allocation unit of its own, so a level with two of
//! them cannot be built below two units however small the quota is. A maximum
//! under that total is unsatisfiable: the quota is raised to the minimum at
//! construction and would then sit above the maximum, so the level comes up
//! already unable to resize to the size it reports.
TEST(KvCacheManagerV2ColdPageTest, MaxQuotaBelowTheStructuralMinimumIsRejected)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    // Two cold pool groups, from the two distinct sliding windows.
    size_t const kOneUnit = HostMem::sizeGranularity();

    auto makeConfig = [kOneUnit](std::optional<size_t> maxQuota)
    {
        auto config = makeSplitColdGroupingConfig();
        config.cacheTiers[1] = HostCacheTierConfig{kOneUnit, maxQuota};
        return config;
    };
    // Cold pool groups are keyed by (reuse policy, cold page bytes), so the split
    // comes from the codec reporting different page sizes, not from the config
    // alone. Without it both lifecycles share one group and one unit suffices.
    auto build = [&makeConfig](std::optional<size_t> maxQuota) {
        return KvCacheManager{makeConfig(maxQuota), nullptr, std::make_unique<SplitColdPageCodec>()};
    };

    EXPECT_THROW(build(kOneUnit), std::invalid_argument) << "a maximum of one unit cannot cover two pool groups";

    // Controls: enough for both groups, and no maximum at all.
    EXPECT_NO_THROW(build(2 * kOneUnit));
    EXPECT_NO_THROW(build(std::nullopt));
}

TEST(KvCacheManagerV2ColdPageTest, FallenPagesRetainHighestPriorityAcrossLifecycleQueues)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeSingleColdSlotConfig();
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto& storage = manager->storage();
    CacheLevel const coldLevel{1};
    ASSERT_EQ(storage.numPoolGroups(kHotLevel), PoolGroupIndex{1});
    ASSERT_EQ(storage.numPoolGroups(coldLevel), PoolGroupIndex{1});
    ASSERT_EQ(storage.getStatistics(coldLevel).total, 1);

    TypedVec<LifeCycleId, SlotCount> oneSlotPerLifeCycle(LifeCycleId{2}, 1);
    auto hotSlots = storage.newGpuSlots(oneSlotPerLifeCycle);
    auto highPriorityPage = makeCommittedPage(
        *manager, storage, kHotLevel, hotSlots[LifeCycleId{0}].front(), LifeCycleId{0}, /*priority=*/100);
    auto lowPriorityPage = makeCommittedPage(
        *manager, storage, kHotLevel, hotSlots[LifeCycleId{1}].front(), LifeCycleId{1}, /*priority=*/1);

    TypedVec<PoolGroupIndex, SlotCount> evictBoth(storage.numPoolGroups(kHotLevel), 2);
    storage.forceEvict(kHotLevel, evictBoth);

    EXPECT_EQ(highPriorityPage->cacheLevel, coldLevel);
    EXPECT_EQ(lowPriorityPage->cacheLevel, kHotLevel);
}

TEST(KvCacheManagerV2ColdPageTest, RecursiveFallenPageMergeResortsByPriority)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeSingleColdSlotConfig();
    config.cacheTiers.emplace_back(HostCacheTierConfig{HostMem::sizeGranularity()});
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto& storage = manager->storage();
    CacheLevel const firstColdLevel{1};
    CacheLevel const lastColdLevel{2};
    ASSERT_EQ(storage.getStatistics(firstColdLevel).total, 1);
    ASSERT_EQ(storage.getStatistics(lastColdLevel).total, 1);

    TypedVec<LifeCycleId, SlotCount> oneSlotForSecondLifeCycle(LifeCycleId{2}, 0);
    oneSlotForSecondLifeCycle[LifeCycleId{1}] = 1;
    auto coldSlots = storage.newSlots(firstColdLevel, oneSlotForSecondLifeCycle);
    auto highestPriorityPage = makeCommittedPage(*manager, storage, firstColdLevel, coldSlots[LifeCycleId{1}].front(),
        LifeCycleId{1}, /*priority=*/100, /*tokenBase=*/0);

    TypedVec<LifeCycleId, SlotCount> oneSlotPerLifeCycle(LifeCycleId{2}, 1);
    auto hotSlots = storage.newGpuSlots(oneSlotPerLifeCycle);
    auto middlePriorityPage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[LifeCycleId{0}].front(),
        LifeCycleId{0}, /*priority=*/50, /*tokenBase=*/100);
    auto lowestPriorityPage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[LifeCycleId{1}].front(),
        LifeCycleId{1}, /*priority=*/1, /*tokenBase=*/200);

    TypedVec<PoolGroupIndex, SlotCount> evictBoth(storage.numPoolGroups(kHotLevel), 2);
    storage.forceEvict(kHotLevel, evictBoth);

    EXPECT_EQ(middlePriorityPage->cacheLevel, firstColdLevel);
    EXPECT_EQ(highestPriorityPage->cacheLevel, lastColdLevel);
    EXPECT_EQ(lowestPriorityPage->cacheLevel, kHotLevel);
}

TEST(KvCacheManagerV2ColdPageTest, SparseHistoryPassesThroughTheCodec)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<RecordingColdPageCodec>();
    auto* codecPtr = codec.get();
    auto manager = std::make_shared<KvCacheManager>(makeSparseTieredConfig(), nullptr, std::move(codec));
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    LayerGroupId const layerGroup{lifeCycle.value()};
    CacheLevel const coldLevel{1};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);

    // The arrangement this test is about: the sparse buffer's *hot* pages are host memory,
    // sitting in a GPU-tier group beside a device-backed pool. Without this the test would
    // pass for a dense life cycle too, and prove nothing.
    PoolGroupIndex const hotPg = storage.getPoolGroupIndex(kHotLevel, lifeCycle);
    ASSERT_EQ(storage.cacheTier(kHotLevel), CacheTier::GPU_MEM);
    auto const hotPools = storage.slotSize(kHotLevel, hotPg);
    ASSERT_EQ(hotPools.size(), PoolIndex{2}) << "a sparse pool beside a device-backed one";
    int sparsePools = 0;
    for (PoolIndex pool{0}; pool < hotPools.size(); ++pool)
    {
        // A pool whose medium differs from its level's tier is the host-backed one.
        sparsePools += storage.poolMedium(kHotLevel, hotPg, pool) != storage.cacheTier(kHotLevel) ? 1 : 0;
    }
    ASSERT_EQ(sparsePools, 1) << "exactly one of the group's pools holds host-backed pages";

    // A hot page of the sparse life cycle. Host-backed, but an ordinary hot page: locked at
    // the hot level and migratable like any other.
    auto hotSlots = storage.newSlots(kHotLevel, oneSlot);
    auto page = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[lifeCycle].front());
    storage.excludeFromEviction(*page);
    EXPECT_EQ(codecPtr->encoded(layerGroup), 0U);

    // Out of the hot tier: crossing the boundary is what invokes the codec.
    storage.batchedMigrate(coldLevel, {page}, {});
    EXPECT_EQ(page->cacheLevel, coldLevel);
    EXPECT_EQ(codecPtr->encoded(layerGroup), 1U) << "sparse history must reach the codec on demotion";

    // And back, which is the decode half.
    storage.batchedMigrate(kHotLevel, {page}, {});
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(codecPtr->decoded(layerGroup), 1U) << "sparse history must reach the codec on promotion";
    page->readyEvent.synchronize();
}

} // namespace

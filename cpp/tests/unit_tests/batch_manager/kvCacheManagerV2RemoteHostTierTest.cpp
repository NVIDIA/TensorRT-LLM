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

// The remote host tier: a last HOST_MEM level over host memory chunks another process lends as fabric handles.
// Here this process stands in for the lender: it creates HOST_NUMA + FABRIC chunks, writes the verification word,
// exports them and hands the handles to the level, which imports them again. Tests that need the lent memory skip
// when the driver cannot create or import fabric host memory (no IMEX channel, no multi-node NVLink); with
// TLLM_TEST_REMOTE_HOST_REQUIRE_FABRIC=1 in the environment they fail instead, so a run on a node with fabric host
// memory cannot pass without covering the level.

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/coldPageCodec.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/common.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/exceptions.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/page.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::common::TllmException;

constexpr int kTokensPerBlock = 4;
constexpr size_t kPageBytes = 4096;
constexpr uint64_t kNonce = 0x5245'4d4f'5445'0001ULL;
constexpr CacheLevel kLocalHostLevel{1};
constexpr CacheLevel kRemoteLevel{2};
constexpr size_t kDummyChunkBytes = 2UL << 20;
constexpr size_t kLargeChunkBytes = 1UL << 30;

//! TLLM_TEST_REMOTE_HOST_REQUIRE_FABRIC=1: a missing fabric export or import fails the test instead of skipping it.
bool requireFabric()
{
    char const* value = std::getenv("TLLM_TEST_REMOTE_HOST_REQUIRE_FABRIC");
    return value != nullptr && std::string(value) == "1";
}

//! Resident set size of this process in bytes, from /proc/self/status.
size_t residentBytes()
{
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key)
    {
        if (key == "VmRSS:")
        {
            size_t kib = 0;
            status >> kib;
            return kib << 10;
        }
        std::string rest;
        std::getline(status, rest);
    }
    return 0;
}

//! Lends host memory chunks of this process the way a lending server does: HOST_NUMA + FABRIC, no RDMA flag,
//! first 8 bytes of chunk i = nonce ^ i. Holds the chunks until destroyed.
class LocalLender
{
public:
    //! `chunkBytes` 0 means the smallest chunk the driver allows (at least 2 MiB).
    static std::unique_ptr<LocalLender> create(size_t numChunks, std::string& whyNot, size_t chunkBytes = 0)
    {
        CUdevice device{};
        if (cuCtxGetDevice(&device) != CUDA_SUCCESS)
        {
            whyNot = "no current CUDA context";
            return nullptr;
        }
        int numaId = -1;
        if (cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device) != CUDA_SUCCESS || numaId < 0)
        {
            numaId = 0;
        }
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id = numaId;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.allocFlags.gpuDirectRDMACapable = 0;
        size_t granularity = 0;
        if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS
            || granularity == 0)
        {
            whyNot = "HOST_NUMA + FABRIC allocations are not supported";
            return nullptr;
        }
        size_t const minChunk = std::max<size_t>(granularity, 2UL << 20);
        if (chunkBytes != 0 && chunkBytes % granularity != 0)
        {
            whyNot = "chunk size " + std::to_string(chunkBytes) + " is not a multiple of the granularity";
            return nullptr;
        }
        std::unique_ptr<LocalLender> lender(new LocalLender(chunkBytes != 0 ? chunkBytes : minChunk, device));
        for (size_t index = 0; index < numChunks; ++index)
        {
            CUmemGenericAllocationHandle handle{};
            CUresult const err = cuMemCreate(&handle, lender->mChunkBytes, &prop, 0);
            if (err != CUDA_SUCCESS)
            {
                whyNot = "cuMemCreate(HOST_NUMA, FABRIC) failed with " + std::to_string(static_cast<int>(err));
                return nullptr;
            }
            lender->mHandles.push_back(handle);
            CUmemFabricHandle fabric{};
            if (cuMemExportToShareableHandle(&fabric, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0) != CUDA_SUCCESS)
            {
                whyNot = "fabric handle export failed";
                return nullptr;
            }
            lender->mFabricHandles.emplace_back(reinterpret_cast<char const*>(fabric.data), sizeof(fabric.data));
            if (!lender->writeHead(handle, kNonce ^ static_cast<uint64_t>(index)))
            {
                whyNot = "writing the verification word failed";
                return nullptr;
            }
        }
        return lender;
    }

    ~LocalLender()
    {
        for (auto handle : mHandles)
        {
            cuMemRelease(handle);
        }
    }

    LocalLender(LocalLender const&) = delete;
    LocalLender& operator=(LocalLender const&) = delete;

    [[nodiscard]] RemoteHostTierConfig remoteConfig() const
    {
        return RemoteHostTierConfig{mFabricHandles, mChunkBytes, kNonce};
    }

    [[nodiscard]] size_t capacity() const noexcept
    {
        return mChunkBytes * mHandles.size();
    }

    [[nodiscard]] size_t chunkBytes() const noexcept
    {
        return mChunkBytes;
    }

    //! Overwrites the verification word of chunk `index` (a stale or foreign lender).
    bool corrupt(size_t index)
    {
        return writeHead(mHandles.at(index), ~(kNonce ^ static_cast<uint64_t>(index)));
    }

    //! Writes the expected verification word of chunk `index` again.
    bool repair(size_t index)
    {
        return writeHead(mHandles.at(index), kNonce ^ static_cast<uint64_t>(index));
    }

private:
    LocalLender(size_t chunkBytes, CUdevice device)
        : mChunkBytes(chunkBytes)
        , mDevice(device)
    {
    }

    bool writeHead(CUmemGenericAllocationHandle handle, uint64_t value) const
    {
        CUdeviceptr va = 0;
        if (cuMemAddressReserve(&va, mChunkBytes, 0, 0, 0) != CUDA_SUCCESS)
        {
            return false;
        }
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = mDevice;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        bool const mapped = cuMemMap(va, mChunkBytes, 0, handle, 0) == CUDA_SUCCESS;
        bool ok = mapped && cuMemSetAccess(va, mChunkBytes, &access, 1) == CUDA_SUCCESS;
        ok = ok && cuMemcpyHtoD(va, &value, sizeof(value)) == CUDA_SUCCESS;
        ok = ok && cuCtxSynchronize() == CUDA_SUCCESS;
        if (mapped)
        {
            ok = cuMemUnmap(va, mChunkBytes) == CUDA_SUCCESS && ok;
        }
        return cuMemAddressFree(va, mChunkBytes) == CUDA_SUCCESS && ok;
    }

    size_t mChunkBytes;
    CUdevice mDevice;
    std::vector<CUmemGenericAllocationHandle> mHandles;
    std::vector<std::string> mFabricHandles;
};

//! GPU, local host and remote host levels; one 4 KiB full-attention buffer, plus a sliding-window layer with 8 KiB
//! pages when `twoPoolGroups`. The sliding window gives that layer its own life cycle and so a second pool group at
//! every level (a second full-attention layer would share the first one's life cycle and pool group).
KVCacheManagerConfig remoteConfig(
    RemoteHostTierConfig remote, size_t remoteQuota, bool twoPoolGroups = false, bool enableStats = true)
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = kTokensPerBlock;
    config.enableStats = enableStats;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4UL << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{2UL << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{remoteQuota, std::move(remote)});
    AttentionLayerConfig layer;
    layer.layerId = 0;
    layer.buffers.push_back(BufferConfig{"key", kPageBytes, std::nullopt});
    config.layers.emplace_back(std::move(layer));
    if (twoPoolGroups)
    {
        AttentionLayerConfig wide;
        wide.layerId = 1;
        wide.slidingWindowSize = 2 * kTokensPerBlock;
        wide.buffers.push_back(BufferConfig{"key", 2 * kPageBytes, std::nullopt});
        config.layers.emplace_back(std::move(wide));
    }
    return config;
}

//! Dummy handles of the right size, for checks that never import them.
RemoteHostTierConfig dummyRemote(size_t numChunks, size_t chunkBytes = kDummyChunkBytes)
{
    return RemoteHostTierConfig{
        std::vector<std::string>(numChunks, std::string(kFabricHandleBytes, '\0')), chunkBytes, kNonce};
}

//! A committed page in its own root-level block at `level`, scheduled for eviction.
SharedPtr<CommittedPage> makeCommittedPage(
    KvCacheManager& manager, CacheLevel level, int tokenBase, LifeCycleId lifeCycle = LifeCycleId{0})
{
    auto& storage = manager.storage();
    TypedVec<LifeCycleId, SlotCount> counts(storage.numLifeCycles(), 0);
    counts[lifeCycle] = 1;
    auto slots = storage.newSlots(level, counts);
    RootBlock& root = manager.radixTree().addOrGetExisting({});
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < manager.tokensPerBlock(); ++token)
    {
        tokens.emplace_back(TokenId{tokenBase + token});
    }
    auto block = addOrGetExistingBlock(&root, std::move(tokens), /*knownNoDigest=*/true);
    auto page = makeShared<CommittedPage>(
        &storage, block, lifeCycle, level, static_cast<int>(block->tokens.size()), kPriorityDefault);
    page->setSlot(slots[lifeCycle].front());
    block->storage[lifeCycle] = page.get();
    storage.scheduleForEviction(*page);
    return page;
}

void evict(StorageManager& storage, CacheLevel level, int count, LifeCycleId lifeCycle = LifeCycleId{0})
{
    TypedVec<PoolGroupIndex, SlotCount> numPages(storage.numPoolGroups(level), 0);
    numPages[storage.getPoolGroupIndex(level, lifeCycle)] = count;
    storage.forceEvict(level, numPages);
}

class KvCacheManagerV2RemoteHostTierTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaFree(nullptr), cudaSuccess); // make the primary context current for the driver API
        ASSERT_EQ(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking), cudaSuccess);
    }

    void TearDown() override
    {
        EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    //! A lender of `numChunks` chunks; nullptr (and the test marked skipped) when fabric host memory is unavailable.
    std::unique_ptr<LocalLender> lend(size_t numChunks, size_t chunkBytes = 0)
    {
        std::string whyNot;
        auto lender = LocalLender::create(numChunks, whyNot, chunkBytes);
        if (!lender)
        {
            mSkipReason = "fabric host memory unavailable: " + whyNot;
        }
        return lender;
    }

    //! A manager over the lender's chunks; nullptr (and the test marked skipped) when this driver cannot import
    //! fabric handles into the process that exported them.
    std::shared_ptr<KvCacheManager> makeManager(
        LocalLender const& lender, bool twoPoolGroups = false, bool enableStats = true)
    {
        try
        {
            return std::make_shared<KvCacheManager>(
                remoteConfig(lender.remoteConfig(), lender.capacity(), twoPoolGroups, enableStats));
        }
        catch (CuError const& error)
        {
            if (isImportUnavailable(error))
            {
                mSkipReason = std::string("same-process fabric import unavailable: ") + error.what();
                return nullptr;
            }
            throw;
        }
    }

    static bool isImportUnavailable(CuError const& error)
    {
        return error.errorCode == CUDA_ERROR_NOT_SUPPORTED || error.errorCode == CUDA_ERROR_NOT_PERMITTED;
    }

    //! Commits `tokens` through a new cache and closes it, leaving its pages reusable.
    void commitAndClose(KvCacheManager& manager, std::vector<TokenIdExt> const& tokens)
    {
        auto cache = manager.createKvCache({}, toSpan(tokens));
        ASSERT_TRUE(cache->resume(stream()));
        int const size = static_cast<int>(tokens.size());
        int const reused = cache->numCommittedTokens();
        ASSERT_TRUE(cache->resize(size));
        if (reused < size)
        {
            cache->commit(TokenSpan{tokens.data() + reused, size - reused});
        }
        cache->close();
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    }

    static std::vector<TokenIdExt> makeTokens(int base, int count)
    {
        std::vector<TokenIdExt> tokens;
        for (int i = 0; i < count; ++i)
        {
            tokens.emplace_back(TokenId{base + i});
        }
        return tokens;
    }

    //! The cache level of every block page matched by `tokens`.
    static std::vector<CacheLevel> chainLevels(KvCacheManager& manager, std::vector<TokenIdExt> const& tokens)
    {
        auto const apiLock = manager.lockExclusive();
        auto const match = manager.matchReuse({}, toSpan(tokens), /*knownNoDigest=*/true);
        std::vector<CacheLevel> levels;
        for (Block* block : match.blocks)
        {
            levels.push_back(block->getPage(LifeCycleId{0})->cacheLevel);
        }
        return levels;
    }

    CUstream stream() const
    {
        return reinterpret_cast<CUstream>(mStream);
    }

    static MemAddress pageAddress(StorageManager& storage, CacheLevel level, SlotId slot)
    {
        PoolGroupIndex const pg = storage.getPoolGroupIndex(level, LifeCycleId{0});
        return std::get<MemAddress>(storage.slotAddress(level, pg, slot, PoolIndex{0}));
    }

    void fill(MemAddress address, uint8_t value)
    {
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(address), value, kPageBytes, mStream), cudaSuccess);
    }

    bool holds(MemAddress address, uint8_t value)
    {
        std::vector<uint8_t> bytes(kPageBytes);
        // Migrations run on the manager's own streams; wait for all of them before reading.
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(bytes.data(), reinterpret_cast<void const*>(address), kPageBytes, cudaMemcpyDefault),
            cudaSuccess);
        return std::all_of(bytes.begin(), bytes.end(), [value](uint8_t byte) { return byte == value; });
    }

    cudaStream_t mStream{};
    std::string mSkipReason;
};

#define REMOTE_HOST_TIER_SKIP_IF_NULL(ptr)                                                                             \
    if (!(ptr))                                                                                                        \
    {                                                                                                                  \
        if (requireFabric())                                                                                           \
        {                                                                                                              \
            FAIL() << mSkipReason << " (TLLM_TEST_REMOTE_HOST_REQUIRE_FABRIC=1)";                                      \
        }                                                                                                              \
        GTEST_SKIP() << mSkipReason;                                                                                   \
    }

TEST(KvCacheManagerV2RemoteHostTierConfigTest, ValidatesPlacementAndHandles)
{
    size_t const quota = 4 * kDummyChunkBytes;
    // Accepted: GPU, local host, remote host.
    EXPECT_NO_THROW(remoteConfig(dummyRemote(4), quota).validate());

    // The remote tier must be last and must follow exactly one local host tier.
    auto notLast = remoteConfig(dummyRemote(4), quota);
    notLast.cacheTiers.emplace_back(HostCacheTierConfig{2UL << 20});
    EXPECT_THROW(notLast.validate(), std::invalid_argument);

    auto noLocalHost = remoteConfig(dummyRemote(4), quota);
    noLocalHost.cacheTiers.erase(noLocalHost.cacheTiers.begin() + 1);
    EXPECT_THROW(noLocalHost.validate(), std::invalid_argument);

    auto twoRemote = remoteConfig(dummyRemote(4), quota);
    twoRemote.cacheTiers[1] = HostCacheTierConfig{quota, dummyRemote(4)};
    EXPECT_THROW(twoRemote.validate(), std::invalid_argument);

    // Handles must be CUmemFabricHandle-sized, and the quota cannot exceed the lent capacity.
    auto badHandle = dummyRemote(4);
    badHandle.fabricHandles[2].resize(kFabricHandleBytes - 1);
    EXPECT_THROW(remoteConfig(badHandle, quota).validate(), std::invalid_argument);
    EXPECT_THROW(remoteConfig(dummyRemote(4), quota + kDummyChunkBytes).validate(), std::invalid_argument);
    EXPECT_THROW(remoteConfig(dummyRemote(0), kDummyChunkBytes).validate(), std::invalid_argument);
    EXPECT_THROW(remoteConfig(dummyRemote(4, 0), kDummyChunkBytes).validate(), std::invalid_argument);

    // A local host tier is unchanged.
    EXPECT_FALSE(HostCacheTierConfig{2UL << 20}.isRemote());
    EXPECT_NO_THROW(HostCacheTierConfig{2UL << 20}.assertValid());
}

TEST_F(KvCacheManagerV2RemoteHostTierTest, RefusesColdPageCodec)
{
    // Refused before anything is imported, so dummy handles suffice; an explicit default codec counts too.
    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(
                remoteConfig(dummyRemote(4), 4 * kDummyChunkBytes), nullptr, createDefaultKvCacheColdPageCodec());
            (void) manager;
        },
        TllmException);
}

TEST_F(KvCacheManagerV2RemoteHostTierTest, RejectsStaleChunks)
{
    auto lender = lend(4);
    REMOTE_HOST_TIER_SKIP_IF_NULL(lender);
    ASSERT_TRUE(lender->corrupt(3));
    try
    {
        auto manager = std::make_shared<KvCacheManager>(remoteConfig(lender->remoteConfig(), lender->capacity()));
        ADD_FAILURE() << "a chunk with the wrong verification word was accepted";
    }
    catch (CuError const& error)
    {
        if (isImportUnavailable(error) && !requireFabric())
        {
            GTEST_SKIP() << "same-process fabric import unavailable: " << error.what();
        }
        ADD_FAILURE() << "unexpected CUDA error " << error.what();
    }
    catch (TllmException const& error)
    {
        EXPECT_NE(std::string(error.what()).find("stale or foreign"), std::string::npos) << error.what();
    }
}

TEST_F(KvCacheManagerV2RemoteHostTierTest, ImportsEveryChunkAndExposesRanges)
{
    auto lender = lend(8);
    REMOTE_HOST_TIER_SKIP_IF_NULL(lender);
    auto manager = makeManager(*lender);
    REMOTE_HOST_TIER_SKIP_IF_NULL(manager);
    auto& storage = manager->storage();
    ASSERT_EQ(storage.numCacheLevels(), CacheLevel{3});
    EXPECT_EQ(storage.cacheTier(kRemoteLevel), CacheTier::HOST_MEM);
    EXPECT_TRUE(storage.isRemoteHostLevel(kRemoteLevel));
    EXPECT_FALSE(storage.isRemoteHostLevel(kLocalHostLevel));
    auto const stats = manager->remoteHostTierStats().storage;
    EXPECT_EQ(stats.level, kRemoteLevel.value());
    EXPECT_EQ(stats.numChunks, 8U);
    EXPECT_EQ(stats.lentBytes, lender->capacity());
    EXPECT_LE(manager->getQuota(kRemoteLevel), lender->capacity());
    auto const ranges = manager->remoteHostAddressRanges();
    ASSERT_EQ(ranges.size(), 1U);
    EXPECT_EQ(ranges.front().second, lender->capacity());
}

TEST_F(KvCacheManagerV2RemoteHostTierTest, CopiesThroughEveryRouteOnTheGpu)
{
    auto lender = lend(4);
    REMOTE_HOST_TIER_SKIP_IF_NULL(lender);
    auto manager = makeManager(*lender);
    REMOTE_HOST_TIER_SKIP_IF_NULL(manager);
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    TypedVec<LifeCycleId, SlotCount> one(storage.numLifeCycles(), 1);
    TypedVec<LifeCycleId, SlotCount> two(storage.numLifeCycles(), 2);
    auto gpu = storage.newSlots(kHotLevel, one);
    auto local = storage.newSlots(kLocalHostLevel, one);
    auto remote = storage.newSlots(kRemoteLevel, two);
    SlotId const gpuSlot = gpu[lc].front().slotId();
    SlotId const localSlot = local[lc].front().slotId();
    SlotId const remoteA = remote[lc][0].slotId();
    SlotId const remoteB = remote[lc][1].slotId();
    auto const before = manager->remoteHostTierStats().storage;

    // GPU -> local host -> remote (cascade demotion; the second hop is host-to-host).
    fill(pageAddress(storage, kHotLevel, gpuSlot), 0x5A);
    storage.copySlotData(lc, kLocalHostLevel, kHotLevel, localSlot, gpuSlot, stream());
    storage.copySlotData(lc, kRemoteLevel, kLocalHostLevel, remoteA, localSlot, stream());
    // Inside the remote level, as defragmentation moves a page.
    storage.copySlotData(lc, kRemoteLevel, kRemoteLevel, remoteB, remoteA, stream());
    // Remote -> GPU directly (promotion skips the local host level).
    fill(pageAddress(storage, kHotLevel, gpuSlot), 0x00);
    storage.copySlotData(lc, kHotLevel, kRemoteLevel, gpuSlot, remoteB, stream());
    EXPECT_TRUE(holds(pageAddress(storage, kHotLevel, gpuSlot), 0x5A));
    // Remote -> local host (promotion one level up), checked through the GPU.
    fill(pageAddress(storage, kHotLevel, gpuSlot), 0x00);
    storage.copySlotData(lc, kLocalHostLevel, kHotLevel, localSlot, gpuSlot, stream()); // clear the local copy
    storage.copySlotData(lc, kLocalHostLevel, kRemoteLevel, localSlot, remoteB, stream());
    storage.copySlotData(lc, kHotLevel, kLocalHostLevel, gpuSlot, localSlot, stream());
    EXPECT_TRUE(holds(pageAddress(storage, kHotLevel, gpuSlot), 0x5A));

    auto const after = manager->remoteHostTierStats().storage;
    // local->remote, remote->remote and remote->local ran on the GPU kernel; nothing used the host function.
    EXPECT_EQ(after.gpuKernelHostCopies - before.gpuKernelHostCopies, 3U);
    EXPECT_EQ(after.hostFunctionHostCopies, before.hostFunctionHostCopies);
    EXPECT_EQ(after.demotedPages - before.demotedPages, 1U);
    EXPECT_EQ(after.promotedPages - before.promotedPages, 2U);
    EXPECT_EQ(after.demotedBytes - before.demotedBytes, kPageBytes);

    storage.releaseSlot(lc, kHotLevel, std::move(gpu[lc].front()));
    storage.releaseSlot(lc, kLocalHostLevel, std::move(local[lc].front()));
    for (auto& slot : remote[lc])
    {
        storage.releaseSlot(lc, kRemoteLevel, std::move(slot));
    }
}

TEST_F(KvCacheManagerV2RemoteHostTierTest, PagesCascadeIntoRemoteAndHeldPagesStay)
{
    auto lender = lend(4);
    REMOTE_HOST_TIER_SKIP_IF_NULL(lender);
    auto manager = makeManager(*lender);
    REMOTE_HOST_TIER_SKIP_IF_NULL(manager);
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();

    auto page = makeCommittedPage(*manager, kHotLevel, 0);
    auto held = makeCommittedPage(*manager, kHotLevel, 100);
    fill(pageAddress(storage, kHotLevel, page->slotId()), 0xC3);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    auto holder = held->hold(); // a long-chain pin holds its pages this way
    ASSERT_EQ(held->status(), PageStatus::HELD);

    // Evict both down: GPU -> local host -> remote.
    evict(storage, kHotLevel, 2);
    ASSERT_EQ(page->cacheLevel, kLocalHostLevel);
    ASSERT_EQ(held->cacheLevel, kLocalHostLevel);
    evict(storage, kLocalHostLevel, 2);
    ASSERT_EQ(page->cacheLevel, kRemoteLevel);
    ASSERT_EQ(held->cacheLevel, kRemoteLevel);
    EXPECT_TRUE(holds(pageAddress(storage, kRemoteLevel, page->slotId()), 0xC3));

    // The remote level is the last level: a held page is not evictable there; a droppable page is.
    EXPECT_FALSE(storage.isEvictable(*held));
    EXPECT_TRUE(storage.isEvictable(*page));

    // Direct promotion of the remote page back to the GPU keeps its bytes.
    storage.excludeFromEviction(*page);
    storage.batchedMigrate(kHotLevel, {page}, {});
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_TRUE(holds(pageAddress(storage, kHotLevel, page->slotId()), 0xC3));
    storage.scheduleForEviction(*page);

    auto const stats = manager->remoteHostTierStats().storage;
    EXPECT_GE(stats.demotedPages, 2U);
    EXPECT_GE(stats.promotedPages, 1U);
    EXPECT_EQ(stats.hostFunctionHostCopies, 0U);
    holder.reset();
}

TEST_F(KvCacheManagerV2RemoteHostTierTest, DefragmentsAndResizesTheRemoteLevel)
{
    auto lender = lend(8);
    REMOTE_HOST_TIER_SKIP_IF_NULL(lender);
    auto manager = makeManager(*lender, /*twoPoolGroups=*/true);
    REMOTE_HOST_TIER_SKIP_IF_NULL(manager);
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    ASSERT_EQ(storage.numPoolGroups(kRemoteLevel), PoolGroupIndex{2});
    auto const before = manager->remoteHostTierStats().storage;
    PoolGroupIndex const pg = storage.getPoolGroupIndex(kRemoteLevel, LifeCycleId{0});
    ASSERT_GT(storage.numSlots(pg, kRemoteLevel), SlotCount{8});

    // Fill the pool group, keep the four pages in its highest slots and drop the rest, then shrink it to four slots:
    // the kept pages move down inside the level.
    std::vector<SharedPtr<CommittedPage>> pages;
    for (int i = 0; storage.getStatistics(kRemoteLevel, pg).free > 0; ++i)
    {
        pages.push_back(makeCommittedPage(*manager, kRemoteLevel, 1000 + 10 * i));
    }
    std::sort(pages.begin(), pages.end(), [](auto const& a, auto const& b) { return a->slotId() < b->slotId(); });
    std::vector<SharedPtr<CommittedPage>> kept(pages.end() - 4, pages.end());
    std::vector<uint8_t> values;
    for (size_t i = 0; i < kept.size(); ++i)
    {
        values.push_back(static_cast<uint8_t>(0x10 + i));
        fill(pageAddress(storage, kRemoteLevel, kept[i]->slotId()), values.back());
        storage.excludeFromEviction(*kept[i]);
    }
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int const numDropped = static_cast<int>(pages.size()) - 4;
    pages.clear();
    evict(storage, kRemoteLevel, numDropped);
    for (auto const& page : kept)
    {
        storage.scheduleForEviction(*page);
    }

    std::vector<SharedPtr<Page>> persistent;
    storage.shrinkPoolGroup(kRemoteLevel, pg, SlotCount{4}, persistent);
    for (size_t i = 0; i < kept.size(); ++i)
    {
        ASSERT_EQ(kept[i]->cacheLevel, kRemoteLevel);
        EXPECT_LT(kept[i]->slotId(), SlotId{4});
        EXPECT_TRUE(holds(pageAddress(storage, kRemoteLevel, kept[i]->slotId()), values[i]));
    }
    auto const afterDefrag = manager->remoteHostTierStats().storage;
    EXPECT_GT(afterDefrag.gpuKernelHostCopies, before.gpuKernelHostCopies);
    EXPECT_EQ(afterDefrag.hostFunctionHostCopies, before.hostFunctionHostCopies);

    // Ratio adjustment of the last level maps and unmaps lent chunks between the pool groups.
    size_t const quota = lender->capacity();
    for (float first : {0.25F, 0.75F, 0.5F})
    {
        TypedVec<PoolGroupIndex, float> ratio(PoolGroupIndex{2}, 0.F);
        ratio[PoolGroupIndex{0}] = first;
        ratio[PoolGroupIndex{1}] = 1.F - first;
        storage.adjustCacheLevel(kRemoteLevel, quota, ratio, nullptr);
        EXPECT_LE(manager->getQuota(kRemoteLevel), lender->capacity());
        for (size_t i = 0; i < kept.size(); ++i)
        {
            EXPECT_TRUE(holds(pageAddress(storage, kRemoteLevel, kept[i]->slotId()), values[i]));
        }
    }
}

TEST_F(KvCacheManagerV2RemoteHostTierTest, CountsRemoteHitsOnReuseWithStatsOnAndOff)
{
    // Commit a chain, push it down to the remote level, then match it again: the hit is counted whether or not
    // iteration stats are enabled.
    for (bool const enableStats : {true, false})
    {
        SCOPED_TRACE(enableStats ? "stats on" : "stats off");
        auto lender = lend(4);
        REMOTE_HOST_TIER_SKIP_IF_NULL(lender);
        auto manager = makeManager(*lender, /*twoPoolGroups=*/false, enableStats);
        REMOTE_HOST_TIER_SKIP_IF_NULL(manager);
        int const numBlocks = 3;
        auto const tokens = makeTokens(500, numBlocks * kTokensPerBlock);
        commitAndClose(*manager, tokens);
        if (HasFatalFailure())
        {
            return;
        }
        ASSERT_EQ(chainLevels(*manager, tokens).size(), static_cast<size_t>(numBlocks));
        {
            auto const apiLock = manager->lockExclusive();
            evict(manager->storage(), kHotLevel, numBlocks);
            evict(manager->storage(), kLocalHostLevel, numBlocks);
        }
        for (CacheLevel const level : chainLevels(*manager, tokens))
        {
            ASSERT_EQ(level, kRemoteLevel);
        }
        auto const before = manager->remoteHostTierStats();
        EXPECT_EQ(before.hitTokens, 0U);

        auto cache = manager->createKvCache({}, toSpan(tokens));
        ASSERT_GT(cache->cachedTokensByLevel().size(), kRemoteLevel);
        int64_t const remoteTokens = cache->cachedTokensByLevel().at(kRemoteLevel);
        EXPECT_GT(remoteTokens, 0);
        ASSERT_TRUE(cache->resume(stream()));
        EXPECT_GT(cache->numCommittedTokens(), 0);
        cache->close();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        auto const after = manager->remoteHostTierStats();
        EXPECT_EQ(after.hitTokens - before.hitTokens, static_cast<uint64_t>(remoteTokens));
        EXPECT_GE(after.storage.demotedPages, static_cast<uint64_t>(numBlocks));
        EXPECT_GT(after.storage.promotedPages, before.storage.promotedPages);
        EXPECT_GT(after.storage.gpuKernelHostCopies, 0U);
        EXPECT_EQ(after.storage.hostFunctionHostCopies, 0U);
    }
}

TEST_F(KvCacheManagerV2RemoteHostTierTest, RefusesQuotaAboveTheLentCapacity)
{
    auto lender = lend(8);
    REMOTE_HOST_TIER_SKIP_IF_NULL(lender);
    auto manager = makeManager(*lender, /*twoPoolGroups=*/true);
    REMOTE_HOST_TIER_SKIP_IF_NULL(manager);
    auto& storage = manager->storage();
    size_t const quotaBefore = manager->getQuota(kRemoteLevel);
    std::vector<SlotCount> slotsBefore;
    for (PoolGroupIndex pg{0}; pg < storage.numPoolGroups(kRemoteLevel); ++pg)
    {
        slotsBefore.push_back(storage.numSlots(pg, kRemoteLevel));
    }
    size_t const tooMuch = lender->capacity() + lender->chunkBytes();
    {
        auto const apiLock = manager->lockExclusive();
        TypedVec<PoolGroupIndex, float> ratio(PoolGroupIndex{2}, 0.5F);
        EXPECT_THROW(storage.adjustCacheLevel(kRemoteLevel, tooMuch, ratio, nullptr), std::invalid_argument);
    }
    // resize() reports the refusal; nothing was shrunk on the way.
    EXPECT_FALSE(manager->resize(kRemoteLevel, tooMuch));
    EXPECT_EQ(manager->getQuota(kRemoteLevel), quotaBefore);
    for (PoolGroupIndex pg{0}; pg < storage.numPoolGroups(kRemoteLevel); ++pg)
    {
        EXPECT_EQ(storage.numSlots(pg, kRemoteLevel), slotsBefore.at(pg.value()));
    }
    EXPECT_TRUE(manager->resize(kRemoteLevel, lender->capacity()));
}

TEST_F(KvCacheManagerV2RemoteHostTierTest, FailedImportCleansUpAndRetries)
{
    // Chunk 2 of 4 fails verification after chunks 0-2 were imported and chunk heads 0-1 were mapped and read. Every
    // failed attempt must release what it imported and unmap what it mapped, so repeating it changes nothing and a
    // later import of the same (repaired) handles works.
    auto lender = lend(4);
    REMOTE_HOST_TIER_SKIP_IF_NULL(lender);
    ASSERT_TRUE(lender->corrupt(2));
    for (int attempt = 0; attempt < 8; ++attempt)
    {
        try
        {
            auto manager = std::make_shared<KvCacheManager>(remoteConfig(lender->remoteConfig(), lender->capacity()));
            FAIL() << "a chunk with the wrong verification word was accepted";
        }
        catch (CuError const& error)
        {
            if (isImportUnavailable(error) && !requireFabric())
            {
                GTEST_SKIP() << "same-process fabric import unavailable: " << error.what();
            }
            FAIL() << "unexpected CUDA error " << error.what();
        }
        catch (TllmException const& error)
        {
            EXPECT_NE(std::string(error.what()).find("chunk 2"), std::string::npos) << error.what();
        }
    }
    ASSERT_TRUE(lender->repair(2));
    auto manager = makeManager(*lender);
    REMOTE_HOST_TIER_SKIP_IF_NULL(manager);
    EXPECT_EQ(manager->remoteHostTierStats().storage.numChunks, 4U);
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    TypedVec<LifeCycleId, SlotCount> one(storage.numLifeCycles(), 1);
    auto gpu = storage.newSlots(kHotLevel, one);
    auto remote = storage.newSlots(kRemoteLevel, one);
    fill(pageAddress(storage, kHotLevel, gpu[lc].front().slotId()), 0x3C);
    storage.copySlotData(lc, kRemoteLevel, kHotLevel, remote[lc].front().slotId(), gpu[lc].front().slotId(), stream());
    EXPECT_TRUE(holds(pageAddress(storage, kRemoteLevel, remote[lc].front().slotId()), 0x3C));
    storage.releaseSlot(lc, kHotLevel, std::move(gpu[lc].front()));
    storage.releaseSlot(lc, kRemoteLevel, std::move(remote[lc].front()));
}

TEST_F(KvCacheManagerV2RemoteHostTierTest, ImportsLargeChunks)
{
    // A large chunk (1 GiB): the pool reservation must be aligned to it. Skipped when this node cannot
    // allocate 2 GiB of fabric host memory, whatever TLLM_TEST_REMOTE_HOST_REQUIRE_FABRIC says.
    std::string whyNot;
    auto lender = LocalLender::create(2, whyNot, kLargeChunkBytes);
    if (!lender)
    {
        GTEST_SKIP() << "large fabric host chunks unavailable: " << whyNot;
    }
    auto manager = makeManager(*lender);
    REMOTE_HOST_TIER_SKIP_IF_NULL(manager);
    auto const stats = manager->remoteHostTierStats().storage;
    EXPECT_EQ(stats.numChunks, 2U);
    EXPECT_EQ(stats.lentBytes, 2 * kLargeChunkBytes);
    for (auto const& [base, bytes] : manager->remoteHostAddressRanges())
    {
        EXPECT_EQ(base % kLargeChunkBytes, 0U);
        EXPECT_EQ(bytes, 2 * kLargeChunkBytes);
    }

    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    PoolGroupIndex const pg = storage.getPoolGroupIndex(kRemoteLevel, lc);
    SlotCount const numSlots = storage.numSlots(pg, kRemoteLevel);
    TypedVec<LifeCycleId, SlotCount> one(storage.numLifeCycles(), 1);
    auto gpu = storage.newSlots(kHotLevel, one);
    auto local = storage.newSlots(kLocalHostLevel, one);
    // The last slot of the level lies in the second chunk.
    TypedVec<LifeCycleId, SlotCount> all(storage.numLifeCycles(), numSlots);
    auto remote = storage.newSlots(kRemoteLevel, all);
    auto const last = std::max_element(
        remote[lc].begin(), remote[lc].end(), [](Slot const& a, Slot const& b) { return a.slotId() < b.slotId(); });
    ASSERT_NE(last, remote[lc].end());
    SlotId const gpuSlot = gpu[lc].front().slotId();
    SlotId const localSlot = local[lc].front().slotId();
    fill(pageAddress(storage, kHotLevel, gpuSlot), 0x77);
    storage.copySlotData(lc, kLocalHostLevel, kHotLevel, localSlot, gpuSlot, stream());
    storage.copySlotData(lc, kRemoteLevel, kLocalHostLevel, last->slotId(), localSlot, stream());
    fill(pageAddress(storage, kHotLevel, gpuSlot), 0x00);
    storage.copySlotData(lc, kHotLevel, kRemoteLevel, gpuSlot, last->slotId(), stream());
    EXPECT_TRUE(holds(pageAddress(storage, kHotLevel, gpuSlot), 0x77));
    storage.releaseSlot(lc, kHotLevel, std::move(gpu[lc].front()));
    storage.releaseSlot(lc, kLocalHostLevel, std::move(local[lc].front()));
    for (auto& slot : remote[lc])
    {
        storage.releaseSlot(lc, kRemoteLevel, std::move(slot));
    }
}

TEST_F(KvCacheManagerV2RemoteHostTierTest, LogsResidentGrowthPerMillionRemoteSlots)
{
    auto lender = lend(32);
    REMOTE_HOST_TIER_SKIP_IF_NULL(lender);
    auto manager = makeManager(*lender);
    REMOTE_HOST_TIER_SKIP_IF_NULL(manager);
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    PoolGroupIndex const pg = storage.getPoolGroupIndex(kRemoteLevel, LifeCycleId{0});
    std::vector<SharedPtr<CommittedPage>> pages;
    pages.reserve(slotCountToSizeT(storage.numSlots(pg, kRemoteLevel)));
    size_t const rssBefore = residentBytes();
    for (int i = 0; storage.getStatistics(kRemoteLevel, pg).free > 0; ++i)
    {
        pages.push_back(makeCommittedPage(*manager, kRemoteLevel, 10 * i));
    }
    size_t const rssAfter = residentBytes();
    ASSERT_GT(pages.size(), 1000U);
    size_t const growth = rssAfter > rssBefore ? rssAfter - rssBefore : 0;
    double const gibPerMillion
        = static_cast<double>(growth) / static_cast<double>(pages.size()) * 1e6 / static_cast<double>(1UL << 30);
    // Sizes the context server's local host tier: KVCMv2 page, block and slot metadata per remote page.
    std::cout << "[remote_host_tier] resident growth " << growth << " B for " << pages.size()
              << " remote pages: " << gibPerMillion << " GiB per million remote slots" << std::endl;
    RecordProperty("rss_gib_per_million_remote_slots", std::to_string(gibPerMillion));
    pages.clear();
}

} // namespace

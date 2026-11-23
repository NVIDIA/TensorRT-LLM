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

// The store level: a last DISK-tier level whose slots are objects of an external store, reached through an IO table
// (storeLevelIo.h). Here the table is an in-memory fake with its own worker thread: objects are a map keyed by
// (pool group, slot), each op can be delayed or its registration refused, and every row is checked against the
// registered ranges and against another op of the same key still in flight.

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/coldPageCodec.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/common.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/exceptions.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/page.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storeLevel.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storeLevelIo.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::common::TllmException;
using namespace std::chrono_literals;

constexpr int kTokensPerBlock = 4;
constexpr size_t kPageBytes = 4096;
constexpr CacheLevel kHostLevel{1};
constexpr CacheLevel kStoreLevel{2};

// ---------------------------------------------------------------------------
// FakeStoreIo — an in-memory TllmKvStoreLevelIoV1.
// ---------------------------------------------------------------------------

struct FakeStoreIoOptions
{
    uint32_t maxRowsPerOp = 64;
    //! 0-based index of the registerMemory call that fails with registerCode; -1: none fails.
    int failRegisterAt = -1;
    int32_t registerCode = -600;
    int32_t configureCode = 0;
};

class FakeStoreIo
{
public:
    using Options = FakeStoreIoOptions;

    explicit FakeStoreIo(Options options = {})
        : mOptions(options)
    {
        mTable.abiVersion = TLLM_KV_STORE_LEVEL_IO_ABI_VERSION;
        mTable.structBytes = sizeof(TllmKvStoreLevelIoV1);
        mTable.maxRowsPerOp = options.maxRowsPerOp;
        mTable.ctx = this;
        mTable.registerMemory = [](void* ctx, void* base, uint64_t bytes)
        { return static_cast<FakeStoreIo*>(ctx)->registerMemory(base, bytes); };
        mTable.unregisterMemory
            = [](void* ctx, void* base) { return static_cast<FakeStoreIo*>(ctx)->unregisterMemory(base); };
        mTable.configure = [](void* ctx, uint32_t numPoolGroups, uint64_t const* slotBytes, uint64_t const* numSlots)
        { return static_cast<FakeStoreIo*>(ctx)->configure(numPoolGroups, slotBytes, numSlots); };
        mTable.submit = [](void* ctx, TllmKvStoreOp const* op) { static_cast<FakeStoreIo*>(ctx)->submit(*op); };
        mTable.drain = [](void* ctx, double timeoutS) { return static_cast<FakeStoreIo*>(ctx)->drain(timeoutS); };
        mTable.stats = [](void* ctx, TllmKvStoreLevelIoStats* out) { static_cast<FakeStoreIo*>(ctx)->stats(out); };
        mWorker = std::thread([this]() { run(); });
    }

    ~FakeStoreIo()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mStop = true;
        }
        mCv.notify_all();
        mWorker.join();
    }

    FakeStoreIo(FakeStoreIo const&) = delete;
    FakeStoreIo& operator=(FakeStoreIo const&) = delete;

    TllmKvStoreLevelIoV1& table() noexcept
    {
        return mTable;
    }

    std::uintptr_t address() noexcept
    {
        return reinterpret_cast<std::uintptr_t>(&mTable);
    }

    void setLatency(std::chrono::microseconds put, std::chrono::microseconds get)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mPutLatency = put;
        mGetLatency = get;
    }

    //! From now on every put fails as a degrade IO answers it: nothing stored, TLLM_KV_STORE_DONE_FAILED set.
    void setFailPuts(bool fail)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mFailPuts = fail;
    }

    [[nodiscard]] uint64_t numGetOps() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mStats.getOps;
    }

    [[nodiscard]] std::optional<std::vector<uint8_t>> object(uint32_t pg, uint64_t slot) const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto const it = mObjects.find({pg, slot});
        if (it == mObjects.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    [[nodiscard]] size_t numRegistered() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mRanges.size();
    }

    [[nodiscard]] size_t numRegisterCalls() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mRegisterCalls;
    }

    [[nodiscard]] size_t numUnregisterCalls() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mUnregisterCalls;
    }

    [[nodiscard]] uint64_t numOps() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mStats.putOps + mStats.getOps;
    }

    //! Ops submitted while another op of one of their keys was still in flight.
    [[nodiscard]] uint64_t numOverlaps() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mOverlaps;
    }

    //! Rows whose address range is not inside one registered range, and gets of keys never put.
    [[nodiscard]] uint64_t numBadRows() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mBadRows;
    }

    [[nodiscard]] std::vector<uint64_t> configuredSlotBytes() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mSlotBytes;
    }

    [[nodiscard]] std::vector<uint64_t> configuredNumSlots() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mNumSlots;
    }

private:
    using Key = std::pair<uint32_t, uint64_t>;

    struct QueuedOp
    {
        TllmKvStoreOp op;
        std::vector<TllmKvStoreRow> rows;
    };

    int32_t registerMemory(void* base, uint64_t bytes)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        int const index = static_cast<int>(mRegisterCalls++);
        if (index == mOptions.failRegisterAt)
        {
            return mOptions.registerCode;
        }
        if (base == nullptr || bytes == 0 || !mRanges.emplace(reinterpret_cast<std::uintptr_t>(base), bytes).second)
        {
            return -100010;
        }
        return 0;
    }

    int32_t unregisterMemory(void* base)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        ++mUnregisterCalls;
        return mRanges.erase(reinterpret_cast<std::uintptr_t>(base)) == 1 ? 0 : -100010;
    }

    int32_t configure(uint32_t numPoolGroups, uint64_t const* slotBytes, uint64_t const* numSlots)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mSlotBytes.assign(slotBytes, slotBytes + numPoolGroups);
        mNumSlots.assign(numSlots, numSlots + numPoolGroups);
        return mOptions.configureCode;
    }

    void submit(TllmKvStoreOp const& op)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            QueuedOp queued{op, std::vector<TllmKvStoreRow>(op.rows, op.rows + op.numRows)};
            queued.op.rows = nullptr;
            for (auto const& row : queued.rows)
            {
                if (mInFlight[Key{row.poolGroup, row.slot}]++ > 0)
                {
                    ++mOverlaps;
                }
            }
            mQueue.push_back(std::move(queued));
            mStats.queuedOps = mQueue.size();
            mStats.maxQueuedOps = std::max<uint64_t>(mStats.maxQueuedOps, mQueue.size());
        }
        mCv.notify_all();
    }

    uint64_t drain(double timeoutS)
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait_for(
            lock, std::chrono::duration<double>(timeoutS), [this]() { return mQueue.empty() && mRunning == 0; });
        return mQueue.size() + mRunning;
    }

    void stats(TllmKvStoreLevelIoStats* out)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        *out = mStats;
    }

    //! True when [address, address + bytes) lies in one registered range. Caller holds mMutex.
    [[nodiscard]] bool isRegistered(std::uintptr_t address, uint64_t bytes) const
    {
        auto it = mRanges.upper_bound(address);
        if (it == mRanges.begin())
        {
            return false;
        }
        --it;
        return address + bytes <= it->first + it->second;
    }

    void run()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        for (;;)
        {
            mCv.wait(lock, [this]() { return mStop || !mQueue.empty(); });
            if (mQueue.empty())
            {
                return;
            }
            QueuedOp queued = std::move(mQueue.front());
            mQueue.pop_front();
            mStats.queuedOps = mQueue.size();
            ++mRunning;
            bool const isPut = queued.op.kind == TLLM_KV_STORE_OP_PUT;
            auto const latency = isPut ? mPutLatency : mGetLatency;
            lock.unlock();
            if (latency.count() > 0)
            {
                std::this_thread::sleep_for(latency);
            }
            lock.lock();
            // As a degrade IO does: a failed put stores nothing and completes with the failure flag.
            bool const failed = isPut && mFailPuts;
            for (auto const& row : queued.rows)
            {
                Key const key{row.poolGroup, row.slot};
                uint64_t const bytes = row.poolGroup < mSlotBytes.size() ? mSlotBytes[row.poolGroup] : 0;
                if (bytes == 0 || !isRegistered(row.address, bytes))
                {
                    ++mBadRows;
                }
                else if (failed)
                {
                    mObjects.erase(key);
                }
                else if (isPut)
                {
                    auto const* source = reinterpret_cast<uint8_t const*>(row.address);
                    mObjects[key].assign(source, source + bytes);
                    mStats.putBytes += bytes;
                }
                else if (auto const it = mObjects.find(key); it != mObjects.end())
                {
                    std::memcpy(reinterpret_cast<void*>(row.address), it->second.data(), bytes);
                    mStats.getBytes += bytes;
                }
                else
                {
                    ++mBadRows;
                }
                --mInFlight[key];
            }
            (isPut ? mStats.putOps : mStats.getOps) += 1;
            (isPut ? mStats.putRows : mStats.getRows) += queued.rows.size();
            --mRunning;
            uint32_t const done = failed ? (queued.op.doneValue | TLLM_KV_STORE_DONE_FAILED) : queued.op.doneValue;
            std::atomic_ref<uint32_t>(*queued.op.doneWord).store(done, std::memory_order_release);
            mCv.notify_all();
        }
    }

    bool mFailPuts = false;
    Options mOptions;
    TllmKvStoreLevelIoV1 mTable{};
    mutable std::mutex mMutex;
    std::condition_variable mCv;
    std::deque<QueuedOp> mQueue;
    size_t mRunning = 0;
    bool mStop = false;
    std::chrono::microseconds mPutLatency{0};
    std::chrono::microseconds mGetLatency{0};
    std::map<Key, std::vector<uint8_t>> mObjects;
    std::map<Key, int> mInFlight;
    std::map<std::uintptr_t, uint64_t> mRanges;
    std::vector<uint64_t> mSlotBytes;
    std::vector<uint64_t> mNumSlots;
    size_t mRegisterCalls = 0;
    size_t mUnregisterCalls = 0;
    uint64_t mOverlaps = 0;
    uint64_t mBadRows = 0;
    TllmKvStoreLevelIoStats mStats{};
    std::thread mWorker;
};

// ---------------------------------------------------------------------------
// Configurations and helpers
// ---------------------------------------------------------------------------

//! GPU, local host and store levels; one 4 KiB full-attention buffer, plus a sliding-window layer with 8 KiB pages
//! when `twoPoolGroups` (its own life cycle, and so a second pool group at every level).
KVCacheManagerConfig storeConfig(
    bool twoPoolGroups = false, size_t hostQuota = 64UL << 10, size_t storeQuota = 2UL << 20)
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = kTokensPerBlock;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4UL << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{hostQuota, std::nullopt});
    config.cacheTiers.emplace_back(DiskCacheTierConfig{storeQuota, "", StoreTierConfig{"fake", 0}});
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

size_t pageBytesOf(StorageManager& storage, CacheLevel level, LifeCycleId lifeCycle)
{
    return storage.slotSize(level, storage.getPoolGroupIndex(level, lifeCycle)).at(PoolIndex{0});
}

MemAddress slotAddressOf(StorageManager& storage, CacheLevel level, LifeCycleId lifeCycle, SlotId slot)
{
    PoolGroupIndex const pg = storage.getPoolGroupIndex(level, lifeCycle);
    return std::get<MemAddress>(storage.slotAddress(level, pg, slot, PoolIndex{0}));
}

std::vector<uint8_t> pattern(size_t bytes, uint8_t seed)
{
    std::vector<uint8_t> out(bytes);
    for (size_t i = 0; i < bytes; ++i)
    {
        out[i] = static_cast<uint8_t>(seed + i * 7);
    }
    return out;
}

std::vector<uint8_t> readHost(MemAddress address, size_t bytes)
{
    auto const* begin = reinterpret_cast<uint8_t const*>(address);
    return std::vector<uint8_t>(begin, begin + bytes);
}

class KvCacheManagerV2StoreLevelTest : public ::testing::Test
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

    CUstream stream() const
    {
        return reinterpret_cast<CUstream>(mStream);
    }

    void sync()
    {
        // Migrations run on the manager's temporary streams; wait for all of them.
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
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

    cudaStream_t mStream{};
};

// ---------------------------------------------------------------------------
// (g) Configuration refusals
// ---------------------------------------------------------------------------

TEST(KvCacheManagerV2StoreLevelConfigTest, ValidatesPlacement)
{
    EXPECT_NO_THROW(storeConfig().validate());
    EXPECT_TRUE((DiskCacheTierConfig{2UL << 20, "", StoreTierConfig{}}.isStore()));
    EXPECT_FALSE((DiskCacheTierConfig{2UL << 20, "/tmp", std::nullopt}.isStore()));

    auto expectRefusal = [](KVCacheManagerConfig const& config, std::string const& text)
    {
        try
        {
            config.validate();
            ADD_FAILURE() << "accepted; expected: " << text;
        }
        catch (std::invalid_argument const& error)
        {
            EXPECT_NE(std::string(error.what()).find(text), std::string::npos) << error.what();
        }
    };

    // The store tier must be last.
    auto notLast = storeConfig();
    notLast.cacheTiers.emplace_back(HostCacheTierConfig{64UL << 10, std::nullopt});
    expectRefusal(notLast, "the store tier must be the last cache tier");

    // It must follow exactly one local host tier.
    auto noLocalHost = storeConfig();
    noLocalHost.cacheTiers.erase(noLocalHost.cacheTiers.begin() + 1);
    expectRefusal(noLocalHost, "the store tier must follow a GPU tier and one local host tier");

    // Never together with a remote host tier (the remote tier in the host tier's place), or with a disk tier.
    auto withRemote = storeConfig();
    withRemote.cacheTiers[1]
        = HostCacheTierConfig{2UL << 20, RemoteHostTierConfig{{std::string(kFabricHandleBytes, '\0')}, 2UL << 20, 0}};
    expectRefusal(withRemote, "the remote host tier and the store tier are mutually exclusive");

    auto withDisk = storeConfig();
    withDisk.cacheTiers.insert(withDisk.cacheTiers.begin() + 2, DiskCacheTierConfig{4UL << 20, "/tmp", std::nullopt});
    expectRefusal(withDisk, "the store tier cannot be combined with a disk tier");

    // The tier itself: a quota, and no path.
    EXPECT_THROW((DiskCacheTierConfig{0, "", StoreTierConfig{}}.assertValid()), std::invalid_argument);
    EXPECT_THROW((DiskCacheTierConfig{2UL << 20, "/tmp", StoreTierConfig{}}.assertValid()), std::invalid_argument);
    EXPECT_NO_THROW((DiskCacheTierConfig{2UL << 20, "", StoreTierConfig{}}.assertValid()));
}

TEST(KvCacheManagerV2StoreLevelConfigTest, StoreSlotPoolHasFixedSlotsAndNoMemory)
{
    StoreSlotPool pool(kPageBytes, 8);
    EXPECT_EQ(pool.numSlots(), 8);
    EXPECT_NO_THROW(pool.resize(8));
    EXPECT_THROW(pool.resize(4), LogicError);
    EXPECT_THROW(pool.resize(16), LogicError);
    auto const address = std::get<DiskAddress>(pool.slotAddress(SlotId{3}));
    EXPECT_EQ(address.fd, kStoreFd);
    EXPECT_EQ(address.pos, static_cast<ssize_t>(3 * kPageBytes));
}

TEST_F(KvCacheManagerV2StoreLevelTest, RefusesColdPageCodecAndResize)
{
    // An explicit codec is refused even when it is the default lossless one.
    EXPECT_THROW(
        {
            auto manager
                = std::make_shared<KvCacheManager>(storeConfig(), nullptr, createDefaultKvCacheColdPageCodec());
            (void) manager;
        },
        TllmException);

    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig(/*twoPoolGroups=*/true));
    ASSERT_EQ(manager->storage().storeLevel(), std::optional<CacheLevel>{kStoreLevel});
    EXPECT_EQ(manager->storeLevelStats().storage.level, kStoreLevel.value());
    EXPECT_FALSE(manager->storeLevelStats().ioAttached);

    // Neither the store level nor the host level above it can be resized or adjusted.
    EXPECT_THROW(manager->resize(kStoreLevel, 8UL << 20), std::invalid_argument);
    EXPECT_THROW(manager->resize(kHostLevel, 128UL << 10), std::invalid_argument);
    {
        auto const apiLock = manager->lockExclusive();
        TypedVec<PoolGroupIndex, float> ratio(PoolGroupIndex{2}, 0.5F);
        EXPECT_THROW(
            manager->storage().adjustCacheLevel(kStoreLevel, std::nullopt, ratio, nullptr), std::invalid_argument);
        EXPECT_THROW(
            manager->storage().adjustCacheLevel(kHostLevel, std::nullopt, ratio, nullptr), std::invalid_argument);
    }
    EXPECT_FALSE(manager->needAdjustment());
}

TEST_F(KvCacheManagerV2StoreLevelTest, RefusesBadTablesMigrationBeforeAttachAndASecondAttach)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    {
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        LifeCycleId const lc{0};
        TypedVec<LifeCycleId, SlotCount> one(storage.numLifeCycles(), 1);
        auto host = storage.newSlots(kHostLevel, one);
        auto store = storage.newSlots(kStoreLevel, one);
        // No IO yet: a store route throws instead of losing the page.
        EXPECT_THROW(storage.copySlotData(
                         lc, kStoreLevel, kHostLevel, store[lc].front().slotId(), host[lc].front().slotId(), stream()),
            LogicError);
        storage.releaseSlot(lc, kHostLevel, std::move(host[lc].front()));
        storage.releaseSlot(lc, kStoreLevel, std::move(store[lc].front()));
    }

    // Bad tables change nothing.
    fake.table().abiVersion = TLLM_KV_STORE_LEVEL_IO_ABI_VERSION + 1;
    EXPECT_THROW(manager->attachStoreLevelIo(fake.address()), std::invalid_argument);
    fake.table().abiVersion = TLLM_KV_STORE_LEVEL_IO_ABI_VERSION;
    fake.table().structBytes = sizeof(TllmKvStoreLevelIoV1) - 8;
    EXPECT_THROW(manager->attachStoreLevelIo(fake.address()), std::invalid_argument);
    fake.table().structBytes = sizeof(TllmKvStoreLevelIoV1);
    fake.table().maxRowsPerOp = 0;
    EXPECT_THROW(manager->attachStoreLevelIo(fake.address()), std::invalid_argument);
    fake.table().maxRowsPerOp = 64;
    EXPECT_EQ(fake.numRegisterCalls(), 0U);

    ASSERT_NO_THROW(manager->attachStoreLevelIo(fake.address()));
    EXPECT_TRUE(manager->storeLevelStats().ioAttached);
    EXPECT_THROW(manager->attachStoreLevelIo(fake.address()), std::invalid_argument);

    // A manager without a store level has nothing to attach.
    auto plain = storeConfig();
    plain.cacheTiers.pop_back();
    auto plainManager = std::make_shared<KvCacheManager>(plain);
    EXPECT_THROW(plainManager->attachStoreLevelIo(fake.address()), std::invalid_argument);
    EXPECT_EQ(plainManager->storeLevelStats().storage.level, -1);
}

TEST_F(KvCacheManagerV2StoreLevelTest, AttachRegistersHostPoolsAndStagingAndUnwindsOnFailure)
{
    {
        // The second registration fails: the first is unregistered again, and configure is never reached.
        FakeStoreIo fake(FakeStoreIo::Options{.failRegisterAt = 1});
        auto manager = std::make_shared<KvCacheManager>(storeConfig(/*twoPoolGroups=*/true));
        try
        {
            manager->attachStoreLevelIo(fake.address());
            ADD_FAILURE() << "attach accepted a failed registration";
        }
        catch (std::runtime_error const& error)
        {
            EXPECT_NE(std::string(error.what()).find("store level: registerMemory failed with -600"), std::string::npos)
                << error.what();
        }
        EXPECT_EQ(fake.numRegistered(), 0U);
        EXPECT_TRUE(fake.configuredSlotBytes().empty());
        EXPECT_FALSE(manager->storeLevelStats().ioAttached);
    }
    {
        // configure refuses (an overcommitted store): every range is unregistered.
        FakeStoreIo fake(FakeStoreIo::Options{.configureCode = -200});
        auto manager = std::make_shared<KvCacheManager>(storeConfig(/*twoPoolGroups=*/true));
        try
        {
            manager->attachStoreLevelIo(fake.address());
            ADD_FAILURE() << "attach accepted a failed configure";
        }
        catch (std::runtime_error const& error)
        {
            EXPECT_NE(std::string(error.what()).find("store level: configure failed with -200"), std::string::npos)
                << error.what();
        }
        EXPECT_EQ(fake.numRegistered(), 0U);
        EXPECT_EQ(fake.numUnregisterCalls(), fake.numRegisterCalls());
    }
    {
        FakeStoreIo fake;
        auto manager = std::make_shared<KvCacheManager>(storeConfig(/*twoPoolGroups=*/true));
        manager->attachStoreLevelIo(fake.address());
        auto& storage = manager->storage();
        // One range per host pool (one pool per cold pool group) plus the page staging buffer.
        size_t const numHostPools = toSizeT(storage.numPoolGroups(kHostLevel));
        EXPECT_EQ(fake.numRegistered(), numHostPools + 1);
        // The layout is the store level's: page bytes and slot counts per pool group.
        auto const slotBytes = fake.configuredSlotBytes();
        auto const numSlots = fake.configuredNumSlots();
        ASSERT_EQ(slotBytes.size(), toSizeT(storage.numPoolGroups(kStoreLevel)));
        uint64_t totalSlots = 0;
        for (PoolGroupIndex pg{0}; pg < storage.numPoolGroups(kStoreLevel); ++pg)
        {
            EXPECT_EQ(slotBytes[toSizeT(pg)], storage.slotSize(kStoreLevel, pg).at(PoolIndex{0}));
            EXPECT_EQ(numSlots[toSizeT(pg)], static_cast<uint64_t>(storage.numSlots(pg, kStoreLevel)));
            totalSlots += numSlots[toSizeT(pg)];
        }
        auto const stats = manager->storeLevelStats();
        EXPECT_EQ(stats.storage.numSlots, totalSlots);
        EXPECT_GT(stats.storage.quotaBytes, 0U);
    }
}

// ---------------------------------------------------------------------------
// (a), (b) Byte equality through every route
// ---------------------------------------------------------------------------

TEST_F(KvCacheManagerV2StoreLevelTest, HostToStoreAndBackKeepsBytesInBothPoolGroups)
{
    FakeStoreIo fake(FakeStoreIo::Options{.maxRowsPerOp = 1});
    fake.setLatency(2ms, 2ms);
    auto manager = std::make_shared<KvCacheManager>(storeConfig(/*twoPoolGroups=*/true));
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    ASSERT_EQ(storage.numPoolGroups(kStoreLevel), PoolGroupIndex{2});
    TypedVec<LifeCycleId, SlotCount> one(storage.numLifeCycles(), 1);
    auto host = storage.newSlots(kHostLevel, one);
    auto store = storage.newSlots(kStoreLevel, one);
    auto const before = manager->storeLevelStats();

    for (LifeCycleId lc{0}; lc < storage.numLifeCycles(); ++lc)
    {
        SCOPED_TRACE(lc.value());
        size_t const bytes = pageBytesOf(storage, kHostLevel, lc);
        ASSERT_EQ(bytes, pageBytesOf(storage, kStoreLevel, lc));
        SlotId const hostSlot = host[lc].front().slotId();
        SlotId const storeSlot = store[lc].front().slotId();
        MemAddress const hostAddress = slotAddressOf(storage, kHostLevel, lc, hostSlot);
        auto const expected = pattern(bytes, static_cast<uint8_t>(0x11 + lc.value()));
        std::memcpy(reinterpret_cast<void*>(hostAddress), expected.data(), bytes);

        storage.copySlotData(lc, kStoreLevel, kHostLevel, storeSlot, hostSlot, stream());
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        auto const object = fake.object(
            static_cast<uint32_t>(storage.getPoolGroupIndex(kStoreLevel, lc).value()), toSizeT(storeSlot));
        ASSERT_TRUE(object.has_value());
        EXPECT_EQ(*object, expected);

        std::memset(reinterpret_cast<void*>(hostAddress), 0, bytes);
        storage.copySlotData(lc, kHostLevel, kStoreLevel, hostSlot, storeSlot, stream());
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        EXPECT_EQ(readHost(hostAddress, bytes), expected);
    }

    auto const after = manager->storeLevelStats();
    EXPECT_EQ(after.storage.demotedPages - before.storage.demotedPages, 2U);
    EXPECT_EQ(after.storage.promotedPages - before.storage.promotedPages, 2U);
    EXPECT_EQ(after.storage.promotedToGpuPages, before.storage.promotedToGpuPages);
    EXPECT_EQ(after.storage.demotedBytes - before.storage.demotedBytes,
        pageBytesOf(storage, kHostLevel, LifeCycleId{0}) + pageBytesOf(storage, kHostLevel, LifeCycleId{1}));
    EXPECT_EQ(after.storage.opsEnqueued - before.storage.opsEnqueued, 4U);
    EXPECT_EQ(after.storage.opsSubmitted, after.storage.opsEnqueued);
    EXPECT_EQ(after.io.putRows, 2U);
    EXPECT_EQ(after.io.getRows, 2U);
    EXPECT_EQ(fake.numBadRows(), 0U);
    EXPECT_EQ(fake.numOverlaps(), 0U);

    for (LifeCycleId lc{0}; lc < storage.numLifeCycles(); ++lc)
    {
        storage.releaseSlot(lc, kHostLevel, std::move(host[lc].front()));
        storage.releaseSlot(lc, kStoreLevel, std::move(store[lc].front()));
    }
}

TEST_F(KvCacheManagerV2StoreLevelTest, GpuToStoreAndBackGoesThroughStagingInBothPoolGroups)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig(/*twoPoolGroups=*/true));
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    TypedVec<LifeCycleId, SlotCount> one(storage.numLifeCycles(), 1);
    auto gpu = storage.newSlots(kHotLevel, one);
    auto store = storage.newSlots(kStoreLevel, one);
    auto const before = manager->storeLevelStats();

    for (LifeCycleId lc{0}; lc < storage.numLifeCycles(); ++lc)
    {
        SCOPED_TRACE(lc.value());
        size_t const bytes = pageBytesOf(storage, kHotLevel, lc);
        // The default codec stores a cold page as the hot page's bytes.
        ASSERT_EQ(bytes, pageBytesOf(storage, kStoreLevel, lc));
        SlotId const gpuSlot = gpu[lc].front().slotId();
        SlotId const storeSlot = store[lc].front().slotId();
        MemAddress const gpuAddress = slotAddressOf(storage, kHotLevel, lc, gpuSlot);
        auto const expected = pattern(bytes, static_cast<uint8_t>(0x60 + lc.value()));
        ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(gpuAddress), expected.data(), bytes, cudaMemcpyHostToDevice),
            cudaSuccess);

        storage.copySlotData(lc, kStoreLevel, kHotLevel, storeSlot, gpuSlot, stream());
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        auto const object = fake.object(
            static_cast<uint32_t>(storage.getPoolGroupIndex(kStoreLevel, lc).value()), toSizeT(storeSlot));
        ASSERT_TRUE(object.has_value());
        EXPECT_EQ(*object, expected);

        ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(gpuAddress), 0, bytes), cudaSuccess);
        storage.copySlotData(lc, kHotLevel, kStoreLevel, gpuSlot, storeSlot, stream());
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::vector<uint8_t> back(bytes);
        ASSERT_EQ(cudaMemcpy(back.data(), reinterpret_cast<void const*>(gpuAddress), bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_EQ(back, expected);
    }

    auto const after = manager->storeLevelStats();
    EXPECT_EQ(after.storage.demotedPages - before.storage.demotedPages, 2U);
    EXPECT_EQ(after.storage.promotedPages - before.storage.promotedPages, 2U);
    EXPECT_EQ(after.storage.promotedToGpuPages - before.storage.promotedToGpuPages, 2U);
    // Rows pointed into the registered staging buffer, never at GPU memory.
    EXPECT_EQ(fake.numBadRows(), 0U);

    for (LifeCycleId lc{0}; lc < storage.numLifeCycles(); ++lc)
    {
        storage.releaseSlot(lc, kHotLevel, std::move(gpu[lc].front()));
        storage.releaseSlot(lc, kStoreLevel, std::move(store[lc].front()));
    }
}

TEST_F(KvCacheManagerV2StoreLevelTest, StoreToStoreIsRefused)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    TypedVec<LifeCycleId, SlotCount> two(storage.numLifeCycles(), 2);
    auto store = storage.newSlots(kStoreLevel, two);
    EXPECT_THROW(
        storage.copySlotData(lc, kStoreLevel, kStoreLevel, store[lc][1].slotId(), store[lc][0].slotId(), stream()),
        LogicError);
    for (auto& slot : store[lc])
    {
        storage.releaseSlot(lc, kStoreLevel, std::move(slot));
    }
}

// ---------------------------------------------------------------------------
// (c), (d) Slot reuse waits for the store
// ---------------------------------------------------------------------------

TEST_F(KvCacheManagerV2StoreLevelTest, HostSlotFreedByADemotionIsNotOverwrittenBeforeThePutReadIt)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    PoolGroupIndex const hostPg = storage.getPoolGroupIndex(kHostLevel, lc);

    auto page = makeCommittedPage(*manager, kHostLevel, 0);
    SlotId const hostSlot = page->slotId();
    MemAddress const hostAddress = slotAddressOf(storage, kHostLevel, lc, hostSlot);
    auto const expected = pattern(kPageBytes, 0xA1);
    std::memcpy(reinterpret_cast<void*>(hostAddress), expected.data(), kPageBytes);

    fake.setLatency(300ms, 0us);
    evict(storage, kHostLevel, 1);
    ASSERT_EQ(page->cacheLevel, kStoreLevel);
    SlotId const storeSlot = page->slotId();

    // Take every free host slot; the one the page left is among them, its put still running.
    auto slots = storage.newSlotsForPoolGroup(kHostLevel, hostPg, storage.getStatistics(kHostLevel, hostPg).free);
    auto it = std::find_if(slots.begin(), slots.end(), [&](Slot const& slot) { return slot.slotId() == hostSlot; });
    ASSERT_NE(it, slots.end());
    EXPECT_FALSE(it->queryReady());

    // A new owner writes only after the slot's readyEvent, as every migration does.
    ASSERT_EQ(cuStreamWaitEvent(stream(), it->readyEvent.handle(), 0), CUDA_SUCCESS);
    ASSERT_EQ(cudaLaunchHostFunc(
                  mStream, [](void* address) { std::memset(address, 0xFF, kPageBytes); },
                  reinterpret_cast<void*>(hostAddress)),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    auto const object
        = fake.object(static_cast<uint32_t>(storage.getPoolGroupIndex(kStoreLevel, lc).value()), toSizeT(storeSlot));
    ASSERT_TRUE(object.has_value());
    EXPECT_EQ(*object, expected);
    EXPECT_EQ(readHost(hostAddress, kPageBytes), std::vector<uint8_t>(kPageBytes, 0xFF));

    for (auto& slot : slots)
    {
        storage.releaseSlot(lc, kHostLevel, std::move(slot));
    }
}

TEST_F(KvCacheManagerV2StoreLevelTest, AStreamWaitingOnAStoreOpIsNotHandedToOtherMigrations)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    PoolGroupIndex const hostPg = storage.getPoolGroupIndex(kHostLevel, lc);
    StoreLevel& io = storage.storeIo();

    auto page = makeCommittedPage(*manager, kHostLevel, 100);
    storage.excludeFromEviction(*page);
    auto slow = makeCommittedPage(*manager, kHostLevel, 0);
    fake.setLatency(2s, 0us);
    evict(storage, kHostLevel, 1);
    ASSERT_EQ(slow->cacheLevel, kStoreLevel);
    EXPECT_EQ(io.numParkedStreams(), 1U);
    EXPECT_EQ(manager->storeLevelStats().storage.parkedStreams, 1U);
    // Hold every free host slot (the one the slow put reads among them), so the page below only ever moves into the
    // host slot it left itself.
    auto held = storage.newSlotsForPoolGroup(kHostLevel, hostPg, storage.getStatistics(kHostLevel, hostPg).free);

    // More host <-> GPU migrations than the stream pool caches, none of them touching the store. A pooled stream still
    // waiting for the put would hold one of them, and every later one through the page's readyEvent.
    int const numMigrations = 2 * (CudaStreamPool::instance().cachedCount() / 2) + 32;
    auto const start = std::chrono::steady_clock::now();
    for (int i = 0; i < numMigrations; ++i)
    {
        storage.batchedMigrate(i % 2 == 0 ? kHotLevel : kHostLevel, {page}, {});
    }
    ASSERT_EQ(page->cacheLevel, kHostLevel);
    page->readyEvent.synchronize();
    EXPECT_LT(std::chrono::steady_clock::now() - start, 1s);
    EXPECT_FALSE(slow->readyEvent.queryComplete()) << "the put finished before the migrations: the test proves nothing";
    EXPECT_EQ(io.numParkedStreams(), 1U);

    sync();
    io.reapParkedStreams();
    EXPECT_EQ(io.numParkedStreams(), 0U);
    EXPECT_EQ(fake.numOverlaps(), 0U);
    EXPECT_EQ(fake.numBadRows(), 0U);
    for (auto& slot : held)
    {
        storage.releaseSlot(lc, kHostLevel, std::move(slot));
    }
    storage.scheduleForEviction(*page);
}

TEST_F(KvCacheManagerV2StoreLevelTest, StoreSlotIsReusedOnlyAfterItsGetFinished)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    PoolGroupIndex const storePg = storage.getPoolGroupIndex(kStoreLevel, lc);

    // Leave exactly one free store slot, so the next demotions must take the slot the get is still reading.
    auto held
        = storage.newSlotsForPoolGroup(kStoreLevel, storePg, storage.getStatistics(kStoreLevel, storePg).free - 1);

    auto first = makeCommittedPage(*manager, kHostLevel, 0);
    auto const firstBytes = pattern(kPageBytes, 0x21);
    std::memcpy(reinterpret_cast<void*>(slotAddressOf(storage, kHostLevel, lc, first->slotId())), firstBytes.data(),
        kPageBytes);
    evict(storage, kHostLevel, 1);
    ASSERT_EQ(first->cacheLevel, kStoreLevel);
    SlotId const storeSlot = first->slotId();

    // Promote it back with a slow get; its store slot is free again at once, its readyEvent pending.
    fake.setLatency(0us, 300ms);
    storage.excludeFromEviction(*first);
    storage.batchedMigrate(kHostLevel, {first}, {});
    ASSERT_EQ(first->cacheLevel, kHostLevel);

    // Demote another page: it can only land in the slot the get is reading.
    auto second = makeCommittedPage(*manager, kHostLevel, 100);
    auto const secondBytes = pattern(kPageBytes, 0x42);
    std::memcpy(reinterpret_cast<void*>(slotAddressOf(storage, kHostLevel, lc, second->slotId())), secondBytes.data(),
        kPageBytes);
    evict(storage, kHostLevel, 1);
    ASSERT_EQ(second->cacheLevel, kStoreLevel);
    EXPECT_EQ(second->slotId(), storeSlot);
    sync();

    EXPECT_EQ(fake.numOverlaps(), 0U);
    EXPECT_EQ(fake.numBadRows(), 0U);
    EXPECT_EQ(readHost(slotAddressOf(storage, kHostLevel, lc, first->slotId()), kPageBytes), firstBytes);
    auto const object = fake.object(static_cast<uint32_t>(storePg.value()), toSizeT(storeSlot));
    ASSERT_TRUE(object.has_value());
    EXPECT_EQ(*object, secondBytes);

    storage.scheduleForEviction(*first);
    for (auto& slot : held)
    {
        storage.releaseSlot(lc, kStoreLevel, std::move(slot));
    }
}

TEST_F(KvCacheManagerV2StoreLevelTest, ManyDemotionsAndPromotionsNeverOverlapOnAKey)
{
    FakeStoreIo fake(FakeStoreIo::Options{.maxRowsPerOp = 3});
    fake.setLatency(200us, 300us);
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    PoolGroupIndex const storePg = storage.getPoolGroupIndex(kStoreLevel, lc);
    // Four free store slots, so slots are reused while their previous ops may still run.
    auto held
        = storage.newSlotsForPoolGroup(kStoreLevel, storePg, storage.getStatistics(kStoreLevel, storePg).free - 4);

    std::vector<SharedPtr<CommittedPage>> pages;
    std::vector<std::vector<uint8_t>> contents;
    for (int i = 0; i < 4; ++i)
    {
        pages.push_back(makeCommittedPage(*manager, kHostLevel, 1000 + 10 * i));
        contents.push_back(pattern(kPageBytes, static_cast<uint8_t>(i)));
        std::memcpy(reinterpret_cast<void*>(slotAddressOf(storage, kHostLevel, lc, pages.back()->slotId())),
            contents.back().data(), kPageBytes);
    }
    for (int round = 0; round < 20; ++round)
    {
        evict(storage, kHostLevel, 4);
        for (auto const& page : pages)
        {
            ASSERT_EQ(page->cacheLevel, kStoreLevel);
            storage.excludeFromEviction(*page);
        }
        // Promote in a rotated order, so the next round's demotions reuse the slots in another order.
        std::vector<SharedPtr<Page>> order;
        for (size_t i = 0; i < pages.size(); ++i)
        {
            order.push_back(pages[(i + round) % pages.size()]);
        }
        storage.batchedMigrate(kHostLevel, order, {});
        for (auto const& page : pages)
        {
            ASSERT_EQ(page->cacheLevel, kHostLevel);
            storage.scheduleForEviction(*page);
        }
    }
    sync();
    EXPECT_EQ(fake.numOverlaps(), 0U);
    EXPECT_EQ(fake.numBadRows(), 0U);
    for (size_t i = 0; i < pages.size(); ++i)
    {
        EXPECT_EQ(readHost(slotAddressOf(storage, kHostLevel, lc, pages[i]->slotId()), kPageBytes), contents[i]);
    }
    for (auto& slot : held)
    {
        storage.releaseSlot(lc, kStoreLevel, std::move(slot));
    }
}

// ---------------------------------------------------------------------------
// (e) Prefetch and the lock path
// ---------------------------------------------------------------------------

TEST_F(KvCacheManagerV2StoreLevelTest, PrefetchMovesStorePagesToHostAndResumeStagesTheRestOnHost)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    int const numBlocks = 3;
    auto const tokens = makeTokens(500, numBlocks * kTokensPerBlock);
    commitAndClose(*manager, tokens);
    ASSERT_EQ(chainLevels(*manager, tokens).size(), static_cast<size_t>(numBlocks));
    auto pushToStore = [&]()
    {
        auto const apiLock = manager->lockExclusive();
        for (CacheLevel level : {kHotLevel, kHostLevel})
        {
            auto const levels = chainLevels(*manager, tokens);
            int const count = static_cast<int>(std::count(levels.begin(), levels.end(), level));
            if (count > 0)
            {
                evict(manager->storage(), level, count);
            }
        }
    };
    pushToStore();
    for (CacheLevel const level : chainLevels(*manager, tokens))
    {
        ASSERT_EQ(level, kStoreLevel);
    }

    auto const before = manager->storeLevelStats();
    {
        auto cache = manager->createKvCache({}, toSpan(tokens));
        EXPECT_TRUE(cache->prefetch(kHostLevel));
        cache->close();
    }
    sync();
    for (CacheLevel const level : chainLevels(*manager, tokens))
    {
        EXPECT_EQ(level, kHostLevel);
    }
    auto const afterPrefetch = manager->storeLevelStats();
    EXPECT_EQ(afterPrefetch.storage.prefetchedPages - before.storage.prefetchedPages, static_cast<uint64_t>(numBlocks));
    EXPECT_EQ(afterPrefetch.storage.promotedToGpuPages, before.storage.promotedToGpuPages);
    // A prefetch from the store is not a disk prefetch.
    EXPECT_EQ(manager->getAndResetIterationDiskPrefetchBlocks(), 0);

    // Without a prefetch, resuming moves the store pages to the host level (zero-copy gets into host slots), then to
    // the GPU: none goes through the staging ring.
    pushToStore();
    {
        auto cache = manager->createKvCache({}, toSpan(tokens));
        ASSERT_TRUE(cache->resume(stream()));
        EXPECT_EQ(cache->numCommittedTokens(), numBlocks * kTokensPerBlock);
        cache->close();
    }
    sync();
    auto const afterResume = manager->storeLevelStats();
    EXPECT_EQ(
        afterResume.storage.promotedPages - afterPrefetch.storage.promotedPages, static_cast<uint64_t>(numBlocks));
    EXPECT_EQ(afterResume.storage.promotedToGpuPages, afterPrefetch.storage.promotedToGpuPages);
    EXPECT_EQ(afterResume.storage.stagingFallbacks, 0U);

    // A prefetch into the GPU takes the same route.
    pushToStore();
    {
        auto cache = manager->createKvCache({}, toSpan(tokens));
        EXPECT_TRUE(cache->prefetch(kHotLevel));
        cache->close();
    }
    sync();
    auto const afterGpuPrefetch = manager->storeLevelStats();
    EXPECT_EQ(afterGpuPrefetch.storage.prefetchedPages - afterResume.storage.prefetchedPages,
        static_cast<uint64_t>(numBlocks));
    EXPECT_EQ(afterGpuPrefetch.storage.promotedToGpuPages, afterPrefetch.storage.promotedToGpuPages);
    for (CacheLevel const level : chainLevels(*manager, tokens))
    {
        EXPECT_EQ(level, kHotLevel);
    }
    EXPECT_EQ(fake.numBadRows(), 0U);
    EXPECT_EQ(fake.numOverlaps(), 0U);
}

TEST_F(KvCacheManagerV2StoreLevelTest, ResumeReadsStorePagesThroughStagingWhenTheHostLevelHasNoRoom)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    int const numBlocks = 2;
    auto const tokens = makeTokens(700, numBlocks * kTokensPerBlock);
    commitAndClose(*manager, tokens);
    LifeCycleId const lc{0};
    std::vector<Slot> held;
    {
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        evict(storage, kHotLevel, numBlocks);
        evict(storage, kHostLevel, numBlocks);
        // No free and no evictable host slot left.
        PoolGroupIndex const hostPg = storage.getPoolGroupIndex(kHostLevel, lc);
        ASSERT_EQ(storage.getStatistics(kHostLevel, hostPg).evictable, 0);
        held = storage.newSlotsForPoolGroup(kHostLevel, hostPg, storage.getStatistics(kHostLevel, hostPg).free);
    }
    for (CacheLevel const level : chainLevels(*manager, tokens))
    {
        ASSERT_EQ(level, kStoreLevel);
    }
    auto const before = manager->storeLevelStats();
    {
        auto cache = manager->createKvCache({}, toSpan(tokens));
        ASSERT_TRUE(cache->resume(stream()));
        EXPECT_EQ(cache->numCommittedTokens(), numBlocks * kTokensPerBlock);
        cache->close();
    }
    sync();
    auto const after = manager->storeLevelStats();
    EXPECT_EQ(after.storage.stagingFallbacks - before.storage.stagingFallbacks, 1U);
    EXPECT_EQ(after.storage.promotedToGpuPages - before.storage.promotedToGpuPages, static_cast<uint64_t>(numBlocks));
    EXPECT_EQ(fake.numBadRows(), 0U);
    auto const apiLock = manager->lockExclusive();
    for (auto& slot : held)
    {
        manager->storage().releaseSlot(lc, kHostLevel, std::move(slot));
    }
}

// ---------------------------------------------------------------------------
// (f) Free-slot reserve
// ---------------------------------------------------------------------------

TEST_F(KvCacheManagerV2StoreLevelTest, KeepFreeBytesKeepsTheTargetAndLaterAllocationsTakeReadySlots)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    PoolGroupIndex const hostPg = storage.getPoolGroupIndex(kHostLevel, lc);

    std::vector<SharedPtr<CommittedPage>> pages;
    for (int i = 0; storage.getStatistics(kHostLevel, hostPg).free > 0; ++i)
    {
        pages.push_back(makeCommittedPage(*manager, kHostLevel, 1000 + 10 * i));
    }
    ASSERT_GE(pages.size(), 12U);

    // Four pages' worth: four LRU pages go down to the store.
    fake.setLatency(200ms, 0us);
    EXPECT_EQ(manager->keepFreeBytes(kHostLevel, 4 * kPageBytes), 4);
    EXPECT_EQ(storage.getStatistics(kHostLevel, hostPg).free, 4);
    sync(); // their puts finished: four ready slots

    // Eight pages' worth: four more go down, their puts still running.
    EXPECT_EQ(manager->keepFreeBytes(kHostLevel, 8 * kPageBytes), 4);
    EXPECT_EQ(storage.getStatistics(kHostLevel, hostPg).free, 8);
    auto slots = storage.newSlotsForPoolGroup(kHostLevel, hostPg, 4);
    for (auto& slot : slots)
    {
        EXPECT_TRUE(slot.queryReady());
    }
    for (auto& slot : slots)
    {
        storage.releaseSlot(lc, kHostLevel, std::move(slot));
    }

    // Already satisfied: nothing moves.
    EXPECT_EQ(manager->keepFreeBytes(kHostLevel, 8 * kPageBytes), 0);
    auto const stats = manager->storeLevelStats();
    EXPECT_EQ(stats.storage.reserveCalls, 3U);
    EXPECT_EQ(stats.storage.reserveEvictedSlots, 8U);
    EXPECT_EQ(stats.storage.demotedPages, 8U);

    // Capped by the evictable pages: never throws.
    for (auto const& page : pages)
    {
        if (page->cacheLevel == kHostLevel && page->scheduledForEviction())
        {
            storage.excludeFromEviction(*page);
        }
    }
    EXPECT_EQ(manager->keepFreeBytes(kHostLevel, 1UL << 30), 0);
    for (auto const& page : pages)
    {
        if (page->cacheLevel == kHostLevel)
        {
            storage.scheduleForEviction(*page);
        }
    }
    sync();
}

TEST_F(KvCacheManagerV2StoreLevelTest, KeepFreeBytesMovesNoMoreHeldPagesThanTheStoreCanTake)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    PoolGroupIndex const hostPg = storage.getPoolGroupIndex(kHostLevel, lc);
    PoolGroupIndex const storePg = storage.getPoolGroupIndex(kStoreLevel, lc);

    // Held host pages (a reservation or a long-chain pin): evictable from the host level, never droppable.
    std::vector<SharedPtr<CommittedPage>> pages;
    std::vector<SharedPtr<PageHolder>> holders;
    for (int i = 0; i < 6; ++i)
    {
        pages.push_back(makeCommittedPage(*manager, kHostLevel, 3000 + 10 * i));
        holders.push_back(pages.back()->hold());
    }
    // Two free store slots and nothing droppable in the store.
    auto storeSlots
        = storage.newSlotsForPoolGroup(kStoreLevel, storePg, storage.getStatistics(kStoreLevel, storePg).free - 2);
    SlotCount const freeBefore = storage.getStatistics(kHostLevel, hostPg).free;
    size_t const target = static_cast<size_t>(freeBefore + 6) * kPageBytes;

    // Six more free host slots asked: only the two the store can take move.
    EXPECT_EQ(manager->keepFreeBytes(kHostLevel, target), 2);
    EXPECT_EQ(storage.getStatistics(kHostLevel, hostPg).free, freeBefore + 2);
    EXPECT_EQ(
        std::count_if(pages.begin(), pages.end(), [](auto const& page) { return page->cacheLevel == kStoreLevel; }), 2);
    // The store holds only held pages now: nothing moves and nothing throws.
    EXPECT_EQ(manager->keepFreeBytes(kHostLevel, target), 0);
    auto const stats = manager->storeLevelStats();
    EXPECT_EQ(stats.storage.reserveFailures, 0U);
    EXPECT_EQ(stats.storage.reserveEvictedSlots, 2U);
    sync();

    holders.clear();
    for (auto& slot : storeSlots)
    {
        storage.releaseSlot(lc, kStoreLevel, std::move(slot));
    }
}

// ---------------------------------------------------------------------------
// (h) Shutdown, (i) completion ring, (j) drops
// ---------------------------------------------------------------------------

TEST_F(KvCacheManagerV2StoreLevelTest, ShutdownDrainsAndUnregistersEveryRange)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig(/*twoPoolGroups=*/true));
    manager->attachStoreLevelIo(fake.address());
    size_t const numRanges = fake.numRegistered();
    ASSERT_GT(numRanges, 1U);
    fake.setLatency(50ms, 0us);
    {
        auto const apiLock = manager->lockExclusive();
        for (int i = 0; i < 6; ++i)
        {
            makeCommittedPage(*manager, kHostLevel, 2000 + 10 * i);
        }
        evict(manager->storage(), kHostLevel, 6);
    }
    // The puts are still running: shutdown waits for them before the host tier is freed.
    manager->shutdown();
    EXPECT_EQ(fake.numRegistered(), 0U);
    EXPECT_EQ(fake.numUnregisterCalls(), numRanges);
    auto const stats = manager->storeLevelStats();
    EXPECT_FALSE(stats.ioAttached);
    EXPECT_EQ(stats.storage.opsSubmitted, stats.storage.opsEnqueued);
    EXPECT_EQ(stats.io.putRows, 6U); // the snapshot taken at detach
    EXPECT_NO_THROW(manager->detachStoreLevelIo());
}

TEST_F(KvCacheManagerV2StoreLevelTest, CompletionRingWaitsForAWordWhosePreviousLapIsInFlight)
{
    FakeStoreIo fake(FakeStoreIo::Options{.maxRowsPerOp = 1});
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    TypedVec<LifeCycleId, SlotCount> one(storage.numLifeCycles(), 1);
    auto host = storage.newSlots(kHostLevel, one);
    TllmKvStoreRow const row{static_cast<uint32_t>(storage.getPoolGroupIndex(kStoreLevel, lc).value()), 0, 0,
        static_cast<uint64_t>(slotAddressOf(storage, kHostLevel, lc, host[lc].front().slotId()))};
    StoreLevel& io = storage.storeIo();

    // One full lap of single-row ops on one stream (each waits for the previous one).
    std::vector<TllmKvStoreRow> rows(StoreLevel::kDoneWords, row);
    io.enqueue(stream(), TLLM_KV_STORE_OP_PUT, rows.data(), rows.size());
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(io.nextSeq(), StoreLevel::kDoneWords);
    auto* word = const_cast<uint32_t*>(io.doneWord(io.nextSeq()));
    ASSERT_EQ(*word, 1U);

    // Pretend the op one lap earlier on the next word is still in flight: the next op waits for it.
    std::atomic_ref<uint32_t>(*word).store(0, std::memory_order_release);
    std::thread completer(
        [word]()
        {
            std::this_thread::sleep_for(50ms);
            std::atomic_ref<uint32_t>(*word).store(1, std::memory_order_release);
        });
    io.enqueue(stream(), TLLM_KV_STORE_OP_PUT, &row, 1);
    completer.join();
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(io.nextSeq(), uint64_t{StoreLevel::kDoneWords} + 1);
    EXPECT_EQ(std::atomic_ref<uint32_t>(*word).load(std::memory_order_acquire), 2U);

    // One that stays in flight past the wait timeout fails the enqueue.
    word = const_cast<uint32_t*>(io.doneWord(io.nextSeq()));
    std::atomic_ref<uint32_t>(*word).store(0, std::memory_order_release);
    io.setLapWaitTimeout(0.05);
    try
    {
        io.enqueue(stream(), TLLM_KV_STORE_OP_PUT, &row, 1);
        ADD_FAILURE() << "an op took a word whose previous lap was in flight";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("more than 65536 store ops in flight"), std::string::npos)
            << error.what();
    }
    EXPECT_EQ(io.nextSeq(), uint64_t{StoreLevel::kDoneWords} + 1);

    // Once it completes, the next lap takes the word and completes it with the lap's value.
    std::atomic_ref<uint32_t>(*word).store(1, std::memory_order_release);
    io.enqueue(stream(), TLLM_KV_STORE_OP_PUT, &row, 1);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    EXPECT_EQ(std::atomic_ref<uint32_t>(*word).load(std::memory_order_acquire), 2U);
    EXPECT_EQ(fake.numOverlaps(), 0U);
    EXPECT_EQ(manager->storeLevelStats().storage.opsSubmitted, uint64_t{StoreLevel::kDoneWords} + 2);

    storage.releaseSlot(lc, kHostLevel, std::move(host[lc].front()));
}

TEST_F(KvCacheManagerV2StoreLevelTest, DroppingAStorePageIssuesNoStoreOp)
{
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    {
        auto page = makeCommittedPage(*manager, kHostLevel, 0);
        evict(storage, kHostLevel, 1);
        ASSERT_EQ(page->cacheLevel, kStoreLevel);
    }
    sync();
    auto const before = manager->storeLevelStats();
    uint64_t const opsBefore = fake.numOps();
    evict(storage, kStoreLevel, 1);
    sync();
    auto const after = manager->storeLevelStats();
    EXPECT_EQ(after.storage.opsEnqueued, before.storage.opsEnqueued);
    EXPECT_EQ(fake.numOps(), opsBefore);
    EXPECT_EQ(storage.getStatistics(kStoreLevel, storage.getPoolGroupIndex(kStoreLevel, LifeCycleId{0})).free,
        storage.numSlots(storage.getPoolGroupIndex(kStoreLevel, LifeCycleId{0}), kStoreLevel));
}

// ---------------------------------------------------------------------------
// Degrade fault mode (TLLM_KV_STORE_TIER_FAULT_MODE=degrade)
// ---------------------------------------------------------------------------

//! Sets the degrade fault mode's variables for one test (read at attach) and clears them after.
class DegradeFaultModeEnv
{
public:
    explicit DegradeFaultModeEnv(char const* breakerS)
    {
        setenv("TLLM_KV_STORE_TIER_FAULT_MODE", "degrade", 1);
        setenv("TLLM_KV_STORE_TIER_BREAKER_S", breakerS, 1);
    }

    ~DegradeFaultModeEnv()
    {
        unsetenv("TLLM_KV_STORE_TIER_FAULT_MODE");
        unsetenv("TLLM_KV_STORE_TIER_BREAKER_S");
    }

    DegradeFaultModeEnv(DegradeFaultModeEnv const&) = delete;
    DegradeFaultModeEnv& operator=(DegradeFaultModeEnv const&) = delete;
};

TEST_F(KvCacheManagerV2StoreLevelTest, DegradeModeFailedPutsBecomeMissesAndAreNeverRead)
{
    DegradeFaultModeEnv const env("0");
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    ASSERT_TRUE(manager->storeLevelStats().storage.faultDegrade);
    int const numBlocks = 3;
    auto pushToStore = [&](std::vector<TokenIdExt> const& tokens)
    {
        auto const apiLock = manager->lockExclusive();
        for (CacheLevel level : {kHotLevel, kHostLevel})
        {
            auto const levels = chainLevels(*manager, tokens);
            int const count = static_cast<int>(std::count(levels.begin(), levels.end(), level));
            if (count > 0)
            {
                evict(manager->storage(), level, count);
            }
        }
    };
    PoolGroupIndex const storePg = manager->storage().getPoolGroupIndex(kStoreLevel, LifeCycleId{0});
    auto storeSlotsUsed = [&]()
    {
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        return storage.numSlots(storePg, kStoreLevel) - storage.getStatistics(kStoreLevel, storePg).free;
    };

    // Every put fails: the pages stay in the tree at the store level, but no match reuses them and no get is issued.
    auto const lostTokens = makeTokens(700, numBlocks * kTokensPerBlock);
    commitAndClose(*manager, lostTokens);
    fake.setFailPuts(true);
    pushToStore(lostTokens);
    sync();
    EXPECT_EQ(storeSlotsUsed(), numBlocks);
    {
        auto cache = manager->createKvCache({}, toSpan(lostTokens));
        ASSERT_TRUE(cache->resume(stream()));
        EXPECT_EQ(cache->numCommittedTokens(), 0); // a miss: recomputed, not read
        cache->close();
    }
    sync();
    auto const failed = manager->storeLevelStats();
    EXPECT_GE(failed.storage.failedPutOps, 1U);
    EXPECT_EQ(failed.storage.lostSlots, static_cast<uint64_t>(numBlocks));
    EXPECT_GE(failed.storage.unreadableMatches, 1U);
    EXPECT_EQ(failed.storage.failedGetOps, 0U);
    EXPECT_EQ(fake.numGetOps(), 0U);

    // The store works again: a new chain is put, confirmed and read back (no breaker window with 0 s).
    fake.setFailPuts(false);
    auto const goodTokens = makeTokens(900, numBlocks * kTokensPerBlock);
    commitAndClose(*manager, goodTokens);
    pushToStore(goodTokens);
    sync();
    {
        auto cache = manager->createKvCache({}, toSpan(goodTokens));
        ASSERT_TRUE(cache->resume(stream()));
        EXPECT_EQ(cache->numCommittedTokens(), numBlocks * kTokensPerBlock);
        cache->close();
    }
    sync();
    EXPECT_GE(fake.numGetOps(), 1U);
    EXPECT_EQ(fake.numBadRows(), 0U);
    EXPECT_EQ(fake.numOverlaps(), 0U);
}

TEST_F(KvCacheManagerV2StoreLevelTest, DegradeModeBreakerWindowRefusesEveryStorePage)
{
    DegradeFaultModeEnv const env("3600");
    FakeStoreIo fake;
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto good = makeCommittedPage(*manager, kHostLevel, 0);
    evict(storage, kHostLevel, 1);
    sync();
    storage.reapStoreOps();
    ASSERT_EQ(good->cacheLevel, kStoreLevel);
    EXPECT_TRUE(storage.storePageReadable(*good));
    fake.setFailPuts(true);
    auto bad = makeCommittedPage(*manager, kHostLevel, 100);
    evict(storage, kHostLevel, 1);
    EXPECT_FALSE(storage.storePageReadable(*bad)); // unconfirmed
    sync();
    storage.reapStoreOps();
    EXPECT_FALSE(storage.storePageReadable(*bad));  // lost
    EXPECT_FALSE(storage.storePageReadable(*good)); // a put failed within the window: no store page is reused
    EXPECT_EQ(manager->storeLevelStats().storage.lostSlots, 1U);
}

TEST_F(KvCacheManagerV2StoreLevelTest, DegradeModeResetsAFailedWordBeforeItsNextLap)
{
    DegradeFaultModeEnv const env("0");
    FakeStoreIo fake(FakeStoreIo::Options{.maxRowsPerOp = 1});
    auto manager = std::make_shared<KvCacheManager>(storeConfig());
    manager->attachStoreLevelIo(fake.address());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    LifeCycleId const lc{0};
    TypedVec<LifeCycleId, SlotCount> one(storage.numLifeCycles(), 1);
    auto host = storage.newSlots(kHostLevel, one);
    TllmKvStoreRow const row{static_cast<uint32_t>(storage.getPoolGroupIndex(kStoreLevel, lc).value()), 0, 0,
        static_cast<uint64_t>(slotAddressOf(storage, kHostLevel, lc, host[lc].front().slotId()))};
    StoreLevel& io = storage.storeIo();

    // One full lap of failing single-row puts: every word ends with the failure flag.
    fake.setFailPuts(true);
    std::vector<TllmKvStoreRow> rows(StoreLevel::kDoneWords, row);
    io.enqueue(stream(), TLLM_KV_STORE_OP_PUT, rows.data(), rows.size());
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    auto* word = const_cast<uint32_t*>(io.doneWord(io.nextSeq()));
    ASSERT_EQ(std::atomic_ref<uint32_t>(*word).load(std::memory_order_acquire), 1U | TLLM_KV_STORE_DONE_FAILED);

    // The next lap's op must still wait for its own completion: the flag (above any lap value) is cleared first.
    fake.setLatency(200ms, 0us);
    io.enqueue(stream(), TLLM_KV_STORE_OP_PUT, &row, 1);
    std::this_thread::sleep_for(50ms); // long enough for a wait satisfied by the stale flag to have passed
    EXPECT_EQ(cudaStreamQuery(mStream), cudaErrorNotReady);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    EXPECT_EQ(std::atomic_ref<uint32_t>(*word).load(std::memory_order_acquire), 2U | TLLM_KV_STORE_DONE_FAILED);
    io.reapCompletedOps();
    EXPECT_EQ(io.failureStats().failedPutOps, uint64_t{StoreLevel::kDoneWords} + 1);
    EXPECT_EQ(io.failureStats().lostSlots, 1U);
    EXPECT_EQ(fake.numOverlaps(), 0U);

    storage.releaseSlot(lc, kHostLevel, std::move(host[lc].front()));
}

} // namespace

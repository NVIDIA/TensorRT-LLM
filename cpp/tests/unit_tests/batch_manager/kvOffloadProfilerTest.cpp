/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "kvCacheManagerV2TestUtils.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"
#include "tensorrt_llm/common/profiler.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;
namespace legacy = tensorrt_llm::batch_manager::kv_cache_manager;
namespace v2 = tensorrt_llm::batch_manager::kv_cache_manager_v2;

namespace tensorrt_llm::testing
{
class KVCacheTransferManagerTestAccess
{
public:
    static void copy(legacy::KVCacheTransferManager& manager, legacy::BlockPtr const& src, legacy::BlockPtr const& dst,
        std::vector<legacy::KVCacheBlockPool> const& pools, bool offload, int tokens)
    {
        manager.copyBlock(src, dst, pools, offload, tokens);
    }
};
} // namespace tensorrt_llm::testing

namespace
{
struct Observation
{
    std::string name;
    std::string metadata;
};

std::vector<Observation>& observations()
{
    static thread_local std::vector<Observation> values;
    return values;
}

bool& observerNeedsInputs()
{
    static thread_local bool value{true};
    return value;
}

std::unique_ptr<at::ObserverContext> observe(at::RecordFunction const& record)
{
    if (std::string_view{record.name()}.find("trtllm::kv_cache_") == 0)
    {
        observations().push_back({record.name(),
            !observerNeedsInputs() || record.inputs().empty() ? "" : record.inputs().front().toStringRef()});
    }
    return nullptr;
}

class CallbackGuard
{
public:
    explicit CallbackGuard(bool inputs = true)
        : mHandle(at::addThreadLocalCallback(
            at::RecordFunctionCallback(observe, nullptr).needsInputs(inputs).scopes({at::RecordScope::USER_SCOPE})))
    {
        observations().clear();
        observerNeedsInputs() = inputs;
    }

    ~CallbackGuard()
    {
        at::removeCallback(mHandle);
    }

private:
    at::CallbackHandle mHandle;
};

void expectField(Observation const& value, std::string const& field)
{
    EXPECT_NE(value.metadata.find(field), std::string::npos) << value.metadata;
}

class KvOffloadProfilerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        {
            GTEST_SKIP() << "CUDA device required for actual native transfer boundaries";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        if (std::getenv("TLLM_PROFILE_CUSTOM_REGIONS") == nullptr)
        {
            setenv("TLLM_PROFILE_CUSTOM_REGIONS", "kv", 1);
        }
        ASSERT_FALSE(tc::initializeProfiler().has_value());
        if (!tc::customRegionEnabled(tc::ProfilerRegion::kKv))
        {
            GTEST_SKIP() << "Positive observation cases require kv selection at process startup";
        }
    }
};

class V2KvOffloadProfilerTest : public KvOffloadProfilerTest, public ::testing::WithParamInterface<bool>
{
};

void expectV2RoundTrip(v2::CacheTier coldTier, bool inputs, bool selected)
{
    auto config = v2::test::makeConfig();
    switch (coldTier)
    {
    case v2::CacheTier::HOST_MEM: config.cacheTiers.emplace_back(v2::HostCacheTierConfig{4 << 20}); break;
    case v2::CacheTier::GPU_MEM: config.cacheTiers.emplace_back(v2::GpuCacheTierConfig{4 << 20}); break;
    case v2::CacheTier::DISK: config.cacheTiers.emplace_back(v2::DiskCacheTierConfig{4 << 20, "/tmp"}); break;
    }

    auto manager = std::make_shared<v2::KvCacheManager>(config);
    auto stream = std::make_shared<tr::CudaStream>();
    auto& storage = manager->storage();
    auto const lifeCycle = manager->getLayerGroupId(v2::LayerId{0});
    auto const hotGroup = storage.getPoolGroupIndex(v2::kHotLevel, lifeCycle);
    auto const coldLevel = v2::CacheLevel{1};
    auto const coldGroup = storage.getPoolGroupIndex(coldLevel, lifeCycle);
    auto const coldBytes = storage.slotSize(coldLevel, coldGroup).at(v2::PoolIndex{0});
    auto const hotBytes = storage.slotSize(v2::kHotLevel, hotGroup).at(v2::PoolIndex{0});
    std::vector<v2::TokenIdExt> tokens;
    for (int index = 0; index < 8; ++index)
    {
        tokens.emplace_back(v2::TokenId{index});
    }
    v2::TokenSpan const span{tokens.data(), static_cast<int>(tokens.size())};
    auto source = manager->createKvCache({}, {}, 1);
    ASSERT_TRUE(source->resume(stream->get()));
    ASSERT_TRUE(source->resize(8));
    for (int page = 0; page < 2; ++page)
    {
        auto const address = std::get<v2::MemAddress>(storage.slotAddress(
            v2::kHotLevel, hotGroup, v2::SlotId{source->getBasePageIndices(lifeCycle)[page]}, v2::PoolIndex{0}));
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(address), 0x5A, hotBytes, stream->get()), cudaSuccess);
    }
    source->commit(span);
    source->close();
    CallbackGuard callback(inputs);
    v2::TypedVec<v2::PoolGroupIndex, v2::SlotCount> goals(storage.numPoolGroups(v2::kHotLevel), 0);
    goals[hotGroup] = storage.getStatistics(v2::kHotLevel, hotGroup).total;
    storage.prepareFreeSlots(v2::kHotLevel, goals);
    ASSERT_EQ(storage.getStatistics(coldLevel, coldGroup).evictable, 2);
    // Repeating the satisfied requirement must not submit another migration.
    storage.prepareFreeSlots(v2::kHotLevel, goals);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // Overwrite recycled hot memory so a missing restore cannot reuse the original bytes.
    auto const base
        = std::get<v2::MemAddress>(storage.slotAddress(v2::kHotLevel, hotGroup, v2::SlotId{0}, v2::PoolIndex{0}));
    ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(base), 0, hotBytes * goals[hotGroup]), cudaSuccess);
    auto restored = manager->createKvCache({}, span, 2);
    ASSERT_EQ(restored->numCommittedTokens(), 8);
    // Drive StorageManager::prefetch through the public cache API, including a no-work repeat.
    ASSERT_TRUE(restored->prefetch(v2::kHotLevel));
    ASSERT_TRUE(restored->prefetch(v2::kHotLevel));
    ASSERT_TRUE(restored->resume(stream->get()));
    stream->synchronize();
    for (int page = 0; page < 2; ++page)
    {
        auto const address = std::get<v2::MemAddress>(storage.slotAddress(
            v2::kHotLevel, hotGroup, v2::SlotId{restored->getBasePageIndices(lifeCycle)[page]}, v2::PoolIndex{0}));
        std::vector<unsigned char> bytes(hotBytes);
        ASSERT_EQ(cudaMemcpy(bytes.data(), reinterpret_cast<void const*>(address), hotBytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [](auto byte) { return byte == 0x5A; }));
    }
    if (!selected || coldTier != v2::CacheTier::HOST_MEM)
    {
        EXPECT_TRUE(observations().empty());
        restored->close();
        manager->shutdown();
        return;
    }
    ASSERT_EQ(observations().size(), 2);
    EXPECT_EQ(observations()[0].name, "trtllm::kv_cache_offload");
    EXPECT_EQ(observations()[1].name, "trtllm::kv_cache_restore");
    for (auto const& value : observations())
    {
        if (!inputs)
        {
            EXPECT_TRUE(value.metadata.empty());
            continue;
        }
        expectField(value, "'kv.manager_version':'v2'");
        expectField(value, "'kv.copy_count':2");
        expectField(value, "'kv.bytes_per_copy':" + std::to_string(coldBytes));
        expectField(value, "'kv.payload_bytes':" + std::to_string(2 * coldBytes));
        expectField(value, "'kv.cuda_stream_id':");
    }
    restored->close();
    manager->shutdown();
}

TEST_P(V2KvOffloadProfilerTest, ProductionEvictionAndOnboardingPreservePayload)
{
    expectV2RoundTrip(v2::CacheTier::HOST_MEM, GetParam(), true);
}

INSTANTIATE_TEST_SUITE_P(CallbackInputs, V2KvOffloadProfilerTest, ::testing::Bool());

class V2OtherTierProfilerTest : public KvOffloadProfilerTest, public ::testing::WithParamInterface<v2::CacheTier>
{
};

TEST_P(V2OtherTierProfilerTest, ProductionMigrationDoesNotMislabelOtherTiersOrEmptyWork)
{
    expectV2RoundTrip(GetParam(), true, true);
}

INSTANTIATE_TEST_SUITE_P(
    ColdTier, V2OtherTierProfilerTest, ::testing::Values(v2::CacheTier::GPU_MEM, v2::CacheTier::DISK));

class KvOffloadDisabledTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        {
            GTEST_SKIP() << "CUDA device required";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        // gtest_discover_tests launches each case in a fresh process, before selector caching.
#if defined(_WIN32)
        ASSERT_EQ(_putenv_s("TLLM_PROFILE_CUSTOM_REGIONS", "none"), 0);
#else
        ASSERT_EQ(setenv("TLLM_PROFILE_CUSTOM_REGIONS", "none", 1), 0);
#endif
        ASSERT_FALSE(tc::initializeProfiler().has_value());
        ASSERT_FALSE(tc::customRegionEnabled(tc::ProfilerRegion::kKv));
    }
};

TEST_F(KvOffloadDisabledTest, V2ProductionRoundTripPreservesPayload)
{
    expectV2RoundTrip(v2::CacheTier::HOST_MEM, true, false);
}

class LegacyKvOffloadProfilerTest
    : public KvOffloadProfilerTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool, int, int, tensorrt_llm::DataType>>
{
};

void expectLegacyRoundTrip(
    bool layerFirst, bool scales, int tokens, int expectedBytes, tensorrt_llm::DataType dtype, bool selected)
{
    auto stream = std::make_shared<tr::CudaStream>();
    tr::BufferManager buffers(stream);
    tr::ITensor::SharedPtr primary = buffers.gpu(tr::ITensor::makeShape({2, 2, 2, 32}), dtype);
    tr::ITensor::SharedPtr secondary = tr::BufferManager::pinned(
        layerFirst ? tr::ITensor::makeShape({2, 3, 2, 32}) : tr::ITensor::makeShape({3, 2, 2, 32}), dtype);
    legacy::KVCacheBlockPool pool(2, 2, 1, 4, 8, primary, secondary, scales);
    pool.layerFirstLayout = layerFirst;
    legacy::KVCacheTransferManager manager(buffers);
    auto src = std::make_shared<legacy::KVCacheBlock>(11, tensorrt_llm::kernels::KVCacheIndex(1));
    auto dst = std::make_shared<legacy::KVCacheBlock>(12, tensorrt_llm::kernels::KVCacheIndex(2, true));
    ASSERT_EQ(cudaMemset(primary->data(), 0x5A, primary->getSizeInBytes()), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::memset(secondary->data(), 0, secondary->getSizeInBytes());
    CallbackGuard callback;
    tensorrt_llm::testing::KVCacheTransferManagerTestAccess::copy(manager, src, dst, {pool}, true, tokens);
    manager.syncTransfers();
    stream->synchronize();
    bool const packed = dtype == tensorrt_llm::DataType::kINT4 || dtype == tensorrt_llm::DataType::kFP4;
    std::size_t const blockBytes = packed ? 64 : 512;
    std::size_t const layerBytes = packed ? 32 : 256;
    bool const partial = !layerFirst && !scales && !packed && tokens == 3;
    auto payloadAt = [&](std::size_t offset) { return !partial || offset % 128 < 48; };
    auto const* hostBytes = static_cast<unsigned char const*>(secondary->data());
    for (std::size_t index = 0; index < secondary->getSizeInBytes(); ++index)
    {
        bool const copied = layerFirst
            ? (index >= 2 * layerBytes && index < 3 * layerBytes) || (index >= 5 * layerBytes && index < 6 * layerBytes)
            : index >= 2 * blockBytes && index < 3 * blockBytes && payloadAt(index - 2 * blockBytes);
        EXPECT_EQ(hostBytes[index], copied ? 0x5A : 0) << "host byte " << index;
    }
    if (selected)
    {
        ASSERT_EQ(observations().size(), 1);
        expectField(observations()[0], "'kv.payload_bytes':" + std::to_string(expectedBytes));
        expectField(observations()[0], "'kv.manager_version':'legacy'");
        expectField(observations()[0], "'kv.pool_index':0");
        expectField(observations()[0], "'kv.source_block_id':11");
        EXPECT_EQ(observations()[0].name, "trtllm::kv_cache_offload");
    }
    else
    {
        EXPECT_TRUE(observations().empty());
    }
    ASSERT_EQ(cudaMemset(primary->data(), 0, primary->getSizeInBytes()), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    tensorrt_llm::testing::KVCacheTransferManagerTestAccess::copy(manager, dst, src, {pool}, false, tokens);
    manager.syncTransfers();
    stream->synchronize();
    if (selected)
    {
        ASSERT_EQ(observations().size(), 2);
        EXPECT_EQ(observations()[1].name, "trtllm::kv_cache_restore");
        expectField(observations()[1], "'kv.payload_bytes':" + std::to_string(expectedBytes));
    }
    else
    {
        EXPECT_TRUE(observations().empty());
    }
    auto check = tr::BufferManager::pinned(primary->getSizeInBytes());
    ASSERT_EQ(cudaMemcpy(check->data(), primary->data(), check->getSizeInBytes(), cudaMemcpyDeviceToHost), cudaSuccess);
    auto const* restored = static_cast<unsigned char const*>(check->data());
    for (std::size_t index = 0; index < check->getSizeInBytes(); ++index)
    {
        bool const copied = layerFirst
            ? (index >= layerBytes && index < 2 * layerBytes) || (index >= 3 * layerBytes && index < 4 * layerBytes)
            : index >= blockBytes && index < 2 * blockBytes && payloadAt(index - blockBytes);
        EXPECT_EQ(restored[index], copied ? 0x5A : 0) << "restored byte " << index;
    }
}

TEST_P(LegacyKvOffloadProfilerTest, ReportsActualPoolPayloadAndPreservesRoundTrip)
{
    auto const [layerFirst, scales, tokens, expectedBytes, dtype] = GetParam();
    expectLegacyRoundTrip(layerFirst, scales, tokens, expectedBytes, dtype, true);
}

TEST_F(KvOffloadDisabledTest, LegacyProductionRoundTripPreservesPayload)
{
    expectLegacyRoundTrip(false, false, 3, 192, tensorrt_llm::DataType::kFLOAT, false);
}

TEST_F(KvOffloadProfilerTest, LegacyDeviceReuseIsNotOffload)
{
    auto stream = std::make_shared<tr::CudaStream>();
    tr::BufferManager buffers(stream);
    tr::ITensor::SharedPtr primary = buffers.gpu(tr::ITensor::makeShape({2, 2, 2, 32}), tensorrt_llm::DataType::kFLOAT);
    legacy::KVCacheBlockPool pool(2, 2, 1, 4, 8, primary);
    legacy::KVCacheTransferManager manager(buffers);
    auto src = std::make_shared<legacy::KVCacheBlock>(11, tensorrt_llm::kernels::KVCacheIndex(0));
    auto dst = std::make_shared<legacy::KVCacheBlock>(12, tensorrt_llm::kernels::KVCacheIndex(1));
    CallbackGuard callback;
    tensorrt_llm::testing::KVCacheTransferManagerTestAccess::copy(manager, src, dst, {pool}, true, 3);
    manager.syncTransfers();
    stream->synchronize();
    EXPECT_TRUE(observations().empty());
}

INSTANTIATE_TEST_SUITE_P(PayloadPaths, LegacyKvOffloadProfilerTest,
    ::testing::Values(std::tuple{false, false, 0, 512, tensorrt_llm::DataType::kFLOAT},
        std::tuple{false, false, 8, 512, tensorrt_llm::DataType::kFLOAT},
        std::tuple{false, false, 3, 192, tensorrt_llm::DataType::kFLOAT},
        std::tuple{false, true, 3, 512, tensorrt_llm::DataType::kFLOAT},
        std::tuple{true, false, 3, 512, tensorrt_llm::DataType::kFLOAT},
        std::tuple{false, false, 3, 64, tensorrt_llm::DataType::kINT4},
        std::tuple{false, false, 3, 64, tensorrt_llm::DataType::kFP4}));
} // namespace

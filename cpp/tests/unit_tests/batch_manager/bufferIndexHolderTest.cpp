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

#include "tensorrt_llm/batch_manager/baseTransBuffer.h"
#include "tensorrt_llm/batch_manager/cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include <NvInferRuntime.h>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::runtime;

namespace
{

// Subclass that exposes concurrence counters so tests can assert how many
// slots are currently held without relying on indirect observable effects.
class ObservableTransBufferManager : public CacheTransBufferManager
{
public:
    using CacheTransBufferManager::CacheTransBufferManager;

    [[nodiscard]] int sendInUse() const
    {
        return mConcurrenceSendResource.mConcurrence.load();
    }

    [[nodiscard]] int recvInUse() const
    {
        return mConcurrenceRecvResource.mConcurrence.load();
    }
};

} // namespace

enum class Side
{
    Send,
    Recv
};

struct HolderCase
{
    std::string name;
    Side side;
};

class BufferIndexHolderLifecycleTest : public ::testing::TestWithParam<HolderCase>
{
protected:
    void SetUp() override
    {
        setenv("TRTLLM_USE_UCX_KVCACHE", "1", 1);
        // Move-assignment coverage requires two simultaneously held indices from either pool.
        setenv("TRTLLM_REQUEST_KV_CACHE_CONCURRENT", "1", 1);
        setenv("TRTLLM_KVCACHE_RECV_BUFFER_COUNT", "2", 1);
        setenv("TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM", "2", 1);

        int constexpr numLayers = 2;
        int constexpr numHeads = 2;
        int constexpr sizePerHead = 8;
        int constexpr tokensPerBlock = 4;
        SizeType32 constexpr maxBlocksPerSeq = 4;
        SizeType32 constexpr maxBeamWidth = 1;
        SizeType32 constexpr maxNumSequences = 4;
        SizeType32 constexpr sinkTokenLength = 0;
        auto stream = std::make_shared<CudaStream>();
        auto const kvMaxNumTokens = tokensPerBlock * maxBlocksPerSeq;
        auto const totalNumBlocks = maxNumSequences * maxBlocksPerSeq;
        using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;
        auto const blocksPerWindow = BlocksPerWindow{{kvMaxNumTokens, {totalNumBlocks, 0}}};

        mKv = std::make_unique<KVCacheManager>(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow,
            maxNumSequences, maxBeamWidth, std::vector<BlockManager::SizeType32>{kvMaxNumTokens},
            nvinfer1::DataType::kFLOAT, sinkTokenLength, stream, kvMaxNumTokens, kvMaxNumTokens,
            /*enableBlockReuse=*/true, CacheType::kSELF, std::nullopt, nullptr, true);
        mKv->allocatePools(false);
        mTrans = std::make_unique<ObservableTransBufferManager>(mKv.get(), std::optional<size_t>{kvMaxNumTokens});
    }

    void TearDown() override
    {
        mTrans.reset();
        mKv.reset();
    }

    [[nodiscard]] std::optional<int> acquire() const
    {
        return GetParam().side == Side::Send ? mTrans->assignBufferIndexForSend() : mTrans->assignBufferIndexForRecv();
    }

    [[nodiscard]] int inUse() const
    {
        return GetParam().side == Side::Send ? mTrans->sendInUse() : mTrans->recvInUse();
    }

    [[nodiscard]] bool isRecv() const
    {
        return GetParam().side == Side::Recv;
    }

    [[nodiscard]] ObservableTransBufferManager& mgr() const
    {
        return *mTrans;
    }

    std::unique_ptr<KVCacheManager> mKv;
    std::unique_ptr<ObservableTransBufferManager> mTrans;
};

// Default-constructed holder owns nothing; destruction is a no-op.
TEST_P(BufferIndexHolderLifecycleTest, DefaultConstructedHolderReleasesNothing)
{
    int const before = inUse();
    {
        BufferIndexHolder holder;
        EXPECT_FALSE(holder.held());
        EXPECT_FALSE(holder.index().has_value());
    }
    EXPECT_EQ(inUse(), before);
}

// Explicit release() on a default-constructed holder (no manager bound)
// must not dereference the null manager pointer.
TEST_P(BufferIndexHolderLifecycleTest, DefaultConstructedExplicitReleaseIsNoOp)
{
    int const before = inUse();
    BufferIndexHolder holder;
    holder.release();
    EXPECT_FALSE(holder.held());
    EXPECT_EQ(inUse(), before);
}

// Holder built from a nullopt index is disarmed (mHeld == false).
TEST_P(BufferIndexHolderLifecycleTest, NulloptIndexReleasesNothing)
{
    int const before = inUse();
    {
        BufferIndexHolder holder{mgr(), std::nullopt, isRecv()};
        EXPECT_FALSE(holder.held());
    }
    EXPECT_EQ(inUse(), before);
}

// RAII: a held slot is released when the holder goes out of scope.
TEST_P(BufferIndexHolderLifecycleTest, ValidIndexReleasedOnDestruction)
{
    int const before = inUse();
    auto idx = acquire();
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(inUse(), before + 1);
    {
        BufferIndexHolder holder{mgr(), idx, isRecv()};
        EXPECT_TRUE(holder.held());
        EXPECT_EQ(holder.index(), idx);
    }
    EXPECT_EQ(inUse(), before);
}

// release() frees the slot immediately and disarms the destructor.
TEST_P(BufferIndexHolderLifecycleTest, ExplicitReleaseFreesSlotEagerly)
{
    int const before = inUse();
    auto idx = acquire();
    ASSERT_TRUE(idx.has_value());
    BufferIndexHolder holder{mgr(), idx, isRecv()};
    EXPECT_EQ(inUse(), before + 1);
    holder.release();
    EXPECT_EQ(inUse(), before);
    EXPECT_FALSE(holder.held());
}

// release() is idempotent: a second call is a no-op.
TEST_P(BufferIndexHolderLifecycleTest, DoubleReleaseIsSafe)
{
    int const before = inUse();
    auto idx = acquire();
    ASSERT_TRUE(idx.has_value());
    BufferIndexHolder holder{mgr(), idx, isRecv()};
    holder.release();
    EXPECT_EQ(inUse(), before);
    holder.release();
    EXPECT_EQ(inUse(), before);
}

// detach() returns the index and disarms the destructor. The caller is
// responsible for freeing the slot — we free it manually to keep the pool
// balanced for subsequent tests.
TEST_P(BufferIndexHolderLifecycleTest, DetachDisarmsDestructor)
{
    int const before = inUse();
    auto idx = acquire();
    ASSERT_TRUE(idx.has_value());
    {
        BufferIndexHolder holder{mgr(), idx, isRecv()};
        auto detached = holder.detach();
        EXPECT_EQ(detached, idx);
        EXPECT_FALSE(holder.held());
    }
    // Destructor was disarmed: slot still in use.
    EXPECT_EQ(inUse(), before + 1);

    // Manually free so the harness is balanced.
    if (isRecv())
    {
        mgr().freeBufferIndexForRecv(idx);
    }
    else
    {
        mgr().freeBufferIndexForSend(idx);
    }
    EXPECT_EQ(inUse(), before);
}

// Move construction transfers ownership; the moved-from holder is disarmed.
TEST_P(BufferIndexHolderLifecycleTest, MoveConstructTransfersOwnership)
{
    int const before = inUse();
    auto idx = acquire();
    ASSERT_TRUE(idx.has_value());
    {
        BufferIndexHolder source{mgr(), idx, isRecv()};
        BufferIndexHolder sink{std::move(source)};
        EXPECT_FALSE(source.held());
        EXPECT_TRUE(sink.held());
        EXPECT_EQ(sink.index(), idx);
        EXPECT_EQ(inUse(), before + 1);
    }
    // Sink released on scope exit; source was disarmed.
    EXPECT_EQ(inUse(), before);
}

// Move assignment releases the prior holding before taking new ownership.
TEST_P(BufferIndexHolderLifecycleTest, MoveAssignReleasesPriorThenTransfers)
{
    int const before = inUse();
    ASSERT_GE(isRecv() ? mgr().getRecvBufferCount() : mgr().getSendBufferCount(), 2);
    auto firstIdx = acquire();
    auto secondIdx = acquire();
    ASSERT_TRUE(firstIdx.has_value());
    ASSERT_TRUE(secondIdx.has_value());
    EXPECT_EQ(inUse(), before + 2);
    {
        BufferIndexHolder sink{mgr(), firstIdx, isRecv()};
        BufferIndexHolder source{mgr(), secondIdx, isRecv()};
        sink = std::move(source);
        // Sink now owns secondIdx; firstIdx was released by the move-assign.
        EXPECT_EQ(inUse(), before + 1);
        EXPECT_EQ(sink.index(), secondIdx);
        EXPECT_FALSE(source.held());
    }
    EXPECT_EQ(inUse(), before);
}

// Exception unwind through a scope containing a held holder still releases
// the slot.
TEST_P(BufferIndexHolderLifecycleTest, ExceptionUnwindStillReleases)
{
    int const before = inUse();
    auto idx = acquire();
    ASSERT_TRUE(idx.has_value());
    try
    {
        BufferIndexHolder holder{mgr(), idx, isRecv()};
        EXPECT_EQ(inUse(), before + 1);
        throw std::runtime_error("forced unwind");
    }
    catch (std::runtime_error const&)
    {
        // Holder destructor ran during unwind.
    }
    EXPECT_EQ(inUse(), before);
}

INSTANTIATE_TEST_SUITE_P(SideVariants, BufferIndexHolderLifecycleTest,
    ::testing::Values(HolderCase{"send", Side::Send}, HolderCase{"recv", Side::Recv}),
    [](::testing::TestParamInfo<HolderCase> const& info) { return info.param.name; });

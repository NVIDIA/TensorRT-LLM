/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <gtest/gtest.h>
#include <memory>
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::runtime;

class CacheTransBufferTest : public ::testing::Test
{
protected:
    void SetUpCacheTransBuffer(int numLayers, int numHeads, int sizePerHead, int tokensPerBlock, CacheType cacheType,
        std::optional<size_t> maxNumTokens, SizeType32 maxBlocksPerSeq)
    {
        setenv("TRTLLM_USE_UCX_KVCACHE", "1", 1);
        // Initialize KVCacheManager with required parameters
        auto hiddenSize = numHeads * sizePerHead;
        auto constexpr maxBeamWidth = 4;
        auto constexpr sinkTokenLength = 0;
        auto constexpr maxNumSequences = 8;
        auto const stream = std::make_shared<CudaStream>();

        auto kvMaxNumTokens = tokensPerBlock * maxBlocksPerSeq;
        auto maxAttentionWindow = kvMaxNumTokens;
        auto inputLength = kvMaxNumTokens - tokensPerBlock - 1;
        auto numSharedBlocks = inputLength / tokensPerBlock;
        auto numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

        auto totalNumBlocks = maxNumSequences * numBlocksPerSeq;
        auto constexpr blocksInSecondaryPool = 0;

        auto constexpr enableBlockReuse = true;
        auto constexpr onboardBlocks = true;
        auto constexpr dataType = nvinfer1::DataType::kFLOAT;

        using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;
        auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

        mCacheManager = std::make_unique<KVCacheManager>(numLayers, numHeads, sizePerHead, tokensPerBlock,
            blocksPerWindow, maxNumSequences, maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow},
            std::nullopt, dataType, sinkTokenLength, stream, kvMaxNumTokens, enableBlockReuse, onboardBlocks, cacheType,
            std::nullopt, nullptr, true);

        mCacheManager->allocatePools(false);

        TLLM_LOG_INFO("kvCacheManager created");
        mTransBufferManager = std::make_unique<CacheTransBufferManager>(mCacheManager.get(), maxNumTokens);
        TLLM_LOG_INFO("CacheTransBufferManager created");
    }

    void TearDown() override
    {
        mTransBufferManager.reset();
        mCacheManager.reset();
    }

    size_t kvCacheSizePerToken(int numLayers, int numHeads, int sizePerHead, CacheType cacheType)
    {
        if (cacheType == CacheType::kSELFKONLY)
        {
            // data type is float
            return numLayers * numHeads * sizePerHead * 4;
        }
        else
        {
            return numLayers * numHeads * sizePerHead * 2 * 4;
        }
    }

    std::unique_ptr<KVCacheManager> mCacheManager;
    std::unique_ptr<CacheTransBufferManager> mTransBufferManager;
};

TEST_F(CacheTransBufferTest, TestPreAllocBufferSize)
{

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "Fork failed";

    if (pid == 0)
    {

        // Child process
        SizeType32 maxBlocksPerSeq = 10;
        SizeType32 tokensPerBlock = 8;
        std::optional<size_t> maxNumTokens = maxBlocksPerSeq * tokensPerBlock;
        SetUpCacheTransBuffer(4, 2, 64, tokensPerBlock, CacheType::kSELFKONLY, maxNumTokens, maxBlocksPerSeq);
        size_t recvbufferCount = tensorrt_llm::common::getEnvRequestKVCacheConcurrent()
            ? tensorrt_llm::common::getEnvKVCacheRecvBufferCount()
            : 1;
        size_t sendBufferCount = tensorrt_llm::common::getEnvKVCacheSendMaxConcurrenceNum();
        size_t cacheSizeBytesPerToken = kvCacheSizePerToken(4, 2, 64, CacheType::kSELFKONLY);
        std::map<SizeType32, SizeType32> cacheSizeBytesPerTokenPerWindow{
            {maxBlocksPerSeq * tokensPerBlock, cacheSizeBytesPerToken}};
        tensorrt_llm::executor::CacheTransceiverConfig cacheTransceiverConfig{
            tensorrt_llm::executor::CacheTransceiverConfig::BackendType::UCX, maxNumTokens};
        size_t bufferSizeBytes = CacheTransBufferManager::preAllocBufferSize(
            cacheSizeBytesPerTokenPerWindow, tokensPerBlock, cacheTransceiverConfig);
        auto bufferId = mTransBufferManager->assignBufferIndexForSend();
        EXPECT_TRUE(bufferId.has_value());
        EXPECT_EQ(bufferId.value(), 0);
        EXPECT_EQ(bufferSizeBytes,
            mTransBufferManager->getSendBuffer(bufferId)->getSizeInBytes() * (recvbufferCount + sendBufferCount));
        mTransBufferManager->freeBufferIndexForSend(bufferId);
        exit(testing::Test::HasFailure() ? 1 : 0);
    }
    else
    {
        // Parent process
        int status;
        ASSERT_NE(-1, waitpid(pid, &status, 0)) << "waitpid failed";
        ASSERT_TRUE(WIFEXITED(status)) << "Child process terminated abnormally";
        ASSERT_EQ(0, WEXITSTATUS(status)) << "Test in child process failed";
    }
}

TEST_F(CacheTransBufferTest, TestPreAllocBufferSize2)
{
    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "Fork failed";

    if (pid == 0)
    {
        // Child process

        SizeType32 maxBlocksPerSeq = 10;
        SizeType32 tokensPerBlock = 8;
        std::optional<size_t> maxNumTokens = maxBlocksPerSeq * tokensPerBlock;
        SetUpCacheTransBuffer(4, 2, 64, tokensPerBlock, CacheType::kSELF, maxNumTokens, maxBlocksPerSeq);
        size_t recvbufferCount = tensorrt_llm::common::getEnvRequestKVCacheConcurrent()
            ? tensorrt_llm::common::getEnvKVCacheRecvBufferCount()
            : 1;
        size_t sendBufferCount = tensorrt_llm::common::getEnvKVCacheSendMaxConcurrenceNum();
        size_t cacheSizeBytesPerToken = kvCacheSizePerToken(4, 2, 64, CacheType::kSELF);
        tensorrt_llm::executor::CacheTransceiverConfig cacheTransceiverConfig{
            tensorrt_llm::executor::CacheTransceiverConfig::BackendType::UCX, maxNumTokens};
        std::map<SizeType32, SizeType32> cacheSizeBytesPerTokenPerWindow{
            {maxBlocksPerSeq * tokensPerBlock, cacheSizeBytesPerToken}};
        size_t bufferSizeBytes = CacheTransBufferManager::preAllocBufferSize(
            cacheSizeBytesPerTokenPerWindow, tokensPerBlock, cacheTransceiverConfig);
        auto bufferId = mTransBufferManager->assignBufferIndexForSend();
        EXPECT_TRUE(bufferId.has_value());
        EXPECT_EQ(bufferId.value(), 0);
        EXPECT_EQ(bufferSizeBytes,
            mTransBufferManager->getSendBuffer(bufferId)->getSizeInBytes() * (recvbufferCount + sendBufferCount));
        mTransBufferManager->freeBufferIndexForSend(bufferId);
        exit(testing::Test::HasFailure() ? 1 : 0);
    }
    else
    {
        int status;
        ASSERT_NE(-1, waitpid(pid, &status, 0)) << "waitpid failed";
        ASSERT_TRUE(WIFEXITED(status)) << "Child process terminated abnormally";
        ASSERT_EQ(0, WEXITSTATUS(status)) << "Test in child process failed";
    }
}

TEST_F(CacheTransBufferTest, TestBufferIndexAssignment0)
{
    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "Fork failed";

    if (pid == 0)
    {
        // Child process

        SizeType32 maxBlocksPerSeq = 10;
        SizeType32 tokensPerBlock = 8;
        std::optional<size_t> maxNumTokens = maxBlocksPerSeq * tokensPerBlock;
        SetUpCacheTransBuffer(4, 2, 64, tokensPerBlock, CacheType::kSELF, maxNumTokens, maxBlocksPerSeq);

        auto bufferId = mTransBufferManager->assignBufferIndexForSend();
        EXPECT_TRUE(bufferId.has_value());
        EXPECT_EQ(bufferId.value(), 0);
        mTransBufferManager->freeBufferIndexForSend(bufferId);
        auto time = std::chrono::steady_clock::now();

        bufferId = mTransBufferManager->assignBufferIndexForSend();

        auto thread = std::thread(
            [this, time]()
            {
                auto bufferId = mTransBufferManager->assignBufferIndexForSend();
                auto duration = std::chrono::steady_clock::now() - time;
                EXPECT_TRUE(bufferId.has_value());
                EXPECT_EQ(bufferId.value(), 0);
                EXPECT_GT(duration, std::chrono::milliseconds(200));
                mTransBufferManager->freeBufferIndexForSend(bufferId);
            });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        mTransBufferManager->freeBufferIndexForSend(bufferId);

        thread.join();

        // Test receive buffer index assignment
        time = std::chrono::steady_clock::now();

        auto recvBufferId = mTransBufferManager->assignBufferIndexForRecv();
        EXPECT_TRUE(recvBufferId.has_value());
        EXPECT_GE(recvBufferId.value(), 0);

        auto thread2 = std::thread(
            [this, time]()
            {
                auto recvBufferId = mTransBufferManager->assignBufferIndexForRecv();
                auto duration = std::chrono::steady_clock::now() - time;
                EXPECT_TRUE(recvBufferId.has_value());
                EXPECT_EQ(recvBufferId.value(), 0);
                EXPECT_GT(duration, std::chrono::milliseconds(200));
                mTransBufferManager->freeBufferIndexForRecv(recvBufferId);
            });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        mTransBufferManager->freeBufferIndexForRecv(recvBufferId);
        thread2.join();
        // Free buffer indices
        mTransBufferManager->freeBufferIndexForRecv(recvBufferId);
        exit(testing::Test::HasFailure() ? 1 : 0);
    }
    else
    {
        int status;
        ASSERT_NE(-1, waitpid(pid, &status, 0)) << "waitpid failed";
        ASSERT_TRUE(WIFEXITED(status)) << "Child process terminated abnormally";
        ASSERT_EQ(0, WEXITSTATUS(status)) << "Test in child process failed";
    }
}

TEST_F(CacheTransBufferTest, TestBufferIndexAssignment1)
{

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "Fork failed";
    if (pid == 0)
    {
        SizeType32 maxBlocksPerSeq = 10;
        SizeType32 tokensPerBlock = 8;
        std::optional<size_t> maxNumTokens = maxBlocksPerSeq * tokensPerBlock;
        setenv("TRTLLM_REQUEST_KV_CACHE_CONCURRENT", "1", 1);
        setenv("TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM", "2", 1);
        SetUpCacheTransBuffer(4, 2, 64, tokensPerBlock, CacheType::kSELF, maxNumTokens, maxBlocksPerSeq);
        auto bufferId = mTransBufferManager->assignBufferIndexForSend();
        EXPECT_TRUE(bufferId.has_value());
        EXPECT_EQ(bufferId.value(), 0);
        auto bufferId2 = mTransBufferManager->assignBufferIndexForSend();
        EXPECT_TRUE(bufferId2.has_value());
        EXPECT_EQ(bufferId2.value(), 1);
        auto time = std::chrono::steady_clock::now();

        auto thread0 = std::thread(
            [this, time]()
            {
                auto bufferId2 = mTransBufferManager->assignBufferIndexForSend();
                EXPECT_TRUE(bufferId2.has_value());
                EXPECT_EQ(bufferId2.value(), 0);
                auto duration = std::chrono::steady_clock::now() - time;
                EXPECT_GT(duration, std::chrono::milliseconds(200));
                mTransBufferManager->freeBufferIndexForSend(bufferId2);
            });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        mTransBufferManager->freeBufferIndexForSend(bufferId);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        mTransBufferManager->freeBufferIndexForSend(bufferId2);
        thread0.join();
        exit(testing::Test::HasFailure() ? 1 : 0);

        auto recvBufferId = mTransBufferManager->assignBufferIndexForRecv();
        EXPECT_TRUE(recvBufferId.has_value());
        EXPECT_EQ(recvBufferId.value(), 0);
        auto recvBufferId2 = mTransBufferManager->assignBufferIndexForRecv();
        EXPECT_TRUE(recvBufferId2.has_value());
        EXPECT_EQ(recvBufferId2.value(), 1);
        auto time2 = std::chrono::steady_clock::now();

        auto thread1 = std::thread(
            [this, time2]()
            {
                auto recvBufferId2 = mTransBufferManager->assignBufferIndexForRecv();
                EXPECT_TRUE(recvBufferId2.has_value());
                EXPECT_EQ(recvBufferId2.value(), 1);
                auto duration = std::chrono::steady_clock::now() - time2;
                EXPECT_GT(duration, std::chrono::milliseconds(200));
                mTransBufferManager->freeBufferIndexForRecv(recvBufferId2);
            });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        mTransBufferManager->freeBufferIndexForRecv(recvBufferId2);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        mTransBufferManager->freeBufferIndexForRecv(recvBufferId);
        thread1.join();
        exit(testing::Test::HasFailure() ? 1 : 0);
    }
    else
    {
        int status;
        ASSERT_NE(-1, waitpid(pid, &status, 0)) << "waitpid failed";
        ASSERT_TRUE(WIFEXITED(status)) << "Child process terminated abnormally";
        ASSERT_EQ(0, WEXITSTATUS(status)) << "Test in child process failed";
    }
}

// TODO: test for numtoken is nullopt

TEST_F(CacheTransBufferTest, TestForNullOptAndNoneTransSize)
{

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "Fork failed";
    if (pid == 0)
    {
        std::optional<size_t> maxNumTokens = std::nullopt;
        SizeType32 maxBlocksPerSeq = 10;
        SizeType32 tokensPerBlock = 8;
        setenv("TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE", "0B", 1);
        SetUpCacheTransBuffer(4, 2, 64, tokensPerBlock, CacheType::kSELF, maxNumTokens, maxBlocksPerSeq);
        auto bufferId = mTransBufferManager->assignBufferIndexForSend();
        EXPECT_FALSE(bufferId.has_value());
        mTransBufferManager->freeBufferIndexForSend(bufferId);
        auto bufferId2 = mTransBufferManager->assignBufferIndexForRecv();
        EXPECT_FALSE(bufferId2.has_value());
        mTransBufferManager->freeBufferIndexForRecv(bufferId2);
        auto bufferId3 = mTransBufferManager->assignBufferIndexForSend();
        EXPECT_FALSE(bufferId3.has_value());
        auto bufferManager = tensorrt_llm::runtime::BufferManager{std::make_shared<CudaStream>()};
        auto targetNum = 2;
        auto targetSize = 1024;
        std::vector<size_t> targetSizeVec = std::vector<size_t>(targetNum, targetSize);
        auto [sendBuffers, bufferCoverTargetNum, onlyUseDynamicBuffer]
            = mTransBufferManager->getOrAllocateSendBuffers(bufferId3, targetNum, targetSizeVec, bufferManager);
        EXPECT_EQ(sendBuffers.size(), targetNum);
        EXPECT_EQ(bufferCoverTargetNum, targetNum);
        EXPECT_EQ(onlyUseDynamicBuffer, true);
        mTransBufferManager->freeBufferIndexForSend(bufferId3);
        EXPECT_EQ(sendBuffers.at(0)->getSize(), targetSize);

        EXPECT_EQ(mTransBufferManager->getSendBuffer(bufferId3), nullptr);
        exit(testing::Test::HasFailure() ? 1 : 0);
    }
    else
    {
        int status;
        ASSERT_NE(-1, waitpid(pid, &status, 0)) << "waitpid failed";
        ASSERT_TRUE(WIFEXITED(status)) << "Child process terminated abnormally";
        ASSERT_EQ(0, WEXITSTATUS(status)) << "Test in child process failed";
    }
}

TEST_F(CacheTransBufferTest, TestForNullOptAndDefaultTransSize)
{

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "Fork failed";
    if (pid == 0)
    {
        std::optional<size_t> maxNumTokens = std::nullopt;
        SizeType32 maxBlocksPerSeq = 10;
        SizeType32 tokensPerBlock = 8;
        SetUpCacheTransBuffer(4, 2, 64, tokensPerBlock, CacheType::kSELF, maxNumTokens, maxBlocksPerSeq);
        auto defaultTransSize = tensorrt_llm::common::getEnvMemSizeForKVCacheTransferBuffer();
        TLLM_LOG_INFO("defaultTransSize: %d", defaultTransSize);
        EXPECT_GT(defaultTransSize, 0);
        auto bufferId = mTransBufferManager->assignBufferIndexForSend();
        EXPECT_TRUE(bufferId.has_value());
        EXPECT_EQ(bufferId.value(), 0);
        mTransBufferManager->freeBufferIndexForSend(bufferId);
        auto bufferId2 = mTransBufferManager->assignBufferIndexForRecv();
        EXPECT_TRUE(bufferId2.has_value());
        EXPECT_EQ(bufferId2.value(), 0);
        mTransBufferManager->freeBufferIndexForRecv(bufferId2);
        auto bufferId3 = mTransBufferManager->assignBufferIndexForSend();
        EXPECT_TRUE(bufferId3.has_value());
        auto bufferManager = tensorrt_llm::runtime::BufferManager{std::make_shared<CudaStream>()};
        auto targetNum = 2;
        auto targetSize = 1024;
        std::vector<size_t> targetSizeVec = std::vector<size_t>(targetNum, targetSize);
        auto [sendBuffers, bufferCoverTargetNum, onlyUseDynamicBuffer]
            = mTransBufferManager->getOrAllocateSendBuffers(bufferId3, targetNum, targetSizeVec, bufferManager);
        EXPECT_EQ(sendBuffers.size(), targetNum);
        EXPECT_EQ(bufferCoverTargetNum, targetNum);
        EXPECT_EQ(onlyUseDynamicBuffer, false);
        EXPECT_EQ(mTransBufferManager->getSendBuffer(bufferId3)->getSizeInBytes(), defaultTransSize);
        mTransBufferManager->freeBufferIndexForSend(bufferId3);
        EXPECT_EQ(sendBuffers.at(0)->getSize(), targetSize);

        targetNum = 4;
        targetSize = defaultTransSize / 4 / 2; // float 4 bytes
        auto bufferId4 = mTransBufferManager->assignBufferIndexForSend();
        EXPECT_TRUE(bufferId4.has_value());
        EXPECT_EQ(bufferId4.value(), 0);
        targetSizeVec = std::vector<size_t>(targetNum, targetSize);
        auto [sendBuffers2, bufferCoverTargetNum2, onlyUseDynamicBuffer2]
            = mTransBufferManager->getOrAllocateSendBuffers(bufferId4, targetNum, targetSizeVec, bufferManager);
        EXPECT_EQ(sendBuffers2.size(), targetNum);
        EXPECT_EQ(bufferCoverTargetNum2, targetNum / 2);
        EXPECT_EQ(onlyUseDynamicBuffer2, false);
        mTransBufferManager->freeBufferIndexForSend(bufferId4);

        targetSize = defaultTransSize / 4 / 8;
        auto bufferId5 = mTransBufferManager->assignBufferIndexForSend();
        EXPECT_TRUE(bufferId5.has_value());
        EXPECT_EQ(bufferId5.value(), 0);
        targetSizeVec = std::vector<size_t>(targetNum, targetSize);
        auto [sendBuffers3, bufferCoverTargetNum3, onlyUseDynamicBuffer3]
            = mTransBufferManager->getOrAllocateSendBuffers(bufferId5, targetNum, targetSizeVec, bufferManager);
        EXPECT_EQ(sendBuffers3.size(), targetNum);
        EXPECT_EQ(bufferCoverTargetNum3, targetNum);
        EXPECT_EQ(onlyUseDynamicBuffer3, false);
        mTransBufferManager->freeBufferIndexForSend(bufferId5);
        exit(testing::Test::HasFailure() ? 1 : 0);
    }
    else
    {
        int status;
        ASSERT_NE(-1, waitpid(pid, &status, 0)) << "waitpid failed";
        ASSERT_TRUE(WIFEXITED(status)) << "Child process terminated abnormally";
        ASSERT_EQ(0, WEXITSTATUS(status)) << "Test in child process failed";
    }
}

// TODO: pybinding

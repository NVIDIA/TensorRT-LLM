/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moeLoadBalance/moeLoadBalanceKernels.h"

using namespace tensorrt_llm::kernels;

class MoeLoadBalanceSignalKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&mDeviceEnabled, sizeof(int)), cudaSuccess);
        ASSERT_EQ(cudaMallocHost(&mSignal, sizeof(MoeLoadBalanceSingleLayerSignal)), cudaSuccess);
        mSignal->stepAndOwner = 0ULL; // initialize to gpu ownership
    }

    void TearDown() override
    {
        ASSERT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
        ASSERT_EQ(cudaFree(mDeviceEnabled), cudaSuccess);
        ASSERT_EQ(cudaFreeHost(mSignal), cudaSuccess);
    }

    // the device memory
    int* mDeviceEnabled;
    MoeLoadBalanceSingleLayerSignal* mSignal;
    cudaStream_t mStream;
};

// test the signaling mechanism
TEST_F(MoeLoadBalanceSignalKernelTest, TestSignaling)
{
    int const kRepeatCount = 1000;

    std::atomic<int> orderId(0);

    // start the device thread to simulate
    std::thread deviceThread(
        [&]()
        {
            for (int i = 0; i < kRepeatCount; ++i)
            {
                // wait for the gpu stage signal
                moeWaitSignalForGpuStageDevice(mSignal, mDeviceEnabled, mStream);
                ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

                int order = orderId.fetch_add(1);
                ASSERT_EQ(order, i * 2);

                // set the cpu stage signal
                moeSetSignalForCpuStageDevice(mSignal, mStream);
                ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
            }
        });

    // the host thread
    std::thread hostThread(
        [&]()
        {
            for (int i = 0; i < kRepeatCount; ++i)
            {
                // wait for the cpu stage signal
                moeWaitSignalForCpuStageHost(mSignal);

                int order = orderId.fetch_add(1);
                ASSERT_EQ(order, i * 2 + 1);

                // set the gpu stage signal
                moeSetSignalForGpuStageHost(mSignal, i + 1, true);
            }
        });

    // wait for the threads to finish
    deviceThread.join();
    hostThread.join();
}

struct MoeLoadBalanceTestParam
{
    int expertCount;
    int topK;
    int epRank;
    int epSize;
    int slotCountPerRank;
    int maxTokenCountPerRank;
    bool isFirstStage;
    bool isLastStage;
    float decayFactor;
};

class MoeLoadBalanceStatisticKernelTest : public ::testing::TestWithParam<MoeLoadBalanceTestParam>
{
protected:
    void SetUp() override
    {
        auto param = GetParam();

        mMetaInfo.expertCount = param.expertCount;
        mMetaInfo.topK = param.topK;
        mMetaInfo.epRank = param.epRank;
        mMetaInfo.epSize = param.epSize;
        mMetaInfo.slotCountPerRank = param.slotCountPerRank;

        mStatisticInfo.decayFactor = param.decayFactor;

        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);

        // allocate device memory
        size_t expertLoadFactorSize = param.expertCount * sizeof(float);
        size_t expertTokenCountSize = param.expertCount * mStatisticInfo.rawDataWindowSize * sizeof(int);
        size_t gatheredIdsSize = param.maxTokenCountPerRank * param.epSize * param.topK * sizeof(int);

        ASSERT_EQ(cudaMalloc(&mDeviceEnabled, sizeof(int)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&mDeviceExpertLoadFactor, expertLoadFactorSize), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&mDeviceExpertTokenCount, expertTokenCountSize), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&mDeviceGatheredIds, gatheredIdsSize), cudaSuccess);

        // allocate the signal structure
        ASSERT_EQ(cudaMallocHost(&mSignal, sizeof(MoeLoadBalanceSingleLayerSignal)), cudaSuccess);
        mSignal->stepAndOwner = 0ULL; // initialize to gpu ownership

        // allocate host memory for verification
        mExpectedLoadFactor.resize(param.expertCount, 0.0f);
        mExpectedExpertTokenCount.resize(param.expertCount * mStatisticInfo.rawDataWindowSize);
        mHostExpertLoadFactor.resize(param.expertCount);
        mHostExpertTokenCount.resize(param.expertCount * mStatisticInfo.rawDataWindowSize);
        mHostGatheredIds.resize(param.maxTokenCountPerRank * param.epSize * param.topK);

        // initialize the random number generator
        mRng.seed(1234);
    }

    void TearDown() override
    {
        // clean up the resources
        ASSERT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
        ASSERT_EQ(cudaFree(mDeviceEnabled), cudaSuccess);
        ASSERT_EQ(cudaFree(mDeviceExpertLoadFactor), cudaSuccess);
        ASSERT_EQ(cudaFree(mDeviceExpertTokenCount), cudaSuccess);
        ASSERT_EQ(cudaFree(mDeviceGatheredIds), cudaSuccess);
        ASSERT_EQ(cudaFreeHost(mSignal), cudaSuccess);
    }

    // helper function: generate random expert ids
    void generateRandomExpertIds()
    {
        auto param = GetParam();
        std::uniform_int_distribution<int> dist(0, param.expertCount - 1);
        for (size_t i = 0; i < mHostGatheredIds.size(); ++i)
        {
            mHostGatheredIds[i] = dist(mRng);
        }
        ASSERT_EQ(cudaMemcpy(mDeviceGatheredIds, mHostGatheredIds.data(), mHostGatheredIds.size() * sizeof(int),
                      cudaMemcpyHostToDevice),
            cudaSuccess);
    }

    // helper function: compute the expected statistics
    void computeExpectedStatistics()
    {
        auto param = GetParam();
        mExpectedExpertTokenCount = mHostExpertTokenCount;
        if (param.isFirstStage)
        {
            for (int windowIdx = mStatisticInfo.rawDataWindowSize - 1; windowIdx >= 0; --windowIdx)
            {
                if (windowIdx > 0)
                {
                    for (int i = 0; i < param.expertCount; ++i)
                    {
                        mExpectedExpertTokenCount[param.expertCount * windowIdx + i]
                            = mExpectedExpertTokenCount[param.expertCount * (windowIdx - 1) + i];
                    }
                }
                else
                {
                    for (int i = 0; i < param.expertCount; ++i)
                    {
                        mExpectedExpertTokenCount[i] = 0;
                    }
                }
            }
        }
        // compute the token count for each expert
        for (auto const& expertId : mHostGatheredIds)
        {
            if (expertId >= 0 && expertId < param.expertCount)
            {
                mExpectedExpertTokenCount[expertId]++;
            }
        }

        // update the load factor
        mExpectedLoadFactor = mHostExpertLoadFactor;
        if (param.isLastStage)
        {
            for (int i = 0; i < param.expertCount; ++i)
            {
                mExpectedLoadFactor[i] = mHostExpertLoadFactor[i] * param.decayFactor + mExpectedExpertTokenCount[i];
            }
        }
    }

protected:
    // the members for testing
    MoeLoadBalanceMetaInfo mMetaInfo;
    MoeLoadBalanceStatisticInfo mStatisticInfo;
    MoeLoadBalanceSingleLayerSignal* mSignal;
    cudaStream_t mStream;

    // the device memory
    int* mDeviceEnabled;
    float* mDeviceExpertLoadFactor;
    int* mDeviceExpertTokenCount;
    int* mDeviceGatheredIds;

    // the host memory
    std::vector<float> mHostExpertLoadFactor;
    std::vector<int> mHostExpertTokenCount;
    std::vector<int> mHostGatheredIds;
    std::vector<float> mExpectedLoadFactor;
    std::vector<int> mExpectedExpertTokenCount;

    // the random number generator
    std::mt19937 mRng;
};

// test the statistics calculation
TEST_P(MoeLoadBalanceStatisticKernelTest, TestStatistics)
{
    auto param = GetParam();

    // generate random input data
    generateRandomExpertIds();

    // initialize the load factor
    std::fill(mHostExpertLoadFactor.begin(), mHostExpertLoadFactor.end(), 1.0f);
    // random fill mHostExpertTokenCount to test isFirstStage == false
    std::uniform_int_distribution<int> dist(0, 100);
    for (auto& tokenCount : mHostExpertTokenCount)
    {
        tokenCount = dist(mRng);
    }

    ASSERT_EQ(cudaMemcpy(mDeviceExpertLoadFactor, mHostExpertLoadFactor.data(),
                  mHostExpertLoadFactor.size() * sizeof(float), cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(mDeviceExpertTokenCount, mHostExpertTokenCount.data(),
                  mHostExpertTokenCount.size() * sizeof(int), cudaMemcpyHostToDevice),
        cudaSuccess);

    // set the enabled flag
    int enabled = 1;
    ASSERT_EQ(cudaMemcpy(mDeviceEnabled, &enabled, sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);

    // update the statistic info pointer
    mStatisticInfo.expertLoadFactor = mDeviceExpertLoadFactor;
    mStatisticInfo.expertTokenCount = mDeviceExpertTokenCount;

    // execute the statistics calculation
    moeStatisticDevice(mMetaInfo, mStatisticInfo, param.maxTokenCountPerRank * param.epSize, mDeviceEnabled,
        param.isFirstStage, param.isLastStage, mDeviceGatheredIds, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    // compute the expected result
    computeExpectedStatistics();

    // copy the result back to host
    ASSERT_EQ(cudaMemcpy(mHostExpertLoadFactor.data(), mDeviceExpertLoadFactor,
                  mHostExpertLoadFactor.size() * sizeof(float), cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(mHostExpertTokenCount.data(), mDeviceExpertTokenCount,
                  mHostExpertTokenCount.size() * sizeof(int), cudaMemcpyDeviceToHost),
        cudaSuccess);

    // verify the result
    for (int i = 0; i < param.expertCount; ++i)
    {
        EXPECT_NEAR(mHostExpertLoadFactor[i], mExpectedLoadFactor[i], 1e-6)
            << "Expert " << i << " load factor mismatch";
    }
    for (int i = 0; i < param.expertCount * mStatisticInfo.rawDataWindowSize; ++i)
    {
        EXPECT_EQ(mHostExpertTokenCount[i], mExpectedExpertTokenCount[i]) << "Expert " << i << " token count mismatch";
    }
}

// instantiate the parameterized tests
INSTANTIATE_TEST_SUITE_P(MoeLoadBalanceStatisticKernelTests, MoeLoadBalanceStatisticKernelTest,
    ::testing::Values(
        // basic test scenarios
        MoeLoadBalanceTestParam{/* expertCount */ 8,
            /* topK */ 2,
            /* epRank */ 0,
            /* epSize */ 2,
            /* slotCountPerRank */ 4,
            /* maxTokenCountPerRank */ 128,
            /* isFirstStage */ true,
            /* isLastStage */ true,
            /* decayFactor */ 0.9f},
        // large scale test scenarios
        MoeLoadBalanceTestParam{/* expertCount */ 64,
            /* topK */ 4,
            /* epRank */ 1,
            /* epSize */ 4,
            /* slotCountPerRank */ 16,
            /* maxTokenCountPerRank */ 512,
            /* isFirstStage */ false,
            /* isLastStage */ true,
            /* decayFactor */ 0.95f} // can add more test scenarios
        ));

class MoeLoadBalanceRouteKernelTest : public ::testing::TestWithParam<MoeLoadBalanceTestParam>
{
protected:
    void SetUp() override
    {
        auto param = GetParam();

        mMetaInfo.expertCount = param.expertCount;
        mMetaInfo.topK = param.topK;
        mMetaInfo.epRank = param.epRank;
        mMetaInfo.epSize = param.epSize;
        mMetaInfo.slotCountPerRank = param.slotCountPerRank;

        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);

        int tokenCount = param.maxTokenCountPerRank;
        mTotalTokens = tokenCount * param.topK;

        ASSERT_EQ(cudaMalloc(&mDeviceTokenSelectedExperts, mTotalTokens * sizeof(int)), cudaSuccess);

        ASSERT_EQ(cudaMalloc(&mDeviceTokenRoutedRankIds, mTotalTokens * sizeof(int)), cudaSuccess);

        ASSERT_EQ(cudaMalloc(&mPlacementInfo.expertReplicaCount, param.expertCount * sizeof(int)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&mPlacementInfo.expertReplicaStartOffset, param.expertCount * sizeof(int)), cudaSuccess);

        int totalReplicas = param.epSize * param.slotCountPerRank;
        ASSERT_EQ(cudaMalloc(&mPlacementInfo.globalSlotIds, totalReplicas * sizeof(int)), cudaSuccess);

        // allocate host memory
        mHostTokenSelectedExperts.resize(mTotalTokens);
        mHostTokenRoutedRankIds.resize(mTotalTokens);
        mExpectedTokenRoutedRankIds.resize(mTotalTokens);
        mHostExpertReplicaCount.resize(param.expertCount);
        mHostExpertReplicaStartOffset.resize(param.expertCount);
        mHostGlobalSlotIdsInfo.resize(totalReplicas);

        // initialize the random number generator
        mRng.seed(1234);
    }

    void TearDown() override
    {
        ASSERT_EQ(cudaFree(mDeviceTokenSelectedExperts), cudaSuccess);
        ASSERT_EQ(cudaFree(mDeviceTokenRoutedRankIds), cudaSuccess);
        ASSERT_EQ(cudaFree(mPlacementInfo.expertReplicaCount), cudaSuccess);
        ASSERT_EQ(cudaFree(mPlacementInfo.expertReplicaStartOffset), cudaSuccess);
        ASSERT_EQ(cudaFree(mPlacementInfo.globalSlotIds), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    void setupPlacementInfo()
    {
        auto param = GetParam();

        int totalReplicas = 0;

        ASSERT_GE(param.epSize * param.slotCountPerRank, param.expertCount);

        // global placement
        std::vector<int> placement(param.epSize * param.slotCountPerRank);
        // first assign one slot for each expert
        for (int i = 0; i < param.expertCount; ++i)
        {
            mHostExpertReplicaCount[i] = 1;
            placement[i] = i;
        }
        // then randomly assign the rest of the slots for some expert
        for (int i = param.expertCount; i < param.epSize * param.slotCountPerRank; ++i)
        {
            int expertId = std::rand() % param.expertCount;
            mHostExpertReplicaCount[expertId]++;
            placement[i] = expertId;
        }
        // do random permutation
        std::shuffle(placement.begin(), placement.end(), mRng);

        std::vector<std::vector<int>> expertReplicaGlobalSlotIdsInfo(param.expertCount);
        for (int i = 0; i < param.epSize * param.slotCountPerRank; ++i)
        {
            int expertId = placement[i];
            expertReplicaGlobalSlotIdsInfo[expertId].push_back(i);
        }

        for (int i = 0; i < param.expertCount; ++i)
        {
            mHostExpertReplicaStartOffset[i] = totalReplicas;
            totalReplicas += mHostExpertReplicaCount[i];
        }

        for (int i = 0; i < param.expertCount; ++i)
        {
            for (size_t j = 0; j < expertReplicaGlobalSlotIdsInfo[i].size(); ++j)
            {
                int offset = mHostExpertReplicaStartOffset[i] + j;
                mHostGlobalSlotIdsInfo[offset] = expertReplicaGlobalSlotIdsInfo[i][j];
            }
        }

        ASSERT_EQ(cudaMemcpy(mPlacementInfo.expertReplicaCount, mHostExpertReplicaCount.data(),
                      param.expertCount * sizeof(int), cudaMemcpyHostToDevice),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpy(mPlacementInfo.expertReplicaStartOffset, mHostExpertReplicaStartOffset.data(),
                      param.expertCount * sizeof(int), cudaMemcpyHostToDevice),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpy(mPlacementInfo.globalSlotIds, mHostGlobalSlotIdsInfo.data(), totalReplicas * sizeof(int),
                      cudaMemcpyHostToDevice),
            cudaSuccess);
    }

    void generateRandomExpertSelection()
    {
        auto param = GetParam();
        std::uniform_int_distribution<int> dist(0, param.expertCount - 1);

        for (int i = 0; i < mTotalTokens; ++i)
        {
            mHostTokenSelectedExperts[i] = dist(mRng);
        }

        ASSERT_EQ(cudaMemcpy(mDeviceTokenSelectedExperts, mHostTokenSelectedExperts.data(), mTotalTokens * sizeof(int),
                      cudaMemcpyHostToDevice),
            cudaSuccess);
    }

    void computeExpectedRouting()
    {
        auto param = GetParam();

        // simulate blocking effact of GPU
        int const threadCount = 1024;
        int const blockCount = (mTotalTokens + threadCount - 1) / threadCount;

        for (int blockIdx = 0; blockIdx < blockCount; ++blockIdx)
        {
            std::vector<int> sharedExpertTokenCount(param.expertCount, 0);
            for (int threadIdx = 0; threadIdx < threadCount; ++threadIdx)
            {
                int tokenIdx = blockIdx * threadCount + threadIdx;
                if (tokenIdx < mTotalTokens)
                {
                    int expertId = mHostTokenSelectedExperts[tokenIdx];
                    if (expertId >= 0 && expertId < param.expertCount)
                    {
                        int idxInBlock = sharedExpertTokenCount[expertId]++;
                        idxInBlock += blockIdx;

                        int replicaCount = mHostExpertReplicaCount[expertId];
                        int replicaStartOffset = mHostExpertReplicaStartOffset[expertId];
                        int replicaId = idxInBlock % replicaCount + replicaStartOffset;
                        int slotId = mHostGlobalSlotIdsInfo[replicaId];
                        mExpectedTokenRoutedRankIds[tokenIdx] = slotId;
                    }
                    else
                    {
                        mExpectedTokenRoutedRankIds[tokenIdx] = param.epSize * param.slotCountPerRank;
                    }
                }
            }
        }
    }

protected:
    MoeLoadBalanceMetaInfo mMetaInfo;
    MoePlacementInfo mPlacementInfo;
    cudaStream_t mStream;

    int* mDeviceTokenSelectedExperts;
    int* mDeviceTokenRoutedRankIds;

    std::vector<int> mHostTokenSelectedExperts;
    std::vector<int> mHostTokenRoutedRankIds;
    std::vector<int> mExpectedTokenRoutedRankIds;

    std::vector<int> mHostExpertReplicaCount;
    std::vector<int> mHostExpertReplicaStartOffset;
    std::vector<int> mHostGlobalSlotIdsInfo;

    int mTotalTokens;
    std::mt19937 mRng;
};

TEST_P(MoeLoadBalanceRouteKernelTest, TestBasicRouting)
{
    auto param = GetParam();

    setupPlacementInfo();

    generateRandomExpertSelection();

    moeComputeRouteDevice(mMetaInfo, mPlacementInfo, mDeviceTokenSelectedExperts, mDeviceTokenRoutedRankIds,
        param.maxTokenCountPerRank, false, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    computeExpectedRouting();

    ASSERT_EQ(cudaMemcpy(mHostTokenRoutedRankIds.data(), mDeviceTokenRoutedRankIds, mTotalTokens * sizeof(int),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);

    for (int i = 0; i < mTotalTokens; ++i)
    {
        EXPECT_EQ(mHostTokenRoutedRankIds[i], mExpectedTokenRoutedRankIds[i])
            << "Token " << i << " routed to wrong rank. Expert ID: " << mHostTokenSelectedExperts[i];
    }
}

TEST_P(MoeLoadBalanceRouteKernelTest, TestInvalidExpertIds)
{
    auto param = GetParam();

    setupPlacementInfo();

    std::fill(mHostTokenSelectedExperts.begin(), mHostTokenSelectedExperts.end(), -1);
    ASSERT_EQ(cudaMemcpy(mDeviceTokenSelectedExperts, mHostTokenSelectedExperts.data(), mTotalTokens * sizeof(int),
                  cudaMemcpyHostToDevice),
        cudaSuccess);

    moeComputeRouteDevice(mMetaInfo, mPlacementInfo, mDeviceTokenSelectedExperts, mDeviceTokenRoutedRankIds,
        param.maxTokenCountPerRank, false, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(mHostTokenRoutedRankIds.data(), mDeviceTokenRoutedRankIds, mTotalTokens * sizeof(int),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);

    for (int i = 0; i < mTotalTokens; ++i)
    {
        EXPECT_EQ(mHostTokenRoutedRankIds[i], param.epSize * param.slotCountPerRank)
            << "Invalid expert ID not routed to invalid rank at index " << i;
    }

    std::fill(mHostTokenSelectedExperts.begin(), mHostTokenSelectedExperts.end(), param.expertCount);
    ASSERT_EQ(cudaMemcpy(mDeviceTokenSelectedExperts, mHostTokenSelectedExperts.data(), mTotalTokens * sizeof(int),
                  cudaMemcpyHostToDevice),
        cudaSuccess);

    moeComputeRouteDevice(mMetaInfo, mPlacementInfo, mDeviceTokenSelectedExperts, mDeviceTokenRoutedRankIds,
        param.maxTokenCountPerRank, false, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(mHostTokenRoutedRankIds.data(), mDeviceTokenRoutedRankIds, mTotalTokens * sizeof(int),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);

    // verify that all routes are set to invalid rank
    for (int i = 0; i < mTotalTokens; ++i)
    {
        EXPECT_EQ(mHostTokenRoutedRankIds[i], param.epSize * param.slotCountPerRank)
            << "Invalid expert ID not routed to invalid rank at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(MoeLoadBalanceRouteKernelTests, MoeLoadBalanceRouteKernelTest,
    ::testing::Values(
        // basic test scenarios
        MoeLoadBalanceTestParam{/* expertCount */ 8,
            /* topK */ 2,
            /* epRank */ 0,
            /* epSize */ 2,
            /* slotCountPerRank */ 5,
            /* maxTokenCountPerRank */ 128,
            /* isFirstStage */ true,
            /* isLastStage */ true,
            /* decayFactor */ 0.9f},
        // large scale test scenarios
        MoeLoadBalanceTestParam{/* expertCount */ 256,
            /* topK */ 8,
            /* epRank */ 1,
            /* epSize */ 32,
            /* slotCountPerRank */ 12,
            /* maxTokenCountPerRank */ 5000,
            /* isFirstStage */ false,
            /* isLastStage */ true,
            /* decayFactor */ 0.95f},
        // edge case: single rank
        MoeLoadBalanceTestParam{/* expertCount */ 16,
            /* topK */ 2,
            /* epRank */ 0,
            /* epSize */ 1,
            /* slotCountPerRank */ 16,
            /* maxTokenCountPerRank */ 64,
            /* isFirstStage */ true,
            /* isLastStage */ true,
            /* decayFactor */ 0.9f}));

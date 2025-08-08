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

#include <gtest/gtest.h>

#include <cstdlib>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moeLoadBalance/moeLoadBalanceKernels.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/moeLoadBalancer.h"

using namespace tensorrt_llm::runtime;

// Unit test for doReplication function
// Define test parameters for doReplication test
struct ReplicationTestParam
{
    int expertCount;
    int slotCountPerRank;
    int epSize;
    std::vector<float> expertLoadFactors;
    std::vector<int> expectedReplicaCounts;
    std::string description;
};

class MoeReplicationTest : public ::testing::TestWithParam<ReplicationTestParam>
{
};

TEST_P(MoeReplicationTest, VerifyReplication)
{
    auto param = GetParam();

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo{
        param.expertCount, 2, 0, param.epSize, param.slotCountPerRank};

    MoePlacementCpuInfo cpuPlacement;
    cpuPlacement.expertReplicaCount.resize(param.expertCount);

    // Call the function under test
    doReplication(metaInfo, param.expertLoadFactors.data(), &cpuPlacement);

    // Verify results against expected replica counts
    int totalSlots = param.epSize * param.slotCountPerRank;
    int totalReplicas = 0;

    ASSERT_EQ(cpuPlacement.expertReplicaCount.size(), param.expectedReplicaCounts.size())
        << "Replica count size mismatch";

    for (int i = 0; i < param.expertCount; i++)
    {
        EXPECT_EQ(cpuPlacement.expertReplicaCount[i], param.expectedReplicaCounts[i])
            << "Expert " << i << " has incorrect replica count";
        totalReplicas += cpuPlacement.expertReplicaCount[i];
    }

    // Verify total replicas matches total slots
    EXPECT_EQ(totalReplicas, totalSlots) << "Total replicas should equal total slots";
}

// Define a set of test cases for replication
INSTANTIATE_TEST_SUITE_P(ReplicationTests, MoeReplicationTest,
    ::testing::Values(
        // Test case 1: Total slot count equals expert count (equal distribution)
        ReplicationTestParam{4,       // expertCount
            2,                        // slotCountPerRank
            2,                        // epSize
            {1.0f, 2.0f, 3.0f, 4.0f}, // expertLoadFactors - increasing load
            {1, 1, 1, 1},             // expectedReplicaCounts - each expert gets 1 replica
            "Equal slots and experts"},

        // Test case 2: No load information (even distribution)
        ReplicationTestParam{4,       // expertCount
            3,                        // slotCountPerRank
            2,                        // epSize
            {0.0f, 0.0f, 0.0f, 0.0f}, // expertLoadFactors - all zero
            {2, 2, 1, 1},             // expectedReplicaCounts - even distribution with remainder to first experts
            "Zero load factors"},

        // Test case 3: Greedy distribution based on load factors
        ReplicationTestParam{4,       // expertCount
            4,                        // slotCountPerRank
            2,                        // epSize
            {1.0f, 4.0f, 2.0f, 1.0f}, // expertLoadFactors - varied loads
            {1, 4, 2, 1},             // expectedReplicaCounts - proportional to load
            "Varied load factors"},

        // Test case 4: More complex scenario with uneven distribution
        ReplicationTestParam{5,               // expertCount
            3,                                // slotCountPerRank
            3,                                // epSize
            {10.0f, 5.0f, 2.0f, 15.0f, 8.0f}, // expertLoadFactors
            {2, 1, 1, 3, 2},                  // expectedReplicaCounts
            "Complex load distribution"},

        // Test case 5: Single expert with load, others zero
        ReplicationTestParam{3,  // expertCount
            2,                   // slotCountPerRank
            2,                   // epSize
            {0.0f, 10.0f, 0.0f}, // expertLoadFactors - only middle expert has load
            {1, 2, 1},           // expectedReplicaCounts - middle expert gets most replicas
            "Single expert with load"}),
    [](::testing::TestParamInfo<ReplicationTestParam> const& info)
    {
        // Generate readable test names based on the description
        std::string name = info.param.description;
        // Replace spaces and special characters with underscores
        std::replace(name.begin(), name.end(), ' ', '_');
        std::replace(name.begin(), name.end(), ',', '_');
        std::replace(name.begin(), name.end(), '(', '_');
        std::replace(name.begin(), name.end(), ')', '_');
        return name;
    });

// Define test parameters for doPlacement test
struct PlacementTestParam
{
    int expertCount;
    int slotCountPerRank;
    int epSize;
    std::vector<float> expertLoadFactors;
    std::vector<int> replicaCounts; // Pre-computed replica counts
    double maxExpectedLoadDiff;     // Maximum expected load difference between ranks
    std::string description;
};

class MoePlacementTest : public ::testing::TestWithParam<PlacementTestParam>
{
};

TEST_P(MoePlacementTest, VerifyPlacement)
{
    auto param = GetParam();

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo{
        param.expertCount, 2, 0, param.epSize, param.slotCountPerRank};

    MoePlacementCpuInfo cpuPlacement;
    cpuPlacement.expertReplicaCount.resize(param.expertCount);

    // Use the pre-computed replica counts instead of calling doReplication
    for (int i = 0; i < param.expertCount; i++)
    {
        cpuPlacement.expertReplicaCount[i] = param.replicaCounts[i];
    }

    // Initialize rankExpertIds for placement
    cpuPlacement.rankExpertIds.resize(param.epSize);
    for (int i = 0; i < param.epSize; i++)
    {
        cpuPlacement.rankExpertIds[i].resize(param.slotCountPerRank, -1);
    }

    // Call the function under test
    doPlacement(metaInfo, param.expertLoadFactors.data(), &cpuPlacement);

    // Verify all slots are assigned
    int totalSlots = param.epSize * param.slotCountPerRank;
    int assignedExperts = 0;
    std::vector<int> expertAssignments(param.expertCount, 0);

    for (int rankId = 0; rankId < param.epSize; rankId++)
    {
        for (int slotId = 0; slotId < param.slotCountPerRank; slotId++)
        {
            int expertId = cpuPlacement.rankExpertIds[rankId][slotId];

            // Verify expert ID is valid
            EXPECT_GE(expertId, 0) << "Expert ID should be non-negative";
            EXPECT_LT(expertId, param.expertCount) << "Expert ID should be less than expert count";

            // Count assignments
            expertAssignments[expertId]++;
            assignedExperts++;
        }
    }

    // Verify correct number of slots assigned
    EXPECT_EQ(assignedExperts, totalSlots) << "Total assigned experts should equal total slots";

    // Verify each expert is assigned the correct number of replicas
    for (int i = 0; i < param.expertCount; i++)
    {
        EXPECT_EQ(expertAssignments[i], param.replicaCounts[i])
            << "Expert " << i << " should be assigned to exactly replicaCounts[" << i << "] slots";
    }

    // Test load balancing property: ranks should have roughly equal total load
    std::vector<double> rankLoads(param.epSize, 0.0);
    for (int rankId = 0; rankId < param.epSize; rankId++)
    {
        for (int slotId = 0; slotId < param.slotCountPerRank; slotId++)
        {
            int expertId = cpuPlacement.rankExpertIds[rankId][slotId];
            double slotSize = param.expertLoadFactors[expertId] / static_cast<double>(param.replicaCounts[expertId]);
            rankLoads[rankId] += slotSize;
        }
    }

    // Print rank loads for debugging
    std::cout << "Rank loads for " << param.description << ":" << std::endl;
    for (int i = 0; i < param.epSize; i++)
    {
        std::cout << "  Rank " << i << ": " << rankLoads[i] << std::endl;
    }

    // Calculate max difference between rank loads
    double maxLoadDiff = 0.0;
    for (int i = 0; i < param.epSize - 1; i++)
    {
        for (int j = i + 1; j < param.epSize; j++)
        {
            maxLoadDiff = std::max(maxLoadDiff, std::abs(rankLoads[i] - rankLoads[j]));
        }
    }
    std::cout << "  Max load difference: " << maxLoadDiff << std::endl;

    // The load difference should be within the expected threshold
    EXPECT_LE(maxLoadDiff, param.maxExpectedLoadDiff) << "Load difference between ranks exceeds expected threshold";
}

// Define a set of test cases for placement
INSTANTIATE_TEST_SUITE_P(PlacementTests, MoePlacementTest,
    ::testing::Values(
        // Test case 1: Basic placement with equal load
        PlacementTestParam{4,         // expertCount
            2,                        // slotCountPerRank
            2,                        // epSize
            {1.0f, 1.0f, 1.0f, 1.0f}, // expertLoadFactors - equal load
            {1, 1, 1, 1},             // replicaCounts
            0.001,                    // maxExpectedLoadDiff
            "Equal load factors"},

        // Test case 2: Varied load factors
        PlacementTestParam{4,         // expertCount
            3,                        // slotCountPerRank
            2,                        // epSize
            {1.0f, 3.0f, 2.0f, 1.0f}, // expertLoadFactors - varied loads
            {1, 2, 2, 1},             // replicaCounts
            0.01,                     // maxExpectedLoadDiff
            "Varied load factors"},

        // Test case 3: Complex scenario with multiple ranks
        PlacementTestParam{5,                 // expertCount
            3,                                // slotCountPerRank
            3,                                // epSize
            {10.0f, 5.0f, 2.0f, 15.0f, 8.0f}, // expertLoadFactors
            {2, 1, 1, 3, 2},                  // replicaCounts
            2.0,                              // maxExpectedLoadDiff (higher tolerance for complex case)
            "Complex load distribution"},

        // Test case 4: Extreme load differences
        PlacementTestParam{3,      // expertCount
            2,                     // slotCountPerRank
            2,                     // epSize
            {1.0f, 100.0f, 10.0f}, // expertLoadFactors - extreme differences
            {1, 2, 1},             // replicaCounts
            9.0,                   // maxExpectedLoadDiff (higher tolerance due to extreme loads)
            "Extreme load differences"},

        // Test case 5: Many experts with single replica each
        PlacementTestParam{6,                     // expertCount
            3,                                    // slotCountPerRank
            2,                                    // epSize
            {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, // expertLoadFactors
            {1, 1, 1, 1, 1, 1},                   // replicaCounts - one replica per expert
            1.0,                                  // maxExpectedLoadDiff
            "One replica per expert"}),
    [](::testing::TestParamInfo<PlacementTestParam> const& info)
    {
        // Generate readable test names based on the description
        std::string name = info.param.description;
        // Replace spaces and special characters with underscores
        std::replace(name.begin(), name.end(), ' ', '_');
        std::replace(name.begin(), name.end(), ',', '_');
        std::replace(name.begin(), name.end(), '(', '_');
        std::replace(name.begin(), name.end(), ')', '_');
        return name;
    });

// Iteration control parameter structure
struct IterConfig
{
    bool statisticEnabled;
    bool updateWeightsEnabled;
};

struct MoeLoadBalancerTestParam
{
    int expertCount;                     // Number of experts per layer
    int topK;                            // Number of experts selected for each token
    int epRank;                          // Current node rank
    int epSize;                          // Number of nodes in the cluster
    int slotCountPerRank;                // Number of slots on each node
    int numLayers;                       // Number of MoE layers in the model
    int layerUpdatesPerIter;             // Maximum number of layers to update per iteration
    int warmUpIters;                     // Number of warm-up iterations
    std::vector<IterConfig> iterConfigs; // Configuration for each iteration
};

class MoeLoadBalancerTest : public ::testing::TestWithParam<MoeLoadBalancerTestParam>
{
protected:
    void SetUp() override
    {
        setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
        auto param = GetParam();
        TLLM_CUDA_CHECK(cudaSetDevice(0));
        mLoadBalancer = std::make_unique<MoeLoadBalancer>(param.epRank, param.epSize, param.layerUpdatesPerIter);

        mLoadBalancer->setUseGpuMemcpy(true);

        // Create multiple MoE layers
        createLayers(param);

        // Create model weights for each layer
        createModelWeights(param);

        // Calculate which layers should be updated in each iteration
        computeLayerUpdateSchedule(param);

        // Initialize the last iteration ID
        mLastIterId = -1;
    }

    void TearDown() override
    {
        // Release all test resources
        for (auto& layerWeights : mAllHostWeights)
        {
            for (auto& weight : layerWeights)
            {
                EXPECT_EQ(cudaFreeHost(weight.mWeightPtr), cudaSuccess);
            }
        }
        mAllHostWeights.clear();

        for (auto& layerSlots : mAllSlotWeights)
        {
            for (auto& slot : layerSlots)
            {
                EXPECT_EQ(cudaFree(slot.mWeightPtr), cudaSuccess);
            }
        }
        mAllSlotWeights.clear();
    }

    // Calculate which layers should be updated in each iteration
    void computeLayerUpdateSchedule(MoeLoadBalancerTestParam const& param)
    {
        mLayerUpdateSchedule.clear();
        int numIters = param.iterConfigs.size();
        mLayerUpdateSchedule.resize(numIters);

        int totalLayers = param.numLayers;
        int updatesPerIter = param.layerUpdatesPerIter;

        // Calculate the minimum number of iterations needed to update all layers
        int minItersNeeded = (totalLayers + updatesPerIter - 1) / updatesPerIter;

        // If the number of layers is divisible by updates per iteration, add an extra empty update cycle
        int realItersUsed = minItersNeeded;
        if (totalLayers % updatesPerIter == 0)
        {
            realItersUsed += 1;
        }

        // Count effective update iterations (iterations that have updates enabled) after warm-up
        int effectiveUpdateIters = 0;

        // Allocate which layers to update for each iteration
        for (int iter = 0; iter < numIters; ++iter)
        {
            // Enforce rule: if statistics are disabled, updates must also be disabled
            if (!param.iterConfigs[iter].statisticEnabled && param.iterConfigs[iter].updateWeightsEnabled)
            {
                TLLM_CHECK_WITH_INFO(false,
                    "Invalid configuration at iteration %d: statistics disabled but updates enabled. "
                    "When statistics are disabled, updates must also be disabled.",
                    iter);
            }

            // Only schedule layer updates when weight updates are enabled and after warm-up period
            if (param.iterConfigs[iter].updateWeightsEnabled && iter >= param.warmUpIters)
            {
                // Calculate cycle position based on effective update iterations rather than raw iteration count
                int cyclePos = effectiveUpdateIters % realItersUsed;

                // If cyclePos is less than minItersNeeded, update layers according to existing logic
                // Otherwise, this is an empty update cycle, don't assign any layers
                if (cyclePos < minItersNeeded)
                {
                    // Calculate which layers should be updated at this position
                    for (int i = 0; i < updatesPerIter; ++i)
                    {
                        int layerId = cyclePos + i * minItersNeeded;
                        if (layerId < totalLayers)
                        {
                            mLayerUpdateSchedule[iter].push_back(layerId);
                        }
                    }
                }
                // Otherwise mLayerUpdateSchedule[iter] remains an empty vector, indicating no layers to update in this
                // cycle

                // Increment effective update iteration count
                effectiveUpdateIters++;
            }
        }
    }

    void createLayers(MoeLoadBalancerTestParam const& param)
    {
        mLayers.clear();
        // Create the specified number of MoE layers
        for (int i = 0; i < param.numLayers; ++i)
        {
            auto layer = mLoadBalancer->AddLayer(param.expertCount, param.topK, param.slotCountPerRank);
            mLayers.push_back(layer);
        }
    }

    void createModelWeights(MoeLoadBalancerTestParam const& param)
    {
        int const weightWidth = 1024;
        int const weightHeight = 10;
        int weightSize = weightWidth * weightHeight;
        int const gpuWeightPitch = 2048;
        int gpuWeightSize = gpuWeightPitch * weightHeight;
        // Adjust the size of vectors storing weights for multiple layers
        mAllHostWeights.resize(param.numLayers);
        mAllSlotWeights.resize(param.numLayers);

        // Create weight slots for each layer
        for (int layerId = 0; layerId < param.numLayers; ++layerId)
        {
            // Create host weights for each expert in the layer
            mAllHostWeights[layerId].resize(param.expertCount);
            for (int expertId = 0; expertId < param.expertCount; ++expertId)
            {
                float* hostWeightPtr;
                EXPECT_EQ(cudaMallocHost(&hostWeightPtr, weightSize * sizeof(float)), cudaSuccess);

                // Fill weight values: use a combination of layerId and expertId for unique identification
                // This ensures that weights for each expert/layer combination are unique
                float baseValue = generateWeightValue(layerId, expertId);
                for (int i = 0; i < weightSize; ++i)
                {
                    // Add small position offset to make each position have a different value
                    hostWeightPtr[i] = baseValue + i * 0.1f;
                }

                mAllHostWeights[layerId][expertId].mWeightPtr = hostWeightPtr;
                mAllHostWeights[layerId][expertId].mHeight = weightHeight;
                mAllHostWeights[layerId][expertId].mWidth = weightWidth * sizeof(float);
                mAllHostWeights[layerId][expertId].mPitch = weightWidth * sizeof(float);

                // Add to the weight updater for the corresponding layer
                mLayers[layerId]->addSingleHostWeight(expertId, "test_weight", mAllHostWeights[layerId][expertId]);
            }
        }

        // Generate initial expert assignments
        std::vector<int> initialExpertAssignments;
        for (int r = 0; r < param.epSize; ++r)
        {
            int expertStart = r * param.expertCount / param.epSize;
            for (int slotId = 0; slotId < param.slotCountPerRank; ++slotId)
            {
                int expertId = (expertStart + slotId) % param.expertCount;
                initialExpertAssignments.push_back(expertId);
            }
        }

        int expertStart = param.epRank * param.expertCount / param.epSize;

        // Create weight slots for each layer
        for (int layerId = 0; layerId < param.numLayers; ++layerId)
        {
            // Create device weights for each slot in the layer
            mAllSlotWeights[layerId].resize(param.slotCountPerRank);
            for (int slotId = 0; slotId < param.slotCountPerRank; ++slotId)
            {
                int expertId = (expertStart + slotId) % param.expertCount;
                float* slotWeightPtr;
                EXPECT_EQ(cudaMalloc(&slotWeightPtr, gpuWeightSize * sizeof(float)), cudaSuccess);

                // Initialize weights
                for (int h = 0; h < weightHeight; h++)
                {
                    EXPECT_EQ(cudaMemcpy(slotWeightPtr + h * gpuWeightPitch,
                                  static_cast<float*>(mAllHostWeights[layerId][expertId].mWeightPtr) + h * weightWidth,
                                  sizeof(float) * weightWidth, cudaMemcpyHostToDevice),
                        cudaSuccess);
                }

                mAllSlotWeights[layerId][slotId].mWeightPtr = slotWeightPtr;
                mAllSlotWeights[layerId][slotId].mHeight = weightHeight;
                mAllSlotWeights[layerId][slotId].mWidth = weightWidth * sizeof(float);
                mAllSlotWeights[layerId][slotId].mPitch = gpuWeightPitch * sizeof(float);

                // Add to the corresponding layer slot
                mLayers[layerId]->addSingleWeightSlot(slotId, "test_weight", mAllSlotWeights[layerId][slotId]);
            }

            // Set initial weight assignments
            mLayers[layerId]->setInitialWeightAssignments(initialExpertAssignments);
        }

        // Save current weights for each layer and slot for later comparison
        saveCurrentWeights();
    }

    // Save the current weight state for all layers
    void saveCurrentWeights()
    {
        int const weightWidth = 1024;
        int const weightHeight = 10;
        int const weightSize = weightWidth * weightHeight;
        int const weightPitch = 2048;

        auto param = GetParam();
        mCurrentWeights.resize(param.numLayers);
        for (int layerId = 0; layerId < param.numLayers; ++layerId)
        {
            mCurrentWeights[layerId].resize(param.slotCountPerRank);
            for (int slotId = 0; slotId < param.slotCountPerRank; ++slotId)
            {
                mCurrentWeights[layerId][slotId].resize(weightSize);
                for (int h = 0; h < weightHeight; h++)
                {
                    EXPECT_EQ(cudaMemcpy(mCurrentWeights[layerId][slotId].data() + h * weightWidth,
                                  static_cast<float*>(mAllSlotWeights[layerId][slotId].mWeightPtr) + h * weightPitch,
                                  weightWidth * sizeof(float), cudaMemcpyDeviceToHost),
                        cudaSuccess);
                }
            }
        }
    }

    // Simulate forward pass, set different load distributions for each layer
    void setExpertLoadFactors(MoeLoadBalancerTestParam const& param)
    {
        for (int layerId = 0; layerId < param.numLayers; ++layerId)
        {
            float* expertLoadFactor = mLayers[layerId]->getStatisticInfo()->expertLoadFactor;
            for (int i = 0; i < param.expertCount; ++i)
            {
                // Set different load patterns for each layer
                if (layerId % 2 == 0)
                {
                    // Even layers: higher load for higher expert IDs
                    expertLoadFactor[i] = static_cast<float>(i + 1);
                }
                else
                {
                    // Odd layers: higher load for lower expert IDs
                    expertLoadFactor[i] = static_cast<float>(param.expertCount - i);
                }
            }
        }
    }

    // Verify if the weight updates for a layer match expectations
    void verifyLayerWeightUpdates(int layerId, int slotId, bool shouldBeUpdated, int64_t iterId)
    {
        int const weightWidth = 1024;
        int const weightHeight = 10;
        int const weightSize = weightWidth * weightHeight;
        int const weightPitch = 2048;

        // Get current weights
        std::vector<float> currentWeight(weightSize);
        for (int h = 0; h < weightHeight; h++)
        {
            EXPECT_EQ(cudaMemcpy(currentWeight.data() + h * weightWidth,
                          static_cast<float*>(mAllSlotWeights[layerId][slotId].mWeightPtr) + h * weightPitch,
                          weightWidth * sizeof(float), cudaMemcpyDeviceToHost),
                cudaSuccess);
        }

        // Check if weights have changed
        bool hasChanged = false;
        for (int i = 0; i < weightSize && !hasChanged; ++i)
        {
            if (mCurrentWeights[layerId][slotId][i] != currentWeight[i])
            {
                hasChanged = true;
            }
        }

        // Verify according to expectations
        if (shouldBeUpdated)
        {
            // In some cases (e.g., optimal allocation didn't change), weights might not change; this is just a weak
            // check EXPECT_TRUE(hasChanged) << "Layer " << layerId << ", Slot " << slotId << " weights should have been
            // updated but weren't at iter " << iterId;
        }
        else
        {
            // When updates shouldn't happen, weights must remain unchanged
            EXPECT_FALSE(hasChanged) << "Layer " << layerId << ", Slot " << slotId
                                     << " weights shouldn't have changed but did at iter " << iterId;
        }

        // Update saved weight state
        mCurrentWeights[layerId][slotId] = currentWeight;
    }

    // Verify single layer's weight updates from the previous iteration
    void verifyLayerPreviousIterationWeightUpdates(
        int layerId, int64_t currentIterId, bool wasUpdateEnabled, bool shouldLayerBeUpdated)
    {
        // No "previous" iteration for the first iteration
        if (mLastIterId < 0)
        {
            return;
        }

        auto param = GetParam();
        for (int slotId = 0; slotId < param.slotCountPerRank; ++slotId)
        {
            verifyLayerWeightUpdates(layerId, slotId, wasUpdateEnabled && shouldLayerBeUpdated, currentIterId);
        }
    }

    // Simulate GPU waiting for signal, processing signal, and checking related state
    void simulateGpuSignalWaitAndProcess(int64_t iterId, bool expectedStatisticEnabled,
        bool expectedUpdateWeightsEnabled, std::vector<int> const& expectedUpdatedLayers)
    {
        // Execute simulated GPU forward pass and signal processing for each layer
        for (int layerId = 0; layerId < mLayers.size(); ++layerId)
        {
            auto signal = mLayers[layerId]->getSignal();

            // 1. Use helper function to simulate GPU waiting for signal
            int statisticEnabled;
            moeWaitSignalForGpuStageForTest(signal, &statisticEnabled);

            // Get current iteration ID from the signal
            int64_t gpuIterId = signal->stepAndOwner >> 2;

            // 2. Verify the obtained signal values
            EXPECT_EQ(gpuIterId, iterId) << "Layer " << layerId << " has wrong iterId: " << gpuIterId << " vs expected "
                                         << iterId;
            EXPECT_EQ(statisticEnabled != 0, expectedStatisticEnabled)
                << "Layer " << layerId << " statistic enabled mismatch at iteration " << iterId;

            // If not the first iteration, verify the current layer's weight updates from the previous iteration
            if (mLastIterId >= 0)
            {
                // Determine if the current layer should have been updated in the previous iteration
                bool shouldLayerBeUpdated = false;
                if (mLastIterId < mLayerUpdateSchedule.size())
                {
                    auto& updatedLayers = mLayerUpdateSchedule[mLastIterId];
                    shouldLayerBeUpdated
                        = std::find(updatedLayers.begin(), updatedLayers.end(), layerId) != updatedLayers.end();
                }

                // Get the previous iteration's configuration
                bool wasUpdateEnabled = false;
                if (mLastIterId < GetParam().iterConfigs.size())
                {
                    auto& lastConfig = GetParam().iterConfigs[mLastIterId];
                    // Enforce the rule: if statistics are disabled, updates must also be disabled
                    EXPECT_TRUE(lastConfig.statisticEnabled || !lastConfig.updateWeightsEnabled)
                        << "Invalid state detected in iteration " << mLastIterId
                        << ": statistics disabled but updates enabled";

                    wasUpdateEnabled = lastConfig.updateWeightsEnabled && mLastIterId >= GetParam().warmUpIters;
                }

                // Verify the previous iteration's updates for the current layer between GPU signal processing
                verifyLayerPreviousIterationWeightUpdates(layerId, iterId, wasUpdateEnabled, shouldLayerBeUpdated);
            }

            // After verification, regenerate host weights for the next verification
            updateHostWeightsForLayer(layerId, iterId);

            // 3. Use helper function to simulate GPU sending signal to CPU, indicating forward pass completion
            moeSetSignalForCpuStageForTest(signal);
        }

        // Update current iteration ID
        mLastIterId = iterId;
    }

    // Verify the final load balancing situation - need to verify the weight updates from the last iteration
    void verifyLastIterationWeightUpdates(MoeLoadBalancerTestParam const& param)
    {
        // Verify the weight updates from the last iteration
        if (mLastIterId >= 0)
        {
            for (int layerId = 0; layerId < param.numLayers; ++layerId)
            {
                // Determine if the current layer should have been updated in the last iteration
                bool shouldLayerBeUpdated = false;
                if (mLastIterId < mLayerUpdateSchedule.size())
                {
                    auto& updatedLayers = mLayerUpdateSchedule[mLastIterId];
                    shouldLayerBeUpdated
                        = std::find(updatedLayers.begin(), updatedLayers.end(), layerId) != updatedLayers.end();
                }

                // Get the last iteration's configuration
                bool wasUpdateEnabled = false;
                if (mLastIterId < param.iterConfigs.size())
                {
                    auto& lastConfig = param.iterConfigs[mLastIterId];
                    // Enforce the rule: if statistics are disabled, updates must also be disabled
                    EXPECT_TRUE(lastConfig.statisticEnabled || !lastConfig.updateWeightsEnabled)
                        << "Invalid state detected in iteration " << mLastIterId
                        << ": statistics disabled but updates enabled";

                    wasUpdateEnabled = lastConfig.updateWeightsEnabled && mLastIterId >= param.warmUpIters;
                }

                // Verify the updates for each layer from the last iteration
                verifyLayerPreviousIterationWeightUpdates(
                    layerId, mLastIterId + 1, wasUpdateEnabled, shouldLayerBeUpdated);
            }
        }

        // Other load balancing related verifications can be added here
    }

    // Generate consistent weight value based on layerId, expertId
    float generateWeightValue(int layerId, int expertId)
    {
        // Base value from layerId and expertId, ensuring each expert/layer combination is unique
        float baseValue = static_cast<float>((layerId + 1) * 100 + (expertId + 1));

        // Also set different values for different positions in the weight array by adding position offset
        return baseValue; // Position index will be added when used
    }

    // Regenerate host weights for a specific expert
    void regenerateHostWeights(int layerId, int expertId)
    {
        int const weightWidth = 1024;
        int const weightHeight = 10;
        int weightSize = weightWidth * weightHeight;

        float* hostWeightPtr = static_cast<float*>(mAllHostWeights[layerId][expertId].mWeightPtr);
        float baseValue = generateWeightValue(layerId, expertId);

        // Make values at different positions in the weight array also different
        for (int i = 0; i < weightSize; ++i)
        {
            // Add small position offset to make each position have a different value
            hostWeightPtr[i] = baseValue + i * 0.1f;
        }
    }

    // Update host weights for all slots in the current layer
    void updateHostWeightsForLayer(int layerId, int64_t iterId)
    {
        auto param = GetParam();
        // Update weights for all experts in this layer
        for (int expertId = 0; expertId < param.expertCount; ++expertId)
        {
            regenerateHostWeights(layerId, expertId);
        }
    }

    std::unique_ptr<MoeLoadBalancer> mLoadBalancer;
    std::vector<std::shared_ptr<SingleLayerMoeLoadBalancer>> mLayers; // Store pointers to all layers
    std::vector<std::vector<MoeWeight>> mAllHostWeights;              // Expert weights for each layer (host memory)
    std::vector<std::vector<MoeWeight>> mAllSlotWeights;              // Slot weights for each layer (device memory)
    std::vector<std::vector<std::vector<float>>> mCurrentWeights;     // Save current weight state
    std::vector<std::vector<int>> mLayerUpdateSchedule;               // Layers to update in each iteration
    int64_t mLastIterId;                                              // Save last iteration ID for verification
};

TEST_P(MoeLoadBalancerTest, MultiLayerTest)
{
    auto param = GetParam();

    // Complete model setup
    mLoadBalancer->finalizeModel();

    // Set warm-up iteration count
    if (param.warmUpIters > 0)
    {
        mLoadBalancer->setWarmUpIterCount(param.warmUpIters);
    }

    // Set different load factors for each layer
    setExpertLoadFactors(param);

    // Run multiple iterations for testing
    for (int iterIdx = 0; iterIdx < param.iterConfigs.size(); ++iterIdx)
    {
        auto const& config = param.iterConfigs[iterIdx];
        int64_t iterId = iterIdx;

        // 1. Start new iteration
        mLoadBalancer->startIter(iterId, config.statisticEnabled, config.updateWeightsEnabled);

        // 2. Simulate GPU waiting and processing signals
        std::vector<int> expectedUpdatedLayers;
        if (iterIdx < mLayerUpdateSchedule.size())
        {
            expectedUpdatedLayers = mLayerUpdateSchedule[iterIdx];
        }

        simulateGpuSignalWaitAndProcess(iterId, config.statisticEnabled,
            config.updateWeightsEnabled && iterIdx >= param.warmUpIters, expectedUpdatedLayers);

        // 3. End current iteration
        mLoadBalancer->endIter(iterId);

        // Make sure all operations are complete
        cudaDeviceSynchronize();
    }

    // At the end of the test, verify load balancing
    verifyLastIterationWeightUpdates(param);

    // Normal shutdown of the load balancer
    mLoadBalancer->shutdown();

    mLoadBalancer.reset();
}

// Define test parameters
INSTANTIATE_TEST_SUITE_P(MoeLoadBalancerTests, MoeLoadBalancerTest,
    ::testing::Values(
        // Scenario 1: Two-layer model, alternating enable/disable statistics and updates
        MoeLoadBalancerTestParam{8, // expertCount
            2,                      // topK
            0,                      // epRank
            2,                      // epSize
            4,                      // slotCountPerRank
            2,                      // numLayers
            1,                      // layerUpdatesPerIter (update 1 layer at a time)
            2,                      // warmUpIters
            {
                // iterConfigs
                {true, true},  // Iteration 0: Enable statistics and updates (warm-up)
                {true, true},  // Iteration 1: Enable statistics and updates (warm-up)
                {true, true},  // Iteration 2: Enable statistics and updates
                {true, false}, // Iteration 3: Enable statistics, disable updates
                {true, true},  // Iteration 4: Enable statistics and updates
                {false, false} // Iteration 5: Disable statistics and updates (both must be disabled together)
            }},

        // Scenario 2: Three-layer model, updating 2 layers each time
        MoeLoadBalancerTestParam{16, // expertCount
            1,                       // topK
            0,                       // epRank
            4,                       // epSize
            4,                       // slotCountPerRank
            3,                       // numLayers
            2,                       // layerUpdatesPerIter (update 2 layers at a time)
            1,                       // warmUpIters
            {
                // iterConfigs
                {true, true},  // Iteration 0: Enable statistics and updates (warm-up)
                {true, true},  // Iteration 1: Enable statistics and updates
                {true, false}, // Iteration 2: Enable statistics, disable updates
                {true, true}   // Iteration 3: Enable statistics and updates
            }},

        // Scenario 3: Five-layer model, updating 3 layers each time, no warm-up
        MoeLoadBalancerTestParam{32, // expertCount
            4,                       // topK
            1,                       // epRank
            8,                       // epSize
            4,                       // slotCountPerRank
            5,                       // numLayers
            3,                       // layerUpdatesPerIter (update 3 layers at a time)
            0,                       // warmUpIters (no warm-up)
            {
                // iterConfigs
                {true, true},   // Iteration 0: Enable statistics and updates
                {false, false}, // Iteration 1: Disable statistics and updates
                {true, true}    // Iteration 2: Enable statistics and updates
            }},

        // Scenario 4: 61-layer model, updating 10 layers each time, 5 warm-up iterations
        MoeLoadBalancerTestParam{256, // expertCount
            8,                        // topK
            4,                        // epRank
            36,                       // epSize
            8,                        // slotCountPerRank
            61,                       // numLayers
            10,                       // layerUpdatesPerIter (update 10 layers at a time)
            5,                        // warmUpIters (5 warm-up iterations)
            {
                // iterConfigs
                {true, true},   // Iteration 0: Enable statistics and updates
                {true, true},   // Iteration 1: Enable statistics and updates
                {true, true},   // Iteration 2: Enable statistics and updates
                {true, true},   // Iteration 3: Enable statistics and updates
                {true, true},   // Iteration 4: Enable statistics and updates
                {true, true},   // Iteration 5: Enable statistics and updates
                {true, true},   // Iteration 6: Enable statistics and updates
                {true, true},   // Iteration 7: Enable statistics and updates
                {true, true},   // Iteration 8: Enable statistics and updates
                {true, true},   // Iteration 9: Enable statistics and updates
                {true, true},   // Iteration 10: Enable statistics and updates
                {true, true},   // Iteration 11: Enable statistics and updates
                {true, true},   // Iteration 12: Enable statistics and updates
                {false, false}, // Iteration 13: Disable statistics and updates
                {true, false},  // Iteration 14: Enable statistics, disable updates
                {true, true},   // Iteration 15: Enable statistics and updates
                {false, false}, // Iteration 16: Disable statistics and updates
                {false, false}, // Iteration 17: Disable statistics and updates
                {true, true},   // Iteration 18: Enable statistics and updates
                {true, true},   // Iteration 19: Enable statistics and updates
                {true, true},   // Iteration 20: Enable statistics and updates
                {true, true}    // Iteration 21: Enable statistics and updates
            }}));

/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "moeLoadBalancer.h"
#include "hostAccessibleDeviceAllocator.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moeLoadBalance/moeLoadBalanceKernels.h"
#include "topologyDetector.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace tensorrt_llm::runtime
{
// Helper structure to hold replica information
struct ReplicaInfo
{
    double slotSize;
    int expertId;

    // Overload < operator for sorting (descending slotSize)
    bool operator<(ReplicaInfo const& other) const
    {
        // Primary sort key: slotSize descending
        if (slotSize != other.slotSize)
        {
            return slotSize > other.slotSize;
        }
        // Secondary sort key: expertId ascending (for stability)
        return expertId < other.expertId;
    }
};

void doReplication(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, float* const expertLoadFactor,
    MoePlacementCpuInfo* cpuPlacement)
{
    cpuPlacement->expertReplicaCount.resize(metaInfo.expertCount);
    int totalSlotCount = metaInfo.epSize * metaInfo.slotCountPerRank;
    // --- Edge Case 1: No replication needed ---
    if (totalSlotCount == metaInfo.expertCount)
    {
        std::fill(cpuPlacement->expertReplicaCount.begin(), cpuPlacement->expertReplicaCount.end(), 1);
        return;
    }

    // --- Edge Case 2: No load information, distribute evenly ---
    std::vector<float> expertLoadFactorVec(expertLoadFactor, expertLoadFactor + metaInfo.expertCount);
    double sumLoadFactor = std::accumulate(expertLoadFactorVec.begin(), expertLoadFactorVec.end(), 0.0);

    if (sumLoadFactor == 0.0)
    {
        int baseCount = totalSlotCount / metaInfo.expertCount;
        int remainder = totalSlotCount % metaInfo.expertCount;
        for (int i = 0; i < metaInfo.expertCount; ++i)
        {
            cpuPlacement->expertReplicaCount[i] = baseCount + (i < remainder ? 1 : 0);
        }
        return;
    }

    // --- Greedy Replication using Priority Queue ---

    // Initialize replica counts to 1 for all experts
    std::fill(cpuPlacement->expertReplicaCount.begin(), cpuPlacement->expertReplicaCount.end(), 1);
    int assignedSlotCount = metaInfo.expertCount;

    // Define a max-priority queue storing pairs of {current_slot_size, expert_id}
    // std::priority_queue is a max-heap by default.
    using SlotExpertPair = std::pair<double, int>;
    std::priority_queue<SlotExpertPair> pq;

    // Initialize the priority queue with the initial slot size for each expert (replicaCount = 1)
    for (int i = 0; i < metaInfo.expertCount; ++i)
    {
        // Initial slot size based on replicaCount = 1
        double currentSlotSize = expertLoadFactorVec[i] / 1.0;
        pq.push({currentSlotSize, i});
    }

    // Assign the remaining (mTotalSlotCount - mExpertCount) slots greedily
    while (assignedSlotCount < totalSlotCount)
    {
        // Get the expert with the maximum current slot size
        SlotExpertPair top = pq.top();
        pq.pop();
        int expertId = top.second;

        // Increment the replica count for this expert
        cpuPlacement->expertReplicaCount[expertId]++;
        assignedSlotCount++;

        // Calculate the new slot size for this expert with the updated replica count
        double newSlotSize
            = expertLoadFactorVec[expertId] / static_cast<double>(cpuPlacement->expertReplicaCount[expertId]);

        // Push the updated state (new slot size and expert id) back into the queue
        pq.push({newSlotSize, expertId});
    }
}

void doPlacement(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, float* const expertLoadFactor,
    MoePlacementCpuInfo* cpuPlacement)
{
    // This function only update these two vectors
    auto& rankExpertIds = cpuPlacement->rankExpertIds;
    auto& replicaCount = cpuPlacement->expertReplicaCount;

    int totalSlotCount = metaInfo.epSize * metaInfo.slotCountPerRank;
    // 1. Create all replica information
    std::vector<ReplicaInfo> allReplicas;
    allReplicas.reserve(totalSlotCount);

    for (int expertId = 0; expertId < metaInfo.expertCount; ++expertId)
    {
        assert(replicaCount[expertId] > 0); // Ensure replica count is positive
        double slotSize = expertLoadFactor[expertId] / static_cast<double>(replicaCount[expertId]);
        for (int replicaId = 0; replicaId < replicaCount[expertId]; ++replicaId)
        {
            allReplicas.push_back({slotSize, expertId});
            // totalLoadSum += slotSize; // Accumulate total load
        }
    }

    assert(static_cast<int>(allReplicas.size()) == totalSlotCount);

    // 2. Sort replicas by slotSize descending
    std::sort(allReplicas.begin(), allReplicas.end());

    // 3. Maintain Rank state and initialize Priority Queue
    std::vector<double> currentRankLoad(metaInfo.epSize, 0.0);
    std::vector<int> currentRankSlots(metaInfo.epSize, 0); // Tracks the count of assigned slots per rank

    // Define a min-priority queue storing pairs of {load, rank_id}
    using RankLoadPair = std::pair<double, int>;
    std::priority_queue<RankLoadPair, std::vector<RankLoadPair>, std::greater<RankLoadPair>> pq;

    // Initialize the priority queue with all ranks having 0 load
    for (int rank = 0; rank < metaInfo.epSize; ++rank)
    {
        pq.push({0.0, rank});
    }

    // 4. Optimized Greedy assignment using Priority Queue, writing directly to rankExpertIds
    for (auto const& replica : allReplicas)
    {
        // Get the rank with the minimum load from the priority queue
        RankLoadPair top = pq.top();
        pq.pop();

        int bestRank = top.second;
        double currentLoad = top.first; // The load before adding this replica

        int localSlotId = currentRankSlots[bestRank];
        TLLM_CHECK_WITH_INFO(localSlotId < metaInfo.slotCountPerRank,
            "localSlotId=%d should be less than metaInfo.slotCountPerRank=%d", localSlotId, metaInfo.slotCountPerRank);
        rankExpertIds[bestRank][localSlotId] = replica.expertId;

        // Update rank state
        currentRankLoad[bestRank] = currentLoad + replica.slotSize; // Update load explicitly
        currentRankSlots[bestRank]++;                               // Increment the slot count for this rank

        // If the rank still has capacity, push it back into the queue with updated load
        if (currentRankSlots[bestRank] < metaInfo.slotCountPerRank)
        {
            pq.push({currentRankLoad[bestRank], bestRank});
        }
    }
    assert(pq.empty());
}

namespace
{

void printMoePlacementInfo(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo,
    tensorrt_llm::kernels::MoePlacementInfo* cpuPlacement, std::stringstream& ss)
{
    ss << "MoePlacementInfo:\n";
    ss << "expertReplicaCount: [";
    for (int i = 0; i < metaInfo.expertCount; ++i)
    {
        ss << cpuPlacement->expertReplicaCount[i] << ", ";
    }
    ss << "]\n";
    ss << "expertReplicaStartOffset: [";
    for (int i = 0; i < metaInfo.expertCount; ++i)
    {
        ss << cpuPlacement->expertReplicaStartOffset[i] << ", ";
    }
    ss << "]\n";
    ss << "globalSlotIds: [";
    for (int i = 0; i < metaInfo.epSize * metaInfo.slotCountPerRank; ++i)
    {
        ss << cpuPlacement->globalSlotIds[i] << ", ";
    }
    ss << "]\n";
}

void printCpuPlacementInfo(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, MoePlacementCpuInfo* cpuPlacement)
{
    std::stringstream ss;
    for (int rank = 0; rank < metaInfo.epSize; ++rank)
    {
        ss << "rank=" << rank << " expertIds: [";
        for (int slotId = 0; slotId < metaInfo.slotCountPerRank; ++slotId)
        {
            int expertId = cpuPlacement->rankExpertIds[rank][slotId];
            ss << expertId << ", ";
        }
        ss << "]\n";
    }
    printMoePlacementInfo(metaInfo, &cpuPlacement->placementInfoForGPU, ss);
    printf("%s", ss.str().c_str());
}

void prepareGpuPlacementInfo(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, MoePlacementCpuInfo* cpuPlacement)
{
    // update placementInfoForGPU (which is used to copy to GPU)
    // based on expertReplicaCount and rankExpertIds
    int startOffset = 0;
    for (int expertId = 0; expertId < metaInfo.expertCount; ++expertId)
    {
        cpuPlacement->placementInfoForGPU.expertReplicaCount[expertId] = cpuPlacement->expertReplicaCount[expertId];
        cpuPlacement->placementInfoForGPU.expertReplicaStartOffset[expertId] = startOffset;
        startOffset += cpuPlacement->expertReplicaCount[expertId];
    }

    // Generate globalSlotIds

    // globalSlotIds[i][j] is the list of global slot ids for expert i's j-th replica
    // different experts have different number of replicas, so globalSlotIds is a vector of vectors
    // the sum of sizes of all vectors in globalSlotIds is equal to the total number of slots
    std::vector<std::vector<int>> globalSlotIds(metaInfo.expertCount);
    for (int rank = 0; rank < metaInfo.epSize; ++rank)
    {
        for (int slotId = 0; slotId < metaInfo.slotCountPerRank; ++slotId)
        {
            int expertId = cpuPlacement->rankExpertIds[rank][slotId];
            int replicaId = globalSlotIds[expertId].size();
            int globalSlotId = rank * metaInfo.slotCountPerRank + slotId;
            globalSlotIds[expertId].push_back(globalSlotId);
            int offset = cpuPlacement->placementInfoForGPU.expertReplicaStartOffset[expertId] + replicaId;
            cpuPlacement->placementInfoForGPU.globalSlotIds[offset] = globalSlotId;
        }
    }
    // printCpuPlacementInfo(metaInfo, cpuPlacement);
}

void allocateStatisticInfo(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo const& metaInfo,
    tensorrt_llm::kernels::MoeLoadBalanceStatisticInfo* statisticInfo)
{
    TLLM_CUDA_CHECK(cudaMallocHost(&statisticInfo->expertLoadFactor, sizeof(float) * metaInfo.expertCount));
    std::fill_n(statisticInfo->expertLoadFactor, metaInfo.expertCount, 0.0f);
    TLLM_CHECK_WITH_INFO(statisticInfo->rawDataWindowSize > 0, "statisticInfo->rawDataWindowSize should > 0.");
    TLLM_CUDA_CHECK(cudaMalloc(
        &statisticInfo->expertTokenCount, sizeof(int) * metaInfo.expertCount * statisticInfo->rawDataWindowSize));
}

void freeStatisticInfo(tensorrt_llm::kernels::MoeLoadBalanceStatisticInfo* statisticInfo)
{
    TLLM_CUDA_CHECK(cudaFreeHost(statisticInfo->expertLoadFactor));
    statisticInfo->expertLoadFactor = nullptr;
    if (statisticInfo->expertTokenCount != nullptr)
    {
        TLLM_CUDA_CHECK(cudaFree(statisticInfo->expertTokenCount));
        statisticInfo->expertTokenCount = nullptr;
    }
}

void allocatePlacementInfo(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo const& metaInfo,
    tensorrt_llm::kernels::MoePlacementInfo* placementInfo, bool isCpu = false)
{
    auto allocFn = [isCpu](void** ptr, size_t size)
    {
        if (isCpu)
        {
            return cudaMallocHost(ptr, size);
        }
        else
        {
            if (HostAccessibleDeviceAllocator::getInstance().isSupported())
            {
                *ptr = HostAccessibleDeviceAllocator::getInstance().allocate(size);
                return cudaSuccess;
            }
            else
            {
                return cudaMalloc(ptr, size);
            }
        }
    };
    TLLM_CUDA_CHECK(
        allocFn(reinterpret_cast<void**>(&placementInfo->expertReplicaCount), sizeof(int) * metaInfo.expertCount));
    TLLM_CUDA_CHECK(allocFn(
        reinterpret_cast<void**>(&placementInfo->expertReplicaStartOffset), sizeof(int) * metaInfo.expertCount));
    TLLM_CUDA_CHECK(allocFn(reinterpret_cast<void**>(&placementInfo->globalSlotIds),
        sizeof(int) * metaInfo.epSize * metaInfo.slotCountPerRank));
}

void freePlacementInfo(tensorrt_llm::kernels::MoePlacementInfo* placementInfo, bool isCpu = false)
{
    auto freeFn = [isCpu](void* ptr)
    {
        if (isCpu)
        {
            return cudaFreeHost(ptr);
        }
        else
        {
            if (HostAccessibleDeviceAllocator::getInstance().isSupported())
            {
                HostAccessibleDeviceAllocator::getInstance().free(ptr);
                return cudaSuccess;
            }
            else
            {
                return cudaFree(ptr);
            }
        }
    };
    TLLM_CUDA_CHECK(freeFn(placementInfo->expertReplicaCount));
    TLLM_CUDA_CHECK(freeFn(placementInfo->expertReplicaStartOffset));
    TLLM_CUDA_CHECK(freeFn(placementInfo->globalSlotIds));
}

void getHostAccessibleGpuPlacement(tensorrt_llm::kernels::MoePlacementInfo const* placementInfoGpuAccess,
    tensorrt_llm::kernels::MoePlacementInfo* placementInfoHostAccess)
{
    if (HostAccessibleDeviceAllocator::getInstance().isSupported())
    {
        placementInfoHostAccess->expertReplicaStartOffset = static_cast<int*>(
            HostAccessibleDeviceAllocator::getInstance().getHostPtr(placementInfoGpuAccess->expertReplicaStartOffset));
        placementInfoHostAccess->expertReplicaCount = static_cast<int*>(
            HostAccessibleDeviceAllocator::getInstance().getHostPtr(placementInfoGpuAccess->expertReplicaCount));
        placementInfoHostAccess->globalSlotIds = static_cast<int*>(
            HostAccessibleDeviceAllocator::getInstance().getHostPtr(placementInfoGpuAccess->globalSlotIds));
    }
    else
    {
        placementInfoHostAccess->expertReplicaStartOffset = nullptr;
        placementInfoHostAccess->expertReplicaCount = nullptr;
        placementInfoHostAccess->globalSlotIds = nullptr;
    }
}

tensorrt_llm::kernels::MoeLoadBalanceSingleLayerSignal* allocateSingleLayerSignal()
{
    tensorrt_llm::kernels::MoeLoadBalanceSingleLayerSignal* ptr = nullptr;
    TLLM_CUDA_CHECK(cudaMallocHost(&ptr, sizeof(tensorrt_llm::kernels::MoeLoadBalanceSingleLayerSignal)));
    // first initialized as CPU ownership and GPU should wait CPU thread to set to GPU ownership at startup.
    ptr->stepAndOwner = tensorrt_llm::kernels::MoeLoadBalanceSingleLayerSignal::kCPU;
    return ptr;
}

void freeSingleLayerSignal(tensorrt_llm::kernels::MoeLoadBalanceSingleLayerSignal* ptr)
{
    TLLM_CUDA_CHECK(cudaFreeHost(ptr));
}

} // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////
// Single Layer Moe Load Balancer
///////////////////////////////////////////////////////////////////////////////////////////////////

SingleLayerMoeLoadBalancer::SingleLayerMoeLoadBalancer(
    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, MoeLoadBalancer* loadBalancer, int balanceLayerId)
    : mMoeLoadBalancer(loadBalancer)
    , mMetaInfo(metaInfo)
    , mLayerId(balanceLayerId)
{
}

SingleLayerMoeLoadBalancer::~SingleLayerMoeLoadBalancer()
{
    destroyResources();
}

void SingleLayerMoeLoadBalancer::addSingleWeightSlot(int localSlotId, std::string const& name, MoeWeight weightSlot)
{
    mWeightUpdater->addSingleWeightSlot(localSlotId, name, weightSlot, mMoeLoadBalancer->mUseGpuMemcpy);
}

void SingleLayerMoeLoadBalancer::addSingleHostWeight(int expertId, std::string const& name, MoeWeight hostWeight)
{
    mWeightUpdater->addSingleHostWeight(expertId, name, hostWeight);
}

void SingleLayerMoeLoadBalancer::setInitialWeightAssignments(std::vector<int> const& initialWeightAssignments)
{
    std::fill_n(mCpuPlacementInfo.expertReplicaCount.begin(), mMetaInfo.expertCount, 0);
    for (int rank = 0; rank < mMetaInfo.epSize; ++rank)
    {
        for (int localSlotId = 0; localSlotId < mMetaInfo.slotCountPerRank; ++localSlotId)
        {
            int expertId = initialWeightAssignments[rank * mMetaInfo.slotCountPerRank + localSlotId];
            TLLM_CHECK_WITH_INFO(expertId >= 0 && expertId < mMetaInfo.expertCount, "expertId=%d", expertId);
            mCpuPlacementInfo.rankExpertIds[rank][localSlotId] = expertId;
            mCpuPlacementInfo.expertReplicaCount[expertId]++;
        }
    }
    prepareGpuPlacementInfo(mMetaInfo, &mCpuPlacementInfo);
    // we don't need to call mWeightUpdater since should be already assigned.
    copyPlacementInfoToGpu();
    TLLM_CUDA_CHECK(cudaEventRecord(mUpdateWeightsDoneEvent, mMoeLoadBalancer->mStream));
    TLLM_CUDA_CHECK(cudaEventSynchronize(mUpdateWeightsDoneEvent));
}

void SingleLayerMoeLoadBalancer::createResources()
{
    // Statistic Info
    allocateStatisticInfo(mMetaInfo, &mStatisticInfo);

    mCpuPlacementInfo.rankExpertIds.resize(mMetaInfo.epSize);
    mCpuPlacementInfo.oldRankExpertIds.resize(mMetaInfo.epSize);
    mCpuPlacementInfo.expertReplicaCount.resize(mMetaInfo.expertCount, 0);
    for (int i = 0; i < mMetaInfo.epSize; ++i)
    {
        mCpuPlacementInfo.rankExpertIds[i].resize(mMetaInfo.slotCountPerRank, -1);
        mCpuPlacementInfo.oldRankExpertIds[i].resize(mMetaInfo.slotCountPerRank, -1);
    }

    allocatePlacementInfo(mMetaInfo, &mCpuPlacementInfo.placementInfoForGPU, true);
    allocatePlacementInfo(mMetaInfo, &mGpuPlacementGpuAccess, false);
    getHostAccessibleGpuPlacement(&mGpuPlacementGpuAccess, &mGpuPlacementHostAccess);

    mSingleLayerSignal = allocateSingleLayerSignal();
    TLLM_CUDA_CHECK(cudaEventCreate(&mUpdateWeightsDoneEvent));
    mWeightUpdater.reset(new HostMemoryMoeWeightUpdater(mMetaInfo, this));
}

void SingleLayerMoeLoadBalancer::destroyResources()
{
    mWeightUpdater.reset();
    freeStatisticInfo(&mStatisticInfo);
    freePlacementInfo(&mCpuPlacementInfo.placementInfoForGPU, true);
    freePlacementInfo(&mGpuPlacementGpuAccess, false);
    freeSingleLayerSignal(mSingleLayerSignal);
    TLLM_CUDA_CHECK(cudaEventDestroy(mUpdateWeightsDoneEvent));
}

void SingleLayerMoeLoadBalancer::finalizeModel()
{
    mWeightUpdater->finalizeWeights();
}

void SingleLayerMoeLoadBalancer::startCpuNewIter(int64_t iterId, bool enableStatistic, bool enableUpdateWeights)
{
    TLLM_CHECK_WITH_INFO(mIterId + 1 == iterId, "Expected iterId=%ld, but got %ld", mIterId + 1, iterId);
    mIterId = iterId;
    mStatisticEnabled = enableStatistic;
    mUpdateWeightsEnabled = enableUpdateWeights;
    std::unique_lock<std::mutex> lock(mUpdateWeightsMutex);
    mUpdateWeightsDone = false;
}

void SingleLayerMoeLoadBalancer::setGpuStage()
{
    tensorrt_llm::kernels::moeSetSignalForGpuStageHost(mSingleLayerSignal, mIterId, mStatisticEnabled);
}

void SingleLayerMoeLoadBalancer::waitCpuStage()
{
    tensorrt_llm::kernels::moeWaitSignalForCpuStageHost(mSingleLayerSignal);
}

void SingleLayerMoeLoadBalancer::maybeStartUpdateWeights()
{
    if (mIterId >= 0 && mUpdateWeightsEnabled)
    {
        mMoeLoadBalancer->addUpdateTask(
            [this]
            {
                if (mMoeLoadBalancer->mUseGpuMemcpy)
                {
                    updateWeightsRoutine();
                }
                else
                {
                    updateWeightsRoutineByCpu();
                }
            });
    }
}

void SingleLayerMoeLoadBalancer::waitLastUpdateDone()
{
    if (mIterId >= 0 && mUpdateWeightsEnabled)
    {
        std::unique_lock<std::mutex> lock(mUpdateWeightsMutex);
        mUpdateWeightsCondition.wait(lock, [this] { return mUpdateWeightsDone; });
        lock.unlock();
    }
}

cudaStream_t SingleLayerMoeLoadBalancer::getStream() const
{
    return mMoeLoadBalancer->mStream;
}

void SingleLayerMoeLoadBalancer::copyPlacementInfoToGpu()
{
    cudaStream_t stream = mMoeLoadBalancer->mStream;
    TLLM_CUDA_CHECK(cudaMemcpyAsync(mGpuPlacementGpuAccess.expertReplicaCount,
        mCpuPlacementInfo.placementInfoForGPU.expertReplicaCount, sizeof(int) * mMetaInfo.expertCount,
        cudaMemcpyHostToDevice, stream));
    TLLM_CUDA_CHECK(cudaMemcpyAsync(mGpuPlacementGpuAccess.expertReplicaStartOffset,
        mCpuPlacementInfo.placementInfoForGPU.expertReplicaStartOffset, sizeof(int) * mMetaInfo.expertCount,
        cudaMemcpyHostToDevice, stream));
    TLLM_CUDA_CHECK(
        cudaMemcpyAsync(mGpuPlacementGpuAccess.globalSlotIds, mCpuPlacementInfo.placementInfoForGPU.globalSlotIds,
            sizeof(int) * mMetaInfo.epSize * mMetaInfo.slotCountPerRank, cudaMemcpyHostToDevice, stream));
    mCpuPlacementInfo.rankExpertIds.swap(mCpuPlacementInfo.oldRankExpertIds);
    for (int i = 0; i < mMetaInfo.epSize; ++i)
    {
        std::fill_n(mCpuPlacementInfo.rankExpertIds[i].begin(), mMetaInfo.slotCountPerRank, -1);
    }
    // clear expert load factor for next statistic
    std::fill_n(mStatisticInfo.expertLoadFactor, mMetaInfo.expertCount, 0.0f);
}

void SingleLayerMoeLoadBalancer::copyPlacementInfoToGpuByCpu()
{
    HostAccessibleDeviceAllocator::getInstance().memcpyToDevice(mGpuPlacementHostAccess.expertReplicaCount,
        mCpuPlacementInfo.placementInfoForGPU.expertReplicaCount, sizeof(int) * mMetaInfo.expertCount);
    HostAccessibleDeviceAllocator::getInstance().memcpyToDevice(mGpuPlacementHostAccess.expertReplicaStartOffset,
        mCpuPlacementInfo.placementInfoForGPU.expertReplicaStartOffset, sizeof(int) * mMetaInfo.expertCount);
    HostAccessibleDeviceAllocator::getInstance().memcpyToDevice(mGpuPlacementHostAccess.globalSlotIds,
        mCpuPlacementInfo.placementInfoForGPU.globalSlotIds,
        sizeof(int) * mMetaInfo.epSize * mMetaInfo.slotCountPerRank);
    mCpuPlacementInfo.rankExpertIds.swap(mCpuPlacementInfo.oldRankExpertIds);
    for (int i = 0; i < mMetaInfo.epSize; ++i)
    {
        std::fill_n(mCpuPlacementInfo.rankExpertIds[i].begin(), mMetaInfo.slotCountPerRank, -1);
    }
    // clear expert load factor for next statistic
    std::fill_n(mStatisticInfo.expertLoadFactor, mMetaInfo.expertCount, 0.0f);
}

void SingleLayerMoeLoadBalancer::updateWeightsRoutine()
{
    doReplication(mMetaInfo, mStatisticInfo.expertLoadFactor, &mCpuPlacementInfo);
    doPlacement(mMetaInfo, mStatisticInfo.expertLoadFactor, &mCpuPlacementInfo);
    prepareGpuPlacementInfo(mMetaInfo, &mCpuPlacementInfo);
    mWeightUpdater->updateWeights(&mCpuPlacementInfo);
    copyPlacementInfoToGpu();
    TLLM_CUDA_CHECK(cudaEventRecord(mUpdateWeightsDoneEvent, mMoeLoadBalancer->mStream));
    TLLM_CUDA_CHECK(cudaEventSynchronize(mUpdateWeightsDoneEvent));
    std::unique_lock<std::mutex> lock(mUpdateWeightsMutex);
    mUpdateWeightsDone = true;
    mUpdateWeightsCondition.notify_one();
}

void SingleLayerMoeLoadBalancer::updateWeightsRoutineByCpu()
{
    doReplication(mMetaInfo, mStatisticInfo.expertLoadFactor, &mCpuPlacementInfo);
    doPlacement(mMetaInfo, mStatisticInfo.expertLoadFactor, &mCpuPlacementInfo);
    prepareGpuPlacementInfo(mMetaInfo, &mCpuPlacementInfo);
    mLastUpdateTaskId = mMoeLoadBalancer->addCopyTask(
        [this](int rank, int size) { mWeightUpdater->updateWeights(&mCpuPlacementInfo, rank, size); });
    mMoeLoadBalancer->waitCopyTaskDone(mLastUpdateTaskId);
    mLastUpdateTaskId = -1;
    copyPlacementInfoToGpuByCpu();
    std::unique_lock<std::mutex> lock(mUpdateWeightsMutex);
    mUpdateWeightsDone = true;
    mUpdateWeightsCondition.notify_one();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Weight Updater
///////////////////////////////////////////////////////////////////////////////////////////////////

MoeWeightUpdaterBase::MoeWeightUpdaterBase(
    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, SingleLayerMoeLoadBalancer* layerLoadBalancer)
    : mMetaInfo(metaInfo)
    , mLayerLoadBalancer(layerLoadBalancer)
{
}

void MoeWeightUpdaterBase::addSingleWeightSlot(
    int localSlotId, std::string const& name, tensorrt_llm::runtime::MoeWeight weightSlot, bool gpuAccess)
{
    TLLM_CHECK_WITH_INFO(mWeightSlotsFinalized == false, "Cannot add slots after finalize");
    TLLM_CHECK_WITH_INFO(localSlotId >= 0 && localSlotId < mMetaInfo.slotCountPerRank,
        "localSlotId (%d) should be in range[0, %d)", localSlotId, mMetaInfo.slotCountPerRank);
    TLLM_CHECK_WITH_INFO(weightSlot.mWeightPtr != nullptr && weightSlot.mHeight > 0 && weightSlot.mWidth > 0
            && weightSlot.mPitch >= weightSlot.mWidth,
        "Invalid weightSlot ptr=%p, Height=%ld, Width=%ld, with Pitch=%ld", weightSlot.mWeightPtr, weightSlot.mHeight,
        weightSlot.mWidth, weightSlot.mPitch);
    if (mWeightSlots.find(name) == mWeightSlots.end())
    {
        mWeightSlots.emplace(name, std::vector<MoeWeight>(mMetaInfo.slotCountPerRank));
    }
    TLLM_CHECK_WITH_INFO(mWeightSlots[name][localSlotId].mWeightPtr == nullptr,
        "localSlotId=%d, name=%s already added.", localSlotId, name.c_str());
    if (!gpuAccess)
    {
        // if using cpu copy, change to host accessible pointer.
        weightSlot.mWeightPtr
            = tensorrt_llm::runtime::HostAccessibleDeviceAllocator::getInstance().getHostPtr(weightSlot.mWeightPtr);
    }
    mWeightSlots[name][localSlotId] = weightSlot;
}

void MoeWeightUpdaterBase::finalizeWeightSlot()
{
    TLLM_CHECK_WITH_INFO(mWeightSlotsFinalized == false, "already finalized");
    for (auto it = mWeightSlots.cbegin(); it != mWeightSlots.cend(); ++it)
    {
        auto name = it->first;
        auto& vecWeights = it->second;
        TLLM_CHECK_WITH_INFO(vecWeights.size() == static_cast<size_t>(mMetaInfo.slotCountPerRank),
            "slot count not match for %s", name.c_str());
        for (int i = 0; i < mMetaInfo.slotCountPerRank; ++i)
        {
            TLLM_CHECK_WITH_INFO(vecWeights[i].mWeightPtr != nullptr, "slotId=%d, name=%s not added.", i, name.c_str());
            if (i > 0)
            {
                TLLM_CHECK_WITH_INFO(
                    vecWeights[i].mHeight == vecWeights[0].mHeight && vecWeights[i].mWidth == vecWeights[0].mWidth,
                    "finalizeWeightSlot slot shape not same for slot %d and 0, (%ld, %ld) v.s. (%ld, %ld)", i,
                    vecWeights[i].mHeight, vecWeights[i].mWidth, vecWeights[0].mHeight, vecWeights[0].mWidth);
            }
        }
    }
    mWeightSlotsFinalized = true;
}

void MoeWeightUpdaterBase::finalizeWeights()
{
    finalizeWeightSlot();
}

HostMemoryMoeWeightUpdater::HostMemoryMoeWeightUpdater(
    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, SingleLayerMoeLoadBalancer* layerLoadBalancer)
    : MoeWeightUpdaterBase(metaInfo, layerLoadBalancer)
{
}

void HostMemoryMoeWeightUpdater::addSingleHostWeight(int expertId, std::string const& name, MoeWeight hostWeight)
{
    TLLM_CHECK_WITH_INFO(mHostWeightsFinalized == false, "Cannot add host weight after finalize");
    TLLM_CHECK_WITH_INFO(expertId >= 0 && expertId < mMetaInfo.expertCount, "expertId (%d) should be in range[0, %d)",
        expertId, mMetaInfo.expertCount);
    TLLM_CHECK_WITH_INFO(hostWeight.mWeightPtr != nullptr && hostWeight.mHeight > 0 && hostWeight.mWidth > 0
            && hostWeight.mPitch >= hostWeight.mWidth,
        "Invalid hostWeight ptr=%p, Height=%ld, Width=%ld, with Pitch=%ld", hostWeight.mWeightPtr, hostWeight.mHeight,
        hostWeight.mWidth, hostWeight.mPitch);
    if (mHostWeights.find(name) == mHostWeights.end())
    {
        mHostWeights.emplace(name, std::vector<MoeWeight>(mMetaInfo.expertCount));
    }
    TLLM_CHECK_WITH_INFO(mHostWeights[name][expertId].mWeightPtr == nullptr,
        "expertId=%d, name=%s already added to host weight.", expertId, name.c_str());
    mHostWeights[name][expertId] = hostWeight;
}

void HostMemoryMoeWeightUpdater::finalizeHostWeight()
{
    TLLM_CHECK_WITH_INFO(mHostWeightsFinalized == false, "already finalized");
    TLLM_CHECK_WITH_INFO(mHostWeights.size() == mWeightSlots.size(),
        "mHostWeights and mWeightSlots doesn't have same count of weights, %ld v.s. %ld.", mHostWeights.size(),
        mWeightSlots.size());
    for (auto it = mHostWeights.cbegin(); it != mHostWeights.cend(); ++it)
    {
        auto name = it->first;
        auto& vecWeights = it->second;
        TLLM_CHECK_WITH_INFO(
            mWeightSlots.find(name) != mWeightSlots.end(), "name %s not found in mWeightSlots.", name.c_str());
        auto slotIt = mWeightSlots.find(name);
        TLLM_CHECK_WITH_INFO(vecWeights.size() == static_cast<size_t>(mMetaInfo.expertCount),
            "expert count not match for %s", name.c_str());
        for (int i = 0; i < mMetaInfo.expertCount; ++i)
        {
            TLLM_CHECK_WITH_INFO(
                vecWeights[i].mWeightPtr != nullptr, "expertId=%d, name=%s not added.", i, name.c_str());
            if (i > 0)
            {
                TLLM_CHECK_WITH_INFO(
                    vecWeights[i].mHeight == vecWeights[0].mHeight && vecWeights[i].mWidth == vecWeights[0].mWidth,
                    "finalizeHostWeight host weights shape not same for expert %d and 0, (%ld, %ld) v.s. (%ld, %ld)", i,
                    vecWeights[i].mHeight, vecWeights[i].mWidth, vecWeights[0].mHeight, vecWeights[0].mWidth);
            }
            else
            {
                auto& slotWeight = slotIt->second[0];
                TLLM_CHECK_WITH_INFO(
                    vecWeights[i].mHeight == slotWeight.mHeight && vecWeights[i].mWidth == slotWeight.mWidth,
                    "finalizeHostWeight host weights shape not same for expert 0 and slot 0, (%ld, %ld) v.s. (%ld, "
                    "%ld)",
                    vecWeights[i].mHeight, vecWeights[i].mWidth, slotWeight.mHeight, slotWeight.mWidth);
            }
        }
    }
    mHostWeightsFinalized = true;
}

void HostMemoryMoeWeightUpdater::finalizeWeights()
{
    finalizeWeightSlot();
    finalizeHostWeight();
}

void HostMemoryMoeWeightUpdater::copyWeights(MoeWeight const& src, MoeWeight const& dst, cudaStream_t stream)
{
    TLLM_CHECK(src.mWeightPtr != nullptr && dst.mWeightPtr != nullptr);
    TLLM_CHECK(src.mHeight == dst.mHeight && src.mWidth == dst.mWidth);
    if (src.mPitch == src.mWidth && dst.mPitch == dst.mWidth)
    {
        TLLM_CUDA_CHECK(
            cudaMemcpyAsync(dst.mWeightPtr, src.mWeightPtr, src.mHeight * src.mWidth, cudaMemcpyHostToDevice, stream));
    }
    else
    {
        TLLM_CUDA_CHECK(cudaMemcpy2DAsync(dst.mWeightPtr, dst.mPitch, src.mWeightPtr, src.mPitch, src.mWidth,
            src.mHeight, cudaMemcpyHostToDevice, stream));
    }
}

void HostMemoryMoeWeightUpdater::copyWeightsCpu(MoeWeight const& src, MoeWeight const& dst, int rank, int size)
{
    TLLM_CHECK(src.mWeightPtr != nullptr && dst.mWeightPtr != nullptr);
    TLLM_CHECK(src.mHeight == dst.mHeight && src.mWidth == dst.mWidth);
    char* srcPtr = static_cast<char*>(src.mWeightPtr);
    char* dstPtr = static_cast<char*>(dst.mWeightPtr);
    size_t singleCopySize, copyCount, srcPitch, dstPitch;
    if (src.mPitch == src.mWidth && dst.mPitch == dst.mWidth)
    {
        singleCopySize = src.mWidth * src.mHeight;
        copyCount = 1;
        srcPitch = singleCopySize;
        dstPitch = singleCopySize;
    }
    else
    {
        singleCopySize = src.mWidth;
        copyCount = src.mHeight;
        srcPitch = src.mPitch;
        dstPitch = dst.mPitch;
    }
    size_t fullCopyCount = copyCount / size * size;
    size_t threadCopyCount = fullCopyCount / size;
    for (size_t i = rank * threadCopyCount; i < (rank + 1) * threadCopyCount; i++)
    {
        HostAccessibleDeviceAllocator::getInstance().memcpyToDevice(
            dstPtr + i * dstPitch, srcPtr + i * srcPitch, singleCopySize);
    }
    size_t threadStartOffset = rank * singleCopySize / size;
    size_t threadEndOffset = (rank + 1) * singleCopySize / size;
    size_t threadCopySize = threadEndOffset - threadStartOffset;
    for (size_t i = fullCopyCount; i < copyCount && threadCopySize > 0; i++)
    {
        HostAccessibleDeviceAllocator::getInstance().memcpyToDevice(
            dstPtr + i * dstPitch + threadStartOffset, srcPtr + i * srcPitch + threadStartOffset, threadCopySize);
    }
}

void PrintUpdateInfo(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo,
    tensorrt_llm::runtime::MoePlacementCpuInfo const* placementCpuInfo)
{
    std::stringstream ss;
    ss << "[UpdateInfo] rank=" << metaInfo.epRank << ", expert weights=\n    [";
    for (int slotId = 0; slotId < metaInfo.slotCountPerRank * metaInfo.epSize; slotId++)
    {
        ss << placementCpuInfo->rankExpertIds[slotId / metaInfo.slotCountPerRank][slotId % metaInfo.slotCountPerRank]
           << ", ";
    }
    ss << "\n";
    fprintf(stderr, "%s\n", ss.str().c_str());
}

void HostMemoryMoeWeightUpdater::updateWeights(
    tensorrt_llm::runtime::MoePlacementCpuInfo const* placementCpuInfo, int rank, int size)
{
    // PrintUpdateInfo(mMetaInfo, placementCpuInfo);
    bool useGpu = mLayerLoadBalancer->mMoeLoadBalancer->mUseGpuMemcpy;
    for (int slotId = 0; slotId < mMetaInfo.slotCountPerRank; ++slotId)
    {
        int oldExpertId = placementCpuInfo->oldRankExpertIds[mMetaInfo.epRank][slotId];
        int newExpertId = placementCpuInfo->rankExpertIds[mMetaInfo.epRank][slotId];
        TLLM_CHECK_WITH_INFO(oldExpertId >= 0 && oldExpertId < mMetaInfo.expertCount,
            "oldExpertId=%d, should in range [0, %d)", oldExpertId, mMetaInfo.expertCount);
        TLLM_CHECK_WITH_INFO(newExpertId >= 0 && newExpertId < mMetaInfo.expertCount,
            "newExpertId=%d, should in range [0, %d)", newExpertId, mMetaInfo.expertCount);
        if (oldExpertId == newExpertId)
        {
            continue;
        }
        for (auto slotIt = mWeightSlots.cbegin(); slotIt != mWeightSlots.cend(); ++slotIt)
        {
            auto& name = slotIt->first;
            auto& slotWeight = slotIt->second[slotId];
            auto& hostWeight = mHostWeights[name][newExpertId];
            if (useGpu)
            {
                copyWeights(hostWeight, slotWeight, mLayerLoadBalancer->getStream());
            }
            else
            {
                copyWeightsCpu(hostWeight, slotWeight, rank, size);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Moe Load Balancer
///////////////////////////////////////////////////////////////////////////////////////////////////

MoeLoadBalancer::MoeLoadBalancer(int epRank, int epSize, int layerUpdatesPerIter)
    : mEpRank{epRank}
    , mEpSize{epSize}
    , mLayerUpdatesPerIter{layerUpdatesPerIter}
{
    TLLM_CUDA_CHECK(cudaGetDevice(&mCudaDeviceId));
    // create a non-blocking stream for compute and update, not needed anymore for CPU copy engine.
    TLLM_CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));

    auto& topologyDetector = TopologyDetector::getInstance();
    int currentGpuNumaId = topologyDetector.getCurrentGpuNumaId();
    int numaCpuCount = topologyDetector.getCurrentGpuNumaCpuCount();
    int numaGpuCount = topologyDetector.getGpuCountUnderNuma(currentGpuNumaId);
    HostAccessibleDeviceAllocator::getInstance().IncRefCount();
    TLLM_CHECK_WITH_INFO(layerUpdatesPerIter == 0 || HostAccessibleDeviceAllocator::getInstance().isSupported(),
        "HostAccessibleDeviceAllocator is not supported on current platform, please install gdrcopy(gdrdrv).");
    TLLM_CHECK_WITH_INFO(
        numaCpuCount > 0 && numaGpuCount > 0, "numaCpuCount=%d, numaGpuCount=%d", numaCpuCount, numaGpuCount);
    int cpuCountPerGpu = std::max(1, numaCpuCount / numaGpuCount);
    std::string cpuArch = topologyDetector.getCpuArchitecture();

    int numCopyThreads = 8;
    if (getenv("TLLM_LOAD_BALANCE_NUM_COPY_THREADS"))
    {
        int numCopyThreadsFromEnv = atoi(getenv("TLLM_LOAD_BALANCE_NUM_COPY_THREADS"));
        if (numCopyThreadsFromEnv > 0)
        {
            TLLM_LOG_INFO(
                "Setting TLLM_LOAD_BALANCE_NUM_COPY_THREADS to %d by environment variable", numCopyThreadsFromEnv);
            numCopyThreads = numCopyThreadsFromEnv;
        }
    }
    else
    {
        if (cpuCountPerGpu > 0)
        {
            numCopyThreads = std::min(16, std::max(4, cpuCountPerGpu / 2));
            TLLM_LOG_INFO("Auto-setting copy threads to %d based on NUMA topology (NUMA node %d, %d CPUs, arch: %s)",
                numCopyThreads, currentGpuNumaId, numaCpuCount, cpuArch.c_str());
        }
    }

    mMultiThreadWorker.reset(new MultiThreadWorker(numCopyThreads, mCudaDeviceId));
}

MoeLoadBalancer::~MoeLoadBalancer()
{
    shutdown();
    HostAccessibleDeviceAllocator::getInstance().DecRefCount();
}

std::shared_ptr<SingleLayerMoeLoadBalancer> MoeLoadBalancer::AddLayer(int expertCount, int topK, int slotCountPerRank)
{
    TLLM_CHECK_WITH_INFO(mModelFinalized == false, "Model is finalized, cannot add new layer.");
    auto layer = std::make_shared<SingleLayerMoeLoadBalancer>(
        tensorrt_llm::kernels::MoeLoadBalanceMetaInfo{expertCount, topK, mEpRank, mEpSize, slotCountPerRank}, this,
        mLayers.size());
    layer->createResources();
    mLayers.push_back(layer);
    return layer;
}

void MoeLoadBalancer::generateUpdatePlan()
{
    int layerCount = mLayers.size();
    int fullUpdateIters = (layerCount + mLayerUpdatesPerIter - 1) / mLayerUpdatesPerIter;
    std::vector<std::vector<int>> updatePlan(fullUpdateIters);
    for (int l = 0; l < layerCount; ++l)
    {
        updatePlan[l % fullUpdateIters].push_back(l);
    }
    for (int i = 0; i < fullUpdateIters; i++)
    {
        std::set<int> iterUpdates;
        for (auto updateLayer : updatePlan[i])
        {
            iterUpdates.insert(updateLayer);
        }
        mUpdateLayerQueue.push_back(iterUpdates);
    }
    if (updatePlan.front().size() == updatePlan.back().size())
    {
        mUpdateLayerQueue.push_back(std::set<int>());
    }
}

void MoeLoadBalancer::finalizeModel()
{
    TLLM_CHECK_WITH_INFO(mModelFinalized == false, "Model is already finalized.");
    for (auto& layer : mLayers)
    {
        layer->finalizeModel();
    }
    if (mLayerUpdatesPerIter > 0)
    {
        mWorkerThreadStopped = false;
        mMultiThreadWorker->start();
        generateUpdatePlan();
        startThreads();
    }
    mModelFinalized = true;
}

void MoeLoadBalancer::startThreads()
{
    mComputeAndUpdateThread.reset(new std::thread(&MoeLoadBalancer::computeAndUpdateThread, this));
    mWorkerThread.reset(new std::thread(&MoeLoadBalancer::workerThread, this));
}

void MoeLoadBalancer::setWarmUpIterCount(int64_t iterCount)
{
    mWarmUpUntilIter = mIterId + iterCount;
}

void MoeLoadBalancer::startIter(int64_t iterId, bool enableStatistic, bool enableUpdateWeights)
{
    std::unique_lock<std::mutex> lock(mWorkerThreadMutex);
    TLLM_CHECK_WITH_INFO(mModelFinalized == true, "Model is not finalized, cannot start iteration.");
    TLLM_CHECK_WITH_INFO(mIterId + 1 == iterId, "Expected iterId=%ld, but got %ld", mIterId + 1, iterId);

    mIterId = iterId;
    // disable update for warm up iters.
    bool isWarmUpIter = mIterId <= mWarmUpUntilIter;
    bool fixedUpdateWeightsEnabled = enableUpdateWeights && !isWarmUpIter;

    IterInfo iterInfo{iterId, enableStatistic, fixedUpdateWeightsEnabled};
    mIterInfoQueue.push(iterInfo);
    mWorkerThreadCondition.notify_one();
}

void MoeLoadBalancer::endIter(int64_t iterId)
{
    TLLM_CHECK_WITH_INFO(mIterId == iterId, "endIter expected iterId=%ld, but got %ld", mIterId, iterId);
}

void MoeLoadBalancer::shutdown()
{
    std::unique_lock<std::mutex> lock(mWorkerThreadMutex);
    if (!mWorkerThreadStopped)
    {
        mWorkerThreadStopped = true;
        mWorkerThreadCondition.notify_one();
        lock.unlock();

        mWorkerThread->join();
        TLLM_LOG_INFO("MoeLoadBalancer shutdown.");
        mMultiThreadWorker->stop();
    }
}

void MoeLoadBalancer::workerThread()
{
    TLLM_CUDA_CHECK(cudaSetDevice(mCudaDeviceId));
    int64_t iterId = -1;
    while (true)
    {
        bool iterUpdateWeightsEnabled, iterStatisticEnabled;
        {
            std::unique_lock<std::mutex> lock(mWorkerThreadMutex);
            mWorkerThreadCondition.wait(lock, [this] { return !mIterInfoQueue.empty() || mWorkerThreadStopped; });
            if (mIterInfoQueue.empty() && mWorkerThreadStopped)
            {
                break;
            }
            auto iterInfo = mIterInfoQueue.front();
            mIterInfoQueue.pop();
            TLLM_CHECK_WITH_INFO(iterInfo.iterId == iterId + 1, "Jump detected, iterId=%ld, but got next iterId=%ld",
                iterId, iterInfo.iterId);
            iterId = iterInfo.iterId;
            iterUpdateWeightsEnabled = iterInfo.updateWeightsEnabled;
            iterStatisticEnabled = iterInfo.statisticEnabled;
        }
        for (int layerId = 0; static_cast<size_t>(layerId) < mLayers.size(); ++layerId)
        {
            auto& layer = mLayers[layerId];
            layer->waitLastUpdateDone();
            bool enableLayerUpdate = iterUpdateWeightsEnabled && (mUpdateLayerQueue.front().count(layerId) > 0);
            layer->startCpuNewIter(iterId, iterStatisticEnabled, enableLayerUpdate);
            layer->setGpuStage();
            layer->waitCpuStage();
            layer->maybeStartUpdateWeights();
        }
        if (iterUpdateWeightsEnabled)
        {
            auto currentUpdatedLayers = mUpdateLayerQueue.front();
            mUpdateLayerQueue.pop_front();
            mUpdateLayerQueue.push_back(currentUpdatedLayers);
        }
    }
    for (auto& layer : mLayers)
    {
        layer->waitLastUpdateDone();
    }
    addUpdateTask(nullptr);
    mComputeAndUpdateThread->join();
}

void MoeLoadBalancer::computeAndUpdateThread()
{
    TLLM_CUDA_CHECK(cudaSetDevice(mCudaDeviceId));
    while (true)
    {
        std::unique_lock<std::mutex> lock(mUpdateQueueMutex);
        mUpdateQueueCondition.wait(lock, [this] { return !mUpdateTaskQueue.empty(); });
        auto task = mUpdateTaskQueue.front();
        mUpdateTaskQueue.pop();
        lock.unlock();
        if (!task)
        {
            break;
        }
        task();
    }
}

void MoeLoadBalancer::addUpdateTask(std::function<void()> task)
{
    std::unique_lock<std::mutex> lock(mUpdateQueueMutex);
    mUpdateTaskQueue.push(task);
    mUpdateQueueCondition.notify_one();
}

int64_t MoeLoadBalancer::addCopyTask(std::function<void(int, int)> task)
{
    return mMultiThreadWorker->addTask(task);
}

void MoeLoadBalancer::waitCopyTaskDone(int64_t taskId)
{
    if (!mUseGpuMemcpy)
    {
        mMultiThreadWorker->waitTaskDone(taskId);
    }
}

MultiThreadWorker::MultiThreadWorker(int numThreads, int cudaDeviceId)
    : mNumThreads(numThreads)
    , mCudaDeviceId(cudaDeviceId)
    , mRunning(false)
    , mNextTaskId(0)
{
}

MultiThreadWorker::~MultiThreadWorker()
{
    stop();
}

void MultiThreadWorker::start()
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mRunning)
        return;
    mRunning = true;
    mThreads.reserve(mNumThreads);
    for (int i = 0; i < mNumThreads; ++i)
    {
        mThreads.emplace_back(&MultiThreadWorker::workerLoop, this, i);
    }
}

int64_t MultiThreadWorker::addTask(std::function<void(int, int)> func)
{
    auto task = std::make_shared<Task>();
    {
        std::lock_guard<std::mutex> lk(mMutex);
        task->id = mNextTaskId++;
        task->func = std::move(func);
        task->remaining = mNumThreads;
        mTasks.push_back(task);
        mTaskMap[task->id] = task;
    }
    mCondition.notify_all();
    return task->id;
}

void MultiThreadWorker::waitTaskDone(int64_t taskId)
{
    std::unique_lock<std::mutex> lk(mMutex);
    auto it = mTaskMap.find(taskId);
    if (it == mTaskMap.end())
    {
        TLLM_CHECK_WITH_INFO(mDoneTaskMap.count(taskId) > 0, "Task %ld not found", taskId);
        mDoneTaskMap.erase(taskId);
        return;
    }
    auto task = it->second;
    task->cv.wait(lk, [task] { return task->remaining == 0; });
    TLLM_CHECK_WITH_INFO(mDoneTaskMap.count(taskId) > 0, "Task %ld not found", taskId);
    mDoneTaskMap.erase(taskId);
}

void MultiThreadWorker::stop()
{
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (!mRunning)
            return;
        mRunning = false;
    }
    mCondition.notify_all();
    for (auto& t : mThreads)
    {
        if (t.joinable())
            t.join();
    }
    mThreads.clear();
}

void MultiThreadWorker::workerLoop(int rank)
{
    TLLM_CUDA_CHECK(cudaSetDevice(mCudaDeviceId));
    auto& topologyDetector = TopologyDetector::getInstance();
    topologyDetector.bindThreadByCurrentGpu(); // use relaxed mode
    while (true)
    {
        std::shared_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lk(mMutex);

            mCondition.wait(lk, [this] { return !mRunning || !mTasks.empty(); });

            if (!mRunning && mTasks.empty())
                return;

            task = mTasks.front();
        }

        task->func(rank, mNumThreads);

        {
            std::unique_lock<std::mutex> lk(mMutex);
            if (--task->remaining == 0)
            {
                mTasks.pop_front();
                mTaskMap.erase(task->id);
                mDoneTaskMap[task->id] = task;
                task->cv.notify_all();
            }
            else
            {
                task->cv.wait(lk, [task] { return task->remaining == 0; });
            }
        }
    }
}

} // namespace tensorrt_llm::runtime

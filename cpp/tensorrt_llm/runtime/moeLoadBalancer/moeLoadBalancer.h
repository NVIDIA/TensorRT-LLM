/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moe/loadBalance/moeLoadBalanceCommon.h"

namespace tensorrt_llm::runtime
{

struct MoeWeight
{
    // use 2D layout to support stride tensor using cudaMemcpy2D
    void* mWeightPtr = nullptr;
    size_t mHeight = 0;
    size_t mWidth = 0;
    size_t mPitch = 0;

    int64_t getWeightPtr() const
    {
        return reinterpret_cast<int64_t>(mWeightPtr);
    }

    void setWeightPtr(int64_t weightPtr)
    {
        mWeightPtr = reinterpret_cast<void*>(weightPtr);
    }
};

struct MoePlacementCpuInfo
{
    std::vector<int> expertReplicaCount;

    // rankExpertIds[i][j] is the list of expert ids for rank i's j-th slot
    // all ranks have the same number of slots, so rankExpertIds is a vector of vectors.
    // it can also be a tensor of shape [epSize, slotCountPerRank]
    std::vector<std::vector<int>> rankExpertIds;

    // oldRankExpertIds[i][j] is the list of old rank ids for expert i's j-th replica
    // same as rankExpertIds but holding the last iteration's rank ids
    std::vector<std::vector<int>> oldRankExpertIds;

    tensorrt_llm::kernels::MoePlacementInfo placementInfoForGPU;
};

class SingleLayerMoeLoadBalancer;

class MoeWeightUpdaterBase
{
public:
    MoeWeightUpdaterBase(
        tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, SingleLayerMoeLoadBalancer* layerLoadBalancer);

    virtual ~MoeWeightUpdaterBase() {}

    // A CPU-copied slot keeps its host pointer, or with fault tolerance its device address, since a spare slot gets its
    // host mapping only when a restore backs it.
    void addSingleWeightSlot(
        int localSlotId, std::string const& name, MoeWeight weightSlot, bool gpuAccess, bool faultTolerance);
    virtual void addSingleHostWeight(int expertId, std::string const& name, MoeWeight hostWeight) = 0;
    virtual void finalizeWeights();
    virtual void updateWeights(MoePlacementCpuInfo const* placementCpuInfo, int rank = 0, int size = 1) = 0;

protected:
    void finalizeWeightSlot();
    bool mWeightSlotsFinalized = false;
    std::map<std::string, std::vector<MoeWeight>> mWeightSlots;
    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo mMetaInfo;
    SingleLayerMoeLoadBalancer* mLayerLoadBalancer;
};

class HostMemoryMoeWeightUpdater : public MoeWeightUpdaterBase
{
public:
    HostMemoryMoeWeightUpdater(
        tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, SingleLayerMoeLoadBalancer* layerLoadBalancer);

    ~HostMemoryMoeWeightUpdater() {}

    void addSingleHostWeight(int expertId, std::string const& name, MoeWeight hostWeight) override;
    void finalizeWeights() override;

    void updateWeights(MoePlacementCpuInfo const* placementCpuInfo, int rank = 0, int size = 1) override;

private:
    static void copyWeights(MoeWeight const& src, MoeWeight const& dst, cudaStream_t stream);
    static void copyWeightsCpu(MoeWeight const& src, MoeWeight const& dst, int rank, int size, bool dstIsDeviceAddress);
    void finalizeHostWeight();
    bool mHostWeightsFinalized = false;
    std::map<std::string, std::vector<MoeWeight>> mHostWeights;
};

class MoeLoadBalancer;

class SingleLayerMoeLoadBalancer
{
public:
    SingleLayerMoeLoadBalancer(
        tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, MoeLoadBalancer* loadBalancer, int balanceLayerId);
    ~SingleLayerMoeLoadBalancer();

    // interface for weights management
    // should bind to python
    void addSingleWeightSlot(int slotId, std::string const& name, MoeWeight weightSlot);
    // should bind to python
    void addSingleHostWeight(int expertId, std::string const& name, MoeWeight hostWeight);

    // set initial weight assignments for each slot
    // index is the global slot id, value is the expert id, or -1 for an empty slot
    // should bind to python
    void setInitialWeightAssignments(std::vector<int> const& initialWeightAssignments);

    int64_t getSelfPtr() const
    {
        return reinterpret_cast<int64_t>(this);
    }

    int getLayerId() const
    {
        return mLayerId;
    }

    std::vector<std::vector<int>> getOldRankExpertIds() const
    {
        // This API is only used in test_moe_module to check expert placement.
        return mCpuPlacementInfo.oldRankExpertIds;
    }

    // should bind to python
    // Return the local slots that the initial or the last installed placement leaves empty, which EPLB's updates leave
    // empty too. Unlike getOldRankExpertIds, they can be read while an update runs.
    std::vector<int> getUnusableLocalSlots() const;

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo getMetaInfo() const
    {
        return mMetaInfo;
    }

    cudaStream_t getStream() const;

    MoePlacementCpuInfo* getPlacementCpuInfo()
    {
        return &mCpuPlacementInfo;
    }

    tensorrt_llm::kernels::MoePlacementInfo getGpuPlacementInfo()
    {
        return mGpuPlacementGpuAccess;
    }

    // Return whether routing uses the direct expert-to-slot kernel, because every expert has a single replica. Under
    // fault tolerance this is fixed by the initial placement: CUDA graphs capture the routing kernel, so every later
    // placement must keep a single replica per expert (checkKeepsSingleReplicaPerExpert).
    bool usesSingleReplicaRouting() const;

    tensorrt_llm::kernels::MoeLoadBalanceSingleLayerSignal* getSignal()
    {
        return mSingleLayerSignal;
    }

    // interfaces for tests
    tensorrt_llm::kernels::MoeLoadBalanceStatisticInfo* getStatisticInfo()
    {
        return &mStatisticInfo;
    }

private:
    friend class MoeLoadBalancer;
    friend class HostMemoryMoeWeightUpdater;

    void createResources();
    void destroyResources();
    void finalizeModel();

    // interface for worker thread from MoeLoadBalancer
    void startCpuNewIter(int64_t iterId, bool enableStatistic, bool enableUpdateWeights);
    void setGpuStage();
    void waitCpuStage();
    void maybeStartUpdateWeights();
    void waitLastUpdateDone();
    void resetAfterAbandon();
    void validateMaskOnly(std::vector<uint8_t> const& deadRankMask) const;
    void reconfigureMaskOnly(std::vector<uint8_t> const& deadRankMask);
    std::vector<int> computeMaskOnlyReplicaCounts(std::vector<uint8_t> const& deadRankMask) const;
    std::vector<int> computeReplicaCounts(std::vector<uint8_t> const& deadRankMask,
        std::vector<std::vector<int>> const& rankExpertIds, char const* operation) const;
    void validatePlacement(
        std::vector<uint8_t> const& deadRankMask, std::vector<std::vector<int>> const& rankExpertIds) const;
    void installPlacement(std::vector<uint8_t> const& deadRankMask, std::vector<std::vector<int>> const& rankExpertIds);
    void checkKeepsSingleReplicaPerExpert(std::vector<int> const& expertReplicaCount) const;
    void updateSingleReplicaPerExpert();

    MoeLoadBalancer* mMoeLoadBalancer = nullptr;
    // Whether the initial placement has a single replica per expert.
    bool mSingleReplicaPerExpert = false;

    std::unique_ptr<MoeWeightUpdaterBase> mWeightUpdater;

    int64_t mIterId = -1;
    bool mStatisticEnabled = true;
    bool mUpdateWeightsEnabled = true;

    void copyPlacementInfoToGpu();
    void copyPlacementInfoToGpuByCpu();
    void updateWeightsRoutine();
    void updateWeightsRoutineByCpu();

    int64_t mLastUpdateTaskId = -1;

    // Per global slot, 1 if a live rank's slot holds no expert (-1 in the initial or an installed placement), so EPLB's
    // updates must leave it empty. Slots of dead ranks are excluded by the dead rank mask instead. Empty if no slot is
    // unusable.
    std::vector<uint8_t> mUnusableSlots;

    cudaEvent_t mUpdateWeightsDoneEvent = nullptr;
    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo mMetaInfo;
    tensorrt_llm::kernels::MoeLoadBalanceStatisticInfo mStatisticInfo;
    MoePlacementCpuInfo mCpuPlacementInfo;
    tensorrt_llm::kernels::MoePlacementInfo mGpuPlacementHostAccess;
    tensorrt_llm::kernels::MoePlacementInfo mGpuPlacementGpuAccess;
    tensorrt_llm::kernels::MoeLoadBalanceSingleLayerSignal* mSingleLayerSignal = nullptr;

    std::mutex mUpdateWeightsMutex;
    std::condition_variable mUpdateWeightsCondition;
    bool mUpdateWeightsDone = false;

    int mLayerId = -1;
};

class MultiThreadWorker
{
public:
    explicit MultiThreadWorker(int numThreads, int cudaDeviceId);
    ~MultiThreadWorker();

    void start();
    int64_t addTask(std::function<void(int, int)> func);
    void waitTaskDone(int64_t taskId);
    void stop();

private:
    struct Task
    {
        int64_t id;
        std::function<void(int, int)> func;
        int remaining;
        std::condition_variable cv;
    };

    void workerLoop(int rank);

    int mNumThreads;
    int mCudaDeviceId;
    std::vector<std::thread> mThreads;
    std::mutex mMutex;
    std::condition_variable mCondition;

    std::deque<std::shared_ptr<Task>> mTasks;

    std::unordered_map<int64_t, std::shared_ptr<Task>> mTaskMap;
    std::unordered_map<int64_t, std::shared_ptr<Task>> mDoneTaskMap;

    bool mRunning;
    int64_t mNextTaskId;
};

class MoeLoadBalancer
{
public:
    MoeLoadBalancer(int epRank, int epSize, int layerUpdatesPerIter, bool faultTolerance = false);
    ~MoeLoadBalancer();

    // Add a new layer to the load balancer
    // Should be called in order, and only once for each layer
    // should bind to python
    std::shared_ptr<SingleLayerMoeLoadBalancer> AddLayer(int expertCount, int topK, int slotCountPerRank);
    // should bind to python
    void finalizeModel();

    // should bind to python
    void setWarmUpIterCount(int64_t iterCount);

    // should bind to python
    void startIter(int64_t iterId, bool enableStatistic, bool enableUpdateWeights);
    // should bind to python
    void endIter(int64_t iterId);

    // should bind to python
    void shutdown();

    // should bind to python
    // This API only validates placement safety, i.e. every expert keeps at least one surviving replica.
    // The caller must separately gate degraded-mode capacity/HBM headroom before invoking it.
    void reconfigureMaskOnly(std::vector<int> const& deadRanks);

    // should bind to python
    // Set the failed ranks and install a placement per layer, while no iteration runs. Every rank must finish restoring
    // before any rank routes with the new placement. The caller loads every local slot whose expert the placement
    // changes before the call.
    void restorePlacement(
        std::vector<int> const& deadRanks, std::vector<std::vector<std::vector<int>>> const& layerRankExpertIds);

    // should bind to python
    // End the online EPLB work of the iterations started before a rank failure, and restart the statistics and the
    // update plan so that every survivor updates the same layers. Every launched kernel must be able to finish.
    void abandonIterations();

    // should bind to python
    // Check that no iteration runs or is queued and the online EPLB worker is done with the last one, so that the
    // caller may write local slots.
    void checkSlotsWritable();

    // Test interface to use GPU to do memcpy test functionality
    void setUseGpuMemcpy(bool useGpuMemcpy = false)
    {
        mUseGpuMemcpy = useGpuMemcpy;
    }

private:
    friend class SingleLayerMoeLoadBalancer;
    friend class HostMemoryMoeWeightUpdater;

    void startThreads();

    void checkRecoveryCall(char const* action) const;

    // Like checkRecoveryCall, and also check that no iteration is queued and the online EPLB worker is done with the
    // last one. Call with mWorkerThreadMutex held.
    void checkUpdatesEnded(char const* action) const;

    // worker thread is used to wait for gpu update done signal
    // and also used to trigger cpu update and wait for it done and signal gpu
    void workerThread();
    std::mutex mWorkerThreadMutex;
    std::condition_variable mWorkerThreadCondition;
    // Wide-EP fault tolerance: slots may hold no expert (-1), a layer with a single replica per expert uses the direct
    // routing kernel, and the recovery calls (restorePlacement, abandonIterations, checkSlotsWritable) are allowed.
    bool mFaultTolerance = false;
    int64_t mWorkerDoneIterId = -1;
    std::condition_variable mWorkerDoneCondition;
    std::atomic<bool> mAbandoning{false};
    bool mWorkerThreadStopped = true;
    int64_t mWarmUpUntilIter = -1;

    // we use a separate thread to compute and update weights to avoid possible blocking for next layer due to slow
    // compute.
    void computeAndUpdateThread();
    std::mutex mUpdateQueueMutex;
    std::condition_variable mUpdateQueueCondition;
    std::queue<std::function<void()>> mUpdateTaskQueue;
    void addUpdateTask(std::function<void()> task);
    int64_t addCopyTask(std::function<void(int, int)> task);
    void waitCopyTaskDone(int64_t taskId);
    std::vector<uint8_t> getDeadRankMaskSnapshot();

    std::vector<std::shared_ptr<SingleLayerMoeLoadBalancer>> mLayers;

    int64_t mIterId = -1;

    struct IterInfo
    {
        int64_t iterId = -1;
        bool statisticEnabled = true;
        bool updateWeightsEnabled = true;
    };

    std::queue<IterInfo> mIterInfoQueue;

    bool mModelFinalized = false;
    bool mIterActive = false;

    int mEpRank = 0;
    int mEpSize = 1;

    cudaStream_t mStream = nullptr;
    int mCudaDeviceId = -1;

    std::unique_ptr<std::thread> mWorkerThread;
    std::unique_ptr<std::thread> mComputeAndUpdateThread;

    std::unique_ptr<MultiThreadWorker> mMultiThreadWorker;

    std::mutex mDeadRankMaskMutex;
    std::vector<uint8_t> mDeadRankMask;

    // update plan member and function
    int mLayerUpdatesPerIter = 1;
    std::deque<std::set<int>> mUpdateLayerQueue;
    void generateUpdatePlan();

    bool mUseGpuMemcpy = false;
};

// functions exposed for testing
// Slots that unusableSlotMask marks are left -1.
void doReplication(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, float* const expertLoadFactor,
    MoePlacementCpuInfo* cpuPlacement, std::vector<uint8_t> const* deadRankMask = nullptr,
    std::vector<uint8_t> const* unusableSlotMask = nullptr);

void doPlacement(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, float* const expertLoadFactor,
    MoePlacementCpuInfo* cpuPlacement, std::vector<uint8_t> const* deadRankMask = nullptr,
    std::vector<uint8_t> const* unusableSlotMask = nullptr);

} // namespace tensorrt_llm::runtime

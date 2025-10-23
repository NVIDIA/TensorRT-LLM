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
#include "tensorrt_llm/kernels/moeLoadBalance/moeLoadBalanceCommon.h"

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

    void addSingleWeightSlot(int localSlotId, std::string const& name, MoeWeight weightSlot, bool gpuAccess);
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
    static void copyWeightsCpu(MoeWeight const& src, MoeWeight const& dst, int rank, int size);
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
    // index is the global slot id, value is the expert id
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

    MoeLoadBalancer* mMoeLoadBalancer = nullptr;

    std::unique_ptr<MoeWeightUpdaterBase> mWeightUpdater;

    int64_t mIterId = -1;
    bool mStatisticEnabled = true;
    bool mUpdateWeightsEnabled = true;

    void copyPlacementInfoToGpu();
    void copyPlacementInfoToGpuByCpu();
    void updateWeightsRoutine();
    void updateWeightsRoutineByCpu();

    int64_t mLastUpdateTaskId = -1;

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
    MoeLoadBalancer(int epRank, int epSize, int layerUpdatesPerIter);
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

    // Test interface to use GPU to do memcpy test functionality
    void setUseGpuMemcpy(bool useGpuMemcpy = false)
    {
        mUseGpuMemcpy = useGpuMemcpy;
    }

private:
    friend class SingleLayerMoeLoadBalancer;
    friend class HostMemoryMoeWeightUpdater;

    void startThreads();

    // worker thread is used to wait for gpu update done signal
    // and also used to trigger cpu update and wait for it done and signal gpu
    void workerThread();
    std::mutex mWorkerThreadMutex;
    std::condition_variable mWorkerThreadCondition;
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

    int mEpRank = 0;
    int mEpSize = 1;

    cudaStream_t mStream = nullptr;
    int mCudaDeviceId = -1;

    std::unique_ptr<std::thread> mWorkerThread;
    std::unique_ptr<std::thread> mComputeAndUpdateThread;

    std::unique_ptr<MultiThreadWorker> mMultiThreadWorker;

    // update plan member and function
    int mLayerUpdatesPerIter = 1;
    std::deque<std::set<int>> mUpdateLayerQueue;
    void generateUpdatePlan();

    bool mUseGpuMemcpy = false;
};

// functions exposed for testing
void doReplication(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, float* const expertLoadFactor,
    MoePlacementCpuInfo* cpuPlacement);

void doPlacement(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo, float* const expertLoadFactor,
    MoePlacementCpuInfo* cpuPlacement);

} // namespace tensorrt_llm::runtime

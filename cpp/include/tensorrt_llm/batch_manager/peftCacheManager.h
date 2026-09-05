/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/peftCacheManagerConfig.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/runtime/loraCache.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/workerPool.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include "tensorrt_llm/common/tllmDataType.h"

#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::batch_manager
{

using runtime::SizeType32;

class PeftTaskNotCachedException : public runtime::LoraExpectedException
{
public:
    explicit PeftTaskNotCachedException(std::string const& msg);
    ~PeftTaskNotCachedException() noexcept override;
};

/**
 * Per-iteration view of PEFT cache activity.
 *
 * The transition counters are cleared when read, so each drain reports only the
 * activity since the previous one. The gauges are sampled at drain time and
 * describe the cache as it stands.
 */
struct PeftCacheIterationStats
{
    //! Ownership transitions, counted per (adapter, request) pair. A request pausing
    //! does not by itself free anything -- other requests may still hold the adapter.
    std::uint64_t requestsPaused{0};
    std::uint64_t requestsResumed{0};
    std::uint64_t requestsTerminated{0};

    //! Adapters whose last holder let go, making their pages evictable. This is the
    //! transition that actually frees capacity: requestsPaused counts intent,
    //! tasksReleasedDevice counts effect.
    std::uint64_t tasksReleasedDevice{0};
    //! Adapters dropped from the host cache too, i.e. released with no paused holder
    //! left to come back for them.
    std::uint64_t tasksReleasedHost{0};

    std::uint64_t tasksEvictedDevice{0};
    std::uint64_t pagesEvictedDevice{0};
    std::uint64_t tasksEvictedHost{0};
    std::uint64_t pagesEvictedHost{0};

    SizeType32 devicePagesTotal{0};
    SizeType32 devicePagesAvailable{0};
    SizeType32 hostPagesTotal{0};
    SizeType32 hostPagesAvailable{0};
    //! Device-cache adapters still held by a request, and those marked done (evictable).
    SizeType32 deviceTasksInProgress{0};
    SizeType32 deviceTasksDone{0};
    //! Distinct adapters with at least one active request, and with at least one paused request.
    SizeType32 activeTasks{0};
    SizeType32 pausedTasks{0};
};

/**
 * BasePeftCacheManager
 *
 * Manages caches of PEFT (Parameter Efficient Fine Tuning) weights.
 * Does cache updates during execution loop moving weights to device as needed.
 */
class BasePeftCacheManager
{
public:
    using LlmRequestPtr = std::shared_ptr<LlmRequest>;
    using RequestVector = std::vector<LlmRequestPtr>;
    using PeftTable = std::unordered_map<uint64_t, std::vector<runtime::LoraCache::TaskLayerModuleConfig>>;
    using TaskPeftTable = std::unordered_map<uint64_t, std::vector<runtime::LoraCache::TaskLayerModuleConfig>>;
    using TaskIdToReqIds = std::unordered_map<uint64_t, std::vector<uint64_t>>;
    using EnsureBatchTaskResult = std::tuple<TaskPeftTable, TaskIdToReqIds>;

    virtual ~BasePeftCacheManager() = default;

    /**
     * \brief add PEFT weights from llmRequest if any.  This will kickoff background copy tasks.
     * \param[in] llmRequest: the request
     * \param[in] tryGpuCache: if true try to load weights into gpu cache
     */
    virtual void addRequestPeft(LlmRequestPtr llmRequest, bool tryGpuCache = true) = 0;

    /**
     * \brief ensures device cache has all the weights needed to execute batch as specified by requests.
     * This acts as sync for the copy tasks started by addRequestPeft
     * \param[in] contextRequests: current context requests
     * \param[in] genRequests: current generation requests
     * \param[in] resetGpuCache: reset (make all tasks evictable)
     * \returns -- a PeftTable
     */
    virtual PeftTable ensureBatch(
        RequestVector const& contextRequests, RequestVector const& generationRequests, bool resetGpuCache = false)
        = 0;

    /**
     * \brief mark all the tasks in device cache as done
     */
    virtual void resetDeviceCache() = 0;

    virtual void markRequestDone(LlmRequest const& llmReq, bool pause = false) = 0;

    [[nodiscard]] virtual SizeType32 getMaxDevicePages() const = 0;

    [[nodiscard]] virtual SizeType32 getMaxHostPages() const = 0;

    [[nodiscard]] virtual SizeType32 determineNumPages(std::shared_ptr<LlmRequest> llmRequest) const = 0;

    [[nodiscard]] virtual bool enabled() const = 0;

    //! \brief Read and clear this iteration's cache activity. Called once per iteration
    //! from the executor loop; the transition counters reset on read.
    [[nodiscard]] virtual PeftCacheIterationStats getAndResetIterationStats() = 0;
};

class PeftCacheManager : public BasePeftCacheManager
{
public:
    using EnsureBatchTaskResult = BasePeftCacheManager::EnsureBatchTaskResult;

    //! \param[in] enableStats: record the per-iteration counters reported by
    //! getAndResetIterationStats(). Off by default; the executor turns it on when
    //! iteration performance stats are enabled.
    PeftCacheManager(PeftCacheManagerConfig const& config, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager, bool enableStats = false);

    ~PeftCacheManager() override = default;

    void addRequestPeft(std::shared_ptr<LlmRequest> llmRequest, bool tryGpuCache = true) override;

    PeftTable ensureBatch(RequestVector const& contextRequests, RequestVector const& generationRequests,
        bool resetGpuCache = false) override;

    EnsureBatchTaskResult ensureBatchMapTaskId(
        RequestVector const& contextRequests, RequestVector const& generationRequests, bool resetGpuCache = false);

    [[nodiscard]] bool isTaskCached(uint64_t taskId) const;

    [[nodiscard]] bool isTaskDone(uint64_t taskId) const;

    [[nodiscard]] bool isTaskDoneDevice(uint64_t taskId) const;

    [[nodiscard]] bool isTaskCachedDevice(uint64_t const taskId) const;

    [[nodiscard]] tensorrt_llm::DataType getDataType() const;

    //! Configure the homogeneous LoRA weight dtype before inserting an adapter.
    void configureDataType(tensorrt_llm::DataType dataType);

    void resetDeviceCache() override;

    void markRequestDone(LlmRequest const& llmReq, bool pause = false) override;

    [[nodiscard]] SizeType32 getMaxDevicePages() const override;

    [[nodiscard]] SizeType32 getMaxHostPages() const override;

    [[nodiscard]] SizeType32 determineNumPages(std::shared_ptr<LlmRequest> llmRequest) const override;

    inline bool enabled() const override
    {
        return true;
    }

    [[nodiscard]] PeftCacheIterationStats getAndResetIterationStats() override;

    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> const& getActiveTasks() const;

    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> const& getPausedTasks() const;

    void updateTaskState(uint64_t taskId, uint64_t reqId, bool terminate = false, bool pause = false);

    static std::pair<uint64_t, uint64_t> getMaxNumSlots(PeftCacheManagerConfig const& config,
        tensorrt_llm::DataType dataType, uint64_t pageWidth, uint64_t max1dModSize,
        runtime::BufferManager const& bufferManager);

    static std::pair<runtime::LoraCachePageManagerConfig, runtime::LoraCachePageManagerConfig> getPageManagerConfig(
        PeftCacheManagerConfig const& config, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager);

    void prefetchLoraWeights(std::string const& modelDir, runtime::BufferManager const& bufferManager);

private:
    static std::pair<uint64_t, uint64_t> getMaxNumSlots(PeftCacheManagerConfig const& config,
        tensorrt_llm::DataType dataType, uint64_t pageWidth, uint64_t max1dModSize,
        std::optional<uint64_t> deviceCacheByteBudget);

    static std::pair<runtime::LoraCachePageManagerConfig, runtime::LoraCachePageManagerConfig> getPageManagerConfig(
        PeftCacheManagerConfig const& config, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, tensorrt_llm::DataType dataType,
        std::optional<uint64_t> deviceCacheByteBudget);

    std::unique_ptr<runtime::LoraCache> mHostLoraCache;
    std::unique_ptr<runtime::LoraCache> mDeviceLoraCache;

    std::shared_ptr<runtime::WorkerPool> mPutWorkerPool;
    std::unique_ptr<runtime::WorkerPool> mEnsureWorkerPool;

    mutable std::mutex mPutFuturesMutex;
    std::unordered_map<std::uint64_t, std::future<void>> mPutFutures;

    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> mTaskIdToReqIds;
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> mTaskIdToPausedReqIds;

    // Ownership-transition counters for the current iteration. Plain integers because
    // they are written only from updateTaskState, which mutates mTaskIdToReqIds without
    // a lock and is therefore already confined to the executor thread. The LoraCache
    // eviction counters they are combined with are atomic, since eviction also runs on
    // the put/ensure worker pools.
    bool mEnableStats{false};
    std::uint64_t mIterRequestsPaused{0};
    std::uint64_t mIterRequestsResumed{0};
    std::uint64_t mIterRequestsTerminated{0};
    std::uint64_t mIterTasksReleasedDevice{0};
    std::uint64_t mIterTasksReleasedHost{0};

    std::tuple<std::unordered_map<uint64_t, std::future<void>>, TaskIdToReqIds> getTaskMaps(
        RequestVector const& contextRequests, RequestVector const& generationRequests);

    runtime::ModelConfig mModelConfig;
    runtime::WorldConfig mWorldConfig;
    PeftCacheManagerConfig mConfig;
    std::optional<uint64_t> mDeviceCacheByteBudget;

    mutable std::mutex mDataTypeMutex;
    std::optional<tensorrt_llm::DataType> mDataType;

    int mDevice{-1};
};

class NoOpPeftCacheManager : public BasePeftCacheManager
{
public:
    ~NoOpPeftCacheManager() override = default;

private:
    void addRequestPeft(std::shared_ptr<LlmRequest> llmRequest, bool tryGpuCache = true) override;

    PeftTable ensureBatch(RequestVector const& contextRequests, RequestVector const& generationRequests,
        bool resetGpuCache = false) override;

    void resetDeviceCache() override;

    void markRequestDone(LlmRequest const& llmReq, bool pause = false) override;

    [[nodiscard]] SizeType32 getMaxDevicePages() const override;

    [[nodiscard]] SizeType32 getMaxHostPages() const override;

    [[nodiscard]] SizeType32 determineNumPages(std::shared_ptr<LlmRequest> llmRequest) const override;

    inline bool enabled() const override
    {
        return false;
    }

    [[nodiscard]] PeftCacheIterationStats getAndResetIterationStats() override
    {
        return {};
    }
};
} // namespace tensorrt_llm::batch_manager

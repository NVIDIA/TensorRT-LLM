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

#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/loraCache.h"
#include "tensorrt_llm/runtime/loraUtils.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tensorrt_llm/runtime/utils/runtimeUtils.h"
#include "tensorrt_llm/runtime/workerPool.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace tensorrt_llm::batch_manager
{

PeftTaskNotCachedException::PeftTaskNotCachedException(std::string const& msg)
    : runtime::LoraExpectedException(msg)
{
}

PeftTaskNotCachedException::~PeftTaskNotCachedException() noexcept = default;

std::pair<uint64_t, uint64_t> PeftCacheManager::getMaxNumSlots(PeftCacheManagerConfig const& config,
    nvinfer1::DataType dataType, uint64_t pageWidth, uint64_t max1dModSize, runtime::BufferManager const& bufferManager)
{
    TLLM_LOG_DEBUG("max1dModeSize=%llu", max1dModSize);
    TLLM_LOG_DEBUG("pageWidth=%llu", pageWidth);
    uint64_t const pageWidthSize = pageWidth * static_cast<uint64_t>(runtime::BufferDataType(dataType).getSize());
    uint64_t totalHostSlots, totalDeviceSlots;
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    if (config.numHostModuleLayer > 0)
    {
        totalHostSlots = static_cast<uint64_t>(config.numHostModuleLayer) * common::ceilDiv(max1dModSize, pageWidth);
        TLLM_LOG_DEBUG("totalHostSlots=%llu", totalHostSlots);
    }
    else
    {
        auto const memSize = config.hostCacheSize.value_or(PeftCacheManagerConfig::kDefaultHostCacheSize);
        totalHostSlots = memSize / pageWidthSize;
        TLLM_LOG_DEBUG("memSize: %llu, totalHostSlots: %llu", memSize, totalHostSlots);
    }

    if (config.numDeviceModuleLayer > 0)
    {
        totalDeviceSlots
            = static_cast<uint64_t>(config.numDeviceModuleLayer) * common::ceilDiv(max1dModSize, pageWidth);
    }
    else
    {
        auto const memPercent = config.deviceCachePercent.value_or(PeftCacheManagerConfig::kDefaultDeviceCachePercent);
        auto const [freeMem, totalMem] = common::getDeviceMemoryInfo(false);
        totalDeviceSlots = static_cast<uint64_t>(static_cast<double>(memPercent)
            * static_cast<double>(freeMem + bufferManager.memoryPoolFree()) / static_cast<double>(pageWidthSize));
    }

    auto hostMem = totalHostSlots * pageWidthSize;
    auto deviceMem = totalDeviceSlots * pageWidthSize;

    TLLM_LOG_INFO("Using " + std::to_string(hostMem) + " bytes for LoRA host cache");
    TLLM_LOG_INFO("Using " + std::to_string(deviceMem) + " bytes for LoRA device cache");

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);

    return std::make_pair(totalHostSlots, totalDeviceSlots);
}

std::pair<runtime::LoraCachePageManagerConfig, runtime::LoraCachePageManagerConfig>
PeftCacheManager::getPageManagerConfig(PeftCacheManagerConfig const& config, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager)
{

    auto const tpSize = worldConfig.getTensorParallelism();
    auto const ppSize = worldConfig.getPipelineParallelism();
    auto const ppRank = worldConfig.getPipelineParallelRank();
    auto const numLocalLayers = modelConfig.getNbAttentionLayers(ppSize, ppRank);
    uint64_t min1dModSize = std::numeric_limits<uint64_t>::max(); // used to setup the pageWidth
    uint64_t total1dModSize = 0;
    uint64_t total1lSlots = 0; // the slots we need for each layer, summing the slots of all modules
    uint64_t max1dModSize = 0; // used to compute the totalHostSlots and totalDeviceSlots
    for (auto const& module : modelConfig.getLoraModules())
    {
        uint64_t const oneDSize = static_cast<uint64_t>(module.localInDim(tpSize) + module.localOutDim(tpSize));
        TLLM_LOG_DEBUG("oneDSize=%llu", oneDSize);
        min1dModSize = std::min(min1dModSize, oneDSize);
        max1dModSize = std::max(max1dModSize, oneDSize);
        total1dModSize += oneDSize;
    }
    TLLM_LOG_DEBUG("total1dModSize=%llu", total1dModSize);
    auto const pageWidth = min1dModSize;

    for (auto const& module : modelConfig.getLoraModules())
    {
        uint64_t const oneDSize = static_cast<uint64_t>(module.localInDim(tpSize) + module.localOutDim(tpSize));
        total1lSlots += config.optimalAdapterSize * common::ceilDiv(oneDSize, pageWidth);
    }
    uint64_t const max1dLoraSize = total1dModSize * static_cast<uint64_t>(numLocalLayers);
    uint64_t const totalSlots = total1lSlots * static_cast<uint64_t>(numLocalLayers);
    uint64_t const maxLoraSize = config.maxAdapterSize * max1dLoraSize;
    uint64_t const minNumSlots = common::ceilDiv(config.maxAdapterSize * max1dModSize, pageWidth);
    uint64_t const numSlotsPerPage = std::max(totalSlots, minNumSlots);
    uint64_t const minTotalSlots = common::ceilDiv(config.maxAdapterSize * max1dLoraSize, pageWidth);
    TLLM_LOG_DEBUG("max1dModSize=%llu", max1dModSize);

    auto [totalHostSlots, totalDeviceSlots]
        = getMaxNumSlots(config, modelConfig.getDataType(), pageWidth, max1dModSize, bufferManager);

    TLLM_CHECK_WITH_INFO(totalHostSlots >= minTotalSlots,
        "Not enough space allocated to host LoRA cache to hold 1 max sized LoRA %lu < %lu", totalHostSlots,
        minTotalSlots);
    TLLM_CHECK_WITH_INFO(totalDeviceSlots >= minTotalSlots,
        "Not enough space allocated to device LoRA cache to hold 1 max sized LoRA %lu < %lu", totalDeviceSlots,
        minTotalSlots);

    uint64_t const totalHostPages = common::ceilDiv(totalHostSlots, numSlotsPerPage);
    uint64_t const totalDevicePages = common::ceilDiv(totalDeviceSlots, numSlotsPerPage);

    uint64_t const totalMaxLorasHost = totalHostPages * numSlotsPerPage * pageWidth / maxLoraSize;
    uint64_t const totalMaxLorasDevice = totalDevicePages * numSlotsPerPage * pageWidth / maxLoraSize;

    TLLM_LOG_INFO("Max LoRA size is %llu", maxLoraSize);

    TLLM_LOG_INFO("LoRA host Cache can hold %llu max sized LoRAs", totalMaxLorasHost);
    TLLM_LOG_INFO("LoRA device Cache can hold %llu max sized LoRAs", totalMaxLorasDevice);

    TLLM_CHECK(std::numeric_limits<SizeType32>::max() >= totalHostPages);
    TLLM_CHECK(std::numeric_limits<SizeType32>::max() >= totalDevicePages);
    TLLM_CHECK(std::numeric_limits<SizeType32>::max() >= numSlotsPerPage);
    TLLM_CHECK(std::numeric_limits<SizeType32>::max() >= pageWidth);

    runtime::LoraCachePageManagerConfig hostPageConfig(runtime::MemoryType::kCPU, modelConfig.getDataType(),
        totalHostPages, config.maxPagesPerBlockHost, numSlotsPerPage, pageWidth, 0);
    runtime::LoraCachePageManagerConfig devicePageConfig(runtime::MemoryType::kGPU, modelConfig.getDataType(),
        totalDevicePages, config.maxPagesPerBlockDevice, numSlotsPerPage, pageWidth, config.numCopyStreams);
    return std::make_pair(hostPageConfig, devicePageConfig);
}

void PeftCacheManager::prefetchLoraWeights(std::string const& modelDir, runtime::BufferManager const& bufferManager)
{
    // This function loads LoRA weights from modelDir. In the folder, users can put many
    // folders for different lora tasks.
    // For example, assume we want to store lora weights in modelDir and there are three
    // lora tasks `0`, `1` and `3`, then the architecture of the folder would be like
    // modelDir
    // ├── 0
    // │   ├── model.lora_config.npy
    // │   └── model.lora_weights.npy
    // ├── 1
    // │   ├── model.lora_config.npy
    // │   └── model.lora_weights.npy
    // └── 3
    //     ├── model.lora_config.npy
    //     └── model.lora_weights.npy
    //
    // If the name of the lora task is not digit, will print warning and
    // skip loading lora weight from the folder
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    namespace fs = std::filesystem;
    std::vector<std::string> tasks;

    if (!fs::exists(modelDir) || !fs::is_directory(modelDir))
    {
        TLLM_LOG_DEBUG("Cannot find the %s, skipping prefetching the lora weights.", modelDir.c_str());
        return;
    }
    // collect the lora tasks under modelDir
    for (auto const& entry : fs::directory_iterator(modelDir))
    {
        if (fs::is_directory(entry.path()))
        {
            std::string task_name = entry.path().filename().string();
            if (all_of(task_name.begin(), task_name.end(), ::isdigit))
            {
                tasks.push_back(task_name);
            }
            else
            {
                TLLM_LOG_WARNING(
                    "lora task name %s is invalid, skipping to load lora weight from this folder.", task_name.c_str());
            }
        }
    }

    TLLM_LOG_DEBUG("find %u lora tasks to prefetch.", tasks.size());
    // load lora tasks one by one
    using TensorPtr = runtime::ITensor::SharedPtr;
    auto const multiLoraPath = fs::path{modelDir};
    for (std::uint32_t taskId = 0; taskId < tasks.size(); ++taskId)
    {
        auto const weightsFn = (multiLoraPath / tasks[taskId] / "model.lora_weights.npy").string();
        auto const configFn = (multiLoraPath / tasks[taskId] / "model.lora_config.npy").string();
        TensorPtr weights = runtime::utils::loadNpy(bufferManager, weightsFn, runtime::MemoryType::kCPU);
        TensorPtr config = runtime::utils::loadNpy(bufferManager, configFn, runtime::MemoryType::kCPU);
        TLLM_LOG_DEBUG("prefetch lora task %s", tasks[taskId].c_str());
        mHostLoraCache->put(std::stoi(tasks[taskId]), weights, config, true);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

PeftCacheManager::PeftCacheManager(PeftCacheManagerConfig const& config, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager)
    : mModelConfig(modelConfig)
    , mWorldConfig(worldConfig)
    , mDevice{runtime::utils::initDevice(worldConfig)}
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto cfg = config;
    cfg.optimalAdapterSize = std::min(cfg.optimalAdapterSize, modelConfig.getMaxLoraRank());
    cfg.maxAdapterSize = std::min(cfg.maxAdapterSize, modelConfig.getMaxLoraRank());
    auto [hostCacheConfig, deviceCacheConfig] = getPageManagerConfig(cfg, modelConfig, worldConfig, bufferManager);
    mHostLoraCache = std::make_unique<runtime::LoraCache>(hostCacheConfig, modelConfig, worldConfig, bufferManager);
    mDeviceLoraCache = std::make_unique<runtime::LoraCache>(deviceCacheConfig, modelConfig, worldConfig, bufferManager);

    mPutWorkerPool = std::make_shared<runtime::WorkerPool>(cfg.numPutWorkers, mDevice);
    mEnsureWorkerPool = std::make_unique<runtime::WorkerPool>(config.numEnsureWorkers, mDevice);

    if (config.loraPrefetchDir.has_value())
    {
        prefetchLoraWeights(config.loraPrefetchDir.value(), bufferManager);
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void PeftCacheManager::addRequestPeft(std::shared_ptr<LlmRequest> llmRequest, bool tryGpuCache)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto optTaskId = llmRequest->getLoraTaskId();
    auto optLoraWeights = llmRequest->getLoraWeights();
    auto optLoraConfig = llmRequest->getLoraConfig();
    if (optTaskId || optLoraWeights || optLoraConfig)
    {
        runtime::lora::loraValidateRequestTensors(optTaskId, optLoraWeights, optLoraConfig, mModelConfig, mWorldConfig);
    }
    else
    {
        // no lora to add to cache so we're done
        return;
    }

    auto const taskId = optTaskId.value();
    if (!optLoraWeights || !optLoraConfig)
    {
        // Throw special exception that's logged as warning in executor
        if (!isTaskCached(taskId))
        {
            std::string errMsg
                = "LoRA task " + std::to_string(taskId) + " not found in cache. Please send LoRA weights with request";
            throw PeftTaskNotCachedException(errMsg);
        }
    }

    auto const reqId = llmRequest->mRequestId;
    TLLM_LOG_DEBUG("addRequestPeft taskId=" + std::to_string(taskId) + " reqId=" + std::to_string(reqId));

    updateTaskState(taskId, reqId);
    {
        // if we are already processing this task we are done
        std::lock_guard<std::mutex> lk(mPutFuturesMutex);
        if (mPutFutures.count(taskId))
        {
            TLLM_LOG_DEBUG(
                "addRequestPeft haveFuture skip taskId=" + std::to_string(taskId) + " reqId=" + std::to_string(reqId));
            return;
        }
    }

    bool loadNeeded;
    try
    {
        if (optLoraWeights && optLoraConfig)
        {
            TLLM_LOG_DEBUG("addRequestPeft put taskId=" + std::to_string(taskId) + " reqId=" + std::to_string(reqId));
            mHostLoraCache->put(taskId, optLoraWeights.value(), optLoraConfig.value(), false);
            loadNeeded = true;
        }
        else
        {
            TLLM_LOG_DEBUG("addRequestPeft bump taskId=" + std::to_string(taskId) + " reqId=" + std::to_string(reqId));
            mHostLoraCache->bump(taskId);
            loadNeeded = false;
        }
    }
    catch (runtime::LoraCacheFullException const& e)
    {
        std::string errMsg("PEFT Cache is full. Could not store taskId=" + std::to_string(taskId));
        TLLM_LOG_ERROR(errMsg);
        updateTaskState(taskId, reqId, true, false);
        // re-throw with better error message
        throw runtime::LoraCacheFullException(errMsg);
    }
    catch (std::runtime_error const& e)
    {
        updateTaskState(taskId, reqId, true, false);
        TLLM_THROW("Error storing task=%lu in PEFT cache -- %s", taskId, e.what());
    }

    auto fn = [taskId, req = llmRequest, loadNeeded, this]()
    {
        auto optWeights = req->getLoraWeights();
        auto optConfig = req->getLoraConfig();
        if (loadNeeded && optWeights.has_value() && optConfig.has_value())
        {
            TLLM_LOG_DEBUG(
                "addRequestPeft load taskId=" + std::to_string(taskId) + " reqId=" + std::to_string(req->mRequestId));
            mHostLoraCache->loadWeights(taskId, optWeights.value(), optConfig.value());
            // free memory associated with lora weights in llmRequest
            req->clearLoraWeights();
            req->clearLoraConfig();
        }

        // TODO (grclark) pre-loading nvbug 4547061
        if (false)
        {
            try
            {
                mHostLoraCache->copyTask(taskId, *mDeviceLoraCache, false);
            }
            catch (std::runtime_error& e)
            {
                TLLM_LOG_DEBUG("failed to copy task " + std::to_string(taskId) + " to gpu cache -- " + e.what());
            }
        }
#ifndef NDEBUG
        if (!mHostLoraCache->isLoaded(taskId))
        {
            throw std::runtime_error("Expected task to be loaded at the end of put " + std::to_string(taskId));
        }
        if (mHostLoraCache->isDone(taskId))
        {
            throw std::runtime_error("Expected task to be in progress at the end of put " + std::to_string(taskId));
        }
#endif
    };

    auto putFuture = mPutWorkerPool->enqueue(fn);
    {
        std::lock_guard<std::mutex> lk(mPutFuturesMutex);
        mPutFutures.try_emplace(taskId, std::move(putFuture));
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

std::tuple<std::map<uint64_t, std::future<void>>, std::map<uint64_t, std::vector<uint64_t>>>
PeftCacheManager::getTaskMaps(RequestVector const& contextRequests, RequestVector const& generationRequests)
{
    std::map<uint64_t, std::vector<uint64_t>> taskIdToReqIds;
    std::map<uint64_t, std::future<void>> taskIdToFuture;
    std::lock_guard<std::mutex> futuresLock(mPutFuturesMutex);
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            if (llmReq->getLoraTaskId().has_value())
            {
                auto const taskId = llmReq->getLoraTaskId().value();
                if (!taskIdToReqIds.count(taskId))
                {
                    taskIdToReqIds.try_emplace(taskId, std::vector<uint64_t>{});
                }
                taskIdToReqIds.at(taskId).push_back(llmReq->mRequestId);
                if (mPutFutures.count(taskId))
                {
                    TLLM_LOG_DEBUG("Ensure batch, has future for " + std::to_string(taskId));
                    taskIdToFuture.try_emplace(taskId, std::move(mPutFutures.at(taskId)));
                    mPutFutures.erase(taskId);
                }
                else if (!taskIdToFuture.count(taskId))
                {
                    /*
                     * if we don't find a future in mPutFutures, we may have already put one in
                     * taskIdToFutures (ie if 2 requests have the same taskId)
                     * If no future is found we create a dummy future for the task
                     */
                    TLLM_LOG_DEBUG("Ensure batch, do not have future for " + std::to_string(taskId));
                    std::promise<void> p;
                    p.set_value();
                    taskIdToFuture.try_emplace(taskId, p.get_future());
                }
            }
        }
    }
    return {std::move(taskIdToFuture), taskIdToReqIds};
}

PeftCacheManager::PeftTable PeftCacheManager::ensureBatch(
    RequestVector const& contextRequests, RequestVector const& generationRequests, bool resetGpuCache)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    if (resetGpuCache)
    {
        resetDeviceCache();
    }
    auto [taskIdToFuture_, taskIdToReqIds] = getTaskMaps(contextRequests, generationRequests);
    auto taskIdToFuture = std::move(taskIdToFuture_); // captured structured bindings are a C++20 extension

    std::map<uint64_t, std::future<std::vector<runtime::LoraCache::TaskLayerModuleConfig>>> ensureFutures;
    for (auto& [taskId, taskFuture] : taskIdToFuture)
    {
        auto fn = [&taskIdToFuture, taskId = taskId, this]() -> std::vector<runtime::LoraCache::TaskLayerModuleConfig>
        {
            // TODO (grclark) it should be possible to move capture taskFuture instead of doing a second lookup
            // And you can, which required this lambda to be mutable, which doesn't work with WorkerPool
            auto&& taskFuture = taskIdToFuture.at(taskId);
            try
            {
                taskFuture.get();
            }
            catch (std::runtime_error& e)
            {
                throw std::runtime_error("Caught Exception while storing peft weights -- " + std::string(e.what()));
            }

            try
            {
                mHostLoraCache->copyTask(taskId, *mDeviceLoraCache);
            }
            catch (std::runtime_error& e)
            {
                throw std::runtime_error("Caught exception during copyTask ensure batch -- " + std::string(e.what()));
            }
            return mDeviceLoraCache->get(taskId);
        };
        auto f = mEnsureWorkerPool->enqueue(fn);
        ensureFutures.try_emplace(taskId, std::move(f));
    }

    PeftTable peftTable{};
    for (auto const& [taskId, reqIds] : taskIdToReqIds)
    {
        auto&& f = ensureFutures.at(taskId);
        auto const values = f.get();
        for (auto const& reqId : reqIds)
        {
            peftTable.try_emplace(reqId, values);
        }
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return peftTable;
}

bool PeftCacheManager::isTaskCached(uint64_t taskId) const
{
    return mHostLoraCache->has(taskId);
}

bool PeftCacheManager::isTaskDone(uint64_t taskId) const
{
    return mHostLoraCache->isDone(taskId);
}

bool PeftCacheManager::isTaskDoneDevice(uint64_t taskId) const
{
    return mDeviceLoraCache->isDone(taskId);
}

void PeftCacheManager::updateTaskState(uint64_t taskId, uint64_t reqId, bool terminate, bool pause)
{
    if (!terminate)
    {
        if (!mTaskIdToReqIds.count(taskId))
        {
            mTaskIdToReqIds.try_emplace(taskId, std::unordered_set<uint64_t>{});
        }
        mTaskIdToReqIds.at(taskId).insert(reqId);
        if (mTaskIdToPausedReqIds.count(taskId))
        {
            mTaskIdToPausedReqIds.at(taskId).erase(reqId);
            if (mTaskIdToPausedReqIds.at(taskId).empty())
            {
                mTaskIdToPausedReqIds.erase(taskId);
            }
        }
    }
    else
    {
        auto activeTaskIt = mTaskIdToReqIds.find(taskId);
        auto pauseTakskIt = mTaskIdToPausedReqIds.find(taskId);
        if (activeTaskIt != mTaskIdToReqIds.end())
        {
            activeTaskIt->second.erase(reqId);
            if (activeTaskIt->second.empty())
            {
                mTaskIdToReqIds.erase(taskId);
            }
        }

        if (pause)
        {
            if (pauseTakskIt == mTaskIdToPausedReqIds.end())
            {
                mTaskIdToPausedReqIds.try_emplace(taskId, std::unordered_set<uint64_t>{});
            }
            mTaskIdToPausedReqIds.at(taskId).insert(reqId);
        }
        else
        {
            if (pauseTakskIt != mTaskIdToPausedReqIds.end())
            {
                pauseTakskIt->second.erase(reqId);
                if (pauseTakskIt->second.empty())
                {
                    mTaskIdToPausedReqIds.erase(taskId);
                }
            }
        }

        if (!mTaskIdToReqIds.count(taskId))
        {
            // paused taskIds get removed from gpu cache but not host cache
            mDeviceLoraCache->markTaskDone(taskId);
            if (!mTaskIdToPausedReqIds.count(taskId))
            {
                {
                    std::lock_guard<std::mutex> lk(mPutFuturesMutex);
                    mPutFutures.erase(taskId);
                    TLLM_LOG_DEBUG(
                        "erase task " + std::to_string(taskId) + " future size=" + std::to_string(mPutFutures.size()));
                }
                mHostLoraCache->markTaskDone(taskId);
            }
        }
    }
}

void PeftCacheManager::resetDeviceCache()
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    mDeviceLoraCache->markAllDone();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void PeftCacheManager::markRequestDone(LlmRequest const& llmReq, bool pause)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    // mDeviceLoraCache->markAllDone();
    if (!llmReq.getLoraTaskId().has_value())
    {
        return;
    }
    auto const taskId = llmReq.getLoraTaskId().value();
    auto const reqId = llmReq.mRequestId;
    updateTaskState(taskId, reqId, true, pause);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

SizeType32 PeftCacheManager::getMaxDevicePages() const
{
    return mDeviceLoraCache->getNumPages();
}

SizeType32 PeftCacheManager::getMaxHostPages() const
{
    return mHostLoraCache->getNumPages();
}

SizeType32 PeftCacheManager::determineNumPages(std::shared_ptr<LlmRequest> llmRequest) const
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    if (llmRequest->getLoraTaskId().has_value())
    {
        try
        {
            return mHostLoraCache->determineNumPages(llmRequest->getLoraTaskId().value());
        }
        catch (std::runtime_error& e)
        {
            if (llmRequest->getLoraConfig().has_value())
            {
                return mHostLoraCache->determineNumPages(llmRequest->getLoraConfig().value());
            }
            throw;
        }
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return 0;
}

std::unordered_map<uint64_t, std::unordered_set<uint64_t>> const& PeftCacheManager::getActiveTasks() const
{
    return mTaskIdToReqIds;
}

std::unordered_map<uint64_t, std::unordered_set<uint64_t>> const& PeftCacheManager::getPausedTasks() const
{
    return mTaskIdToPausedReqIds;
}

void NoOpPeftCacheManager::addRequestPeft(std::shared_ptr<LlmRequest> llmRequest, bool tryGpuCache) {}

PeftCacheManager::PeftTable NoOpPeftCacheManager::ensureBatch(
    RequestVector const& contextRequests, RequestVector const& generationRequests, bool resetGpuCache)
{
    return PeftTable{};
}

void NoOpPeftCacheManager::resetDeviceCache() {}

void NoOpPeftCacheManager::markRequestDone(LlmRequest const& llmReq, bool pause) {}

SizeType32 NoOpPeftCacheManager::getMaxDevicePages() const
{
    return std::numeric_limits<SizeType32>::max();
}

SizeType32 NoOpPeftCacheManager::getMaxHostPages() const
{
    return std::numeric_limits<SizeType32>::max();
}

SizeType32 NoOpPeftCacheManager::determineNumPages(std::shared_ptr<LlmRequest> llmReqeust) const
{
    return 0;
}
} // namespace tensorrt_llm::batch_manager

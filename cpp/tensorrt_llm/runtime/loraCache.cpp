/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/loraCache.h"
#include "bufferManager.h"
#include "cudaEvent.h"
#include "cudaStream.h"
#include "iBuffer.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/loraUtils.h"
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

namespace tensorrt_llm::runtime
{

LoraExpectedException::LoraExpectedException(std::string const& msg)
    : std::runtime_error(msg)
{
}

LoraExpectedException::~LoraExpectedException() noexcept = default;

LoraCacheFullException::LoraCacheFullException(std::string const& msg)
    : LoraExpectedException(msg)
{
}

LoraCacheFullException::~LoraCacheFullException() noexcept = default;

LoraCachePageManager::LoraCachePageManager(LoraCachePageManagerConfig const& config, BufferManager const& bufferManager)
    : mConfig(config)
{
    initialize(bufferManager);
}

void LoraCachePageManager::initialize(BufferManager const& bufferManager)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    TLLM_LOG_DEBUG("pageConfig: " + to_string(mConfig));

    std::size_t pageIdx = 0;
    while (pageIdx < static_cast<size_t>(mConfig.getTotalNumPages()))
    {
        auto const numLocalPages = std::min<SizeType32>(
            mConfig.getTotalNumPages() - static_cast<SizeType32>(pageIdx), mConfig.getMaxPagesPerBlock());
        auto const blockShape = ITensor::makeShape({numLocalPages, mConfig.getSlotsPerPage(), mConfig.getPageWidth()});
        TensorPtr block = bufferManager.allocate(mConfig.getMemoryType(), blockShape, mConfig.getDataType());
        bufferManager.setZero(*block);
        mPageBlocks.push_back(block);
        for (SizeType32 i = 0; i < numLocalPages; ++i)
        {
            mFreePageIds.push_back(pageIdx);
            ++pageIdx;
        }
    }
    mIsPageFree.assign(pageIdx, 1);

    TLLM_LOG_DEBUG("%s allocated %d blocks and %d pages", __PRETTY_FUNCTION__, mPageBlocks.size(), pageIdx);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

std::optional<std::vector<std::size_t>> LoraCachePageManager::claimPages(SizeType32 numPages)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    if (numPages <= static_cast<SizeType32>(mFreePageIds.size()))
    {
        std::vector<std::size_t> outputPages{};
        outputPages.reserve(numPages);
        for (auto it = mFreePageIds.begin();
             outputPages.size() < static_cast<std::size_t>(numPages) && it != mFreePageIds.end();
             it = mFreePageIds.erase(it))
        {
            mIsPageFree.at(*it) = 0;
            outputPages.push_back(*it);
        }
        return std::make_optional(std::move(outputPages));
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return std::nullopt;
}

SizeType32 LoraCachePageManager::numAvailablePages() const
{
    return static_cast<SizeType32>(mFreePageIds.size());
}

void LoraCachePageManager::releasePages(std::vector<std::size_t> const& pageIds)
{
    for (auto pageId : pageIds)
    {
        if (pageId >= mIsPageFree.size() || mIsPageFree[pageId])
        {
            TLLM_LOG_WARNING("Attempted to release already free lora cache page");
        }
        else
        {
            mFreePageIds.push_front(pageId);
            mIsPageFree.at(pageId) = 1;
        }
    }
}

ITensor::SharedConstPtr LoraCachePageManager::blockPtr(SizeType32 blockIdx) const
{
    return mPageBlocks.at(blockIdx);
}

ITensor::SharedConstPtr LoraCachePageManager::pagePtr(std::size_t pageIdx) const
{
    auto blockIdx = pageIdx / mConfig.getMaxPagesPerBlock();
    auto blockPageIdx = pageIdx % mConfig.getMaxPagesPerBlock();

    return ITensor::view(ITensor::slice(mPageBlocks.at(blockIdx), blockPageIdx, 1),
        ITensor::makeShape({mConfig.getSlotsPerPage(), mConfig.getPageWidth()}));
}

ITensor::SharedPtr LoraCachePageManager::mutablePagePtr(std::size_t pageIdx)
{
    auto blockIdx = pageIdx / mConfig.getMaxPagesPerBlock();
    auto blockPageIdx = pageIdx % mConfig.getMaxPagesPerBlock();

    return ITensor::view(ITensor::slice(mPageBlocks.at(blockIdx), blockPageIdx, 1),
        ITensor::makeShape({mConfig.getSlotsPerPage(), mConfig.getPageWidth()}));
}

void LoraCache::put(TaskIdType taskId, TensorPtr sourceWeights, TensorPtr sourceConfig, bool load)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto taskValuePtr = [&]() -> std::optional<TaskValuePtr>
    {
        std::lock_guard<std::mutex> cacheLock(mCacheMutex);
        if (kVALUE_STATUS_MISSING != getStatus(taskId))
        {
            bumpTaskInProgress(taskId);
            TLLM_LOG_DEBUG("%s return nullopt", __PRETTY_FUNCTION__);
            return std::nullopt;
        }

        mInProgressTasks.push_front(taskId);
        TaskValuePtr cacheV = std::make_shared<TaskValue>(std::vector<std::size_t>{}, TaskLayerModuleConfigListPtr(),
            mInProgressTasks.begin(), true, false, false, true);
        mCacheMap.try_emplace(taskId, std::move(cacheV));
        TLLM_LOG_DEBUG("%s return mCacheMap.at(taskId)", __PRETTY_FUNCTION__);
        return mCacheMap.at(taskId);
    }();
    if (!taskValuePtr)
    {
        TLLM_LOG_DEBUG("%s return", __PRETTY_FUNCTION__);
        return;
    }
    auto taskValue = taskValuePtr.value();

    TensorPtr config = sourceConfig->getShape().nbDims == 2
        ? sourceConfig
        : ITensor::view(
            sourceConfig, ITensor::makeShape({sourceConfig->getShape().d[1], sourceConfig->getShape().d[2]}));

    TensorPtr weights = sourceWeights->getShape().nbDims == 2
        ? sourceWeights
        : ITensor::view(
            sourceWeights, ITensor::makeShape({sourceWeights->getShape().d[1], sourceWeights->getShape().d[2]}));

    auto neededPages = determineNumPages(config);
    std::vector<size_t> pageIds{};
    try
    {
        pageIds = claimPagesWithEvict(neededPages);
    }
    catch (std::runtime_error& e)
    {
        std::lock_guard<std::mutex> lk(mCacheMutex);
        mInProgressTasks.erase(taskValue->it);
        mCacheMap.erase(taskId);
        throw e;
    }

    taskValue->pageIds = std::move(pageIds);
    {
        std::lock_guard<std::mutex> lk(mCacheMutex);
        taskValue->loadInProgress = false;
    }

    if (load)
    {
        loadWeights(*taskValue, weights, config);
    }

    bool isDone;
    {
        std::lock_guard<std::mutex> lk(mCacheMutex);
        isDone = taskValue->done;
    }
    if (isDone)
    {
        markTaskDone(taskId);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void LoraCache::loadWeights(TaskIdType taskId, TensorPtr sourceWeights, TensorPtr sourceConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto taskValuePtr = [&]() -> std::optional<TaskValuePtr>
    {
        std::lock_guard<std::mutex> cacheLock(mCacheMutex);
        auto taskStatus = getStatus(taskId);
        if (kVALUE_STATUS_MISSING == taskStatus)
        {
            throw std::runtime_error("task " + std::to_string(taskId) + " has not been added to cache. call put first");
        }
        else if (kVALUE_STATUS_LOADED == taskStatus)
        {
            return std::nullopt;
        }

        auto taskValue = mCacheMap.at(taskId);
        if (taskValue->loadInProgress)
        {
            return std::nullopt;
        }
        taskValue->loadInProgress = true;
        return taskValue;
    }();
    if (!taskValuePtr)
    {
        return;
    }
    auto taskValue = taskValuePtr.value();

    TensorPtr config = sourceConfig->getShape().nbDims == 2
        ? sourceConfig
        : ITensor::view(
            sourceConfig, ITensor::makeShape({sourceConfig->getShape().d[1], sourceConfig->getShape().d[2]}));

    TensorPtr weights = sourceWeights->getShape().nbDims == 2
        ? sourceWeights
        : ITensor::view(
            sourceWeights, ITensor::makeShape({sourceWeights->getShape().d[1], sourceWeights->getShape().d[2]}));

    loadWeights(*taskValue, weights, config);

    bool isDone;
    {
        std::lock_guard<std::mutex> lk(mCacheMutex);
        isDone = taskValue->done;
    }
    if (isDone)
    {
        markTaskDone(taskId);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void LoraCache::loadWeights(TaskValue& taskValue, TensorPtr weights, TensorPtr config)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    std::vector<TensorPtr> pagePtrs{};
    pagePtrs.reserve(taskValue.pageIds.size());
    for (auto id : taskValue.pageIds)
    {
        pagePtrs.push_back(mCachePageManager->mutablePagePtr(id));
    }

    taskValue.configs = std::make_shared<std::vector<TaskLayerModuleConfig>>(copyToPages(
        weights, config, mModelConfig, mWorldConfig, mModuleIdToModule, *mBufferManager, pagePtrs, taskValue.pageIds));
    {
        std::lock_guard<std::mutex> lk(mCacheMutex);
        taskValue.loadInProgress = false;
        taskValue.loaded = true;
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

std::vector<std::size_t> LoraCache::claimPagesWithEvict(SizeType32 numPages)
{
    TLLM_LOG_DEBUG(
        "%s start, mPagedManagerConfig: %s", __PRETTY_FUNCTION__, runtime::to_string(mPageManagerConfig).c_str());
    TLLM_LOG_DEBUG("trying to claim " + std::to_string(numPages));
    std::lock_guard<std::mutex> pageLock(mPagesMutex);
    auto const availablePages = mCachePageManager->numAvailablePages();
    if (numPages <= availablePages)
    {
        auto pageIds = mCachePageManager->claimPages(numPages);
        TLLM_CHECK(pageIds.has_value());
        return pageIds.value();
    }

    std::lock_guard<std::mutex> cacheLock(mCacheMutex);
    std::vector<size_t> pageIdsToEvict;
    std::vector<uint64_t> taskIdsToEvict;
    auto neededPages = numPages - availablePages;
    auto it = mDoneTasks.rbegin();
    for (auto it = mDoneTasks.rbegin(); it != mDoneTasks.rend() && neededPages > 0; it = std::next(it))
    {
        auto const taskId = *it;
        taskIdsToEvict.push_back(taskId);
        auto const& taskValue = *(mCacheMap.at(taskId));
        pageIdsToEvict.insert(pageIdsToEvict.end(), taskValue.pageIds.begin(), taskValue.pageIds.end());
        neededPages -= taskValue.pageIds.size();
    }
    if (it == mDoneTasks.rend())
    {
        throw LoraCacheFullException("Cache is full. There are no done tasks to evict");
    }

    TLLM_LOG_DEBUG("evicting " + std::to_string(taskIdsToEvict.size()));
    for (size_t i = 0; i < taskIdsToEvict.size(); ++i)
    {

        TLLM_LOG_DEBUG("evicting taskId" + std::to_string(taskIdsToEvict.at(i)));
        mDoneTasks.pop_back();
        mCacheMap.erase(taskIdsToEvict.at(i));
    }
    mCachePageManager->releasePages(pageIdsToEvict);
    auto pageIds = mCachePageManager->claimPages(numPages);
    TLLM_CHECK(pageIds.has_value());
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return pageIds.value();
}

void LoraCache::markTaskDone(TaskIdType taskId)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_DEBUG("markTaskDone " + std::to_string(taskId));
    std::lock_guard<std::mutex> lock(mCacheMutex);
    if (mCacheMap.find(taskId) == mCacheMap.end())
    {
        return;
    }
    auto& taskValue = *(mCacheMap.at(taskId));
    bool inProgress = taskValue.inProgress;
    bool loaded = taskValue.loaded;
    if (inProgress)
    {
        if (loaded)
        {
            mInProgressTasks.erase(taskValue.it);
            mDoneTasks.push_front(taskId);
            taskValue.it = mDoneTasks.begin();
            taskValue.inProgress = false;
        }
    }
    taskValue.done = true;
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void LoraCache::markAllDone()
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    std::lock_guard<std::mutex> lock(mCacheMutex);
    for (auto it = mInProgressTasks.rbegin(), nit = it; it != mInProgressTasks.rend(); it = nit)
    {
        nit = std::next(it);
        auto taskId = *it;
        auto& taskValue = *(mCacheMap.at(*it));
        bool inProgress = taskValue.inProgress;
        bool loaded = taskValue.loaded;
        if (inProgress && loaded)
        {
            nit = decltype(it){mInProgressTasks.erase(taskValue.it)};
            mDoneTasks.push_front(taskId);
            taskValue.it = mDoneTasks.begin();
            taskValue.inProgress = false;
        }
        taskValue.done = true;
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

std::vector<LoraCache::TaskLayerModuleConfig> const& LoraCache::get(TaskIdType taskId)
{
    std::lock_guard<std::mutex> lock(mCacheMutex);
    if (kVALUE_STATUS_LOADED != getStatus(taskId))
    {
        throw std::runtime_error("taskid not loaded");
    }

    bumpTaskInProgress(taskId);
    return *mCacheMap.at(taskId)->configs;
}

void LoraCache::bump(TaskIdType taskId)
{
    std::lock_guard<std::mutex> lk(mCacheMutex);
    bumpTaskInProgress(taskId);
}

void LoraCache::bumpTaskInProgress(TaskIdType taskId)
{
    auto it = mCacheMap.find(taskId);
    if (it != mCacheMap.end())
    {
        auto& taskValue = *(it->second);
        if (taskValue.inProgress)
        {
            mInProgressTasks.erase(taskValue.it);
        }
        else
        {
            mDoneTasks.erase(taskValue.it);
        }
        mInProgressTasks.push_front(taskId);
        taskValue.it = mInProgressTasks.begin();
        taskValue.inProgress = true;
        taskValue.done = false;
    }
}

LoraCache::ValueStatus LoraCache::getStatus(TaskIdType taskId) const
{
    auto it = mCacheMap.find(taskId);
    if (it != mCacheMap.end())
    {
        return it->second->loaded ? kVALUE_STATUS_LOADED : kVALUE_STATUS_PROCESSING;
    }
    return kVALUE_STATUS_MISSING;
}

SizeType32 LoraCache::determineNumPages(TaskIdType taskId) const
{
    std::lock_guard<std::mutex> lk(mCacheMutex);
    if (kVALUE_STATUS_MISSING == getStatus(taskId))
    {
        throw std::runtime_error("task " + std::to_string(taskId) + " not found in cache call put first");
    }

    return mCacheMap.at(taskId)->pageIds.size();
}

SizeType32 LoraCache::determineNumPages(TensorPtr loraConfig) const
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const localNumLayers = mModelConfig.getNbAttentionLayers(
        mWorldConfig.getPipelineParallelism(), mWorldConfig.getPipelineParallelRank());
    auto const firstLayerId = mWorldConfig.getPipelineParallelRank() * localNumLayers;
    auto const lastLayerId = firstLayerId + localNumLayers;

    SizeType32 currPage = 0;
    SizeType32 currSlot = 0;
    SizeType32 const slotsPerPage = mPageManagerConfig.getSlotsPerPage();
    SizeType32 const pageWidth = mPageManagerConfig.getPageWidth();
    for (SizeType32 row = 0; row < loraConfig->getShape().d[0]; ++row)
    {
        lora::LoraModuleConfig const config(ITensor::slice(loraConfig, row, 1));
        auto const layerId = config.layerId;
        if (layerId >= firstLayerId && layerId < lastLayerId)
        {
            auto const adapterSize = config.adapterSize;
            bool const isDora = config.isDora;
            auto const& module = mModuleIdToModule.at(config.moduleId);
            auto const localSize = module.localTotalSize(adapterSize, mWorldConfig.getTensorParallelism(), isDora);
            auto const numSlots = common::ceilDiv(localSize, pageWidth);
            if (numSlots + currSlot > slotsPerPage)
            {
                currSlot = 0;
                ++currPage;
            }

            currSlot += numSlots;
        }
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return currPage + 1;
}

LoraCache::LoraCache(LoraCachePageManagerConfig const& pageManagerConfig, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, BufferManager const& bufferManager)
    : mPageManagerConfig(pageManagerConfig)
    , mModelConfig(modelConfig)
    , mWorldConfig(worldConfig)
{
    mCachePageManager = std::make_unique<LoraCachePageManager>(mPageManagerConfig, bufferManager);

    auto modules = modelConfig.getLoraModules();
    for (auto const& m : modules)
    {
        mModuleIdToModule[m.value()] = m;
    }

    mBufferManager = std::make_unique<BufferManager>(std::make_shared<CudaStream>());

    for (size_t i = 0; i < static_cast<size_t>(mPageManagerConfig.getNumCopyStreams()); ++i)
    {
        mDeviceBufferManagers.push_back(std::make_unique<BufferManager>(std::make_shared<CudaStream>()));
    }
}

template <typename T>
void LoraCache::splitTransposeCpuInner(ITensor& output, ITensor const& input, SizeType32 tpSize, SizeType32 tpRank)
{
    auto const adapterSize = input.getShape().d[0];
    auto const hiddenSize = input.getShape().d[1];
    auto const splitHiddenSize = static_cast<SizeType32>(hiddenSize / tpSize);

    auto outputPtr = bufferCast<T>(output);
    auto const inputPtr = bufferCast<T>(input);

    for (SizeType32 adapterIdx = 0; adapterIdx < adapterSize; ++adapterIdx)
    {
        for (SizeType32 hiddenIdx = 0; hiddenIdx < splitHiddenSize; ++hiddenIdx)
        {
            auto outputIdx = common::flat_index2(adapterIdx, hiddenIdx, splitHiddenSize);
            auto inputIdx = common::flat_index2(adapterIdx, hiddenIdx + tpRank * splitHiddenSize, hiddenSize);
            outputPtr[outputIdx] = inputPtr[inputIdx];
        }
    }
}

void LoraCache::splitTransposeCpu(ITensor& output, ITensor const& input, SizeType32 tpSize, SizeType32 tpRank)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    switch (input.getDataType())
    {
    case nvinfer1::DataType::kINT32: splitTransposeCpuInner<SizeType32>(output, input, tpSize, tpRank); break;
    case nvinfer1::DataType::kFLOAT: splitTransposeCpuInner<float>(output, input, tpSize, tpRank); break;
    case nvinfer1::DataType::kHALF: splitTransposeCpuInner<half>(output, input, tpSize, tpRank); break;
    case nvinfer1::DataType::kINT8: splitTransposeCpuInner<int8_t>(output, input, tpSize, tpRank); break;
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8: splitTransposeCpuInner<__nv_fp8_e4m3>(output, input, tpSize, tpRank); break;
#endif // ENABLE_FP8
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: splitTransposeCpuInner<__nv_bfloat16>(output, input, tpSize, tpRank); break;
#endif // ENABLE_BF16
    default: TLLM_CHECK_WITH_INFO(false, "data type not supported");
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

std::vector<LoraCache::TaskLayerModuleConfig> LoraCache::copyToPages(TensorPtr sourceWeights, TensorPtr sourceConfig,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    std::unordered_map<SizeType32, LoraModule> moduleIdToModule, BufferManager const& manager,
    std::vector<TensorPtr> const& pages, std::vector<std::size_t> const& pageIds)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(!pages.empty(), "empty pages");

    TensorPtr weights = sourceWeights->getShape().nbDims == 2
        ? sourceWeights
        : ITensor::view(
            sourceWeights, ITensor::makeShape({sourceWeights->getShape().d[1], sourceWeights->getShape().d[2]}));

    TensorPtr config = sourceConfig->getShape().nbDims == 2
        ? sourceConfig
        : ITensor::view(
            sourceConfig, ITensor::makeShape({sourceConfig->getShape().d[1], sourceConfig->getShape().d[2]}));

    TLLM_CHECK(pages[0]->getShape().nbDims == 2);
    auto const slotsPerPage = pages[0]->getShape().d[0];
    auto const pageWidth = pages[0]->getShape().d[1];

    auto const tpSize = worldConfig.getTensorParallelism();
    auto const tpRank = worldConfig.getTensorParallelRank();
    auto const ppSize = worldConfig.getPipelineParallelism();
    auto const ppRank = worldConfig.getPipelineParallelRank();
    // TODO(oargov): why *attention* layers?
    auto const localNumLayers = modelConfig.getNbAttentionLayers(ppSize, ppRank);
    auto const firstLayerId = ppRank * localNumLayers;
    auto const lastLayerId = firstLayerId + localNumLayers;

    SizeType32 currPage = 0;
    SizeType32 currSlot = 0;

    std::vector<SizeType32> rowPage;
    std::vector<SizeType32> rowSlot;
    std::vector<SizeType32> rowIndices;

    auto const numRows = config->getShape().d[0];
    for (SizeType32 row = 0; row < numRows; ++row)
    {
        lora::LoraModuleConfig const loraConfig(ITensor::slice(config, row, 1));
        auto const layerId = loraConfig.layerId;
        if (layerId >= firstLayerId && layerId < lastLayerId)
        {
            auto const adapterSize = loraConfig.adapterSize;
            bool const isDora = loraConfig.isDora;

            auto const modId = loraConfig.moduleId;
            auto const& module = moduleIdToModule.at(modId);
            auto const localSize = module.localTotalSize(adapterSize, tpSize, isDora);
            auto const rowSlots = common::ceilDiv(localSize, pageWidth);
            if (currSlot + rowSlots > slotsPerPage)
            {
                currSlot = 0;
                ++currPage;
            }

            rowIndices.push_back(row);
            rowSlot.push_back(currSlot);
            rowPage.push_back(currPage);
            currSlot += rowSlots;
        }
    }

    std::vector<LoraCache::TaskLayerModuleConfig> pageLocations(rowIndices.size());
    for (SizeType32 i = 0; i < static_cast<SizeType32>(rowIndices.size()); ++i)
    {
        auto copyFn = [i = i, &rowIndices, &rowPage, &rowSlot, &pageLocations, weights, config, &pages,
                          &moduleIdToModule, &manager, pageWidth, tpSize, tpRank, pageIds]()
        {
            auto const row = rowIndices[i];
            auto const currPage = rowPage[i];
            auto const currSlot = rowSlot[i];

            lora::LoraModuleConfig const loraConfig(ITensor::slice(config, row, 1));

            auto const layerId = loraConfig.layerId;
            auto const adapterSize = loraConfig.adapterSize;

            bool const isDora = loraConfig.isDora;
            auto const modId = loraConfig.moduleId;

            auto const& module = moduleIdToModule.at(modId);
            auto const localSize = module.localTotalSize(adapterSize, tpSize, isDora);
            auto const rowSlots = common::ceilDiv(localSize, pageWidth);

            auto const inDim = module.inDim();
            auto const outDim = module.outDim();
            auto const localOutDim = module.localOutDim(tpSize);
            auto const inSize = module.inSize(adapterSize);
            auto const outSize = module.outSize(adapterSize);
            auto const localInSize = module.localInSize(adapterSize, tpSize);
            auto const localOutSize = module.localOutSize(adapterSize, tpSize);
            auto const magnitudeVecSize = isDora ? outDim : 0;
            auto const localMagnitudeVecSize = module.localScalesSize(tpSize, isDora);

            TLLM_CHECK(module.inDimFirst() == false);
            TLLM_CHECK(module.outDimFirst() == true);
            TLLM_CHECK(module.inTpSplitDim() == 1 || module.inTpSplitDim() == -1);
            TLLM_CHECK(module.outTpSplitDim() == 0 || module.outTpSplitDim() == -1);

            auto const splitIn = module.inTpSplitDim() == 1;
            auto const splitOut = module.outTpSplitDim() == 0;

            TensorPtr rowWeights = ITensor::view(
                ITensor::slice(weights, row, 1), ITensor::makeShape({inSize + outSize + magnitudeVecSize}));
            TensorPtr weightsIn
                = ITensor::view(ITensor::slice(rowWeights, 0, inSize), ITensor::makeShape({adapterSize, inDim}));
            TensorPtr weightsOut
                = ITensor::view(ITensor::slice(rowWeights, inSize, outSize), ITensor::makeShape({outDim, adapterSize}));
            TensorPtr magnitudeVec = ITensor::view(
                ITensor::slice(rowWeights, inSize + outSize, magnitudeVecSize), ITensor::makeShape({magnitudeVecSize}));

            TensorPtr pageSlice = ITensor::slice(pages.at(currPage), currSlot, rowSlots);
            SizeType32 pageSliceSize = ITensor::volume(pageSlice->getShape());
            TensorPtr pageFlatView = ITensor::view(pageSlice, ITensor::makeShape({pageSliceSize}));
            TensorPtr targetWeightsIn = ITensor::slice(pageFlatView, 0, localInSize);
            TensorPtr targetWeightsOut = ITensor::slice(pageFlatView, localInSize, localOutSize);
            TensorPtr targetMagnitudeVec
                = ITensor::slice(pageFlatView, localInSize + localOutSize, localMagnitudeVecSize);

            if (!splitIn)
            {
                manager.copy(*weightsIn, *targetWeightsIn);
            }
            else
            {
                splitTransposeCpu(*targetWeightsIn, *weightsIn, tpSize, tpRank);
            }

            if (!splitOut)
            {
                manager.copy(*weightsOut, *targetWeightsOut);
                if (isDora)
                {
                    manager.copy(*magnitudeVec, *targetMagnitudeVec);
                }
            }
            else
            {
                TensorPtr source = ITensor::view(
                    ITensor::slice(
                        ITensor::view(weightsOut, ITensor::makeShape({tpSize, localOutDim, adapterSize})), tpRank, 1),
                    ITensor::makeShape({localOutDim, adapterSize}));
                manager.copy(*source, *targetWeightsOut);
                if (isDora)
                {
                    TensorPtr magSource = ITensor::view(
                        ITensor::slice(ITensor::view(magnitudeVec, ITensor::makeShape({tpSize, localMagnitudeVecSize})),
                            tpRank, 1),
                        ITensor::makeShape({localMagnitudeVecSize}));
                    manager.copy(*magSource, *targetMagnitudeVec);
                }
            }

            pageLocations[i] = LoraCache::TaskLayerModuleConfig{pageIds.at(currPage), currSlot, localInSize,
                localOutSize, modId, layerId, adapterSize, static_cast<SizeType32>(rowSlots),
                reinterpret_cast<std::int64_t>(targetWeightsIn->data()),
                reinterpret_cast<std::int64_t>(targetWeightsOut->data()),
                isDora ? std::optional<std::int64_t>(reinterpret_cast<std::int64_t>(targetMagnitudeVec->data()))
                       : std::nullopt};
        };
        copyFn();
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return pageLocations;
}

std::map<size_t, std::pair<size_t, SizeType32>> LoraCache::copyTaskMapPages(TaskValue& targetTaskValue,
    TaskValue const& sourceTaskValue, std::vector<size_t> const& targetPageIds, LoraCache const& targetCache)
{
    auto const& pageIds = sourceTaskValue.pageIds;

    // collect mapping from oldPageId to (newPageId, num used slots in page)
    std::map<size_t, std::pair<size_t, SizeType32>> oldToNewPageIds{};
    for (size_t i = 0; i < pageIds.size(); ++i)
    {
        oldToNewPageIds.insert_or_assign(pageIds[i], std::make_pair(targetPageIds[i], 0));
    }

    targetTaskValue.configs = std::make_shared<std::vector<TaskLayerModuleConfig>>(*sourceTaskValue.configs);
    targetTaskValue.pageIds = targetPageIds;
    for (size_t i = 0; i < sourceTaskValue.configs->size(); ++i)
    {
        auto const& sourceConfigs = *(sourceTaskValue.configs);
        auto& targetConfigs = *(targetTaskValue.configs);
        auto& newPagePair = oldToNewPageIds.at(sourceConfigs[i].pageId);
        newPagePair.second += sourceConfigs[i].numSlots;
        targetConfigs[i].pageId = newPagePair.first;
        bool const isDora = sourceConfigs[i].scalingVecPointer.has_value();
        auto page = targetCache.mCachePageManager->mutablePagePtr(targetConfigs[i].pageId);
        auto const slotId = targetConfigs[i].slotIdx;
        auto const numSlots = targetConfigs[i].numSlots;
        auto const inSize = targetConfigs[i].inSize;
        auto const outSize = targetConfigs[i].outSize;
        auto const outDim = outSize / targetConfigs[i].adapterSize;
        TensorPtr slot = ITensor::view(ITensor::slice(page, slotId, numSlots),
            ITensor::makeShape({numSlots * targetCache.mPageManagerConfig.getPageWidth()}));
        targetConfigs[i].weightsInPointer = reinterpret_cast<std::int64_t>(
            ITensor::view(ITensor::slice(slot, 0, inSize), ITensor::makeShape({inSize}))->data());
        targetConfigs[i].weightsOutPointer = reinterpret_cast<std::int64_t>(
            ITensor::view(ITensor::slice(slot, inSize, outSize), ITensor::makeShape({outSize}))->data());
        targetConfigs[i].scalingVecPointer = isDora
            ? std::optional<std::int64_t>(reinterpret_cast<std::int64_t>(
                ITensor::view(ITensor::slice(slot, inSize + outSize, outDim), ITensor::makeShape({outDim}))->data()))
            : std::nullopt;
    }

    return oldToNewPageIds;
}

void LoraCache::copyTask(TaskIdType taskId, LoraCache& deviceCache, bool markDone)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_DEBUG("copyTask " + std::to_string(taskId));

    TLLM_CHECK_WITH_INFO(deviceCache.mPageManagerConfig.getMemoryType() == runtime::MemoryType::kGPU
            && !deviceCache.mDeviceBufferManagers.empty(),
        "The deviceCache must hold GPU memory and have at least one bufferManager / copy stream");

    // First get the taskValue from this cache
    // TaskValue& taskValue = copyTaskGetThisTaskValue(taskId);
    TaskValuePtr taskValue = [&]() -> TaskValuePtr
    {
        std::lock_guard<std::mutex> cacheLock(mCacheMutex);
        auto status = getStatus(taskId);
        if (kVALUE_STATUS_PROCESSING == status)
        {
            throw std::runtime_error("can't move a processing task taskId=" + std::to_string(taskId));
        }
        else if (status == kVALUE_STATUS_MISSING)
        {
            throw std::runtime_error("can't move a missing task" + std::to_string(taskId));
        }
        auto taskValue = mCacheMap.at(taskId);
        // mark task unloaded so we can evict the task while the copy in in progress
        taskValue->loaded = false;
        bumpTaskInProgress(taskId);
        return taskValue;
    }();

    auto& pageIds = taskValue->pageIds;
    auto neededPages = pageIds.size();

    // Now create put the task in the target cache
    // TaskValue* otherTaskValuePtr = copyTaskGetOtherTaskValue(taskId, taskValue, deviceCache, markDone);
    std::optional<TaskValuePtr> optOtherTaskValuePtr = [&]() -> std::optional<TaskValuePtr>
    {
        std::lock_guard<std::mutex> deviceCacheLock(deviceCache.mCacheMutex);
        auto otherStatus = deviceCache.getStatus(taskId);
        if (kVALUE_STATUS_MISSING != otherStatus)
        {
            deviceCache.bumpTaskInProgress(taskId);
            taskValue->loaded = true;
            return std::nullopt;
        }

        deviceCache.mInProgressTasks.push_front(taskId);
        auto cacheV = std::make_shared<TaskValue>(std::vector<std::size_t>{}, TaskLayerModuleConfigListPtr(),
            deviceCache.mInProgressTasks.begin(), true, false, markDone, true);
        deviceCache.mCacheMap.try_emplace(taskId, std::move(cacheV));
        auto otherTaskValue = deviceCache.mCacheMap.at(taskId);
        // TODO (grclark) return shared_ptr
        return otherTaskValue;
    }();
    if (!optOtherTaskValuePtr)
    {
        return;
    }
    TaskValuePtr otherTaskValue = optOtherTaskValuePtr.value();

    std::vector<size_t> newPageIds{};
    try
    {
        newPageIds = deviceCache.claimPagesWithEvict(neededPages);
    }
    catch (std::runtime_error& e)
    {
        {
            std::lock_guard<std::mutex> lk(deviceCache.mCacheMutex);
            deviceCache.mInProgressTasks.erase(otherTaskValue->it);
            deviceCache.mCacheMap.erase(taskId);
            taskValue->loaded = true;
            throw std::runtime_error("Couldn't claim pages during copyTask -- " + std::string(e.what()));
        }
    }

    auto oldToNewPageIds = copyTaskMapPages(*otherTaskValue, *taskValue, newPageIds, deviceCache);

    auto const flatPageShape
        = ITensor::makeShape({mPageManagerConfig.getPageWidth() * mPageManagerConfig.getSlotsPerPage()});
    size_t bufferManagerOffset = taskId % deviceCache.mDeviceBufferManagers.size();
    std::vector<CudaEvent> copyEvents(otherTaskValue->pageIds.size());
    size_t eventIdx = 0;
    for (auto const& [oldPageId, newPagePair] : oldToNewPageIds)
    {
        auto const newPageId = newPagePair.first;
        auto const copySize = newPagePair.second * mPageManagerConfig.getPageWidth();
        auto const copyShape = ITensor::makeShape({copySize});
        TLLM_LOG_DEBUG("copy page (task " + std::to_string(taskId) + ") " + std::to_string(oldPageId) + " -> "
            + std::to_string(newPageId) + " size: " + std::to_string(copySize));
        TensorPtr oldPagePtr = mCachePageManager->mutablePagePtr(oldPageId);
        TensorPtr newPagePtr = deviceCache.mCachePageManager->mutablePagePtr(newPageId);
        TensorPtr source
            = ITensor::view(ITensor::slice(ITensor::view(oldPagePtr, flatPageShape), 0, copySize), copyShape);
        TensorPtr dest
            = ITensor::view(ITensor::slice(ITensor::view(newPagePtr, flatPageShape), 0, copySize), copyShape);
        deviceCache.mDeviceBufferManagers[bufferManagerOffset]->copy(*source, *dest);
        deviceCache.mDeviceBufferManagers[bufferManagerOffset]->getStream().record(copyEvents[eventIdx++]);
        bufferManagerOffset = (bufferManagerOffset + 1) % deviceCache.mDeviceBufferManagers.size();
    }
    for (auto const& event : copyEvents)
    {
        event.synchronize();
    }

    bool otherIsDone;
    {
        std::lock_guard<std::mutex> lk(mCacheMutex);
        otherIsDone = otherTaskValue->done;
        otherTaskValue->loadInProgress = false;
        otherTaskValue->loaded = true;
    }
    if (otherIsDone)
    {
        deviceCache.markTaskDone(taskId);
    }

    bool isDone;
    {
        std::lock_guard<std::mutex> lk(mCacheMutex);
        isDone = taskValue->done;
        taskValue->loaded = true;
    }
    if (isDone)
    {
        markTaskDone(taskId);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

ITensor::SharedConstPtr LoraCache::getPagePtr(size_t pageId) const
{
    return mCachePageManager->pagePtr(pageId);
}

SizeType32 LoraCache::getNumPages() const
{
    return mPageManagerConfig.getTotalNumPages();
}

bool LoraCache::fits(TensorPtr config) const
{
    auto const neededPages = determineNumPages(config);
    SizeType32 availablePages;
    {
        std::lock_guard<std::mutex> lk(mPagesMutex);
        availablePages = mCachePageManager->numAvailablePages();
    }
    return neededPages < availablePages;
}

std::string to_string(LoraCache::TaskLayerModuleConfig const& v)
{
    std::stringstream sstream;
    sstream << "{pageIdx=" << v.pageId << "; "
            << "slotIdx=" << v.slotIdx << "; "
            << "inSize=" << v.inSize << "; "
            << "outSize=" << v.outSize << "; "
            << "moduleId=" << v.moduleId << "; "
            << "layerId=" << v.layerId << "; "
            << "adapterSize=" << v.adapterSize << "; "
            << "numSlots=" << v.numSlots << "}";
    return sstream.str();
}

std::ostream& operator<<(std::ostream& os, LoraCache::TaskLayerModuleConfig const& v)
{
    os << to_string(v);
    return os;
}

bool LoraCache::TaskLayerModuleConfig::operator==(LoraCache::TaskLayerModuleConfig const& o) const
{
    return (pageId == o.pageId && slotIdx == o.slotIdx && inSize == o.inSize && outSize == o.outSize
        && moduleId == o.moduleId && layerId == o.layerId && adapterSize == o.adapterSize && numSlots == o.numSlots);
}

bool LoraCache::isDone(TaskIdType taskId) const
{
    std::lock_guard<std::mutex> lk(mCacheMutex);
    if (mCacheMap.count(taskId))
    {
        auto const taskValue = mCacheMap.at(taskId);
        return !taskValue->inProgress;
    }
    return false;
}
} // namespace tensorrt_llm::runtime

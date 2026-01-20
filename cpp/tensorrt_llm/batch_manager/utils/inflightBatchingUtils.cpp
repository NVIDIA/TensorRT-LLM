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

#include "inflightBatchingUtils.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

namespace tensorrt_llm::batch_manager::utils
{
using ITensor = runtime::ITensor;

TensorPtr collectRequestIds(RequestVector const& contextRequests, RequestVector const& generationRequests)
{
    auto const numRequests = static_cast<ITensor::DimType64>(contextRequests.size() + generationRequests.size());
    auto requestIds
        = runtime::BufferManager::cpu(ITensor::makeShape({numRequests}), runtime::TRTDataType<RequestIdType>::value);
    auto requestIdsRange = runtime::BufferRange<RequestIdType>(*requestIds);
    auto batchIdx{0};
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& request : requests)
        {
            requestIdsRange[batchIdx++] = request->mRequestId;
        }
    }
    return requestIds;
}

void sortRequests(RequestVector& contextRequests, RequestVector& generationRequests, bool chunksPresent)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto sortByLoraId = [](RequestVector::iterator begin, RequestVector::iterator end)
    {
        std::sort(
            begin, end, [](auto const& lhs, auto const& rhs) { return lhs->getLoraTaskId() < rhs->getLoraTaskId(); });
    };

    if (chunksPresent)
    {
        // Move context requests that reached the last context chunk to the end of the vector.
        // This order is required for moveFinishedContextRequestsToGeneration.
        auto firstFinished = std::partition(contextRequests.begin(), contextRequests.end(),
            [](auto const& llmReq) { return !llmReq->isLastContextChunk(); });

        // Sort context requests by lora task id, but keep finished requests separate.
        sortByLoraId(contextRequests.begin(), firstFinished);
        sortByLoraId(firstFinished, contextRequests.end());
    }
    else
    {
        sortByLoraId(contextRequests.begin(), contextRequests.end());
    }
    sortByLoraId(generationRequests.begin(), generationRequests.end());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void moveFinishedContextRequestsToGeneration(ScheduledRequests& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& contextRequests = scheduledRequests.contextRequests;
    auto& generationRequests = scheduledRequests.generationRequests;
    auto firstFinished = std::find_if(
        contextRequests.begin(), contextRequests.end(), [](auto const& llmReq) { return llmReq->isContextFinished(); });
    TLLM_LOG_DEBUG(
        "Found %ld unfinished chunked context requests. Found %ld finished context requests, moving them to "
        "generation.",
        std::distance(contextRequests.begin(), firstFinished), std::distance(firstFinished, contextRequests.end()));
    generationRequests.insert(generationRequests.begin(), std::make_move_iterator(firstFinished),
        std::make_move_iterator(contextRequests.end()));
    contextRequests.erase(firstFinished, contextRequests.end());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void copyGenerationLogits(RuntimeBuffers::GenerationLogitsCache& generationLogitsCache,
    runtime::BufferManager const& bufferManager, LlmRequest& llmReq, bool beforeDecoder,
    std::vector<SizeType32> const& numDroppedTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(
        !beforeDecoder || numDroppedTokens.empty(), "numDroppedTokens are only possible after decoder.");

    auto const reqBeamWidth = llmReq.getBeamWidthByIter();
    TLLM_CHECK_WITH_INFO(numDroppedTokens.empty() || numDroppedTokens.size() == static_cast<size_t>(reqBeamWidth),
        "Dropped tokens have to be defined for all beams.");

    auto const fragmentSize = llmReq.getGenerationLogitsFragmentsSize();

    // Merge logits fragments on device
    auto const& transposeBufferPtr = generationLogitsCache.transposedLogits;
    auto const& cachePointerDevice = generationLogitsCache.fragmentPointerDevice;
    auto const& cachePointerHost = generationLogitsCache.getFragmentPointerHost();
    tensorrt_llm::runtime::kernels::mergeLogitsFragments(bufferManager, *transposeBufferPtr,
        llmReq.getGenerationLogitsFragments(), *cachePointerDevice, *cachePointerHost, 0, 1, reqBeamWidth,
        bufferManager.getStream(), 0);
    llmReq.clearGenerationLogitsFragments();

    // Copy logits to host
    for (SizeType32 beam = 0; beam < reqBeamWidth; beam++)
    {
        auto const droppedSize = !numDroppedTokens.empty() ? numDroppedTokens.at(beam) : 0;
        // Ignore logits of dropped tokens
        auto const beamFragmentSize = fragmentSize - droppedSize;
        // If this function is called before the decoder, the request does not contain the generated token of the
        // current iteration, so we add 1 to the number of tokens.
        auto const numGenerationToken
            = static_cast<SizeType32>(beforeDecoder) + llmReq.getNumTokens(beam) - llmReq.mPromptLen;
        auto const hostOffset = numGenerationToken - beamFragmentSize;

        // [beamWidth, GENERATION_LOGITS_BUFFER_LENGTH, vocabSizePadded] -> [beamFragmentSize, vocabSizePadded]
        auto beamDeviceTensorPtr = ITensor::slice(transposeBufferPtr, {beam, 0}, beamFragmentSize);
        // [beamWidth, mMaxNewTokens, vocabSizePadded] -> [beamFragmentSize, vocabSizePadded]
        auto beamHostTensorPtr = ITensor::slice(llmReq.getGenerationLogitsHost(), {beam, hostOffset}, beamFragmentSize);
        bufferManager.copy(*beamDeviceTensorPtr, *beamHostTensorPtr);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

namespace
{

std::pair<TensorPtr const&, bool> findOutputTensor(std::string const& outputTensorName,
    std::vector<executor::AdditionalModelOutput> const& additionalModelOutputs,
    RuntimeBuffers::TensorMap const& outputMap, bool isContext)
{
    auto const aoIter = std::find_if(additionalModelOutputs.cbegin(), additionalModelOutputs.cend(),
        [&outputTensorName](auto const& ao) { return ao.name == outputTensorName; });
    TLLM_CHECK_WITH_INFO(aoIter != additionalModelOutputs.cend(), "Additional %s output tensor not found: %s",
        isContext ? "context" : "generation", outputTensorName.c_str());

    auto const gatherContext = aoIter->gatherContext;
    if (isContext)
    {
        TLLM_CHECK_WITH_INFO(
            gatherContext, "Additional context output tensor not gathered: %s", outputTensorName.c_str());
    }

    auto const tensorIt = outputMap.find(outputTensorName);
    TLLM_CHECK_WITH_INFO(tensorIt != outputMap.end(), "Additional %s output tensor not found: %s",
        isContext ? "context" : "generation", outputTensorName.c_str());

    return {tensorIt->second, gatherContext};
}

} // namespace

void copyAdditionalOutputs(std::vector<executor::AdditionalModelOutput> const& additionalModelOutputs,
    RequestVector const& contextRequests, RequestVector const& generationRequests,
    RuntimeBuffers::TensorMap const& outputMap, runtime::BufferManager const& manager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // One index shared across all output tensors that have gatherContext
    SizeType32 srcTensorIndexWithContext{0};
    // One index shared across all output tensors that do not have gatherContext
    SizeType32 srcTensorIndexWithoutContext{0};

    for (auto const& llmReq : contextRequests)
    {
        auto numContextTokens = llmReq->getContextChunkSize();
        for (auto const& outputTensor : llmReq->getAdditionalContextOutputs())
        {
            auto const& [tensor, gatherContext]
                = findOutputTensor(outputTensor.first, additionalModelOutputs, outputMap, true);

            auto const srcTensorIndex = srcTensorIndexWithContext;
            auto srcView = ITensor::slice(tensor, srcTensorIndex, numContextTokens);
            auto dstView = ITensor::slice(outputTensor.second, llmReq->getContextCurrentPosition(), numContextTokens);
            manager.copy(*srcView, *dstView);
        }
        srcTensorIndexWithContext += numContextTokens;
        srcTensorIndexWithoutContext += 1;

        // Copy output of last token to generation outputs
        if (llmReq->isLastContextChunk())
        {
            for (auto const& outputTensor : llmReq->getAdditionalGenerationOutputs())
            {
                auto const& [tensor, gatherContext]
                    = findOutputTensor(outputTensor.first, additionalModelOutputs, outputMap, false);

                auto const srcTensorIndex = gatherContext ? srcTensorIndexWithContext : srcTensorIndexWithoutContext;
                auto srcView = ITensor::slice(tensor, srcTensorIndex - 1, 1);
                for (SizeType32 beam = 0; beam < llmReq->getBeamWidthByIter(); beam++)
                {
                    auto dstView = ITensor::slice(outputTensor.second, {beam, 0}, 1);
                    manager.copy(*srcView, *dstView);
                }
            }
        }
    }

    for (auto const& llmReq : generationRequests)
    {
        auto const reqBeamWidth = llmReq->getBeamWidthByIter();
        for (auto const& outputTensor : llmReq->getAdditionalGenerationOutputs())
        {
            auto const& [tensor, gatherContext]
                = findOutputTensor(outputTensor.first, additionalModelOutputs, outputMap, false);

            auto const srcTensorIndex = gatherContext ? srcTensorIndexWithContext : srcTensorIndexWithoutContext;
            for (SizeType32 beam = 0; beam < reqBeamWidth; beam++)
            {
                auto const generatedLength = llmReq->getNumTokens(beam) - llmReq->getPromptLen();
                TLLM_CHECK(generatedLength >= 1);
                auto srcView = ITensor::slice(tensor, srcTensorIndex + beam, 1);
                auto dstView = ITensor::slice(outputTensor.second, {beam, generatedLength}, 1);
                manager.copy(*srcView, *dstView);
            }
        }
        srcTensorIndexWithContext += reqBeamWidth;
        srcTensorIndexWithoutContext += reqBeamWidth;
    }

    // Check final indices
    for (auto const& outputTensor : additionalModelOutputs)
    {
        auto const& outputTensorName = outputTensor.name;
        auto const gatherContext = outputTensor.gatherContext;

        auto const tensorIt = outputMap.find(outputTensorName);
        TLLM_CHECK_WITH_INFO(
            tensorIt != outputMap.end(), "Additional output tensor not found: %s", outputTensorName.c_str());

        auto const& outputShape = tensorIt->second->getShape();
        auto const outputSize = outputShape.d[0];
        auto const finalIndex = gatherContext ? srcTensorIndexWithContext : srcTensorIndexWithoutContext;

        TLLM_CHECK_WITH_INFO(finalIndex == outputSize, "Additional %s output tensor final index mismatch %d != %ld: %s",
            gatherContext ? "context" : "generation", finalIndex, outputSize, outputTensorName.c_str());
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void terminateRequest(SequenceSlotManager& seqSlotManager, LlmRequest& llmReq, SizeType32 maxInputLen,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager> peftCacheManager, bool pause)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // If a sequence slot is associated with this request id, free it
    seqSlotManager.freeSequenceSlot(llmReq.mRequestId);
    // Remove the sequence from kvCacheManager
    auto const requestId = llmReq.mRequestId;
    if (kvCacheManager)
    {
        (void) kvCacheManager->removeSequence(requestId, llmReq);
    }
    if (crossKvCacheManager)
    {
        (void) crossKvCacheManager->removeSequence(requestId, llmReq);
    }
    if (pause && !llmReq.isGenerationCompleteState())
    {
        llmReq.pause(maxInputLen);
    }
    else
    {
        TLLM_LOG_DEBUG("terminated: request ID %lu, paused: %d", requestId, pause);
    }

    if (peftCacheManager)
    {
        peftCacheManager->markRequestDone(llmReq, pause);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::vector<SizeType32> getRequestBeamWidths(
    RequestVector const& contextRequests, RequestVector const& generationRequests)
{
    std::vector<SizeType32> beamWidths{};
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            beamWidths.push_back(llmReq->getBeamWidthByIter());
        }
    }
    return beamWidths;
}

void CudaGraphExecutor::create(cudaGraph_t const& graph)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    assert(mInstance == nullptr);
    TLLM_CUDA_CHECK(cudaGraphInstantiate(&mInstance, graph, nullptr, nullptr, 0));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CudaGraphExecutor::uploadToStream(runtime::CudaStream const& stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    assert(hasInstance());
    TLLM_CUDA_CHECK(cudaGraphUpload(mInstance, stream.get()));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CudaGraphExecutor::launch(runtime::CudaStream const& stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CUDA_CHECK(cudaGraphLaunch(mInstance, stream.get()));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

bool CudaGraphExecutor::update(cudaGraph_t const& graph)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    return cudaGraphExecUpdate(mInstance, graph, nullptr) != cudaSuccess;
}

void CudaGraphExecutor::clear()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (mInstance != nullptr)
    {
        TLLM_CUDA_CHECK(cudaGraphExecDestroy(mInstance));
        mInstance = nullptr;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CudaGraphExecutor::prepareNextGraph(std::unique_ptr<runtime::TllmRuntime>& runtime, SizeType32 nextContextId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& stream = runtime->getStream();

    cudaGraph_t nextGraph;
    TLLM_CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    runtime->executeContext(nextContextId);
    TLLM_CUDA_CHECK(cudaStreamEndCapture(stream.get(), &nextGraph));

    if (hasInstance())
    {
        if (update(nextGraph))
        {
            clear();
            create(nextGraph);
        }
    }
    else
    {
        create(nextGraph);
    }

    TLLM_CUDA_CHECK(cudaGraphDestroy(nextGraph));
    uploadToStream(stream);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::optional<std::shared_ptr<CudaGraphExecutor>> CudaGraphExecutorCache::get(BatchState const& state)
{
    auto it = mMap.find(state);
    if (it == mMap.end())
    {
        return std::nullopt;
    }
    mCache.splice(mCache.begin(), mCache, it->second);
    return it->second->second;
}

void CudaGraphExecutorCache::put(BatchState const& state, std::shared_ptr<CudaGraphExecutor> const& value)
{
    auto it = mMap.find(state);
    if (it != mMap.end())
    {
        mCache.erase(it->second);
    }
    mCache.emplace_front(state, value);
    mMap[state] = mCache.begin();

    if (static_cast<runtime::SizeType32>(mMap.size()) > mCapacity)
    {
        auto lastState = mCache.back().first;
        mCache.pop_back();
        mMap.erase(lastState);
    }
}

} // namespace tensorrt_llm::batch_manager::utils

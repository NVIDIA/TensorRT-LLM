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

#include "trtEncoderModel.h"
#include "encoderBuffers.h"
#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/runtimeUtils.h"

#include <algorithm>
#include <cstddef>
#include <vector>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::mpi;

namespace tensorrt_llm::batch_manager
{

TrtEncoderModel::TrtEncoderModel(runtime::ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    runtime::RawEngine const& rawEngine, std::shared_ptr<nvinfer1::ILogger> logger,
    executor::ExecutorConfig const& executorConfig)
    : TrtGptModel(modelConfig, worldConfig, executorConfig)
    , mModelConfig{modelConfig}
    , mWorldConfig{worldConfig}
    , mDevice{runtime::utils::initDevice(worldConfig)}
    , mLogger{logger ? std::move(logger) : std::make_shared<TllmLogger>()}
    , mRuntime{std::make_shared<TllmRuntime>(
          rawEngine, mLogger.get(), executorConfig.getUseGpuDirectStorage(), executorConfig.getGpuWeightsPercent())}
    , mNumMicroBatches{1}
    , mNumBuffers{mNumMicroBatches}
    , mCopyBufferManager{std::make_shared<CudaStream>()}
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(
        !mWorldConfig.isPipelineParallel(), "Pipeline parallelism is currently not supported for encoder models.");

    createRuntimeContexts();

    createBuffers();

    if (mWorldConfig.isPipelineParallel())
    {
        auto const& commSession = COMM_SESSION;
        mMpiCommPipelinePara = std::make_shared<tensorrt_llm::mpi::MpiComm>(
            commSession.split(mWorldConfig.getTensorParallelRank(), mWorldConfig.getPipelineParallelRank()));
    }

    mMicroBatchScheduledRequests.resize(mNumMicroBatches);
    // mEncoderWaitEvents.resize(mNumMicroBatches);

    // set noScheduleUntilState to LlmRequestState::kENCODER_INIT for encoder model
    // when null kv cache manager is given, request scheduler will use MaxRequests as capacity scheduler, i.e. no
    // handling of maximizing utilization or pause/evict
    // TODO: finer control on encoder requests scheduling
    mCapacityScheduler = std::make_unique<tensorrt_llm::batch_manager::CapacityScheduler>(
        getMaxBatchSize() * mNumMicroBatches, executorConfig.getSchedulerConfig().getCapacitySchedulerPolicy(), false,
        false, LlmRequestState::kENCODER_INIT, LlmRequestState::kCONTEXT_INIT);

    mMicroBatchScheduler = std::make_unique<tensorrt_llm::batch_manager::MicroBatchScheduler>(
        std::nullopt, mModelConfig.getMaxInputLen(), LlmRequestState::kENCODER_INIT, LlmRequestState::kCONTEXT_INIT);

    mHiddenSize = modelConfig.getHiddenSize();

    mMaxInputLen = mModelConfig.getMaxInputLen();
    TLLM_LOG_INFO("TRTEncoderModel mMaxInputLen: reset to %d from build config.", mMaxInputLen);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

BufferManager const& TrtEncoderModel::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

BufferManager::CudaStreamPtr TrtEncoderModel::getRuntimeStreamPtr() const
{
    return mRuntime->getStreamPtr();
}

nvinfer1::DataType TrtEncoderModel::getTensorDataType(std::string const& name) const
{
    auto const& engine = mRuntime->getEngine();
    return engine.getTensorDataType(name.c_str());
}

nvinfer1::Dims TrtEncoderModel::getTensorShape(std::string const& name) const
{
    auto const& engine = mRuntime->getEngine();
    return engine.getTensorShape(name.c_str());
}

void TrtEncoderModel::getCurrentIterationStats(executor::IterationStats& stats) const
{
    stats.iter = mIterCounter;
}

void TrtEncoderModel::getCurrentRequestStats(executor::RequestStatsPerIteration& stats) const
{
    stats.iter = mIterCounter;
}

executor::DebugTensorsPerIteration TrtEncoderModel::getCurrentDebugTensors() const
{
    executor::DebugTensorsPerIteration debugTensors;
    debugTensors.iter = mIterCounter;

    TLLM_LOG_WARNING("TrtEncoderModel doesn't support getting debug tensors.");

    return debugTensors;
}

void TrtEncoderModel::setLayerProfiler()
{
    TLLM_CHECK(mRuntime);
    mRuntime->setLayerProfiler();
}

std::string TrtEncoderModel::getLayerProfileInfo() const
{
    TLLM_CHECK(mRuntime);
    return mRuntime->getLayerProfileInfo();
}

void TrtEncoderModel::createRuntimeContexts()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mRuntime->clearContexts();
    auto const numProfiles = mRuntime->getNbProfiles();
    TLLM_CHECK_WITH_INFO(numProfiles == 1, "Encoder only expects one optimization profile");
    for (auto i = 0; i < numProfiles; ++i)
    {
        mRuntime->addContext(i);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::executeContext(SizeType32 runtimeContextId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeContext);
    auto enqueueSuccessful = mRuntime->executeContext(runtimeContextId);
    if (!enqueueSuccessful)
    {
        throw std::runtime_error("Executing TRT engine failed!");
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::createBuffers()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    for (SizeType32 i = 0; i < mNumBuffers; ++i)
    {
        mBuffers.emplace_back(
            std::make_shared<EncoderBuffers>(getMaxBatchSize(), mModelConfig, mWorldConfig, *mRuntime));
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::executeBatch(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeBatch);

    // encoder model only have one optimization profile for now, so no optimization profile switch
    SizeType32 optProfileIndex = 0;
    auto const bufferId = getBufferId();
    if (!scheduledRequests.contextRequests.empty())
    {
        // engine I/O
        auto [inputMap, outputMap]
            = mBuffers[bufferId]->prepareIO(scheduledRequests.contextRequests, mModelConfig, mWorldConfig, *mRuntime);
        mRuntime->setInputTensors(optProfileIndex, inputMap);
        mRuntime->setOutputTensors(optProfileIndex, outputMap);

        // engine run
        executeContext(optProfileIndex);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::rearrangeOutputs(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(rearrangeOutputs);

    auto const bufferId = getBufferId();
    if (!scheduledRequests.contextRequests.empty())
    {
        mBuffers[bufferId]->rearrangeOutputs(scheduledRequests.contextRequests, mModelConfig, mWorldConfig, *mRuntime);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::forwardSync()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE_WITH_NAME(range, "TrtEncoderModel::forwardSync");

    auto const device = mWorldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));

    auto& currRequests = mMicroBatchScheduledRequests.at(mMicroBatchId);
    // auto& encoderWaitEvent = mEncoderWaitEvents.at(mMicroBatchId);

    if (!currRequests.empty())
    {
        if (!mWorldConfig.isPipelineParallel() || !mWorldConfig.isLastPipelineParallelRank())
        {
            // TLLM_CHECK_WITH_INFO(mEncStepAsyncSndHdl.get() == nullptr, "encoderSync handle must be nullptr.");
            // // Wait for encoding for requests in flight for the current micro batch
            // mEncStepAsyncSndHdl = encoderSync(currRequests, encoderWaitEvent);
        }
        else
        {
        }

        NVTX3_SCOPED_RANGE(pauseFlaggedCurrRequests);
        for (auto const& requests : {currRequests.contextRequests})
        {
            for (auto const& llmReq : requests)
            {
                auto const reqId = llmReq->mRequestId;
                mInflightReqIds.erase(reqId);
                TLLM_LOG_DEBUG("request ID %u removed from ENCODER inflight set", reqId);

                // If a request in encoder phase had been flagged to be paused, pause it right away
                if (mReqIdsToPause.find(reqId) != mReqIdsToPause.end())
                {
                    terminateRequest(llmReq, true);
                    mReqIdsToPause.erase(reqId);
                }
            }
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::forwardAsync(RequestList const& activeRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE_WITH_NAME(range, "TrtEncoderModel::ForwardAsync");
    auto const device = mWorldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));

    try
    {
        auto& currRequests = mMicroBatchScheduledRequests.at(mMicroBatchId);
        // auto& encoderWaitEvent = mEncoderWaitEvents.at(mMicroBatchId);

        // Get a new set of requests for encoder
        // The scheduler will not include any requests that are already in flight for encoder models
        // TODO: add pause handling logic
        TLLM_LOG_DEBUG("Running ENCODER request scheduler");

        auto [fittingRequests, fittingDisaggeGenInitReuqests, requestsToPause] = (*mCapacityScheduler)(activeRequests);

        TLLM_CHECK_WITH_INFO(
            fittingDisaggeGenInitReuqests.empty(), "Disaggregated servering is not support by encoder model.");

        std::tie(currRequests.contextRequests, std::ignore) = (*mMicroBatchScheduler)(
            fittingRequests, mInflightReqIds, getMaxBatchSize(), mModelConfig.getMaxNumTokens());

        {
            NVTX3_SCOPED_RANGE(pauseRequestsFlaggedByScheduler);
            // Loop over requests flagged to be paused, and if not in flight pause it right away
            for (auto const& llmReq : requestsToPause)
            {
                auto const reqId = llmReq->mRequestId;
                if (mInflightReqIds.find(reqId) == mInflightReqIds.end())
                {
                    // Not in flight, can terminate right away
                    terminateRequest(llmReq, true);
                }
                else
                {
                    // In flight, add to set for pausing later
                    mReqIdsToPause.insert(reqId);
                }
            }
        }

        TLLM_CHECK(currRequests.size() <= static_cast<size_t>(getMaxBatchSize()));

        if (!currRequests.empty())
        {
            TLLM_LOG_DEBUG("Running ENCODER model with batch size: %u", currRequests.size());
            {
                NVTX3_SCOPED_RANGE(updateInflightReqIds);
                // Add to set of requests in flight
                for (auto const& requests : {currRequests.contextRequests})
                {
                    for (auto const& llmReq : requests)
                    {
                        TLLM_LOG_DEBUG("request ID %u added to ENCODER inflight set", llmReq->mRequestId);
                        mInflightReqIds.insert(llmReq->mRequestId);
                    }
                }
            }

            executeBatch(currRequests);

            sync_check_cuda_error(mRuntime->getStream().get());

            rearrangeOutputs(currRequests);

            sync_check_cuda_error(mRuntime->getStream().get());

            // encoderWaitEvent = encoderStepAsync(currRequests);

            for (auto const& requests : {currRequests.contextRequests})
            {
                for (auto const& llmReq : requests)
                {
                    if (llmReq->isEncoderInitState())
                    {
                        llmReq->setState(LlmRequestState::kCONTEXT_INIT);
                        TLLM_LOG_DEBUG("request ID: %u finishes encoder phase", llmReq->mRequestId);
                    }
                }
            }
        }

        // TODO: PP handling
        if (!currRequests.empty())
        {
            if (mWorldConfig.isPipelineParallel() && mWorldConfig.isLastPipelineParallelRank())
            {
                // TLLM_CHECK_WITH_INFO(mEncStepAsyncSndHdl.get() == nullptr, "decoderSync handle must be nullptr.");
                // Wait for encoding for requests in flight for the current micro batch
                // mEncStepAsyncSndHdl = encoderSync(currRequests, encoderWaitEvent);
            }
        }

        // Update the micro batch ID
        mMicroBatchId = (mMicroBatchId + 1) % mNumMicroBatches;
    }
    // In case of error, we need to free the batch slot associated with those requests
    catch (std::exception const& e)
    {
        for (auto const& llmReq : activeRequests)
        {
            terminateRequest(llmReq);
        }
        throw;
    }

    ++mIterCounter;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::terminateRequest(std::shared_ptr<LlmRequest> const& llmReq, bool pause)
{
    // For encoder-only models, just change req state here. might need to do more when using an asynced forward
    // For enc-dec models, only remove cross kv cache after decoder
    // genenration has finished
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (llmReq->isEncoderInitState())
    {
        llmReq->setState(LlmRequestState::kCONTEXT_INIT);
    }
    else
    {
        TLLM_LOG_DEBUG("Non-encoder request terminated in encoder model: id %lu", llmReq->mRequestId);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::terminateRequestSync(
    std::shared_ptr<LlmRequest> const& llmReq, executor::FinishReason finishReason)
{
    terminateRequest(llmReq, false);
    llmReq->finishByReason(finishReason);
    llmReq->clearGeneratedTokens();
}

void TrtEncoderModel::fillEncoderOutputSync(RequestVector const& requestList, TensorMap outputTensors)
{
    auto const totalTokensNb = outputTensors["encoder_output"]->getShape().d[0];
    auto const encoderOutputDtype = mRuntime->getEngine().getTensorDataType("encoder_output");
    SizeType32 const bytesPerValue = (encoderOutputDtype == nvinfer1::DataType::kFLOAT) ? 4 : 2;
    std::vector<std::byte> encoderOutputHost(
        totalTokensNb * mHiddenSize * bytesPerValue * mWorldConfig.getTensorParallelism());
    TLLM_CHECK_WITH_INFO(encoderOutputHost.size() > 0, "Encoder output size is 0!");
    getBufferManager().copy(*(outputTensors["encoder_output"]), reinterpret_cast<void*>(encoderOutputHost.data()));
    getBufferManager().getStream().synchronize(); // TODO: change engine call to async to improve perf. Also
                                                  // need to store output buffers, cuda events, etc.

    auto encoderOutputHostPtr = encoderOutputHost.data();
    for (auto const& llmReq : requestList)
    {
        SizeType32 const seqLen = llmReq->getEncoderOutputLen();
        TensorPtr currentEncoderOutput
            = mCopyBufferManager.copyFrom(reinterpret_cast<half const*>(encoderOutputHostPtr),
                ITensor::makeShape({seqLen, mHiddenSize * mWorldConfig.getTensorParallelism()}), MemoryType::kCPU);
        llmReq->setEncoderOutputHost(currentEncoderOutput);
        encoderOutputHostPtr += seqLen * mHiddenSize * bytesPerValue * mWorldConfig.getTensorParallelism();

        if (llmReq->isEncoderInitState())
        {
            llmReq->setState(LlmRequestState::kCONTEXT_INIT);
        }
        else
        {
            TLLM_LOG_DEBUG("Non-encoder request terminated in encoder model: id %lu", llmReq->mRequestId);
        }
    }
}

void TrtEncoderModel::executeBatch(RequestVector const& requestList)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeBatch);

    auto const modelName = mModelConfig.getModelName();
    TLLM_CHECK_WITH_INFO(modelName == "EncoderModel" || modelName == "WhisperEncoder", "Model not supported.");
    TensorMap inputTensors;
    TensorMap outputTensors;
    TensorPtr rankOutput;

    std::vector<TokenIdType> inputIdsHost;
    std::vector<SizeType32> positionIdsHost;
    SizeType32 totalOutputLength = 0;
    SizeType32 totalInputLength = 0;
    std::vector<SizeType32> inputLengthsHost;
    std::vector<std::byte> inputFeaturesHost;

    inputLengthsHost.reserve(requestList.size());
    SizeType32 maxInputLengthHost = 0;

    for (auto const& llmReq : requestList)
    {
        SizeType32 length = 0;
        if (mModelConfig.getModelName() == "EncoderModel")
        {
            auto const& reqTokens = *(llmReq->getEncoderTokens().value());
            length = reqTokens.size();

            inputIdsHost.insert(inputIdsHost.end(), reqTokens.begin(), reqTokens.end());
            maxInputLengthHost = std::max(maxInputLengthHost, static_cast<SizeType32>(length));
        }
        else if (mModelConfig.getModelName() == "WhisperEncoder")
        {
            auto const& reqFeatures = llmReq->getEncoderInputFeatures(); // [length, featureDim]
            length = reqFeatures->getShape().d[0];

            auto const curFeatureBytes = reqFeatures->getSizeInBytes();
            auto const srcPtr = reinterpret_cast<std::byte*>(reqFeatures->data());
            inputFeaturesHost.insert(inputFeaturesHost.end(), srcPtr, srcPtr + curFeatureBytes);
        }
        positionIdsHost.reserve(positionIdsHost.size() + length);
        auto const newReqPosBegin = positionIdsHost.end();
        positionIdsHost.resize(positionIdsHost.size() + length);
        std::iota(newReqPosBegin, positionIdsHost.end(), 0);

        totalOutputLength += llmReq->getEncoderOutputLen();
        totalInputLength += length;
        inputLengthsHost.push_back(length);
    }

    TensorPtr hiddenStatesInput;
    TensorPtr inputLengths = getBufferManager().copyFrom(
        inputLengthsHost, ITensor::makeShape({static_cast<SizeType32>(inputLengthsHost.size())}), MemoryType::kGPU);
    inputTensors.emplace("input_lengths", inputLengths);

    if (mModelConfig.getModelName() == "EncoderModel")
    {
        // use shape of maxInputLength to indicates max length, content is not important
        TensorPtr maxInputLength
            = getBufferManager().gpu(ITensor::makeShape({maxInputLengthHost}), nvinfer1::DataType::kINT32);
        inputTensors.emplace("max_input_length", maxInputLength);
    }

    // engine outputs
    rankOutput = getBufferManager().gpu(
        ITensor::makeShape({totalOutputLength, mHiddenSize * mWorldConfig.getTensorParallelism()}),
        mModelConfig.getDataType());

    if (mWorldConfig.isFirstPipelineParallelRank())
    {
        if (mModelConfig.getModelName() == "EncoderModel")
        {
            // Engine inputs
            TensorPtr inputIds
                = getBufferManager().copyFrom(inputIdsHost, ITensor::makeShape({totalInputLength}), MemoryType::kGPU);
            TensorPtr positionIds = getBufferManager().copyFrom(
                positionIdsHost, ITensor::makeShape({totalInputLength}), MemoryType::kGPU);
            inputTensors.emplace("input_ids", inputIds);
            inputTensors.emplace("position_ids", positionIds);
        }
        else if (mModelConfig.getModelName() == "WhisperEncoder")
        {
            auto inputFeaturesHostPtr = inputFeaturesHost.data();
            auto const featureDim = requestList.front()->getEncoderInputFeatures()->getShape().d[1];
            auto const dtype = requestList.front()->getEncoderInputFeatures()->getDataType();
            TensorPtr inputFeatures = getBufferManager().gpu(ITensor::makeShape({totalInputLength, featureDim}), dtype);
            getBufferManager().copy(
                reinterpret_cast<void const*>(inputFeaturesHostPtr), *inputFeatures, runtime::MemoryType::kCPU);
            TensorPtr positionIds = getBufferManager().copyFrom(
                positionIdsHost, ITensor::makeShape({totalOutputLength}), MemoryType::kGPU);
            inputTensors.emplace("input_features", inputFeatures);
            inputTensors.emplace("position_ids", positionIds);
        }
    }
    else
    {
        SizeType32 length = mModelConfig.getModelName() == "WhisperEncoder" ? totalOutputLength : totalInputLength;
        hiddenStatesInput
            = getBufferManager().gpu(ITensor::makeShape({length, mHiddenSize * mWorldConfig.getTensorParallelism()}),
                mModelConfig.getDataType());

        inputTensors.emplace("hidden_states_input", hiddenStatesInput);
    }

    auto const outputName = mWorldConfig.isLastPipelineParallelRank() ? "encoder_output" : "hidden_states_output";
    outputTensors.emplace(outputName, rankOutput);

    // Set input / output tensors to context, encoder model only have one context
    mRuntime->setInputTensors(0, inputTensors);
    mRuntime->setOutputTensors(0, outputTensors);

    executeContext(0);

    // copy encoder output to llmRequest, if last PP rank
    // dispatch result to each llmReq, only needed by the last PP rank
    // TODO: more dtypes support
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        fillEncoderOutputSync(requestList, outputTensors);
    }
    else
    {
        getBufferManager().getStream().synchronize();
    }

    // Update the micro batch ID for next microbatches
    mMicroBatchId = (mMicroBatchId + 1) % mWorldConfig.getPipelineParallelism();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::forward(RequestVector& activeRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const device = mWorldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));

    try
    {
        if (activeRequests.empty())
        {
            return;
        }

        executeBatch(activeRequests);
    }
    catch (std::exception const& e)
    {
        for (auto& req : activeRequests)
        {
            terminateRequest(req);
        }
        throw;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::setLogitsPostProcessorBatched(
    std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched)
{
    TLLM_CHECK_WITH_INFO(!logitsPostProcessorBatched.has_value(), "TrtEncoderModel does not use logits processor.");
}

void TrtEncoderModel::setReplicateLogitsPostProcessor(bool replicateLogitsPostProcessor)
{
    TLLM_THROW("TrtEncoderModel does not use logits processor.");
}

bool TrtEncoderModel::getReplicateLogitsPostProcessor() const
{
    TLLM_THROW("TrtEncoderModel does not use logits processor.");
}

TrtEncoderModel::~TrtEncoderModel() = default;

} //  namespace tensorrt_llm::batch_manager

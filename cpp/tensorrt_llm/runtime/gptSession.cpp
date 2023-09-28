//
// Created by martinma on 5/24/23.
//
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

#include "tensorrt_llm/runtime/gptSession.h"

#include "iBuffer.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/runtime/gptDecoderBatch.h"
#include "tensorrt_llm/runtime/ncclCommunicator.h"
#include "tensorrt_llm/runtime/runtimeBuffers.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/statefulGptDecoder.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

#include <cstdint>
#include <fstream>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace bmkv = tensorrt_llm::batch_manager::kv_cache_manager;

GptSession::GptSession(GptModelConfig const& modelConfig, WorldConfig const& worldConfig, void const* engineBuffer,
    std::size_t engineSize, LoggerPtr logger)
    : mModelConfig{modelConfig}
    , mWorldConfig{worldConfig}
    , mDevice{utils::initDevice(worldConfig)}
    , mLogger{logger ? std::move(logger) : std::make_shared<TllmLogger>()}
    , mRuntime{std::make_shared<TllmRuntime>(engineBuffer, engineSize, *mLogger)}
    , mDecoder{}
    , mBuffers{std::make_shared<RuntimeBuffers>()}
    , mCudaGraphInstances{}
{
    createContexts();
    mBuffers->create(*mRuntime, mModelConfig, mWorldConfig);

    if (mWorldConfig.isPipelineParallel())
    {
        mPipelineComm = NcclCommunicator::createPipelineComm(mWorldConfig, *mLogger);
    }

    // TODO compare expected and runtime tensor names?
}

nvinfer1::ILogger& GptSession::getLogger() const
{
    return *mLogger;
}

BufferManager& GptSession::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

void GptSession::createContexts()
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    mRuntime->clearContexts();
    auto numProfiles = mRuntime->getNbProfiles();
    TLLM_CHECK_WITH_INFO(
        numProfiles == 1 || numProfiles == 2, "GPT only expects one optimization profile or two optimization profiles");
    // Instantiate two contexts for flip-flopping
    if (numProfiles == 1)
    {
        mRuntime->addContext(0);
        mRuntime->addContext(0);
    }
    else
    {
        mRuntime->addContext(1);
        mRuntime->addContext(1);
        mRuntime->addContext(0);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::createDecoder(bool decoderPerRequest)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const vocabSize = mModelConfig.getVocabSize();
    auto const vocabSizePadded = mModelConfig.getVocabSizePadded(mWorldConfig.getSize());
    auto const& stream = mRuntime->getStreamPtr();

    if (decoderPerRequest)
        mDecoder = std::make_shared<GptDecoderBatch>(vocabSize, vocabSizePadded, stream);
    else
        mDecoder = std::make_shared<StatefulGptDecoder>(vocabSize, vocabSizePadded, stream);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::setup(SizeType const batchSize, SizeType const beamWidth, SizeType const maxSequenceLength,
    bool decoderPerRequest, std::optional<SizeType> maxTokensInPagedKvCache)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    // Store this param related to deocder buffer size and kv cache manager to check against
    // the input shape with the params given in generate().
    // gptDecoderBatch does not resize buffers, but allows smaller batchSize and beamWidth.
    // TODO (rkobus) refactor batch manager to remove dependency on maxSequenceLength.
    mDecoderMaxSequenceLength = maxSequenceLength;

    if (mModelConfig.usePagedKvCache())
    {
        auto const numLayers = mModelConfig.getNbLayers();
        auto const nbHeads = mModelConfig.getNbHeads();
        auto const nbKvHeads = mModelConfig.getNbKvHeads();
        auto const hiddenSize = mModelConfig.getHiddenSize();
        auto const tokensPerBlock = mModelConfig.getTokensPerBlock();

        auto const maxBlocksPerSeq = tc::divUp(maxSequenceLength, tokensPerBlock);
        auto const maxNumTokens
            = maxTokensInPagedKvCache.value_or(batchSize * beamWidth * maxBlocksPerSeq * tokensPerBlock);
        auto const maxNumBlocks = tc::divUp(maxNumTokens, tokensPerBlock);
        auto const kvDtype = mBuffers->presentKeysVals.at(0)->getDataType();

        // init KV cache block manager
        mKvCacheManager = std::make_shared<bmkv::KVCacheManager>(numLayers, nbHeads, nbKvHeads, hiddenSize,
            tokensPerBlock, maxNumBlocks, batchSize, kvDtype, mRuntime->getStreamPtr());
    }

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto const logitsType = mRuntime->getEngine().getTensorDataType("logits");
        createDecoder(decoderPerRequest);
        mDecoder->setup(batchSize, beamWidth, maxSequenceLength, logitsType);
    }

    // reshape does not care about maxInputLength or maxNewTokens
    auto const generationConfig = RuntimeBuffers::GenerationConfig{batchSize, beamWidth, 0, 0, maxSequenceLength};
    mBuffers->reshape(generationConfig, mModelConfig, mWorldConfig);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::generate(
    GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(inputs.packed == mModelConfig.usePackedInput(),
        "The chosen model requires a packed input tensor (did you set packed?).");
    TLLM_CHECK_WITH_INFO(inputs.lengths->getShape().nbDims == 1, "Input lengths tensor must be one-dimensional.");

    auto& manager = mRuntime->getBufferManager();
    auto& stream = mRuntime->getStream();
    auto& buffers = *mBuffers;

    buffers.contextLengthsDevice = inputs.lengths;
    buffers.contextLengthsHost->reshape(inputs.lengths->getShape());
    manager.copy(*buffers.contextLengthsDevice, *buffers.contextLengthsHost);
    manager.getStream().synchronize();

    auto const generationConfig = RuntimeBuffers::GenerationConfig::fromInput(inputs.ids, buffers.contextLengthsHost,
        inputs.packed, samplingConfig.beamWidth, mDecoderMaxSequenceLength, inputs.maxNewTokens, manager);

    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    auto const maxInputLength = generationConfig.maxInputLength;
    auto const maxNewTokens = generationConfig.maxNewTokens;

    TLLM_CHECK_WITH_INFO(buffers.allocated, "Buffers not allocated, please call setup first!");

    buffers.reshape(generationConfig, mModelConfig, mWorldConfig);

    if (mModelConfig.usePagedKvCache())
    {
        auto const contextLengthsHost = bufferCast<SizeType const>(*buffers.contextLengthsHost);
        for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            mKvCacheManager->addSequence(batchIdx, contextLengthsHost[batchIdx], beamWidth);
        }
    }

    RuntimeBuffers::TensorMap inputBuffers[2];
    RuntimeBuffers::TensorMap outputBuffers[2];
    auto& onTokenGenerated = outputs.onTokenGenerated;
    outputs.ids->reshape(ITensor::makeShape({batchSize, beamWidth, mDecoderMaxSequenceLength}));
    ITensor::SharedPtr newTokens;
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        mDecoder->newBatch(inputs, samplingConfig);
        newTokens = mDecoder->getNewTokens();
    }
    else if (mWorldConfig.isFirstPipelineParallelRank())
    {
        newTokens = manager.gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32);
    }

    for (SizeType step = 0; step < maxNewTokens; ++step)
    {
        auto const contextId = step % 2;
        bool enqueueSuccessful = false;
        if (step == 0)
        {
            SizeType contextIdForContextPhase = 0;
            if (mRuntime->getNbProfiles() == 2)
            {
                contextIdForContextPhase = 2;
            }
            buffers.prepareContextStep(
                inputs.ids, inputs.padId, manager, *mKvCacheManager, generationConfig, mModelConfig, mWorldConfig);
            buffers.getRuntimeBuffers(inputBuffers[contextId], outputBuffers[contextId], step, inputs.ids,
                *mKvCacheManager, mModelConfig, mWorldConfig);
            mRuntime->setInputTensors(contextIdForContextPhase, inputBuffers[contextId]);
            mRuntime->setOutputTensors(contextIdForContextPhase, outputBuffers[contextId]);

            if (isCudaGraphMode())
            {
                for (auto& instance : mCudaGraphInstances)
                {
                    instance.clear();
                }
            }
            enqueueSuccessful = mRuntime->executeContext(contextIdForContextPhase);
        }
        else
        {
            if (isCudaGraphMode() && mCudaGraphInstances[contextId].hasInstance())
            {
                mCudaGraphInstances[contextId].launch(stream);
                enqueueSuccessful = true;
            }
            else
            {
                enqueueSuccessful = mRuntime->executeContext(contextId);
            }
        }

        TLLM_CHECK_WITH_INFO(enqueueSuccessful, "Executing TRT engine failed!");
        sync_check_cuda_error();

        if (step == 0)
        {
            buffers.postContextStep(manager, generationConfig, mModelConfig, mWorldConfig);
        }

        std::swap(buffers.cacheIndirectionDecoderInput, buffers.cacheIndirectionDecoderOutput);

        if (step < maxNewTokens - 1)
        {
            auto const nextStep = step + 1;
            auto const nextContextId = nextStep % 2;
            auto nextInputIds = buffers.prepareNextStep(
                step, newTokens, manager, *mKvCacheManager, generationConfig, mModelConfig, mWorldConfig);
            buffers.getRuntimeBuffers(inputBuffers[nextContextId], outputBuffers[nextContextId], nextStep, nextInputIds,
                *mKvCacheManager, mModelConfig, mWorldConfig);
            mRuntime->setInputTensors(nextContextId, inputBuffers[nextContextId]);
            mRuntime->setOutputTensors(nextContextId, outputBuffers[nextContextId]);

            if (isCudaGraphMode())
            {
                mCudaGraphInstances[nextContextId].prepareNextGraph(*mRuntime, nextContextId);
            }
        }

        sync_check_cuda_error();

        // FIXME(nkorobov): this synchronize is important to get logits right
        // manager.getStream().synchronize();

        auto shouldStop = executeDecoderStep(outputs.ids, newTokens, maxInputLength + step);

        if (mWorldConfig.isFirstPipelineParallelRank())
        {
            if (onTokenGenerated)
            {
                // TODO(rkobus) use getNewTokens(), remove step from Callback?
                ITensor::SharedPtr outputIds
                    = mWorldConfig.isPipelineParallel() ? outputs.ids : mDecoder->getOutputIds();
                onTokenGenerated(outputIds, step, shouldStop || step == maxNewTokens - 1);
            }
        }

        if (shouldStop)
        {
            mLogger->log(nvinfer1::ILogger::Severity::kVERBOSE, "GPT decoding finished early");
            break;
        }
    }

    if (mModelConfig.usePagedKvCache())
    {
        for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            mKvCacheManager->removeSequence(batchIdx);
        }
    }

    finalizeOutputIds(*outputs.ids);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

bool GptSession::executeDecoderStep(ITensor::SharedPtr& outputIds, ITensor::SharedPtr& newTokens, SizeType decoderStep)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& stream = mRuntime->getStream();
    auto& buffers = *mBuffers;

    auto shouldStopPtr = bufferCast<std::uint8_t>(*buffers.shouldStop);
    auto& shouldStop = *shouldStopPtr;
    shouldStop = false;
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        decoder::Input decodingInput{buffers.logits};
        decoder::Output decodingOutput{};
        decodingInput.cacheIndirection = buffers.cacheIndirectionDecoderInput;
        decodingOutput.cacheIndirection = buffers.cacheIndirectionDecoderOutput;
        decodingOutput.sequenceLengths = buffers.sequenceLengths;

        shouldStop = mDecoder->forward(decodingOutput, decodingInput);
    }

    if (mWorldConfig.isPipelineParallel())
    {
        if (mWorldConfig.isLastPipelineParallelRank())
        {
            for (auto peer = 0; peer < mWorldConfig.getPipelineParallelism() - 1; ++peer)
            {
                mPipelineComm->send(shouldStopPtr, 1, peer, stream, *mLogger);
            }
            mPipelineComm->send(bufferCast<std::int32_t>(*newTokens), newTokens->getSize(), 0, stream, *mLogger);
        }
        else
        {
            auto const peer = mWorldConfig.getPipelineParallelism() - 1;
            mPipelineComm->receive(shouldStopPtr, 1, peer, stream, *mLogger);

            if (mWorldConfig.isFirstPipelineParallelRank())
            {
                mPipelineComm->receive(
                    bufferCast<std::int32_t>(*newTokens), newTokens->getSize(), peer, stream, *mLogger);

                auto const& newTokensShape = newTokens->getShape();
                auto newTokensView
                    = ITensor::view(outputIds, ITensor::makeShape({1, newTokensShape.d[0] * newTokensShape.d[1]}));
                auto const& outputIdsShape = outputIds->getShape();
                auto outputIdsView = ITensor::view(
                    outputIds, ITensor::makeShape({outputIdsShape.d[0] * outputIdsShape.d[1], outputIdsShape.d[2]}));
                kernels::invokeTransposeWithOutputOffset(*outputIdsView, *newTokensView, decoderStep, stream);
            }
        }
    }
    sync_check_cuda_error();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return shouldStop;
}

void GptSession::finalizeOutputIds(ITensor& outputIds)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& manager = mRuntime->getBufferManager();
    auto& stream = mRuntime->getStream();

    ITensor::SharedPtr finalOutputIds;
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        finalOutputIds = mDecoder->getFinalOutputIds();
        if (mWorldConfig.isPipelineParallel())
        {
            mPipelineComm->send(
                bufferCast<std::int32_t>(*finalOutputIds), finalOutputIds->getSize(), 0, stream, *mLogger);
        }
    }
    if (mWorldConfig.isFirstPipelineParallelRank())
    {
        if (mWorldConfig.isPipelineParallel())
        {
            auto const peer = mWorldConfig.getPipelineParallelism() - 1;
            mPipelineComm->receive(bufferCast<std::int32_t>(outputIds), outputIds.getSize(), peer, stream, *mLogger);
        }
        else
        {
            manager.copy(*finalOutputIds, outputIds);
        }
    }
    sync_check_cuda_error();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::CudaGraphExecutor::create(cudaGraph_t const& graph)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    assert(mInstance == nullptr);
    TLLM_CUDA_CHECK(cudaGraphInstantiate(&mInstance, graph, nullptr, nullptr, 0));
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::CudaGraphExecutor::uploadToStream(CudaStream const& stream)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    assert(hasInstance());
    TLLM_CUDA_CHECK(cudaGraphUpload(mInstance, stream.get()));
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::CudaGraphExecutor::launch(CudaStream const& stream)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_CUDA_CHECK(cudaGraphLaunch(mInstance, stream.get()));
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

bool GptSession::CudaGraphExecutor::update(cudaGraph_t const& graph)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    return cudaGraphExecUpdate(mInstance, graph, nullptr) != cudaSuccess;
}

void GptSession::CudaGraphExecutor::clear()
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    if (mInstance != nullptr)
    {
        TLLM_CUDA_CHECK(cudaGraphExecDestroy(mInstance));
        mInstance = nullptr;
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::CudaGraphExecutor::prepareNextGraph(TllmRuntime const& runtime, SizeType nextContextId)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& stream = runtime.getStream();

    cudaGraph_t nextGraph;
    TLLM_CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    runtime.executeContext(nextContextId);
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
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

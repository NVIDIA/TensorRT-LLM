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
    TLLM_CHECK_WITH_INFO(mRuntime->getNbProfiles() == 1, "GPT only expects one optimization profile");
    createContexts();
    mBuffers->create(*mRuntime, mModelConfig);
    // TODO compare expected and runtime tensor names?
}

nvinfer1::ILogger& tensorrt_llm::runtime::GptSession::getLogger() const
{
    return *mLogger;
}

BufferManager& tensorrt_llm::runtime::GptSession::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

void GptSession::createContexts()
{
    mRuntime->clearContexts();
    // Instantiate two contexts for flip-flopping
    mRuntime->addContext(0);
    mRuntime->addContext(0);
}

void GptSession::createDecoder(bool decoderPerRequest)
{
    auto const vocabSize = mModelConfig.getVocabSize();
    auto const vocabSizePadded = mModelConfig.getVocabSizePadded(mWorldConfig.getSize());
    auto const& stream = mRuntime->getStreamPtr();

    if (decoderPerRequest)
        mDecoder = std::make_shared<GptDecoderBatch>(vocabSize, vocabSizePadded, stream);
    else
        mDecoder = std::make_shared<StatefulGptDecoder>(vocabSize, vocabSizePadded, stream);
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

    auto const logitsType = utils::getTensorDataType(mRuntime->getEngine(), "logits");

    createDecoder(decoderPerRequest);
    mDecoder->setup(batchSize, beamWidth, maxSequenceLength, logitsType);

    // reshape does not care about maxInputLength or maxNewTokens
    auto const generationConfig = RuntimeBuffers::GenerationConfig{batchSize, beamWidth, 0, 0, maxSequenceLength};
    mBuffers->reshape(generationConfig, mModelConfig, mWorldConfig.getSize());
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
    auto const maxSeqLength = generationConfig.maxSeqLength;
    auto finalSeqLength = maxSeqLength;

    TLLM_CHECK_WITH_INFO(buffers.allocated, "Buffers not allocated, please call setup first!");

    buffers.reshape(generationConfig, mModelConfig, mWorldConfig.getSize());

    if (mModelConfig.usePackedInput())
    {
        buffers.inputOffsets->reshape(ITensor::makeShape({batchSize + 1}));
        manager.setZero(*buffers.inputOffsets);
        kernels::invokeInclusiveSum(
            *ITensor::slice(buffers.inputOffsets, 1), *buffers.contextLengthsDevice, manager, stream);
    }
    if (mModelConfig.usePagedKvCache())
    {
        auto const contextLengthsHost = bufferCast<SizeType const>(*buffers.contextLengthsHost);
        for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            mKvCacheManager->addSequence(batchIdx, contextLengthsHost[batchIdx], beamWidth);
        }
    }

    mDecoder->newBatch(inputs, samplingConfig);

    RuntimeBuffers::TensorMap inputBuffers[2];
    RuntimeBuffers::TensorMap outputBuffers[2];
    auto& onTokenGenerated = outputs.onTokenGenerated;

    for (SizeType step = 0; step < maxNewTokens; ++step)
    {
        auto const contextId = step % 2;
        if (step == 0)
        {
            buffers.prepareContextStep(
                inputs.ids, inputs.padId, manager, *mKvCacheManager, generationConfig, mModelConfig);
            buffers.getRuntimeBuffers(
                inputBuffers[contextId], outputBuffers[contextId], step, inputs.ids, *mKvCacheManager, mModelConfig);
            mRuntime->setInputTensors(contextId, inputBuffers[contextId]);
            mRuntime->setOutputTensors(contextId, outputBuffers[contextId]);
            if (isCudaGraphMode())
            {
                for (auto& instance : mCudaGraphInstances)
                {
                    instance.clear();
                }
            }
        }
        bool enqueueSuccessful = false;
        if (isCudaGraphMode() && mCudaGraphInstances[contextId].hasInstance())
        {
            mCudaGraphInstances[contextId].launch(stream);
            enqueueSuccessful = true;
        }
        else
        {
            enqueueSuccessful = mRuntime->executeContext(contextId);
        }

        TLLM_CHECK_WITH_INFO(enqueueSuccessful, "Executing TRT engine failed!");
        sync_check_cuda_error();

        if (step == 0)
        {
            buffers.postContextStep(manager, generationConfig, mModelConfig);
        }

        std::swap(buffers.cacheIndirectionDecoderInput, buffers.cacheIndirectionDecoderOutput);

        decoder::Input decodingInput{buffers.logits};
        decoder::Output decodingOutput{};
        decodingInput.cacheIndirection = buffers.cacheIndirectionDecoderInput;
        decodingOutput.cacheIndirection = buffers.cacheIndirectionDecoderOutput;

        if (step < maxNewTokens - 1)
        {
            auto const nextStep = step + 1;
            auto const nextContextId = nextStep % 2;
            auto nextInputIds = buffers.prepareNextStep(
                step, mDecoder->getNewTokens(), manager, *mKvCacheManager, generationConfig, mModelConfig);
            buffers.getRuntimeBuffers(inputBuffers[nextContextId], outputBuffers[nextContextId], nextStep, nextInputIds,
                *mKvCacheManager, mModelConfig);
            mRuntime->setInputTensors(nextContextId, inputBuffers[nextContextId]);
            mRuntime->setOutputTensors(nextContextId, outputBuffers[nextContextId]);

            if (isCudaGraphMode())
            {
                // capture cuda graph
                cudaGraph_t next_graph;
                TLLM_CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
                mRuntime->executeContext(nextContextId);
                TLLM_CUDA_CHECK(cudaStreamEndCapture(stream.get(), &next_graph));

                if (mCudaGraphInstances[nextContextId].hasInstance())
                {
                    if (mCudaGraphInstances[nextContextId].update(next_graph))
                    {
                        mCudaGraphInstances[nextContextId].clear();
                        mCudaGraphInstances[nextContextId].create(next_graph);
                    }
                }
                else
                {
                    mCudaGraphInstances[nextContextId].create(next_graph);
                }

                TLLM_CUDA_CHECK(cudaGraphDestroy(next_graph));
                mCudaGraphInstances[nextContextId].uploadToStream(stream);
            }
        }

        sync_check_cuda_error();

        // FIXME(nkorobov): this synchronize is important to get logits right
        // manager.getStream().synchronize();

        auto const shouldStop = mDecoder->forward(decodingOutput, decodingInput);

        if (onTokenGenerated)
        {
            // TODO(rkobus) use getNewTokens(), remove step from Callback?
            onTokenGenerated(mDecoder->getOutputIds(), step, shouldStop || step == maxNewTokens - 1);
        }

        if (shouldStop)
        {
            finalSeqLength = maxInputLength + step + 1;
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

    outputs.ids->reshape(ITensor::makeShape({batchSize, beamWidth, finalSeqLength}));
    manager.copy(*mDecoder->getFinalOutputIds(), *outputs.ids);
    sync_check_cuda_error();
}

void GptSession::CudaGraphExecutor::create(cudaGraph_t const& graph)
{
    assert(mInstance == nullptr);
    TLLM_CUDA_CHECK(cudaGraphInstantiate(&mInstance, graph, nullptr, nullptr, 0));
}

void GptSession::CudaGraphExecutor::uploadToStream(CudaStream const& stream)
{
    assert(mInstance.hasInstance());
    TLLM_CUDA_CHECK(cudaGraphUpload(mInstance, stream.get()));
}

void GptSession::CudaGraphExecutor::launch(CudaStream const& stream)
{
    TLLM_CUDA_CHECK(cudaGraphLaunch(mInstance, stream.get()));
}

bool GptSession::CudaGraphExecutor::update(cudaGraph_t const& graph)
{
    return cudaGraphExecUpdate(mInstance, graph, nullptr) != cudaSuccess;
}

void GptSession::CudaGraphExecutor::clear()
{
    if (mInstance != nullptr)
    {
        TLLM_CUDA_CHECK(cudaGraphExecDestroy(mInstance));
        mInstance = nullptr;
    }
}

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
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/runtime/gptDecoderBatch.h"
#include "tensorrt_llm/runtime/ipcUtils.h"
#include "tensorrt_llm/runtime/ncclCommunicator.h"
#include "tensorrt_llm/runtime/runtimeBuffers.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/statefulGptDecoder.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>

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
    , mNumMicroBatches{worldConfig.getPipelineParallelism()}
    , mDecoders{}
    , mBuffers{}
    , mCudaGraphInstances{}
{
    if (mWorldConfig.isPipelineParallel())
    {
        mPipelineComm = NcclCommunicator::createPipelineComm(mWorldConfig, *mLogger);
        mCommStream = std::make_shared<CudaStream>();
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

void GptSession::createContexts(SizeType numMicroBatches)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    mRuntime->clearContexts();
    // Instantiate multiple contexts for flip-flopping
    auto const numContextsPerPhase = std::max(2, numMicroBatches);
    auto const numProfiles = mRuntime->getNbProfiles();
    TLLM_CHECK_WITH_INFO(
        numProfiles == 1 || numProfiles == 2, "GPT only expects one optimization profile or two optimization profiles");
    auto constexpr ctxContextId = 0;
    auto constexpr genContextId = 1;
    if (numProfiles == 2)
    {
        for (auto i = 0; i < numContextsPerPhase; ++i)
            mRuntime->addContext(genContextId);
    }
    for (auto i = 0; i < numContextsPerPhase; ++i)
        mRuntime->addContext(ctxContextId);

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::createBuffers(SizeType numMicroBatches)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    mBuffers.clear();

    for (SizeType i = 0; i < numMicroBatches; ++i)
    {
        mBuffers.emplace_back(std::make_shared<RuntimeBuffers>());
        mBuffers.back()->create(*mRuntime, mModelConfig, mWorldConfig);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::createDecoders(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength,
    nvinfer1::DataType logitsType, bool decoderPerRequest, SizeType numMicroBatches)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const vocabSize = mModelConfig.getVocabSize();
    auto const vocabSizePadded = mModelConfig.getVocabSizePadded(mWorldConfig.getSize());
    auto const& stream = mRuntime->getStreamPtr();

    mDecoders.clear();

    for (SizeType i = 0; i < numMicroBatches; ++i)
    {
        if (decoderPerRequest)
            mDecoders.emplace_back(std::make_shared<GptDecoderBatch>(vocabSize, vocabSizePadded, stream));
        else
            mDecoders.emplace_back(std::make_shared<StatefulGptDecoder>(vocabSize, vocabSizePadded, stream));
        mDecoders.back()->setup(batchSize, beamWidth, maxSequenceLength, logitsType);
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::createKvCacheManagers(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength,
    SizeType numMicroBatches, std::optional<SizeType> maxTokensInPagedKvCache)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const localNbLayers = mModelConfig.getNbLayers(mWorldConfig.getPipelineParallelism());
    auto const nbHeads = mModelConfig.getNbHeads();
    auto const nbKvHeads = mModelConfig.getNbKvHeads();
    auto const hiddenSize = mModelConfig.getHiddenSize();
    auto const tokensPerBlock = mModelConfig.getTokensPerBlock();

    auto const maxBlocksPerSeq = tc::divUp(maxSequenceLength, tokensPerBlock);
    auto const maxNumTokens
        = maxTokensInPagedKvCache.value_or(batchSize * beamWidth * maxBlocksPerSeq * tokensPerBlock);
    auto const maxNumBlocks = tc::divUp(maxNumTokens, tokensPerBlock);

    nvinfer1::DataType kvDtype;
    if (mModelConfig.getQuantMode().hasFp8KvCache())
    {
        kvDtype = nvinfer1::DataType::kFP8;
    }
    else if (mModelConfig.getQuantMode().hasInt8KvCache())
    {
        kvDtype = nvinfer1::DataType::kINT8;
    }
    else
    {
        kvDtype = mModelConfig.getDataType();
    }

    mKvCacheManagers.clear();

    for (SizeType i = 0; i < numMicroBatches; ++i)
    {
        mKvCacheManagers.emplace_back(
            std::make_shared<bmkv::KVCacheManager>(localNbLayers, nbHeads, nbKvHeads, hiddenSize, tokensPerBlock,
                maxNumBlocks, batchSize, beamWidth, maxBlocksPerSeq, kvDtype, mRuntime->getStreamPtr()));
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::createCustomAllReduceWorkspace(
    SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength)
{
    setPeerAccess(mWorldConfig, true);

    auto& manager = mRuntime->getBufferManager();
    for (const auto& buffer : mBuffers)
    {
        buffer->createCustomAllReduceWorkspace(
            maxBatchSize, maxBeamWidth, maxSequenceLength, mModelConfig.getHiddenSize(), mWorldConfig, manager);
    }
}

void GptSession::setup(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength, bool decoderPerRequest,
    std::optional<SizeType> maxTokensInPagedKvCache, std::optional<SizeType> numMicroBatches)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    if (numMicroBatches)
        mNumMicroBatches = numMicroBatches.value();
    createContexts(mNumMicroBatches);
    createBuffers(mNumMicroBatches);

    auto const microBatchSize = tc::ceilDiv(maxBatchSize, mNumMicroBatches);
    // Store this param related to deocder buffer size and kv cache manager to check against
    // the input shape with the params given in generate().
    // gptDecoderBatch does not resize buffers, but allows smaller batchSize and beamWidth.
    // TODO refactor batch manager to remove dependency on maxSequenceLength.
    mDecoderMaxSequenceLength = maxSequenceLength;

    if (mModelConfig.usePagedKvCache())
    {
        createKvCacheManagers(
            microBatchSize, maxBeamWidth, maxSequenceLength, mNumMicroBatches, maxTokensInPagedKvCache);
    }

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto const logitsType = mRuntime->getEngine().getTensorDataType("logits");
        createDecoders(
            microBatchSize, maxBeamWidth, maxSequenceLength, logitsType, decoderPerRequest, mNumMicroBatches);
    }

    if (mWorldConfig.isPipelineParallel())
    {
        mReceivedEvents.clear();
        for (SizeType i = 0; i < mNumMicroBatches; ++i)
            mReceivedEvents.emplace_back();
    }

    if (mWorldConfig.isTensorParallel() && mModelConfig.useCustomAllReduce())
    {
        createCustomAllReduceWorkspace(microBatchSize, maxBeamWidth, maxSequenceLength);
    }

    // we don't know maxInputLength and maxNewTokens yet and ignore those for pre-allocation
    auto const generationConfig
        = RuntimeBuffers::GenerationConfig{microBatchSize, maxBeamWidth, 0, 0, maxSequenceLength};

    for (auto& buffers : mBuffers)
        buffers->reshape(generationConfig, mModelConfig, mWorldConfig);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::generateSingleBatch(
    GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(inputs.packed == mModelConfig.usePackedInput(),
        "The chosen model requires a packed input tensor (did you set packed?).");
    auto const& inputLengths = inputs.lengths;
    TLLM_CHECK_WITH_INFO(inputLengths->getShape().nbDims == 1, "Input lengths tensor must be one-dimensional.");

    auto constexpr microBatchId = 0;
    auto& manager = mRuntime->getBufferManager();

    // Initialize and reshape buffers
    auto& buffers = *mBuffers.at(microBatchId);
    TLLM_CHECK_WITH_INFO(buffers.allocated, "Buffers not allocated, please call setup first!");
    buffers.initContextLengths(inputLengths, manager);
    auto const generationConfig = RuntimeBuffers::GenerationConfig::fromInput(*inputs.ids, *buffers.contextLengthsHost,
        inputs.packed, samplingConfig.beamWidth, mDecoderMaxSequenceLength, inputs.maxNewTokens);

    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    auto const maxInputLength = generationConfig.maxInputLength;
    auto const maxNewTokens = generationConfig.maxNewTokens;

    if (mWorldConfig.isLastPipelineParallelRank() && mModelConfig.computeContextLogits())
    {
        auto const vocabSizePadded = mModelConfig.getVocabSizePadded(mWorldConfig.getSize());

        TLLM_CHECK_WITH_INFO(outputs.contextLogits,
            "outputs.contextLogits is nullptr. It must be allocated when computeContextLogits() is enabled.");
        outputs.contextLogits->reshape(ITensor::makeShape({batchSize, maxInputLength, vocabSizePadded}));
        auto const contextLogitsShape = outputs.contextLogits->getShape();
        TLLM_CHECK_WITH_INFO(contextLogitsShape.d[0] == batchSize, "Invalid dim[0]");
        TLLM_CHECK_WITH_INFO(contextLogitsShape.d[1] == maxInputLength, "Invalid dim[1]");
        TLLM_CHECK_WITH_INFO(contextLogitsShape.d[2] == vocabSizePadded, "Invalid dim[2]");

        buffers.logits = outputs.contextLogits;
    }

    buffers.reshape(generationConfig, mModelConfig, mWorldConfig);
    kvCacheAddSequences(beamWidth, microBatchId);
    ITensor::SharedPtr newTokens{initNewTokens(inputs, samplingConfig, microBatchId)};

    auto& onTokenGenerated = outputs.onTokenGenerated;
    outputs.ids->reshape(ITensor::makeShape({batchSize, beamWidth, mDecoderMaxSequenceLength}));

    auto kvCacheManager = mModelConfig.usePagedKvCache() ? mKvCacheManagers.at(microBatchId).get() : nullptr;
    RuntimeBuffers::TensorMap inputBuffers[2];
    RuntimeBuffers::TensorMap outputBuffers[2];
    for (SizeType step = 0; step < maxNewTokens; ++step)
    {
        auto const contextId = step % 2;
        if (step == 0)
        {
            SizeType const contextIdForContextPhase
                = mRuntime->getNbProfiles() == 2 ? mRuntime->getNbContexts() / 2 : 0;
            buffers.prepareContextStep(
                inputs.ids, inputs.padId, manager, kvCacheManager, generationConfig, mModelConfig, mWorldConfig);
            buffers.getRuntimeBuffers(
                inputBuffers[contextId], outputBuffers[contextId], step, inputs.ids, mModelConfig, mWorldConfig);
            mRuntime->setInputTensors(contextIdForContextPhase, inputBuffers[contextId]);
            mRuntime->setOutputTensors(contextIdForContextPhase, outputBuffers[contextId]);

            if (isCudaGraphMode())
            {
                for (auto& instance : mCudaGraphInstances)
                {
                    instance.clear();
                }
            }
            TLLM_CHECK_WITH_INFO(
                mRuntime->executeContext(contextIdForContextPhase), "Executing TRT engine in context phase failed!");
        }
        else
        {
            if (isCudaGraphMode() && mCudaGraphInstances[contextId].hasInstance())
            {
                mCudaGraphInstances[contextId].launch(mRuntime->getStream());
            }
            else
            {
                TLLM_CHECK_WITH_INFO(
                    mRuntime->executeContext(contextId), "Executing TRT engine in generation phase failed!");
            }
        }

        sync_check_cuda_error();

        if (step == 0)
        {
            buffers.postContextStep(manager, generationConfig, mModelConfig, mWorldConfig);
        }

        std::swap(buffers.cacheIndirectionDecoderInput, buffers.cacheIndirectionDecoderOutput);

        if (step < maxNewTokens - 1) // this is not the last step
        {                            // preparing the next step
            auto const nextStep = step + 1;
            auto const nextContextId = nextStep % 2;
            auto nextInputIds = buffers.prepareNextStep(
                step, newTokens, manager, kvCacheManager, generationConfig, mModelConfig, mWorldConfig);
            buffers.getRuntimeBuffers(inputBuffers[nextContextId], outputBuffers[nextContextId], nextStep, nextInputIds,
                mModelConfig, mWorldConfig);
            mRuntime->setInputTensors(nextContextId, inputBuffers[nextContextId]);
            mRuntime->setOutputTensors(nextContextId, outputBuffers[nextContextId]);

            if (isCudaGraphMode())
            {
                mCudaGraphInstances[nextContextId].prepareNextGraph(*mRuntime, nextContextId);
            }
        }

        sync_check_cuda_error();

        // FIXME: this synchronize is important to get logits right
        // manager.getStream().synchronize();

        decoderStepAsync(outputs.ids, newTokens, maxInputLength + step, microBatchId);
        auto const shouldStop = shouldStopSync(batchSize, beamWidth, microBatchId);

        if (mWorldConfig.isFirstPipelineParallelRank())
        {
            if (onTokenGenerated)
            {
                // TODO use getNewTokens(), remove step from Callback?
                ITensor::SharedPtr outputIds
                    = mWorldConfig.isPipelineParallel() ? outputs.ids : mDecoders.at(microBatchId)->getOutputIds();
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
            kvCacheManager->removeSequence(batchIdx);
        }
    }

    finalizeOutputIds(*outputs.ids, microBatchId);
    manager.getStream().synchronize();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::kvCacheAddSequences(SizeType beamWidth, SizeType microBatchId)
{
    if (mModelConfig.usePagedKvCache())
    {
        auto& kvCacheManager = mKvCacheManagers.at(microBatchId);
        TLLM_CHECK(kvCacheManager);
        auto contextLengthsHost = mBuffers.at(microBatchId)->contextLengthsHost;
        TLLM_CHECK(contextLengthsHost);
        auto const contextLengthsPtr = bufferCast<SizeType const>(*contextLengthsHost);
        auto const contextLengthsSize = static_cast<SizeType>(contextLengthsHost->getSize());
        for (SizeType batchIdx = 0; batchIdx < contextLengthsSize; ++batchIdx)
        {
            kvCacheManager->addSequence(batchIdx, contextLengthsPtr[batchIdx], beamWidth);
        }
    }
}

ITensor::SharedPtr GptSession::initNewTokens(
    GenerationInput const& inputs, SamplingConfig const& samplingConfig, SizeType microBatchId)
{
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto& decoder = mDecoders.at(microBatchId);
        decoder->newBatch(inputs, samplingConfig);
        return decoder->getNewTokens();
    }
    else if (mWorldConfig.isFirstPipelineParallelRank())
    {
        auto const beamWidth = samplingConfig.beamWidth;
        auto const batchSize = static_cast<SizeType>(inputs.lengths->getSize());
        return mRuntime->getBufferManager().gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32);
    }
    else
    {
        return ITensor::SharedPtr{};
    }
}

namespace
{
std::vector<GenerationInput> splitInputs(
    GenerationInput const& inputs, SizeType numMicroBatches, BufferManager& manager)
{
    std::vector<GenerationInput> inputBatches;
    auto const numRequests = inputs.lengths->getShape().d[0];
    auto const microBatchSize = tc::ceilDiv(numRequests, numMicroBatches);

    if (inputs.packed)
    {
        auto contextLengthsHost = manager.copyFrom(*inputs.lengths, MemoryType::kCPU);
        ITensor::SharedPtr inputIdsView = ITensor::view(inputs.ids);
        inputIdsView->squeeze(0);
        auto contextLengthsRange = BufferRange<SizeType>(*contextLengthsHost);

        auto tokensBegin = 0;
        for (auto offset = 0; offset < numRequests; offset += microBatchSize)
        {
            auto batchSize = std::min(microBatchSize, numRequests - offset);
            auto numTokens = std::accumulate(
                contextLengthsRange.begin() + offset, contextLengthsRange.begin() + offset + batchSize, 0);

            ITensor::SharedPtr batchInputs = ITensor::slice(inputIdsView, tokensBegin, numTokens);
            batchInputs->reshape(ITensor::makeShape({1, numTokens}));

            inputBatches.emplace_back(inputs.endId, inputs.padId, batchInputs,
                ITensor::slice(inputs.lengths, offset, batchSize), inputs.packed);

            tokensBegin += numTokens;
        }
    }
    else
    {
        for (auto offset = 0; offset < numRequests; offset += microBatchSize)
        {
            auto batchSize = std::min(microBatchSize, numRequests - offset);
            inputBatches.emplace_back(inputs.endId, inputs.padId, ITensor::slice(inputs.ids, offset, batchSize),
                ITensor::slice(inputs.lengths, offset, batchSize), inputs.packed);
        }
    }

    for (auto& batch : inputBatches)
    {
        if (inputs.embeddingBiasOpt)
            batch.embeddingBiasOpt = inputs.embeddingBiasOpt;
        if (inputs.badWordsList)
            batch.badWordsList = inputs.badWordsList;
        if (inputs.stopWordsList)
            batch.stopWordsList = inputs.stopWordsList;
        if (inputs.maxNewTokens)
            batch.maxNewTokens = inputs.maxNewTokens;
    }

    return inputBatches;
}

void updateOutputIds(
    ITensor::SharedPtr& outputIds, ITensor::SharedPtr& newTokens, SizeType decoderStep, CudaStream const& stream)
{ // assemble outputIds of all micro batches
    auto const& newTokensShape = newTokens->getShape();
    auto newTokensView = ITensor::view(newTokens, ITensor::makeShape({1, newTokensShape.d[0] * newTokensShape.d[1]}));
    auto const& outputIdsShape = outputIds->getShape();
    auto outputIdsView = ITensor::view(
        outputIds, ITensor::makeShape({outputIdsShape.d[0] * outputIdsShape.d[1], outputIdsShape.d[2]}));
    kernels::invokeTransposeWithOutputOffset(*outputIdsView, *newTokensView, decoderStep, stream);
}
} // namespace

void GptSession::generateMultiBatch(
    GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(inputs.packed == mModelConfig.usePackedInput(),
        "The chosen model requires a packed input tensor (did you set packed?).");
    auto const& inputLengths = inputs.lengths;
    TLLM_CHECK_WITH_INFO(inputLengths->getShape().nbDims == 1, "Input lengths tensor must be one-dimensional.");

    auto& manager = mRuntime->getBufferManager();

    auto const batchSize = static_cast<SizeType>(inputLengths->getSize());
    auto const beamWidth = samplingConfig.beamWidth;
    outputs.ids->reshape(ITensor::makeShape({batchSize, beamWidth, mDecoderMaxSequenceLength}));
    auto& onTokenGenerated = outputs.onTokenGenerated;

    auto const numMicroBatches = std::min(batchSize, mNumMicroBatches);
    auto microBatches = splitInputs(inputs, numMicroBatches, manager);

    std::vector<RuntimeBuffers::GenerationConfig> generationConfigs;
    std::vector<ITensor::SharedPtr> newTokensPerBatch;
    std::vector<ITensor::SharedPtr> outputIdsPerBatch;

    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto const& microBatchInputs = microBatches.at(microBatchId);

        // Initialize and reshape buffers
        auto& buffers = *mBuffers.at(microBatchId);
        TLLM_CHECK_WITH_INFO(buffers.allocated, "Buffers not allocated, please call setup first!");
        buffers.initContextLengths(microBatchInputs.lengths, manager);
        generationConfigs.emplace_back(RuntimeBuffers::GenerationConfig::fromInput(*microBatchInputs.ids,
            *buffers.contextLengthsHost, microBatchInputs.packed, samplingConfig.beamWidth, mDecoderMaxSequenceLength,
            microBatchInputs.maxNewTokens));
        auto const& generationConfig = generationConfigs.back();

        auto const beamWidth = generationConfig.beamWidth;

        buffers.reshape(generationConfig, mModelConfig, mWorldConfig);
        kvCacheAddSequences(beamWidth, microBatchId);
        newTokensPerBatch.emplace_back(initNewTokens(microBatchInputs, samplingConfig, microBatchId));
    }

    auto maxNewTokens = generationConfigs.front().maxNewTokens;
    auto microBatchSize = generationConfigs.front().batchSize;
    auto offset = 0;
    outputIdsPerBatch.emplace_back(ITensor::slice(outputs.ids, offset, microBatchSize));
    offset += microBatchSize;
    for (auto microBatchId = 1; microBatchId < numMicroBatches; ++microBatchId)
    {
        maxNewTokens = std::min(maxNewTokens, generationConfigs.at(microBatchId).maxNewTokens);
        auto microBatchSize = generationConfigs.at(microBatchId).batchSize;
        outputIdsPerBatch.emplace_back(ITensor::slice(outputs.ids, offset, microBatchSize));
        offset += microBatchSize;
    }

    // TODO(micro batching) do we need 1 or 2 per micro batch?
    std::vector<RuntimeBuffers::TensorMap> inputBuffers(numMicroBatches * 2);
    std::vector<RuntimeBuffers::TensorMap> outputBuffers(numMicroBatches * 2);
    std::vector<bool> microBatchesFinished(numMicroBatches, false);
    for (SizeType step = 0; step < maxNewTokens; ++step)
    {
        if (std::all_of(microBatchesFinished.begin(), microBatchesFinished.end(), [](bool x) { return x; }))
            break;

        for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
        {
            auto& buffers = *mBuffers.at(microBatchId);
            auto kvCacheManager = mModelConfig.usePagedKvCache() ? mKvCacheManagers.at(microBatchId).get() : nullptr;
            auto& newTokens = newTokensPerBatch.at(microBatchId);
            auto& generationConfig = generationConfigs.at(microBatchId);
            auto& outputIds = outputIdsPerBatch.at(microBatchId);

            if (microBatchesFinished.at(microBatchId))
                continue;

            if (step > 0)
            {
                auto const microBatchSize = generationConfig.batchSize;
                auto const beamWidth = generationConfig.beamWidth;
                auto const shouldStop = shouldStopSync(microBatchSize, beamWidth, microBatchId);

                if (mWorldConfig.isFirstPipelineParallelRank() && onTokenGenerated
                    && microBatchId == numMicroBatches - 1)
                {
                    onTokenGenerated(outputs.ids, step - 1, shouldStop);
                }

                if (shouldStop)
                {
                    mLogger->log(nvinfer1::ILogger::Severity::kVERBOSE, "GPT decoding finished early");
                    microBatchesFinished.at(microBatchId) = true;
                    continue;
                }
            }

            auto const contextId = microBatchId;
            if (step == 0)
            {
                SizeType const contextIdForContextPhase
                    = contextId + (mRuntime->getNbProfiles() == 2 ? mNumMicroBatches : 0);
                auto const& inputs = microBatches.at(microBatchId);
                buffers.prepareContextStep(
                    inputs.ids, inputs.padId, manager, kvCacheManager, generationConfig, mModelConfig, mWorldConfig);
                buffers.getRuntimeBuffers(
                    inputBuffers[contextId], outputBuffers[contextId], step, inputs.ids, mModelConfig, mWorldConfig);
                mRuntime->setInputTensors(contextIdForContextPhase, inputBuffers[contextId]);
                mRuntime->setOutputTensors(contextIdForContextPhase, outputBuffers[contextId]);

                TLLM_CHECK_WITH_INFO(
                    mRuntime->executeContext(contextIdForContextPhase), "Executing TRT engine failed!");

                buffers.postContextStep(manager, generationConfig, mModelConfig, mWorldConfig);
            }
            else
            {
                auto nextInputIds = buffers.prepareNextStep(
                    step - 1, newTokens, manager, kvCacheManager, generationConfig, mModelConfig, mWorldConfig);
                buffers.getRuntimeBuffers(
                    inputBuffers[contextId], outputBuffers[contextId], step, nextInputIds, mModelConfig, mWorldConfig);
                mRuntime->setInputTensors(contextId, inputBuffers[contextId]);
                mRuntime->setOutputTensors(contextId, outputBuffers[contextId]);

                TLLM_CHECK_WITH_INFO(mRuntime->executeContext(contextId), "Executing TRT engine failed!");
            }
            sync_check_cuda_error();

            std::swap(buffers.cacheIndirectionDecoderInput, buffers.cacheIndirectionDecoderOutput);

            auto const maxInputLength = generationConfigs.at(microBatchId).maxInputLength;
            auto const decoderStep = maxInputLength + step;
            decoderStepAsync(outputIds, newTokens, decoderStep, microBatchId);
            if (!mWorldConfig.isPipelineParallel() && mNumMicroBatches > 1)
            {
                updateOutputIds(outputIds, newTokens, decoderStep, mRuntime->getStream());
            }
        }
    }

    // TODO(micro batching) move into loop above?
    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto const& generationConfig = generationConfigs.at(microBatchId);
        auto const microBatchSize = generationConfig.batchSize;

        auto kvCacheManager = mModelConfig.usePagedKvCache() ? mKvCacheManagers.at(microBatchId).get() : nullptr;
        auto& outputIds = outputIdsPerBatch.at(microBatchId);

        // TODO(micro batching) sync receive event
        if (mWorldConfig.isFirstPipelineParallelRank() && onTokenGenerated && microBatchId == numMicroBatches - 1)
        {
            onTokenGenerated(outputs.ids, maxNewTokens - 1, true);
        }

        if (mModelConfig.usePagedKvCache())
        {
            for (auto batchIdx = 0; batchIdx < microBatchSize; ++batchIdx)
            {
                kvCacheManager->removeSequence(batchIdx);
            }
        }

        // TODO(micro batching) use mCommStream?
        finalizeOutputIds(*outputIds, microBatchId);
    }
    manager.getStream().synchronize();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::decoderStepAsync(
    ITensor::SharedPtr& outputIds, ITensor::SharedPtr& newTokens, SizeType decoderStep, SizeType microBatchId)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& stream = mRuntime->getStream();
    auto& buffers = *mBuffers.at(microBatchId);

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto& decoder = *mDecoders.at(microBatchId);

        decoder::Input decodingInput{buffers.logits};
        decoder::Output decodingOutput{};
        decodingInput.cacheIndirection = buffers.cacheIndirectionDecoderInput;
        decodingOutput.cacheIndirection = buffers.cacheIndirectionDecoderOutput;
        decodingOutput.sequenceLengths = buffers.sequenceLengths;

        decoder.forwardAsync(decodingOutput, decodingInput);
        if (mWorldConfig.isPipelineParallel())
        { // send shouldStop to all previous ranks and newTokens to the first rank
            stream.record(mCommEvent.get());
            mCommStream->wait(mCommEvent.get());
            auto const pipelineGroup = mWorldConfig.getPipelineParallelGroup();

            auto& cacheIndirection = *buffers.cacheIndirectionDecoderOutput;
            auto& sequenceLengths = *buffers.sequenceLengths;
            auto const beamWidth = cacheIndirection.getShape().d[1];
            for (auto peerIdx = 0; peerIdx < mWorldConfig.getPipelineParallelism() - 1; ++peerIdx)
            {
                mPipelineComm->send<SizeType>(*decoder.getNbFinished(), pipelineGroup[peerIdx], *mCommStream, *mLogger);
                if (beamWidth > 1)
                {
                    mPipelineComm->send<SizeType>(cacheIndirection, pipelineGroup[peerIdx], *mCommStream, *mLogger);
                }
                mPipelineComm->send<SizeType>(sequenceLengths, pipelineGroup[peerIdx], *mCommStream, *mLogger);
            }
            mPipelineComm->send<TokenIdType>(*decoder.getNewTokens(), pipelineGroup.front(), *mCommStream, *mLogger);
        }
    }
    else // pipeline parallel mode
    {    // receive shouldStop from the last rank on a separate stream
        stream.record(mCommEvent.get());
        mCommStream->wait(mCommEvent.get());
        auto const pipelineGroup = mWorldConfig.getPipelineParallelGroup();
        auto const peer = pipelineGroup.back();
        mPipelineComm->receive<SizeType>(*buffers.nbFinished, peer, *mCommStream, *mLogger);

        auto& cacheIndirection = *buffers.cacheIndirectionDecoderOutput;
        auto& sequenceLengths = *buffers.sequenceLengths;
        auto const beamWidth = cacheIndirection.getShape().d[1];
        if (beamWidth > 1)
        {
            mPipelineComm->receive<SizeType>(cacheIndirection, peer, *mCommStream, *mLogger);
        }
        mPipelineComm->receive<SizeType>(sequenceLengths, peer, *mCommStream, *mLogger);
        if (mWorldConfig.isFirstPipelineParallelRank())
        { // receive newTokens from last rank on a separate stream
            mPipelineComm->receive<TokenIdType>(*newTokens, peer, *mCommStream, *mLogger);
            updateOutputIds(outputIds, newTokens, decoderStep, *mCommStream);
        }
        mCommStream->record(mReceivedEvents.at(microBatchId).get());
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

bool GptSession::shouldStopSync(SizeType batchSize, SizeType beamWidth, SizeType microBatchId)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    SizeType nbFinished = 0;

    if (mWorldConfig.isLastPipelineParallelRank())
    { // read the Finished flag from the decoder
        auto& decoder = *mDecoders.at(microBatchId);
        decoder.isFinishedSync();
        nbFinished = *bufferCast<SizeType>(*decoder.getNbFinished());
    }
    else
    { // ensure all information has been received
        TLLM_CUDA_CHECK(cudaEventSynchronize(mReceivedEvents.at(microBatchId).get()));
        nbFinished = *bufferCast<SizeType>(*mBuffers.at(microBatchId)->nbFinished);
    }
    sync_check_cuda_error();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return nbFinished == batchSize * beamWidth;
}

void GptSession::finalizeOutputIds(ITensor& outputIds, SizeType microBatchId)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& manager = mRuntime->getBufferManager();

    if (mWorldConfig.isPipelineParallel())
    {
        auto& stream = mRuntime->getStream();
        auto const pipelineGroup = mWorldConfig.getPipelineParallelGroup();

        if (mWorldConfig.isLastPipelineParallelRank())
        { // send ids from last to first
            auto const peer = pipelineGroup.front();
            auto const finalOutputIds = mDecoders.at(microBatchId)->getFinalOutputIds();
            mPipelineComm->send(
                bufferCast<std::int32_t>(*finalOutputIds), finalOutputIds->getSize(), peer, stream, *mLogger);
        }
        else if (mWorldConfig.isFirstPipelineParallelRank())
        { // receive ids from last on first
            auto const peer = pipelineGroup.back();
            mPipelineComm->receive(bufferCast<std::int32_t>(outputIds), outputIds.getSize(), peer, stream, *mLogger);
        }
    }
    else
    {
        manager.copy(*mDecoders.at(microBatchId)->getFinalOutputIds(), outputIds);
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

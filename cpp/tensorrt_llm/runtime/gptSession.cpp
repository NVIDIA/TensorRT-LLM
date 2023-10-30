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
#include <limits>
#include <memory>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace bmkv = tensorrt_llm::batch_manager::kv_cache_manager;

GptSession::GptSession(Config const& sessionConfig, GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
    void const* engineBuffer, std::size_t engineSize, LoggerPtr logger)
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

    setup(sessionConfig);
}

nvinfer1::ILogger& GptSession::getLogger() const
{
    return *mLogger;
}

BufferManager& GptSession::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

void GptSession::createContexts(SizeType numMicroBatches, bool useCudaGraphs)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    mRuntime->clearContexts();

    if (useCudaGraphs)
    {
        // Instantiate multiple graph instances for flip-flopping
        mCudaGraphInstances.resize(2 * numMicroBatches);
    }

    auto const numProfiles = mRuntime->getNbProfiles();
    TLLM_CHECK_WITH_INFO(
        numProfiles == 1 || numProfiles == 2, "GPT only expects one optimization profile or two optimization profiles");

    if (numProfiles == 2)
    {
        auto constexpr ctxContextId = 0;
        auto constexpr genContextId = 1;
        // Instantiate 2 contexts for flip-flopping
        for (auto i = 0; i < 2 * numMicroBatches; ++i)
            mRuntime->addContext(genContextId);
        // Instantiate 1 context for context phase
        for (auto i = 0; i < numMicroBatches; ++i)
            mRuntime->addContext(ctxContextId);
    }
    else
    {
        auto constexpr contextId = 0;
        // Instantiate 2 contexts for flip-flopping
        for (auto i = 0; i < 2 * numMicroBatches; ++i)
            mRuntime->addContext(contextId);
    }
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

void GptSession::createKvCacheManager(
    SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength, KvCacheConfig const& config)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const localNbLayers = mModelConfig.getNbLayers(mWorldConfig.getPipelineParallelism());
    auto const nbHeads = mModelConfig.getNbHeads();
    auto const nbKvHeads = mModelConfig.getNbKvHeads();
    auto const hiddenSize = mModelConfig.getHiddenSize();
    auto const tokensPerBlock = mModelConfig.getTokensPerBlock();

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

    auto const maxNumTokens = bmkv::KVCacheManager::getMaxNumTokens(config, kvDtype, mModelConfig, mWorldConfig);
    TLLM_LOG_INFO("Using %d tokens in paged KV cache.", maxNumTokens);
    auto const maxNumBlocks = tc::ceilDiv(maxNumTokens, tokensPerBlock);
    auto const maxBlocksPerSeq = tc::ceilDiv(maxSequenceLength, tokensPerBlock);

    mKvCacheManager = std::make_shared<bmkv::KVCacheManager>(localNbLayers, nbHeads, nbKvHeads, hiddenSize,
        tokensPerBlock, maxNumBlocks, batchSize, beamWidth, maxBlocksPerSeq, kvDtype, mRuntime->getStreamPtr());
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

void GptSession::setup(Config const& sessionConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    mCudaGraphMode = sessionConfig.cudaGraphMode;

    auto const maxBatchSize = sessionConfig.maxBatchSize;
    auto const maxBeamWidth = sessionConfig.maxBeamWidth;
    auto const maxSequenceLength = sessionConfig.maxSequenceLength;

    if (sessionConfig.numMicroBatches)
        mNumMicroBatches = sessionConfig.numMicroBatches.value();
    createContexts(mNumMicroBatches, sessionConfig.cudaGraphMode);
    createBuffers(mNumMicroBatches);

    auto const microBatchSize = tc::ceilDiv(maxBatchSize, mNumMicroBatches);
    // Store this param related to decoder buffer size and kv cache manager to check against
    // the input shape with the params given in generate().
    // gptDecoderBatch does not resize buffers, but allows smaller batchSize and beamWidth.
    // TODO refactor batch manager to remove dependency on maxSequenceLength.
    mDecoderMaxSequenceLength = maxSequenceLength;

    if (mModelConfig.usePagedKvCache())
    {
        createKvCacheManager(maxBatchSize, maxBeamWidth, maxSequenceLength, sessionConfig.kvCacheConfig);
    }

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto const logitsType = mRuntime->getEngine().getTensorDataType("logits");
        createDecoders(microBatchSize, maxBeamWidth, maxSequenceLength, logitsType, sessionConfig.decoderPerRequest,
            mNumMicroBatches);
    }

    if (mWorldConfig.isPipelineParallel() || mNumMicroBatches > 1)
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

void GptSession::kvCacheAddSequences(SizeType beamWidth, SizeType microBatchId, SizeType firstBatchIdx)
{
    if (mModelConfig.usePagedKvCache())
    {
        TLLM_CHECK(mKvCacheManager);
        auto contextLengthsHost = mBuffers.at(microBatchId)->contextLengthsHost;
        TLLM_CHECK(contextLengthsHost);
        auto const contextLengthsPtr = bufferCast<SizeType const>(*contextLengthsHost);
        auto const contextLengthsSize = static_cast<SizeType>(contextLengthsHost->getSize());
        for (SizeType batchIdx = firstBatchIdx; batchIdx < firstBatchIdx + contextLengthsSize; ++batchIdx)
        {
            mKvCacheManager->addSequence(batchIdx, contextLengthsPtr[batchIdx], beamWidth);
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

void updateOutputIds(ITensor::SharedPtr const& outputIds, ITensor::SharedPtr const& newTokens, SizeType decoderStep,
    CudaStream const& stream)
{ // assemble outputIds of all micro batches
    auto const& newTokensShape = newTokens->getShape();
    auto newTokensView = ITensor::view(newTokens, ITensor::makeShape({1, newTokensShape.d[0] * newTokensShape.d[1]}));
    auto const& outputIdsShape = outputIds->getShape();
    auto outputIdsView = ITensor::view(
        outputIds, ITensor::makeShape({outputIdsShape.d[0] * outputIdsShape.d[1], outputIdsShape.d[2]}));
    kernels::invokeTransposeWithOutputOffset(*outputIdsView, *newTokensView, decoderStep, stream);
    sync_check_cuda_error();
}
} // namespace

void GptSession::generate(
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
    outputs.lengths->reshape(ITensor::makeShape({batchSize, beamWidth}));
    if (mWorldConfig.isLastPipelineParallelRank() && mModelConfig.computeContextLogits())
    {
        TLLM_CHECK_WITH_INFO(outputs.contextLogits,
            "outputs.contextLogits is nullptr. It must be allocated when computeContextLogits() is enabled.");
        auto const vocabSizePadded = mModelConfig.getVocabSizePadded(mWorldConfig.getSize());
        auto const inputLengthsHost = manager.copyFrom(*inputLengths, MemoryType::kCPU);
        auto const inputLengthsRange = BufferRange<SizeType>(*inputLengthsHost);
        auto const maxInputLength = *std::max_element(inputLengthsRange.begin(), inputLengthsRange.end());
        outputs.contextLogits->reshape(ITensor::makeShape({batchSize, maxInputLength, vocabSizePadded}));
    }

    auto const numMicroBatches = std::min(batchSize, mNumMicroBatches);
    if (numMicroBatches == 1)
    {
        std::vector<GenerationInput> microBatches{inputs};
        generateBatched(outputs, microBatches, samplingConfig);
    }
    else
    {
        auto const microBatches = splitInputs(inputs, numMicroBatches, manager);
        generateBatched(outputs, microBatches, samplingConfig);
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

std::function<void(SizeType microBatchId, SizeType step, bool finished)> GptSession::createOnTokenGeneratedCallback(
    GenerationOutput& outputs, SizeType numMicroBatches)
{
    if (outputs.onTokenGenerated && mWorldConfig.isFirstPipelineParallelRank())
    {
        ITensor::SharedPtr outputIds{mWorldConfig.isPipelineParallel() || mNumMicroBatches > 1
                ? outputs.ids
                : mDecoders.front()->getOutputIds()};
        auto const lastMicroBatchId = numMicroBatches - 1;
        return [onTokenGenerated = outputs.onTokenGenerated, outputIds = std::move(outputIds), lastMicroBatchId](
                   SizeType microBatchId, SizeType step, bool finished)
        {
            if (microBatchId == lastMicroBatchId)
                onTokenGenerated(outputIds, step, finished);
        };
    }
    else
    {
        return [](SizeType microBatchId, SizeType step, bool finished) {};
    }
}

void GptSession::generateBatched(
    GenerationOutput& outputs, std::vector<GenerationInput> const& microBatches, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto& manager = mRuntime->getBufferManager();
    auto const numMicroBatches = static_cast<SizeType>(microBatches.size());
    TLLM_CHECK(numMicroBatches > 0);
    TLLM_CHECK(numMicroBatches <= mNumMicroBatches);
    SizeType const beamWidth{samplingConfig.beamWidth};

    // Initialize and reshape buffers
    std::vector<RuntimeBuffers::GenerationConfig> generationConfigs;
    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto const& microBatchInputs = microBatches.at(microBatchId);
        auto& buffers = *mBuffers.at(microBatchId);
        TLLM_CHECK_WITH_INFO(buffers.allocated, "Buffers not allocated, please call setup first!");
        buffers.initContextLengths(microBatchInputs.lengths, manager);
        generationConfigs.emplace_back(
            RuntimeBuffers::GenerationConfig::fromInput(*microBatchInputs.ids, *buffers.contextLengthsHost,
                microBatchInputs.packed, beamWidth, mDecoderMaxSequenceLength, microBatchInputs.maxNewTokens));
        buffers.reshape(generationConfigs.back(), mModelConfig, mWorldConfig);
    }

    auto minMaxNewTokens = std::numeric_limits<SizeType>::max();
    std::vector<SizeType> microBatchOffsets(1, 0);
    microBatchOffsets.reserve(numMicroBatches + 1);
    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto const& generationConfig = generationConfigs.at(microBatchId);
        minMaxNewTokens = std::min(minMaxNewTokens, generationConfig.maxNewTokens);
        microBatchOffsets.emplace_back(microBatchOffsets.back() + generationConfig.batchSize);
    }

    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto& buffers = *mBuffers.at(microBatchId);
        auto const& generationConfig = generationConfigs.at(microBatchId);
        auto const batchOffset = microBatchOffsets.at(microBatchId);
        kvCacheAddSequences(beamWidth, microBatchId, batchOffset);
        auto const& microBatchInputs = microBatches.at(microBatchId);
        buffers.newTokens = initNewTokens(microBatchInputs, samplingConfig, microBatchId);
        auto const microBatchSize = generationConfig.batchSize;
        buffers.outputIds = ITensor::slice(outputs.ids, batchOffset, microBatchSize);
        buffers.outputLengths = ITensor::slice(outputs.lengths, batchOffset, microBatchSize);
        if (mWorldConfig.isLastPipelineParallelRank() && mModelConfig.computeContextLogits())
        {
            buffers.logits = ITensor::slice(outputs.contextLogits, batchOffset, microBatchSize);
        }
    }

    // Prepare the onTokenGenerated callback
    auto const onTokenGenerated = createOnTokenGeneratedCallback(outputs, numMicroBatches);

    if (useCudaGraphs())
    {
        for (auto& instance : mCudaGraphInstances)
        {
            instance.clear();
        }
    }

    auto kvCacheManager = mModelConfig.usePagedKvCache() ? mKvCacheManager.get() : nullptr;

    std::vector<RuntimeBuffers::TensorMap> inputBuffers(numMicroBatches * 2);
    std::vector<RuntimeBuffers::TensorMap> outputBuffers(numMicroBatches * 2);
    std::vector<bool> microBatchesFinished(numMicroBatches, false);
    auto notFinished = [&microBatchesFinished]()
    { return std::any_of(microBatchesFinished.begin(), microBatchesFinished.end(), [](bool x) { return !x; }); };

    for (SizeType step = 0; step < minMaxNewTokens && notFinished(); ++step)
    {
        auto const flipFlopId = step % 2;
        for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
        {
            if (microBatchesFinished.at(microBatchId))
                continue;

            auto& buffers = *mBuffers.at(microBatchId);
            auto& generationConfig = generationConfigs.at(microBatchId);

            auto const contextId = flipFlopId * numMicroBatches + microBatchId;
            auto& inputBuffer = inputBuffers[contextId];
            auto& outputBuffer = outputBuffers[contextId];

            if (step == 0)
            {
                SizeType const contextIdForContextPhase
                    = (mRuntime->getNbProfiles() == 2 ? 2 * mNumMicroBatches : 0) + microBatchId;

                auto const& microBatchInputs = microBatches.at(microBatchId);
                buffers.prepareContextStep(microBatchInputs.ids, microBatchInputs.padId, manager, kvCacheManager,
                    microBatchOffsets.at(microBatchId), generationConfig, mModelConfig, mWorldConfig);
                buffers.getRuntimeBuffers(
                    inputBuffer, outputBuffer, step, microBatchInputs.ids, mModelConfig, mWorldConfig);
                mRuntime->setInputTensors(contextIdForContextPhase, inputBuffer);
                mRuntime->setOutputTensors(contextIdForContextPhase, outputBuffer);

                TLLM_CHECK_WITH_INFO(
                    mRuntime->executeContext(contextIdForContextPhase), "Executing TRT engine in context step failed!");
                sync_check_cuda_error();

                buffers.postContextStep(manager, generationConfig, mModelConfig, mWorldConfig);
                sync_check_cuda_error();
            }
            else
            {
                auto nextInputIds = buffers.prepareNextStep(step - 1, manager, kvCacheManager,
                    microBatchOffsets.at(microBatchId), generationConfig, mModelConfig, mWorldConfig);
                buffers.getRuntimeBuffers(inputBuffer, outputBuffer, step, nextInputIds, mModelConfig, mWorldConfig);
                mRuntime->setInputTensors(contextId, inputBuffer);
                mRuntime->setOutputTensors(contextId, outputBuffer);

                if (useCudaGraphs())
                {
                    mCudaGraphInstances.at(contextId).prepareNextGraph(*mRuntime, contextId);
                }

                // check decoder result of previous iteration
                auto const microBatchSize = generationConfig.batchSize;
                auto const shouldStop = shouldStopSync(microBatchSize, beamWidth, microBatchId);
                onTokenGenerated(microBatchId, step - 1, shouldStop);

                if (shouldStop)
                {
                    mLogger->log(nvinfer1::ILogger::Severity::kVERBOSE, "GPT decoding finished early");
                    microBatchesFinished.at(microBatchId) = true;
                    continue;
                }

                if (useCudaGraphs())
                {
                    auto& cudaGraphInstance = mCudaGraphInstances.at(contextId);
                    TLLM_CHECK(cudaGraphInstance.hasInstance());
                    cudaGraphInstance.launch(mRuntime->getStream());
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(mRuntime->executeContext(contextId),
                        tc::fmtstr("Executing TRT engine in step %d failed!", step));
                }
                sync_check_cuda_error();
            }

            std::swap(buffers.cacheIndirectionDecoderInput, buffers.cacheIndirectionDecoderOutput);

            auto const maxInputLength = generationConfigs.at(microBatchId).maxInputLength;
            auto const decoderStep = maxInputLength + step;
            decoderStepAsync(decoderStep, microBatchId);
        }
    }

    // Collect the results for the last step
    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto const& generationConfig = generationConfigs.at(microBatchId);
        auto const microBatchSize = generationConfig.batchSize;
        auto const shouldStop = shouldStopSync(microBatchSize, beamWidth, microBatchId);
        onTokenGenerated(microBatchId, minMaxNewTokens - 1, shouldStop);

        auto const firstBatchIdx = microBatchOffsets.at(microBatchId);
        if (mModelConfig.usePagedKvCache())
        {
            for (auto batchIdx = firstBatchIdx; batchIdx < firstBatchIdx + microBatchSize; ++batchIdx)
            {
                kvCacheManager->removeSequence(batchIdx);
            }
        }

        // TODO(micro batching) use mCommStream?
        if (beamWidth > 1)
            finalizeOutputIds(microBatchId);
        else if (!mWorldConfig.isPipelineParallel())
            manager.copy(*mDecoders.at(microBatchId)->getOutputIds(), *mBuffers.at(microBatchId)->outputIds);
    }
    manager.getStream().synchronize();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::decoderStepAsync(SizeType decoderStep, SizeType microBatchId)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& stream = mRuntime->getStream();
    auto& buffers = *mBuffers.at(microBatchId);
    auto const& outputIds = buffers.outputIds;
    auto const& newTokens = buffers.newTokens;

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

    if (!mWorldConfig.isPipelineParallel() && mNumMicroBatches > 1)
    {
        updateOutputIds(outputIds, newTokens, decoderStep, stream);
        stream.record(mReceivedEvents.at(microBatchId).get());
    }

    sync_check_cuda_error();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

bool GptSession::shouldStopSync(SizeType batchSize, SizeType beamWidth, SizeType microBatchId)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    SizeType nbFinished = 0;

    if (mWorldConfig.isLastPipelineParallelRank())
    { // read the Finished flag from the decoder
        auto& decoder = *mDecoders.at(microBatchId);
        decoder.forwardSync();
        nbFinished = *bufferCast<SizeType>(*decoder.getNbFinished());

        if (!mWorldConfig.isPipelineParallel() && mNumMicroBatches > 1)
        {
            // ensure outputIds have been updated
            mReceivedEvents.at(microBatchId).synchronize();
        }
    }
    else
    { // ensure all information has been received
        mReceivedEvents.at(microBatchId).synchronize();
        nbFinished = *bufferCast<SizeType>(*mBuffers.at(microBatchId)->nbFinished);
    }
    sync_check_cuda_error();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return nbFinished == batchSize * beamWidth;
}

void GptSession::finalizeOutputIds(SizeType microBatchId)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& manager = mRuntime->getBufferManager();
    auto& outputIds = *mBuffers.at(microBatchId)->outputIds;
    auto& sequenceLengths = *mBuffers.at(microBatchId)->sequenceLengths;

    if (mWorldConfig.isPipelineParallel())
    {
        auto& stream = mRuntime->getStream();
        auto const pipelineGroup = mWorldConfig.getPipelineParallelGroup();

        if (mWorldConfig.isLastPipelineParallelRank())
        { // send ids from last to first
            auto const peer = pipelineGroup.front();
            auto const finalOutputIds = mDecoders.at(microBatchId)->getFinalOutputIds();
            mPipelineComm->send<TokenIdType>(*finalOutputIds, peer, stream, *mLogger);
            mPipelineComm->send<SizeType>(sequenceLengths, peer, stream, *mLogger);
        }
        else if (mWorldConfig.isFirstPipelineParallelRank())
        { // receive ids from last on first
            auto const peer = pipelineGroup.back();
            mPipelineComm->receive<TokenIdType>(outputIds, peer, stream, *mLogger);
            mPipelineComm->receive<SizeType>(sequenceLengths, peer, stream, *mLogger);
        }
    }
    else
    {
        manager.copy(*mDecoders.at(microBatchId)->getFinalOutputIds(), outputIds);
        // sequenceLengths are already updated by decoder
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

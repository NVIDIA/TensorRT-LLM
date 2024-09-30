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

#include "tensorrt_llm/runtime/gptSession.h"

#include "common.h"
#include "iBuffer.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/ipcUtils.h"
#include "tensorrt_llm/runtime/ncclCommunicator.h"
#include "tensorrt_llm/runtime/runtimeBuffers.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/statefulGptDecoder.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

#include <algorithm>
#include <cstdlib> // std::getenv
#include <cstring>
#include <cuda_profiler_api.h>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace bmkv = tensorrt_llm::batch_manager::kv_cache_manager;

namespace
{

std::unordered_set<std::int32_t> populateMicrobatchIndexes()
{
    auto const* profileMbIdxChar = std::getenv("TLLM_GPTS_PROFILE_START_STOP");
    std::unordered_set<std::int32_t> idxSet;
    if (profileMbIdxChar != nullptr)
    {
        std::istringstream iss{profileMbIdxChar};
        std::int32_t idx;
        char c;
        while (iss >> idx)
        {
            idxSet.insert(idx);
            iss >> c;
        }
    }

    return idxSet;
}

auto const kProfileMbIdxs = populateMicrobatchIndexes();

GptSession::Config setPath(GptSession::Config const& original, std::string const& path)
{
    GptSession::Config config = original;
    config.enginePath = std::filesystem::path(path);
    return config;
}

} // namespace

GptSession::GptSession(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    RawEngine const& rawEngine, LoggerPtr logger)
    : mModelConfig{modelConfig}
    , mWorldConfig{worldConfig}
    , mDevice{utils::initDevice(worldConfig)}
    , mLogger{logger ? std::move(logger) : std::make_shared<TllmLogger>()}
    , mRuntime{std::make_shared<TllmRuntime>(rawEngine, mLogger.get(), sessionConfig.gpuWeightsPercent)}
{
    TLLM_LOG_WARNING(
        "GptSession is deprecated and will be removed in a future release."
        " Please use the executor API instead (cpp/include/tensorrt_llm/executor).");

    if (mWorldConfig.isPipelineParallel())
    {
        mPipelineComm = std::make_shared<NcclCommunicator>(mWorldConfig);
        mCommStream = std::make_shared<CudaStream>();
    }

    TLLM_CHECK_WITH_INFO(!(mModelConfig.usePromptTuning() && !mModelConfig.useGptAttentionPlugin()),
        "Prompt tuning is only enabled with GPT attention plugin.");

    // TODO compare expected and runtime tensor names?

    setup(sessionConfig);

    if (mModelConfig.getManageWeightsType() != ModelConfig::ManageWeightsType::kDisabled)
    {
        mRuntime->loadManagedWeights(rawEngine, mWorldConfig.getLocalRank());
    }
}

GptSession::GptSession(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    std::string const& engineFile, LoggerPtr logger)
    : GptSession(
        setPath(sessionConfig, engineFile), modelConfig, worldConfig, utils::loadEngine(engineFile), std::move(logger))
{
}

nvinfer1::ILogger& GptSession::getLogger() const
{
    return *mLogger;
}

BufferManager const& GptSession::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

BufferManager::CudaStreamPtr GptSession::getRuntimeStreamPtr() const
{
    return mRuntime->getStreamPtr();
}

nvinfer1::DataType GptSession::getLogitDataType() const
{
    return mRuntime->getEngine().getTensorDataType("logits");
}

nvinfer1::IEngineInspector& GptSession::getEngineInspector() const
{
    return mRuntime->getEngineInspector();
}

void GptSession::createContexts()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mRuntime->clearContexts();

    auto const numProfiles = mRuntime->getNbProfiles();
    TLLM_CHECK_WITH_INFO(numProfiles == 1 || numProfiles == 2,
        "GptSession only expects 1 or 2 optimization profiles, set --multiple_profiles=disable when calling "
        "trtllm-build to disable the feature. Please also note that, GptSession is going to be deprecated in the "
        "future.");

    // Instantiate 1 execution context for each profile
    for (auto contextId = 0; contextId < numProfiles; ++contextId)
    {
        mRuntime->addContext(contextId);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::createBuffers(SizeType32 numMicroBatches)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mBuffers.clear();

    for (SizeType32 i = 0; i < numMicroBatches; ++i)
    {
        mBuffers.emplace_back(std::make_shared<RuntimeBuffers>());
        mBuffers.back()->create(*mRuntime, mModelConfig, mWorldConfig);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::createDecoders(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxAttentionWindow,
    SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, nvinfer1::DataType logitsType, bool decoderPerRequest,
    SizeType32 numMicroBatches, executor::DecodingMode const& decodingMode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const vocabSize = mModelConfig.getVocabSize();
    auto const vocabSizePadded = mModelConfig.getVocabSizePadded(mWorldConfig.getSize());
    auto const& stream = mRuntime->getStreamPtr();

    mDecoders.clear();

    for (SizeType32 i = 0; i < numMicroBatches; ++i)
    {
        if (decoderPerRequest)
        {
            mDecoders.emplace_back(std::make_shared<GptDecoderBatched>(
                vocabSize, vocabSizePadded, stream, mModelConfig.getSpeculativeDecodingMode(), logitsType));
        }
        else
        {
            mDecoders.emplace_back(std::make_shared<StatefulGptDecoder>(vocabSize, vocabSizePadded, stream));
        }
        constexpr SizeType32 maxTokensPerStep = 1;
        mDecoders.back()->setup(decodingMode, batchSize, beamWidth, maxAttentionWindow, sinkTokenLength,
            maxSequenceLength, maxTokensPerStep, logitsType, mModelConfig);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::createKvCacheManager(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
    SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, KvCacheConfig const& kvCacheConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const tokensPerBlock = mModelConfig.getTokensPerBlock();

    auto const kvDtype = mModelConfig.getKvDataType();

    auto const [blocksInPrimaryPool, blocksInSecondaryPool] = bmkv::KVCacheManager::calculateMaxNumBlocks(
        kvCacheConfig, kvDtype, mModelConfig, mWorldConfig, getBufferManager());

    // If maxBeamWidth > 1, use one more block for each sequence in the paged kv cache to avoid dropping the needed
    // tokens, when enabling cyclic kv cache.
    auto const useOneMoreBlock = maxBeamWidth > 1 && maxSequenceLength > maxAttentionWindow;

    auto const localNbLayers = mModelConfig.getNbAttentionLayers(mWorldConfig.getPipelineParallelism());
    auto const nbKvHeads = mModelConfig.getNbKvHeads();
    auto const sizePerHead = mModelConfig.getSizePerHead();
    bool constexpr enableBlockReuse{false};
    bool enableDiffMaxAttenWin = false;
    for (SizeType32 maxAttenWin : mDecoderMaxAttentionWindowVec)
    {
        if (maxAttenWin != maxAttentionWindow)
        {
            enableDiffMaxAttenWin = true;
            break;
        }
    }
    TLLM_CHECK_WITH_INFO(maxBeamWidth == 1 || !enableDiffMaxAttenWin,
        "Can't support layer-wise max_attention_window with beam search. Please use a unified max_attention_window for "
        "all layers.");
    mKvCacheManager = std::make_shared<bmkv::KVCacheManager>(localNbLayers, nbKvHeads, sizePerHead, tokensPerBlock,
        blocksInPrimaryPool, blocksInSecondaryPool, maxBatchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength,
        useOneMoreBlock, mRuntime->getStreamPtr(), enableBlockReuse, kvCacheConfig.onboardBlocks);

    auto const maxBlocksPerSeq = mKvCacheManager->getMaxBlocksPerSeq();

    TLLM_CHECK(mBuffers.size() == static_cast<size_t>(mMicroBatchConfig.numGenBatches));
    for (auto& buffers : mBuffers)
    {
        TLLM_CHECK(buffers->transformerBuffers);
        buffers->transformerBuffers->reshapeKvTensors(maxBatchSize, maxBeamWidth, maxBlocksPerSeq, *mRuntime);
    }

    mKvCacheManager->allocatePools(kvDtype, kvCacheConfig.useUvm);

    for (auto& buffers : mBuffers)
    {
        buffers->transformerBuffers->setKvPoolPointers(mKvCacheManager.get());
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::createCustomAllReduceWorkspace(
    SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxSequenceLength)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& manager = mRuntime->getBufferManager();
    auto const hiddenSize = mModelConfig.getHiddenSize();

    mAllReduceBuffers = std::make_shared<AllReduceBuffers>(
        maxBatchSize, maxBeamWidth, maxSequenceLength, hiddenSize, manager, mWorldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

GptSession::MicroBatchConfig::MicroBatchConfig(SizeType32 maxBatchSize, SizeType32 pipelineParallelism,
    std::optional<SizeType32> genMicroBatchSize, std::optional<SizeType32> ctxMicroBatchSize)
{
    if (genMicroBatchSize || ctxMicroBatchSize)
    {
        genBatchSize = genMicroBatchSize.value_or(maxBatchSize);
        TLLM_CHECK(genBatchSize <= maxBatchSize);
        ctxBatchSize = ctxMicroBatchSize.value_or(genBatchSize);
        TLLM_CHECK_WITH_INFO(genBatchSize % ctxBatchSize == 0,
            "Generation batch size (%d) must be divisible by context batch size (%d)", genBatchSize, ctxBatchSize);
        numGenBatches = tc::ceilDiv(maxBatchSize, genBatchSize);
        numCtxBatches = numGenBatches * (genBatchSize / ctxBatchSize);
    }
    else
    {
        numCtxBatches = numGenBatches = pipelineParallelism;
        ctxBatchSize = genBatchSize = tc::ceilDiv(maxBatchSize, numGenBatches);
    }
}

void GptSession::setup(Config const& sessionConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mCudaGraphMode = sessionConfig.cudaGraphMode;

    auto const maxBatchSize = sessionConfig.maxBatchSize;
    auto const maxBeamWidth = sessionConfig.maxBeamWidth;
    auto const maxSequenceLength = sessionConfig.maxSequenceLength;
    std::vector<SizeType32> maxAttentionWindowVec;
    SizeType32 maxAttentionWindow = 0;
    if (sessionConfig.kvCacheConfig.maxAttentionWindowVec.has_value())
    {
        bool warning = false;
        for (SizeType32 maxAttenWin : sessionConfig.kvCacheConfig.maxAttentionWindowVec.value())
        {
            maxAttentionWindowVec.push_back(std::min(maxAttenWin, maxSequenceLength));
            maxAttentionWindow = std::max(maxAttentionWindow, maxAttentionWindowVec.back());
            if (maxAttenWin > maxSequenceLength)
                warning = true;
        }
        if (warning)
            TLLM_LOG_WARNING(
                "The value of maxAttentionWindow cannot exceed maxSequenceLength. "
                "Therefore, it has been adjusted to match the value of maxSequenceLength.");
    }
    else
    {
        maxAttentionWindowVec.push_back(maxSequenceLength);
        maxAttentionWindow = maxSequenceLength;
    }
    auto const sinkTokenLength = sessionConfig.kvCacheConfig.sinkTokenLength.has_value()
        ? sessionConfig.kvCacheConfig.sinkTokenLength.value()
        : 0;

    mMicroBatchConfig = MicroBatchConfig(maxBatchSize, mWorldConfig.getPipelineParallelism(),
        sessionConfig.genMicroBatchSize, sessionConfig.ctxMicroBatchSize);

    if (sessionConfig.cudaGraphMode)
    {
        // Instantiate 2 graph instances for flip-flopping of each generation batch
        mCudaGraphInstances.resize(2 * mMicroBatchConfig.numGenBatches);
    }
    createContexts();
    createBuffers(mMicroBatchConfig.numGenBatches);

    mNormalizeLogProbs = sessionConfig.normalizeLogProbs;

    // Store this param related to decoder buffer size and kv cache manager to check against
    // the input shape with the params given in generate().
    // GptDecoderBatched does not resize buffers, but allows smaller batchSize and beamWidth.
    // TODO refactor batch manager to remove dependency on maxSequenceLength.
    mDecoderMaxSequenceLength = maxSequenceLength;
    mDecoderMaxAttentionWindowVec = maxAttentionWindowVec;
    mDecoderMaxAttentionWindow = maxAttentionWindow;
    mDecoderSinkTokenLength = sinkTokenLength;

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto const logitsType = mRuntime->getEngine().getTensorDataType("logits");
        executor::DecodingMode decodingMode = sessionConfig.decodingMode.value_or(
            maxBeamWidth == 1 ? executor::DecodingMode::TopKTopP() : executor::DecodingMode::BeamSearch());
        createDecoders(mMicroBatchConfig.genBatchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength,
            maxSequenceLength, logitsType, sessionConfig.decoderPerRequest, mMicroBatchConfig.numGenBatches,
            decodingMode);
    }

    if (mWorldConfig.isPipelineParallel() || mMicroBatchConfig.numGenBatches > 1)
    {
        mReceivedEvents.clear();
        for (SizeType32 i = 0; i < mMicroBatchConfig.numGenBatches; ++i)
        {
            mReceivedEvents.emplace_back();
        }
    }

    if (mWorldConfig.isTensorParallel())
    {
        createCustomAllReduceWorkspace(mMicroBatchConfig.genBatchSize, maxBeamWidth, maxSequenceLength);
    }

    for (auto& buffers : mBuffers)
    {
        // we don't know maxInputLength yet and ignore it for pre-allocation
        buffers->generationConfig = GenerationConfig{mMicroBatchConfig.genBatchSize, maxBeamWidth, 0,
            maxAttentionWindowVec, maxAttentionWindow, sinkTokenLength, maxSequenceLength};
        buffers->reshape(mModelConfig, mWorldConfig);
    }

    if (shouldUseKVCacheManager())
    {
        createKvCacheManager(maxBatchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSequenceLength,
            sessionConfig.kvCacheConfig);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::kvCacheAddSequences(SizeType32 beamWidth, SizeType32 microBatchId, SizeType32 firstBatchIdx)
{
    if (shouldUseKVCacheManager())
    {
        TLLM_CHECK(mKvCacheManager);
        auto contextLengthsHost = mBuffers.at(microBatchId)->contextLengthsHost;
        TLLM_CHECK(contextLengthsHost);
        auto const* const contextLengthsPtr = bufferCast<SizeType32 const>(*contextLengthsHost);
        auto const contextLengthsSize = static_cast<SizeType32>(contextLengthsHost->getSize());
        for (SizeType32 batchIdx = 0; batchIdx < contextLengthsSize; ++batchIdx)
        {
            mKvCacheManager->addSequence(firstBatchIdx + batchIdx, contextLengthsPtr[batchIdx], beamWidth);
        }
    }
}

ITensor::SharedPtr GptSession::initDecoder(ITensor& outputIds, GenerationInput const& inputs,
    GenerationOutput const& outputs, SamplingConfig const& samplingConfig, SizeType32 microBatchId) const
{
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto& decoder = *mDecoders.at(microBatchId);
        decoder.newBatch(inputs, outputs, samplingConfig);
        return decoder.getNewTokens();
    }
    else if (mWorldConfig.isFirstPipelineParallelRank())
    {
        auto& manager = mRuntime->getBufferManager();
        auto const& stream = mRuntime->getStreamPtr();

        auto const inputLengths = inputs.lengths;
        auto const batchSize = static_cast<SizeType32>(inputLengths->getSize());
        auto const inputLengthsHost = manager.copyFrom(*inputLengths, MemoryType::kCPU);
        auto const* inputLengthsData = bufferCast<SizeType32>(*inputLengthsHost);
        SizeType32 const maxInputLength
            = *std::max_element(inputLengthsData, inputLengthsData + inputLengths->getSize());

        ITensor::SharedPtr inputOffsets = manager.emptyTensor(MemoryType::kGPU, TRTDataType<SizeType32>::value);
        if (inputs.packed)
        {
            inputOffsets->reshape(ITensor::makeShape({batchSize + 1}));
            manager.setZero(*inputOffsets);
            kernels::invokeInclusiveSum(*ITensor::slice(inputOffsets, 1), *inputLengths, manager, *stream);
        }

        kernels::initOutputIds(outputIds, *inputs.ids, *inputLengths, *inputOffsets, inputs.padId, inputs.endId,
            maxInputLength, inputs.packed, *stream);

        auto const beamWidth = samplingConfig.beamWidth;
        return manager.gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32);
    }
    else
    {
        return ITensor::SharedPtr{};
    }
}

namespace
{
std::tuple<std::vector<ITensor::SharedPtr>, std::vector<ITensor::SharedPtr>, std::vector<SizeType32>> splitInputIds(
    GenerationInput const& inputs, SizeType32 microBatchSize, BufferManager& manager,
    std::optional<SizeType32> maxNumTokens)
{
    auto const numRequests = static_cast<SizeType32>(inputs.lengths->getShape().d[0]);

    std::vector<ITensor::SharedPtr> inputIds;
    std::vector<ITensor::SharedPtr> inputLengths;
    std::vector<SizeType32> microBatchOffsets(1, 0);
    if (inputs.packed)
    {
        auto const contextLengthsHost = manager.copyFrom(*inputs.lengths, MemoryType::kCPU);
        ITensor::SharedPtr inputIdsView = ITensor::view(inputs.ids);
        if (inputIdsView->getShape().nbDims == 2)
        { // For users still pass inputs.ids with shape [1, num_tokens], do squeeze for them.
            inputIdsView->squeeze(0);
        }
        TLLM_CHECK(inputIdsView->getShape().nbDims == 1);
        auto const contextLengthsRange = BufferRange<SizeType32>(*contextLengthsHost);

        auto tokensBegin = 0;
        for (auto offset = 0; offset < numRequests; offset += microBatchSize)
        {
            auto const batchSize = std::min(microBatchSize, numRequests - offset);
            auto const numTokens = std::accumulate(
                contextLengthsRange.begin() + offset, contextLengthsRange.begin() + offset + batchSize, 0);
            if (maxNumTokens)
                TLLM_CHECK_WITH_INFO(numTokens <= maxNumTokens.value(),
                    "Micro-batch %d with %d token exceeds max_num_tokens=%d, consider to use larger value when "
                    "building engine",
                    offset / microBatchSize, numTokens, maxNumTokens.value());
            ITensor::SharedPtr batchInputs = ITensor::slice(inputIdsView, tokensBegin, numTokens);
            TLLM_CHECK(batchInputs->getShape().nbDims == 1);
            TLLM_CHECK(batchInputs->getShape().d[0] == numTokens);

            inputIds.emplace_back(std::move(batchInputs));
            inputLengths.emplace_back(ITensor::slice(inputs.lengths, offset, batchSize));
            microBatchOffsets.emplace_back(offset + batchSize);

            tokensBegin += numTokens;
        }
    }
    else
    {
        for (auto offset = 0; offset < numRequests; offset += microBatchSize)
        {
            auto const batchSize = std::min(microBatchSize, numRequests - offset);

            inputIds.emplace_back(ITensor::slice(inputs.ids, offset, batchSize));
            inputLengths.emplace_back(ITensor::slice(inputs.lengths, offset, batchSize));
            microBatchOffsets.emplace_back(offset + batchSize);
        }
    }

    return {inputIds, inputLengths, microBatchOffsets};
}

std::vector<GenerationInput> splitInputs(GenerationInput const& inputs, SizeType32 microBatchSize,
    BufferManager& manager, std::optional<SizeType32> maxNumTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto [inputIds, inputLengths, microBatchOffsets] = splitInputIds(inputs, microBatchSize, manager, maxNumTokens);

    std::vector<GenerationInput> inputBatches;
    for (std::size_t batchId = 0; batchId < inputIds.size(); ++batchId)
    {
        inputBatches.emplace_back(
            inputs.endId, inputs.padId, std::move(inputIds[batchId]), std::move(inputLengths[batchId]), inputs.packed);
    }

    for (std::size_t batchId = 0; batchId < inputBatches.size(); ++batchId)
    {
        auto& batch = inputBatches[batchId];
        auto const offset = microBatchOffsets[batchId];
        auto const batchSize = microBatchOffsets[batchId + 1] - offset;

        if (inputs.embeddingBias)
        {
            batch.embeddingBias = inputs.embeddingBias;
        }

        if (inputs.badWordsList)
        {
            auto const& shape = inputs.badWordsList->getShape();
            if (shape.nbDims == 2)
            {
                batch.badWordsList = inputs.badWordsList;
            }
            else
            {
                assert(shape.nbDims == 3);
                batch.badWordsList = ITensor::slice(inputs.badWordsList, offset, batchSize);
            }
        }
        if (inputs.stopWordsList)
        {
            batch.stopWordsList = ITensor::slice(inputs.stopWordsList, offset, batchSize);
        }
        if (inputs.maxNewTokens)
        {
            batch.maxNewTokens = inputs.maxNewTokens;
        }

        if (inputs.promptTuningParams.embeddingTable)
        {
            batch.promptTuningParams.embeddingTable = inputs.promptTuningParams.embeddingTable;
        }
        if (inputs.promptTuningParams.tasks)
        {
            batch.promptTuningParams.tasks = ITensor::slice(inputs.promptTuningParams.tasks, offset, batchSize);
        }
        if (inputs.promptTuningParams.vocabSize)
        {
            batch.promptTuningParams.vocabSize = inputs.promptTuningParams.vocabSize;
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return inputBatches;
}

std::vector<GenerationOutput> splitOutputs(
    GenerationOutput& outputs, SizeType32 microBatchSize, WorldConfig const& mWorldConfig)
{
    auto const numRequests = static_cast<SizeType32>(outputs.ids->getShape().d[0]);

    std::vector<GenerationOutput> outputBatches;
    for (auto batchOffset = 0; batchOffset < numRequests; batchOffset += microBatchSize)
    {
        auto const batchSize = std::min(microBatchSize, numRequests - batchOffset);

        outputBatches.emplace_back(ITensor::slice(outputs.ids, batchOffset, batchSize),
            ITensor::slice(outputs.lengths, batchOffset, batchSize));

        if (outputs.cumLogProbs)
        {
            outputBatches.back().cumLogProbs = ITensor::slice(outputs.cumLogProbs, batchOffset, batchSize);
        }
        if (outputs.logProbs)
        {
            outputBatches.back().logProbs = ITensor::slice(outputs.logProbs, batchOffset, batchSize);
        }
        if (outputs.contextLogits && mWorldConfig.isLastPipelineParallelRank())
        {
            outputBatches.back().contextLogits = ITensor::slice(outputs.contextLogits, batchOffset, batchSize);
        }
        if (outputs.generationLogits && mWorldConfig.isLastPipelineParallelRank())
        {
            outputBatches.back().generationLogits = ITensor::slice(outputs.generationLogits, batchOffset, batchSize);
        }
    }

    return outputBatches;
}

void updateOutputIds(ITensor::SharedPtr const& outputIds, ITensor::SharedPtr const& newTokens, SizeType32 decoderStep,
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

void GptSession::generate(GenerationOutput& outputs, GenerationInput const& inputs,
    SamplingConfig const& samplingConfig, std::shared_ptr<GenerationProfiler> const generationProfiler)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(inputs.packed == mModelConfig.usePackedInput(),
        "The chosen model requires a packed input tensor (did you set packed?).");
    auto const& inputLengths = inputs.lengths;
    TLLM_CHECK_WITH_INFO(inputLengths->getShape().nbDims == 1, "Input lengths tensor must be one-dimensional.");

    auto& manager = mRuntime->getBufferManager();

    auto const batchSize = static_cast<SizeType32>(inputLengths->getSize());

    auto const beamWidth = samplingConfig.beamWidth;
    outputs.ids->reshape(ITensor::makeShape({batchSize, beamWidth, mDecoderMaxSequenceLength}));
    outputs.lengths->reshape(ITensor::makeShape({batchSize, beamWidth}));
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        if (outputs.cumLogProbs)
        {
            TLLM_CHECK_WITH_INFO(outputs.cumLogProbs,
                "outputs.cumLogProbs is nullptr. It must be allocated when computeLogProbs is true");
            outputs.cumLogProbs->reshape(ITensor::makeShape({batchSize, beamWidth}));
        }
        if (outputs.logProbs)
        {
            TLLM_CHECK_WITH_INFO(
                outputs.logProbs, "outputs.logProbs is nullptr. It must be allocated when computeLogProbs is true");
            outputs.logProbs->reshape(ITensor::makeShape({batchSize, beamWidth, mDecoderMaxSequenceLength}));
        }
        if (mModelConfig.computeContextLogits() || mModelConfig.computeGenerationLogits())
        {
            auto const vocabSizePadded = mModelConfig.getVocabSizePadded(mWorldConfig.getSize());
            auto const inputLengthsHost = manager.copyFrom(*inputLengths, MemoryType::kCPU);
            auto const inputLengthsRange = BufferRange<SizeType32>(*inputLengthsHost);
            auto const maxInputLength = *std::max_element(inputLengthsRange.begin(), inputLengthsRange.end());

            if (mModelConfig.computeContextLogits())
            {
                if (!outputs.contextLogits)
                {
                    outputs.contextLogits = manager.emptyTensor(MemoryType::kGPU, getLogitDataType());
                }
                outputs.contextLogits->reshape(ITensor::makeShape({batchSize, maxInputLength, vocabSizePadded}));
            }

            // Initialize the output generation logits buffer
            if (mModelConfig.computeGenerationLogits())
            {
                SizeType32 maxNewTokens = 0;
                if (inputs.maxNewTokens)
                {
                    maxNewTokens = inputs.maxNewTokens.value();
                }
                else
                {
                    for (auto iter : inputLengthsRange)
                    {
                        maxNewTokens = std::max(maxNewTokens, mDecoderMaxSequenceLength - iter);
                    }
                }

                TLLM_CHECK_WITH_INFO(maxNewTokens, "maxNewTokens is null");

                if (!outputs.generationLogits)
                {
                    outputs.generationLogits = manager.emptyTensor(MemoryType::kGPU, getLogitDataType());
                }
                outputs.generationLogits->reshape(
                    ITensor::makeShape({batchSize, beamWidth, maxNewTokens, vocabSizePadded}));

                auto const generationLogitsShape = outputs.generationLogits->getShape();
                TLLM_CHECK_WITH_INFO(generationLogitsShape.d[0] == batchSize, "Invalid dim[0]");
                TLLM_CHECK_WITH_INFO(generationLogitsShape.d[1] == beamWidth, "Invalid dim[1]");
                TLLM_CHECK_WITH_INFO(generationLogitsShape.d[2] == maxNewTokens, "Invalid dim[2]");
                TLLM_CHECK_WITH_INFO(generationLogitsShape.d[3] == vocabSizePadded, "Invalid dim[3]");
            };
        }
    }

    // callbacks
    auto const onTokenGenerated = createOnTokenGeneratedCallback(outputs);
    if (batchSize <= mMicroBatchConfig.genBatchSize)
    {
        std::vector<GenerationInput> microBatchesInputs{inputs};
        std::vector<GenerationOutput> microBatchesOutputs{outputs};
        generateBatched(microBatchesOutputs, microBatchesInputs, samplingConfig, onTokenGenerated, generationProfiler);
    }
    else
    {
        auto const microBatchesInputs
            = splitInputs(inputs, mMicroBatchConfig.genBatchSize, manager, mModelConfig.getMaxNumTokens());
        auto microBatchesOutputs = splitOutputs(outputs, mMicroBatchConfig.genBatchSize, mWorldConfig);
        generateBatched(microBatchesOutputs, microBatchesInputs, samplingConfig, onTokenGenerated, generationProfiler);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

GptSession::TokenGeneratedCallback GptSession::createOnTokenGeneratedCallback(GenerationOutput& outputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (outputs.onTokenGenerated && mWorldConfig.isFirstPipelineParallelRank())
    {
        ITensor::SharedPtr outputIds{mWorldConfig.isPipelineParallel() || mMicroBatchConfig.numGenBatches > 1
                ? outputs.ids
                : mDecoders.front()->getIds()};
        return [onTokenGenerated = outputs.onTokenGenerated, outputIds = std::move(outputIds)](
                   SizeType32 step, bool finished) { onTokenGenerated(outputIds, step, finished); };
    }
    else
    {
        return [](SizeType32 step, bool finished) {};
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

bool GptSession::shouldUseKVCacheManager() const
{
    // KVCacheManager is only used for Transformer-based models with paged KV cache enabled.
    return mModelConfig.isTransformerBased() && mModelConfig.isPagedKVCache();
}

void GptSession::generateBatched(std::vector<GenerationOutput>& microBatchesOutputs,
    std::vector<GenerationInput> const& microBatchesInputs, SamplingConfig const& samplingConfig,
    TokenGeneratedCallback const& onTokenGenerated, std::shared_ptr<GenerationProfiler> const generationProfiler)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& manager = mRuntime->getBufferManager();
    TLLM_CHECK(microBatchesInputs.size() == microBatchesOutputs.size());
    auto const numMicroBatches = static_cast<SizeType32>(microBatchesInputs.size());
    TLLM_CHECK(numMicroBatches > 0);
    TLLM_CHECK(numMicroBatches <= mMicroBatchConfig.numGenBatches);
    SizeType32 const beamWidth{samplingConfig.beamWidth};

    auto* kvCacheManager = shouldUseKVCacheManager() ? mKvCacheManager.get() : nullptr;

    // Initialize and reshape buffers
    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto const& microBatchInputs = microBatchesInputs.at(microBatchId);
        auto& buffers = *mBuffers.at(microBatchId);
        buffers.initFromInput(*microBatchInputs.ids, microBatchInputs.lengths, microBatchInputs.packed, beamWidth,
            mDecoderMaxAttentionWindowVec, mDecoderMaxAttentionWindow, mDecoderSinkTokenLength,
            mDecoderMaxSequenceLength, manager);
        buffers.reshape(mModelConfig, mWorldConfig);
        buffers.reset(manager);
    }

    std::vector<SizeType32> microBatchOffsets(1, 0);
    microBatchOffsets.reserve(numMicroBatches + 1);
    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto const& generationConfig = mBuffers.at(microBatchId)->generationConfig;
        microBatchOffsets.emplace_back(microBatchOffsets.back() + generationConfig.batchSize);
    }

    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto& buffers = *mBuffers.at(microBatchId);
        auto const batchOffset = microBatchOffsets.at(microBatchId);
        kvCacheAddSequences(beamWidth, microBatchId, batchOffset);
        auto const& microBatchInputs = microBatchesInputs.at(microBatchId);
        auto& microBatchOutputs = microBatchesOutputs.at(microBatchId);
        buffers.outputIds = microBatchOutputs.ids;
        buffers.outputLengths = microBatchOutputs.lengths;
        buffers.newTokens
            = initDecoder(*buffers.outputIds, microBatchInputs, microBatchOutputs, samplingConfig, microBatchId);

        if (mWorldConfig.isLastPipelineParallelRank())
        {
            buffers.cumLogProbs = nullptr;
            if (microBatchOutputs.cumLogProbs)
            {
                buffers.cumLogProbs = microBatchOutputs.cumLogProbs;
            }
            buffers.logProbs = nullptr;
            if (microBatchOutputs.logProbs)
            {
                buffers.logProbs = microBatchOutputs.logProbs;
            }
            if (mModelConfig.computeContextLogits())
            {
                buffers.logits = microBatchOutputs.contextLogits;
            }
        }
        if (mModelConfig.usePromptTuning())
        {
            buffers.promptTuningParams = microBatchInputs.promptTuningParams;
        }
    }

    if (useCudaGraphs())
    {
        for (auto& instance : mCudaGraphInstances)
        {
            instance.clear();
        }
    }

    auto const profileContext = !kProfileMbIdxs.empty() && kProfileMbIdxs.count(0) > 0;
    if (profileContext)
        cudaProfilerStart();
    executeContextStep(microBatchesInputs, microBatchOffsets, kvCacheManager);
    if (profileContext)
        cudaProfilerStop();

    std::vector<bool> microBatchesFinished(numMicroBatches, false);
    SizeType32 numBatchesFinished{0};
    SizeType32 step{0};

    if (generationProfiler)
    {
        manager.getStream().record(generationProfiler->getStart());
    }

    while (numBatchesFinished < numMicroBatches)
    {
        ++step;

        auto const profileStep = !kProfileMbIdxs.empty() && kProfileMbIdxs.count(step) > 0;
        if (profileStep)
            cudaProfilerStart();

        numBatchesFinished += executeGenerationStep(
            step, microBatchesInputs, microBatchesOutputs, microBatchOffsets, kvCacheManager, microBatchesFinished);

        onTokenGenerated(step - 1, numBatchesFinished == numMicroBatches);

        if (profileStep)
            cudaProfilerStop();
    }

    if (generationProfiler)
    {
        manager.getStream().record(generationProfiler->getEnd());
    }

    // Collect the results for the last step
    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto const& generationConfig = mBuffers.at(microBatchId)->generationConfig;
        auto const microBatchSize = generationConfig.batchSize;

        auto const firstBatchIdx = microBatchOffsets.at(microBatchId);
        if (shouldUseKVCacheManager())
        {
            for (auto batchIdx = firstBatchIdx; batchIdx < firstBatchIdx + microBatchSize; ++batchIdx)
            {
                kvCacheManager->removeSequence(batchIdx);
            }
        }

        // TODO(micro batching) use mCommStream?
        if (beamWidth > 1)
        {
            finalize(microBatchId, samplingConfig);
        }
        else if (!mWorldConfig.isPipelineParallel())
        {
            auto& buffers = *mBuffers.at(microBatchId);
            auto& decoder = *mDecoders.at(microBatchId);
            manager.copy(*decoder.getIds(), *buffers.outputIds);

            auto& cumLogProbs = buffers.cumLogProbs;
            if (cumLogProbs)
            {
                manager.copy(*decoder.getCumLogProbs(), *buffers.cumLogProbs);
            }
            auto& logProbs = buffers.logProbs;
            if (logProbs)
            {
                manager.copy(*decoder.getLogProbs(), *buffers.logProbs);
            }
        }
        // copy generation logits fragments into a single generationLogits tensor
        if (mWorldConfig.isLastPipelineParallelRank() && mModelConfig.computeGenerationLogits())
        {
            auto& buffers = *mBuffers.at(microBatchId);
            auto& microBatchOutputs = microBatchesOutputs.at(microBatchId);

            auto const beamWidth = generationConfig.beamWidth;
            TensorPtr cachePointerDevice = ITensor::slice(buffers.cacheGenerationFragmentPointerDevice, 0, 1);
            TensorPtr cachePointerHost = ITensor::slice(buffers.cacheGenerationFragmentPointerHost, 0, 1);

            tensorrt_llm::runtime::kernels::mergeLogitsFragments(manager, *microBatchOutputs.generationLogits,
                *buffers.generationLogitsFragments, *cachePointerDevice, *cachePointerHost, 0, microBatchSize,
                beamWidth, manager.getStream(), 0);
            buffers.generationLogitsFragments->clear();
        }
    }

    manager.getStream().synchronize();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::executeContextStep(std::vector<GenerationInput> const& generationBatchesInputs,
    std::vector<SizeType32> const& generationBatchesOffsets, KvCacheManager const* kvCacheManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& manager = mRuntime->getBufferManager();

    auto allReduceCommPtrs = mAllReduceBuffers ? mAllReduceBuffers->mAllReduceCommPtrs : TensorPtr{};

    auto const numGenerationBatches = static_cast<SizeType32>(generationBatchesInputs.size());
    auto constexpr step = 0;
    auto constexpr contextId = 0;
    for (auto generationBatchId = 0; generationBatchId < numGenerationBatches; ++generationBatchId)
    {
        auto const& generationBatchInputs = generationBatchesInputs.at(generationBatchId);
        auto& generationBuffers = *mBuffers.at(generationBatchId);

        auto const contextBatchSize = mMicroBatchConfig.ctxBatchSize;
        auto [inputIds, inputLengths, contextBatchOffsets]
            = splitInputIds(generationBatchInputs, contextBatchSize, manager, mModelConfig.getMaxNumTokens());
        auto contextBuffers = generationBuffers.split(contextBatchSize, mModelConfig, mWorldConfig);
        TLLM_CHECK(inputIds.size() == contextBuffers.size());
        auto const numContextBatches = static_cast<SizeType32>(contextBuffers.size());

        for (auto contextBatchId = 0; contextBatchId < numContextBatches; ++contextBatchId)
        {
            auto batchOffset = generationBatchesOffsets.at(generationBatchId) + contextBatchOffsets.at(contextBatchId);
            auto& buffers = contextBuffers.at(contextBatchId);
            auto& inputBuffer = buffers.inputBuffers[0];
            auto& outputBuffer = buffers.outputBuffers[0];

            buffers.prepareContextStep(inputIds.at(contextBatchId), generationBatchInputs.padId, manager,
                kvCacheManager, batchOffset, mModelConfig, mWorldConfig);
            buffers.getRuntimeBuffers(inputBuffer, outputBuffer, step, inputIds.at(contextBatchId), allReduceCommPtrs,
                mModelConfig, mWorldConfig);
            mRuntime->setInputTensors(contextId, inputBuffer);
            mRuntime->setOutputTensors(contextId, outputBuffer);

            TLLM_CHECK_WITH_INFO(mRuntime->executeContext(contextId), "Executing TRT engine in context step failed!");
            sync_check_cuda_error();
            buffers.clearTensorMaps(); // inputBuffer and outputBuffer are not needed anymore, we explicitly clear them
                                       // to release memory
        }

        generationBuffers.postContextStep(contextBuffers, manager, mModelConfig, mWorldConfig);
        sync_check_cuda_error();

        // Save the last token logits of context into generation logits
        if (mWorldConfig.isLastPipelineParallelRank() && mModelConfig.computeGenerationLogits())
        {
            auto& buffers = *mBuffers.at(generationBatchId);
            buffers.generationLogitsFragments->push_back(generationBuffers.logits);
        }

        std::swap(generationBuffers.cacheIndirectionDecoderInput, generationBuffers.cacheIndirectionDecoderOutput);

        auto const decoderStep = generationBuffers.generationConfig.maxInputLength + step;

        decoderStepAsync(decoderStep, generationBatchId);

        if (mWorldConfig.isLastPipelineParallelRank() && mModelConfig.computeGenerationLogits())
        {
            TensorPtr newLogitBuffer = ITensor::slice(generationBuffers.allGenerationLogits, 1, 1);
            newLogitBuffer->squeeze(0);
            generationBuffers.logits = newLogitBuffer;
        }
    }
    if (mRuntime->hasLayerProfiler(contextId))
    {
        mRuntime->reportToProfiler(contextId);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

SizeType32 GptSession::executeGenerationStep(SizeType32 step, std::vector<GenerationInput> const& microBatchesInputs,
    std::vector<GenerationOutput>& microBatchesOutputs, std::vector<SizeType32> const& microBatchOffsets,
    KvCacheManager* kvCacheManager, std::vector<bool>& microBatchesFinished)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(microBatchesInputs.size() == microBatchesOutputs.size());
    auto& manager = mRuntime->getBufferManager();

    auto allReduceCommPtrs = mAllReduceBuffers ? mAllReduceBuffers->mAllReduceCommPtrs : TensorPtr{};

    auto const numMicroBatches = static_cast<SizeType32>(microBatchesInputs.size());
    SizeType32 numBatchesFinished{0};

    auto const flipFlopId = step % 2;
    auto const contextId = mRuntime->getNbProfiles() - 1;
    for (auto generationBatchId = 0; generationBatchId < numMicroBatches; ++generationBatchId)
    {
        if (microBatchesFinished.at(generationBatchId))
            continue;

        auto& buffers = *mBuffers.at(generationBatchId);
        auto const& generationConfig = buffers.generationConfig;

        auto const graphId = mMicroBatchConfig.getGenGraphId(flipFlopId, generationBatchId);
        auto& inputBuffer = buffers.inputBuffers[flipFlopId];
        auto& outputBuffer = buffers.outputBuffers[flipFlopId];

        auto nextInputIds = buffers.prepareNextStep(
            step - 1, manager, kvCacheManager, microBatchOffsets.at(generationBatchId), mModelConfig, mWorldConfig);
        buffers.getRuntimeBuffers(
            inputBuffer, outputBuffer, step, nextInputIds, allReduceCommPtrs, mModelConfig, mWorldConfig);
        mRuntime->setInputTensors(contextId, inputBuffer);
        mRuntime->setOutputTensors(contextId, outputBuffer);

        if (useCudaGraphs())
        {
            // RNNBased only create cuda graph instance once.
            if (mModelConfig.isRnnBased())
            {
                // step > 3 is WAR for TRT 9.x.
                // Which is fixed in TRT 10.x.
                if (step > 3 && !mCudaGraphInstances.at(graphId).hasInstance())
                {
                    mCudaGraphInstances.at(graphId).prepareNextGraph(*mRuntime, contextId);
                }
            }
            else
            {
                mCudaGraphInstances.at(graphId).prepareNextGraph(*mRuntime, contextId);
            }
        }

        // check decoder result of previous iteration
        if (shouldStopSync(generationConfig.batchSize, generationConfig.beamWidth, generationBatchId))
        {
            mLogger->log(nvinfer1::ILogger::Severity::kVERBOSE,
                tc::fmtstr("GPT decoding finished for step %d and microBatchId %d", step, generationBatchId).c_str());
            microBatchesFinished.at(generationBatchId) = true;
            numBatchesFinished += 1;
            continue;
        }

        // report profile data
        if (mRuntime->hasLayerProfiler(contextId))
        {
            mRuntime->reportToProfiler(contextId);
        }

        if (useCudaGraphs() && mCudaGraphInstances.size() > (size_t) graphId
            && mCudaGraphInstances.at(graphId).hasInstance())
        {
            auto& cudaGraphInstance = mCudaGraphInstances.at(graphId);
            cudaGraphInstance.launch(mRuntime->getStream());
        }
        else
        {
            TLLM_CHECK_WITH_INFO(
                mRuntime->executeContext(contextId), tc::fmtstr("Executing TRT engine in step %d failed!", step));
        }
        sync_check_cuda_error();

        if (mWorldConfig.isLastPipelineParallelRank() && mModelConfig.computeGenerationLogits())
        {
            auto& buffers = *mBuffers.at(generationBatchId);
            buffers.generationLogitsFragments->push_back(buffers.logits);
        }
        sync_check_cuda_error();

        std::swap(buffers.cacheIndirectionDecoderInput, buffers.cacheIndirectionDecoderOutput);

        auto const decoderStep = generationConfig.maxInputLength + step;

        decoderStepAsync(decoderStep, generationBatchId);

        if (mWorldConfig.isLastPipelineParallelRank() && mModelConfig.computeGenerationLogits()
            && buffers.allGenerationLogits->getShape().d[0] > step + 1)
        {
            TensorPtr newLogitBuffer = ITensor::slice(buffers.allGenerationLogits, step + 1, 1);
            newLogitBuffer->squeeze(0);
            buffers.logits = newLogitBuffer;
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return numBatchesFinished;
}

void GptSession::decoderStepAsync(SizeType32 decoderStep, SizeType32 microBatchId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& stream = mRuntime->getStream();
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
                mPipelineComm->send(*decoder.getNbFinished(), pipelineGroup[peerIdx], *mCommStream);
                if (beamWidth > 1)
                {
                    mPipelineComm->send(cacheIndirection, pipelineGroup[peerIdx], *mCommStream);
                }
                mPipelineComm->send(sequenceLengths, pipelineGroup[peerIdx], *mCommStream);
            }
            mPipelineComm->send(*decoder.getNewTokens(), pipelineGroup.front(), *mCommStream);
        }
    }
    else // pipeline parallel mode
    {    // receive shouldStop from the last rank on a separate stream
        stream.record(mCommEvent.get());
        mCommStream->wait(mCommEvent.get());
        auto const pipelineGroup = mWorldConfig.getPipelineParallelGroup();
        auto const peer = pipelineGroup.back();
        mPipelineComm->receive(*buffers.nbFinished, peer, *mCommStream);

        auto& cacheIndirection = *buffers.cacheIndirectionDecoderOutput;
        auto& sequenceLengths = *buffers.sequenceLengths;
        auto const beamWidth = cacheIndirection.getShape().d[1];
        if (beamWidth > 1)
        {
            mPipelineComm->receive(cacheIndirection, peer, *mCommStream);
        }
        mPipelineComm->receive(sequenceLengths, peer, *mCommStream);
        if (mWorldConfig.isFirstPipelineParallelRank())
        { // receive newTokens from last rank on a separate stream
            mPipelineComm->receive(*newTokens, peer, *mCommStream);
            updateOutputIds(outputIds, newTokens, decoderStep, *mCommStream);
        }
        mCommStream->record(mReceivedEvents.at(microBatchId).get());
    }

    if (!mWorldConfig.isPipelineParallel() && mMicroBatchConfig.numGenBatches > 1)
    {
        updateOutputIds(outputIds, newTokens, decoderStep, stream);
        stream.record(mReceivedEvents.at(microBatchId).get());
    }

    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

bool GptSession::shouldStopSync(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 microBatchId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 nbFinished = 0;

    if (mWorldConfig.isLastPipelineParallelRank())
    { // read the Finished flag from the decoder
        auto& decoder = *mDecoders.at(microBatchId);
        decoder.forwardSync();
        nbFinished = *bufferCast<SizeType32>(*decoder.getNbFinished());

        if (!mWorldConfig.isPipelineParallel() && mMicroBatchConfig.numGenBatches > 1)
        {
            // ensure outputIds have been updated
            mReceivedEvents.at(microBatchId).synchronize();
        }
    }
    else
    { // ensure all information has been received
        mReceivedEvents.at(microBatchId).synchronize();
        nbFinished = *bufferCast<SizeType32>(*mBuffers.at(microBatchId)->nbFinished);
    }
    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return nbFinished == batchSize * beamWidth;
}

void GptSession::finalize(SizeType32 microBatchId, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& manager = mRuntime->getBufferManager();
    auto& buffers = *mBuffers.at(microBatchId);
    auto& outputIds = buffers.outputIds;
    auto& cumLogProbs = buffers.cumLogProbs;
    auto& logProbs = buffers.logProbs;
    auto& sequenceLengths = buffers.sequenceLengths;

    if (mWorldConfig.isPipelineParallel())
    {
        auto& stream = mRuntime->getStream();
        auto const pipelineGroup = mWorldConfig.getPipelineParallelGroup();

        if (mWorldConfig.isLastPipelineParallelRank())
        {                                               // send ids from last to first
            auto& decoder = mDecoders.at(microBatchId); // only the last rank has the decoder
            decoder->finalize(samplingConfig);
            auto finalOutputIds = decoder->getGatheredIds();

            auto const peer = pipelineGroup.front();
            mPipelineComm->send(*finalOutputIds, peer, stream);
            mPipelineComm->send(*sequenceLengths, peer, stream);
            manager.copy(*finalOutputIds, *outputIds);

            if (cumLogProbs)
            {
                auto finalCumLogProbs = decoder->getCumLogProbs();
                mPipelineComm->send(*finalCumLogProbs, peer, stream);
                manager.copy(*finalCumLogProbs, *cumLogProbs);
            }
            if (logProbs)
            {
                auto finalLogProbs = decoder->getLogProbs();
                mPipelineComm->send(*finalLogProbs, peer, stream);
                manager.copy(*finalLogProbs, *logProbs);
            }
        }
        else if (mWorldConfig.isFirstPipelineParallelRank())
        { // receive ids from last on first
            auto const peer = pipelineGroup.back();
            mPipelineComm->receive(*outputIds, peer, stream);
            mPipelineComm->receive(*sequenceLengths, peer, stream);
            if (cumLogProbs)
            {
                mPipelineComm->receive(*cumLogProbs, peer, stream);
            }
            if (logProbs)
            {
                mPipelineComm->receive(*logProbs, peer, stream);
            }
        }
    }
    else
    {
        auto& decoder = mDecoders.at(microBatchId); // all ranks have the decoder
        decoder->finalize(samplingConfig);
        auto finalOutputIds = decoder->getGatheredIds();
        manager.copy(*finalOutputIds, *outputIds);
        if (cumLogProbs)
        {
            auto finalCumLogProbs = decoder->getCumLogProbs();
            manager.copy(*finalCumLogProbs, *cumLogProbs);
        }
        if (logProbs)
        {
            auto finalLogProbs = decoder->getLogProbs();
            manager.copy(*finalLogProbs, *logProbs);
        }
        // sequenceLengths are already updated by decoder
    }

    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::setLayerProfiler()
{
    TLLM_CHECK(mRuntime);
    mRuntime->setLayerProfiler();
}

std::string GptSession::getLayerProfileInfo() const
{
    TLLM_CHECK(mRuntime);
    return mRuntime->getLayerProfileInfo();
}

void GptSession::CudaGraphExecutor::create(cudaGraph_t const& graph)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    assert(mInstance == nullptr);
    TLLM_CUDA_CHECK(cudaGraphInstantiate(&mInstance, graph, nullptr, nullptr, 0));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::CudaGraphExecutor::uploadToStream(CudaStream const& stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    assert(hasInstance());
    TLLM_CUDA_CHECK(cudaGraphUpload(mInstance, stream.get()));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::CudaGraphExecutor::launch(CudaStream const& stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CUDA_CHECK(cudaGraphLaunch(mInstance, stream.get()));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

bool GptSession::CudaGraphExecutor::update(cudaGraph_t const& graph)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    return cudaGraphExecUpdate(mInstance, graph, nullptr) != cudaSuccess;
}

void GptSession::CudaGraphExecutor::clear()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (mInstance != nullptr)
    {
        TLLM_CUDA_CHECK(cudaGraphExecDestroy(mInstance));
        mInstance = nullptr;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptSession::CudaGraphExecutor::prepareNextGraph(TllmRuntime const& runtime, SizeType32 nextContextId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
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
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

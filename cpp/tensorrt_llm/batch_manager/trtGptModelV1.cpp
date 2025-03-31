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

#include "trtGptModelV1.h"

#include "promptTuningBuffers.h"
#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <algorithm>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::batch_manager
{

bool TrtGptModelV1::optionalParamsAreValid(
    runtime::ModelConfig const& modelConfig, TrtGptModelOptionalParams const& optionalParams)
{
    // Make sure logic in this function matches fixOptionalParams
    if (optionalParams.kvCacheConfig.enableBlockReuse)
    {
        return false;
    }
    if (optionalParams.schedulerConfig.getCapacitySchedulerPolicy()
        != executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
    {
        return false;
    }
    return true;
}

TrtGptModelOptionalParams TrtGptModelV1::fixOptionalParams(
    runtime::ModelConfig const& modelConfig, TrtGptModelOptionalParams const& optionalParams)
{
    // Make sure logic in this function matches optionalParamsAreValid
    auto fixedOptionalParams = TrtGptModelOptionalParams(optionalParams);
    if (fixedOptionalParams.kvCacheConfig.enableBlockReuse)
    {
        TLLM_LOG_WARNING("Fix optionalParams : KV cache reuse disabled because V1 does not support it");
        fixedOptionalParams.kvCacheConfig.enableBlockReuse = false;
    }
    if (fixedOptionalParams.schedulerConfig.getCapacitySchedulerPolicy()
        != executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
    {
        TLLM_LOG_WARNING(
            "Fix optionalParams : Changed scheduler policy to GUARANTEED_NO_EVICT because it is the only one supported "
            "by V1");
        fixedOptionalParams.schedulerConfig
            = executor::SchedulerConfig(executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
                optionalParams.schedulerConfig.getContextChunkingPolicy(),
                optionalParams.schedulerConfig.getDynamicBatchConfig());
    }
    return fixedOptionalParams;
}

namespace
{
SamplingConfig initBatchSamplingConfig(SamplingConfig const& baseSamplingConfig)
{
    SamplingConfig batchSamplingConfig{baseSamplingConfig.beamWidth};

    auto initOptional = [](auto& batch, auto const& base)
    {
        using T = typename std::remove_reference_t<decltype(base)>::value_type;
        if (base)
            batch.emplace(T{});
    };

    initOptional(batchSamplingConfig.temperature, baseSamplingConfig.temperature);
    initOptional(batchSamplingConfig.minLength, baseSamplingConfig.minLength);
    initOptional(batchSamplingConfig.repetitionPenalty, baseSamplingConfig.repetitionPenalty);
    initOptional(batchSamplingConfig.presencePenalty, baseSamplingConfig.presencePenalty);
    initOptional(batchSamplingConfig.frequencyPenalty, baseSamplingConfig.frequencyPenalty);
    initOptional(batchSamplingConfig.noRepeatNgramSize, baseSamplingConfig.noRepeatNgramSize);
    // sampling layers
    initOptional(batchSamplingConfig.topK, baseSamplingConfig.topK);
    initOptional(batchSamplingConfig.topP, baseSamplingConfig.topP);
    initOptional(batchSamplingConfig.randomSeed, baseSamplingConfig.randomSeed);
    initOptional(batchSamplingConfig.topPDecay, baseSamplingConfig.topPDecay);
    initOptional(batchSamplingConfig.topPMin, baseSamplingConfig.topPMin);
    initOptional(batchSamplingConfig.topPResetIds, baseSamplingConfig.topPResetIds);

    // beam search layer
    batchSamplingConfig.beamSearchDiversityRate = baseSamplingConfig.beamSearchDiversityRate;
    batchSamplingConfig.lengthPenalty = baseSamplingConfig.lengthPenalty;
    batchSamplingConfig.earlyStopping = baseSamplingConfig.earlyStopping;

    batchSamplingConfig.normalizeLogProbs = baseSamplingConfig.normalizeLogProbs;

    return batchSamplingConfig;
}

void addToSamplingConfig(SamplingConfig& batchSamplingConfig, SamplingConfig const& addSamplingConfig)
{
    TLLM_CHECK(batchSamplingConfig.beamWidth == addSamplingConfig.beamWidth);
    TLLM_CHECK(batchSamplingConfig.beamSearchDiversityRate == addSamplingConfig.beamSearchDiversityRate);
    TLLM_CHECK(batchSamplingConfig.lengthPenalty == addSamplingConfig.lengthPenalty);
    TLLM_CHECK(batchSamplingConfig.earlyStopping == addSamplingConfig.earlyStopping);
    TLLM_CHECK(batchSamplingConfig.beamWidthArray == addSamplingConfig.beamWidthArray);

    auto addOptional = [](auto& batch, auto const& add, char const* name)
    {
        if (batch)
        {
            TLLM_CHECK_WITH_INFO(add, "Sampling configs have different optional %s.", name);
            batch->push_back(add->at(0));
        }
        else
        {
            TLLM_CHECK_WITH_INFO(!add, "Sampling configs have different optional %s.", name);
        }
    };

    addOptional(batchSamplingConfig.temperature, addSamplingConfig.temperature, "temperature");
    addOptional(batchSamplingConfig.minLength, addSamplingConfig.minLength, "minLength");
    addOptional(batchSamplingConfig.repetitionPenalty, addSamplingConfig.repetitionPenalty, "repetitionPenalty");
    addOptional(batchSamplingConfig.presencePenalty, addSamplingConfig.presencePenalty, "presencePenalty");
    addOptional(batchSamplingConfig.frequencyPenalty, addSamplingConfig.frequencyPenalty, "frequencyPenalty");
    addOptional(batchSamplingConfig.noRepeatNgramSize, addSamplingConfig.noRepeatNgramSize, "noRepeatNgramSize");

    addOptional(batchSamplingConfig.topK, addSamplingConfig.topK, "topK");
    addOptional(batchSamplingConfig.topP, addSamplingConfig.topP, "topP");
    addOptional(batchSamplingConfig.randomSeed, addSamplingConfig.randomSeed, "randomSeed");
    addOptional(batchSamplingConfig.topPDecay, addSamplingConfig.topPDecay, "topPDecay");
    addOptional(batchSamplingConfig.topPMin, addSamplingConfig.topPMin, "topPMin");
    addOptional(batchSamplingConfig.topPResetIds, addSamplingConfig.topPResetIds, "topPResetIds");
}

bool emptyWordsList(GenerationInput::TensorPtr const& wordsList)
{
    auto shape = wordsList->getShape();
    if (shape.nbDims != 3 || shape.d[0] != 1 || shape.d[1] != 2)
    {
        TLLM_THROW("Unexpected shape for badWords tensor, expecting shape [1, 2, length]");
    }

    bool emptyWordsList = false;
    if (shape.d[2] <= 0)
    {
        emptyWordsList = true;
    }
    // If only one offset, check if zero. If so ignore, consider empty
    else if (shape.d[2] == 1)
    {
        auto wordsListPtr = bufferCast<int32_t>(*wordsList);
        int32_t offset = wordsListPtr[1];
        emptyWordsList = (offset <= 0);
    }
    else
    {
        emptyWordsList = false;
    }

    return emptyWordsList;
}

GenerationInput::TensorPtr fillWordsTensor(
    std::vector<std::optional<GenerationInput::TensorPtr>> const& optWordsLists, runtime::BufferManager const& manager)
{
    using TensorPtr = GenerationInput::TensorPtr;

    bool hasWords = false;
    SizeType32 maxWordsLength = 0;
    SizeType32 batchSize = optWordsLists.size();
    std::vector<bool> emptyWordsLists(batchSize, true);

    for (std::size_t bid = 0; bid < optWordsLists.size(); ++bid)
    {
        auto const& optWordsList = optWordsLists[bid];
        if (optWordsList.has_value())
        {
            auto wordsList = optWordsList.value();
            if (wordsList && !emptyWordsList(wordsList))
            {
                hasWords = true;
                emptyWordsLists[bid] = false;

                auto length = wordsList->getShape().d[2];
                maxWordsLength = length > maxWordsLength ? length : maxWordsLength;
            }
        }
    }
    if (!hasWords)
    {
        return nullptr;
    }

    TensorPtr output = manager.gpu(ITensor::makeShape({batchSize, 2, maxWordsLength}), nvinfer1::DataType::kINT32);

    // Fill with -1, special value used to indicate end of offset
    runtime::kernels::invokeFill<std::int32_t>(*output, -1, manager.getStream());

    // TODO: If perf critical, use one kernel to do multiple copies below
    for (std::size_t bid = 0; bid < optWordsLists.size(); ++bid)
    {
        auto const& optWordsList = optWordsLists[bid];
        if (!emptyWordsLists.at(bid))
        {
            auto wordsList = optWordsList.value();
            auto length = wordsList->getShape().d[2];

            // Create an input/output view for that batch entry, removing batch dim
            TensorPtr outputView = ITensor::slice(output, bid, 1);
            outputView->squeeze(0);
            TensorPtr inputView = ITensor::slice(wordsList, 0, 1);
            inputView->squeeze(0);

            for (int row = 0; row < 2; ++row)
            {
                TensorPtr inputRowView = ITensor::slice(inputView, row, 1);
                TensorPtr outputRowView = ITensor::slice(outputView, row, 1);
                outputRowView->reshape(ITensor::makeShape({1, length}));
                manager.copy(*inputRowView, *outputRowView);
            }
        }
    }
    return output;
}

GenerationInput::TensorPtr fillEmbeddingBias(
    std::vector<std::optional<GenerationInput::TensorPtr>> const& optEmbeddingBiases,
    BufferManager const& bufferManager)
{
    using TensorPtr = GenerationInput::TensorPtr;

    TensorPtr output = nullptr;
    // TODO:
    for (auto const& optEmbeddingBias : optEmbeddingBiases)
    {
        if (optEmbeddingBias.has_value())
        {
            TLLM_THROW("V1 doesn't support embedding_bias tensor yet.");
        }
    }

    return output;
}

} // namespace

TrtGptModelV1::TrtGptModelV1(std::shared_ptr<nvinfer1::ILogger> logger, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, RawEngine const& rawEngine, TrtGptModelOptionalParams const& optionalParams)
    : TrtGptModel(modelConfig, worldConfig, optionalParams)
    , mPeftCacheManager{std::make_shared<NoOpPeftCacheManager>()}
{
    mPpTimesMaxBatchSize = worldConfig.getPipelineParallelism() * getMaxBatchSize();

    runtime::GptSession::Config sessionConfig{mPpTimesMaxBatchSize, getMaxBeamWidth(), getMaxSequenceLen()};
    sessionConfig.decoderPerRequest = true;
    sessionConfig.kvCacheConfig = optionalParams.kvCacheConfig;
    sessionConfig.normalizeLogProbs = optionalParams.normalizeLogProbs;
    sessionConfig.gpuWeightsPercent = optionalParams.gpuWeightsPercent;
    sessionConfig.cudaGraphMode = true;
    sessionConfig.gatherGenerationLogits = optionalParams.gatherGenerationLogits;

    mSession = std::make_shared<GptSession>(sessionConfig, modelConfig, worldConfig, rawEngine, logger);

    // Here we use the microBatchScheduler but we don't need any microBatching capabilities
    // Since micro batches are created in gptSession
    std::optional<SizeType32> maxNumTokens = modelConfig.getMaxNumTokens();
    TLLM_CHECK_WITH_INFO(maxNumTokens, "Max number of tokens is not set.");

    mCapacityScheduler = std::make_unique<tensorrt_llm::batch_manager::CapacityScheduler>(mPpTimesMaxBatchSize,
        optionalParams.schedulerConfig.getCapacitySchedulerPolicy(), static_cast<bool>(mSession->mKvCacheManager));

    mMicroBatchScheduler = std::make_unique<tensorrt_llm::batch_manager::MicroBatchScheduler>();
}

runtime::ModelConfig const& TrtGptModelV1::getModelConfig() const
{
    return mSession->getModelConfig();
}

[[nodiscard]] bool TrtGptModelV1::getGatherGenerationLogits() const
{
    return mSession->getGatherGenerationLogits();
}

[[nodiscard]] TrtGptModelV1::IterationStatsV1 TrtGptModelV1::getLastIterationStats() const
{
    return mLastIterationStatsV1;
}

runtime::WorldConfig const& TrtGptModelV1::getWorldConfig() const
{
    return mSession->getWorldConfig();
}

runtime::BufferManager const& TrtGptModelV1::getBufferManager() const
{
    return mSession->getBufferManager();
}

runtime::BufferManager::CudaStreamPtr TrtGptModelV1::getRuntimeStreamPtr() const
{
    return mSession->getRuntimeStreamPtr();
}

SizeType32 TrtGptModelV1::getNumMicroBatches() const
{
    return 1;
}

nvinfer1::DataType TrtGptModelV1::getLogitDataType() const
{
    return mSession->getLogitDataType();
}

TrtGptModelV1::IterationStatsV1 TrtGptModelV1::fillIterationStats(
    RequestVector const& scheduledRequests, SizeType32 cappedMaxNewTokens, RequestVector const& requestsToTerminate)
{
    IterationStatsV1 iterationStatsV1;
    iterationStatsV1.numCtxTokensInBatch = 0;
    iterationStatsV1.numGenTokensInBatch = 0;

    for (auto const& llmReq : scheduledRequests)
    {
        auto requestMaxNewTokens = std::min(llmReq->mMaxNewTokens, cappedMaxNewTokens);
        iterationStatsV1.numCtxTokensInBatch += llmReq->getNumTokens(0);
        iterationStatsV1.numGenTokensInBatch += requestMaxNewTokens;
        iterationStatsV1.scheduledRequests.insert(llmReq->mRequestId);
    }

    iterationStatsV1.numScheduledRequests = scheduledRequests.size();
    iterationStatsV1.emptyGenSlots
        = (scheduledRequests.size() * cappedMaxNewTokens) - iterationStatsV1.numGenTokensInBatch;

    for (auto const& llmReq : requestsToTerminate)
    {
        iterationStatsV1.pausedRequests.insert(llmReq->mRequestId);
    }

    return iterationStatsV1;
}

std::tuple<runtime::GenerationInput, runtime::SamplingConfig> TrtGptModelV1::fillGenInputAndSamplingConfig(
    RequestVector const& scheduledRequests, BufferManager const& bufferManager, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, SizeType32 maxSeqLen, SizeType32 maxBatchSize, bool normalizeLogProbs)
{
    SizeType32 batchSize = scheduledRequests.size();
    auto const& firstLlmReq = scheduledRequests.front();
    auto const firstEndId = firstLlmReq->mEndId;
    auto const firstPadId = firstLlmReq->mPadId;
    firstLlmReq->mSamplingConfig.normalizeLogProbs = normalizeLogProbs;
    auto batchSamplingConfig = initBatchSamplingConfig(firstLlmReq->mSamplingConfig);

    std::vector<std::vector<TokenIdType>> inputTokens;
    inputTokens.reserve(batchSize);
    std::vector<SizeType32> inputSeqLengths;
    inputSeqLengths.reserve(batchSize);
    for (auto const& llmReq : scheduledRequests)
    {
        TLLM_CHECK_WITH_INFO(llmReq->hasDraftTokens() == false,
            "Speculative decoding is not supported in V1. Use inflight batching mode.");
        inputTokens.push_back(llmReq->getTokens(0));
        inputSeqLengths.push_back(llmReq->getNumTokens(0));
        TLLM_CHECK(llmReq->mEndId == firstEndId);
        TLLM_CHECK(llmReq->mPadId == firstPadId);
        addToSamplingConfig(batchSamplingConfig, llmReq->mSamplingConfig);
    }

    SizeType32 endId = firstEndId.value_or(-1);
    SizeType32 padId = firstPadId.value_or(-1);

    SizeType32 const maxInputLength = *std::max_element(inputSeqLengths.begin(), inputSeqLengths.end());

    // Now compute maxNewTokens
    // If a request has maxNewTokens > maxSeqLen - maxInputLength, cap it
    SizeType32 maxNewTokens = 0;
    for (auto const& llmReq : scheduledRequests)
    {
        auto requestMaxNewTokens = llmReq->mMaxNewTokens;
        if (requestMaxNewTokens > (maxSeqLen - maxInputLength))
        {
            TLLM_LOG_WARNING(
                "Requested number of tokens exceeds maxSeqLen - maxInputLength, setting to maxSequenceLength");
            requestMaxNewTokens = maxSeqLen - maxInputLength;
        }

        if (requestMaxNewTokens > maxNewTokens)
        {
            maxNewTokens = requestMaxNewTokens;
        }
    }

    auto inputLengths = bufferManager.copyFrom(inputSeqLengths, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    auto const usePackedInput = modelConfig.usePackedInput();
    GenerationInput::TensorPtr inputIds;
    if (usePackedInput)
    {
        std::vector<SizeType32> inputOffsetsHost(batchSize + 1);
        tc::stl_utils::inclusiveScan(inputSeqLengths.begin(), inputSeqLengths.end(), inputOffsetsHost.begin() + 1);
        auto const totalInputSize = inputOffsetsHost.back();

        std::vector<TokenIdType> inputsHost(totalInputSize);
        for (SizeType32 i = 0; i < batchSize; ++i)
        {
            std::copy(inputTokens[i].begin(), inputTokens[i].end(), inputsHost.begin() + inputOffsetsHost[i]);
        }
        inputIds = bufferManager.copyFrom(inputsHost, ITensor::makeShape({totalInputSize}), MemoryType::kGPU);
    }
    else
    {
        std::vector<TokenIdType> inputsHost(batchSize * maxInputLength, padId);
        for (SizeType32 i = 0; i < batchSize; ++i)
        {
            std::copy(inputTokens[i].begin(), inputTokens[i].end(), inputsHost.begin() + i * maxInputLength);
        }
        inputIds
            = bufferManager.copyFrom(inputsHost, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);
    }

    GenerationInput generationInput{endId, padId, std::move(inputIds), std::move(inputLengths), usePackedInput};
    generationInput.maxNewTokens = maxNewTokens;

    std::vector<std::optional<GenerationInput::TensorPtr>> optBadWordsLists, optStopWordsLists, optBiasTensors;
    for (auto const& llmReq : scheduledRequests)
    {
        optBadWordsLists.push_back(llmReq->getBadWordsList());
        optStopWordsLists.push_back(llmReq->getStopWordsList());
        optBiasTensors.push_back(llmReq->getEmbeddingBias());
    }

    generationInput.badWordsList = fillWordsTensor(optBadWordsLists, bufferManager);
    generationInput.stopWordsList = fillWordsTensor(optStopWordsLists, bufferManager);
    generationInput.embeddingBias = fillEmbeddingBias(optBiasTensors, bufferManager);

    // Create a prompt embedding table from the request embedding tables
    if (modelConfig.usePromptTuning())
    {
        // Create the prompt table from the requests individual prompt tables
        auto promptTuningBuffers = PromptTuningBuffers(maxBatchSize, bufferManager, modelConfig, worldConfig);
        promptTuningBuffers.fill(scheduledRequests, {}, bufferManager, false);

        generationInput.promptTuningParams = promptTuningBuffers.mPromptTuningParams;

        // Reshape tasks from [batchSize, 1] to [batchSize] which is what GenerationInput expects
        auto const& tasks = generationInput.promptTuningParams.tasks;
        auto const batchSize = tasks->getShape().d[0];
        tasks->reshape(ITensor::makeShape({batchSize}));
    }
    return {generationInput, batchSamplingConfig};
}

void TrtGptModelV1::resetIterationStats()
{
    mLastIterationStatsV1 = fillIterationStats({}, 0, {});
}

void TrtGptModelV1::forwardSync() {}

void TrtGptModelV1::forwardAsync(RequestList const& activeRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const device = getWorldConfig().getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));

    auto [fittingRequests, fittingDisaggGenInitRequests, pausedRequests]
        = (*mCapacityScheduler)(activeRequests, mSession->mKvCacheManager);
    TLLM_CHECK_WITH_INFO(fittingDisaggGenInitRequests.empty(), "Disaggregated servering is not support by V1 batcher.");

    auto [scheduledRequests, genRequests]
        = (*mMicroBatchScheduler)(fittingRequests, {}, mPpTimesMaxBatchSize, getModelConfig().getMaxNumTokens());

    TLLM_CHECK(genRequests.empty());

    if (scheduledRequests.empty())
    {
        return;
    }

    auto const& bufferManager = mSession->getBufferManager();

    auto [generationInput, batchSamplingConfig] = fillGenInputAndSamplingConfig(scheduledRequests, bufferManager,
        getModelConfig(), getWorldConfig(), getMaxSequenceLen(), getMaxBatchSize(), mSession->getNormalizeLogProbs());
    auto maxNewTokens = generationInput.maxNewTokens.value_or(0);

    // Assign callback iteration stats
    mLastIterationStatsV1 = fillIterationStats(scheduledRequests, maxNewTokens, pausedRequests);

    GenerationOutput generationOutput{bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
        bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};

    if (getWorldConfig().isLastPipelineParallelRank())
    {
        for (auto const& llmReq : scheduledRequests)
        {
            if (llmReq->returnLogProbs())
            {
                generationOutput.cumLogProbs = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
                generationOutput.logProbs = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
                break;
            }
            TLLM_CHECK_WITH_INFO(llmReq->mLogitsPostProcessor == std::nullopt,
                "Logits processor callback isn't supported with V1 batcher.");
            TLLM_CHECK_WITH_INFO(!llmReq->mApplyLogitsPostProcessorBatched,
                "Batched logits processor callback isn't supported with V1 batcher.");
        }
    }

    SizeType32 numSteps = 0;
    generationOutput.onTokenGenerated = [&numSteps]([[maybe_unused]] GenerationOutput::TensorPtr const& outputIds,
                                            [[maybe_unused]] SizeType32 step, bool finished) { ++numSteps; };

    mSession->generate(generationOutput, generationInput, batchSamplingConfig);

    auto const& outputIds = generationOutput.ids;
    auto const& outputDims = outputIds->getShape();

    auto outputHost = bufferManager.copyFrom(*outputIds, MemoryType::kPINNEDPOOL);
    auto output = bufferCast<TokenIdType>(*outputHost);
    auto outputLengthsHost = bufferManager.copyFrom(*generationOutput.lengths, MemoryType::kPINNEDPOOL);
    auto outputLengths = bufferCast<SizeType32>(*outputLengthsHost);

    float* cumLogProbsHostData = nullptr;
    float* logProbsHostData = nullptr;
    GenerationOutput::TensorPtr cumLogProbsHost, logProbsHost;
    if (generationOutput.cumLogProbs)
    {
        cumLogProbsHost = bufferManager.copyFrom(*generationOutput.cumLogProbs, MemoryType::kPINNEDPOOL);
        cumLogProbsHostData = bufferCast<float>(*cumLogProbsHost);
    }
    if (generationOutput.logProbs)
    {
        logProbsHost = bufferManager.copyFrom(*generationOutput.logProbs, MemoryType::kPINNEDPOOL);
        logProbsHostData = bufferCast<float>(*logProbsHost);
    }
    bufferManager.getStream().synchronize();

    int bid = 0;
    for (auto const& llmReq : scheduledRequests)
    {
        auto const inputLength = llmReq->mPromptLen;
        std::vector<std::vector<int32_t>> generatedTokens(batchSamplingConfig.beamWidth);
        for (SizeType32 beam = 0; beam < batchSamplingConfig.beamWidth; ++beam)
        {
            auto const lengthsIndex = tc::flat_index2(bid, beam, batchSamplingConfig.beamWidth);
            auto const numGeneratedTokens = std::min(outputLengths[lengthsIndex] - inputLength, llmReq->mMaxNewTokens);
            auto const beginIndex = tc::flat_index3(
                bid, beam, inputLength, batchSamplingConfig.beamWidth, static_cast<SizeType32>(outputDims.d[2]));
            auto const endIndex = beginIndex + numGeneratedTokens;
            generatedTokens[beam].assign(output + beginIndex, output + endIndex);

            if (llmReq->returnLogProbs())
            {
                llmReq->setCumLogProb(cumLogProbsHostData[bid * getMaxBeamWidth() + beam], beam);

                auto const offset = (bid * getMaxBeamWidth() + beam) * getMaxSequenceLen();
                std::vector<float> logProbs(logProbsHostData + offset + inputLength,
                    logProbsHostData + offset + inputLength + numGeneratedTokens);
                llmReq->setLogProbs(logProbs, beam);
            }
        }

        llmReq->setGeneratedTokens(generatedTokens);
        llmReq->setState(LlmRequestState::kGENERATION_COMPLETE);
        bid++;
    }
    outputIds->release();

    // Save context logits
    if (mSession->getModelConfig().computeContextLogits())
    {
        TLLM_CHECK(mSession->getModelConfig().usePackedInput());

        // generationOutput.contextLogits shape: [packedSize, vocabSize]
        SizeType32 inputOffset = 0;
        for (auto const& llmReq : scheduledRequests)
        {
            if (llmReq->getReturnContextLogits())
            {
                TensorPtr contextLogitsView
                    = ITensor::slice(generationOutput.contextLogits, inputOffset, llmReq->mPromptLen);

                TensorPtr contextLogitsHost = bufferManager.copyFrom(*contextLogitsView, MemoryType::kPINNEDPOOL);
                llmReq->setContextLogitsHost(contextLogitsHost);
            }
            inputOffset += llmReq->mPromptLen;
        }
    }

    // Save generation logits
    if (mSession->getGatherGenerationLogits())
    {
        auto const& generationLogitsShape = generationOutput.generationLogits->getShape();
        TLLM_CHECK_WITH_INFO(generationOutput.generationLogits->getShape().d[0] == int(scheduledRequests.size()),
            "Mismatch output generation logit batch size, different with request list size");

        auto const beamWidth = generationLogitsShape.d[1];
        auto const maxOutputLen = generationLogitsShape.d[2];
        auto const vocabSizePadded = generationLogitsShape.d[3];

        SizeType32 reqOffset = 0;
        for (auto const& llmReq : scheduledRequests)
        {
            if (llmReq->getReturnGenerationLogits())
            {
                auto const llmReqBeamWidth = llmReq->mSamplingConfig.beamWidth;
                auto const llmReqOutLen = llmReq->mMaxNewTokens;

                TLLM_CHECK_WITH_INFO(beamWidth == llmReqBeamWidth,
                    "Mismatch output generation logit shape: beamWidth, different with request's tensor shape");
                TLLM_CHECK_WITH_INFO(maxOutputLen >= llmReqOutLen,
                    "Mismatch output generation logit shape: maxOutputLen is smaller than request's tensor shape");

                TensorPtr genLogitsViewWithPad = ITensor::slice(generationOutput.generationLogits, reqOffset, 1);
                genLogitsViewWithPad->reshape(ITensor::makeShape({beamWidth * maxOutputLen, vocabSizePadded}));

                TensorPtr genLogitsViewNoPad = ITensor::slice(genLogitsViewWithPad, 0, llmReqBeamWidth * llmReqOutLen);
                genLogitsViewNoPad->reshape(ITensor::makeShape({llmReqBeamWidth, llmReqOutLen, vocabSizePadded}));

                TensorPtr genLogitsHost = bufferManager.copyFrom(*genLogitsViewNoPad, MemoryType::kPINNEDPOOL);
                llmReq->setGenerationLogitsHost(genLogitsHost);
            }
            reqOffset += 1;
        }
    }

    ++mIterCounter;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::shared_ptr<kv_cache_manager::BaseKVCacheManager> TrtGptModelV1::getKVCacheManager()
{
    return mSession->mKvCacheManager;
}

void TrtGptModelV1::setLayerProfiler()
{
    mSession->setLayerProfiler();
}

std::string TrtGptModelV1::getLayerProfileInfo() const
{
    return mSession->getLayerProfileInfo();
}

std::shared_ptr<kv_cache_manager::BaseKVCacheManager const> TrtGptModelV1::getKVCacheManager() const
{
    return mSession->mKvCacheManager;
}

void TrtGptModelV1::getCurrentIterationStats(executor::IterationStats& stats) const
{
    stats.iter = mIterCounter;
    // KVCacheManager statistics
    auto const& kvCacheManager = getKVCacheManager();
    if (kvCacheManager)
    {
        executor::KvCacheStats kvStats;
        auto kvCacheStats = kvCacheManager->getKvCacheStats();
        kvStats.maxNumBlocks = kvCacheStats.maxNumBlocks;
        kvStats.freeNumBlocks = kvCacheStats.freeNumBlocks;
        kvStats.usedNumBlocks = kvCacheStats.usedNumBlocks;
        kvStats.tokensPerBlock = kvCacheStats.toksPerBlock;
        kvStats.allocTotalBlocks = kvCacheStats.allocTotalBlocks;
        kvStats.allocNewBlocks = kvCacheStats.allocNewBlocks;
        kvStats.reusedBlocks = kvCacheStats.reusedBlocks;
        kvStats.missedBlocks = kvCacheStats.missedBlocks;
        kvStats.cacheHitRate = kvCacheStats.cacheHitRate;
        stats.kvCacheStats = kvStats;
    }
    executor::StaticBatchingStats modelStats;
    modelStats.numScheduledRequests = mLastIterationStatsV1.numScheduledRequests;
    modelStats.numContextRequests = mLastIterationStatsV1.numScheduledRequests;
    modelStats.numCtxTokens = mLastIterationStatsV1.numCtxTokensInBatch;
    modelStats.numGenTokens = mLastIterationStatsV1.numGenTokensInBatch;
    modelStats.emptyGenSlots = mLastIterationStatsV1.emptyGenSlots;
    stats.staticBatchingStats = modelStats;
}

void TrtGptModelV1::getCurrentRequestStats(executor::RequestStatsPerIteration& stats) const
{
    stats.iter = mIterCounter;
    for (auto& requestStat : stats.requestStats)
    {
        requestStat.scheduled
            = mLastIterationStatsV1.scheduledRequests.count(static_cast<RequestIdType>(requestStat.id));
        requestStat.paused = mLastIterationStatsV1.pausedRequests.count(static_cast<RequestIdType>(requestStat.id));
    }
}

executor::DebugTensorsPerIteration TrtGptModelV1::getCurrentDebugTensors() const
{
    executor::DebugTensorsPerIteration debugTensors;
    debugTensors.iter = mIterCounter;

    TLLM_LOG_WARNING("TrtGptModelV1 doesn't support getting debug tensors.");

    return debugTensors;
}

void TrtGptModelV1::setLogitsPostProcessorBatched(std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched)
{
    TLLM_CHECK_WITH_INFO(
        !logitsPostProcessorBatched.has_value(), "Batched logits post processor is not supported in V1 batcher");
}

void TrtGptModelV1::setReplicateLogitsPostProcessor(bool replicateLogitsPostProcessor)
{
    TLLM_THROW("Logits post processor is not supported in V1 batcher.");
}

bool TrtGptModelV1::getReplicateLogitsPostProcessor() const
{
    TLLM_THROW("Logits post processor is not supported in V1 batcher.");
}

nvinfer1::DataType TrtGptModelV1::getTensorDataType(std::string const& name) const
{
    return mSession->getTensorDataType(name);
}

nvinfer1::Dims TrtGptModelV1::getTensorShape(std::string const& name) const
{
    return mSession->getTensorShape(name);
}

TrtGptModelV1::~TrtGptModelV1() = default;

} // namespace tensorrt_llm::batch_manager

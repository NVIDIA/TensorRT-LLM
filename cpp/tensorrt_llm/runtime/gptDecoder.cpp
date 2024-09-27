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

#include "tensorrt_llm/runtime/gptDecoder.h"

#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/externalDraftTokensKernels.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/dynamicDecodeLayer.h"
#include "tensorrt_llm/runtime/decodingLayerWorkspace.h"

#include <memory>

#include <NvInferRuntime.h>

namespace tle = tensorrt_llm::executor;
namespace tl = tensorrt_llm::layers;
namespace tksd = tensorrt_llm::kernels::speculative_decoding;

using namespace tensorrt_llm::runtime;

using BufferConstPtr = IBuffer::SharedConstPtr;
using BufferPtr = IBuffer::SharedPtr;
using TensorConstPtr = ITensor::SharedConstPtr;
using TensorPtr = ITensor::SharedPtr;

template <typename T>
GptDecoder<T>::GptDecoder(executor::DecodingMode const& mode, size_t maxBatchSize, size_t maxBeamWidth,
    size_t vocabSize, size_t vocabSizePadded, size_t maxSequenceLength, CudaStreamPtr const& stream,
    std::shared_ptr<SpeculativeDecodingModule const> speculativeDecodingModule)
    : mManager{std::make_shared<BufferManager>(stream)}
    , mMaxBatchSize(maxBatchSize)
    , mDecodingMode{mode}
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const decodingDomain = tensorrt_llm::layers::DecoderDomain(
        maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded, speculativeDecodingModule);
    mDynamicDecodeLayer = std::make_shared<tensorrt_llm::layers::DynamicDecodeLayer<T>>(mode, decodingDomain, mManager);

    mDecodingLayerWorkspace = std::make_unique<tensorrt_llm::runtime::DecodingLayerWorkspace>(
        mManager, decodingDomain, TRTDataType<T>::value, mDynamicDecodeLayer->getWorkspaceSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GptDecoder<T>::setup(SamplingConfig const& samplingConfig, size_t batchSize, TensorConstPtr const& batchSlots,
    std::optional<DecodingOutput> const& output,
    std::optional<std::vector<decoder_batch::Request> const> const& requestsOpt)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mSamplingConfig = samplingConfig;
    auto setupParams = std::make_shared<layers::DynamicDecodeSetupParams>();

    TLLM_CHECK_WITH_INFO(mSamplingConfig.validate(), "Sampling config is invalid");
    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots are mandatory to set up the decoder.");

    auto penaltyParams = std::make_shared<tl::PenaltySetupParams>();
    penaltyParams->repetitionPenalty = mSamplingConfig.repetitionPenalty;
    penaltyParams->presencePenalty = mSamplingConfig.presencePenalty;
    penaltyParams->frequencyPenalty = mSamplingConfig.frequencyPenalty;
    penaltyParams->temperature = mSamplingConfig.temperature;
    penaltyParams->minLength = mSamplingConfig.minLength;

    setupParams->penaltyParams = std::move(penaltyParams);

    auto banWordsParams = std::make_shared<tl::BanWordsSetupParams>();
    banWordsParams->noRepeatNgramSize = mSamplingConfig.noRepeatNgramSize;

    setupParams->banWordsParams = std::move(banWordsParams);

    if (mDecodingMode.isTopKorTopP())
    {
        auto samplingParams = std::make_shared<tl::SamplingSetupParams>();
        samplingParams->normalizeLogProbs = mSamplingConfig.normalizeLogProbs;
        // signed to unsigned
        if (mSamplingConfig.topK)
        {
            auto const& topK = mSamplingConfig.topK.value();
            samplingParams->runtimeTopK = std::vector<SizeType32>(std::begin(topK), std::end(topK));
        }

        samplingParams->runtimeTopP = mSamplingConfig.topP;
        samplingParams->topPDecay = mSamplingConfig.topPDecay;
        samplingParams->topPMin = mSamplingConfig.topPMin;
        samplingParams->topPResetIds = mSamplingConfig.topPResetIds;
        samplingParams->outputLogProbs = mSamplingConfig.outputLogProbs;
        samplingParams->cumLogProbs = mSamplingConfig.cumLogProbs;

        setupParams->decodingParams = std::move(samplingParams);
    }
    else if (mDecodingMode.isBeamSearch())
    {
        auto beamSearchParams = std::make_shared<tl::BeamSearchSetupParams>();
        beamSearchParams->beamSearchDiversityRate = mSamplingConfig.beamSearchDiversityRate;
        beamSearchParams->lengthPenalty = mSamplingConfig.lengthPenalty;
        beamSearchParams->earlyStopping = mSamplingConfig.earlyStopping;

        setupParams->decodingParams = std::move(beamSearchParams);
    }
    else if (mDecodingMode.isMedusa())
    {
        auto medusaParams = std::make_shared<tl::MedusaSetupParams>();
        // signed to unsigned
        if (mSamplingConfig.topK)
        {
            auto const& topK = mSamplingConfig.topK.value();
            medusaParams->runtimeTopK = std::vector<SizeType32>(std::begin(topK), std::end(topK));
        }
        medusaParams->runtimeHeadsTopK = mSamplingConfig.topKMedusaHeads;

        setupParams->decodingParams = std::move(medusaParams);
    }
    else if (mDecodingMode.isExplicitDraftTokens())
    {
        TLLM_CHECK_WITH_INFO(output.has_value(), "Output tensors must be provided for ExplicitDraftTokens");
        auto explicitDraftTokensParams = std::make_shared<tl::ExplicitDraftTokensSetupParams>();
        explicitDraftTokensParams->temperature = mSamplingConfig.temperature;
        explicitDraftTokensParams->randomDataSample = output->explicitDraftTokensBuffers->randomDataSample;
        explicitDraftTokensParams->temperatures = output->explicitDraftTokensBuffers->temperatures;
        TLLM_CHECK(requestsOpt);
        // Ignore the dtype from all other requests assuming that it is the same for all.
        explicitDraftTokensParams->dtype = requestsOpt.value()[0].dtype;

        setupParams->decodingParams = explicitDraftTokensParams;
    }
    else if (mDecodingMode.isLookahead())
    {
        TLLM_CHECK_WITH_INFO(output.has_value(), "Output tensors must be provided for Lookahead decoding");
        TLLM_LOG_DEBUG("gptDecoder setup lookahead, batchSize=%d", batchSize);
        auto lookaheadParams = std::make_shared<tl::LookaheadSetupParams>();

        TLLM_CHECK(requestsOpt);
        auto& requests = requestsOpt.value();
        lookaheadParams->prompt.resize(0);
        lookaheadParams->prompt.reserve(batchSize);
        lookaheadParams->algoConfigs.resize(0);
        lookaheadParams->algoConfigs.reserve(batchSize);
        for (size_t bi = 0; bi < batchSize; bi++)
        {
            lookaheadParams->prompt.emplace_back(ITensor::slice(requests[bi].ids, 0, requests[bi].inputLen));
            TLLM_CHECK(requests[bi].lookaheadRuntimeConfig);
            lookaheadParams->algoConfigs.emplace_back(requests[bi].lookaheadRuntimeConfig.value());
        }
        lookaheadParams->generationLengths = output->lookaheadOutputs->generationLengths;
        lookaheadParams->positionOffsets = output->lookaheadOutputs->positionOffsets;
        lookaheadParams->attentionPackedMasks = output->lookaheadOutputs->packedMasks;
        setupParams->decodingParams = std::move(lookaheadParams);
    }
    setupParams->decodingParams->randomSeed = mSamplingConfig.randomSeed;

    mDecodingLayerWorkspace->setDeviceBatchSlots(batchSlots);
    mDynamicDecodeLayer->setup(batchSize, mSamplingConfig.beamWidth, batchSlots, setupParams, mDecodingLayerWorkspace);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

namespace
{

std::shared_ptr<tl::BanWordsDecodingInputs> prepareBanWordsInputs(DecodingInput const& input)
{
    auto banWordsParams = std::make_shared<tl::BanWordsDecodingInputs>(input.batchSize);
    if (input.badWordsPtrs)
    {
        TLLM_CHECK_WITH_INFO(input.badWordsPtrs, "Bad word lengths must be provided when badWordsPtrs is given");
        banWordsParams->badWordsPtr = input.badWordsPtrs;
        banWordsParams->badWordsLengths = input.badWordsLens;
        banWordsParams->maxBadWordsLen = input.maxBadWordsLen;
    }

    return banWordsParams;
}

std::shared_ptr<tl::StopCriteriaDecodingInputs> prepareStopCriteriaInputs(DecodingInput const& input)
{
    auto stopCriteriaParams = std::make_shared<tl::StopCriteriaDecodingInputs>(input.batchSize);
    if (input.stopWordsPtrs)
    {
        TLLM_CHECK_WITH_INFO(input.stopWordsLens, "Stop word lengths must be provided when stopWordsPtrs is given");

        stopCriteriaParams->stopWordsPtr = input.stopWordsPtrs;
        stopCriteriaParams->stopWordsLengths = input.stopWordsLens;
        stopCriteriaParams->maxStopWordsLen = input.maxStopWordsLen;
    }

    if (input.sequenceLimitLength)
    {
        stopCriteriaParams->sequenceLimitLength = input.sequenceLimitLength;
    }

    return stopCriteriaParams;
}

void prepareMedusaInputs(
    DecodingInput const& inputs, size_t maxBatchSize, std::shared_ptr<tl::DecodingInputs>& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputParams = std::dynamic_pointer_cast<tl::MedusaDecodingInputs>(baseInputs);

    auto const& medusaInputs = inputs.medusaInputs.value();

    inputParams->curTokensPerStep = medusaInputs.medusaCurTokensPerStep;
    inputParams->targetTokensPerStep = medusaInputs.medusaTargetTokensPerStep;
    inputParams->paths = medusaInputs.medusaPaths;
    inputParams->treeIds = medusaInputs.medusaTreeIds;
    auto const batchSlots = bufferCast<SizeType32>(*inputs.batchSlots);
    if (medusaInputs.medusaLogits.size())
    {
        std::vector<std::vector<TensorPtr>> medusaLogits;
        auto const batchSize = medusaInputs.medusaLogits.size();
        medusaLogits.resize(maxBatchSize);
        for (size_t bi = 0; bi < batchSize; ++bi)
        {
            auto const slot = batchSlots[bi];
            auto const& logitsHeads = medusaInputs.medusaLogits.at(slot);
            auto const medusaHeads = logitsHeads.size();
            medusaLogits[slot].resize(medusaHeads);
            for (size_t hi = 0; hi < medusaHeads; ++hi)
            {
                if (logitsHeads[hi])
                {
                    medusaLogits[slot][hi] = logitsHeads[hi];
                }
            }
        }
        inputParams->medusaLogits = medusaLogits;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void prepareExplicitDraftTokensInput(DecodingInput const& inputs, std::shared_ptr<tl::DecodingInputs>& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputParams = std::dynamic_pointer_cast<tl::ExplicitDraftTokensInputs>(baseInputs);

    auto& explicitDraftTokensInputs = inputs.explicitDraftTokensInputs;

    TLLM_CHECK_WITH_INFO(explicitDraftTokensInputs.has_value(), "ExplicitDraftTokensInputs are not set");

    inputParams->nextDraftTokens = explicitDraftTokensInputs->nextDraftTokens;
    inputParams->nextFlatTokens = explicitDraftTokensInputs->nextFlatTokens;
    inputParams->nextDraftIndices = explicitDraftTokensInputs->nextDraftIndices;
    inputParams->nextDraftProbs = explicitDraftTokensInputs->nextDraftProbs;
    inputParams->lastDraftTokens = explicitDraftTokensInputs->lastDraftTokens;
    inputParams->lastDraftIndices = explicitDraftTokensInputs->lastDraftIndices;
    inputParams->masks = explicitDraftTokensInputs->masks;
    inputParams->packedPosIds = explicitDraftTokensInputs->packedPositionIds;
    inputParams->bestPathLengths = explicitDraftTokensInputs->bestPathLengths;
    inputParams->bestPathIndices = explicitDraftTokensInputs->bestPathIndices;
    inputParams->generationLengths = explicitDraftTokensInputs->nextGenerationLengths;
    inputParams->positionIdsBase = explicitDraftTokensInputs->lastPositionIdsBase;
    inputParams->lastGenerationLengths = explicitDraftTokensInputs->lastGenerationLengths;
    inputParams->maxGenLengthDevice = explicitDraftTokensInputs->maxGenLengthDevice;
    inputParams->seqSlots = explicitDraftTokensInputs->seqSlots;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void prepareLookaheadInputs(
    DecodingInput const& inputs, size_t maxBatchSize, std::shared_ptr<tl::DecodingInputs>& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputParams = std::dynamic_pointer_cast<tl::LookaheadDecodingInputs>(baseInputs);
    auto const& lookaheadInputs = inputs.lookaheadInputs.value();
    inputParams->curTokensPerStep = lookaheadInputs.tokensPerStep;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
std::shared_ptr<tl::BaseDecodingInputs> prepareInputs(
    DecodingInput const& input, size_t maxBatchSize, tle::DecodingMode const& decodingMode)
{
    auto constexpr ite = 0;

    TLLM_CHECK_WITH_INFO(input.batchSlots != nullptr, "Batch slots are mandatory to call the decoder.");
    std::shared_ptr<tl::DecodingInputs> forwardParams;
    if (decodingMode.isTopKorTopP())
    {
        forwardParams
            = std::make_shared<tl::SamplingInputs>(input.endIds, input.batchSlots, input.step, ite, input.batchSize);
    }
    else if (decodingMode.isBeamSearch())
    {
        forwardParams = std::make_shared<tl::DecodingInputs>(input.endIds, input.batchSlots, input.step, ite,
            input.batchSize, input.maxAttentionWindow, input.sinkTokenLength);
    }
    else if (decodingMode.isMedusa())
    {
        forwardParams = std::make_shared<tl::MedusaDecodingInputs>(input.endIds, input.batchSlots, input.batchSize);
    }
    else if (decodingMode.isLookahead())
    {
        forwardParams = std::make_shared<tl::LookaheadDecodingInputs>(input.endIds, input.batchSlots);
    }
    else if (decodingMode.isExplicitDraftTokens())
    {
        forwardParams
            = std::make_shared<tl::ExplicitDraftTokensInputs>(input.endIds, input.batchSlots, input.batchSize);
    }

    // No logits for explicit draft tokens
    if (!decodingMode.isExplicitDraftTokens())
    {
        if (input.logitsVec)
        {
            std::vector<TensorConstPtr> logitsVec;
            for (auto const& logits : input.logitsVec.value())
            {
                TLLM_CHECK(logits->getDataType() == TRTDataType<T>::value);
                logitsVec.push_back(logits);
            }
            forwardParams->logitsVec = logitsVec;
        }
        else if (input.logits)
        {
            TLLM_CHECK(input.logits->getDataType() == TRTDataType<T>::value);
            forwardParams->logits = input.logits;
        }
    }

    if (input.cacheIndirection)
    {
        forwardParams->srcCacheIndirection = input.cacheIndirection;
    }

    if (input.embeddingBias)
    {
        forwardParams->embeddingBias = input.embeddingBias;
    }

    if (input.lengths)
    {
        forwardParams->inputLengths = input.lengths;
    }

    forwardParams->banWordsInputs = prepareBanWordsInputs(input);

    forwardParams->stopCriteriaInputs = prepareStopCriteriaInputs(input);

    if (input.finishReasons)
    {
        forwardParams->finished = input.finishReasons;
    }

    // Medusa
    if (decodingMode.isMedusa())
    {
        prepareMedusaInputs(input, maxBatchSize, forwardParams);
    }

    // Explicit draft tokens
    if (decodingMode.isExplicitDraftTokens())
    {
        prepareExplicitDraftTokensInput(input, forwardParams);
    }

    if (input.lookaheadInputs)
    {
        prepareLookaheadInputs(input, maxBatchSize, forwardParams);
        forwardParams->localBatchSize = input.batchSize;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);

    return forwardParams;
}

void prepareBeamSearchOutputs(DecodingOutput& output, std::shared_ptr<tl::BaseDecodingOutputs>& baseOutputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto outputParams = std::dynamic_pointer_cast<tl::BeamSearchOutputs>(baseOutputs);
    outputParams->beamHypotheses = std::make_unique<tensorrt_llm::kernels::BeamHypotheses>();
    if (output.beamHypotheses.outputIdsCBA)
    {
        outputParams->beamHypotheses->outputIdsCBA = bufferCast<int>(*output.beamHypotheses.outputIdsCBA);
    }
    if (output.beamHypotheses.logProbsCBA)
    {
        outputParams->beamHypotheses->logProbsCBA = bufferCast<float>(*output.beamHypotheses.logProbsCBA);
    }
    if (output.beamHypotheses.sequenceLengthsCBA)
    {
        outputParams->beamHypotheses->sequenceLengthsCBA = bufferCast<int>(*output.beamHypotheses.sequenceLengthsCBA);
    }
    if (output.beamHypotheses.cumLogProbsCBA)
    {
        outputParams->beamHypotheses->cumLogProbsCBA = bufferCast<float>(*output.beamHypotheses.cumLogProbsCBA);
    }
    if (output.beamHypotheses.normedScoresCBA)
    {
        outputParams->beamHypotheses->normedScoresCBA = bufferCast<float>(*output.beamHypotheses.normedScoresCBA);
    }
    if (output.beamHypotheses.numBeamsCBA)
    {
        outputParams->beamHypotheses->numBeamsCBA = bufferCast<int>(*output.beamHypotheses.numBeamsCBA);
    }
    if (output.beamHypotheses.minNormedScoresCBA)
    {
        outputParams->beamHypotheses->minNormedScoresCBA = bufferCast<float>(*output.beamHypotheses.minNormedScoresCBA);
    }
    if (output.beamHypotheses.batchDones)
    {
        outputParams->beamHypotheses->batchDones = bufferCast<bool>(*output.beamHypotheses.batchDones);
    }

    if (output.cacheIndirection)
    {
        outputParams->tgtCacheIndirection = output.cacheIndirection;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void prepareSpeculativeDecodingOutputs(DecodingOutput& output, std::shared_ptr<tl::BaseDecodingOutputs>& baseOutputs,
    tle::DecodingMode const& decodingMode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto outputParams = std::dynamic_pointer_cast<tl::SpeculativeDecodingOutputs>(baseOutputs);

    auto const& speculativeDecodingOutputs = output.speculativeDecodingOutputs;
    TLLM_CHECK_WITH_INFO(speculativeDecodingOutputs.has_value(), "speculativeDecodingOutputs is not set");

    outputParams->nextDraftTokens = speculativeDecodingOutputs->nextDraftTokens;
    outputParams->numNewTokens = speculativeDecodingOutputs->acceptedTokensLen;
    outputParams->numNewTokensCumSum = speculativeDecodingOutputs->acceptedLengthsCumSum;
    outputParams->pathsOffsets = speculativeDecodingOutputs->pathsOffsets;
    if (speculativeDecodingOutputs->nextDraftTokensLen)
    {
        outputParams->nextDraftLengths = speculativeDecodingOutputs->nextDraftTokensLen;
    }
    if (speculativeDecodingOutputs->prevDraftTokensLen)
    {
        outputParams->prevDraftLengths = speculativeDecodingOutputs->prevDraftTokensLen;
    }

    if (decodingMode.isExplicitDraftTokens())
    {
        auto outputParams = std::dynamic_pointer_cast<tl::ExplicitDraftTokensOutputs>(baseOutputs);
        auto const& explicitDraftTokensBuffers = output.explicitDraftTokensBuffers;
        TLLM_CHECK_WITH_INFO(explicitDraftTokensBuffers.has_value(), "explicitDraftTokensBuffers is not set");
        outputParams->packedMasks = explicitDraftTokensBuffers->packedMasks;
        outputParams->nextDraftPosIds = explicitDraftTokensBuffers->positionIds;

        outputParams->unpackedNextDraftTokens = explicitDraftTokensBuffers->draftTokens;
        outputParams->unpackedNextDraftIndices = explicitDraftTokensBuffers->draftIndices;
        outputParams->nextDraftProbs = explicitDraftTokensBuffers->draftProbs;
        outputParams->positionIdsBase = explicitDraftTokensBuffers->positionIdsBase;
        outputParams->randomDataSample = explicitDraftTokensBuffers->randomDataSample;
        outputParams->randomDataValidation = explicitDraftTokensBuffers->randomDataValidation;
        outputParams->temperatures = explicitDraftTokensBuffers->temperatures;
        outputParams->generationLengths = explicitDraftTokensBuffers->generationLengths;
        outputParams->generationLengthsHost = explicitDraftTokensBuffers->generationLengthsHost;
        outputParams->maxGenLengthHost = explicitDraftTokensBuffers->maxGenLengthHost;
    }
    if (decodingMode.isLookahead())
    {
        TLLM_CHECK(output.lookaheadOutputs);
        auto outputParams = std::dynamic_pointer_cast<tl::LookaheadDecodingOutputs>(baseOutputs);
        outputParams->packedMasks = output.lookaheadOutputs->packedMasks;
        outputParams->positionIds = output.lookaheadOutputs->positionIds;
        outputParams->positionOffsets = output.lookaheadOutputs->positionOffsets;
        outputParams->generationLengths = output.lookaheadOutputs->generationLengths;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::shared_ptr<tl::BaseDecodingOutputs> prepareOutputs(DecodingOutput& output, tle::DecodingMode const& decodingMode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    std::shared_ptr<tl::BaseDecodingOutputs> outputParams;

    if (decodingMode.isBeamSearch())
    {
        outputParams = std::make_shared<tl::BeamSearchOutputs>(output.ids);
    }
    else if (decodingMode.isMedusa())
    {
        outputParams = std::make_shared<tl::SpeculativeDecodingOutputs>(output.ids);
    }
    else if (decodingMode.isLookahead())
    {
        outputParams = std::make_shared<tl::LookaheadDecodingOutputs>(output.ids);
    }
    else if (decodingMode.isExplicitDraftTokens())
    {
        outputParams = std::make_shared<tl::ExplicitDraftTokensOutputs>(output.ids);
    }
    else
    {
        outputParams = std::make_shared<tl::BaseDecodingOutputs>(output.ids);
    }

    // Common outputs
    outputParams->newTokens = output.newTokens;

    if (output.cumLogProbs)
    {
        outputParams->cumLogProbs = output.cumLogProbs;
    }

    if (output.parentIds)
    {
        outputParams->parentIds = output.parentIds;
    }

    if (output.finishReasons)
    {
        outputParams->finished = output.finishReasons;
    }

    if (output.finishedSum)
    {
        outputParams->finishedSum = output.finishedSum;
    }

    if (output.lengths)
    {
        outputParams->sequenceLength = output.lengths;
    }

    if (output.logProbs)
    {
        outputParams->outputLogProbs = output.logProbs;
        outputParams->outputLogProbsTiled = output.logProbsTiled;
    }

    // Beam search outputs
    if (decodingMode.isBeamSearch())
    {
        prepareBeamSearchOutputs(output, outputParams);
    }

    // Speculative decoding outputs
    if (decodingMode.isMedusa() || decodingMode.isLookahead() || decodingMode.isExplicitDraftTokens())
    {
        prepareSpeculativeDecodingOutputs(output, outputParams, decodingMode);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return outputParams;
}

} // namespace

template <typename T>
void GptDecoder<T>::forwardAsync(DecodingOutput& output, DecodingInput const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto forwardParams = prepareInputs<T>(input, mMaxBatchSize, mDecodingMode);
    auto outputParams = prepareOutputs(output, mDecodingMode);

    mDynamicDecodeLayer->forwardAsync(outputParams, forwardParams, mDecodingLayerWorkspace);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GptDecoder<T>::forwardSync(DecodingOutput& output, DecodingInput const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto forwardParams = prepareInputs<T>(input, mMaxBatchSize, mDecodingMode);
    auto outputParams = prepareOutputs(output, mDecodingMode);

    mDynamicDecodeLayer->forwardSync(outputParams, forwardParams, mDecodingLayerWorkspace);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

namespace tensorrt_llm::runtime
{
template class GptDecoder<float>;
template class GptDecoder<half>;
} // namespace tensorrt_llm::runtime

void IGptDecoder::acceptDraftTokensByIds(ITensor const& targetTokenIds, ITensor const& draftTokenIds,
    ITensor const& contextLengths, ITensor const& numDraftTokens, ITensor& sequenceLengths, ITensor const& finishedVec,
    ITensor& finishedFinal, ITensor& finishedSum, ITensor const& batchSlots, BufferManager::CudaStreamPtr const& stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const finishedVecShape = finishedVec.getShape();
    auto const maxBatchSize = finishedVecShape.d[1];
    auto const batchSlotsShape = batchSlots.getShape();
    auto const batchSize = batchSlotsShape.d[0];
    auto const targetTokenIdsShape = targetTokenIds.getShape();
    auto const beamWidth = targetTokenIdsShape.d[1];
    auto const maxSeqLength = targetTokenIdsShape.d[2];
    auto const maxDraftTokens = draftTokenIds.getDimension<1>();

    TLLM_CHECK_WITH_INFO(beamWidth == 1,
        common::fmtstr("Beam width (" FMT_DIM ") > 1 is not supported for the speculative decoding", beamWidth));

    TLLM_CHECK_WITH_INFO(batchSize <= maxBatchSize,
        common::fmtstr("Batch size (" FMT_DIM ") is not smaller or equal to max batch size (" FMT_DIM ")", batchSize,
            maxBatchSize));

    TLLM_CHECK_WITH_INFO(draftTokenIds.getDimension<0>() == maxBatchSize,
        common::fmtstr("Draft tokens batch size (" FMT_DIM ") is not equal to target batch size (" FMT_DIM ")",
            draftTokenIds.getDimension<0>(), maxBatchSize));

    TLLM_CHECK_WITH_INFO(contextLengths.getDimension<0>() == maxBatchSize,
        common::fmtstr("Context length batch size (" FMT_DIM ") is not equal to batch size (" FMT_DIM ")",
            contextLengths.getDimension<0>(), maxBatchSize));

    TLLM_CHECK_WITH_INFO(numDraftTokens.getDimension<0>() == maxBatchSize,
        common::fmtstr("Num draft tokens batch size (" FMT_DIM ") is not equal to batch size (" FMT_DIM ")",
            numDraftTokens.getDimension<0>(), maxBatchSize));

    TLLM_CHECK_WITH_INFO(sequenceLengths.getDimension<0>() == maxBatchSize,
        common::fmtstr("Sequence length batch size (" FMT_DIM ") is not equal to batch size (" FMT_DIM ")",
            sequenceLengths.getDimension<0>(), maxBatchSize));

    tksd::invokeAcceptDraftTokensByIds(bufferCast<TokenIdType>(draftTokenIds), bufferCast<TokenIdType>(targetTokenIds),
        bufferCast<SizeType32>(contextLengths), bufferCast<SizeType32>(numDraftTokens),
        bufferCast<SizeType32>(sequenceLengths),
        reinterpret_cast<tensorrt_llm::kernels::FinishedState const*>(
            bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(finishedVec)),
        reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
            bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(finishedFinal)),
        bufferCast<int>(finishedSum), bufferCast<SizeType32>(batchSlots), batchSize, maxBatchSize, beamWidth,
        maxSeqLength, maxDraftTokens, stream->get());

    sync_check_cuda_error();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void IGptDecoder::acceptDraftTokensByLogits(ITensor& draftLogits, ITensor const& targetLogits, ITensor& draftProbs,
    ITensor& targetProbs, ITensor const& numDraftTokens, ITensor& finished, ITensor const& batchSlots,
    SizeType32 vocabSize, SizeType32 vocabSizePadded, bool useRandomAcceptThreshold, float randomAcceptThreshold,
    curandState_t* curandState, BufferManager::CudaStreamPtr const& stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const draftLogitsShape = draftLogits.getShape();
    auto const maxBatchSize = draftLogitsShape.d[0];
    auto const maxTokensPerStep = draftLogitsShape.d[1];
    auto const batchSlotsShape = batchSlots.getShape();
    auto const batchSize = batchSlotsShape.d[0];
    auto constexpr beamWidth = 1;

    TLLM_CHECK_WITH_INFO(
        beamWidth == 1, common::fmtstr("Beam width (%d) > 1 is not supported for the speculative decoding", beamWidth));

    TLLM_CHECK(draftLogitsShape.d[2] == vocabSize);

    if (draftLogits.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        tksd::acceptDraftTokensByLogits(bufferCast<float>(draftLogits),
            const_cast<float**>(reinterpret_cast<float const* const*>(bufferCast<int64_t>(targetLogits))),
            bufferCast<float>(draftProbs), bufferCast<float>(targetProbs), bufferCast<SizeType32>(numDraftTokens),
            reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
                bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(finished)),
            curandState, bufferCast<SizeType32>(batchSlots), batchSize, maxBatchSize, beamWidth, vocabSize,
            vocabSizePadded, maxTokensPerStep, useRandomAcceptThreshold, randomAcceptThreshold, stream->get());
    }
    else if (draftLogits.getDataType() == nvinfer1::DataType::kHALF)
    {
        tksd::acceptDraftTokensByLogits(bufferCast<half>(draftLogits),
            const_cast<half**>(reinterpret_cast<half const* const*>(bufferCast<int64_t>(targetLogits))),
            bufferCast<half>(draftProbs), bufferCast<half>(targetProbs), bufferCast<SizeType32>(numDraftTokens),
            reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
                bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(finished)),
            curandState, bufferCast<SizeType32>(batchSlots), batchSize, maxBatchSize, beamWidth, vocabSize,
            vocabSizePadded, maxTokensPerStep, useRandomAcceptThreshold, randomAcceptThreshold, stream->get());
    }
    else
    {
        TLLM_THROW("Incorrect logits dtype. Only float32 and float16 are supported");
    }

    sync_check_cuda_error();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

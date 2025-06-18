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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/dynamicDecodeLayer.h"
#include "tensorrt_llm/runtime/decodingLayerWorkspace.h"

#include <NvInferRuntime.h>

#include <memory>

namespace tle = tensorrt_llm::executor;
namespace tl = tensorrt_llm::layers;

using namespace tensorrt_llm::runtime;

using BufferConstPtr = IBuffer::SharedConstPtr;
using BufferPtr = IBuffer::SharedPtr;
using TensorConstPtr = ITensor::SharedConstPtr;
using TensorPtr = ITensor::SharedPtr;

template <typename T>
GptDecoder<T>::GptDecoder(executor::DecodingMode const& mode, size_t maxNumSequences, size_t maxBeamWidth,
    size_t vocabSize, size_t vocabSizePadded, CudaStreamPtr const& stream,
    std::shared_ptr<SpeculativeDecodingModule const> speculativeDecodingModule)
    : mManager{std::make_shared<BufferManager>(stream)}
    , mMaxNumSequences(maxNumSequences)
    , mVocabSize(vocabSize)
    , mVocabSizePadded(vocabSizePadded)
    , mDecodingMode{mode}
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const decodingDomain = tensorrt_llm::layers::DecoderDomain(
        maxNumSequences, maxBeamWidth, vocabSize, vocabSizePadded, speculativeDecodingModule);
    mDynamicDecodeLayer = std::make_shared<tensorrt_llm::layers::DynamicDecodeLayer<T>>(mode, decodingDomain, mManager);

    mDecodingLayerWorkspace = std::make_unique<tensorrt_llm::runtime::DecodingLayerWorkspace>(
        mManager, decodingDomain, TRTDataType<T>::value, mDynamicDecodeLayer->getWorkspaceSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GptDecoder<T>::disableLookahead(
    std::optional<SamplingConfig> const& samplingConfig, SizeType32 batchSize, TensorConstPtr batchSlots)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mDecodingMode = executor::DecodingMode::TopKTopP();
    auto const decodingDomain
        = tensorrt_llm::layers::DecoderDomain(mMaxNumSequences, 1, mVocabSize, mVocabSizePadded, nullptr);

    auto setupParams = std::make_shared<layers::DynamicDecodeSetupParams>();

    if (batchSize == 0)
    {
        mDynamicDecodeLayer->disableLookahead(
            decodingDomain, batchSize, batchSlots, setupParams, mDecodingLayerWorkspace);
        return;
    }

    mSamplingConfig = samplingConfig.value();
    TLLM_CHECK_WITH_INFO(mSamplingConfig.validate(), "Sampling config is invalid");
    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots are mandatory to set up the decoder.");
    // penalty parameters
    auto penaltyParams = std::make_shared<tl::PenaltySetupParams>();
    penaltyParams->repetitionPenalty = mSamplingConfig.repetitionPenalty;
    penaltyParams->presencePenalty = mSamplingConfig.presencePenalty;
    penaltyParams->frequencyPenalty = mSamplingConfig.frequencyPenalty;
    penaltyParams->temperature = mSamplingConfig.temperature;
    penaltyParams->minLength = mSamplingConfig.minLength;

    // banwords parameters
    auto banWordsParams = std::make_shared<tl::BanWordsSetupParams>();
    banWordsParams->noRepeatNgramSize = mSamplingConfig.noRepeatNgramSize;

    // sampling parameters
    auto samplingParams = std::make_shared<tl::SamplingSetupParams>();
    samplingParams->normalizeLogProbs = mSamplingConfig.normalizeLogProbs;
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
    samplingParams->runtimeMinP = mSamplingConfig.minP;

    // get setup parameters
    setupParams->penaltyParams = std::move(penaltyParams);
    setupParams->banWordsParams = std::move(banWordsParams);
    setupParams->decodingParams = std::move(samplingParams);

    mDecodingLayerWorkspace->setDeviceBatchSlots(batchSlots);
    mDynamicDecodeLayer->disableLookahead(decodingDomain, batchSize, batchSlots, setupParams, mDecodingLayerWorkspace);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GptDecoder<T>::setup(SamplingConfig const& samplingConfig, size_t batchSize, TensorConstPtr const& batchSlots,
    std::optional<DecodingOutput> const& output, std::optional<nvinfer1::DataType> explicitDraftTokensDType,
    std::optional<std::vector<TensorConstPtr>> const& lookaheadPrompt,
    std::optional<std::vector<tle::LookaheadDecodingConfig>> const& lookaheadAlgoConfigs)
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
        samplingParams->runtimeMinP = mSamplingConfig.minP;

        setupParams->decodingParams = std::move(samplingParams);
    }
    else if (mDecodingMode.isBeamSearch())
    {
        auto beamSearchParams = std::make_shared<tl::BeamSearchSetupParams>();
        beamSearchParams->beamSearchDiversityRate = mSamplingConfig.beamSearchDiversityRate;
        beamSearchParams->lengthPenalty = mSamplingConfig.lengthPenalty;
        beamSearchParams->earlyStopping = mSamplingConfig.earlyStopping;
        beamSearchParams->beamWidthArray = mSamplingConfig.beamWidthArray;

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
        TLLM_CHECK(explicitDraftTokensDType.has_value());
        explicitDraftTokensParams->dtype = explicitDraftTokensDType.value();

        setupParams->decodingParams = explicitDraftTokensParams;
    }
    else if (mDecodingMode.isLookahead())
    {
        TLLM_LOG_DEBUG("gptDecoder setup lookahead, batchSize=%d", batchSize);
        auto lookaheadParams = std::make_shared<tl::LookaheadSetupParams>();

        TLLM_CHECK_WITH_INFO(lookaheadPrompt.has_value(), "Lookahead prompt must be provided");
        lookaheadParams->prompt = lookaheadPrompt.value();
        TLLM_CHECK_WITH_INFO(lookaheadAlgoConfigs.has_value(), "Lookahead algo configs must be provided");
        lookaheadParams->algoConfigs = lookaheadAlgoConfigs.value();
        TLLM_CHECK_WITH_INFO(output.has_value(), "Output tensors must be provided for Lookahead decoding");
        lookaheadParams->generationLengths = output->lookaheadOutputs->generationLengths;
        lookaheadParams->positionOffsets = output->lookaheadOutputs->positionOffsets;
        lookaheadParams->attentionPackedMasks = output->lookaheadOutputs->packedMasks;

        setupParams->decodingParams = std::move(lookaheadParams);
    }
    else if (mDecodingMode.isExternalDraftTokens())
    {
        auto externalDraftTokensParams = std::make_shared<tl::ExternalDraftTokensSetupParams>();
        // signed to unsigned
        if (mSamplingConfig.topK)
        {
            auto const& topK = mSamplingConfig.topK.value();
            externalDraftTokensParams->runtimeTopK = std::vector<SizeType32>(std::begin(topK), std::end(topK));
        }
        externalDraftTokensParams->runtimeTopP = mSamplingConfig.topP;
        setupParams->decodingParams = std::move(externalDraftTokensParams);
    }
    else if (mDecodingMode.isEagle())
    {
        TLLM_CHECK_WITH_INFO(output.has_value(), "Output tensors must be provided for Eagle");
        auto eagleParams = std::make_shared<tl::EagleSetupParams>();
        eagleParams->temperature = mSamplingConfig.originalTemperature;
        eagleParams->randomDataSample = output->eagleBuffers->randomDataSample;
        eagleParams->temperatures = output->eagleBuffers->temperatures;

        setupParams->decodingParams = eagleParams;
    }

    setupParams->decodingParams->randomSeed = mSamplingConfig.randomSeed;

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
    DecodingInput const& inputs, size_t maxNumSequences, std::shared_ptr<tl::DecodingInputs>& baseInputs)
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
        medusaLogits.resize(maxNumSequences);
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

void prepareExternalDraftTokensInputs(DecodingInput const& inputs, std::shared_ptr<tl::DecodingInputs>& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputParams = std::dynamic_pointer_cast<tl::ExternalDraftTokensInputs>(baseInputs);
    auto const& externalDraftTokensInputs = inputs.externalDraftTokensInputs.value();

    inputParams->draftLogits = externalDraftTokensInputs.draftLogits;
    inputParams->draftProbs = externalDraftTokensInputs.draftProbs;
    inputParams->targetProbs = externalDraftTokensInputs.targetProbs;
    inputParams->numDraftTokens = externalDraftTokensInputs.numDraftTokens;
    inputParams->numDraftTokensHost = externalDraftTokensInputs.numDraftTokensHost;
    inputParams->draftTokenIds = externalDraftTokensInputs.draftTokenIds;
    inputParams->constantThreshold = externalDraftTokensInputs.constantThreshold;
    inputParams->useRandomAcceptanceThreshold = externalDraftTokensInputs.useRandomAcceptanceThreshold;
    inputParams->step = externalDraftTokensInputs.step;
    inputParams->useDraftLogits = externalDraftTokensInputs.useDraftLogits;
    inputParams->useDraftLogitsHost = externalDraftTokensInputs.useDraftLogitsHost;

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

void prepareLookaheadInputs(DecodingInput const& inputs, std::shared_ptr<tl::DecodingInputs>& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputParams = std::dynamic_pointer_cast<tl::LookaheadDecodingInputs>(baseInputs);
    auto const& lookaheadInputs = inputs.lookaheadInputs.value();
    inputParams->curTokensPerStep = lookaheadInputs.tokensPerStep;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void prepareEagleInput(DecodingInput const& inputs, std::shared_ptr<tl::DecodingInputs>& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputParams = std::dynamic_pointer_cast<tl::EagleInputs>(baseInputs);

    auto& eagleInputs = inputs.eagleInputs;

    TLLM_CHECK_WITH_INFO(eagleInputs.has_value(), "EagleInputs are not set");

    inputParams->nextDraftTokens = eagleInputs->nextDraftTokens;
    inputParams->nextDraftLens = eagleInputs->nextDraftLens;
    inputParams->nextDraftPaths = eagleInputs->nextDraftPaths;
    inputParams->lastDraftTokens = eagleInputs->lastDraftTokens;
    inputParams->lastDraftLens = eagleInputs->lastDraftLens;
    inputParams->lastDraftPaths = eagleInputs->lastDraftPaths;
    inputParams->acceptedTokens = eagleInputs->acceptedTokens;
    inputParams->acceptedLens = eagleInputs->acceptedLens;
    inputParams->acceptedPathIds = eagleInputs->acceptedPathIds;
    inputParams->chunkedContextNextTokens = eagleInputs->chunkedContextNextTokens;
    inputParams->seqSlots = eagleInputs->seqSlots;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
std::shared_ptr<tl::BaseDecodingInputs> prepareInputs(
    DecodingInput const& input, size_t maxNumSequences, tle::DecodingMode const& decodingMode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

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

        if (input.cacheIndirection)
        {
            forwardParams->srcCacheIndirection = input.cacheIndirection;
        }
        forwardParams->beamSearchSteps = input.generationSteps;
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
    else if (decodingMode.isExternalDraftTokens())
    {
        forwardParams = std::make_shared<tl::ExternalDraftTokensInputs>(
            input.endIds, input.batchSlots, input.step, ite, input.batchSize);
    }
    else if (decodingMode.isEagle())
    {
        auto& eagleInputs = input.eagleInputs;

        TLLM_CHECK_WITH_INFO(eagleInputs.has_value(), "EagleInputs are not set");

        forwardParams = std::make_shared<tl::EagleInputs>(input.endIds, input.batchSlots, input.batchSize,
            eagleInputs->nextDraftTokens, eagleInputs->nextDraftLens, eagleInputs->nextDraftPaths,
            eagleInputs->lastDraftTokens, eagleInputs->lastDraftLens, eagleInputs->lastDraftPaths,
            eagleInputs->acceptedTokens, eagleInputs->acceptedLens, eagleInputs->acceptedPathIds,
            eagleInputs->chunkedContextNextTokens, eagleInputs->seqSlots);
    }

    // No logits for explicit draft tokens and eagle
    if (!decodingMode.isExplicitDraftTokens() && !decodingMode.isEagle())
    {
        for (auto const& logits : input.logitsVec)
        {
            TLLM_CHECK(logits->getDataType() == TRTDataType<T>::value);
        }
        forwardParams->logitsVec = input.logitsVec;
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

    // Speculative decoding
    if (decodingMode.isMedusa())
    {
        prepareMedusaInputs(input, maxNumSequences, forwardParams);
    }
    else if (decodingMode.isExplicitDraftTokens())
    {
        prepareExplicitDraftTokensInput(input, forwardParams);
    }
    else if (decodingMode.isLookahead() && input.lookaheadInputs)
    {
        prepareLookaheadInputs(input, forwardParams);
        forwardParams->localBatchSize = input.batchSize;
    }
    else if (decodingMode.isExternalDraftTokens())
    {
        prepareExternalDraftTokensInputs(input, forwardParams);
    }
    else if (decodingMode.isEagle())
    {
        prepareEagleInput(input, forwardParams);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);

    return forwardParams;
}

void prepareBeamSearchOutputs(DecodingOutput& output, std::shared_ptr<tl::BaseDecodingOutputs>& baseOutputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& bhSrc = output.beamHypotheses;
    auto bhOutputs = std::dynamic_pointer_cast<tl::BeamSearchOutputs>(baseOutputs);
    bhOutputs->beamHypotheses = std::make_unique<tensorrt_llm::kernels::BeamHypotheses>();
    auto& bhDst = bhOutputs->beamHypotheses;

    if (bhSrc.outputIdsCBA)
    {
        bhDst->outputIdsCBA = bufferCast<int>(*bhSrc.outputIdsCBA);
    }
    if (bhSrc.logProbsCBA)
    {
        bhDst->logProbsCBA = bufferCast<float>(*bhSrc.logProbsCBA);
    }
    if (bhSrc.sequenceLengthsCBA)
    {
        bhDst->sequenceLengthsCBA = bufferCast<int>(*bhSrc.sequenceLengthsCBA);
    }
    if (bhSrc.cumLogProbsCBA)
    {
        bhDst->cumLogProbsCBA = bufferCast<float>(*bhSrc.cumLogProbsCBA);
    }
    if (bhSrc.normedScoresCBA)
    {
        bhDst->normedScoresCBA = bufferCast<float>(*bhSrc.normedScoresCBA);
    }
    if (bhSrc.numBeamsCBA)
    {
        bhDst->numBeamsCBA = bufferCast<int>(*bhSrc.numBeamsCBA);
    }
    if (bhSrc.minNormedScoresCBA)
    {
        bhDst->minNormedScoresCBA = bufferCast<float>(*bhSrc.minNormedScoresCBA);
    }
    if (bhSrc.batchDones)
    {
        bhDst->batchDones = bufferCast<bool>(*bhSrc.batchDones);
    }
    if (output.cacheIndirection)
    {
        bhOutputs->tgtCacheIndirection = output.cacheIndirection;
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
    else if (decodingMode.isLookahead())
    {
        TLLM_CHECK(output.lookaheadOutputs);
        auto outputParams = std::dynamic_pointer_cast<tl::LookaheadDecodingOutputs>(baseOutputs);
        outputParams->packedMasks = output.lookaheadOutputs->packedMasks;
        outputParams->positionIds = output.lookaheadOutputs->positionIds;
        outputParams->positionOffsets = output.lookaheadOutputs->positionOffsets;
        outputParams->generationLengths = output.lookaheadOutputs->generationLengths;
    }
    else if (decodingMode.isEagle())
    {
        auto outputParams = std::dynamic_pointer_cast<tl::EagleOutputs>(baseOutputs);
        auto const& eagleBuffers = output.eagleBuffers;
        TLLM_CHECK_WITH_INFO(eagleBuffers.has_value(), "eagleBuffers is not set");

        outputParams->temperatures = eagleBuffers->temperatures;
        outputParams->unpackedNextDraftTokens = eagleBuffers->draftTokens;
        outputParams->nextDraftPaths = eagleBuffers->draftPaths;
        outputParams->generationLengths = eagleBuffers->specDecodingGenerationLengths;
        outputParams->generationLengthsHost = eagleBuffers->specDecodingGenerationLengthsHost;
        outputParams->nextDraftPosIds = eagleBuffers->specDecodingPositionOffsets;
        outputParams->packedMasks = eagleBuffers->specDecodingPackedMasks;
        outputParams->randomDataSample = eagleBuffers->randomDataSample;
        outputParams->randomDataValidation = eagleBuffers->randomDataValidation;

        outputParams->eagleNetCtxRequestTypesHost = eagleBuffers->eagleNetCtxRequestTypesHost;
        outputParams->eagleNetCtxContextLengthsHost = eagleBuffers->eagleNetCtxContextLengthsHost;
        outputParams->eagleNetCtxPastKeyValueLengthsHost = eagleBuffers->eagleNetCtxPastKeyValueLengthsHost;
        outputParams->eagleNetGenRequestTypesHost = eagleBuffers->eagleNetGenRequestTypesHost;
        outputParams->eagleNetGenContextLengthsHost = eagleBuffers->eagleNetGenContextLengthsHost;
        outputParams->eagleNetGenPastKeyValueLengthsHost = eagleBuffers->eagleNetGenPastKeyValueLengthsHost;
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
    else if (decodingMode.isEagle())
    {
        outputParams = std::make_shared<tl::EagleOutputs>(output.ids);
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
    if (decodingMode.isMedusa() || decodingMode.isLookahead() || decodingMode.isExplicitDraftTokens()
        || decodingMode.isEagle())
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

    auto forwardParams = prepareInputs<T>(input, mMaxNumSequences, mDecodingMode);
    auto outputParams = prepareOutputs(output, mDecodingMode);
    mDynamicDecodeLayer->forwardAsync(outputParams, forwardParams, mDecodingLayerWorkspace);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GptDecoder<T>::forwardSync(DecodingOutput& output, DecodingInput const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto forwardParams = prepareInputs<T>(input, mMaxNumSequences, mDecodingMode);
    auto outputParams = prepareOutputs(output, mDecodingMode);

    mDynamicDecodeLayer->forwardSync(outputParams, forwardParams, mDecodingLayerWorkspace);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

namespace tensorrt_llm::runtime
{
template class GptDecoder<float>;
template class GptDecoder<half>;
} // namespace tensorrt_llm::runtime

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

#include "tensorrt_llm/common/cudaAllocator.h"
#include "tensorrt_llm/common/tensorConversion.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/externalDraftTokensKernels.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/dynamicDecodeLayer.h"

#include <memory>

#include <NvInferRuntime.h>

namespace tle = tensorrt_llm::executor;
namespace tc = tensorrt_llm::common;
namespace tl = tensorrt_llm::layers;
namespace tcc = tensorrt_llm::common::conversion;
namespace tksd = tensorrt_llm::kernels::speculative_decoding;

using namespace tensorrt_llm::runtime;

template <typename T>
GptDecoder<T>::GptDecoder(executor::DecodingMode const& mode, size_t maxBatchSize, size_t maxBeamWidth,
    size_t vocabSize, size_t vocabSizePadded, size_t maxSequenceLength, CudaStreamPtr const& stream,
    std::shared_ptr<SpeculativeDecodingModule const> speculativeDecodingModule)
    : mManager{stream}
    , mMaxBatchSize(maxBatchSize)
    , mDecodingMode{mode}
{
    auto const decodingDomain = tensorrt_llm::layers::DecoderDomain(
        maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded, speculativeDecodingModule);
    auto allocator = std::make_shared<common::CudaAllocator>(mManager);
    mDynamicDecodeLayer = std::make_shared<tensorrt_llm::layers::DynamicDecodeLayer<T>>(
        mode, decodingDomain, stream->get(), std::move(allocator));

    auto constexpr nvFloatType = TRTDataType<float>::value;
    mLogProbsTiled = mManager.gpu(ITensor::makeShape({static_cast<SizeType32>(maxSequenceLength),
                                      static_cast<SizeType32>(maxBatchSize), static_cast<SizeType32>(maxBeamWidth)}),
        nvFloatType);
    mManager.setZero(*mLogProbsTiled);
}

template <typename T>
void GptDecoder<T>::setup(SamplingConfig const& samplingConfig, size_t batchSize, SizeType32 const* batchSlots,
    std::optional<DecodingOutput> const& output)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mSamplingConfig = samplingConfig;
    auto setupParams = std::make_shared<layers::DynamicDecodeSetupParams>();

    TLLM_CHECK_WITH_INFO(mSamplingConfig.validate(), "Sampling config is invalid");

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
        explicitDraftTokensParams->randomDataSample
            = tcc::toTllmTensor(*output->explicitDraftTokensBuffers->randomDataSample);
        explicitDraftTokensParams->temperatures = tcc::toTllmTensor(*output->explicitDraftTokensBuffers->temperatures);

        setupParams->decodingParams = explicitDraftTokensParams;
    }

    setupParams->decodingParams->randomSeed = mSamplingConfig.randomSeed;

    mDynamicDecodeLayer->setup(batchSize, mSamplingConfig.beamWidth, batchSlots, setupParams);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

namespace
{
void safeInsert(tc::TensorMap& map, std::string const& key, DecodingOutput::TensorPtr const& tensor)
{
    if (tensor)
    {
        ITensor const& t{*tensor};
        map.insert({key, tcc::toTllmTensor(t)});
    }
}

std::shared_ptr<tl::BanWordsDecodingInputs> prepareBanWordsInputs(DecodingInput const& input)
{
    auto banWordsParams = std::make_shared<tl::BanWordsDecodingInputs>(input.batchSize);
    if (input.badWordsPtrs)
    {
        TLLM_CHECK_WITH_INFO(input.badWordsPtrs, "Bad word lengths must be provided when badWordsPtrs is given");
        banWordsParams->badWordsPtr = tcc::toTllmTensor(*input.badWordsPtrs);
        banWordsParams->badWordsLengths = tcc::toTllmTensor(*input.badWordsLens);
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

        stopCriteriaParams->stopWordsPtr = tcc::toTllmTensor(*input.stopWordsPtrs);
        stopCriteriaParams->stopWordsLengths = tcc::toTllmTensor(*input.stopWordsLens);
        stopCriteriaParams->maxStopWordsLen = input.maxStopWordsLen;
    }

    if (input.sequenceLimitLength)
    {
        stopCriteriaParams->sequenceLimitLength = tcc::toTllmTensor(*input.sequenceLimitLength);
    }

    return stopCriteriaParams;
}

void prepareMedusaInputs(
    DecodingInput const& inputs, size_t maxBatchSize, std::shared_ptr<tl::DecodingInputs>& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputParams = std::dynamic_pointer_cast<tl::MedusaDecodingInputs>(baseInputs);

    auto const& medusaInputs = inputs.medusaInputs.value();

    inputParams->curTokensPerStep = tcc::toTllmTensor(*medusaInputs.medusaCurTokensPerStep);
    inputParams->targetTokensPerStep = tcc::toTllmTensor(*medusaInputs.medusaTargetTokensPerStep);
    inputParams->paths = tcc::toTllmTensor(*medusaInputs.medusaPaths);
    inputParams->treeIds = tcc::toTllmTensor(*medusaInputs.medusaTreeIds);
    auto const batchSlots = bufferCast<SizeType32>(*inputs.batchSlots);
    if (medusaInputs.medusaLogits.size())
    {
        std::vector<std::vector<tc::Tensor>> medusaLogits;
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
                    medusaLogits[slot][hi] = tcc::toTllmTensor(*logitsHeads[hi]);
                }
            }
        }
        inputParams->medusaLogits = medusaLogits;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void prepareExplicitDraftTokensInput(
    DecodingInput const& inputs, size_t maxBatchSize, std::shared_ptr<tl::DecodingInputs>& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputParams = std::dynamic_pointer_cast<tl::ExplicitDraftTokensInputs>(baseInputs);

    auto& explicitDraftTokensInputs = inputs.explicitDraftTokensInputs;

    TLLM_CHECK_WITH_INFO(explicitDraftTokensInputs.has_value(), "ExplicitDraftTokensInputs are not set");

    inputParams->nextDraftTokens = tcc::toTllmTensor(*explicitDraftTokensInputs->nextDraftTokens);
    inputParams->nextFlatTokens = tcc::toTllmTensor(*explicitDraftTokensInputs->nextFlatTokens);
    inputParams->nextDraftIndices = tcc::toTllmTensor(*explicitDraftTokensInputs->nextDraftIndices);
    inputParams->nextDraftProbs = tcc::toTllmTensor(*explicitDraftTokensInputs->nextDraftProbs);
    inputParams->lastDraftTokens = tcc::toTllmTensor(*explicitDraftTokensInputs->lastDraftTokens);
    inputParams->lastDraftIndices = tcc::toTllmTensor(*explicitDraftTokensInputs->lastDraftIndices);
    inputParams->masks = tcc::toTllmTensor(*explicitDraftTokensInputs->masks);
    inputParams->packedPosIds = tcc::toTllmTensor(*explicitDraftTokensInputs->packedPositionIds);
    inputParams->bestPathLengths = tcc::toTllmTensor(*explicitDraftTokensInputs->bestPathLengths);
    inputParams->bestPathIndices = tcc::toTllmTensor(*explicitDraftTokensInputs->bestPathIndices);
    inputParams->generationLengths = tcc::toTllmTensor(*explicitDraftTokensInputs->nextGenerationLengths);
    inputParams->positionIdsBase = tcc::toTllmTensor(*explicitDraftTokensInputs->lastPositionIdsBase);
    inputParams->lastGenerationLengths = tcc::toTllmTensor(*explicitDraftTokensInputs->lastGenerationLengths);
    inputParams->maxGenLengthDevice = tcc::toTllmTensor(*explicitDraftTokensInputs->maxGenLengthDevice);
    inputParams->seqSlots = tcc::toTllmTensor(*explicitDraftTokensInputs->seqSlots);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
std::shared_ptr<tl::BaseDecodingInputs> prepareInputs(
    DecodingInput const& input, size_t maxBatchSize, tle::DecodingMode const& decodingMode)
{
    auto constexpr ite = 0;

    std::shared_ptr<tl::DecodingInputs> forwardParams;
    if (decodingMode.isTopKorTopP())
    {
        forwardParams
            = std::make_shared<tl::SamplingInputs>(tcc::toTllmTensor(*input.endIds), input.step, ite, input.batchSize);
    }
    else if (decodingMode.isBeamSearch())
    {
        forwardParams = std::make_shared<tl::DecodingInputs>(tcc::toTllmTensor(*input.endIds), input.step, ite,
            input.batchSize, input.maxAttentionWindow, input.sinkTokenLength);
    }
    else if (decodingMode.isMedusa())
    {
        forwardParams = std::make_shared<tl::MedusaDecodingInputs>(tcc::toTllmTensor(*input.endIds), input.batchSize);
    }
    else if (decodingMode.isLookahead())
    {
        // TODO add lookahead inputs
    }
    else if (decodingMode.isExplicitDraftTokens())
    {
        forwardParams
            = std::make_shared<tl::ExplicitDraftTokensInputs>(tcc::toTllmTensor(*input.endIds), input.batchSize);
    }

    // No logits for explicit draft tokens
    if (!decodingMode.isExplicitDraftTokens())
    {
        if (input.logitsVec)
        {
            std::vector<tc::Tensor> logitsVec;
            for (auto const& logits : input.logitsVec.value())
            {
                TLLM_CHECK(logits->getDataType() == TRTDataType<T>::value);
                logitsVec.push_back(tcc::toTllmTensor(*logits));
            }
            forwardParams->logitsVec = logitsVec;
        }
        else if (input.logits)
        {
            TLLM_CHECK(input.logits->getDataType() == TRTDataType<T>::value);
            forwardParams->logits = tcc::toTllmTensor(*input.logits);
        }
    }

    if (input.cacheIndirection)
    {
        forwardParams->srcCacheIndirection = tcc::toTllmTensor(*input.cacheIndirection);
    }

    if (input.embeddingBias)
    {
        forwardParams->embeddingBias = tcc::toTllmTensor(*input.embeddingBias);
    }

    if (input.lengths)
    {
        forwardParams->inputLengths = tcc::toTllmTensor(*input.lengths);
    }

    forwardParams->banWordsInputs = prepareBanWordsInputs(input);

    forwardParams->stopCriteriaInputs = prepareStopCriteriaInputs(input);

    if (input.finished)
    {
        forwardParams->finished = tcc::toTllmTensor(*input.finished);
    }

    if (input.batchSlots)
    {
        forwardParams->batchSlots = tcc::toTllmTensor(*input.batchSlots);
    }

    // Medusa
    if (decodingMode.isMedusa())
    {
        prepareMedusaInputs(input, maxBatchSize, forwardParams);
    }

    // Explicit draft tokens
    if (decodingMode.isExplicitDraftTokens())
    {
        prepareExplicitDraftTokensInput(input, maxBatchSize, forwardParams);
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
        outputParams->tgtCacheIndirection = tcc::toTllmTensor(*output.cacheIndirection);
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

    outputParams->nextDraftTokens = tcc::toTllmTensor(*speculativeDecodingOutputs->nextDraftTokens);
    outputParams->numNewTokens = tcc::toTllmTensor(*speculativeDecodingOutputs->acceptedTokensLen);
    outputParams->numNewTokensCumSum = tcc::toTllmTensor(*speculativeDecodingOutputs->acceptedLengthsCumSum);
    outputParams->pathsOffsets = tcc::toTllmTensor(*speculativeDecodingOutputs->pathsOffsets);
    if (speculativeDecodingOutputs->nextDraftTokensLen)
    {
        outputParams->nextDraftLengths = tcc::toTllmTensor(*speculativeDecodingOutputs->nextDraftTokensLen);
    }
    if (speculativeDecodingOutputs->prevDraftTokensLen)
    {
        outputParams->prevDraftLengths = tcc::toTllmTensor(*speculativeDecodingOutputs->prevDraftTokensLen);
    }

    if (decodingMode.isExplicitDraftTokens())
    {
        auto outputParams = std::dynamic_pointer_cast<tl::ExplicitDraftTokensOutputs>(baseOutputs);
        auto const& explicitDraftTokensBuffers = output.explicitDraftTokensBuffers;
        TLLM_CHECK_WITH_INFO(explicitDraftTokensBuffers.has_value(), "explicitDraftTokensBuffers is not set");
        outputParams->packedMasks = tcc::toTllmTensor(*explicitDraftTokensBuffers->packedMasks);
        outputParams->nextDraftPosIds = tcc::toTllmTensor(*explicitDraftTokensBuffers->positionIds);

        outputParams->unpackedNextDraftTokens = tcc::toTllmTensor(*explicitDraftTokensBuffers->draftTokens);
        outputParams->unpackedNextDraftIndices = tcc::toTllmTensor(*explicitDraftTokensBuffers->draftIndices);
        outputParams->nextDraftProbs = tcc::toTllmTensor(*explicitDraftTokensBuffers->draftProbs);
        outputParams->positionIdsBase = tcc::toTllmTensor(*explicitDraftTokensBuffers->positionIdsBase);
        outputParams->randomDataSample = tcc::toTllmTensor(*explicitDraftTokensBuffers->randomDataSample);
        outputParams->randomDataValidation = tcc::toTllmTensor(*explicitDraftTokensBuffers->randomDataValidation);
        outputParams->temperatures = tcc::toTllmTensor(*explicitDraftTokensBuffers->temperatures);
        outputParams->generationLengths = tcc::toTllmTensor(*explicitDraftTokensBuffers->generationLengths);
        outputParams->maxGenLengthHost = tcc::toTllmTensor(*explicitDraftTokensBuffers->maxGenLengthHost);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::shared_ptr<tl::BaseDecodingOutputs> prepareOutputs(
    DecodingOutput& output, DecodingOutput::TensorPtr& logProbsTiled, tle::DecodingMode const& decodingMode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    std::shared_ptr<tl::BaseDecodingOutputs> outputParams;

    if (decodingMode.isBeamSearch())
    {
        outputParams = std::make_shared<tl::BeamSearchOutputs>(tcc::toTllmTensor(*output.ids));
    }
    else if (decodingMode.isMedusa() || decodingMode.isLookahead())
    {
        outputParams = std::make_shared<tl::SpeculativeDecodingOutputs>(tcc::toTllmTensor(*output.ids));
    }
    else if (decodingMode.isExplicitDraftTokens())
    {
        outputParams = std::make_shared<tl::ExplicitDraftTokensOutputs>(tcc::toTllmTensor(*output.ids));
    }
    else
    {
        outputParams = std::make_shared<tl::BaseDecodingOutputs>(tcc::toTllmTensor(*output.ids));
    }

    // Common outputs
    outputParams->newTokens = tcc::toTllmTensor(*output.newTokens);

    if (output.cumLogProbs)
    {
        outputParams->cumLogProbs = tcc::toTllmTensor(*output.cumLogProbs);
    }

    if (output.parentIds)
    {
        outputParams->parentIds = tcc::toTllmTensor(*output.parentIds);
    }

    if (output.finished)
    {
        outputParams->finished = tcc::toTllmTensor(*output.finished);
    }

    if (output.finishedSum)
    {
        outputParams->finishedSum = tcc::toTllmTensor(*output.finishedSum);
    }

    if (output.lengths)
    {
        outputParams->sequenceLength = tcc::toTllmTensor(*output.lengths);
    }

    if (output.logProbs)
    {
        outputParams->outputLogProbs = tcc::toTllmTensor(*output.logProbs);
        outputParams->outputLogProbsTiled = tcc::toTllmTensor(*logProbsTiled);
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
    auto outputParams = prepareOutputs(output, mLogProbsTiled, mDecodingMode);

    mDynamicDecodeLayer->forwardAsync(outputParams, forwardParams);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GptDecoder<T>::forwardSync(DecodingOutput& output, DecodingInput const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto forwardParams = prepareInputs<T>(input, mMaxBatchSize, mDecodingMode);
    auto outputParams = prepareOutputs(output, mLogProbsTiled, mDecodingMode);

    mDynamicDecodeLayer->forwardSync(outputParams, forwardParams);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

// Must be similar to [cpp/tensorrt_llm/thop/gatherTreeOp.cpp] gatherTree
template <typename T>
void GptDecoder<T>::gatherTree(ITensor& finalOutputIds, DecodingOutput const& decodingOutput,
    DecodingInput const& decodingInput, BufferManager const& manager,
    std::optional<std::reference_wrapper<SamplingConfig const>> samplingConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& finalOutputIdsShape = finalOutputIds.getShape();
    auto const& decodingOutputIdsShape = decodingOutput.ids->getShape();
    auto const batchSize = finalOutputIdsShape.d[0];
    auto const beamWidth = finalOutputIdsShape.d[1];
    auto const maxSeqLength = finalOutputIdsShape.d[2];

    TLLM_CHECK_WITH_INFO(beamWidth > 1, "gatherTree is only needed for beam search.");

    TLLM_CHECK_WITH_INFO(decodingOutputIdsShape.d[0] == batchSize,
        common::fmtstr("Decoder batch size (" FMT_DIM ") does not match final batch size (" FMT_DIM ")",
            decodingOutputIdsShape.d[0], batchSize));
    TLLM_CHECK_WITH_INFO(decodingOutputIdsShape.d[1] == beamWidth,
        common::fmtstr("Decoder beam width (" FMT_DIM ") does not match final beam width (" FMT_DIM ")",
            decodingOutputIdsShape.d[1], beamWidth));
    TLLM_CHECK_WITH_INFO(decodingOutputIdsShape.d[2] <= maxSeqLength,
        common::fmtstr("Decoder seq length size (" FMT_DIM ") is too large for final seq length (" FMT_DIM ")",
            decodingOutputIdsShape.d[2], maxSeqLength));

    auto const& stream = manager.getStream().get();

    tensorrt_llm::kernels::invokeInitializeOutput(bufferCast<TokenIdType>(finalOutputIds),
        bufferCast<TokenIdType>(*decodingInput.endIds), batchSize * beamWidth, maxSeqLength, stream);
    sync_check_cuda_error();

    // Prepare length penalty, use the value from samplingConfig or 1.0f by default
    SamplingConfig const& samplingConf = samplingConfig ? (*samplingConfig).get() : mSamplingConfig;
    std::vector<float> lengthPenaltyVec;
    TensorPtr lengthPenaltyPtr
        = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize}), TRTDataType<float>::value));
    if (!samplingConf.lengthPenalty.has_value() || samplingConf.lengthPenalty.value().size() == 0)
    {
        lengthPenaltyVec = std::vector<float>(batchSize, 1.0f);
    }
    else if (long int const size = samplingConf.lengthPenalty.value().size(); size == 1)
    {
        lengthPenaltyVec = std::vector<float>(batchSize, samplingConf.lengthPenalty.value()[0]);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(size == batchSize,
            common::fmtstr("Size of lengthPenalty in SamplingConfig (" FMT_DIM ") is different from batchSize (" FMT_DIM
                           ")",
                size, batchSize));
        lengthPenaltyVec = samplingConf.lengthPenalty.value();
    }

    lengthPenaltyPtr = manager.copyFrom(lengthPenaltyVec, ITensor::makeShape({batchSize}), runtime::MemoryType::kGPU);

    tensorrt_llm::kernels::BeamHypotheses bh;
    bh.nMaxBatchSize = batchSize;
    bh.nBatchSize = batchSize;
    bh.nBeamWidth = beamWidth;
    bh.nMaxSeqLen = maxSeqLength;
    bh.lengthPenalties = bufferCast<float>(*lengthPenaltyPtr);
    bh.inputLengths = bufferCast<SizeType32>(*decodingInput.lengths);
    bh.outputIds = bufferCast<TokenIdType>(finalOutputIds);
    bh.logProbs = (decodingOutput.logProbs == nullptr) ? nullptr : bufferCast<float>(*decodingOutput.logProbs);
    bh.logProbsTiled = bufferCast<float>(*mLogProbsTiled);
    bh.sequenceLengths = bufferCast<SizeType32>(*decodingOutput.lengths);
    bh.cumLogProbs = bufferCast<float>(*decodingOutput.cumLogProbs);
    bh.outputIdsCBA = bufferCast<TokenIdType>(*decodingOutput.beamHypotheses.outputIdsCBA);
    bh.logProbsCBA = bufferCast<float>(*decodingOutput.beamHypotheses.logProbsCBA);
    bh.sequenceLengthsCBA = bufferCast<SizeType32>(*decodingOutput.beamHypotheses.sequenceLengthsCBA);
    bh.cumLogProbsCBA = bufferCast<float>(*decodingOutput.beamHypotheses.cumLogProbsCBA);
    bh.normedScoresCBA = bufferCast<float>(*decodingOutput.beamHypotheses.normedScoresCBA);
    bh.numBeamsCBA = bufferCast<SizeType32>(*decodingOutput.beamHypotheses.numBeamsCBA);
    bh.minNormedScoresCBA = bufferCast<float>(*decodingOutput.beamHypotheses.minNormedScoresCBA);
    bh.batchDones = bufferCast<bool>(*decodingOutput.beamHypotheses.batchDones);
    bh.finished = reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
        bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*decodingOutput.finished));
    bh.outputIdsUnfinish = bufferCast<TokenIdType>(*decodingOutput.ids);
    bh.parentIdsUnfinish = bufferCast<TokenIdType>(*decodingOutput.parentIds);

    // This is where transpose is done
    tensorrt_llm::kernels::invokeInsertUnfinishedPath(bh, stream);
    sync_check_cuda_error();

    tensorrt_llm::kernels::invokeFinalize(bh, stream);
    sync_check_cuda_error();

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
    auto const maxDraftTokens = draftTokenIds.getShape().d[1];

    TLLM_CHECK_WITH_INFO(beamWidth == 1,
        common::fmtstr("Beam width (" FMT_DIM ") > 1 is not supported for the speculative decoding", beamWidth));

    TLLM_CHECK_WITH_INFO(batchSize <= maxBatchSize,
        common::fmtstr("Batch size (" FMT_DIM ") is not smaller or equal to max batch size (" FMT_DIM ")", batchSize,
            maxBatchSize));

    TLLM_CHECK_WITH_INFO(draftTokenIds.getShape().d[0] == maxBatchSize,
        common::fmtstr("Draft tokens batch size (" FMT_DIM ") is not equal to target batch size (" FMT_DIM ")",
            draftTokenIds.getShape().d[0], maxBatchSize));

    TLLM_CHECK_WITH_INFO(contextLengths.getShape().d[0] == maxBatchSize,
        common::fmtstr("Context length batch size (" FMT_DIM ") is not equal to batch size (" FMT_DIM ")",
            contextLengths.getShape().d[0], maxBatchSize));

    TLLM_CHECK_WITH_INFO(numDraftTokens.getShape().d[0] == maxBatchSize,
        common::fmtstr("Num draft tokens batch size (" FMT_DIM ") is not equal to batch size (" FMT_DIM ")",
            numDraftTokens.getShape().d[0], maxBatchSize));

    TLLM_CHECK_WITH_INFO(sequenceLengths.getShape().d[0] == maxBatchSize,
        common::fmtstr("Sequence length batch size (" FMT_DIM ") is not equal to batch size (" FMT_DIM ")",
            sequenceLengths.getShape().d[0], maxBatchSize));

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

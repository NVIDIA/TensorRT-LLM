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
void GptDecoder<T>::setup(
    SamplingConfig const& samplingConfig, size_t batchSize, std::optional<TensorPtr> const& batchSlots)
{
    mSamplingConfig = samplingConfig;
    auto setupParams = std::make_shared<layers::DynamicDecodeSetupParams>();

    TLLM_CHECK_WITH_INFO(mSamplingConfig.validate(), "Sampling config is invalid");

    setupParams->penaltyParams.repetitionPenalty = mSamplingConfig.repetitionPenalty;
    setupParams->penaltyParams.presencePenalty = mSamplingConfig.presencePenalty;
    setupParams->penaltyParams.frequencyPenalty = mSamplingConfig.frequencyPenalty;
    setupParams->penaltyParams.temperature = mSamplingConfig.temperature;
    setupParams->penaltyParams.minLength = mSamplingConfig.minLength;
    setupParams->penaltyParams.noRepeatNgramSize = mSamplingConfig.noRepeatNgramSize;

    setupParams->randomSeed = mSamplingConfig.randomSeed;

    setupParams->samplingParams.normalize_log_probs = mSamplingConfig.normalizeLogProbs;
    // signed to unsigned
    if (mSamplingConfig.topK)
    {
        auto const& topK = mSamplingConfig.topK.value();
        setupParams->samplingParams.runtime_top_k = std::vector<SizeType32>(std::begin(topK), std::end(topK));
    }

    setupParams->samplingParams.runtime_top_p = mSamplingConfig.topP;
    setupParams->samplingParams.top_p_decay = mSamplingConfig.topPDecay;
    setupParams->samplingParams.top_p_min = mSamplingConfig.topPMin;
    setupParams->samplingParams.top_p_reset_ids = mSamplingConfig.topPResetIds;
    setupParams->samplingParams.outputLogProbs = mSamplingConfig.outputLogProbs;
    setupParams->samplingParams.cumLogProbs = mSamplingConfig.cumLogProbs;

    setupParams->beamSearchParams.beam_search_diversity_rate = mSamplingConfig.beamSearchDiversityRate;
    setupParams->beamSearchParams.length_penalty = mSamplingConfig.lengthPenalty;
    setupParams->beamSearchParams.early_stopping = mSamplingConfig.earlyStopping;

    setupParams->medusaParams.topKMedusaHeads = mSamplingConfig.topKMedusaHeads;

    auto const batchSlotsPtr = batchSlots.has_value() ? bufferCast<SizeType32>(*(batchSlots.value())) : nullptr;
    mDynamicDecodeLayer->setup(batchSize, mSamplingConfig.beamWidth, batchSlotsPtr, setupParams);
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

template <typename T>
tl::DynamicDecodeInputParams::MedusaInputs prepareMedusaInputs(DecodingInput const& inputs, size_t maxBatchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& medusaInputs = inputs.medusaInputs.value();

    tl::DynamicDecodeInputParams::MedusaInputs medusaDecodingInputs;
    medusaDecodingInputs.medusaCurTokensPerStep = tcc::toTllmTensor(*medusaInputs.medusaCurTokensPerStep);
    medusaDecodingInputs.medusaTargetTokensPerStep = tcc::toTllmTensor(*medusaInputs.medusaTargetTokensPerStep);
    medusaDecodingInputs.medusaPaths = tcc::toTllmTensor(*medusaInputs.medusaPaths);
    medusaDecodingInputs.medusaTreeIds = tcc::toTllmTensor(*medusaInputs.medusaTreeIds);
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
        medusaDecodingInputs.medusaLogits = medusaLogits;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return medusaDecodingInputs;
}

template <typename T>
tl::DynamicDecodeInputParams::ExplicitDraftTokensInputs prepareExplicitDraftTokensInput(
    DecodingInput const& inputs, size_t maxBatchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return tl::DynamicDecodeInputParams::ExplicitDraftTokensInputs{};
}

template <typename T>
std::shared_ptr<tl::DynamicDecodeInputParams> prepareInputs(
    DecodingInput const& input, size_t maxBatchSize, tle::DecodingMode const& decodingMode)
{
    auto constexpr ite = 0; // no pipeline parallelism
    auto forwardParams = std::make_shared<tl::DynamicDecodeInputParams>(input.step, ite, input.maxLength,
        input.maxAttentionWindow, input.sinkTokenLength, input.maxBatchSize, tcc::toTllmTensor(*input.endIds));

    if (input.logitsVec)
    {
        std::vector<tc::Tensor> logitsVec;
        for (auto const& logits : input.logitsVec.value())
        {
            TLLM_CHECK(logits->getDataType() == TRTDataType<T>::value);
            logitsVec.push_back(tcc::toTllmTensor(*logits));
        }
        forwardParams->logits_vec = logitsVec;
    }
    else
    {
        TLLM_CHECK(input.logits->getDataType() == TRTDataType<T>::value);
        forwardParams->logits = tcc::toTllmTensor(*input.logits);
    }

    if (input.cacheIndirection)
    {
        forwardParams->src_cache_indirection = tcc::toTllmTensor(*input.cacheIndirection);
    }

    if (input.sequenceLimitLength)
    {
        forwardParams->sequence_limit_length = tcc::toTllmTensor(*input.sequenceLimitLength);
    }

    if (input.embeddingBias)
    {
        forwardParams->embedding_bias = tcc::toTllmTensor(*input.embeddingBias);
    }

    if (input.lengths)
    {
        forwardParams->input_lengths = tcc::toTllmTensor(*input.lengths);
    }

    if (input.badWordsPtrs)
    {
        TLLM_CHECK_WITH_INFO(input.badWordsPtrs, "Bad word lengths must be provided when badWordsPtrs is given");
        forwardParams->bad_words_ptr = tcc::toTllmTensor(*input.badWordsPtrs);
        forwardParams->bad_words_lengths = tcc::toTllmTensor(*input.badWordsLens);
        forwardParams->max_bad_words_len = input.maxBadWordsLen;
    }

    if (input.stopWordsPtrs)
    {
        TLLM_CHECK_WITH_INFO(input.stopWordsLens, "Stop word lengths must be provided when stopWordsPtrs is given");
        forwardParams->stop_words_ptr = tcc::toTllmTensor(*input.stopWordsPtrs);
        forwardParams->stop_words_lengths = tcc::toTllmTensor(*input.stopWordsLens);
        forwardParams->max_stop_words_len = input.maxStopWordsLen;
    }

    if (input.finished)
    {
        forwardParams->finished = tcc::toTllmTensor(*input.finished);
    }

    if (input.batchSlots)
    {
        forwardParams->batch_slots = tcc::toTllmTensor(*input.batchSlots);
    }

    // Medusa
    if (decodingMode.isMedusa())
    {
        forwardParams->medusaInputs = prepareMedusaInputs<T>(input, maxBatchSize);
    }

    // Explicit draft tokens
    if (decodingMode.isExplicitDraftTokens())
    {
        forwardParams->explicitDraftTokensInputs = prepareExplicitDraftTokensInput<T>(input, maxBatchSize);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);

    return forwardParams;
}

template <typename T>
tl::DynamicDecodeOutputParams::SpeculativeDecodingOutputs prepareSpeculativeDecodingOutputs(
    DecodingOutput::SpeculativeDecodingOutputs& output)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    tl::DynamicDecodeOutputParams::SpeculativeDecodingOutputs speculativeDecodingOutputs;
    speculativeDecodingOutputs.nextDraftTokens = tcc::toTllmTensor(*output.nextDraftTokens);
    speculativeDecodingOutputs.acceptedLengths = tcc::toTllmTensor(*output.acceptedTokensLen);
    speculativeDecodingOutputs.acceptedLengthsCumSum = tcc::toTllmTensor(*output.acceptedLengthsCumSum);
    speculativeDecodingOutputs.pathsOffsets = tcc::toTllmTensor(*output.pathsOffsets);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return speculativeDecodingOutputs;
}

template <typename T>
std::shared_ptr<tl::DynamicDecodeOutputParams> prepareOutputs(
    DecodingOutput& output, DecodingOutput::TensorPtr& logProbsTiled, tle::DecodingMode const& decodingMode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto outputParams = std::make_shared<tl::DynamicDecodeOutputParams>(tcc::toTllmTensor(*output.ids));

    outputParams->newTokens = tcc::toTllmTensor(*output.newTokens);

    if (output.cumLogProbs)
    {
        outputParams->cum_log_probs = tcc::toTllmTensor(*output.cumLogProbs);
    }

    if (output.parentIds)
    {
        outputParams->parent_ids = tcc::toTllmTensor(*output.parentIds);
    }

    if (output.cacheIndirection)
    {
        outputParams->tgt_cache_indirection = tcc::toTllmTensor(*output.cacheIndirection);
    }

    if (output.finished)
    {
        outputParams->finished = tcc::toTllmTensor(*output.finished);
    }

    if (output.finishedSum)
    {
        outputParams->finished_sum = tcc::toTllmTensor(*output.finishedSum);
    }

    if (output.lengths)
    {
        outputParams->sequence_length = tcc::toTllmTensor(*output.lengths);
    }

    if (output.logProbs)
    {
        outputParams->output_log_probs = tcc::toTllmTensor(*output.logProbs);
        outputParams->output_log_probs_tiled = tcc::toTllmTensor(*logProbsTiled);
    }

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

    // Speculative decoding
    if (decodingMode.isMedusa() || decodingMode.isLookahead() || decodingMode.isExplicitDraftTokens())
    {
        outputParams->speculativeDecodingOutputs
            = prepareSpeculativeDecodingOutputs<T>(output.speculativeDecodingOutputs.value());
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
    auto outputParams = prepareOutputs<T>(output, mLogProbsTiled, mDecodingMode);

    mDynamicDecodeLayer->forwardAsync(outputParams, forwardParams);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GptDecoder<T>::forwardSync(DecodingOutput& output, DecodingInput const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto forwardParams = prepareInputs<T>(input, mMaxBatchSize, mDecodingMode);
    auto outputParams = prepareOutputs<T>(output, mLogProbsTiled, mDecodingMode);

    mDynamicDecodeLayer->forwardSync(outputParams, forwardParams);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

// Must be similar to [cpp/tensorrt_llm/thop/gatherTreeOp.cpp] gatherTree
template <typename T>
void GptDecoder<T>::gatherTree(ITensor& finalOutputIds, DecodingOutput const& decodingOutput,
    DecodingInput const& decodingInput, BufferManager const& manager)
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

    // Prepare length penalty, use the value from mSamplingConfig or 1.0f by default
    std::vector<float> lengthPenaltyVec;
    TensorPtr lengthPenaltyPtr
        = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize}), TRTDataType<float>::value));
    if (!mSamplingConfig.lengthPenalty.has_value() || mSamplingConfig.lengthPenalty.value().size() == 0)
    {
        lengthPenaltyVec = std::vector<float>(batchSize, 1.0f);
    }
    else if (long int const size = mSamplingConfig.lengthPenalty.value().size(); size == 1)
    {
        lengthPenaltyVec = std::vector<float>(batchSize, mSamplingConfig.lengthPenalty.value()[0]);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(size == batchSize,
            common::fmtstr("Size of lengthPenalty in SimplingConfig (" FMT_DIM ") is different from batchSize (" FMT_DIM
                           ")",
                size, batchSize));
        lengthPenaltyVec = mSamplingConfig.lengthPenalty.value();
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

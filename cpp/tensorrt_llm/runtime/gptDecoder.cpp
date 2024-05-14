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
#include "tensorrt_llm/kernels/parallelDecoding/kvCacheUpdateKernels.h"
#include "tensorrt_llm/layers/dynamicDecodeLayer.h"

#include <memory>

#include <NvInferRuntime.h>

namespace tc = tensorrt_llm::common;
namespace tl = tensorrt_llm::layers;
namespace tcc = tensorrt_llm::common::conversion;

using namespace tensorrt_llm::runtime;

template <typename T>
GptDecoder<T>::GptDecoder(DecodingMode const& mode, size_t maxBatchSize, size_t maxBeamWidth, size_t vocabSize,
    size_t vocabSizePadded, size_t maxSequenceLength, CudaStreamPtr const& stream,
    std::optional<runtime::SizeType32> maxTokensPerStep, std::optional<runtime::SizeType32> maxNumMedusaHeads)
    : mManager{stream}
    , mMaxBatchSize(maxBatchSize)
{
    auto const decodingDomain = tensorrt_llm::layers::DecoderDomain(
        maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded, maxTokensPerStep, maxNumMedusaHeads);
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
void GptDecoder<T>::setup(SamplingConfig const& samplingConfig, size_t batchSize, SizeType32 maxSequenceLength,
    std::optional<TensorPtr> const& batchSlots)
{
    mSamplingConfig = samplingConfig;
    auto setupParams = std::make_shared<layers::DynamicDecodeSetupParams>();

    setupParams->penaltyParams.repetitionPenalty = samplingConfig.repetitionPenalty;
    setupParams->penaltyParams.presencePenalty = samplingConfig.presencePenalty;
    setupParams->penaltyParams.frequencyPenalty = samplingConfig.frequencyPenalty;
    setupParams->penaltyParams.temperature = samplingConfig.temperature;
    setupParams->penaltyParams.minLength = samplingConfig.minLength;

    setupParams->randomSeed = samplingConfig.randomSeed;

    setupParams->samplingParams.normalize_log_probs = samplingConfig.normalizeLogProbs;
    // signed to unsigned
    if (samplingConfig.topK)
    {
        auto const& topK = samplingConfig.topK.value();
        setupParams->samplingParams.runtime_top_k = std::vector<SizeType32>(std::begin(topK), std::end(topK));
    }

    setupParams->samplingParams.runtime_top_p = samplingConfig.topP;
    setupParams->samplingParams.top_p_decay = samplingConfig.topPDecay;
    setupParams->samplingParams.top_p_min = samplingConfig.topPMin;
    setupParams->samplingParams.top_p_reset_ids = samplingConfig.topPResetIds;

    setupParams->beamSearchParams.beam_search_diversity_rate = samplingConfig.beamSearchDiversityRate;
    setupParams->beamSearchParams.length_penalty = samplingConfig.lengthPenalty;
    setupParams->beamSearchParams.early_stopping = samplingConfig.earlyStopping;

    setupParams->medusaParams.topKMedusaHeads = samplingConfig.topKMedusaHeads;

    auto const batchSlotsPtr = batchSlots.has_value() ? bufferCast<SizeType32>(*(batchSlots.value())) : nullptr;
    mDynamicDecodeLayer->setup(batchSize, samplingConfig.beamWidth, batchSlotsPtr, setupParams);
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
std::shared_ptr<tl::DynamicDecodeInputParams> prepareInputs(DecodingInput const& input, size_t maxBatchSize)
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
    if (input.medusaInputs)
    {
        forwardParams->medusaInputs = prepareMedusaInputs<T>(input, maxBatchSize);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);

    return forwardParams;
}

template <typename T>
tl::DynamicDecodeOutputParams::MedusaOutputs prepareMedusaOutputs(DecodingOutput::MedusaOutputs& output)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    tl::DynamicDecodeOutputParams::MedusaOutputs medusaOutputs;
    medusaOutputs.nextDraftTokens = tcc::toTllmTensor(*output.medusaNextDraftTokens);
    medusaOutputs.acceptedLengths = tcc::toTllmTensor(*output.medusaAcceptedTokensLen);
    medusaOutputs.acceptedLengthsCumSum = tcc::toTllmTensor(*output.medusaAcceptedLengthsCumSum);
    medusaOutputs.pathsOffsets = tcc::toTllmTensor(*output.medusaPathsOffsets);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return medusaOutputs;
}

template <typename T>
std::shared_ptr<tl::DynamicDecodeOutputParams> prepareOutputs(
    DecodingOutput& output, DecodingInput::TensorPtr const& inputLengths, DecodingOutput::TensorPtr& logProbsTiled)
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
    if (output.beamHypotheses.batchDones)
    {
        outputParams->beamHypotheses->batchDones = bufferCast<bool>(*output.beamHypotheses.batchDones);
    }
    if (output.beamHypotheses.cumLogProbsCBA)
    {
        outputParams->beamHypotheses->cumLogProbsCBA = bufferCast<float>(*output.beamHypotheses.cumLogProbsCBA);
    }
    if (output.beamHypotheses.logProbsCBA)
    {
        outputParams->beamHypotheses->logProbsCBA = bufferCast<float>(*output.beamHypotheses.logProbsCBA);
    }
    if (output.beamHypotheses.minNormedScoresCBA)
    {
        outputParams->beamHypotheses->minNormedScoresCBA = bufferCast<float>(*output.beamHypotheses.minNormedScoresCBA);
    }
    if (output.beamHypotheses.normedScoresCBA)
    {
        outputParams->beamHypotheses->normedScoresCBA = bufferCast<float>(*output.beamHypotheses.normedScoresCBA);
    }
    if (output.beamHypotheses.numBeamsCBA)
    {
        outputParams->beamHypotheses->numBeamsCBA = bufferCast<int>(*output.beamHypotheses.numBeamsCBA);
    }
    if (output.beamHypotheses.outputIdsCBA)
    {
        outputParams->beamHypotheses->outputIdsCBA = bufferCast<int>(*output.beamHypotheses.outputIdsCBA);
    }
    if (output.beamHypotheses.sequenceLengthsCBA)
    {
        outputParams->beamHypotheses->sequenceLengthsCBA = bufferCast<int>(*output.beamHypotheses.sequenceLengthsCBA);
    }
    if (inputLengths)
    {
        outputParams->beamHypotheses->inputLengths = bufferCast<int32_t>(*inputLengths);
    }

    // Medusa
    if (output.medusaOutputs)
    {
        outputParams->medusaOutputs = prepareMedusaOutputs<T>(output.medusaOutputs.value());
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return outputParams;
}

} // namespace

template <typename T>
bool GptDecoder<T>::forward(DecodingOutput& output, DecodingInput const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto forwardParams = prepareInputs<T>(input, mMaxBatchSize);
    auto outputParams = prepareOutputs<T>(output, input.lengths, mLogProbsTiled);
    auto const maxBatchSize = input.maxBatchSize;

    BufferManager::ITensorPtr finishedSum;
    std::int32_t* finishedSumHost = nullptr;
    if (input.sequenceLimitLength && output.finished)
    {
        if (output.finishedSum)
        {
            finishedSumHost = bufferCast<std::int32_t>(*output.finishedSum);
        }
        else
        {
            finishedSum = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
            outputParams->finished_sum = tcc::toTllmTensor(*finishedSum);
            finishedSumHost = bufferCast<std::int32_t>(*finishedSum);
        }
        for (SizeType32 bi = 0; bi < maxBatchSize; ++bi)
        {
            finishedSumHost[bi] = 0;
        }
    }

    mDynamicDecodeLayer->forward(outputParams, forwardParams);

    if (finishedSumHost)
    {
        auto const numToFinish = output.finished->getSize();
        TLLM_CUDA_CHECK(::cudaStreamSynchronize(mDynamicDecodeLayer->getStream()));

        SizeType32 finishedSum = 0;
        for (SizeType32 bi = 0; bi < maxBatchSize; ++bi)
        {
            finishedSum += finishedSumHost[bi];
        }
        return numToFinish == static_cast<std::size_t>(finishedSum);
    }
    else
    {
        return false;
    }
}

template <typename T>
void GptDecoder<T>::forwardAsync(DecodingOutput& output, DecodingInput const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto forwardParams = prepareInputs<T>(input, mMaxBatchSize);
    auto outputParams = prepareOutputs<T>(output, input.lengths, mLogProbsTiled);

    mDynamicDecodeLayer->forward(outputParams, forwardParams);

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

    tensorrt_llm::kernels::BeamHypotheses bh;
    bh.nBatchSize = batchSize;
    bh.nBeamWidth = beamWidth;
    bh.nMaxSeqLen = maxSeqLength;
    bh.lengthPenalties = nullptr; // TODO (bhsueh): A gpu tensor used in invokeInsertUnfinishedPath
                                  // default value (1.0f) will be used when it is nullptr
    bh.inputLengths = bufferCast<SizeType32>(*decodingInput.lengths);
    bh.outputIds = bufferCast<TokenIdType>(finalOutputIds);
    bh.logProbs = bufferCast<float>(*mLogProbsTiled);
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

    tensorrt_llm::kernels::invokeAcceptDraftTokensByIds(bufferCast<TokenIdType>(draftTokenIds),
        bufferCast<TokenIdType>(targetTokenIds), bufferCast<SizeType32>(contextLengths),
        bufferCast<SizeType32>(numDraftTokens), bufferCast<SizeType32>(sequenceLengths),
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
        tensorrt_llm::kernels::acceptDraftTokensByLogits(bufferCast<float>(draftLogits),
            const_cast<float**>(reinterpret_cast<float const* const*>(bufferCast<int64_t>(targetLogits))),
            bufferCast<float>(draftProbs), bufferCast<float>(targetProbs), bufferCast<SizeType32>(numDraftTokens),
            reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
                bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(finished)),
            curandState, bufferCast<SizeType32>(batchSlots), batchSize, maxBatchSize, beamWidth, vocabSize,
            vocabSizePadded, maxTokensPerStep, useRandomAcceptThreshold, randomAcceptThreshold, stream->get());
    }
    else if (draftLogits.getDataType() == nvinfer1::DataType::kHALF)
    {
        tensorrt_llm::kernels::acceptDraftTokensByLogits(bufferCast<half>(draftLogits),
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

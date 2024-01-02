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

#include "tensorrt_llm/common/tensorConversion.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/layers/dynamicDecodeLayer.h"

#include <memory>

#include <NvInferRuntime.h>

namespace tc = tensorrt_llm::common;
namespace tl = tensorrt_llm::layers;
namespace tcc = tensorrt_llm::common::conversion;

using namespace tensorrt_llm::runtime;

template <typename T>
GptDecoder<T>::GptDecoder(size_t vocabSize, size_t vocabSizePadded, CudaStreamPtr const& stream)
    : mManager{stream}
{
    bool isFreeBufferAfterForward{false};
    cudaDeviceProp prop;
    tc::check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    auto allocator = std::make_shared<common::CudaAllocator>(mManager);

    mDynamicDecodeLayer = std::make_shared<tensorrt_llm::layers::DynamicDecodeLayer<T>>(
        vocabSize, vocabSizePadded, stream->get(), std::move(allocator), isFreeBufferAfterForward, &prop);

    auto constexpr nvFloatType = TRTDataType<float>::value;
    mLogProbsTiled = mManager.emptyTensor(MemoryType::kGPU, nvFloatType);
}

template <typename T>
void GptDecoder<T>::setup(SamplingConfig const& samplingConfig, size_t batchSize, SizeType maxSequenceLength)
{
    mSamplingConfig = samplingConfig;

    typename layers::DynamicDecodeLayer<T>::SetupParams setupParams;

    setupParams.randomSeed = samplingConfig.randomSeed;

    setupParams.repetition_penalty = samplingConfig.repetitionPenalty;
    setupParams.presence_penalty = samplingConfig.presencePenalty;
    setupParams.frequency_penalty = samplingConfig.frequencyPenalty;
    setupParams.temperature = samplingConfig.temperature;
    setupParams.min_length = samplingConfig.minLength;
    setupParams.normalize_log_probs = samplingConfig.normalizeLogProbs;

    // signed to unsigned
    if (samplingConfig.topK)
    {
        auto const& topK = samplingConfig.topK.value();
        setupParams.runtime_top_k = std::vector<uint32_t>(std::begin(topK), std::end(topK));
    }

    setupParams.runtime_top_p = samplingConfig.topP;
    setupParams.top_p_decay = samplingConfig.topPDecay;
    setupParams.top_p_min = samplingConfig.topPMin;
    setupParams.top_p_reset_ids = samplingConfig.topPResetIds;

    setupParams.beam_search_diversity_rate = samplingConfig.beamSearchDiversityRate;
    setupParams.length_penalty = samplingConfig.lengthPenalty;

    mDynamicDecodeLayer->setup(batchSize, samplingConfig.beamWidth, setupParams);

    mLogProbsTiled->reshape(
        ITensor::makeShape({maxSequenceLength, static_cast<SizeType>(batchSize), samplingConfig.beamWidth}));
    mManager.setZero(*mLogProbsTiled);
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
typename tl::DynamicDecodeLayer<T>::ForwardParams prepareInputs(DecodingInput const& input)
{
    TLLM_CHECK(input.logits->getDataType() == TRTDataType<T>::value);

    auto constexpr ite = 0; // no pipeline parallelism
    typename tl::DynamicDecodeLayer<T>::ForwardParams forwardParams{input.step, ite, input.maxLength,
        input.maxAttentionWindow, input.sinkTokenLength, input.batchSize, tcc::toTllmTensor(*input.logits),
        tcc::toTllmTensor(*input.endIds)};

    if (input.cacheIndirection)
    {
        forwardParams.src_cache_indirection = tcc::toTllmTensor(*input.cacheIndirection);
    }

    if (input.sequenceLimitLength)
    {
        forwardParams.sequence_limit_length = tcc::toTllmTensor(*input.sequenceLimitLength);
    }

    if (input.embeddingBias)
    {
        forwardParams.embedding_bias = tcc::toTllmTensor(*input.embeddingBias);
    }

    if (input.lengths)
    {
        forwardParams.input_lengths = tcc::toTllmTensor(*input.lengths);
    }

    if (input.badWordsList)
    {
        forwardParams.bad_words_list = tcc::toTllmTensor(*input.badWordsList);
    }

    if (input.stopWordsList)
    {
        forwardParams.stop_words_list = tcc::toTllmTensor(*input.stopWordsList);
    }

    if (input.finished)
    {
        forwardParams.finished = tcc::toTllmTensor(*input.finished);
    }

    return forwardParams;
}

template <typename T>
typename tl::DynamicDecodeLayer<T>::OutputParams prepareOutputs(
    DecodingOutput& output, DecodingInput::TensorPtr const& inputLengths, DecodingOutput::TensorPtr& logProbsTiled)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    typename tl::DynamicDecodeLayer<T>::OutputParams outputParams(tcc::toTllmTensor(*output.ids));

    outputParams.newTokens = tcc::toTllmTensor(*output.newTokens);

    if (output.cumLogProbs)
    {
        outputParams.cum_log_probs = tcc::toTllmTensor(*output.cumLogProbs);
    }

    if (output.parentIds)
    {
        outputParams.parent_ids = tcc::toTllmTensor(*output.parentIds);
    }

    if (output.cacheIndirection)
    {
        outputParams.tgt_cache_indirection = tcc::toTllmTensor(*output.cacheIndirection);
    }

    if (output.finished)
    {
        outputParams.finished = tcc::toTllmTensor(*output.finished);
    }

    if (output.finishedSum)
    {
        outputParams.finished_sum = tcc::toTllmTensor(*output.finishedSum);
    }

    if (output.lengths)
    {
        outputParams.sequence_length = tcc::toTllmTensor(*output.lengths);
    }

    if (output.logProbs)
    {
        outputParams.output_log_probs = tcc::toTllmTensor(*output.logProbs);
        outputParams.output_log_probs_tiled = tcc::toTllmTensor(*logProbsTiled);
    }

    outputParams.beamHypotheses = std::make_shared<tensorrt_llm::kernels::BeamHypotheses>();
    if (output.beamHypotheses.outputIdsTgt)
    {
        outputParams.beamHypotheses->output_ids_tgt = bufferCast<int>(*output.beamHypotheses.outputIdsTgt);
    }
    if (output.beamHypotheses.sequenceLengthsTgt)
    {
        outputParams.beamHypotheses->sequence_lengths_tgt = bufferCast<int>(*output.beamHypotheses.sequenceLengthsTgt);
    }
    if (output.beamHypotheses.cumLogProbs)
    {
        outputParams.beamHypotheses->cum_log_probs = bufferCast<float>(*output.beamHypotheses.cumLogProbs);
    }
    if (output.beamHypotheses.normedScores)
    {
        outputParams.beamHypotheses->normed_scores = bufferCast<float>(*output.beamHypotheses.normedScores);
    }
    if (output.beamHypotheses.logProbs)
    {
        outputParams.beamHypotheses->log_probs = bufferCast<float>(*output.beamHypotheses.logProbs);
    }
    if (output.beamHypotheses.minNormedScores)
    {
        outputParams.beamHypotheses->min_normed_scores = bufferCast<float>(*output.beamHypotheses.minNormedScores);
    }
    if (output.beamHypotheses.numBeams)
    {
        outputParams.beamHypotheses->num_beams = bufferCast<int>(*output.beamHypotheses.numBeams);
    }
    if (output.beamHypotheses.isDone)
    {
        outputParams.beamHypotheses->is_done = bufferCast<bool>(*output.beamHypotheses.isDone);
    }
    if (inputLengths)
    {
        outputParams.beamHypotheses->input_lengths = bufferCast<int32_t>(*inputLengths);
    }

    return outputParams;
}

} // namespace

template <typename T>
bool GptDecoder<T>::forward(DecodingOutput& output, DecodingInput const& input)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto forwardParams = prepareInputs<T>(input);
    auto outputParams = prepareOutputs<T>(output, input.lengths, mLogProbsTiled);

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
            finishedSum = BufferManager::pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
            outputParams.finished_sum = tcc::toTllmTensor(*finishedSum);
            finishedSumHost = bufferCast<std::int32_t>(*finishedSum);
        }
        *finishedSumHost = 0;
    }

    mDynamicDecodeLayer->forward(outputParams, forwardParams);

    if (finishedSumHost)
    {
        auto const numToFinish = output.finished->getSize();
        TLLM_CUDA_CHECK(::cudaStreamSynchronize(mDynamicDecodeLayer->getStream()));
        return numToFinish == static_cast<std::size_t>(*finishedSumHost);
    }
    else
    {
        return false;
    }
}

template <typename T>
void GptDecoder<T>::forwardAsync(DecodingOutput& output, DecodingInput const& input)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto forwardParams = prepareInputs<T>(input);
    auto outputParams = prepareOutputs<T>(output, input.lengths, mLogProbsTiled);

    mDynamicDecodeLayer->forward(outputParams, forwardParams);
}

// this should be similar to gatherTree in cpp/tensorrt_llm/thop/gatherTreeOp.cpp
template <typename T>
void GptDecoder<T>::gatherTree(ITensor& finalOutputIds, DecodingOutput const& decodingOutput,
    DecodingInput const& decodingInput, BufferManager const& manager)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const& finalOutputIdsShape = finalOutputIds.getShape();
    auto const& decodingOutputIdsShape = decodingOutput.ids->getShape();
    auto const batchSize = finalOutputIdsShape.d[0];
    auto const beamWidth = finalOutputIdsShape.d[1];
    auto const maxSeqLength = finalOutputIdsShape.d[2];

    TLLM_CHECK_WITH_INFO(beamWidth > 1, "gatherTree is only needed for beam search.");

    TLLM_CHECK_WITH_INFO(decodingOutputIdsShape.d[0] == batchSize,
        common::fmtstr(
            "Decoder batch size (%d) does not match final batch size (%d)", decodingOutputIdsShape.d[0], batchSize));
    TLLM_CHECK_WITH_INFO(decodingOutputIdsShape.d[1] == beamWidth,
        common::fmtstr(
            "Decoder beam width (%d) does not match final beam width (%d)", decodingOutputIdsShape.d[1], beamWidth));
    TLLM_CHECK_WITH_INFO(decodingOutputIdsShape.d[2] <= maxSeqLength,
        common::fmtstr("Decoder seq length size (%d) is too large for final seq length (%d)",
            decodingOutputIdsShape.d[2], maxSeqLength));

    auto const& stream = manager.getStream();

    tensorrt_llm::kernels::invokeInitializeOutput(bufferCast<TokenIdType>(finalOutputIds),
        bufferCast<TokenIdType>(*decodingInput.endIds), batchSize * beamWidth, maxSeqLength, stream.get());
    sync_check_cuda_error();

    tensorrt_llm::kernels::BeamHypotheses beamHypotheses;
    beamHypotheses.sequence_lengths_src = bufferCast<SizeType>(*decodingOutput.lengths);
    beamHypotheses.parent_ids_src = bufferCast<TokenIdType>(*decodingOutput.parentIds);
    beamHypotheses.output_ids_src = bufferCast<TokenIdType>(*decodingOutput.ids);
    beamHypotheses.log_probs_src = bufferCast<float>(*mLogProbsTiled);
    beamHypotheses.max_seq_len = maxSeqLength;
    beamHypotheses.length_penalties
        = nullptr; // TODO (bhsueh) should set length penalties, this should be a gpu tensor When it is set as
                   // nullptr, the kernel will use default value (1.0f) automatically.

    beamHypotheses.output_ids_tgt = bufferCast<TokenIdType>(*decodingOutput.beamHypotheses.outputIdsTgt);
    beamHypotheses.sequence_lengths_tgt = bufferCast<SizeType>(*decodingOutput.beamHypotheses.sequenceLengthsTgt);
    beamHypotheses.cum_log_probs = bufferCast<float>(*decodingOutput.beamHypotheses.cumLogProbs);
    beamHypotheses.normed_scores = bufferCast<float>(*decodingOutput.beamHypotheses.normedScores);
    beamHypotheses.log_probs = bufferCast<float>(*decodingOutput.beamHypotheses.logProbs);
    beamHypotheses.min_normed_scores = bufferCast<float>(*decodingOutput.beamHypotheses.minNormedScores);
    beamHypotheses.num_beams = bufferCast<SizeType>(*decodingOutput.beamHypotheses.numBeams);
    beamHypotheses.is_done = bufferCast<bool>(*decodingOutput.beamHypotheses.isDone);
    beamHypotheses.input_lengths = bufferCast<SizeType>(*decodingInput.lengths);

    // This is where transpose is done
    tensorrt_llm::kernels::invokeInsertUnfinishedPath(beamHypotheses,
        reinterpret_cast<const tensorrt_llm::kernels::FinishedState*>(
            bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*decodingOutput.finished)),
        bufferCast<float>(*decodingOutput.cumLogProbs), batchSize, beamWidth, stream.get());
    sync_check_cuda_error();

    tensorrt_llm::kernels::invokeFinalize(bufferCast<TokenIdType>(finalOutputIds),
        bufferCast<SizeType>(*decodingOutput.lengths), bufferCast<float>(*decodingOutput.cumLogProbs),
        decodingOutput.logProbs ? bufferCast<float>(*decodingOutput.logProbs) : nullptr, beamHypotheses.output_ids_tgt,
        beamHypotheses.sequence_lengths_tgt, beamHypotheses.normed_scores, beamHypotheses.cum_log_probs,
        beamHypotheses.log_probs, beamHypotheses.num_beams, beamHypotheses.input_lengths, beamWidth, maxSeqLength,
        batchSize, stream.get());
    sync_check_cuda_error();

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

namespace tensorrt_llm::runtime
{
template class GptDecoder<float>;
template class GptDecoder<half>;
} // namespace tensorrt_llm::runtime

void IGptDecoder::acceptDraftTokensByIds(const ITensor& targetTokenIds, const ITensor& draftTokenIds,
    const ITensor& contextLengths, const ITensor& numDraftTokens, ITensor& sequenceLengths, const ITensor& finishedVec,
    ITensor& finishedFinal, ITensor& finishedSum, BufferManager::CudaStreamPtr const& stream)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto const targetTokenIdsShape = targetTokenIds.getShape();
    auto const batchSize = targetTokenIdsShape.d[0];
    auto const beamWidth = targetTokenIdsShape.d[1];
    auto const maxSeqLength = targetTokenIdsShape.d[2];
    auto const maxDraftTokens = draftTokenIds.getShape().d[2];

    TLLM_CHECK_WITH_INFO(
        beamWidth == 1, common::fmtstr("Beam width (%d) > 1 is not supported for the speculative decoding", beamWidth));

    TLLM_CHECK_WITH_INFO(draftTokenIds.getShape().d[0] == batchSize,
        common::fmtstr("Draft tokens batch size (%d) is not equal to target batch size (%d)",
            draftTokenIds.getShape().d[0], batchSize));

    TLLM_CHECK_WITH_INFO(contextLengths.getShape().d[0] == batchSize,
        common::fmtstr("Context length batch size (%d) is not equal to batch size (%d)", contextLengths.getShape().d[0],
            batchSize));

    TLLM_CHECK_WITH_INFO(numDraftTokens.getShape().d[0] == batchSize,
        common::fmtstr("Num draft tokens batch size (%d) is not equal to batch size (%d)",
            numDraftTokens.getShape().d[0], batchSize));

    TLLM_CHECK_WITH_INFO(sequenceLengths.getShape().d[0] == batchSize,
        common::fmtstr("Sequence length batch size (%d) is not equal to batch size (%d)",
            sequenceLengths.getShape().d[0], batchSize));

    tensorrt_llm::kernels::invokeAcceptDraftTokensByIds(bufferCast<SizeType>(draftTokenIds),
        bufferCast<SizeType>(targetTokenIds), bufferCast<SizeType>(contextLengths),
        bufferCast<SizeType>(numDraftTokens), bufferCast<SizeType>(sequenceLengths),
        reinterpret_cast<const tensorrt_llm::kernels::FinishedState*>(
            bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(finishedVec)),
        reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
            bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(finishedFinal)),
        bufferCast<int>(finishedSum), batchSize, beamWidth, maxSeqLength, maxDraftTokens, stream->get());

    sync_check_cuda_error();

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void IGptDecoder::acceptDraftTokensByLogits(ITensor& draftLogits, const ITensor& targetLogits, ITensor& draftProbs,
    ITensor& targetProbs, const ITensor& numDraftTokens, ITensor& finished, SizeType vocabSize,
    SizeType vocabSizePadded, bool useRandomAcceptThreshold, float randomAcceptThreshold, curandState_t* curandState,
    BufferManager::CudaStreamPtr const& stream)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto const draftLogitsShape = draftLogits.getShape();
    auto const maxDraftTokens = draftLogitsShape.d[0];
    auto const batchSize = draftLogitsShape.d[1];
    auto const beamWidth = draftLogitsShape.d[2];

    TLLM_CHECK_WITH_INFO(
        beamWidth == 1, common::fmtstr("Beam width (%d) > 1 is not supported for the speculative decoding", beamWidth));

    TLLM_CHECK(draftLogitsShape.d[3] == vocabSize);

    if (draftLogits.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        tensorrt_llm::kernels::acceptDraftTokensByLogits(bufferCast<float>(draftLogits),
            const_cast<float*>(bufferCast<float>(targetLogits)), bufferCast<float>(draftProbs),
            bufferCast<float>(targetProbs), bufferCast<SizeType>(numDraftTokens),
            reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
                bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(finished)),
            curandState, batchSize, beamWidth, vocabSize, vocabSizePadded, maxDraftTokens, useRandomAcceptThreshold,
            randomAcceptThreshold, stream->get());
    }
    else if (draftLogits.getDataType() == nvinfer1::DataType::kHALF)
    {
        tensorrt_llm::kernels::acceptDraftTokensByLogits(bufferCast<half>(draftLogits),
            const_cast<half*>(bufferCast<half>(targetLogits)), bufferCast<half>(draftProbs),
            bufferCast<half>(targetProbs), bufferCast<SizeType>(numDraftTokens),
            reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
                bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(finished)),
            curandState, batchSize, beamWidth, vocabSize, vocabSizePadded, maxDraftTokens, useRandomAcceptThreshold,
            randomAcceptThreshold, stream->get());
    }
    else
    {
        TLLM_THROW("Incorrect logits dtype. Only float32 and float16 are supported");
    }

    sync_check_cuda_error();

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

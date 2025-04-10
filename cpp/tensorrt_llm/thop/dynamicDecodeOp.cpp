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

#include "tensorrt_llm/thop/dynamicDecodeOp.h"

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <c10/cuda/CUDAFunctions.h>
#include <cstdint>

namespace th = torch;

namespace tle = tensorrt_llm::executor;
namespace tr = tensorrt_llm::runtime;
namespace tl = tensorrt_llm::layers;
namespace tk = tensorrt_llm::kernels;

namespace torch_ext
{

template <typename T>
FtDynamicDecode<T>::FtDynamicDecode(size_t const maxBatchSize, size_t const maxBeamWidth, size_t const vocabSize,
    size_t const vocabSizePadded, int const tensorParaSize, int const pipelineParaSize)
{
    TLLM_CHECK_WITH_INFO(vocabSizePadded % tensorParaSize == 0,
        tensorrt_llm::common::fmtstr(
            "vocabSize (%ld) is not multiple of tensorParaSize (%d).", vocabSizePadded, tensorParaSize));

    auto const decodingDomain = tl::DecoderDomain(maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto const currentDeviceId = c10::cuda::current_device();
    auto cudaStreamPtr = std::make_shared<tensorrt_llm::runtime::CudaStream>(stream, currentDeviceId, false);
    auto bufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(cudaStreamPtr);

    mFinishedSum = bufferManager->pinnedPool(
        tr::ITensor::makeShape({static_cast<int32_t>(maxBatchSize)}), nvinfer1::DataType::kINT32);
    mDynamicDecodeLayer
        = std::make_shared<tl::DynamicDecodeLayer<T>>(tle::DecodingMode::Auto(), decodingDomain, bufferManager);
    mBatchSlots = tr::getDefaultBatchSlots(maxBatchSize);
    mDecodingWorkspace = std::make_unique<tensorrt_llm::runtime::DecodingLayerWorkspace>(bufferManager, decodingDomain,
        tensorrt_llm::runtime::TRTDataType<T>::value, mDynamicDecodeLayer->getWorkspaceSize());
}

namespace
{

template <typename T>
void safeInsert(th::optional<th::Tensor>& tensor, std::optional<std::vector<T>>& arg)
{
    if (tensor.has_value())
    {
        auto shape = convert_shape(tensor.value());
        size_t const size = tensorrt_llm::runtime::ITensor::volume(shape);
        auto ptr = get_ptr<T>(tensor.value());
        arg = std::vector<T>(ptr, ptr + size);
    }
}

template <typename T>
void safeUpdate(th::optional<th::Tensor>& tensor, std::optional<tr::ITensor::SharedPtr>& arg)
{
    if (tensor.has_value())
    {
        arg = convert_tensor<T>(tensor.value());
    }
}

template <typename T>
void safeUpdate(th::optional<th::Tensor>& tensor, std::optional<tr::ITensor::SharedConstPtr>& arg)
{
    if (tensor.has_value())
    {
        arg = convert_tensor<T>(tensor.value());
    }
}

template <typename T>
void safeUpdateScalar(th::optional<th::Tensor>& tensor, std::optional<T>& arg, std::string const& name)
{
    if (tensor.has_value())
    {
        auto accessor = tensor->accessor<T, 1>();
        TLLM_CHECK_WITH_INFO(accessor.size(0) == 1, name + " must be a scalar");
        arg = accessor[0];
    }
}

template <typename T>
void safeUpdatePtr(th::optional<th::Tensor>& tensor, T*& ptr)
{
    if (tensor.has_value())
    {
        ptr = get_ptr<T>(tensor.value());
    }
}

} // namespace

template <typename T>
void FtDynamicDecode<T>::setup(size_t const batch_size, size_t const beam_width,
    th::optional<th::Tensor> runtime_top_k_opt, th::optional<th::Tensor> runtime_top_p_opt,
    th::optional<th::Tensor> temperature_opt, th::optional<th::Tensor> repetition_penalty_opt,
    th::optional<th::Tensor> presence_penalty_opt, th::optional<th::Tensor> frequency_penalty_opt,
    th::optional<th::Tensor> min_length_opt, th::optional<th::Tensor> length_penalty_opt,
    th::optional<th::Tensor> early_stopping_opt, th::optional<th::Tensor> beam_search_diversity_rate_opt,
    th::optional<th::Tensor> random_seed_opt, th::optional<th::Tensor> top_p_decay_opt,
    th::optional<th::Tensor> top_p_min_opt, th::optional<th::Tensor> top_p_reset_ids_opt,
    th::optional<th::Tensor> no_repeat_ngram_size_opt, th::optional<th::Tensor> min_p_opt, bool output_log_probs,
    bool cum_log_probs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mBeamWidth = beam_width;

    auto setupParams = std::make_shared<tl::DynamicDecodeSetupParams>();
    auto penaltyParams = std::make_shared<tl::PenaltySetupParams>();
    auto banWordsParams = std::make_shared<tl::BanWordsSetupParams>();
    safeInsert(temperature_opt, penaltyParams->temperature);
    safeInsert(repetition_penalty_opt, penaltyParams->repetitionPenalty);
    safeInsert(presence_penalty_opt, penaltyParams->presencePenalty);
    safeInsert(frequency_penalty_opt, penaltyParams->frequencyPenalty);
    safeInsert(min_length_opt, penaltyParams->minLength);
    safeInsert(no_repeat_ngram_size_opt, banWordsParams->noRepeatNgramSize);
    if (beam_width == 1)
    {
        auto decodingParams = std::make_shared<tl::SamplingSetupParams>();
        safeInsert(runtime_top_k_opt, decodingParams->runtimeTopK);
        safeInsert(runtime_top_p_opt, decodingParams->runtimeTopP);
        safeInsert(top_p_decay_opt, decodingParams->topPDecay);
        safeInsert(top_p_min_opt, decodingParams->topPMin);
        safeInsert(top_p_reset_ids_opt, decodingParams->topPResetIds);
        safeInsert(min_p_opt, decodingParams->runtimeMinP);
        decodingParams->outputLogProbs = std::vector<bool>({output_log_probs});
        decodingParams->cumLogProbs = std::vector<bool>({cum_log_probs});
        safeInsert(random_seed_opt, decodingParams->randomSeed);

        setupParams->decodingParams = decodingParams;
    }
    else
    {
        auto decodingParams = std::make_shared<tl::BeamSearchSetupParams>();
        safeInsert(beam_search_diversity_rate_opt, decodingParams->beamSearchDiversityRate);
        safeInsert(length_penalty_opt, decodingParams->lengthPenalty);
        safeInsert(early_stopping_opt, decodingParams->earlyStopping);
        decodingParams->outputLogProbs = std::vector<bool>({output_log_probs});
        decodingParams->cumLogProbs = std::vector<bool>({cum_log_probs});
        safeInsert(random_seed_opt, decodingParams->randomSeed);

        setupParams->decodingParams = decodingParams;
    }

    // TODO: insert "normalizeLogProbs" and "topKMedusaHeads"

    setupParams->penaltyParams = penaltyParams;
    setupParams->banWordsParams = banWordsParams;
    mDynamicDecodeLayer->setup(
        batch_size, beam_width, tr::ITensor::slice(mBatchSlots, 0, batch_size), setupParams, mDecodingWorkspace);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void FtDynamicDecode<T>::forward(th::Tensor const& logits, int const step, int const maxInputLength,
    int const maxAttentionWindow, int const sinkTokenLength, uint64_t const ite, int const localBatchSize,
    th::Tensor endId, th::optional<th::Tensor> embeddingBiasOpt, th::optional<th::Tensor> inputLengthsOpt,
    th::optional<th::Tensor> sequenceLimitLengthOpt, th::optional<th::Tensor> stopWordsListPtrsOpt,
    th::optional<th::Tensor> stopWordsLensOpt, int32_t const maxStopWordsLen,
    th::optional<th::Tensor> badWordsListPtrsOpt, th::optional<th::Tensor> badWordsLensOpt,
    int32_t const maxBadWordsLen, th::optional<th::Tensor> srcCacheIndirectionOpt, th::Tensor& outputTokenIds,
    th::Tensor& newTokens, th::Tensor& shouldStop, th::optional<th::Tensor> finishedInput,
    th::optional<th::Tensor> finishedOutput, th::optional<th::Tensor> sequenceLengthsOpt,
    th::optional<th::Tensor> cumLogProbsOpt, th::optional<th::Tensor> outputLogProbsOpt,
    th::optional<th::Tensor> outputLogProbsTiledOpt, th::optional<th::Tensor> parentIdsOpt,
    th::optional<th::Tensor> tgtCacheIndirectionOpt, th::optional<th::Tensor> beamHypsOutputIdsCbaOpt,
    th::optional<th::Tensor> beamHypsSeqLenCbaOpt, th::optional<th::Tensor> beamHypsCumLogProbsCbaOpt,
    th::optional<th::Tensor> beamHypsNormedScoresCbaOpt, th::optional<th::Tensor> beamHypsLogProbsCbaOpt,
    th::optional<th::Tensor> beamHypsMinNormedScoresOpt, th::optional<th::Tensor> beamHypsNumBeamsOpt,
    th::optional<th::Tensor> beamHypsIsDoneOpt, bool const useBeamHyps)
{
    TLLM_CHECK_WITH_INFO(mBeamWidth.has_value(), "Beam width is not set. setup() must be called before forward()");
    auto const isBeamSearch = mBeamWidth.value() > 1;

    std::shared_ptr<tl::DecodingInputs> forwardParams;
    tr::ITensor::SharedConstPtr batchSlotsSlice = tr::ITensor::slice(mBatchSlots, 0, localBatchSize);
    if (isBeamSearch)
    {
        forwardParams = std::make_shared<tl::DecodingInputs>(convert_tensor<int>(endId), batchSlotsSlice, step,
            static_cast<int>(ite), localBatchSize, maxAttentionWindow, sinkTokenLength);
    }
    else
    {
        forwardParams = std::make_shared<tl::SamplingInputs>(
            convert_tensor<int>(endId), batchSlotsSlice, step, static_cast<int>(ite), localBatchSize);
    }

    forwardParams->logits = convert_tensor<T>(logits);
    forwardParams->stopCriteriaInputs = std::make_shared<tl::StopCriteriaDecodingInputs>(localBatchSize);
    forwardParams->banWordsInputs = std::make_shared<tl::BanWordsDecodingInputs>(localBatchSize);

    safeUpdate<T>(embeddingBiasOpt, forwardParams->embeddingBias);
    safeUpdate<tr::SizeType32>(inputLengthsOpt, forwardParams->inputLengths);
    safeUpdate<tr::SizeType32>(sequenceLimitLengthOpt, forwardParams->stopCriteriaInputs->sequenceLimitLength);
    safeUpdate<tr::TokenIdType*>(stopWordsListPtrsOpt, forwardParams->stopCriteriaInputs->stopWordsPtr);
    safeUpdate<tr::SizeType32>(stopWordsLensOpt, forwardParams->stopCriteriaInputs->stopWordsLengths);
    forwardParams->stopCriteriaInputs->maxStopWordsLen = maxStopWordsLen;
    safeUpdate<tr::TokenIdType*>(badWordsListPtrsOpt, forwardParams->banWordsInputs->badWordsPtr);
    safeUpdate<tr::SizeType32>(badWordsLensOpt, forwardParams->banWordsInputs->badWordsLengths);
    forwardParams->banWordsInputs->maxBadWordsLen = maxBadWordsLen;
    safeUpdate<tr::SizeType32>(srcCacheIndirectionOpt, forwardParams->srcCacheIndirection);

    tr::ITensor::SharedPtr outputIdsConverted = convert_tensor<tr::TokenIdType>(outputTokenIds);

    std::shared_ptr<tl::BaseDecodingOutputs> outputParams;
    if (isBeamSearch)
    {
        outputParams = std::make_shared<tl::BeamSearchOutputs>(outputIdsConverted);
    }
    else
    {
        outputParams = std::make_shared<tl::BaseDecodingOutputs>(outputIdsConverted);
    }
    outputParams->newTokens = convert_tensor<tr::TokenIdType>(newTokens);
    safeUpdate<tk::FinishedState::UnderlyingType>(finishedInput, forwardParams->finished);
    safeUpdate<tk::FinishedState::UnderlyingType>(finishedOutput, outputParams->finished);
    safeUpdate<tr::SizeType32>(sequenceLengthsOpt, outputParams->sequenceLength);
    safeUpdate<float>(cumLogProbsOpt, outputParams->cumLogProbs);
    safeUpdate<float>(outputLogProbsOpt, outputParams->outputLogProbs);
    safeUpdate<float>(outputLogProbsTiledOpt, outputParams->outputLogProbsTiled);
    safeUpdate<tr::TokenIdType>(parentIdsOpt, outputParams->parentIds);

    tr::SizeType32* finishedSumHost = nullptr;
    if (forwardParams->stopCriteriaInputs->sequenceLimitLength && outputParams->finished.has_value())
    {
        // Skip the initialization and later calculation if there is no limit of sequence length or no finished beam
        outputParams->finishedSum = mFinishedSum;
        finishedSumHost = tr::bufferCast<tr::SizeType32>(*mFinishedSum);
        for (int32_t bi = 0; bi < localBatchSize; ++bi)
        {
            finishedSumHost[bi] = 0;
        }
    }

    if (isBeamSearch)
    {
        auto outputsBeamSearch = std::dynamic_pointer_cast<tl::BeamSearchOutputs>(outputParams);
        TLLM_CHECK_WITH_INFO(tgtCacheIndirectionOpt.has_value(), "tgtCacheIndirection must be set for beam search");
        outputsBeamSearch->tgtCacheIndirection = convert_tensor<int>(tgtCacheIndirectionOpt.value());
        if (useBeamHyps)
        {
            // Additional parameters for beam search
            outputsBeamSearch->beamHypotheses = std::make_unique<tensorrt_llm::kernels::BeamHypotheses>();
            safeUpdatePtr<bool>(beamHypsIsDoneOpt, outputsBeamSearch->beamHypotheses->batchDones);
            safeUpdatePtr<float>(beamHypsCumLogProbsCbaOpt, outputsBeamSearch->beamHypotheses->cumLogProbsCBA);
            safeUpdatePtr<float>(beamHypsLogProbsCbaOpt, outputsBeamSearch->beamHypotheses->logProbsCBA);
            safeUpdatePtr<float>(beamHypsMinNormedScoresOpt, outputsBeamSearch->beamHypotheses->minNormedScoresCBA);
            safeUpdatePtr<float>(beamHypsNormedScoresCbaOpt, outputsBeamSearch->beamHypotheses->normedScoresCBA);
            safeUpdatePtr<tr::SizeType32>(beamHypsNumBeamsOpt, outputsBeamSearch->beamHypotheses->numBeamsCBA);
            safeUpdatePtr<tr::TokenIdType>(beamHypsOutputIdsCbaOpt, outputsBeamSearch->beamHypotheses->outputIdsCBA);
            safeUpdatePtr<tr::SizeType32>(beamHypsSeqLenCbaOpt, outputsBeamSearch->beamHypotheses->sequenceLengthsCBA);
        }
    }

    mDynamicDecodeLayer->forwardAsync(outputParams, forwardParams, mDecodingWorkspace);

    if (finishedSumHost)
    {
        TLLM_CUDA_CHECK(::cudaStreamSynchronize(mDynamicDecodeLayer->getStream()));
        uint32_t numRealFinished = 0;
        for (int32_t bi = 0; bi < localBatchSize; ++bi)
        {
            numRealFinished += finishedSumHost[bi];
        }
        auto const numToFinish = outputParams->finished.value()->getSize();
        auto shouldStopAccessor = shouldStop.accessor<bool, 1>();
        shouldStopAccessor[0] = numToFinish == numRealFinished;
    }
}

DynamicDecodeOp::DynamicDecodeOp(int64_t const maxBatchSize, int64_t const maxBeamWidth, int64_t const vocabSize,
    int64_t const vocabSizePadded, int64_t const tensorParaSize, int64_t const pipelineParaSize,
    at::ScalarType const scalarType)
    : maxBatchSize_(static_cast<tr::SizeType32>(maxBatchSize))
    , maxBeamWidth_(static_cast<tr::SizeType32>(maxBeamWidth))
    , vocabSize_(static_cast<tr::SizeType32>(vocabSize))
    , vocabSizePadded_(static_cast<tr::SizeType32>(vocabSizePadded))
    , tensorParaSize_(static_cast<int>(tensorParaSize))
    , pipelineParaSize_(static_cast<int>(pipelineParaSize))
    , scalarType_(scalarType)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    createInstance();
}

void DynamicDecodeOp::createInstance()
{
    dynamicDecode_.reset();
    switch (scalarType_)
    {
    case at::ScalarType::Float:
        dynamicDecode_ = std::make_unique<FtDynamicDecode<float>>(
            maxBatchSize_, maxBeamWidth_, vocabSize_, vocabSizePadded_, tensorParaSize_, pipelineParaSize_);
        break;
    case at::ScalarType::Half:
        dynamicDecode_ = std::make_unique<FtDynamicDecode<half>>(
            maxBatchSize_, maxBeamWidth_, vocabSize_, vocabSizePadded_, tensorParaSize_, pipelineParaSize_);
        break;
    default: throw std::runtime_error("Wrong tensor type.");
    }
}

void DynamicDecodeOp::setup(int64_t const batchSize, int64_t const beamWidth, th::optional<th::Tensor> runtimeTopKOpt,
    th::optional<th::Tensor> runtimeTopPOpt, th::optional<th::Tensor> temperatureOpt,
    th::optional<th::Tensor> repetitionPenaltyOpt, th::optional<th::Tensor> presencePenaltyOpt,
    th::optional<th::Tensor> frequencyPenaltyOpt, th::optional<th::Tensor> minLengthOpt,
    th::optional<th::Tensor> lengthPenaltyOpt, th::optional<th::Tensor> earlyStoppingOpt,
    th::optional<th::Tensor> beamSearchDiversityRateOpt, th::optional<th::Tensor> randomSeedOpt,
    th::optional<th::Tensor> topPDecayOpt, th::optional<th::Tensor> topPMinOpt,
    th::optional<th::Tensor> topPResetIdsOpt, th::optional<th::Tensor> noRepeatNgramSizeOpt,
    th::optional<th::Tensor> minPOpt, bool outputLogProbs, bool cumLogProbs)
{
    // TODO: Revise DynamicDecodeLayer and make the decode arguments consistent.
    // TODO: add parameters "normalizeLogProbs" and "topKMedusaHeads"
    CHECK_OPTIONAL_CPU_INPUT(runtimeTopKOpt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(runtimeTopPOpt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(temperatureOpt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(repetitionPenaltyOpt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(presencePenaltyOpt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(frequencyPenaltyOpt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(minLengthOpt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(lengthPenaltyOpt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(earlyStoppingOpt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(beamSearchDiversityRateOpt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(randomSeedOpt, torch::kInt64);
    CHECK_OPTIONAL_INPUT(topPDecayOpt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(topPMinOpt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(topPResetIdsOpt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(noRepeatNgramSizeOpt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(minPOpt, torch::kFloat);

    dynamicDecode_->setup(static_cast<tr::SizeType32>(batchSize), static_cast<tr::SizeType32>(beamWidth),
        runtimeTopKOpt, runtimeTopPOpt, temperatureOpt, repetitionPenaltyOpt, presencePenaltyOpt, frequencyPenaltyOpt,
        minLengthOpt, lengthPenaltyOpt, earlyStoppingOpt, beamSearchDiversityRateOpt, randomSeedOpt, topPDecayOpt,
        topPMinOpt, topPResetIdsOpt, noRepeatNgramSizeOpt, minPOpt, outputLogProbs, cumLogProbs);
}

th::Tensor DynamicDecodeOp::forward(
    // Inputs  BS: batchSize, BM: beamWidth, MSL: maxSeqLength, V: vocabSize, VP: vocabSizePadded
    th::Tensor const& logits,                        // [BS, BM, VP], T, variables for input
    int64_t const step,                              //
    int64_t const maxInputLength,                    //
    int64_t const maxAttentionWindow,                //
    int64_t const sinkTokenLength,                   //
    int64_t const ite,                               //
    int64_t const localBatchSize,                    //
    th::Tensor const endId,                          // [BS*BM], int
    th::optional<th::Tensor> embeddingBiasOpt,       // [VP], T
    th::optional<th::Tensor> inputLengthsOpt,        // [BS*BM], int, length of input contexts
    th::optional<th::Tensor> sequenceLimitLengthOpt, // [BS, 1], int
    th::optional<th::Tensor> stopWordsListPtrsOpt,   // [BS][2, stopWordsLength], int64
    th::optional<th::Tensor> stopWordsLensOpt,       // [BS], int
    int64_t const maxStopWordsLen,                   //
    th::optional<th::Tensor> badWordsListPtrsOpt,    // [BS][2, badWordsLength], int64
    th::optional<th::Tensor> badWordsLensOpt,        // [BS], int
    int64_t const maxBadWordsLen,                    //
    th::optional<th::Tensor> srcCacheIndirectionOpt, // [localBS, BM, MSL], int
    // Outputs
    th::Tensor outputTokenIds,                           // [BS, BM, MSL], variables for output
    th::Tensor newTokens,                                // [BS, BM, 1], int
    th::optional<th::Tensor> finishedInput,              // [BS, BM], uint8
    th::optional<th::Tensor> finishedOutput,             // [BS, BM], uint8
    th::optional<th::Tensor> sequenceLengthsOpt,         // [BS*BM], int, length of the current sequences
    th::optional<th::Tensor> cumLogProbsOpt,             // [BS, BM], float
    th::optional<th::Tensor> outputLogProbsOpt,          // [BS, BM, MSL], float
    th::optional<th::Tensor> outputLogProbsTiledOpt,     // [MSL, BS, BM], float, transpose of outputLogProbsOpt
    th::optional<th::Tensor> parentIdsOpt,               // [BS, BM, MSL], int
    th::optional<th::Tensor> tgtCacheIndirectionOpt,     // [localBS, BM, MSL], int
    th::optional<th::Tensor> beamHypsOutputIdsCbaOpt,    // [BS, BM*2, MSL], int
    th::optional<th::Tensor> beamHypsSeqLenCbaOpt,       // [BS, BM*2], int
    th::optional<th::Tensor> beamHypsCumLogProbsCbaOpt,  // [BS, BM*2], float
    th::optional<th::Tensor> beamHypsNormedScoresCbaOpt, // [BS, BM*2], float
    th::optional<th::Tensor> beamHypsLogProbsCbaOpt,     // [BS, BM*2, MSL], float
    th::optional<th::Tensor> beamHypsMinNormedScoresOpt, // [BS], float
    th::optional<th::Tensor> beamHypsNumBeamsOpt,        // [BS], int
    th::optional<th::Tensor> beamHypsIsDoneOpt,          // [BS], bool
    bool const useBeamHyps                               //
)
{
    CHECK_INPUT(logits, scalarType_);
    TLLM_CHECK_WITH_INFO(logits.dim() == 3,
        "logits is of shape (batchSize, beamWidth, vocabSizePadded), but got dim=%d shape=%s", (int) logits.dim(),
        tensorrt_llm::runtime::ITensor::toString(convert_shape(logits)).c_str());
    TLLM_CHECK_WITH_INFO(static_cast<size_t>(logits.size(2)) == vocabSizePadded_,
        "logits is of shape (batchSize, beamWidth, vocabSize(%ld)), but got the last dim=%ld.", vocabSizePadded_,
        static_cast<size_t>(logits.size(2)));
    CHECK_INPUT(endId, torch::kInt32);
    CHECK_OPTIONAL_INPUT(embeddingBiasOpt, scalarType_);
    CHECK_OPTIONAL_INPUT(inputLengthsOpt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(sequenceLimitLengthOpt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(stopWordsListPtrsOpt, torch::kInt64);
    CHECK_OPTIONAL_INPUT(stopWordsLensOpt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(badWordsListPtrsOpt, torch::kInt64);
    CHECK_OPTIONAL_INPUT(badWordsLensOpt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(srcCacheIndirectionOpt, torch::kInt32);
    CHECK_INPUT(outputTokenIds, torch::kInt32);
    CHECK_INPUT(newTokens, torch::kInt32);
    CHECK_OPTIONAL_INPUT(finishedInput, torch::kUInt8);
    CHECK_OPTIONAL_INPUT(finishedOutput, torch::kUInt8);
    CHECK_OPTIONAL_INPUT(sequenceLengthsOpt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(cumLogProbsOpt, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(outputLogProbsOpt, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(outputLogProbsTiledOpt, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(parentIdsOpt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(tgtCacheIndirectionOpt, torch::kInt32);

    th::Tensor shouldStop = torch::zeros({1}, torch::dtype(torch::kBool).requires_grad(false));

    dynamicDecode_->forward(
        // Inputs
        logits, static_cast<int>(step), static_cast<int>(maxInputLength), static_cast<int>(maxAttentionWindow),
        static_cast<int>(sinkTokenLength), static_cast<uint32_t>(ite), static_cast<int>(localBatchSize), endId,
        embeddingBiasOpt, inputLengthsOpt, sequenceLimitLengthOpt, stopWordsListPtrsOpt, stopWordsLensOpt,
        static_cast<int32_t>(maxStopWordsLen), badWordsListPtrsOpt, badWordsLensOpt,
        static_cast<int32_t>(maxBadWordsLen), srcCacheIndirectionOpt,
        // Outputs
        outputTokenIds, newTokens, shouldStop, finishedInput, finishedOutput, sequenceLengthsOpt, cumLogProbsOpt,
        outputLogProbsOpt, outputLogProbsTiledOpt, parentIdsOpt, tgtCacheIndirectionOpt, beamHypsOutputIdsCbaOpt,
        beamHypsSeqLenCbaOpt, beamHypsCumLogProbsCbaOpt, beamHypsNormedScoresCbaOpt, beamHypsLogProbsCbaOpt,
        beamHypsMinNormedScoresOpt, beamHypsNumBeamsOpt, beamHypsIsDoneOpt, useBeamHyps);

    return shouldStop;
}

} // namespace torch_ext

static auto trtllmGptContextDecoderTHS
    = torch::jit::class_<torch_ext::DynamicDecodeOp>("trtllm", "DynamicDecodeOp")
          .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, at::ScalarType>())
          .def("setup", &torch_ext::DynamicDecodeOp::setup)
          .def("forward", &torch_ext::DynamicDecodeOp::forward);

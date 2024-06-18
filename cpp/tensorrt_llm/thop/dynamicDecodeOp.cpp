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

#include "tensorrt_llm/common/tensorConversion.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "tensorrt_llm/thop/torchAllocator.h"

namespace th = torch;

namespace tle = tensorrt_llm::executor;
namespace tr = tensorrt_llm::runtime;
namespace tcc = tensorrt_llm::common::conversion;
namespace tl = tensorrt_llm::layers;

namespace torch_ext
{

template <typename T>
FtDynamicDecode<T>::FtDynamicDecode(size_t const maxBatchSize, size_t const maxBeamWidth, size_t const vocabSize,
    size_t const vocabSizePadded, int const tensorParaSize, int const pipelineParaSize)
    : mFinishedSum(tr::BufferManager::pinned(
        tr::ITensor::makeShape({static_cast<int32_t>(maxBatchSize)}), nvinfer1::DataType::kINT32))
{
    TLLM_CHECK_WITH_INFO(vocabSizePadded % tensorParaSize == 0,
        tensorrt_llm::common::fmtstr(
            "vocabSize (%ld) is not multiple of tensorParaSize (%d).", vocabSizePadded, tensorParaSize));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto allocator = std::make_shared<tensorrt_llm::thop::TorchAllocator>(stream);

    auto const decodingDomain = tl::DecoderDomain(maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded);

    mDynamicDecodeLayer = std::make_shared<tl::DynamicDecodeLayer<T>>(
        tle::DecodingMode::Auto(), decodingDomain, stream, std::move(allocator));
}

namespace
{

template <typename T>
void safeInsert(th::optional<th::Tensor>& tensor, std::optional<std::vector<T>>& arg)
{
    using valueType = T;
    if (tensor.has_value())
    {
        auto ptr = get_ptr<valueType>(tensor.value());
        auto shape = convert_shape(tensor.value());
        size_t const size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        arg = std::vector<valueType>(ptr, ptr + size);
    }
}

template <typename T>
void safeUpdate(th::optional<th::Tensor>& tensor, std::optional<tc::Tensor>& arg)
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
void FtDynamicDecode<T>::setup(size_t const batchSize, size_t const beamWidth, th::optional<th::Tensor> runtimeTopKOpt,
    th::optional<th::Tensor> runtimeTopPOpt, th::optional<th::Tensor> temperatureOpt,
    th::optional<th::Tensor> repetitionPenaltyOpt, th::optional<th::Tensor> presencePenaltyOpt,
    th::optional<th::Tensor> frequencyPenaltyOpt, th::optional<th::Tensor> minLengthOpt,
    th::optional<th::Tensor> lengthPenaltyOpt, th::optional<th::Tensor> earlyStoppingOpt,
    th::optional<th::Tensor> beamSearchDiversityRateOpt, th::optional<th::Tensor> randomSeedOpt,
    th::optional<th::Tensor> topPDecayOpt, th::optional<th::Tensor> topPMinOpt,
    th::optional<th::Tensor> topPResetIdsOpt, th::optional<th::Tensor> noRepeatNgramSizeOpt, bool outputLogProbs,
    bool cumLogProbs)
{
    mBeamWidth = beamWidth;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mDynamicDecodeLayer->setStream(stream);

    auto setupParams = std::make_shared<tl::DynamicDecodeSetupParams>();
    auto penaltyParams = std::make_shared<tl::PenaltySetupParams>();
    auto banWordsParams = std::make_shared<tl::BanWordsSetupParams>();
    safeInsert(temperatureOpt, penaltyParams->temperature);
    safeInsert(repetitionPenaltyOpt, penaltyParams->repetitionPenalty);
    safeInsert(presencePenaltyOpt, penaltyParams->presencePenalty);
    safeInsert(frequencyPenaltyOpt, penaltyParams->frequencyPenalty);
    safeInsert(minLengthOpt, penaltyParams->minLength);

    safeInsert(noRepeatNgramSizeOpt, banWordsParams->noRepeatNgramSize);

    if (beamWidth == 1)
    {
        auto decodingParams = std::make_shared<tl::SamplingSetupParams>();
        safeInsert(runtimeTopKOpt, decodingParams->runtimeTopK);
        safeInsert(runtimeTopPOpt, decodingParams->runtimeTopP);
        safeInsert(topPDecayOpt, decodingParams->topPDecay);
        safeInsert(topPMinOpt, decodingParams->topPMin);
        safeInsert(topPResetIdsOpt, decodingParams->topPResetIds);
        decodingParams->outputLogProbs = std::vector<bool>({outputLogProbs});
        decodingParams->cumLogProbs = std::vector<bool>({cumLogProbs});
        safeInsert(randomSeedOpt, decodingParams->randomSeed);

        setupParams->decodingParams = decodingParams;
    }
    else
    {
        auto decodingParams = std::make_shared<tl::BeamSearchSetupParams>();
        safeInsert(beamSearchDiversityRateOpt, decodingParams->beamSearchDiversityRate);
        safeInsert(lengthPenaltyOpt, decodingParams->lengthPenalty);
        safeInsert(earlyStoppingOpt, decodingParams->earlyStopping);
        decodingParams->outputLogProbs = std::vector<bool>({outputLogProbs});
        decodingParams->cumLogProbs = std::vector<bool>({cumLogProbs});
        safeInsert(randomSeedOpt, decodingParams->randomSeed);

        setupParams->decodingParams = decodingParams;
    }

    // TODO: insert "normalizeLogProbs" and "topKMedusaHeads"

    setupParams->penaltyParams = penaltyParams;
    setupParams->banWordsParams = banWordsParams;
    mDynamicDecodeLayer->setup(batchSize, beamWidth, nullptr, setupParams);
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
    if (isBeamSearch)
    {
        forwardParams = std::make_shared<tl::DecodingInputs>(convert_tensor<int>(endId), step, static_cast<int>(ite),
            localBatchSize, maxAttentionWindow, sinkTokenLength);
    }
    else
    {
        forwardParams = std::make_shared<tl::SamplingInputs>(
            convert_tensor<int>(endId), step, static_cast<int>(ite), localBatchSize);
    }

    forwardParams->logits = convert_tensor<float>(logits);
    forwardParams->stopCriteriaInputs = std::make_shared<tl::StopCriteriaDecodingInputs>(localBatchSize);
    forwardParams->banWordsInputs = std::make_shared<tl::BanWordsDecodingInputs>(localBatchSize);

    safeUpdate<T>(embeddingBiasOpt, forwardParams->embeddingBias);
    safeUpdate<int>(inputLengthsOpt, forwardParams->inputLengths);
    safeUpdate<int>(sequenceLimitLengthOpt, forwardParams->stopCriteriaInputs->sequenceLimitLength);
    safeUpdate<uint64_t>(stopWordsListPtrsOpt, forwardParams->stopCriteriaInputs->stopWordsPtr);
    safeUpdate<int>(stopWordsLensOpt, forwardParams->stopCriteriaInputs->stopWordsLengths);
    forwardParams->stopCriteriaInputs->maxStopWordsLen = maxStopWordsLen;
    safeUpdate<uint64_t>(badWordsListPtrsOpt, forwardParams->banWordsInputs->badWordsPtr);
    safeUpdate<int>(badWordsLensOpt, forwardParams->banWordsInputs->badWordsLengths);
    forwardParams->banWordsInputs->maxBadWordsLen = maxBadWordsLen;
    safeUpdate<int>(srcCacheIndirectionOpt, forwardParams->srcCacheIndirection);

    auto const& outputIdsConverted = convert_tensor<int>(outputTokenIds);

    std::shared_ptr<tl::BaseDecodingOutputs> outputParams;
    if (isBeamSearch)
    {
        outputParams = std::make_shared<tl::BeamSearchOutputs>(outputIdsConverted);
    }
    else
    {
        outputParams = std::make_shared<tl::BaseDecodingOutputs>(outputIdsConverted);
    }
    outputParams->newTokens = std::move(convert_tensor<int>(newTokens));
    safeUpdate<uint8_t>(finishedInput, forwardParams->finished);
    safeUpdate<uint8_t>(finishedOutput, outputParams->finished);
    safeUpdate<int>(sequenceLengthsOpt, outputParams->sequenceLength);
    safeUpdate<float>(cumLogProbsOpt, outputParams->cumLogProbs);
    safeUpdate<float>(outputLogProbsOpt, outputParams->outputLogProbs);
    safeUpdate<float>(outputLogProbsTiledOpt, outputParams->outputLogProbsTiled);
    safeUpdate<int>(parentIdsOpt, outputParams->parentIds);

    std::int32_t* finishedSumHost = nullptr;
    if (forwardParams->stopCriteriaInputs->sequenceLimitLength && outputParams->finished.has_value())
    {
        // Skip the initialization and later calculation if there is no limit of sequence length or no finished beam
        outputParams->finishedSum = tcc::toTllmTensor(*mFinishedSum);
        finishedSumHost = tr::bufferCast<std::int32_t>(*mFinishedSum);
        for (int32_t bi = 0; bi < localBatchSize; ++bi)
        {
            finishedSumHost[bi] = 0;
        }
    }

    if (isBeamSearch)
    {
        auto outputsBeamSearch = std::dynamic_pointer_cast<tl::BeamSearchOutputs>(outputParams);
        TLLM_CHECK_WITH_INFO(tgtCacheIndirectionOpt.has_value(), "tgtCacheIndirection must be set for beam search");
        outputsBeamSearch->tgtCacheIndirection = std::move(convert_tensor<int>(tgtCacheIndirectionOpt.value()));
        if (useBeamHyps)
        {
            // Additional parameters for beam search
            outputsBeamSearch->beamHypotheses = std::make_unique<tensorrt_llm::kernels::BeamHypotheses>();
            safeUpdatePtr<bool>(beamHypsIsDoneOpt, outputsBeamSearch->beamHypotheses->batchDones);
            safeUpdatePtr<float>(beamHypsCumLogProbsCbaOpt, outputsBeamSearch->beamHypotheses->cumLogProbsCBA);
            safeUpdatePtr<float>(beamHypsLogProbsCbaOpt, outputsBeamSearch->beamHypotheses->logProbsCBA);
            safeUpdatePtr<float>(beamHypsMinNormedScoresOpt, outputsBeamSearch->beamHypotheses->minNormedScoresCBA);
            safeUpdatePtr<float>(beamHypsNormedScoresCbaOpt, outputsBeamSearch->beamHypotheses->normedScoresCBA);
            safeUpdatePtr<int>(beamHypsNumBeamsOpt, outputsBeamSearch->beamHypotheses->numBeamsCBA);
            safeUpdatePtr<int>(beamHypsOutputIdsCbaOpt, outputsBeamSearch->beamHypotheses->outputIdsCBA);
            safeUpdatePtr<int>(beamHypsSeqLenCbaOpt, outputsBeamSearch->beamHypotheses->sequenceLengthsCBA);
        }
    }

    mDynamicDecodeLayer->forwardAsync(outputParams, forwardParams);

    if (finishedSumHost)
    {
        TLLM_CUDA_CHECK(::cudaStreamSynchronize(mDynamicDecodeLayer->getStream()));
        int32_t numRealFinished = 0;
        for (int32_t bi = 0; bi < localBatchSize; ++bi)
        {
            numRealFinished += finishedSumHost[bi];
        }
        auto const numToFinish = outputParams->finished->size();
        auto shouldStopAccessor = shouldStop.accessor<bool, 1>();
        shouldStopAccessor[0] = numToFinish == numRealFinished;
    }
}

DynamicDecodeOp::DynamicDecodeOp(int64_t const maxBatchSize, int64_t const maxBeamWidth, int64_t const vocabSize,
    int64_t const vocabSizePadded, int64_t const tensorParaSize, int64_t const pipelineParaSize,
    at::ScalarType const scalarType)
    : maxBatchSize_(static_cast<size_t>(maxBatchSize))
    , maxBeamWidth_(static_cast<size_t>(maxBeamWidth))
    , vocabSize_(static_cast<size_t>(vocabSize))
    , vocabSizePadded_(static_cast<size_t>(vocabSizePadded))
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
    th::optional<th::Tensor> topPResetIdsOpt, th::optional<th::Tensor> noRepeatNgramSizeOpt, bool outputLogProbs,
    bool cumLogProbs)
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
    CHECK_OPTIONAL_CPU_INPUT(noRepeatNgramSizeOpt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(beamSearchDiversityRateOpt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(randomSeedOpt, torch::kInt64);
    CHECK_OPTIONAL_INPUT(topPDecayOpt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(topPMinOpt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(topPResetIdsOpt, torch::kInt32);

    dynamicDecode_->setup(static_cast<size_t>(batchSize), static_cast<size_t>(beamWidth), runtimeTopKOpt,
        runtimeTopPOpt, temperatureOpt, repetitionPenaltyOpt, presencePenaltyOpt, frequencyPenaltyOpt, minLengthOpt,
        lengthPenaltyOpt, earlyStoppingOpt, beamSearchDiversityRateOpt, randomSeedOpt, topPDecayOpt, topPMinOpt,
        topPResetIdsOpt, noRepeatNgramSizeOpt, outputLogProbs, cumLogProbs);
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
        tensorrt_llm::common::vec2str(convert_shape(logits)).c_str());
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

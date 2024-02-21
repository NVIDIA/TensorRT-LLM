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

#include "tests/layers/dynamicDecodeLayerTest.h"
#include <algorithm>

namespace tensorrt_llm::tests::layers::sampling
{

// TODO(nkorobov):
// Add tests for
// - finished states
// - finished sum
// - max length
// - repeat n grams
// - padded vocab
// - beam search

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::layers;
using namespace tensorrt_llm::common;

namespace tk = tensorrt_llm::kernels;
namespace tcc = tensorrt_llm::common::conversion;
namespace trk = tensorrt_llm::runtime::kernels;

constexpr float EPSILON = 1e-20f;

inline bool almostEqual(float a, float b, float atol = 1e-5, float rtol = 1e-8)
{
    // Params: a = value to compare and b = reference
    // This function follows implementation of numpy.isclose(), which checks
    //   abs(a - b) <= (atol + rtol * abs(b)).
    // Note that the inequality above is asymmetric where b is considered as
    // a reference value. To account into both absolute/relative errors, it
    // uses absolute tolerance and relative tolerance at the same time. The
    // default values of atol and rtol borrowed from numpy.isclose(). For the
    // case of nan value, the result will be true.
    if (isnan(a) && isnan(b))
    {
        return true;
    }

    if (isinf(a) && isinf(b))
    {
        return true;
    }
    return fabs(a - b) <= (atol + rtol * fabs(b));
}

template <typename T>
bool compareValues(T* out, T* ref, size_t size)
{
    bool isFp32 = sizeof(T) == 4;
    float atol = isFp32 ? 1e-4f : 1e-3f;
    float rtol = isFp32 ? 1e-2f : 1e-1f;

    size_t failures = 0;
    float relativeGap = 0.0f;

    for (size_t i = 0; i < size; ++i)
    {
        // The values for the output and the reference.
        float a = (float) out[i];
        float b = (float) ref[i];

        bool ok = almostEqual(a, b, atol, rtol);
        // Print the error.
        if (!ok && failures < 4)
        {
            TLLM_LOG_DEBUG(">> invalid result for i=%lu:", i);
            TLLM_LOG_DEBUG(">>    found......: %10.6f", a);
            TLLM_LOG_DEBUG(">>    expected...: %10.6f", b);
            TLLM_LOG_DEBUG(">>    error......: %.6f", fabsf(a - b));
            TLLM_LOG_DEBUG(">>    tol........: %.6f", atol + rtol * fabs(b));
        }
        // Update the number of failures.
        failures += ok ? 0 : 1;
        // Update the relative gap.
        relativeGap += fabsf(a - b) / (fabsf(b) + EPSILON);
    }

    relativeGap /= size;

    // Allow not matched up to 0% elements.
    size_t tolFailures = (size_t) (0.0 * size);
    TLLM_LOG_DEBUG("check... : %-50s (failures: %.2f%% atol: %.2e rtol: %.2e rel_gap: %.2e%%)",
        failures <= tolFailures ? "....OK" : "FAILED", 100. * failures / size, atol, rtol, 100. * relativeGap);
    return failures <= tolFailures;
}

template bool compareValues(float* out, float* ref, size_t size);
template bool compareValues(half* out, half* ref, size_t size);

template <typename T>
void DynamicDecodeLayerTest<T>::SetUp()
{
    mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
    mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);

    mAllocator = std::make_shared<tensorrt_llm::common::CudaAllocator>(*mBufferManager);

    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&mDeviceProp, device);

    auto const decodingMode = mBeamWidth == 1 ? DecodingMode::TopKTopP() : DecodingMode::BeamSearch();

    mDecodeLayer = std::make_shared<tensorrt_llm::layers::DynamicDecodeLayer<T>>(decodingMode, mMaxBatchSize,
        mBeamWidth, mVocabSize, mVocabSizePadded, mStream->get(), mAllocator, &mDeviceProp);
}

template <typename T>
void DynamicDecodeLayerTest<T>::setup(uint64_t seed, SamplingParams const& params)
{
    auto const dataType = TRTDataType<T>::value;

    // clang-format off

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1, 0.0)
    mTestLogitsInit = {
            -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, // step 0
            -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, // step 1
            -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, -FLT_MAX, // step 2
            -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX  // step 3
    };

    // clang-format on

    mLogitsDevice = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mBeamWidth, mVocabSizePadded}), dataType);
    mRuntimeLogitsHost
        = mBufferManager->pinned(ITensor::makeShape({mBatchSize, mBeamWidth, mVocabSizePadded}), dataType);

    mSeqLengthsDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
    mContextLengthDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
    mFinishedDevice = mBufferManager->gpu(
        ITensor::makeShape({mMaxBatchSize}), TRTDataType<tk::FinishedState::UnderlyingType>::value);
    mFinishedSumDevice = mBufferManager->pinned(ITensor::makeShape({1}), nvinfer1::DataType::kFLOAT);
    mOutputIdsDevice
        = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize, mBeamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);
    mNewTokens = mBufferManager->pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
    mEndIdsDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

    mEmbeddingBiasHost = mBufferManager->pinned(ITensor::makeShape({mMaxBatchSize, mVocabSizePadded}), dataType);
    mEmbeddingBiasDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize, mVocabSizePadded}), dataType);

    mRefLogProbsHost
        = mBufferManager->pinned(ITensor::makeShape({mMaxBatchSize, mMaxSeqLen}), nvinfer1::DataType::kFLOAT);
    mOutputLogProbsDevice
        = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize, mMaxSeqLen}), nvinfer1::DataType::kFLOAT);
    mOutputLogProbsTiledDevice
        = mBufferManager->gpu(ITensor::makeShape({mMaxSeqLen, mMaxBatchSize}), nvinfer1::DataType::kFLOAT);

    mCumLogProbsDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kFLOAT);

    mMaxBadWordsLen = getMaxWordsLen(params.badWords);
    mMaxStopWordsLen = getMaxWordsLen(params.stopWords);

    mBadWords
        = mBufferManager->pinned(ITensor::makeShape({mMaxBatchSize, 2, mMaxBadWordsLen}), nvinfer1::DataType::kINT32);
    mBadWordsLens = mBufferManager->pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
    mBadWordsPtrs = mBufferManager->pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT64);

    mStopWords
        = mBufferManager->pinned(ITensor::makeShape({mMaxBatchSize, 2, mMaxStopWordsLen}), nvinfer1::DataType::kINT32);
    mStopWordsLens = mBufferManager->pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
    mStopWordsPtrs = mBufferManager->pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT64);

    mBatchSlots = mBufferManager->pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

    trk::invokeFill(*mSeqLengthsDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mContextLengthDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mFinishedDevice, uint8_t{0}, *mStream);
    trk::invokeFill(*mOutputIdsDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mEmbeddingBiasDevice, T{0.0f}, *mStream);
    trk::invokeFill(*mCumLogProbsDevice, float{0.0f}, *mStream);
    trk::invokeFill(*mOutputLogProbsDevice, float{0.0f}, *mStream);
    trk::invokeFill(*mOutputLogProbsTiledDevice, float{0.0f}, *mStream);
    trk::invokeFill(*mRefLogProbsHost, float{0.0f}, *mStream);
    trk::invokeFill(*mEndIdsDevice, int32_t{mEndId}, *mStream);

    auto batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
    for (SizeType bi = 0; bi < mBatchSize; ++bi)
    {
        batchSlotsPtr[bi] = 2 * bi;
    }

    if (params.useBias)
    {
        auto embeddingBiasHostPtr = bufferCast<T>(*mEmbeddingBiasHost);
        for (SizeType bi = 0; bi < mMaxBatchSize; bi++)
        {
            for (SizeType vi = 0; vi < mVocabSizePadded; vi++)
            {
                embeddingBiasHostPtr[bi * mVocabSizePadded + vi] = 2 <= vi && vi < 6 ? T{2.0f} : T{0.0f};
            }
        }
        mBufferManager->copy(*mEmbeddingBiasHost, *mEmbeddingBiasDevice);
    }

    mLogitsVec.resize(mBatchSize);
    for (SizeType bi = 0; bi < mBatchSize; ++bi)
    {
        auto logitsSlice = ITensor::slice(mLogitsDevice, bi, 1);
        mLogitsVec[bi] = tcc::toTllmTensor(*logitsSlice);
    }

    typename DynamicDecodeLayer<T>::SetupParams setupParams;
    setupParams.randomSeed = std::make_optional<std::vector<uint64_t>>({seed});
    setupParams.temperature
        = params.temperatures.size() ? std::make_optional<std::vector<float>>(params.temperatures) : std::nullopt;
    setupParams.runtime_top_k
        = params.topKs.size() ? std::make_optional<std::vector<uint32_t>>(params.topKs) : std::nullopt;
    setupParams.runtime_top_p
        = params.topPs.size() ? std::make_optional<std::vector<float>>(params.topPs) : std::nullopt;
    setupParams.repetition_penalty = params.repetitionPenalties.size()
        ? std::make_optional<std::vector<float>>(params.repetitionPenalties)
        : std::nullopt;
    setupParams.presence_penalty = params.presencePenalties.size()
        ? std::make_optional<std::vector<float>>(params.presencePenalties)
        : std::nullopt;
    setupParams.frequency_penalty = params.frequencyPenalties.size()
        ? std::make_optional<std::vector<float>>(params.frequencyPenalties)
        : std::nullopt;
    setupParams.min_length
        = params.minLengths.size() ? std::make_optional<std::vector<int32_t>>(params.minLengths) : std::nullopt;
    setupParams.top_p_decay = params.decay.size() ? std::make_optional<std::vector<float>>(params.decay) : std::nullopt;
    setupParams.top_p_min
        = params.minTopP.size() ? std::make_optional<std::vector<float>>(params.minTopP) : std::nullopt;
    setupParams.top_p_reset_ids
        = params.topPResetIds.size() ? std::make_optional<std::vector<int32_t>>(params.topPResetIds) : std::nullopt;
    setupParams.normalize_log_probs = {false};

    initXWordsTensors(batchSlotsPtr, bufferCast<SizeType>(*mBadWords),
        reinterpret_cast<SizeType**>(bufferCast<int64_t>(*mBadWordsPtrs)), bufferCast<SizeType>(*mBadWordsLens),
        mMaxBadWordsLen, params.badWords);
    initXWordsTensors(batchSlotsPtr, bufferCast<SizeType>(*mStopWords),
        reinterpret_cast<SizeType**>(bufferCast<int64_t>(*mStopWordsPtrs)), bufferCast<SizeType>(*mStopWordsLens),
        mMaxStopWordsLen, params.stopWords);

    mDecodeLayer->setup(mBatchSize, mBeamWidth, batchSlotsPtr, setupParams);

    mStream->synchronize();
}

template <typename T>
SizeType DynamicDecodeLayerTest<T>::getMaxWordsLen(std::vector<std::vector<std::vector<SizeType>>> const& inputWords)
{
    SizeType maxWordsLen = 0;
    for (const auto& batchWords : inputWords)
    {
        SizeType wordsLen = 0;
        for (const auto& words : batchWords)
        {
            wordsLen += words.size();
        }
        if (wordsLen == batchWords.size())
        {
            wordsLen += 1;
        }
        maxWordsLen = std::max(maxWordsLen, wordsLen);
    }
    return maxWordsLen;
}

template <typename T>
void DynamicDecodeLayerTest<T>::initXWordsTensors(SizeType* batchSlotsPtr, SizeType* wordsData, SizeType** wordsPtr,
    SizeType* wordsLenData, SizeType maxWordsLen, std::vector<std::vector<std::vector<SizeType>>> const& inputWords)
{
    std::fill(wordsData, wordsData + mMaxBatchSize * 2 * maxWordsLen, -1);
    for (SizeType bi = 0; bi < inputWords.size(); bi++)
    {
        auto const batchSlot = batchSlotsPtr[bi];
        SizeType totalLen = 0;
        for (SizeType wi = 0; wi < inputWords[bi].size(); ++wi)
        {
            for (SizeType si = 0; si < inputWords[bi][wi].size(); ++si)
            {
                wordsData[batchSlot * 2 * maxWordsLen + 0 * maxWordsLen + totalLen + si] = inputWords[bi][wi][si];
            }
            totalLen += inputWords[bi][wi].size();
            // Do not add value if words is empty
            if (totalLen > 0)
            {
                wordsData[batchSlot * 2 * maxWordsLen + 1 * maxWordsLen + wi] = totalLen;
            }
        }
    }

    for (SizeType bi = 0; bi < inputWords.size(); bi++)
    {
        auto const batchSlot = batchSlotsPtr[bi];
        wordsPtr[batchSlot] = wordsData + batchSlot * 2 * maxWordsLen;

        wordsLenData[batchSlot] = maxWordsLen;
    }
}

template <typename T>
typename DynamicDecodeLayer<T>::ForwardParams DynamicDecodeLayerTest<T>::createInputTensors(int32_t step)
{
    constexpr int32_t ite = 0;
    typename DynamicDecodeLayer<T>::ForwardParams forwardParams(
        step, ite, mMaxInputLen, mMaxSeqLen, mSinkTokenLength, mBatchSize, tcc::toTllmTensor(*mEndIdsDevice));

    forwardParams.embedding_bias = tcc::toTllmTensor(*mEmbeddingBiasDevice);

    forwardParams.finished = tcc::toTllmTensor(*mFinishedDevice);

    forwardParams.batch_slots = tcc::toTllmTensor(*mBatchSlots);

    if (mUseLogitsVec)
    {
        forwardParams.logits_vec = mLogitsVec;
    }
    else
    {
        forwardParams.logits = tcc::toTllmTensor(*mLogitsDevice);
    }

    forwardParams.bad_words_ptr = tcc::toTllmTensor(*mBadWordsPtrs);
    forwardParams.bad_words_lengths = tcc::toTllmTensor(*mBadWordsLens);
    forwardParams.max_bad_words_len = mMaxBadWordsLen;

    forwardParams.stop_words_ptr = tcc::toTllmTensor(*mStopWordsPtrs);
    forwardParams.stop_words_lengths = tcc::toTllmTensor(*mStopWordsLens);
    forwardParams.max_stop_words_len = mMaxStopWordsLen;

    // TODO(nkorobov): extend to
    // std::optional<tc::Tensor> src_cache_indirection;
    // std::optional<tc::Tensor> sequence_limit_length;
    // std::optional<tc::Tensor> input_lengths;
    // std::optional<tc::Tensor> no_repeat_ngram_size;
    // std::optional<std::vector<tc::Tensor>> logits_vec;

    return forwardParams;
}

template <typename T>
typename DynamicDecodeLayer<T>::OutputParams DynamicDecodeLayerTest<T>::createOutputTensors()
{
    typename DynamicDecodeLayer<T>::OutputParams outputParams(tcc::toTllmTensor(*mOutputIdsDevice));

    outputParams.sequence_length = tcc::toTllmTensor(*mSeqLengthsDevice);

    outputParams.finished = tcc::toTllmTensor(*mFinishedDevice);

    outputParams.finished_sum = tcc::toTllmTensor(*mFinishedSumDevice);

    outputParams.cum_log_probs = tcc::toTllmTensor(*mCumLogProbsDevice);

    outputParams.newTokens = tcc::toTllmTensor(*mNewTokens);

    outputParams.output_log_probs = tcc::toTllmTensor(*mOutputLogProbsDevice);

    outputParams.output_log_probs_tiled = tcc::toTllmTensor(*mOutputLogProbsTiledDevice);

    // TODO(nkorobov): extend to
    // std::optional<tc::Tensor> parent_ids;
    // std::optional<tc::Tensor> tgt_cache_indirection;
    // std::shared_ptr<kernels::BeamHypotheses> beamHypotheses;

    return outputParams;
}

template <typename T>
void DynamicDecodeLayerTest<T>::batchCopy(int32_t step)
{
    const auto logitsHost = ITensor::wrap(mTestLogitsInit.data() + step * mVocabSizePadded,
        std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF,
        ITensor::makeShape({1, mVocabSizePadded}));
    for (int32_t bi = 0; bi < mBatchSize; ++bi)
    {
        auto logitsDeviceView = ITensor::slice(mLogitsDevice, bi, 1);
        mBufferManager->copy(*logitsHost, *logitsDeviceView);
    }
    mLogitsRefHost = mBufferManager->copyFrom(*mLogitsDevice, tensorrt_llm::runtime::MemoryType::kCPU);
}

template <typename T>
bool DynamicDecodeLayerTest<T>::checkResult(int32_t* outputIds, std::vector<std::set<int32_t>> const& expectedIds,
    int32_t* seqLens, int32_t leadingDim, int32_t stride, int32_t step)
{
    assert(expectedIds.size() == leadingDim * stride);
    int failures = 0;
    auto const batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
    for (int32_t i = 0; i < leadingDim * stride; ++i)
    {
        int32_t s = i / stride;
        int32_t b = i % stride;
        auto const batchSlot = batchSlotsPtr[b];
        if (seqLens[batchSlot] <= step)
        {
            continue;
        }
        std::set<int32_t> expts = expectedIds.at(i + step * stride);
        auto bid = batchSlot;
        const auto outputId = outputIds[bid * leadingDim + s];
        if (expts.count(outputId) == 0)
        {
            if (failures < 10)
            {
                std::stringstream ss;
                ss << " - Fail "
                   << " (step=" << s << ", batch=" << b << ") "
                   << "actual=" << outputId << ", expected";
                for (auto& expt : expts)
                {
                    ss << " " << expt;
                }
                TLLM_LOG_DEBUG("%s", ss.str().c_str());
            }
            ++failures;
        }
    }
    TLLM_LOG_DEBUG(
        "check...%6s : failures: %d / %d", failures == 0 ? "....OK" : "FAILED", failures, leadingDim * stride);
    return failures == 0;
}

template <typename T>
void DynamicDecodeLayerTest<T>::fillRefLogits(
    int32_t const* seqLenHost, std::vector<std::set<int32_t>> const& expectedOutputIds, SizeType step)
{
    auto const batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
    auto const runtimeLogitsHost = bufferCast<T>(*mRuntimeLogitsHost);
    for (SizeType bi = 0; bi < mBatchBeam; ++bi)
    {
        auto const batchSlot = batchSlotsPtr[bi];
        if (seqLenHost[batchSlot] <= step)
        {
            continue;
        }
        auto& expectedSet = expectedOutputIds[step * mBatchBeam + bi];
        TLLM_CHECK(expectedSet.size() == 1);
        auto expectedToken = *expectedSet.begin();
        bufferCast<float>(*mRefLogProbsHost)[batchSlot * mMaxSeqLen + step]
            = logf(runtimeLogitsHost[bi * mVocabSizePadded + expectedToken]);
    }
}

template <typename T>
void DynamicDecodeLayerTest<T>::runTestImpl(
    std::vector<std::set<int32_t>> const& expectedOutputIds, SamplingParams const& params, int32_t endId)
{
    mEndId = endId == -1 ? mVocabSize - 1 : endId;

    bool greedySearch
        = std::all_of(expectedOutputIds.begin(), expectedOutputIds.end(), [](auto v) { return v.size() == 1; });
    for (uint64_t seed = 0; seed < mMaxSeed; ++seed)
    {
        setup(seed, params);

        int32_t step = mMaxInputLen;
        auto inputTensors = createInputTensors(step);
        auto outputTensors = createOutputTensors();

        for (step = mMaxInputLen; step < mMaxOutputLen; ++step)
        {
            // Reset by the test value since the sampling layer internally update the logit buffer.
            batchCopy(step);
            inputTensors.step = step;
            mDecodeLayer->forward(outputTensors, inputTensors);
            mStream->synchronize();
            auto const newTokensHost = mBufferManager->copyFrom(*mNewTokens, tensorrt_llm::runtime::MemoryType::kCPU);
            auto const seqLenHost
                = mBufferManager->copyFrom(*mSeqLengthsDevice, tensorrt_llm::runtime::MemoryType::kCPU);
            auto const logitsHost = mBufferManager->copyFrom(*mLogitsDevice, tensorrt_llm::runtime::MemoryType::kCPU);
            mBufferManager->copy(
                mDecodeLayer->getRuntimeLogitsDevice(), *mRuntimeLogitsHost, tensorrt_llm::runtime::MemoryType::kGPU);
            mStream->synchronize();

            if (greedySearch)
            {
                fillRefLogits(bufferCast<int32_t>(*seqLenHost), expectedOutputIds, step);
            }

            {
                bool passed = checkResult(bufferCast<int32_t>(*newTokensHost), expectedOutputIds,
                    bufferCast<int32_t>(*seqLenHost), 1, mBatchBeam, step);
                EXPECT_TRUE(passed) << "New tokens check failed at seed " << seed;
                if (!passed)
                {
                    std::stringstream ss;
                    ss << "New tokens ids:" << std::endl << *newTokensHost;
                    TLLM_LOG_DEBUG(ss.str());
                }
            }

            // Check if logits were not modified in-place
            {
                bool passed = compareValues(bufferCast<T>(*mLogitsRefHost), bufferCast<T>(*logitsHost),
                    mBatchSize * mBeamWidth * mVocabSizePadded);
                EXPECT_TRUE(passed) << "Unmodified logits check failed at seed " << seed;
            }
        }

        mStream->synchronize();

        auto const outputIdsHost = mBufferManager->copyFrom(*mOutputIdsDevice, tensorrt_llm::runtime::MemoryType::kCPU);
        auto const seqLenHost = mBufferManager->copyFrom(*mSeqLengthsDevice, tensorrt_llm::runtime::MemoryType::kCPU);
        auto const logProbsHost
            = mBufferManager->copyFrom(*mOutputLogProbsDevice, tensorrt_llm::runtime::MemoryType::kCPU);

        {
            bool passed = checkResult(bufferCast<int32_t>(*outputIdsHost), expectedOutputIds,
                bufferCast<int32_t>(*seqLenHost), mMaxSeqLen, mBatchBeam, /* step */ 0);
            EXPECT_TRUE(passed) << "Output Ids check failed at seed " << seed;
            if (!passed)
            {
                std::stringstream ss;
                ss << "Actual output ids:" << std::endl << *outputIdsHost;
                TLLM_LOG_DEBUG(ss.str());
            }
        }

        if (greedySearch)
        {
            bool passed = compareValues(
                bufferCast<float>(*logProbsHost), bufferCast<float>(*mRefLogProbsHost), mMaxSeqLen * mMaxBatchSize);
            EXPECT_TRUE(passed) << "Log probs check failed at seed " << seed;
        }
    }
}

template <typename T>
void DynamicDecodeLayerTest<T>::runTest(
    std::vector<std::set<int32_t>> const& expectedOutputIds, SamplingParams const& params, int32_t endId)
{
    TLLM_LOG_DEBUG("Run test with linear logits");
    mUseLogitsVec = false;
    runTestImpl(expectedOutputIds, params, endId);
    TLLM_LOG_DEBUG("Run test with vectorized logits");
    mUseLogitsVec = true;
    runTestImpl(expectedOutputIds, params, endId);
}

template class DynamicDecodeLayerTest<float>;
template class DynamicDecodeLayerTest<half>;

TYPED_TEST_SUITE(DynamicDecodeLayerTest, FloatAndHalfTypes);

TYPED_TEST(DynamicDecodeLayerTest, TopK)
{
    uint32_t topK = 2;
    float topP = 0.0f;
    SamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, // step 0
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, // step 1
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, // step 2
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopK1TopP0)
{
    uint32_t topK = 1;
    float topP = 0.0f;
    SamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, BatchTopK)
{
    std::vector<uint32_t> topKs = {2, 1, 1, 2, 1, 1};
    SamplingParams params;
    params.topKs = topKs;
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4}, {4}, {4, 5}, {4}, {4}, // step 0
        {0, 1}, {0}, {0}, {0, 1}, {0}, {0}, // step 1
        {2, 3}, {2}, {2}, {2, 3}, {2}, {2}, // step 2
        {0, 1}, {0}, {0}, {0, 1}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKTopP)
{
    uint32_t topK = 2;
    float topP = 0.3;
    SamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, BatchTopKTopP)
{
    std::vector<uint32_t> topKs = {2, 2, 1, 2, 2, 1};
    float topP = 0.3;
    SamplingParams params;
    params.topKs = topKs;
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKBatchTopP)
{
    uint32_t topK = 2;
    std::vector<float> topPs = {0.5, 0.3, 0.5, 0.5, 0.3, 0.5};
    SamplingParams params;
    params.topKs = {topK};
    params.topPs = topPs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4}, {4, 5}, {4, 5}, {4}, {4, 5}, // step 0
        {0, 1}, {0}, {0, 1}, {0, 1}, {0}, {0, 1}, // step 1
        {2, 3}, {2}, {2, 3}, {2, 3}, {2}, {2, 3}, // step 2
        {0, 1}, {0}, {0, 1}, {0, 1}, {0}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, BatchTopKBatchTopP)
{
    std::vector<uint32_t> topKs = {2, 2, 0, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    SamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4}, {4, 5}, {4, 5}, {4}, {4}, // step 0
        {0, 1}, {0}, {0, 1}, {0, 1}, {0}, {0}, // step 1
        {2, 3}, {2}, {2, 3}, {2, 3}, {2}, {2}, // step 2
        {0, 1}, {0}, {0, 1}, {0, 1}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, InvalidArgsZeroTopK)
{
    uint32_t topK = 0;
    SamplingParams params;
    params.topKs = {topK};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, InvalidArgsZeroTopP)
{
    float topP = 0;
    SamplingParams params;
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, InvalidArgsZeroTopKTopP)
{
    uint32_t topK = 0;
    float topP = 0;
    SamplingParams params;
    params.topPs = {topP};
    params.topKs = {topK};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, InvalidArgsZeroBatchTopKTopP)
{
    std::vector<uint32_t> topKs = {0, 0, 0, 0, 0, 0};
    float topP = 0;
    SamplingParams params;
    params.topPs = {topP};
    params.topKs = topKs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, InvalidArgsZeroTopKBatchTopP)
{
    uint32_t topK = 0;
    std::vector<float> topPs = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    SamplingParams params;
    params.topPs = topPs;
    params.topKs = {topK};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, InvalidArgsBatchTopKContainZero)
{
    std::vector<uint32_t> topKs = {2, 1, 0, 0, 2, 1};
    SamplingParams params;
    params.topKs = topKs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4}, {4}, {4}, {4, 5}, {4}, // step 0
        {0, 1}, {0}, {0}, {0}, {0, 1}, {0}, // step 1
        {2, 3}, {2}, {2}, {2}, {2, 3}, {2}, // step 2
        {0, 1}, {0}, {0}, {0}, {0, 1}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, InvalidArgsBatchTopKTopPContainZero)
{
    std::vector<uint32_t> topKs = {2, 2, 1, 0, 2, 0};
    float topP = 0.0;
    SamplingParams params;
    params.topPs = {topP};
    params.topKs = topKs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4, 5}, {4}, {4}, {4, 5}, {4}, // step 0
        {0, 1}, {0, 1}, {0}, {0}, {0, 1}, {0}, // step 1
        {2, 3}, {2, 3}, {2}, {2}, {2, 3}, {2}, // step 2
        {0, 1}, {0, 1}, {0}, {0}, {0, 1}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, InvalidArgsBatchTopKBatchTopPContainZero)
{
    std::vector<uint32_t> topKs = {0, 2, 1, 2, 2, 0};
    std::vector<float> topPs = {0.0, 0.3, 0.9, 0.0, 0.3, 0.5};
    SamplingParams params;
    params.topPs = topPs;
    params.topKs = topKs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4, 5}, {4}, {4, 5}, // step 0
        {0}, {0}, {0}, {0, 1}, {0}, {0, 1}, // step 1
        {2}, {2}, {2}, {2, 3}, {2}, {2, 3}, // step 2
        {0}, {0}, {0}, {0, 1}, {0}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPTemperature)
{
    float temperature = 0.05f;
    SamplingParams params;
    params.temperatures = {temperature};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPTemperatureBatch)
{
    std::vector<float> temperatures = {0.05f, 1e3f, 1.0f, 1.0f, 0.05f, 1.0f};
    SamplingParams params;
    params.temperatures = temperatures;
    params.topPs = {0.5f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        {4}, {4, 5, 6, 7}, {4, 5}, {4, 5}, {4}, {4, 5}, // step 0
        {0}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0, 1}, // step 1
        {2}, {2, 3, 4, 5}, {2, 3}, {2, 3}, {2}, {2, 3}, // step 2
        {0}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPRepetitionPenalty)
{
    uint32_t topK = 1;
    float repetitionPenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPRepetitionPenaltiesBatch)
{
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = repetitionPenalties;
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPPresencePenalty)
{
    float presencePenalty = 1e9f;
    SamplingParams params;
    params.presencePenalties = {presencePenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPPresencePenaltiesBatch)
{
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.presencePenalties = presencePenalties;
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPFrequencyPenalty)
{
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.frequencyPenalties = {frequencyPenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPFrequencyPenaltiesBatch)
{
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.frequencyPenalties = frequencyPenalties;
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPRepetitionPresencePenalty)
{
    float repetitionPenalty = 1e9f;
    float presencePenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.presencePenalties = {presencePenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPRepetitionPresencePenaltiesBatch)
{
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalties};
    params.presencePenalties = {presencePenalties};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPRepetitionFrequencyPenalty)
{
    float repetitionPenalty = 1e9f;
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.frequencyPenalties = {frequencyPenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPRepetitionFrequencyPenaltiesBatch)
{
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalties};
    params.frequencyPenalties = {frequencyPenalties};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPPresenceFrequencyPenalty)
{
    float presencePenalty = 1e9f;
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.presencePenalties = {presencePenalty};
    params.frequencyPenalties = {frequencyPenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPPresenceFrequencyPenaltiesBatch)
{
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.presencePenalties = {presencePenalties};
    params.frequencyPenalties = {frequencyPenalties};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPFullPenalty)
{
    float repetitionPenalty = 1e9f;
    float presencePenalty = 1e9f;
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.presencePenalties = {presencePenalty};
    params.frequencyPenalties = {frequencyPenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPFullPenaltiesBatch)
{
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalties};
    params.presencePenalties = {presencePenalties};
    params.frequencyPenalties = {frequencyPenalties};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPMinLengthBatch)
{
    std::vector<int32_t> minLengths = {3, 1, 1, 3, 0, 3};
    SamplingParams params;
    params.minLengths = minLengths;
    params.topPs = {0.3f};
    int32_t const endId = 0;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {1}, {0}, {0}, {1}, {0}, {1}, // step 1
        {2}, {0}, {0}, {2}, {0}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params, endId);
}

TYPED_TEST(DynamicDecodeLayerTest, TopPBias)
{
    SamplingParams params;
    params.topPs = {0.5f};
    params.useBias = true;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, // step 0
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, // step 1
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, // step 2
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKTemperature)
{
    uint32_t topK = 2;
    float temperature = 0.05f;
    SamplingParams params;
    params.temperatures = {temperature};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKTemperatureBatch)
{
    uint32_t topK = 2;
    std::vector<float> temperatures = {0.05f, 1e3f, 1.0f, 0.5f, 0.05f, 1.0f};
    SamplingParams params;
    params.temperatures = temperatures;
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        {4}, {4, 5, 6, 7}, {4, 5}, {4, 5}, {4}, {4, 5}, // step 0
        {0}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0, 1}, // step 1
        {2}, {2, 3, 4, 5}, {2, 3}, {2, 3}, {2}, {2, 3}, // step 2
        {0}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKRepetitionPenalty)
{
    uint32_t topK = 1;
    float repetitionPenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKRepetitionPenaltiesBatch)
{
    uint32_t topK = 1;
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = repetitionPenalties;
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKPresencePenalty)
{
    uint32_t topK = 1;
    float presencePenalty = 1e9f;
    SamplingParams params;
    params.presencePenalties = {presencePenalty};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKPresencePenaltiesBatch)
{
    uint32_t topK = 1;
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.presencePenalties = presencePenalties;
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKFrequencyPenalty)
{
    uint32_t topK = 1;
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.frequencyPenalties = {frequencyPenalty};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKFrequencyPenaltiesBatch)
{
    uint32_t topK = 1;
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.frequencyPenalties = frequencyPenalties;
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKRepetitionPresencePenalty)
{
    uint32_t topK = 1;
    float repetitionPenalty = 1e9f;
    float presencePenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.presencePenalties = {presencePenalty};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKRepetitionPresencePenaltiesBatch)
{
    uint32_t topK = 1;
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalties};
    params.presencePenalties = {presencePenalties};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKRepetitionFrequencyPenalty)
{
    uint32_t topK = 1;
    float repetitionPenalty = 1e9f;
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.frequencyPenalties = {frequencyPenalty};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKRepetitionFrequencyPenaltiesBatch)
{
    uint32_t topK = 1;
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalties};
    params.frequencyPenalties = {frequencyPenalties};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKPresenceFrequencyPenalty)
{
    uint32_t topK = 1;
    float presencePenalty = 1e9f;
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.presencePenalties = {presencePenalty};
    params.frequencyPenalties = {frequencyPenalty};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKPresenceFrequencyPenaltiesBatch)
{
    uint32_t topK = 1;
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.presencePenalties = {presencePenalties};
    params.frequencyPenalties = {frequencyPenalties};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKFullPenalty)
{
    uint32_t topK = 1;
    float repetitionPenalty = 1e9f;
    float presencePenalty = 1e9f;
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.presencePenalties = {presencePenalty};
    params.frequencyPenalties = {frequencyPenalty};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKFullPenaltiesBatch)
{
    uint32_t topK = 1;
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalties};
    params.presencePenalties = {presencePenalties};
    params.frequencyPenalties = {frequencyPenalties};
    params.topKs = {topK};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKMinLengthBatch)
{
    uint32_t topK = 1;
    std::vector<int32_t> minLengths = {3, 1, 1, 3, 0, 3};
    SamplingParams params;
    params.minLengths = minLengths;
    params.topKs = {topK};
    params.topPs = {1.0f};
    int32_t const endId = 0;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {1}, {0}, {0}, {1}, {0}, {1}, // step 1
        {2}, {0}, {0}, {2}, {0}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params, endId);
}

TYPED_TEST(DynamicDecodeLayerTest, TopKBias)
{
    uint32_t topK = 2;
    SamplingParams params;
    params.topKs = {topK};
    params.topPs = {1.0f};
    params.useBias = true;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, // step 0
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, // step 1
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, // step 2
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, BadWords)
{
    uint32_t topK = 1;
    SamplingParams params;
    params.topKs = {topK};
    params.topPs = {1.0f};
    params.badWords = {{{4, 0}, {2}}, {{0, 2}}, {{4, 0, 2}, {4, 0, 3, 0}}, {{3}}, {{4}, {5}}, {{0}, {3}}};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {6}, {4}, // step 0
        {1}, {0}, {0}, {0}, {0}, {1}, // step 1
        {3}, {3}, {3}, {2}, {2}, {2}, // step 2
        {0}, {0}, {1}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(DynamicDecodeLayerTest, StopWords)
{
    uint32_t topK = 1;
    SamplingParams params;
    params.topKs = {topK};
    params.topPs = {1.0f};
    params.stopWords = {{{4, 0}, {2}}, {{0, 2}}, {{4, 0, 2}}, {{3}}, {{4}, {5}}, {{4, 0, 2, 0}}};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {0}, {2}, {2}, {2}, {0}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}
} // namespace tensorrt_llm::tests::layers::sampling

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
#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include <gtest/gtest.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include <curand_kernel.h>
#include <random>
#include <unordered_set>

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
namespace trk = tensorrt_llm::runtime::kernels;

using namespace tensorrt_llm::runtime;

namespace
{

inline bool almostEqual(float a, float b, float atol = 1e-2, float rtol = 1e-3)
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
    return fabs(a - b) <= (atol + rtol * fabs(b));
}

std::vector<float> calculateGaussianKernel(float sigma, int size)
{
    std::vector<float> kernel(size);
    float sum = 0.f;

    for (int i = 0; i < size; ++i)
    {
        int x = i - size / 2;
        kernel[i] = std::exp(-0.5f * (x * x) / (sigma * sigma));
        sum += kernel[i];
    }

    // Normalize the kernel
    for (int i = 0; i < size; ++i)
    {
        kernel[i] /= sum;
    }

    return kernel;
}

template <typename T>
void applyGaussianFilter(T* result, float const* input, int n, float sigma)
{
    int size = static_cast<int>(std::ceil(6.f * sigma));
    size = (size % 2 == 0) ? size + 1 : size;

    std::vector<float> kernel = calculateGaussianKernel(sigma, size);
    int halfSize = size / 2;

    for (int i = 0; i < n; ++i)
    {
        result[i] = T{0};
    }

    // Convolution operation
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            int k = i - halfSize + j;
            if (k >= 0 && k < n)
            {
                result[i] += input[k] * kernel[j];
            }
        }
    }
}

template void applyGaussianFilter(float* result, float const* input, int n, float sigma);
template void applyGaussianFilter(__half* result, float const* input, int n, float sigma);

template <typename T>
void probsToLogits(T const* probs, T* logits, SizeType n)
{
    constexpr float eps = 1e-6f;
    for (SizeType ni = 0; ni < n; ++ni)
    {
        auto const prob = std::max(eps, static_cast<float>(probs[ni]));
        logits[ni] = std::log(prob / (1.f - prob));
    }
}

template <typename T>
void softmax(T const* logits, T* probs, int n)
{
    float epsilon = 1e-6f;

    // Find the maximum logit value
    float maxLogits = -std::numeric_limits<float>::max();
    for (int ii = 0; ii < n; ++ii)
    {
        maxLogits = std::max(maxLogits, static_cast<float>(logits[ii]));
    }

    // Calculate the numerator of the softmax formula
    float expSum = 0.0;
    for (int ii = 0; ii < n; ++ii)
    {
        expSum += std::exp(static_cast<float>(logits[ii]) - maxLogits);
    }

    // Calculate softmax probabilities
    for (int ii = 0; ii < n; ++ii)
    {
        float prob = std::exp(static_cast<float>(logits[ii]) - maxLogits) / (expSum + epsilon);
        probs[ii] = prob;
    }
}

template void probsToLogits(float const* probs, float* logits, SizeType n);
template void probsToLogits(__half const* probs, __half* logits, SizeType n);

enum AcceptKernelMode
{
    BY_IDS,
    BY_LOGITS,
    BY_IDS_WITH_PATH
};

struct DecodingKernelTestParam
{
    SizeType mBatchSize{128};
    SizeType mMaxBatchSize{2 * mBatchSize};
    SizeType mBeamWidth{1};
    SizeType mMaxSeqLen{16};
    SizeType mVocabSize{32};
    SizeType mMaxDraftTokens{8};
    SizeType mMaxNumHeads{0};
    SizeType mMaxDraftSeqPerStep{1};
    AcceptKernelMode mAcceptMode{AcceptKernelMode::BY_IDS};

    DecodingKernelTestParam& setBatchSize(SizeType bs)
    {
        mBatchSize = bs;
        mMaxBatchSize = 2 * mBatchSize;
        return *this;
    }

    DecodingKernelTestParam& setVocabSize(SizeType vs)
    {
        mVocabSize = vs;
        return *this;
    }

    DecodingKernelTestParam& setMaxSeqLen(SizeType msl)
    {
        mMaxSeqLen = msl;
        return *this;
    }

    DecodingKernelTestParam& setMaxDraftTokens(SizeType dt)
    {
        mMaxDraftTokens = dt;
        return *this;
    }

    DecodingKernelTestParam& setMaxNumHeads(SizeType mnh)
    {
        mMaxNumHeads = mnh;
        return *this;
    }

    DecodingKernelTestParam& setMaxDraftSeqPerStep(SizeType tps)
    {
        mMaxDraftSeqPerStep = tps;
        return *this;
    }

    DecodingKernelTestParam& setAcceptMode(AcceptKernelMode const& mode)
    {
        mAcceptMode = mode;
        return *this;
    }
};

template <typename T>
class DecodingKernelsTest : public testing::Test
{
public:
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

    void SetUp() override
    {
        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);
    }

    void TearDown() override {}

    void createBuffers()
    {
        auto const dataType = TRTDataType<T>::value;
        auto const ptrType = TRTDataType<T*>::value;

        mDraftTokens
            = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize, mMaxDraftSeqlen}), nvinfer1::DataType::kINT32);
        mTargetTokens
            = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize, mMaxSeqLen}), nvinfer1::DataType::kINT32);
        mNumsDraftTokens = BufferManager::pinned(
            ITensor::makeShape({mMaxBatchSize, mMaxDraftSeqPerStep}), nvinfer1::DataType::kINT32);
        mSequenceLengths = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mContextLengths = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mDraftContextLengths = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mFinishedSteps = BufferManager::pinned(ITensor::makeShape({mMaxDraftTokens + 1, mMaxBatchSize}),
            TRTDataType<tk::FinishedState::UnderlyingType>::value);
        mFinishedFinal = BufferManager::pinned(
            ITensor::makeShape({mMaxBatchSize}), TRTDataType<tk::FinishedState::UnderlyingType>::value);
        mFinishedSum = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mPaths = BufferManager::pinned(
            ITensor::makeShape({mMaxBatchSize, mMaxDraftSeqPerStep, mMaxDraftTokens}), nvinfer1::DataType::kINT32);
        mEndIds = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mBatchSlots = BufferManager::pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

        mCurandStates = mBufferManager->gpu(
            ITensor::makeShape({mMaxBatchSize, sizeof(curandState_t)}), nvinfer1::DataType::kINT8);

        mAcceptedLen.resize(mMaxBatchSize);
        mOutputLen.resize(mMaxBatchSize);
        mAcceptedFinished.resize(mMaxBatchSize, tk::FinishedState::empty());

        // Buffers only for Logits comparison
        if (mAcceptMode == AcceptKernelMode::BY_LOGITS)
        {
            mDraftLogits = BufferManager::pinned(
                ITensor::makeShape({mMaxBatchSize, mMaxTotalDraftTokens, mVocabSize}), dataType);
            mTargetLogits = BufferManager::pinned(
                ITensor::makeShape({mMaxBatchSize, mMaxTotalDraftTokens, mVocabSize}), dataType);
            mTargetLogitsPtrs = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), ptrType);
            mRefTargetLogits = BufferManager::pinned(
                ITensor::makeShape({mMaxBatchSize, mMaxTotalDraftTokens, mVocabSize}), dataType);

            mDraftProbs = BufferManager::pinned(
                ITensor::makeShape({mMaxBatchSize, mMaxTotalDraftTokens, mVocabSize}), dataType);
            mTargetProbs = BufferManager::pinned(
                ITensor::makeShape({mMaxBatchSize, mMaxTotalDraftTokens, mVocabSize}), dataType);
        }

        if (mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH)
        {
            mMedusaLogitsPtrs = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize, mMaxNumHeads}), ptrType);
        }
    }

    void initData(SizeType seed)
    {
        std::mt19937 generator(seed);
        std::uniform_int_distribution<SizeType> contextLenDistr(0, std::max(mMaxSeqLen - mMaxTotalDraftTokens, 0));
        std::uniform_int_distribution<SizeType> draftContextLenDistr(
            0, std::max(mMaxDraftSeqlen - mMaxTotalDraftTokens, 0));
        std::uniform_int_distribution<SizeType> numTotalDraftTokensDistr(1, mMaxTotalDraftTokens);
        std::uniform_int_distribution<SizeType> numDraftTokensDistr(0, mMaxDraftTokens);
        std::uniform_int_distribution<SizeType> vocabDistr(1, mVocabSize - 1);
        std::uniform_real_distribution<float> acceptTokenDistr(0.f, 1.f);

        trk::invokeFill(*mPaths, int32_t{-1}, *mStream);
        trk::invokeFill(*mFinishedFinal, tk::FinishedState::UnderlyingType{0}, *mStream);

        auto sequenceLengthsPtr = BufferRange<SizeType>(*mSequenceLengths);
        auto contextLengthsPtr = BufferRange<SizeType>(*mContextLengths);
        auto draftContextLengthsPtr = BufferRange<SizeType>(*mDraftContextLengths);
        auto numsDraftTokensPtr = BufferRange<SizeType>(*mNumsDraftTokens);
        auto draftTokensPtr = BufferRange<SizeType>(*mDraftTokens);
        auto targetTokensPtr = BufferRange<SizeType>(*mTargetTokens);
        auto finishedStepsPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps));
        auto pathsPtr = BufferRange<SizeType>(*mPaths);
        auto endIdsPtr = BufferRange<SizeType>(*mEndIds);

        auto batchSlotsPtr = BufferRange<SizeType>(*mBatchSlots);

        tk::invokeCurandInitialize(reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStates)), nullptr,
            mMaxBatchSize, seed, this->mStream->get());

        auto generateAvoidingValues
            = [&vocabDistr, &generator](std::uniform_int_distribution<SizeType>& distr,
                  std::unordered_set<SizeType> const& tokensToAvoid, SizeType maxTries = -1, SizeType defaultValue = -1)
        {
            // Avoid generating endId.
            auto token = distr(generator);
            SizeType tries = 0;
            while (tokensToAvoid.count(token) != 0 && ((maxTries >= 0 && tries < maxTries) || maxTries < 0))
            {
                token = distr(generator);
                tries++;
            }
            if (tries == maxTries)
            {
                token = defaultValue;
            }
            return token;
        };

        // Init batch slots
        for (SizeType bi = 0; bi < mBatchSize; ++bi)
        {
            batchSlotsPtr[bi] = 2 * bi;
        }

        // Init end ids
        for (SizeType bi = 0; bi < mMaxBatchSize; ++bi)
        {
            endIdsPtr[bi] = generateAvoidingValues(vocabDistr, {mPadId});
            TLLM_LOG_DEBUG("bi %d endIdsPtr[bi] %d", bi, endIdsPtr[bi]);

            // Randomly init context len for target and draft
            contextLengthsPtr[bi] = contextLenDistr(generator);
            draftContextLengthsPtr[bi] = draftContextLenDistr(generator);
        }

        std::fill(draftTokensPtr.begin(), draftTokensPtr.begin() + mMaxBatchSize * mMaxDraftSeqlen, mPadId);
        std::fill(targetTokensPtr.begin(), targetTokensPtr.begin() + mMaxBatchSize * mMaxSeqLen, mPadId);
        std::fill(pathsPtr.begin(), pathsPtr.begin() + mMaxBatchSize * mMaxDraftSeqPerStep * mMaxDraftTokens, -1);

        // Generate paths
        for (SizeType bi = 0; bi < mMaxBatchSize; ++bi)
        {
            auto const numTotalDraftTokens = std::min(mMaxDraftTokens, numTotalDraftTokensDistr(generator));
            std::uniform_int_distribution<SizeType> pathIdDistr(0, numTotalDraftTokens);
            for (SizeType pi = 0; pi < mMaxDraftSeqPerStep; ++pi)
            {
                std::unordered_set<SizeType> pathIds;
                auto const numDraftTokensAtStep = numDraftTokensDistr(generator);
                numsDraftTokensPtr[bi * mMaxDraftSeqPerStep + pi] = numDraftTokensAtStep;

                for (SizeType ti = 0; ti < numDraftTokensAtStep; ++ti)
                {
                    auto const pathIdx = tc::flat_index3(bi, pi, ti, mMaxDraftSeqPerStep, mMaxDraftTokens);
                    // Single linear path for BY_IDS and BY_LOGITS modes
                    auto const pathId = mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH
                        ? generateAvoidingValues(pathIdDistr, pathIds, mMaxDraftTokens * 5, -1)
                        : ti;
                    pathsPtr[pathIdx] = pathId;
                    pathIds.insert(pathId);
                }
                TLLM_LOG_DEBUG("bi %d pi %d numsDraftTokensPtr[bi] %d", bi, pi, numDraftTokensAtStep);
            }
        }

        for (SizeType ti = 0; ti < mMaxDraftSeqPerStep; ++ti)
        {
            std::vector<SizeType> targetPredictedLen(mMaxBatchSize);
            std::vector<SizeType> targetAcceptedLen(mMaxBatchSize);

            // Init number of draft tokens
            for (SizeType bi = 0; bi < mMaxBatchSize; ++bi)
            {
                // It can be shorter than num of draft tokens due to the EOS generation
                std::uniform_int_distribution<SizeType> realDraftTokensDistr(
                    0, numsDraftTokensPtr[bi * mMaxDraftSeqPerStep + ti]);
                targetPredictedLen[bi] = realDraftTokensDistr(generator);
                // Accept ~ half of the tokens on avergae
                std::poisson_distribution<SizeType> targetAcceptedDistr(targetPredictedLen[bi] / 2);
                targetAcceptedLen[bi] = std::min(targetAcceptedDistr(generator), targetPredictedLen[bi]);

                TLLM_LOG_DEBUG(
                    "bi %d ti %d targetPredictedLen[bi] %d targetAcceptedLen[bi] %d draftContextLengthsPtr[bi] %d", bi,
                    ti, targetPredictedLen[bi], targetAcceptedLen[bi], draftContextLengthsPtr[bi]);
            }

            // Fill draft tokens
            for (SizeType bi = 0; bi < mMaxBatchSize; ++bi)
            {
                for (SizeType si = 0; si < numsDraftTokensPtr[bi * mMaxDraftSeqPerStep + ti]; ++si)
                {
                    auto const pathIdx = tc::flat_index3(bi, ti, si, mMaxDraftSeqPerStep, mMaxDraftTokens);
                    auto const draftTokenIdx = bi * mMaxDraftSeqlen + draftContextLengthsPtr[bi] + pathsPtr[pathIdx];
                    // Avoid generating endId. We'll insert in manually later if needed.
                    draftTokensPtr[draftTokenIdx] = generateAvoidingValues(vocabDistr, {mPadId, endIdsPtr[bi]});
                    TLLM_LOG_DEBUG("bi %d ti %d si %d pathId %d draftToken %d", bi, ti, si, pathsPtr[pathIdx],
                        draftTokensPtr[draftTokenIdx]);
                }
            }

            for (SizeType bi = 0; bi < mMaxBatchSize; ++bi)
            {
                sequenceLengthsPtr[bi] = contextLengthsPtr[bi] + targetPredictedLen[bi];

                // Initialize finished states
                for (int di = 0; di < numsDraftTokensPtr[bi * mMaxDraftSeqPerStep + ti]; ++di)
                {
                    finishedStepsPtr[di * mMaxBatchSize + bi]
                        = (di < targetPredictedLen[bi]) ? tk::FinishedState::empty() : tk::FinishedState::finished();
                }

                // Init helper vectors
                mAcceptedLen[bi] = contextLengthsPtr[bi] + std::max(targetAcceptedLen[bi], 0);
                mOutputLen[bi] = std::min(sequenceLengthsPtr[bi], std::min(mAcceptedLen[bi] + 1, mMaxSeqLen));
                mAcceptedFinished[bi] = finishedStepsPtr[std::max(targetAcceptedLen[bi], 0) * mMaxBatchSize + bi];

                TLLM_LOG_DEBUG(
                    "bi %d ti %d contextLengthsPtr[bi] %d sequenceLengthsPtr[bi] %d mAcceptedLen[bi] %d mOutputLen[bi] "
                    "%d",
                    bi, ti, contextLengthsPtr[bi], sequenceLengthsPtr[bi], mAcceptedLen[bi], mOutputLen[bi]);
            }

            // Fill token arrays
            for (SizeType bi = 0; bi < mMaxBatchSize; ++bi)
            {
                // Draft:  [padId, padId, for draftContextLengthsPtr[bi] ... padId,
                //         d0, d1, d2, ... for numsDraftTokensPtr[bi] ... , dK,
                //         padId, padId, .. to mMaxDraftSeqlen]
                // Target: [padId, padId, ... for contextLengthsPtr[bi] ... padId,
                //         d0, d1, d2, ... for targetAcceptedLen[bi],
                //         ti (!= di), ti+1 (!= di+1), ... for (targetPredictedLen[bi] - targetAcceptedLen[bi]),
                //         EOS, EOS, EOS, ... for (numsDraftTokensPtr[bi] - targetPredictedLen[bi])
                //         padId, padId, .. to mMaxSeqLen]
                for (SizeType si = 0; si < numsDraftTokensPtr[bi * mMaxDraftSeqPerStep + ti]; ++si)
                {
                    auto const pathIdx = tc::flat_index3(bi, ti, si, mMaxDraftSeqPerStep, mMaxDraftTokens);
                    auto const pathId = pathsPtr[pathIdx];
                    if (pathId == -1)
                    {
                        continue;
                    }
                    auto const draftTokenIdx = bi * mMaxDraftSeqlen + draftContextLengthsPtr[bi] + pathId;
                    auto const targetTokenIdx = bi * mMaxSeqLen + contextLengthsPtr[bi] + pathId;
                    auto targetToken = mPadId;

                    if (0 <= si && si < targetAcceptedLen[bi])
                    {
                        // Use draft token up to the accepted len
                        targetToken = draftTokensPtr[draftTokenIdx];
                    }
                    else if (targetAcceptedLen[bi] <= si && si < targetPredictedLen[bi])
                    {
                        // Do not use draft token token up to the generated len
                        targetToken = generateAvoidingValues(
                            vocabDistr, {mPadId, endIdsPtr[bi], draftTokensPtr[draftTokenIdx]});
                    }
                    else if (targetPredictedLen[bi] <= si && si < numsDraftTokensPtr[bi])
                    {
                        // Fill with EOS from generated len to the draft len
                        targetToken = endIdsPtr[bi];
                    }
                    targetTokensPtr[targetTokenIdx] = targetToken;
                    TLLM_LOG_DEBUG("bi %d ti %d si %d pathId %d targetToken %d", bi, ti, si, pathId, targetToken);
                }
            }
        }

        if (mAcceptMode == AcceptKernelMode::BY_LOGITS)
        {
            initDataAndReferenceAcceptByLogits();
        }

        if (mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH)
        {
            initDataAndReferenceAcceptByIdsWithPaths();
        }
    }

    void initDataAndReferenceAcceptByIdsWithPaths()
    {
        auto const dataType = TRTDataType<T>::value;
        auto const ptrType = TRTDataType<T*>::value;

        auto pathsPtr = BufferRange<SizeType>(*mPaths);
        auto endIdsPtr = BufferRange<SizeType>(*mEndIds);
        auto contextLengthsPtr = BufferRange<SizeType>(*mContextLengths);
        auto draftContextLengthsPtr = BufferRange<SizeType>(*mDraftContextLengths);
        auto draftTokensPtr = BufferRange<SizeType>(*mDraftTokens);
        auto targetTokensPtr = BufferRange<SizeType>(*mTargetTokens);

        trk::invokeFill(*mMedusaLogitsPtrs, int64_t{0}, *mStream);

        mAcceptedLen.resize(mMaxBatchSize);
        mAcceptedPathIdx.resize(mMaxBatchSize);
        mRefAcceptedTokens.resize(mMaxBatchSize);
        mFinishedByIdsPaths.resize(mMaxBatchSize);
        for (SizeType bi = 0; bi < mMaxBatchSize; ++bi)
        {
            SizeType maxAcceptedLen = -1;
            SizeType maxAcceptedPath = -1;
            bool maxFinished = false;
            std::vector<SizeType> maxAcceptedTokens;
            for (SizeType ti = 0; ti < mMaxDraftSeqPerStep; ++ti)
            {
                std::vector<SizeType> acceptedTokens;
                SizeType curAcceptedLen = mMaxDraftTokens;
                SizeType curAcceptedPath = -1;
                bool curFinished = false;
                for (SizeType di = 0; di < mMaxDraftTokens; ++di)
                {
                    auto const pathIdx = tc::flat_index3(bi, ti, di, mMaxDraftSeqPerStep, mMaxDraftTokens);
                    auto const pathId = pathsPtr[pathIdx];
                    if (pathId == -1)
                    {
                        curAcceptedLen = di;
                        curAcceptedPath = ti;
                        curFinished = false;
                        break;
                    }
                    auto const draftTokenIdx = bi * mMaxDraftSeqlen + draftContextLengthsPtr[bi] + pathId;
                    auto const targetTokenIdx = bi * mMaxSeqLen + contextLengthsPtr[bi] + pathId;
                    auto const draftToken = draftTokensPtr[draftTokenIdx];
                    auto const targetToken = targetTokensPtr[targetTokenIdx];
                    bool const hasEnd = targetToken == endIdsPtr[bi];
                    if (!hasEnd)
                    {
                        acceptedTokens.push_back(targetToken);
                    }
                    if (draftToken != targetToken || hasEnd)
                    {
                        auto const curLen = hasEnd ? di : di + 1;
                        curAcceptedLen = curLen;
                        curAcceptedPath = ti;
                        curFinished = hasEnd;
                        break;
                    }
                }
                if (curAcceptedLen > maxAcceptedLen)
                {
                    maxAcceptedLen = curAcceptedLen;
                    maxAcceptedPath = curAcceptedPath;
                    maxAcceptedTokens = acceptedTokens;
                    maxFinished = curFinished;
                }
            }
            mAcceptedLen[bi] = maxAcceptedLen;
            mAcceptedPathIdx[bi] = maxAcceptedPath;
            mRefAcceptedTokens[bi] = maxAcceptedTokens;
            mFinishedByIdsPaths[bi] = maxFinished;
            TLLM_LOG_DEBUG("bi %d maxAcceptedLen %d maxAcceptedPath %d", bi, maxAcceptedLen, maxAcceptedPath);
            std::ostringstream ss;
            for (auto& tk : maxAcceptedTokens)
            {
                ss << tk << " ";
            }
            TLLM_LOG_DEBUG(ss.str().c_str());
        }
        mDraftContextLengthsCopy = mBufferManager->copyFrom(*mDraftContextLengths, MemoryType::kCPU);
    }

    void initDataAndReferenceAcceptByLogits()
    {
        auto contextLengthsPtr = BufferRange<SizeType>(*mContextLengths);
        auto numsDraftTokensPtr = BufferRange<SizeType>(*mNumsDraftTokens);
        auto draftTokensPtr = BufferRange<SizeType>(*mDraftTokens);
        auto targetTokensPtr = BufferRange<SizeType>(*mTargetTokens);

        auto draftProbsPtr = BufferRange<T>(*mDraftProbs);
        auto targetProbsPtr = BufferRange<T>(*mTargetProbs);

        auto draftLogitsPtr = BufferRange<T>(*mDraftLogits);
        auto targetLogitsPtr = BufferRange<T>(*mTargetLogits);
        auto targetLogitsPtrsPtr = BufferRange<T*>(*mTargetLogitsPtrs);
        auto refTargetLogitsPtr = BufferRange<T>(*mRefTargetLogits);
        auto batchSlotsPtr = BufferRange<SizeType>(*mBatchSlots);

        for (SizeType bi = 0; bi < mMaxBatchSize; ++bi)
        {
            // Init draft and target logits and probabilities
            for (SizeType si = 0; si < numsDraftTokensPtr[bi]; ++si)
            {
                std::vector<float> peakDraftProb(mVocabSize, 0.f);
                std::vector<float> peakTargetProb(mVocabSize, 0.f);

                auto const targetToken = targetTokensPtr[bi * mMaxSeqLen + contextLengthsPtr[bi] + si] % mVocabSize;
                auto const draftToken = draftTokensPtr[bi * mMaxDraftTokens + si] % mVocabSize;

                peakDraftProb[draftToken] = 1.f;
                peakTargetProb[targetToken] = 1.f;

                auto const logitsOffset = bi * mMaxDraftTokens * mVocabSize + si * mVocabSize;
                // Emulate some distribution around target token
                applyGaussianFilter(
                    draftProbsPtr.begin() + logitsOffset, peakDraftProb.data(), peakDraftProb.size(), 1.0f);
                applyGaussianFilter(
                    targetProbsPtr.begin() + logitsOffset, peakTargetProb.data(), peakTargetProb.size(), 1.0f);

                // Probabilities to logits
                probsToLogits(draftProbsPtr.begin() + logitsOffset, draftLogitsPtr.begin() + logitsOffset, mVocabSize);
                probsToLogits(
                    targetProbsPtr.begin() + logitsOffset, targetLogitsPtr.begin() + logitsOffset, mVocabSize);

                // Do softmax conversion back to emulate kernels accuracy
                softmax(draftLogitsPtr.begin() + logitsOffset, draftProbsPtr.begin() + logitsOffset, mVocabSize);
                softmax(targetLogitsPtr.begin() + logitsOffset, targetProbsPtr.begin() + logitsOffset, mVocabSize);
            }
        }

        for (SizeType bi = 0; bi < mMaxBatchSize; ++bi)
        {
            for (SizeType si = 0; si < mMaxDraftTokens; ++si)
            {
                auto const logitsOffset = bi * mMaxDraftTokens * mVocabSize + si * mVocabSize;
                auto const outputLen = mOutputLen[bi] - contextLengthsPtr[bi];
                auto const acceptedLen = mAcceptedLen[bi] - contextLengthsPtr[bi];
                if (si < acceptedLen)
                {
                    auto logitsStart = targetLogitsPtr.begin() + logitsOffset;
                    std::copy(logitsStart, logitsStart + mVocabSize, refTargetLogitsPtr.begin() + logitsOffset);
                }
                else if (si == acceptedLen)
                {
                    // When token is not accepted, correct probabilities and compute updated logits
                    float sumProb = 1e-6f;
                    for (SizeType vi = 0; vi < mVocabSize; ++vi)
                    {
                        auto const correctedProb = std::max(
                            static_cast<float>(targetProbsPtr[logitsOffset + vi] - draftProbsPtr[logitsOffset + vi]),
                            0.f);
                        sumProb += correctedProb;
                    }
                    for (SizeType vi = 0; vi < mVocabSize; ++vi)
                    {
                        auto prob = std::max(static_cast<float>(
                                                 targetProbsPtr[logitsOffset + vi] - draftProbsPtr[logitsOffset + vi]),
                                        0.f)
                            / sumProb;
                        if (prob < 1e-8)
                        {
                            prob = 0.f;
                        }
                        refTargetLogitsPtr[logitsOffset + vi] = std::log(prob / (1.f - prob));
                    }
                }
            }
        }
        for (SizeType bi = 0; bi < mBatchSize; ++bi)
        {
            targetLogitsPtrsPtr[bi] = targetLogitsPtr.begin() + batchSlotsPtr[bi] * mMaxDraftTokens * mVocabSize;
        }
    }

    void callAcceptByIds()
    {
        tk::invokeAcceptDraftTokensByIds(bufferCast<SizeType>(*mDraftTokens), bufferCast<SizeType>(*mTargetTokens),
            bufferCast<SizeType>(*mContextLengths), bufferCast<SizeType>(*mNumsDraftTokens),
            bufferCast<SizeType>(*mSequenceLengths),
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps)),
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedFinal)),
            bufferCast<SizeType>(*mFinishedSum), bufferCast<SizeType>(*mBatchSlots), mBatchSize, mMaxBatchSize,
            mBeamWidth, mMaxSeqLen, mMaxDraftTokens, mStream->get());
    }

    void callAcceptByLogits()
    {
        tk::acceptDraftTokensByLogits(bufferCast<T>(*mDraftLogits),
            reinterpret_cast<T**>(bufferCast<int64_t>(*mTargetLogitsPtrs)), bufferCast<T>(*mDraftProbs),
            bufferCast<T>(*mTargetProbs), bufferCast<SizeType>(*mNumsDraftTokens),
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps)),
            reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStates)), bufferCast<SizeType>(*mBatchSlots),
            mBatchSize, mMaxBatchSize, mBeamWidth, mVocabSize, mVocabSize, mMaxDraftTokens, false, 0.9f,
            mStream->get());
    }

    void callAcceptByIdsWithPaths()
    {
        tk::acceptDraftTokensByIdsWithPaths(bufferCast<SizeType>(*mDraftTokens), bufferCast<SizeType>(*mTargetTokens),
            bufferCast<SizeType>(*mDraftContextLengths),
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedFinal)),
            bufferCast<SizeType>(*mBatchSlots), bufferCast<SizeType>(*mPaths), bufferCast<SizeType>(*mEndIds),
            static_cast<T const*>(nullptr), reinterpret_cast<T const**>(bufferCast<int64_t>(*mMedusaLogitsPtrs)),
            mBatchSize, mVocabSize, mMaxBatchSize, mMaxDraftSeqlen, mMaxTotalDraftTokens, mMaxNumHeads,
            mMaxDraftSeqPerStep, mStream->get());
    }

    void callTestedKernel()
    {
        switch (mAcceptMode)
        {
        case AcceptKernelMode::BY_IDS: callAcceptByIds(); break;
        case AcceptKernelMode::BY_LOGITS: callAcceptByLogits(); break;
        case AcceptKernelMode::BY_IDS_WITH_PATH: callAcceptByIdsWithPaths(); break;
        default: TLLM_CHECK(false); // Should never be here
        }
    }

    void verifyAcceptByIdsResults(SizeType seed)
    {
        auto finishedFinalPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedFinal));
        auto sequenceLengthsPtr = BufferRange<SizeType>(*mSequenceLengths);
        auto finishedSumPtr = BufferRange<SizeType>(*mFinishedSum);
        auto batchSlotsPtr = BufferRange<SizeType>(*mBatchSlots);
        // Verify seqLen for accepted tokens
        for (SizeType bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            EXPECT_EQ(mOutputLen[batchSlot], sequenceLengthsPtr[batchSlot]) << " bi " << bi << " seed " << seed;
            EXPECT_EQ(mAcceptedFinished[batchSlot].isFinished(), finishedFinalPtr[batchSlot].isFinished())
                << " bi " << bi << " seed " << seed;
            EXPECT_EQ(mAcceptedFinished[batchSlot].isSkipDecoding(), finishedFinalPtr[batchSlot].isSkipDecoding())
                << " bi " << bi << " seed " << seed;
            EXPECT_EQ(static_cast<SizeType>(mAcceptedFinished[batchSlot].isFinished()), finishedSumPtr[batchSlot]);
        }
    }

    void verifyAcceptByLogitsResults(SizeType seed)
    {
        auto finishedStepsPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps));
        auto contextLengthsPtr = BufferRange<SizeType>(*mContextLengths);
        auto outLogitsPtr = BufferRange<T>(*mTargetLogits);
        auto refLogitsPtr = BufferRange<T>(*mRefTargetLogits);
        auto numsDraftTokensPtr = BufferRange<SizeType>(*mNumsDraftTokens);
        auto batchSlotsPtr = BufferRange<SizeType>(*mBatchSlots);

        for (SizeType bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (SizeType si = 0; si < numsDraftTokensPtr[batchSlot]; ++si)
            {
                auto const outFinishedState = finishedStepsPtr[si * mMaxBatchSize + batchSlot];
                auto const logitsOffset = batchSlot * mMaxDraftTokens * mVocabSize + si * mVocabSize;
                if (si <= mAcceptedLen[batchSlot] - contextLengthsPtr[batchSlot])
                {
                    EXPECT_FALSE(outFinishedState.isSkipDecoding())
                        << " bi: " << bi << " si: " << si << " seed: " << seed;
                    for (SizeType vi = 0; vi < mVocabSize; ++vi)
                    {
                        auto const outLogit = static_cast<float>(outLogitsPtr[logitsOffset + vi]);
                        auto const refLogit = static_cast<float>(refLogitsPtr[logitsOffset + vi]);
                        EXPECT_FALSE((refLogit > -10) ^ (outLogit > -10))
                            << " bi: " << bi << " si: " << si << " vi: " << vi << " seed: " << seed;
                        if (refLogit > -10 && outLogit > -10)
                        {
                            if (!almostEqual(outLogit, refLogit, 1e-1, 1e-2))
                            {
                                std::cout << refLogit << " " << outLogit << std::endl;
                            }
                            ASSERT_TRUE(almostEqual(outLogit, refLogit, 1e-1, 1e-2))
                                << " bi: " << bi << " si: " << si << " vi: " << vi << " seed: " << seed;
                        }
                    }
                }
                else
                {
                    EXPECT_TRUE(outFinishedState.isSkipDecoding())
                        << " bi: " << bi << " si: " << si << " seed: " << seed;
                }
            }
        }
    }

    void verifyAcceptByIdsWithPathsResults(SizeType seed)
    {
        auto medusaLogitsPtrsPtr = BufferRange<T*>(*mMedusaLogitsPtrs);
        auto batchSlotsPtr = BufferRange<SizeType>(*mBatchSlots);
        auto draftContextLengths = BufferRange<SizeType>(*mDraftContextLengths);
        auto draftContextLengthsInit = BufferRange<SizeType>(*mDraftContextLengthsCopy);
        auto draftTokensPtr = BufferRange<SizeType>(*mDraftTokens);
        auto finishedFinalPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedFinal));

        for (SizeType bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            auto const bestPathIdx = mAcceptedPathIdx[batchSlot];
            auto const acceptedLen = mAcceptedLen[batchSlot];
            auto acceptedTokens = mRefAcceptedTokens[batchSlot];

            for (int32_t hi = 0; hi < mMaxNumHeads; ++hi)
            {
                auto refOffset
                    = tc::flat_index4(hi, bi, acceptedLen, 0, mMaxBatchSize, mMaxDraftSeqPerStep, mVocabSize);
                auto outOffset
                    = static_cast<SizeType>(medusaLogitsPtrsPtr[bi * mMaxNumHeads + hi] - static_cast<T*>(nullptr));
                EXPECT_EQ(outOffset, refOffset) << " bi: " << bi << " hi: " << hi << " seed: " << seed;
            }
            EXPECT_EQ(draftContextLengths[batchSlot], draftContextLengthsInit[batchSlot] + acceptedLen)
                << " bi: " << bi << " seed: " << seed << " out: " << draftContextLengths[batchSlot]
                << " ref: " << draftContextLengthsInit[batchSlot] + acceptedLen;

            for (SizeType ti = 0; ti < acceptedLen; ++ti)
            {
                ASSERT_EQ(mRefAcceptedTokens[batchSlot].size(), acceptedLen)
                    << " bi: " << bi << " ti: " << ti << " seed: " << seed;
                EXPECT_EQ(draftTokensPtr[batchSlot * mMaxDraftSeqlen + draftContextLengthsInit[batchSlot] + ti],
                    mRefAcceptedTokens[batchSlot][ti])
                    << " bi: " << bi << " ti: " << ti << " seed: " << seed;
            }
            EXPECT_EQ(finishedFinalPtr[batchSlot].isFinished(), mFinishedByIdsPaths[batchSlot])
                << " bi: " << bi << " seed: " << seed;
        }
    }

    void verifyResult(SizeType seed)
    {
        switch (mAcceptMode)
        {
        case AcceptKernelMode::BY_IDS: verifyAcceptByIdsResults(seed); break;
        case AcceptKernelMode::BY_LOGITS: verifyAcceptByLogitsResults(seed); break;
        case AcceptKernelMode::BY_IDS_WITH_PATH: verifyAcceptByIdsWithPathsResults(seed); break;
        default: TLLM_CHECK(false); // Should never be here
        }
    }

    void runTest(DecodingKernelTestParam const& params)
    {
        mAcceptMode = params.mAcceptMode;

        mBatchSize = params.mBatchSize;
        mMaxBatchSize = params.mMaxBatchSize;
        mBeamWidth = params.mBeamWidth;
        mVocabSize = params.mVocabSize;
        mMaxDraftTokens = params.mMaxDraftTokens;

        mMaxNumHeads = params.mMaxNumHeads;
        if (mMaxNumHeads > 1 && mAcceptMode != AcceptKernelMode::BY_IDS_WITH_PATH)
        {
            GTEST_SKIP() << "MaxNumHeads > 1 is only supported for AcceptKernelMode::BY_IDS_WITH_PATH";
        }

        mMaxDraftSeqPerStep = params.mMaxDraftSeqPerStep;
        if (mMaxDraftSeqPerStep > 1 && mAcceptMode != AcceptKernelMode::BY_IDS_WITH_PATH)
        {
            GTEST_SKIP() << "MaxDraftSeqPerStep > 1 is only supported for AcceptKernelMode::BY_IDS_WITH_PATH";
        }

        mMaxTotalDraftTokens = mMaxDraftSeqPerStep * mMaxDraftTokens;
        mPadId = mVocabSize - 1;

        mMaxDraftSeqlen = mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH ? params.mMaxSeqLen : mMaxDraftTokens;
        mMaxSeqLen = mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH ? mMaxTotalDraftTokens : params.mMaxSeqLen;

        createBuffers();

        for (SizeType seed = 0; seed < mSeeds; ++seed)
        {
            TLLM_LOG_DEBUG("Seed %d", seed);

            initData(seed);

            mStream->synchronize();

            callTestedKernel();

            mStream->synchronize();

            verifyResult(seed);
        }
    }

protected:
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;

    TensorPtr mDraftTokens;
    TensorPtr mTargetTokens;

    TensorPtr mDraftLogits;
    TensorPtr mTargetLogits;
    TensorPtr mTargetLogitsPtrs;
    TensorPtr mRefTargetLogits;

    TensorPtr mDraftProbs;
    TensorPtr mTargetProbs;

    TensorPtr mNumsDraftTokens;
    TensorPtr mSequenceLengths;
    TensorPtr mContextLengths;
    TensorPtr mDraftContextLengthsCopy;
    TensorPtr mDraftContextLengths;
    TensorPtr mFinishedSteps;
    TensorPtr mFinishedFinal;
    TensorPtr mFinishedSum;
    TensorPtr mBatchSlots;

    TensorPtr mPaths;
    TensorPtr mEndIds;
    TensorPtr mMedusaLogitsPtrs;

    TensorPtr mCurandStates;

    std::vector<SizeType> mAcceptedLen;
    std::vector<SizeType> mOutputLen;
    std::vector<tk::FinishedState> mAcceptedFinished;
    std::vector<SizeType> mAcceptedPathIdx;
    std::vector<std::vector<SizeType>> mRefAcceptedTokens;
    std::vector<bool> mFinishedByIdsPaths;

    SizeType mBatchSize;
    SizeType mMaxBatchSize;
    SizeType mBeamWidth;
    SizeType mMaxSeqLen;
    SizeType mVocabSize;
    SizeType mMaxDraftTokens;
    SizeType mMaxTotalDraftTokens;
    SizeType mMaxDraftSeqlen;
    SizeType mMaxNumHeads;
    SizeType mMaxDraftSeqPerStep;
    AcceptKernelMode mAcceptMode;
    SizeType mPadId;
    static constexpr SizeType mSeeds = 64;
};

template class DecodingKernelsTest<float>;
template class DecodingKernelsTest<half>;

typedef testing::Types<float, half> FloatAndHalfTypes;

TYPED_TEST_SUITE(DecodingKernelsTest, FloatAndHalfTypes);

TYPED_TEST(DecodingKernelsTest, acceptDraftTokensByIdsKernelSmall)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(1)
                      .setMaxSeqLen(16)
                      .setVocabSize(32)
                      .setMaxDraftTokens(8)
                      .setMaxDraftSeqPerStep(1)
                      .setAcceptMode(AcceptKernelMode::BY_IDS));
}

TYPED_TEST(DecodingKernelsTest, acceptDraftTokensByIdsKernelLarge)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(128)
                      .setMaxSeqLen(128)
                      .setVocabSize(52000)
                      .setMaxDraftTokens(8)
                      .setMaxDraftSeqPerStep(1)
                      .setAcceptMode(AcceptKernelMode::BY_IDS));
}

TYPED_TEST(DecodingKernelsTest, acceptDraftTokensByLogitsKernelSmall)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(1)
                      .setMaxSeqLen(16)
                      .setVocabSize(32)
                      .setMaxDraftTokens(8)
                      .setMaxDraftSeqPerStep(1)
                      .setAcceptMode(AcceptKernelMode::BY_LOGITS));
}

TYPED_TEST(DecodingKernelsTest, acceptDraftTokensByLogitsKernelLarge)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(64)
                      .setMaxSeqLen(64)
                      .setVocabSize(4000)
                      .setMaxDraftTokens(8)
                      .setMaxDraftSeqPerStep(1)
                      .setAcceptMode(AcceptKernelMode::BY_LOGITS));
}

TYPED_TEST(DecodingKernelsTest, acceptDraftTokensByIdsWithPathsKernelSmall)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(1)
                      .setMaxSeqLen(128)
                      .setVocabSize(32)
                      .setMaxDraftTokens(5)
                      .setMaxDraftSeqPerStep(4)
                      .setMaxNumHeads(4)
                      .setAcceptMode(AcceptKernelMode::BY_IDS_WITH_PATH));
}

TYPED_TEST(DecodingKernelsTest, acceptDraftTokensByIdsWithPathsKernelLarge)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(128)
                      .setMaxSeqLen(1024)
                      .setVocabSize(4000)
                      .setMaxDraftTokens(8)
                      .setMaxDraftSeqPerStep(64)
                      .setMaxNumHeads(7)
                      .setAcceptMode(AcceptKernelMode::BY_IDS_WITH_PATH));
}
} // end of namespace

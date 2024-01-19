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

#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include <curand_kernel.h>
#include <random>

namespace tk = tensorrt_llm::kernels;

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
void applyGaussianFilter(T* result, const float* input, int n, float sigma)
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

template void applyGaussianFilter(float* result, const float* input, int n, float sigma);
template void applyGaussianFilter(__half* result, const float* input, int n, float sigma);

template <typename T>
void probsToLogits(const T* probs, T* logits, SizeType n)
{
    constexpr float eps = 1e-6f;
    for (SizeType ni = 0; ni < n; ++ni)
    {
        const auto prob = std::max(eps, static_cast<float>(probs[ni]));
        logits[ni] = std::log(prob / (1.f - prob));
    }
}

template <typename T>
void softmax(const T* logits, T* probs, int n)
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

template void probsToLogits(const float* probs, float* logits, SizeType n);
template void probsToLogits(const __half* probs, __half* logits, SizeType n);

template <typename T>
class DecodingKernelsTest : public testing::Test
{
public:
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

    void SetUp() override
    {
        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);

        cudaMalloc(&mCurandStates, sizeof(curandState_t) * batchSize);
    }

    void TearDown() override
    {
        cudaFree(mCurandStates);
    }

    void initData(SizeType seed)
    {
        std::mt19937 generator(seed);
        std::uniform_int_distribution<int> contextLenDistr(0, maxSeqLen - maxDraftTokens);
        std::uniform_int_distribution<int> numDraftTokensDistr(1, maxDraftTokens);
        std::uniform_int_distribution<int> vocabDistr(1, vocabSize - 1);
        std::uniform_real_distribution<float> acceptTokenDistr(0.f, 1.f);

        mDraftTokens = mBufferManager->pinned(
            ITensor::makeShape({batchSize, beamWidth, maxDraftTokens}), nvinfer1::DataType::kINT32);
        mTargetTokens
            = mBufferManager->pinned(ITensor::makeShape({batchSize, beamWidth, maxSeqLen}), nvinfer1::DataType::kINT32);

        mDraftLogits = mBufferManager->pinned(ITensor::makeShape({maxDraftTokens, batchSize * beamWidth, vocabSize}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
        mTargetLogits = mBufferManager->pinned(ITensor::makeShape({maxDraftTokens, batchSize * beamWidth, vocabSize}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
        mRefTargetLogits
            = mBufferManager->pinned(ITensor::makeShape({maxDraftTokens, batchSize * beamWidth, vocabSize}),
                std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);

        mDraftProbs = mBufferManager->pinned(ITensor::makeShape({maxDraftTokens, batchSize * beamWidth, vocabSize}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
        mTargetProbs = mBufferManager->pinned(ITensor::makeShape({maxDraftTokens, batchSize * beamWidth, vocabSize}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);

        mNumsDraftTokens = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
        mSequenceLengths
            = mBufferManager->pinned(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32);
        mContextLengths
            = mBufferManager->pinned(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32);
        mFinishedSteps = mBufferManager->pinned(ITensor::makeShape({maxDraftTokens, batchSize, beamWidth}),
            TRTDataType<tk::FinishedState::UnderlyingType>::value);
        mFinishedFinal = mBufferManager->pinned(
            ITensor::makeShape({batchSize, beamWidth}), TRTDataType<tk::FinishedState::UnderlyingType>::value);
        mFinishedSum = mBufferManager->pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

        mAcceptedLen.resize(batchSize * beamWidth);
        mOutputLen.resize(batchSize * beamWidth);
        for (SizeType bi = 0; bi < batchSize * beamWidth; ++bi)
        {
            mAcceptedFinished.emplace_back(tk::FinishedState::empty());
        }

        auto sequenceLengthsPtr = bufferCast<SizeType>(*mSequenceLengths);
        auto contextLengthsPtr = bufferCast<SizeType>(*mContextLengths);
        auto numsDraftTokensPtr = bufferCast<SizeType>(*mNumsDraftTokens);
        auto draftTokensPtr = bufferCast<SizeType>(*mDraftTokens);
        auto targetTokensPtr = bufferCast<SizeType>(*mTargetTokens);
        auto finishedStepsPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps));
        auto finishedFinalPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedFinal));
        auto finishedSumPtr = bufferCast<SizeType>(*mFinishedSum);

        auto draftProbsPtr = bufferCast<T>(*mDraftProbs);
        auto targetProbsPtr = bufferCast<T>(*mTargetProbs);

        auto draftLogitsPtr = bufferCast<T>(*mDraftLogits);
        auto targetLogitsPtr = bufferCast<T>(*mTargetLogits);
        auto refTargetLogitsPtr = bufferCast<T>(*mRefTargetLogits);

        tk::invokeCurandInitialize(mCurandStates, batchSize, seed, this->mStream->get());

        // Init number of draft tokens
        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            numsDraftTokensPtr[bi] = numDraftTokensDistr(generator);
        }

        for (SizeType bi = 0; bi < batchSize * beamWidth; ++bi)
        {
            const SizeType batchIdx = bi / beamWidth;
            // Randomly init context len
            contextLengthsPtr[bi] = contextLenDistr(generator);

            // Sequence len is at most numsDraftTokensPtr[bi] away from context len (it can be closer if e.g. endId is
            // generated)
            std::uniform_int_distribution<int> realDraftTokensDistr(0, numsDraftTokensPtr[batchIdx]);
            const auto realLen = realDraftTokensDistr(generator);
            sequenceLengthsPtr[bi] = contextLengthsPtr[bi] + realLen;

            // Initialize finished states
            for (int i = 0; i < realLen; ++i)
            {
                finishedStepsPtr[i * batchSize * beamWidth + bi] = tk::FinishedState::empty();
            }
            for (int i = realLen; i <= numsDraftTokensPtr[batchIdx]; ++i)
            {
                finishedStepsPtr[i * batchSize * beamWidth + bi] = tk::FinishedState::finished();
            }

            // Init helper vector with max value
            mAcceptedLen[bi] = sequenceLengthsPtr[bi];
            mOutputLen[bi] = sequenceLengthsPtr[bi];
            mAcceptedFinished[bi] = finishedStepsPtr[realLen * batchSize * beamWidth + bi];
        }
        // Fill token arrays
        for (SizeType bi = 0; bi < batchSize * beamWidth; ++bi)
        {
            // Draft: [d0, d1, d2, ... for numsDraftTokensPtr[bi] ... , dN]
            // Target: [vocabSize - 1, vocabSize - 1, ... for contextLengthsPtr[bi] ... vocabSize - 1,
            //         t0, t1, t2, ... for numsDraftTokensPtr[bi] ... , tN,
            //         vocabSize - 1, vocabSize - 1, .. to maxSeqLen]
            for (SizeType si = 0; si < contextLengthsPtr[bi]; ++si)
            {
                targetTokensPtr[bi * maxSeqLen + si] = vocabSize - 1;
            }

            for (SizeType si = contextLengthsPtr[bi]; si < sequenceLengthsPtr[bi]; ++si)
            {
                const auto draftToken = vocabDistr(generator);
                const auto draftTokenIdx = si - contextLengthsPtr[bi];
                const auto targetToken
                    = acceptTokenDistr(generator) < 1.f / (draftTokenIdx + 1e-6) ? draftToken : vocabDistr(generator);
                draftTokensPtr[bi * maxDraftTokens + draftTokenIdx] = draftToken;
                targetTokensPtr[bi * maxSeqLen + si] = targetToken;
                if (draftToken != targetToken)
                {
                    mAcceptedLen[bi] = std::min(mAcceptedLen[bi], std::min(si, maxSeqLen));
                    mOutputLen[bi] = std::min(mOutputLen[bi], std::min(si + 1, maxSeqLen));
                    mAcceptedFinished[bi] = finishedStepsPtr[draftTokenIdx * batchSize * beamWidth + bi];
                }
            }

            for (SizeType si = sequenceLengthsPtr[bi]; si < maxSeqLen; ++si)
            {
                targetTokensPtr[bi * maxSeqLen + si] = vocabSize - 1;
            }

            for (SizeType si = sequenceLengthsPtr[bi] - contextLengthsPtr[bi]; si < maxDraftTokens; ++si)
            {
                draftTokensPtr[bi * maxDraftTokens + si] = 0;
            }

            // Init draft and target logits and probabilities
            for (SizeType si = 0; si < numsDraftTokensPtr[bi]; ++si)
            {
                std::vector<float> peakDraftProb(vocabSize, 0.f);
                std::vector<float> peakTargetProb(vocabSize, 0.f);

                const auto targetToken = targetTokensPtr[bi * maxSeqLen + contextLengthsPtr[bi] + si] % vocabSize;
                const auto draftToken = draftTokensPtr[bi * maxDraftTokens + si] % vocabSize;

                peakDraftProb[draftToken] = 1.f;
                peakTargetProb[targetToken] = 1.f;

                const int logitsOffset = si * batchSize * beamWidth * vocabSize + bi * vocabSize;
                // Emulate some distribution around target token
                applyGaussianFilter(draftProbsPtr + logitsOffset, peakDraftProb.data(), peakDraftProb.size(), 1.0f);
                applyGaussianFilter(targetProbsPtr + logitsOffset, peakTargetProb.data(), peakTargetProb.size(), 1.0f);

                // Probabilities to logits
                probsToLogits(draftProbsPtr + logitsOffset, draftLogitsPtr + logitsOffset, vocabSize);
                probsToLogits(targetProbsPtr + logitsOffset, targetLogitsPtr + logitsOffset, vocabSize);

                // Do softmax conversion back to emulate kernels accuracy
                softmax(draftLogitsPtr + logitsOffset, draftProbsPtr + logitsOffset, vocabSize);
                softmax(targetLogitsPtr + logitsOffset, targetProbsPtr + logitsOffset, vocabSize);
            }

            for (SizeType si = 0; si < maxDraftTokens; ++si)
            {
                const int logitsOffset = si * batchSize * beamWidth * vocabSize + bi * vocabSize;
                const auto outputLen = mOutputLen[bi] - contextLengthsPtr[bi];
                const auto acceptedLen = mAcceptedLen[bi] - contextLengthsPtr[bi];
                if (si < acceptedLen)
                {
                    std::memcpy(
                        refTargetLogitsPtr + logitsOffset, targetLogitsPtr + logitsOffset, vocabSize * sizeof(T));
                }
                else if (si == acceptedLen)
                {
                    // When token is not accepted, correct probabilities and compute updated logits
                    float sumProb = 1e-6f;
                    for (SizeType vi = 0; vi < vocabSize; ++vi)
                    {
                        const auto correctedProb = std::max(
                            static_cast<float>(targetProbsPtr[logitsOffset + vi] - draftProbsPtr[logitsOffset + vi]),
                            0.f);
                        sumProb += correctedProb;
                    }
                    for (SizeType vi = 0; vi < vocabSize; ++vi)
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
    }

    void verifyAcceptByIdsResults(SizeType seed)
    {
        mStream->synchronize();

        auto finishedFinalPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedFinal));
        auto sequenceLengthsPtr = bufferCast<SizeType>(*mSequenceLengths);
        auto finishedSumPtr = bufferCast<SizeType>(*mFinishedSum);

        // Verify seqLen for accepted tokens
        int finishedSumRef = 0;
        for (SizeType bi = 0; bi < batchSize * beamWidth; ++bi)
        {
            EXPECT_EQ(mOutputLen[bi], sequenceLengthsPtr[bi]) << " bi " << bi << " seed " << seed;
            EXPECT_EQ(mAcceptedFinished[bi].isFinished(), finishedFinalPtr[bi].isFinished())
                << " bi " << bi << " seed " << seed;
            EXPECT_EQ(mAcceptedFinished[bi].isSkipDecoding(), finishedFinalPtr[bi].isSkipDecoding())
                << " bi " << bi << " seed " << seed;
            finishedSumRef += static_cast<SizeType>(mAcceptedFinished[bi].isFinished());
        }
        EXPECT_EQ(finishedSumRef, finishedSumPtr[0]);
    }

    void verifyAcceptByLogitsResults(SizeType seed)
    {
        mStream->synchronize();

        auto finishedStepsPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps));
        auto contextLengthsPtr = bufferCast<SizeType>(*mContextLengths);
        auto outLogitsPtr = bufferCast<T>(*mTargetLogits);
        auto refLogitsPtr = bufferCast<T>(*mRefTargetLogits);
        auto numsDraftTokensPtr = bufferCast<SizeType>(*mNumsDraftTokens);

        for (SizeType bi = 0; bi < batchSize * beamWidth; ++bi)
        {
            for (SizeType si = 0; si < numsDraftTokensPtr[bi]; ++si)
            {
                const auto outFinishedState = finishedStepsPtr[si * batchSize * beamWidth + bi];
                const auto logitsOffset = si * batchSize * beamWidth * vocabSize + bi * vocabSize;
                if (si <= mAcceptedLen[bi] - contextLengthsPtr[bi])
                {
                    EXPECT_FALSE(outFinishedState.isSkipDecoding())
                        << " bi: " << bi << " si: " << si << " seed: " << seed;
                    for (SizeType vi = 0; vi < vocabSize; ++vi)
                    {
                        const auto outLogit = static_cast<float>(outLogitsPtr[logitsOffset + vi]);
                        const auto refLogit = static_cast<float>(refLogitsPtr[logitsOffset + vi]);
                        EXPECT_FALSE((refLogit > -10) ^ (outLogit > -10));
                        if (refLogit > -10 && outLogit > -10)
                        {
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

    void runAcceptByIdsTest(SizeType seed)
    {
        initData(seed);
        tk::invokeAcceptDraftTokensByIds(bufferCast<SizeType>(*mDraftTokens), bufferCast<SizeType>(*mTargetTokens),
            bufferCast<SizeType>(*mContextLengths), bufferCast<SizeType>(*mNumsDraftTokens),
            bufferCast<SizeType>(*mSequenceLengths),
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps)),
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedFinal)),
            bufferCast<SizeType>(*mFinishedSum), batchSize, beamWidth, maxSeqLen, maxDraftTokens, mStream->get());
        verifyAcceptByIdsResults(seed);
    }

    void runAcceptByLogitsTest(SizeType seed)
    {
        initData(seed);
        tk::acceptDraftTokensByLogits(bufferCast<T>(*mDraftLogits), bufferCast<T>(*mTargetLogits),
            bufferCast<T>(*mDraftProbs), bufferCast<T>(*mTargetProbs), bufferCast<SizeType>(*mNumsDraftTokens),
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps)),
            mCurandStates, batchSize, beamWidth, vocabSize, vocabSize, maxDraftTokens, true, 0, mStream->get());

        verifyAcceptByLogitsResults(seed);
    }

protected:
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;

    TensorPtr mDraftTokens;
    TensorPtr mTargetTokens;

    TensorPtr mDraftLogits;
    TensorPtr mTargetLogits;
    TensorPtr mRefTargetLogits;

    TensorPtr mDraftProbs;
    TensorPtr mTargetProbs;

    TensorPtr mNumsDraftTokens;
    TensorPtr mSequenceLengths;
    TensorPtr mContextLengths;
    TensorPtr mFinishedSteps;
    TensorPtr mFinishedFinal;
    TensorPtr mFinishedSum;

    std::vector<int> mAcceptedLen;
    std::vector<int> mOutputLen;
    std::vector<tk::FinishedState> mAcceptedFinished;

    curandState_t* mCurandStates;

    static constexpr SizeType batchSize{8};
    static constexpr SizeType beamWidth{1};
    static constexpr SizeType maxSeqLen{16};
    static constexpr SizeType vocabSize{32};
    static constexpr SizeType maxDraftTokens{8};
};

template class DecodingKernelsTest<float>;
template class DecodingKernelsTest<half>;

typedef testing::Types<float, half> FloatAndHalfTypes;

TYPED_TEST_SUITE(DecodingKernelsTest, FloatAndHalfTypes);

TYPED_TEST(DecodingKernelsTest, acceptDraftTokensByIdsKernel)
{
    constexpr SizeType seeds = 64;
    for (SizeType seed = 0; seed < seeds; ++seed)
    {
        this->runAcceptByIdsTest(seed);
    }
}

TYPED_TEST(DecodingKernelsTest, acceptDraftTokensByLogitsKernel)
{
    constexpr SizeType seeds = 64;
    for (SizeType seed = 0; seed < seeds; ++seed)
    {
        this->runAcceptByLogitsTest(seed);
    }
}

} // end of namespace

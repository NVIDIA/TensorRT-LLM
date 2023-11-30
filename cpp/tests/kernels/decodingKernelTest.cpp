/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include <random>

namespace tk = tensorrt_llm::kernels;

using namespace tensorrt_llm::runtime;

namespace
{

void runAcceptedTokensTest(SizeType seed)
{
    constexpr SizeType batchSize{8};
    constexpr SizeType beamWidth{1};
    constexpr SizeType maxSeqLen{16};
    constexpr SizeType vocabSize{32};
    constexpr SizeType maxDraftTokens{8};

    auto stream = std::make_shared<CudaStream>();
    BufferManager manager(stream);

    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> contextLenDistr(0, maxSeqLen - maxDraftTokens);
    std::uniform_int_distribution<int> numDraftTokensDistr(1, maxDraftTokens);
    std::uniform_int_distribution<int> vocabDistr(1, vocabSize);
    std::uniform_real_distribution<float> acceptTokenDistr(0.f, 1.f);

    auto draftTokens
        = manager.pinned(ITensor::makeShape({batchSize, beamWidth, maxDraftTokens}), nvinfer1::DataType::kINT32);
    auto targetTokens
        = manager.pinned(ITensor::makeShape({batchSize, beamWidth, maxSeqLen}), nvinfer1::DataType::kINT32);
    auto numsDraftTokens = manager.pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
    auto sequenceLengths = manager.pinned(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32);
    auto contextLengths = manager.pinned(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32);
    auto finishedSteps
        = manager.pinned(ITensor::makeShape({maxDraftTokens, batchSize, beamWidth}), nvinfer1::DataType::kBOOL);
    auto finishedFinal = manager.pinned(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kBOOL);
    auto finishedSum = manager.pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

    std::vector<int> acceptedLen(batchSize * beamWidth);
    std::vector<bool> acceptedFinished(batchSize * beamWidth);

    auto sequenceLengthsPtr = bufferCast<SizeType>(*sequenceLengths);
    auto contextLengthsPtr = bufferCast<SizeType>(*contextLengths);
    auto numsDraftTokensPtr = bufferCast<SizeType>(*numsDraftTokens);
    auto draftTokensPtr = bufferCast<SizeType>(*draftTokens);
    auto targetTokensPtr = bufferCast<SizeType>(*targetTokens);
    auto finishedStepsPtr = bufferCast<bool>(*finishedSteps);
    auto finishedFinalPtr = bufferCast<bool>(*finishedFinal);
    auto finishedSumPtr = bufferCast<SizeType>(*finishedSum);
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

        for (int i = 0; i < realLen; ++i)
        {
            finishedStepsPtr[i * batchSize * beamWidth + bi] = false;
        }
        for (int i = realLen; i <= numsDraftTokensPtr[batchIdx]; ++i)
        {
            finishedStepsPtr[i * batchSize * beamWidth + bi] = true;
        }

        // Init helper vector with max value
        acceptedLen[bi] = sequenceLengthsPtr[bi];
        acceptedFinished[bi] = finishedStepsPtr[realLen * batchSize * beamWidth + bi];
    }
    // Fill token arrays
    for (SizeType bi = 0; bi < batchSize * beamWidth; ++bi)
    {
        // Draft: [d0, d1, d2, ... for numsDraftTokensPtr[bi] ... , dN]
        // Target: [vocabSize + 1, vocabSize + 1, ... for contextLengthsPtr[bi] ... vocabSize + 1,
        //         t0, t1, t2, ... for numsDraftTokensPtr[bi] ... , tN,
        //         vocabSize + 1, vocabSize + 1, .. to maxSeqLen]
        for (SizeType si = 0; si < contextLengthsPtr[bi]; ++si)
        {
            targetTokensPtr[bi * maxSeqLen + si] = vocabSize + 1;
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
                acceptedLen[bi] = std::min(acceptedLen[bi], std::min(si + 1, maxSeqLen));
                acceptedFinished[bi] = finishedStepsPtr[draftTokenIdx * batchSize * beamWidth + bi];
            }
        }
        for (SizeType si = sequenceLengthsPtr[bi]; si < maxSeqLen; ++si)
        {
            targetTokensPtr[bi * maxSeqLen + si] = vocabSize + 1;
        }
    }

    // Call function
    tk::invokeAcceptTokens(draftTokensPtr, targetTokensPtr, contextLengthsPtr, numsDraftTokensPtr, sequenceLengthsPtr,
        finishedStepsPtr, finishedFinalPtr, finishedSumPtr, batchSize, beamWidth, maxSeqLen, maxDraftTokens,
        stream->get());

    stream->synchronize();

    // Verify seqLen for accepted tokens
    int finishedSumRef = 0;
    for (SizeType bi = 0; bi < batchSize * beamWidth; ++bi)
    {
        EXPECT_EQ(acceptedLen[bi], sequenceLengthsPtr[bi]) << " bi " << bi << " seed " << seed;
        EXPECT_EQ(acceptedFinished[bi], finishedFinalPtr[bi]) << " bi " << bi << " seed " << seed;
        finishedSumRef += static_cast<SizeType>(acceptedFinished[bi]);
    }
    EXPECT_EQ(finishedSumRef, finishedSumPtr[0]);
}

TEST(DecodingKernelsTest, acceptTokensKernel)
{
    constexpr SizeType seeds = 64;
    for (SizeType seed = 0; seed < seeds; ++seed)
    {
        runAcceptedTokensTest(seed);
    }
}

} // end of namespace

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
#include "tensorrt_llm/kernels/stopCriteriaKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include <algorithm>
#include <curand_kernel.h>
#include <random>

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

using namespace tensorrt_llm::runtime;

namespace
{

class StopCriteriaKernelsTest : public testing::Test
{
public:
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

    void SetUp() override
    {
        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);
    }

    void TearDown() override {}

    void initData(SizeType seed, const std::vector<std::vector<std::vector<SizeType>>>& stopWords,
        SizeType stopWordsLen, SizeType batchSize, SizeType beamWidth)
    {
        std::mt19937 generator(seed);
        std::uniform_int_distribution<int> seqLenDistr(0, mMaxSeqLen);

        mSequenceLengths
            = mBufferManager->pinned(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32);
        mSequenceLengthLimits = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
        mFinished = mBufferManager->pinned(
            ITensor::makeShape({batchSize, beamWidth}), TRTDataType<tk::FinishedState::UnderlyingType>::value);
        mFinishedSum = mBufferManager->pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

        mOutputIds = mBufferManager->pinned(
            ITensor::makeShape({batchSize, beamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);
        mOutputIdsPtr = mBufferManager->pinned(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT64);

        mParentIds = mBufferManager->pinned(
            ITensor::makeShape({batchSize, beamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);
        mParentIdsPtr = mBufferManager->pinned(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT64);

        mRefOutputIds = mBufferManager->pinned(
            ITensor::makeShape({batchSize, beamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);

        mStopWords
            = mBufferManager->pinned(ITensor::makeShape({batchSize, 2, stopWordsLen}), nvinfer1::DataType::kINT32);

        auto sequenceLengthsPtr = bufferCast<SizeType>(*mSequenceLengths);
        auto sequenceLengthLimitsPtr = bufferCast<SizeType>(*mSequenceLengthLimits);
        auto finishedPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished));
        auto finishedSumPtr = bufferCast<SizeType>(*mFinishedSum);

        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            for (SizeType ri = 0; ri < beamWidth; ri++)
            {
                sequenceLengthsPtr[bi * beamWidth + ri]
                    = stopWordsLen == 0 ? seqLenDistr(generator) : mMaxSeqLen - (bi + ri) % mMaxSeqLen;
                finishedPtr[bi * beamWidth + ri] = tk::FinishedState::empty();
            }
        }
        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            sequenceLengthLimitsPtr[bi] = stopWordsLen == 0 ? seqLenDistr(generator) : mMaxSeqLen - bi % mMaxSeqLen;
        }
        finishedSumPtr[0] = 0;

        auto outputIdsPtrsData = reinterpret_cast<void**>(bufferCast<int64_t>(*mOutputIdsPtr));
        auto parentIdsPtrsData = reinterpret_cast<void**>(bufferCast<int64_t>(*mParentIdsPtr));
        auto outputIdsData = bufferCast<int32_t>(*mOutputIds);
        auto refOutputIdsData = bufferCast<int32_t>(*mRefOutputIds);
        auto parentIdsData = bufferCast<int32_t>(*mParentIds);

        // Tokens ids are
        // bi: 0, ri: 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        // bi: 0, ri: 1: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        // bi: 1, ri: 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        // bi: 1, ri: 1: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        // bi: 2, ri: 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        // bi: 2, ri: 1: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        // bi: 3, ri: 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        // bi: 3, ri: 1: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        for (SizeType bi = 0; bi < batchSize; bi++)
        {
            for (SizeType ri = 0; ri < beamWidth; ri++)
            {
                for (SizeType si = 0; si < mMaxSeqLen; si++)
                {
                    auto const idx = tc::flat_index3(bi, ri, si, beamWidth, mMaxSeqLen);
                    outputIdsData[idx] = ri * mMaxSeqLen + si;
                    parentIdsData[idx] = 0;
                }
            }
        }

        for (SizeType bi = 0; bi < batchSize; bi++)
        {
            outputIdsPtrsData[bi] = outputIdsData + bi * beamWidth * mMaxSeqLen;
            parentIdsPtrsData[bi] = parentIdsData + bi * beamWidth * mMaxSeqLen;
        }

        // Init stop words tensor
        auto stopWordsData = bufferCast<int32_t>(*mStopWords);
        std::fill(stopWordsData, stopWordsData + batchSize * 2 * stopWordsLen, -1);
        for (SizeType bi = 0; bi < stopWords.size(); bi++)
        {
            SizeType totalLen = 0;
            for (SizeType wi = 0; wi < stopWords[bi].size(); ++wi)
            {
                for (SizeType si = 0; si < stopWords[bi][wi].size(); ++si)
                {
                    stopWordsData[bi * 2 * stopWordsLen + 0 * stopWordsLen + totalLen + si] = stopWords[bi][wi][si];
                }
                totalLen += stopWords[bi][wi].size();
                // Do not add value if stop words is empty
                if (totalLen > 0)
                {
                    stopWordsData[bi * 2 * stopWordsLen + 1 * stopWordsLen + wi] = totalLen;
                }
            }
            // Special case when all stop words are of single token length
            if (stopWords[bi].size() == totalLen)
            {
                stopWordsData[bi * 2 * stopWordsLen + 1 * stopWordsLen + totalLen] = totalLen + 1;
            }
        }
    }

    void verifyMaxSeqLenStopCriteriaResults(SizeType seed, SizeType batchSize, SizeType beamWidth)
    {
        mStream->synchronize();

        auto sequenceLengthsPtr = bufferCast<SizeType>(*mSequenceLengths);
        auto sequenceLengthLimitsPtr = bufferCast<SizeType>(*mSequenceLengthLimits);
        auto finishedPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished));
        auto finishedSumPtr = bufferCast<SizeType>(*mFinishedSum);

        int32_t refSumFinished = 0;
        for (SizeType bi = 0; bi < batchSize * beamWidth; ++bi)
        {
            auto const batchIdx = bi / beamWidth;
            auto const beamIdx = bi % beamWidth;
            const auto limitExceeded = sequenceLengthsPtr[bi] >= sequenceLengthLimitsPtr[batchIdx];
            refSumFinished += limitExceeded;
            if (limitExceeded)
            {
                EXPECT_TRUE(finishedPtr[bi].isFinishedMaxLength())
                    << " batchIdx: " << batchIdx << " beamIdx: " << beamIdx << " seed: " << seed;
            }
        }
        EXPECT_EQ(refSumFinished, finishedSumPtr[0]);
    }

    bool isSubsequence(const SizeType* sequence, SizeType n, const std::vector<int>& subsequence)
    {
        auto it = std::search(sequence, sequence + n, subsequence.begin(), subsequence.end());
        return it != sequence + n;
    }

    void verifyStopWordsStopCriteriaResults(SizeType seed,
        const std::vector<std::vector<std::vector<SizeType>>>& stopWords, SizeType stopWordsLen, SizeType batchSize,
        SizeType beamWidth)
    {
        mStream->synchronize();

        auto outputIdsData = bufferCast<int32_t>(*mOutputIds);
        auto finishedPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished));
        auto sequenceLengthsPtr = bufferCast<SizeType>(*mSequenceLengths);

        for (SizeType bi = 0; bi < batchSize; bi++)
        {
            for (SizeType bwi = 0; bwi < beamWidth; bwi++)
            {
                auto outputIdsBatchBeam = outputIdsData + bi * beamWidth * mMaxSeqLen + bwi * mMaxSeqLen;
                bool found = false;
                for (SizeType wi = 0; wi < stopWords[bi].size(); ++wi)
                {
                    auto const wordLen = stopWords[bi][wi].size();
                    auto const seqLen = sequenceLengthsPtr[bi * beamWidth + bwi];
                    auto const offset = seqLen - wordLen;
                    found |= isSubsequence(outputIdsBatchBeam + offset, wordLen, stopWords[bi][wi]);
                    if (found)
                    {
                        EXPECT_TRUE(finishedPtr[bi * beamWidth + bwi].isFinishedStopWords());
                        break;
                    }
                }
                if (!found)
                {
                    EXPECT_FALSE(finishedPtr[bi * beamWidth + bwi].isFinished());
                }
            }
        }
    }

    void runStopWordsCriteriaTest(
        const std::vector<std::vector<std::vector<SizeType>>>& stopWords, SizeType batchSize, SizeType beamWidth)
    {
        SizeType maxStopWordsLen = 0;
        for (const auto& batchStopWords : stopWords)
        {
            SizeType stopWordsLen = 0;
            for (const auto& words : batchStopWords)
            {
                stopWordsLen += words.size();
            }
            if (stopWordsLen == batchStopWords.size())
            {
                stopWordsLen += 1;
            }
            maxStopWordsLen = std::max(maxStopWordsLen, stopWordsLen);
        }

        initData(0, stopWords, maxStopWordsLen, batchSize, beamWidth);

        tk::invokeStopWordsCriterion(reinterpret_cast<const int**>(bufferCast<int64_t>(*mOutputIdsPtr)),
            reinterpret_cast<const int**>(bufferCast<int64_t>(*mParentIdsPtr)), bufferCast<SizeType>(*mStopWords),
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished)),
            bufferCast<SizeType>(*mSequenceLengths), maxStopWordsLen, batchSize, beamWidth, mMaxSeqLen, mStream->get());

        verifyStopWordsStopCriteriaResults(0, stopWords, maxStopWordsLen, batchSize, beamWidth);
    }

    void runMaxLengthCriteriaTest(SizeType seed, SizeType batchSize, SizeType beamWidth)
    {
        initData(seed, {}, 0, batchSize, beamWidth);

        tk::invokeLengthCriterion(
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished)),
            bufferCast<SizeType>(*mFinishedSum),
            reinterpret_cast<const uint32_t*>(bufferCast<SizeType>(*mSequenceLengthLimits)),
            bufferCast<SizeType>(*mSequenceLengths), batchSize, beamWidth, mStream->get());

        verifyMaxSeqLenStopCriteriaResults(seed, batchSize, beamWidth);
    }

protected:
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;

    TensorPtr mSequenceLengths;
    TensorPtr mSequenceLengthLimits;
    TensorPtr mFinished;
    TensorPtr mFinishedSum;

    TensorPtr mOutputIds;
    TensorPtr mRefOutputIds;
    TensorPtr mOutputIdsPtr;
    TensorPtr mParentIds;
    TensorPtr mParentIdsPtr;
    TensorPtr mStopWords;

    static constexpr SizeType mMaxSeqLen{16};
    static constexpr SizeType mVocabSize{32};
};

TEST_F(StopCriteriaKernelsTest, maxLengthCriteriaBS1BW1Test)
{
    constexpr SizeType seeds = 64;
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    for (SizeType seed = 0; seed < seeds; ++seed)
    {
        this->runMaxLengthCriteriaTest(seed, batchSize, beamWidth);
    }
}

TEST_F(StopCriteriaKernelsTest, maxLengthCriteriaBS1BW2Test)
{
    constexpr SizeType seeds = 64;
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 2;
    for (SizeType seed = 0; seed < seeds; ++seed)
    {
        this->runMaxLengthCriteriaTest(seed, batchSize, beamWidth);
    }
}

TEST_F(StopCriteriaKernelsTest, maxLengthCriteriaBS1024BW1Test)
{
    constexpr SizeType seeds = 64;
    constexpr SizeType batchSize = 1024;
    constexpr SizeType beamWidth = 1;
    for (SizeType seed = 0; seed < seeds; ++seed)
    {
        this->runMaxLengthCriteriaTest(seed, batchSize, beamWidth);
    }
}

TEST_F(StopCriteriaKernelsTest, maxLengthCriteriaBS1024BW2Test)
{
    constexpr SizeType seeds = 64;
    constexpr SizeType batchSize = 1024;
    constexpr SizeType beamWidth = 2;
    for (SizeType seed = 0; seed < seeds; ++seed)
    {
        this->runMaxLengthCriteriaTest(seed, batchSize, beamWidth);
    }
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS1SingleTokenSingleWordTest)
{
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    // Expected to not match any word
    this->runStopWordsCriteriaTest({{{2}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS1SingleTokenMultipleWordsTest)
{
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    // Expected to match 15
    this->runStopWordsCriteriaTest({{{145}, {4}, {1}, {15}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS1MultipleTokensSingleWordTest)
{
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    // Expected to not match any word
    this->runStopWordsCriteriaTest({{{2, 3}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS1MultipleTokensMultipleWordsMatchTest)
{
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    // Expected to match {13, 14, 15}
    this->runStopWordsCriteriaTest({{{1, 4}, {2, 3}, {13, 14, 15}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS1MultipleTokensMultipleWordsNotMatchTest)
{
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    // Expected to not match any word
    this->runStopWordsCriteriaTest({{{1, 4}, {2, 3}, {12, 14, 15}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS4MultipleTokensMultipleWordsTest)
{
    constexpr SizeType batchSize = 4;
    constexpr SizeType beamWidth = 1;
    // Expected to match {12, 13} for the 3rd instance in the batch
    this->runStopWordsCriteriaTest({{{2}}, {{}}, {{15}, {12, 13}}, {{1}, {8, 9}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS4BW2MultipleTokensMultipleWordsTest)
{
    constexpr SizeType batchSize = 4;
    constexpr SizeType beamWidth = 2;
    // Expected to match {12, 13} to {bi, bw}={{2, 0}}
    // Expected to match {11, 12} to {bi, bw}={{3, 0}}
    // Expected to match {27} to {bi, bw}={{3, 1}}
    this->runStopWordsCriteriaTest({{{2}}, {{}}, {{11}, {12, 13}}, {{27}, {11, 12}}}, batchSize, beamWidth);
}

} // end of namespace

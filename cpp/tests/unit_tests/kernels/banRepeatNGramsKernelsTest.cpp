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

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/banRepeatNgram.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <curand_kernel.h>
#include <gtest/gtest.h>

#include <algorithm>

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
namespace trk = tensorrt_llm::runtime::kernels;

using namespace tensorrt_llm::runtime;

namespace
{

class BanRepeatNgramKernelsTest : public testing::Test
{
public:
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

    void SetUp() override
    {
        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);
    }

    void TearDown() override {}

    void initData(std::vector<std::vector<SizeType32>> const& outputIds, std::vector<SizeType32> const& nGramSizes)
    {
        auto const ptrType = TRTDataType<void*>::value;
        SizeType32 const batchSize = outputIds.size();
        auto const maxBatchSize = 2 * batchSize;

        mLogits = BufferManager::pinned(ITensor::makeShape({batchSize, mVocabSizePadded}), nvinfer1::DataType::kFLOAT);

        mSequenceLengths
            = BufferManager::pinned(ITensor::makeShape({maxBatchSize, mBeamWidth}), nvinfer1::DataType::kINT32);
        mFinished = BufferManager::pinned(
            ITensor::makeShape({maxBatchSize, mBeamWidth}), TRTDataType<tk::FinishedState::UnderlyingType>::value);

        mOutputIds = BufferManager::pinned(
            ITensor::makeShape({maxBatchSize, mBeamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);
        mOutputIdsPtr = BufferManager::pinned(ITensor::makeShape({maxBatchSize, mBeamWidth}), ptrType);

        mParentIds = BufferManager::pinned(
            ITensor::makeShape({maxBatchSize, mBeamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);
        mParentIdsPtr = BufferManager::pinned(ITensor::makeShape({maxBatchSize, mBeamWidth}), ptrType);

        mNGramSizes = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

        mBatchSlots = BufferManager::pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);

        auto batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            batchSlotsPtr[bi] = 2 * bi;
        }

        auto logitsPtr = bufferCast<float>(*mLogits);
        auto sequenceLengthsPtr = bufferCast<SizeType32>(*mSequenceLengths);
        auto finishedPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished));

        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (SizeType32 ri = 0; ri < mBeamWidth; ri++)
            {
                sequenceLengthsPtr[batchSlot * mBeamWidth + ri] = outputIds[bi].size() - 1;
                finishedPtr[batchSlot * mBeamWidth + ri] = tk::FinishedState::empty();
            }
        }

        trk::invokeFill(*mLogits, 0.f, *mStream);
        trk::invokeFill(*mNGramSizes, int32_t{0}, *mStream);
        mStream->synchronize();

        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const lastId = outputIds[bi].back();
            logitsPtr[bi * mVocabSizePadded + lastId] = 1.f;
        }

        auto outputIdsPtrsData = BufferRange<void*>(*mOutputIdsPtr);
        auto parentIdsPtrsData = BufferRange<void*>(*mParentIdsPtr);
        auto outputIdsData = bufferCast<int32_t>(*mOutputIds);
        auto parentIdsData = bufferCast<int32_t>(*mParentIds);

        for (SizeType32 bi = 0; bi < batchSize; bi++)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (SizeType32 ri = 0; ri < mBeamWidth; ri++)
            {
                for (SizeType32 si = 0; si < outputIds[bi].size(); si++)
                {
                    auto const idx = tc::flat_index3(batchSlot, ri, si, mBeamWidth, mMaxSeqLen);
                    outputIdsData[idx] = outputIds[bi][si];
                    parentIdsData[idx] = 0;
                }
            }
        }

        for (SizeType32 bi = 0; bi < maxBatchSize; bi++)
        {
            outputIdsPtrsData[bi] = outputIdsData + bi * mBeamWidth * mMaxSeqLen;
            parentIdsPtrsData[bi] = parentIdsData + bi * mBeamWidth * mMaxSeqLen;
        }

        auto nGramSizesPtr = bufferCast<int32_t>(*mNGramSizes);
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            nGramSizesPtr[batchSlot] = nGramSizes[bi];
        }
    }

    void verifyBanRepeatNGramResults(
        std::vector<SizeType32> const& nGramSizes, std::vector<SizeType32> const& expectedLastId)
    {

        auto const batchSize = expectedLastId.size();
        auto const maxBatchSize = 2 * batchSize;

        auto batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
        auto logitsPtr = bufferCast<float>(*mLogits);

        for (SizeType32 bi = 0; bi < batchSize; bi++)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            auto const lastId = expectedLastId[bi];
            auto const shouldHave = lastId > 0;
            auto const lastIdAbs = std::abs(lastId);

            auto const outLogit = logitsPtr[bi * mVocabSizePadded + lastIdAbs];
            auto const expectedLogit = shouldHave ? 1.f : -INFINITY;
            EXPECT_EQ(outLogit, expectedLogit) << "bi: " << bi << " vi: " << lastId;
        }
    }

    void runBanRepeatNGramTest(std::vector<std::vector<SizeType32>> const& outputIds,
        std::vector<SizeType32> const& nGramSizes, std::vector<SizeType32> const& expectedLastId)
    {
        auto const batchSize = expectedLastId.size();
        int32_t maxStep = 0;
        for (auto const& ids : outputIds)
        {
            maxStep = std::max(maxStep, static_cast<int32_t>(ids.size() - 1));
        }
        initData(outputIds, nGramSizes);

        tk::invokeBanRepeatNgram(bufferCast<float>(*mLogits),
            reinterpret_cast<int32_t const**>(bufferCast<int64_t>(*mOutputIdsPtr)),
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished)),
            reinterpret_cast<int32_t const**>(bufferCast<int64_t>(*mParentIdsPtr)), bufferCast<int32_t>(*mBatchSlots),
            bufferCast<int32_t>(*mSequenceLengths), batchSize, mBeamWidth, mMaxSeqLen,
            bufferCast<int32_t>(*mNGramSizes), mVocabSizePadded, maxStep, mStream->get());

        mStream->synchronize();

        verifyBanRepeatNGramResults(nGramSizes, expectedLastId);
    }

protected:
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;

    TensorPtr mLogits;
    TensorPtr mSequenceLengths;
    TensorPtr mFinished;

    TensorPtr mOutputIds;
    TensorPtr mRefOutputIds;
    TensorPtr mOutputIdsPtr;
    TensorPtr mParentIds;
    TensorPtr mParentIdsPtr;
    TensorPtr mNGramSizes;
    TensorPtr mBatchSlots;

    static constexpr SizeType32 mMaxSeqLen{16};
    static constexpr SizeType32 mVocabSizePadded{32};
    // TODO: add beam width
    static constexpr SizeType32 mBeamWidth{1};
};

TEST_F(BanRepeatNgramKernelsTest, noRepeatNGramsBS1BW1Test)
{
    std::vector<std::vector<std::vector<SizeType32>>> outputIds
        = {{{1, 2, 3, 4, 5, 6, 2, 3}}, {{1, 2, 3, 4, 5, 6, 2, 3}}};
    std::vector<std::vector<SizeType32>> nGramSizes = {{2}, {3}};
    // Positive value shows expected id of the last token. Negative value shows not-expected id of the last token
    std::vector<std::vector<SizeType32>> expectedOutputIds = {{-3}, {3}};
    for (SizeType32 ti = 0; ti < nGramSizes.size(); ++ti)
    {
        this->runBanRepeatNGramTest(outputIds[ti], nGramSizes[ti], expectedOutputIds[ti]);
    }
}

TEST_F(BanRepeatNgramKernelsTest, noRepeatNGramsBS2BW1Test)
{
    std::vector<std::vector<std::vector<SizeType32>>> outputIds
        = {{{1, 2, 3, 6, 2, 3}, {1, 3, 3, 4, 5, 6, 2, 3}}, {{1, 2, 3, 2, 3}, {1, 2, 3, 4, 5, 6, 2, 3}}};
    std::vector<std::vector<SizeType32>> nGramSizes = {{2, 2}, {3, 2}};
    // Positive value shows expected id of the last token. Negative value shows not-expected id of the last token
    std::vector<std::vector<SizeType32>> expectedOutputIds = {{-3, 3}, {3, -3}};
    for (SizeType32 ti = 0; ti < nGramSizes.size(); ++ti)
    {
        this->runBanRepeatNGramTest(outputIds[ti], nGramSizes[ti], expectedOutputIds[ti]);
    }
}

} // end of namespace

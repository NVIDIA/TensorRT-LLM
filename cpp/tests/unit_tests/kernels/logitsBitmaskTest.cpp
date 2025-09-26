/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/logitsBitmask.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <random>

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

using namespace tensorrt_llm::runtime;

namespace
{

template <typename T>
class LogitsBitmaskTest : public testing::Test
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using BitmaskT = uint32_t;

    void SetUp() override
    {
        mStream = std::make_shared<CudaStream>();
        mBufferManager = std::make_shared<BufferManager>(mStream);
    }

    void TearDown() override {}

    void initData(SizeType32 seed, SizeType32 batchSize, SizeType32 vocabSizePadded)
    {
        std::mt19937 generator(seed);
        std::vector<int32_t> index(batchSize);
        std::iota(index.begin(), index.end(), 0);
        std::shuffle(index.begin(), index.end(), generator);
        std::uniform_int_distribution<BitmaskT> dist(
            std::numeric_limits<BitmaskT>::min(), std::numeric_limits<BitmaskT>::max());

        mBatchSize = batchSize;
        mVocabSizePadded = vocabSizePadded;
        mBitmaskSize = tc::ceilDiv(vocabSizePadded, 32);

        auto constexpr logitsDtype = TRTDataType<T>::value;
        auto constexpr logitsPtrDtype = TRTDataType<T*>::value;
        auto constexpr bitmaskDtype = TRTDataType<BitmaskT>::value;
        auto constexpr bitmaskPtrDtype = TRTDataType<BitmaskT*>::value;

        mLogitsBitmask = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mBitmaskSize}), bitmaskDtype);
        mLogitsBitmaskHost = BufferManager::pinned(ITensor::makeShape({mBatchSize, mBitmaskSize}), bitmaskDtype);
        mLogitsBitmaskPtrVec = mBufferManager->gpu(ITensor::makeShape({mBatchSize}), bitmaskPtrDtype);
        mLogitsBitmaskPtrVecHost = BufferManager::pinned(ITensor::makeShape({mBatchSize}), bitmaskPtrDtype);
        mLogits = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mVocabSizePadded}), logitsDtype);
        mLogitsHost = BufferManager::pinned(ITensor::makeShape({mBatchSize, mVocabSizePadded}), logitsDtype);
        mLogitsPtrVec = mBufferManager->gpu(ITensor::makeShape({mBatchSize}), logitsPtrDtype);
        mLogitsPtrVecHost = BufferManager::pinned(ITensor::makeShape({mBatchSize}), logitsPtrDtype);

        auto logitsHost = BufferRange<T>(*mLogitsHost);
        auto logitsBitmaskHost = BufferRange<BitmaskT>(*mLogitsBitmaskHost);
        auto logitsPtrVecHost = BufferRange<T*>(*mLogitsPtrVecHost);
        auto logitsBitmaskPtrVecHost = BufferRange<BitmaskT*>(*mLogitsBitmaskPtrVecHost);
        for (int i = 0; i < mBatchSize; i++)
        {
            for (int j = 0; j < mVocabSizePadded; j++)
            {
                logitsHost[i * mVocabSizePadded + j] = static_cast<T>(0.0f);
            }
            for (int j = 0; j < mBitmaskSize; j++)
            {
                logitsBitmaskHost[i * mBitmaskSize + j] = dist(generator);
            }
            // Map to randomly shuffled addresses
            logitsPtrVecHost[i] = bufferCast<T>(*ITensor::at(mLogits, {index[i]}));
            logitsBitmaskPtrVecHost[i] = bufferCast<BitmaskT>(*ITensor::at(mLogitsBitmask, {index[i]}));
        }

        mBufferManager->copy(*mLogitsHost, *mLogits);
        mBufferManager->copy(*mLogitsBitmaskHost, *mLogitsBitmask);
        mBufferManager->copy(*mLogitsPtrVecHost, *mLogitsPtrVec);
        mBufferManager->copy(*mLogitsBitmaskPtrVecHost, *mLogitsBitmaskPtrVec);
    }

    void runTest()
    {
        auto logitsPtrVec = bufferCast<T*>(*mLogitsPtrVec);
        auto logitsBitmaskPtrVec = bufferCast<BitmaskT const*>(*mLogitsBitmaskPtrVec);

        tk::invokeLogitsBitmask<T>(logitsPtrVec, logitsBitmaskPtrVec, mBatchSize, mVocabSizePadded, mStream->get());

        mBufferManager->copy(*mLogits, *mLogitsHost);
        mStream->synchronize();

        auto logitsHost = BufferRange<T>(*mLogitsHost);
        auto logitsBitmaskHost = BufferRange<BitmaskT>(*mLogitsBitmaskHost);
        for (int i = 0; i < mBatchSize; i++)
        {
            for (int j = 0; j < mBitmaskSize; j++)
            {
                auto bitmaskVal = logitsBitmaskHost[i * mBitmaskSize + j];
                for (int k = 0; k < 32; k++)
                {
                    if (j * 32 + k >= mVocabSizePadded)
                    {
                        break;
                    }
                    auto logitsVal = static_cast<float>(logitsHost[i * mVocabSizePadded + j * 32 + k]);
                    if (!(bitmaskVal & 1))
                    {
                        EXPECT_LT(logitsVal, -1e6);
                    }
                    else
                    {
                        EXPECT_FLOAT_EQ(logitsVal, 0);
                    }
                    bitmaskVal >>= 1;
                }
            }
        }
    }

protected:
    std::shared_ptr<BufferManager> mBufferManager;
    std::shared_ptr<CudaStream> mStream;

    SizeType32 mBatchSize;
    SizeType32 mVocabSizePadded;
    SizeType32 mBitmaskSize;            // CeilDiv(vocabSizePadded, 32)

    TensorPtr mLogitsBitmask;           // [mBatchSize, mBitmaskSize]
    TensorPtr mLogitsBitmaskHost;       // [mBatchSize, mBitmaskSize]
    TensorPtr mLogitsBitmaskPtrVec;     // [mBatchSize], pointers to the logitsBitmask in a batch
    TensorPtr mLogitsBitmaskPtrVecHost; // [mBatchSize]
    TensorPtr mLogits;                  // [mBatchSize, mVocabSizePadded]
    TensorPtr mLogitsHost;              // [mBatchSize, mVocabSizePadded]
    TensorPtr mLogitsPtrVec;            // [mBatchSize], pointers to the logits in a batch
    TensorPtr mLogitsPtrVecHost;        // [mBatchSize]
};

using TestingTypes = testing::Types<float, half, __nv_bfloat16>;

TYPED_TEST_SUITE(LogitsBitmaskTest, TestingTypes);

TYPED_TEST(LogitsBitmaskTest, Aligned)
{
    this->initData(0, 16, 128000);
    this->runTest();
}

TYPED_TEST(LogitsBitmaskTest, NotAligned)
{
    this->initData(515, 123, 1234);
    this->runTest();
}

} // namespace

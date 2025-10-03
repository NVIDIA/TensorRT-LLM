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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/speculativeDecoding/eagleDecodingKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <NvInferRuntimeBase.h>

#include <algorithm>
#include <cstdint>
#include <random>

namespace
{

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::common;

namespace tk = tensorrt_llm::kernels;
namespace trk = tensorrt_llm::runtime::kernels;
namespace tksd = tensorrt_llm::kernels::speculative_decoding;

class SamplingParams
{
public:
    SamplingParams() {}

    inline void setNumCtxRequests(SizeType32 numCtxRequests)
    {
        mNumCtxRequests = numCtxRequests;
    }

    inline void setNumGenRequests(SizeType32 numGenRequests)
    {
        mNumGenRequests = numGenRequests;
    }

    inline void setMaxPathLen(SizeType32 maxPathLen)
    {
        mMaxPathLen = maxPathLen;
    }

    [[nodiscard]] inline SizeType32 getNumCtxRequests() const
    {
        return mNumCtxRequests;
    }

    [[nodiscard]] inline SizeType32 getNumGenRequests() const
    {
        return mNumGenRequests;
    }

    [[nodiscard]] inline SizeType32 getBatchSize() const
    {
        return getNumCtxRequests() + getNumGenRequests();
    }

    [[nodiscard]] inline SizeType32 getVocabSize() const
    {
        return mVocabSize;
    }

    [[nodiscard]] inline SizeType32 getMaxBatchSize() const
    {
        return 2 * getBatchSize();
    }

    [[nodiscard]] inline SizeType32 getMaxPathLen() const
    {
        return mMaxPathLen;
    }

    [[nodiscard]] inline SizeType32 getMaxDecodingTokens() const
    {
        return mMaxDecodingTokens;
    }

    [[nodiscard]] inline SizeType32 getMaxDecodingDraftTokens() const
    {
        return getMaxDecodingTokens() - 1;
    }

    [[nodiscard]] inline SizeType32 getMaxSeqLen() const
    {
        return getMaxDecodingTokens() * 2;
    }

private:
    SizeType32 mNumCtxRequests{6};
    SizeType32 mNumGenRequests{6};
    SizeType32 mMaxPathLen{4};
    SizeType32 mMaxDecodingTokens{32};
    SizeType32 mVocabSize{256};
};

class EaglePackDataTest : public ::testing::Test
{
public:
    using BufferPtr = IBuffer::SharedPtr;
    using TensorPtr = ITensor::SharedPtr;

    void SetUp() override
    {
        mStream = std::make_shared<CudaStream>();
        mBufferManager = std::make_shared<BufferManager>(mStream);
    }

    void allocateBuffers()
    {
        // inputs
        mBatchSlots = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

        mInputTemperatures = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kFLOAT);

        mInputRandomDataSample = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kFLOAT);

        mInputRandomDataValidation = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingTokens()}),
            nvinfer1::DataType::kFLOAT);

        mInputNextDraftTokens = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingDraftTokens()}),
            nvinfer1::DataType::kINT32);

        mInputNextDraftPaths
            = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getMaxBatchSize(),
                                            mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen()}),
                nvinfer1::DataType::kINT32);

        mInputSpecDecodingGenerationLengths = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

        mInputSpecDecodingPositionOffsets = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingTokens()}),
            nvinfer1::DataType::kINT32);

        auto const numPackedMasks
            = static_cast<SizeType32>(tensorrt_llm::common::divUp(mSamplingParams.getMaxDecodingTokens(), 32));
        mInputSpecDecodingPackedMasks = BufferManager::pinnedPool(
            ITensor::makeShape(
                {mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingTokens(), numPackedMasks}),
            nvinfer1::DataType::kINT32);

        // outputs
        mOutputTemperatures = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kFLOAT);

        mOutputRandomDataSample = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kFLOAT);

        mOutputRandomDataValidation = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getBatchSize(), mSamplingParams.getMaxDecodingTokens()}),
            nvinfer1::DataType::kFLOAT);

        mOutputNextDraftTokens = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getBatchSize(), mSamplingParams.getMaxDecodingDraftTokens()}),
            nvinfer1::DataType::kINT32);

        mOutputNextDraftLens = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

        mOutputNextDraftPaths
            = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize(),
                                            mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen()}),
                nvinfer1::DataType::kINT32);

        mOutputSpecDecodingGenerationLengths = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

        mOutputSpecDecodingPositionOffsets = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getBatchSize(), mSamplingParams.getMaxDecodingTokens()}),
            nvinfer1::DataType::kINT32);

        mOutputSpecDecodingPackedMasks = BufferManager::pinnedPool(
            ITensor::makeShape(
                {mSamplingParams.getBatchSize(), mSamplingParams.getMaxDecodingTokens(), numPackedMasks}),
            nvinfer1::DataType::kINT32);

        // workspace
        mMaxGenerationLength = BufferManager::pinnedPool(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

        mCumSumGenerationLengths = BufferManager::pinnedPool(
            ITensor::makeShape({mSamplingParams.getBatchSize() + 1}), nvinfer1::DataType::kINT32);

        mScanReduceTempStorageBytes = tksd::invokeScanReduceGenerationLengths(
            mSamplingParams.getBatchSize(), nullptr, nullptr, 0, nullptr, nullptr, mStream->get());
        mScanReduceTempStorage = mBufferManager->gpu(mScanReduceTempStorageBytes);
    }

    void initBuffers()
    {
        trk::invokeFill(*mOutputTemperatures, float{0}, *mStream);
        trk::invokeFill(*mOutputRandomDataSample, float{0}, *mStream);
        trk::invokeFill(*mOutputRandomDataValidation, float{0}, *mStream);
        trk::invokeFill(*mOutputNextDraftTokens, TokenIdType{-1}, *mStream);
        trk::invokeFill(*mOutputNextDraftLens, SizeType32{0}, *mStream);
        trk::invokeFill(*mOutputNextDraftPaths, SizeType32{0}, *mStream);
        trk::invokeFill(*mOutputSpecDecodingGenerationLengths, SizeType32{0}, *mStream);
        trk::invokeFill(*mOutputSpecDecodingPositionOffsets, SizeType32{0}, *mStream);
        trk::invokeFill(*mOutputSpecDecodingPackedMasks, SizeType32{0}, *mStream);

        auto batchSlotsPtr = bufferCast<SizeType32>(*mBatchSlots);
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            batchSlotsPtr[bi] = 2 * bi;
        }

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> distr(0.0, 1.0);
        std::uniform_int_distribution<SizeType32> intDistr(0, 1000);
        std::uniform_int_distribution<SizeType32> lenDistr(0, mSamplingParams.getMaxDecodingTokens() - 1);
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            bufferCast<float>(*mInputTemperatures)[batchSlotsPtr[bi]] = distr(gen);
            bufferCast<float>(*mInputRandomDataSample)[batchSlotsPtr[bi]] = distr(gen);
        }

        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            for (SizeType32 ti = 0; ti < mSamplingParams.getMaxDecodingDraftTokens(); ++ti)
            {
                bufferCast<SizeType32>(*mInputNextDraftTokens)[flat_index2(
                    batchSlotsPtr[bi], ti, mSamplingParams.getMaxDecodingDraftTokens())]
                    = intDistr(gen);
            }
            for (SizeType32 ti = 0; ti < mSamplingParams.getMaxDecodingTokens(); ++ti)
            {
                bufferCast<float>(
                    *mInputRandomDataValidation)[batchSlotsPtr[bi] * mSamplingParams.getMaxDecodingTokens() + ti]
                    = distr(gen);
                for (SizeType32 pi = 0; pi < mSamplingParams.getMaxPathLen(); ++pi)
                {
                    bufferCast<SizeType32>(*mInputNextDraftPaths)[flat_index3(batchSlotsPtr[bi], ti, pi,
                        mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen())]
                        = intDistr(gen);
                }
                auto const numPackedMasks
                    = static_cast<SizeType32>(tensorrt_llm::common::divUp(mSamplingParams.getMaxDecodingTokens(), 32));
                for (SizeType32 mi = 0; mi < numPackedMasks; ++mi)
                {
                    bufferCast<SizeType32>(*mInputSpecDecodingPackedMasks)[flat_index3(
                        batchSlotsPtr[bi], ti, mi, mSamplingParams.getMaxDecodingTokens(), numPackedMasks)]
                        = intDistr(gen);
                }
                bufferCast<SizeType32>(*mInputSpecDecodingPositionOffsets)[flat_index2(
                    batchSlotsPtr[bi], ti, mSamplingParams.getMaxDecodingTokens())]
                    = intDistr(gen);
            }
            bufferCast<SizeType32>(*mInputSpecDecodingGenerationLengths)[batchSlotsPtr[bi]] = lenDistr(gen) + 1;
        }
    }

    void callPackData()
    {
        tksd::PackEagleParams params;
        params.batchSize = mSamplingParams.getBatchSize();
        params.maxNumPaths = mSamplingParams.getMaxDecodingTokens();
        params.maxDecodingTokens = mSamplingParams.getMaxDecodingTokens();
        params.maxPathLength = mSamplingParams.getMaxPathLen();
        params.numContextRequests = mSamplingParams.getNumCtxRequests();
        params.numGenerationRequests = mSamplingParams.getNumGenRequests();

        params.batchSlots = bufferCast<SizeType32>(*mBatchSlots);

        // Outputs from decoder -- inputs to the packing kernel
        params.inputTemperatures = bufferCast<float>(*mInputTemperatures);
        params.inputRandomDataSample = bufferCast<float>(*mInputRandomDataSample);
        params.inputRandomDataValidation = bufferCast<float>(*mInputRandomDataValidation);

        params.inputNextDraftTokens = bufferCast<TokenIdType>(*mInputNextDraftTokens);
        params.inputNextDraftPaths = bufferCast<SizeType32>(*mInputNextDraftPaths);

        params.inputSpecDecodingGenerationLengths = bufferCast<SizeType32>(*mInputSpecDecodingGenerationLengths);
        params.inputSpecDecodingPositionOffsets = bufferCast<SizeType32>(*mInputSpecDecodingPositionOffsets);
        params.inputSpecDecodingPackedMasks = bufferCast<int32_t>(*mInputSpecDecodingPackedMasks);

        // Outputs of the packing kernel -- inputs to the engine
        params.outputTemperatures = bufferCast<float>(*mOutputTemperatures);
        params.outputRandomDataSample = bufferCast<float>(*mOutputRandomDataSample);
        params.outputRandomDataValidation = bufferCast<float>(*mOutputRandomDataValidation);

        params.outputNextDraftTokens = bufferCast<TokenIdType>(*mOutputNextDraftTokens);
        params.outputNextDraftLens = bufferCast<SizeType32>(*mOutputNextDraftLens);
        params.outputNextDraftPaths = bufferCast<SizeType32>(*mOutputNextDraftPaths);

        params.outputSpecDecodingGenerationLengths = bufferCast<SizeType32>(*mOutputSpecDecodingGenerationLengths);
        params.outputSpecDecodingPositionOffsets = bufferCast<SizeType32>(*mOutputSpecDecodingPositionOffsets);
        params.outputSpecDecodingPackedMasks = bufferCast<int32_t>(*mOutputSpecDecodingPackedMasks);

        params.maxGenerationLength = bufferCast<SizeType32>(*mMaxGenerationLength);
        params.cumSumGenerationLengths = bufferCast<SizeType32>(*mCumSumGenerationLengths);

        params.checkParams();

        if (mSamplingParams.getNumGenRequests())
        {
            // Pack tensors from batch slot position to continuous array
            tksd::invokePackEagleGenerationLengths(params, mStream->get());

            sync_check_cuda_error(mStream->get());

            // Compute inclusive sum and max
            tksd::invokeScanReduceGenerationLengths(mSamplingParams.getNumGenRequests(),
                bufferCast<SizeType32>(*mOutputSpecDecodingGenerationLengths),
                bufferCast<uint8_t>(*mScanReduceTempStorage), mScanReduceTempStorageBytes,
                bufferCast<SizeType32>(*mCumSumGenerationLengths), bufferCast<SizeType32>(*mMaxGenerationLength),
                mStream->get());

            sync_check_cuda_error(mStream->get());
        }

        mStream->synchronize();

        // Pack tensors from batch slot position to continuous array
        tksd::invokePackEagle(params, mStream->get());

        sync_check_cuda_error(mStream->get());
    }

    void verifyResults()
    {
        auto batchSlotsPtr = bufferCast<SizeType32>(*mBatchSlots);
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            EXPECT_EQ(BufferRange<float>(*mInputTemperatures)[batchSlotsPtr[bi]],
                BufferRange<float>(*mOutputTemperatures)[bi]);
            EXPECT_EQ(BufferRange<float>(*mInputRandomDataSample)[batchSlotsPtr[bi]],
                BufferRange<float>(*mOutputRandomDataSample)[bi]);
        }

        auto const numCtxRequests = mSamplingParams.getNumCtxRequests();
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            for (SizeType32 ti = 0; ti < mSamplingParams.getMaxDecodingTokens(); ++ti)
            {
                EXPECT_EQ(
                    BufferRange<float>(
                        *mInputRandomDataValidation)[batchSlotsPtr[bi] * mSamplingParams.getMaxDecodingTokens() + ti],
                    BufferRange<float>(*mOutputRandomDataValidation)[bi * mSamplingParams.getMaxDecodingTokens() + ti]);
                for (SizeType32 pi = 0; pi < mSamplingParams.getMaxPathLen(); ++pi)
                {
                    EXPECT_EQ(BufferRange<SizeType32>(*mInputNextDraftPaths)[flat_index3(batchSlotsPtr[bi], ti, pi,
                                  mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen())],
                        BufferRange<SizeType32>(*mOutputNextDraftPaths)[flat_index3(
                            bi, ti, pi, mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen())]);
                }
            }
            EXPECT_EQ(BufferRange<SizeType32>(*mOutputNextDraftLens)[bi],
                bi < numCtxRequests
                    ? 0
                    : BufferRange<SizeType32>(*mInputSpecDecodingGenerationLengths)[batchSlotsPtr[bi]] - 1);
        }

        auto const maxGenerationLength = bufferCast<SizeType32>(*mMaxGenerationLength)[0];
        for (SizeType32 bi = 0; bi < mSamplingParams.getNumGenRequests(); ++bi)
        {
            for (SizeType32 ti = 0; ti < mSamplingParams.getMaxDecodingDraftTokens(); ++ti)
            {
                EXPECT_EQ(BufferRange<SizeType32>(*mInputNextDraftTokens)[flat_index2(
                              batchSlotsPtr[numCtxRequests + bi], ti, mSamplingParams.getMaxDecodingDraftTokens())],
                    BufferRange<SizeType32>(*mOutputNextDraftTokens)[flat_index2(
                        numCtxRequests + bi, ti, mSamplingParams.getMaxDecodingDraftTokens())]);
            }
            EXPECT_EQ(BufferRange<SizeType32>(*mInputSpecDecodingGenerationLengths)[batchSlotsPtr[numCtxRequests + bi]],
                BufferRange<SizeType32>(*mOutputSpecDecodingGenerationLengths)[bi]);
            for (SizeType32 ti = 0; ti < maxGenerationLength; ++ti)
            {
                EXPECT_EQ(BufferRange<SizeType32>(*mInputSpecDecodingPositionOffsets)[flat_index2(
                              batchSlotsPtr[numCtxRequests + bi], ti, mSamplingParams.getMaxDecodingTokens())],
                    BufferRange<SizeType32>(
                        *mOutputSpecDecodingPositionOffsets)[flat_index2(bi, ti, maxGenerationLength)])
                    << "bi: " << bi << " ti: " << ti;
            }
            auto const numTokens = (bi == 0) ? bufferCast<SizeType32>(*mCumSumGenerationLengths)[0]
                                             : bufferCast<SizeType32>(*mCumSumGenerationLengths)[bi]
                    - bufferCast<SizeType32>(*mCumSumGenerationLengths)[bi - 1];
            auto const outputStartId = (bi == 0) ? 0 : bufferCast<SizeType32>(*mCumSumGenerationLengths)[bi - 1];
            auto const numPackedMasks
                = static_cast<SizeType32>(tensorrt_llm::common::divUp(mSamplingParams.getMaxDecodingTokens(), 32));
            for (SizeType32 ti = 0; ti < numTokens * numPackedMasks; ++ti)
            {
                EXPECT_EQ(BufferRange<SizeType32>(
                              *mInputSpecDecodingPackedMasks)[flat_index2(batchSlotsPtr[numCtxRequests + bi], ti,
                              mSamplingParams.getMaxDecodingTokens() * numPackedMasks)],
                    BufferRange<SizeType32>(
                        *mOutputSpecDecodingPackedMasks)[flat_index2(outputStartId, ti, numPackedMasks)])
                    << "bi: " << bi << " ti: " << ti;
            }
        }
    }

    void run(SamplingParams samplingParams)
    {
        mSamplingParams = samplingParams;

        allocateBuffers();

        initBuffers();

        callPackData();

        mStream->synchronize();

        verifyResults();
    }

private:
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;

    // input
    TensorPtr mBatchSlots;
    TensorPtr mInputTemperatures;
    TensorPtr mInputRandomDataSample;
    TensorPtr mInputRandomDataValidation;
    TensorPtr mInputNextDraftTokens;
    TensorPtr mInputNextDraftPaths;
    TensorPtr mInputSpecDecodingGenerationLengths;
    TensorPtr mInputSpecDecodingPositionOffsets;
    TensorPtr mInputSpecDecodingPackedMasks;

    // output
    TensorPtr mOutputTemperatures;
    TensorPtr mOutputRandomDataSample;
    TensorPtr mOutputRandomDataValidation;
    TensorPtr mOutputNextDraftTokens;
    TensorPtr mOutputNextDraftLens;
    TensorPtr mOutputNextDraftPaths;
    TensorPtr mOutputSpecDecodingGenerationLengths;
    TensorPtr mOutputSpecDecodingPositionOffsets;
    TensorPtr mOutputSpecDecodingPackedMasks;

    // workspace
    TensorPtr mMaxGenerationLength;
    TensorPtr mCumSumGenerationLengths;

    BufferPtr mScanReduceTempStorage;

    SizeType32 mScanReduceTempStorageBytes;

    SamplingParams mSamplingParams;
};

TEST_F(EaglePackDataTest, Ctx1Gen0)
{
    SamplingParams params;

    params.setNumCtxRequests(1);
    params.setNumGenRequests(0);

    this->run(params);
}

TEST_F(EaglePackDataTest, Ctx0Gen1)
{
    SamplingParams params;

    params.setNumCtxRequests(0);
    params.setNumGenRequests(1);

    this->run(params);
}

TEST_F(EaglePackDataTest, Ctx100Gen0)
{
    SamplingParams params;

    params.setNumCtxRequests(100);
    params.setNumGenRequests(0);

    this->run(params);
}

TEST_F(EaglePackDataTest, Ctx0Gen100)
{
    SamplingParams params;

    params.setNumCtxRequests(0);
    params.setNumGenRequests(100);

    this->run(params);
}

TEST_F(EaglePackDataTest, Ctx100Gen100)
{
    SamplingParams params;

    params.setNumCtxRequests(100);
    params.setNumGenRequests(100);

    this->run(params);
}
} // namespace

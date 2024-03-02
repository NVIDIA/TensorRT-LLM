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
#pragma once

#include <gtest/gtest.h>

#include <memory>

#include "tensorrt_llm/layers/dynamicDecodeLayer.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include "tensorrt_llm/common/cudaAllocator.h"
#include "tensorrt_llm/common/tensorConversion.h"
#include "tensorrt_llm/common/tllmException.h"

namespace tensorrt_llm::tests::layers::sampling
{

struct SamplingParams
{
    std::vector<uint32_t> topKs;
    std::vector<float> topPs;
    std::vector<float> temperatures;
    std::vector<float> repetitionPenalties;
    std::vector<float> presencePenalties;
    std::vector<float> frequencyPenalties;
    std::vector<int32_t> minLengths;
    std::vector<float> decay;
    std::vector<float> minTopP;
    std::vector<int32_t> topPResetIds;
    std::vector<std::vector<std::vector<int32_t>>> badWords;
    std::vector<std::vector<std::vector<int32_t>>> stopWords;
    bool useBias = false;
};

template <typename T>
class DynamicDecodeLayerTest : public testing::Test
{
private:
    void SetUp() override;

    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;
    using BufferPtr = tensorrt_llm::runtime::IBuffer::SharedPtr;

    int32_t seed = 0;
    const static uint64_t mMaxSeed = 32;
    int32_t const mBatchSize = 6;
    int32_t const mMaxBatchSize = 2 * mBatchSize;
    int32_t const mBeamWidth = 1;
    int32_t const mBatchBeam = mBatchSize * mBeamWidth;
    int32_t const mVocabSize = 9;
    int32_t const mVocabSizePadded = mVocabSize;

    int32_t const mMaxInputLen = 0; // has no effect.
    int32_t const mMaxOutputLen = 4;
    int32_t const mMaxSeqLen = mMaxInputLen + mMaxOutputLen;
    int32_t const mSinkTokenLength = 0;
    int32_t mEndId = mVocabSize;

    bool mUseLogitsVec = false;

    TensorPtr mLogitsDevice;
    TensorPtr mRuntimeLogitsHost;
    TensorPtr mLogitsRefHost;
    TensorPtr mContextLengthDevice;
    TensorPtr mSeqLengthsDevice;
    TensorPtr mFinishedDevice;
    TensorPtr mFinishedSumDevice;
    TensorPtr mOutputIdsDevice;
    TensorPtr mNewTokens;
    TensorPtr mEndIdsDevice;
    TensorPtr mBatchSlots;

    TensorPtr mBadWordsLens;
    TensorPtr mBadWords;
    TensorPtr mBadWordsPtrs;

    TensorPtr mStopWordsLens;
    TensorPtr mStopWords;
    TensorPtr mStopWordsPtrs;

    TensorPtr mEmbeddingBiasHost;
    TensorPtr mEmbeddingBiasDevice;

    TensorPtr mRefLogProbsHost;
    TensorPtr mOutputLogProbsDevice;
    TensorPtr mOutputLogProbsTiledDevice;

    TensorPtr mCumLogProbsDevice;

    std::vector<tensorrt_llm::common::Tensor> mLogitsVec;

    struct cudaDeviceProp mDeviceProp;

    const tensorrt_llm::common::DataType data_type = tensorrt_llm::common::getTensorType<T>();

    // Order is important because we pass mAllocator to mDecodeLayer and it is used in destructor
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::common::CudaAllocator> mAllocator;
    std::shared_ptr<tensorrt_llm::layers::DynamicDecodeLayer<T>> mDecodeLayer;

    std::vector<T> mTestLogitsInit;

    int32_t mMaxBadWordsLen{0};
    int32_t mMaxStopWordsLen{0};

private:
    void setup(uint64_t seed, SamplingParams const& params);

    int32_t getMaxWordsLen(std::vector<std::vector<std::vector<int32_t>>> const& inputWords);
    void initXWordsTensors(int32_t* batchSlotsPtr, int32_t* wordsData, int32_t** wordsPtr, int32_t* wordsLenData,
        int32_t maxWordsLen, std::vector<std::vector<std::vector<int32_t>>> const& inputWords);

    typename tensorrt_llm::layers::DynamicDecodeLayer<T>::ForwardParams createInputTensors(int32_t step);

    typename tensorrt_llm::layers::DynamicDecodeLayer<T>::OutputParams createOutputTensors();

    void batchCopy(int32_t step);
    bool checkResult(int32_t* outputIds, std::vector<std::set<int32_t>> const& expectedIds, int32_t* seqLens,
        int32_t leadingDim, int32_t stride, int32_t step);

    void runTestImpl(
        std::vector<std::set<int32_t>> const& expectedOutputIds, SamplingParams const& params, int32_t endId = -1);

    void fillRefLogits(
        int32_t const* seqLenHost, std::vector<std::set<int32_t>> const& expectedOutputIds, int32_t step);

public:
    void runTest(
        std::vector<std::set<int32_t>> const& expectedOutputIds, SamplingParams const& params, int32_t endId = -1);
};

typedef testing::Types<float, half> FloatAndHalfTypes;

} // namespace tensorrt_llm::tests::layers::sampling

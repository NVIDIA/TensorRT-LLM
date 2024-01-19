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

#include "tensorrt_llm/layers/topKSamplingLayer.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"
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
    bool useBias = false;
};

template <typename T>
class SamplingLayerTest : public testing::Test
{
protected:
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

    int32_t seed = 0;
    const static uint64_t mMaxSeed = 32;
    const int32_t mBatchSize = 6;
    const int32_t mBeamWidth = 1;
    const int32_t mBatchBeam = mBatchSize * mBeamWidth;
    const int32_t mVocabSize = 8;

    const int32_t mVocabSizePadded = mVocabSize;

    const int32_t mMaxInputLen = 0; // has no effect.
    const int32_t mMaxOutputLen = 4;
    const int32_t mMaxSeqLen = mMaxInputLen + mMaxOutputLen;
    int32_t mEndId = mVocabSize;

    TensorPtr mLogitsDevice;
    TensorPtr mPenaltyWorkspaceDevice;
    TensorPtr mContextLengthDevice;
    TensorPtr mSeqLengthsDevice;
    TensorPtr mFinishedDevice;
    TensorPtr mOutputIdsDevice;
    TensorPtr mEndIdsDevice;
    TensorPtr mIdsPtrHost;

    TensorPtr mEmbeddingBiasHost;
    TensorPtr mEmbeddingBiasDevice;

    TensorPtr mCumLogProbsDevice;

    const tensorrt_llm::common::DataType data_type = tensorrt_llm::common::getTensorType<T>();

    // Order is important because we pass mAllocator to mSamplingLayer and it is used in destructor
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::common::CudaAllocator> mAllocator;
    std::shared_ptr<tensorrt_llm::layers::BaseSamplingLayer<T>> mSamplingLayer;

    std::vector<T> mTestLogitsInit;

    void setup(uint64_t seed, SamplingParams const& params);

    typename tensorrt_llm::layers::BaseSamplingLayer<T>::ForwardParams createInputTensors(int32_t step);

    tensorrt_llm::layers::DecodingOutputParams createOutputTensors();

    void batchCopy(int32_t step);
    bool checkResult(int32_t* outputIds, std::vector<std::set<int32_t>>& expectedIds);

public:
    void runTest(std::vector<std::set<int32_t>> expectedOutputIds, SamplingParams const& params, int32_t endId = -1);
};

typedef testing::Types<float, half> FloatAndHalfTypes;

} // namespace tensorrt_llm::tests::layers::sampling

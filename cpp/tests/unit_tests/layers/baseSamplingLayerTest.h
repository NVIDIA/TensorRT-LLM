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

#include "tensorrt_llm/layers/beamSearchLayer.h"
#include "tensorrt_llm/layers/externalDraftTokensLayer.h"
#include "tensorrt_llm/layers/samplingLayer.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include "tensorrt_llm/common/tllmException.h"

namespace tensorrt_llm::tests::layers::sampling
{

constexpr float EPSILON = 1e-20f;

template <typename T>
void computeProb(T* probs, T const* logits, int batchSize, int vocabSize)
{
    // Compute the log probability from logits.
    //   logits = batchSize x vocabSize.
    //   probs =  softmax(logits) (softmax along with vocab dimension)
    // float is used for either T=float or half, since operations of half are
    // not fully supported in a host function.
    for (int bidx = 0; bidx < batchSize; ++bidx)
    {
        float maxval = -FLT_MAX;
        for (int i = 0; i < vocabSize; ++i)
        {
            float logit = static_cast<float>(logits[bidx * vocabSize + i]);
            if (logit > maxval)
            {
                maxval = logit;
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < vocabSize; ++i)
        {
            sum += expf(static_cast<float>(logits[bidx * vocabSize + i]) - maxval);
        }
        for (int i = 0; i < vocabSize; ++i)
        {
            int idx = bidx * vocabSize + i;
            float logit = static_cast<float>(logits[idx]) - maxval;
            probs[idx] = static_cast<T>(expf(logit) / (sum + EPSILON));
        }
    }
}

struct TestSamplingParams
{
    std::vector<runtime::SizeType32> topKs;
    std::vector<float> topPs;
    std::vector<float> temperatures;
    std::vector<float> repetitionPenalties;
    std::vector<float> presencePenalties;
    std::vector<float> frequencyPenalties;
    std::vector<int32_t> minLengths;
    std::vector<float> decay;
    std::vector<float> minTopP;
    std::vector<int32_t> topPResetIds;
    int32_t batchSize = 6;
    int32_t beamWidth = 1;
    bool useBias = false;
    bool isExternalDraftTokensLayerTest = false;
    bool useDraftLogits = false;
    bool isAirTopPExternalDraftTokensLayer = false;
};

template <typename T>
class BaseSamplingLayerTest : public testing::Test
{
protected:
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;
    using BufferPtr = tensorrt_llm::runtime::IBuffer::SharedPtr;

    static int32_t constexpr kDoubleBatchIdx = 2;

    int32_t seed = 0;
    int32_t mBatchSize = -1; // setup by runTest
    int32_t mBeamWidth = 1;
    static int32_t constexpr mBatchSizeBadPad = 512;
    uint64_t mMaxSeed = 32;
    int32_t const mVocabSize = 8;
    int32_t const mVocabSizePadded = mVocabSize;

    int32_t const mMaxInputLen = 0; // has no effect.
    static int32_t constexpr mMaxOutputLen = 4;
    int32_t const mMaxSeqLen = mMaxInputLen + mMaxOutputLen;
    int32_t const mMaxTokensPerEngineStep = mMaxOutputLen;

    int32_t mEndId = mVocabSize;

    bool mComputeProbs = false;

    TensorPtr mContextLengthDevice;
    TensorPtr mSeqLengthsDevice;
    TensorPtr mFinishedDevice;
    TensorPtr mOutputIdsDevice;
    TensorPtr mEndIdsDevice;
    TensorPtr mIdsPtrHost;
    TensorPtr mBatchSlots;

    TensorPtr mEmbeddingBiasHost;
    TensorPtr mEmbeddingBiasDevice;

    TensorPtr mCumLogProbsDevice;
    TensorPtr mOutputLogProbsDevice;

    TensorPtr mCurandStatesDevice;
    TensorPtr mPenaltyWorkspaceDevice;

    // For Beam Search
    TensorPtr mSrcCacheIndirection;
    TensorPtr mTgtCacheIndirection;
    TensorPtr mParentIds;
    TensorPtr mOutputIdsCBA;
    TensorPtr mLogProbsCBA;
    TensorPtr mSequenceLengthsCBA;
    TensorPtr mCumLogProbsCBA;
    TensorPtr mNormedScoresCBA;
    TensorPtr mNumBeamsCBA;
    TensorPtr mMinNormedScoresCBA;
    TensorPtr mBatchDones;
    TensorPtr mOutputIdsPtr;
    TensorPtr mParentIdsPtr;

    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::layers::BaseLayer> mSamplingLayer;
    std::shared_ptr<tensorrt_llm::runtime::DecodingLayerWorkspace> mDecodingWorkspace;

    std::vector<T> mTestLogitsInit;

    int32_t maxBatchSize() const
    {
        return kDoubleBatchIdx * mBatchSize;
    }

    int32_t batchBeam() const
    {
        return mBatchSize * mBeamWidth;
    }

    void setup(uint64_t seed, TestSamplingParams const& params);

    virtual void initLayer(TestSamplingParams const& params) = 0;

    virtual std::shared_ptr<tensorrt_llm::layers::DecodingInputs> createInputTensors(int32_t step);

    std::shared_ptr<tensorrt_llm::layers::BaseDecodingOutputs> createOutputTensors();

    void batchCopy(int32_t step);
    bool checkResult(int32_t const* outputIds, std::vector<std::set<int32_t>> const& expectedIds);

public:
    void runTest(
        std::vector<std::set<int32_t>> const& expectedOutputIds, TestSamplingParams const& params, int32_t endId = -1);
};

typedef testing::Types<float, half> FloatAndHalfTypes;

} // namespace tensorrt_llm::tests::layers::sampling

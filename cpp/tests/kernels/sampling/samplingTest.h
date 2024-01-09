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
#include <random>

#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

namespace tensorrt_llm::tests::kernels::sampling
{

typedef testing::Types<float, half> FloatAndHalfTypes;

constexpr float EPSILON = 1e-20f;

inline bool almostEqual(float a, float b, float atol = 1e-5, float rtol = 1e-8)
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

template <typename T>
void initRandom(T* ptr, size_t size, float minval, float maxval)
{
    for (size_t i = 0; i < size; ++i)
    {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        val *= (maxval - minval);
        ptr[i] = static_cast<T>(minval + val);
    }
}

inline void initRandomInt(int* ptr, size_t size, int minval, int maxval)
{
    assert(minval < maxval);
    int mod = maxval - minval;
    for (size_t i = 0; i < size; ++i)
    {
        ptr[i] = minval + rand() % mod;
    }
}

template <typename T>
bool checkResult(std::string name, T* out, T* ref, size_t size)
{
    bool isFp32 = sizeof(T) == 4;
    float atol = isFp32 ? 1e-4f : 1e-3f;
    float rtol = isFp32 ? 1e-2f : 1e-1f;

    size_t failures = 0;
    float relativeGap = 0.0f;
    ;

    for (size_t i = 0; i < size; ++i)
    {
        // The values for the output and the reference.
        float a = (float) out[i];
        float b = (float) ref[i];

        bool ok = almostEqual(a, b, atol, rtol);
        // Print the error.
        if (!ok && failures < 4)
        {
            TLLM_LOG_ERROR(">> invalid result for i=%lu:", i);
            TLLM_LOG_ERROR(">>    found......: %10.6f", a);
            TLLM_LOG_ERROR(">>    expected...: %10.6f", b);
            TLLM_LOG_ERROR(">>    error......: %.6f", fabsf(a - b));
            TLLM_LOG_ERROR(">>    tol........: %.6f", atol + rtol * fabs(b));
        }
        // Update the number of failures.
        failures += ok ? 0 : 1;
        // Update the relative gap.
        relativeGap += fabsf(a - b) / (fabsf(b) + EPSILON);
    }

    relativeGap /= size;

    // Allow not matched up to 1% elements.
    size_t tolFailures = (size_t) (0.0 * size);
    TLLM_LOG_DEBUG("check...%6s : %-50s (failures: %.2f%% atol: %.2e rtol: %.2e rel_gap: %.2e%%)",
        failures <= tolFailures ? "....OK" : "FAILED", name.c_str(), 100. * failures / size, atol, rtol,
        100. * relativeGap);
    return failures <= tolFailures;
}

/////////////////////////////////// Tests //////////////////////////////////////////

template <typename T>
void computeProb(T* probs, const T* logits, int batchSize, int vocabSize)
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

template <typename T>
void computeLogProb(T* logprobs, const T* logits, int batchSize, int vocabSize)
{
    // Compute the log probability from logits.
    //   logits = batchSize x vocabSize.
    //   logprobs = log(softmax(logits)) (softmax along with vocab dimension)
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
            logprobs[idx] = static_cast<T>(logit - logf(sum + EPSILON));
        }
    }
}

struct SamplingKernelTestParam
{
    int32_t batchSize;
    int32_t vocabSize;
    uint32_t topK;
    float topP;
    int32_t outputLen;
    bool normalizeLogProbs = false;

    SamplingKernelTestParam& setBatchSize(int32_t bs)
    {
        batchSize = bs;
        return *this;
    }

    SamplingKernelTestParam& setVocabSize(int32_t vs)
    {
        vocabSize = vs;
        return *this;
    }

    SamplingKernelTestParam& setTopK(uint32_t tk)
    {
        topK = tk;
        return *this;
    }

    SamplingKernelTestParam& setTopP(float tp)
    {
        topP = tp;
        return *this;
    }

    SamplingKernelTestParam& setOutputLen(int32_t ol)
    {
        outputLen = ol;
        return *this;
    }

    std::string toString() const
    {
        return tensorrt_llm::common::fmtstr("SamplingKernelTestParam[batch=%d, vocab=%d, k=%u, p=%3.1f, output_len=%d]",
            batchSize, vocabSize, topK, topP, outputLen);
    }
};

template <typename T>
class SamplingKernelTest : public testing::Test
{
public:
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

    void SetUp() override;
    void TearDown() override;

    void runTest(const SamplingKernelTestParam& param);

protected:
    virtual size_t getWorkspaceSize(const SamplingKernelTestParam& param)
    {
        throw std::logic_error("Not implemented");
    };

    virtual void callTestedFunction(
        const SamplingKernelTestParam& param, bool hasDiffRuntimeArgs, size_t workspaceSize, TensorPtr& workspaceDevice)
    {
        throw std::logic_error("Not implemented");
    }

    void allocateBuffers(int32_t batchSize, int32_t vocabSize, int32_t maxSeqLen, int32_t outputLen);

    void setupBuffers(int32_t batchSize, int32_t vocabSize, int32_t maxSeqLen, int32_t outputLen, int32_t topK,
        float topP, bool useSkipDecode, bool hasDiffRuntimeArgs, std::mt19937& gen,
        std::uniform_int_distribution<>& endIdsDistr);

    void verifyCurrentStep(int32_t batchSize, int32_t vocabSize, int32_t maxSeqLen, int32_t step, bool greedySearch,
        bool useSkipDecode, bool hasDiffRuntimeArgs, std::vector<tensorrt_llm::kernels::FinishedState>& refFinished,
        std::vector<int32_t>& refSeqLength,
        const std::vector<tensorrt_llm::kernels::FinishedState>& finishedCurrentStep);

private:
    void runTest(const SamplingKernelTestParam& param, bool hasDiffRuntimeArgs, bool useSkipDecode);

protected:
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    uint32_t mSeed = 0;

    struct cudaDeviceProp mDeviceProp;

    TensorPtr mSeqLengthsHost;
    TensorPtr mSeqLengthsDevice;

    TensorPtr mFinishedHost;
    TensorPtr mFinishedDevice;

    TensorPtr mOutputIdsHost;
    TensorPtr mOutputIdsDevice;

    TensorPtr mProbsHost;
    TensorPtr mProbsDevice;

    TensorPtr mCumLogProbsDevice;
    TensorPtr mOutputLogProbsDevice;
    TensorPtr mTopPIdValsDevice;
    TensorPtr mZeroParentIdsDevice;
    TensorPtr mBeginOffsetsDevice;
    TensorPtr mEndOffsetsDevice;

    TensorPtr mLogitsHost;
    TensorPtr mLogProbsHost;
    TensorPtr mIdsPtrHost;

    TensorPtr mEndIdsHost;
    TensorPtr mEndIdsDevice;

    TensorPtr mTopKsHost;
    TensorPtr mTopKsDevice;

    TensorPtr mTopPsHost;
    TensorPtr mTopPsDevice;

    TensorPtr mSkipDecodeHost;
    TensorPtr mSkipDecodeDevice;

    TensorPtr mExpectedCumLogProbsHost;

    int32_t mMaxTopK;
    float mMaxTopP;

    curandState_t* mCurandStatesDevice;
};

} // namespace tensorrt_llm::tests::kernels::sampling

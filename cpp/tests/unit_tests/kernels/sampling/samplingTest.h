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
constexpr float HALF_FLT_MAX = 65504.f;

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

template <typename T>
void initConstant(T* ptr, size_t size, T val)
{
    for (size_t i = 0; i < size; ++i)
    {
        ptr[i] = val;
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

template <typename T>
void computeLogProb(T* logprobs, T const* logits, int batchSize, int vocabSize)
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
    uint32_t topK{1};
    float topP{0.f};
    bool normalizeLogProbs{false};
    bool logitsHasProbs{true};
    int32_t maxTokensPerStep{1};
    bool returnAllSelectedTokens{false};
    bool useLogitsPtrs{false};
    bool isDeterministicTopP{false};

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

    SamplingKernelTestParam& setMaxTokensPerStep(int32_t ts)
    {
        maxTokensPerStep = ts;
        return *this;
    }

    SamplingKernelTestParam& setReturnAllSelectedTokens()
    {
        returnAllSelectedTokens = true;
        return *this;
    }

    SamplingKernelTestParam& setUseLogitsPtrs()
    {
        useLogitsPtrs = true;
        return *this;
    }

    SamplingKernelTestParam& setDeterministicTopP(bool isDeter)
    {
        isDeterministicTopP = isDeter;
        return *this;
    }

    std::string toString() const
    {
        return tensorrt_llm::common::fmtstr(
            "SamplingKernelTestParam[batch=%d, vocab=%d, k=%u, p=%3.1f, tokens_per_step=%d]", batchSize, vocabSize,
            topK, topP, maxTokensPerStep);
    }
};

template <typename T>
class SamplingKernelTest : public testing::Test
{
public:
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

    void SetUp() override;
    void TearDown() override;

    void runTest(SamplingKernelTestParam const& param);

protected:
    virtual size_t getWorkspaceSize(SamplingKernelTestParam const& param)
    {
        throw std::logic_error("Not implemented");
    };

    virtual void callTestedFunction(SamplingKernelTestParam const& param, TensorPtr& workspaceDevice)
    {
        throw std::logic_error("Not implemented");
    }

private:
    void allocateBuffers(SamplingKernelTestParam const& param);

    void setupBuffers(SamplingKernelTestParam const& param);

    void verifyResult(SamplingKernelTestParam const& param);

    std::vector<int32_t> computeTopKTopPVariants(SamplingKernelTestParam const& param, int32_t bi, int32_t batchSlot,
        int32_t ti, int32_t tokensPerStep, int32_t vocabSize);

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
    TensorPtr mProbsPtrsDevice;

    TensorPtr mCumLogProbsDevice;
    TensorPtr mOutputLogProbsDevice;
    TensorPtr mZeroParentIdsDevice;

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

    TensorPtr mTokensPerStep;

    TensorPtr mBatchSlots;

    TensorPtr mExpectedCumLogProbsHost;

    TensorPtr mExpectedLogProbsHost;

    TensorPtr mCurandStatesDevice;

    int32_t mMaxTopK;
    static constexpr int32_t mMaxSeqLen = 2048;
    float mMaxTopP;
};

} // namespace tensorrt_llm::tests::kernels::sampling

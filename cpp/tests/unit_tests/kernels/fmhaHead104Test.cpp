/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/fmhaDispatcher.h"

namespace
{

namespace tk = tensorrt_llm::kernels;

struct TestCase
{
    char const* name;
    tk::Data_type inputType;
    tk::ContextAttentionMaskType mask;
    bool forceFixed = false;
    bool forceRuntime = false;
};

class FmhaHead104Test : public ::testing::TestWithParam<TestCase>
{
protected:
    static constexpr int kHeadDim = 104;
    static constexpr int kHeads = 2;
    static constexpr float kQScaling = 1.25F;

    void SetUp() override
    {
        int const sm = tensorrt_llm::common::getSMVersion();
        bool const isSm100 = tensorrt_llm::common::isSM100Family(sm);
        if (sm != 90 && sm != 120 && !isSm100)
        {
            GTEST_SKIP() << "Head-104 packed-QKV kernels require SM90, SM100-family, or SM120.";
        }
        TLLM_CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    void TearDown() override
    {
        if (mStream != nullptr)
        {
            cudaStreamSynchronize(mStream);
        }
        for (void* allocation : mAllocations)
        {
            cudaFree(allocation);
        }
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    template <typename T>
    T* allocate(size_t count)
    {
        T* pointer = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
        mAllocations.push_back(pointer);
        return pointer;
    }

    template <typename T>
    T* upload(std::vector<T> const& values)
    {
        auto* pointer = allocate<T>(values.size());
        TLLM_CUDA_CHECK(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
        return pointer;
    }

    static std::vector<float> reference(std::vector<float> const& qkv, std::vector<int> const& lengths, bool causal)
    {
        int const tokens = std::accumulate(lengths.begin(), lengths.end(), 0);
        int const hidden = kHeads * kHeadDim;
        std::vector<float> output(tokens * hidden);
        int offset = 0;
        for (int const length : lengths)
        {
            std::vector<float> scores(length);
            for (int q = 0; q < length; ++q)
            {
                int const keys = causal ? q + 1 : length;
                for (int head = 0; head < kHeads; ++head)
                {
                    for (int k = 0; k < keys; ++k)
                    {
                        float dot = 0.0F;
                        for (int d = 0; d < kHeadDim; ++d)
                        {
                            dot += qkv[(offset + q) * 3 * hidden + head * kHeadDim + d]
                                * qkv[(offset + k) * 3 * hidden + hidden + head * kHeadDim + d];
                        }
                        scores[k] = dot / (std::sqrt(float(kHeadDim)) * kQScaling);
                    }
                    float const maximum = *std::max_element(scores.begin(), scores.begin() + keys);
                    float sum = 0.0F;
                    for (int k = 0; k < keys; ++k)
                    {
                        scores[k] = std::exp(scores[k] - maximum);
                        sum += scores[k];
                    }
                    for (int d = 0; d < kHeadDim; ++d)
                    {
                        float value = 0.0F;
                        for (int k = 0; k < keys; ++k)
                        {
                            value
                                += scores[k] / sum * qkv[(offset + k) * 3 * hidden + 2 * hidden + head * kHeadDim + d];
                        }
                        output[(offset + q) * hidden + head * kHeadDim + d] = value;
                    }
                }
            }
            offset += length;
        }
        return output;
    }

    template <typename T>
    void run(std::vector<int> const& lengths)
    {
        TestCase const& test = GetParam();
        int const tokens = std::accumulate(lengths.begin(), lengths.end(), 0);
        int const maximum = *std::max_element(lengths.begin(), lengths.end());
        size_t const outputSize = tokens * kHeads * kHeadDim;
        std::vector<T> qkv(3 * outputSize);
        std::vector<float> roundedQkv(qkv.size());
        std::mt19937 generator(6665906);
        std::normal_distribution<float> normal;
        for (size_t i = 0; i < qkv.size(); ++i)
        {
            qkv[i] = T(normal(generator));
            roundedQkv[i] = static_cast<float>(qkv[i]);
        }
        std::vector<int> cumulative(lengths.size() + 1, 0);
        std::partial_sum(lengths.begin(), lengths.end(), cumulative.begin() + 1);

        tk::MHARunnerFixedParams fixed{};
        fixed.dataType = test.inputType;
        fixed.dataTypeKv = test.inputType;
        fixed.dataTypeOut = test.inputType;
        fixed.forceFp32Acc = test.forceFixed;
        fixed.attentionMaskType = test.mask;
        fixed.attentionInputLayout = tk::AttentionInputLayout::PACKED_QKV;
        fixed.numQHeads = kHeads;
        fixed.numKvHeads = kHeads;
        fixed.headSize = kHeadDim;
        fixed.headSizeV = kHeadDim;
        fixed.qScaling = kQScaling;
        tk::FmhaDispatcher dispatcher(fixed);
        ASSERT_TRUE(dispatcher.isSupported());

        auto* output = allocate<T>(outputSize);
        TLLM_CUDA_CHECK(cudaMemset(output, 0xff, outputSize * sizeof(T)));
        tk::MHARunnerParams params{};
        params.b = lengths.size();
        params.qSeqLen = maximum;
        params.kvSeqLen = maximum;
        params.totalQSeqLen = tokens;
        params.totalKvSeqLen = tokens;
        params.slidingWindowSize = std::numeric_limits<int>::max();
        params.qkvPtr = upload(qkv);
        params.outputPtr = output;
        params.cuQSeqLenPtr = upload(cumulative);
        params.cuKvSeqLenPtr = params.cuQSeqLenPtr;
        params.kvSeqLenPtr = upload(lengths);
        params.tileCounterPtr = upload<int>({0});
        params.forceFp32Acc = test.forceRuntime;
        params.stream = mStream;
        // Direct dispatch must launch a fused kernel; there is no unfused implementation in this test.
        dispatcher.run(params);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));

        std::vector<T> actual(outputSize);
        TLLM_CUDA_CHECK(cudaMemcpy(actual.data(), output, outputSize * sizeof(T), cudaMemcpyDeviceToHost));
        auto const expected = reference(roundedQkv, lengths, test.mask == tk::ContextAttentionMaskType::CAUSAL);
        double squaredError = 0.0;
        double squaredReference = 0.0;
        float maxError = 0.0F;
        float referencePeak = 0.0F;
        for (size_t i = 0; i < outputSize; ++i)
        {
            float const value = static_cast<float>(actual[i]);
            ASSERT_TRUE(std::isfinite(value)) << "Unwritten or nonfinite output at " << i;
            float const error = value - expected[i];
            squaredError += double(error) * error;
            squaredReference += double(expected[i]) * expected[i];
            maxError = std::max(maxError, std::abs(error));
            referencePeak = std::max(referencePeak, std::abs(expected[i]));
        }
        bool const bf16 = test.inputType == tk::DATA_TYPE_BF16;
        float const relativeRmseLimit = bf16 ? 0.015F : 0.005F;
        float const peakErrorLimit = bf16 ? 0.025F : 0.01F;
        EXPECT_LE(std::sqrt(squaredError / std::max(squaredReference, 1e-16)), relativeRmseLimit);
        EXPECT_LE(maxError, std::max(0.01F, referencePeak * peakErrorLimit));
    }

    cudaStream_t mStream = nullptr;
    std::vector<void*> mAllocations;
};

TEST_P(FmhaHead104Test, PackedRaggedNumerics)
{
    for (auto const& lengths : {std::vector<int>{31, 7}, std::vector<int>{257, 65}})
    {
        SCOPED_TRACE("Maximum sequence length: " + std::to_string(lengths.front()));
        switch (GetParam().inputType)
        {
        case tk::DATA_TYPE_FP16: run<half>(lengths); break;
        case tk::DATA_TYPE_BF16: run<__nv_bfloat16>(lengths); break;
        default: FAIL() << "Unexpected input type";
        }
        ASSERT_FALSE(HasFatalFailure());
    }
}

INSTANTIATE_TEST_SUITE_P(Head104, FmhaHead104Test,
    ::testing::Values(TestCase{"Fp16Full", tk::DATA_TYPE_FP16, tk::ContextAttentionMaskType::PADDING},
        TestCase{"Fp16Causal", tk::DATA_TYPE_FP16, tk::ContextAttentionMaskType::CAUSAL},
        TestCase{"Bf16Full", tk::DATA_TYPE_BF16, tk::ContextAttentionMaskType::PADDING},
        TestCase{"Bf16Causal", tk::DATA_TYPE_BF16, tk::ContextAttentionMaskType::CAUSAL},
        TestCase{"Fp16FixedFp32Full", tk::DATA_TYPE_FP16, tk::ContextAttentionMaskType::PADDING, true},
        TestCase{"Fp16RuntimeFp32Full", tk::DATA_TYPE_FP16, tk::ContextAttentionMaskType::PADDING, false, true}),
    [](auto const& info) { return info.param.name; });

} // namespace

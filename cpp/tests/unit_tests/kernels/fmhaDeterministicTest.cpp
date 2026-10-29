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
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"

namespace
{

using tensorrt_llm::kernels::DATA_TYPE_BF16;
using tensorrt_llm::kernels::DATA_TYPE_E4M3;
using tensorrt_llm::kernels::FmhaKernelType;
using tensorrt_llm::kernels::QkvLayout;
using tensorrt_llm::kernels::SparseType;
using tensorrt_llm::kernels::TileScheduler;
using tensorrt_llm::kernels::TllmGenFmhaRunner;
using tensorrt_llm::kernels::TllmGenFmhaRunnerParams;
using tensorrt_llm::kernels::TrtllmGenAttentionMaskType;

class FmhaDeterministicTest : public ::testing::Test
{
protected:
    static constexpr int32_t kHeadDim = 512;
    static constexpr int32_t kNumHeads = 32;
    static constexpr int32_t kSparseTopK = 640;
    static constexpr int32_t kCaptureKvLength = 9216;

    void SetUp() override
    {
        if (!tensorrt_llm::common::isSM100Family(tensorrt_llm::common::getSMVersion()))
        {
            GTEST_SKIP() << "TRTLLM-Gen sparse MLA requires Blackwell or newer.";
        }
        TLLM_CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    void TearDown() override
    {
        if (mStream != nullptr)
        {
            cudaStreamSynchronize(mStream);
        }
        if (mGraphExec != nullptr)
        {
            cudaGraphExecDestroy(mGraphExec);
        }
        if (mGraph != nullptr)
        {
            cudaGraphDestroy(mGraph);
        }
        for (void* ptr : mAllocations)
        {
            cudaFree(ptr);
        }
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    template <typename T>
    T* allocate(size_t count)
    {
        T* ptr = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
        mAllocations.push_back(ptr);
        return ptr;
    }

    template <typename T>
    void copyToDevice(T* dst, std::vector<T> const& src)
    {
        TLLM_CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    std::vector<uint16_t> readOutput(uint16_t const* output)
    {
        std::vector<uint16_t> result(kNumHeads * kHeadDim);
        TLLM_CUDA_CHECK(
            cudaMemcpyAsync(result.data(), output, result.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, mStream));
        TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
        return result;
    }

    TllmGenFmhaRunnerParams makeParams(int32_t numHeads = kNumHeads)
    {
        TllmGenFmhaRunnerParams params{};
        params.mQkvLayout = QkvLayout::PagedKv;
        params.mMaskType = TrtllmGenAttentionMaskType::Dense;
        params.mKernelType = FmhaKernelType::Generation;
        params.mTileScheduler = TileScheduler::Persistent;
        params.mMultiCtasKvMode = false;
        params.mHeadDimQk = kHeadDim;
        params.mHeadDimV = kHeadDim;
        params.mNumHeadsQ = numHeads;
        params.mNumHeadsKv = 1;
        params.mNumHeadsQPerKv = numHeads;
        params.mBatchSize = 1;
        params.mMaxSeqLenCacheKv = kCaptureKvLength;
        params.mMaxSeqLenQ = 1;
        params.mMaxSeqLenKv = kCaptureKvLength;
        // Negative reaches are unbounded, which is what a full-attention decode wants.
        params.mLeftSlidingWindow = -1;
        params.mRightSlidingWindow = -1;
        params.mChunkedAttentionSize = std::numeric_limits<int32_t>::max();
        params.mSumOfSeqLensQ = 1;
        params.mSumOfSeqLensKv = kCaptureKvLength;
        params.mMaxNumPagesPerSeqKv = kSparseTopK;
        params.mNumTokensPerPage = 1;
        params.mNumPagesInMemPool = kSparseTopK;
        params.mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
        params.mScaleQ = 1.0F;
        params.mSparseAttention = SparseType::DynamicTokenSparse;
        params.mSparseTopK = kSparseTopK;
        params.stream = mStream;
        return params;
    }

    cudaStream_t mStream{};
    cudaGraph_t mGraph{};
    cudaGraphExec_t mGraphExec{};
    std::vector<void*> mAllocations;
};

TEST_F(FmhaDeterministicTest, LongSparseKvKeepsOneReductionPartition)
{
    TllmGenFmhaRunner runner(DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 0, 0, 0, 0, false);
    for (int32_t const numHeads : {32, 64})
    {
        for (int32_t const kvLength : {129, 640, kCaptureKvLength})
        {
            auto params = makeParams(numHeads);
            params.mMaxSeqLenKv = kvLength;
            auto const [supported, info] = runner.isSupportedWithInfo(params);
            ASSERT_TRUE(supported) << info;
            EXPECT_NE(info.find("multiCtasKvMode=0"), std::string::npos)
                << "heads=" << numHeads << ", kvLength=" << kvLength << ": " << info;
        }
    }
}

TEST_F(FmhaDeterministicTest, CudaGraphMatchesEagerForChangingSparseLengths)
{
    auto params = makeParams();
    auto* q = allocate<uint8_t>(kNumHeads * kHeadDim);
    auto* kv = allocate<uint8_t>(kSparseTopK * kHeadDim);
    auto* output = allocate<uint16_t>(kNumHeads * kHeadDim);
    auto* sequenceLength = allocate<int32_t>(1);
    auto* topkLength = allocate<int32_t>(1);
    auto* indices = allocate<int32_t>(kSparseTopK);
    auto* scale = allocate<float>(1);
    auto* softmaxScale = allocate<float>(1);
    std::vector<uint8_t> qValues(kNumHeads * kHeadDim);
    std::vector<uint8_t> kvValues(kSparseTopK * kHeadDim);
    for (size_t i = 0; i < qValues.size(); ++i)
    {
        qValues[i] = static_cast<uint8_t>(24 + (i * 7 % 24) + ((i / 3 % 2) << 7));
    }
    for (size_t i = 0; i < kvValues.size(); ++i)
    {
        kvValues[i] = static_cast<uint8_t>(24 + (i * 11 % 24) + ((i / 5 % 2) << 7));
    }
    std::vector<int32_t> indexValues(kSparseTopK);
    for (int32_t i = 0; i < kSparseTopK; ++i)
    {
        indexValues[i] = i;
    }
    copyToDevice(q, qValues);
    copyToDevice(kv, kvValues);
    copyToDevice(indices, indexValues);
    copyToDevice(scale, std::vector<float>{1.0F});
    copyToDevice(softmaxScale, std::vector<float>{static_cast<float>(std::log2(std::exp(1.0)) / std::sqrt(kHeadDim))});
    copyToDevice(sequenceLength, std::vector<int32_t>{kCaptureKvLength});
    copyToDevice(topkLength, std::vector<int32_t>{kSparseTopK});
    params.qPtr = q;
    params.kvPtr = kv;
    params.slidingWindowKvPoolBasePtr = kv;
    params.oPtr = output;
    params.seqLensKvPtr = sequenceLength;
    params.kvPageIdxPtr = indices;
    params.ptrSparseMlaTopKLens = topkLength;
    params.outputScalePtr = scale;
    params.scaleSoftmaxLog2Ptr = softmaxScale;
    params.kvSfScalePtr = scale;
    params.oSfScalePtr = scale;
    TllmGenFmhaRunner runner(DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 0, 0, 0, 0, false);
    runner.run(params);
    TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
    TLLM_CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal));
    runner.run(params);
    TLLM_CUDA_CHECK(cudaStreamEndCapture(mStream, &mGraph));
    TLLM_CUDA_CHECK(cudaGraphInstantiate(&mGraphExec, mGraph, nullptr, nullptr, 0));

    for (int32_t const kvLength : {127, 128, 129, 160, 513, 2049, 8193})
    {
        SCOPED_TRACE(kvLength);
        int32_t const selected = std::min(kvLength, 128) + std::min(kvLength / 4, 512);
        copyToDevice(sequenceLength, std::vector<int32_t>{kvLength});
        copyToDevice(topkLength, std::vector<int32_t>{selected});
        params.mMaxSeqLenKv = kvLength;
        params.mSumOfSeqLensKv = kvLength;
        runner.run(params);
        auto const eager = readOutput(output);
        ASSERT_TRUE(
            std::all_of(eager.begin(), eager.end(), [](uint16_t value) { return (value & 0x7f80U) != 0x7f80U; }));
        TLLM_CUDA_CHECK(cudaGraphLaunch(mGraphExec, mStream));
        auto const graph = readOutput(output);
        EXPECT_EQ(eager, graph);
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (setenv("FORCE_ATTENTION_KERNEL_DETERMINISTIC", "1", 1) != 0)
    {
        return EXIT_FAILURE;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

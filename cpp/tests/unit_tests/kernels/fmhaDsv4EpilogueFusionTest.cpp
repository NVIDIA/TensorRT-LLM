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
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"

namespace
{

using tensorrt_llm::kernels::FmhaKernelType;
using tensorrt_llm::kernels::QkvLayout;
using tensorrt_llm::kernels::SparseType;
using tensorrt_llm::kernels::TileScheduler;
using tensorrt_llm::kernels::TrtllmGenAttentionMaskType;
using tensorrt_llm::kernels::TllmGenFmhaRunner;
using tensorrt_llm::kernels::TllmGenFmhaRunnerParams;
using tensorrt_llm::kernels::DATA_TYPE_BF16;
using tensorrt_llm::kernels::DATA_TYPE_E4M3;

class FmhaDsv4EpilogueFusionTest : public ::testing::Test
{
protected:
    static constexpr int32_t kNumTokens = 5;
    static constexpr int32_t kScaleBufM = 8;
    static constexpr int32_t kNumHeads = 128;
    static constexpr int32_t kNumKvHeads = 1;
    static constexpr int32_t kHeadDim = 512;
    static constexpr int32_t kRopeStart = 448;
    static constexpr int32_t kRopeHalf = 32;
    static constexpr int32_t kNumGroups = 16;
    static constexpr int32_t kHeadsPerGroup = 8;
    static constexpr int32_t kNumScaleBlocksPerHead = kHeadDim / 128;
    static constexpr int32_t kSparseTopK = 4;
    static constexpr int32_t kScaleGuardElts = 256;
    static constexpr uint32_t kSentinelBits = 0x7fc00000U;
    static constexpr float kFp8Max = 448.0F;
    static constexpr float kScaleRtol = 0.01F;
    static constexpr float kValueAtol = 1.0e-3F;
    static constexpr float kValueRtol = 0.01F;

    void SetUp() override
    {
        if (tensorrt_llm::common::getSMVersion() != 107)
        {
            GTEST_SKIP() << "DSv4 fused FMHA epilogue cubins are SM107-specific.";
        }
        TLLM_CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    void TearDown() override
    {
        if (mStream != nullptr)
        {
            cudaStreamSynchronize(mStream);
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

    template <typename T>
    std::vector<T> copyToHost(T const* src, size_t count)
    {
        std::vector<T> dst(count);
        TLLM_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost, mStream));
        TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
        return dst;
    }

    static float bf16ToFloat(uint16_t value)
    {
        uint32_t bits = static_cast<uint32_t>(value) << 16;
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    static float e4m3ToFloat(uint8_t value)
    {
        int const sign = (value & 0x80U) != 0U ? -1 : 1;
        int const exponent = (value >> 3) & 0x0f;
        int const mantissa = value & 0x07;
        if (exponent == 0)
        {
            return sign * std::ldexp(static_cast<float>(mantissa) / 8.0F, -6);
        }
        if (exponent == 0x0f && mantissa == 0x07)
        {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
    }

    static uint32_t floatBits(float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    static float inverseRopeReference(std::vector<uint16_t> const& bf16Output, size_t headBase, int32_t dim)
    {
        float const value = bf16ToFloat(bf16Output[headBase + dim]);
        if (dim < kRopeStart)
        {
            return value;
        }

        int32_t const pairStart = dim & ~1;
        float const first = bf16ToFloat(bf16Output[headBase + pairStart]);
        float const second = bf16ToFloat(bf16Output[headBase + pairStart + 1]);
        return (dim & 1) == 0 ? first * 0.6F + second * 0.8F : second * 0.6F - first * 0.8F;
    }

    TllmGenFmhaRunnerParams makeParams()
    {
        TllmGenFmhaRunnerParams params{};
        params.mQkvLayout = QkvLayout::PagedKv;
        params.mMaskType = TrtllmGenAttentionMaskType::Causal;
        params.mKernelType = FmhaKernelType::Generation;
        params.mTileScheduler = TileScheduler::Persistent;
        params.mMultiCtasKvMode = false;
        params.mUseBlockSparseAttention = false;
        params.qPtr = mQ;
        params.kvPtr = mKv;
        params.slidingWindowKvPoolBasePtr = mSlidingWindowKv;
        params.oPtr = mBf16Output;
        params.softmaxStatsPtr = mSoftmaxStats;
        params.seqLensQPtr = mSeqLens;
        params.seqLensKvPtr = mSeqLens;
        params.cumSeqLensQPtr = mCumSeqLens;
        params.cumSeqLensKvPtr = mCumSeqLens;
        params.kvPageIdxPtr = mSparseIndices;
        params.ptrSparseMlaTopKLens = mSparseTopKLens;
        params.outputScalePtr = mUnitScale;
        params.scaleSoftmaxLog2Ptr = mSoftmaxScaleLog2;
        params.kvSfScalePtr = mUnitScale;
        params.oSfScalePtr = mUnitScale;
        params.mHeadDimQk = kHeadDim;
        params.mHeadDimV = kHeadDim;
        params.mHeadDimQkNope = kRopeStart;
        params.mNumHeadsQ = kNumHeads;
        params.mNumHeadsKv = kNumKvHeads;
        params.mNumHeadsQPerKv = kNumHeads / kNumKvHeads;
        params.mBatchSize = kNumTokens;
        params.mMaxSeqLenCacheKv = 1;
        params.mMaxSeqLenQ = 1;
        params.mMaxSeqLenKv = 1;
        params.mAttentionWindowSize = std::numeric_limits<int32_t>::max();
        params.mChunkedAttentionSize = std::numeric_limits<int32_t>::max();
        params.mSumOfSeqLensQ = kNumTokens;
        params.mSumOfSeqLensKv = kNumTokens;
        params.mMaxNumPagesPerSeqKv = 1;
        params.mNumTokensPerPage = 1;
        params.mNumPagesInMemPool = kNumTokens;
        params.mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
        params.mScaleQ = 1.0F;
        params.mSparseAttention = SparseType::DynamicTokenSparse;
        params.mSparseTopK = kSparseTopK;
        params.stream = mStream;
        return params;
    }

    void allocateAndInitializeBuffers()
    {
        size_t constexpr qElts = static_cast<size_t>(kNumTokens) * kNumHeads * kHeadDim;
        size_t constexpr kvElts = static_cast<size_t>(kNumTokens) * kHeadDim;
        size_t constexpr scaleElts
            = static_cast<size_t>(kNumGroups) * kHeadsPerGroup * kNumScaleBlocksPerHead * kScaleBufM;

        mQ = allocate<uint8_t>(qElts);
        mKv = allocate<uint8_t>(kvElts);
        mSlidingWindowKv = allocate<uint8_t>(kvElts);
        mBf16Output = allocate<uint16_t>(qElts);
        mFp8Output = allocate<uint8_t>(qElts);
        mOutputScale = allocate<float>(scaleElts + kScaleGuardElts);
        mCosSin = allocate<float>(kScaleBufM * kRopeHalf * 2);
        mSoftmaxStats = allocate<float2>(static_cast<size_t>(kNumTokens) * kNumHeads);
        mSeqLens = allocate<int32_t>(kNumTokens);
        mCumSeqLens = allocate<int32_t>(kNumTokens + 1);
        mSparseIndices = allocate<int32_t>(kNumTokens * kSparseTopK);
        mSparseTopKLens = allocate<int32_t>(kNumTokens);
        mUnitScale = allocate<float>(1);
        mSoftmaxScaleLog2 = allocate<float>(1);

        TLLM_CUDA_CHECK(cudaMemsetAsync(mQ, 0, qElts, mStream));
        // Use a distinct exact E4M3 value for every (token, 128-element block). With one selected KV token,
        // this makes each token/block expected scale distinct and catches token-stride or block-index remapping bugs.
        std::vector<uint8_t> kv(kvElts);
        for (int32_t token = 0; token < kNumTokens; ++token)
        {
            for (int32_t block = 0; block < kNumScaleBlocksPerHead; ++block)
            {
                uint8_t const value = static_cast<uint8_t>(((4 + token) << 3) | block);
                std::fill_n(
                    kv.data() + (static_cast<size_t>(token) * kNumScaleBlocksPerHead + block) * 128, 128, value);
            }
        }
        copyToDevice(mKv, kv);
        copyToDevice(mSlidingWindowKv, kv);
        TLLM_CUDA_CHECK(cudaMemsetAsync(mBf16Output, 0xff, qElts * sizeof(uint16_t), mStream));
        TLLM_CUDA_CHECK(cudaMemsetAsync(mFp8Output, 0xff, qElts, mStream));

        float sentinel;
        std::memcpy(&sentinel, &kSentinelBits, sizeof(sentinel));
        copyToDevice(mOutputScale, std::vector<float>(scaleElts + kScaleGuardElts, sentinel));

        std::vector<float> cosSin(kScaleBufM * kRopeHalf * 2);
        for (int32_t pos = 0; pos < kScaleBufM; ++pos)
        {
            std::fill_n(cosSin.data() + pos * kRopeHalf * 2, kRopeHalf, 0.6F);
            std::fill_n(cosSin.data() + pos * kRopeHalf * 2 + kRopeHalf, kRopeHalf, 0.8F);
        }
        copyToDevice(mCosSin, cosSin);

        std::vector<int32_t> seqLens(kNumTokens, 1);
        std::vector<int32_t> cumSeqLens(kNumTokens + 1);
        std::vector<int32_t> sparseIndices(kNumTokens * kSparseTopK);
        for (int32_t token = 0; token < kNumTokens; ++token)
        {
            cumSeqLens[token] = token;
            std::fill_n(sparseIndices.data() + token * kSparseTopK, kSparseTopK, token);
        }
        cumSeqLens[kNumTokens] = kNumTokens;
        copyToDevice(mSeqLens, seqLens);
        copyToDevice(mCumSeqLens, cumSeqLens);
        copyToDevice(mSparseIndices, sparseIndices);
        copyToDevice(mSparseTopKLens, seqLens);
        copyToDevice(mUnitScale, std::vector<float>{1.0F});
        copyToDevice(
            mSoftmaxScaleLog2, std::vector<float>{static_cast<float>(1.4426950408889634 / std::sqrt(kHeadDim))});
        TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
    }

    cudaStream_t mStream{};
    std::vector<void*> mAllocations;
    uint8_t* mQ{};
    uint8_t* mKv{};
    uint8_t* mSlidingWindowKv{};
    uint16_t* mBf16Output{};
    uint8_t* mFp8Output{};
    float* mOutputScale{};
    float* mCosSin{};
    float2* mSoftmaxStats{};
    int32_t* mSeqLens{};
    int32_t* mCumSeqLens{};
    int32_t* mSparseIndices{};
    int32_t* mSparseTopKLens{};
    float* mUnitScale{};
    float* mSoftmaxScaleLog2{};
};

TEST_F(FmhaDsv4EpilogueFusionTest, SelectsAndRunsFusedAndNonFusedCubins)
{
    allocateAndInitializeBuffers();
    auto params = makeParams();

    TllmGenFmhaRunner nonFusedRunner(DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 0, 0, 0, 0, false);
    auto const [nonFusedSupported, nonFusedInfo] = nonFusedRunner.isSupportedWithInfo(params);
    ASSERT_TRUE(nonFusedSupported) << nonFusedInfo;
    nonFusedRunner.run(params);
    TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));

    TllmGenFmhaRunner fusedRunner(DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 0, 0, 0, 0, true);

    // The E4M3-output dynamic-sparse tactic only exists with RopeQuant enabled. Looking it up through the
    // non-fused namespace must fail rather than accidentally returning the fused cubin with the same 64-bit tactic
    // hash.
    auto const [plainE4m3Supported, plainE4m3Info] = fusedRunner.isSupportedWithInfo(params);
    EXPECT_FALSE(plainE4m3Supported) << plainE4m3Info;

    params.oPtr = mFp8Output;
    params.oSfPtr = mOutputScale;
    params.mDsv4EpilogueFusion.enabled = true;
    params.mDsv4EpilogueFusion.cosSinCache = mCosSin;
    params.mDsv4EpilogueFusion.scaleBufM = kScaleBufM;
    auto const [fusedSupported, fusedInfo] = fusedRunner.isSupportedWithInfo(params);
    ASSERT_TRUE(fusedSupported) << fusedInfo;
    fusedRunner.run(params);
    TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));

    size_t constexpr outputElts = static_cast<size_t>(kNumTokens) * kNumHeads * kHeadDim;
    size_t constexpr scaleElts = static_cast<size_t>(kNumGroups) * kHeadsPerGroup * kNumScaleBlocksPerHead * kScaleBufM;
    auto const bf16Output = copyToHost(mBf16Output, outputElts);
    auto const fp8Output = copyToHost(mFp8Output, outputElts);
    auto const outputScale = copyToHost(mOutputScale, scaleElts + kScaleGuardElts);

    for (int32_t group = 0; group < kNumGroups; ++group)
    {
        for (int32_t token = 0; token < kNumTokens; ++token)
        {
            for (int32_t headInGroup = 0; headInGroup < kHeadsPerGroup; ++headInGroup)
            {
                int32_t const head = group * kHeadsPerGroup + headInGroup;
                size_t const bf16HeadBase = (static_cast<size_t>(token) * kNumHeads + head) * kHeadDim;
                size_t const fp8HeadBase
                    = ((static_cast<size_t>(group) * kNumTokens + token) * kHeadsPerGroup + headInGroup) * kHeadDim;
                for (int32_t block = 0; block < kNumScaleBlocksPerHead; ++block)
                {
                    int32_t const blockStart = block * 128;
                    float expectedAmax = 0.0F;
                    for (int32_t dim = blockStart; dim < blockStart + 128; ++dim)
                    {
                        expectedAmax
                            = std::max(expectedAmax, std::fabs(inverseRopeReference(bf16Output, bf16HeadBase, dim)));
                    }
                    float const expectedScale = expectedAmax / kFp8Max;
                    size_t const scaleBlock = static_cast<size_t>(headInGroup) * kNumScaleBlocksPerHead + block;
                    size_t const scaleIndex
                        = (static_cast<size_t>(group) * kHeadsPerGroup * kNumScaleBlocksPerHead + scaleBlock)
                            * kScaleBufM
                        + token;
                    float const scale = outputScale[scaleIndex];
                    ASSERT_NEAR(scale, expectedScale, expectedScale * kScaleRtol)
                        << "group=" << group << ", token=" << token << ", headInGroup=" << headInGroup
                        << ", block=" << block << ", scaleIndex=" << scaleIndex;

                    for (int32_t dim = blockStart; dim < blockStart + 128; ++dim)
                    {
                        float const reference = inverseRopeReference(bf16Output, bf16HeadBase, dim);
                        float const dequantized = e4m3ToFloat(fp8Output[fp8HeadBase + dim]) * scale;
                        float const tolerance = kValueAtol + kValueRtol * std::fabs(reference);
                        EXPECT_NEAR(dequantized, reference, tolerance)
                            << "group=" << group << ", token=" << token << ", headInGroup=" << headInGroup
                            << ", block=" << block << ", dim=" << dim;
                    }
                }
            }
        }
    }

    // All logical scale entries must be written using the padded physical token stride. The guard catches
    // an oversized stride while untouched expected entries catch a missing or compact (numTokens) stride.
    for (size_t index = scaleElts; index < scaleElts + kScaleGuardElts; ++index)
    {
        EXPECT_EQ(floatBits(outputScale[index]), kSentinelBits) << "guard index=" << index;
    }
}

} // namespace

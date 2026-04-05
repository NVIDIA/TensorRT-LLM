/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include <chrono>
#include <cmath>
#include <memory> //@todo check the usage of this
#include <random> //@todo check the usage of this

namespace tensorrt_llm::tests::kernels::routing
{

typedef testing::Types<float, __nv_bfloat16> FloatAndBf16Types;
typedef testing::Types<__nv_bfloat16> Bf16Types;

using RoutingMethodType = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::RoutingMethodType;
using RoutingPreprocessType = moe::dev::routing::RoutingPreprocessType;
using RoutingPostprocessType = moe::dev::routing::RoutingPostprocessType;
using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;
using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace trk = tensorrt_llm::runtime::kernels;

using PackedFloat = moe::dev::routing::PackedScoreIdx<float>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
constexpr T mulLog2(T a, T bLog2)
{
    return a << bLog2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
constexpr T divUpLog2(T a, T bLog2)
{
    return ((a + (1 << bLog2) - 1) >> bLog2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
constexpr T divUpMulLog2(T a, T bLog2)
{
    return mulLog2<T>(divUpLog2<T>(a, bLog2), bLog2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__ constexpr T mulTileN(T a, T tileN)
{
    return a * tileN;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__host__ __device__ constexpr T divUpTileN(T a, T tileN)
{
    return (a + tileN - 1) / tileN;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__host__ __device__ constexpr T divUpMulTileN(T a, T tileN)
{
    return divUpTileN(a, tileN) * tileN;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
constexpr void initData(T* data, int num, int rngSeed)
{
    std::default_random_engine rng(rngSeed);
    std::normal_distribution dist{0.F, 1.F};
    for (int ii = 0; ii < num; ++ii)
    {
        data[ii] = static_cast<T>(dist(rng));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
inline void printSet(std::set<int32_t> const& s)
{
    for (auto it = s.begin(); it != s.end(); ++it)
        TLLM_LOG_DEBUG("%d%s ", *it, std::distance(it, s.end()) == 1 ? "" : ", ");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
inline bool checkSetEqual(int ie, std::set<int32_t> const& exp, std::set<int32_t> const& test, std::string const& name)
{
    bool result = true;
    if (exp != test)
    {
        TLLM_LOG_DEBUG("Expert %d expected %s:\t", ie, name.c_str());
        printSet(exp);
        TLLM_LOG_DEBUG("\t but got:\t");
        printSet(test);
        TLLM_LOG_DEBUG("\n");
        result = false;
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline bool assertEqual(T* hostData, T* hostTest, int n, std::string const& name)
{
    TLLM_LOG_DEBUG("assertEqual %s", name.c_str());
    bool result = true;
    int count = 0;
    for (int ii = 0; ii < n; ++ii)
    {
        EXPECT_EQ(hostTest[ii], hostData[ii]);
        if (hostTest[ii] != hostData[ii])
        {
            TLLM_LOG_ERROR("Unexpected result for '%s' at idx %d: exp %d != %d\n", name.c_str(), ii, (int) hostData[ii],
                (int) hostTest[ii]);
            result = false;
            count++;

            if (count > 10)
            {
                break;
            }
        }
    }
    return result;
}

template <typename T>
inline bool printTwoVar(T* hostData, T* hostTest, int n, std::string const& name)
{
    TLLM_LOG_INFO("printTwoVar %s", name.c_str());
    for (int ii = 0; ii < n; ++ii)
    {
        std::cout << "[" << ii << "] " << hostData[ii] << " " << hostTest[ii] << " | ";
        if (ii % 10 == 0)
        {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    std::free(hostTest);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline bool isClose(T* hostData, T* hostTest, int n, std::string const& name, float atol = 0.001, float rtol = 0.01)
{
    TLLM_LOG_DEBUG("isClose %s", name.c_str());
    auto numErrors = 0;
    float maxAbsDiff = 0.F;
    float maxRelDiff = 0.F;
    bool result = true;
    int count = 0;
    for (int ii = 0; ii < n; ++ii)
    {
        auto exp = float{hostData[ii]};
        auto test = float{hostTest[ii]};
        float absDiff = std::abs(exp - test);
        float relDiff = absDiff / std::abs(exp);

        // Compute the tolerance for this one.
        float tol = atol + rtol * std::abs(exp);
        // Detect errors and report.
        bool error = std::isnan(test) || std::isinf(test) || (absDiff > tol);
        if (error)
        {
            TLLM_LOG_ERROR("Unexpected result for '%s' at idx %d: exp %f, got %f, atol %f rtol %f\n", name.c_str(), ii,
                exp, test, atol, rtol);
            maxAbsDiff = std::max(maxAbsDiff, absDiff);
            maxRelDiff = std::max(maxRelDiff, relDiff);
            numErrors++;
            result = false;
            count++;

            if (count > 10)
            {
                break;
            }
        }
    }
    if (numErrors > 0)
    {
        TLLM_LOG_ERROR("'%s': max abs diff: %f, max rel diff: %f\n", name.c_str(), maxAbsDiff, maxRelDiff);
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline auto comp = [](PackedFloat const& a, PackedFloat const& b)
{
    return ((a.score > b.score) || (a.score == b.score && a.idx < b.idx)); //@TODO: check if this is correct
};

////////////////////////////////////////////////////////////////////////////////////////////////////
struct RoutingKernelTestParam
{
    RoutingMethodType routingMethod{RoutingMethodType::Renormalize};
    int32_t numTokens{0};
    int32_t numExperts{0};
    uint32_t topK{1};

    int32_t localExpertsStartIdx{0};
    int32_t localExpertsStrideLog2{0};
    int32_t numLocalExperts{0};
    int32_t paddingLog2{3};
    int32_t tileTokensDim{1};

    int32_t singleClusterTokenNum{1024};
    bool usePdl{true};
    bool getExpWeights{true};

    int requiredComputeCapability{9};

    // Check the input parameters
    bool useTopKAsInput{false};       // When true, mPtrTopKIds + mPtrTopKWeights are provided as input
    bool useTopKPackedAsInput{false}; // When true, mPtrTopKPacked is provided as input (without mPtrScores)
    bool hasInvalidTopKInput{false};
    int32_t invalidExpertIdValue{-1}; // Value used to mark invalid topK entries: -1 or numExperts

    // Special for renormalize routing method
    bool doSoftmaxBeforeTopK{false};
    bool normTopkProb{true};

    // Policy type selection for routingCustom (set automatically by build() if not overridden)
    RoutingPreprocessType preprocessType{RoutingPreprocessType::None};
    RoutingPostprocessType postprocessType{RoutingPostprocessType::Softmax};

    // Special for deepseek routing method
    int32_t nGroup{0};
    int32_t topkGroup{0};
    float routedScalingFactor{1.0f};

    // Default constructor
    RoutingKernelTestParam() = default;

    // Copy / move constructors and assignment operators
    RoutingKernelTestParam(RoutingKernelTestParam const& other) = default;
    RoutingKernelTestParam(RoutingKernelTestParam&& other) = default;
    RoutingKernelTestParam& operator=(RoutingKernelTestParam const& other) = default;
    RoutingKernelTestParam& operator=(RoutingKernelTestParam&& other) = default;
    ~RoutingKernelTestParam() = default;

    //
    // Fluent builder methods — each returns *this so calls can be chained.
    // Usage:
    //   auto param = RoutingKernelTestParam()
    //       .withRoutingMethod(RoutingMethodType::Renormalize)
    //       .withNumTokens(4)
    //       .withNumExperts(128)
    //       .withTopK(8)
    //       .build();
    //

    RoutingKernelTestParam& withRoutingMethod(RoutingMethodType val)
    {
        routingMethod = val;
        return *this;
    }

    RoutingKernelTestParam& withNumTokens(int32_t val)
    {
        numTokens = val;
        return *this;
    }

    RoutingKernelTestParam& withNumExperts(int32_t val)
    {
        numExperts = val;
        return *this;
    }

    RoutingKernelTestParam& withTopK(uint32_t val)
    {
        topK = val;
        return *this;
    }

    RoutingKernelTestParam& withExpertParallelization(int32_t ep, int32_t epId = 0)
    {
        mExpertParallelization = ep;
        mExpertParallelizationId = epId;
        return *this;
    }

    RoutingKernelTestParam& withTileTokensDim(int32_t val)
    {
        tileTokensDim = val;
        return *this;
    }

    RoutingKernelTestParam& withPaddingLog2(int32_t val)
    {
        paddingLog2 = val;
        return *this;
    }

    RoutingKernelTestParam& withLocalExpertsStrideLog2(int32_t val)
    {
        localExpertsStrideLog2 = val;
        return *this;
    }

    RoutingKernelTestParam& withUsePdl(bool val)
    {
        usePdl = val;
        return *this;
    }

    RoutingKernelTestParam& withGetExpWeights(bool val)
    {
        getExpWeights = val;
        return *this;
    }

    RoutingKernelTestParam& withUseTopKAsInput(bool val)
    {
        useTopKAsInput = val;
        return *this;
    }

    RoutingKernelTestParam& withUseTopKPackedAsInput(bool val)
    {
        useTopKPackedAsInput = val;
        return *this;
    }

    RoutingKernelTestParam& withHasInvalidTopKInput(bool val)
    {
        hasInvalidTopKInput = val;
        return *this;
    }

    RoutingKernelTestParam& withInvalidExpertIdValue(int32_t val)
    {
        invalidExpertIdValue = val;
        return *this;
    }

    RoutingKernelTestParam& withNGroup(int32_t val)
    {
        nGroup = val;
        return *this;
    }

    RoutingKernelTestParam& withTopkGroup(int32_t val)
    {
        topkGroup = val;
        return *this;
    }

    RoutingKernelTestParam& withRoutedScalingFactor(float val)
    {
        routedScalingFactor = val;
        return *this;
    }

    RoutingKernelTestParam& withPreprocessType(RoutingPreprocessType val)
    {
        preprocessType = val;
        mPreprocessTypeOverridden = true;
        return *this;
    }

    RoutingKernelTestParam& withPostprocessType(RoutingPostprocessType val)
    {
        postprocessType = val;
        mPostprocessTypeOverridden = true;
        return *this;
    }

    RoutingKernelTestParam& withNormTopkProb(bool val)
    {
        normTopkProb = val;
        mNormTopkProbOverridden = true;
        return *this;
    }

    RoutingKernelTestParam& withRequiredComputeCapability(int val)
    {
        requiredComputeCapability = val;
        return *this;
    }

    /// Finalize and validate. Must be called after all `with*()` setters.
    RoutingKernelTestParam& build()
    {
        // Validate routing method
        if (routingMethod != RoutingMethodType::Default && routingMethod != RoutingMethodType::Renormalize
            && routingMethod != RoutingMethodType::RenormalizeNaive && routingMethod != RoutingMethodType::Llama4
            && routingMethod != RoutingMethodType::DeepSeekV3 && routingMethod != RoutingMethodType::MiniMax2
            && routingMethod != RoutingMethodType::SigmoidRenorm)
        {
            throw std::invalid_argument("Invalid routing method");
        }

        // Derive expert parallelization parameters
        numLocalExperts = numExperts / mExpertParallelization;
        localExpertsStartIdx = numLocalExperts * mExpertParallelizationId;

        // Apply routing-method-specific settings
        if (routingMethod == RoutingMethodType::Default)
        {
            doSoftmaxBeforeTopK = true;
            normTopkProb = false;
        }
        else if (routingMethod == RoutingMethodType::RenormalizeNaive)
        {
            doSoftmaxBeforeTopK = true;
            normTopkProb = true;
        }
        else if (routingMethod == RoutingMethodType::SigmoidRenorm)
        {
            doSoftmaxBeforeTopK = false;
            if (!mNormTopkProbOverridden)
            {
                normTopkProb = true;
            }
        }
        else
        {
            doSoftmaxBeforeTopK = false;
            normTopkProb = false;
        }

        // Derive policy types from routing method when not explicitly set
        if (!mPreprocessTypeOverridden)
        {
            if (routingMethod == RoutingMethodType::Default || routingMethod == RoutingMethodType::RenormalizeNaive)
            {
                preprocessType = RoutingPreprocessType::Softmax;
            }
            else if (routingMethod == RoutingMethodType::MiniMax2)
            {
                preprocessType = RoutingPreprocessType::SigmoidBias;
            }
            else if (routingMethod == RoutingMethodType::SigmoidRenorm)
            {
                preprocessType = RoutingPreprocessType::Sigmoid;
            }
            else
            {
                preprocessType = RoutingPreprocessType::None;
            }
        }
        if (!mPostprocessTypeOverridden)
        {
            if (routingMethod == RoutingMethodType::Default)
            {
                postprocessType = RoutingPostprocessType::None;
            }
            else if (routingMethod == RoutingMethodType::RenormalizeNaive)
            {
                postprocessType = RoutingPostprocessType::SumNormalize;
            }
            else if (routingMethod == RoutingMethodType::MiniMax2)
            {
                postprocessType = RoutingPostprocessType::ScaledSumNormalize;
            }
            else if (routingMethod == RoutingMethodType::SigmoidRenorm)
            {
                postprocessType = RoutingPostprocessType::SumNormalize;
            }
            else
            {
                postprocessType = RoutingPostprocessType::Softmax;
            }
        }

        // Set singleClusterTokenNum
        if (routingMethod == RoutingMethodType::DeepSeekV3)
        {
            singleClusterTokenNum = 1024;
        }
        else
        {
            singleClusterTokenNum = 256;
        }

        // Cross-field validation
        if (hasInvalidTopKInput && !useTopKAsInput)
        {
            throw std::invalid_argument("hasInvalidTopKInput is only supported when useTopKAsInput is true");
        }
        if (useTopKAsInput && useTopKPackedAsInput)
        {
            throw std::invalid_argument("useTopKAsInput and useTopKPackedAsInput are mutually exclusive");
        }

        return *this;
    }

    std::string toString() const
    {
        return tensorrt_llm::common::fmtstr(
            "RoutingKernelTestParam[num_tokens=%d, num_experts=%d, topK=%u, doSoftmaxBeforeTopK=%d, normTopkProb=%d, "
            "localExpertsStartIdx=%d, localExpertsStrideLog2=%d, numLocalExperts=%d, usePdl=%d, useTopKAsInput=%d, "
            "useTopKPackedAsInput=%d, hasInvalidTopKInput=%d]",
            numTokens, numExperts, topK, doSoftmaxBeforeTopK, normTopkProb, localExpertsStartIdx,
            localExpertsStrideLog2, numLocalExperts, usePdl, useTopKAsInput, useTopKPackedAsInput, hasInvalidTopKInput);
    }

private:
    // Builder state — used by build() to derive public fields.
    int32_t mExpertParallelization{1};
    int32_t mExpertParallelizationId{0};
    bool mPreprocessTypeOverridden{false};
    bool mPostprocessTypeOverridden{false};
    bool mNormTopkProbOverridden{false};
};

template <typename T>
class RoutingKernelTest : public testing::Test
{
public:
    // Add a method to generate time-based seed
    static uint32_t generateTimeBasedSeed()
    {
        std::random_device rd;
        uint32_t seed = rd();
        TLLM_LOG_DEBUG("Random device seed: %u", seed);
        return seed;
    }

    // Method to set seed after construction
    void setSeed(uint32_t seed)
    {
        mSeed = seed;
    }

    // Method to reset to time-based seed
    void resetToTimeBasedSeed()
    {
        mSeed = generateTimeBasedSeed();
    }

    void SetUp() override;
    void TearDown() override;

    void runTest(RoutingKernelTestParam const& param);

protected:
    using PackedType = moe::dev::routing::PackedScoreIdx<T>;

    virtual size_t getDeviceWorkspaceSize(RoutingKernelTestParam const& param)
    {
        return sizeof(int32_t);
    };

    virtual void callTestedFunction(RoutingKernelTestParam const& param, TensorPtr& workspaceDevice)
    {
        throw std::logic_error("Not implemented");
    }

    virtual void computeTopKExperts(RoutingKernelTestParam const& param)
    {
        throw std::logic_error("Not implemented");
    }

    virtual void computePermutation(RoutingKernelTestParam const& param);

    virtual void callHostFunction(RoutingKernelTestParam const& param);

    virtual void verifyExpertRoutingIndices(RoutingKernelTestParam const& param);

    virtual void allocateBuffers(RoutingKernelTestParam const& param);

    virtual void setupBuffers(RoutingKernelTestParam const& param);

    inline int32_t computeLog2(int32_t val, std::string const& name = "")
    {
        int32_t n = val;
        int32_t out = 0;
        while (n >>= 1)
        {
            ++out;
        }
        if ((1 << out) != val)
        {
            out = -1;
        }
        return out;
    }

    template <typename RoutingData>
    inline void setCommonParams(RoutingKernelTestParam const& param, RoutingData& routingData)
    {
        // Set common parameters that are shared across different routing methods
        routingData.mNumTokens = param.numTokens;
        routingData.mNumExperts = param.numExperts;
        routingData.mTopK = param.topK;
        routingData.mTileTokensDim = param.tileTokensDim;
        routingData.mPaddingLog2 = computeLog2(param.tileTokensDim);
        routingData.mLocalExpertsStartIdx = param.localExpertsStartIdx;
        routingData.mLocalExpertsStrideLog2 = param.localExpertsStrideLog2;
        routingData.mNumLocalExperts = param.numLocalExperts;
        routingData.mUsePdl = param.usePdl;

        // Set output pointers
        routingData.mPtrExpertCounts = bufferCast<int32_t>(*mPtrExpertCountsDevice);
        routingData.mPtrPermutedIdxSize = bufferCast<int32_t>(*mPtrPermutedIdxSizeDevice);
        routingData.mPtrExpandedIdxToPermutedIdx = bufferCast<int32_t>(*mPtrExpandedIdxToPermutedIdxDevice);
        routingData.mPtrPermutedIdxToTokenIdx = bufferCast<int32_t>(*mPtrPermutedIdxToTokenIdxDevice);
        routingData.mPtrTopKWeights = bufferCast<T>(*mPtrTopKWeightsDevice);
        routingData.mPtrTopKPacked = reinterpret_cast<PackedType*>(bufferCast<int8_t>(*mPtrTopKPackedDevice));

        // Set grouped gemm launch config buffers
        routingData.mPtrCtaIdxXyToBatchIdx = bufferCast<int32_t>(*mPtrCtaIdxXyToBatchIdxDevice);
        routingData.mPtrCtaIdxXyToMnLimit = bufferCast<int32_t>(*mPtrCtaIdxXyToMnLimitDevice);
        routingData.mPtrNumNonExitingCtas = bufferCast<int32_t>(*mPtrNumNonExitingCtasDevice);
    }

private:
    void verifyResult(RoutingKernelTestParam const& param);

protected:
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    TensorPtr mCurandStatesDevice;
    uint32_t mSeed = 42;

    struct cudaDeviceProp mDeviceProp;

    // optional: if `nullptr`, `mPtrTopKPacked` must be provided.
    // If it is given, it represents the scores without sigmoid activation for
    // each token and expert.
    // note: if it is provided, we always re-compute the top1 scores
    // dim: [mNumTokens, mNumExperts]
    TensorPtr mPtrScoresHost;
    TensorPtr mPtrScoresDevice;
    // optional: if `nullptr`, scores are used directly as input.
    // If it is given, it must represent a packed value s.t. the most significant
    // 16/32 bits represent the score without sigmoid activation and
    // the least significant 16 bits represent the index of the chosen expert (unsigned).
    // note: this is required if the number of tokens is large.
    // dim: [mNumTokens, mTopK]
    TensorPtr mPtrTopKPackedHost;
    TensorPtr mPtrTopKPackedDevice;

    // optional: Add another input format.
    // dim: [mNumTokens, mTopK]
    TensorPtr mPtrTopKIdsHost;
    TensorPtr mPtrTopKIdsDevice;

    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens, mTopK]
    // Note: this might be reused as input when we take topk_ids as input.
    TensorPtr mPtrTopKWeightsHost;
    TensorPtr mPtrTopKWeightsDevice;

    // note: at least one of the optional outputs below must be provided
    // optional: only used as an intermediate buffer when the number of tokens is large.
    // dim: [2, mNumExperts]
    TensorPtr mPtrExpertCountsHost;
    TensorPtr mPtrExpertCountsDevice;
    // dim: [1]
    TensorPtr mPtrPermutedIdxSizeHost;
    TensorPtr mPtrPermutedIdxSizeDevice;
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens * mTopK]
    TensorPtr mPtrExpandedIdxToPermutedIdxHost;
    TensorPtr mPtrExpandedIdxToPermutedIdxDevice;
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens * mTopK + (mNumExperts << mPaddingLog2) - mNumExperts]
    TensorPtr mPtrPermutedIdxToTokenIdxHost;
    TensorPtr mPtrPermutedIdxToTokenIdxDevice;

    //
    // Grouped Gemm Launch Config Buffers
    //
    TensorPtr mPtrCtaIdxXyToBatchIdxHost;
    TensorPtr mPtrCtaIdxXyToBatchIdxDevice;
    TensorPtr mPtrCtaIdxXyToMnLimitHost;
    TensorPtr mPtrCtaIdxXyToMnLimitDevice;
    TensorPtr mPtrNumNonExitingCtasHost;
    TensorPtr mPtrNumNonExitingCtasDevice;
};

} // namespace tensorrt_llm::tests::kernels::routing

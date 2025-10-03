/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <benchmark/benchmark.h>

#include <nlohmann/json.hpp>

#ifdef USING_OSS_CUTLASS_MOE_GEMM
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
#else
#include "moe_kernels.h"
#endif

#include "tensorrt_llm/kernels/cutlass_kernels/include/cutlass_kernel_selector.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <algorithm>
#include <cuda.h>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::cutlass_extensions;

using namespace CUTLASS_MOE_GEMM_KERNELS_NAMESPACE;
using CUTLASS_MOE_GEMM_NAMESPACE::TmaWarpSpecializedGroupedGemmInput;
using CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::CutlassMoeFCRunner;
using CUTLASS_MOE_GEMM_NAMESPACE::ActivationType;
using CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::ActivationParams;
using CUTLASS_MOE_GEMM_NAMESPACE::isGatedActivation;

static BufferManager::CudaStreamPtr streamPtr;
static std::unique_ptr<BufferManager> bufferManager;
static int deviceCount;
static char* workloadFile = nullptr;
static bool useCudaGraph = true;

enum VERBOSE_LEVEL
{
    SILENT = 0,
    ERROR = 1,
    INFO = 2,
    VERBOSE = 3
};

constexpr int LOG_LEVEL = ERROR;

enum class GemmToProfile : int
{
    GEMM_1 = static_cast<int>(GemmProfilerBackend::GemmToProfile::GEMM_1),
    GEMM_2 = static_cast<int>(GemmProfilerBackend::GemmToProfile::GEMM_2),
    LAYER = static_cast<int>(3),
};

namespace
{
// Abstract class for routing config
struct RoutingConfig
{
    virtual void start(){};
    virtual void setRouting(int* selected_experts, int64_t num_experts, int64_t k, int64_t num_tokens) = 0;
    virtual std::string getName() = 0;
    virtual bool isDeterministic() const = 0;
    virtual bool supportsConfig(int64_t num_experts, int64_t k, int64_t num_tokens) const = 0;
};

/**
 * Generates a perfectly balanced routing configuration
 */
struct LoadBalancedRoutingConfig : public RoutingConfig
{
    std::string getName() override
    {
        return "balanced";
    }

    bool isDeterministic() const override
    {
        return true;
    }

    void setRouting(int* selected_experts, int64_t num_experts, int64_t k, int64_t num_tokens) override
    {
        std::vector<int> h_selected_experts(k * num_tokens, 0);
        int stride = tensorrt_llm::common::ceilDiv(num_experts, k);
        for (int token = 0; token < num_tokens; token++)
        {
            for (int i = 0; i < k; i++)
            {
                h_selected_experts[token * k + i] = (token + i * stride) % num_experts;
            }
        }

        check_cuda_error(cudaMemcpyAsync(selected_experts, h_selected_experts.data(),
            h_selected_experts.size() * sizeof(int), cudaMemcpyHostToDevice, streamPtr->get()));
        check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
    }

    bool supportsConfig(int64_t, int64_t, int64_t) const override
    {
        return true;
    }
};

/**
 * Selects experts according to given random distribution
 */
struct RandomDistributionRoutingConfig : public RoutingConfig
{
    using ElementType = float;
    std::mt19937_64 twister{0xD5};
    std::vector<float> probabilities;
    int64_t num_experts;
    int64_t k;
    std::string name;

    RandomDistributionRoutingConfig(std::vector<ElementType> const& in_probabilities, int64_t num_experts, int64_t k,
        std::string name = "random_distribution")
        : probabilities(std::move(in_probabilities))
        , num_experts(num_experts)
        , k(k)
        , name(std::move(name))
    {
        TLLM_CHECK_WITH_INFO(num_experts == probabilities.size(),
            "Cannot create random routing distribution. Number of experts does not match the number of weights");
    }

    void start()
    {
        twister.seed(0xD5);
    }

    std::string getName() override
    {
        return name;
    }

    bool isDeterministic() const override
    {
        return false;
    }

    void doSample(float& curr_max, std::vector<int>& selected)
    {
        std::uniform_real_distribution<float> dist(0, curr_max);
        float value = dist(twister);
        float running_sum = 0;
        for (int expert = 0; expert < probabilities.size(); expert++)
        {
            if (std::find(selected.begin(), selected.end(), expert) != selected.end())
                continue; // Already picked
            float prob = probabilities[expert];
            running_sum += prob;
            if (value < running_sum)
            {
                curr_max -= prob;
                selected.push_back(expert);
                return;
            }
        }
    }

    void setRouting(int* selected_experts, int64_t num_experts, int64_t k, int64_t num_tokens) override
    {
        TLLM_CHECK(num_experts == probabilities.size());
        std::vector<int> h_selected_experts(k * num_tokens, 0);
        std::vector<int> selected;
        float max = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
        // TODO Put this on the GPU
        for (int token = 0; token < num_tokens; token++)
        {
            selected.clear();
            float curr_max = max;
            for (int choice = 0; choice < k; choice++)
            {
                doSample(curr_max, selected);
            }
            std::copy(selected.begin(), selected.end(), h_selected_experts.begin() + token * k);
        }
        check_cuda_error(cudaMemcpyAsync(selected_experts, h_selected_experts.data(),
            h_selected_experts.size() * sizeof(int), cudaMemcpyHostToDevice, streamPtr->get()));
        check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
    }

    bool supportsConfig(int64_t num_experts_, int64_t, int64_t) const override
    {
        return num_experts == num_experts_;
    }
};

/**
 * Generates routing values by sampling a uniform distribution [-1,1)
 */
struct UniformRoutingConfig : public RoutingConfig
{
    std::mt19937_64 twister{0xD5};

    void start()
    {
        twister.seed(0xD5);
    }

    std::string getName() override
    {
        return "uniform";
    }

    bool isDeterministic() const override
    {
        return false;
    }

    void setRouting(int* selected_experts, int64_t num_experts, int64_t k, int64_t num_tokens) override
    {
        std::uniform_int_distribution<int> dist(0, num_experts - 1);
        std::vector<int> input(k * num_tokens);
        for (int i = 0; i < num_tokens; i++)
        {
            for (int j = 0; j < k; j++)
            {
                while (true)
                {
                    int expert_id = dist(twister);
                    bool valid = true;
                    for (int prev_j = 0; prev_j < j; prev_j++)
                    {
                        if (expert_id == input[i * k + prev_j])
                        {
                            valid = false;
                            break;
                        }
                    }
                    if (valid)
                    {
                        input[i * k + j] = expert_id;
                        break;
                    }
                }
            }
        }
        check_cuda_error(cudaMemcpyAsync(
            selected_experts, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice, streamPtr->get()));
        check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
    }

    bool supportsConfig(int64_t, int64_t, int64_t) const override
    {
        return true;
    }
};

/**
 * Stores a specific routing configuration
 */
struct VectoredRoutingConfig : public RoutingConfig
{
    using ElementType = int;
    std::vector<int> config;
    int64_t num_experts;
    int64_t k;
    std::string name;

    VectoredRoutingConfig(
        std::vector<ElementType> config, int64_t num_experts, int64_t k, std::string name = "vectored")
        : config(config)
        , num_experts(num_experts)
        , k(k)
        , name(name)
    {
    }

    std::string getName() override
    {
        return name;
    }

    bool isDeterministic() const override
    {
        return true;
    }

    void setRouting(int* selected_experts, int64_t num_experts, int64_t k, int64_t num_tokens) override
    {
        for (int64_t i = 0; i < num_tokens; i += config.size())
        {
            int64_t num_to_copy = std::min(num_tokens - i, (int64_t) config.size());
            check_cuda_error(cudaMemcpyAsync(
                selected_experts, config.data(), num_to_copy * sizeof(int), cudaMemcpyHostToDevice, streamPtr->get()));
            selected_experts += num_to_copy;
        }
        check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
    }

    bool supportsConfig(int64_t num_experts_, int64_t k_, int64_t) const override
    {
        return num_experts_ == num_experts && k_ == k;
    }
};

}; // namespace

constexpr int LOAD_BALANCED_ROUTING_CONFIG = 0;
constexpr int UNIFORM_ROUTING_CONFIG = 1;
std::vector<std::shared_ptr<RoutingConfig>> routingConfigCache{
    std::static_pointer_cast<RoutingConfig>(std::make_shared<LoadBalancedRoutingConfig>()),
    std::static_pointer_cast<RoutingConfig>(std::make_shared<UniformRoutingConfig>()),
};

#ifdef ENABLE_FP8
using SafeFP8 = __nv_fp8_e4m3;
#else
using SafeFP8 = void;
#endif

#ifdef ENABLE_FP4
using SafeFP4 = __nv_fp4_e2m1;
#else
using SafeFP4 = void;
#endif

template <class TypeTuple_>
class MixtureOfExpertsBenchmark : public ::benchmark::Fixture
{
public:
    using DataType = typename TypeTuple_::DataType;
    using WeightType = typename TypeTuple_::WeightType;
    using OutputType = typename TypeTuple_::OutputType;
    constexpr static bool INT4 = std::is_same_v<WeightType, cutlass::uint4b_t>;
    constexpr static bool NVFP4 = std::is_same_v<DataType, SafeFP4> && std::is_same_v<WeightType, SafeFP4>;
    constexpr static bool FP8 = std::is_same_v<DataType, SafeFP8> && std::is_same_v<WeightType, SafeFP8>;
    constexpr static bool WFP4AFP8 = std::is_same_v<WeightType, SafeFP4> && std::is_same_v<DataType, SafeFP8>;
    constexpr static bool INT_QUANT = !std::is_same_v<DataType, WeightType>
        && (std::is_same_v<WeightType, cutlass::uint4b_t> || std::is_same_v<WeightType, uint8_t>);
    constexpr static bool ANY_FP4 = NVFP4 || WFP4AFP8;
    using InputType = std::conditional_t<NVFP4, OutputType, DataType>;
    using WeightStorage = std::conditional_t<INT_QUANT || ANY_FP4, uint8_t, WeightType>;
    constexpr static int WEIGHT_ELEM_PER_BYTE = (INT4 || ANY_FP4) ? 2 : 1;
    int const BASE_HIDDEN_SIZE = 64 / sizeof(WeightType) * WEIGHT_ELEM_PER_BYTE;

    constexpr static int64_t FP4_VECTOR_SIZE = NVFP4 ? TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                                                     : TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize;
    constexpr static int64_t MinNDimAlignment = NVFP4 ? TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4
                                                      : TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX;
    constexpr static int64_t MinKDimAlignment = NVFP4 ? TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4
                                                      : TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX;

    std::vector<BufferManager::IBufferPtr> managed_buffers;
    int* mSelectedExperts{};
    DataType* mInputTensor{};

    int64_t mHiddenSize{};
    int64_t mNumExperts{};
    int64_t mNumExpertsPerNode{};
    int64_t mK{};

    constexpr static nvinfer1::DataType toDTypeID()
    {
        if (FP8 || WFP4AFP8)
            return nvinfer1::DataType::kFP8;
        if (NVFP4)
            return nvinfer1::DataType::kFP4;
        if (INT_QUANT && INT4)
            return nvinfer1::DataType::kINT4;
        if (INT_QUANT)
            return nvinfer1::DataType::kINT8;
        if (std::is_same_v<DataType, float>)
            return nvinfer1::DataType::kFLOAT;
        if (std::is_same_v<DataType, half>)
            return nvinfer1::DataType::kHALF;
#ifdef ENABLE_BF16
        if (std::is_same_v<DataType, nv_bfloat16>)
            return nvinfer1::DataType::kBF16;
#endif
        TLLM_THROW("Unrecognised format");
    };

    constexpr static nvinfer1::DataType toWTypeID()
    {
        if (FP8)
            return nvinfer1::DataType::kFP8;
        if (NVFP4 || WFP4AFP8)
            return nvinfer1::DataType::kFP4;
        if (INT_QUANT && INT4)
            return nvinfer1::DataType::kINT4;
        if (INT_QUANT)
            return nvinfer1::DataType::kINT8;
        if (std::is_same_v<DataType, float>)
            return nvinfer1::DataType::kFLOAT;
        if (std::is_same_v<DataType, half>)
            return nvinfer1::DataType::kHALF;
#ifdef ENABLE_BF16
        if (std::is_same_v<DataType, nv_bfloat16>)
            return nvinfer1::DataType::kBF16;
#endif
        TLLM_THROW("Unrecognised format");
    };

    template <class T>
    constexpr static auto typeToDtypeID()
    {
        if constexpr (std::is_same_v<T, SafeFP8>)
        {
            return nvinfer1::DataType::kFP8;
        }
        else if constexpr (std::is_same_v<T, SafeFP4>)
        {
            return nvinfer1::DataType::kFP4;
        }
        else if constexpr (std::is_same_v<T, uint8_t>)
        {
            return nvinfer1::DataType::kINT8;
        }
        else if constexpr (std::is_same_v<T, cutlass::uint4b_t>)
        {
            return nvinfer1::DataType::kINT4;
        }
        else if constexpr (std::is_same_v<T, nv_bfloat16>)
        {
            return nvinfer1::DataType::kBF16;
        }
        else if constexpr (std::is_same_v<T, half>)
        {
            return nvinfer1::DataType::kHALF;
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return nvinfer1::DataType::kFLOAT;
        }
        else
        {
            // sizeof(T) to make the static assert dependent on the template
            static_assert(sizeof(T) == 0, "Unrecognised data type");
        }
    }

    static bool shouldSkip()
    {
#ifndef ENABLE_FP8
        static_assert(!FP8, "FP8 Tests enabled on unsupported CUDA version");
#endif
#ifndef ENABLE_FP4
        static_assert(!ANY_FP4, "FP4 Tests enabled on unsupported CUDA version");
#endif
        bool should_skip_unsupported_fp8 = getSMVersion() < 89 && FP8;
        bool should_skip_unsupported_fp4 = (getSMVersion() < 100 || getSMVersion() >= 120) && ANY_FP4;
        return should_skip_unsupported_fp8 || should_skip_unsupported_fp4;
    }

    // Deprecated, just here to suppress warnings
    void SetUp(benchmark::State const& s) override
    {
        abort();
    }

    void TearDown(benchmark::State const& s) override
    {
        abort();
    }

    cudaEvent_t mStartEvent, mEndEvent;

    void SetUp(benchmark::State& s) override
    {
        assert(bufferManager);
        if (shouldSkip())
        {
            s.SkipWithMessage("GPU does not support dtype");
        }

        // Makes sure nothing from a previous iteration hangs around
        check_cuda_error(cudaDeviceSynchronize());
        check_cuda_error(cudaEventCreate(&mStartEvent));
        check_cuda_error(cudaEventCreate(&mEndEvent));
    }

    void TearDown(benchmark::State& s) override
    {
        managed_buffers.clear();

        check_cuda_error(cudaEventDestroy(mStartEvent));
        check_cuda_error(cudaEventDestroy(mEndEvent));
        check_cuda_error(cudaDeviceSynchronize());
    }

    CutlassMoeFCRunner<DataType, WeightType, OutputType, InputType> mMoERunner{};
    GemmProfilerBackend mGemmProfilerBackend{};
    char* mGemmProfilerWorkspace{};
    char* mWorkspace{};
    float* mScaleProbs{};
    WeightStorage* mExpertWeight1{};
    WeightStorage* mExpertWeight2{};
    DataType* mExpertIntScale1{};
    DataType* mExpertIntScale2{};

    float* mExpertFP8Scale1{};
    float* mExpertFP8Scale2{};
    float* mExpertFP8Scale3{};

    float* mExpertFP4ActScale1{};
    using ElementSF = TmaWarpSpecializedGroupedGemmInput::ElementSF;
    ElementSF* mExpertFP4WeightSf1{};
    float* mExpertFP4GlobalScale1{};
    float* mExpertFP4ActScale2{};
    ElementSF* mExpertFP4WeightSf2{};
    float* mExpertFP4GlobalScale2{};

    DataType* mExpertBias1{};
    DataType* mExpertBias2{};

    OutputType* mFinalOutput{};
    int* mSourceToExpandedMap;
    int64_t mInterSize{};
    int64_t mTotalTokens{};

    int mRoutingConfigIndex = 0;

    bool mUseBias = true;
    bool mUseFinalScale = true;
    bool mIsGated = false;
    int mGatedMultiplier = 1;

    ActivationType mActType = ActivationType::Relu;

    constexpr static int64_t NUM_BUFFERS = 32;
    int64_t mNumWorkspaceBuffers = NUM_BUFFERS;
    int64_t mNumInputBuffers = NUM_BUFFERS;
    int64_t mNumGemmProfilerBuffers = NUM_BUFFERS;

    std::array<QuantParams, NUM_BUFFERS> mQuantParams{};
    bool mUseLora = false;
    bool mUsePrequantScale = false;
    int mGroupSize = -1;
    std::array<LoraParams, NUM_BUFFERS> mLoraParams{};

    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> mSelectedConfig = std::nullopt;

    int64_t mBufferIndex = 0;
    size_t mGemmProfilerWorkspaceSize = 0;
    size_t mWorkspaceSize = 0;
    size_t mExpertWeight1Size = 0;
    size_t mExpertWeight2Size = 0;
    size_t mExpertBias1Size = 0;
    size_t mExpertBias2Size = 0;
    size_t mInputTensorSize = 0;
    size_t mFinalOutputSize = 0;
    size_t mSourceToExpandedMapSize = 0;
    size_t mScaleProbsSize = 0;
    size_t mSelectedExpertsSize = 0;
    size_t mExpertFP4WeightSf1Size = 0;
    size_t mExpertFP4WeightSf2Size = 0;
    size_t mExpertIntScale1Size = 0;
    size_t mExpertIntScale2Size = 0;

    size_t padSize(size_t size)
    {
        return ceilDiv(size, 128) * 128;
    }

    template <class T>
    T* allocBuffer(size_t size)
    {
        size_t size_padded = padSize(size) * sizeof(T);
        auto i_buffer = bufferManager->gpu(size_padded);
        check_cuda_error(cudaGetLastError());
        managed_buffers.emplace_back(std::move(i_buffer));
        T* ptr = static_cast<T*>(managed_buffers.back()->data());
        populateRandomBuffer(ptr, size_padded, streamPtr->get());
        return ptr;
    }

    void initBuffersPermute(int64_t num_tokens, int64_t hidden_size, int64_t inter_size, int64_t num_experts, int64_t k,
        int64_t routing_config, MOEParallelismConfig parallelism_config, GemmToProfile gemm_to_profile)
    {
        assert(hidden_size % BASE_HIDDEN_SIZE == 0);

        managed_buffers.clear();

        mTotalTokens = num_tokens;
        mHiddenSize = hidden_size;
        mInterSize = inter_size / parallelism_config.tp_size;
        mNumExperts = num_experts;
        mNumExpertsPerNode = num_experts / parallelism_config.ep_size;
        mK = k;
        mIsGated = isGatedActivation(mActType);
        mGatedMultiplier = mIsGated ? 2 : 1;
        auto const gated_inter = mInterSize * mGatedMultiplier;
        size_t const expert_matrix_size = padSize(mNumExpertsPerNode * mHiddenSize * mInterSize);

        bool need_weight_1 = gemm_to_profile == GemmToProfile::GEMM_1 || gemm_to_profile == GemmToProfile::LAYER;
        bool need_weight_2 = gemm_to_profile == GemmToProfile::GEMM_2 || gemm_to_profile == GemmToProfile::LAYER;
        mExpertWeight1Size = need_weight_1 ? expert_matrix_size * mGatedMultiplier / WEIGHT_ELEM_PER_BYTE : 0;
        mExpertWeight2Size = need_weight_2 ? expert_matrix_size / WEIGHT_ELEM_PER_BYTE : 0;
        mExpertWeight1 = need_weight_1 ? allocBuffer<WeightStorage>(mExpertWeight1Size * NUM_BUFFERS) : nullptr;
        mExpertWeight2 = need_weight_2 ? allocBuffer<WeightStorage>(mExpertWeight2Size * NUM_BUFFERS) : nullptr;

        if (gemm_to_profile == GemmToProfile::LAYER)
        {
            mWorkspaceSize = mMoERunner.getWorkspaceSize(mTotalTokens, mHiddenSize, mInterSize, mNumExperts, mK,
                mActType, parallelism_config, mUseLora, /*use_deepseek_fp8_block_scale=*/false,
                /*min_latency_mode=*/false, mUsePrequantScale);

            mNumWorkspaceBuffers = mWorkspaceSize > 1024 * 1024 * 1024 ? 2 : NUM_BUFFERS;
            mWorkspace = allocBuffer<char>(mWorkspaceSize * mNumWorkspaceBuffers);

            mExpertBias1 = nullptr;
            mExpertBias2 = nullptr;
            if (mUseBias)
            {
                mExpertBias1Size = padSize(mNumExpertsPerNode * gated_inter);
                mExpertBias2Size = padSize(mNumExpertsPerNode * mHiddenSize);
                mExpertBias1 = allocBuffer<DataType>(mExpertBias1Size * NUM_BUFFERS);
                mExpertBias2 = allocBuffer<DataType>(mExpertBias2Size * NUM_BUFFERS);
            }

            if constexpr (INT_QUANT)
            {
                mExpertIntScale1Size = padSize(mNumExpertsPerNode * gated_inter);
                mExpertIntScale2Size = padSize(mNumExpertsPerNode * mHiddenSize);
                mExpertIntScale1 = allocBuffer<DataType>(mExpertIntScale1Size * NUM_BUFFERS);
                mExpertIntScale2 = allocBuffer<DataType>(mExpertIntScale2Size * NUM_BUFFERS);

                for (int i = 0; i < NUM_BUFFERS; i++)
                {
                    mQuantParams[i] = QuantParams::Int(
                        mExpertIntScale1 + mExpertIntScale1Size * i, mExpertIntScale2 + mExpertIntScale2Size * i);
                }
            }
            else if constexpr (FP8)
            {
                mExpertFP8Scale1 = allocBuffer<float>(mNumExpertsPerNode);
                mExpertFP8Scale2 = allocBuffer<float>(1);
                mExpertFP8Scale3 = allocBuffer<float>(mNumExpertsPerNode);

                for (int i = 0; i < NUM_BUFFERS; i++)
                {
                    mQuantParams[i] = QuantParams::FP8(mExpertFP8Scale1, mExpertFP8Scale2, mExpertFP8Scale3);
                }
            }
            else if constexpr (ANY_FP4)
            {
                mExpertFP4ActScale1 = allocBuffer<float>(mNumExpertsPerNode);
                mExpertFP4WeightSf1Size = mNumExpertsPerNode
                    * TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gated_inter, MinNDimAlignment)
                    * TmaWarpSpecializedGroupedGemmInput::alignToSfDim(mHiddenSize, MinKDimAlignment) / FP4_VECTOR_SIZE;
                mExpertFP4WeightSf1 = allocBuffer<ElementSF>(mExpertFP4WeightSf1Size * NUM_BUFFERS);
                mExpertFP4GlobalScale1 = allocBuffer<float>(mNumExpertsPerNode);

                mExpertFP4ActScale2 = allocBuffer<float>(mNumExpertsPerNode);
                mExpertFP4WeightSf2Size = mNumExpertsPerNode
                    * TmaWarpSpecializedGroupedGemmInput::alignToSfDim(mInterSize, MinNDimAlignment)
                    * TmaWarpSpecializedGroupedGemmInput::alignToSfDim(mHiddenSize, MinKDimAlignment) / FP4_VECTOR_SIZE;
                mExpertFP4WeightSf2 = allocBuffer<ElementSF>(mExpertFP4WeightSf2Size * NUM_BUFFERS);
                mExpertFP4GlobalScale2 = allocBuffer<float>(mNumExpertsPerNode);

                auto func = NVFP4 ? QuantParams::FP4 : QuantParams::FP8MXFP4;
                for (int i = 0; i < NUM_BUFFERS; i++)
                {
                    mQuantParams[i] = func(mExpertFP4ActScale1, mExpertFP4WeightSf1 + mExpertFP4WeightSf1Size * i,
                        mExpertFP4GlobalScale1, mExpertFP4ActScale2, mExpertFP4WeightSf2 + mExpertFP4WeightSf2Size * i,
                        mExpertFP4GlobalScale2, false, false);
                }
            }

            mSelectedExpertsSize = padSize(mTotalTokens * mK);
            mSelectedExperts = allocBuffer<int>(mSelectedExpertsSize * NUM_BUFFERS);
            mScaleProbsSize = padSize(mTotalTokens * mK);
            mScaleProbs = allocBuffer<float>(mScaleProbsSize * NUM_BUFFERS);
            mInputTensorSize = padSize(mTotalTokens * mHiddenSize);
            mNumInputBuffers = mInputTensorSize > 1024 * 1024 * 1024 ? 2 : NUM_BUFFERS;
            mInputTensor = allocBuffer<DataType>(mInputTensorSize * mNumInputBuffers);
            mFinalOutputSize = padSize(mTotalTokens * mHiddenSize);
            mFinalOutput = allocBuffer<OutputType>(mFinalOutputSize * mNumInputBuffers);

            mSourceToExpandedMapSize = padSize(mTotalTokens * mK);
            mSourceToExpandedMap = allocBuffer<int>(mSourceToExpandedMapSize * NUM_BUFFERS);
            mRoutingConfigIndex = routing_config;
            auto tactic = routingConfigCache.at(routing_config);
            tactic->start();
            for (int i = 0; i < NUM_BUFFERS; i++)
            {
                tactic->setRouting(mSelectedExperts + mSelectedExpertsSize * i, mNumExperts, mK, mTotalTokens);
            }
        }

#ifdef USING_OSS_CUTLASS_MOE_GEMM
        mGemmProfilerBackend.init(mMoERunner, GemmProfilerBackend::GemmToProfile::Undefined, typeToDtypeID<DataType>(),
            typeToDtypeID<WeightType>(), typeToDtypeID<OutputType>(), mNumExperts, mK, mHiddenSize, mHiddenSize,
            mInterSize, mGroupSize, mActType, mUseBias, mUseLora, /*min_latency_mode=*/false,
            /*need_weights=*/false, parallelism_config, /*enable_alltoall=*/false);
#else
        mGemmProfilerBackend.init(mMoERunner, GemmProfilerBackend::GemmToProfile::Undefined, typeToDtypeID<DataType>(),
            typeToDtypeID<WeightType>(), typeToDtypeID<OutputType>(), mNumExperts, mK, mHiddenSize, mHiddenSize,
            mInterSize, mGroupSize, mActType, mUseBias, mUseLora, /*min_latency_mode=*/false,
            /*need_weights=*/false, parallelism_config);
#endif

        mGemmProfilerWorkspaceSize = 0;
        if (gemm_to_profile == GemmToProfile::GEMM_1 || gemm_to_profile == GemmToProfile::LAYER)
        {
            mGemmProfilerBackend.mGemmToProfile = GemmProfilerBackend::GemmToProfile::GEMM_1;
            mGemmProfilerWorkspaceSize
                = std::max(mGemmProfilerWorkspaceSize, mGemmProfilerBackend.getWorkspaceSize(mTotalTokens));
        }

        if (gemm_to_profile == GemmToProfile::GEMM_2 || gemm_to_profile == GemmToProfile::LAYER)
        {
            mGemmProfilerBackend.mGemmToProfile = GemmProfilerBackend::GemmToProfile::GEMM_2;
            mGemmProfilerWorkspaceSize
                = std::max(mGemmProfilerWorkspaceSize, mGemmProfilerBackend.getWorkspaceSize(mTotalTokens));
        }

        mGemmProfilerWorkspaceSize = padSize(mGemmProfilerWorkspaceSize);
        mNumGemmProfilerBuffers = mGemmProfilerWorkspaceSize > 1024 * 1024 * 1024 ? 2 : NUM_BUFFERS;
        mNumGemmProfilerBuffers = gemm_to_profile == GemmToProfile::LAYER ? 1 : mNumGemmProfilerBuffers;
        mGemmProfilerWorkspace = mGemmProfilerWorkspaceSize > 0
            ? allocBuffer<char>(mGemmProfilerWorkspaceSize * mNumGemmProfilerBuffers)
            : nullptr;

        check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
    }

    void prepareGemmProfiler(GemmToProfile gemm_to_profile)
    {
        if (gemm_to_profile == GemmToProfile::LAYER)
            return;
        mGemmProfilerBackend.mGemmToProfile = static_cast<GemmProfilerBackend::GemmToProfile>(gemm_to_profile);
        auto* expert_weights = gemm_to_profile == GemmToProfile::GEMM_1 ? mExpertWeight1 : mExpertWeight2;
        auto expert_weights_size = gemm_to_profile == GemmToProfile::GEMM_1 ? mExpertWeight1Size : mExpertWeight2Size;
        mGemmProfilerBackend.prepare(mTotalTokens,
            mGemmProfilerWorkspace + mGemmProfilerWorkspaceSize * (mBufferIndex % mNumGemmProfilerBuffers),
            /*expert_weights=*/expert_weights + expert_weights_size * mBufferIndex, streamPtr->get());
    }

    std::array<cudaGraph_t, NUM_BUFFERS> mGraph{};

    std::array<cudaGraphExec_t, NUM_BUFFERS> mGraphInstance{};

    void createGraph(MOEParallelismConfig parallelism_config, GemmToProfile gemm_to_profile)
    {
        if (!useCudaGraph)
            return;

        NVTX3_SCOPED_RANGE(BuildGraph);

        for (int i = 0; i < NUM_BUFFERS; i++)
        {
            mBufferIndex = i;
            // Each buffer will have a different routing config for the gemm profiler
            prepareGemmProfiler(gemm_to_profile);
            check_cuda_error(cudaGraphCreate(&mGraph[i], 0));
            check_cuda_error(cudaStreamBeginCapture(streamPtr->get(), cudaStreamCaptureModeThreadLocal));
            runMoEPermute(parallelism_config, gemm_to_profile);
            check_cuda_error(cudaStreamEndCapture(streamPtr->get(), &mGraph[i]));
            check_cuda_error(cudaGraphInstantiate(&mGraphInstance[i], mGraph[i], nullptr, nullptr, 0));
        }
    }

    void destroyGraph()
    {
        if (!useCudaGraph)
            return;

        NVTX3_SCOPED_RANGE(DestroyGraph);

        for (int i = 0; i < NUM_BUFFERS; i++)
        {
            check_cuda_error(cudaGraphExecDestroy(mGraphInstance[i]));
            check_cuda_error(cudaGraphDestroy(mGraph[i]));
        }
    }

    float benchmarkLoop(MOEParallelismConfig parallelism_config, GemmToProfile gemm_to_profile)
    {
        mBufferIndex = (mBufferIndex + 1) % NUM_BUFFERS;

        // Setup the profiler state for this iteration. CUDA Graphs will do this when it captures the graph.
        if (gemm_to_profile != GemmToProfile::LAYER && !useCudaGraph)
        {
            prepareGemmProfiler(gemm_to_profile);
        }
        else if (gemm_to_profile == GemmToProfile::LAYER)
        {
            auto tactic = routingConfigCache.at(mRoutingConfigIndex);
            if (!tactic->isDeterministic())
            {
                tactic->setRouting(
                    mSelectedExperts + mSelectedExpertsSize * mBufferIndex, mNumExperts, mK, mTotalTokens);
            }
        }

        {
            NVTX3_SCOPED_RANGE(BenchmarkLoopIteration);
            check_cuda_error(cudaEventRecord(mStartEvent, streamPtr->get()));
            if (useCudaGraph)
            {
                cudaGraphLaunch(mGraphInstance[mBufferIndex], streamPtr->get());
            }
            else
            {
                runMoEPermute(parallelism_config, gemm_to_profile);
            }
            check_cuda_error(cudaEventRecord(mEndEvent, streamPtr->get()));
            check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
        }

        float ms;
        check_cuda_error(cudaEventElapsedTime(&ms, mStartEvent, mEndEvent));
        return ms;
    }

    // An imprecise benchmark pass for picking the best tactic.
    // Runs for 3 iterations or 1 second and picks the best option
    int pickBestTactic(MOEParallelismConfig parallelism_config, GemmToProfile gemm_to_profile)
    {
        auto tactics = mMoERunner.getTactics(static_cast<MoeGemmId>(gemm_to_profile));
        ::nvtx3::scoped_range nvtx(tensorrt_llm::common::nvtx::nextColor(),
            "Tactic Profiling GEMM " + std::to_string(static_cast<int>(gemm_to_profile)));
        // We save space by reusing the same workspace buffer for all tactics when doing full layer profiling. So we
        // need to hardcode the buffer index to 0.
        auto old_buffer_index = mBufferIndex;
        mBufferIndex = 0;
        prepareGemmProfiler(gemm_to_profile);
        mBufferIndex = old_buffer_index;

        auto* mGemmProfilerExpertWeights = gemm_to_profile == GemmToProfile::GEMM_1 ? mExpertWeight1 : mExpertWeight2;

        float best_time = INFINITY;
        int best_idx = -1;
        for (int tidx = 0; tidx < tactics.size(); tidx++)
        {
            ::nvtx3::scoped_range nvtx(
                tensorrt_llm::common::nvtx::nextColor(), "Tactic Profiling Tactic Index: " + std::to_string(tidx));
            try
            {
                // Set the tactic
                auto const& t = tactics[tidx];

                {
                    ::nvtx3::scoped_range nvtx(tensorrt_llm::common::nvtx::nextColor(), "Tactic Profiling Warm-Up");
                    // Warm-Up run
                    mGemmProfilerBackend.runProfiler(mTotalTokens, t, mGemmProfilerWorkspace,
                        /*expert_weights=*/mGemmProfilerExpertWeights, streamPtr->get());
                    check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
                }

                // Profile all samples or for 1 sec
                int const max_iters = mGemmProfilerBackend.NUM_ROUTING_SAMPLES * 2;
                float const max_time_ms = 1000.f;

                float time = 0.f;
                int iter = 0;
                while (iter < max_iters && time < max_time_ms)
                {
                    {
                        ::nvtx3::scoped_range nvtx(tensorrt_llm::common::nvtx::nextColor(),
                            "Tactic Profiling Iteration " + std::to_string(iter));

                        check_cuda_error(cudaEventRecord(mStartEvent, streamPtr->get()));
                        mGemmProfilerBackend.runProfiler(mTotalTokens, t, mGemmProfilerWorkspace,
                            /*expert_weights=*/mGemmProfilerExpertWeights, streamPtr->get());
                        check_cuda_error(cudaEventRecord(mEndEvent, streamPtr->get()));
                        check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
                    }

                    float ms;
                    check_cuda_error(cudaEventElapsedTime(&ms, mStartEvent, mEndEvent));
                    time += ms;

                    iter++;
                }
                // Get average time per iteration
                time /= static_cast<float>(iter);

                if (LOG_LEVEL >= VERBOSE)
                {
                    std::cout << "Tactic " << tidx << " for GEMM" << (int) gemm_to_profile << ":\n"
                              << t.toString() << "\ntook: " << time << "ms\n";
                }

                // Update the best
                if (time < best_time)
                {
                    best_idx = tidx;
                    best_time = time;
                }
            }
            catch (std::exception const& e)
            {
                // Sync to tidy up
                if (LOG_LEVEL >= ERROR)
                    std::cout << "Tactic failed to run with: " << e.what() << std::endl;
                check_cuda_error(cudaDeviceSynchronize());
                // skip invalid tactic
                continue;
            }
        }

        return best_idx;
    }

    int mBestTacticGemm1 = -1;
    int mBestTacticGemm2 = -1;

    std::pair<int, int> setTactic(
        int tactic_idx1, int tactic_idx2, MOEParallelismConfig parallelism_config, GemmToProfile gemm_to_profile)
    {
        auto tactics1 = mMoERunner.getTactics(MoeGemmId::GEMM_1);
        auto tactics2 = mMoERunner.getTactics(MoeGemmId::GEMM_2);
        std::vector<std::pair<std::reference_wrapper<int>, GemmToProfile>> tactics_to_profile{
            {tactic_idx1, GemmToProfile::GEMM_1}, {tactic_idx2, GemmToProfile::GEMM_2}};
        for (auto& combo : tactics_to_profile)
        {
            auto& t = combo.first.get();
            auto& tactics = combo.second == GemmToProfile::GEMM_1 ? tactics1 : tactics2;
            if (combo.second != gemm_to_profile && gemm_to_profile != GemmToProfile::LAYER)
            {
                t = 0; // Unneeded tactic, set to 0
                continue;
            }
            if (t == -1)
            {
                t = pickBestTactic(parallelism_config, combo.second);
            }

            if (t < 0 || t >= tactics.size())
            {
                return {-1, -1};
            }
        }

        mMoERunner.setTactic(tactics1[tactic_idx1], tactics2[tactic_idx2]);
        mBestTacticGemm1 = tactic_idx1;
        mBestTacticGemm2 = tactic_idx2;
        return {tactic_idx1, tactic_idx2};
    }

    void runMoEPermute(MOEParallelismConfig parallelism_config, GemmToProfile gemm_to_profile)
    {
        switch (gemm_to_profile)
        {
        case GemmToProfile::GEMM_1:
        case GemmToProfile::GEMM_2:
        {
            auto tactic_idx = gemm_to_profile == GemmToProfile::GEMM_1 ? mBestTacticGemm1 : mBestTacticGemm2;
            auto* expert_weights = gemm_to_profile == GemmToProfile::GEMM_1 ? mExpertWeight1 : mExpertWeight2;
            auto expert_weights_size
                = gemm_to_profile == GemmToProfile::GEMM_1 ? mExpertWeight1Size : mExpertWeight2Size;

            auto tactics = mMoERunner.getTactics(static_cast<MoeGemmId>(gemm_to_profile))[tactic_idx];
            if (static_cast<int>(gemm_to_profile) != static_cast<int>(mGemmProfilerBackend.mGemmToProfile))
            {
                throw std::runtime_error("Configuration mismatch between mGemmProfilerBackend and runMoEPermute");
            }
            mGemmProfilerBackend.mSampleIndex = mBufferIndex % mGemmProfilerBackend.NUM_ROUTING_SAMPLES;
            mGemmProfilerBackend.runProfiler(mTotalTokens, tactics,
                mGemmProfilerWorkspace + mGemmProfilerWorkspaceSize * (mBufferIndex % mNumGemmProfilerBuffers),
                /*expert_weights=*/expert_weights + expert_weights_size * mBufferIndex, streamPtr->get());
            break;
        }
        case GemmToProfile::LAYER:
        {
            auto stream = streamPtr->get();
            MoeMinLatencyParams min_latency_params;
#ifdef USING_OSS_CUTLASS_MOE_GEMM
            mMoERunner.runMoe(mInputTensor + mInputTensorSize * (mBufferIndex % mNumInputBuffers), nullptr, true,
                mSelectedExperts + mSelectedExpertsSize * mBufferIndex,
                mUseFinalScale ? mScaleProbs + mScaleProbsSize * mBufferIndex : nullptr,
                mExpertWeight1 + mExpertWeight1Size * mBufferIndex, mExpertBias1 + mExpertBias1Size * mBufferIndex,
                ActivationParams(mActType), mExpertWeight2 + mExpertWeight2Size * mBufferIndex,
                mExpertBias2 + mExpertBias2Size * mBufferIndex, mQuantParams[mBufferIndex], mTotalTokens, mHiddenSize,
                mHiddenSize, mInterSize, mNumExperts, mK,
                mWorkspace + mWorkspaceSize * (mBufferIndex % mNumWorkspaceBuffers),
                mFinalOutput + mFinalOutputSize * (mBufferIndex % mNumInputBuffers),
                mSourceToExpandedMap + mSourceToExpandedMapSize * mBufferIndex, parallelism_config,
                /*enable_alltoall=*/false, mUseLora, mLoraParams[mBufferIndex],
                /*use_fp8_block_scaling=*/false, /*min_latency_mode=*/false, min_latency_params, stream);
#else
            mMoERunner.runMoe(mInputTensor + mInputTensorSize * (mBufferIndex % mNumInputBuffers), nullptr, true,
                mSelectedExperts + mSelectedExpertsSize * mBufferIndex,
                mUseFinalScale ? mScaleProbs + mScaleProbsSize * mBufferIndex : nullptr,
                mExpertWeight1 + mExpertWeight1Size * mBufferIndex, mExpertBias1 + mExpertBias1Size * mBufferIndex,
                ActivationParams(mActType), mExpertWeight2 + mExpertWeight2Size * mBufferIndex,
                mExpertBias2 + mExpertBias2Size * mBufferIndex, mQuantParams[mBufferIndex], mTotalTokens, mHiddenSize,
                mHiddenSize, mInterSize, mNumExperts, mK,
                mWorkspace + mWorkspaceSize * (mBufferIndex % mNumWorkspaceBuffers),
                mFinalOutput + mFinalOutputSize * (mBufferIndex % mNumInputBuffers),
                mSourceToExpandedMap + mSourceToExpandedMapSize * mBufferIndex, parallelism_config,
                /*enable_alltoall=*/false, mUseLora, mLoraParams[mBufferIndex],
                /*use_fp8_block_scaling=*/false, /*min_latency_mode=*/false, min_latency_params, stream);
#endif
            break;
        }
        }
    }

    void runBenchmark(benchmark::State& state);
};

template <class TypeTuple_>
void MixtureOfExpertsBenchmark<TypeTuple_>::runBenchmark(benchmark::State& state)
{
    NVTX3_SCOPED_RANGE(FullBenchmark);
    int const num_experts = state.range(0);
    int const top_k = state.range(1);
    int const hidden_size = state.range(2);
    int const inter_size = state.range(3);
    int const tp_size = state.range(4);
    int const ep_size = state.range(5);
    int const world_rank = state.range(6);
    int const num_tokens = state.range(7);
    mUseBias = state.range(8);
    mUseFinalScale = state.range(9);
    mActType = static_cast<ActivationType>(state.range(10));
    int tactic_idx1 = state.range(11);
    int tactic_idx2 = state.range(12);
    int const routing_config = state.range(13);
    GemmToProfile const gemm_to_profile = static_cast<GemmToProfile>(state.range(14));

    state.counters["num_experts"] = num_experts;
    state.counters["top_k"] = top_k;
    state.counters["hidden_size"] = hidden_size;
    state.counters["inter_size"] = inter_size;
    state.counters["tp_size"] = tp_size;
    state.counters["ep_size"] = ep_size;
    state.counters["world_rank"] = world_rank;
    state.counters["num_tokens"] = num_tokens;
    state.counters["use_bias"] = (int) mUseBias;
    state.counters["use_final_scale"] = (int) mUseFinalScale;
    state.counters["act_fn"] = (int) mActType;
    state.counters["routing_config"] = (int) routing_config;
    state.counters["dtype"] = (int) toDTypeID();
    state.counters["wtype"] = (int) toWTypeID();
    state.counters["gemm_to_profile"] = (int) gemm_to_profile;

    std::stringstream ss;
    ss << "Experts,K,Hidden,Inter,TP,EP,Rank,Tokens,Bias,Scale,Actfn,Tactic1,Tactic2,Gemm,Routing=";
    for (auto v : {num_experts, top_k, hidden_size, inter_size, tp_size, ep_size, world_rank, num_tokens,
             (int) mUseBias, (int) mUseFinalScale, (int) mActType, tactic_idx1, tactic_idx2, (int) gemm_to_profile})
    {
        ss << v << ",";
    }
    ss << routingConfigCache.at(routing_config)->getName();
    // state.SetLabel(ss.str());
    state.SetLabel(routingConfigCache.at(routing_config)->getName());

    // Always use EP size for moe config until we support TP+EP, we just divide the inter size for TP
    MOEParallelismConfig parallelism_config{tp_size, world_rank / ep_size, ep_size, world_rank % ep_size};
    initBuffersPermute(
        num_tokens, hidden_size, inter_size, num_experts, top_k, routing_config, parallelism_config, gemm_to_profile);

    // Parse the tactic, does checks for "auto" mode and out of range
    std::tie(tactic_idx1, tactic_idx2) = setTactic(tactic_idx1, tactic_idx2, parallelism_config, gemm_to_profile);
    if (tactic_idx1 < 0 || tactic_idx2 < 0)
    {
        state.SkipWithMessage("Out of range tactic");
        return;
    }
    auto tactics1 = mMoERunner.getTactics(MoeGemmId::GEMM_1);
    auto tactics2 = mMoERunner.getTactics(MoeGemmId::GEMM_2);
    if (LOG_LEVEL >= INFO)
    {
        std::cout << "Selected tactic #1: " << tactic_idx1 << "/" << tactics1.size() << "\n"
                  << tactics1[tactic_idx1].toString() << std::endl;
        std::cout << "Selected tactic #2: " << tactic_idx2 << "/" << tactics2.size() << "\n"
                  << tactics2[tactic_idx2].toString() << std::endl;
    }
    state.counters["tactic_idx1"] = tactic_idx1;
    state.counters["tactic_idx2"] = tactic_idx2;

    state.counters["t1_sm_version"] = tactics1[tactic_idx1].sm_version;
    state.counters["t1_tile_shape"] = tactics1[tactic_idx1].getTileConfigAsInt();
    state.counters["t1_cluster_shape"] = (int) tactics1[tactic_idx1].cluster_shape;
    state.counters["t1_dynamic_cluster_shape"] = (int) tactics1[tactic_idx1].dynamic_cluster_shape;
    state.counters["t1_fallback_cluster_shape"] = (int) tactics1[tactic_idx1].fallback_cluster_shape;
    state.counters["t1_epilogue_schedule"] = (int) tactics1[tactic_idx1].epilogue_schedule;

    state.counters["t2_sm_version"] = tactics2[tactic_idx2].sm_version;
    state.counters["t2_tile_shape"] = tactics2[tactic_idx2].getTileConfigAsInt();
    state.counters["t2_cluster_shape"] = (int) tactics2[tactic_idx2].cluster_shape;
    state.counters["t2_dynamic_cluster_shape"] = (int) tactics2[tactic_idx2].dynamic_cluster_shape;
    state.counters["t2_fallback_cluster_shape"] = (int) tactics2[tactic_idx2].fallback_cluster_shape;
    state.counters["t2_epilogue_schedule"] = (int) tactics2[tactic_idx2].epilogue_schedule;

    createGraph(parallelism_config, gemm_to_profile);

    {
        ::nvtx3::scoped_range nvtx(tensorrt_llm::common::nvtx::nextColor(), "BenchmarkRun " + ss.str());
        for (auto _ : state)
        {
            float ms = benchmarkLoop(parallelism_config, gemm_to_profile);
            state.SetIterationTime(ms / 1000.f);
        }
    }

    destroyGraph();

    state.SetItemsProcessed(state.iterations() * num_tokens);

    // Cleanup all the benchmark state
    managed_buffers.clear();
    check_cuda_error(cudaDeviceSynchronize());
}

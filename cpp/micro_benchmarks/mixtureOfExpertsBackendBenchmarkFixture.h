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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
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

static BufferManager::CudaStreamPtr streamPtr;
static std::unique_ptr<BufferManager> bufferManager;
static int deviceCount;
static char* workloadFile = nullptr;

constexpr bool VERBOSE = false;

namespace
{
// Abstract class for routing config
struct RoutingConfig
{
    virtual void setRouting(float* routing_output, int64_t num_experts, int64_t k, int64_t num_tokens) = 0;
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

    void setRouting(float* routing_output, int64_t num_experts, int64_t k, int64_t num_tokens) override
    {
        nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
        makeLoadBalancedRoutingConfiguration(routing_output, num_experts, num_tokens, k, type, streamPtr->get());
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
    std::mt19937_64 twister{0xD5};
    std::vector<float> probabilities;
    std::pair<int64_t, int64_t> shape;
    std::string name;

    RandomDistributionRoutingConfig(std::vector<float> const& in_probabilities, std::pair<int64_t, int64_t> shape,
        std::string name = "random_distribution")
        : probabilities(std::move(in_probabilities))
        , shape(std::move(shape))
        , name(std::move(name))
    {
        TLLM_CHECK_WITH_INFO(shape.second == probabilities.size(),
            "Cannot create random routing distribution. Number of experts does not match the number of weights");
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

    void setRouting(float* routing_output, int64_t num_experts, int64_t k, int64_t num_tokens) override
    {
        TLLM_CHECK(num_experts == probabilities.size());
        std::vector<float> input(num_experts * num_tokens, 0);
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
            for (auto selection : selected)
            {
                input[token * num_experts + selection] = 1.f;
            }
        }
        check_cuda_error(cudaMemcpyAsync(
            routing_output, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, streamPtr->get()));
        check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
    }

    bool supportsConfig(int64_t num_experts, int64_t, int64_t) const override
    {
        return num_experts == shape.second;
    }
};

/**
 * Generates routing values by sampling a uniform distribution [-1,1)
 */
struct UniformRoutingConfig : public RoutingConfig
{
    std::mt19937_64 twister{0xD5};

    std::string getName() override
    {
        return "uniform";
    }

    bool isDeterministic() const override
    {
        return false;
    }

    void setRouting(float* routing_output, int64_t num_experts, int64_t k, int64_t num_tokens) override
    {
        std::uniform_real_distribution<float> dist(-1, 1);
        std::vector<float> input(num_experts * num_tokens);
        std::generate(input.begin(), input.end(), [&] { return dist(twister); });
        check_cuda_error(cudaMemcpyAsync(
            routing_output, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, streamPtr->get()));
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
    std::vector<float> config;
    std::pair<int64_t, int64_t> shape;
    std::string name;

    VectoredRoutingConfig(std::vector<float> config, std::pair<int64_t, int64_t> shape, std::string name = "vectored")
        : config(config)
        , shape(shape)
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

    void setRouting(float* routing_output, int64_t num_experts, int64_t k, int64_t num_tokens) override
    {
        assert(shape.second == num_experts);
        for (int64_t i = 0; i < num_tokens; i += shape.first)
        {
            int num_to_copy = std::min(num_tokens - i, shape.first);
            check_cuda_error(cudaMemcpyAsync(routing_output + i * num_experts, config.data(),
                num_to_copy * num_experts * sizeof(float), cudaMemcpyHostToDevice, streamPtr->get()));
        }
        check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
    }

    bool supportsConfig(int64_t num_experts, int64_t, int64_t) const override
    {
        return shape.second == num_experts;
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

template <class TypeTuple_>
class MixtureOfExpertsBenchmark : public ::benchmark::Fixture
{
public:
    using DataType = typename TypeTuple_::DataType;
    using WeightType = typename TypeTuple_::WeightType;
    using OutputType = typename TypeTuple_::OutputType;
    constexpr static bool INT4 = std::is_same_v<WeightType, cutlass::uint4b_t>;
    constexpr static bool FP8 = std::is_same_v<DataType, SafeFP8>;
    constexpr static bool INT_QUANT = !std::is_same_v<DataType, WeightType>;
    using WeightStorage = std::conditional_t<INT_QUANT, uint8_t, WeightType>;
    constexpr static int WEIGHT_ELEM_PER_BYTE = INT4 ? 2 : 1;
    int const BASE_HIDDEN_SIZE = 64 / sizeof(WeightType) * WEIGHT_ELEM_PER_BYTE;

    std::vector<BufferManager::IBufferPtr> managed_buffers;
    float* mInputProbabilities{};
    DataType* mInputTensor{};

    int64_t mHiddenSize{};
    int64_t mNumExperts{};
    int64_t mK{};

    constexpr static nvinfer1::DataType toDTypeID()
    {
        if (FP8)
            return nvinfer1::DataType::kFP8;
        if (INT_QUANT && INT4)
            return nvinfer1::DataType::kUINT8; // Hack to distinguish int4, use unsigned
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
        return nvinfer1::DataType::kBOOL;
    };

    static bool shouldSkip()
    {
#ifndef ENABLE_FP8
        static_assert(!FP8, "FP8 Tests enabled on unsupported CUDA version");
#endif
        bool should_skip_unsupported_fp8 = getSMVersion() < 90 && FP8;
        return should_skip_unsupported_fp8;
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
            s.SkipWithMessage("GPU does not support FP8");
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

    CutlassMoeFCRunner<DataType, WeightType, OutputType> mMoERunner{};
    char* mWorkspace{};
    float* mScaleProbs{};
    WeightStorage* mExpertWeight1{};
    WeightStorage* mExpertWeight2{};
    DataType* mExpertIntScale1{};
    DataType* mExpertIntScale2{};

    float* mExpertFP8Scale1{};
    float* mExpertFP8Scale2{};
    float* mExpertFP8Scale3{};

    DataType* mExpertBias1{};
    DataType* mExpertBias2{};

    OutputType* mFinalOutput{};
    int* mSourceToExpandedMap;
    int* mSelectedExpert;
    int64_t mInterSize{};
    int64_t mTotalTokens{};

    int mRoutingConfigIndex = 0;

    bool mUseBias = true;

    bool mIsGated = false;
    int mGatedMultiplier = 1;

    tensorrt_llm::ActivationType mActType = tensorrt_llm::ActivationType::Relu;
    MOEExpertScaleNormalizationMode mNormMode = MOEExpertScaleNormalizationMode::NONE;

    QuantParams mQuantParams{};

    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> mSelectedConfig = std::nullopt;

    template <class T>
    T* allocBuffer(size_t size)
    {
        auto i_buffer = bufferManager->gpu(size * sizeof(T));
        check_cuda_error(cudaGetLastError());
        managed_buffers.emplace_back(std::move(i_buffer));
        T* ptr = static_cast<T*>(managed_buffers.back()->data());
        check_cuda_error(cudaMemsetAsync(ptr, 0x0, size * sizeof(T), streamPtr->get()));
        return ptr;
    }

    void initBuffersPermute(int64_t num_tokens, int64_t hidden_size, int64_t inter_size, int64_t num_experts, int64_t k,
        int64_t routing_config, MOEParallelismConfig parallelism_config)
    {
        assert(hidden_size % BASE_HIDDEN_SIZE == 0);

        managed_buffers.clear();

        mTotalTokens = num_tokens;
        mHiddenSize = hidden_size;
        mInterSize = inter_size / parallelism_config.tp_size;
        mNumExperts = num_experts;
        mK = k;
        mIsGated = tensorrt_llm::isGatedActivation(mActType);
        mGatedMultiplier = mIsGated ? 2 : 1;
        auto const gated_inter = mInterSize * mGatedMultiplier;

        size_t workspace_size
            = mMoERunner.getWorkspaceSize(mTotalTokens, mHiddenSize, mInterSize, mNumExperts, mK, mActType, {});

        mWorkspace = allocBuffer<char>(workspace_size);
        size_t const expert_matrix_size = mNumExperts * mHiddenSize * mInterSize;

        mExpertWeight1
            = allocBuffer<WeightStorage>(expert_matrix_size * mGatedMultiplier / WEIGHT_ELEM_PER_BYTE - 8192);
        mExpertWeight2 = allocBuffer<WeightStorage>(expert_matrix_size / WEIGHT_ELEM_PER_BYTE - 8192);

        mExpertBias1 = nullptr;
        mExpertBias2 = nullptr;
        if (mUseBias)
        {
            mExpertBias1 = allocBuffer<DataType>(mNumExperts * gated_inter);
            mExpertBias2 = allocBuffer<DataType>(mNumExperts * mHiddenSize);
        }

        if constexpr (INT_QUANT)
        {
            mExpertIntScale1 = allocBuffer<DataType>(mNumExperts * gated_inter);
            mExpertIntScale2 = allocBuffer<DataType>(mNumExperts * mHiddenSize);

            mQuantParams = QuantParams::Int(mExpertIntScale1, mExpertIntScale2);
        }
        else if constexpr (FP8)
        {
            mExpertFP8Scale1 = allocBuffer<float>(mNumExperts);
            mExpertFP8Scale2 = allocBuffer<float>(1);
            mExpertFP8Scale3 = allocBuffer<float>(mNumExperts);

            mQuantParams = QuantParams::FP8(mExpertFP8Scale1, mExpertFP8Scale2, mExpertFP8Scale3);
        }

        mInputProbabilities = allocBuffer<float>(mTotalTokens * mNumExperts);
        mScaleProbs = allocBuffer<float>(mTotalTokens * mK);
        mInputTensor = allocBuffer<DataType>(mTotalTokens * mHiddenSize);
        mFinalOutput = allocBuffer<OutputType>(mTotalTokens * mHiddenSize);

        mSourceToExpandedMap = allocBuffer<int>(mTotalTokens * mK);
        mSelectedExpert = allocBuffer<int>(mTotalTokens * mK);

        mRoutingConfigIndex = routing_config;
        auto tactic = routingConfigCache.at(routing_config);
        tactic->setRouting(mInputProbabilities, mNumExperts, mK, mTotalTokens);

        check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
    }

    float benchmarkLoop(MOEParallelismConfig parallelism_config)
    {
        auto tactic = routingConfigCache.at(mRoutingConfigIndex);
        if (!tactic->isDeterministic())
        {
            tactic->setRouting(mInputProbabilities, mNumExperts, mK, mTotalTokens);
        }

        {
            NVTX3_SCOPED_RANGE(BenchmarkLoopIteration);
            check_cuda_error(cudaEventRecord(mStartEvent, streamPtr->get()));
            runMoEPermute(parallelism_config);
            check_cuda_error(cudaEventRecord(mEndEvent, streamPtr->get()));
            check_cuda_error(cudaStreamSynchronize(streamPtr->get()));
        }

        float ms;
        check_cuda_error(cudaEventElapsedTime(&ms, mStartEvent, mEndEvent));
        return ms;
    }

    // An imprecise benchmark pass for picking the best tactic.
    // Runs for 3 iterations or 1 second and picks the best option
    int pickBestTactic(MOEParallelismConfig parallelism_config)
    {
        NVTX3_SCOPED_RANGE(WarmUpRun);
        auto tactics = mMoERunner.getTactics();

        float best_time = INFINITY;
        int best_idx = -1;
        for (int tidx = 0; tidx < tactics.size(); tidx++)
        {
            try
            {
                // Set the tactic
                auto const& t = tactics[tidx];
                mMoERunner.setTactic(t);

                // Warm-Up run
                benchmarkLoop(parallelism_config);

                float const max_time_ms = 1000.f;
                int const max_iters = 3;
                float time = 0.f;
                int iter = 0;
                while (iter < max_iters && time < max_time_ms)
                {
                    time += benchmarkLoop(parallelism_config);
                    iter++;
                }
                // Get average time per iteration
                time /= static_cast<float>(iter);

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
                if (VERBOSE)
                    std::cout << "Tactic failed to run with: " << e.what() << std::endl;
                check_cuda_error(cudaDeviceSynchronize());
                // skip invalid tactic
                continue;
            }
        }

        // Check if all tactics failed
        if (best_idx < 0)
            return -1;

        auto const& best_tactic = tactics[best_idx];
        mMoERunner.setTactic(best_tactic);
        return best_idx;
    }

    int setTactic(int tactic_idx, MOEParallelismConfig parallelism_config)
    {
        if (tactic_idx == -1)
        {
            return pickBestTactic(parallelism_config);
        }

        auto tactics = mMoERunner.getTactics();
        if (tactic_idx < 0 || tactic_idx >= tactics.size())
        {
            return -1;
        }
        auto selected_tactic = tactics[tactic_idx];
        mMoERunner.setTactic(selected_tactic);
        return tactic_idx;
    }

    void runMoEPermute(MOEParallelismConfig parallelism_config)
    {
        auto stream = streamPtr->get();
        mMoERunner.runMoe(mInputTensor, mInputProbabilities, mExpertWeight1, mExpertBias1, mActType, mExpertWeight2,
            mExpertBias2, mQuantParams, mTotalTokens, mHiddenSize, mInterSize, mNumExperts, mK, mWorkspace,
            mFinalOutput, nullptr, mTotalTokens, mScaleProbs, mSourceToExpandedMap, mSelectedExpert, parallelism_config,
            mNormMode, stream);
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
    mActType = static_cast<tensorrt_llm::ActivationType>(state.range(9));
    mNormMode = static_cast<MOEExpertScaleNormalizationMode>(state.range(10));
    int tactic_idx = state.range(11);
    int const routing_config = state.range(12);

    state.counters["num_experts"] = num_experts;
    state.counters["top_k"] = top_k;
    state.counters["hidden_size"] = hidden_size;
    state.counters["inter_size"] = inter_size;
    state.counters["tp_size"] = tp_size;
    state.counters["ep_size"] = ep_size;
    state.counters["world_rank"] = world_rank;
    state.counters["num_tokens"] = num_tokens;
    state.counters["use_bias"] = (int) mUseBias;
    state.counters["act_fn"] = (int) mActType;
    state.counters["norm_mode"] = (int) mNormMode;
    state.counters["routing_config"] = (int) routing_config;
    state.counters["dtype"] = (int) toDTypeID();

    std::stringstream ss;
    ss << "Experts,K,Hidden,Inter,TP,EP,Rank,Tokens,Bias,Actfn,Norm Mode,Tactic,Routing=";
    for (auto v : {num_experts, top_k, hidden_size, inter_size, tp_size, ep_size, world_rank, num_tokens,
             (int) mUseBias, (int) mActType, (int) mNormMode, tactic_idx})
    {
        ss << v << ",";
    }
    ss << routingConfigCache.at(routing_config)->getName();
    // state.SetLabel(ss.str());
    state.SetLabel(routingConfigCache.at(routing_config)->getName());

    // Always use EP size for moe config until we support TP+EP, we just divide the inter size for TP
    MOEParallelismConfig parallelism_config{tp_size, world_rank / ep_size, ep_size, world_rank % ep_size};
    initBuffersPermute(num_tokens, hidden_size, inter_size, num_experts, top_k, routing_config, parallelism_config);

    // Parse the tactic, does checks for "auto" mode and out of range
    tactic_idx = setTactic(tactic_idx, parallelism_config);
    if (tactic_idx < 0)
    {
        state.SkipWithMessage("Out of range tactic");
        return;
    }
    if (VERBOSE)
    {
        auto tactics = mMoERunner.getTactics();
        std::cout << "Selected " << tactic_idx << "/" << tactics.size() << "\n"
                  << tactics[tactic_idx].toString() << std::endl;
    }
    state.counters["tactic_idx"] = tactic_idx;

    {
        NVTX3_SCOPED_RANGE(BenchmarkRun);
        for (auto _ : state)
        {
            float ms = benchmarkLoop(parallelism_config);
            state.SetIterationTime(ms / 1000.f);
        }
    }

    state.SetItemsProcessed(state.iterations() * num_tokens);

    // Cleanup all the benchmark state
    managed_buffers.clear();
    check_cuda_error(cudaDeviceSynchronize());
}

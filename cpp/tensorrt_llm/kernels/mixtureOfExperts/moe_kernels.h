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
#include "cutlass/gemm/gemm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include <cuda_runtime_api.h>
#include <optional>
#include <random>

namespace tensorrt_llm::kernels
{

static inline size_t pad_to_multiple_of_16(size_t const& input)
{
    static constexpr int ALIGNMENT = 16;
    return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

class CubKeyValueSorter
{
public:
    CubKeyValueSorter();

    CubKeyValueSorter(int const num_experts);

    void updateNumExperts(int const num_experts);

    static size_t getWorkspaceSize(size_t const num_key_value_pairs, int const num_experts);

    void run(void* workspace, size_t const workspace_size, int const* keys_in, int* keys_out, int const* values_in,
        int* values_out, size_t const num_key_value_pairs, cudaStream_t stream);

private:
    int num_experts_;
    int num_bits_;
};

enum class MOEExpertScaleNormalizationMode : int
{
    NONE = 0,    //!< Run the softmax on all scales and select the topk
    RENORMALIZE, //!< Renormalize the selected scales so they sum to one. This is equivalent to only running softmax on
                 //!< the topk selected experts
};

/**
 * \brief Describes what parallelism mode the MoE is using
 *
 * Tensor Parallelism refers to the mode where the weight matrices for each expert are sliced up between nodes.
 * Each node will handle part of each expert, the final result is achieved by summing the result.
 * The inter_size dimension should be divided by the number of nodes prior to passing it to the MoE plugin, only the
 * required slice of the weights should be provided to the plugin FC1 is a ColumnLinear and FC2 is a RowLinear, see
 * tensorrt_llm/mlp/mlp.py for an example of how this works for a single MLP
 *
 * NOTE: The bias for fc2 is only applied on rank 0. If we added it on all nodes the allreduce() would contain multiple
 * copies of the bias. The bias on other node will be ignored, and may be set to nullptr
 *
 * Expert Parallelism refers to the mode where experts are divided between the nodes. Each node will handle only the
 * tokens that are routed to the experts it is assigned to. Only the weights for the node's experts should be provided
 * to the plugin For example, with #experts = 8, expert parallelism = 2: Node 0 would handle experts 0-3, and node 1
 * would handle experts 4-7
 *
 * Regardless of parallelism mode:
 *  * The input routing values must be the complete routing for all tokens/experts (required for softmax)
 *  * An allreduce must be run on the result to combine the results from different nodes if parallelism > 1
 */
struct MOEParallelismConfig
{
    int tp_size = 1;
    int tp_rank = 0;
    int ep_size = 1;
    int ep_rank = 0;

    MOEParallelismConfig() = default;

    MOEParallelismConfig(int tp_size, int tp_rank, int ep_size, int ep_rank)
        : tp_size(tp_size)
        , tp_rank(tp_rank)
        , ep_size(ep_size)
        , ep_rank(ep_rank)
    {
        // Do some basic sanity checks
        TLLM_CHECK(tp_rank < tp_size);
        TLLM_CHECK(tp_rank >= 0);
        TLLM_CHECK(tp_size >= 1);
        TLLM_CHECK(ep_rank < ep_size);
        TLLM_CHECK(ep_rank >= 0);
        TLLM_CHECK(ep_size >= 1);
    }

    bool operator==(MOEParallelismConfig const& other) const
    {
        return tp_size == other.tp_size && tp_rank == other.tp_rank && ep_size == other.ep_size
            && ep_rank == other.ep_rank;
    }

    friend std::ostream& operator<<(std::ostream& os, MOEParallelismConfig const& config)
    {
        os << "tp_size: " << config.tp_size << ", tp_rank: " << config.tp_rank << ", ep_size: " << config.ep_size
           << ", ep_rank: " << config.ep_rank;
        return os;
    }
};

struct QuantParams
{
    // Int weight only quantization params
    void const* fc1_weight_scales = nullptr;
    void const* fc2_weight_scales = nullptr;

    // FP8 quantization params
    float const* dequant_fc1 = nullptr;
    float const* quant_fc2 = nullptr;
    float const* dequant_fc2 = nullptr;
    float const* quant_final = nullptr;

    static QuantParams FP8(
        float const* dequant_fc1, float const* quant_fc2, float const* dequant_fc2, float const* quant_final = nullptr)
    {
        return QuantParams{nullptr, nullptr, dequant_fc1, quant_fc2, dequant_fc2, quant_final};
    }

    static QuantParams Int(void const* fc1_weight_scales, void const* fc2_weight_scales)
    {
        return QuantParams{fc1_weight_scales, fc2_weight_scales, nullptr, nullptr, nullptr, nullptr};
    }
};

class CutlassMoeFCRunnerInterface
{
public:
    virtual ~CutlassMoeFCRunnerInterface() = default;
    virtual size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
        int const num_experts, int const k, ActivationType activation_type,
        MOEParallelismConfig parallelism_config) const
        = 0;
    virtual void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
        std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config)
        = 0;
    virtual std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() = 0;

    virtual void runMoe(void const* input_activations, float const* gating_output, void const* fc1_expert_weights,
        void const* fc1_expert_biases, ActivationType fc1_activation_type, void const* fc2_expert_weights,
        void const* fc2_expert_biases, QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size,
        int64_t const inter_size, int const num_experts, int const k, char* workspace_ptr, void* final_output,
        bool const* finished, int64_t const active_rows, void* expert_scales,
        int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row,
        MOEParallelismConfig parallelism_config, MOEExpertScaleNormalizationMode normalization_mode,
        cudaStream_t stream)
        = 0;

    // Aliases for profiling the gemms
    virtual void gemm1(void const* const input, void* const output, void* const intermediate_result,
        int64_t const* const total_rows_before_expert, HopperGroupedGemmInput hopper_input_template,
        void const* const fc1_expert_weights, void const* const fc1_expert_biases,
        int64_t const* const num_valid_tokens_ptr, void const* const fc1_int_scales, float const* const fc1_fp8_dequant,
        float const* const fc2_fp8_quant, int64_t const expanded_num_rows, int64_t const hidden_size,
        int64_t const inter_size, int const num_experts_per_node, ActivationType fc1_activation_type,
        cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config)
        = 0;

    virtual void gemm2(void const* const input, void* const output, int64_t const* const total_rows_before_expert,
        HopperGroupedGemmInput hopper_input_template, void const* const fc2_expert_weights,
        void const* const fc2_int_scales, float const* const fc2_fp8_dequant, int64_t const expanded_num_rows,
        int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node, cudaStream_t stream,
        cutlass_extensions::CutlassGemmConfig config)
        = 0;

    virtual size_t getGemmWorkspaceSize(int num_experts) const = 0;

    bool is_profiler = false;
};

// Assumes inputs activations are row major. Weights need to be preprocessed by th_op/weight_quantize.cc .
// Nested in a class to avoid multiple calls to cudaGetDeviceProperties as this call can be expensive.
// Avoid making several duplicates of this class.
template <typename T,        /*The type used for activations/scales/compute*/
    typename WeightType,     /* The type for the MoE weights */
    typename OutputType = T, /* The type for the MoE final output */
    typename Enable = void>
class CutlassMoeFCRunner : public CutlassMoeFCRunnerInterface
{
    using Self = CutlassMoeFCRunner<T, WeightType, OutputType>;

public:
    CutlassMoeFCRunner() = default;

    ~CutlassMoeFCRunner() override = default;

    static_assert(
        std::is_same_v<T, WeightType> || !std::is_same_v<T, float>, "Does not support float with quantized weights");

    size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size, int64_t const fc1_output_size,
        int const num_experts, int const k, ActivationType activation_type,
        MOEParallelismConfig parallelism_config) const override;

    void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
        std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config) override
    {
        gemm1_config_ = std::move(gemm1_config);
        gemm2_config_ = std::move(gemm2_config);
    }

    std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() override
    {
        return moe_gemm_runner_.getConfigs();
    }

    static std::vector<cutlass_extensions::CutlassGemmConfig> getTactics(int sm)
    {
        using RunnerType = decltype(moe_gemm_runner_);
        return RunnerType::getConfigs(sm);
    }

    void runMoe(void const* input_activations, float const* gating_output, void const* fc1_expert_weights,
        void const* fc1_expert_biases, ActivationType fc1_activation_type, void const* fc2_expert_weights,
        void const* fc2_expert_biases, QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size,
        int64_t const inter_size, int const num_experts, int const k, char* workspace_ptr, void* final_output,
        bool const* finished, int64_t const active_rows, void* expert_scales,
        int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row,
        MOEParallelismConfig parallelism_config, MOEExpertScaleNormalizationMode normalization_mode,
        cudaStream_t stream) override;

    // We make these GEMM1 & GEMM2 static because they need to be stateless for the profiler to work
    static void gemm1(MoeGemmRunner<T, WeightType>& gemm_runner, T const* const input, T* const output,
        void* const intermediate_result, int64_t const* const total_rows_before_expert,
        HopperGroupedGemmInput const hopper_input_template, WeightType const* const fc1_expert_weights,
        T const* const fc1_expert_biases, int64_t const* const num_valid_tokens_ptr, T const* const fc1_int_scales,
        float const* const fc1_fp8_dequant, float const* const fc2_fp8_quant, int64_t const expanded_num_rows,
        int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
        ActivationType fc1_activation_type, cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config);

    static void gemm2(MoeGemmRunner<T, WeightType>& gemm_runner, T const* const input, void* const output,
        int64_t const* const total_rows_before_expert, HopperGroupedGemmInput hopper_input_template,
        WeightType const* const fc2_expert_weights, T const* const fc2_int_scales, float const* const fc2_fp8_dequant,
        int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
        int const num_experts_per_node, cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config);

    // Overrides to allow us to forward on to the internal functions with the pointers using the correct type
    void gemm1(void const* const input, void* const output, void* const intermediate_result,
        int64_t const* const total_rows_before_expert, HopperGroupedGemmInput hopper_input_template,
        void const* const fc1_expert_weights, void const* const fc1_expert_biases,
        int64_t const* const num_valid_tokens_ptr, void const* const fc1_int_scales, float const* const fc1_fp8_dequant,
        float const* const fc2_fp8_quant, int64_t const expanded_num_rows, int64_t const hidden_size,
        int64_t const inter_size, int const num_experts_per_node, ActivationType fc1_activation_type,
        cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config) override
    {
        return Self::gemm1(moe_gemm_runner_, static_cast<T const*>(input), static_cast<T*>(output), intermediate_result,
            total_rows_before_expert, hopper_input_template, static_cast<WeightType const*>(fc1_expert_weights),
            static_cast<T const*>(fc1_expert_weights), num_valid_tokens_ptr, static_cast<T const*>(fc1_int_scales),
            fc1_fp8_dequant, fc2_fp8_quant, expanded_num_rows, hidden_size, inter_size, num_experts_per_node,
            fc1_activation_type, stream, config);
    }

    void gemm2(void const* const input, void* const output, int64_t const* const total_rows_before_expert,
        HopperGroupedGemmInput hopper_input_template, void const* const fc2_expert_weights,
        void const* const fc2_int_scales, float const* const fc2_fp8_dequant, int64_t const expanded_num_rows,
        int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node, cudaStream_t stream,
        cutlass_extensions::CutlassGemmConfig config) override
    {
        return Self::gemm2(moe_gemm_runner_, static_cast<T const*>(input), output, total_rows_before_expert,
            hopper_input_template, static_cast<WeightType const*>(fc2_expert_weights),
            static_cast<T const*>(fc2_int_scales), fc2_fp8_dequant, expanded_num_rows, hidden_size, inter_size,
            num_experts_per_node, stream, config);
    }

    virtual size_t getGemmWorkspaceSize(int num_experts) const override
    {
        return moe_gemm_runner_.getMaxWorkspaceSize(num_experts);
    }

private:
    using HopperGemmOutputType = typename HopperGroupedGemmInput::OutputTypeAdaptor_t<T>;

    static void computeTotalRowsBeforeExpert(int const* sorted_indices, int const total_indices, int const num_experts,
        int64_t* total_rows_before_expert, cudaStream_t stream);
    static HopperGroupedGemmInput computeStridesHopper(int64_t const* total_rows_before_expert,
        HopperGroupedGemmInput layout_info, int64_t gemm_n, int64_t gemm_k, int const num_experts, T const* in,
        WeightType const* weights, float const* fp8_dequant, T const* bias, HopperGemmOutputType* output,
        cudaStream_t stream);
    std::vector<size_t> getWorkspaceBufferSizes(int64_t const num_rows, int64_t const hidden_size,
        int64_t const inter_size, int const num_experts, int const num_experts_per_node, int const k,
        ActivationType activation_type) const;
    void configureWsPtrs(char* ws_ptr, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
        int const num_experts, int const num_experts_per_node, int const k, ActivationType activation_type);

private:
    bool mayHaveDifferentGEMMOutputType() const
    {
        // We just check if its supported because we need to know when calculating workspace size
        return moe_gemm_runner_.supportsHopperSpecialisation() && !std::is_same_v<T, HopperGemmOutputType>;
    }

    CubKeyValueSorter sorter_;
    MoeGemmRunner<T, WeightType> moe_gemm_runner_;

    std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config_;
    std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config_;

    // Pointers
    int* source_rows_{};
    int* permuted_rows_{};
    int* permuted_experts_{};
    char* sorter_ws_{};
    T* permuted_data_{};
    float* softmax_out_{};

    int64_t* total_rows_before_expert_{};

    void* glu_inter_result_{};
    void* fc2_result_{};
    T* fc1_result_{};

    HopperGroupedGemmInput hopper_grouped_gemm_input_;
};

void makeLoadBalancedRoutingConfiguration(
    void* data_void, int num_experts, int num_tokens, int k, nvinfer1::DataType type, cudaStream_t stream);

struct GemmProfilerBackend
{
public:
    using Config = cutlass_extensions::CutlassGemmConfig;
    enum class GemmToProfile
    {
        Undefined = 0,
        GEMM_1,
        GEMM_2
    };

    void init(CutlassMoeFCRunnerInterface& runner, GemmToProfile gemm_to_profile, nvinfer1::DataType dtype,
        nvinfer1::DataType wtype, int num_experts, int k, int64_t hidden_size, int64_t inter_size,
        ActivationType activation_type, bool bias, MOEParallelismConfig parallelism_config)
    {
        mInterface = &runner;
        mGemmToProfile = gemm_to_profile;
        mDType = dtype;
        mWType = wtype;
        mNumExperts = num_experts;
        mNumExpertsPerNode = num_experts / parallelism_config.ep_size;
        mK = k;
        mExpertHiddenSize = hidden_size;
        mExpertInterSize = inter_size;
        mActivationType = activation_type;
        mBias = bias;
        mParallelismConfig = parallelism_config;
        mSM = common::getSMVersion();
        // Reset the seed so the distributions are reproducible
        mTwister = std::mt19937_64{0xD5};
    }

    void prepare(int num_tokens, char* workspace, cudaStream_t stream);

    std::vector<size_t> getProfilerWorkspaces(int maxM, bool is_hopper);
    size_t getWorkspaceSize(int maxM);

    void runProfiler(int num_tokens, Config const& tactic, char* workspace_ptr_char, cudaStream_t const& stream);

    CutlassMoeFCRunnerInterface* mInterface;
    GemmToProfile mGemmToProfile = GemmToProfile::Undefined;
    std::vector<Config> mAllTacticsSaved;
    int mSM{};
    int64_t mNumExperts{};
    int64_t mNumExpertsPerNode{};
    int64_t mK{};
    int64_t mExpertHiddenSize{};
    int64_t mExpertInterSize{};
    ActivationType mActivationType{};
    MOEParallelismConfig mParallelismConfig{};

    int mSampleIndex = 0;

    std::mt19937_64 mTwister{};

    nvinfer1::DataType mDType;
    nvinfer1::DataType mWType;

    // This will be a unique value for every iteration of warmup and actual bench
    constexpr static int NUM_ROUTING_SAMPLES = 16;

    bool mBias{};
};

} // namespace tensorrt_llm::kernels

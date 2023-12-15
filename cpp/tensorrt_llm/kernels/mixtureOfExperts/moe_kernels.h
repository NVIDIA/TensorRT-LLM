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
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include <cuda_runtime_api.h>
#include <optional>

namespace tensorrt_llm::kernels
{

static inline size_t pad_to_multiple_of_16(const size_t& input)
{
    static constexpr int ALIGNMENT = 16;
    return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

/*
  Launches the topk gating softmax required for the MoE layers.

  Params:
  input - a [num_rows x num_experts]
  finished - [num_rows] vector with 1 if the sentence at this row is done translating and 0 otherwise.
  output - a buffer of shape [num_rows x k] containing the top-k values of the softmax for each row.
  indices - a matrix of shape [num_rows x k] containing the top-k experts each row should get routed to.
  source_rows - a matrix of shape [num_rows x k] used internally for permuting. source_rows[row][k] =  k * num_rows +
  row. It is constructed like this so we can track where each of the original rows end up in order to perform the
                "k-way" reduction later in the routing.

  num_rows - The number of rows in the matrix
  num_experts - The number of expert layers present
  k - k value in topk
*/
template <typename T>
void topk_gating_softmax_kernelLauncher(const T* input, const bool* finished, T* output, T* softmax_temp_out,
    int* indices, int* source_row, const int num_rows, const int num_experts, const int k, cudaStream_t stream);

class CubKeyValueSorter
{
public:
    CubKeyValueSorter();

    CubKeyValueSorter(const int num_experts);

    void updateNumExperts(const int num_experts);

    static size_t getWorkspaceSize(const size_t num_key_value_pairs, const int num_experts);

    void run(void* workspace, const size_t workspace_size, const int* keys_in, int* keys_out, const int* values_in,
        int* values_out, const size_t num_key_value_pairs, cudaStream_t stream);

private:
    int num_experts_;
    int num_bits_;
};

enum class MOEParallelismMode : int
{
    NONE = 0,           //!< Ignore parallelism and duplicate the work across all nodes
    EXPERT_PARALLELISM, //!< Divide the experts between each node. The number of experts must be a multiple of
                        //!< parallelism
    TENSOR_PARALLELISM, //!< Divide the weight matrices between the nodes. The hidden dimension must be a multiple of
                        //!< parallelism
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
    constexpr static MOEParallelismConfig TensorParallelism(int tp_size, int tp_rank)
    {
        return {tp_size, tp_rank, 1, 0};
    }

    constexpr static MOEParallelismConfig ExpertParallelism(int ep_size, int ep_rank)
    {
        return {1, 0, ep_size, ep_rank};
    }

    const int tp_size = 1;
    const int tp_rank = 0;
    const int ep_size = 1;
    const int ep_rank = 0;
};

class CutlassMoeFCRunnerInterface
{
public:
    virtual ~CutlassMoeFCRunnerInterface() = default;
    virtual size_t getWorkspaceSize(const int num_rows, const int hidden_size, const int fc1_output_size,
        const int num_experts, const int k, ActivationType activation_type,
        MOEParallelismConfig parallelism_config) const
        = 0;
    virtual void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm_config) = 0;
    virtual std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() = 0;

    virtual void runMoe(const void* input_activations, const float* gating_output, const void* fc1_expert_weights,
        const void* fc1_scales, const void* fc1_expert_biases, ActivationType fc1_activation_type,
        const void* fc2_expert_weights, const void* fc2_scales, const void* fc2_expert_biases, const int num_rows,
        const int hidden_size, const int inter_size, const int num_experts, const int k, char* workspace_ptr,
        void* final_output, void* fc2_result, const bool* finished, const int active_rows, void* expert_scales,
        int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row,
        MOEParallelismConfig parallelism_config, MOEExpertScaleNormalizationMode normalization_mode,
        cudaStream_t stream)
        = 0;
};

// Assumes inputs activations are row major. Weights need to be preprocessed by th_op/weight_quantize.cc .
// Nested in a class to avoid multiple calls to cudaGetDeviceProperties as this call can be expensive.
// Avoid making several duplicates of this class.
template <typename T,    /*The type used for activations/scales/compute*/
    typename WeightType, /* The type for the MoE weights */
    typename Enable = void>
class CutlassMoeFCRunner : public CutlassMoeFCRunnerInterface
{
public:
    CutlassMoeFCRunner() = default;
    ~CutlassMoeFCRunner() override = default;

    size_t getWorkspaceSize(const int num_rows, const int hidden_size, const int fc1_output_size, const int num_experts,
        const int k, ActivationType activation_type, MOEParallelismConfig parallelism_config) const override;

    void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm_config) override
    {
        moe_gemm_runner_.setBestConfig(std::move(gemm_config));
    }

    std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() override
    {
        return moe_gemm_runner_.getConfigs();
    }

    void runMoe(const void* input_activations, const float* gating_output, const void* fc1_expert_weights,
        const void* fc1_scales, const void* fc1_expert_biases, ActivationType fc1_activation_type,
        const void* fc2_expert_weights, const void* fc2_scales, const void* fc2_expert_biases, const int num_rows,
        const int hidden_size, const int inter_size, const int num_experts, const int k, char* workspace_ptr,
        void* final_output, void* fc2_result, const bool* finished, const int active_rows, void* expert_scales,
        int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row,
        MOEParallelismConfig parallelism_config, MOEExpertScaleNormalizationMode normalization_mode,
        cudaStream_t stream) override;

private:
    void computeTotalRowsBeforeExpert(const int* sorted_indices, const int total_indices, const int num_experts,
        int64_t* total_rows_before_expert, cudaStream_t stream);
    std::vector<size_t> getWorkspaceBufferSizes(const int num_rows, const int hidden_size, const int inter_size,
        const int num_experts, const int num_experts_per_node, const int k, ActivationType activation_type) const;
    void configureWsPtrs(char* ws_ptr, const int num_rows, const int hidden_size, const int inter_size,
        const int num_experts, const int num_experts_per_node, const int k, ActivationType activation_type);

private:
    CubKeyValueSorter sorter_;
    MoeGemmRunner<T, WeightType> moe_gemm_runner_;

    // Pointers
    int* source_rows_;
    int* permuted_rows_;
    int* permuted_experts_;
    char* sorter_ws_;
    T* permuted_data_;
    float* softmax_out_;

    int64_t* total_rows_before_expert_;

    T* fc1_result_;
    T* glu_inter_result_;
};

template <typename WeightType>
class CutlassMoeFCRunner<float, WeightType, typename std::enable_if_t<!std::is_same<float, WeightType>::value>>
    : public CutlassMoeFCRunnerInterface
{
public:
    CutlassMoeFCRunner() = default;

    size_t getWorkspaceSize(const int num_rows, const int hidden_size, const int fc1_output_size, const int num_experts,
        const int k, ActivationType activation_type, MOEParallelismConfig parallelism_config) const override
    {
        return 0;
    }

    void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm_config) override
    {
        return;
    }

    void runMoe(const void* input_activations, const float* gating_output, const void* fc1_expert_weights,
        const void* fc1_scales, const void* fc1_expert_biases, ActivationType fc1_activation_type,
        const void* fc2_expert_weights, const void* fc2_scales, const void* fc2_expert_biases, const int num_rows,
        const int hidden_size, const int inter_size, const int num_experts, const int k, char* workspace_ptr,
        void* final_output, void* fc2_result, const bool* finished, const int active_rows, void* expert_scales,
        int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row,
        MOEParallelismConfig parallelism_config, MOEExpertScaleNormalizationMode normalization_mode,
        cudaStream_t stream) override
    {
        TLLM_THROW("FP32 MoE with different precision weights is not supported.");
    }
};

void makeLoadBalancedRoutingConfiguration(
    void* data_void, int num_experts, int num_tokens, int k, nvinfer1::DataType type, cudaStream_t stream);

} // namespace tensorrt_llm::kernels

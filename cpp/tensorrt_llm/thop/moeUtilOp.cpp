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

#include "tensorrt_llm/kernels/moeUtilOp.h"
#include "moe_gemm_kernels.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/native/cuda/Resize.h>

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

namespace common = tensorrt_llm::common;
namespace kernels = tensorrt_llm::kernels;
namespace cutlass_kernels = tensorrt_llm::kernels::cutlass_kernels;

namespace torch_ext
{

// input_activations: [num_tokens, hidden_size]
// input: token_topk_unpermuted_scales, [num_tokens, k]
// output: permuted_data_, [num_token * k, hidden_size]
// output: permuted_token_final_scales_, [num_tokens, k]
template <typename T>
void runPermute(void const* input_activations_void, void const* input_sf_void, int const* token_selected_experts,
    float const* token_final_scales, void const* fc1_expert_weights_void, void const* fc1_expert_biases_void,
    tensorrt_llm::ActivationType fc1_activation_type, void const* fc2_expert_weights_void,
    void const* fc2_expert_biases_void, cutlass_kernels::QuantParams quant_params, int64_t const num_rows,
    int64_t const hidden_size, int const full_num_experts, int const experts_per_token,
    int* unpermuted_token_selected_experts_, int* unpermuted_source_token_ids_, int* permuted_source_token_ids_,
    int* permuted_token_selected_experts_, T* permuted_data_, char* sorter_ws_, int64_t* expert_first_token_offset_,
    float* permuted_token_final_scales_, int* expanded_source_row_to_expanded_dest_row,
    cutlass_kernels::MOEParallelismConfig parallelism_config, cutlass_kernels::CubKeyValueSorter sorter_, bool use_lora,
    kernels::LoraParams& lora_params, bool use_fp8_block_scaling, bool min_latency_mode,
    cutlass_kernels::MoeMinLatencyParams& min_latency_params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(experts_per_token * full_num_experts <= std::numeric_limits<int>::max(),
        "experts_per_token * num_experts is too large");

    auto const* input_activations = static_cast<T const*>(input_activations_void);
    auto const* input_sf = input_sf_void
        ? reinterpret_cast<tensorrt_llm::TmaWarpSpecializedGroupedGemmInput::ElementSF const*>(input_sf_void)
        : nullptr;
    int const num_experts_per_node = full_num_experts / parallelism_config.ep_size;
    int start_expert = num_experts_per_node * parallelism_config.ep_rank;
    int end_expert = start_expert + num_experts_per_node;

    bool const needs_num_valid = parallelism_config.ep_size > 1;
    // Note: expert_first_token_offset_[num_experts_per_node] stores the total number of expanded tokens
    int64_t const* num_valid_tokens_ptr = needs_num_valid ? expert_first_token_offset_ + num_experts_per_node : nullptr;

    bool use_w4afp8 = false;
    bool fused_prologue_result = false;
    if (!use_w4afp8)
    {
        // WAR: fusedBuildExpertMapsSortFirstToken kernel will lead to illegal memory access for W4AFP8
        // input: token_selected_experts, [num_tokens, k]
        // output: unpermuted_token_selected_experts_, [num_tokens, k]
        // output: permuted_source_token_ids_, [num_tokens, k]
        // output: expert_first_token_offset_, [num_experts_per_node + 1]
        fused_prologue_result = kernels::fusedBuildExpertMapsSortFirstToken(token_selected_experts,
            unpermuted_token_selected_experts_, permuted_source_token_ids_, expert_first_token_offset_, num_rows,
            num_experts_per_node, experts_per_token, start_expert, end_expert, stream);
    }
    if (!fused_prologue_result)
    {
        TLLM_LOG_TRACE("Falling back to unfused prologue");
        kernels::buildExpertMaps(token_selected_experts, unpermuted_token_selected_experts_,
            unpermuted_source_token_ids_, num_rows, num_experts_per_node, experts_per_token, start_expert, end_expert,
            stream);
        sync_check_cuda_error(stream);

        kernels::generateTokenPermutation(unpermuted_token_selected_experts_, unpermuted_source_token_ids_,
            permuted_token_selected_experts_, permuted_source_token_ids_, expert_first_token_offset_, num_rows,
            num_experts_per_node, experts_per_token, sorter_, static_cast<void*>(sorter_ws_), stream);
    }
    sync_check_cuda_error(stream);

    // using ExpandedActivationsType = std::conditional_t<use_w4afp8, BackBoneType, T>;
    using ExpandedActivationsType = T;
    // input_activations: [num_tokens, hidden_size]
    // output: permuted_data_, [num_token * k, hidden_size]
    // input: token_topk_unpermuted_scales, [num_tokens, k]
    // output: permuted_token_final_scales_, [num_tokens * k]
    // input: permuted_source_token_ids_, [num_tokens, k]
    // output: expanded_source_row_to_expanded_dest_row, [num_tokens, k]
    float const* token_topk_unpermuted_scales = token_final_scales;
    kernels::expandInputRowsKernelLauncher(input_activations,
        reinterpret_cast<ExpandedActivationsType*>(permuted_data_), token_topk_unpermuted_scales,
        permuted_token_final_scales_, permuted_source_token_ids_, expanded_source_row_to_expanded_dest_row, num_rows,
        num_valid_tokens_ptr, hidden_size, experts_per_token, num_experts_per_node,
        quant_params.fp4.fc1.act_global_scale, expert_first_token_offset_,
        /* fc1_fp4_act_scale_ */ nullptr, input_sf, stream);
    sync_check_cuda_error(stream);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor>
moe_permute_op(torch::Tensor const& input, torch::Tensor const& token_selected_experts,
    torch::optional<torch::Tensor> token_final_scales, torch::Tensor const& fc1_expert_weights,
    torch::Tensor const& fc2_expert_weights, torch::optional<c10::ArrayRef<torch::Tensor>> quant_scales,
    torch::optional<torch::Tensor> input_sf, int64_t const num_experts_on_rank, int64_t const tp_size,
    int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
    int64_t const cluster_rank, bool min_latency_mode, bool use_fp8_block_scaling)
{
    cutlass_kernels::CubKeyValueSorter sorter_;

    TORCH_CHECK(cluster_size == 1 && cluster_rank == 0, "smart_router is supported in min_latency mode");
    TORCH_CHECK(min_latency_mode == false, "min_latency_mode is not supported now");

    CHECK_INPUT(token_selected_experts, at::ScalarType::Int)
    if (token_final_scales)
    {
        CHECK_INPUT(token_final_scales.value(), at::ScalarType::Float)
    }

    TORCH_CHECK(input.dim() == 2, "input must be 2D.");
    TORCH_CHECK(token_selected_experts.dim() == 2, "token_selected_experts must be 2D.");

    TORCH_CHECK(input.sizes()[0] == token_selected_experts.sizes()[0],
        "input and token_selected_experts must have the same num tokens.");
    if (token_final_scales)
    {
        TORCH_CHECK(token_final_scales.value().dim() == 2, "token_selected_experts_probs must be 2D.");
        TORCH_CHECK(input.sizes()[0] == token_final_scales.value().sizes()[0],
            "input and token_selected_experts_probs must have the same num tokens.");
        TORCH_CHECK(token_selected_experts.sizes()[1] == token_final_scales.value().sizes()[1],
            "token_selected_experts and token_final_scales must have the same number of experts per token.");
    }

    int experts_per_token = token_selected_experts.sizes()[1];
    int64_t num_rows = input.sizes()[0];
    int64_t hidden_size = input.sizes()[1];
    auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
    auto parallelism_config = cutlass_kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
    auto activation_type = tensorrt_llm::ActivationType::Swiglu;

    int const num_experts_per_node = num_experts_on_rank;
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    int64_t num_moe_inputs = static_cast<int64_t>(experts_per_token * num_rows);

    auto unpermuted_token_selected_experts_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    auto unpermuted_source_token_ids_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    auto permuted_source_token_ids_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    auto permuted_token_selected_experts_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    auto permuted_data_tensor = torch::empty({num_moe_inputs, hidden_size}, input.options().requires_grad(false));

    auto permuted_token_final_scales_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));

    auto expert_first_token_offset_tensor = torch::empty(
        {num_experts_per_node + 1}, torch::dtype(torch::kInt64).device(torch::kCUDA).requires_grad(false));

    size_t const sorter_size = min_latency_mode
        ? 0
        : cutlass_kernels::CubKeyValueSorter::getWorkspaceSize(num_rows * experts_per_token, num_experts_per_node);
    auto sorter_ws_tensor = torch::empty(
        {static_cast<int64_t>(sorter_size)}, torch::dtype(torch::kChar).device(torch::kCUDA).requires_grad(false));

    auto src_to_dest_map_tensor = torch::empty({static_cast<int64_t>(experts_per_token * num_rows)},
        torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    cutlass_kernels::QuantParams quant_params{};
    cutlass_kernels::MoeMinLatencyParams min_latency_params{};

    kernels::LoraParams lora_params{};

    auto data_type = input.scalar_type();
    switch (data_type)
    {
    case torch::kFloat32:
        runPermute<float>(input.const_data_ptr(), input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            /*fc1_expert_weights.const_data_ptr()*/ nullptr, nullptr, activation_type,
            /*fc2_expert_weights.const_data_ptr()*/ nullptr, nullptr, quant_params, num_rows, hidden_size,
            num_experts_total, static_cast<int>(experts_per_token),
            static_cast<int*>(unpermuted_token_selected_experts_tensor.data_ptr()),
            static_cast<int*>(unpermuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_token_selected_experts_tensor.data_ptr()),
            static_cast<float*>(permuted_data_tensor.data_ptr()), static_cast<char*>(sorter_ws_tensor.data_ptr()),
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            static_cast<int*>(src_to_dest_map_tensor.data_ptr()), parallelism_config, sorter_, false, lora_params,
            use_fp8_block_scaling, min_latency_mode, min_latency_params, stream);
        break;
    case torch::kBFloat16:
        runPermute<__nv_bfloat16>(input.const_data_ptr(),
            input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            /*fc1_expert_weights.const_data_ptr()*/ nullptr, nullptr, activation_type,
            /*fc2_expert_weights.const_data_ptr()*/ nullptr, nullptr, quant_params, num_rows, hidden_size,
            num_experts_total, static_cast<int>(experts_per_token),
            static_cast<int*>(unpermuted_token_selected_experts_tensor.data_ptr()),
            static_cast<int*>(unpermuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_token_selected_experts_tensor.data_ptr()),
            static_cast<__nv_bfloat16*>(permuted_data_tensor.data_ptr()),
            static_cast<char*>(sorter_ws_tensor.data_ptr()),
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            static_cast<int*>(src_to_dest_map_tensor.data_ptr()), parallelism_config, sorter_, false, lora_params,
            use_fp8_block_scaling, min_latency_mode, min_latency_params, stream);
        break;
    case torch::kHalf:
        runPermute<half>(input.const_data_ptr(), input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            /*fc1_expert_weights.const_data_ptr()*/ nullptr, nullptr, activation_type,
            /*fc2_expert_weights.const_data_ptr()*/ nullptr, nullptr, quant_params, num_rows, hidden_size,
            num_experts_total, static_cast<int>(experts_per_token),
            static_cast<int*>(unpermuted_token_selected_experts_tensor.data_ptr()),
            static_cast<int*>(unpermuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_token_selected_experts_tensor.data_ptr()),
            static_cast<half*>(permuted_data_tensor.data_ptr()), static_cast<char*>(sorter_ws_tensor.data_ptr()),
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            static_cast<int*>(src_to_dest_map_tensor.data_ptr()), parallelism_config, sorter_, false, lora_params,
            use_fp8_block_scaling, min_latency_mode, min_latency_params, stream);
        break;
    default:
        throw std::invalid_argument(
            "Invalid dtype, only supports input tensor with float32, float16 and bfloat16 dtype");
        break;
    }
    return std::make_tuple(unpermuted_token_selected_experts_tensor, unpermuted_source_token_ids_tensor,
        permuted_source_token_ids_tensor, permuted_token_selected_experts_tensor, permuted_data_tensor,
        expert_first_token_offset_tensor, permuted_token_final_scales_tensor, src_to_dest_map_tensor);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_moe_expand_op(torch::Tensor const& input,
    torch::optional<torch::Tensor> token_final_scales, torch::Tensor const& permuted_source_token_ids,
    int64_t const num_rows, torch::Tensor& expert_first_token_offset_tensor, int64_t const hidden_size,
    int64_t const experts_per_token, int64_t const num_experts_per_node, int64_t const tp_size, int64_t const tp_rank,
    int64_t const ep_size, int64_t const ep_rank, bool use_fp8_block_scaling)
{
    auto parallelism_config = cutlass_kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);

    bool const needs_num_valid = parallelism_config.ep_size > 1;
    int64_t const* num_valid_tokens_ptr = needs_num_valid
        ? static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()) + num_experts_per_node
        : nullptr;

    int64_t num_moe_inputs = static_cast<int64_t>(experts_per_token * num_rows);
    auto permuted_data_tensor = torch::empty({num_moe_inputs, hidden_size}, input.options().requires_grad(false));
    auto permuted_token_final_scales_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    auto expanded_source_row_to_expanded_dest_row
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    cutlass_kernels::QuantParams quant_params{};

    float const* token_topk_unpermuted_scales = token_final_scales.has_value()
        ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
        : nullptr;
    auto data_type = input.scalar_type();
    switch (data_type)
    {
    case torch::kFloat32:
        kernels::expandInputRowsKernelLauncher<float, float>(static_cast<float const*>(input.const_data_ptr()),
            reinterpret_cast<float*>(permuted_data_tensor.data_ptr()), token_topk_unpermuted_scales,
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            static_cast<int const*>(permuted_source_token_ids.const_data_ptr()),
            static_cast<int*>(expanded_source_row_to_expanded_dest_row.data_ptr()), num_rows, num_valid_tokens_ptr,
            hidden_size, experts_per_token, num_experts_per_node, quant_params.fp4.fc1.act_global_scale,
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            /* fc1_fp4_act_scale_ */ nullptr, /*input_sf*/ nullptr, stream);
        break;
    case torch::kBFloat16:
        kernels::expandInputRowsKernelLauncher<__nv_bfloat16, __nv_bfloat16>(
            static_cast<__nv_bfloat16 const*>(input.const_data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(permuted_data_tensor.data_ptr()), token_topk_unpermuted_scales,
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            static_cast<int const*>(permuted_source_token_ids.const_data_ptr()),
            static_cast<int*>(expanded_source_row_to_expanded_dest_row.data_ptr()), num_rows, num_valid_tokens_ptr,
            hidden_size, experts_per_token, num_experts_per_node, quant_params.fp4.fc1.act_global_scale,
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            /* fc1_fp4_act_scale_ */ nullptr, /*input_sf*/ nullptr, stream);
        break;
    case torch::kHalf:
        kernels::expandInputRowsKernelLauncher<half, half>(static_cast<half const*>(input.const_data_ptr()),
            reinterpret_cast<half*>(permuted_data_tensor.data_ptr()), token_topk_unpermuted_scales,
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            static_cast<int const*>(permuted_source_token_ids.const_data_ptr()),
            static_cast<int*>(expanded_source_row_to_expanded_dest_row.data_ptr()), num_rows, num_valid_tokens_ptr,
            hidden_size, experts_per_token, num_experts_per_node, quant_params.fp4.fc1.act_global_scale,
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            /* fc1_fp4_act_scale_ */ nullptr, /*input_sf*/ nullptr, stream);
        break;
    default:
        throw std::invalid_argument(
            "Invalid dtype, only supports input tensor with float32, float16 and bfloat16 dtype");
        break;
    }
    return std::make_tuple(
        permuted_data_tensor, permuted_token_final_scales_tensor, expanded_source_row_to_expanded_dest_row);
}

template <class UnfusedGemmOutputType, class ScaleBiasType, class OutputType>
void runMoEFinalizeScaleOp(UnfusedGemmOutputType const* const gemm2_output,
    ScaleBiasType const* const fc2_expert_biases, float const* const unpermuted_final_scales,
    int const* const expanded_source_row_to_expanded_dest_row, int const* const expert_for_source_row,
    int64_t const* const num_valid_tokens_ptr, int64_t const num_rows, /*int64_t const expanded_num_rows,*/
    int64_t const hidden_size, /*int64_t const inter_size, int const num_experts_per_node,*/
    int64_t const experts_per_token, cutlass_kernels::MOEParallelismConfig parallelism_config, cudaStream_t stream,
    OutputType* const final_output)
{
    kernels::finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
        static_cast<UnfusedGemmOutputType const*>(gemm2_output), final_output, fc2_expert_biases,
        unpermuted_final_scales, expanded_source_row_to_expanded_dest_row, expert_for_source_row, num_rows, hidden_size,
        experts_per_token, num_valid_tokens_ptr, parallelism_config, stream);
}

torch::Tensor run_moe_finalize_scale_op(torch::Tensor const& gemm2_output, torch::Tensor const& fc2_expert_biases,
    torch::Tensor const& unpermuted_final_scales, torch::Tensor const& expanded_source_row_to_expanded_dest_row,
    torch::Tensor const& expert_for_source_row, torch::Tensor const& expert_first_token_offset_tensor,
    c10::SymInt num_rows_param, c10::SymInt hidden_size_param, int64_t const experts_per_token,
    int64_t const num_experts_per_node, int64_t const tp_size, int64_t const tp_rank, int64_t const ep_size,
    int64_t const ep_rank)
{
    int64_t num_rows = num_rows_param.guard_int(__FILE__, __LINE__);
    int64_t hidden_size = hidden_size_param.guard_int(__FILE__, __LINE__);

    TORCH_CHECK(gemm2_output.dim() == 2, "gemm2_output must be 2D.");
    TORCH_CHECK(unpermuted_final_scales.dim() == 2, "unpermuted_final_scales must be 2D.");
    TORCH_CHECK(
        expanded_source_row_to_expanded_dest_row.dim() == 1, "expanded_source_row_to_expanded_dest_row must be 1D.");
    TORCH_CHECK(expert_for_source_row.dim() == 1, "expert_for_source_row must be 1D.");
    TORCH_CHECK(expert_first_token_offset_tensor.dim() == 1, "expert_first_token_offset_tensor must be 1D.");

    TORCH_CHECK(gemm2_output.sizes()[0] == expert_for_source_row.sizes()[0],
        "gemm2_output and expert_for_source_row must have the same expanded num tokens.");
    TORCH_CHECK(unpermuted_final_scales.sizes()[0] == num_rows, "unpermuted_final_scales[0] should equal to num_rows.");
    TORCH_CHECK(unpermuted_final_scales.sizes()[1] == experts_per_token,
        "unpermuted_final_scales[1] should equal to experts_per_token.");
    TORCH_CHECK(expert_for_source_row.sizes()[0] == gemm2_output.sizes()[0],
        "expert_for_source_row and gemm2_output must have the same expanded num tokens.");
    TORCH_CHECK(expert_first_token_offset_tensor.sizes()[0] == num_experts_per_node + 1,
        "expert_first_token_offset_tensor[0] should equal to num_experts_per_node + 1.");

    auto parallelism_config = cutlass_kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);

    bool const needs_num_valid = parallelism_config.ep_size > 1;
    int64_t const* num_valid_tokens_ptr = needs_num_valid
        ? static_cast<int64_t const*>(expert_first_token_offset_tensor.const_data_ptr()) + num_experts_per_node
        : nullptr;

    auto final_output = torch::empty({num_rows, hidden_size}, gemm2_output.options());

    auto stream = at::cuda::getCurrentCUDAStream(gemm2_output.get_device());
    auto data_type = gemm2_output.scalar_type();
    switch (data_type)
    {
    case torch::kFloat32:
        runMoEFinalizeScaleOp<float, float, float>(static_cast<float const*>(gemm2_output.const_data_ptr()),
            // static_cast<float const*>(fc2_expert_biases.const_data_ptr()),
            nullptr, static_cast<float const*>(unpermuted_final_scales.const_data_ptr()),
            static_cast<int const*>(expanded_source_row_to_expanded_dest_row.const_data_ptr()),
            static_cast<int const*>(expert_for_source_row.const_data_ptr()), num_valid_tokens_ptr, num_rows,
            hidden_size, experts_per_token, parallelism_config, stream, static_cast<float*>(final_output.data_ptr()));
        break;
    case torch::kBFloat16:
        runMoEFinalizeScaleOp<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
            static_cast<__nv_bfloat16 const*>(gemm2_output.const_data_ptr()),
            // static_cast<__nv_bfloat16 const*>(fc2_expert_biases.const_data_ptr()),
            nullptr, static_cast<float const*>(unpermuted_final_scales.const_data_ptr()),
            static_cast<int const*>(expanded_source_row_to_expanded_dest_row.const_data_ptr()),
            static_cast<int const*>(expert_for_source_row.const_data_ptr()), num_valid_tokens_ptr, num_rows,
            hidden_size, experts_per_token, parallelism_config, stream,
            static_cast<__nv_bfloat16*>(final_output.data_ptr()));
        break;
    case torch::kHalf:
        runMoEFinalizeScaleOp<half, half, half>(static_cast<half const*>(gemm2_output.const_data_ptr()),
            // static_cast<half const*>(fc2_expert_biases.const_data_ptr()),
            nullptr, static_cast<float const*>(unpermuted_final_scales.const_data_ptr()),
            static_cast<int const*>(expanded_source_row_to_expanded_dest_row.const_data_ptr()),
            static_cast<int const*>(expert_for_source_row.const_data_ptr()), num_valid_tokens_ptr, num_rows,
            hidden_size, experts_per_token, parallelism_config, stream, static_cast<half*>(final_output.data_ptr()));
        break;
    default:
        throw std::invalid_argument(
            "Invalid dtype, only supports input tensor with float32, float16 and bfloat16 dtype");
        break;
    }
    return final_output;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_permute_op(Tensor input, Tensor token_selected_experts, Tensor? token_final_scales, Tensor "
        "fc1_expert_weights, Tensor fc2_expert_weights, Tensor[]? quant_scales, Tensor? input_sf, int "
        "num_experts_on_rank, int tp_size, int tp_rank, int ep_size, int ep_rank, int cluster_size, int cluster_rank, "
        "bool min_latency_mode, bool use_fp8_block_scaling)"
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "moe_finalize_scale_op(Tensor gemm2_output, Tensor fc2_expert_biases, Tensor unpermuted_final_scales, Tensor "
        "expanded_source_row_to_expanded_dest_row, Tensor expert_for_source_row, Tensor "
        "expert_first_token_offset_tensor, SymInt num_rows, SymInt hidden_size, int experts_per_token, int "
        "num_experts_per_node, int tp_size, int tp_rank, int ep_size, int ep_rank)"
        "-> (Tensor)");
    m.def(
        "moe_expand_op(Tensor input, Tensor? token_final_scales, Tensor permuted_source_token_ids, int num_rows, "
        "Tensor expert_first_token_offset_tensor, int hidden_size, int experts_per_token, int num_experts_per_node, "
        "int tp_size, int tp_rank, int ep_size, int ep_rank, bool use_fp8_block_scaling)"
        "-> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_permute_op", &torch_ext::moe_permute_op);
    m.impl("moe_finalize_scale_op", &torch_ext::run_moe_finalize_scale_op);
    m.impl("moe_expand_op", &torch_ext::run_moe_expand_op);
}

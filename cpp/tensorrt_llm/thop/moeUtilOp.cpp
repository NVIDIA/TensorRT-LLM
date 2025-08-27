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

#include "moe_gemm_kernels.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h"
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

template <typename T>
void runPermute(void const* input_activations_void, void const* input_sf_void, int const* token_selected_experts,
    float const* token_final_scales, void const* fc1_expert_weights_void, void const* fc1_expert_biases_void,
    tensorrt_llm::ActivationType fc1_activation_type, void const* fc2_expert_weights_void,
    void const* fc2_expert_biases_void, cutlass_kernels::QuantParams quant_params, int64_t const num_rows,
    int64_t const hidden_size, int const full_num_experts, int const experts_per_token,
    int* permuted_row_to_unpermuted_row_, int* permuted_token_selected_experts_, T* permuted_data_,
    int64_t* expert_first_token_offset_, float* permuted_token_final_scales_, int* unpermuted_row_to_permuted_row,
    int* blocked_expert_counts_, int* blocked_expert_counts_cumsum_, int* blocked_row_to_unpermuted_row_,
    cutlass_kernels::MOEParallelismConfig parallelism_config, bool use_lora, kernels::LoraParams& lora_params,
    bool use_fp8_block_scaling, bool min_latency_mode, cutlass_kernels::MoeMinLatencyParams& min_latency_params,
    cudaStream_t stream)
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

    bool use_w4afp8 = false;
    bool fused_prologue_result = false;
    if (!use_w4afp8)
    {
        fused_prologue_result = cutlass_kernels::fusedBuildExpertMapsSortFirstToken(token_selected_experts,
            permuted_row_to_unpermuted_row_, unpermuted_row_to_permuted_row, expert_first_token_offset_, num_rows,
            num_experts_per_node, experts_per_token, start_expert, end_expert, stream);
    }
    if (!fused_prologue_result)
    {
        TLLM_LOG_TRACE("Falling back to unfused prologue");
        cutlass_kernels::threeStepBuildExpertMapsSortFirstToken(token_selected_experts,
            permuted_token_selected_experts_, permuted_row_to_unpermuted_row_, unpermuted_row_to_permuted_row,
            expert_first_token_offset_, blocked_expert_counts_, blocked_expert_counts_cumsum_,
            blocked_row_to_unpermuted_row_, num_rows, num_experts_per_node, experts_per_token, start_expert, stream);
    }
    sync_check_cuda_error(stream);

    using ExpandedActivationsType = T;
    float const* token_topk_unpermuted_scales = token_final_scales;
    cutlass_kernels::expandInputRowsKernelLauncher(input_activations,
        reinterpret_cast<ExpandedActivationsType*>(permuted_data_), token_topk_unpermuted_scales,
        permuted_token_final_scales_, permuted_row_to_unpermuted_row_, num_rows, hidden_size, experts_per_token,
        num_experts_per_node, quant_params, /*use_per_expert_act_scale*/ false, expert_first_token_offset_,
        /* fc1_fp4_act_scale_ */ nullptr, input_sf, true, /* prequant_scales */ nullptr, stream);
    sync_check_cuda_error(stream);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_permute_op(
    torch::Tensor const& input, torch::Tensor const& token_selected_experts,
    torch::optional<torch::Tensor> token_final_scales, torch::Tensor const& fc1_expert_weights,
    torch::Tensor const& fc2_expert_weights, torch::optional<c10::ArrayRef<torch::Tensor>> quant_scales,
    torch::optional<torch::Tensor> input_sf, int64_t const num_experts_on_rank, int64_t const tp_size,
    int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
    int64_t const cluster_rank, bool min_latency_mode, bool use_fp8_block_scaling)
{
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

    auto permuted_row_to_unpermuted_row_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    auto permuted_token_selected_experts_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    auto permuted_data_tensor = torch::empty({num_moe_inputs, hidden_size}, input.options().requires_grad(false));
    auto permuted_token_final_scales_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    auto expert_first_token_offset_tensor = torch::empty(
        {num_experts_per_node + 1}, torch::dtype(torch::kInt64).device(torch::kCUDA).requires_grad(false));
    auto unpermuted_row_to_permuted_row_tensor = torch::empty({static_cast<int64_t>(experts_per_token * num_rows)},
        torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    int64_t const num_tokens_per_block = cutlass_kernels::computeNumTokensPerBlock(num_rows, num_experts_per_node);
    int64_t const num_blocks_per_seq = tensorrt_llm::common::ceilDiv(num_rows, num_tokens_per_block);
    auto blocked_expert_counts_tensor = torch::empty({num_experts_per_node * num_blocks_per_seq},
        torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    auto blocked_expert_counts_cumsum_tensor = torch::empty({num_experts_per_node * num_blocks_per_seq},
        torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    auto blocked_row_to_unpermuted_row_tensor = torch::empty(
        {num_experts_per_node * num_rows}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

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
            /*fc1_expert_weights.const_data_ptr()*/ nullptr, /*fc1_expert_biases.const_data_ptr()*/ nullptr,
            activation_type,
            /*fc2_expert_weights.const_data_ptr()*/ nullptr, /*fc2_expert_biases.const_data_ptr()*/ nullptr,
            quant_params, num_rows, hidden_size, num_experts_total, static_cast<int>(experts_per_token),
            static_cast<int*>(permuted_row_to_unpermuted_row_tensor.data_ptr()),
            static_cast<int*>(permuted_token_selected_experts_tensor.data_ptr()),
            static_cast<float*>(permuted_data_tensor.data_ptr()),
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            static_cast<int*>(unpermuted_row_to_permuted_row_tensor.data_ptr()),
            static_cast<int*>(blocked_expert_counts_tensor.data_ptr()),
            static_cast<int*>(blocked_expert_counts_cumsum_tensor.data_ptr()),
            static_cast<int*>(blocked_row_to_unpermuted_row_tensor.data_ptr()), parallelism_config, /*use_lora*/ false,
            lora_params, use_fp8_block_scaling, min_latency_mode, min_latency_params, stream);
        break;
    case torch::kBFloat16:
        runPermute<__nv_bfloat16>(input.const_data_ptr(),
            input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            /*fc1_expert_weights.const_data_ptr()*/ nullptr, /*fc1_expert_biases.const_data_ptr()*/ nullptr,
            activation_type,
            /*fc2_expert_weights.const_data_ptr()*/ nullptr, /*fc2_expert_biases.const_data_ptr()*/ nullptr,
            quant_params, num_rows, hidden_size, num_experts_total, static_cast<int>(experts_per_token),
            static_cast<int*>(permuted_row_to_unpermuted_row_tensor.data_ptr()),
            static_cast<int*>(permuted_token_selected_experts_tensor.data_ptr()),
            static_cast<__nv_bfloat16*>(permuted_data_tensor.data_ptr()),
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            static_cast<int*>(unpermuted_row_to_permuted_row_tensor.data_ptr()),
            static_cast<int*>(blocked_expert_counts_tensor.data_ptr()),
            static_cast<int*>(blocked_expert_counts_cumsum_tensor.data_ptr()),
            static_cast<int*>(blocked_row_to_unpermuted_row_tensor.data_ptr()), parallelism_config, /*use_lora*/ false,
            lora_params, use_fp8_block_scaling, min_latency_mode, min_latency_params, stream);
        break;
    case torch::kHalf:
        runPermute<half>(input.const_data_ptr(), input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            /*fc1_expert_weights.const_data_ptr()*/ nullptr, /*fc1_expert_biases.const_data_ptr()*/ nullptr,
            activation_type,
            /*fc2_expert_weights.const_data_ptr()*/ nullptr, /*fc2_expert_biases.const_data_ptr()*/ nullptr,
            quant_params, num_rows, hidden_size, num_experts_total, static_cast<int>(experts_per_token),
            static_cast<int*>(permuted_row_to_unpermuted_row_tensor.data_ptr()),
            static_cast<int*>(permuted_token_selected_experts_tensor.data_ptr()),
            static_cast<half*>(permuted_data_tensor.data_ptr()),
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            static_cast<int*>(unpermuted_row_to_permuted_row_tensor.data_ptr()),
            static_cast<int*>(blocked_expert_counts_tensor.data_ptr()),
            static_cast<int*>(blocked_expert_counts_cumsum_tensor.data_ptr()),
            static_cast<int*>(blocked_row_to_unpermuted_row_tensor.data_ptr()), parallelism_config, /*use_lora*/ false,
            lora_params, use_fp8_block_scaling, min_latency_mode, min_latency_params, stream);
        break;
    default:
        throw std::invalid_argument(
            "Invalid dtype, only supports input tensor with float32, float16 and bfloat16 dtype");
        break;
    }
    return std::make_tuple(permuted_row_to_unpermuted_row_tensor, permuted_token_selected_experts_tensor,
        permuted_data_tensor, expert_first_token_offset_tensor, permuted_token_final_scales_tensor,
        unpermuted_row_to_permuted_row_tensor);
}

template <class UnfusedGemmOutputType, class ScaleBiasType, class OutputType>
void runMoEFinalizeScaleOp(UnfusedGemmOutputType const* const gemm2_output, ScaleBiasType const* const biases,
    float const* const unpermuted_final_scales, int const* const unpermuted_row_to_permuted_row,
    int const* const permuted_row_to_unpermuted_row, int const* const token_selected_experts,
    int64_t const* const expert_first_token_offset, int64_t const num_rows, int64_t const hidden_size,
    int64_t const unpadded_hidden_size, int64_t const experts_per_token, int const num_experts_per_node,
    cutlass_kernels::MOEParallelismConfig parallelism_config, bool enable_alltoall, cudaStream_t stream,
    OutputType* const final_output)
{
    cutlass_kernels::finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
        static_cast<UnfusedGemmOutputType const*>(gemm2_output), final_output, biases, unpermuted_final_scales,
        unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row, token_selected_experts,
        expert_first_token_offset, num_rows, hidden_size, unpadded_hidden_size, experts_per_token, num_experts_per_node,
        parallelism_config, enable_alltoall, stream);
}

torch::Tensor run_moe_finalize_scale_op(torch::Tensor const& gemm2_output, torch::optional<torch::Tensor> biases,
    torch::Tensor const& unpermuted_final_scales, torch::Tensor const& unpermuted_row_to_permuted_row,
    torch::Tensor const& permuted_row_to_unpermuted_row, torch::Tensor const& token_selected_experts,
    torch::Tensor const& expert_first_token_offset_tensor, bool enable_alltoall, c10::SymInt num_rows_param,
    c10::SymInt hidden_size_param, c10::SymInt unpadded_hidden_size_param, int64_t const experts_per_token,
    int64_t const num_experts_per_node, int64_t const tp_size, int64_t const tp_rank, int64_t const ep_size,
    int64_t const ep_rank)
{
    int64_t num_rows = num_rows_param.guard_int(__FILE__, __LINE__);
    int64_t hidden_size = hidden_size_param.guard_int(__FILE__, __LINE__);
    int64_t unpadded_hidden_size = unpadded_hidden_size_param.guard_int(__FILE__, __LINE__);

    TORCH_CHECK(gemm2_output.dim() == 2, "gemm2_output must be 2D.");
    TORCH_CHECK(unpermuted_final_scales.dim() == 2, "unpermuted_final_scales must be 2D.");
    TORCH_CHECK(token_selected_experts.dim() == 2, "token_selected_experts must be 2D.");
    TORCH_CHECK(unpermuted_row_to_permuted_row.dim() == 1, "unpermuted_row_to_permuted_row must be 1D.");
    TORCH_CHECK(permuted_row_to_unpermuted_row.dim() == 1, "permuted_row_to_unpermuted_row must be 1D.");
    TORCH_CHECK(expert_first_token_offset_tensor.dim() == 1, "expert_first_token_offset_tensor must be 1D.");

    TORCH_CHECK(unpermuted_final_scales.sizes()[0] == num_rows, "unpermuted_final_scales[0] should equal to num_rows.");
    TORCH_CHECK(unpermuted_final_scales.sizes()[1] == experts_per_token,
        "unpermuted_final_scales[1] should equal to experts_per_token.");
    TORCH_CHECK(token_selected_experts.sizes()[0] == num_rows, "token_selected_experts[0] should equal to num_rows.");
    TORCH_CHECK(token_selected_experts.sizes()[1] == experts_per_token,
        "token_selected_experts[1] should equal to experts_per_token.");
    TORCH_CHECK(gemm2_output.sizes()[0] == unpermuted_row_to_permuted_row.sizes()[0],
        "gemm2_output and unpermuted_row_to_permuted_row must have the same expanded num tokens.");
    TORCH_CHECK(gemm2_output.sizes()[0] == permuted_row_to_unpermuted_row.sizes()[0],
        "gemm2_output.sizes()[0] should equal to permuted_row_to_unpermuted_row.sizes()[0].");
    TORCH_CHECK(expert_first_token_offset_tensor.sizes()[0] == num_experts_per_node + 1,
        "expert_first_token_offset_tensor[0] should equal to num_experts_per_node + 1.");

    auto parallelism_config = cutlass_kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);

    auto final_output = torch::empty({num_rows, unpadded_hidden_size}, gemm2_output.options());

    auto stream = at::cuda::getCurrentCUDAStream(gemm2_output.get_device());
    auto data_type = gemm2_output.scalar_type();
    switch (data_type)
    {
    case torch::kFloat32:
        runMoEFinalizeScaleOp<float, float, float>(static_cast<float const*>(gemm2_output.const_data_ptr()),
            biases.has_value() ? static_cast<float const*>(biases.value().const_data_ptr()) : nullptr,
            static_cast<float const*>(unpermuted_final_scales.const_data_ptr()),
            static_cast<int const*>(unpermuted_row_to_permuted_row.const_data_ptr()),
            static_cast<int const*>(permuted_row_to_unpermuted_row.const_data_ptr()),
            static_cast<int const*>(token_selected_experts.const_data_ptr()),
            static_cast<int64_t const*>(expert_first_token_offset_tensor.const_data_ptr()), num_rows, hidden_size,
            unpadded_hidden_size, experts_per_token, num_experts_per_node, parallelism_config, enable_alltoall, stream,
            static_cast<float*>(final_output.data_ptr()));
        break;
    case torch::kBFloat16:
        runMoEFinalizeScaleOp<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
            static_cast<__nv_bfloat16 const*>(gemm2_output.const_data_ptr()),
            biases.has_value() ? static_cast<__nv_bfloat16 const*>(biases.value().const_data_ptr()) : nullptr,
            static_cast<float const*>(unpermuted_final_scales.const_data_ptr()),
            static_cast<int const*>(unpermuted_row_to_permuted_row.const_data_ptr()),
            static_cast<int const*>(permuted_row_to_unpermuted_row.const_data_ptr()),
            static_cast<int const*>(token_selected_experts.const_data_ptr()),
            static_cast<int64_t const*>(expert_first_token_offset_tensor.const_data_ptr()), num_rows, hidden_size,
            unpadded_hidden_size, experts_per_token, num_experts_per_node, parallelism_config, enable_alltoall, stream,
            static_cast<__nv_bfloat16*>(final_output.data_ptr()));
        break;
    case torch::kHalf:
        runMoEFinalizeScaleOp<half, half, half>(static_cast<half const*>(gemm2_output.const_data_ptr()),
            biases.has_value() ? static_cast<half const*>(biases.value().const_data_ptr()) : nullptr,
            static_cast<float const*>(unpermuted_final_scales.const_data_ptr()),
            static_cast<int const*>(unpermuted_row_to_permuted_row.const_data_ptr()),
            static_cast<int const*>(permuted_row_to_unpermuted_row.const_data_ptr()),
            static_cast<int const*>(token_selected_experts.const_data_ptr()),
            static_cast<int64_t const*>(expert_first_token_offset_tensor.const_data_ptr()), num_rows, hidden_size,
            unpadded_hidden_size, experts_per_token, num_experts_per_node, parallelism_config, enable_alltoall, stream,
            static_cast<half*>(final_output.data_ptr()));
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
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "moe_finalize_scale_op(Tensor gemm2_output, Tensor? biases, Tensor unpermuted_final_scales, Tensor "
        "unpermuted_row_to_permuted_row, Tensor permuted_row_to_unpermuted_row, Tensor token_selected_experts, Tensor "
        "expert_first_token_offset_tensor, bool enable_alltoall, SymInt num_rows, SymInt hidden_size, SymInt "
        "unpadded_hidden_size, int "
        "experts_per_token, int "
        "num_experts_per_node, int tp_size, int tp_rank, int ep_size, int ep_rank)"
        "-> (Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_permute_op", &torch_ext::moe_permute_op);
    m.impl("moe_finalize_scale_op", &torch_ext::run_moe_finalize_scale_op);
}

/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

// #include "tensorrt_llm/kernels/moeUtilOp.h"
// #include "tensorrt_llm/common/cudaTypeUtils.cuh"
// #include "tensorrt_llm/common/opUtils.h"
// #include "tensorrt_llm/runtime/torchUtils.h"

// #include "cutlass/epilogue/thread/activation.h"

#include "tensorrt_llm/kernels/moeUtilOp.h"
#include "moe_gemm_kernels.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/native/cuda/Resize.h>

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

namespace common = tensorrt_llm::common;
// namespace kernels = tensorrt_llm::kernels;
namespace kernels = tensorrt_llm::kernels;
namespace cutlass_kernels = tensorrt_llm::kernels::cutlass_kernels;

namespace torch_ext
{

std::tuple<at::Tensor, at::Tensor> renorm_permutate_op(th::Tensor const& router_logits, int64_t topk)
{
    auto data_type = router_logits.scalar_type();
    auto input_size = router_logits.sizes();
    int64_t num_tokens = input_size[0];
    int64_t num_experts = input_size[1];
    TORCH_CHECK(input_size.size() == 2, "router_logits must be a 2D Tensor");
    TORCH_CHECK(topk <= 8, "topk should be smaller than or equal to 8 for now"); //@todo: remove this restriction later
    TORCH_CHECK(num_experts <= 128, "expert number should be smaller than or equal to 128 for now");

    th::Tensor topk_values = th::empty({num_tokens, topk}, th::dtype(torch::kFloat32).device(torch::kCUDA));
    th::Tensor topk_indices = th::empty({num_tokens, topk}, th::dtype(torch::kInt32).device(torch::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream(router_logits.get_device());

    switch (data_type)
    {
    case torch::kFloat32:
        // Handle Float32
        tk::invokeRenormMoeRouting<float, float, int32_t>(reinterpret_cast<float*>(router_logits.mutable_data_ptr()),
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
            reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, topk, stream);
        break;
    case torch::kBFloat16:
        // Handle BFloat16
        tk::invokeRenormMoeRouting<__nv_bfloat16, float, int32_t>(
            reinterpret_cast<__nv_bfloat16*>(router_logits.mutable_data_ptr()),
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
            reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, topk, stream);
        break;
    case torch::kHalf:
        // Handle Half
        tk::invokeRenormMoeRouting<half, float, int32_t>(reinterpret_cast<half*>(router_logits.mutable_data_ptr()),
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
            reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, topk, stream);
        break;
    default:
        // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float32, float16 and bfloat16");
        break;
    }
    return {topk_indices, topk_values};
}

#if 0
struct WorkspaceInfo
{
    void* workspace{};
    void* src_to_dest_map{};
};

// constexpr bool isGatedActivation(ActivationType activation_type)
// {
//     return activation_type == ActivationType::Swiglu || activation_type == ActivationType::Geglu;
// }

// bool mayHaveDifferentGEMMOutputType() const
// {
//     // We just check if its supported because we need to know when calculating workspace size
//     return ((moe_gemm_runner_.supportsTmaWarpSpecialized() && !std::is_same_v<T, UnfusedGemmOutputType>) || use_fp8);
// }

// bool mayHaveFinalizeFused() const
// {
//     return moe_gemm_runner_.supportsTmaWarpSpecialized() && moe_gemm_runner_.getSM() == 90
//         && !use_deterministic_hopper_reduce_;
// }

// limin-todo: UnfusedGemmOutputType
template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
std::vector<size_t> getWorkspaceDeviceBufferSizes(
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
    int const experts_per_token, tensorrt_llm::ActivationType activation_type, bool use_lora, bool use_fp8_block_scaling,
    bool min_latency_mode, bool use_awq, bool use_fp4 = false, bool use_w4afp8 = false)
{
    size_t num_moe_inputs
        = use_fp8_block_scaling ? (experts_per_token * num_rows + 3) / 4 * 4 : experts_per_token * num_rows;
    num_moe_inputs = min_latency_mode ? num_experts_per_node * num_rows : num_moe_inputs;
    size_t const permuted_elems = num_moe_inputs * hidden_size;
    size_t const interbuf_elems = num_moe_inputs * inter_size;
    size_t glu_inter_elems = 0;
    // bool is_gated_activation = isGatedActivation(activation_type);
    bool is_gated_activation = false;
    if (is_gated_activation)
    {
        glu_inter_elems = interbuf_elems * 2;
    }
    // else if (mayHaveDifferentGEMMOutputType())
    // {
    //     // In this case we are using activation quantization, and some intermediate buffers will be unquantized
    //     // We need to have separate memory for these as we can no longer alias the output buffer for reuse
    //     glu_inter_elems = interbuf_elems;
    // }

    // bool using_tma_ws = moe_gemm_runner_.supportsTmaWarpSpecialized();
    bool using_tma_ws = false;

    size_t const gemm_output_dtype = sizeof(UnfusedGemmOutputType);

    constexpr float dtype_size = use_fp4 ? 0.5f : (use_w4afp8 ? 2.0f : sizeof(T));

    size_t const unpermuted_token_selected_experts_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    size_t const unpermuted_source_token_ids_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    size_t const permuted_source_token_ids_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    size_t const permuted_token_selected_experts_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    size_t const permuted_data_size = permuted_elems * dtype_size;
    size_t const expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t);
    // limin-todo: 
    // size_t const permuted_token_final_scales_size = mayHaveFinalizeFused() ? num_moe_inputs * sizeof(float) : 0;
    size_t const permuted_token_final_scales_size = num_moe_inputs * sizeof(float);
    size_t const glu_inter_size = glu_inter_elems * gemm_output_dtype; // May be an intermediate type for quantization
    size_t const fc1_result_size = interbuf_elems * dtype_size;        // Activation quantizes so back to dtype_size
    size_t const sorter_size
        = min_latency_mode ? 0 : cutlass_kernels::CubKeyValueSorter::getWorkspaceSize(num_rows, num_experts_per_node);
    size_t const fc2_result_size = min_latency_mode
        ? 0
        : num_moe_inputs * hidden_size * gemm_output_dtype; // May be an intermediate type for quantization

    // NOTE: We should skip allocation for all non-FP4 cases (e.g.
    // fc1_fp4_act_scale_size = use_fp4 ? scale_size : 0), but
    // this will cause BF16 unit tests to fail. One of the kernels
    // could be buggy/relying on the extra space.
    // size_t const fc1_fp4_act_scale_size = use_fp4 ? getOffsetFlatSFArray(num_experts_per_node, num_rows, hidden_size)
    //         * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
    //                                               : 0;
    // size_t const fc2_fp4_act_scale_size = use_fp4 ? getOffsetFlatSFArray(num_experts_per_node, num_rows, inter_size)
    //         * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
    //                                               : 0;
    size_t const fc1_fp4_act_scale_size = 0;
    size_t const fc2_fp4_act_scale_size = 0;

    size_t const fp4_act_scale_size = std::max(fc1_fp4_act_scale_size, fc2_fp4_act_scale_size);

    // size_t const tma_ws_size
    //     = using_tma_ws ? TmaWarpSpecializedGroupedGemmInput::workspaceSize(num_experts_per_node) : 0;

    // size_t const gemm_workspace_size = moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);
    size_t const tma_ws_size = 0;
    size_t const gemm_workspace_size = 0;

   // // lora related
   //  size_t const lora_input_size
   //      = (use_lora && use_fp8) ? std::max(permuted_elems, interbuf_elems) * sizeof(ScaleBiasType) : 0;
   //  size_t const lora_fc1_result_size = use_lora
   //      ? (is_gated_activation ? 2 * interbuf_elems * sizeof(ScaleBiasType) : interbuf_elems * sizeof(ScaleBiasType))
   //      : 0;
   //  size_t const lora_add_bias_size = use_lora ? lora_fc1_result_size : 0;
   //  size_t const lora_fc2_result_size = use_lora ? permuted_elems * sizeof(ScaleBiasType) : 0; 
    size_t const lora_input_size = 0;
    size_t const lora_fc1_result_size = 0;
    size_t const lora_add_bias_size = 0;
    size_t const lora_fc2_result_size = 0;

    // We do some overlapping of the large workspace buffers. Although we could overlap some of the other buffers, they
    // are small enough (i.e no factor of hidden size) they will only be a couple MiB at most, so we don't bother
    // in the case of fused activation we overlap permuted_data and fc2_result
    // in the case of unfused activation we overlap permuted_data and fc1_result
    // we need to calculate the max possible size, so use the max of all three
    size_t overlapped_gemm1_gemm2_inputs = std::max(permuted_data_size, fc2_result_size);
    // When glu_inter_elems is 0 we are always fused, otherwise we may need the un-fused case
    if (glu_inter_elems > 0)
    {
        overlapped_gemm1_gemm2_inputs = std::max(overlapped_gemm1_gemm2_inputs, fc1_result_size);
    }

    size_t const alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float*);

    // if we have glu_inter we overlap it with fc2_result, otherwise we use fc1_result by itself
    size_t overlapped_gemm1_gemm2_outputs = fc1_result_size;
    if (glu_inter_elems > 0)
    {
        overlapped_gemm1_gemm2_outputs
            = std::max(std::max(glu_inter_size, fc2_result_size), overlapped_gemm1_gemm2_outputs);
    }

    // size_t smoothed_act_size = use_awq ? std::max(permuted_elems, interbuf_elems) * sizeof(T) * 2
    //                                    : 0; // Extra workspace required by AWQ for smoothing activations
    size_t smoothed_act_size = 0;
    size_t deepseek_fc_workspace_size = 0;
    if (use_fp8_block_scaling)
    {
        size_t factor = is_gated_activation ? 2 : 1;
        size_t blockscale_fc1_output_size = factor * interbuf_elems * gemm_output_dtype;
        size_t blockscale_fc2_output_size = permuted_elems * gemm_output_dtype;
        overlapped_gemm1_gemm2_inputs
            = std::max(std::max(permuted_data_size, fc1_result_size), blockscale_fc2_output_size);
        overlapped_gemm1_gemm2_outputs = blockscale_fc1_output_size;

        // auto* blockscale_gemm_runner = getBlockScaleGemmRunner();
        // TLLM_CHECK(blockscale_gemm_runner != nullptr);
        // auto deepseek_fc1_workspace_size = blockscale_gemm_runner->getWorkspaceSize(
        //     num_rows, factor * inter_size, hidden_size, experts_per_token, num_experts_per_node);
        // auto deepseek_fc2_workspace_size = blockscale_gemm_runner->getWorkspaceSize(
        //     num_rows, hidden_size, inter_size, experts_per_token, num_experts_per_node);
        size_t deepseek_fc1_workspace_size = 0;
        size_t deepseek_fc2_workspace_size = 0;
        deepseek_fc_workspace_size
            = std::max(deepseek_fc1_workspace_size, deepseek_fc2_workspace_size);
    }

    // limin-todo:
    std::vector<size_t> workspace{unpermuted_source_token_ids_size, unpermuted_token_selected_experts_size,
        permuted_source_token_ids_size, permuted_token_selected_experts_size, expert_first_token_offset_size,
        permuted_token_final_scales_size, sorter_size,
        // These pointers reuse the same memory
        overlapped_gemm1_gemm2_inputs, overlapped_gemm1_gemm2_outputs, alpha_scale_ptr_array_size, fp4_act_scale_size,
        tma_ws_size, gemm_workspace_size, lora_input_size, lora_fc1_result_size, lora_add_bias_size,
        lora_fc2_result_size, deepseek_fc_workspace_size,
        // End of reused memory
        smoothed_act_size};
    return workspace;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
size_t getWorkspaceSize(
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts,
    int const experts_per_token, tensorrt_llm::ActivationType activation_type, cutlass_kernels::MOEParallelismConfig parallelism_config, bool use_lora,
    bool use_fp8_block_scaling, bool min_latency_mode, bool use_awq)
{
    int const ep_size = parallelism_config.ep_size;
    TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0, "Number of experts must be a multiple of ep size");
    auto workspace = getWorkspaceDeviceBufferSizes(num_rows, hidden_size, inter_size, num_experts / ep_size,
        experts_per_token, activation_type, use_lora, use_fp8_block_scaling, min_latency_mode, use_awq);
    auto ws_size = tensorrt_llm::common::calculateTotalWorkspaceSize(workspace.data(), workspace.size());
    TLLM_LOG_DEBUG("Mixture Of Experts Plugin requires workspace of %2f MiB", ws_size / 1024.f / 1024.f);
    return ws_size;
}

// limin-todo: 
WorkspaceInfo getWorkspaceInfo(int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
    int num_experts, int experts_per_token, tensorrt_llm::ActivationType activation_type,
    cutlass_kernels::MOEParallelismConfig const& parallelismConfig, bool min_latency_mode)
{
    // extract this function.
    size_t moe_workspace_size = getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts,
        experts_per_token, activation_type, parallelismConfig, /* use_lora */ false, /* mUseFp8BlockScaling */ true,
        /* min_latency_mode */ false, /* mUseW4A8GroupScaling */ false);
    // limin-todo:
    // size_t src_to_dest_map_size = experts_per_token * num_rows * sizeof(int);
    size_t src_to_dest_map_size = 0;

    std::vector<size_t> workspaces{moe_workspace_size, src_to_dest_map_size};

    size_t total_workspace_size = common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    auto workspace = torch::empty({static_cast<long>(total_workspace_size)},
        torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

    WorkspaceInfo info{};
    info.workspace = workspace.data_ptr();
    info.src_to_dest_map = common::nextWorkspacePtr(static_cast<int8_t*>(workspace.data_ptr()), moe_workspace_size);

    return info;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
void configureWsPtrs(char* ws_ptr, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts_per_node, int const experts_per_token, tensorrt_llm::ActivationType activation_type,
    cutlass_kernels::MOEParallelismConfig parallelism_config, bool use_lora, bool use_fp8_block_scaling, bool min_latency_mode,
    bool use_awq, int* &unpermuted_token_selected_experts_, int* &unpermuted_source_token_ids_, 
    int* &permuted_source_token_ids_, int* &permuted_token_selected_experts_, char* &sorter_ws_, T* &permuted_data_, 
    float* &permuted_token_final_scales_, int64_t* &expert_first_token_offset_)
{
    auto ws_sizes = getWorkspaceDeviceBufferSizes(num_rows, hidden_size, inter_size, num_experts_per_node,
        experts_per_token, activation_type, use_lora, use_fp8_block_scaling, min_latency_mode, use_awq);

    std::vector<int8_t*> ws_sliced{(int8_t*) ws_ptr};
    for (auto size : ws_sizes)
    {
        ws_sliced.push_back(nextWorkspacePtr(ws_sliced.back(), size));
    }
    ws_sliced.pop_back();

    unpermuted_source_token_ids_ = (int*) ws_sliced[0];
    unpermuted_token_selected_experts_ = (int*) ws_sliced[1];
    permuted_source_token_ids_ = (int*) ws_sliced[2];
    permuted_token_selected_experts_ = (int*) ws_sliced[3];

    expert_first_token_offset_ = (int64_t*) ws_sliced[4];

    // // We check if the provided config uses fused finalize and disable it if it does not
    // bool const gemm2_using_tma_ws = moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_);
    // permuted_token_final_scales_ = (gemm2_using_tma_ws && mayHaveFinalizeFused()) ? (float*) ws_sliced[5] : nullptr;
    permuted_token_final_scales_ = (float*) ws_sliced[5];

    sorter_ws_ = (char*) ws_sliced[6];

    // Always same index, but overlapped with either fc1_result_ or fc2_result_
    permuted_data_ = (T*) ws_sliced[7];

    // bool const is_gated_activation = isGatedActivation(activation_type);
    // bool const gemm1_using_fused_moe
    //     = moe_gemm_runner_.isFusedGatedActivation(*gemm1_config_, is_gated_activation, inter_size, hidden_size);
    // bool const gemm1_using_tma_ws = moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_);
    // bool const tma_ws_has_glu = gemm1_using_tma_ws && (mayHaveDifferentGEMMOutputType() || is_gated_activation);
    // // We always use fused path if we can
    // bool const non_tma_ws_has_glu = !gemm1_using_fused_moe && is_gated_activation;
    // bool const has_glu_inter_result = tma_ws_has_glu || non_tma_ws_has_glu || use_fp8;
    // // Always same index, ignored if not needed
    // glu_inter_result_ = has_glu_inter_result ? (T*) ws_sliced[8] : nullptr;

    // // fc1 and fc2 alias one of the above pointers, but it depends on if actfn is fused/unfused which is overlapped
    // // NOTE: It is important to get the order of these correct as the wrong order will cause the buffer to be used as an
    // // input and output for the same gemm, which will cause corruption
    // fc1_result_ = has_glu_inter_result ? (T*) ws_sliced[7] : (T*) ws_sliced[8];
    // fc2_result_ = has_glu_inter_result ? (T*) ws_sliced[8] : (T*) ws_sliced[7];

    // if (use_fp8_block_scaling)
    // {
    //     permuted_data_ = (T*) ws_sliced[7];
    //     fc1_result_ = (T*) ws_sliced[7];
    //     glu_inter_result_ = (T*) ws_sliced[8];
    //     fc2_result_ = (T*) ws_sliced[7];
    // }

    // alpha_scale_ptr_array_ = reinterpret_cast<float const**>(ws_sliced[9]);

    // // NOTE: We alias these, but if we fuse the quantization for GEMM2 into GEMM1 they will need separated
    // fc1_fp4_act_scale_
    //     = reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(ws_sizes[10] > 0 ? ws_sliced[10] : nullptr);
    // fc2_fp4_act_scale_ = fc1_fp4_act_scale_;

    // tma_ws_grouped_gemm_input_ = {};
    // if (moe_gemm_runner_.supportsTmaWarpSpecialized())
    // {
    //     tma_ws_grouped_gemm_input_.configureWorkspace(ws_sliced[11], num_experts_per_node, ws_sliced[12], ws_sizes[12]);
    // }

    // lora_fc1_result_ = {};
    // lora_add_bias_ = {};
    // lora_fc2_result_ = {};

    // if (use_lora)
    // {
    //     lora_input_ = (ScaleBiasType*) ws_sliced[13];
    //     lora_fc1_result_ = (ScaleBiasType*) ws_sliced[14];
    //     lora_add_bias_ = (ScaleBiasType*) ws_sliced[15];
    //     lora_fc2_result_ = (ScaleBiasType*) ws_sliced[16];
    // }

    // if (use_fp8_block_scaling)
    // {
    //     auto* blockscale_gemm_runner = getBlockScaleGemmRunner();
    //     TLLM_CHECK(blockscale_gemm_runner != nullptr);
    //     blockscale_gemm_runner->configureWorkspace((char*) ws_sliced[17]);
    // }

    // if (use_awq)
    // {
    //     smoothed_act_ = (void*) ws_sliced[18];
    // }
}
#endif

void print(int* tensor, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        std::cout << tensor[i] << " ";
    }
}

template <typename T>
void runPermute(void const* input_activations_void, void const* input_sf_void, int const* token_selected_experts,
    float const* token_final_scales, void const* fc1_expert_weights_void, void const* fc1_expert_biases_void,
    tensorrt_llm::ActivationType fc1_activation_type, void const* fc2_expert_weights_void,
    void const* fc2_expert_biases_void, cutlass_kernels::QuantParams quant_params, int64_t const num_rows,
    int64_t const hidden_size, int const full_num_experts,
    int const experts_per_token, /*char* workspace_ptr, void* final_output_void, */
    int* unpermuted_token_selected_experts_, int* unpermuted_source_token_ids_, int* permuted_source_token_ids_,
    int* permuted_token_selected_experts_, T* permuted_data_, char* sorter_ws_, int64_t* expert_first_token_offset_,
    float* permuted_token_final_scales_, int* expanded_source_row_to_expanded_dest_row,
    cutlass_kernels::MOEParallelismConfig parallelism_config, cutlass_kernels::CubKeyValueSorter sorter_, bool use_lora,
    kernels::LoraParams& lora_params, bool use_fp8_block_scaling, bool min_latency_mode,
    cutlass_kernels::MoeMinLatencyParams& min_latency_params, cudaStream_t stream)
{
    // std::cout << "enter runPermute\n";
    TLLM_CHECK_WITH_INFO(experts_per_token * full_num_experts <= std::numeric_limits<int>::max(),
        "experts_per_token * num_experts is too large");

    auto const* input_activations = static_cast<T const*>(input_activations_void);
    auto const* input_sf = input_sf_void
        ? reinterpret_cast<tensorrt_llm::TmaWarpSpecializedGroupedGemmInput::ElementSF const*>(input_sf_void)
        : nullptr;
    // std::cout << "line 439\n";
    // limin-todo: 看看输入输出分别是什么?
    // input_activations: [num_tokens, hidden_size]
    // output: permuted_data_, [num_token * k, hidden_size]
    // input: token_topk_unpermuted_scales, [num_tokens, k]
    // output: permuted_token_final_scales_, [num_tokens, k]
    int const num_experts_per_node = full_num_experts / parallelism_config.ep_size;
    int start_expert = num_experts_per_node * parallelism_config.ep_rank;
    int end_expert = start_expert + num_experts_per_node;

    bool const needs_num_valid = parallelism_config.ep_size > 1;
    // TODO: expert_first_token_offset_[num_experts_per_node]存储了所有展开后的token总数
    int64_t const* num_valid_tokens_ptr = needs_num_valid ? expert_first_token_offset_ + num_experts_per_node : nullptr;

    bool use_w4afp8 = false;
    bool fused_prologue_result = false;
    if (!use_w4afp8)
    {
        // WAR: fusedBuildExpertMapsSortFirstToken kernel will lead to illegal memory access for W4AFP8
        // input: token_selected_experts, [num_tokens, k]
        // output: unpermuted_token_selected_experts_, [num_tokens, k]
        // output: permuted_source_token_ids_, [num_tokens, k], ????
        // output: expert_first_token_offset_, [num_experts_per_node + 1]
        // std::cout << "before call fusedBuildExpertMapsSortFirstToken" << std::endl;
        fused_prologue_result = kernels::fusedBuildExpertMapsSortFirstToken(token_selected_experts,
            unpermuted_token_selected_experts_, permuted_source_token_ids_, expert_first_token_offset_, num_rows,
            num_experts_per_node, experts_per_token, start_expert, end_expert, stream);
        // std::cout << "after call fusedBuildExpertMapsSortFirstToken" << std::endl;
    }
    // std::cout << "fused_prologue_result = " << fused_prologue_result << std::endl;
    if (!fused_prologue_result)
    {
        TLLM_LOG_TRACE("Falling back to unfused prologue");
        // std::cout << "before call buildExpertMaps" << std::endl;
        kernels::buildExpertMaps(token_selected_experts, unpermuted_token_selected_experts_,
            unpermuted_source_token_ids_, num_rows, num_experts_per_node, experts_per_token, start_expert, end_expert,
            stream);
        sync_check_cuda_error(stream);
        // torch::cuda::synchronize();
        // std::cout << "after call buildExpertMaps" << std::endl;

        // std::cout << "before call generateTokenPermutation" << std::endl;
        kernels::generateTokenPermutation(unpermuted_token_selected_experts_, unpermuted_source_token_ids_,
            permuted_token_selected_experts_, permuted_source_token_ids_, expert_first_token_offset_, num_rows,
            num_experts_per_node, experts_per_token, sorter_, static_cast<void*>(sorter_ws_), stream);
        // std::cout << "after call generateTokenPermutation" << std::endl;
    }
    sync_check_cuda_error(stream);
    // torch::cuda::synchronize();
    // std::cout << "after generateTOkenPermute/fusedBuildExpertMapsSortFirstToken sync" << std::endl;
    // size_t num_moe_inputs
    //     = use_fp8_block_scaling ? (experts_per_token * num_rows + 3) / 4 * 4 : experts_per_token * num_rows;
    // print(permuted_source_token_ids_, num_moe_inputs);
#if 1
    // using ExpandedActivationsType = std::conditional_t<use_w4afp8, BackBoneType, T>;
    using ExpandedActivationsType = T;
    // input_activations: [num_tokens, hidden_size]
    // output: permuted_data_, [num_token * k, hidden_size]
    // input: token_topk_unpermuted_scales, [num_tokens, k]
    // output: permuted_token_final_scales_, [num_tokens * k]
    // input: permuted_source_token_ids_, [num_tokens, k]
    // output: expanded_source_row_to_expanded_dest_row, [num_tokens, k]
    // input: num_rows, num_valid_tokens_ptr, hidden_size, experts_per_token, num_experts_per_node,
    // quant_params.fp4.fc1.act_global_scale, expert_first_token_offset_, fc1_fp4_act_scale_, input_sf
    float const* token_topk_unpermuted_scales = token_final_scales;
    kernels::expandInputRowsKernelLauncher(input_activations,
        reinterpret_cast<ExpandedActivationsType*>(permuted_data_), token_topk_unpermuted_scales,
        permuted_token_final_scales_, permuted_source_token_ids_, expanded_source_row_to_expanded_dest_row, num_rows,
        num_valid_tokens_ptr, hidden_size, experts_per_token, num_experts_per_node,
        quant_params.fp4.fc1.act_global_scale, expert_first_token_offset_,
        /* fc1_fp4_act_scale_ */ nullptr, input_sf, stream);
#endif

    sync_check_cuda_error(stream);
    // torch::cuda::synchronize();
}

// 打印torch tensor的工具函数
template <typename T>
void printTorchTensor(torch::Tensor const& tensor, std::string const& name)
{
    torch::cuda::synchronize();
    try
    {
        std::cout << "Tensor " << name << ":" << std::endl;

        // 1. 检查张量是否有效
        if (!tensor.defined())
        {
            std::cout << "Tensor is not defined" << std::endl;
            return;
        }

        // 2. 检查内存是否已分配
        if (!tensor.is_contiguous() || !tensor.data_ptr())
        {
            std::cout << "Tensor memory is not allocated" << std::endl;
            return;
        }

        // 2. 打印形状
        std::cout << "Shape: [";
        for (int i = 0; i < tensor.dim(); i++)
        {
            std::cout << tensor.size(i);
            if (i < tensor.dim() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // 3. 打印设备信息
        std::cout << "Device: " << tensor.device() << std::endl;

        // 4. 打印数据类型
        std::cout << "Dtype: " << tensor.dtype() << std::endl;

        // 5. 打印值
        std::cout << "Values: " << std::endl;

        // 检查张量是否在CUDA上
        if (tensor.device().is_cuda())
        {
            try
            {
                // 使用clone()创建新的张量，避免共享内存问题
                auto tensor_cpu = tensor.clone().cpu();

                // 如果是1维张量
                if (tensor_cpu.dim() == 1)
                {
                    try
                    {
                        auto accessor = tensor_cpu.accessor<T, 1>();
                        for (int i = 0; i < std::min(tensor_cpu.size(0), 100L); i++)
                        {
                            std::cout << static_cast<float>(accessor[i]) << " ";
                        }
                        if (tensor_cpu.size(0) > 100)
                        {
                            std::cout << "... (total elements: " << tensor_cpu.size(0) << ")";
                        }
                        std::cout << std::endl;
                    }
                    catch (std::exception const& e)
                    {
                        std::cout << "Error accessing 1D tensor: " << e.what() << std::endl;
                    }
                }
                // 如果是2维张量
                else if (tensor_cpu.dim() == 2)
                {
                    try
                    {
                        auto accessor = tensor_cpu.accessor<T, 2>();
                        for (int i = 0; i < std::min(tensor_cpu.size(0), 10L); i++)
                        {
                            for (int j = 0; j < std::min(tensor_cpu.size(1), 10L); j++)
                            {
                                std::cout << static_cast<float>(accessor[i][j]) << " ";
                            }
                            if (tensor_cpu.size(1) > 10)
                            {
                                std::cout << "...";
                            }
                            std::cout << std::endl;
                        }
                        if (tensor_cpu.size(0) > 10)
                        {
                            std::cout << "... (total rows: " << tensor_cpu.size(0) << ")" << std::endl;
                        }
                    }
                    catch (std::exception const& e)
                    {
                        std::cout << "Error accessing 2D tensor: " << e.what() << std::endl;
                    }
                }
                // 如果是高维张量
                else
                {
                    std::cout << "Tensor has " << tensor_cpu.dim() << " dimensions. Only printing shape." << std::endl;
                }

                // 6. 打印统计信息
                try
                {
                    auto tensor_float = tensor_cpu.to(torch::kFloat);
                    std::cout << "Statistics:" << std::endl;
                    std::cout << "Min: " << tensor_float.min().item<float>() << std::endl;
                    std::cout << "Max: " << tensor_float.max().item<float>() << std::endl;
                    std::cout << "Mean: " << tensor_float.mean().item<float>() << std::endl;
                    std::cout << "Std: " << tensor_float.std().item<float>() << std::endl;
                }
                catch (std::exception const& e)
                {
                    std::cout << "Error calculating statistics: " << e.what() << std::endl;
                }
            }
            catch (std::exception const& e)
            {
                std::cout << "Error moving tensor to CPU: " << e.what() << std::endl;
            }
        }
        else
        {
            std::cout << "Tensor is not on CUDA device" << std::endl;
        }

        std::cout << std::endl;
    }
    catch (std::exception const& e)
    {
        std::cout << "Error in printTorchTensor: " << e.what() << std::endl;
    }
}

// limin-todo: remove fc1_expert_weights, fc2_expert_weights
// 4个gpu：
// clsuter_size:
// cluster_rank:
// large EP:
// chunk:
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor>
moe_permute_op(torch::Tensor const& input, torch::Tensor const& token_selected_experts,
    torch::optional<torch::Tensor> token_final_scales, torch::Tensor const& fc1_expert_weights,
    torch::Tensor const& fc2_expert_weights, torch::optional<c10::ArrayRef<torch::Tensor>> quant_scales,
    torch::optional<torch::Tensor> input_sf, int64_t const num_experts_on_rank, int64_t const tp_size,
    int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
    int64_t const cluster_rank, bool min_latency_mode, bool use_fp8_block_scaling)
{
    // // limin-todo: is this usefull?
    // std::lock_guard<std::mutex> lock(mMutex);
    // // Free the profile workspace to save memory
    // freeProfileWorkspace();

    // int64_t mInnerDimMultiplier = 1;
    cutlass_kernels::CubKeyValueSorter sorter_;

    TORCH_CHECK(cluster_size == 1 && cluster_rank == 0, "smart_router is supported in min_latency mode");
    TORCH_CHECK(min_latency_mode == false, "min_latency_mode is not supported now");

    // limin-todo:
    // CHECK_INPUT(input, mActivationDtype)
    CHECK_INPUT(token_selected_experts, at::ScalarType::Int)
    if (token_final_scales)
    {
        CHECK_INPUT(token_final_scales.value(), at::ScalarType::Float)
    }
    // CHECK_INPUT(fc1_expert_weights, mWeightDtype)
    // CHECK_INPUT(fc2_expert_weights, mWeightDtype)

    TORCH_CHECK(input.dim() == 2, "input must be 2D.");
    TORCH_CHECK(token_selected_experts.dim() == 2, "token_selected_experts must be 2D.");

    // TORCH_CHECK(fc1_expert_weights.dim() == 3, "fc1_expert_weights must be 3D.");
    // TORCH_CHECK(fc2_expert_weights.dim() == 3, "fc2_expert_weights must be 3D.");
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
    // TORCH_CHECK(fc1_expert_weights.sizes()[0] == fc2_expert_weights.sizes()[0],
    //     "fc1_expert_weights and fc2_expert_weights must have the same number of experts.");
    // TORCH_CHECK(fc1_expert_weights.sizes()[1] == fc2_expert_weights.sizes()[2] * mInnerDimMultiplier * 2,
    //     "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.");

    int experts_per_token = token_selected_experts.sizes()[1];
    int64_t num_rows = input.sizes()[0];
    int64_t hidden_size = input.sizes()[1];
    // int64_t hidden_size = fc2_expert_weights.sizes()[1];
    // int64_t inter_size = fc2_expert_weights.sizes()[2] * mInnerDimMultiplier;
    // int const num_experts_on_rank = fc2_expert_weights.sizes()[0];
    auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
    auto parallelism_config = cutlass_kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
    auto activation_type = tensorrt_llm::ActivationType::Swiglu;

    int const num_experts_per_node = num_experts_on_rank;

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    // // limin-todo: what's the output?
    // size_t num_moe_inputs
    //     = use_fp8_block_scaling ? (experts_per_token * num_rows + 3) / 4 * 4 : experts_per_token * num_rows;
    size_t num_moe_inputs = experts_per_token * num_rows;
    // num_moe_inputs = min_latency_mode ? num_experts_per_node * num_rows : num_moe_inputs;
    // size_t const permuted_elems = num_moe_inputs * hidden_size;
    // std::cout << "limin: moe_permute_op, num_rows, experts_per_token, num_moe_inputs: " << num_rows << ", " <<
    // experts_per_token << ", " << num_moe_inputs << std::endl;

    // size_t const unpermuted_token_selected_experts_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    // size_t const unpermuted_source_token_ids_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    // size_t const permuted_source_token_ids_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    // size_t const permuted_token_selected_experts_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    // size_t const permuted_data_size = permuted_elems * dtype_size;
    // size_t const expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t);
    // size_t const permuted_token_final_scales_size = num_moe_inputs * sizeof(float);

    // std::vector<int> unpermuted_token_selected_experts_shape = {num_moe_inputs};
    auto unpermuted_token_selected_experts_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    // std::vector<int> unpermuted_source_token_ids_shape = {num_moe_inputs};
    auto unpermuted_source_token_ids_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    // std::vector<int> permuted_source_token_ids_shape = {num_moe_inputs};
    auto permuted_source_token_ids_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    // std::vector<int> permuted_token_selected_experts_shape = {num_moe_inputs};
    auto permuted_token_selected_experts_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    // std::vector<int> permuted_data_shape = {num_moe_inputs, hidden_size};
    auto permuted_data_tensor = torch::empty({num_moe_inputs, hidden_size}, input.options().requires_grad(false));

    // std::vector<int> permuted_token_final_scales_shape = {num_moe_inputs};
    auto permuted_token_final_scales_tensor
        = torch::empty({num_moe_inputs}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));

    // std::vector<int64_t> expert_first_token_offset_shape = {num_experts_per_node + 1};
    // num_tokens * tok_k
    // [0, 2, 5, 5, 7] => [4, 5, 6]
    // m_indices: 00001111
    auto expert_first_token_offset_tensor = torch::empty(
        {num_experts_per_node + 1}, torch::dtype(torch::kInt64).device(torch::kCUDA).requires_grad(false));

    size_t const sorter_size = min_latency_mode
        ? 0
        : cutlass_kernels::CubKeyValueSorter::getWorkspaceSize(num_rows * experts_per_token, num_experts_per_node);
    // std::vector<int> sorter_ws_shape = {sorter_size};
    // limin-todo: use kInt8 or kChar?
    auto sorter_ws_tensor
        = torch::empty({sorter_size}, torch::dtype(torch::kChar).device(torch::kCUDA).requires_grad(false));
    // std::cout << "sorter_ws_tensor.size() = " << sorter_ws_tensor.sizes() << std::endl;

    // experts_per_token * num_rows * sizeof(int);
    // std::vector<int64_t> src_to_dest_map_shape = {experts_per_token, num_rows};
    auto src_to_dest_map_tensor = torch::empty(
        {experts_per_token * num_rows}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    // // limin-todo:
    // WorkspaceInfo workspace_info = getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total,
    //     static_cast<int>(experts_per_token), activation_type, parallelism_config, min_latency_mode);

    // limin-todo:
    // auto const quant_params = getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales);
    cutlass_kernels::QuantParams quant_params{};
    cutlass_kernels::MoeMinLatencyParams min_latency_params{};

    // TODO: support lora in the future
    kernels::LoraParams lora_params{};
    // TODO: limin-todo
    // std::cout << "limin: moeOp.cpp, runMoe, call mKernelRunner->runMoe" << std::endl;
    // std::cout << "limin: moeOp.cpp, runMoe, quant_params: " << quant_params << std::endl;
    // error: no matching function for call to ‘runPermute(void const*, void const*, int const*, float const*,
    // void const*, std::nullptr_t, tensorrt_llm::ActivationType&, void const*, std::nullptr_t,
    // tensorrt_llm::kernels::QuantParams&, int64_t&, int64_t&, int64_t&, int const&, int,
    // void*, void*, void*, void*, void*, void*, void*, void*, void*, tensorrt_llm::kernels::MOEParallelismConfig&,
    // bool, tensorrt_llm::kernels::LoraParams&, bool&, bool&, tensorrt_llm::kernels::MoeMinLatencyParams&,
    // c10::cuda::CUDAStream&)’
    // void runPermute(void const* input_activations_void, void const* input_sf_void, int const* token_selected_experts,
    // float const* token_final_scales, void const* fc1_expert_weights_void, void const* fc1_expert_biases_void,
    // tensorrt_llm::ActivationType fc1_activation_type, void const* fc2_expert_weights_void,
    // void const* fc2_expert_biases_void, kernels::QuantParams quant_params, int64_t const num_rows,
    // int64_t const hidden_size, int64_t const inter_size, int const full_num_experts,
    // int const experts_per_token, /*char* workspace_ptr, void* final_output_void, */
    // int* unpermuted_token_selected_experts_, int* unpermuted_source_token_ids_, int* permuted_source_token_ids_,
    // int* permuted_token_selected_experts_, char* sorter_ws_, T* permuted_data_, float* permuted_token_final_scales_,
    // int64_t* expert_first_token_offset_, int* expanded_source_row_to_expanded_dest_row,
    // kernels::MOEParallelismConfig parallelism_config, bool use_lora, kernels::LoraParams& lora_params,
    // bool use_fp8_block_scaling, bool min_latency_mode, kernels::MoeMinLatencyParams& min_latency_params,
    // cudaStream_t stream)

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
            /* static_cast<char*>(workspace_info.workspace), */
            /* output.data_ptr(), */
            static_cast<int*>(unpermuted_token_selected_experts_tensor.data_ptr()),
            static_cast<int*>(unpermuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_token_selected_experts_tensor.data_ptr()),
            static_cast<float*>(permuted_data_tensor.data_ptr()), static_cast<char*>(sorter_ws_tensor.data_ptr()),
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            /* static_cast<int*>(workspace_info.src_to_dest_map)*/
            static_cast<int*>(src_to_dest_map_tensor.data_ptr()), parallelism_config, sorter_, false, lora_params,
            /* mUseFp8BlockScaling */ use_fp8_block_scaling, min_latency_mode, min_latency_params, stream);
        break;
    case torch::kBFloat16:
        // std::cout << "before runPermute<__nv_bfloat16>" << std::endl;
        runPermute<__nv_bfloat16>(input.const_data_ptr(),
            input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            /*fc1_expert_weights.const_data_ptr()*/ nullptr, nullptr, activation_type,
            /*fc2_expert_weights.const_data_ptr()*/ nullptr, nullptr, quant_params, num_rows, hidden_size,
            num_experts_total, static_cast<int>(experts_per_token),
            /* static_cast<char*>(workspace_info.workspace), */
            /* output.data_ptr(), */
            static_cast<int*>(unpermuted_token_selected_experts_tensor.data_ptr()),
            static_cast<int*>(unpermuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_token_selected_experts_tensor.data_ptr()),
            static_cast<__nv_bfloat16*>(permuted_data_tensor.data_ptr()),
            static_cast<char*>(sorter_ws_tensor.data_ptr()),
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            /* static_cast<int*>(workspace_info.src_to_dest_map)*/
            static_cast<int*>(src_to_dest_map_tensor.data_ptr()), parallelism_config, sorter_, false, lora_params,
            /* mUseFp8BlockScaling */ use_fp8_block_scaling, min_latency_mode, min_latency_params, stream);
        // std::cout << "after runPermute<__nv_bfloat16>" << std::endl;
        break;
    case torch::kHalf:
        runPermute<half>(input.const_data_ptr(), input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            /*fc1_expert_weights.const_data_ptr()*/ nullptr, nullptr, activation_type,
            /*fc2_expert_weights.const_data_ptr()*/ nullptr, nullptr, quant_params, num_rows, hidden_size,
            num_experts_total, static_cast<int>(experts_per_token),
            /* static_cast<char*>(workspace_info.workspace), */
            /* output.data_ptr(), */
            static_cast<int*>(unpermuted_token_selected_experts_tensor.data_ptr()),
            static_cast<int*>(unpermuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_source_token_ids_tensor.data_ptr()),
            static_cast<int*>(permuted_token_selected_experts_tensor.data_ptr()),
            static_cast<half*>(permuted_data_tensor.data_ptr()), static_cast<char*>(sorter_ws_tensor.data_ptr()),
            static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()),
            static_cast<float*>(permuted_token_final_scales_tensor.data_ptr()),
            /* static_cast<int*>(workspace_info.src_to_dest_map)*/
            static_cast<int*>(src_to_dest_map_tensor.data_ptr()), parallelism_config, sorter_, false, lora_params,
            /* mUseFp8BlockScaling */ use_fp8_block_scaling, min_latency_mode, min_latency_params, stream);
        break;
    default:
        // Handle other data types
        throw std::invalid_argument(
            "Invalid dtype, only supports intput tensor with float32, float16 and bfloat16 dtype");
        break;
    }
    // printTorchTensor<int>(unpermuted_token_selected_experts_tensor, "unpermuted_token_selected_experts_tensor");
    // printTorchTensor<int>(unpermuted_source_token_ids_tensor, "unpermuted_source_token_ids_tensor");
    // printTorchTensor<int>(permuted_token_selected_experts_tensor, "permuted_token_selected_experts_tensor");
    // printTorchTensor<int>(permuted_source_token_ids_tensor, "permuted_source_token_ids_tensor");
    // printTorchTensor<int64_t>(expert_first_token_offset_tensor, "expert_first_token_offset_tensor");
    // printTorchTensor<int>(src_to_dest_map_tensor, "src_to_dest_map_tensor");
    // final op 需要的input需要作为这个permute op的输出返回
    // src_to_dest_map_tensor, unpermuted_token_selected_experts
    // group gemm需要的input需要作为这个permute op的输出
    // permuted_data_tensor, expert_first_token_offset_tensor
    // fusedgemm_finalize op需要使用permuted_token_final_scales_tensor
    return std::make_tuple(unpermuted_token_selected_experts_tensor, unpermuted_source_token_ids_tensor,
        permuted_source_token_ids_tensor, permuted_token_selected_experts_tensor, permuted_data_tensor,
        expert_first_token_offset_tensor, permuted_token_final_scales_tensor, src_to_dest_map_tensor);
    // template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
    // void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::runMoe(
    //     void const* input_activations_void, void const* input_sf_void, int const* token_selected_experts,
    //     float const* token_final_scales, void const* fc1_expert_weights_void, void const* fc1_expert_biases_void,
    //     ActivationType fc1_activation_type, void const* fc2_expert_weights_void, void const* fc2_expert_biases_void,
    //     QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
    //     int const full_num_experts, int const experts_per_token, char* workspace_ptr, void* final_output_void,
    //     int* expanded_source_row_to_expanded_dest_row, MOEParallelismConfig parallelism_config, bool use_lora,
    //     LoraParams& lora_params, bool use_fp8_block_scaling, bool min_latency_mode,
    //     MoeMinLatencyParams& min_latency_params, cudaStream_t stream)
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_moe_expand_op(torch::Tensor const& input,
    torch::optional<torch::Tensor> token_final_scales, torch::Tensor const& permuted_source_token_ids,
    int64_t const num_rows, torch::Tensor& expert_first_token_offset_tensor, int64_t const hidden_size,
    int64_t const experts_per_token, int64_t const num_experts_per_node, int64_t const tp_size, int64_t const tp_rank,
    int64_t const ep_size, int64_t const ep_rank, bool use_fp8_block_scaling)
{
    // input_activations: [num_tokens, hidden_size]
    // output: permuted_data_, [num_token * k, hidden_size]
    // input: token_topk_unpermuted_scales, [num_tokens, k]
    // output: permuted_token_final_scales_, [num_tokens * k]
    // input: permuted_source_token_ids_, [num_tokens, k]
    // output: expanded_source_row_to_expanded_dest_row, [num_tokens, k]
    // input: num_rows, num_valid_tokens_ptr, hidden_size, experts_per_token, num_experts_per_node,
    // quant_params.fp4.fc1.act_global_scale, expert_first_token_offset_, fc1_fp4_act_scale_, input_sf
    auto parallelism_config = cutlass_kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);

    bool const needs_num_valid = parallelism_config.ep_size > 1;
    // TODO: expert_first_token_offset_[num_experts_per_node]存储了所有展开后的token总数
    int64_t const* num_valid_tokens_ptr = needs_num_valid
        ? static_cast<int64_t*>(expert_first_token_offset_tensor.data_ptr()) + num_experts_per_node
        : nullptr;

    size_t num_moe_inputs
        = use_fp8_block_scaling ? (experts_per_token * num_rows + 3) / 4 * 4 : experts_per_token * num_rows;
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
            "Invalid dtype, only supports intput tensor with float32, float16 and bfloat16 dtype");
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

// gemm2_output: [expanded_num_tokens, hidden_size]
// unpermuted_final_scales: token_topk_unpermuted_scales[num_tokens, k]
// expanded_source_row_to_expanded_dest_row:[expanded_num_tokens]
// expert_for_source_row: unpermuted_token_selected_experts_[expaned_num_tokens]
// num_rows: num_token for original input/output
// experts_per_token: topk's k
torch::Tensor run_moe_finalize_scale_op(torch::Tensor const& gemm2_output, torch::Tensor const& fc2_expert_biases,
    torch::Tensor const& unpermuted_final_scales, torch::Tensor const& expanded_source_row_to_expanded_dest_row,
    torch::Tensor const& expert_for_source_row,
    /*torch::Tensor const& num_valid_tokens_ptr,*/ torch::Tensor const& expert_first_token_offset_tensor,
    int64_t const num_rows, int64_t const hidden_size, int64_t const experts_per_token,
    int64_t const num_experts_per_node, int64_t const tp_size, int64_t const tp_rank, int64_t const ep_size,
    int64_t const ep_rank)
{
    // TODO: check on inputs, shape, dtype, etc.
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
    // TODO: expert_first_token_offset_[num_experts_per_node]存储了所有展开后的token总数
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
            "Invalid dtype, only supports intput tensor with float32, float16 and bfloat16 dtype");
        break;
    }
    return final_output;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "renorm_permutate_op(Tensor router_logits, int topk"
        ") -> (Tensor, Tensor)");
    m.def(
        "moe_permute_op(Tensor input, Tensor token_selected_experts, Tensor? token_final_scales, Tensor "
        "fc1_expert_weights, Tensor fc2_expert_weights, Tensor[]? quant_scales, Tensor? input_sf, int "
        "num_experts_on_rank, int tp_size, int tp_rank, int ep_size, int ep_rank, int cluster_size, int cluster_rank, "
        "bool min_latency_mode, bool use_fp8_block_scaling)"
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "moe_finalize_scale_op(Tensor gemm2_output, Tensor fc2_expert_biases, Tensor unpermuted_final_scales, Tensor "
        "expanded_source_row_to_expanded_dest_row, Tensor expert_for_source_row, Tensor "
        "expert_first_token_offset_tensor, int num_rows, int hidden_size, int experts_per_token, int "
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
    m.impl("renorm_permutate_op", &torch_ext::renorm_permutate_op);
    m.impl("moe_permute_op", &torch_ext::moe_permute_op);
    m.impl("moe_finalize_scale_op", &torch_ext::run_moe_finalize_scale_op);
    m.impl("moe_expand_op", &torch_ext::run_moe_expand_op);
}
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/ops/index_select.h>

namespace torch_ext
{
namespace btg = batchedGemm::trtllm::gen;
using tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::RoutingMethodType;
using MoeRunnerType = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner;

std::vector<torch::Tensor> run_fp4_block_scale_moe_runner(torch::optional<torch::Tensor> const& routing_logits,
    torch::optional<torch::Tensor> const& routing_bias, torch::Tensor const& hidden_states,
    torch::optional<torch::Tensor> const& hidden_states_scale, torch::Tensor const& gemm1_weights,
    torch::Tensor const& gemm1_weights_scale, torch::Tensor const& gemm2_weights,
    torch::Tensor const& gemm2_weights_scale, torch::Tensor const& output1_scales_scalar,
    torch::Tensor const& output1_scales_gate_scalar, torch::Tensor const& output2_scales_scalar,
    int64_t const num_experts, int64_t const top_k, std::optional<int64_t> const n_group,
    std::optional<int64_t> const topk_group, int64_t const intermediate_size, int64_t const local_expert_offset,
    int64_t const local_num_experts, std::optional<double> const routed_scaling_factor, int64_t const tile_tokens_dim,
    int64_t const routing_method_type, bool const do_finalize, btg::Dtype const dtype, MoeRunnerType& moe_runner,
    int64_t const moeConfigIndex, torch::optional<torch::Tensor> const& topk_weights,
    torch::optional<torch::Tensor> const& topk_ids)
{
    TORCH_CHECK(dtype == btg::Dtype::E4m3 || dtype == btg::Dtype::E2m1, "dtype can only be e4m3 or e2m1.");
    TORCH_CHECK(tensorrt_llm::common::isSM100Family(), "Only SM100f is supported by FP4 block scale MOE");
    TORCH_CHECK(tile_tokens_dim == 8 || tile_tokens_dim == 16 || tile_tokens_dim == 32 || tile_tokens_dim == 64
            || tile_tokens_dim == 128,
        "tile_tokens_dim must be 8, 16, 32, 64, 128");

    if (topk_ids.has_value() && topk_weights.has_value())
    {
        TORCH_CHECK(topk_ids.value().scalar_type() == at::ScalarType::Int, "topk_ids must be int");
        TORCH_CHECK(topk_weights.value().scalar_type() == at::ScalarType::BFloat16, "topk_weights must be bfloat16.");
        TORCH_CHECK(topk_ids.value().dim() == 2, "topk_ids must be 2D.");
        TORCH_CHECK(topk_ids.value().sizes()[0] == hidden_states.sizes()[0],
            "topk_ids and hidden_states must have the same number of tokens.");
        TORCH_CHECK(topk_ids.value().sizes()[1] == top_k, "topk_ids dim1 must match top_k.");
        TORCH_CHECK(topk_weights.value().dim() == 2, "topk_weights must be 2D.");
        TORCH_CHECK(topk_weights.value().sizes()[0] == hidden_states.sizes()[0],
            "topk_weights and hidden_states must have the same number of tokens.");
        TORCH_CHECK(topk_weights.value().sizes()[1] == top_k, "topk_weights dim1 must match top_k.");
    }
    else if (routing_logits.has_value())
    {
        if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::DeepSeekV3)
        {
            TORCH_CHECK(routing_logits.value().scalar_type() == at::ScalarType::Float, "routing_logits must be float");
        }
        else
        {
            TORCH_CHECK(
                routing_logits.value().scalar_type() == at::ScalarType::BFloat16, "routing_logits must be bfloat16");
        }
        TORCH_CHECK(routing_logits.value().dim() == 2, "routing_logits must be 2D.");
        TORCH_CHECK(routing_logits.value().sizes()[1] == num_experts, "routing_logits has incorrect shape.");
    }
    else
    {
        TORCH_CHECK(false, "routing_logits or (topk_ids and topk_weights) must be provided.");
    }

    if (topk_ids.has_value() && topk_weights.has_value() && routing_logits.has_value())
    {
        TLLM_LOG_WARNING(
            "When logits and (topk_ids and topk_weights) are both provided, we only use (topk_ids and topk_weights).");
    }

    if (topk_ids.has_value())
    {
        TORCH_CHECK(topk_ids.value().sizes()[0] == hidden_states.sizes()[0],
            "topk_ids and hidden_states must have the same number of tokens.");
    }
    else
    {
        TORCH_CHECK(routing_logits.value().sizes()[0] == hidden_states.sizes()[0],
            "routing_logits and hidden_states must have the same number of tokens.");
    }

    if (routing_bias.has_value())
    {
        TORCH_CHECK(routing_bias.value().scalar_type() == at::ScalarType::BFloat16, "routing_bias must be bfloat16.");
        TORCH_CHECK(routing_bias.value().dim() == 1, "routing_bias must be 1D.");
        TORCH_CHECK(routing_bias.value().sizes()[0] == num_experts, "routing_bias has incorrect shape.");
    }

    if (n_group.has_value() && n_group.value() != 0)
    {
        TORCH_CHECK(static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::DeepSeekV3,
            "Routing kernel with groups implies DeepSeekV3 routing method.");
        TORCH_CHECK(topk_group.has_value(), "if n_group is given, topk_group must be given");
        TORCH_CHECK(num_experts % n_group.value() == 0, "num_experts must be divisible by n_group");
        TORCH_CHECK(top_k <= 8 && top_k > 0, "Current routing kernel (with groups) only supports top_k<=8 && top_k>0.");
        TORCH_CHECK(topk_group.value() <= 4 && topk_group.value() > 0,
            "Current routing kernel only (with groups) supports topk_group<=4 && topk_group > 0.");
        TORCH_CHECK(topk_group.value() <= n_group.value(), "n_group must not be smaller than topk_group.");
        // This check ensures we have enough experts in the selected groups to handle the top_k routing
        TORCH_CHECK(top_k < (topk_group.value() * num_experts / n_group.value()),
            "top_k must be less than total number of experts in selected groups");
    }
    else if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Renormalize
        || static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::RenormalizeNaive)
    {
        TORCH_CHECK(top_k <= 10 && top_k > 0,
            "Current routing kernel (no groups, renormalize) only supports top_k<=10 && top_k>0.");
    }
    else if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Llama4)
    {
        TORCH_CHECK(top_k == 1, "Current routing kernel (no groups, Llama4) only supports top_k=1.");
    }

    TORCH_CHECK(num_experts % 4 == 0, "Routing kernel expects that num_experts must be divisible by 4");
    TORCH_CHECK(num_experts > top_k, "num_experts must be greater than top_k");
    TORCH_CHECK(num_experts <= 512, "num_experts must be less than or equal to 512");

    // If both routing inputs are provided, they must be on the same device
    if (routing_logits.has_value() && topk_ids.has_value())
    {
        TORCH_CHECK(
            routing_logits->device() == topk_ids->device(), "routing_logits and topk_ids must be on the same device");
    }

    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoERunnerArgs args;
    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoEWorkspace workspace;

    // setup args
    // note: the assumption is that output data type is always Bfloat16 (the default)
    auto const routing_bias_dtype
        = routing_bias.has_value() ? routing_bias.value().scalar_type() : at::ScalarType::BFloat16;
    args.mDtypeElt = dtype;
    args.mDtypeExpW = routing_bias_dtype == at::ScalarType::Float ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
    args.routing_logits = routing_logits.has_value() ? routing_logits.value().data_ptr() : nullptr;
    args.routing_bias = routing_bias.has_value() ? routing_bias.value().data_ptr() : nullptr;
    args.hidden_states = hidden_states.data_ptr();
    args.hidden_states_scale = hidden_states_scale.has_value() ? hidden_states_scale.value().data_ptr() : nullptr;

    args.topk_weights = topk_weights.has_value() ? topk_weights.value().data_ptr() : nullptr;
    args.topk_ids = topk_ids.has_value() ? static_cast<int32_t*>(topk_ids.value().data_ptr()) : nullptr;

    args.gemm1_weights = gemm1_weights.data_ptr();
    args.gemm1_weights_scale = gemm1_weights_scale.data_ptr();
    args.gemm2_weights = gemm2_weights.data_ptr();
    args.gemm2_weights_scale = gemm2_weights_scale.data_ptr();
    args.num_tokens = hidden_states.sizes()[0];
    args.num_experts = num_experts;
    if (dtype == btg::Dtype::E4m3)
    {
        args.hidden_size = hidden_states.sizes()[1];
    }
    else // E2m1
    {
        // * 2 to compensate for the fact that sizeof(hidden_states.dtype) is 1 because we pack 2 e2m1 into 1 byte.
        args.hidden_size = hidden_states.sizes()[1] * 2;
    }
    args.top_k = top_k;
    args.n_group = n_group.value_or(0);
    args.topk_group = topk_group.value_or(0);
    args.local_expert_offset = local_expert_offset;
    args.local_num_experts = local_num_experts;
    args.routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args.intermediate_size = intermediate_size;

    // allocate workspace for routing kernel
    if (routing_logits.has_value() && topk_ids.has_value())
    {
        TORCH_CHECK(routing_logits.value().device() == topk_ids.value().device(),
            "routing_logits and topk_ids must be on the same device");
    }
    auto routing_device = routing_logits.has_value() ? routing_logits.value().device() : topk_ids.value().device();
    at::Tensor num_tokens_per_expert
        = at::detail::empty_cuda({num_experts}, at::ScalarType::Int, routing_device, std::nullopt);
    int32_t max_num_padded_tokens
        = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::getMaxPermutedPaddedCount(
            args.num_tokens, top_k, num_experts, tile_tokens_dim);
    int32_t max_num_padded_tokens_gemm1
        = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::maybeGetMinTokenCount(
            max_num_padded_tokens, args.intermediate_size, btg::dtypeGetNumBits(args.mDtypeElt));
    int32_t max_num_padded_tokens_gemm2
        = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::maybeGetMinTokenCount(
            max_num_padded_tokens, args.hidden_size, btg::dtypeGetNumBits(args.mDtypeOut));
    at::Tensor total_num_padded_tokens
        = at::empty({}, at::TensorOptions().device(routing_device).dtype(at::ScalarType::Int));
    at::Tensor expanded_idx_to_permuted_idx
        = at::detail::empty_cuda({args.num_tokens, args.top_k}, at::ScalarType::Int, routing_device, std::nullopt);

    at::Tensor permuted_idx_to_token_idx
        = at::detail::empty_cuda({max_num_padded_tokens}, at::ScalarType::Int, routing_device, std::nullopt);
    at::Tensor expert_weights
        = at::detail::empty_cuda({args.num_tokens, args.top_k}, routing_bias_dtype, routing_device, std::nullopt);
    at::Tensor expert_indexes
        = at::detail::empty_cuda({args.num_tokens, args.top_k}, at::ScalarType::Int, routing_device, std::nullopt);
    int64_t const size_of_expert_count_histogram = std::max(num_experts * 2, int64_t(256 * 2));
    at::Tensor expert_count_histogram
        = at::detail::empty_cuda({size_of_expert_count_histogram}, at::ScalarType::Int, routing_device, std::nullopt);

    // allocate workspace for activation/gemm/finalize kernels
    at::Tensor gemm1_output;
    at::Tensor gemm1_output_scale;
    if (dtype == btg::Dtype::E4m3)
    {
        gemm1_output = at::detail::empty_cuda({max_num_padded_tokens_gemm1, intermediate_size},
            at::ScalarType::Float8_e4m3fn, hidden_states.device(), std::nullopt);
    }
    else
    {
        gemm1_output = at::detail::empty_cuda({max_num_padded_tokens_gemm1, intermediate_size / 2},
            at::ScalarType::Float8_e4m3fn, routing_device, std::nullopt);

        int64_t sf_size
            = tensorrt_llm::computeSwizzledLayoutSFSize(max_num_padded_tokens_gemm1, intermediate_size / 16);
        gemm1_output_scale
            = at::detail::empty_cuda({sf_size}, at::ScalarType::Float8_e4m3fn, routing_device, std::nullopt);
    }

    at::Tensor gemm2_output = at::detail::empty_cuda(
        {max_num_padded_tokens_gemm2, args.hidden_size}, at::ScalarType::BFloat16, routing_device, std::nullopt);

    int32_t max_num_ctas = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::getMaxNumCtasInBatchDim(
        args.num_tokens, args.top_k, args.num_experts, tile_tokens_dim);
    at::Tensor cta_idx_xy_to_batch_idx
        = at::detail::empty_cuda({max_num_ctas}, at::ScalarType::Int, routing_device, std::nullopt);
    at::Tensor cta_idx_xy_to_mn_limit
        = at::detail::empty_cuda({max_num_ctas}, at::ScalarType::Int, routing_device, std::nullopt);
    at::Tensor num_non_exiting_ctas
        = at::empty({}, at::TensorOptions().device(routing_device).dtype(at::ScalarType::Int));

    // Set the optional pointer to the expert weights and expert ids
    void* expert_weights_ptr = args.topk_weights ? args.topk_weights : expert_weights.data_ptr();

    //
    // TopK routing
    //

    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::Runner routing_runner(tile_tokens_dim);
    auto const& stream = at::cuda::getCurrentCUDAStream(
        routing_logits.has_value() ? routing_logits.value().get_device() : topk_ids.value().get_device());
    routing_runner.run(args.routing_logits, args.routing_bias, args.num_tokens, args.num_experts, args.top_k,
        args.n_group, args.topk_group, args.local_expert_offset, args.local_num_experts, args.routed_scaling_factor,
        expert_indexes.data_ptr<int>(), expert_count_histogram.data_ptr<int>(), total_num_padded_tokens.data_ptr<int>(),
        expanded_idx_to_permuted_idx.data_ptr<int>(), nullptr, /*permuted_idx_to_expanded_idx.data_ptr<int>(),*/
        permuted_idx_to_token_idx.data_ptr<int>(), expert_weights_ptr, args.topk_ids,
        num_tokens_per_expert.data_ptr<int>(), cta_idx_xy_to_batch_idx.data_ptr<int>(),
        cta_idx_xy_to_mn_limit.data_ptr<int>(), num_non_exiting_ctas.data_ptr<int>(), args.mDtypeElt,
        false /* use_routing_scales_on_input */, false /* use_deep_seek_fp8 */,
        static_cast<RoutingMethodType>(routing_method_type), stream);

    //
    // FC13 (gemm1) + FC2 (gemm2)
    //

    if (dtype == btg::Dtype::E4m3)
    {
        TORCH_CHECK(hidden_states.scalar_type() == at::ScalarType::Float8_e4m3fn, "hidden_states must be e4m3.");
        TORCH_CHECK(!hidden_states_scale.has_value(), "hidden_states_scale must be nullopt.");
        TORCH_CHECK(gemm1_weights.sizes()[2] * 2 == hidden_states.sizes()[1],
            "the third dimension of weights must be equal to hidden_size.");
        TORCH_CHECK(
            gemm1_weights_scale.sizes()[2] == args.hidden_size / 32, "gemm1_weights_scale has incorrect dim 2.");
        TORCH_CHECK(
            gemm2_weights_scale.sizes()[2] == intermediate_size / 32, "gemm2_weights_scale has incorrect dim 2.");
    }
    else
    {
        TORCH_CHECK(hidden_states.scalar_type() == FLOAT4_E2M1X2, "hidden_states must be byte.");
        TORCH_CHECK(hidden_states_scale.value().scalar_type() == at::ScalarType::Float8_e4m3fn,
            "hidden_states_scale must be fp8.");

        TORCH_CHECK(hidden_states_scale.value().dim() == 1, "hidden_states_scale must be 1D.");
        TORCH_CHECK(hidden_states_scale.value().sizes()[0]
                == tensorrt_llm::computeLinearLayoutSFSize(args.num_tokens, args.hidden_size / 16),
            "hidden_states_scale has incorrect size");
        // This check passes even though the actual shape of the weights[2] and hidden_states[1] is
        // 2 times larger due to the fact that 2 e2m1 are packed into 1 byte.
        TORCH_CHECK(gemm1_weights.sizes()[2] == hidden_states.sizes()[1],
            "the third dimension of weights must be equal to hidden_size.");
        TORCH_CHECK(
            gemm1_weights_scale.sizes()[2] == args.hidden_size / 16, "gemm1_weights_scale has incorrect dim 2.");
        TORCH_CHECK(
            gemm2_weights_scale.sizes()[2] == intermediate_size / 16, "gemm2_weights_scale has incorrect dim 2.");
    }
    TORCH_CHECK(gemm1_weights.scalar_type() == FLOAT4_E2M1X2, "gemm1_weights must be byte.");

    TORCH_CHECK(gemm1_weights.dim() == 3, "gemm1_weights must be 3D.");
    TORCH_CHECK(gemm1_weights.sizes()[1] % 2 == 0, "the second dimension of weights must be even.");
    TORCH_CHECK(intermediate_size == gemm1_weights.sizes()[1] / 2, "intermediate_size has incorrect dim 1.");

    TORCH_CHECK(gemm1_weights_scale.scalar_type() == at::ScalarType::Float8_e4m3fn, "gemm1_weights_scale must be fp8.");

    TORCH_CHECK(gemm1_weights_scale.dim() == 3, "gemm1_weights_scale must be 3D.");
    TORCH_CHECK(gemm1_weights_scale.sizes()[0] == local_num_experts, "gemm1_weights_scale has incorrect dim 0.");
    TORCH_CHECK(intermediate_size % 16 == 0, "the second dimension of weights must be a multiple of 16.");
    TORCH_CHECK(gemm1_weights_scale.sizes()[1] == 2 * intermediate_size, "gemm1_weights_scale has incorrect dim 1.");

    TORCH_CHECK(gemm2_weights.scalar_type() == FLOAT4_E2M1X2, "gemm2_weights must be byte.");

    TORCH_CHECK(gemm2_weights.dim() == 3, "gemm2_weights must be 3D.");
    // / 2 to compensate for the fact that we pack 2 e2m1 into 1 byte.
    TORCH_CHECK(gemm2_weights.sizes()[2] == intermediate_size / 2,
        "the third dimension of weights must be equal to intermediate_size.");

    TORCH_CHECK(gemm2_weights_scale.scalar_type() == at::ScalarType::Float8_e4m3fn, "gemm2_weights_scale must be fp8.");

    TORCH_CHECK(gemm2_weights_scale.dim() == 3, "gemm2_weights_scale must be 3D.");
    TORCH_CHECK(gemm2_weights_scale.sizes()[0] == local_num_experts, "gemm2_weights_scale has incorrect dim 0.");
    TORCH_CHECK(gemm2_weights_scale.sizes()[1] == args.hidden_size, "gemm2_weights_scale has incorrect dim 1.");

    TORCH_CHECK(output1_scales_scalar.scalar_type() == at::ScalarType::Float, "output1_scales_scalar must be float.");
    TORCH_CHECK(output1_scales_scalar.dim() == 1, "output1_scales_scalar must be 1D.");
    TORCH_CHECK(output1_scales_scalar.sizes()[0] == local_num_experts, "output1_scales_scalar has incorrect dim 0.");

    TORCH_CHECK(
        output1_scales_gate_scalar.scalar_type() == at::ScalarType::Float, "output1_scales_gate_scalar must be float.");
    TORCH_CHECK(output1_scales_gate_scalar.dim() == 1, "output1_scales_gate_scalar must be 1D.");
    TORCH_CHECK(
        output1_scales_gate_scalar.sizes()[0] == local_num_experts, "output1_scales_gate_scalar has incorrect dim 0.");

    TORCH_CHECK(output2_scales_scalar.scalar_type() == at::ScalarType::Float, "output2_scales_scalar must be float.");
    TORCH_CHECK(output2_scales_scalar.dim() == 1, "output2_scales_scalar must be 1D.");
    TORCH_CHECK(output2_scales_scalar.sizes()[0] == local_num_experts, "output2_scales_scalar has incorrect dim 0.");

    // allocate output
    at::Tensor output = at::detail::empty_cuda(
        {args.num_tokens, args.hidden_size}, at::ScalarType::BFloat16, hidden_states.device(), std::nullopt);

    // setup workspace
    workspace.total_num_padded_tokens = total_num_padded_tokens.data_ptr<int>();
    workspace.total_max_padded_tokens = std::max(max_num_padded_tokens_gemm1, max_num_padded_tokens_gemm2);
    workspace.ProjUpTileN = tile_tokens_dim;
    workspace.routing_expert_indexes = expert_indexes.data_ptr<int>();
    workspace.permuted_idx_size = total_num_padded_tokens.data_ptr<int>();
    workspace.expanded_idx_to_permuted_idx
        = expanded_idx_to_permuted_idx.data_ptr<int>(); // Needed by permute/finalize kernels
    workspace.permuted_idx_to_token_idx = permuted_idx_to_token_idx.data_ptr<int>(); // Needed by permuteGemm1 kernel

    workspace.expert_weights = expert_weights_ptr;                                   // Consumed by finalize kernel

    workspace.cta_idx_xy_to_batch_idx = cta_idx_xy_to_batch_idx.data_ptr<int>();
    workspace.cta_idx_xy_to_mn_limit = cta_idx_xy_to_mn_limit.data_ptr<int>();
    workspace.num_non_exiting_ctas = num_non_exiting_ctas.data_ptr<int>();

    // gemm1 intermediate ws
    workspace.gemm1_output = gemm1_output.data_ptr();
    if (dtype == btg::Dtype::E4m3)
    {
        workspace.gemm1_output_scale = nullptr;
    }
    else // E2m1
    {
        workspace.gemm1_output_scale = reinterpret_cast<float*>(gemm1_output_scale.data_ptr());
    }

    // gemm2 intermediate ws
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
    args.output = output.data_ptr();
    args.output_scale = nullptr;
    args.output1_scales_scalar = output1_scales_scalar.data_ptr<float>();
    args.output1_scales_gate_scalar = output1_scales_gate_scalar.data_ptr<float>();
    args.output2_scales_scalar = output2_scales_scalar.data_ptr<float>();
    args.do_finalize = do_finalize;

    auto const workspace_sizes = moe_runner.getWorkspaceSizeInBytes(args, moeConfigIndex);

    at::Tensor workspace_fc1 = at::detail::empty_cuda(
        {std::get<0>(workspace_sizes)}, at::ScalarType::Char, hidden_states.device(), std::nullopt);
    at::Tensor workspace_fc2 = at::detail::empty_cuda(
        {std::get<1>(workspace_sizes)}, at::ScalarType::Char, hidden_states.device(), std::nullopt);
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();
    auto const& moe_stream = at::cuda::getCurrentCUDAStream(hidden_states.get_device());
    moe_runner.run(args, workspace, hidden_states.get_device(), moe_stream, moeConfigIndex);

    if (!do_finalize)
    {
        return {gemm2_output, expert_weights, expanded_idx_to_permuted_idx};
    }

    return {output};
}

// Wrapped the TRTLLM-Gen kernel runner in a Torch custom class to allow
// use with the torch workflow autotuner class.
class FP4BlockScaleMoeRunner : public torch::CustomClassHolder
{
public:
    explicit FP4BlockScaleMoeRunner()
        // Update this as new cubins come in
        : mSupportedTileN{8, 16, 32, 64, 128}
    {
        for (int tileN : mSupportedTileN)
        {
            mRunners.emplace(tileN, std::make_unique<RunnerType>(mDtypeElt, mUseDeepSeekFp8, tileN));
        }
    }

    [[nodiscard]] std::vector<std::vector<int64_t>> getValidConfigs(
        int64_t topK, int64_t hiddenSize, int64_t intermediateSize, int64_t numLocalExperts, int64_t numTokens) const
    {
        // returns (tileN, config)
        std::vector<std::vector<int64_t>> tactics;
        for (auto& [tileN, runner] : mRunners)
        {
            auto config_indices_per_runner
                = runner->getValidConfigIndices(topK, hiddenSize, intermediateSize, numLocalExperts, numTokens);
            for (auto cfg : config_indices_per_runner)
            {
                tactics.push_back({tileN, cfg});
            }
        }
        return tactics;
    }

    [[nodiscard]] std::vector<torch::Tensor> run(torch::optional<torch::Tensor> const& routing_logits,
        torch::optional<torch::Tensor> const& routing_bias, torch::Tensor const& hidden_states,
        torch::Tensor const& hidden_states_scale, torch::Tensor const& gemm1_weights,
        torch::Tensor const& gemm1_weights_scale, torch::Tensor const& gemm2_weights,
        torch::Tensor const& gemm2_weights_scale, torch::Tensor const& output1_scales_scalar,
        torch::Tensor const& output1_scales_gate_scalar, torch::Tensor const& output2_scales_scalar,
        int64_t const num_experts, int64_t const top_k, std::optional<int64_t> const n_group,
        std::optional<int64_t> const topk_group, int64_t const intermediate_size, int64_t const local_expert_offset,
        int64_t const local_num_experts, std::optional<double> const routed_scaling_factor,
        int64_t const routing_method_type, bool const do_finalize, std::vector<int64_t> moeConfigIndex,
        torch::optional<torch::Tensor> const& topk_weights, torch::optional<torch::Tensor> const& topk_ids)
    {
        // moeConfigIndex corresponds to pair (tileN, config)
        auto [tileN, config] = std::tie(moeConfigIndex[0], moeConfigIndex[1]);

        // Autotuner has requested a default or 'fallback' config index
        if (tileN == -1 || config == -1)
        {
            auto const num_tokens = hidden_states.sizes()[0];

            // 2x FP4 per byte element
            auto const hidden_size = 2 * hidden_states.sizes()[1];

            float const avg_tokens_per_expert = static_cast<float>(num_tokens * top_k) / local_num_experts;
            tileN = std::clamp(nextPowerOfTwo(avg_tokens_per_expert), mSupportedTileN.front(), mSupportedTileN.back());

            config = mRunners[tileN]->getDefaultValidConfigIndex(
                top_k, hidden_size, intermediate_size, local_num_experts, num_tokens);
        }

        return run_fp4_block_scale_moe_runner(routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, output1_scales_scalar,
            output1_scales_gate_scalar, output2_scales_scalar, num_experts, top_k, n_group, topk_group,
            intermediate_size, local_expert_offset, local_num_experts, routed_scaling_factor, tileN,
            routing_method_type, do_finalize, mDtypeElt, *mRunners[tileN], config, topk_weights, topk_ids);
    }

private:
    using RunnerType = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner;

    std::vector<int32_t> const mSupportedTileN;
    std::unordered_map<int32_t, std::unique_ptr<RunnerType>> mRunners;

    btg::Dtype mDtypeElt{btg::Dtype::E2m1};
    bool mUseDeepSeekFp8{false};
};

// Wrapped the TRTLLM-Gen kernel runner in a Torch custom class to allow
// use with the torch workflow autotuner class.
class FP8FP4BlockScaleMoeRunner : public torch::CustomClassHolder
{
public:
    explicit FP8FP4BlockScaleMoeRunner(int64_t actType)
        // Update this as new cubins come in
        : mSupportedTileN{8, 16, 32, 64}
    {
        for (int tileN : mSupportedTileN)
        {
            mRunners.emplace(tileN,
                std::make_unique<RunnerType>(mDtypeAct, mDtypeWeights, mUseDeepSeekFp8, tileN,
                    static_cast<tensorrt_llm::kernels::ActType>(actType)));
        }
    }

    [[nodiscard]] std::vector<std::vector<int64_t>> getValidConfigs(
        int64_t topK, int64_t hiddenSize, int64_t intermediateSize, int64_t numLocalExperts, int64_t numTokens) const
    {
        // returns (tileN, config)
        std::vector<std::vector<int64_t>> tactics;
        for (auto& [tileN, runner] : mRunners)
        {
            auto config_indices_per_runner
                = runner->getValidConfigIndices(topK, hiddenSize, intermediateSize, numLocalExperts, numTokens);
            for (auto cfg : config_indices_per_runner)
            {
                tactics.push_back({tileN, cfg});
            }
        }
        return tactics;
    }

    [[nodiscard]] std::vector<torch::Tensor> run(torch::optional<torch::Tensor> const& routing_logits,
        torch::optional<torch::Tensor> const& routing_bias, torch::Tensor const& hidden_states,
        torch::Tensor const& gemm1_weights, torch::Tensor const& gemm1_weights_scale,
        torch::Tensor const& gemm2_weights, torch::Tensor const& gemm2_weights_scale,
        torch::Tensor const& output1_scales_scalar, torch::Tensor const& output1_scales_gate_scalar,
        torch::Tensor const& output2_scales_scalar, int64_t const num_experts, int64_t const top_k,
        std::optional<int64_t> const n_group, std::optional<int64_t> const topk_group, int64_t const intermediate_size,
        int64_t const local_expert_offset, int64_t const local_num_experts,
        std::optional<double> const routed_scaling_factor, int64_t const routing_method_type, bool const do_finalize,
        std::vector<int64_t> moeConfigIndex, torch::optional<torch::Tensor> const& topk_weights,
        torch::optional<torch::Tensor> const& topk_ids)
    {
        // moeConfigIndex corresponds to pair (tileN, config)
        auto [tileN, config] = std::tie(moeConfigIndex[0], moeConfigIndex[1]);

        // Autotuner has requested a default or 'fallback' config index
        if (tileN == -1 || config == -1)
        {
            auto const num_tokens = hidden_states.sizes()[0];

            auto const hidden_size = hidden_states.sizes()[1];

            float const avg_tokens_per_expert = static_cast<float>(num_tokens * top_k) / local_num_experts;
            tileN = std::clamp(nextPowerOfTwo(avg_tokens_per_expert), mSupportedTileN.front(), mSupportedTileN.back());

            config = mRunners[tileN]->getDefaultValidConfigIndex(
                top_k, hidden_size, intermediate_size, local_num_experts, num_tokens);
        }

        return run_fp4_block_scale_moe_runner(routing_logits, routing_bias, hidden_states,
            std::nullopt /*hidden_states_scale*/, gemm1_weights, gemm1_weights_scale, gemm2_weights,
            gemm2_weights_scale, output1_scales_scalar, output1_scales_gate_scalar, output2_scales_scalar, num_experts,
            top_k, n_group, topk_group, intermediate_size, local_expert_offset, local_num_experts,
            routed_scaling_factor, tileN, routing_method_type, do_finalize, mDtypeAct, *mRunners[tileN], config,
            topk_weights, topk_ids);
    }

private:
    using RunnerType = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner;

    std::vector<int32_t> const mSupportedTileN;
    std::unordered_map<int32_t, std::unique_ptr<RunnerType>> mRunners;

    btg::Dtype mDtypeAct{btg::Dtype::E4m3};
    btg::Dtype mDtypeWeights{btg::Dtype::E2m1};
    bool mUseDeepSeekFp8{false};
};

torch::Tensor shuffleMatrix(torch::Tensor matrix, torch::Tensor permuteIndices)
{
    return torch::index_select(matrix, 0, permuteIndices);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::FP4BlockScaleMoeRunner>("FP4BlockScaleMoERunner")
        .def(torch::init<>())
        .def("get_valid_configs", &torch_ext::FP4BlockScaleMoeRunner::getValidConfigs)
        .def("run_moe", &torch_ext::FP4BlockScaleMoeRunner::run);
    m.class_<torch_ext::FP8FP4BlockScaleMoeRunner>("FP8FP4BlockScaleMoERunner")
        .def(torch::init<int64_t>())
        .def("get_valid_configs", &torch_ext::FP8FP4BlockScaleMoeRunner::getValidConfigs)
        .def("run_moe", &torch_ext::FP8FP4BlockScaleMoeRunner::run);
}

// Accepts both CPU and CUDA tensors
static auto shuffle_matrix = torch::RegisterOperators("trtllm::shuffle_matrix", &torch_ext::shuffleMatrix);

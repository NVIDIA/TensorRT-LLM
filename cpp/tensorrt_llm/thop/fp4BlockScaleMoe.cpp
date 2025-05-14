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

#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <ATen/cuda/EmptyTensor.h>

namespace torch_ext
{
namespace tg = trtllm::gen;

torch::Tensor fp4_block_scale_moe_runner(torch::Tensor const& routing_logits, torch::Tensor const& routing_bias,
    torch::Tensor const& hidden_states, torch::Tensor const& hidden_states_scale, torch::Tensor const& gemm1_weights,
    torch::Tensor const& gemm1_weights_scale, torch::Tensor const& gemm2_weights,
    torch::Tensor const& gemm2_weights_scale, torch::Tensor const& output1_scales_scalar,
    torch::Tensor const& output1_scales_gate_scalar, torch::Tensor const& output2_scales_scalar,
    int64_t const num_experts, int64_t const top_k, int64_t const n_group, int64_t const topk_group,
    int64_t const intermediate_size, int64_t const local_expert_offset, int64_t const local_num_experts,
    double const routed_scaling_factor)
{
    auto const sm = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sm == 100, "Only SM100 is supported by FP4 block scale MOE");
    TORCH_CHECK(routing_logits.scalar_type() == at::ScalarType::Float, "routing_logits must be float.");
    TORCH_CHECK(routing_logits.dim() == 2, "routing_logits must be 2D.");
    TORCH_CHECK(routing_logits.sizes()[1] == num_experts, "routing_logits has incorrect shape.");
    TORCH_CHECK(routing_bias.scalar_type() == at::ScalarType::BFloat16, "routing_bias must be bfloat16.");
    TORCH_CHECK(routing_bias.dim() == 1, "routing_bias must be 1D.");
    TORCH_CHECK(routing_bias.sizes()[0] == num_experts, "routing_bias has incorrect shape.");

    TORCH_CHECK(top_k == 8, "Current routing kernel only supports top_k=8.");
    TORCH_CHECK(topk_group == 4, "Current routing kernel only supports topk_group=4.");
    TORCH_CHECK(num_experts % 4 == 0, "Routing kernel expects that num_experts must be divisible by 4");
    TORCH_CHECK(num_experts % n_group == 0, "num_experts must be divisible by n_group");
    TORCH_CHECK(num_experts > top_k, "num_experts must be greater than top_k");
    // This check ensures we have enough experts in the selected groups to handle the top_k routing
    TORCH_CHECK(top_k < (topk_group * num_experts / n_group),
        "top_k must be less than total number of experts in selected groups");
    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoERunnerArgs args;
    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoEWorkspace workspace;

    // setup args
    args.mDtypeElt = tg::Dtype::E2m1;
    args.routing_logits = routing_logits.data_ptr<float>();
    args.routing_bias = routing_bias.data_ptr();
    args.hidden_states = hidden_states.data_ptr();
    args.hidden_states_scale = hidden_states_scale.data_ptr();
    args.gemm1_weights = gemm1_weights.data_ptr();
    args.gemm1_weights_scale = gemm1_weights_scale.data_ptr();
    args.gemm2_weights = gemm2_weights.data_ptr();
    args.gemm2_weights_scale = gemm2_weights_scale.data_ptr();
    args.num_tokens = hidden_states.sizes()[0];
    args.num_experts = num_experts;
    // * 2 to compensate for the fact that sizeof(hidden_states.dtype) is 1 because we pack 2 e2m1 into 1 byte.
    args.hidden_size = hidden_states.sizes()[1] * 2;
    args.top_k = top_k;
    args.n_group = n_group;
    args.topk_group = topk_group;
    args.local_expert_offset = local_expert_offset;
    args.local_num_experts = local_num_experts;
    args.routed_scaling_factor = routed_scaling_factor;
    args.intermediate_size = intermediate_size;

    // allocate workspace for routing kernel
    at::Tensor num_tokens_per_expert
        = at::detail::empty_cuda({num_experts}, at::ScalarType::Int, routing_logits.device(), std::nullopt);
    const int32_t tileN = 8; // hard code to mProjUp->mTileN=padding=8
    int32_t max_num_padded_tokens
        = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::getMaxPermutedPaddedCount(
            args.num_tokens, top_k, num_experts, tileN);
    at::Tensor total_num_padded_tokens
        = at::empty({}, at::TensorOptions().device(routing_logits.device()).dtype(at::ScalarType::Int));
    at::Tensor expanded_idx_to_permuted_idx = at::detail::empty_cuda(
        {args.num_tokens * args.top_k}, at::ScalarType::Int, routing_logits.device(), std::nullopt);

    at::Tensor permuted_idx_to_token_idx
        = at::detail::empty_cuda({max_num_padded_tokens}, at::ScalarType::Int, routing_logits.device(), std::nullopt);
    at::Tensor expert_weights = at::detail::empty_cuda(
        {args.num_tokens, args.top_k}, at::ScalarType::BFloat16, routing_logits.device(), std::nullopt);
    at::Tensor expert_indexes = at::detail::empty_cuda(
        {args.num_tokens, args.top_k}, at::ScalarType::Int, routing_logits.device(), std::nullopt);
    at::Tensor expert_count_histogram = at::detail::empty_cuda({2 * 256},
        at::ScalarType::Int, // 256 is the max number of threads per block and max number of experts
        routing_logits.device(), std::nullopt);

    // allocate workspace for activation/gemm/finalize kernels
    at::Tensor gemm1_output = at::detail::empty_cuda({max_num_padded_tokens, intermediate_size / 2},
        at::ScalarType::Float8_e4m3fn, hidden_states.device(), std::nullopt);

    at::Tensor gemm1_output_scale = at::detail::empty_cuda({max_num_padded_tokens, intermediate_size / 16},
        at::ScalarType::Float8_e4m3fn, hidden_states.device(), std::nullopt);

    at::Tensor gemm2_output = at::detail::empty_cuda(
        {max_num_padded_tokens, args.hidden_size}, at::ScalarType::BFloat16, hidden_states.device(), std::nullopt);

    int32_t max_num_ctas = (args.num_tokens + tileN - 1) / tileN * args.num_experts;
    at::Tensor cta_idx_xy_to_batch_idx
        = at::detail::empty_cuda({max_num_ctas}, at::ScalarType::Int, routing_logits.device(), std::nullopt);
    at::Tensor cta_idx_xy_to_mn_limit
        = at::detail::empty_cuda({max_num_ctas}, at::ScalarType::Int, routing_logits.device(), std::nullopt);
    at::Tensor num_non_exiting_ctas
        = at::empty({}, at::TensorOptions().device(routing_logits.device()).dtype(at::ScalarType::Int));

    // FIXME: check shape
    auto const hidden_states_scale_linear_size
        = tensorrt_llm::computeFP4LinearLayoutSFSize(args.num_tokens, args.hidden_size / 16);
    at::Tensor hidden_states_scale_linear = at::detail::empty_cuda(
        hidden_states_scale_linear_size, at::ScalarType::Float8_e4m3fn, hidden_states.device(), std::nullopt);

    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::Runner routing_runner;
    auto const& stream = at::cuda::getCurrentCUDAStream(routing_logits.get_device());
    routing_runner.run(routing_logits.data_ptr<float>(), routing_bias.data_ptr(), args.num_tokens, args.num_experts,
        args.top_k, args.n_group, args.topk_group, args.local_expert_offset, args.local_num_experts,
        args.routed_scaling_factor, expert_indexes.data_ptr<int>(), expert_count_histogram.data_ptr<int>(),
        total_num_padded_tokens.data_ptr<int>(), expanded_idx_to_permuted_idx.data_ptr<int>(),
        nullptr, /*permuted_idx_to_expanded_idx.data_ptr<int>(),*/
        permuted_idx_to_token_idx.data_ptr<int>(), expert_weights.data_ptr(), num_tokens_per_expert.data_ptr<int>(),
        cta_idx_xy_to_batch_idx.data_ptr<int>(), cta_idx_xy_to_mn_limit.data_ptr<int>(),
        num_non_exiting_ctas.data_ptr<int>(), args.mDtypeElt, false /* use_routing_scales_on_input */,
        false /* use_deep_seek_fp8 */, stream);

    // MoE kernel except routing
    TORCH_CHECK(hidden_states.scalar_type() == FLOAT4_E2M1X2, "hidden_states must be byte.");
    TORCH_CHECK(hidden_states_scale.scalar_type() == at::ScalarType::Float8_e4m3fn, "hidden_states_scale must be fp8.");

    TORCH_CHECK(hidden_states_scale.dim() == 1, "hidden_states_scale must be 1D.");
    TORCH_CHECK(hidden_states_scale.sizes()[0]
            == tensorrt_llm::computeFP4LinearLayoutSFSize(args.num_tokens, args.hidden_size / 16),
        "hidden_states_scale has incorrect size");

    TORCH_CHECK(gemm1_weights.scalar_type() == FLOAT4_E2M1X2, "gemm1_weights must be byte.");

    TORCH_CHECK(gemm1_weights.dim() == 3, "gemm1_weights must be 3D.");
    TORCH_CHECK(gemm1_weights.sizes()[1] % 2 == 0, "the second dimension of weights must be even.");
    TORCH_CHECK(intermediate_size == gemm1_weights.sizes()[1] / 2, "intermediate_size has incorrect dim 1.");
    // This check passes even though the actual shape of the weights[2] and hidden_states[1] is
    // 2 times larger due to the fact that 2 e2m1 are packed into 1 byte.
    TORCH_CHECK(gemm1_weights.sizes()[2] == hidden_states.sizes()[1],
        "the third dimension of weights must be equal to hidden_size.");

    TORCH_CHECK(gemm1_weights_scale.scalar_type() == at::ScalarType::Float8_e4m3fn, "gemm1_weights_scale must be fp8.");

    TORCH_CHECK(gemm1_weights_scale.dim() == 3, "gemm1_weights_scale must be 3D.");
    TORCH_CHECK(gemm1_weights_scale.sizes()[0] == local_num_experts, "gemm1_weights_scale has incorrect dim 0.");
    TORCH_CHECK(intermediate_size % 16 == 0, "the second dimension of weights must be a multiple of 16.");
    TORCH_CHECK(gemm1_weights_scale.sizes()[1] == 2 * intermediate_size, "gemm1_weights_scale has incorrect dim 1.");
    TORCH_CHECK(gemm1_weights_scale.sizes()[2] == args.hidden_size / 16, "gemm1_weights_scale has incorrect dim 2.");

    TORCH_CHECK(gemm2_weights.scalar_type() == FLOAT4_E2M1X2, "gemm2_weights must be byte.");

    TORCH_CHECK(gemm2_weights.dim() == 3, "gemm2_weights must be 3D.");
    // / 2 to compensate for the fact that we pack 2 e2m1 into 1 byte.
    TORCH_CHECK(gemm2_weights.sizes()[2] == intermediate_size / 2,
        "the third dimension of weights must be equal to intermediate_size.");

    TORCH_CHECK(gemm2_weights_scale.scalar_type() == at::ScalarType::Float8_e4m3fn, "gemm2_weights_scale must be fp8.");

    TORCH_CHECK(gemm2_weights_scale.dim() == 3, "gemm2_weights_scale must be 3D.");
    TORCH_CHECK(gemm2_weights_scale.sizes()[0] == local_num_experts, "gemm2_weights_scale has incorrect dim 0.");
    TORCH_CHECK(gemm2_weights_scale.sizes()[1] == args.hidden_size, "gemm2_weights_scale has incorrect dim 1.");
    TORCH_CHECK(gemm2_weights_scale.sizes()[2] == intermediate_size / 16, "gemm2_weights_scale has incorrect dim 2.");

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
    workspace.total_max_padded_tokens = max_num_padded_tokens;
    workspace.ProjUpTileN = tileN;
    workspace.routing_expert_indexes = expert_indexes.data_ptr<int>();
    workspace.permuted_idx_size = total_num_padded_tokens.data_ptr<int>();
    workspace.expanded_idx_to_permuted_idx
        = expanded_idx_to_permuted_idx.data_ptr<int>(); // Needed by permute/finalize kernels
    workspace.permuted_idx_to_token_idx = permuted_idx_to_token_idx.data_ptr<int>(); // Needed by permuteGemm1 kernel
    workspace.expert_weights = expert_weights.data_ptr();                            // Consumed by finalize kernel

    workspace.cta_idx_xy_to_batch_idx = cta_idx_xy_to_batch_idx.data_ptr<int>();
    workspace.cta_idx_xy_to_mn_limit = cta_idx_xy_to_mn_limit.data_ptr<int>();
    workspace.num_non_exiting_ctas = num_non_exiting_ctas.data_ptr<int>();

    workspace.hidden_states_scale_linear = hidden_states_scale_linear.data_ptr();

    // gemm1 intermediate ws
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = reinterpret_cast<float*>(gemm1_output_scale.data_ptr());

    // gemm2 intermediate ws
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
    args.output = output.data_ptr();
    args.output_scale = nullptr;
    args.output1_scales_scalar = output1_scales_scalar.data_ptr<float>();
    args.output1_scales_gate_scalar = output1_scales_gate_scalar.data_ptr<float>();
    args.output2_scales_scalar = output2_scales_scalar.data_ptr<float>();

    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner moe_runner(args.mDtypeElt, args.mUseDeepSeekFp8);
    auto workspace_sizes = moe_runner.getWorkspaceSizeInBytes(args);
    at::Tensor workspace_fc1 = at::detail::empty_cuda(
        {std::get<0>(workspace_sizes)}, at::ScalarType::Char, hidden_states.device(), std::nullopt);
    at::Tensor workspace_fc2 = at::detail::empty_cuda(
        {std::get<1>(workspace_sizes)}, at::ScalarType::Char, hidden_states.device(), std::nullopt);
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();
    auto const& moe_stream = at::cuda::getCurrentCUDAStream(hidden_states.get_device());
    moe_runner.run(args, workspace, hidden_states.get_device(), moe_stream);
    return output;
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fp4_block_scale_moe_runner("
        "Tensor routing_logits,"
        "Tensor routing_bias,"
        "Tensor hidden_states,"
        "Tensor hidden_states_scale,"
        "Tensor gemm1_weights,"
        "Tensor gemm1_weights_scale,"
        "Tensor gemm2_weights,"
        "Tensor gemm2_weights_scale,"
        "Tensor output1_scale_scalar,"
        "Tensor output1_scale_gate_scalar,"
        "Tensor output2_scale_scalar,"
        "int num_experts,"
        "int top_k,"
        "int n_group,"
        "int topk_group,"
        "int intermediate_size,"
        "int local_expert_offset,"
        "int local_num_experts,"
        "float routed_scaling_factor) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_block_scale_moe_runner", &torch_ext::fp4_block_scale_moe_runner);
}

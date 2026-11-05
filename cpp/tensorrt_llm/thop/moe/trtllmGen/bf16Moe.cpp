/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/moe/trtllmGen/runner.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <ATen/cuda/EmptyTensor.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace btg = batchedGemm::trtllm::gen;
using tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::RoutingMethodType;
using tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::computeSelectedTileN;
namespace routing = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing;

namespace
{
using MoeRunnerType = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner;

// The BF16 cubins read their weights in the shuffled BlockMajorK layout: per expert
// [K / kBlockK, M, kBlockK], with 128-byte (64-element) K blocks.
constexpr int64_t kBlockK = 64;

void checkBlockMajorKWeights(torch::Tensor const& weights, char const* name, int64_t numExperts, int64_t m, int64_t k)
{
    TORCH_CHECK(weights.scalar_type() == at::ScalarType::BFloat16, name, " must be bfloat16, got ",
        c10::toString(weights.scalar_type()), ".");
    TORCH_CHECK(weights.dim() == 4 && weights.sizes()[0] == numExperts && weights.sizes()[1] * kBlockK == k
            && weights.sizes()[2] == m && weights.sizes()[3] == kBlockK,
        name, " must be BlockMajorK [", numExperts, ", ", k / kBlockK, ", ", m, ", ", kBlockK, "], got ",
        weights.sizes(), ".");
    TORCH_CHECK(weights.is_contiguous(), name, " must be contiguous.");
}

std::vector<torch::Tensor> runBf16Moe(torch::optional<torch::Tensor> const& routing_logits,
    torch::optional<torch::Tensor> const& routing_bias, torch::Tensor const& hidden_states,
    torch::Tensor const& gemm1_weights, torch::Tensor const& gemm2_weights, int64_t const num_experts,
    int64_t const top_k, std::optional<int64_t> const n_group, std::optional<int64_t> const topk_group,
    int64_t const intermediate_size, int64_t const local_expert_offset, int64_t const local_num_experts,
    std::optional<double> const routed_scaling_factor, int64_t const tile_tokens_dim, int64_t const routing_method_type,
    bool const is_gated_activation, bool const do_finalize, MoeRunnerType& moe_runner, int64_t const moe_config_index,
    torch::optional<torch::Tensor> const& topk_weights, torch::optional<torch::Tensor> const& topk_ids,
    torch::optional<torch::Tensor> const& out_tensor)
{
    TORCH_CHECK(tensorrt_llm::common::isSM100Family(), "Only SM100f is supported by the BF16 TRTLLM-Gen MoE.");

    if (topk_ids.has_value() && topk_weights.has_value())
    {
        TORCH_CHECK(topk_ids->scalar_type() == at::ScalarType::Int, "topk_ids must be int32.");
        TORCH_CHECK(topk_weights->scalar_type() == at::ScalarType::BFloat16, "topk_weights must be bfloat16.");
        TORCH_CHECK(
            topk_ids->dim() == 2 && topk_ids->sizes()[0] == hidden_states.sizes()[0] && topk_ids->sizes()[1] == top_k,
            "topk_ids must be [num_tokens, top_k].");
        TORCH_CHECK(topk_weights->dim() == 2 && topk_weights->sizes()[0] == hidden_states.sizes()[0]
                && topk_weights->sizes()[1] == top_k,
            "topk_weights must be [num_tokens, top_k].");
    }
    else
    {
        TORCH_CHECK(routing_logits.has_value(), "routing_logits or (topk_ids and topk_weights) must be provided.");
        TORCH_CHECK(routing_logits->scalar_type() == at::ScalarType::BFloat16
                || routing_logits->scalar_type() == at::ScalarType::Float,
            "routing_logits must be bfloat16 or float32.");
        TORCH_CHECK(routing_logits->dim() == 2 && routing_logits->sizes()[0] == hidden_states.sizes()[0]
                && routing_logits->sizes()[1] == num_experts,
            "routing_logits must be [num_tokens, num_experts].");
    }

    if (routing_bias.has_value())
    {
        TORCH_CHECK(routing_bias->scalar_type() == at::ScalarType::BFloat16
                || routing_bias->scalar_type() == at::ScalarType::Float,
            "routing_bias must be bfloat16 or float32.");
        TORCH_CHECK(
            routing_bias->dim() == 1 && routing_bias->sizes()[0] == num_experts, "routing_bias must be [num_experts].");
    }

    if (n_group.has_value() && n_group.value() > 1)
    {
        TORCH_CHECK(static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::DeepSeekV3,
            "Routing kernel with groups implies DeepSeekV3 routing method.");
        TORCH_CHECK(topk_group.has_value(), "if n_group is given, topk_group must be given");
        TORCH_CHECK(num_experts % n_group.value() == 0, "num_experts must be divisible by n_group");
        TORCH_CHECK(top_k <= 8 && top_k > 0, "Current routing kernel (with groups) only supports top_k<=8 && top_k>0.");
        TORCH_CHECK(topk_group.value() <= 4 && topk_group.value() > 0,
            "Current routing kernel only (with groups) supports topk_group<=4 && topk_group > 0.");
        TORCH_CHECK(topk_group.value() <= n_group.value(), "n_group must not be smaller than topk_group.");
        TORCH_CHECK(top_k < (topk_group.value() * num_experts / n_group.value()),
            "top_k must be less than total number of experts in selected groups");
    }
    else if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Renormalize
        || static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::RenormalizeNaive)
    {
        TORCH_CHECK(top_k <= 32 && top_k > 0,
            "Current routing kernel (no groups, renormalize) only supports top_k<=32 && top_k>0.");
    }
    else if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Llama4)
    {
        TORCH_CHECK(top_k == 1, "Current routing kernel (no groups, Llama4) only supports top_k=1.");
    }
    TORCH_CHECK(num_experts > top_k, "num_experts must be greater than top_k");
    TORCH_CHECK(num_experts <= 2048, "num_experts must be less than or equal to 2048");

    TORCH_CHECK(hidden_states.scalar_type() == at::ScalarType::BFloat16, "hidden_states must be bfloat16, got ",
        c10::toString(hidden_states.scalar_type()), ".");
    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D.");
    int64_t const hidden_size = hidden_states.sizes()[1];
    TORCH_CHECK(hidden_size % kBlockK == 0 && intermediate_size % kBlockK == 0,
        "hidden_size and intermediate_size must be multiples of ", kBlockK, " for BlockMajorK weights, got ",
        hidden_size, " and ", intermediate_size, ".");
    checkBlockMajorKWeights(gemm1_weights, "gemm1_weights", local_num_experts,
        is_gated_activation ? 2 * intermediate_size : intermediate_size, hidden_size);
    checkBlockMajorKWeights(gemm2_weights, "gemm2_weights", local_num_experts, hidden_size, intermediate_size);

    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoERunnerArgs args;
    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoEWorkspace workspace;

    auto const routing_bias_dtype = routing_bias.has_value() ? routing_bias->scalar_type() : at::ScalarType::BFloat16;
    args.mDtypeElt = btg::Dtype::Bfloat16;
    args.mDtypeBias = routing_bias_dtype == at::ScalarType::Float ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
    args.routing_logits = routing_logits.has_value() ? routing_logits->data_ptr() : nullptr;
    args.routing_bias = routing_bias.has_value() ? routing_bias->data_ptr() : nullptr;
    args.hidden_states = hidden_states.data_ptr();
    args.topk_weights = topk_weights.has_value() ? topk_weights->data_ptr() : nullptr;
    args.topk_ids = topk_ids.has_value() ? static_cast<int32_t*>(topk_ids->data_ptr()) : nullptr;
    args.gemm1_weights = gemm1_weights.data_ptr();
    args.gemm2_weights = gemm2_weights.data_ptr();
    args.num_tokens = hidden_states.sizes()[0];
    args.num_experts = num_experts;
    args.hidden_size = hidden_size;
    args.top_k = top_k;
    args.n_group = n_group.value_or(0);
    args.topk_group = topk_group.value_or(0);
    args.local_expert_offset = local_expert_offset;
    args.local_num_experts = local_num_experts;
    args.routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args.intermediate_size = intermediate_size;

    auto const routing_device = routing_logits.has_value() ? routing_logits->device() : topk_ids->device();
    at::Tensor num_tokens_per_expert
        = at::detail::empty_cuda({num_experts}, at::ScalarType::Int, routing_device, std::nullopt);
    // Only the local experts' rows are materialized, so size the permuted buffers by them.
    int32_t const max_num_padded_tokens
        = routing::getMaxPermutedPaddedCount(args.num_tokens, top_k, local_num_experts, tile_tokens_dim);
    int32_t const max_num_padded_tokens_gemm1 = routing::maybeGetMinTokenCount(
        max_num_padded_tokens, args.intermediate_size, btg::dtypeGetNumBits(args.mDtypeElt));
    int32_t const max_num_padded_tokens_gemm2
        = routing::maybeGetMinTokenCount(max_num_padded_tokens, args.hidden_size, btg::dtypeGetNumBits(args.mDtypeOut));
    int32_t const max_num_ctas
        = routing::getMaxNumCtasInBatchDim(args.num_tokens, args.top_k, args.local_num_experts, tile_tokens_dim);

    at::Tensor total_num_padded_tokens
        = at::empty({}, at::TensorOptions().device(routing_device).dtype(at::ScalarType::Int));
    at::Tensor expanded_idx_to_permuted_idx
        = at::detail::empty_cuda({args.num_tokens, args.top_k}, at::ScalarType::Int, routing_device, std::nullopt);
    // Sized via getRouteMapAllocCount: the gemm1 cubins read one element past the end of the route
    // map, see the comment on that helper.
    at::Tensor permuted_idx_to_token_idx = at::detail::empty_cuda(
        {routing::getRouteMapAllocCount(max_num_padded_tokens)}, at::ScalarType::Int, routing_device, std::nullopt);
    // The routing kernel's top-k weights feed moe_finalize, which reads them in the output dtype.
    at::Tensor expert_weights
        = at::detail::empty_cuda({args.num_tokens, args.top_k}, at::ScalarType::BFloat16, routing_device, std::nullopt);
    at::Tensor expert_indexes
        = at::detail::empty_cuda({args.num_tokens, args.top_k}, at::ScalarType::Int, routing_device, std::nullopt);
    int64_t const size_of_expert_count_histogram = std::max(num_experts * 2, int64_t(256 * 2));
    at::Tensor expert_count_histogram
        = at::detail::empty_cuda({size_of_expert_count_histogram}, at::ScalarType::Int, routing_device, std::nullopt);
    at::Tensor cta_idx_xy_to_batch_idx
        = at::detail::empty_cuda({max_num_ctas}, at::ScalarType::Int, routing_device, std::nullopt);
    at::Tensor cta_idx_xy_to_mn_limit
        = at::detail::empty_cuda({max_num_ctas}, at::ScalarType::Int, routing_device, std::nullopt);
    at::Tensor num_non_exiting_ctas
        = at::empty({}, at::TensorOptions().device(routing_device).dtype(at::ScalarType::Int));

    // The gated activation is fused into gemm1, so its output is intermediate_size wide either way.
    at::Tensor gemm1_output = at::detail::empty_cuda(
        {max_num_padded_tokens_gemm1, intermediate_size}, at::ScalarType::BFloat16, routing_device, std::nullopt);
    at::Tensor gemm2_output = at::detail::empty_cuda(
        {max_num_padded_tokens_gemm2, args.hidden_size}, at::ScalarType::BFloat16, routing_device, std::nullopt);

    void* expert_weights_ptr = args.topk_weights ? args.topk_weights : expert_weights.data_ptr();

    routing::Runner routing_runner(tile_tokens_dim);
    auto const& routing_stream = at::cuda::getCurrentCUDAStream(routing_device.index());
    auto const dtype_routing_logits
        = routing_logits.has_value() && routing_logits->scalar_type() == at::ScalarType::Float ? btg::Dtype::Fp32
                                                                                               : btg::Dtype::Bfloat16;
    routing_runner.run(args.routing_logits, args.routing_bias, args.num_tokens, args.num_experts, args.top_k,
        /* num_fused_shared_expert */ 0, args.n_group, args.topk_group, args.local_expert_offset,
        args.local_num_experts, args.routed_scaling_factor, expert_indexes.data_ptr<int>(),
        expert_count_histogram.data_ptr<int>(), total_num_padded_tokens.data_ptr<int>(),
        expanded_idx_to_permuted_idx.data_ptr<int>(), nullptr, permuted_idx_to_token_idx.data_ptr<int>(),
        expert_weights_ptr, args.topk_ids, num_tokens_per_expert.data_ptr<int>(),
        cta_idx_xy_to_batch_idx.data_ptr<int>(), cta_idx_xy_to_mn_limit.data_ptr<int>(),
        num_non_exiting_ctas.data_ptr<int>(), args.mDtypeElt, false /* use_routing_scales_on_input */,
        false /* use_deep_seek_fp8 */, static_cast<RoutingMethodType>(routing_method_type), routing_stream,
        dtype_routing_logits, args.mDtypeBias);

    at::Tensor output;
    if (out_tensor.has_value())
    {
        TORCH_CHECK(do_finalize, "out_tensor is only supported when do_finalize=true.");
        TORCH_CHECK(out_tensor->scalar_type() == at::ScalarType::BFloat16, "out_tensor must be bfloat16.");
        TORCH_CHECK(out_tensor->dim() == 2 && out_tensor->sizes()[0] == args.num_tokens
                && out_tensor->sizes()[1] == args.hidden_size,
            "out_tensor must be [num_tokens, hidden_size].");
        TORCH_CHECK(out_tensor->device() == hidden_states.device(), "out_tensor must be on the same device as inputs.");
        output = out_tensor.value();
    }
    else
    {
        output = at::detail::empty_cuda(
            {args.num_tokens, args.hidden_size}, at::ScalarType::BFloat16, hidden_states.device(), std::nullopt);
    }

    workspace.total_num_padded_tokens = total_num_padded_tokens.data_ptr<int>();
    workspace.total_max_padded_tokens = std::max(max_num_padded_tokens_gemm1, max_num_padded_tokens_gemm2);
    workspace.routing_expert_indexes = expert_indexes.data_ptr<int>();
    workspace.permuted_idx_size = total_num_padded_tokens.data_ptr<int>();
    workspace.expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx.data_ptr<int>();
    workspace.permuted_idx_to_token_idx = permuted_idx_to_token_idx.data_ptr<int>();
    workspace.expert_weights = expert_weights_ptr;
    workspace.cta_idx_xy_to_batch_idx = cta_idx_xy_to_batch_idx.data_ptr<int>();
    workspace.cta_idx_xy_to_mn_limit = cta_idx_xy_to_mn_limit.data_ptr<int>();
    workspace.num_non_exiting_ctas = num_non_exiting_ctas.data_ptr<int>();
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = nullptr;
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
    args.output = output.data_ptr();
    args.output_scale = nullptr;
    args.do_finalize = do_finalize;

    auto const workspace_sizes = moe_runner.getWorkspaceSizeInBytes(args, moe_config_index);
    at::Tensor workspace_fc1 = at::detail::empty_cuda(
        {std::get<0>(workspace_sizes)}, at::ScalarType::Char, hidden_states.device(), std::nullopt);
    at::Tensor workspace_fc2 = at::detail::empty_cuda(
        {std::get<1>(workspace_sizes)}, at::ScalarType::Char, hidden_states.device(), std::nullopt);
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();
    auto const& moe_stream = at::cuda::getCurrentCUDAStream(hidden_states.get_device());
    moe_runner.run(args, workspace, hidden_states.get_device(), moe_stream, moe_config_index);

    if (!do_finalize)
    {
        // With caller-supplied top-k the routing kernel reads topk_weights and leaves expert_weights unwritten.
        // Copy rather than return topk_weights: an op output must not alias its inputs.
        if (topk_weights.has_value())
        {
            expert_weights.copy_(topk_weights.value());
        }
        return {gemm2_output, expert_weights, expanded_idx_to_permuted_idx};
    }
    return {output};
}
} // namespace

// Wrapped the TRTLLM-Gen kernel runner in a Torch custom class to allow
// use with the torch workflow autotuner class.
class Bf16MoeRunner : public torch::CustomClassHolder
{
public:
    explicit Bf16MoeRunner(int64_t actType)
        // Tiles the BF16 x BF16 cubins ship: low-latency 8/16/32 and 2-CTA high-throughput 64/128.
        : mSupportedTileN{8, 16, 32, 64, 128}
        , mActType{static_cast<tensorrt_llm::kernels::ActType>(actType)}
    {
        for (int32_t tileN : mSupportedTileN)
        {
            mRunners.emplace(tileN,
                std::make_unique<MoeRunnerType>(
                    btg::Dtype::Bfloat16, btg::Dtype::Bfloat16, /* useDeepSeekFp8 */ false, tileN, mActType));
        }
    }

    // Returns (tileN, config) pairs.
    [[nodiscard]] std::vector<std::vector<int64_t>> getValidConfigs(
        int64_t topK, int64_t hiddenSize, int64_t intermediateSize, int64_t numLocalExperts, int64_t numTokens) const
    {
        auto const chosen = computeSelectedTileN(mSupportedTileN, numTokens, topK, numLocalExperts);
        std::vector<std::vector<int64_t>> tactics;
        for (auto const& [tileN, runner] : mRunners)
        {
            if (chosen.find(tileN) == chosen.end())
            {
                continue;
            }
            for (auto config :
                runner->getValidConfigIndices(topK, hiddenSize, intermediateSize, numLocalExperts, numTokens))
            {
                tactics.push_back({tileN, config});
            }
        }
        return tactics;
    }

    [[nodiscard]] std::vector<torch::Tensor> run(torch::optional<torch::Tensor> const& routing_logits,
        torch::optional<torch::Tensor> const& routing_bias, torch::Tensor const& hidden_states,
        torch::Tensor const& gemm1_weights, torch::Tensor const& gemm2_weights, int64_t const num_experts,
        int64_t const top_k, std::optional<int64_t> const n_group, std::optional<int64_t> const topk_group,
        int64_t const intermediate_size, int64_t const local_expert_offset, int64_t const local_num_experts,
        std::optional<double> const routed_scaling_factor, int64_t const routing_method_type, bool const do_finalize,
        std::vector<int64_t> moeConfigIndex, torch::optional<torch::Tensor> const& topk_weights,
        torch::optional<torch::Tensor> const& topk_ids, torch::optional<torch::Tensor> const& output = torch::nullopt)
    {
        TORCH_CHECK(moeConfigIndex.size() == 2, "moeConfigIndex must be a (tileN, config) pair.");
        int64_t tileN = moeConfigIndex[0];
        int64_t config = moeConfigIndex[1];
        // No autotuner: select the tile and its default config.
        if (tileN == -1 || config == -1)
        {
            auto const num_tokens = hidden_states.sizes()[0];
            float const avg_tokens_per_expert = static_cast<float>(num_tokens * top_k) / local_num_experts;
            tileN = std::clamp(nextPowerOfTwo(avg_tokens_per_expert), mSupportedTileN.front(), mSupportedTileN.back());
            config = mRunners.at(tileN)->getDefaultValidConfigIndex(
                top_k, hidden_states.sizes()[1], intermediate_size, local_num_experts, num_tokens);
        }
        auto const it = mRunners.find(tileN);
        TORCH_CHECK(it != mRunners.end(), "No BF16 TRTLLM-Gen MoE runner for tileN=", tileN, ".");
        return runBf16Moe(routing_logits, routing_bias, hidden_states, gemm1_weights, gemm2_weights, num_experts, top_k,
            n_group, topk_group, intermediate_size, local_expert_offset, local_num_experts, routed_scaling_factor,
            tileN, routing_method_type, tensorrt_llm::kernels::isGatedActType(mActType), do_finalize, *it->second,
            config, topk_weights, topk_ids, output);
    }

private:
    std::vector<int32_t> const mSupportedTileN;
    tensorrt_llm::kernels::ActType const mActType;
    std::map<int32_t, std::unique_ptr<MoeRunnerType>> mRunners;
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<tensorrt_llm::torch_ext::Bf16MoeRunner>("Bf16MoERunner")
        .def(torch::init<int64_t>())
        .def("get_valid_configs", &tensorrt_llm::torch_ext::Bf16MoeRunner::getValidConfigs)
        .def("run_moe", &tensorrt_llm::torch_ext::Bf16MoeRunner::run);
}

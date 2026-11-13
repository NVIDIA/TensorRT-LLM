/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/csrc/routeStageSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! DEP_ROUTER_STOCK_BOUNDARY: the live rows' routing of the stage a boundary began (dep2ctx:
//! minimax_m31_ctx_boundary_normq_out), from the stock router's logits (fp32 [M, E], unit column stride, E <= 256 =
//! bias.numel()): bit for bit what noaux_tc_fp4_quantize_stage_out(logits, bias, x, ..., pad_mask) + stage_appended
//! write into rows 0..M-1 of topk_indices / topk_weights (the top-k ids and weights; -1 / 0 on pad-mask rows; with
//! appended_slot >= 0 the slot / 1.0 in column topk). With fp32_weight (NOAUX_TC_FP32, default) the weights take the
//! FP32 form the family's noaux_tc_fp4_quantize_stage_out uses (the same values as the FP64 expression); false (its
//! kill switch) evaluates the FP64 expression for every weight.
void minimaxM31RouteStage(torch::Tensor const& logits, torch::Tensor const& bias, std::optional<torch::Tensor> padMask,
    int64_t topk, double routedScalingFactor, int64_t appendedSlot, torch::Tensor& topkIndices,
    torch::Tensor& topkWeights, bool pdl, bool fp32Weight)
{
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2 && logits.stride(1) == 1
            && logits.size(0) <= 65535,
        "minimax_m31_route_stage: logits must be fp32 [M, E] with unit column stride");
    int64_t const rows = logits.size(0);
    TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kFloat && bias.dim() == 1 && bias.is_contiguous()
            && bias.numel() == logits.size(1) && bias.numel() >= 1 && bias.numel() <= 256
            && bias.device() == logits.device(),
        "minimax_m31_route_stage: bias must be a contiguous fp32 [E], E <= 256");
    int64_t const appended = appendedSlot >= 0 ? 1 : 0;
    TORCH_CHECK(topk >= 1 && topk <= 32 && topk <= bias.numel(), "minimax_m31_route_stage: 1 <= topk <= 32, E");
    for (torch::Tensor const* t : {&topkIndices, &topkWeights})
    {
        TORCH_CHECK(t->is_cuda() && t->device() == logits.device() && t->dim() == 2 && t->stride(1) == 1
                && t->size(0) >= rows && t->size(1) >= topk + appended
                && (rows <= 1 || t->stride(0) >= topk + appended),
            "minimax_m31_route_stage: stage tensors must be 2-D, unit column stride, >= M rows, topk (+ 1) columns, "
            "rows not overlapping");
    }
    TORCH_CHECK(topkIndices.scalar_type() == at::kInt && topkWeights.scalar_type() == at::kFloat,
        "minimax_m31_route_stage: int32 topk_indices, fp32 topk_weights");
    uint8_t const* padPtr = nullptr;
    if (padMask.has_value())
    {
        TORCH_CHECK(padMask->is_cuda() && padMask->device() == logits.device() && padMask->scalar_type() == at::kByte
                && padMask->dim() == 1 && padMask->size(0) == rows && padMask->is_contiguous(),
            "minimax_m31_route_stage: pad_mask must be a contiguous uint8 [M]");
        padPtr = padMask->data_ptr<uint8_t>();
    }
    TRTLLM_MV2_NS::kernels::minimaxM31::RouteStageParams p{};
    p.logits = logits.data_ptr<float>();
    p.bias = bias.data_ptr<float>();
    p.padMask = padPtr;
    p.topkIndices = topkIndices.data_ptr<int32_t>();
    p.topkWeights = topkWeights.data_ptr<float>();
    p.logitStride = logits.stride(0);
    p.indexStride = topkIndices.stride(0);
    p.weightStride = topkWeights.stride(0);
    p.numRows = static_cast<int>(rows);
    p.numExperts = static_cast<int>(bias.numel());
    p.topk = static_cast<int>(topk);
    p.appendedSlot = static_cast<int>(appendedSlot);
    p.routedScalingFactor = routedScalingFactor;
    p.fp32Weight = fp32Weight;
    at::cuda::CUDAGuard const guard(logits.device());
    cudaError_t const e = TRTLLM_MV2_NS::kernels::minimaxM31::invokeRouteStage(
        p, pdl, at::cuda::getCurrentCUDAStream(logits.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_route_stage launch: ", cudaGetErrorString(e));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_route_stage(Tensor logits, Tensor bias, Tensor? pad_mask, int topk, float routed_scaling_factor, "
        "int appended_slot, Tensor(a!) topk_indices, Tensor(b!) topk_weights, bool pdl=True, bool fp32_weight=True) "
        "-> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_route_stage", &TRTLLM_MV2_NS::torch_ext::minimaxM31RouteStage);
}

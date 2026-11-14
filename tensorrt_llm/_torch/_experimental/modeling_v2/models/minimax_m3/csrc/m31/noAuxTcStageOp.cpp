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

#include "minimax_m3/csrc/m31/noAuxTcStage.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <optional>
#include <torch/extension.h>

namespace th = torch;
namespace tk = TRTLLM_MV2_NS::kernels::minimaxM31;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
tk::NoAuxTcFp4QuantStage makeNoAuxTcFp4QuantStage(th::Tensor& topkIndices, th::Tensor& topkWeights, th::Tensor& fp4Out,
    th::Tensor& sfOut, int64_t numRows, int64_t topk, int64_t hiddenSize, int64_t device)
{
    for (th::Tensor const* out : {&topkIndices, &topkWeights, &fp4Out, &sfOut})
    {
        TORCH_CHECK(out->is_cuda() && out->get_device() == device && out->dim() == 2 && out->stride(1) == 1,
            "noaux_tc_fp4_quantize_stage_out: stage tensors must be 2D CUDA tensors on the device of scores with "
            "unit column stride");
    }
    TORCH_CHECK(topkIndices.scalar_type() == torch::kInt32 && topkIndices.size(1) >= topk,
        "noaux_tc_fp4_quantize_stage_out: topk_indices must be int32 with at least topk columns");
    TORCH_CHECK(topkWeights.scalar_type() == torch::kFloat32 && topkWeights.size(1) >= topk,
        "noaux_tc_fp4_quantize_stage_out: topk_weights must be float32 with at least topk columns");
    TORCH_CHECK(fp4Out.element_size() == 1 && fp4Out.size(1) == hiddenSize / 2 && fp4Out.stride(0) % 4 == 0
            && reinterpret_cast<uintptr_t>(fp4Out.data_ptr()) % 4 == 0,
        "noaux_tc_fp4_quantize_stage_out: fp4_out must hold H / 2 bytes per row, 4-byte aligned");
    TORCH_CHECK(sfOut.element_size() == 1 && sfOut.size(1) == hiddenSize / 16,
        "noaux_tc_fp4_quantize_stage_out: sf_out must hold H / 16 bytes per row");
    TORCH_CHECK(numRows >= 0 && numRows <= topkIndices.size(0) && numRows <= topkWeights.size(0)
            && numRows <= fp4Out.size(0) && numRows <= sfOut.size(0),
        "noaux_tc_fp4_quantize_stage_out: a stage holds fewer rows than it receives");
    return tk::NoAuxTcFp4QuantStage{reinterpret_cast<int32_t*>(topkIndices.mutable_data_ptr()),
        reinterpret_cast<float*>(topkWeights.mutable_data_ptr()),
        reinterpret_cast<uint32_t*>(fp4Out.mutable_data_ptr()), reinterpret_cast<uint8_t*>(sfOut.mutable_data_ptr()),
        topkIndices.stride(0), topkWeights.stride(0), fp4Out.stride(0) / 4, sfOut.stride(0),
        static_cast<int32_t>(numRows), static_cast<int32_t>(topkIndices.size(0)),
        static_cast<int32_t>(topkWeights.size(0))};
}
} // namespace

// noaux_tc_op (single group) and fp4_quantize (NVFP4, 16-element blocks, linear scale factors) of the same M tokens in
// one launch, written into two MegaMoE input stages: tokens [0, rows0) to stage 0 rows [0, rows0), tokens [rows0, M) to
// stage 1 rows [0, M - rows0). Each stage's ids past its tokens become -1 and its weights 0, through the ids' and
// weights' first dimension, and so do the ids and weights of every token t with a nonzero pad_mask[t]. H must be a
// multiple of 64 so the scale-factor rows carry no padding.
void noaux_tc_fp4_quantize_stage_out(th::Tensor const& scores, th::Tensor const& bias, th::Tensor const& hiddenStates,
    th::Tensor const& globalScale, int64_t topk, double routed_scaling_factor, int64_t rows0, th::Tensor& topkIndices0,
    th::Tensor& topkWeights0, th::Tensor& fp4Out0, th::Tensor& sfOut0, th::Tensor& topkIndices1,
    th::Tensor& topkWeights1, th::Tensor& fp4Out1, th::Tensor& sfOut1, std::optional<th::Tensor> const& padMask)
{
    int const smVersion = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(smVersion >= 100 && smVersion < 110, "noaux_tc_fp4_quantize_stage_out requires an SM10x architecture");
    TORCH_CHECK(scores.is_cuda() && bias.is_cuda() && hiddenStates.is_cuda() && globalScale.is_cuda(),
        "noaux_tc_fp4_quantize_stage_out: all inputs must be CUDA tensors");
    auto const device = scores.get_device();
    TORCH_CHECK(
        bias.get_device() == device && hiddenStates.get_device() == device && globalScale.get_device() == device,
        "noaux_tc_fp4_quantize_stage_out: all inputs must be on the same device");
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32
            && globalScale.scalar_type() == torch::kFloat32 && globalScale.numel() >= 1,
        "noaux_tc_fp4_quantize_stage_out: float32 scores, bias and global_scale expected");
    TORCH_CHECK(hiddenStates.scalar_type() == torch::kBFloat16, "noaux_tc_fp4_quantize_stage_out: bf16 hidden_states");
    TORCH_CHECK(scores.is_contiguous() && bias.is_contiguous() && hiddenStates.is_contiguous(),
        "noaux_tc_fp4_quantize_stage_out: contiguous scores, bias and hidden_states expected");
    TORCH_CHECK(scores.dim() == 2 && hiddenStates.dim() == 2 && hiddenStates.size(0) == scores.size(0),
        "noaux_tc_fp4_quantize_stage_out: scores [M, E] and hidden_states [M, H] expected");
    int64_t const numTokens = scores.size(0);
    int64_t const numExperts = scores.size(1);
    int64_t const hiddenSize = hiddenStates.size(1);
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == numExperts, "noaux_tc_fp4_quantize_stage_out: bias must be [E]");
    TORCH_CHECK(hiddenSize > 0 && hiddenSize % 64 == 0, "noaux_tc_fp4_quantize_stage_out: H must be a multiple of 64");
    TORCH_CHECK(rows0 >= 0 && rows0 <= numTokens, "noaux_tc_fp4_quantize_stage_out: rows0 must be in [0, M]");
    uint8_t const* padMaskPtr = nullptr;
    if (padMask.has_value())
    {
        TORCH_CHECK(padMask->is_cuda() && padMask->get_device() == device && padMask->scalar_type() == torch::kUInt8
                && padMask->dim() == 1 && padMask->size(0) == numTokens && padMask->is_contiguous(),
            "noaux_tc_fp4_quantize_stage_out: pad_mask must be a contiguous uint8 [M] tensor on the device of scores");
        padMaskPtr = padMask->data_ptr<uint8_t>();
    }

    auto const stage0
        = makeNoAuxTcFp4QuantStage(topkIndices0, topkWeights0, fp4Out0, sfOut0, rows0, topk, hiddenSize, device);
    auto const stage1 = makeNoAuxTcFp4QuantStage(
        topkIndices1, topkWeights1, fp4Out1, sfOut1, numTokens - rows0, topk, hiddenSize, device);
    auto stream = at::cuda::getCurrentCUDAStream(device);
    tk::invokeNoAuxTcFp4QuantStage(reinterpret_cast<float const*>(scores.data_ptr()),
        reinterpret_cast<float const*>(bias.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(hiddenStates.data_ptr()),
        reinterpret_cast<float const*>(globalScale.data_ptr()), padMaskPtr, stage0, stage1, numTokens, numExperts, topk,
        hiddenSize, routed_scaling_factor, stream);
}

// The ids and weights noaux_tc_fp4_quantize_stage_out writes into a single stage (rows0 = M), in one launch without
// the NVFP4 activations: token t goes to row t, and the ids past M through topk_indices' first dimension become -1 and
// the weights past M through topk_weights' first dimension 0, as do those of every token with a nonzero pad_mask[t].
void noaux_tc_route_stage_out(th::Tensor const& scores, th::Tensor const& bias, int64_t topk,
    double routed_scaling_factor, th::Tensor& topkIndices, th::Tensor& topkWeights,
    std::optional<th::Tensor> const& padMask)
{
    int const smVersion = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(smVersion >= 100 && smVersion < 110, "noaux_tc_route_stage_out requires an SM10x architecture");
    TORCH_CHECK(scores.is_cuda() && bias.is_cuda(), "noaux_tc_route_stage_out: all inputs must be CUDA tensors");
    auto const device = scores.get_device();
    TORCH_CHECK(bias.get_device() == device, "noaux_tc_route_stage_out: all inputs must be on the same device");
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32,
        "noaux_tc_route_stage_out: float32 scores and bias expected");
    TORCH_CHECK(scores.is_contiguous() && bias.is_contiguous(),
        "noaux_tc_route_stage_out: contiguous scores and bias expected");
    TORCH_CHECK(scores.dim() == 2, "noaux_tc_route_stage_out: scores [M, E] expected");
    int64_t const numTokens = scores.size(0);
    int64_t const numExperts = scores.size(1);
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == numExperts, "noaux_tc_route_stage_out: bias must be [E]");
    for (th::Tensor const* out : {&topkIndices, &topkWeights})
    {
        TORCH_CHECK(out->is_cuda() && out->get_device() == device && out->dim() == 2 && out->stride(1) == 1,
            "noaux_tc_route_stage_out: topk_indices and topk_weights must be 2D CUDA tensors on the device of scores "
            "with unit column stride");
        TORCH_CHECK(out->size(1) >= topk && out->size(0) >= numTokens,
            "noaux_tc_route_stage_out: the stage holds fewer rows or columns than it receives");
    }
    TORCH_CHECK(topkIndices.scalar_type() == torch::kInt32 && topkWeights.scalar_type() == torch::kFloat32,
        "noaux_tc_route_stage_out: int32 topk_indices and float32 topk_weights expected");
    uint8_t const* padMaskPtr = nullptr;
    if (padMask.has_value())
    {
        TORCH_CHECK(padMask->is_cuda() && padMask->get_device() == device && padMask->scalar_type() == torch::kUInt8
                && padMask->dim() == 1 && padMask->size(0) == numTokens && padMask->is_contiguous(),
            "noaux_tc_route_stage_out: pad_mask must be a contiguous uint8 [M] tensor on the device of scores");
        padMaskPtr = padMask->data_ptr<uint8_t>();
    }
    tk::NoAuxTcFp4QuantStage const stage{reinterpret_cast<int32_t*>(topkIndices.mutable_data_ptr()),
        reinterpret_cast<float*>(topkWeights.mutable_data_ptr()), nullptr, nullptr, topkIndices.stride(0),
        topkWeights.stride(0), 0, 0, static_cast<int32_t>(numTokens), static_cast<int32_t>(topkIndices.size(0)),
        static_cast<int32_t>(topkWeights.size(0))};
    auto stream = at::cuda::getCurrentCUDAStream(device);
    tk::invokeNoAuxTcRouteStage(reinterpret_cast<float const*>(scores.data_ptr()),
        reinterpret_cast<float const*>(bias.data_ptr()), padMaskPtr, stage, numExperts, topk, routed_scaling_factor,
        stream);
}
} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "noaux_tc_fp4_quantize_stage_out(Tensor scores, Tensor bias, Tensor hidden_states, Tensor global_scale, int "
        "topk, float routed_scaling_factor, int rows0, Tensor(a!) topk_indices0, Tensor(b!) topk_weights0, Tensor(c!) "
        "fp4_out0, Tensor(d!) sf_out0, Tensor(e!) topk_indices1, Tensor(f!) topk_weights1, Tensor(g!) fp4_out1, "
        "Tensor(h!) sf_out1, Tensor? pad_mask=None) -> ()");
    m.def(
        "noaux_tc_route_stage_out(Tensor scores, Tensor bias, int topk, float routed_scaling_factor, Tensor(a!) "
        "topk_indices, Tensor(b!) topk_weights, Tensor? pad_mask=None) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("noaux_tc_fp4_quantize_stage_out", &TRTLLM_MV2_NS::torch_ext::noaux_tc_fp4_quantize_stage_out);
    m.impl("noaux_tc_route_stage_out", &TRTLLM_MV2_NS::torch_ext::noaux_tc_route_stage_out);
}

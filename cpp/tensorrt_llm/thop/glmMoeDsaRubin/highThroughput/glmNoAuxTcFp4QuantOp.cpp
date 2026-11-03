/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/highThroughput/glmNoAuxTcFp4Quant.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;
TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
void glm_rubin_noaux_tc_fp4_quantize_out(th::Tensor const& scores, th::Tensor const& bias,
    th::Tensor const& hiddenStates, th::Tensor const& globalScale, int64_t n_group, int64_t topk_group, int64_t topk,
    double routed_scaling_factor, th::Tensor& topkValues, th::Tensor& topkIndices, th::Tensor& fp4Out,
    th::Tensor& sfOut)
{
    TORCH_CHECK(scores.is_cuda() && bias.is_cuda() && scores.get_device() == bias.get_device(),
        "scores and bias must be on the same CUDA device");
    TORCH_CHECK(scores.dim() == 2 && scores.size(1) == 256 && bias.dim() == 1 && bias.numel() == 256,
        "expected scores [M, 256] and bias [256]");
    TORCH_CHECK(topk >= 1 && topk <= 32, "topk must be in [1, 32]");
    c10::cuda::CUDAGuard guard(scores.device());
    TORCH_CHECK(n_group == 1 && topk_group == 1, "glm_rubin_noaux_tc_fp4_quantize_out: single-group routing only");
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32,
        "glm_rubin_noaux_tc_fp4_quantize_out: float32 scores and bias expected");
    TORCH_CHECK(scores.is_contiguous() && bias.is_contiguous(),
        "glm_rubin_noaux_tc_fp4_quantize_out: contiguous scores and bias");
    int64_t const numTokens = scores.size(0);
    int64_t const numExperts = scores.size(1);
    TORCH_CHECK(numTokens > 0, "glm_rubin_noaux_tc_fp4_quantize_out: at least one token expected");
    TORCH_CHECK(hiddenStates.is_cuda() && hiddenStates.get_device() == scores.get_device()
            && hiddenStates.scalar_type() == torch::kBFloat16 && hiddenStates.is_contiguous() && hiddenStates.dim() == 2
            && hiddenStates.size(0) == numTokens,
        "glm_rubin_noaux_tc_fp4_quantize_out: hidden_states must be a contiguous bf16 [M, H] tensor on the device of "
        "scores");
    int64_t const hiddenSize = hiddenStates.size(1);
    TORCH_CHECK(
        hiddenSize > 0 && hiddenSize % 16 == 0, "glm_rubin_noaux_tc_fp4_quantize_out: H must be a multiple of 16");
    TORCH_CHECK(globalScale.is_cuda() && globalScale.get_device() == scores.get_device()
            && globalScale.scalar_type() == torch::kFloat32 && globalScale.numel() == 1 && globalScale.is_contiguous(),
        "glm_rubin_noaux_tc_fp4_quantize_out: global_scale must be a float32 tensor on the device of scores");
    for (th::Tensor const* out : {&topkValues, &topkIndices, &fp4Out, &sfOut})
    {
        TORCH_CHECK(out->is_cuda() && out->get_device() == scores.get_device() && out->is_contiguous(),
            "glm_rubin_noaux_tc_fp4_quantize_out: outputs must be contiguous tensors on the device of scores");
    }
    TORCH_CHECK(topkValues.scalar_type() == torch::kFloat32 && topkValues.dim() == 2 && topkValues.size(0) == numTokens
            && topkValues.size(1) == topk,
        "glm_rubin_noaux_tc_fp4_quantize_out: topk_values must be a float32 [M, topk] tensor");
    TORCH_CHECK(topkIndices.scalar_type() == torch::kInt32 && topkIndices.dim() == 2 && topkIndices.size(0) == numTokens
            && topkIndices.size(1) == topk,
        "glm_rubin_noaux_tc_fp4_quantize_out: topk_indices must be an int32 [M, topk] tensor");
    TORCH_CHECK(fp4Out.element_size() == 1 && fp4Out.numel() == numTokens * hiddenSize / 2,
        "glm_rubin_noaux_tc_fp4_quantize_out: fp4_out must hold M * H / 2 bytes");
    TORCH_CHECK(sfOut.element_size() == 1 && sfOut.numel() == numTokens * hiddenSize / 16,
        "glm_rubin_noaux_tc_fp4_quantize_out: sf_out must hold M * H / 16 bytes");

    for (th::Tensor const* out : {&topkValues, &topkIndices, &fp4Out, &sfOut})
    {
        for (th::Tensor const* in : {&scores, &bias, &hiddenStates, &globalScale})
        {
            at::assert_no_overlap(*out, *in);
        }
    }
    at::assert_no_overlap(topkValues, topkIndices);
    at::assert_no_overlap(topkValues, fp4Out);
    at::assert_no_overlap(topkValues, sfOut);
    at::assert_no_overlap(topkIndices, fp4Out);
    at::assert_no_overlap(topkIndices, sfOut);
    at::assert_no_overlap(fp4Out, sfOut);
    auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    tk::invokeGlmRubinNoAuxTcFp4Quant(reinterpret_cast<float*>(scores.mutable_data_ptr()),
        reinterpret_cast<float*>(bias.mutable_data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(hiddenStates.data_ptr()),
        reinterpret_cast<float const*>(globalScale.data_ptr()), reinterpret_cast<float*>(topkValues.mutable_data_ptr()),
        reinterpret_cast<int32_t*>(topkIndices.mutable_data_ptr()),
        reinterpret_cast<uint32_t*>(fp4Out.mutable_data_ptr()), reinterpret_cast<uint8_t*>(sfOut.mutable_data_ptr()),
        numTokens, numExperts, topk, hiddenSize, routed_scaling_factor, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm_rubin_noaux_tc_fp4_quantize_out(Tensor scores, Tensor bias, Tensor hidden_states, Tensor global_scale, "
        "int n_group, int topk_group, int topk, float routed_scaling_factor, Tensor(a!) topk_values, "
        "Tensor(b!) topk_indices, Tensor(c!) fp4_out, Tensor(d!) sf_out) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_rubin_noaux_tc_fp4_quantize_out", &tensorrt_llm::torch_ext::glm_rubin_noaux_tc_fp4_quantize_out);
}

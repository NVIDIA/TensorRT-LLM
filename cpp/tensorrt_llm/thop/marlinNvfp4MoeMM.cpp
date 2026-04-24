/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/kernels/marlin/marlin_nvfp4_moe_gemm.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <torch/extension.h>

using torch::Tensor;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// W4A16 Fused MoE Marlin NVFP4 GEMM: BF16 activations + FP4 weights
//
// a:                      [M, K] BF16 activations
// b_q_weight:             [num_experts, K/tile_size, N*tile_size/pack_factor] Marlin-packed FP4
// b_scales:               [num_experts, num_groups, N] FP8 E4M3 block scales
// global_scale:           [num_experts] BF16 per-expert global scales
// workspace:              int32 lock buffer
// sorted_token_ids:       [max_num_tokens_padded] int32
// expert_ids:             [max_num_tokens_padded / block_size] int32
// num_tokens_past_padded: [1] int32 device tensor
// topk_weights:           [M, top_k] float32 router weights
// moe_block_size:         MoE block size (typically 16)
// top_k:                  experts per token
// mul_topk_weights:       whether to multiply topk weights in-kernel
// size_n:                 output dimension N
// size_k:                 reduction dimension K
// out_dtype:              output data type
// use_fp32_reduce:        use FP32 for intermediate reduction
Tensor marlin_nvfp4_moe_gemm(Tensor const& a, Tensor const& b_q_weight, Tensor const& b_scales,
    Tensor const& global_scale, Tensor const& workspace, Tensor const& sorted_token_ids, Tensor const& expert_ids,
    Tensor const& num_tokens_past_padded, Tensor const& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, int64_t size_n, int64_t size_k, std::optional<c10::ScalarType> out_dtype,
    bool use_fp32_reduce = false)
{
    CHECK_INPUT(a, at::kBFloat16);
    CHECK_INPUT(b_q_weight, at::kLong); // 16x nvfp4
    CHECK_INPUT(b_scales, at::kInt);
    CHECK_TH_CUDA(global_scale);
    CHECK_TH_CUDA(workspace);
    CHECK_INPUT(sorted_token_ids, at::kInt);
    CHECK_INPUT(expert_ids, at::kInt);
    CHECK_INPUT(num_tokens_past_padded, at::kInt);
    CHECK_TH_CUDA(topk_weights);

    TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
    TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be 3D [num_experts, ...]");

    int64_t size_m = a.size(0);

    auto const out_dtype_ = out_dtype.value_or(at::ScalarType::BFloat16);
    TORCH_CHECK(out_dtype_ == at::ScalarType::BFloat16, "Output must be bf16");

    // Output: [max_num_tokens_padded, N] (MoE output with sorted token ordering)
    auto out = at::zeros({size_m * top_k, size_n}, a.options().dtype(out_dtype_));

    if (size_m == 0)
        return out;

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    cudaDataType_t outType = convert_torch_dtype(out.scalar_type());

    // Compute num_groups from b_scales shape [num_experts, num_groups, N]
    int num_groups = b_scales.size(1);
    int group_size = (num_groups > 1) ? (static_cast<int>(size_k) / num_groups) : -1;

    // Allocate C_tmp for FP32 reduce
    Tensor c_tmp;
    if (use_fp32_reduce)
    {
        int sms = -1;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a.get_device());
        long max_c_tmp_size = std::min((long) size_n * sorted_token_ids.size(0),
            (long) sms * 4 * moe_block_size * 256); // max_thread_n = 256
        if (moe_block_size == 8)
            max_c_tmp_size *= 2;
        c_tmp = at::empty({max_c_tmp_size}, a.options().dtype(at::kFloat));
    }
    else
    {
        c_tmp = at::empty({0}, a.options().dtype(at::kFloat));
    }

    bool use_atomic_add = false;

    ::marlin_nvfp4::marlinNvfp4MoeGemmDispatcher(a.data_ptr(), b_q_weight.data_ptr(), out.data_ptr(), c_tmp.data_ptr(),
        b_scales.data_ptr(), global_scale.data_ptr(), sorted_token_ids.data_ptr(), expert_ids.data_ptr(),
        num_tokens_past_padded.data_ptr(), topk_weights.data_ptr(), static_cast<int>(moe_block_size),
        static_cast<int>(top_k), mul_topk_weights, static_cast<int>(size_m), static_cast<int>(size_n),
        static_cast<int>(size_k), const_cast<void*>(workspace.data_ptr()), num_groups, group_size, use_fp32_reduce,
        use_atomic_add, outType, stream);

    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "marlin_nvfp4_moe_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, Tensor global_scale,"
        " Tensor workspace, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_past_padded,"
        " Tensor topk_weights, int moe_block_size, int top_k, bool mul_topk_weights,"
        " int size_n, int size_k, ScalarType? out_dtype, bool use_fp32_reduce=False) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("marlin_nvfp4_moe_gemm", &tensorrt_llm::torch_ext::marlin_nvfp4_moe_gemm);
}

/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/ulyssesPostUnscatterKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Post-Ulysses A2A unscatter: take Q/K/V tensors of shape [P, B, Sp, H, D]
// (output of the head-dim -> seq-dim all-to-all) and produce their HND form
// [B, H, P*Sp, D] expected by SDPA. Replaces the eager chain
//     t.permute(1, 0, 2, 3, 4).reshape(B, P * Sp, H, D).contiguous()
//     .transpose(1, 2).contiguous()
// for Q, K, V in one kernel launch.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ulysses_post_unscatter_qkv(
    torch::Tensor& q_in, // [P, B, Sp, H, D]
    torch::Tensor& k_in, // [P, B, Sp, H, D]
    torch::Tensor& v_in) // [P, B, Sp, H, D]
{
    TORCH_CHECK(q_in.dim() == 5 && k_in.dim() == 5 && v_in.dim() == 5,
        "ulysses_post_unscatter_qkv expects 5D tensors [P, B, Sp, H, D]");
    TORCH_CHECK(q_in.sizes() == k_in.sizes() && q_in.sizes() == v_in.sizes(), "Q/K/V must share the same shape");

    CHECK_INPUT(q_in, torch::kBFloat16);
    CHECK_INPUT(k_in, torch::kBFloat16);
    CHECK_INPUT(v_in, torch::kBFloat16);

    int64_t const P = q_in.size(0);
    int64_t const B = q_in.size(1);
    int64_t const Sp = q_in.size(2);
    int64_t const H = q_in.size(3);
    int64_t const D = q_in.size(4);

    auto opts = q_in.options();
    auto q_out = torch::empty({B, H, P * Sp, D}, opts);
    auto k_out = torch::empty({B, H, P * Sp, D}, opts);
    auto v_out = torch::empty({B, H, P * Sp, D}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();
    tensorrt_llm::kernels::launchUlyssesPostUnscatter(q_in.data_ptr(), k_in.data_ptr(), v_in.data_ptr(),
        q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(), static_cast<int>(P), static_cast<int>(B),
        static_cast<int>(Sp), static_cast<int>(H), static_cast<int>(D), stream);

    return std::make_tuple(q_out, k_out, v_out);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("ulysses_post_unscatter_qkv(Tensor q_in, Tensor k_in, Tensor v_in) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("ulysses_post_unscatter_qkv", &ulysses_post_unscatter_qkv);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

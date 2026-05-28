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
// (output of the head-dim -> seq-dim all-to-all) and produce SDPA-ready Q/K/V
// in the layout selected by ``layout``:
//   layout=0 → HND [B, H, P*Sp, D]  (VANILLA / torch SDPA)
//   layout=1 → NHD [B, P*Sp, H, D]  (TRTLLM / FA4)
// Replaces the eager chain
//     t.permute(1, 0, 2, 3, 4).reshape(B, P * Sp, H, D).contiguous()           // NHD
//     [.transpose(1, 2).contiguous()]                                          // HND extra step
// for Q, K, V in one kernel launch.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ulysses_post_unscatter_qkv(
    torch::Tensor& q_in, // [P, B, Sp, H, D]
    torch::Tensor& k_in, // [P, B, Sp, H, D]
    torch::Tensor& v_in, // [P, B, Sp, H, D]
    int64_t layout)      // 0 = HND, 1 = NHD
{
    TORCH_CHECK(q_in.dim() == 5 && k_in.dim() == 5 && v_in.dim() == 5,
        "ulysses_post_unscatter_qkv expects 5D tensors [P, B, Sp, H, D]");
    TORCH_CHECK(q_in.sizes() == k_in.sizes() && q_in.sizes() == v_in.sizes(), "Q/K/V must share the same shape");
    TORCH_CHECK(layout == 0 || layout == 1, "layout must be 0 (HND) or 1 (NHD), got ", layout);

    CHECK_INPUT(q_in, torch::kBFloat16);
    CHECK_INPUT(k_in, torch::kBFloat16);
    CHECK_INPUT(v_in, torch::kBFloat16);

    // D % 8 enforced here at op boundary (mirrors sibling ulysses_permute_scatter).
    // Without this, torch::empty allocates the three output tensors before the
    // kernel launcher's TLLM_CHECK_WITH_INFO fires, producing a less-actionable
    // error path. Vec width is 8 elements for bf16 (16-byte vectorized stores).
    TORCH_CHECK(q_in.size(-1) % 8 == 0, "D (last dim) must be divisible by 8 (bf16 vec=8)");

    int64_t const P = q_in.size(0);
    int64_t const B = q_in.size(1);
    int64_t const Sp = q_in.size(2);
    int64_t const H = q_in.size(3);
    int64_t const D = q_in.size(4);

    bool const is_hnd = (layout == 0);
    auto opts = q_in.options();
    auto const out_shape = is_hnd ? std::vector<int64_t>{B, H, P * Sp, D} : std::vector<int64_t>{B, P * Sp, H, D};
    auto q_out = torch::empty(out_shape, opts);
    auto k_out = torch::empty(out_shape, opts);
    auto v_out = torch::empty(out_shape, opts);

    // Empty-tensor no-op: P=0/B=0/Sp=0 produces zero grid extent in the
    // kernel launcher (undefined cuLaunchKernel behavior across CUDA versions).
    // The three output tensors above are already empty-shaped via P*Sp=0 or B=0,
    // so returning them directly preserves the output-shape contract.
    if (q_in.numel() == 0)
    {
        return std::make_tuple(q_out, k_out, v_out);
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    tensorrt_llm::kernels::launchUlyssesPostUnscatter(q_in.data_ptr(), k_in.data_ptr(), v_in.data_ptr(),
        q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(), static_cast<int>(P), static_cast<int>(B),
        static_cast<int>(Sp), static_cast<int>(H), static_cast<int>(D), is_hnd, stream);

    return std::make_tuple(q_out, k_out, v_out);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    // layout: 0 = HND [B, H, P*Sp, D], 1 = NHD [B, P*Sp, H, D]. Default 0 keeps
    // backward compatibility with the original HND-only callers.
    m.def(
        "ulysses_post_unscatter_qkv(Tensor q_in, Tensor k_in, Tensor v_in, int layout=0) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("ulysses_post_unscatter_qkv", &ulysses_post_unscatter_qkv);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

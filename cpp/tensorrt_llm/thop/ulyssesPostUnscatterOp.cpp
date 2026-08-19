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
#include <limits>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void checkInt32Dim(char const* name, int64_t value)
{
    TORCH_CHECK(value <= std::numeric_limits<int>::max(), name, " must fit in int32, got ", value);
}

void checkInt32Product(char const* name, int64_t lhs, int64_t rhs)
{
    TORCH_CHECK(lhs == 0 || rhs <= std::numeric_limits<int>::max() / lhs, name, " must fit in int32, got ", lhs, "*",
        rhs);
}

void checkInt32SumProduct(char const* name, int64_t lhs, int64_t a, int64_t b, int64_t c)
{
    TORCH_CHECK(a <= std::numeric_limits<int64_t>::max() - b && a + b <= std::numeric_limits<int64_t>::max() - c,
        name, " sum overflows int64");
    checkInt32Product(name, lhs, a + b + c);
}

} // namespace

// Post-Ulysses A2A unscatter: take Q/K/V tensors of shape [P, B, Sp, H, D]
// (output of the head-dim -> seq-dim all-to-all) and produce SDPA-ready Q/K/V.
// The kernel writes NHD-contig storage [B, P*Sp, H, D]; the return depends on ``layout``:
//   layout=0 (HND) → transpose-view [B, H, P*Sp, D] (HND-shape, NHD-stride, non-contig)
//   layout=1 (NHD) → storage as-is [B, P*Sp, H, D] (NHD-contig)
// Equivalent eager: t.permute(1,0,2,3,4).reshape(B, P*Sp, H, D).contiguous()[.transpose(1,2) if HND].
// Q/K/V may differ in (Sp, H) (cross-attn: Q=audio vs K/V=video) — one packed launch handles both.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ulysses_post_unscatter_qkv(
    torch::Tensor& q_in, // [P, B, Sp, H, D]
    torch::Tensor& k_in, // [P, B, Sp, H, D]
    torch::Tensor& v_in, // [P, B, Sp, H, D]
    int64_t layout)      // 0 = HND, 1 = NHD
{
    TORCH_CHECK(q_in.dim() == 5 && k_in.dim() == 5 && v_in.dim() == 5,
        "ulysses_post_unscatter_qkv expects 5D tensors [P, B, Sp, H, D]");
    TORCH_CHECK(layout == 0 || layout == 1, "layout must be 0 (HND) or 1 (NHD), got ", layout);

    CHECK_INPUT(q_in, torch::kBFloat16);
    CHECK_INPUT(k_in, torch::kBFloat16);
    CHECK_INPUT(v_in, torch::kBFloat16);

    // P (ulysses degree), B (batch), D (head dim) are shared across Q/K/V; Sp (seq shard)
    // and H (heads/rank) may differ (cross-attn: Q=audio vs K/V=video). The launcher's
    // packed grid handles equal or different shapes in one launch.
    int64_t const P = q_in.size(0);
    int64_t const B = q_in.size(1);
    int64_t const D = q_in.size(4);
    TORCH_CHECK(k_in.size(0) == P && v_in.size(0) == P, "Q/K/V must share P (ulysses degree)");
    TORCH_CHECK(k_in.size(1) == B && v_in.size(1) == B, "Q/K/V must share B (batch)");
    TORCH_CHECK(k_in.size(4) == D && v_in.size(4) == D, "Q/K/V must share D (head dim)");
    // D % 8 enforced at the op boundary: vec width is 8 bf16 (16-byte vectorized stores).
    TORCH_CHECK(D % 8 == 0, "D (last dim) must be divisible by 8 (bf16 vec=8)");

    int64_t const Sp_q = q_in.size(2), H_q = q_in.size(3);
    int64_t const Sp_k = k_in.size(2), H_k = k_in.size(3);
    int64_t const Sp_v = v_in.size(2), H_v = v_in.size(3);
    checkInt32Dim("P", P);
    checkInt32Dim("B", B);
    checkInt32Dim("D", D);
    checkInt32Dim("Sp_q", Sp_q);
    checkInt32Dim("H_q", H_q);
    checkInt32Dim("Sp_k", Sp_k);
    checkInt32Dim("H_k", H_k);
    checkInt32Dim("Sp_v", Sp_v);
    checkInt32Dim("H_v", H_v);
    checkInt32Product("P*Sp_q", P, Sp_q);
    checkInt32Product("P*Sp_k", P, Sp_k);
    checkInt32Product("P*Sp_v", P, Sp_v);
    checkInt32SumProduct("P*(Sp_q+Sp_k+Sp_v)", P, Sp_q, Sp_k, Sp_v);

    bool const is_hnd = (layout == 0);
    auto opts = q_in.options();
    // Per-tensor NHD-contig storage [B, P*Sp, H, D].
    auto q_out = torch::empty({B, P * Sp_q, H_q, D}, opts);
    auto k_out = torch::empty({B, P * Sp_k, H_k, D}, opts);
    auto v_out = torch::empty({B, P * Sp_v, H_v, D}, opts);

    // Empty-tensor no-op: only when ALL three are empty (P=0 / B=0 / all Sp=0) is the grid
    // extent zero (UB in the launcher). If any tensor is non-empty the launch is valid and
    // the kernel bounds-checks each tensor's own (Sp, H), so a per-tensor empty is fine.
    if (q_in.numel() == 0 && k_in.numel() == 0 && v_in.numel() == 0)
    {
        if (is_hnd)
        {
            return std::make_tuple(q_out.transpose(1, 2), k_out.transpose(1, 2), v_out.transpose(1, 2));
        }
        return std::make_tuple(q_out, k_out, v_out);
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    tensorrt_llm::kernels::launchUlyssesPostUnscatter(q_in.data_ptr(), k_in.data_ptr(), v_in.data_ptr(),
        q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(), static_cast<int>(P), static_cast<int>(B),
        static_cast<int>(D), static_cast<int>(Sp_q), static_cast<int>(H_q), static_cast<int>(Sp_k),
        static_cast<int>(H_k), static_cast<int>(Sp_v), static_cast<int>(H_v), stream);

    // HND callers get a transpose-view of the NHD storage (zero-copy stride
    // reinterpretation). NHD callers get the storage as-is.
    if (is_hnd)
    {
        return std::make_tuple(q_out.transpose(1, 2), k_out.transpose(1, 2), v_out.transpose(1, 2));
    }
    return std::make_tuple(q_out, k_out, v_out);
}

// Self-attention-only variant that consumes the raw packed 5D all-to-all receive buffer
// [P, B, Sp, 3, H, D]. This removes the Python-side packed-QKV permute/contiguous/unbind
// chain before VANILLA/HND SDPA.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ulysses_packed_qkv_post_unscatter(
    torch::Tensor& qkv_in, int64_t layout)
{
    TORCH_CHECK(qkv_in.dim() == 6, "ulysses_packed_qkv_post_unscatter expects [P, B, Sp, 3, H, D]");
    TORCH_CHECK(qkv_in.size(3) == 3, "ulysses_packed_qkv_post_unscatter expects qkv dim size 3, got ",
        qkv_in.size(3));
    TORCH_CHECK(layout == 0 || layout == 1, "layout must be 0 (HND) or 1 (NHD), got ", layout);

    CHECK_INPUT(qkv_in, torch::kBFloat16);

    int64_t const P = qkv_in.size(0);
    int64_t const B = qkv_in.size(1);
    int64_t const Sp = qkv_in.size(2);
    int64_t const H = qkv_in.size(4);
    int64_t const D = qkv_in.size(5);
    TORCH_CHECK(D % 8 == 0, "D (last dim) must be divisible by 8 (bf16 vec=8)");
    checkInt32Dim("P", P);
    checkInt32Dim("B", B);
    checkInt32Dim("Sp", Sp);
    checkInt32Dim("H", H);
    checkInt32Dim("D", D);
    checkInt32Product("P*Sp", P, Sp);
    checkInt32Product("3*P*Sp", 3, P * Sp);

    bool const is_hnd = (layout == 0);
    auto opts = qkv_in.options();
    auto q_out = torch::empty({B, P * Sp, H, D}, opts);
    auto k_out = torch::empty({B, P * Sp, H, D}, opts);
    auto v_out = torch::empty({B, P * Sp, H, D}, opts);

    if (qkv_in.numel() != 0)
    {
        auto stream = at::cuda::getCurrentCUDAStream();
        tensorrt_llm::kernels::launchUlyssesPackedQkvPostUnscatter(qkv_in.data_ptr(), q_out.data_ptr(),
            k_out.data_ptr(), v_out.data_ptr(), static_cast<int>(P), static_cast<int>(B), static_cast<int>(Sp),
            static_cast<int>(H), static_cast<int>(D), stream);
    }

    if (is_hnd)
    {
        return std::make_tuple(q_out.transpose(1, 2), k_out.transpose(1, 2), v_out.transpose(1, 2));
    }
    return std::make_tuple(q_out, k_out, v_out);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    // layout: 0 = HND [B, H, P*Sp, D], 1 = NHD [B, P*Sp, H, D]. Default 0 keeps
    // backward compatibility with the original HND-only callers.
    m.def(
        "ulysses_post_unscatter_qkv(Tensor q_in, Tensor k_in, Tensor v_in, int layout=0) -> (Tensor, Tensor, Tensor)");
    m.def("ulysses_packed_qkv_post_unscatter(Tensor qkv_in, int layout=0) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("ulysses_post_unscatter_qkv", &ulysses_post_unscatter_qkv);
    m.impl("ulysses_packed_qkv_post_unscatter", &ulysses_packed_qkv_post_unscatter);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

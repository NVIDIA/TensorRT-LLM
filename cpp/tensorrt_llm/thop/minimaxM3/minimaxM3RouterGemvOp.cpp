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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/minimaxM3/routerGemv.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <optional>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kGemvMaxM = 16;
constexpr int64_t kTcMaxM = 48;
constexpr int64_t kTcExperts = 128;

void checkSmVersion(char const* op)
{
    int const sm = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sm == 100 || sm == 103, op, " requires SM100 or SM103, got SM", sm);
}

void checkActivation(torch::Tensor const& x, torch::Tensor const& w, int64_t maxM)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1,
        "x must be a 2-D bf16 CUDA tensor with unit column stride");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= maxM, "x must hold 1..", maxM, " rows, got ", x.size(0));
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.is_contiguous() && w.dim() == 2 && w.size(1) == x.size(1),
        "w must be a contiguous fp32 [E, K] tensor");
    TORCH_CHECK(x.size(1) % 1024 == 0 && (x.stride(0) * 2) % 16 == 0, "K must be a multiple of 1024 with 16-byte rows");
}

} // namespace

//! fp32 router logits `x @ w^T` for 1..16 bf16 rows (exact fp32 FMA, one CTA per expert).
torch::Tensor minimaxM3RouterGemv(torch::Tensor const& x, torch::Tensor const& w, int64_t rows, bool pdl)
{
    checkSmVersion("minimax_m3_router_gemv");
    checkActivation(x, w, kGemvMaxM);
    TORCH_CHECK(rows == 1, "minimax_m3_router_gemv: only rows=1 is built, got ", rows);
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    auto out = torch::empty({x.size(0), w.size(0)}, x.options().dtype(at::kFloat));
    tensorrt_llm::kernels::invokeMinimaxM3RouterGemv(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        static_cast<int32_t>(x.stride(0)), static_cast<int32_t>(x.size(0)), static_cast<int32_t>(x.size(1)),
        w.data_ptr<float>(), static_cast<int32_t>(w.size(0)), out.data_ptr<float>(), nullptr, nullptr, nullptr, pdl,
        at::cuda::getCurrentCUDAStream(x.get_device()));
    return out;
}

//! Router logits as `minimax_m3_router_gemv` plus the NVFP4 activation: [logits fp32 [M, E], packed e2m1
//! [M, K / 2] uint8, linear E4M3 scales [M * K / 16] uint8].
std::vector<torch::Tensor> minimaxM3RouterGemvFp4(
    torch::Tensor const& x, torch::Tensor const& w, torch::Tensor const& sfScale, bool pdl)
{
    checkSmVersion("minimax_m3_router_gemv_fp4");
    checkActivation(x, w, kGemvMaxM);
    TORCH_CHECK(sfScale.scalar_type() == at::kFloat && sfScale.numel() == 1 && sfScale.is_cuda(),
        "sf_scale must be a one-element fp32 CUDA tensor");
    int64_t const M = x.size(0), K = x.size(1);
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    auto out = torch::empty({M, w.size(0)}, x.options().dtype(at::kFloat));
    auto q = torch::empty({M, K / 2}, x.options().dtype(at::kByte));
    auto sf = torch::empty({M * (K / 16)}, x.options().dtype(at::kByte));
    tensorrt_llm::kernels::invokeMinimaxM3RouterGemv(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        static_cast<int32_t>(x.stride(0)), static_cast<int32_t>(M), static_cast<int32_t>(K), w.data_ptr<float>(),
        static_cast<int32_t>(w.size(0)), out.data_ptr<float>(), q.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(),
        sfScale.data_ptr<float>(), pdl, at::cuda::getCurrentCUDAStream(x.get_device()));
    return {out, q, sf};
}

//! TF32 tensor-core router logits for 128 experts and 1..48 rows: [logits fp32 [M, 128]] or, with `sf_scale`,
//! [logits, packed e2m1 [M, K / 2], linear E4M3 scales [M * K / 16]].
std::vector<torch::Tensor> minimaxM3RouterTc(
    torch::Tensor const& x, torch::Tensor const& w, std::optional<torch::Tensor> const& sfScale, bool pdl)
{
    checkSmVersion("minimax_m3_router_tc");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1,
        "x must be a 2-D bf16 CUDA tensor with unit column stride");
    int64_t const M = x.size(0), K = x.size(1);
    TORCH_CHECK(M >= 1 && M <= kTcMaxM && w.scalar_type() == at::kFloat && w.dim() == 2 && w.size(0) == kTcExperts
            && w.size(1) == K && w.is_contiguous(),
        "minimax_m3_router_tc takes 1..48 rows and a contiguous fp32 [128, K] weight");
    TORCH_CHECK(K % 1024 == 0 && (x.stride(0) * 2) % 16 == 0, "K must be a multiple of 1024 with 16-byte rows");
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    auto out = torch::empty({M, w.size(0)}, x.options().dtype(at::kFloat));
    torch::Tensor q, sf;
    uint8_t* qp = nullptr;
    uint8_t* sfp = nullptr;
    float const* scp = nullptr;
    if (sfScale.has_value())
    {
        TORCH_CHECK(sfScale->scalar_type() == at::kFloat && sfScale->numel() == 1 && sfScale->is_cuda(),
            "sf_scale must be a one-element fp32 CUDA tensor");
        q = torch::empty({M, K / 2}, x.options().dtype(at::kByte));
        sf = torch::empty({M * (K / 16)}, x.options().dtype(at::kByte));
        qp = q.data_ptr<uint8_t>();
        sfp = sf.data_ptr<uint8_t>();
        scp = sfScale->data_ptr<float>();
    }
    tensorrt_llm::kernels::invokeMinimaxM3RouterTc(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        static_cast<int32_t>(x.stride(0)), static_cast<int32_t>(M), static_cast<int32_t>(K), w.data_ptr<float>(),
        out.data_ptr<float>(), qp, sfp, scp, pdl, at::cuda::getCurrentCUDAStream(x.get_device()));
    if (sfScale.has_value())
    {
        return {out, q, sf};
    }
    return {out};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m3_router_gemv(Tensor x, Tensor w, int rows, bool pdl) -> Tensor");
    m.def("minimax_m3_router_gemv_fp4(Tensor x, Tensor w, Tensor sf_scale, bool pdl) -> Tensor[]");
    m.def("minimax_m3_router_tc(Tensor x, Tensor w, Tensor? sf_scale, bool pdl) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_router_gemv", &tensorrt_llm::torch_ext::minimaxM3RouterGemv);
    m.impl("minimax_m3_router_gemv_fp4", &tensorrt_llm::torch_ext::minimaxM3RouterGemvFp4);
    m.impl("minimax_m3_router_tc", &tensorrt_llm::torch_ext::minimaxM3RouterTc);
}

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

#include "gateResidRmsnorm.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <tuple>

namespace tensorrt_llm::visual_gen::minimax_h3
{

namespace
{

bool aligned16(at::Tensor const& t)
{
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

void checkModulator(at::Tensor const& t, int64_t d, char const* name)
{
    TORCH_CHECK(t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.size(1) == d && t.stride(1) == 1
            && t.stride(0) % 8 == 0 && aligned16(t),
        name, " must be a bf16 [M, ", d, "] row view with unit column stride and 16-byte aligned rows");
}

std::tuple<at::Tensor, at::Tensor> gateResidRmsnormIndexedShiftScale(at::Tensor const& x, at::Tensor const& attn,
    at::Tensor const& weight, at::Tensor const& gate, at::Tensor const& shift, at::Tensor const& scale,
    at::Tensor const& indices, double eps)
{
    int64_t const d = kGateResidRmsnormHidden;
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == d,
        "x must be bf16 CUDA [T, ", d, "]");
    TORCH_CHECK(attn.scalar_type() == at::kBFloat16 && attn.sizes() == x.sizes(), "attn must match x");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16 && weight.numel() == d, "weight must be bf16 [", d, "]");
    checkModulator(gate, d, "gate");
    checkModulator(shift, d, "shift");
    checkModulator(scale, d, "scale");
    TORCH_CHECK(scale.stride(0) == shift.stride(0), "scale and shift must share a row stride");
    TORCH_CHECK(indices.numel() == x.size(0), "indices must hold one row per token");
    auto const xc = x.contiguous();
    auto const ac = attn.contiguous();
    auto const wc = weight.contiguous();
    auto const idx = indices.to(at::kLong).contiguous();
    TORCH_CHECK(aligned16(xc) && aligned16(ac) && aligned16(wc), "x, attn and weight must be 16-byte aligned");
    auto y = at::empty_like(xc);
    auto h = at::empty_like(xc);
    if (xc.size(0) == 0)
    {
        return {y, h};
    }
    c10::cuda::CUDAGuard guard(x.device());
    auto const bf = [](at::Tensor const& t) { return reinterpret_cast<__nv_bfloat16 const*>(t.data_ptr()); };
    C10_CUDA_CHECK(invokeGateResidRmsnormIndexedShiftScale(reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h.data_ptr()), bf(xc), bf(ac), bf(wc), bf(gate), bf(scale), bf(shift),
        idx.data_ptr<int64_t>(), gate.stride(0), scale.stride(0), static_cast<int>(xc.size(0)), static_cast<float>(eps),
        at::cuda::getCurrentCUDAStream()));
    return {y, h};
}

} // namespace

} // namespace tensorrt_llm::visual_gen::minimax_h3

TORCH_LIBRARY_FRAGMENT(vg_minimax_h3, m)
{
    m.def(
        "gate_resid_rmsnorm_indexed_shift_scale(Tensor x, Tensor attn, Tensor weight, Tensor gate, Tensor shift, "
        "Tensor scale, Tensor indices, float eps) -> (Tensor, Tensor)",
        {at::Tag::needs_fixed_stride_order});
}

TORCH_LIBRARY_IMPL(vg_minimax_h3, CUDA, m)
{
    m.impl("gate_resid_rmsnorm_indexed_shift_scale",
        &tensorrt_llm::visual_gen::minimax_h3::gateResidRmsnormIndexedShiftScale);
}

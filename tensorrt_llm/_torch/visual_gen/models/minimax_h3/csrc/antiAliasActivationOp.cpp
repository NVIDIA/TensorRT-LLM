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

#include "antiAliasActivation.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace tensorrt_llm::visual_gen::minimax_h3
{

namespace
{

at::Tensor antiAliasSnakeBeta(at::Tensor const& x, at::Tensor const& upFilter, at::Tensor const& downFilter,
    at::Tensor const& logAlpha, at::Tensor const& logBeta)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.dim() == 3, "x must be fp32 CUDA [B, C, L]");
    int64_t const channels = x.size(1);
    int64_t const length = x.size(2);
    for (auto const* f : {&upFilter, &downFilter})
    {
        TORCH_CHECK(f->scalar_type() == at::kFloat && f->numel() == kAntiAliasTaps, "filters must be fp32 with ",
            kAntiAliasTaps, " taps");
    }
    for (auto const* p : {&logAlpha, &logBeta})
    {
        TORCH_CHECK(p->scalar_type() == at::kFloat && p->numel() == channels, "alpha and beta must be fp32 [C]");
    }
    TORCH_CHECK(x.size(0) * channels <= 65535 && length <= (1 << 29), "x is too large");
    auto const xc = x.contiguous();
    auto out = at::empty_like(xc);
    if (xc.numel() == 0)
    {
        return out;
    }
    c10::cuda::CUDAGuard guard(x.device());
    C10_CUDA_CHECK(
        invokeAntiAliasSnakeBeta(out.data_ptr<float>(), xc.data_ptr<float>(), upFilter.contiguous().data_ptr<float>(),
            downFilter.contiguous().data_ptr<float>(), logAlpha.contiguous().data_ptr<float>(),
            logBeta.contiguous().data_ptr<float>(), static_cast<int>(x.size(0) * channels), static_cast<int>(channels),
            static_cast<int>(length), at::cuda::getCurrentCUDAStream()));
    return out;
}

} // namespace

} // namespace tensorrt_llm::visual_gen::minimax_h3

TORCH_LIBRARY_FRAGMENT(vg_minimax_h3, m)
{
    m.def(
        "anti_alias_snake_beta(Tensor x, Tensor up_filter, Tensor down_filter, Tensor log_alpha, Tensor log_beta) "
        "-> Tensor");
}

TORCH_LIBRARY_IMPL(vg_minimax_h3, CUDA, m)
{
    m.impl("anti_alias_snake_beta", &tensorrt_llm::visual_gen::minimax_h3::antiAliasSnakeBeta);
}

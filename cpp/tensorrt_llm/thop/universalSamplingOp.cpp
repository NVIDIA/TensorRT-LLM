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

// Compiled both by CMake (as part of the wheel) and by torch.utils.cpp_extension during
// development; it must build identically either way, so it includes only its own kernel
// header and ATen.

#include "tensorrt_llm/kernels/universalSamplingKernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <tuple>

namespace torch_ext
{

namespace
{

namespace tk = tensorrt_llm::kernels;

//! Three call shapes share one kernel; which outputs are wanted is decided by which
//! pointers are set, so the kernel is instantiated without the half it does not need.
tk::UniversalSamplingParams buildParams(torch::Tensor const& logits, torch::Tensor const& temperatures,
    torch::Tensor const& topKs, torch::Tensor const& topPs, torch::Tensor const& minPs,
    std::optional<torch::Tensor> const& seed, std::optional<torch::Tensor> const& offset)
{
    TORCH_CHECK(logits.dim() == 2, "logits must be [numRows, vocabSize], got ", logits.dim(), "D");
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");

    auto const numRows = static_cast<int32_t>(logits.size(0));
    auto const vocabSize = static_cast<int32_t>(logits.size(1));

    auto checkRowVector = [numRows](torch::Tensor const& t, char const* name, torch::ScalarType dtype)
    {
        TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
        TORCH_CHECK(t.dim() == 1 && t.size(0) >= numRows, name, " must be a 1D tensor of at least numRows entries");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    };
    checkRowVector(temperatures, "temperatures", torch::kFloat32);
    checkRowVector(topKs, "top_ks", torch::kInt32);
    checkRowVector(topPs, "top_ps", torch::kFloat32);
    checkRowVector(minPs, "min_ps", torch::kFloat32);

    tk::UniversalSamplingParams params;
    params.logits = logits.const_data_ptr();
    params.temperatures = temperatures.const_data_ptr<float>();
    params.topKs = topKs.const_data_ptr<int32_t>();
    params.topPs = topPs.const_data_ptr<float>();
    params.minPs = minPs.const_data_ptr<float>();
    params.numRows = numRows;
    params.vocabSize = vocabSize;

    if (seed.has_value() && offset.has_value())
    {
        auto const& s = seed.value();
        auto const& o = offset.value();
        TORCH_CHECK(s.scalar_type() == torch::kInt64 && o.scalar_type() == torch::kInt64,
            "seed and offset must be int64 tensors (scalars are not CUDA-graph legal)");
        TORCH_CHECK(s.numel() == o.numel(), "seed and offset must have the same length");
        TORCH_CHECK(s.numel() == 1 || s.numel() >= numRows, "seed/offset must hold 1 or numRows entries");
        params.seed = reinterpret_cast<uint64_t const*>(s.const_data_ptr<int64_t>());
        params.offset = reinterpret_cast<uint64_t const*>(o.const_data_ptr<int64_t>());
        params.perRowRng = s.numel() > 1;
    }
    return params;
}

void dispatchByDtype(torch::Tensor const& logits, tk::UniversalSamplingParams const& params)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (logits.scalar_type())
    {
    case torch::kFloat32: tk::invokeUniversalSampling<float>(params, stream); break;
    case torch::kFloat16: tk::invokeUniversalSampling<__half>(params, stream); break;
    case torch::kBFloat16: tk::invokeUniversalSampling<__nv_bfloat16>(params, stream); break;
    default: TORCH_CHECK(false, "universal sampling does not support logits dtype ", logits.scalar_type());
    }
}

} // namespace

torch::Tensor universal_sample_from_logits(torch::Tensor const& logits, torch::Tensor const& temperatures,
    torch::Tensor const& topKs, torch::Tensor const& topPs, torch::Tensor const& minPs,
    std::optional<torch::Tensor> const& seed, std::optional<torch::Tensor> const& offset)
{
    auto params = buildParams(logits, temperatures, topKs, topPs, minPs, seed, offset);
    auto tokens = torch::empty({logits.size(0)}, logits.options().dtype(torch::kInt32));
    params.outputTokens = tokens.data_ptr<int32_t>();
    dispatchByDtype(logits, params);
    return tokens;
}

std::tuple<torch::Tensor, torch::Tensor> universal_sample_from_logits_with_probs(torch::Tensor const& logits,
    torch::Tensor const& temperatures, torch::Tensor const& topKs, torch::Tensor const& topPs,
    torch::Tensor const& minPs, std::optional<torch::Tensor> const& seed, std::optional<torch::Tensor> const& offset)
{
    auto params = buildParams(logits, temperatures, topKs, topPs, minPs, seed, offset);
    auto tokens = torch::empty({logits.size(0)}, logits.options().dtype(torch::kInt32));
    auto probs = torch::empty_like(logits, logits.options().dtype(torch::kFloat32));
    params.outputTokens = tokens.data_ptr<int32_t>();
    params.outputProbs = probs.data_ptr<float>();
    dispatchByDtype(logits, params);
    return {tokens, probs};
}

torch::Tensor universal_compute_probs_from_logits(torch::Tensor const& logits, torch::Tensor const& temperatures,
    torch::Tensor const& topKs, torch::Tensor const& topPs, torch::Tensor const& minPs)
{
    auto params = buildParams(logits, temperatures, topKs, topPs, minPs, std::nullopt, std::nullopt);
    auto probs = torch::empty_like(logits, logits.options().dtype(torch::kFloat32));
    params.outputProbs = probs.data_ptr<float>();
    dispatchByDtype(logits, params);
    return probs;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "universal_sample_from_logits(Tensor logits, Tensor temperatures, Tensor top_ks, "
        "Tensor top_ps, Tensor min_ps, Tensor? seed=None, Tensor? offset=None) -> Tensor");
    m.def(
        "universal_sample_from_logits_with_probs(Tensor logits, Tensor temperatures, Tensor top_ks, "
        "Tensor top_ps, Tensor min_ps, Tensor? seed=None, Tensor? offset=None) -> (Tensor, Tensor)");
    m.def(
        "universal_compute_probs_from_logits(Tensor logits, Tensor temperatures, Tensor top_ks, "
        "Tensor top_ps, Tensor min_ps) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("universal_sample_from_logits", &torch_ext::universal_sample_from_logits);
    m.impl("universal_sample_from_logits_with_probs", &torch_ext::universal_sample_from_logits_with_probs);
    m.impl("universal_compute_probs_from_logits", &torch_ext::universal_compute_probs_from_logits);
}

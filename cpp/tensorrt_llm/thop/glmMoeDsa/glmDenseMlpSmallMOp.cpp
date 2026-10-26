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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/glmMoeDsa/glmDenseMlpSmallM.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace tkd = tensorrt_llm::kernels::glm_moe_dsa::dense_mlp_small_m;

//! The kernels use clusters, bulk copies and st.async; below sm_90 their bodies compile to nothing.
constexpr int kMinSmVersion = 90;

void checkBf16Matrix(at::Tensor const& t, char const* name)
{
    CHECK_INPUT(t, at::ScalarType::BFloat16);
    TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

} // namespace

//! Whether the kernels serve M rows of a dense MLP with this hidden and rank-local intermediate size (host only).
bool glmDenseMlpSmallMSupported(int64_t numRows, int64_t hidden, int64_t inter)
{
    return tkd::isGateUpSupported(static_cast<int>(numRows), static_cast<int>(hidden), static_cast<int>(inter))
        && tkd::isDownSupported(static_cast<int>(numRows), static_cast<int>(inter), static_cast<int>(hidden));
}

//! The rank-local partial of a BF16 gated MLP with plain SwiGLU (no output all-reduce): x [M, H],
//! gate_up_weight [2 I, H] (gate rows first), down_weight [H, I] -> [M, H]. gate_up_tiling / down_tiling select a
//! compiled tiling (-1: the default of the shape).
at::Tensor glmDenseMlpSmallM(at::Tensor const& x, at::Tensor const& gateUpWeight, at::Tensor const& downWeight,
    int64_t gateUpTiling, int64_t downTiling)
{
    checkBf16Matrix(x, "x");
    checkBf16Matrix(gateUpWeight, "gate_up_weight");
    checkBf16Matrix(downWeight, "down_weight");
    TORCH_CHECK(gateUpWeight.device() == x.device() && downWeight.device() == x.device(),
        "glm_dense_mlp_small_m: all tensors must be on the device of x");
    TORCH_CHECK(tensorrt_llm::common::getSMVersion() >= kMinSmVersion, "glm_dense_mlp_small_m needs sm_90 or newer");
    int64_t const numRows = x.size(0);
    int64_t const hidden = x.size(1);
    int64_t const inter = downWeight.size(1);
    TORCH_CHECK(gateUpWeight.size(0) == 2 * inter && gateUpWeight.size(1) == hidden && downWeight.size(0) == hidden,
        "glm_dense_mlp_small_m: gate_up_weight ", gateUpWeight.sizes(), " and down_weight ", downWeight.sizes(),
        " do not match x ", x.sizes());
    TORCH_CHECK(tkd::isGateUpSupported(static_cast<int>(numRows), static_cast<int>(hidden), static_cast<int>(inter),
                    static_cast<int>(gateUpTiling))
            && tkd::isDownSupported(static_cast<int>(numRows), static_cast<int>(inter), static_cast<int>(hidden),
                static_cast<int>(downTiling)),
        "glm_dense_mlp_small_m: unsupported shape (M ", numRows, ", hidden ", hidden, ", inter ", inter, ") or tiling");

    at::Tensor h = at::empty({numRows, inter}, x.options());
    at::Tensor out = at::empty({numRows, hidden}, x.options());
    auto const stream = at::cuda::getCurrentCUDAStream(x.get_device());
    bool const enablePdl = tensorrt_llm::common::getEnvEnablePDL();
    TLLM_CUDA_CHECK(tkd::launchGateUpSwiglu(static_cast<__nv_bfloat16 const*>(x.data_ptr()),
        static_cast<__nv_bfloat16 const*>(gateUpWeight.data_ptr()), static_cast<__nv_bfloat16*>(h.data_ptr()),
        static_cast<int>(numRows), static_cast<int>(hidden), static_cast<int>(inter), static_cast<int>(gateUpTiling),
        enablePdl, stream));
    TLLM_CUDA_CHECK(tkd::launchDown(static_cast<__nv_bfloat16 const*>(h.data_ptr()),
        static_cast<__nv_bfloat16 const*>(downWeight.data_ptr()), static_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int>(numRows), static_cast<int>(inter), static_cast<int>(hidden), static_cast<int>(downTiling),
        enablePdl, stream));
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("glm_dense_mlp_small_m_supported(int num_rows, int hidden, int inter) -> bool");
    m.def(
        "glm_dense_mlp_small_m(Tensor x, Tensor gate_up_weight, Tensor down_weight, int gate_up_tiling=-1, "
        "int down_tiling=-1) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("glm_dense_mlp_small_m_supported", &tensorrt_llm::torch_ext::glmDenseMlpSmallMSupported);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_dense_mlp_small_m", &tensorrt_llm::torch_ext::glmDenseMlpSmallM);
}

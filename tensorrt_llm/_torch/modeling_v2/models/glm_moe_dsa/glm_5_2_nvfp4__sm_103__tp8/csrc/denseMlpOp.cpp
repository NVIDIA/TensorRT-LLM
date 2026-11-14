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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/denseMlp.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <cstdint>

namespace tk = TRTLLM_MV2_NS::dense_mlp;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kVecAlignBytes = 16;

void checkBf16Matrix(at::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be BF16");
    TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % kVecAlignBytes == 0, name, " must be 16-byte aligned");
}

void launchGateUp(at::Tensor const& x, at::Tensor const& gateUpWeight, at::Tensor& h, int64_t variant, bool enablePdl)
{
    auto const stream = at::cuda::getCurrentCUDAStream(x.get_device());
    TLLM_CUDA_CHECK(tk::invokeGateUpSwiglu(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(gateUpWeight.data_ptr()), reinterpret_cast<__nv_bfloat16*>(h.data_ptr()),
        static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), static_cast<int>(h.size(1)),
        static_cast<int>(variant), enablePdl, stream));
}

void launchDownProj(at::Tensor const& h, at::Tensor const& downWeight, at::Tensor& out, int64_t variant, bool enablePdl)
{
    auto const stream = at::cuda::getCurrentCUDAStream(h.get_device());
    TLLM_CUDA_CHECK(tk::invokeDown(reinterpret_cast<__nv_bfloat16 const*>(h.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(downWeight.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int>(h.size(0)), static_cast<int>(h.size(1)), static_cast<int>(out.size(1)),
        static_cast<int>(variant), enablePdl, stream));
}

} // namespace

//! SwiGLU of the fused gate/up projection: x [M, H], gate_up_weight [2I, H] -> h [M, I].
at::Tensor denseMlpGateUpSwiglu(at::Tensor const& x, at::Tensor const& gateUpWeight, int64_t variant)
{
    checkBf16Matrix(x, "x");
    checkBf16Matrix(gateUpWeight, "gate_up_weight");
    int64_t const numRows = x.size(0);
    int64_t const hidden = x.size(1);
    TORCH_CHECK(gateUpWeight.size(1) == hidden, "gate_up_weight must be [2 * inter, hidden]");
    TORCH_CHECK(gateUpWeight.size(0) % 2 == 0, "gate_up_weight rows must be even");
    int64_t const inter = gateUpWeight.size(0) / 2;
    TORCH_CHECK(tk::isGateUpSupported(static_cast<int>(numRows), static_cast<int>(hidden), static_cast<int>(inter),
                    static_cast<int>(variant)),
        "glm_moe_dsa_dense_mlp_gate_up_swiglu: unsupported shape or variant");
    at::Tensor h = at::empty({numRows, inter}, x.options());
    launchGateUp(x, gateUpWeight, h, variant, tensorrt_llm::common::getEnvEnablePDL());
    return h;
}

//! Down projection: h [M, I], down_weight [H, I] -> out [M, H] (rank-local partial, no reduction).
at::Tensor denseMlpDown(at::Tensor const& h, at::Tensor const& downWeight, int64_t variant)
{
    checkBf16Matrix(h, "h");
    checkBf16Matrix(downWeight, "down_weight");
    int64_t const numRows = h.size(0);
    int64_t const inter = h.size(1);
    TORCH_CHECK(downWeight.size(1) == inter, "down_weight must be [hidden, inter]");
    int64_t const hidden = downWeight.size(0);
    TORCH_CHECK(tk::isDownSupported(static_cast<int>(numRows), static_cast<int>(inter), static_cast<int>(hidden),
                    static_cast<int>(variant)),
        "glm_moe_dsa_dense_mlp_down: unsupported shape or variant");
    at::Tensor out = at::empty({numRows, hidden}, h.options());
    launchDownProj(h, downWeight, out, variant, tensorrt_llm::common::getEnvEnablePDL());
    return out;
}

//! Dense gated MLP without the output all-reduce: x [M, H] -> rank-local partial [M, H].
at::Tensor denseMlp(at::Tensor const& x, at::Tensor const& gateUpWeight, at::Tensor const& downWeight,
    int64_t gateUpVariant, int64_t downVariant)
{
    at::Tensor const h = denseMlpGateUpSwiglu(x, gateUpWeight, gateUpVariant);
    return denseMlpDown(h, downWeight, downVariant);
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("dense_mlp_gate_up_swiglu(Tensor x, Tensor gate_up_weight, int variant=1) -> Tensor");
    m.def("dense_mlp_down(Tensor h, Tensor down_weight, int variant=0) -> Tensor");
    m.def(
        "dense_mlp(Tensor x, Tensor gate_up_weight, Tensor down_weight, int gate_up_variant=1, "
        "int down_variant=0) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("dense_mlp_gate_up_swiglu", &TRTLLM_MV2_NS::torch_ext::denseMlpGateUpSwiglu);
    m.impl("dense_mlp_down", &TRTLLM_MV2_NS::torch_ext::denseMlpDown);
    m.impl("dense_mlp", &TRTLLM_MV2_NS::torch_ext::denseMlp);
}

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
// Torch ops of the target's CUDA kernels: tensors in, outputs written in place, launches on the current stream.
#include "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/csrc/decodeMoe.h"
#include "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/csrc/mhcArBoundary.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

void decodeMoe(at::Tensor const& x, at::Tensor const& logits, at::Tensor const& bias, double scaling, double limit,
    at::Tensor const& w31, at::Tensor const& s31, at::Tensor const& w2, at::Tensor const& s2, at::Tensor const& sw31,
    at::Tensor const& ss31, at::Tensor const& sw2, at::Tensor const& ss2, at::Tensor& out, bool pdl)
{
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= kernels::kDecodeMoeMaxRows,
        "decode_moe: x must be a CUDA [rows, hidden] tensor of 1 to ", kernels::kDecodeMoeMaxRows, " rows");
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    kernels::DecodeMoeArgs a{};
    a.x = x.data_ptr();
    a.logits = logits.data_ptr<float>();
    a.bias = bias.data_ptr<float>();
    a.w31 = static_cast<uint8_t const*>(w31.data_ptr());
    a.s31 = s31.data_ptr<float>();
    a.w2 = static_cast<uint8_t const*>(w2.data_ptr());
    a.s2 = s2.data_ptr<float>();
    a.sw31 = static_cast<uint8_t const*>(sw31.data_ptr());
    a.ss31 = ss31.data_ptr<float>();
    a.sw2 = static_cast<uint8_t const*>(sw2.data_ptr());
    a.ss2 = ss2.data_ptr<float>();
    a.out = out.data_ptr();
    a.scaling = static_cast<float>(scaling);
    a.limit = static_cast<float>(limit);
    a.rows = static_cast<int>(x.size(0));
    a.pdl = pdl;
    cudaError_t const err = kernels::decodeMoe(a, at::cuda::getCurrentCUDAStream(x.get_device()));
    TORCH_CHECK(err == cudaSuccess, "decode_moe launch failed: ", cudaGetErrorString(err));
}

void mhcArBoundary(at::Tensor const& partial, at::Tensor const& workspace, int64_t rank, at::Tensor const& residual,
    at::Tensor const& post, at::Tensor const& comb, at::Tensor const& fn, at::Tensor const& scale,
    at::Tensor const& base, at::Tensor const& norm_weight, int64_t sinkhorn_iters, double norm_eps, double hc_eps,
    double sinkhorn_eps, double post_mult, double layer_input_eps, at::Tensor& residual_out, at::Tensor& post_out,
    at::Tensor& comb_out, at::Tensor& layer_input, bool pdl)
{
    TORCH_CHECK(partial.is_cuda() && partial.dim() == 2 && partial.size(0) >= 1
            && partial.size(0) <= kernels::kMhcArBoundaryMaxRows,
        "mhc_ar_boundary: partial must be a CUDA [rows, hidden] tensor of 1 to ", kernels::kMhcArBoundaryMaxRows,
        " rows");
    c10::cuda::CUDAGuard const deviceGuard{partial.device()};
    kernels::MhcArBoundaryArgs a{};
    a.workspace = reinterpret_cast<void**>(workspace.data_ptr());
    a.rank = static_cast<int>(rank);
    a.partial = partial.data_ptr();
    a.residual = residual.data_ptr();
    a.post_prev = post.data_ptr<float>();
    a.comb_prev = comb.data_ptr<float>();
    a.fn = fn.data_ptr<float>();
    a.scale = scale.data_ptr<float>();
    a.base = base.data_ptr<float>();
    a.norm_weight = norm_weight.data_ptr();
    a.residual_out = residual_out.data_ptr();
    a.post_out = post_out.data_ptr<float>();
    a.comb_out = comb_out.data_ptr<float>();
    a.layer_input = layer_input.data_ptr();
    a.rows = static_cast<int>(partial.size(0));
    a.sinkhorn_iters = static_cast<int>(sinkhorn_iters);
    a.norm_eps = static_cast<float>(norm_eps);
    a.hc_eps = static_cast<float>(hc_eps);
    a.sinkhorn_eps = static_cast<float>(sinkhorn_eps);
    a.post_mult = static_cast<float>(post_mult);
    a.layer_input_eps = static_cast<float>(layer_input_eps);
    a.pdl = pdl;
    cudaError_t const err = kernels::mhcArBoundary(a, at::cuda::getCurrentCUDAStream(partial.get_device()));
    TORCH_CHECK(err == cudaSuccess, "mhc_ar_boundary launch failed: ", cudaGetErrorString(err));
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "decode_moe(Tensor x, Tensor logits, Tensor bias, float scaling, float limit, Tensor w31, Tensor s31, "
        "Tensor w2, Tensor s2, Tensor sw31, Tensor ss31, Tensor sw2, Tensor ss2, Tensor(a!) out, bool pdl) -> ()");
    m.def(
        "mhc_ar_boundary(Tensor partial, Tensor workspace, int rank, Tensor residual, Tensor post, Tensor comb, "
        "Tensor fn, Tensor scale, Tensor base, Tensor norm_weight, int sinkhorn_iters, float norm_eps, "
        "float hc_eps, float sinkhorn_eps, float post_mult, float layer_input_eps, Tensor(a!) residual_out, "
        "Tensor(b!) post_out, Tensor(c!) comb_out, Tensor(d!) layer_input, bool pdl) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("decode_moe", &TRTLLM_MV2_NS::decodeMoe);
    m.impl("mhc_ar_boundary", &TRTLLM_MV2_NS::mhcArBoundary);
}

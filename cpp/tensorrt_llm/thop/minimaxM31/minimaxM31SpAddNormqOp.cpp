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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/minimaxM31/spAddNormqSm107.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void checkOut(torch::Tensor const& t, int64_t bytes, char const* what)
{
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "minimax_m31_tp2ctx_peer_add_normq_out: ", what,
        " must be a contiguous CUDA tensor");
    TORCH_CHECK(t.numel() * t.element_size() == bytes, "minimax_m31_tp2ctx_peer_add_normq_out: ", what, " holds ",
        t.numel() * t.element_size(), " bytes, expected ", bytes);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "minimax_m31_tp2ctx_peer_add_normq_out: ", what,
        " must be 16-byte aligned");
}

//! TP2CTX_SP_ADDNORM: local_ptr / peer_ptr address this rank's owned rows of its own and the peer's input slot (the
//! peer add's operands); inter_out receives the new residual rows; the other outputs are minimax_m31_ctx_residual_
//! rmsnorm_mxfp8_out's (norm_out optional, required with f32_out; fp4_out / fp4_sf_out iff fp4_global_scale).
//! ctas_per_sm sets the persistent grid (min(rows, SMs * ctas_per_sm)); every grid gives the same bits.
void peerAddNormqOut(int64_t localPtr, int64_t peerPtr, torch::Tensor& interOut, torch::Tensor const& residual,
    torch::Tensor const& weight, double eps, std::optional<torch::Tensor> normOut, torch::Tensor qOut,
    torch::Tensor sfOut, std::optional<torch::Tensor> fp4Out, std::optional<torch::Tensor> fp4SfOut,
    std::optional<torch::Tensor> const& fp4GlobalScale, std::optional<torch::Tensor> f32Out, int64_t ctasPerSm,
    bool enablePdl)
{
    TORCH_CHECK(interOut.is_cuda() && interOut.is_contiguous() && interOut.scalar_type() == at::kBFloat16
        && interOut.dim() >= 2);
    TORCH_CHECK(
        residual.is_contiguous() && residual.scalar_type() == at::kBFloat16 && residual.sizes() == interOut.sizes());
    int64_t const hidden = interOut.size(-1);
    int64_t const rows = interOut.numel() / hidden;
    TORCH_CHECK(weight.is_contiguous() && weight.scalar_type() == at::kBFloat16 && weight.numel() == hidden);
    TORCH_CHECK(hidden % 256 == 0 && hidden / 8 <= 1024, "hidden must be a multiple of 256 and at most 8192");
    TORCH_CHECK(localPtr % 16 == 0 && peerPtr % 16 == 0, "slot rows must be 16-byte aligned");
    TORCH_CHECK(ctasPerSm >= 1 && ctasPerSm <= 2, "ctas_per_sm must be 1 or 2");
    if (normOut.has_value())
    {
        TORCH_CHECK(normOut->scalar_type() == at::kBFloat16, "norm_out must be bfloat16");
        checkOut(*normOut, rows * hidden * 2, "norm_out");
    }
    checkOut(qOut, rows * hidden, "q_out");
    checkOut(sfOut, (rows + 127) / 128 * 128 * ((hidden / 32 + 3) / 4 * 4), "sf_out");
    bool const fp4 = fp4GlobalScale.has_value();
    TORCH_CHECK(fp4 == fp4Out.has_value() && fp4 == fp4SfOut.has_value(),
        "fp4_out / fp4_sf_out are given iff fp4_global_scale is");
    if (fp4)
    {
        TORCH_CHECK(
            fp4GlobalScale->is_cuda() && fp4GlobalScale->scalar_type() == at::kFloat && fp4GlobalScale->numel() == 1);
        checkOut(*fp4Out, rows * hidden / 2, "fp4_out");
        checkOut(*fp4SfOut, rows * (hidden / 16), "fp4_sf_out");
    }
    if (f32Out.has_value())
    {
        TORCH_CHECK(normOut.has_value(), "f32_out requires norm_out");
        TORCH_CHECK(f32Out->scalar_type() == at::kFloat, "f32_out must be float32");
        checkOut(*f32Out, rows * hidden * 4, "f32_out");
    }
    at::cuda::CUDAGuard const guard(interOut.device());
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, interOut.get_device());
    cudaError_t const err
        = tensorrt_llm::kernels::minimaxM31Sm107::invokeSpAddNormq(reinterpret_cast<__nv_bfloat16 const*>(localPtr),
            reinterpret_cast<__nv_bfloat16 const*>(peerPtr), reinterpret_cast<__nv_bfloat16*>(interOut.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr()),
            normOut.has_value() ? reinterpret_cast<__nv_bfloat16*>(normOut->data_ptr()) : nullptr,
            reinterpret_cast<uint64_t*>(qOut.data_ptr()), reinterpret_cast<uint32_t*>(sfOut.data_ptr()),
            fp4 ? reinterpret_cast<uint32_t*>(fp4Out->data_ptr()) : nullptr,
            fp4 ? reinterpret_cast<uint32_t*>(fp4SfOut->data_ptr()) : nullptr,
            fp4 ? fp4GlobalScale->data_ptr<float>() : nullptr, f32Out.has_value() ? f32Out->data_ptr<float>() : nullptr,
            static_cast<int>(rows), static_cast<int>(hidden), static_cast<float>(eps), sms, static_cast<int>(ctasPerSm),
            enablePdl, at::cuda::getCurrentCUDAStream(interOut.get_device()));
    TORCH_CHECK(err == cudaSuccess, "minimax_m31_tp2ctx_peer_add_normq_out: ", cudaGetErrorString(err));
}

//! MB / UT helper: lets the current device read peer_device's memory directly (already-enabled is not an error).
void enablePeer(int64_t peerDevice)
{
    cudaError_t const err = cudaDeviceEnablePeerAccess(static_cast<int>(peerDevice), 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled)
    {
        cudaGetLastError();
        return;
    }
    TORCH_CHECK(err == cudaSuccess, "cudaDeviceEnablePeerAccess: ", cudaGetErrorString(err));
}

} // namespace

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m31_tp2ctx_peer_add_normq_out(int local_ptr, int peer_ptr, Tensor(a!) inter_out, Tensor residual, "
        "Tensor weight, float eps, Tensor(b!)? norm_out, Tensor(c!) q_out, Tensor(d!) sf_out, Tensor(e!)? "
        "fp4_out=None, "
        "Tensor(f!)? fp4_sf_out=None, Tensor? fp4_global_scale=None, Tensor(g!)? f32_out=None, int ctas_per_sm=1, "
        "bool enable_pdl=False) -> ()");
    m.def("minimax_m31_sp_addnorm_enable_peer(int peer_device) -> ()", &tensorrt_llm::torch_ext::enablePeer);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_tp2ctx_peer_add_normq_out", &tensorrt_llm::torch_ext::peerAddNormqOut);
}

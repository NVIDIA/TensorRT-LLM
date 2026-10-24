/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// trtllm::attn_res_kf_fwd (TRTLLM_K3_ATTN_RES_KF): the Kf attn_res (+ add) + RMSNorm kernel,
// a drop-in for attn_res_add_rmsnorm_persistent_fwd at decode token counts. Not bit-identical to it (a
// different reduction order, see attnResKf.h). A separate translation unit, registering only this op, so
// that it can also be built as a standalone library against an image that predates it.

#include "tensorrt_llm/kernels/kimiK3AttnRes/attnResKf.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kHidden = 7168;
// The kernels read and write rows with 16- and 32-byte vector accesses and TMA bulk copies.
constexpr uintptr_t kAlignment = 32;

void check_aligned(at::Tensor const& t, char const* name)
{
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.const_data_ptr()) % kAlignment == 0, "attn_res_kf_fwd: ", name,
        " must be 32-byte aligned");
}

void check_bf16(at::Tensor const& t, at::Tensor const& like, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.device() == like.device(), "attn_res_kf_fwd: ", name,
        " must be on the device of layer_residual");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, "attn_res_kf_fwd: ", name, " must be bf16");
    TORCH_CHECK(t.is_contiguous(), "attn_res_kf_fwd: ", name, " must be contiguous");
    check_aligned(t, name);
}

//! layer_residual [T, 7168], addend [T, 7168] or none, block_residual [K, T, 7168] (K in [1, 8]),
//! weights [7168], all bf16, contiguous and 32-byte aligned. Returns (updated, output), both [T, 7168]:
//! updated = bf16(layer_residual + addend), or a copy of layer_residual without an addend (the kernel
//! writes it either way).
std::tuple<at::Tensor, at::Tensor> attn_res_kf_fwd(at::Tensor layer_residual, std::optional<at::Tensor> addend,
    at::Tensor block_residual, at::Tensor res_weight, at::Tensor rms_weight, at::Tensor output_rms_weight,
    double rms_eps, double output_rms_eps)
{
    TORCH_CHECK(layer_residual.dim() == 2 && layer_residual.size(1) == kHidden,
        "attn_res_kf_fwd: layer_residual must be [T, 7168]");
    check_bf16(layer_residual, layer_residual, "layer_residual");
    bool const has_add = addend.has_value();
    if (has_add)
    {
        TORCH_CHECK(addend->sizes() == layer_residual.sizes(), "attn_res_kf_fwd: addend must match layer_residual");
        check_bf16(*addend, layer_residual, "addend");
    }
    int64_t const T = layer_residual.size(0);
    TORCH_CHECK(block_residual.dim() == 3 && block_residual.size(1) == T && block_residual.size(2) == kHidden,
        "attn_res_kf_fwd: block_residual must be [K, T, 7168]");
    int64_t const K = block_residual.size(0);
    TORCH_CHECK(K >= 1 && K <= 8, "attn_res_kf_fwd: supported K (block snapshots) are [1, 8], got ", K);
    TORCH_CHECK(T * kHidden * K <= std::numeric_limits<int>::max(), "attn_res_kf_fwd: T=", T, " K=", K,
        " exceeds the int row stride");
    check_bf16(block_residual, layer_residual, "block_residual");
    for (auto const* w : {&res_weight, &rms_weight, &output_rms_weight})
    {
        check_bf16(*w, layer_residual, "res_weight / rms_weight / output_rms_weight");
        TORCH_CHECK(w->numel() == kHidden, "attn_res_kf_fwd: weights must have 7168 elements");
    }

    c10::cuda::CUDAGuard device_guard(layer_residual.device());
    // Thread-block clusters and TMA bulk copies, compiled for the sm_100 family only (kimi_k3_attn_res_kf_src).
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 10,
        "attn_res_kf_fwd requires an sm_100-family (datacenter Blackwell / Rubin) GPU");

    auto updated = at::empty_like(layer_residual);
    auto output = at::empty_like(layer_residual);
    if (T == 0)
    {
        return {updated, output};
    }
    kernels::kimiK3AttnRes::invokeAttnResKfFwd(layer_residual.const_data_ptr(),
        has_add ? addend->const_data_ptr() : layer_residual.const_data_ptr(), block_residual.const_data_ptr(),
        res_weight.const_data_ptr(), rms_weight.const_data_ptr(), output_rms_weight.const_data_ptr(),
        updated.data_ptr(), output.data_ptr(), static_cast<int>(T), static_cast<int>(K), has_add,
        static_cast<float>(rms_eps), static_cast<float>(output_rms_eps), at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {updated, output};
}

} // namespace

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "attn_res_kf_fwd(Tensor layer_residual, Tensor? addend, Tensor block_residual, Tensor res_weight, "
        "Tensor rms_weight, Tensor output_rms_weight, float rms_eps, float output_rms_eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("attn_res_kf_fwd", &tensorrt_llm::torch_ext::attn_res_kf_fwd);
}

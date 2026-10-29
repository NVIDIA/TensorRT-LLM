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

// trtllm::attn_res_persistent_ext_fwd (TRTLLM_K3_ATTN_RES_PERSISTENT_FUSE): the persistent attn_res
// + RMSNorm kernel with the MoE residual update folded in (two addends) and/or the FP8 1x128 R128c4
// copy of its output written in the epilogue. Every output is bit-identical to the launches it
// replaces: attn_res_add_rmsnorm_persistent_fwd on residual + bf16(routed + shared)
// (kimi_k3_moe_residual), and fp8_quantize_1x128_cutedsl_ue8m0 of the normalized output.
//
// A separate translation unit from attnResOp.cpp so that it can also be built as a standalone
// extension library (scripts/micro/fx1/p8) against an image that predates it: it registers only
// this op.

#include "tensorrt_llm/kernels/kimiK3AttnRes/attnResFwdPersistentFused.h"

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
// The kernel reads every input with TMA bulk copies, which fault on a source that is not 16-byte aligned.
constexpr uintptr_t kTmaAlignment = 16;

bool is_sm100_family_device()
{
    // tcgen05/TMEM kernel compiled for the sm_100 family only (see attnResOp.cpp). PyTorch caches the
    // properties per device, so this is not a cudaGetDeviceProperties call per launch.
    return at::cuda::getCurrentDeviceProperties()->major == 10;
}

void check_tma_aligned(at::Tensor const& t, char const* name)
{
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.const_data_ptr()) % kTmaAlignment == 0,
        "attn_res_persistent_ext_fwd: ", name, " must be 16-byte aligned");
}

void check_bf16_row_tensor(at::Tensor const& t, at::Tensor const& like, char const* name)
{
    TORCH_CHECK(t.sizes() == like.sizes(), "attn_res_persistent_ext_fwd: ", name, " must match layer_residual");
    TORCH_CHECK(t.is_cuda() && t.device() == like.device(), "attn_res_persistent_ext_fwd: ", name,
        " must be on the device of layer_residual");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, "attn_res_persistent_ext_fwd: ", name, " must be bf16");
    TORCH_CHECK(t.is_contiguous(), "attn_res_persistent_ext_fwd: ", name, " must be contiguous");
    check_tma_aligned(t, name);
}

//! layer_residual [T, H], addend / addend2 [T, H] (addend2 needs addend), block_residual [K, T, H]
//! (K = N - 1 in [1, 8]), weights [H], H = 7168, all bf16 and contiguous.
//! Returns (updated prefix sum, normalized mix, E4M3 [T, H], R128c4 UE8M0 scales
//! [roundUp(T, 128) * H / 32]); updated is layer_residual itself without an addend, and the two FP8
//! tensors are empty without fp8_out.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> attn_res_persistent_ext_fwd(at::Tensor layer_residual,
    std::optional<at::Tensor> addend, std::optional<at::Tensor> addend2, at::Tensor block_residual,
    at::Tensor res_weight, at::Tensor rms_weight, at::Tensor output_rms_weight, double rms_eps, double output_rms_eps,
    bool fp8_out)
{
    TORCH_CHECK(layer_residual.dim() == 2 && layer_residual.size(1) == kHidden,
        "attn_res_persistent_ext_fwd: layer_residual must be [T, 7168]");
    TORCH_CHECK(
        layer_residual.is_cuda() && layer_residual.scalar_type() == at::kBFloat16 && layer_residual.is_contiguous(),
        "attn_res_persistent_ext_fwd: layer_residual must be a contiguous bf16 CUDA tensor");
    check_tma_aligned(layer_residual, "layer_residual");
    bool const has_add = addend.has_value();
    bool const has_add2 = addend2.has_value();
    TORCH_CHECK(!has_add2 || has_add, "attn_res_persistent_ext_fwd: addend2 needs addend");
    if (has_add)
    {
        check_bf16_row_tensor(*addend, layer_residual, "addend");
    }
    if (has_add2)
    {
        check_bf16_row_tensor(*addend2, layer_residual, "addend2");
    }
    int64_t const T = layer_residual.size(0);
    TORCH_CHECK(T >= 1, "attn_res_persistent_ext_fwd: T must be positive");
    TORCH_CHECK(block_residual.dim() == 3 && block_residual.size(1) == T && block_residual.size(2) == kHidden,
        "attn_res_persistent_ext_fwd: block_residual must be [K, T, 7168]");
    int64_t const N = block_residual.size(0) + 1;
    TORCH_CHECK(N >= 2 && N <= 9, "attn_res_persistent_ext_fwd: supported N values are [2, 9], got ", N);
    TORCH_CHECK(T * kHidden * (N > 2 ? N - 2 : 1) <= std::numeric_limits<int>::max(),
        "attn_res_persistent_ext_fwd: T=", T, " N=", N, " exceeds the int candidate stride");
    for (auto const* w : {&block_residual, &res_weight, &rms_weight, &output_rms_weight})
    {
        TORCH_CHECK(w->is_cuda() && w->device() == layer_residual.device() && w->scalar_type() == at::kBFloat16
                && w->is_contiguous(),
            "attn_res_persistent_ext_fwd: block_residual and the weights must be contiguous bf16 tensors on the "
            "device of layer_residual");
        check_tma_aligned(*w, "block_residual / weight");
    }
    TORCH_CHECK(res_weight.numel() == kHidden && rms_weight.numel() == kHidden && output_rms_weight.numel() == kHidden,
        "attn_res_persistent_ext_fwd: weights must have 7168 elements");

    c10::cuda::CUDAGuard device_guard(layer_residual.device());
    TORCH_CHECK(is_sm100_family_device(),
        "attn_res_persistent_ext_fwd requires an sm_100-family (datacenter Blackwell / Rubin) GPU");

    auto updated = has_add ? at::empty_like(layer_residual) : layer_residual;
    auto output = at::empty_like(layer_residual);
    at::Tensor act;
    at::Tensor scale;
    if (fp8_out)
    {
        int64_t const padded_t = (T + 127) / 128 * 128;
        int64_t const sf_cols = (kHidden / 32 + 3) / 4 * 4;
        act = at::empty({T, kHidden}, layer_residual.options().dtype(at::ScalarType::Float8_e4m3fn));
        scale = at::empty({padded_t * sf_cols}, layer_residual.options().dtype(at::ScalarType::Byte));
    }
    else
    {
        act = at::empty({0}, layer_residual.options().dtype(at::ScalarType::Float8_e4m3fn));
        scale = at::empty({0}, layer_residual.options().dtype(at::ScalarType::Byte));
    }

    kernels::kimiK3AttnRes::AttnResPersistentExtParams params{};
    auto& base = params.base;
    base.blockResidual = reinterpret_cast<__nv_bfloat16 const*>(block_residual.const_data_ptr());
    base.layerResidual = reinterpret_cast<__nv_bfloat16 const*>(layer_residual.const_data_ptr());
    base.layerResidualAdd = has_add ? reinterpret_cast<__nv_bfloat16 const*>(addend->const_data_ptr()) : nullptr;
    base.resWeight = reinterpret_cast<__nv_bfloat16 const*>(res_weight.const_data_ptr());
    base.rmsWeight = reinterpret_cast<__nv_bfloat16 const*>(rms_weight.const_data_ptr());
    base.outputRmsWeight = reinterpret_cast<__nv_bfloat16 const*>(output_rms_weight.const_data_ptr());
    base.updatedLayerResidual = has_add ? reinterpret_cast<__nv_bfloat16*>(updated.data_ptr()) : nullptr;
    base.output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    base.numCandidates = static_cast<int>(N);
    base.seqLen = static_cast<int>(T);
    base.batchSize = 1;
    base.hiddenSize = static_cast<int>(kHidden);
    base.rmsEps = static_cast<float>(rms_eps);
    base.outputRmsEps = static_cast<float>(output_rms_eps);
    params.layerResidualAdd2 = has_add2 ? reinterpret_cast<__nv_bfloat16 const*>(addend2->const_data_ptr()) : nullptr;
    params.fp8Output = fp8_out ? act.data_ptr() : nullptr;
    params.fp8Scale = fp8_out ? scale.data_ptr<uint8_t>() : nullptr;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    kernels::kimiK3AttnRes::invokeAttnResPersistentFusedExtFwd(params, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {updated, output, act, scale};
}

} // namespace

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "attn_res_persistent_ext_fwd(Tensor layer_residual, Tensor? addend, Tensor? addend2, "
        "Tensor block_residual, Tensor res_weight, Tensor rms_weight, Tensor output_rms_weight, "
        "float rms_eps, float output_rms_eps, bool fp8_out) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("attn_res_persistent_ext_fwd", &tensorrt_llm::torch_ext::attn_res_persistent_ext_fwd);
}

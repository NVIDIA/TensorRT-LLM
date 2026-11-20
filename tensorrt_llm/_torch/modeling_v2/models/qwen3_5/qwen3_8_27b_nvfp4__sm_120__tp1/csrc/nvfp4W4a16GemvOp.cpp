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

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/nvfp4W4a16Gemv.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <torch/extension.h>

using torch::Tensor;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

// Cuda-core W4A16 NVFP4 GEMV for decode rows (see nvfp4W4a16Gemv.h).
//
// mat_a:        [M, K] BF16 activations, 1 <= M <= 4
// weight:       [N, K/2] uint8 packed E2M1 (fp4_quantize layout)
// weight_sf:    [N, K/16] uint8 (E4M3 bits) linear block scales
// global_scale: FP32 global scale g; out = bf16((mat_a @ dequant(weight)^T) * g)
// Supported (N, K): N even, K in {5120, 6144}, SM120/121.
Tensor nvfp4_w4a16_gemv(Tensor const& mat_a, Tensor const& weight, Tensor const& weight_sf, double global_scale)
{
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(weight);
    CHECK_TH_CUDA(weight_sf);
    TORCH_CHECK(mat_a.dim() == 2 && mat_a.scalar_type() == at::ScalarType::BFloat16 && mat_a.is_contiguous(),
        "mat_a must be a contiguous 2D BF16 tensor");
    TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == at::ScalarType::Byte && weight.is_contiguous(),
        "weight must be a contiguous 2D uint8 tensor");
    TORCH_CHECK(weight_sf.dim() == 2 && weight_sf.scalar_type() == at::ScalarType::Byte && weight_sf.is_contiguous(),
        "weight_sf must be a contiguous 2D uint8 tensor");
    int const m = static_cast<int>(mat_a.size(0));
    int const k = static_cast<int>(mat_a.size(1));
    int const n = static_cast<int>(weight.size(0));
    TORCH_CHECK(weight.size(1) * 2 == k && weight_sf.size(0) == n && weight_sf.size(1) * 16 == k,
        "nvfp4_w4a16_gemv shape mismatch: mat_a ", mat_a.sizes(), ", weight ", weight.sizes(), ", weight_sf ",
        weight_sf.sizes());
    TORCH_CHECK(m >= 1 && m <= kernels::kNvfp4W4a16GemvMaxRows, "nvfp4_w4a16_gemv supports 1..",
        kernels::kNvfp4W4a16GemvMaxRows, " rows, got ", m);

    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(mat_a));
    Tensor out = at::empty({m, n}, mat_a.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
    bool const launched = kernels::nvfp4W4a16Gemv(mat_a.data_ptr(), weight.data_ptr(), weight_sf.data_ptr(),
        out.data_ptr(), static_cast<float>(global_scale), m, n, k, tensorrt_llm::common::getEnvEnablePDL(), stream);
    TORCH_CHECK(launched, "nvfp4_w4a16_gemv: unsupported shape N=", n, " K=", k, " or device");
    return out;
}

// nvfp4_w4a16_gemv with the GDN decode causal_conv1d_update (width 4, no bias, SiLU) fused into its
// epilogue for output columns [0, conv_dim) (see nvfp4W4a16Gemv.h). Bitwise equal to
// nvfp4_w4a16_gemv followed by causal_conv1d_update(out[:, :conv_dim], conv_state, conv_weight, None,
// silu, conv_state_indices=conv_state_indices, pad_slot_id=pad_slot_id). Updates conv_state in place.
// mat_a:              [M, K] BF16, contiguous, 1 <= M <= 3
// conv_state:         [slots, conv_dim, 3] BF16, strides (any, 3, 1)
// conv_weight:        [conv_dim, 4] BF16, contiguous
// conv_state_indices: [M] int32, contiguous
Tensor nvfp4_w4a16_gemv_conv(Tensor const& mat_a, Tensor const& weight, Tensor const& weight_sf, double global_scale,
    Tensor& conv_state, Tensor const& conv_weight, Tensor const& conv_state_indices, int64_t conv_dim,
    int64_t pad_slot_id)
{
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(weight);
    CHECK_TH_CUDA(weight_sf);
    CHECK_TH_CUDA(conv_state);
    CHECK_TH_CUDA(conv_weight);
    CHECK_TH_CUDA(conv_state_indices);
    TORCH_CHECK(mat_a.dim() == 2 && mat_a.scalar_type() == at::ScalarType::BFloat16 && mat_a.is_contiguous(),
        "mat_a must be a contiguous 2D BF16 tensor");
    TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == at::ScalarType::Byte && weight.is_contiguous(),
        "weight must be a contiguous 2D uint8 tensor");
    TORCH_CHECK(weight_sf.dim() == 2 && weight_sf.scalar_type() == at::ScalarType::Byte && weight_sf.is_contiguous(),
        "weight_sf must be a contiguous 2D uint8 tensor");
    int const m = static_cast<int>(mat_a.size(0));
    int const k = static_cast<int>(mat_a.size(1));
    int const n = static_cast<int>(weight.size(0));
    TORCH_CHECK(weight.size(1) * 2 == k && weight_sf.size(0) == n && weight_sf.size(1) * 16 == k,
        "nvfp4_w4a16_gemv_conv shape mismatch: mat_a ", mat_a.sizes(), ", weight ", weight.sizes(), ", weight_sf ",
        weight_sf.sizes());
    TORCH_CHECK(m >= 1 && m <= kernels::kNvfp4W4a16GemvConvMaxRows, "nvfp4_w4a16_gemv_conv supports 1..",
        kernels::kNvfp4W4a16GemvConvMaxRows, " rows, got ", m);
    TORCH_CHECK(
        conv_dim > 0 && conv_dim % 2 == 0 && conv_dim <= n, "conv_dim must be even and in (0, N], got ", conv_dim);
    TORCH_CHECK(conv_state.dim() == 3 && conv_state.scalar_type() == at::ScalarType::BFloat16
            && conv_state.size(1) == conv_dim && conv_state.size(2) == 3 && conv_state.stride(1) == 3
            && conv_state.stride(2) == 1,
        "conv_state must be a [slots, conv_dim, 3] BF16 tensor with strides (any, 3, 1); got sizes ",
        conv_state.sizes(), " strides ", conv_state.strides());
    TORCH_CHECK(conv_weight.dim() == 2 && conv_weight.scalar_type() == at::ScalarType::BFloat16
            && conv_weight.size(0) == conv_dim && conv_weight.size(1) == 4 && conv_weight.is_contiguous()
            && reinterpret_cast<uintptr_t>(conv_weight.data_ptr()) % 8 == 0,
        "conv_weight must be a contiguous, 8-byte aligned [conv_dim, 4] BF16 tensor");
    TORCH_CHECK(conv_state_indices.dim() == 1 && conv_state_indices.scalar_type() == at::ScalarType::Int
            && conv_state_indices.is_contiguous() && conv_state_indices.size(0) == m,
        "conv_state_indices must be a contiguous [M] int32 tensor");

    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(mat_a));
    Tensor out = at::empty({m, n}, mat_a.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
    kernels::Nvfp4GemvConvParams conv{};
    conv.convState = conv_state.data_ptr();
    conv.convWeight = conv_weight.data_ptr();
    conv.convStateIndices = conv_state_indices.data_ptr<int32_t>();
    conv.convStateBatchStride = conv_state.stride(0);
    conv.convDim = static_cast<int>(conv_dim);
    conv.padSlotId = static_cast<int>(pad_slot_id);
    bool const launched
        = kernels::nvfp4W4a16GemvConv(mat_a.data_ptr(), weight.data_ptr(), weight_sf.data_ptr(), out.data_ptr(),
            static_cast<float>(global_scale), m, n, k, conv, tensorrt_llm::common::getEnvEnablePDL(), stream);
    TORCH_CHECK(launched, "nvfp4_w4a16_gemv_conv: unsupported shape N=", n, " K=", k, " or device");
    return out;
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("nvfp4_w4a16_gemv(Tensor mat_a, Tensor weight, Tensor weight_sf, float global_scale) -> Tensor");
    m.def(
        "nvfp4_w4a16_gemv_conv(Tensor mat_a, Tensor weight, Tensor weight_sf, float global_scale,"
        " Tensor(a!) conv_state, Tensor conv_weight, Tensor conv_state_indices, int conv_dim,"
        " int pad_slot_id) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("nvfp4_w4a16_gemv", &TRTLLM_MV2_NS::nvfp4_w4a16_gemv);
    m.impl("nvfp4_w4a16_gemv_conv", &TRTLLM_MV2_NS::nvfp4_w4a16_gemv_conv);
}

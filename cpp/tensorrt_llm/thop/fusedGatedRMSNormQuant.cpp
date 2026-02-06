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

/*
 * PyTorch binding for Fused Gated RMSNorm + NVFP4 Quantization kernel.
 *
 * Date: 2026-02-05
 *
 * This kernel fuses:
 * 1. SiLU gating: x = x * z * sigmoid(z)
 * 2. Group RMSNorm: y = norm(x) * weight
 * 3. NVFP4 quantization with block scaling
 *
 * Input:
 *   x: [M, N] - input tensor (bf16/fp16)
 *   z: [M, N] - gate tensor (bf16/fp16)
 *   weight: [N] - RMSNorm weight (bf16/fp16)
 *   group_size: int - normalization group size (e.g., 1024)
 *   eps: float - epsilon for numerical stability
 *   sf_scale: optional [1] - global scale factor for FP4
 *
 * Output:
 *   y_fp4: [M, N/8] - FP4 quantized output (uint32, 8 FP4 values packed)
 *   sf_out: scale factors in swizzled layout
 */

#include "tensorrt_llm/kernels/fusedGatedRMSNormQuant/fusedGatedRMSNormQuant.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <optional>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Fused Gated RMSNorm + FP4 Quantization
// Returns: (y_fp4, sf_out)
std::tuple<at::Tensor, at::Tensor> fused_gated_rmsnorm_quant(at::Tensor const& x, at::Tensor const& z,
    at::Tensor const& weight, int64_t group_size, double eps, std::optional<at::Tensor> const& sf_scale)
{
    CHECK_TH_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_TH_CUDA(z);
    // z can be non-contiguous (column slice) but must have contiguous inner dim
    TORCH_CHECK(z.stride(-1) == 1, "z must have contiguous inner dimension (stride[-1] == 1)");
    CHECK_TH_CUDA(weight);
    CHECK_CONTIGUOUS(weight);

    // Check GPU architecture - kernel requires SM100+ (Blackwell)
    auto const device = x.get_device();
    cudaDeviceProp props;
    AT_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    TORCH_CHECK(props.major >= 10,
        "fused_gated_rmsnorm_quant requires SM100 (Blackwell) or newer GPU architecture. "
        "Current device: sm_",
        props.major, props.minor);

    auto const& inputShape = x.sizes();
    auto const rank = inputShape.size();

    TORCH_CHECK(rank == 2, "Input x should be 2D tensor [M, N].");
    TORCH_CHECK(z.sizes() == inputShape, "Gate z shape must match input x shape.");

    int64_t const M = inputShape[0];
    int64_t const N = inputShape[1];

    TORCH_CHECK(weight.sizes()[0] == N, "Weight size must match hidden dimension N.");
    TORCH_CHECK(N % group_size == 0, "Hidden dimension N must be divisible by group_size.");
    TORCH_CHECK(group_size >= 256 && group_size <= 8192, "group_size must be between 256 and 8192.");
    TORCH_CHECK(N % 16 == 0, "Hidden dimension N must be divisible by 16 for FP4 quantization.");

    // Validate sf_scale if provided
    float* sfScalePtr = nullptr;
    if (sf_scale.has_value())
    {
        CHECK_INPUT(sf_scale.value(), torch::kFloat32);
        sfScalePtr = sf_scale.value().data_ptr<float>();
    }

    // Allocate output tensors
    // y_fp4: FP4 packed output [M, N/8] as uint32_t (8 FP4 values packed per uint32)
    at::Tensor y_fp4 = at::detail::empty_cuda({M, N / 8}, torch::kInt32, x.device(), std::nullopt);

    // sf_out: scale factors in swizzled layout
    int64_t const sfVecSize = 16;
    int64_t const sfSize = tensorrt_llm::computeSwizzledLayoutSFSize(M, N / sfVecSize);
    at::Tensor sf_out = at::detail::empty_cuda({sfSize}, SF_DTYPE, x.device(), std::nullopt);

    // Get number of SMs
    static int const multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    auto stream = at::cuda::getCurrentCUDAStream(device);

#define LAUNCH_FUSED_GATED_RMSNORM_QUANT(T)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        tensorrt_llm::kernels::FusedGatedRMSNormQuantParams<T> params;                                                 \
        params.x = reinterpret_cast<T const*>(x.data_ptr());                                                           \
        params.z = reinterpret_cast<T const*>(z.data_ptr());                                                           \
        params.weight = reinterpret_cast<T const*>(weight.data_ptr());                                                 \
        params.y_fp4 = reinterpret_cast<uint32_t*>(y_fp4.data_ptr());                                                  \
        params.sf_out = reinterpret_cast<uint32_t*>(sf_out.data_ptr());                                                \
        params.sf_scale = sfScalePtr;                                                                                  \
        params.M = static_cast<int>(M);                                                                                \
        params.N = static_cast<int>(N);                                                                                \
        params.zRowStride = static_cast<int>(z.stride(0));                                                             \
        params.groupSize = static_cast<int>(group_size);                                                               \
        params.eps = static_cast<float>(eps);                                                                          \
        params.stream = stream;                                                                                        \
        tensorrt_llm::kernels::invokeFusedGatedRMSNormQuant<T>(params, multiProcessorCount);                           \
    } while (0)

    if (x.scalar_type() == at::ScalarType::Half)
    {
        LAUNCH_FUSED_GATED_RMSNORM_QUANT(half);
    }
    else if (x.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        LAUNCH_FUSED_GATED_RMSNORM_QUANT(__nv_bfloat16);
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled for fused_gated_rmsnorm_quant with bf16 input.");
#endif
    }
    else
    {
        C10_THROW_ERROR(
            NotImplementedError, "fused_gated_rmsnorm_quant only supports input tensor with dtypes fp16/bf16.");
    }

#undef LAUNCH_FUSED_GATED_RMSNORM_QUANT

    return std::make_tuple(y_fp4, sf_out);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// Register the op with PyTorch
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_gated_rmsnorm_quant(Tensor x, Tensor z, Tensor weight, int group_size, float eps=1e-5, "
        "Tensor? sf_scale=None) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_gated_rmsnorm_quant", &tensorrt_llm::torch_ext::fused_gated_rmsnorm_quant);
}

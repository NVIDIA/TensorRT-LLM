/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/kernels/prefetch.h"
#include <ATen/cuda/CUDAContext.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>

namespace torch_ext
{

static constexpr int PF_TILE_M = 64;
static constexpr int PF_TILE_K
    = 128; // TODO: determine whether we need this, or can just use a single UTMAPF for all of K

// PyTorch wrapper with automatic M,K detection from tensor shape
void cute_host_prefetch_pytorch(
    torch::Tensor tensor, int64_t delay_start, int64_t throttle_time, int64_t throttle_mode, bool pdl)
{
    // Ensure tensor is 2D and on CUDA
    TORCH_CHECK(tensor.dim() == 2, "Input tensor must be 2D (M, K)");
    TORCH_CHECK(tensor.is_cuda(), "Input tensor must be on CUDA");

    int M = tensor.size(0);
    int K = tensor.size(1);
    int strideM = tensor.stride(0);
    int strideK = tensor.stride(1);

    // Check that M and K are divisible by tile sizes
    TORCH_CHECK(M % PF_TILE_M == 0, "M dimension must be divisible by PF_TILE_M (", PF_TILE_M, ")");
    TORCH_CHECK(K % PF_TILE_K == 0, "K dimension must be divisible by PF_TILE_K (", PF_TILE_K, ")");
    TORCH_CHECK(strideM % PF_TILE_M == 0, "M dimension stride must be divisible by PF_TILE_M (", PF_TILE_M, ")");
    TORCH_CHECK(strideK == 1, "K dimension stride must be 1");

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch based on tensor data type
    if (tensor.dtype() == torch::kFloat32)
    {
        tensorrt_llm::kernels::cute_host_prefetch<float, PF_TILE_M, PF_TILE_K>(tensor.data_ptr<float>(), M, K, strideM,
            (int) delay_start, (int) throttle_time, (int) throttle_mode, pdl, stream);
    }
    else if (tensor.dtype() == torch::kFloat16)
    {
        tensorrt_llm::kernels::cute_host_prefetch<cutlass::half_t, PF_TILE_M, PF_TILE_K>(
            reinterpret_cast<cutlass::half_t*>(tensor.data_ptr<at::Half>()), M, K, strideM, (int) delay_start,
            (int) throttle_time, (int) throttle_mode, pdl, stream);
    }
    else if (tensor.dtype() == torch::kBFloat16)
    {
        tensorrt_llm::kernels::cute_host_prefetch<cutlass::bfloat16_t, PF_TILE_M, PF_TILE_K>(
            reinterpret_cast<cutlass::bfloat16_t*>(tensor.data_ptr<at::BFloat16>()), M, K, strideM, (int) delay_start,
            (int) throttle_time, (int) throttle_mode, pdl, stream);
    }
    else if (tensor.dtype() == torch::kFloat8_e4m3fn)
    {
        tensorrt_llm::kernels::cute_host_prefetch<cutlass::float_e4m3_t, PF_TILE_M, PF_TILE_K>(
            reinterpret_cast<cutlass::float_e4m3_t*>(tensor.data_ptr<at::Float8_e4m3fn>()), M, K, strideM,
            (int) delay_start, (int) throttle_time, (int) throttle_mode, pdl, stream);
    }
    else if (tensor.dtype() == torch::kFloat8_e5m2)
    {
        tensorrt_llm::kernels::cute_host_prefetch<cutlass::float_e5m2_t, PF_TILE_M, PF_TILE_K>(
            reinterpret_cast<cutlass::float_e5m2_t*>(tensor.data_ptr<at::Float8_e5m2>()), M, K, strideM,
            (int) delay_start, (int) throttle_time, (int) throttle_mode, pdl, stream);
    }
    else
    {
        TORCH_CHECK(false,
            "Unsupported tensor data type. Supported types: float32, float16, bfloat16, float8_e4m3fn, float8_e5m2");
    }
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("cute_host_prefetch(Tensor tensor, int delay_start, int throttle_time, int throttle_mode, bool pdl) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cute_host_prefetch", &torch_ext::cute_host_prefetch_pytorch);
}

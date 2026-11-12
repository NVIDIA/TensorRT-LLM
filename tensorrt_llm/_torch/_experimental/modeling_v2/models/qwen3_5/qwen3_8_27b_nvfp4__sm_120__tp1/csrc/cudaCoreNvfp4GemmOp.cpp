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

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/cudaCoreNvfp4Gemm.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

using torch::Tensor;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

// Cuda-core W4A4 NVFP4 GEMM for decode rows (see cudaCoreNvfp4Gemm.h).
//
// mat_a:   [M, K/2] uint8 packed E2M1, 1 <= M <= 8
// mat_b:   [N, K/2] uint8 packed E2M1
// scale_a: the activation's 128x4 swizzled E4M3 scales as written by fp4_quantize
// scale_b: the weight's 128x4 swizzled E4M3 scales
// alpha:   [1] FP32
// Returns [M, N] BF16.
Tensor cuda_core_nvfp4_gemm(
    Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b, Tensor const& alpha)
{
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(mat_b);
    CHECK_TH_CUDA(scale_a);
    CHECK_TH_CUDA(scale_b);
    CHECK_TH_CUDA(alpha);
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && mat_a.size(1) == mat_b.size(1),
        "cuda_core_nvfp4_gemm shape mismatch: mat_a ", mat_a.sizes(), ", mat_b ", mat_b.sizes());
    TORCH_CHECK(mat_a.is_contiguous() && mat_b.is_contiguous() && scale_a.is_contiguous() && scale_b.is_contiguous(),
        "cuda_core_nvfp4_gemm operands must be contiguous");
    TORCH_CHECK(alpha.scalar_type() == at::ScalarType::Float && alpha.numel() == 1, "alpha must be one FP32 value");
    int64_t const m = mat_a.size(0);
    int64_t const n = mat_b.size(0);
    int64_t const k = mat_a.size(1) * 2;
    TORCH_CHECK(m >= 1 && m <= kernels::kCudaCoreNvfp4GemmMaxRows, "cuda_core_nvfp4_gemm supports 1..",
        kernels::kCudaCoreNvfp4GemmMaxRows, " rows, got ", m);
    // The swizzled layout pads rows to 128 and scale columns to 4.
    int64_t const sfColsPadded = (k / 16 + 3) / 4 * 4;
    TORCH_CHECK(scale_a.numel() >= (m + 127) / 128 * 128 * sfColsPadded, "scale_a is smaller than the 128x4 layout");
    TORCH_CHECK(scale_b.numel() >= (n + 127) / 128 * 128 * sfColsPadded, "scale_b is smaller than the 128x4 layout");

    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(mat_a));
    Tensor out = at::empty({m, n}, mat_a.options().dtype(at::ScalarType::BFloat16));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
    bool const launched = kernels::cudaCoreNvfp4Gemm(mat_a.data_ptr(), mat_b.data_ptr(), scale_a.data_ptr(),
        scale_b.data_ptr(), alpha.data_ptr<float>(), out.data_ptr(), static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(k), tensorrt_llm::common::getEnvEnablePDL(), stream);
    TORCH_CHECK(launched, "cuda_core_nvfp4_gemm: unsupported shape N=", n, " K=", k);
    return out;
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("cuda_core_nvfp4_gemm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, Tensor alpha) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("cuda_core_nvfp4_gemm", &TRTLLM_MV2_NS::cuda_core_nvfp4_gemm);
}

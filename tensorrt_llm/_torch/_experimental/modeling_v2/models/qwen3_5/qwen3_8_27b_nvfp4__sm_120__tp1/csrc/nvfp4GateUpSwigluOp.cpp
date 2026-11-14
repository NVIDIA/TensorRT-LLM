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

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/nvfp4GateUpSwigluSm120.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

// gate_up NVFP4 W4A4 GEMM (rows pre-permuted, see nvfp4GateUpSwigluSm120.cu)
// + SwiGLU + NVFP4 quantize with down_proj's global input scale `outScale`.
// Returns (fp4 [m, n/4] uint8, swizzled SF [ceil128(m) * ceil4(n/32)] uint8),
// bitwise equal to nvfp4_gemm (BF16) + fused_silu_mul_nvfp4 on the unpermuted
// weights. SM120/121 only.
std::vector<at::Tensor> nvfp4_gate_up_swiglu_nvfp4(at::Tensor const& mat1, at::Tensor const& mat2,
    at::Tensor const& mat1Scale, at::Tensor const& mat2Scale, at::Tensor const& alpha, at::Tensor const& outScale,
    int64_t tile)
{
    CHECK_INPUT(mat1, torch_ext::FLOAT4_E2M1X2);
    CHECK_INPUT(mat2, torch_ext::FLOAT4_E2M1X2);
    CHECK_INPUT(mat1Scale, torch_ext::SF_DTYPE);
    CHECK_INPUT(mat2Scale, torch_ext::SF_DTYPE);
    CHECK_INPUT(alpha, at::ScalarType::Float);
    CHECK_INPUT(outScale, at::ScalarType::Float);
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "mat1 and mat2 must be matrices");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1], "K mismatch");
    // Select the operands' device first: getSMVersion() reads the current device.
    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(mat1));
    TORCH_CHECK(tensorrt_llm::common::getSMVersion() / 10 == 12, "nvfp4_gate_up_swiglu_nvfp4 requires SM12x");
    TORCH_CHECK(tile >= 0 && tile <= 2, "tile must be 0, 1 or 2, got ", tile);
    int64_t const m = mat1.sizes()[0];
    int64_t const n = mat2.sizes()[0];
    int64_t const k = mat2.sizes()[1] * 2;
    TORCH_CHECK(n % 128 == 0 && k % 128 == 0, "n and k must be multiples of 128");
    int64_t const d = n / 2;
    int64_t const mPad = (m + 127) / 128 * 128;
    int64_t const numKTiles = (d / 16 + 3) / 4;
    at::Tensor fp4 = at::detail::empty_cuda({m, d / 2}, at::ScalarType::Byte, mat1.device(), std::nullopt);
    at::Tensor sf = at::detail::empty_cuda({mPad * numKTiles * 4}, at::ScalarType::Byte, mat1.device(), std::nullopt);
    if (m == 0)
    {
        return {fp4, sf};
    }
    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
    int const tileId = static_cast<int>(tile);
    size_t const wsBytes
        = kernels::nvfp4GateUpSwigluSm120(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), tileId, nullptr, 0, stream, false);
    at::Tensor workspace = at::detail::empty_cuda(
        {static_cast<int64_t>(std::max<size_t>(wsBytes, 16))}, at::ScalarType::Char, mat1.device(), std::nullopt);
    kernels::nvfp4GateUpSwigluSm120(mat1.const_data_ptr(), mat2.const_data_ptr(), mat1Scale.const_data_ptr(),
        mat2Scale.const_data_ptr(), alpha.data_ptr<float>(), outScale.data_ptr<float>(), fp4.data_ptr<uint8_t>(),
        sf.data_ptr<uint8_t>(), static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), tileId,
        reinterpret_cast<char*>(workspace.data_ptr()), workspace.numel(), stream,
        tensorrt_llm::common::getEnvEnablePDL());
    return {fp4, sf};
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "nvfp4_gate_up_swiglu_nvfp4(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor alpha, "
        "Tensor outScale, int tile=0) -> Tensor[]");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("nvfp4_gate_up_swiglu_nvfp4", &TRTLLM_MV2_NS::nvfp4_gate_up_swiglu_nvfp4);
}

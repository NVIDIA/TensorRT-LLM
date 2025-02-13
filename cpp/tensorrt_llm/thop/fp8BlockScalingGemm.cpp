/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/internal_cutlass_kernels/include/fp8_blockscale_gemm.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

#include <optional>

using namespace tensorrt_llm::kernels::small_m_gemm;

namespace torch_ext
{

using Fp8BlockScaleGemmRunnerPtr
    = std::unique_ptr<tensorrt_llm::kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunnerInterface>;

namespace
{
void check_input_dtypes(torch::Tensor mat, std::optional<torch::Tensor> matScale)
{
    if (matScale)
    {
        CHECK_INPUT((*matScale), FP8_BLOCK_SCALING_SF_DTYPE);
    }
    if (!matScale.has_value())
    {
        // Mirror the logic from the FP8 current scaling gemm plugin - only support BF16.
        TORCH_CHECK(mat.scalar_type() == at::ScalarType::BFloat16, "Only FP8/BF16 is supported");
    }
    else
    {
        TORCH_CHECK(mat.scalar_type() == at::ScalarType::Float8_e4m3fn,
            "Matrix dtype must be FP8 if scales are provided (the matrix will be dequantized on the fly).");
    }
}

#define DISPATCH_SCALAR_TYPE(scalar_type, ...)                                                                         \
    if (scalar_type == at::ScalarType::BFloat16)                                                                       \
    {                                                                                                                  \
        using DataType = __nv_bfloat16;                                                                                \
        __VA_ARGS__();                                                                                                 \
    }                                                                                                                  \
    else if (scalar_type == at::ScalarType::Float8_e4m3fn)                                                             \
    {                                                                                                                  \
        using DataType = __nv_fp8_e4m3;                                                                                \
        __VA_ARGS__();                                                                                                 \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TORCH_CHECK(false);                                                                                            \
    }

Fp8BlockScaleGemmRunnerPtr get_gemm_runner(at::ScalarType dtype_a, at::ScalarType dtype_b)
{
    Fp8BlockScaleGemmRunnerPtr result;

    DISPATCH_SCALAR_TYPE(dtype_a,
        [&]
        {
            using ADtypeStatic = DataType;
            DISPATCH_SCALAR_TYPE(dtype_b,
                [&]
                {
                    using BDtypeStatic = DataType;
                    result
                        = std::make_unique<CutlassFp8BlockScaleGemmRunner<ADtypeStatic, BDtypeStatic, __nv_bfloat16>>();
                })
        })

    return result;
}

} // namespace

torch::Tensor fp8_block_scaling_gemm(torch::Tensor mat1, torch::Tensor mat2, std::optional<torch::Tensor> mat1Scale,
    std::optional<torch::Tensor> mat2Scale)
{
    check_input_dtypes(mat1, mat1Scale);
    check_input_dtypes(mat2, mat2Scale);

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1], "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0], "x",
        mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

    auto const m = mat1.sizes()[0];
    auto const n = mat2.sizes()[0];
    auto const k = mat1.sizes()[1];
    TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    at::Tensor out = at::detail::empty_cuda({m, n}, at::ScalarType::BFloat16, mat1.device(), std::nullopt);

    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());
    auto ws_size = static_cast<int64_t>(gemm_runner->getWorkspaceSize(m, n, k));
    at::Tensor workspace = at::detail::empty_cuda({ws_size}, at::ScalarType::Char, torch::kCUDA, std::nullopt);

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float const* mat1ScalePtr = mat1Scale.has_value() ? mat1Scale->data_ptr<float>() : nullptr;
    float const* mat2ScalePtr = mat2Scale.has_value() ? mat2Scale->data_ptr<float>() : nullptr;

    gemm_runner->gemm(out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(), m, n, k,
        static_cast<char*>(workspace.data_ptr()), stream, mat1ScalePtr, mat2ScalePtr);

    return out;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fp8_block_scaling_gemm(Tensor mat1, Tensor mat2, Tensor? mat1Scale, Tensor? mat2Scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp8_block_scaling_gemm", &torch_ext::fp8_block_scaling_gemm);
}

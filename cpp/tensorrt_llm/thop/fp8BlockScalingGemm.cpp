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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/internal_cutlass_kernels/include/fp8_blockscale_gemm.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/blockscaleGemm/kernelRunner.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

#include <optional>

using namespace tensorrt_llm::kernels::small_m_gemm;
using namespace tensorrt_llm::kernels;

namespace torch_ext
{

using Fp8BlockScaleGemmRunnerPtr
    = std::unique_ptr<tensorrt_llm::kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunnerInterface>;

namespace
{
void check_input_dtypes(torch::Tensor mat, std::optional<torch::Tensor> matScale)
{
    TORCH_CHECK(matScale.has_value(), "matrix scale must be provided for FP8 matrix");
    CHECK_INPUT((*matScale), FP8_BLOCK_SCALING_SF_DTYPE);
    TORCH_CHECK(mat.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "Matrix dtype must be FP8 (the matrix will be dequantized on the fly).");
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

torch::Tensor fp8_block_scaling_gemm_hopper(torch::Tensor mat1, torch::Tensor mat2,
    std::optional<torch::Tensor> mat1Scale, std::optional<torch::Tensor> mat2Scale)
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

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float const* mat1ScalePtr = mat1Scale.has_value() ? mat1Scale->data_ptr<float>() : nullptr;
    float const* mat2ScalePtr = mat2Scale.has_value() ? mat2Scale->data_ptr<float>() : nullptr;

    gemm_runner->gemm(reinterpret_cast<__nv_fp8_e4m3*>(mat1.data_ptr()), k,
        reinterpret_cast<__nv_fp8_e4m3*>(mat2.data_ptr()), k, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n, m, n,
        k, mat1ScalePtr, mat2ScalePtr, stream);

    return out;
}

torch::Tensor fp8_block_scale_gemm_blackwell(
    torch::Tensor mat1, torch::Tensor mat2, torch::Tensor mat1Scale, torch::Tensor mat2Scale)
{
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix dtype must be FP8.");
    TORCH_CHECK(mat2.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix dtype must be FP8.");
    TORCH_CHECK(mat1Scale.scalar_type() == at::ScalarType::Float, "Scale dtype must be FP32.");
    TORCH_CHECK(mat2Scale.scalar_type() == at::ScalarType::Float, "Scale dtype must be FP32.");

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1], "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0], "x",
        mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

    auto const m = mat1.sizes()[0];
    auto const n = mat2.sizes()[0];
    auto const k = mat1.sizes()[1];
    TORCH_CHECK(m <= std::numeric_limits<int32_t>::max(), "M must be within int32");
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "N must be within int32");
    TORCH_CHECK(k <= std::numeric_limits<int32_t>::max(), "K must be within int32");

    TORCH_CHECK(k % 128 == 0, "K must be a multiple of 128, (K=", k, ")");
    TORCH_CHECK(n % 128 == 0, "N must be a multiple of 128, (N=", n, ")");

    TORCH_CHECK(mat1Scale.dim() == 2, "mat1Scale must be a matrix");
    TORCH_CHECK(mat1Scale.sizes()[0] == k / 128, "mat1Scale must have size K/128 x M");
    TORCH_CHECK(mat1Scale.sizes()[1] == m, "mat1Scale must have size K/128 x M");
    TORCH_CHECK(mat2Scale.dim() == 2, "mat2Scale must be a matrix");
    TORCH_CHECK(mat2Scale.sizes()[0] == n / 128, "mat2Scale must have size N/128 x K/128");
    TORCH_CHECK(mat2Scale.sizes()[1] == k / 128, "mat2Scale must have size N/128 x K/128");

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float const* mat1ScalePtr = mat1Scale.data_ptr<float>();
    float const* mat2ScalePtr = mat2Scale.data_ptr<float>();

    at::Tensor out = at::detail::empty_cuda({m, n}, at::ScalarType::BFloat16, mat1.device(), std::nullopt);
    // The output scale is not used in the current implementation.
    /*
    at::Tensor outScale = at::detail::empty_cuda({n / 128, m}, at::ScalarType::Float, mat1.device(), std::nullopt);
    float* outScalePtr = outScale.data_ptr<float>();
    */
    float* outScalePtr = nullptr;

    tensorrt_llm::kernels::TrtllmGenBlockScaleGemmRunner runner(Data_type::DATA_TYPE_BF16);
    runner.run(
        m, n, k, mat1.data_ptr(), mat1ScalePtr, mat2.data_ptr(), mat2ScalePtr, out.data_ptr(), outScalePtr, stream);

    return out;
}

extern torch::Tensor fp8_block_scaling_gemm(torch::Tensor mat1, torch::Tensor mat2,
    std::optional<torch::Tensor> mat1Scale, std::optional<torch::Tensor> mat2Scale)
{
    auto const sm = tensorrt_llm::common::getSMVersion();
    switch (sm)
    {
    case 100:
        TORCH_CHECK(mat1Scale.has_value(), "mat1Scale must be provided for SM100");
        TORCH_CHECK(mat2Scale.has_value(), "mat2Scale must be provided for SM100");
        return fp8_block_scale_gemm_blackwell(mat1, mat2, mat1Scale.value(), mat2Scale.value());
    case 90: return fp8_block_scaling_gemm_hopper(mat1, mat2, mat1Scale, mat2Scale);
    default: TORCH_CHECK(false, "Unsupported SM version for FP8 block scaling GEMM");
    }
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

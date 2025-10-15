/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/gemm/KernelRunner.h"

#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

#include <optional>

using namespace tensorrt_llm::kernels::fp8_blockscale_gemm;
using namespace tensorrt_llm::kernels;

namespace torch_ext
{

using Fp8BlockScaleGemmRunnerPtr = std::unique_ptr<CutlassFp8BlockScaleGemmRunnerInterface>;

namespace
{
void check_input_dtypes(torch::Tensor const& mat, torch::Tensor const& matScale)
{
    TORCH_CHECK(mat.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "Matrix dtype must be FP8 (the matrix will be dequantized on the fly).");

    CHECK_INPUT(matScale, FP8_BLOCK_SCALING_SF_DTYPE);
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

torch::Tensor fp8_block_scaling_gemm_ada(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale)
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
    TORCH_CHECK(k % 128 == 0, "K must be a multiple of 128, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    at::Tensor out = at::detail::empty_cuda({m, n}, at::ScalarType::BFloat16, mat1.device(), std::nullopt);

    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float const* mat1ScalePtr = mat1Scale.data_ptr<float>();
    float const* mat2ScalePtr = mat2Scale.data_ptr<float>();

    gemm_runner->gemm(reinterpret_cast<__nv_fp8_e4m3*>(mat1.data_ptr()), k,
        reinterpret_cast<__nv_fp8_e4m3*>(mat2.data_ptr()), k, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n, m, n,
        k, mat1ScalePtr, mat2ScalePtr, stream);

    return out;
}

torch::Tensor fp8_block_scaling_gemm_hopper(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale)
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

    float const* mat1ScalePtr = mat1Scale.data_ptr<float>();
    float const* mat2ScalePtr = mat2Scale.data_ptr<float>();

    gemm_runner->gemm(reinterpret_cast<__nv_fp8_e4m3*>(mat1.data_ptr()), k,
        reinterpret_cast<__nv_fp8_e4m3*>(mat2.data_ptr()), k, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n, m, n,
        k, mat1ScalePtr, mat2ScalePtr, stream);

    return out;
}

torch::Tensor fp8_block_scale_gemm_blackwell(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale)
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

    TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

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

    // transposeMmaOutput is hardcoded for now
    tensorrt_llm::kernels::TrtllmGenGemmRunnerOptions options = {.eltTypeA = gemm::trtllm::gen::Dtype::E4m3,
        .outputType = gemm::trtllm::gen::Dtype::Bfloat16,
        .deepSeekFp8 = true,
        .transposeMmaOutput = true};

    tensorrt_llm::kernels::TrtllmGenGemmRunner runner(options);

    int64_t const numBytesWorkspace = runner.getWorkspaceSizeInBytes(m, n, k);
    at::Tensor workspace
        = at::detail::empty_cuda({numBytesWorkspace}, at::ScalarType::Char, torch::kCUDA, std::nullopt);

    runner.run(m, n, k, mat1.const_data_ptr(), mat1ScalePtr, mat2.const_data_ptr(), mat2ScalePtr, out.data_ptr(),
        /* scaleC */ nullptr, outScalePtr, workspace.data_ptr(), stream.stream(), mat1.get_device());

    return out;
}

extern torch::Tensor fp8_block_scaling_gemm(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale)
{
    auto const sm = tensorrt_llm::common::getSMVersion();
    switch (sm)
    {
    case 103: return fp8_block_scale_gemm_blackwell(mat1, mat2, mat1Scale, mat2Scale);
    case 100: return fp8_block_scale_gemm_blackwell(mat1, mat2, mat1Scale, mat2Scale);
    case 90: return fp8_block_scaling_gemm_hopper(mat1, mat2, mat1Scale, mat2Scale);
    case 89: return fp8_block_scaling_gemm_ada(mat1, mat2, mat1Scale, mat2Scale);
    default: TORCH_CHECK(false, "Unsupported SM version for FP8 block scaling GEMM");
    }
}

torch::Tensor fp8_block_scaling_moe_gemm_hopper(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, torch::Tensor const& token_offset)
{
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix dtype must be FP8.");
    TORCH_CHECK(mat2.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix dtype must be FP8.");
    TORCH_CHECK(mat1Scale.scalar_type() == at::ScalarType::Float, "Scale dtype must be FP32.");
    TORCH_CHECK(mat2Scale.scalar_type() == at::ScalarType::Float, "Scale dtype must be FP32.");
    TORCH_CHECK(token_offset.scalar_type() == at::ScalarType::Long, "Token offset dtype must be INT64.");

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix of shape (m_total, k)");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a matrix of shape (num_problems, n, k)");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[2], "mat1 and mat2 shapes cannot be multiplied");

    auto const m_total = mat1.sizes()[0];
    auto const num_problems = mat2.sizes()[0];
    auto const n = mat2.sizes()[1];
    auto const k = mat2.sizes()[2];
    TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    at::Tensor out = at::detail::empty_cuda({m_total, n}, at::ScalarType::BFloat16, mat1.device(), std::nullopt);

    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float const* mat1ScalePtr = mat1Scale.data_ptr<float>();
    float const* mat2ScalePtr = mat2Scale.data_ptr<float>();

    auto workspace_size = static_cast<int64_t>(gemm_runner->getWorkspaceSizeBase(m_total, n, k, num_problems));
    auto workspace = at::detail::empty_cuda({workspace_size}, at::ScalarType::Byte, mat1.device(), std::nullopt);
    void* workspace_ptr = workspace.data_ptr();
    gemm_runner->configureWorkspace(static_cast<char*>(workspace_ptr));
    gemm_runner->moeGemm(out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int64_t*>(token_offset.data_ptr()), num_problems, n, k, stream, mat1ScalePtr, mat2ScalePtr);

    return out;
}

extern torch::Tensor fp8_block_scaling_moe_gemm(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, torch::Tensor const& token_offset)
{
    auto const sm = tensorrt_llm::common::getSMVersion();
    switch (sm)
    {
    case 90: return fp8_block_scaling_moe_gemm_hopper(mat1, mat2, mat1Scale, mat2Scale, token_offset);
    default: TORCH_CHECK(false, "Unsupported SM version for FP8 block scaling MoEGEMM");
    }
}

// All inputs are k-major
torch::Tensor fp8_block_scaling_bmm_out(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, torch::Tensor& out)
{
    check_input_dtypes(mat1, mat1Scale);
    check_input_dtypes(mat2, mat2Scale);

    TORCH_CHECK(mat1.dim() == 3, "mat1 must be a batched matrix");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a batched matrix");
    TORCH_CHECK(mat1.sizes()[0] == mat2.sizes()[0], "mat1 and mat2 batch dim must be the same but got", mat1.sizes()[0],
        ", and ", mat2.sizes()[0]);
    TORCH_CHECK(mat1.sizes()[2] == mat2.sizes()[2], "mat1 and mat2 k dim must be the same but got", mat1.sizes()[2],
        ", and ", mat2.sizes()[2]);

    // mat1 could be strided due to padding

    auto const b = mat1.sizes()[0];
    auto const m = mat1.sizes()[1];
    auto const n = mat2.sizes()[1];
    auto const k = mat1.sizes()[2];
    TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    CHECK_TH_CUDA(out);
    CHECK_TYPE(out, at::ScalarType::BFloat16);
    auto const& out_shape = out.sizes();
    TORCH_CHECK(out_shape[0] == b && out_shape[1] == m && out_shape[2] == n, "out shape must be (", b, ", ", m, ", ", n,
        "), but got (", out_shape[0], ", ", out_shape[1], ", ", out_shape[2], ").");

    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float* mat1ScalePtr = mat1Scale.data_ptr<float>();
    float* mat2ScalePtr = mat2Scale.data_ptr<float>();

    auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    auto* mat1_ptr = reinterpret_cast<__nv_fp8_e4m3*>(mat1.data_ptr());
    auto* mat2_ptr = reinterpret_cast<__nv_fp8_e4m3*>(mat2.data_ptr());

    TORCH_CHECK(out.strides()[2] == 1, "The last stride of out must be 1, not ", out.strides()[2]);
    TORCH_CHECK(mat1.strides()[2] == 1, "The last stride of mat1 must be 1, not ", mat1.strides()[2]);
    TORCH_CHECK(mat2.strides()[2] == 1, "The last stride of mat2 must be 1, not ", mat2.strides()[2]);

    auto const strideD = out.strides()[0]; // m * n
    auto const ldd = out.strides()[1];     // n

    auto const strideA = mat1.strides()[0];
    auto const lda = mat1.strides()[1];

    auto const strideB = mat2.strides()[0];
    auto const ldb = mat2.strides()[1];

    // mat1Scale is a 1D tensor which doesn't carry any stride information
    auto const strideScalesA = ((m + 4 - 1) / 4 * 4) * ((k + 128 - 1) / 128);

    gemm_runner->strideBatchGemm(out_ptr, ldd, strideD, mat1_ptr, lda, strideA, mat2_ptr, ldb, strideB, b, m, n, k,
        stream, mat1ScalePtr, strideScalesA, mat2ScalePtr);

    return out;
}

// All inputs are k-major
torch::Tensor fp8_block_scaling_bmm(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, std::optional<c10::ScalarType> out_dtype)
{
    auto const b = mat1.sizes()[0];
    auto const m = mat1.sizes()[1];
    auto const n = mat2.sizes()[1];

    auto const dtype = out_dtype.value_or(at::ScalarType::BFloat16);

    at::Tensor out = at::detail::empty_cuda({b, m, n}, dtype, mat1.device(), std::nullopt);
    return fp8_block_scaling_bmm_out(mat1, mat2, mat1Scale, mat2Scale, out);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fp8_block_scaling_gemm(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale) -> Tensor");
    m.def(
        "fp8_block_scaling_bmm(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, ScalarType? "
        "out_dtype=None) -> Tensor");
    m.def(
        "fp8_block_scaling_bmm_out(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor(a!) out) -> "
        "Tensor(a!)");
    m.def(
        "fp8_block_scaling_moe_gemm(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor token_offset) "
        "-> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp8_block_scaling_gemm", &torch_ext::fp8_block_scaling_gemm);
    m.impl("fp8_block_scaling_bmm", &torch_ext::fp8_block_scaling_bmm);
    m.impl("fp8_block_scaling_bmm_out", &torch_ext::fp8_block_scaling_bmm_out);
    m.impl("fp8_block_scaling_moe_gemm", &torch_ext::fp8_block_scaling_moe_gemm);
}

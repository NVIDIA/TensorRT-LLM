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
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/gemm/KernelRunner.h"

#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>

#if defined(TRTLLM_ENABLE_DEEP_GEMM_THOP)
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "jit/compiler.hpp"
#include "jit/kernel_runtime.hpp"

#include <dlfcn.h>
#endif

using namespace tensorrt_llm::kernels::fp8_blockscale_gemm;
using namespace tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

using Fp8BlockScaleGemmRunnerPtr = std::unique_ptr<CutlassFp8BlockScaleGemmRunnerInterface>;

namespace
{
constexpr char const* kFp8BlockScalingGemmBackendEnv = "TRTLLM_FP8_BLOCK_SCALING_GEMM_BACKEND";
constexpr char const* kDeepGemmRootEnv = "TRTLLM_DEEP_GEMM_ROOT";

enum class Fp8BlockScalingGemmBackend
{
    Trtllm,
    DeepGemm,
    Auto,
};

std::string normalize_backend_name(std::string backend)
{
    std::transform(backend.begin(), backend.end(), backend.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return backend;
}

Fp8BlockScalingGemmBackend get_fp8_block_scaling_gemm_backend()
{
    char const* env = std::getenv(kFp8BlockScalingGemmBackendEnv);
    if (env == nullptr || env[0] == '\0')
    {
        return Fp8BlockScalingGemmBackend::Trtllm;
    }

    auto const backend = normalize_backend_name(env);
    if (backend == "trtllm" || backend == "default")
    {
        return Fp8BlockScalingGemmBackend::Trtllm;
    }
    if (backend == "direct_deep_gemm" || backend == "deep_gemm" || backend == "deepgemm")
    {
        return Fp8BlockScalingGemmBackend::DeepGemm;
    }
    if (backend == "auto")
    {
        return Fp8BlockScalingGemmBackend::Auto;
    }

    TORCH_CHECK(false, "Unsupported ", kFp8BlockScalingGemmBackendEnv, "=", backend,
        ". Expected 'trtllm', 'direct_deep_gemm', or 'auto'.");
    return Fp8BlockScalingGemmBackend::Trtllm;
}

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

void check_fp8_block_scaling_gemm_common_inputs(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale)
{
    check_input_dtypes(mat1, mat1Scale);
    check_input_dtypes(mat2, mat2Scale);

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1], "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0], "x",
        mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");
}

#if defined(TRTLLM_ENABLE_DEEP_GEMM_THOP)

std::filesystem::path get_shared_library_path()
{
    Dl_info info{};
    if (dladdr(reinterpret_cast<void const*>(&get_shared_library_path), &info) == 0 || info.dli_fname == nullptr)
    {
        return {};
    }
    return std::filesystem::path(info.dli_fname);
}

bool is_deep_gemm_library_root(std::filesystem::path const& path)
{
    return !path.empty() && std::filesystem::exists(path / "include" / "deep_gemm");
}

std::filesystem::path find_deep_gemm_library_root()
{
    if (char const* env = std::getenv(kDeepGemmRootEnv); env != nullptr && env[0] != '\0')
    {
        std::filesystem::path const root(env);
        TORCH_CHECK(is_deep_gemm_library_root(root), kDeepGemmRootEnv, "=", root.string(),
            " is not a DeepGEMM package root containing include/deep_gemm");
        return root;
    }

    auto const shared_library_path = get_shared_library_path();
    if (!shared_library_path.empty())
    {
        auto const trtllm_package_root = shared_library_path.parent_path().parent_path();
        for (auto const& candidate :
            {trtllm_package_root / "deep_gemm", trtllm_package_root / "deep_gemm" / "python" / "deep_gemm"})
        {
            if (is_deep_gemm_library_root(candidate))
            {
                return candidate;
            }
        }
    }

    TORCH_CHECK(false, "Unable to locate the TensorRT-LLM DeepGEMM package root. Set ", kDeepGemmRootEnv,
        " to a directory containing include/deep_gemm.");
    return {};
}

std::filesystem::path find_cuda_home()
{
    for (char const* env_name : {"CUDA_HOME", "CUDA_PATH"})
    {
        if (char const* env = std::getenv(env_name); env != nullptr && env[0] != '\0')
        {
            std::filesystem::path const path(env);
            if (std::filesystem::exists(path / "bin" / "nvcc") && std::filesystem::exists(path / "bin" / "cuobjdump"))
            {
                return path;
            }
        }
    }

    std::filesystem::path const default_path("/usr/local/cuda");
    TORCH_CHECK(std::filesystem::exists(default_path / "bin" / "nvcc")
            && std::filesystem::exists(default_path / "bin" / "cuobjdump"),
        "Unable to locate CUDA_HOME with nvcc and cuobjdump for DeepGEMM JIT. Set CUDA_HOME or CUDA_PATH.");
    return default_path;
}

void init_deep_gemm_runtime_once()
{
    static std::once_flag init_flag;
    std::call_once(init_flag,
        []
        {
            auto const deep_gemm_root = find_deep_gemm_library_root();
            auto const cuda_home = find_cuda_home();
            deep_gemm::Compiler::prepare_init(deep_gemm_root.string(), cuda_home.string());
            deep_gemm::KernelRuntime::prepare_init(cuda_home.string());
        });
}

bool can_use_deep_gemm_hopper(torch::Tensor const& mat1, torch::Tensor const& mat2, torch::Tensor const& mat1Scale,
    torch::Tensor const& mat2Scale)
{
    if (tensorrt_llm::common::getSMVersion() != 90)
    {
        return false;
    }
    if (mat1.scalar_type() != at::ScalarType::Float8_e4m3fn || mat2.scalar_type() != at::ScalarType::Float8_e4m3fn
        || mat1Scale.scalar_type() != at::ScalarType::Float || mat2Scale.scalar_type() != at::ScalarType::Float)
    {
        return false;
    }
    if (mat1.dim() != 2 || mat2.dim() != 2 || mat1.sizes()[1] != mat2.sizes()[1])
    {
        return false;
    }
    auto const m = mat1.sizes()[0];
    auto const n = mat2.sizes()[0];
    auto const k = mat1.sizes()[1];
    if (m <= 0 || n <= 0 || k <= 0 || k % 128 != 0)
    {
        return false;
    }
    if (mat2Scale.dim() != 2 || mat2Scale.sizes()[0] != (n + 127) / 128 || mat2Scale.sizes()[1] != k / 128)
    {
        return false;
    }
    if (mat1Scale.dim() == 2)
    {
        return (mat1Scale.sizes()[0] == m && mat1Scale.sizes()[1] == k / 128)
            || (mat1Scale.sizes()[0] == k / 128 && mat1Scale.sizes()[1] >= m);
    }
    if (mat1Scale.dim() == 1)
    {
        auto const m_padded = ((m + 3) / 4) * 4;
        return mat1Scale.numel() >= (k / 128) * m_padded;
    }
    return false;
}

torch::Tensor to_deep_gemm_a_scale_layout(torch::Tensor const& mat1Scale, int64_t m, int64_t k)
{
    auto const k_blocks = k / 128;

    if (mat1Scale.dim() == 2)
    {
        if (mat1Scale.sizes()[0] == m && mat1Scale.sizes()[1] == k_blocks)
        {
            return mat1Scale;
        }
        if (mat1Scale.sizes()[0] == k_blocks && mat1Scale.sizes()[1] >= m)
        {
            return mat1Scale.slice(1, 0, m).transpose(0, 1);
        }
    }

    if (mat1Scale.dim() == 1)
    {
        auto const m_padded = ((m + 3) / 4) * 4;
        auto const expected_numel = k_blocks * m_padded;
        if (mat1Scale.numel() >= expected_numel)
        {
            return mat1Scale.slice(0, 0, expected_numel).view({k_blocks, m_padded}).slice(1, 0, m).transpose(0, 1);
        }
    }

    TORCH_CHECK(false, "Cannot convert activation scale to DeepGEMM layout: shape=", mat1Scale.sizes(), ", M=", m,
        ", K=", k, ", K_blocks=", k_blocks);
    return {};
}

torch::Tensor fp8_block_scaling_gemm_hopper_deep_gemm(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale)
{
    check_fp8_block_scaling_gemm_common_inputs(mat1, mat2, mat1Scale, mat2Scale);

    auto const m = mat1.sizes()[0];
    auto const n = mat2.sizes()[0];
    auto const k = mat1.sizes()[1];
    TORCH_CHECK(k % 128 == 0, "DeepGEMM backend requires K to be a multiple of 128, (K=", k, ")");
    TORCH_CHECK(can_use_deep_gemm_hopper(mat1, mat2, mat1Scale, mat2Scale),
        "DeepGEMM backend only supports SM90 dense FP8 blockscale GEMM with mat1Scale as TRT raw padded or logical "
        "(M, K/128) FP32 scales and mat2Scale as ((N+127)/128, K/128) FP32 scales.");

    init_deep_gemm_runtime_once();

    at::Tensor out = at::detail::empty_cuda({m, n}, at::ScalarType::BFloat16, mat1.device(), std::nullopt);
    auto const mat1ScaleDeepGemm = to_deep_gemm_a_scale_layout(mat1Scale, m, k);
    auto const recipe = std::make_tuple(1, 128, 128);

    auto const transformed_sfa = deep_gemm::layout::transform_sf_into_required_layout(
        mat1ScaleDeepGemm, static_cast<int>(m), static_cast<int>(k), recipe, std::nullopt, true, true);
    auto const transformed_sfb = deep_gemm::layout::transform_sf_into_required_layout(
        mat2Scale, static_cast<int>(n), static_cast<int>(k), recipe, std::nullopt, false, true);

    auto const major_a = deep_gemm::get_major_type_ab(mat1);
    auto const major_b = deep_gemm::get_major_type_ab(mat2);
    auto const major_sfb = deep_gemm::get_major_type_ab(transformed_sfb);

    TORCH_CHECK(major_a == cute::UMMA::Major::K, "DeepGEMM SM90 backend requires K-major/contiguous mat1.");
    TORCH_CHECK(major_b == cute::UMMA::Major::K, "DeepGEMM SM90 backend requires K-major/contiguous mat2.");
    deep_gemm::check_major_type_cd(out);

    deep_gemm::sm90_fp8_gemm_1d2d(mat1, transformed_sfa, mat2, transformed_sfb, std::nullopt, out, static_cast<int>(m),
        static_cast<int>(n), static_cast<int>(k), major_a, major_b, major_sfb, "nk");
    return out;
}
#endif

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

torch::Tensor fp8_block_scale_gemm_blackwell_geforce(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale)
{
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix dtype must be FP8.");
    TORCH_CHECK(mat2.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix dtype must be FP8.");
    TORCH_CHECK(mat1Scale.scalar_type() == at::ScalarType::Int, "Scale dtype must be Int32.");
    TORCH_CHECK(mat2Scale.scalar_type() == at::ScalarType::Int, "Scale dtype must be Int32.");

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
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    at::Tensor out = at::detail::empty_cuda({m, n}, at::ScalarType::BFloat16, mat1.device(), std::nullopt);

    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float const* mat1ScalePtr = reinterpret_cast<float const*>(mat1Scale.data_ptr());
    float const* mat2ScalePtr = reinterpret_cast<float const*>(mat2Scale.data_ptr());

    gemm_runner->gemm(reinterpret_cast<__nv_fp8_e4m3*>(mat1.data_ptr()), k,
        reinterpret_cast<__nv_fp8_e4m3*>(mat2.data_ptr()), k, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n, m, n,
        k, mat1ScalePtr, mat2ScalePtr, stream);
    return out;
}

torch::Tensor fp8_block_scaling_gemm_hopper(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale)
{
    check_fp8_block_scaling_gemm_common_inputs(mat1, mat2, mat1Scale, mat2Scale);

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
    auto const backend = get_fp8_block_scaling_gemm_backend();
    TORCH_CHECK(backend != Fp8BlockScalingGemmBackend::DeepGemm || sm == 90,
        "DeepGEMM FP8 block scaling backend is currently supported only on SM90/Hopper.");
    switch (sm)
    {
    case 103: return fp8_block_scale_gemm_blackwell(mat1, mat2, mat1Scale, mat2Scale);
    case 100: return fp8_block_scale_gemm_blackwell(mat1, mat2, mat1Scale, mat2Scale);
    case 90:
#if defined(TRTLLM_ENABLE_DEEP_GEMM_THOP)
        if (backend == Fp8BlockScalingGemmBackend::DeepGemm
            || (backend == Fp8BlockScalingGemmBackend::Auto
                && can_use_deep_gemm_hopper(mat1, mat2, mat1Scale, mat2Scale)))
        {
            return fp8_block_scaling_gemm_hopper_deep_gemm(mat1, mat2, mat1Scale, mat2Scale);
        }
#else
        TORCH_CHECK(backend != Fp8BlockScalingGemmBackend::DeepGemm,
            "DeepGEMM backend requested but TensorRT-LLM was built without BUILD_DEEP_GEMM support in th_common.");
#endif
        return fp8_block_scaling_gemm_hopper(mat1, mat2, mat1Scale, mat2Scale);
    case 89: return fp8_block_scaling_gemm_ada(mat1, mat2, mat1Scale, mat2Scale);
    case 120: return fp8_block_scale_gemm_blackwell_geforce(mat1, mat2, mat1Scale, mat2Scale);
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
    auto const expected_m = (m_total + num_problems - 1) / num_problems;
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
        static_cast<int64_t*>(token_offset.data_ptr()), num_problems, expected_m, n, k, stream, mat1ScalePtr,
        mat2ScalePtr);

    return out;
}

torch::Tensor fp8_block_scaling_moe_gemm_blackwell_geforce(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, torch::Tensor const& token_offset)
{
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::BFloat16, "Matrix dtype must be BF16.");
    TORCH_CHECK(mat2.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix dtype must be FP8.");
    TORCH_CHECK(mat1Scale.scalar_type() == at::ScalarType::Int, "Scale dtype must be Int32.");
    TORCH_CHECK(mat2Scale.scalar_type() == at::ScalarType::Int, "Scale dtype must be Int32.");
    TORCH_CHECK(token_offset.scalar_type() == at::ScalarType::Long, "Token offset dtype must be INT64.");

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix of shape (m_total, k)");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a matrix of shape (num_problems, n, k)");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[2], "mat1 and mat2 shapes cannot be multiplied");

    auto const m_total = mat1.sizes()[0];
    auto const num_problems = mat2.sizes()[0];
    auto const n = mat2.sizes()[1];
    auto const k = mat2.sizes()[2];
    auto const expected_m = (m_total + num_problems - 1) / num_problems;
    TORCH_CHECK(k % 128 == 0, "K must be a multiple of 128, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    at::Tensor out = at::detail::empty_cuda({m_total, n}, at::ScalarType::BFloat16, mat1.device(), std::nullopt);

    auto gemm_runner = get_gemm_runner(mat1.scalar_type(), mat2.scalar_type());

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float const* mat1ScalePtr = reinterpret_cast<float const*>(mat1Scale.data_ptr());
    float const* mat2ScalePtr = reinterpret_cast<float const*>(mat2Scale.data_ptr());

    auto workspace_size = static_cast<int64_t>(gemm_runner->getWorkspaceSizeBase(m_total, n, k, num_problems));
    auto workspace = at::detail::empty_cuda({workspace_size}, at::ScalarType::Byte, mat1.device(), std::nullopt);
    void* workspace_ptr = workspace.data_ptr();
    gemm_runner->configureWorkspace(static_cast<char*>(workspace_ptr));
    gemm_runner->moeGemm(out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(),
        static_cast<int64_t*>(token_offset.data_ptr()), num_problems, expected_m, n, k, stream, mat1ScalePtr,
        mat2ScalePtr);

    return out;
}

extern torch::Tensor fp8_block_scaling_moe_gemm(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, torch::Tensor const& token_offset)
{
    auto const sm = tensorrt_llm::common::getSMVersion();
    switch (sm)
    {
    case 90: return fp8_block_scaling_moe_gemm_hopper(mat1, mat2, mat1Scale, mat2Scale, token_offset);
    case 120: return fp8_block_scaling_moe_gemm_blackwell_geforce(mat1, mat2, mat1Scale, mat2Scale, token_offset);
    default: TORCH_CHECK(false, "Unsupported SM version for FP8 block scaling MoEGEMM");
    }
}

// All inputs are k-major
torch::Tensor fp8_block_scaling_bmm_out(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, torch::Tensor& out)
{
    auto const sm = tensorrt_llm::common::getSMVersion();
    if (sm == 120)
    {
        TORCH_CHECK(mat1.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix dtype must be FP8.");
        TORCH_CHECK(mat2.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix dtype must be FP8.");
        TORCH_CHECK(mat1Scale.scalar_type() == at::ScalarType::Int, "Scale dtype must be Int32.");
        TORCH_CHECK(mat2Scale.scalar_type() == at::ScalarType::Int, "Scale dtype must be Int32.");
    }
    else
    {
        check_input_dtypes(mat1, mat1Scale);
        check_input_dtypes(mat2, mat2Scale);
    }

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

    float* mat1ScalePtr = nullptr;
    float* mat2ScalePtr = nullptr;

    if (sm == 120)
    {
        mat1ScalePtr = reinterpret_cast<float*>(mat1Scale.data_ptr());
        mat2ScalePtr = reinterpret_cast<float*>(mat2Scale.data_ptr());
    }
    else
    {
        mat1ScalePtr = mat1Scale.data_ptr<float>();
        mat2ScalePtr = mat2Scale.data_ptr<float>();
    }

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

    // mat1Scale is a 1D tensor which doesn't carry any stride information, no effect on sm120
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

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fp8_block_scaling_gemm_impl(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale) -> Tensor");
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
    m.impl("fp8_block_scaling_gemm_impl", &tensorrt_llm::torch_ext::fp8_block_scaling_gemm);
    m.impl("fp8_block_scaling_bmm", &tensorrt_llm::torch_ext::fp8_block_scaling_bmm);
    m.impl("fp8_block_scaling_bmm_out", &tensorrt_llm::torch_ext::fp8_block_scaling_bmm_out);
    m.impl("fp8_block_scaling_moe_gemm", &tensorrt_llm::torch_ext::fp8_block_scaling_moe_gemm);
}

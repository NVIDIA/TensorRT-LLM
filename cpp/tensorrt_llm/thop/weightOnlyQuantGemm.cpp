/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "cutlass/numeric_types.h"

#include <ATen/cuda/EmptyTensor.h>
#include <optional>

using namespace tensorrt_llm::kernels::cutlass_kernels;
using namespace tensorrt_llm::kernels;

namespace torch_ext
{

using WeightOnlyGemmRunnerPtr = std::shared_ptr<CutlassFpAIntBGemmRunnerInterface>;

namespace
{
void check_input_dtypes(torch::Tensor const& mat_a, torch::Tensor const& mat_b)
{
    TORCH_CHECK(mat_a.scalar_type() == at::ScalarType::BFloat16 || 
                mat_a.scalar_type() == at::ScalarType::Half,
                "Activation matrix dtype must be BF16 or FP16");
    
    TORCH_CHECK(mat_b.scalar_type() == at::ScalarType::Char, "Weight matrix dtype must be INT8");
}

#define DISPATCH_ACTIVATION_TYPE(scalar_type, ...)                                                                   \
    if (scalar_type == at::ScalarType::Half)                                                                         \
    {                                                                                                                \
        using ActivationType = half;                                                                                 \
        __VA_ARGS__();                                                                                               \ 
    }                                                                                                                \
    else if (scalar_type == at::ScalarType::BFloat16)                                                                \
    {                                                                                                                \
        using ActivationType = __nv_bfloat16;                                                                        \
        __VA_ARGS__();                                                                                               \
    }                                                                                                                \
    else                                                                                                             \
    {                                                                                                                \
        TORCH_CHECK(false, "Unsupported activation type");                                                           \
    }

#define DISPATCH_WEIGHT_TYPE(scalar_type, ...)                                                                       \
    if (scalar_type == at::ScalarType::Char)                                                                         \
    {                                                                                                                \
        using WeightType = uint8_t;                                                                                  \
        __VA_ARGS__();                                                                                               \
    }                                                                                                                \
    else if (scalar_type == at::ScalarType::QUInt4x2)                                                                \
    {                                                                                                                \
        using WeightType = cutlass::uint4b_t;                                                                        \
        __VA_ARGS__();                                                                                               \
    }                                                                                                                \
    else                                                                                                             \
    {                                                                                                                \
        TORCH_CHECK(false, "Unsupported weight type");                                                               \
    }

WeightOnlyGemmRunnerPtr get_gemm_runner(at::ScalarType dtype_a, at::ScalarType dtype_b)
{
    WeightOnlyGemmRunnerPtr result;
    
    DISPATCH_ACTIVATION_TYPE(dtype_a,
        [&]
        {
            using ADtypeStatic = ActivationType;
            DISPATCH_WEIGHT_TYPE(dtype_b,
                [&]
                {
                    using BDtypeStatic = WeightType;
                    result = std::make_unique<CutlassFpAIntBGemmRunner<ADtypeStatic, BDtypeStatic, 
                        cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
                })
        })
    
    return result;
}

} // namespace

torch::Tensor weight_only_quant_gemm(
    torch::Tensor const& mat_a, 
    torch::Tensor const& mat_b,
    torch::ScalarType weight_type,
    torch::Tensor const& weight_scales,
    std::optional<c10::ScalarType> out_dtype)
{
    check_input_dtypes(mat_a, mat_b);
    
    TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a matrix");
    TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a matrix");
    TORCH_CHECK(mat_a.sizes()[1] == mat_b.sizes()[0], "mat_a and mat_b shapes cannot be multiplied");
    
    auto const m = mat_a.sizes()[0];
    auto const n = mat_b.sizes()[1];
    auto const k = mat_a.sizes()[1];
    auto real_n = n;
    if (weight_type == at::ScalarType::QUInt4x2)
    {
        real_n = n * 2;
    }
    TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16, (K=", k, ")");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

    auto const dtype = out_dtype.value_or(mat_a.scalar_type());
    at::Tensor out = at::detail::empty_cuda({m, real_n}, dtype, mat_a.device(), std::nullopt);
    auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
    
    auto gemm_runner = get_gemm_runner(mat_a.scalar_type(), weight_type);

    auto workspace_size = gemm_runner->getWorkspaceSize(m, real_n, k);
    at::Tensor workspace;
    char* workspace_ptr = nullptr;
    if (workspace_size > 0)
    {
        workspace = at::detail::empty_cuda({static_cast<int64_t>(workspace_size)}, 
                                          at::ScalarType::Byte, mat_a.device(), std::nullopt);
        workspace_ptr = static_cast<char*>(workspace.data_ptr());
    }
    
    auto configs = gemm_runner->getConfigs();
    TORCH_CHECK(!configs.empty(), "No valid GEMM configurations available");
    auto gemm_config = configs[0];
    
    gemm_runner->gemm(mat_a.data_ptr(), mat_b.data_ptr(), weight_scales.data_ptr(),
                     out.data_ptr(), m, real_n, k, gemm_config,
                     workspace_ptr, workspace_size, stream);

    return out;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("weight_only_quant_gemm(Tensor mat_a, Tensor mat_b, ScalarType weight_type, Tensor weight_scales, "
          "ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("weight_only_quant_gemm", &torch_ext::weight_only_quant_gemm);
}

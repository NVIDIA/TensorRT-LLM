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
#include "weightOnlyQuantGemm.h"
#include "cutlass/numeric_types.h"

#include <ATen/cuda/EmptyTensor.h>
#include <optional>

using namespace tensorrt_llm::kernels::cutlass_kernels;
using namespace tensorrt_llm::kernels;

namespace torch_ext
{

namespace
{
void check_input_dtypes(at::Tensor const& mat_a, at::Tensor const& mat_b)
{
    TORCH_CHECK(mat_a.scalar_type() == at::ScalarType::BFloat16 || mat_a.scalar_type() == at::ScalarType::Half,
        "Activation matrix dtype must be BF16 or FP16");

    TORCH_CHECK(mat_b.scalar_type() == at::ScalarType::Char, "Weight matrix dtype must be INT8");
}

#define DISPATCH_ACTIVATION_TYPE(scalar_type, ...)                                                                     \
    if (scalar_type == at::ScalarType::Half)                                                                           \
    {                                                                                                                  \
        using ActivationType = half;                                                                                   \
        __VA_ARGS__();                                                                                                 \
    }                                                                                                                  \
    else if (scalar_type == at::ScalarType::BFloat16)                                                                  \
    {                                                                                                                  \
        using ActivationType = __nv_bfloat16;                                                                          \
        __VA_ARGS__();                                                                                                 \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TORCH_CHECK(false, "Unsupported activation type");                                                             \
    }

#define DISPATCH_WEIGHT_TYPE(scalar_type, ...)                                                                         \
    if (scalar_type == at::ScalarType::Char)                                                                           \
    {                                                                                                                  \
        using WeightType = uint8_t;                                                                                    \
        __VA_ARGS__();                                                                                                 \
    }                                                                                                                  \
    else if (scalar_type == at::ScalarType::QUInt4x2)                                                                  \
    {                                                                                                                  \
        using WeightType = cutlass::uint4b_t;                                                                          \
        __VA_ARGS__();                                                                                                 \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TORCH_CHECK(false, "Unsupported weight type");                                                                 \
    }

} // namespace

WeightOnlyQuantGemmRunner::WeightOnlyQuantGemmRunner(at::ScalarType activation_dtype, at::ScalarType weight_dtype)
    : mActivationDtype(activation_dtype)
    , mWeightDtype(weight_dtype)
{
    DISPATCH_ACTIVATION_TYPE(activation_dtype,
        [&]
        {
            using ADtypeStatic = ActivationType;
            DISPATCH_WEIGHT_TYPE(weight_dtype,
                [&]
                {
                    using BDtypeStatic = WeightType;
                    mGemmRunner = std::make_shared<CutlassFpAIntBGemmRunner<ADtypeStatic, BDtypeStatic,
                        cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
                })
        })
    mConfigs = mGemmRunner->getConfigs();
    TORCH_CHECK(!mConfigs.empty(), "Failed to get CUTLASS configs for WeightOnlyQuantGemmRunner with activation type ",
        c10::toString(mActivationDtype), ", weight type ", c10::toString(mWeightDtype));
}

at::Tensor WeightOnlyQuantGemmRunner::runGemm(at::Tensor const& mat_a, at::Tensor const& mat_b,
    at::Tensor const& weight_scales, int64_t config_idx, bool to_userbuffers, std::optional<c10::ScalarType> out_dtype)
{
    check_input_dtypes(mat_a, mat_b);

    TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a matrix");
    TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a matrix");
    TORCH_CHECK(mat_a.sizes()[1] == mat_b.sizes()[0], "mat_a and mat_b shapes cannot be multiplied");
    TORCH_CHECK(mat_a.is_cuda() && mat_b.is_cuda() && weight_scales.is_cuda(), "All input tensors must be on CUDA");

    auto const m = mat_a.sizes()[0];
    auto const k = mat_a.sizes()[1];
    auto const n = mat_b.sizes()[1];
    auto real_n = n;
    if (mWeightDtype == at::ScalarType::QUInt4x2)
    {
        real_n = n * 2;
    }

    auto const dtype = out_dtype.value_or(mActivationDtype);
    at::Tensor out;
    if (to_userbuffers)
    {
        out = torch_ext::create_userbuffers_tensor({m, real_n}, dtype).first;
    }
    else
    {
        out = at::detail::empty_cuda({m, real_n}, dtype, mat_a.device(), std::nullopt);
    }

    auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());

    auto workspace_size = mGemmRunner->getWorkspaceSize(m, real_n, k);
    at::Tensor workspace;
    char* workspace_ptr = nullptr;
    if (workspace_size > 0)
    {
        workspace = at::detail::empty_cuda(
            {static_cast<int64_t>(workspace_size)}, at::ScalarType::Byte, mat_a.device(), std::nullopt);
        workspace_ptr = static_cast<char*>(workspace.data_ptr());
    }

    tensorrt_llm::cutlass_extensions::CutlassGemmConfig gemm_config_to_use;
    if (config_idx >= 0 && config_idx < getNumConfigs())
    {
        gemm_config_to_use = mConfigs.at(config_idx);
    }
    else
    {
        gemm_config_to_use = mConfigs.at(0);
    }

    mGemmRunner->gemm(mat_a.data_ptr(), mat_b.data_ptr(), weight_scales.data_ptr(), out.data_ptr(), m, real_n, k,
        gemm_config_to_use, workspace_ptr, workspace_size, stream);

    return out;
}

int64_t WeightOnlyQuantGemmRunner::getNumConfigs() const
{
    TORCH_CHECK(mGemmRunner, "WeightOnlyQuantGemmRunner not initialized properly.");
    return static_cast<int64_t>(mConfigs.size());
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::WeightOnlyQuantGemmRunner>("WeightOnlyQuantGemmRunner")
        .def(torch::init<at::ScalarType, at::ScalarType>())
        .def("run_gemm", &torch_ext::WeightOnlyQuantGemmRunner::runGemm)
        .def("get_num_configs", &torch_ext::WeightOnlyQuantGemmRunner::getNumConfigs);
}

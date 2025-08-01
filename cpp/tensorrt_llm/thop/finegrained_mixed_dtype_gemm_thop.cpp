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

#include "finegrained_mixed_dtype_gemm_thop.h"

#include "cutlass_extensions/gemm_configs.h"
#include "cutlass_extensions/weight_only_quant_op.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if defined(ENABLE_FP8) && defined(TRTLLM_CUDA_FP8_AVAILABLE)
#include <cuda_fp8.h>
#endif

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace torch_ext
{

finegrainedMixedDtypeGemmRunner::finegrainedMixedDtypeGemmRunner(
    at::ScalarType activationDtype, at::ScalarType outputDtype, int64_t quant_mode)
    : mActivationDtype(activationDtype)
    , mOutputDtype(outputDtype)
{
    if (quant_mode == 0)
    {
        if (activationDtype == at::ScalarType::Half)
        {
            TORCH_CHECK(
                outputDtype == activationDtype, "Activation dtype needs to match Output stype", activationDtype);
            mGemmRunner = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<half,
                cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, half, half, half>>();
        }
        else if (activationDtype == at::ScalarType::BFloat16)
        {
            TORCH_CHECK(
                outputDtype == activationDtype, "Activation dtype needs to match Output stype", activationDtype);
            mGemmRunner = std::make_shared<
                tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_bfloat16, cutlass::uint4b_t,
                    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>();
        }

        else if (activationDtype == at::ScalarType::Float8_e4m3fn)
        {
            if (outputDtype == at::ScalarType::BFloat16)
            {
                mGemmRunner = std::make_shared<
                    tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_fp8_e4m3, cutlass::uint4b_t,
                        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, half, __nv_bfloat16, __nv_bfloat16>>();
            }
            else if (outputDtype == at::ScalarType::Half)
            {
                mGemmRunner
                    = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_fp8_e4m3,
                        cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, half, half, half>>();
            }
            else
            {
                TORCH_CHECK(false, "Unsupported output dtype for Float8_e4m3fn activation", outputDtype);
            }
        }
        else
        {
            TORCH_CHECK(false, "Unsupported activation dtype", activationDtype);
        }
    }

    else if (quant_mode == 1)
    {
        if (activationDtype == at::ScalarType::Half)
        {
            TORCH_CHECK(
                outputDtype == activationDtype, "Activation dtype needs to match Output stype", activationDtype);
            mGemmRunner = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<half,
                cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, half, half, half>>();
        }
        else if (activationDtype == at::ScalarType::BFloat16)
        {
            TORCH_CHECK(
                outputDtype == activationDtype, "Activation dtype needs to match Output stype", activationDtype);
            mGemmRunner
                = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_bfloat16,
                    cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, __nv_bfloat16,
                    __nv_bfloat16, __nv_bfloat16>>();
        }
        else if (activationDtype == at::ScalarType::Float8_e4m3fn)
        {
            if (outputDtype == at::ScalarType::BFloat16)
            {
                mGemmRunner = std::make_shared<
                    tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_fp8_e4m3, cutlass::uint4b_t,
                        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, half, __nv_bfloat16, __nv_bfloat16>>();
            }
            else if (outputDtype == at::ScalarType::Half)
            {
                mGemmRunner = std::make_shared<
                    tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_fp8_e4m3, cutlass::uint4b_t,
                        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, half, half, half>>();
            }
            else
            {
                TORCH_CHECK(false, "Unsupported output dtype for Float8_e4m3fn activation", outputDtype);
            }
        }
    }
    else
    {
        TORCH_CHECK(false, "Unsupported quant mode for finegrainedMixedDtypeGemmRunner: ", quant_mode);
    }

    TORCH_CHECK(mGemmRunner, "Failed to create finegrained Mixed Dtype GEMM runner for activation type ",
        c10::toString(activationDtype));
    mConfigs = mGemmRunner->getConfigs(); // Get configs via the interface
    TORCH_CHECK(!mConfigs.empty(), "Failed to get CUTLASS configs for finegrainedMixedDtype GEMM with activation type ",
        c10::toString(activationDtype));
}

at::Tensor finegrainedMixedDtypeGemmRunner::runGemm(at::Tensor const& A, at::Tensor const& B_packed,
    at::Tensor const& scales, int64_t group_size_long, int64_t configIdx, std::optional<at::Tensor> bias,
    std::optional<at::Tensor> zeros, double alpha) const
{
    TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() && scales.is_cuda(), "All input tensors must be on CUDA");
    TORCH_CHECK(A.scalar_type() == mActivationDtype, "Activation tensor A's dtype ", c10::toString(A.scalar_type()),
        " does not match runner's expected dtype ", c10::toString(mActivationDtype));
    TORCH_CHECK(B_packed.scalar_type() == torch::kQUInt4x2 || B_packed.scalar_type() == torch::kInt8
            || B_packed.scalar_type() == torch::kUInt8,
        "B_packed must be quint4x2, int8, or uint8 (view of quantized data)");

    TORCH_CHECK(A.is_contiguous() && B_packed.is_contiguous() && scales.is_contiguous(),
        "All input tensors (A, B_packed, scales) must be contiguous");

    void const* zeros_ptr = nullptr;
    if (zeros.has_value())
    {
        TORCH_CHECK(zeros.value().is_cuda(), "Zeros tensor must be on CUDA");
        TORCH_CHECK(zeros.value().scalar_type() == torch::kFloat16 || zeros.value().scalar_type() == torch::kBFloat16,
            "Zeros must be FP16 or BF16");
        TORCH_CHECK(zeros.value().is_contiguous(), "Zeros tensor must be contiguous");
        zeros_ptr = zeros.value().data_ptr();
    }

    void const* bias_ptr = nullptr;
    if (bias.has_value())
    {
        TORCH_CHECK(bias.value().scalar_type() == torch::kFloat16 || bias.value().scalar_type() == torch::kBFloat16,
            "Bias must be FP16 or BF16");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
        bias_ptr = bias.value().data_ptr();
    }

    int M = 0, K_act = 0;
    // Logic to determine M and K_act from A_tensor dimensions
    if (A.dim() == 2)
    {
        M = A.size(0);
        K_act = A.size(1);
    }
    else
    { // A.dim() >= 3
        M = A.size(0);
        for (int i = 1; i < A.dim() - 1; ++i)
            M *= A.size(i);
        K_act = A.size(A.dim() - 1);
    }

    // Assuming B_packed is [K_weights, N_packed_int4_pairs] or similar
    // K_weights should match K_act. N_orig is 2 * N_packed_int4_pairs
    int K_weights = B_packed.size(0);
    int N_packed_int4 = B_packed.size(1); // This is number of uint8_t elements, each holding two int4
    int N_orig = N_packed_int4 * 2;       // N_orig is the original N dimension

    TORCH_CHECK(K_act == K_weights, "K dimension mismatch: A.shape[-1]=", K_act, " vs B_packed.shape[0]=", K_weights);
    int K = K_act;
    int group_size = static_cast<int>(group_size_long);

    std::vector<int64_t> output_shape_vec;
    if (A.dim() == 2)
    {
        output_shape_vec = {static_cast<int64_t>(M), static_cast<int64_t>(N_orig)};
    }
    else
    {
        output_shape_vec.reserve(A.dim());
        for (int i = 0; i < A.dim() - 1; ++i)
            output_shape_vec.push_back(A.size(i));
        output_shape_vec.push_back(N_orig);
    }

    torch::ScalarType output_dtype;
    if (mOutputDtype == at::ScalarType::Half)
    {
        output_dtype = torch::kFloat16;
    }
    else if (mOutputDtype == at::ScalarType::BFloat16)
    {
        output_dtype = torch::kBFloat16;
    }
    else
    {
        TORCH_CHECK(false, "Unsupported output dtype");
    }

    torch::Tensor C_tensor = torch::empty(output_shape_vec, A.options().dtype(output_dtype));

    void const* A_ptr = A.data_ptr();

    TORCH_CHECK(B_packed.is_contiguous(), "B_packed tensor must be contiguous");
    void const* B_ptr = B_packed.data_ptr();
    void const* scales_ptr = scales.data_ptr();
    void* C_ptr = C_tensor.data_ptr();

    tensorrt_llm::cutlass_extensions::CutlassGemmConfig gemm_config_to_use;
    if (configIdx >= 0 && configIdx < getNumConfigs())
    {
        gemm_config_to_use = mConfigs.at(configIdx);
    }
    else
    {
        gemm_config_to_use = mConfigs.at(0);
    }

    size_t workspace_bytes = mGemmRunner->getWorkspaceSize(M, N_orig, K);
    torch::Tensor workspace_tensor = torch::empty(
        {static_cast<int64_t>(workspace_bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));
    char* workspace_ptr = nullptr;
    if (workspace_bytes > 0)
    {
        workspace_ptr = reinterpret_cast<char*>(workspace_tensor.data_ptr());
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index());

    mGemmRunner->gemm(A_ptr, B_ptr, scales_ptr, zeros_ptr, bias_ptr, static_cast<float>(alpha), C_ptr, M, N_orig, K,
        group_size, gemm_config_to_use, workspace_ptr, workspace_bytes, stream);

    return C_tensor;
}

int64_t finegrainedMixedDtypeGemmRunner::getNumConfigs() const
{
    TORCH_CHECK(mGemmRunner, "finegrainedMixedDtypeGemmRunner not initialized properly.");
    return static_cast<int64_t>(mConfigs.size());
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::finegrainedMixedDtypeGemmRunner>("finegrainedMixedDtypeGemmRunner")
        .def(torch::init<at::ScalarType, at::ScalarType, int64_t>())
        .def("run_gemm", &torch_ext::finegrainedMixedDtypeGemmRunner::runGemm)
        .def("get_num_configs", &torch_ext::finegrainedMixedDtypeGemmRunner::getNumConfigs);
}

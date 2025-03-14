/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/common/customAllReduceUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/allReduceFusionKernels.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/kernels/internal_cutlass_kernels/include/fp4_gemm.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#if ENABLE_MULTI_DEVICE
#include <ATen/cuda/EmptyTensor.h>

#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE
#include <nvml.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace torch_ext
{

#if ENABLE_MULTI_DEVICE

using tensorrt_llm::kernels::AllReduceFusionOp;

namespace
{

class DeepseekAllreduceOp
{
public:
    DeepseekAllreduceOp() {}

    ~DeepseekAllreduceOp() = default;

    std::vector<torch::Tensor> run(torch::Tensor input, torch::optional<torch::Tensor> workspace,
        torch::TensorList reduce_fusion_inputs, int64_t rank, int64_t nranks, double eps, int64_t fusion_op) noexcept
    {
        auto const fusion_op_type = static_cast<AllReduceFusionOp>(int8_t(fusion_op));

        torch::Tensor residual_out;
        torch::Tensor norm_out;
        torch::Tensor quant_out;
        torch::Tensor scale_out;

        tensorrt_llm::kernels::ar_fusion::AllReduceFusionParams allreduce_fusion_params;

        allreduce_fusion_params.quant_out = nullptr;
        allreduce_fusion_params.scale_out = nullptr;
        allreduce_fusion_params.residual_out = nullptr;
        allreduce_fusion_params.norm_out = nullptr;

        if (fusion_op_type == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4)
        {
            TORCH_CHECK(reduce_fusion_inputs.size() == 3, "Pre-MLP fusion should have 3 inputs.");

            int64_t sfVecSize = 16;
            int64_t m = 1;
            auto const& inputShape = input.sizes();
            auto const& r = inputShape.size();
            TORCH_CHECK(r >= 2, "Input should be >=2D tensor.");
            for (size_t i = 0; i < r - 1; i++)
            {
                m *= inputShape[i];
            }
            auto const k = inputShape[r - 1];
            TORCH_CHECK(k % sfVecSize == 0, "Input should be divisible by sfVecSize.");
            std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
            outputShape[r - 1] = k / 2;

            quant_out = at::detail::empty_cuda(outputShape, FLOAT4_E2M1X2, input.device(), std::nullopt);
            scale_out = at::detail::empty_cuda(
                {tensorrt_llm::computeSFSize(m, k / sfVecSize)}, SF_DTYPE, input.device(), std::nullopt);
            residual_out = torch::empty_like(reduce_fusion_inputs[0]);

            allreduce_fusion_params.quant_out = quant_out.mutable_data_ptr();
            allreduce_fusion_params.scale_out = scale_out.mutable_data_ptr();
            allreduce_fusion_params.residual_out = residual_out.mutable_data_ptr();
        }
        else if (fusion_op_type == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {
            norm_out = torch::empty_like(input);
            residual_out = torch::empty_like(reduce_fusion_inputs[0]);

            allreduce_fusion_params.norm_out = norm_out.mutable_data_ptr();
            allreduce_fusion_params.residual_out = residual_out.mutable_data_ptr();
        }
        else
        {

            return std::vector<torch::Tensor>();
        }

        allreduce_fusion_params.nranks = static_cast<int>(nranks);
        allreduce_fusion_params.rank = static_cast<int>(rank);
        allreduce_fusion_params.dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
        allreduce_fusion_params.size = static_cast<int>(input.numel());
        allreduce_fusion_params.hidden_dim = static_cast<int>(input.size(-1));
        allreduce_fusion_params.workspace = reinterpret_cast<void**>(workspace.value().mutable_data_ptr());
        allreduce_fusion_params.allreduce_in = input.data_ptr();
        allreduce_fusion_params.residual_in = reduce_fusion_inputs[0].data_ptr();
        allreduce_fusion_params.rms_gamma = reduce_fusion_inputs[1].data_ptr();
        allreduce_fusion_params.rms_eps = static_cast<float>(eps);

        if (fusion_op_type == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4)
        {
            allreduce_fusion_params.scale_factor = static_cast<float*>(reduce_fusion_inputs[2].data_ptr());
        }
        else
        {
            allreduce_fusion_params.scale_factor = nullptr;
        }

        allreduce_fusion_params.stream = at::cuda::getCurrentCUDAStream(input.get_device());

        tensorrt_llm::kernels::ar_fusion::allreduce_fusion_op(allreduce_fusion_params);

        if (fusion_op_type == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4)
        {
            return std::vector<torch::Tensor>({quant_out, scale_out, residual_out});
        }
        else if (fusion_op_type == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {
            return std::vector<torch::Tensor>({norm_out, residual_out});
        }
        else
        {
            return std::vector<torch::Tensor>();
        }
    }
};

} // namespace

#endif // ENABLE_MULTI_DEVICE

std::vector<torch::Tensor> deepseekAllreduceFusion(torch::Tensor input, torch::optional<torch::Tensor> workspace,
    torch::TensorList reduce_fusion_inputs, int64_t const rank, int64_t const nranks, double const eps,
    int64_t const fusion_op)
{
#if ENABLE_MULTI_DEVICE
    DeepseekAllreduceOp op;
    auto output = op.run(input, workspace, reduce_fusion_inputs, rank, nranks, eps, fusion_op);
    return output;
#else
    return std::vector<torch::Tensor>();
#endif // ENABLE_MULTI_DEVICE
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    // reduce_fusion_inputs includes
    // 0: residual
    // 1. gamma
    // 2. scale_factor: only when fusion_op == RESIDUAL_RMS_NORM_QUANT_NVFP4
    m.def(
        "deepseek_allreduce_fusion(Tensor input, Tensor? workspace, Tensor[] reduce_fusion_inputs, "
        "int rank, int nranks, float eps, int fusion_op) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("deepseek_allreduce_fusion", &torch_ext::deepseekAllreduceFusion);
}

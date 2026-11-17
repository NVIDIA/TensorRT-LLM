/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
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
#include "kimi_k3_vl/csrc/k3MnnvlAllreduce.h"
#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <torch/extension.h>

using tensorrt_llm::kernels::AllReduceFusionOp;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

using MnnvlFusionPattern = tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern;

MnnvlFusionPattern getMnnvlFusionPattern(AllReduceFusionOp fusionOp)
{
    switch (fusionOp)
    {
    case AllReduceFusionOp::NONE: return MnnvlFusionPattern::kAllReduce;
    case AllReduceFusionOp::RMS_NORM: return MnnvlFusionPattern::kARRMSNorm;
    case AllReduceFusionOp::RESIDUAL_RMS_NORM: return MnnvlFusionPattern::kARResidualRMSNorm;
    case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8: return MnnvlFusionPattern::kARResidualRMSNormFP8Quant;
    case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4: return MnnvlFusionPattern::kARResidualRMSNormFP4Quant;
    case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8: return MnnvlFusionPattern::kARResidualRMSNormOutFP8Quant;
    case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4: return MnnvlFusionPattern::kARResidualRMSNormOutFP4Quant;
    default:
        TORCH_CHECK(
            false, "[mnnvlFusionAllReduce] Unsupported fusion operation: " + tensorrt_llm::kernels::toString(fusionOp));
    }
    return MnnvlFusionPattern::kAllReduce;
}

bool isMnnvlQuantFusionOp(AllReduceFusionOp fusionOp)
{
    return fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8
        || fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8
        || fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4
        || fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4;
}

bool isMnnvlNvfp4FusionOp(AllReduceFusionOp fusionOp)
{
    return fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4
        || fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4;
}

bool hasMnnvlNormOutput(AllReduceFusionOp fusionOp)
{
    return fusionOp == AllReduceFusionOp::RMS_NORM || fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM
        || fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8
        || fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4;
}

}

std::vector<torch::Tensor> k3MnnvlFusionAllReduce(torch::Tensor& input, torch::optional<torch::Tensor> const& gamma,
    torch::optional<torch::Tensor> const& residual_in, torch::optional<double> epsilon, torch::Tensor& comm_buffer,
    torch::Tensor& buffer_flags, bool rmsnorm_fusion, torch::optional<torch::Tensor> const& scale, int64_t fusion_op_,
    int64_t one_shot_max_bytes, bool early_trigger)
{
    auto* mcast_mem = tensorrt_llm::common::findMcastDevMemBuffer(comm_buffer.data_ptr());
    TORCH_CHECK(
        mcast_mem != nullptr, "[mnnvlFusionAllReduce] comm_buffer must be obtained from a mcastBuffer instance.");
    TORCH_CHECK(mcast_mem->isMapped(), "[mnnvlFusionAllReduce] MNNVL workspace handles are not attached.");
    TORCH_CHECK(input.is_contiguous(), "[mnnvlFusionAllReduce] input must be contiguous");
    TORCH_CHECK(input.dim() >= 2, "[mnnvlFusionAllReduce] input must have at least 2 dimensions");

    auto const eltsPerThread = sizeof(float4) / input.itemsize();
    auto const hiddenDim = input.size(-1);
    auto const numTokens = input.numel() / hiddenDim;
    TORCH_CHECK(hiddenDim % eltsPerThread == 0,
        "[mnnvlFusionAllReduce] Hidden dimension must be divisible by " + std::to_string(eltsPerThread) + ", got "
            + std::to_string(hiddenDim));

    auto fusionOp = static_cast<AllReduceFusionOp>(int8_t(fusion_op_));
    if (fusionOp == AllReduceFusionOp::NONE && rmsnorm_fusion)
    {
        fusionOp = AllReduceFusionOp::RESIDUAL_RMS_NORM;
    }
    auto const pattern = getMnnvlFusionPattern(fusionOp);
    bool const hasRmsNormFusion = fusionOp != AllReduceFusionOp::NONE;

    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
    torch::Tensor reduceOut;
    torch::Tensor normOut;
    torch::Tensor residualOut;
    torch::Tensor quantOut;
    torch::Tensor scaleOut;

    if (hasRmsNormFusion)
    {

        bool const needsResidual = fusionOp != AllReduceFusionOp::RMS_NORM;
        TORCH_CHECK(gamma.has_value() && epsilon.has_value(),
            "[mnnvlFusionAllReduce] gamma and epsilon must be provided for RMSNorm fusion");
        TORCH_CHECK(!needsResidual || residual_in.has_value(),
            "[mnnvlFusionAllReduce] residual_in must be provided for residual RMSNorm fusion");
        TORCH_CHECK(!residual_in.has_value() || residual_in.value().is_contiguous(),
            "[mnnvlFusionAllReduce] residual_in must be contiguous");
        TORCH_CHECK(gamma.value().is_contiguous(), "[mnnvlFusionAllReduce] gamma must be contiguous");

        if (needsResidual)
        {
            residualOut = torch::empty_like(residual_in.value());
        }
        if (hasMnnvlNormOutput(fusionOp))
        {
            normOut = torch::empty_like(input);
        }
    }
    else
    {
        reduceOut = torch::empty_like(input);
    }

    if (isMnnvlQuantFusionOp(fusionOp))
    {
        TORCH_CHECK(scale.has_value(), "[mnnvlFusionAllReduce] scale is required for quantized fusion");
        TORCH_CHECK(scale.value().scalar_type() == torch::kFloat32,
            "[mnnvlFusionAllReduce] scale must have dtype float32 for quantized fusion");
        TORCH_CHECK(scale.value().is_contiguous(), "[mnnvlFusionAllReduce] scale must be contiguous");

        if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8
            || fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8)
        {
            quantOut = at::detail::empty_cuda(input.sizes(), torch::kFloat8_e4m3fn, input.device(), std::nullopt);
        }
        else if (isMnnvlNvfp4FusionOp(fusionOp))
        {
            int64_t constexpr kSfVecSize = 16;
            TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
                "[mnnvlFusionAllReduce] NVFP4 quantization requires FP16 or BF16 input");
            TORCH_CHECK(hiddenDim % kSfVecSize == 0,
                "[mnnvlFusionAllReduce] hidden dimension must be divisible by 16 for NVFP4 quantization");

            std::vector<int64_t> quantShape(input.sizes().begin(), input.sizes().end());
            quantShape.back() /= 2;
            quantOut = at::detail::empty_cuda(
                quantShape, ::tensorrt_llm::torch_ext::FLOAT4_E2M1X2, input.device(), std::nullopt);
            scaleOut
                = at::detail::empty_cuda({tensorrt_llm::computeSwizzledLayoutSFSize(numTokens, hiddenDim / kSfVecSize)},
                    ::tensorrt_llm::torch_ext::SF_DTYPE, input.device(), std::nullopt);
        }
        else
        {
            TORCH_CHECK(false,
                "[mnnvlFusionAllReduce] Unsupported quantized fusion operation: "
                    + tensorrt_llm::kernels::toString(fusionOp));
        }
    }

    auto allreduce_params = TRTLLM_MV2_NS::kernels::kimiK3MnnvlOneShot::AllReduceFusionParams();
    allreduce_params.nRanks = mcast_mem->getWorldSize();
    allreduce_params.rank = mcast_mem->getRank();
    allreduce_params.dType = dtype;
    allreduce_params.numTokens = numTokens;
    allreduce_params.tokenDim = hiddenDim;
    allreduce_params.bufferPtrsDev = reinterpret_cast<void**>(mcast_mem->getBufferPtrsDev());
    allreduce_params.bufferPtrLocal = comm_buffer.mutable_data_ptr();
    allreduce_params.multicastPtr = mcast_mem->getMulticastPtr();
    allreduce_params.bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags.mutable_data_ptr());
    allreduce_params.input = input.const_data_ptr();
    allreduce_params.pattern = pattern;
    allreduce_params.layout = tensorrt_llm::QuantizationSFLayout::SWIZZLED;

    if (fusionOp == AllReduceFusionOp::NONE)
    {
        allreduce_params.output = reduceOut.mutable_data_ptr();
    }
    else
    {
        allreduce_params.output = normOut.defined() ? normOut.mutable_data_ptr() : nullptr;

        allreduce_params.residualIn = residual_in.has_value() ? residual_in.value().const_data_ptr() : nullptr;
        allreduce_params.gamma = gamma.value().const_data_ptr();
        allreduce_params.epsilon = static_cast<float>(epsilon.value());
        allreduce_params.rmsNormFusion = true;

        allreduce_params.residualOut = residualOut.defined() ? residualOut.mutable_data_ptr() : nullptr;
    }

    if (isMnnvlQuantFusionOp(fusionOp))
    {
        allreduce_params.quantOut = quantOut.mutable_data_ptr();
        allreduce_params.scaleOut = scaleOut.defined() ? scaleOut.mutable_data_ptr() : nullptr;
        allreduce_params.scaleFactor = scale.value().data_ptr<float>();
    }

    allreduce_params.rmsNormFusion = hasRmsNormFusion;
    allreduce_params.stream = at::cuda::getCurrentCUDAStream(input.get_device());

    allreduce_params.earlyTrigger = early_trigger;
    TORCH_CHECK(one_shot_max_bytes >= 0, "[mnnvlFusionAllReduce] one_shot_max_bytes must be non-negative");

    if (numTokens * hiddenDim * allreduce_params.nRanks * input.itemsize() <= static_cast<size_t>(one_shot_max_bytes))
    {
        TRTLLM_MV2_NS::kernels::kimiK3MnnvlOneShot::oneshotAllreduceFusionOp(allreduce_params);
    }
    else
    {
        tensorrt_llm::kernels::mnnvl::twoshotAllreduceFusionOp(allreduce_params);
    }

    switch (fusionOp)
    {
    case AllReduceFusionOp::NONE: return {reduceOut};
    case AllReduceFusionOp::RMS_NORM: return {normOut};
    case AllReduceFusionOp::RESIDUAL_RMS_NORM: return {normOut, residualOut};
    case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8: return {quantOut, residualOut};
    case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8: return {normOut, quantOut, residualOut};
    case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4: return {quantOut, scaleOut, residualOut};
    case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4: return {normOut, quantOut, scaleOut, residualOut};
    default:
        TORCH_CHECK(
            false, "[mnnvlFusionAllReduce] Unsupported fusion operation: " + tensorrt_llm::kernels::toString(fusionOp));
    }
    return {};
}

}

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "k3_mnnvl_fusion_allreduce(Tensor input, Tensor? gamma, Tensor? residual, "
        "float? epsilon, Tensor(a!) comm_buffer, Tensor(b!) buffer_flags, bool rmsnorm_fusion, "
        "Tensor? scale=None, int fusion_op=0, int one_shot_max_bytes=1048576, bool early_trigger=False) -> Tensor[]");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("k3_mnnvl_fusion_allreduce", &TRTLLM_MV2_NS::torch_ext::k3MnnvlFusionAllReduce);
}

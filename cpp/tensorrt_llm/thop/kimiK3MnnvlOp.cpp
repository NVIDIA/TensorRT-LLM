/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.h"
#include "tensorrt_llm/kernels/kimiK3Mnnvl/mnnvlAllGatherKernels.h"
#include "tensorrt_llm/kernels/kimiK3Mnnvl/mnnvlAllreduceAttnRes.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Kimi K3 pre-MoE residual update in the MNNVL one-shot all-reduce epilogue; see
// tensorrt_llm::kernels::kimiK3Mnnvl::oneshotAllreduceAttnResOp. Returns {normed, updated_prefix_sum}.
std::vector<torch::Tensor> mnnvlAllReduceAttnRes(torch::Tensor const& input,
    torch::optional<torch::Tensor> const& prefix_sum, torch::Tensor const& block_residual,
    torch::Tensor const& res_weight, torch::Tensor const& rms_weight, torch::Tensor const& output_rms_weight,
    double rms_eps, double output_rms_eps, torch::Tensor& comm_buffer, torch::Tensor& buffer_flags)
{
    auto* mcast_mem = tensorrt_llm::common::findMcastDevMemBuffer(comm_buffer.data_ptr());
    TORCH_CHECK(
        mcast_mem != nullptr, "[mnnvlAllReduceAttnRes] comm_buffer must be obtained from a mcastBuffer instance.");
    TORCH_CHECK(mcast_mem->isMapped(), "[mnnvlAllReduceAttnRes] MNNVL workspace handles are not attached.");
    auto const checkBf16 = [](torch::Tensor const& tensor, char const* name)
    {
        TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kBFloat16 && tensor.is_contiguous(),
            "[mnnvlAllReduceAttnRes] ", name, " must be a contiguous CUDA bfloat16 tensor");
    };
    checkBf16(input, "input");
    checkBf16(block_residual, "block_residual");
    checkBf16(res_weight, "res_weight");
    checkBf16(rms_weight, "rms_weight");
    checkBf16(output_rms_weight, "output_rms_weight");
    TORCH_CHECK(input.dim() == 2, "[mnnvlAllReduceAttnRes] input must be [num_tokens, hidden]");
    int64_t const numTokens = input.size(0);
    int64_t const hiddenDim = input.size(1);
    if (prefix_sum.has_value())
    {
        checkBf16(prefix_sum.value(), "prefix_sum");
        TORCH_CHECK(prefix_sum.value().sizes() == input.sizes(),
            "[mnnvlAllReduceAttnRes] prefix_sum must have the shape of input");
    }
    TORCH_CHECK(block_residual.dim() == 3 && block_residual.size(1) == numTokens && block_residual.size(2) == hiddenDim,
        "[mnnvlAllReduceAttnRes] block_residual must be [num_snapshots, num_tokens, hidden]");
    for (auto const* weight : {&res_weight, &rms_weight, &output_rms_weight})
    {
        TORCH_CHECK(weight->dim() == 1 && weight->size(0) == hiddenDim,
            "[mnnvlAllReduceAttnRes] res_weight, rms_weight and output_rms_weight must be [hidden]");
    }
    int64_t const nRanks = mcast_mem->getWorldSize();
    TORCH_CHECK(numTokens * hiddenDim * nRanks <= comm_buffer.size(-1),
        "[mnnvlAllReduceAttnRes] the one-shot footprint of ", numTokens * hiddenDim * nRanks,
        " elements exceeds one Lamport buffer of ", comm_buffer.size(-1), " elements");

    torch::Tensor normOut = torch::empty_like(input);
    torch::Tensor prefixOut = torch::empty_like(input);

    auto params = tensorrt_llm::kernels::mnnvl::AllReduceFusionParams();
    params.nRanks = static_cast<int>(nRanks);
    params.rank = mcast_mem->getRank();
    params.dType = tensorrt_llm::DataType::kBF16;
    params.numTokens = static_cast<int>(numTokens);
    params.tokenDim = static_cast<int>(hiddenDim);
    params.bufferPtrsDev = reinterpret_cast<void**>(mcast_mem->getBufferPtrsDev());
    params.bufferPtrLocal = comm_buffer.mutable_data_ptr();
    params.multicastPtr = mcast_mem->getMulticastPtr();
    params.bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags.mutable_data_ptr());
    params.input = input.const_data_ptr();
    params.residualIn = prefix_sum.has_value() ? prefix_sum.value().const_data_ptr() : nullptr;
    params.residualOut = prefixOut.mutable_data_ptr();
    params.output = normOut.mutable_data_ptr();
    params.stream = at::cuda::getCurrentCUDAStream(input.get_device());

    tensorrt_llm::kernels::kimiK3Mnnvl::AttnResEpilogueParams epilogue{};
    epilogue.blockResidual = block_residual.const_data_ptr();
    epilogue.resWeight = res_weight.const_data_ptr();
    epilogue.rmsWeight = rms_weight.const_data_ptr();
    epilogue.outputRmsWeight = output_rms_weight.const_data_ptr();
    epilogue.rmsEps = static_cast<float>(rms_eps);
    epilogue.outputRmsEps = static_cast<float>(output_rms_eps);
    epilogue.numCandidates = static_cast<int>(block_residual.size(0)) + 1;

    tensorrt_llm::kernels::kimiK3Mnnvl::oneshotAllreduceAttnResOp(params, epilogue);
    return {normOut, prefixOut};
}

// One-shot all-gather over the MNNVL workspace of this rank's fp32 rows [num_tokens, columns]:
// the first bf16_columns columns of every rank are gathered as bf16 into [num_tokens, nRanks *
// bf16_columns], the rest as fp32 into [num_tokens, nRanks * (columns - bf16_columns)]; see
// tensorrt_llm::kernels::kimiK3Mnnvl::mnnvlAllGatherSplitOp.
namespace
{

// The all-gather's checks, outputs and params.
tensorrt_llm::kernels::kimiK3Mnnvl::AllGatherSplitParams makeAllGatherSplitParams(torch::Tensor const& input,
    int64_t bf16_columns, int64_t world_size, torch::Tensor& comm_buffer, torch::Tensor& buffer_flags,
    torch::Tensor& bf16Out, torch::Tensor& fp32Out)
{
    namespace k3Mnnvl = tensorrt_llm::kernels::kimiK3Mnnvl;
    auto* mcast_mem = tensorrt_llm::common::findMcastDevMemBuffer(comm_buffer.data_ptr());
    TORCH_CHECK(
        mcast_mem != nullptr, "[mnnvlAllGatherSplit] comm_buffer must be obtained from a mcastBuffer instance.");
    TORCH_CHECK(mcast_mem->isMapped(), "[mnnvlAllGatherSplit] MNNVL workspace handles are not attached.");
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat32 && input.is_contiguous() && input.dim() == 2,
        "[mnnvlAllGatherSplit] input must be a contiguous [num_tokens, columns] fp32 CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16 == 0,
        "[mnnvlAllGatherSplit] input must be 16-byte aligned");
    int64_t const numTokens = input.size(0);
    int64_t const fp32Columns = input.size(1) - bf16_columns;
    TORCH_CHECK(bf16_columns >= 0 && fp32Columns >= 0 && bf16_columns % 8 == 0 && fp32Columns % 4 == 0,
        "[mnnvlAllGatherSplit] needs bf16_columns a multiple of 8 and the remaining columns a multiple of 4");
    int64_t const nRanks = mcast_mem->getWorldSize();
    TORCH_CHECK(world_size == nRanks, "[mnnvlAllGatherSplit] world_size ", world_size, " is not the workspace's ",
        nRanks, " ranks");
    TORCH_CHECK(k3Mnnvl::mnnvlAllGatherSplitFootprint(numTokens, bf16_columns, fp32Columns, nRanks)
            <= comm_buffer.size(-1) * comm_buffer.element_size(),
        "[mnnvlAllGatherSplit] the exchange does not fit in one Lamport buffer");

    auto const options = input.options();
    bf16Out = torch::empty({numTokens, nRanks * bf16_columns}, options.dtype(torch::kBFloat16));
    fp32Out = torch::empty({numTokens, nRanks * fp32Columns}, options);
    k3Mnnvl::AllGatherSplitParams params{};
    params.input = input.const_data_ptr<float>();
    params.bf16Output = reinterpret_cast<__nv_bfloat16*>(bf16Out.mutable_data_ptr());
    params.fp32Output = fp32Out.mutable_data_ptr<float>();
    params.numTokens = static_cast<int>(numTokens);
    params.bf16Columns = static_cast<int>(bf16_columns);
    params.fp32Columns = static_cast<int>(fp32Columns);
    params.nRanks = static_cast<int>(nRanks);
    params.rank = mcast_mem->getRank();
    params.bufferPtrsDev = reinterpret_cast<void**>(mcast_mem->getBufferPtrsDev());
    params.multicastPtr = mcast_mem->getMulticastPtr();
    params.bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags.mutable_data_ptr());
    params.stream = at::cuda::getCurrentCUDAStream(input.get_device());
    return params;
}

} // namespace

std::vector<torch::Tensor> mnnvlAllGatherSplit(torch::Tensor const& input, int64_t bf16_columns, int64_t world_size,
    torch::Tensor& comm_buffer, torch::Tensor& buffer_flags)
{
    torch::Tensor bf16Out;
    torch::Tensor fp32Out;
    auto const params
        = makeAllGatherSplitParams(input, bf16_columns, world_size, comm_buffer, buffer_flags, bf16Out, fp32Out);
    tensorrt_llm::kernels::kimiK3Mnnvl::mnnvlAllGatherSplitOp(params);
    return {bf16Out, fp32Out};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mnnvl_allreduce_attn_res(Tensor input, Tensor? prefix_sum, Tensor block_residual, Tensor res_weight, "
        "Tensor rms_weight, Tensor output_rms_weight, float rms_eps, float output_rms_eps, Tensor(a!) comm_buffer, "
        "Tensor(b!) buffer_flags) -> Tensor[]");
    m.def(
        "mnnvl_allgather_split(Tensor input, int bf16_columns, int world_size, Tensor(a!) comm_buffer, "
        "Tensor(b!) buffer_flags) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mnnvl_allreduce_attn_res", &tensorrt_llm::torch_ext::mnnvlAllReduceAttnRes);
    m.impl("mnnvl_allgather_split", &tensorrt_llm::torch_ext::mnnvlAllGatherSplit);
}

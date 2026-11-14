/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/csrc/glmMtpHeadGather.h"
#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

// All-gather of input along its last dimension over a dedicated MNNVL Lamport workspace (comm_buffer, three buffers,
// and its flags). The output is the rank-ordered concatenation of the shards; BF16 input is widened to FP32 when
// float_output is set.
at::Tensor rubinMtpHeadGather(
    at::Tensor const& input, at::Tensor& comm_buffer, at::Tensor& buffer_flags, int64_t group_size, bool float_output)
{
    auto* mcastMem = tensorrt_llm::common::findMcastDevMemBuffer(comm_buffer.data_ptr());
    TORCH_CHECK(
        mcastMem != nullptr, "glm_rubin_mtp_head_gather: comm_buffer must be obtained from a McastGPUBuffer instance");
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);
    TORCH_CHECK(input.dim() >= 1, "glm_rubin_mtp_head_gather: input must have at least one dimension");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16 || input.scalar_type() == at::ScalarType::Float,
        "glm_rubin_mtp_head_gather supports BF16 and FP32 input, got ", input.scalar_type());
    TORCH_CHECK(static_cast<int64_t>(mcastMem->getWorldSize()) == group_size, "glm_rubin_mtp_head_gather: group_size ",
        group_size, " does not match the workspace group of ", mcastMem->getWorldSize(), " ranks");
    TORCH_CHECK(buffer_flags.element_size() == 4 && buffer_flags.numel() >= 9,
        "glm_rubin_mtp_head_gather: buffer_flags must hold the nine 32-bit Lamport flags");

    int64_t const shardDim = input.size(-1);
    int64_t const numRows = shardDim == 0 ? 0 : input.numel() / shardDim;
    TORCH_CHECK(shardDim * input.element_size() % 16 == 0,
        "glm_rubin_mtp_head_gather: a shard row must be a multiple of 16 bytes");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0,
        "glm_rubin_mtp_head_gather needs a 16-byte aligned input");
    int64_t const gatheredBytes = numRows * group_size * shardDim * input.element_size();
    int64_t const bufferBytes = comm_buffer.numel() * comm_buffer.element_size() / 3;
    TORCH_CHECK(gatheredBytes <= bufferBytes, "glm_rubin_mtp_head_gather: ", gatheredBytes,
        " gathered bytes exceed the Lamport buffer of ", bufferBytes, " bytes");

    std::vector<int64_t> outShape(input.sizes().begin(), input.sizes().end());
    outShape.back() = group_size * shardDim;
    auto const outType = float_output ? at::ScalarType::Float : input.scalar_type();
    at::Tensor out = at::empty(outShape, input.options().dtype(outType));
    if (numRows == 0)
    {
        return out;
    }

    TRTLLM_MV2_NS::kernels::glm_rubin_mtp::AllGatherParams params;
    params.nRanks = static_cast<int>(group_size);
    params.rank = static_cast<int>(mcastMem->getRank());
    params.dType = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
    params.floatOutput = float_output;
    params.numRows = static_cast<int>(numRows);
    params.shardDim = static_cast<int>(shardDim);
    params.input = input.data_ptr();
    params.output = out.data_ptr();
    params.bufferPtrLocal = comm_buffer.data_ptr();
    params.multicastPtr = mcastMem->getMulticastPtr();
    params.bufferFlags = static_cast<uint32_t*>(buffer_flags.data_ptr());
    params.stream = at::cuda::getCurrentCUDAStream(input.get_device());
    TRTLLM_MV2_NS::kernels::glm_rubin_mtp::oneshotAllGatherOp(params);
    return out;
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "glm_rubin_mtp_head_gather(Tensor input, Tensor(a!) comm_buffer, Tensor(b!) buffer_flags, int group_size, "
        "bool float_output=False) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("glm_rubin_mtp_head_gather", &TRTLLM_MV2_NS::torch_ext::rubinMtpHeadGather);
}

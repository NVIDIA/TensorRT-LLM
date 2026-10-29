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

#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmMnnvlAllgather.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// All-gather of BF16 rows along their last dimension, widened to FP32, over a dedicated MNNVL exchange workspace
// (comm_buffer, the unicast view of its three Lamport buffers, and buffer_flags): the rank-ordered concatenation of
// the shards (kernels/glmMoeDsaRubin/glmMnnvlAllgather.h).
at::Tensor glmMnnvlAllgather(
    at::Tensor const& input, at::Tensor& comm_buffer, at::Tensor& buffer_flags, int64_t group_size)
{
    char const* op = "glm_mnnvl_allgather";
    auto* mcastMem = tensorrt_llm::common::findMcastDevMemBuffer(comm_buffer.data_ptr());
    TORCH_CHECK(mcastMem != nullptr, op, ": comm_buffer must be obtained from a McastGPUBuffer instance");
    TORCH_CHECK(mcastMem->isMapped(), op, ": the MNNVL workspace handles are not attached");
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);
    TORCH_CHECK(input.dim() >= 1, op, ": input must have at least one dimension");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16, op, ": input must be BF16, got ", input.scalar_type());
    TORCH_CHECK(static_cast<int64_t>(mcastMem->getWorldSize()) == group_size, op, ": group_size ", group_size,
        " does not match the workspace group of ", mcastMem->getWorldSize(), " ranks");
    TORCH_CHECK(buffer_flags.element_size() == 4 && buffer_flags.numel() >= 9 && buffer_flags.is_contiguous(), op,
        ": buffer_flags must hold the workspace's nine 32-bit Lamport flags");
    TORCH_CHECK(comm_buffer.device() == input.device() && buffer_flags.device() == input.device(), op,
        ": the workspace must be on the device of input");

    int64_t const shardDim = input.size(-1);
    int64_t const numRows = shardDim == 0 ? 0 : input.numel() / shardDim;
    TORCH_CHECK(shardDim * input.element_size() % 16 == 0, op, ": a shard row must be a multiple of 16 bytes");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0, op, ": input must be 16-byte aligned");
    int64_t const gatheredBytes = numRows * group_size * shardDim * input.element_size();
    int64_t const bufferBytes = comm_buffer.numel() * comm_buffer.element_size() / 3;
    TORCH_CHECK(gatheredBytes <= bufferBytes, op, ": ", gatheredBytes, " gathered bytes exceed the Lamport buffer of ",
        bufferBytes, " bytes");

    std::vector<int64_t> outShape(input.sizes().begin(), input.sizes().end());
    outShape.back() = group_size * shardDim;
    at::Tensor out = at::empty(outShape, input.options().dtype(at::ScalarType::Float));
    if (numRows == 0)
    {
        return out;
    }

    tensorrt_llm::kernels::glm_moe_dsa::MnnvlAllgatherParams params{};
    params.nRanks = static_cast<int>(group_size);
    params.rank = static_cast<int>(mcastMem->getRank());
    params.numRows = static_cast<int>(numRows);
    params.shardDim = static_cast<int>(shardDim);
    params.input = input.const_data_ptr();
    params.output = out.data_ptr<float>();
    params.bufferPtrLocal = comm_buffer.data_ptr();
    params.multicastPtr = mcastMem->getMulticastPtr();
    params.bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags.data_ptr());
    params.stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tensorrt_llm::kernels::glm_moe_dsa::mnnvlAllgather(params);
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm_mnnvl_allgather(Tensor input, Tensor(a!) comm_buffer, Tensor(b!) buffer_flags, int group_size) -> "
        "Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_mnnvl_allgather", &tensorrt_llm::torch_ext::glmMnnvlAllgather);
}

/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/ncclUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/thop/userbuffersTensor.h"
#include <ATen/cuda/EmptyTensor.h>
#include <functional>
#include <set>
#include <torch/extension.h>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

using CreateTensorFn = std::function<at::Tensor(at::IntArrayRef shape, at::ScalarType dtype, c10::Device device)>;

enum class OutputBufferKind : int
{
    Default = 0,
    Userbuffers = 1,
    NcclWindow = 2,
};

inline at::Tensor allocate_output(std::vector<int64_t> const& output_size, at::ScalarType dtype, c10::Device device,
    OutputBufferKind output_buffer_kind, c10::optional<torch::List<int64_t>> group = c10::nullopt)
{
    at::Tensor result;
    switch (output_buffer_kind)
    {
    case OutputBufferKind::NcclWindow:
#if ENABLE_MULTI_DEVICE
        if (group.has_value() && group->size() > 0)
        {
            std::set<int> groupSet;
            for (auto const& rank : *group)
            {
                groupSet.insert(static_cast<int>(rank));
            }
            auto commPtr = getComm(groupSet);
            if (commPtr && *commPtr != nullptr)
            {
                auto [tensor, buffer]
                    = tensorrt_llm::common::nccl_util::createNCCLWindowTensor(*commPtr, output_size, dtype);
                if (tensor.defined() && buffer.isValid())
                {
                    result = tensor;
                }
                else
                {
                    TLLM_LOG_DEBUG("[allocate_output] NCCL window alloc failed; tensor_defined=%d buffer_valid=%d",
                        tensor.defined() ? 1 : 0, buffer.isValid() ? 1 : 0);
                }
            }
            else
            {
                TLLM_LOG_DEBUG("[allocate_output] NCCL comm is null; fallback to default output buffer");
            }
        }
        else
        {
            TLLM_LOG_DEBUG(
                "[allocate_output] NCCL window requested but group is empty; fallback to default output buffer");
        }
#else
        (void) group;
        TLLM_LOG_DEBUG(
            "[allocate_output] NCCL window requested but multi-device is disabled; fallback to default output buffer");
#endif // ENABLE_MULTI_DEVICE
        break;
    case OutputBufferKind::Userbuffers: result = torch_ext::create_userbuffers_tensor(output_size, dtype).first; break;
    case OutputBufferKind::Default:
    default: result = at::detail::empty_cuda(output_size, dtype, device, std::nullopt); break;
    }
    if (!result.defined())
    {
        result = at::detail::empty_cuda(output_size, dtype, device, std::nullopt);
    }
    return result;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

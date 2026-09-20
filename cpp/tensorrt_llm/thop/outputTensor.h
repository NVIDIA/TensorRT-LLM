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

// This enum is exposed to Python via nanobind (tensorrt_llm.bindings.internal.thop.BufferKind).
// Integer values must stay in sync with the Python binding.
enum class BufferKind : int
{
    Default = 0,
    Userbuffers = 1,
    NcclWindow = 2,
};

// Allocates an output tensor of the requested kind and returns both the tensor
// and the kind that was *actually* allocated.  For NcclWindow the actual kind
// may be Default when window allocation fails; for Userbuffers and Default the
// actual kind always matches the requested kind.
inline std::pair<at::Tensor, BufferKind> allocate_output(std::vector<int64_t> const& output_size, at::ScalarType dtype,
    c10::Device device, BufferKind output_buffer_kind, c10::optional<torch::List<int64_t>> group)
{
    at::Tensor result;
    BufferKind actual_kind = BufferKind::Default;
    switch (output_buffer_kind)
    {
    case BufferKind::NcclWindow:
#if ENABLE_MULTI_DEVICE
        if (group.has_value() && group->size() > 0)
        {
            std::set<int> groupSet;
            for (auto const& rank : *group)
            {
                groupSet.insert(static_cast<int>(rank));
            }
            std::shared_ptr<ncclComm_t> commPtr;
            try
            {
                commPtr = getComm(groupSet);
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_DEBUG("[allocate_output] getComm threw (MPI disabled?): %s; fallback to default", e.what());
            }
            if (commPtr && *commPtr != nullptr)
            {
                auto [tensor, buffer]
                    = tensorrt_llm::common::nccl_util::createNCCLWindowTensor(commPtr, output_size, dtype);
                if (tensor.defined() && buffer.isValid())
                {
                    result = tensor;
                    actual_kind = BufferKind::NcclWindow;
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
    case BufferKind::Userbuffers:
        result = torch_ext::create_userbuffers_tensor(output_size, dtype).first;
        actual_kind = BufferKind::Userbuffers;
        break;
    case BufferKind::Default:
    default:
        result = at::detail::empty_cuda(output_size, dtype, device, std::nullopt);
        actual_kind = BufferKind::Default;
        break;
    }
    if (!result.defined())
    {
        result = at::detail::empty_cuda(output_size, dtype, device, std::nullopt);
        actual_kind = BufferKind::Default;
    }
    return {result, actual_kind};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

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

// Registers trtllm::allocate_output as a torch custom op.
//
// Signature:
//   allocate_output(Tensor like, int output_buffer_kind,
//                  int[]? group=None, int[]? shape=None,
//                  ScalarType? out_dtype=None) -> (Tensor, int)
//
// Returns the allocated tensor and the BufferKind integer that was *actually*
// used (may differ from output_buffer_kind when NcclWindow allocation falls
// back to Default).  shape and out_dtype default to like.shape / like.dtype
// when not supplied.

#include "tensorrt_llm/thop/outputTensor.h"

#include <torch/extension.h>

#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<at::Tensor, int64_t> allocateOutputOp(at::Tensor const& like, int64_t output_buffer_kind,
    c10::optional<torch::List<int64_t>> group, c10::optional<at::IntArrayRef> shape,
    c10::optional<at::ScalarType> out_dtype)
{
    std::vector<int64_t> const outShape = (shape.has_value() && !shape->empty())
        ? std::vector<int64_t>(shape->begin(), shape->end())
        : std::vector<int64_t>(like.sizes().begin(), like.sizes().end());
    at::ScalarType const dtype = out_dtype.value_or(like.scalar_type());
    auto const [tensor, actual_kind] = torch_ext::allocate_output(
        outShape, dtype, like.device(), static_cast<torch_ext::BufferKind>(output_buffer_kind), group);
    return {tensor, static_cast<int64_t>(actual_kind)};
}

std::tuple<at::Tensor, int64_t, int64_t> allocateOutputWithNcclWindowOp(at::Tensor const& like,
    int64_t output_buffer_kind, c10::optional<torch::List<int64_t>> group, c10::optional<at::IntArrayRef> shape,
    c10::optional<at::ScalarType> out_dtype)
{
    auto const outShape = (shape.has_value() && !shape->empty())
        ? std::vector<int64_t>(shape->begin(), shape->end())
        : std::vector<int64_t>(like.sizes().begin(), like.sizes().end());
    auto const dtype = out_dtype.value_or(like.scalar_type());
#if ENABLE_MULTI_DEVICE
    if (static_cast<BufferKind>(output_buffer_kind) == BufferKind::NcclWindow && group.has_value() && group->size() > 0)
    {
        std::set<int> groupSet;
        for (auto const& rank : *group)
            groupSet.insert(static_cast<int>(rank));
        try
        {
            auto commPtr = getComm(groupSet);
            if (commPtr && *commPtr != nullptr)
            {
                auto [tensor, buffer]
                    = tensorrt_llm::common::nccl_util::createNCCLWindowTensor(commPtr, outShape, dtype);
                if (tensor.defined() && buffer.isValid())
                    return {tensor, static_cast<int64_t>(BufferKind::NcclWindow),
                        static_cast<int64_t>(reinterpret_cast<uintptr_t>(buffer.window))};
            }
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_DEBUG("[allocate_output_with_nccl_window] NCCL window alloc failed: %s", e.what());
        }
    }
#endif
    auto const [tensor, actualKind]
        = allocate_output(outShape, dtype, like.device(), static_cast<BufferKind>(output_buffer_kind), group);
    return {tensor, static_cast<int64_t>(actualKind), 0};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "allocate_output(Tensor like, int output_buffer_kind, int[]? group, "
        "int[]? shape=None, ScalarType? out_dtype=None) -> (Tensor, int)");
    m.def(
        "allocate_output_with_nccl_window(Tensor like, int output_buffer_kind, int[]? group, int[]? shape=None, "
        "ScalarType? out_dtype=None) -> (Tensor, int, int)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("allocate_output", &tensorrt_llm::torch_ext::allocateOutputOp);
    m.impl("allocate_output_with_nccl_window", &tensorrt_llm::torch_ext::allocateOutputWithNcclWindowOp);
}

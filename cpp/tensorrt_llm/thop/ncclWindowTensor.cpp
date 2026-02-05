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
#include "tensorrt_llm/common/ncclUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/common/tllmLogger.h"

#include <torch/extension.h>

#include <set>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<torch::Tensor, bool> create_nccl_window_tensor_op(
    torch::List<int64_t> const& group, at::IntArrayRef shape, torch::ScalarType dtype)
{
#if ENABLE_MULTI_DEVICE
    if (group.size() == 0)
    {
        return {torch::Tensor(), false};
    }

    std::set<int> group_set;
    for (auto const& rank : group)
    {
        group_set.insert(static_cast<int>(rank));
    }

    auto comm_ptr = getComm(group_set);
    if (!comm_ptr || *comm_ptr == nullptr)
    {
        TLLM_LOG_DEBUG("[create_nccl_window_tensor] NCCL comm is null; skipping allocation");
        return {torch::Tensor(), false};
    }

    auto [tensor, buffer] = tensorrt_llm::common::nccl_util::createNCCLWindowTensor(*comm_ptr, shape, dtype);
    if (!tensor.defined() || buffer.invalid())
    {
        return {torch::Tensor(), false};
    }

    return {tensor, true};
#else
    (void) group;
    (void) shape;
    (void) dtype;
    return {torch::Tensor(), false};
#endif
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("create_nccl_window_tensor(int[] group, int[] shape, ScalarType dtype) -> (Tensor out, bool is_valid)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("create_nccl_window_tensor", &tensorrt_llm::torch_ext::create_nccl_window_tensor_op);
}

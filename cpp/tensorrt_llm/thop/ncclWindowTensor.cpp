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
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <torch/extension.h>

#include <set>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<torch::Tensor, bool> createNcclWindowTensorLikeOp(
    torch::Tensor const& like, torch::List<int64_t> const& group, c10::optional<at::IntArrayRef> shape)
{
#if ENABLE_MULTI_DEVICE
    if (!like.defined() || !like.is_cuda())
    {
        return {torch::Tensor(), false};
    }
    if (group.size() == 0)
    {
        return {torch::Tensor(), false};
    }

    std::set<int> groupSet;
    for (auto const& rank : group)
    {
        groupSet.insert(static_cast<int>(rank));
    }

    auto commPtr = getComm(groupSet);
    if (!commPtr || *commPtr == nullptr)
    {
        TLLM_LOG_DEBUG("[create_nccl_window_tensor] NCCL comm is null; skipping allocation");
        return {torch::Tensor(), false};
    }

    at::IntArrayRef const outShape = (shape.has_value() && !shape->empty()) ? *shape : like.sizes();
    auto [tensor, buffer]
        = tensorrt_llm::common::nccl_util::createNCCLWindowTensor(*commPtr, outShape, like.scalar_type());
    if (!tensor.defined() || buffer.invalid())
    {
        return {torch::Tensor(), false};
    }

    return {tensor, true};
#else
    (void) group;
    (void) like;
    (void) shape;
    return {torch::Tensor(), false};
#endif
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("create_nccl_window_tensor(Tensor like, int[] group, int[]? shape=None) -> (Tensor out, bool is_valid)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("create_nccl_window_tensor", &tensorrt_llm::torch_ext::createNcclWindowTensorLikeOp);
}

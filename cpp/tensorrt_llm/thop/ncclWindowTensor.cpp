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
#include "ncclWindowTensor.h"
#include "tensorrt_llm/common/opUtils.h"
#include <set>

namespace torch_ext
{

torch::Tensor create_nccl_window_tensor(
    torch::List<int64_t> const& group_, at::IntArrayRef shape, torch::ScalarType dtype)
{
    // Convert List to set for getComm
    std::set<int> groupSet;
    for (int64_t rank : group_)
    {
        groupSet.insert(static_cast<int>(rank));
    }

    // Get NCCL communicator for the group
    auto comm = getComm(groupSet);

    // Create NCCL window tensor
    using tensorrt_llm::common::nccl_util::createNCCLWindowTensor;
    auto [tensor, buffer] = createNCCLWindowTensor(*comm, shape, dtype);

    // Return just the tensor (Python doesn't need the buffer object)
    return tensor;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("create_nccl_window_tensor", &torch_ext::create_nccl_window_tensor);
}

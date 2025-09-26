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
#include "userbuffersTensor.h"

namespace torch_ext
{

std::pair<torch::Tensor, tensorrt_llm::runtime::ub::UBBuffer> create_userbuffers_tensor(
    at::IntArrayRef shape, torch::ScalarType dtype)
{
    int64_t buffer_size
        = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()) * torch::elementSize(dtype);

    std::vector<int64_t> strides_vec(shape.size());
    strides_vec[shape.size() - 1] = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 1; --i)
    {
        strides_vec[i - 1] = strides_vec[i] * shape[i];
    }

    auto [ptr, ub] = tensorrt_llm::runtime::ub::UserBuffersManager::get_instance().allocate_userbuffers(buffer_size);
    auto& deleter = ptr.get_deleter();
    return std::make_pair(
        torch::from_blob(ptr.release(), shape, strides_vec, deleter, torch::dtype(dtype).device(torch::kCUDA)), ub);
}

// Custom op interface for create_userbuffers_tensor.
// Python side does not need the UBBuffer object.
torch::Tensor create_userbuffers_tensor_op(at::IntArrayRef shape, torch::ScalarType dtype)
{
    return create_userbuffers_tensor(shape, dtype).first;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("create_userbuffers_tensor", &torch_ext::create_userbuffers_tensor_op);
}

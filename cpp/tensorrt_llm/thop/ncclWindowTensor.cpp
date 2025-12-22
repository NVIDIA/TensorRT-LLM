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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/ncclUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/common/stringUtils.h"
#include <set>

namespace tc = tensorrt_llm::common;

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

    TLLM_CHECK(!groupSet.empty());

    // Get NCCL communicator for the group
    auto comm = getComm(groupSet);

    // Create NCCL window tensor
    using tensorrt_llm::common::nccl_util::createNCCLWindowTensor;
    // Return just the tensor (Python doesn't need the buffer object)
    return createNCCLWindowTensor(*comm, shape, dtype).first;
}

torch::Tensor copy_to_nccl_window(torch::Tensor const& input, torch::List<int64_t> const& group)
{
    // Convert IntArrayRef to vector for arr2str
    std::vector<int64_t> shape_vec(input.sizes().begin(), input.sizes().end());
    std::string shape_str = "[" + tc::vec2str(shape_vec) + "]";

    TLLM_LOG_DEBUG(
        "[copy_to_nccl_window] Creating NCCL window tensor and copying input: "
        "shape=%s, dtype=%d, group_size=%zu, buffer_size=%zu bytes",
        shape_str.c_str(), static_cast<int>(input.scalar_type()), group.size(), input.numel() * input.element_size());

    // Create NCCL window tensor with same shape and dtype as input
    auto window_tensor = create_nccl_window_tensor(group, input.sizes(), input.scalar_type());

    // Copy input data to window buffer (async copy)
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    size_t buffer_size_bytes = input.numel() * input.element_size();
    TLLM_LOG_DEBUG("[copy_to_nccl_window] Performing async memcpy: src=%p, dst=%p, size=%zu bytes", input.data_ptr(),
        window_tensor.data_ptr(), buffer_size_bytes);
    TLLM_CUDA_CHECK(cudaMemcpyAsync(
        window_tensor.data_ptr(), input.data_ptr(), buffer_size_bytes, cudaMemcpyDeviceToDevice, stream));

    TLLM_LOG_DEBUG("[copy_to_nccl_window] Copy completed successfully");
    return window_tensor;
}

torch::Tensor matmul_to_nccl_window(torch::Tensor const& a, torch::Tensor const& b, torch::List<int64_t> const& group)
{
    // Calculate output shape: (..., m, n) where a is (..., m, k) and b is (..., k, n)
    std::vector<int64_t> output_shape(a.sizes().vec());
    output_shape.back() = b.size(-1);

    // Convert shapes to strings using helper function
    std::vector<int64_t> a_shape_vec(a.sizes().begin(), a.sizes().end());
    std::vector<int64_t> b_shape_vec(b.sizes().begin(), b.sizes().end());
    std::string a_shape_str = "[" + tc::vec2str(a_shape_vec) + "]";
    std::string b_shape_str = "[" + tc::vec2str(b_shape_vec) + "]";
    std::string out_shape_str = "[" + tc::vec2str(output_shape) + "]";

    TLLM_LOG_DEBUG(
        "[matmul_to_nccl_window] Allocating NCCL window tensor for matmul output: "
        "a_shape=%s, b_shape=%s, output_shape=%s, dtype=%d",
        a_shape_str.c_str(), b_shape_str.c_str(), out_shape_str.c_str(), static_cast<int>(a.scalar_type()));

    // Create NCCL window tensor for output
    auto output_tensor = create_nccl_window_tensor(group, output_shape, a.scalar_type());

    // Perform matmul directly into window tensor
    torch::matmul_out(output_tensor, a, b);

    TLLM_LOG_DEBUG("[matmul_to_nccl_window] Matmul completed into NCCL window tensor");
    return output_tensor;
}

torch::Tensor add_to_nccl_window(torch::Tensor const& a, torch::Tensor const& b, torch::List<int64_t> const& group)
{
    // Calculate output shape: broadcast shape of a and b
    std::vector<int64_t> output_shape;
    size_t max_dim = std::max(a.dim(), b.dim());
    output_shape.reserve(max_dim);
    for (int64_t i = max_dim - 1; i >= 0; --i)
    {
        int64_t a_dim = (i >= static_cast<int64_t>(a.dim())) ? 1 : a.size(i);
        int64_t b_dim = (i >= static_cast<int64_t>(b.dim())) ? 1 : b.size(i);
        output_shape.push_back(std::max(a_dim, b_dim));
    }
    std::reverse(output_shape.begin(), output_shape.end());

    // Convert shapes to strings using helper function
    std::vector<int64_t> a_shape_vec(a.sizes().begin(), a.sizes().end());
    std::vector<int64_t> b_shape_vec(b.sizes().begin(), b.sizes().end());
    std::string a_shape_str = "[" + tc::vec2str(a_shape_vec) + "]";
    std::string b_shape_str = "[" + tc::vec2str(b_shape_vec) + "]";
    std::string out_shape_str = "[" + tc::vec2str(output_shape) + "]";

    TLLM_LOG_DEBUG(
        "[add_to_nccl_window] Allocating NCCL window tensor for add output: "
        "a_shape=%s, b_shape=%s, output_shape=%s, dtype=%d",
        a_shape_str.c_str(), b_shape_str.c_str(), out_shape_str.c_str(), static_cast<int>(a.scalar_type()));

    // Create NCCL window tensor for output
    auto output_tensor = create_nccl_window_tensor(group, output_shape, a.scalar_type());

    // Perform add directly into window tensor
    torch::add_out(output_tensor, a, b);

    TLLM_LOG_DEBUG("[add_to_nccl_window] Add completed into NCCL window tensor");
    return output_tensor;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("create_nccl_window_tensor", &torch_ext::create_nccl_window_tensor);
    m.def("copy_to_nccl_window", &torch_ext::copy_to_nccl_window);
    m.def("matmul_to_nccl_window", &torch_ext::matmul_to_nccl_window);
    m.def("add_to_nccl_window", &torch_ext::add_to_nccl_window);
}

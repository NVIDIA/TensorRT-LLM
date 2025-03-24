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

#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <torch/extension.h>

namespace torch_ext
{

torch::Tensor userbuffers_allreduce_finalize(torch::Tensor input, bool force_applying_finalize)
{
#if ENABLE_MULTI_DEVICE
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    auto output = torch::empty_like(input);
    size_t size = input.numel();
    int hidden_size = input.size(-1);

    TLLM_CHECK_WITH_INFO(tensorrt_llm::runtime::ub::ub_is_initialized(), "UserBuffer has not been initialized!");
    auto ub_buffer1 = tensorrt_llm::runtime::ub::ub_get(1);
    auto ub_comm = tensorrt_llm::runtime::ub::ub_comm();
    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());

    tensorrt_llm::kernels::ub::allgather2_userbuff_residual_launcher(
        ub_buffer1.handle, 0, size, hidden_size, input.data_ptr(), dtype, ub_comm, stream, force_applying_finalize);
    auto const elem_size = tensorrt_llm::common::getDTypeSize(dtype);
    // Force memcpy to avoid ub tensor exposed to torch graph.
    TLLM_CUDA_CHECK(
        cudaMemcpyAsync(output.data_ptr(), ub_buffer1.addr, size * elem_size, cudaMemcpyDeviceToDevice, stream));
    return output;
#else
    return input;
#endif // ENABLE_MULTI_DEVICE
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("userbuffers_allreduce_finalize(Tensor input, bool force_applying_finalize) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("userbuffers_allreduce_finalize", &torch_ext::userbuffers_allreduce_finalize);
}

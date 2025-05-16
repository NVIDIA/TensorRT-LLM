/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/thop/ncclCommunicatorOp.h"

#include "tensorrt_llm/runtime/iBuffer.h"

namespace tr = tensorrt_llm::runtime;

namespace torch_ext
{

NcclCommunicatorOp::NcclCommunicatorOp(int64_t worldSize, int64_t rank)
    : mRank(static_cast<int32_t>(rank))
{
    mPipelineComm = std::make_shared<tensorrt_llm::runtime::NcclCommunicator>(worldSize, rank);
}

void NcclCommunicatorOp::send(th::Tensor tensor, int64_t toRank) const
{
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
    tensorrt_llm::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
    mPipelineComm->send(*tr::IBuffer::wrap(ptr, size), static_cast<int>(toRank), cudaStream);
}

void NcclCommunicatorOp::recv(th::Tensor& tensor, int64_t fromRank) const
{
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
    tensorrt_llm::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
    mPipelineComm->receive(*tr::IBuffer::wrap(ptr, size), static_cast<int>(fromRank), cudaStream);
}

} // namespace torch_ext

static auto trtllmNcclCommunicator = torch::jit::class_<torch_ext::NcclCommunicatorOp>("trtllm", "NcclCommunicatorOp")
                                         .def(torch::jit::init<int64_t, int64_t>())
                                         .def("send", &torch_ext::NcclCommunicatorOp::send)
                                         .def("recv", &torch_ext::NcclCommunicatorOp::recv);

/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
#if TRTLLM_ENABLE_NCCL_PROFILING && ENABLE_MULTI_DEVICE
#include "tensorrt_llm/common/profiler.h"
#include <numeric>
#endif

namespace tr = tensorrt_llm::runtime;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

NcclCommunicatorOp::NcclCommunicatorOp(int64_t worldSize, int64_t rank)
    : mRank(static_cast<int32_t>(rank))
{
#if TRTLLM_ENABLE_NCCL_PROFILING && ENABLE_MULTI_DEVICE
    mWorldSize = static_cast<int32_t>(worldSize);
#endif
    mPipelineComm = std::make_shared<tensorrt_llm::runtime::NcclCommunicator>(worldSize, rank);
}

void NcclCommunicatorOp::send(th::Tensor tensor, int64_t toRank) const
{
#if TRTLLM_ENABLE_NCCL_PROFILING && ENABLE_MULTI_DEVICE
    common::RecordFunctionWithMetadata profiler{common::ProfilerRegion::kNccl};
    profiler.startWithLazyName([]() -> std::string { return "trtllm::nccl_metadata"; },
        [&](common::ProfilerMetadata& metadata)
        {
            std::vector<std::int64_t> group(mWorldSize);
            std::iota(group.begin(), group.end(), std::int64_t{0});
            common::addNcclMetadata(metadata, "send", group, mRank, "pp", "p2p",
                tensor.numel() * static_cast<std::int64_t>(tensor.element_size()),
                at::cuda::getCurrentCUDAStream().stream());
            metadata.add("nccl.src_rank", static_cast<std::int64_t>(mRank));
            metadata.add("nccl.dst_rank", toRank);
            metadata.add("nccl.input_shape", tensor.sizes().vec());
            metadata.add("nccl.dtype", c10::toString(tensor.scalar_type()));
            metadata.add("nccl.selected_strategy", "nccl_send_recv");
        });
#endif
    tensor.record_stream(at::cuda::getCurrentCUDAStream());
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
    tensorrt_llm::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
    mPipelineComm->send(*tr::IBuffer::wrap(ptr, size), static_cast<int>(toRank), cudaStream);
}

void NcclCommunicatorOp::recv(th::Tensor& tensor, int64_t fromRank) const
{
#if TRTLLM_ENABLE_NCCL_PROFILING && ENABLE_MULTI_DEVICE
    common::RecordFunctionWithMetadata profiler{common::ProfilerRegion::kNccl};
    profiler.startWithLazyName([]() -> std::string { return "trtllm::nccl_metadata"; },
        [&](common::ProfilerMetadata& metadata)
        {
            std::vector<std::int64_t> group(mWorldSize);
            std::iota(group.begin(), group.end(), std::int64_t{0});
            common::addNcclMetadata(metadata, "recv", group, mRank, "pp", "p2p",
                tensor.numel() * static_cast<std::int64_t>(tensor.element_size()),
                at::cuda::getCurrentCUDAStream().stream());
            metadata.add("nccl.src_rank", fromRank);
            metadata.add("nccl.dst_rank", static_cast<std::int64_t>(mRank));
            metadata.add("nccl.output_shape", tensor.sizes().vec());
            metadata.add("nccl.dtype", c10::toString(tensor.scalar_type()));
            metadata.add("nccl.selected_strategy", "nccl_send_recv");
        });
#endif
    tensor.record_stream(at::cuda::getCurrentCUDAStream());
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
    tensorrt_llm::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
    mPipelineComm->receive(*tr::IBuffer::wrap(ptr, size), static_cast<int>(fromRank), cudaStream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

static auto trtllmNcclCommunicator
    = torch::jit::class_<tensorrt_llm::torch_ext::NcclCommunicatorOp>("trtllm", "NcclCommunicatorOp")
          .def(torch::jit::init<int64_t, int64_t>())
          .def("send", &tensorrt_llm::torch_ext::NcclCommunicatorOp::send)
          .def("recv", &tensorrt_llm::torch_ext::NcclCommunicatorOp::recv);

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

#if TRTLLM_ENABLE_NCCL_PROFILING && ENABLE_MULTI_DEVICE
namespace
{
//! Describe this communicator's P2P submission without synchronizing the stream.
void addP2pMetadata(common::ProfilerMetadata& metadata, std::string_view operation, th::Tensor const& tensor,
    std::size_t bytes, std::int32_t worldSize, std::int32_t rank, std::int64_t peer, cudaStream_t stream)
{
    bool const send = operation == "send";
    std::vector<std::int64_t> group(worldSize);
    std::iota(group.begin(), group.end(), std::int64_t{0});
    metadata.add("nccl.operation", operation);
    metadata.add("nccl.group", group);
    metadata.add("nccl.rank", rank);
    metadata.add("nccl.parallelism_dimension", "pp");
    metadata.add("nccl.kind", "p2p");
    metadata.add("nccl.logical_bytes", static_cast<std::int64_t>(bytes));
    cudaStreamCaptureStatus captureStatus{};
    auto const captureError = cudaStreamIsCapturing(stream, &captureStatus);
    if (captureError != cudaSuccess)
    {
        static_cast<void>(cudaGetLastError());
    }
    metadata.add("nccl.cuda_graph_state",
        captureError != cudaSuccess                        ? "unknown"
            : captureStatus == cudaStreamCaptureStatusNone ? "eager"
                                                           : "capture");
    metadata.add("nccl.src_rank", send ? rank : peer);
    metadata.add("nccl.dst_rank", send ? peer : rank);
    metadata.add(send ? "nccl.input_shape" : "nccl.output_shape", tensor.sizes().vec());
    metadata.add("nccl.dtype", c10::toString(tensor.scalar_type()));
    metadata.add("nccl.selected_strategy", "nccl_send_recv");
}
} // namespace
#endif

NcclCommunicatorOp::NcclCommunicatorOp(int64_t worldSize, int64_t rank)
    : mRank(static_cast<int32_t>(rank))
{
#if TRTLLM_ENABLE_NCCL_PROFILING && ENABLE_MULTI_DEVICE
    mWorldSize = static_cast<int32_t>(worldSize);
#endif
    mPipelineComm = std::make_shared<tensorrt_llm::runtime::NcclCommunicator>(worldSize, rank);
}

//! Send tensor bytes on the current stream, lazily describing the submission when probes are enabled.
void NcclCommunicatorOp::send(th::Tensor tensor, int64_t toRank) const
{
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
#if TRTLLM_ENABLE_NCCL_PROFILING && ENABLE_MULTI_DEVICE
    common::RecordFunctionWithMetadata scope{common::ProfilerRegion::kNccl};
    scope.startWithLazyName([]() -> std::string { return "trtllm::nccl_send"; },
        [&](common::ProfilerMetadata& metadata)
        {
            addP2pMetadata(
                metadata, "send", tensor, size, mWorldSize, mRank, toRank, at::cuda::getCurrentCUDAStream().stream());
        });
#endif
    tensor.record_stream(at::cuda::getCurrentCUDAStream());
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    tensorrt_llm::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
    mPipelineComm->send(*tr::IBuffer::wrap(ptr, size), static_cast<int>(toRank), cudaStream);
}

//! Receive into tensor on the current stream, lazily describing the submission when probes are enabled.
void NcclCommunicatorOp::recv(th::Tensor& tensor, int64_t fromRank) const
{
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
#if TRTLLM_ENABLE_NCCL_PROFILING && ENABLE_MULTI_DEVICE
    common::RecordFunctionWithMetadata scope{common::ProfilerRegion::kNccl};
    scope.startWithLazyName([]() -> std::string { return "trtllm::nccl_recv"; },
        [&](common::ProfilerMetadata& metadata)
        {
            addP2pMetadata(
                metadata, "recv", tensor, size, mWorldSize, mRank, fromRank, at::cuda::getCurrentCUDAStream().stream());
        });
#endif
    tensor.record_stream(at::cuda::getCurrentCUDAStream());
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
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

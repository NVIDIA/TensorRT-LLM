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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

namespace tr = tensorrt_llm::runtime;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

std::mutex& pipelineCommunicatorMutex()
{
    static auto* mutex = new std::mutex;
    return *mutex;
}

struct PipelineCommunicatorCache
{
    int64_t worldSize{-1};
    int64_t rank{-1};
    std::shared_ptr<tensorrt_llm::runtime::NcclCommunicator> communicator;
};

PipelineCommunicatorCache& pipelineCommunicatorCache()
{
    // Python model-engine reference counts are process-local and therefore
    // cannot define a cross-rank NCCL teardown boundary. Keep the FT PP
    // communicator alive for the process lifetime so rank-skewed wrapper churn
    // never makes one rank reuse an old communicator while a peer enters a
    // fresh-ID rendezvous. Coordinated teardown belongs to the Phase-2 owner.
    static auto* cache = new PipelineCommunicatorCache;
    return *cache;
}

} // namespace

NcclCommunicatorOp::NcclCommunicatorOp(int64_t worldSize, int64_t rank)
    : mRank(static_cast<int32_t>(rank))
{
    TLLM_CHECK_WITH_INFO(worldSize > 0 && worldSize <= std::numeric_limits<int32_t>::max(),
        "NCCL error: world size is out of range: %lld", static_cast<long long>(worldSize));
    TLLM_CHECK_WITH_INFO(rank >= 0 && rank < worldSize && rank <= std::numeric_limits<int32_t>::max(),
        "NCCL error: rank is out of range: %lld", static_cast<long long>(rank));

    if (!mpi::isFaultToleranceModeEnabled())
    {
        // The legacy MPI-broadcast bootstrap assigns independent NCCL IDs, so
        // multiple PP communicators remain valid when fault tolerance is off.
        mPipelineComm = std::make_shared<tensorrt_llm::runtime::NcclCommunicator>(
            static_cast<int>(worldSize), static_cast<int>(rank));
        return;
    }

    std::lock_guard<std::mutex> const lock(pipelineCommunicatorMutex());
    auto& cache = pipelineCommunicatorCache();
    if (cache.communicator == nullptr)
    {
        cache.communicator = std::make_shared<tensorrt_llm::runtime::NcclCommunicator>(
            static_cast<int>(worldSize), static_cast<int>(rank));
        cache.worldSize = worldSize;
        cache.rank = rank;
    }
    else
    {
        TLLM_CHECK_WITH_INFO(cache.worldSize == worldSize && cache.rank == rank,
            "NCCL error: the process-lifetime PP communicator was initialized for world size %lld rank %lld, "
            "not world size %lld rank %lld",
            static_cast<long long>(cache.worldSize), static_cast<long long>(cache.rank),
            static_cast<long long>(worldSize), static_cast<long long>(rank));
    }
    mPipelineComm = cache.communicator;
}

void NcclCommunicatorOp::send(th::Tensor tensor, int64_t toRank) const
{
    tensor.record_stream(at::cuda::getCurrentCUDAStream());
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
    tensorrt_llm::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
    mPipelineComm->send(*tr::IBuffer::wrap(ptr, size), static_cast<int>(toRank), cudaStream);
}

void NcclCommunicatorOp::recv(th::Tensor& tensor, int64_t fromRank) const
{
    tensor.record_stream(at::cuda::getCurrentCUDAStream());
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
    tensorrt_llm::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
    mPipelineComm->receive(*tr::IBuffer::wrap(ptr, size), static_cast<int>(fromRank), cudaStream);
}

void NcclCommunicatorOp::abort()
{
    mPipelineComm->abort();
}

void NcclCommunicatorOp::abortAndReinit(std::vector<int64_t> const& activeRanks, int64_t rendezvousId)
{
    TLLM_CHECK_WITH_INFO(rendezvousId > 0, "NCCL error: recovery rendezvous ID must be positive, got %lld",
        static_cast<long long>(rendezvousId));
    std::vector<int> ranks;
    ranks.reserve(activeRanks.size());
    for (int64_t const rank : activeRanks)
    {
        TLLM_CHECK_WITH_INFO(rank >= 0 && rank <= std::numeric_limits<int>::max(),
            "NCCL error: active rank is out of range: %lld", static_cast<long long>(rank));
        ranks.push_back(static_cast<int>(rank));
    }
    mPipelineComm->abortAndReinit(ranks, static_cast<std::uint64_t>(rendezvousId));
}

std::string NcclCommunicatorOp::getAsyncError() const
{
    return mPipelineComm->getAsyncError();
}

std::vector<int64_t> NcclCommunicatorOp::getActiveRanks() const
{
    auto const ranks = mPipelineComm->getActiveRanks();
    return std::vector<int64_t>(ranks.begin(), ranks.end());
}

void ncclCommAbortAndReinit(
    std::vector<int64_t> const& oldGroup, std::vector<int64_t> const& activeGroup, int64_t rendezvousId)
{
#if ENABLE_MULTI_DEVICE
    TLLM_CHECK_WITH_INFO(rendezvousId > 0, "NCCL error: recovery rendezvous ID must be positive, got %lld",
        static_cast<long long>(rendezvousId));
    std::set<int> oldRanks;
    std::set<int> activeRanks;
    for (auto const rank : oldGroup)
    {
        TLLM_CHECK_WITH_INFO(rank >= 0 && rank <= std::numeric_limits<int>::max(),
            "NCCL error: old-group rank is out of range: %lld", static_cast<long long>(rank));
        oldRanks.insert(static_cast<int>(rank));
    }
    for (auto const rank : activeGroup)
    {
        TLLM_CHECK_WITH_INFO(rank >= 0 && rank <= std::numeric_limits<int>::max(),
            "NCCL error: active-group rank is out of range: %lld", static_cast<long long>(rank));
        activeRanks.insert(static_cast<int>(rank));
    }
    TLLM_CHECK_WITH_INFO(oldRanks.size() == oldGroup.size(), "NCCL error: old group contains duplicate ranks");
    TLLM_CHECK_WITH_INFO(activeRanks.size() == activeGroup.size(), "NCCL error: active group contains duplicate ranks");
    abortAndReinitComm(oldRanks, activeRanks, static_cast<std::uint64_t>(rendezvousId));
#else
    (void) oldGroup;
    (void) activeGroup;
    (void) rendezvousId;
    TLLM_THROW("NCCL error: multi device support is disabled.");
#endif
}

c10::optional<std::string> ncclCommGetAsyncError(std::vector<int64_t> const& group)
{
#if ENABLE_MULTI_DEVICE
    std::set<int> ranks;
    for (auto const rank : group)
    {
        TLLM_CHECK_WITH_INFO(rank >= 0 && rank <= std::numeric_limits<int>::max(),
            "NCCL error: group rank is out of range: %lld", static_cast<long long>(rank));
        ranks.insert(static_cast<int>(rank));
    }
    TLLM_CHECK_WITH_INFO(ranks.size() == group.size(), "NCCL error: group contains duplicate ranks");
    auto error = getCommAsyncError(ranks);
    if (error.has_value())
    {
        return *error;
    }
    return c10::nullopt;
#else
    (void) group;
    return c10::nullopt;
#endif
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

static auto trtllmNcclCommunicator
    = torch::jit::class_<tensorrt_llm::torch_ext::NcclCommunicatorOp>("trtllm", "NcclCommunicatorOp")
          .def(torch::jit::init<int64_t, int64_t>())
          .def("send", &tensorrt_llm::torch_ext::NcclCommunicatorOp::send)
          .def("recv", &tensorrt_llm::torch_ext::NcclCommunicatorOp::recv)
          .def("abort", &tensorrt_llm::torch_ext::NcclCommunicatorOp::abort)
          .def("abort_and_reinit", &tensorrt_llm::torch_ext::NcclCommunicatorOp::abortAndReinit)
          .def("get_async_error", &tensorrt_llm::torch_ext::NcclCommunicatorOp::getAsyncError)
          .def("get_active_ranks", &tensorrt_llm::torch_ext::NcclCommunicatorOp::getActiveRanks);

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("nccl_comm_abort_and_reinit(int[] old_group, int[] active_group, int rendezvous_id) -> ()");
    m.def("nccl_comm_get_async_error(int[] group) -> str?");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("nccl_comm_abort_and_reinit", &tensorrt_llm::torch_ext::ncclCommAbortAndReinit);
    m.impl("nccl_comm_get_async_error", &tensorrt_llm::torch_ext::ncclCommGetAsyncError);
}

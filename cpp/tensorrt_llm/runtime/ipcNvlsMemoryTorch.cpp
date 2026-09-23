/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tensorrt_llm/runtime/ipcNvlsMemoryTorch.h"

#include "ipcNvlsMemoryImpl.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/ipcSocket.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <ctime>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace tensorrt_llm::runtime
{
namespace
{
using detail::IpcCommunicator;
using detail::IpcMemHandle;

void broadcastCpuBytes(c10::intrusive_ptr<c10d::ProcessGroup> const& processGroup, void* data, size_t size, int root)
{
    auto tensor
        = at::from_blob(data, {static_cast<int64_t>(size)}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::vector<at::Tensor> tensors{tensor};
    c10d::BroadcastOptions options;
    options.rootRank = root;
    options.rootTensor = 0;
    PGCHECK_THROW(processGroup->broadcast(tensors, options));
}

void cpuBarrier(c10::intrusive_ptr<c10d::ProcessGroup> const& processGroup)
{
    // A tensor collective selects the CPU/Gloo backend deterministically. A
    // bare ProcessGroup::barrier has no tensor device from which to select a
    // backend on a multi-backend ProcessGroup.
    auto token = at::zeros({1}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::vector<at::Tensor> tensors{token};
    c10d::AllreduceOptions options;
    PGCHECK_THROW(processGroup->allreduce(tensors, options));
}

#if ENABLE_MULTI_DEVICE && !ENABLE_NVSHMEM
class IpcSocketProcessGroupCommunicator final : public IpcCommunicator
{
public:
    explicit IpcSocketProcessGroupCommunicator(c10::intrusive_ptr<c10d::ProcessGroup> processGroup)
        : mProcessGroup(std::move(processGroup))
        , mGroupRank(mProcessGroup->getRank())
        , mGroupSize(mProcessGroup->getSize())
    {
        uint64_t uniqueOpId = 0;
        if (mGroupRank == 0)
        {
            timespec ts{};
            clock_gettime(CLOCK_MONOTONIC, &ts);
            auto const seed = static_cast<unsigned long>(ts.tv_sec) * 1000000000UL + ts.tv_nsec;
            srand(seed);
            uniqueOpId = static_cast<uint64_t>(rand()) ^ (static_cast<uint64_t>(rand()) << 32);
        }
        broadcastCpuBytes(mProcessGroup, &uniqueOpId, sizeof(uniqueOpId), 0);

        mSocket = ncclIpcSocketInit(mGroupRank, uniqueOpId, &mAbortFlag);
        cpuBarrier(mProcessGroup);
    }

    ~IpcSocketProcessGroupCommunicator() override
    {
        ncclIpcSocketClose(mSocket);
    }

    void bcastMemHandle(IpcMemHandle* handle, int root) override
    {
        TLLM_CHECK_WITH_INFO(handle != nullptr, "IPC memory handle must not be null");
        TLLM_CHECK_WITH_INFO(
            root >= 0 && root < mGroupSize, "Root rank %d is outside ProcessGroup of size %d", root, mGroupSize);

        if (mGroupRank == root)
        {
            for (int rank = 0; rank < mGroupSize; ++rank)
            {
                if (rank != root)
                {
                    ncclIpcSocketSendFd(mSocket, handle->fd, rank);
                }
            }
            cpuBarrier(mProcessGroup);
        }
        else
        {
            cpuBarrier(mProcessGroup);
            handle->fd = ncclIpcSocketRecvFd(mSocket);
        }
        cpuBarrier(mProcessGroup);
    }

    [[nodiscard]] int getGroupRank() const override
    {
        return mGroupRank;
    }

    [[nodiscard]] int getGroupSize() const override
    {
        return mGroupSize;
    }

    [[nodiscard]] CUmemAllocationHandleType getMemHandleType() const override
    {
        return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

private:
    c10::intrusive_ptr<c10d::ProcessGroup> mProcessGroup;
    int mGroupRank;
    int mGroupSize;
    uint32_t volatile mAbortFlag{0};
    std::shared_ptr<NcclIpcSocket> mSocket;
};

class IpcFabricProcessGroupCommunicator final : public IpcCommunicator
{
public:
    explicit IpcFabricProcessGroupCommunicator(c10::intrusive_ptr<c10d::ProcessGroup> processGroup)
        : mProcessGroup(std::move(processGroup))
    {
    }

    void bcastMemHandle(IpcMemHandle* handle, int root) override
    {
        TLLM_CHECK_WITH_INFO(handle != nullptr, "IPC memory handle must not be null");
        broadcastCpuBytes(mProcessGroup, &handle->fh, sizeof(handle->fh), root);
    }

    [[nodiscard]] int getGroupRank() const override
    {
        return mProcessGroup->getRank();
    }

    [[nodiscard]] int getGroupSize() const override
    {
        return mProcessGroup->getSize();
    }

    [[nodiscard]] CUmemAllocationHandleType getMemHandleType() const override
    {
        return CU_MEM_HANDLE_TYPE_FABRIC;
    }

private:
    c10::intrusive_ptr<c10d::ProcessGroup> mProcessGroup;
};
#endif // ENABLE_MULTI_DEVICE && !ENABLE_NVSHMEM

class TorchDistIpcNvlsRendezvous final : public IpcNvlsRendezvous
{
public:
    explicit TorchDistIpcNvlsRendezvous(c10::intrusive_ptr<c10d::ProcessGroup> processGroup)
        : mProcessGroup(std::move(processGroup))
    {
        TLLM_CHECK_WITH_INFO(mProcessGroup != nullptr, "ProcessGroup must not be null");
        TLLM_CHECK_WITH_INFO(mProcessGroup->getSize() >= 2, "TorchDist NVLS rendezvous requires at least two ranks");
    }

    IpcNvlsHandle* allocate(size_t bytes) const override
    {
#if ENABLE_MULTI_DEVICE && !ENABLE_NVSHMEM
        TLLM_CHECK_WITH_INFO(ipcNvlsSupported(), "Switch multicast is not supported on this system.");
        TLLM_CHECK_WITH_INFO(bytes > 0, "NVLS allocation size must be positive");

        std::shared_ptr<IpcCommunicator> communicator;
        auto const handleType = detail::getIpcNvlsMemHandleType();
        if (handleType == CU_MEM_HANDLE_TYPE_FABRIC)
        {
            communicator = std::make_shared<IpcFabricProcessGroupCommunicator>(mProcessGroup);
        }
        else
        {
            communicator = std::make_shared<IpcSocketProcessGroupCommunicator>(mProcessGroup);
        }
        auto* handle = detail::ipcNvlsAllocateWithCommunicator(bytes, std::move(communicator));
        TLLM_LOG_INFO("ProcessGroup rank %d NVLS allocate %zu bytes, uc_ptr:%p mc_ptr:%p", rank(), bytes,
            (void*) handle->uc_ptr, (void*) handle->mc_ptr);
        return handle;
#elif ENABLE_NVSHMEM
        TLLM_THROW("TorchDist NVLS rendezvous is not supported with ENABLE_NVSHMEM");
#else
        TLLM_THROW("TorchDist NVLS rendezvous requires ENABLE_MULTI_DEVICE");
#endif
    }

    void barrier() const override
    {
        if (size() > 1)
        {
            cpuBarrier(mProcessGroup);
        }
    }

    [[nodiscard]] int rank() const override
    {
        return mProcessGroup->getRank();
    }

    [[nodiscard]] int size() const override
    {
        return mProcessGroup->getSize();
    }

    [[nodiscard]] uintptr_t identity() const override
    {
        return reinterpret_cast<uintptr_t>(mProcessGroup.get()) | uintptr_t{1};
    }

    [[nodiscard]] IpcNvlsRendezvousKind kind() const override
    {
        return IpcNvlsRendezvousKind::kTorchDist;
    }

    [[nodiscard]] std::string describe() const override
    {
        std::stringstream stream;
        stream << "TorchDistProcessGroup(" << mProcessGroup.get() << ", rank=" << rank() << ", size=" << size() << ')';
        return stream.str();
    }

private:
    c10::intrusive_ptr<c10d::ProcessGroup> mProcessGroup;
};
} // namespace

IpcNvlsRendezvousPtr makeTorchDistIpcNvlsRendezvous(c10::intrusive_ptr<c10d::ProcessGroup> processGroup)
{
    return std::make_shared<TorchDistIpcNvlsRendezvous>(std::move(processGroup));
}

} // namespace tensorrt_llm::runtime

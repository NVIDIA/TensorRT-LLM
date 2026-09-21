/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "tensorrt_llm/runtime/ipcNvlsMemory.h"

#include <memory>

namespace tensorrt_llm::runtime::detail
{

union IpcMemHandle
{
    uint64_t fd;
    CUmemFabricHandle fh;
};

class IpcCommunicator
{
public:
    virtual ~IpcCommunicator() = default;
    virtual void bcastMemHandle(IpcMemHandle* handle, int root) = 0;
    [[nodiscard]] virtual int getGroupRank() const = 0;
    [[nodiscard]] virtual int getGroupSize() const = 0;
    [[nodiscard]] virtual CUmemAllocationHandleType getMemHandleType() const = 0;
};

CUmemAllocationHandleType getIpcNvlsMemHandleType();
IpcNvlsHandle* ipcNvlsAllocateWithCommunicator(size_t size, std::shared_ptr<IpcCommunicator> communicator);

} // namespace tensorrt_llm::runtime::detail

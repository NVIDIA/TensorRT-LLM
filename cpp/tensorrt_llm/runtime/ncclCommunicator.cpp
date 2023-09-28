/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/ncclCommunicator.h"

#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#include <NvInferRuntime.h>
#include <mpi.h>

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

using namespace tensorrt_llm::runtime;

namespace
{
#if ENABLE_MULTI_DEVICE
//! \brief For converting a C++ data type to a Nccl data type.
template <typename T>
struct NcclDataType
{
};

template <>
struct NcclDataType<half>
{
    static constexpr auto value = ncclDataType_t::ncclHalf;
};

template <>
struct NcclDataType<float>
{
    static constexpr auto value = ncclDataType_t::ncclFloat;
};

template <>
struct NcclDataType<std::uint8_t>
{
    static constexpr auto value = ncclDataType_t::ncclUint8;
};

template <>
struct NcclDataType<std::int32_t>
{
    static constexpr auto value = ncclDataType_t::ncclInt32;
};
#endif // ENABLE_MULTI_DEVICE
} // namespace

template <typename T>
void NcclCommunicator::send(
    T* sendbuff, size_t count, int peer, CudaStream const& stream, nvinfer1::ILogger& logger) const
{
#if ENABLE_MULTI_DEVICE
    auto datatype = NcclDataType<T>::value;
    TLLM_NCCL_CHECK(ncclSend(sendbuff, count, datatype, peer, mComm, stream.get()), logger);
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

template void NcclCommunicator::send(std::uint8_t*, size_t, int, CudaStream const&, nvinfer1::ILogger&) const;
template void NcclCommunicator::send(std::int32_t*, size_t, int, CudaStream const&, nvinfer1::ILogger&) const;

template <typename T>
void NcclCommunicator::receive(
    T* sendbuff, size_t count, int peer, CudaStream const& stream, nvinfer1::ILogger& logger) const
{
#if ENABLE_MULTI_DEVICE
    auto datatype = NcclDataType<T>::value;
    TLLM_NCCL_CHECK(ncclRecv(sendbuff, count, datatype, peer, mComm, stream.get()), logger);
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

template void NcclCommunicator::receive(std::uint8_t*, size_t, int, CudaStream const&, nvinfer1::ILogger&) const;
template void NcclCommunicator::receive(std::int32_t*, size_t, int, CudaStream const&, nvinfer1::ILogger&) const;

std::shared_ptr<NcclCommunicator> NcclCommunicator::createPipelineComm(
    WorldConfig const& worldConfig, nvinfer1::ILogger& logger)
{
#if ENABLE_MULTI_DEVICE
    auto ppGroup = worldConfig.getPipelineParallelGroup();

    int myRank = worldConfig.getRank();
    int groupRank = 0;
    for (auto it = ppGroup.begin(); it != ppGroup.end(); ++it)
    {
        if (*it == myRank)
        {
            break;
        }
        ++groupRank;
    }

    ncclUniqueId id;
    if (myRank == ppGroup.front())
    {
        ncclGetUniqueId(&id);
        for (auto it = std::next(std::begin(ppGroup), 1); it != ppGroup.end(); ++it)
        {
            TLLM_MPI_CHECK(MPI_Send(&id, sizeof(id), MPI_BYTE, *it, 0, MPI_COMM_WORLD), logger);
        }
    }
    else
    {
        MPI_Status status;
        TLLM_MPI_CHECK(MPI_Recv(&id, sizeof(id), MPI_BYTE, ppGroup.front(), 0, MPI_COMM_WORLD, &status), logger);
    }

    auto pipelineComm = std::make_shared<NcclCommunicator>();
    TLLM_NCCL_CHECK(ncclCommInitRank(&pipelineComm->mComm, ppGroup.size(), id, groupRank), logger);

    return pipelineComm;
#else
    TLLM_THROW("Multi device support is disabled.");
    return nullptr;
#endif // ENABLE_MULTI_DEVICE
}

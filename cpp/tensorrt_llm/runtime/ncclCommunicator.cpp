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
#include <type_traits>

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
void NcclCommunicator::send(T* sendbuff, size_t count, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    auto datatype = NcclDataType<std::remove_cv_t<T>>::value;
    TLLM_NCCL_CHECK(ncclSend(sendbuff, count, datatype, peer, mComm, stream.get()));
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

template void NcclCommunicator::send(std::uint8_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(std::int32_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(std::uint8_t const*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(std::int32_t const*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(float const*, size_t, int, CudaStream const&) const;

template <typename T>
void NcclCommunicator::receive(T* sendbuff, size_t count, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    auto datatype = NcclDataType<std::remove_cv_t<T>>::value;
    TLLM_NCCL_CHECK(ncclRecv(sendbuff, count, datatype, peer, mComm, stream.get()));
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

template void NcclCommunicator::receive(std::uint8_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::receive(std::int32_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::receive(float*, size_t, int, CudaStream const&) const;

std::shared_ptr<NcclCommunicator> NcclCommunicator::createPipelineComm(WorldConfig const& worldConfig)
{
#if ENABLE_MULTI_DEVICE
    int const myRank = worldConfig.getRank();
    int const worldSize = worldConfig.getSize();

    ncclUniqueId id;
    if (myRank == 0)
    {
        ncclGetUniqueId(&id);
        for (auto peer = 1; peer < worldSize; ++peer)
        {
            TLLM_MPI_CHECK(MPI_Send(&id, sizeof(id), MPI_BYTE, peer, 0, MPI_COMM_WORLD));
        }
    }
    else
    {
        auto constexpr peer = 0;
        MPI_Status status;
        TLLM_MPI_CHECK(MPI_Recv(&id, sizeof(id), MPI_BYTE, peer, 0, MPI_COMM_WORLD, &status));
    }

    auto pipelineComm = std::make_shared<NcclCommunicator>();
    TLLM_NCCL_CHECK(ncclCommInitRank(&pipelineComm->mComm, worldSize, id, myRank));

    return pipelineComm;
#else
    // Python runtime requires instantiation of a communicator even though it may never be used to enable
    // pipeline parallel code-path. To enable this, have an empty communicator with uninitialized state.
    return nullptr;
#endif // ENABLE_MULTI_DEVICE
}

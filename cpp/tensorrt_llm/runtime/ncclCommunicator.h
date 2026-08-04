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

#pragma once

#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

struct ncclComm;
typedef struct ncclComm* ncclComm_t;

namespace tensorrt_llm::runtime
{

class NcclCommunicator
{
public:
    explicit NcclCommunicator(ncclComm_t comm)
        : mComm{comm} {};

    explicit NcclCommunicator(int worldSize, int rank, mpi::MpiComm const& mpiComm = COMM_SESSION)
        : mComm{createComm(worldSize, rank, mpiComm)} {};

    explicit NcclCommunicator(WorldConfig const& worldConfig, mpi::MpiComm const& mpiComm = COMM_SESSION)
        : NcclCommunicator{worldConfig.getSize(), worldConfig.getRank(), mpiComm} {};

    ~NcclCommunicator();

    // no copy
    NcclCommunicator(NcclCommunicator const&) = delete;
    NcclCommunicator& operator=(NcclCommunicator const&) = delete;

    void send(IBuffer const& buf, int peer, CudaStream const& stream) const
    {
        send(buf.data(), buf.getSize(), buf.getDataType(), peer, stream);
    }

    void receive(IBuffer& buf, int peer, CudaStream const& stream) const
    {
        receive(buf.data(), buf.getSize(), buf.getDataType(), peer, stream);
    }

private:
    void send(
        void const* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const;

    void receive(void* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const;

    static ncclComm_t createComm(int worldSize, int rank, mpi::MpiComm const& mpiComm);

    ncclComm_t mComm;
};

} // namespace tensorrt_llm::runtime

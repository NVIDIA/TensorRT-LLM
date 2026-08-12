/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "tensorrt_llm/runtime/mcastGroupComm.h"

namespace tensorrt_llm::runtime
{

MpiMcastGroupComm::MpiMcastGroupComm(int64_t mpiCommFortranHandle)
#if ENABLE_MULTI_DEVICE
    : mComm(MPI_Comm_f2c(mpiCommFortranHandle), false)
#else
    : mComm(nullptr, false)
#endif
{
}

uint32_t MpiMcastGroupComm::getSize() const
{
    return static_cast<uint32_t>(mComm.getSize());
}

uint32_t MpiMcastGroupComm::getRank() const
{
    return static_cast<uint32_t>(mComm.getRank());
}

int MpiMcastGroupComm::getWorldRank() const
{
    return mpi::MpiComm::session().getRank();
}

void MpiMcastGroupComm::allgather(void const* sendBuf, void* recvBuf, size_t bytes)
{
    mComm.allgather(sendBuf, recvBuf, static_cast<int>(bytes), mpi::MpiType::kCHAR);
}

void MpiMcastGroupComm::bcast(void* buf, size_t bytes, int root)
{
    mComm.bcast(buf, bytes, mpi::MpiType::kCHAR, root);
}

void MpiMcastGroupComm::barrier()
{
    mComm.barrier();
}

std::vector<int> MpiMcastGroupComm::getWorldRanks() const
{
    return tensorrt_llm::mpi::getWorldRanks(mComm);
}

} // namespace tensorrt_llm::runtime

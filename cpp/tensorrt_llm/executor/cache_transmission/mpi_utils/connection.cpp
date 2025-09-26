/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace tensorrt_llm::executor::kv_cache
{

MpiConnection::MpiConnection(mpi::MpiComm const* comm, int rank)
    : mComm{comm}
    , mRank{rank}
{
}

void MpiConnection::send(DataContext const& ctx, void const* data, size_t size) const
{
    mComm->sendRawTag(data, size, mpi::MpiType::kCHAR, mRank, ctx.getTag());
}

void MpiConnection::recv(DataContext const& ctx, void* data, size_t size) const
{
    mComm->recvRawTag(data, size, mpi::MpiType::kCHAR, mRank, ctx.getTag());
}

MpiConnectionManager::MpiConnectionManager(mpi::MpiComm const* comm)
    : mComm{comm}
{
    TLLM_CHECK(mComm);
    mCommState = CommState{
        tensorrt_llm::mpi::getWorldRanks(tensorrt_llm::mpi::MpiComm::session()), mpi::MpiComm::session().getRank()};
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "MpiConnectionManager::MpiConnectionManager, commState:%s",
        mCommState.toString().c_str());
}

MpiConnection const* MpiConnectionManager::recvConnect(DataContext const& ctx, void* data, size_t size)
{
#if ENABLE_MULTI_DEVICE
    MPI_Status status;
    MPI_Recv(data, size, MPI_CHAR, MPI_ANY_SOURCE, ctx.getTag(), static_cast<MPI_Comm>(*mComm), std::addressof(status));
    auto&& [it, success] = mConnections.insert({status.MPI_SOURCE, MpiConnection{mComm, status.MPI_SOURCE}});
    return std::addressof(it->second);
#else
    TLLM_THROW("Multi device support is disabled.");
#endif
}

std::vector<Connection const*> MpiConnectionManager::getConnections(CommState const& state)
{
    std::vector<Connection const*> ret;
    TLLM_CHECK(state.isMpiState());
    for (auto rank : state.getMpiState().mRanks)
    {
        auto&& [it, success] = mConnections.insert({rank, MpiConnection{mComm, rank}});
        ret.emplace_back(&it->second);
    }
    TLLM_CHECK(!ret.empty());
    return ret;
}

CommState const& MpiConnectionManager::getCommState() const
{
    return mCommState;
}

} // namespace tensorrt_llm::executor::kv_cache

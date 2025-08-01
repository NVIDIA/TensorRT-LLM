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

#pragma once

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace tensorrt_llm::executor::kv_cache
{

class MpiConnection : public Connection
{
public:
    MpiConnection(mpi::MpiComm const* comm, int rank);
    void send(DataContext const& ctx, void const* data, size_t size) const override;
    void recv(DataContext const& ctx, void* data, size_t size) const override;

private:
    mpi::MpiComm const* mComm{};
    int mRank{-1};
};

class MpiConnectionManager : public ConnectionManager
{
public:
    MpiConnectionManager(mpi::MpiComm const* comm);
    MpiConnection const* recvConnect(DataContext const& ctx, void* data, size_t size) override;
    [[nodiscard]] std::vector<Connection const*> getConnections(CommState const& state) override;
    [[nodiscard]] CommState const& getCommState() const override;

private:
    mpi::MpiComm const* mComm;
    std::map<int, MpiConnection> mConnections;
    CommState mCommState;
};

} // namespace tensorrt_llm::executor::kv_cache

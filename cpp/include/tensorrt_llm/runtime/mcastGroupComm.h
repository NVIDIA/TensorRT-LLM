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
#pragma once

#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensorrt_llm::runtime
{

//! \brief Host-side collective interface used to exchange CUDA memory handles while setting up
//! multicast memory.
//!
//! Both multicast allocation paths need a few host collectives to share memory handles: the
//! multi-node path allgathers unicast fabric handles and broadcasts the multicast handle, and the
//! intra-node NVLS path broadcasts handles (or file descriptors) one rank at a time. Abstracting
//! them lets the same allocation logic run under an MPI job (mpi4py communicator) or under an
//! orchestrator that only provides a PyTorch ProcessGroup, such as Ray.
class McastGroupComm
{
public:
    virtual ~McastGroupComm() = default;

    [[nodiscard]] virtual uint32_t getSize() const = 0;
    [[nodiscard]] virtual uint32_t getRank() const = 0;

    //! Allgather \p bytes bytes from \p sendBuf into \p recvBuf, which must hold bytes * getSize().
    virtual void allgather(void const* sendBuf, void* recvBuf, size_t bytes) = 0;

    //! Broadcast \p bytes bytes of \p buf from \p root to every rank of the group.
    virtual void bcast(void* buf, size_t bytes, int root) = 0;

    //! Block until every rank of the group has arrived.
    virtual void barrier() = 0;

    //! This process's globally unique id. Cheap and local -- unlike getWorldRanks() it involves no
    //! collective, so it is safe to call from a log statement.
    [[nodiscard]] virtual int getWorldRank() const = 0;

    //! Globally unique ids of the group members, ordered by group rank. The NVLS path uses them to
    //! name the per-process IPC sockets it exchanges file descriptors over, so they only need to be
    //! unique and consistent across the job -- MPI session ranks and torch.distributed global ranks
    //! both qualify.
    [[nodiscard]] virtual std::vector<int> getWorldRanks() const = 0;

    //! Whether this backend is MPI-based.
    [[nodiscard]] virtual bool isMpi() const = 0;
};

//! \brief McastGroupComm backed by an MPI communicator.
class MpiMcastGroupComm : public McastGroupComm
{
public:
    //! \param mpiCommFortranHandle Fortran handle of the group communicator (from Python mpi4py).
    explicit MpiMcastGroupComm(int64_t mpiCommFortranHandle);

    [[nodiscard]] uint32_t getSize() const override;
    [[nodiscard]] uint32_t getRank() const override;
    [[nodiscard]] int getWorldRank() const override;
    void allgather(void const* sendBuf, void* recvBuf, size_t bytes) override;
    void bcast(void* buf, size_t bytes, int root) override;
    void barrier() override;
    [[nodiscard]] std::vector<int> getWorldRanks() const override;

    [[nodiscard]] bool isMpi() const override
    {
        return true;
    }

private:
    tensorrt_llm::mpi::MpiComm mComm;
};

} // namespace tensorrt_llm::runtime

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

#include "tensorrt_llm/runtime/mcastGroupComm.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"

#include <exception>
#include <utility>

namespace tensorrt_llm::runtime
{

//! \brief McastGroupComm backed by a PyTorch ProcessGroup.
//!
//! Used by orchestrators that do not provide an MPI communicator (Ray).
//!
//! The payloads are CUDA memory handles, which live in host memory, so the group must have a CPU
//! backend -- TensorRT-LLM's Ray workers build theirs with "cuda:nccl,cpu:gloo". Handles are then
//! exchanged in place over that backend: the collective reads and writes the caller's buffers
//! directly, with no device round-trip and no copies. The exchange happens only when a workspace
//! is created or grown, never on the steady-state path.
//!
//! This header pulls in torch headers and must only be included from translation units that link
//! against torch (the nanobind bindings and thop).
class PgMcastGroupComm : public McastGroupComm
{
public:
    explicit PgMcastGroupComm(c10::intrusive_ptr<c10d::ProcessGroup> pg)
        : mPg(std::move(pg))
        , mHasHostBackend(probeHostBackend(mPg))
    {
        TLLM_CHECK_WITH_INFO(mPg != nullptr, "[PgMcastGroupComm] process group must not be null.");
        TLLM_CHECK_WITH_INFO(mHasHostBackend,
            "[PgMcastGroupComm] the multicast handle exchange needs a process group with a CPU backend, but this "
            "group has none. Build it with a backend string that covers the host, e.g. "
            "init_process_group(backend=\"cuda:nccl,cpu:gloo\").");
    }

    [[nodiscard]] uint32_t getSize() const override
    {
        return static_cast<uint32_t>(mPg->getSize());
    }

    [[nodiscard]] uint32_t getRank() const override
    {
        return static_cast<uint32_t>(mPg->getRank());
    }

    void allgather(void const* sendBuf, void* recvBuf, size_t bytes) override
    {
        doAllgather(sendBuf, recvBuf, bytes);
    }

    void bcast(void* buf, size_t bytes, int root) override
    {
        c10d::BroadcastOptions options;
        options.rootRank = root;
        // Broadcasts in place, so the caller's buffer holds the result.
        std::vector<at::Tensor> payloads{hostView(buf, static_cast<int64_t>(bytes))};
        PGCHECK_THROW(mPg->broadcast(payloads, options));
    }

    void barrier() override
    {
        PGCHECK_THROW(mPg->barrier());
    }

    //! Taken from the world process group when the runtime registered one (TorchDist always does).
    //! Falls back to the group rank, which is still unique within the group that shares an IPC
    //! socket namespace.
    [[nodiscard]] int getWorldRank() const override
    {
        auto worldPg = tensorrt_llm::pg_utils::get_world_pg();
        return worldPg ? worldPg->getRank() : mPg->getRank();
    }

    [[nodiscard]] std::vector<int> getWorldRanks() const override
    {
        int const localId = getWorldRank();
        std::vector<int> ids(mPg->getSize());
        doAllgather(&localId, ids.data(), sizeof(int));
        return ids;
    }

    [[nodiscard]] bool isMpi() const override
    {
        return false;
    }

private:
    //! Gathers straight into the caller's buffer.
    void doAllgather(void const* sendBuf, void* recvBuf, size_t bytes) const
    {
        auto const count = static_cast<int64_t>(bytes);
        auto input = hostView(const_cast<void*>(sendBuf), count);
        auto output = hostView(recvBuf, count * mPg->getSize());
        PGCHECK_THROW(mPg->_allgather_base(output, input));
    }

    //! c10d offers no query for "does this group have a CPU backend", and getBackend() reports the
    //! absence by throwing, so probe once at construction to turn it into a pointed error.
    static bool probeHostBackend(c10::intrusive_ptr<c10d::ProcessGroup> const& pg)
    {
        if (pg == nullptr)
        {
            return false;
        }
        try
        {
            return pg->getBackend(c10::DeviceType::CPU) != nullptr;
        }
        catch (std::exception const&)
        {
            return false;
        }
    }

    static at::Tensor hostView(void* data, int64_t bytes)
    {
        return at::from_blob(data, {bytes}, at::TensorOptions{}.dtype(torch::kChar));
    }

    c10::intrusive_ptr<c10d::ProcessGroup> mPg;
    bool mHasHostBackend;
};

} // namespace tensorrt_llm::runtime

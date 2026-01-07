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

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#include "tensorrt_llm/executor/transferAgent.h"
#include "transfer_engine_c.h"

namespace tensorrt_llm::executor::kv_cache
{

class MooncakeTransferStatus final : public TransferStatus
{
public:
    MooncakeTransferStatus(transfer_engine_t engine, uint64_t batchId, size_t requestCount);

    [[nodiscard]] bool isCompleted() const override;

    TransferState wait(int64_t timeout_ms = -1) const override;

private:
    transfer_engine_t mEngine;
    uint64_t mBatchId;
    size_t mRequestCount;
    mutable bool mBatchFreed = false;
};

class MooncakeMemoryDesc
{
public:
    MooncakeMemoryDesc(MemoryDesc desc)
        : mDesc{std::move(desc)}
        , mRefCnt{0}
    {
    }

    MooncakeMemoryDesc(MooncakeMemoryDesc const& other)
        : mDesc{other.mDesc}
        , mRefCnt{0}
    {
    }

    MooncakeMemoryDesc& operator=(MooncakeMemoryDesc const&) = delete;

    ~MooncakeMemoryDesc() = default;

    void addRef() noexcept
    {
        ++mRefCnt;
    }

    int releaseRef() noexcept
    {
        return --mRefCnt;
    }

    int getRefCount() const noexcept
    {
        return mRefCnt;
    }

    MemoryDesc const& getDesc() const noexcept
    {
        return mDesc;
    }

private:
    MemoryDesc mDesc;
    int mRefCnt;
};

class MooncakeBase64Helper
{
public:
    static std::string encode(std::vector<uint8_t> const& data);
    static std::string encode(std::string const& data);

    static std::vector<uint8_t> decode(std::string const& encoded);
    static std::string decodeToString(std::string const& encoded);

private:
    static const std::string STANDARD_CHARS;

    static std::string encodeInternal(std::vector<uint8_t> const& data, std::string const& chars);
    static std::vector<uint8_t> decodeInternal(std::string const& encoded, std::string const& chars);

    static inline bool isBase64(uint8_t c, std::string const& chars);
    static inline bool isWhitespace(uint8_t c);
};

class MooncakeTransferAgent final : public BaseTransferAgent
{
public:
    MooncakeTransferAgent(BaseAgentConfig const& config);
    ~MooncakeTransferAgent();

    void registerMemory(RegisterDescs const& descs) override;

    void deregisterMemory(RegisterDescs const& descs) override;

    void loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc) override;

    void loadRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo) override;

    void invalidateRemoteAgent(std::string const& name) override;

    AgentDesc getLocalAgentDesc() override;

    ConnectionInfoType getLocalConnectionInfo() override;

    [[nodiscard]] std::unique_ptr<TransferStatus> submitTransferRequests(TransferRequest const& request) override;

    void notifySyncMessage(std::string const& name, SyncMessage const& syncMessage) override;

    [[nodiscard]] std::unordered_map<std::string, std::vector<SyncMessage>> getNotifiedSyncMessages() override;

    bool checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs) override;

private:
    struct AgentInfo
    {
        int segmentId;
    };

    mutable std::mutex mMutex;
    transfer_engine_t mEngine;
    std::unordered_map<uintptr_t, std::shared_ptr<MooncakeMemoryDesc>> mMemRegInfo;
    std::unordered_map<std::string, AgentInfo> mConnectedAgents;
    std::string mLocalAgentName;
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    [[nodiscard]] std::unique_ptr<BaseTransferAgent> createMooncakeTransferAgent(BaseAgentConfig const* config);
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache

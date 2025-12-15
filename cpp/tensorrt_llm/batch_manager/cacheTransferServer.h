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

#include "tensorrt_llm/batch_manager/llmRequest.h"

#include <zmq.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

namespace tensorrt_llm::batch_manager
{

using RequestIdType = LlmRequest::RequestIdType;
// UUID is a 36-character string in the format of "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".
static constexpr size_t kUuidSize = 36;
using UuidType = std::array<char, kUuidSize>;
// Type alias for unique transfer IDs used in KV cache transfer operations.
using TransferTagType = int;
static constexpr TransferTagType kInvalidTransferTag = 0;

class TransferTagGenerator
{
public:
    static uint64_t get()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mReleasedIds.empty())
        {
            uint64_t id = *mReleasedIds.begin();
            mReleasedIds.erase(mReleasedIds.begin());
            return id;
        }
        return mNextId++;
    }

    static void release(uint64_t id)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (id < mNextId)
        {
            mReleasedIds.insert(id);
        }
    }

private:
    static uint64_t mNextId;
    static std::set<uint64_t> mReleasedIds;
    static std::mutex mMutex;
};

inline std::mutex TransferTagGenerator::mMutex;
// 8192 is chosen to avoid conflicts with other MPI tags.
inline uint64_t TransferTagGenerator::mNextId = 8192;
inline std::set<uint64_t> TransferTagGenerator::mReleasedIds;

enum class TransferTagRequestType
{
    kGetTransferTag,
    kReleaseTransferTag
};

struct TransferTagRequest
{
    TransferTagRequestType type;

    union
    {
        struct
        {
            RequestIdType receiverTransferId;
            UuidType receiverServerUuid;
            int32_t expectedRefCount;
        } getTransferTag;

        struct
        {
            RequestIdType receiverTransferId;
            UuidType receiverServerUuid;
            uint64_t transferTag;
        } releaseTransferTag;
    } payload;
};

class TransferTagServer
{
public:
    TransferTagServer();
    ~TransferTagServer();

    TransferTagServer(TransferTagServer const&) = delete;
    TransferTagServer& operator=(TransferTagServer const&) = delete;
    TransferTagServer(TransferTagServer&&) = delete;
    TransferTagServer& operator=(TransferTagServer&&) = delete;

    void waitForReady()
    {
        std::unique_lock<std::mutex> lock(mReadyMutex);
        mReadyCv.wait(lock, [this] { return mReady; });
    }

    std::string getEndpoint() const
    {
        return mEndpoint;
    }

    void stop()
    {
        mRunning = false;
    }

private:
    void loop();
    void handleRequest();

    std::unique_ptr<zmq::context_t> mContext;
    std::unique_ptr<zmq::socket_t> mSocket;
    std::map<std::pair<RequestIdType, UuidType>, std::pair<int32_t, uint64_t>> mTransferTagRefCount;
    std::string mEndpoint;
    std::atomic<bool> mRunning{true};
    std::thread mThread;

    std::mutex mReadyMutex;
    std::condition_variable mReadyCv;
    bool mReady{false};
};

} // namespace tensorrt_llm::batch_manager

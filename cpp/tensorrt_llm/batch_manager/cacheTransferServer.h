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
static constexpr size_t kUuidSize = 36;
using UuidType = std::array<char, kUuidSize>;

class UniqueIdGenerator
{
public:
    static int32_t get()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mReleasedIds.empty())
        {
            int32_t id = *mReleasedIds.begin();
            mReleasedIds.erase(mReleasedIds.begin());
            return id;
        }
        return mNextId++;
    }

    static void release(int32_t id)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (id < mNextId)
        {
            mReleasedIds.insert(id);
        }
    }

private:
    static int32_t mNextId;
    static std::set<int32_t> mReleasedIds;
    static std::mutex mMutex;
};

inline std::mutex UniqueIdGenerator::mMutex;
inline int32_t UniqueIdGenerator::mNextId = 8192;
inline std::set<int32_t> UniqueIdGenerator::mReleasedIds;

enum class CacheTransferRequestType
{
    kGetUniqueId,
    kReleaseUniqueId
};

struct CacheTransferRequest
{
    CacheTransferRequestType type;

    union
    {
        struct
        {
            RequestIdType requestId;
            UuidType serverUuid;
            int32_t expectedRefCount;
        } getUniqueId;

        struct
        {
            RequestIdType requestId;
            UuidType serverUuid;
            int32_t uniqueId;
        } releaseUniqueId;
    } payload;
};

class CacheTransferServer
{
public:
    CacheTransferServer();
    ~CacheTransferServer();

    CacheTransferServer(CacheTransferServer const&) = delete;
    CacheTransferServer& operator=(CacheTransferServer const&) = delete;
    CacheTransferServer(CacheTransferServer&&) = delete;
    CacheTransferServer& operator=(CacheTransferServer&&) = delete;

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
    std::map<std::pair<RequestIdType, UuidType>, std::pair<int32_t, int32_t>> mUniqueIdRefCount;
    std::string mEndpoint;
    std::atomic<bool> mRunning{true};
    std::thread mThread;

    std::mutex mReadyMutex;
    std::condition_variable mReadyCv;
    bool mReady{false};
};

} // namespace tensorrt_llm::batch_manager

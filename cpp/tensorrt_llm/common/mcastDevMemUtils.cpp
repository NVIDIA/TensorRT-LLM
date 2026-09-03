/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "mcastDevMemUtils.h"
#include "tensorrt_llm/common/config.h"

#include <memory>
#include <mutex>
#include <unordered_map>

using McastDeviceMemory = ::tensorrt_llm::runtime::McastDeviceMemory;

TRTLLM_NAMESPACE_BEGIN

namespace common
{

namespace
{
class McastDevMemBufferRegistry
{
public:
    McastDevMemBufferRegistry(McastDevMemBufferRegistry const&) = delete;
    McastDevMemBufferRegistry& operator=(McastDevMemBufferRegistry const&) = delete;

    static McastDevMemBufferRegistry& getInstance()
    {
        static auto* instance = new McastDevMemBufferRegistry;
        return *instance;
    }

    void registerBuffer(void* ptr, std::shared_ptr<McastDeviceMemory> const& buf)
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        mPtrToBuffer[ptr] = buf;
    }

    void unregisterBuffer(McastDeviceMemory* buf)
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        // Potential performance issue! Can use erase-if when we adopt C++20
        // Remove mappings in the table
        for (auto it = mPtrToBuffer.begin(); it != mPtrToBuffer.end();)
        {
            auto const owner = it->second.lock();
            if (owner == nullptr || owner.get() == buf)
            {
                it = mPtrToBuffer.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    std::shared_ptr<McastDeviceMemory> findBuffer(void* ptr)
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        auto const it = mPtrToBuffer.find(ptr);
        if (it == mPtrToBuffer.end())
        {
            return nullptr;
        }

        auto owner = it->second.lock();
        if (owner == nullptr)
        {
            mPtrToBuffer.erase(it);
        }
        return owner;
    }

private:
    McastDevMemBufferRegistry() = default;
    ~McastDevMemBufferRegistry() = default;

    std::mutex mMutex;
    std::unordered_map<void*, std::weak_ptr<McastDeviceMemory>> mPtrToBuffer;
};
} // namespace

void registerMcastDevMemBuffer(void* ptr, std::shared_ptr<McastDeviceMemory> const& buf)
{
    McastDevMemBufferRegistry::getInstance().registerBuffer(ptr, buf);
}

void unregisterMcastDevMemBuffer(McastDeviceMemory* buf)
{
    McastDevMemBufferRegistry::getInstance().unregisterBuffer(buf);
}

std::shared_ptr<McastDeviceMemory> findMcastDevMemBuffer(void* ptr)
{
    return McastDevMemBufferRegistry::getInstance().findBuffer(ptr);
}
} // namespace common

TRTLLM_NAMESPACE_END

/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#include <unordered_map>

namespace tensorrt_llm::common
{
using McastDeviceMemory = tensorrt_llm::runtime::McastDeviceMemory;

namespace
{
class McastDevMemBufferRegistry
{
public:
    McastDevMemBufferRegistry(McastDevMemBufferRegistry const&) = delete;
    McastDevMemBufferRegistry& operator=(McastDevMemBufferRegistry const&) = delete;

    static McastDevMemBufferRegistry& getInstance()
    {
        static McastDevMemBufferRegistry instance;
        return instance;
    }

    void registerBuffer(void* ptr, McastDeviceMemory* buf)
    {
        _ptr_to_buf[ptr] = buf;
    }

    void unregisterBuffer(McastDeviceMemory* buf)
    {
        // Potential performance issue! Can use erase-if when we adopt C++20
        // Remove mappings in the table
        for (auto it = _ptr_to_buf.begin(); it != _ptr_to_buf.end();)
        {
            if (it->second == buf)
            {
                it = _ptr_to_buf.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    McastDeviceMemory* findBuffer(void* ptr)
    {
        auto it = _ptr_to_buf.find(ptr);
        return it == _ptr_to_buf.end() ? nullptr : it->second;
    }

private:
    McastDevMemBufferRegistry() = default;
    ~McastDevMemBufferRegistry() = default;

    std::unordered_map<void*, McastDeviceMemory*> _ptr_to_buf;
};
} // namespace

void registerMcastDevMemBuffer(void* ptr, McastDeviceMemory* buf)
{
    McastDevMemBufferRegistry::getInstance().registerBuffer(ptr, buf);
}

void unregisterMcastDevMemBuffer(McastDeviceMemory* buf)
{
    McastDevMemBufferRegistry::getInstance().unregisterBuffer(buf);
}

McastDeviceMemory* findMcastDevMemBuffer(void* ptr)
{
    return McastDevMemBufferRegistry::getInstance().findBuffer(ptr);
}
} // namespace tensorrt_llm::common

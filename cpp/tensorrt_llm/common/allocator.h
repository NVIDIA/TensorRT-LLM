/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
/**
 * Memory Allocator
 **/

#pragma once

#include <cassert>
#include <string>

#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm
{
namespace common
{

enum class ReallocType
{
    INCREASE,
    REUSE,
    DECREASE,
};

class IAllocator
{
public:
    virtual ~IAllocator() = default;

    // no copying
    IAllocator(const IAllocator&) = delete;
    IAllocator& operator=(const IAllocator&) = delete;

    template <typename T>
    [[nodiscard]] T* reMalloc(T* ptr, size_t sizeBytes, const bool setZero = true)
    {
        TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
        // TODO martinma: why do we need this size extension?
        auto const sizeAligned = ((sizeBytes + 31) / 32) * 32; // make the buffer align with 32 bytes
        if (contains(ptr))
        {
            auto const realloc = reallocType(ptr, sizeAligned);
            if (realloc == ReallocType::INCREASE)
            {
                TLLM_LOG_DEBUG("ReMalloc the buffer %p since it is too small.", ptr);
                free(&ptr);
                return reinterpret_cast<T*>(malloc(sizeAligned, setZero));
            }
            else if (realloc == ReallocType::DECREASE)
            {
                TLLM_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to memory pools.", ptr);
                free(&ptr);
                return reinterpret_cast<T*>(malloc(sizeAligned, setZero));
            }
            else
            {
                assert(realloc == ReallocType::REUSE);
                TLLM_LOG_DEBUG("Reuse original buffer %p with size %d and do nothing for reMalloc.", ptr, sizeAligned);
                if (setZero)
                {
                    memSet(ptr, 0, sizeAligned);
                }
                return ptr;
            }
        }
        else
        {
            TLLM_LOG_DEBUG("Cannot find buffer %p, mallocing new one.", ptr);
            return reinterpret_cast<T*>(malloc(sizeAligned, setZero));
        }
    }

    virtual void free(void** ptr) = 0;

    template <typename T>
    void free(T** ptr)
    {
        free(reinterpret_cast<void**>(ptr));
    }

protected:
    IAllocator() = default;

    virtual void* malloc(std::size_t size, bool setZero) = 0;
    virtual void memSet(void* ptr, int val, std::size_t size) = 0;

    virtual bool contains(void const* ptr) const = 0;
    virtual ReallocType reallocType(void const* ptr, size_t size) const = 0;
};

} // namespace common

} // namespace tensorrt_llm

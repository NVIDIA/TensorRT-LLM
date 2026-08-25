/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "kv_cache_manager_v2/utils/sharedPtr.h"

#include "tensorrt_llm/common/logger.h"

#include <cuda.h>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

template <typename F>
void terminateOnException(char const* context, F&& func) noexcept
{
    try
    {
        std::forward<F>(func)();
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("%s: %s", context, error.what());
        std::terminate();
    }
    catch (...)
    {
        TLLM_LOG_ERROR("%s: unknown error", context);
        std::terminate();
    }
}

// ---------------------------------------------------------------------------
// Exception hierarchy (mirrors _exceptions.py)
// ---------------------------------------------------------------------------

// A retryable failure to obtain cache capacity. OutOfPagesError describes
// exhausted cache slots, while OutOfMemoryError describes allocation failure
// in an underlying memory tier.
class CacheCapacityError : public std::runtime_error
{
public:
    explicit CacheCapacityError(std::string const& msg = "Cache capacity exhausted")
        : std::runtime_error(msg)
    {
    }
};

class OutOfPagesError : public CacheCapacityError
{
public:
    explicit OutOfPagesError(std::string const& msg = "Out of pages")
        : CacheCapacityError(msg)
    {
    }
};

class OutOfMemoryError : public CacheCapacityError
{
public:
    explicit OutOfMemoryError(std::string const& msg = "Out of memory")
        : CacheCapacityError(msg)
    {
    }
};

class HostOOMError : public OutOfMemoryError
{
public:
    explicit HostOOMError(std::string const& msg = "Host out of memory")
        : OutOfMemoryError(msg)
    {
    }
};

class DiskOOMError : public OutOfMemoryError
{
public:
    explicit DiskOOMError(std::string const& msg = "Disk out of memory")
        : OutOfMemoryError(msg)
    {
    }
};

class CuOOMError : public OutOfMemoryError
{
public:
    explicit CuOOMError(std::string const& msg = "CUDA out of memory")
        : OutOfMemoryError(msg)
    {
    }
};

// Indicates a bug in the KV cache manager code.
class LogicError : public std::logic_error
{
public:
    explicit LogicError(std::string const& msg)
        : std::logic_error(msg)
    {
    }
};

// Mirrors a Python `assert` failure: the binding layer translates this to a
// Python AssertionError so shared tests observe the same exception type as the
// pure-Python backend.
class AssertionError : public std::logic_error
{
public:
    explicit AssertionError(std::string const& msg)
        : std::logic_error(msg)
    {
    }
};

// Wraps a CUDA driver API error (CUresult).
class CuError : public std::runtime_error
{
public:
    CUresult errorCode;

    explicit CuError(CUresult result)
        : std::runtime_error(makeMessage(result))
        , errorCode(result)
    {
    }

private:
    static std::string makeMessage(CUresult result)
    {
        char const* errStr = nullptr;
        cuGetErrorString(result, &errStr);
        std::string msg = "CUDA driver error: ";
        msg += errStr ? errStr : "<unknown>";
        return msg;
    }
};

// A resource (e.g., a page lock) is still in use.
class ResourceBusyError : public std::runtime_error
{
public:
    explicit ResourceBusyError(std::string const& msg = "Resource is busy")
        : std::runtime_error(msg)
    {
    }
};

// ---------------------------------------------------------------------------
// Helper: unwrap a weak_ptr, throw LogicError on dangling reference.
// Mirrors Python's unwrap_rawref(_utils.py:163).
// ---------------------------------------------------------------------------
template <typename T>
SharedPtr<T> unwrap(WeakPtr<T> const& ref)
{
    auto ptr = ref.lock();
    if (!ptr)
        throw LogicError("Dereferencing a dangling weak_ptr");
    return ptr;
}

// ---------------------------------------------------------------------------
// Helper: unwrap CUresult, throw CuError/CuOOMError on failure.
// ---------------------------------------------------------------------------
inline void cuCheck(CUresult result)
{
    if (result == CUDA_SUCCESS)
    {
        return;
    }
    if (result == CUDA_ERROR_OUT_OF_MEMORY)
    {
        throw CuOOMError();
    }
    throw CuError(result);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2

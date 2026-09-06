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

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/utils/sharedPtr.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <cuda.h>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// The diagnostic is written to native stderr. pytest's default capture mode (--capture=fd)
// redirects that file descriptor and discards the buffer when the process aborts, so a failure
// under pytest is visible only when the suite runs with --capture=tee-sys or --capture=no.
template <typename F>
void abortOnExcept(char const* context, F&& func) noexcept
{
    try
    {
        std::forward<F>(func)();
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("%s: %s", context, error.what());
        std::abort();
    }
    catch (...)
    {
        TLLM_LOG_ERROR("%s: unknown error", context);
        std::abort();
    }
}

template <typename F>
void logOnExcept(char const* context, F&& func) noexcept
{
    try
    {
        std::forward<F>(func)();
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("%s: %s", context, error.what());
    }
    catch (...)
    {
        TLLM_LOG_ERROR("%s: unknown error", context);
    }
}

// ---------------------------------------------------------------------------
// Exception hierarchy (mirrors _exceptions.py)
// ---------------------------------------------------------------------------

class OutOfMemoryError : public std::runtime_error
{
public:
    explicit OutOfMemoryError(std::string const& msg = "Out of memory")
        : std::runtime_error(msg)
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
        if (cuGetErrorString(result, &errStr) != CUDA_SUCCESS || errStr == nullptr)
        {
            errStr = "<Failed to get error string with cuGetErrorString>";
        }
        return "CUDA driver error: " + std::to_string(static_cast<int>(result)) + " (" + errStr + ")";
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

// Not enough free pages to satisfy an allocation request.
class OutOfPagesError : public std::runtime_error
{
public:
    explicit OutOfPagesError(std::string const& msg = "Out of pages")
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

// Runs a callable and, if it throws, logs the error prefixed with the enclosing function, then
// aborts. Variadic so the callable may contain commas.
#define KVCM2_ABORT_ON_EXCEPT(...)                                                                                     \
    ::tensorrt_llm::batch_manager::kv_cache_manager_v2::abortOnExcept(__PRETTY_FUNCTION__, __VA_ARGS__)

// Runs a callable and, if it throws, logs the error prefixed with the enclosing function and
// returns normally. The remainder of the callable is skipped. Variadic so it may contain commas.
#define KVCM2_LOG_ON_EXCEPT(...)                                                                                       \
    ::tensorrt_llm::batch_manager::kv_cache_manager_v2::logOnExcept(__PRETTY_FUNCTION__, __VA_ARGS__)

// Check variants for noexcept contexts such as destructors: on failure they log the assertion
// message, source location and backtrace captured by TLLM_CHECK, prefixed with the enclosing
// function, then abort.
#define KVCM2_CHECK_FATAL(cond)                                                                                        \
    ::tensorrt_llm::batch_manager::kv_cache_manager_v2::abortOnExcept(__PRETTY_FUNCTION__, [&]() { TLLM_CHECK(cond); })

#define KVCM2_CHECK_FATAL_WITH_INFO(cond, info, ...)                                                                   \
    ::tensorrt_llm::batch_manager::kv_cache_manager_v2::abortOnExcept(                                                 \
        __PRETTY_FUNCTION__, [&]() { TLLM_CHECK_WITH_INFO(cond, info, ##__VA_ARGS__); })

// As above, but only evaluated when debug checks are enabled. The gDebug test comes first so
// nothing is built when checks are off.
#define KVCM2_CHECK_FATAL_DEBUG(cond)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (TLLM_UNLIKELY(::tensorrt_llm::batch_manager::kv_cache_manager_v2::gDebug))                                 \
        {                                                                                                              \
            KVCM2_CHECK_FATAL(cond);                                                                                   \
        }                                                                                                              \
    } while (0)

#define KVCM2_CHECK_FATAL_DEBUG_WITH_INFO(cond, info, ...)                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        if (TLLM_UNLIKELY(::tensorrt_llm::batch_manager::kv_cache_manager_v2::gDebug))                                 \
        {                                                                                                              \
            KVCM2_CHECK_FATAL_WITH_INFO(cond, info, ##__VA_ARGS__);                                                    \
        }                                                                                                              \
    } while (0)

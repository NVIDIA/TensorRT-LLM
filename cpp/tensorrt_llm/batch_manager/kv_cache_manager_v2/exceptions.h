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

#include <cuda.h>
#include <stdexcept>
#include <string>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

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

// Not enough free pages to satisfy an allocation request.
class OutOfPagesError : public std::runtime_error
{
public:
    explicit OutOfPagesError(std::string const& msg = "Out of pages")
        : std::runtime_error(msg)
    {
    }
};

// Block creation rejected because its tokens are fully covered by an existing sibling.
// Mirrors Python's UselessBlockError — carries the sibling block.
// TODO: Once Python is removed and C++ becomes the primary development target,
// replace this exception-based flow with a simple if-condition return in
// addOrGetExistingBlock (returning the sibling block directly instead of throwing).
// The exception pattern exists only to maintain parity with the Python code path.
// Forward-declared; Block definition is in blockRadixTree.h.
struct Block;

class UselessBlockError : public std::runtime_error
{
public:
    SharedPtr<Block> block;

    explicit UselessBlockError(SharedPtr<Block> blk)
        : std::runtime_error("Block is useless — covered by existing sibling")
        , block(std::move(blk))
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

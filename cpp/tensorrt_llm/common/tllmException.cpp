/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"

#include <cstdlib>
#if !defined(_MSC_VER)
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#endif
#include <sstream>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

namespace
{
int constexpr VOID_PTR_SZ = 2 + sizeof(void*) * 2;
}

#if !defined(_MSC_VER)

TllmException::TllmException(std::source_location const& location, std::string_view msg)
    : std::runtime_error{""}
{
    mNbFrames = backtrace(mCallstack.data(), MAX_FRAMES);
    auto const trace = getTrace();
    std::runtime_error::operator=(
        std::runtime_error{std::format("{} ({}:{})\n{}", msg, location.file_name(), location.line(), trace)});
}
#else
TllmException::TllmException(std::source_location const& location, std::string_view msg)
    : mNbFrames{}
    , std::runtime_error{std::format("{} ({}:{})", msg, location.file_name(), location.line())}
{
}
#endif

TllmException::~TllmException() noexcept = default;

std::string TllmException::getTrace() const
{
#if defined(_MSC_VER)
    return "";
#else
    auto const trace = std::unique_ptr<char const*, void (*)(char const**)>(
        const_cast<char const**>(backtrace_symbols(mCallstack.data(), mNbFrames)),
        [](char const** p) { std::free(p); });
    if (trace == nullptr)
    {
        throw std::bad_alloc();
    }
    std::ostringstream buf;
    for (auto i = 1; i < mNbFrames; ++i)
    {
        Dl_info info;
        if (dladdr(mCallstack[i], &info) && info.dli_sname)
        {
            auto const clearName = demangle(info.dli_sname);
            buf << std::format("{:<3d} {:>{}p} {} + {}", i, mCallstack[i], VOID_PTR_SZ, clearName,
                static_cast<char*>(mCallstack[i]) - static_cast<char*>(info.dli_saddr));
        }
        else
        {
            buf << std::format("{:<3d} {:>{}p} {}", i, mCallstack[i], VOID_PTR_SZ, trace.get()[i]);
        }
        if (i < mNbFrames - 1)
            buf << std::endl;
    }

    if (mNbFrames == MAX_FRAMES)
        buf << std::endl << "[truncated]";

    return buf.str();
#endif
}

std::string TllmException::demangle(char const* name)
{
#if defined(_MSC_VER)
    return name;
#else
    std::string clearName{name};
    auto status = -1;
    auto const demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0)
    {
        clearName = demangled;
        std::free(demangled);
    }
    return clearName;
#endif
}

RequestSpecificException::RequestSpecificException(
    std::source_location const& location, std::string_view msg, uint64_t requestID, RequestErrorCode errorCode)
    : std::runtime_error{std::format("{} (Request ID: {}, Error Code: {}) ({}:{})", msg, requestID,
        static_cast<uint32_t>(errorCode), location.file_name(), location.line())}
    , mRequestID{requestID}
    , mErrorCode{errorCode}
{
}

RequestSpecificException::~RequestSpecificException() noexcept = default;

uint64_t RequestSpecificException::getRequestId() const noexcept
{
    return mRequestID;
}

RequestErrorCode RequestSpecificException::getErrorCode() const noexcept
{
    return mErrorCode;
}

} // namespace common

TRTLLM_NAMESPACE_END

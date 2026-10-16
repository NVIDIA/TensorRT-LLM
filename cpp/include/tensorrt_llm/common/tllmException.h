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

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#define TLLM_THROW(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        throw NEW_TLLM_EXCEPTION(__VA_ARGS__);                                                                         \
    } while (0)

#define TLLM_WRAP(ex)                                                                                                  \
    NEW_TLLM_EXCEPTION("{}: {}", tensorrt_llm::common::TllmException::demangle(typeid(ex).name()), ex.what())

#define NEW_TLLM_EXCEPTION(...)                                                                                        \
    tensorrt_llm::common::TllmException(std::source_location::current(), std::format(__VA_ARGS__))

#define TLLM_REQUEST_EXCEPTION(requestID, errorCode, ...)                                                              \
    tensorrt_llm::common::RequestSpecificException(                                                                    \
        std::source_location::current(), std::format(__VA_ARGS__), requestID, errorCode)

TRTLLM_NAMESPACE_BEGIN

namespace common
{

/// @brief Enumeration of different error codes for request-specific exceptions
enum class RequestErrorCode : uint32_t
{
    // General errors (0-999)
    kUNKNOWN_ERROR = 0,

    // Network and communication errors (1000-1999)
    kNETWORK_ERROR = 1000,
};

/// @brief Constant for unknown request ID
static constexpr uint64_t kUNKNOWN_REQUEST_ID = std::numeric_limits<uint64_t>::max();

class TllmException : public std::runtime_error
{
public:
    static auto constexpr MAX_FRAMES = 128;

    explicit TllmException(std::source_location const& location, std::string_view msg);

    ~TllmException() noexcept override;

    [[nodiscard]] std::string getTrace() const;

    static std::string demangle(char const* name);

private:
    std::array<void*, MAX_FRAMES> mCallstack{};
    int mNbFrames;
};

//! Throw a TllmException for an assertion that failed at location.
[[noreturn]] inline void throwRuntimeError(std::source_location const& location, std::string_view info)
{
    throw TllmException(location, std::format("[TensorRT-LLM][ERROR] Assertion failed: {}", info));
}

//! Throw a TllmException for an assertion that failed at location, describing it with a std::format message.
template <typename... Args>
[[noreturn]] void throwRuntimeError(
    std::source_location const& location, std::format_string<Args...> format, Args&&... args)
{
    throwRuntimeError(location, std::format(format, std::forward<Args>(args)...));
}

class RequestSpecificException : public std::runtime_error
{
public:
    explicit RequestSpecificException(
        std::source_location const& location, std::string_view msg, uint64_t requestID, RequestErrorCode errorCode);

    ~RequestSpecificException() noexcept override;

    [[nodiscard]] uint64_t getRequestId() const noexcept;

    [[nodiscard]] RequestErrorCode getErrorCode() const noexcept;

private:
    uint64_t mRequestID;
    RequestErrorCode mErrorCode;
};

} // namespace common

TRTLLM_NAMESPACE_END

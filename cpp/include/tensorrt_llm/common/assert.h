/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/tllmException.h"

class DebugConfig
{
public:
    static bool isCheckDebugEnabled();
};

#if defined(_WIN32)
#define TLLM_LIKELY(x) (__assume((x) == 1), (x))
#define TLLM_UNLIKELY(x) (__assume((x) == 0), (x))
#else
#define TLLM_LIKELY(x) __builtin_expect((x), 1)
#define TLLM_UNLIKELY(x) __builtin_expect((x), 0)
#endif

#define TLLM_CHECK(val)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        TLLM_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                               \
                                            : tensorrt_llm::common::throwRuntimeError(__FILE__, __LINE__, #val);       \
    } while (0)

#define TLLM_CHECK_WITH_INFO(val, info, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        TLLM_LIKELY(static_cast<bool>(val))                                                                            \
        ? ((void) 0)                                                                                                   \
        : tensorrt_llm::common::throwRuntimeError(                                                                     \
            __FILE__, __LINE__, tensorrt_llm::common::fmtstr(info, ##__VA_ARGS__).c_str());                            \
    } while (0)

#define TLLM_CHECK_DEBUG(val)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (TLLM_UNLIKELY(DebugConfig::isCheckDebugEnabled()))                                                         \
        {                                                                                                              \
            TLLM_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                           \
                                                : tensorrt_llm::common::throwRuntimeError(__FILE__, __LINE__, #val);   \
        }                                                                                                              \
    } while (0)

#define TLLM_CHECK_DEBUG_WITH_INFO(val, info, ...)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (TLLM_UNLIKELY(DebugConfig::isCheckDebugEnabled()))                                                         \
        {                                                                                                              \
            TLLM_LIKELY(static_cast<bool>(val))                                                                        \
            ? ((void) 0)                                                                                               \
            : tensorrt_llm::common::throwRuntimeError(                                                                 \
                __FILE__, __LINE__, tensorrt_llm::common::fmtstr(info, ##__VA_ARGS__).c_str());                        \
        }                                                                                                              \
    } while (0)

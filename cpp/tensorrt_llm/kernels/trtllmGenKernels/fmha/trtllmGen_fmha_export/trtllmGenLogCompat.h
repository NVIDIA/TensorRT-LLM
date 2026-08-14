/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
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

// Compatibility layer: trtllm-gen export headers use streaming-style log macros
// (e.g., TLLM_LOG_TRACE("val=", x)) while TensorRT-LLM's logger.h uses printf-style
// (e.g., TLLM_LOG_TRACE("val=%d", x)). This header saves the original definitions
// and replaces them with streaming-compatible versions.
//
// Usage in fmhaKernels.h:
//   #include "tensorrt_llm/common/logger.h"                // TRT-LLM printf-style
//   #include "trtllmGen_fmha_export/trtllmGenLogCompat.h"  // switch to streaming
//   #include "trtllmGen_fmha_export/FmhaOptions.h"         // export headers (streaming)
//   #include "trtllmGen_fmha_export/trtllmGenLogCompatEnd.h" // restore printf-style

#include <sstream>
#include "tensorrt_llm/common/logger.h"

namespace trtllm_gen_log_compat
{

template <typename... Args>
inline void logStreaming(tensorrt_llm::common::Logger::Level level, Args const&... args)
{
    auto* const logger = tensorrt_llm::common::Logger::getLogger();
    if (logger->isEnabled(level))
    {
        std::ostringstream oss;
        (oss << ... << args);
        logger->log(level, "%s", oss.str().c_str());
    }
}

} // namespace trtllm_gen_log_compat

// Save the original TRT-LLM log macro definitions and replace with streaming-compatible versions.
#pragma push_macro("TLLM_LOG_TRACE")
#pragma push_macro("TLLM_LOG_INFO")
#pragma push_macro("TLLM_LOG_WARNING")
#pragma push_macro("TLLM_LOG_ERROR")

#undef TLLM_LOG_TRACE
#undef TLLM_LOG_INFO
#undef TLLM_LOG_WARNING
#undef TLLM_LOG_ERROR

#define TLLM_LOG_TRACE(...)                                                                                            \
    trtllm_gen_log_compat::logStreaming(tensorrt_llm::common::Logger::TRACE, __VA_ARGS__)
#define TLLM_LOG_INFO(...)                                                                                             \
    trtllm_gen_log_compat::logStreaming(tensorrt_llm::common::Logger::INFO, __VA_ARGS__)
#define TLLM_LOG_WARNING(...)                                                                                          \
    trtllm_gen_log_compat::logStreaming(tensorrt_llm::common::Logger::WARNING, __VA_ARGS__)
#define TLLM_LOG_ERROR(...)                                                                                            \
    trtllm_gen_log_compat::logStreaming(tensorrt_llm::common::Logger::ERROR, __VA_ARGS__)

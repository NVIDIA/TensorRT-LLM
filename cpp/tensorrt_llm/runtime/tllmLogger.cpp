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
#include "tllmLogger.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

void TllmLogger::log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg) noexcept
{
    switch (severity)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
    case nvinfer1::ILogger::Severity::kERROR: TLLM_LOG_ERROR(msg); break;
    case nvinfer1::ILogger::Severity::kWARNING: TLLM_LOG_WARNING(msg); break;
    case nvinfer1::ILogger::Severity::kINFO: TLLM_LOG_INFO(msg); break;
    case nvinfer1::ILogger::Severity::kVERBOSE: TLLM_LOG_DEBUG(msg); break;
    default: TLLM_LOG_TRACE(msg); break;
    }
}

nvinfer1::ILogger::Severity TllmLogger::getLevel()
{
    auto* const logger = tc::Logger::getLogger();
    switch (logger->getLevel())
    {
    case tc::Logger::Level::ERROR: return nvinfer1::ILogger::Severity::kERROR;
    case tc::Logger::Level::WARNING: return nvinfer1::ILogger::Severity::kWARNING;
    case tc::Logger::Level::INFO: return nvinfer1::ILogger::Severity::kINFO;
    case tc::Logger::Level::DEBUG:
    case tc::Logger::Level::TRACE: return nvinfer1::ILogger::Severity::kVERBOSE;
    default: return nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
    }
}

void TllmLogger::setLevel(nvinfer1::ILogger::Severity level)
{
    auto* const logger = tc::Logger::getLogger();
    switch (level)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
    case nvinfer1::ILogger::Severity::kERROR: logger->setLevel(tc::Logger::Level::ERROR); break;
    case nvinfer1::ILogger::Severity::kWARNING: logger->setLevel(tc::Logger::Level::WARNING); break;
    case nvinfer1::ILogger::Severity::kINFO: logger->setLevel(tc::Logger::Level::INFO); break;
    case nvinfer1::ILogger::Severity::kVERBOSE: logger->setLevel(tc::Logger::Level::TRACE); break;
    default: TLLM_THROW("Unsupported severity");
    }
}

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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/tllmException.h"
#include <algorithm>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

int Logger::detectRank()
{
    // Try environment variables first (most reliable in distributed settings).
    // TLLM_RANK is TRT-LLM specific; the others are set by common launchers
    // (torchrun, mpirun, Slurm, etc.).
    for (auto const* envVar : {"TLLM_RANK", "RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "PMIX_RANK"})
    {
        auto const* val = std::getenv(envVar);
        if (val != nullptr)
        {
            try
            {
                return std::stoi(val);
            }
            catch (...)
            {
                // Invalid value, try next variable.
            }
        }
    }

    return kRANK_UNSET;
}

bool Logger::parseLevelName(std::string_view name, Level& outLevel)
{
    // Make an uppercase copy for case-insensitive comparison.
    std::string upper{name};
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });

    if (upper == "TRACE")
    {
        outLevel = TRACE;
    }
    else if (upper == "VERBOSE" || upper == "DEBUG")
    {
        outLevel = DEBUG;
    }
    else if (upper == "INFO")
    {
        outLevel = INFO;
    }
    else if (upper == "WARNING")
    {
        outLevel = WARNING;
    }
    else if (upper == "ERROR")
    {
        outLevel = ERROR;
    }
    else
    {
        return false;
    }
    return true;
}

std::map<std::string, Logger::Level, std::less<>> Logger::parseModuleLevels(char const* envValue)
{
    // Format: "level:mod1,mod2;level:mod3,mod4"
    // e.g.   "debug:runtime,kernel;info:plugin;warning:layers"
    std::map<std::string, Level, std::less<>> result;
    std::string_view input{envValue};

    while (!input.empty())
    {
        // Split by ';' to get each level group.
        auto const semiPos = input.find(';');
        auto const group = input.substr(0, semiPos);
        input = (semiPos != std::string_view::npos) ? input.substr(semiPos + 1) : std::string_view{};

        if (group.empty())
        {
            continue;
        }

        // Split by ':' to separate level name from module list.
        auto const colonPos = group.find(':');
        if (colonPos == std::string_view::npos || colonPos == 0)
        {
            // Missing colon or empty level — skip this group.
            std::cerr << "[TRT-LLM][WARNING] TLLM_LOG_LEVEL_BY_MODULE: skipping malformed group \"" << group << "\""
                      << std::endl;
            continue;
        }

        auto const levelStr = group.substr(0, colonPos);
        auto const moduleList = group.substr(colonPos + 1);

        Level level{};
        if (!parseLevelName(levelStr, level))
        {
            std::cerr << "[TRT-LLM][WARNING] TLLM_LOG_LEVEL_BY_MODULE: unknown level \"" << levelStr << "\""
                      << std::endl;
            continue;
        }

        // Split module list by ',' and insert each module.
        auto remaining = moduleList;
        while (!remaining.empty())
        {
            auto const commaPos = remaining.find(',');
            auto const mod = remaining.substr(0, commaPos);
            remaining = (commaPos != std::string_view::npos) ? remaining.substr(commaPos + 1) : std::string_view{};

            if (!mod.empty())
            {
                result[std::string(mod)] = level;
            }
        }
    }

    return result;
}

Logger::Logger()
{
    rank_ = detectRank();

    // Parse per-module log level overrides.
    auto const* moduleLevelEnv = std::getenv("TLLM_LOG_LEVEL_BY_MODULE");
    if (moduleLevelEnv != nullptr)
    {
        moduleLevels_ = parseModuleLevels(moduleLevelEnv);
    }

    char* isFirstRankOnlyChar = std::getenv("TLLM_LOG_FIRST_RANK_ONLY");
    bool isFirstRankOnly = (isFirstRankOnlyChar != nullptr && std::string(isFirstRankOnlyChar) == "ON");

    auto const* levelName = std::getenv("TLLM_LOG_LEVEL");
    if (levelName != nullptr)
    {
        Level level{};
        if (!parseLevelName(levelName, level))
        {
            TLLM_THROW("Invalid log level: %s", levelName);
        }
        // If TLLM_LOG_FIRST_RANK_ONLY=ON, set LOG LEVEL of other device to ERROR
        if (isFirstRankOnly)
        {
            auto const deviceId = getDevice();
            if (deviceId != 1)
            {
                level = ERROR;
            }
        }
        setLevel(level);
    }
}

void Logger::log(std::exception const& ex, Logger::Level level)
{
    log(level, "%s: %s", TllmException::demangle(typeid(ex).name()).c_str(), ex.what());
}

Logger* Logger::getLogger()
{
    thread_local Logger instance;
    return &instance;
}
} // namespace common

TRTLLM_NAMESPACE_END

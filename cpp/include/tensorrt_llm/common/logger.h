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

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace common
{

/// Map a raw module directory name to a short display abbreviation.
/// Every known module gets an explicit entry; unknown modules fall through as-is.
/// Display padding to 8 characters is handled by getPrefix() via printf %-8s formatting.
constexpr std::string_view formatModule(std::string_view module)
{
    // Source modules (cpp/tensorrt_llm/*)
    if (module == "batch_manager")
        return "batchmgr";
    else if (module == "common")
        return "common";
    else if (module == "cutlass_extensions")
        return "cutl_ext";
    else if (module == "deep_ep")
        return "deep_ep";
    else if (module == "deep_gemm")
        return "deepgemm";
    else if (module == "executor")
        return "executor";
    else if (module == "executor_worker")
        return "exec_wkr";
    else if (module == "flash_mla")
        return "flashmla";
    else if (module == "kernels")
        return "kernels";
    else if (module == "layers")
        return "layers";
    else if (module == "nanobind")
        return "nanobind";
    else if (module == "plugins")
        return "plugins";
    else if (module == "runtime")
        return "runtime";
    else if (module == "testing")
        return "testing";
    else if (module == "thop")
        return "thop";
    return module;
}

class Logger
{

// On Windows, the file wingdi.h is included which has
// #define ERROR 0
// This breaks everywhere ERROR is used in the Level enum
#ifdef _WIN32
#undef ERROR
#endif // _WIN32

public:
    enum Level
    {
        TRACE = 0,
        DEBUG = 10,
        INFO = 20,
        WARNING = 30,
        ERROR = 40
    };

    /// Tag type that carries a pre-extracted module name.
    /// Used by the TLLM_LOG macros to disambiguate from char const* format strings.
    struct Module
    {
        std::string_view name;
    };

    static Logger* getLogger();

    /// Extract the first subdirectory after the *last* "tensorrt_llm/" from a __FILE__ path.
    /// e.g. "/code/tensorrt_llm/cpp/tensorrt_llm/runtime/foo.cpp" -> "runtime"
    ///      "/code/tensorrt_llm/cpp/include/tensorrt_llm/common/logger.h" -> "common"
    static constexpr std::string_view extractModule(char const* filePath)
    {
        constexpr std::string_view kMarker{"tensorrt_llm/"};
        std::string_view const path{filePath};
        auto const pos = path.rfind(kMarker);
        if (pos != std::string_view::npos)
        {
            auto const moduleStart = pos + kMarker.size();
            auto const slashPos = path.find('/', moduleStart);
            if (slashPos != std::string_view::npos && slashPos > moduleStart)
            {
                return formatModule(path.substr(moduleStart, slashPos - moduleStart));
            }
        }
        return "  others";
    }

    Logger(Logger const&) = delete;
    void operator=(Logger const&) = delete;

    // ---- log overloads WITHOUT source file (used by internal helpers) --------

#if defined(_MSC_VER)
    template <typename... Args>
    void log(Level const level, char const* format, Args const&... args);

    template <typename... Args>
    void log(Level const level, int const rank, char const* format, Args const&... args);
#else
    template <typename... Args>
    void log(Level const level, char const* format, Args const&... args) __attribute__((format(printf, 3, 0)));

    template <typename... Args>
    void log(Level const level, int const rank, char const* format, Args const&... args)
        __attribute__((format(printf, 4, 0)));
#endif

    template <typename... Args>
    void log(Level const level, std::string const& format, Args const&... args)
    {
        return log(level, format.c_str(), args...);
    }

    template <typename... Args>
    void log(Level const level, int const rank, std::string const& format, Args const&... args)
    {
        return log(level, rank, format.c_str(), args...);
    }

    // ---- log overloads WITH module tag (used by TLLM_LOG macros) -------------

#if defined(_MSC_VER)
    template <typename... Args>
    void log(Level const level, Module module, char const* format, Args const&... args);

    template <typename... Args>
    void log(Level const level, Module module, int const rank, char const* format, Args const&... args);
#else
    template <typename... Args>
    void log(Level const level, Module module, char const* format, Args const&... args)
        __attribute__((format(printf, 4, 0)));

    template <typename... Args>
    void log(Level const level, Module module, int const rank, char const* format, Args const&... args)
        __attribute__((format(printf, 5, 0)));
#endif

    template <typename... Args>
    void log(Level const level, Module module, std::string const& format, Args const&... args)
    {
        return log(level, module, format.c_str(), args...);
    }

    template <typename... Args>
    void log(Level const level, Module module, int const rank, std::string const& format, Args const&... args)
    {
        return log(level, module, rank, format.c_str(), args...);
    }

    // ---- exception logging --------------------------------------------------

    void log(std::exception const& ex, Level level = Level::ERROR);

    Level getLevel() const
    {
        return level_;
    }

    void setLevel(Level const level)
    {
        level_ = level;
        log(INFO, "Set logger level to %s", getLevelName(level));
    }

    /// Check whether logging is enabled at the given level (global check).
    bool isEnabled(Level const level) const
    {
        return level_ <= level;
    }

    /// Check whether logging is enabled at the given level for a specific module.
    /// Per-module overrides (from TLLM_LOG_LEVEL_BY_MODULE) take precedence over the global level.
    bool isEnabled(Level const level, std::string_view module) const
    {
        if (!moduleLevels_.empty() && !module.empty())
        {
            auto const it = moduleLevels_.find(module);
            if (it != moduleLevels_.end())
            {
                return it->second <= level;
            }
        }
        return level_ <= level;
    }

    /// Override the auto-detected rank. Use -1 to disable rank display.
    void setRank(int const rank)
    {
        rank_ = rank;
    }

    int getRank() const
    {
        return rank_;
    }

private:
    static auto constexpr kPREFIX = "[TRT-LLM]";
    static constexpr int kRANK_UNSET = -1;

#ifndef NDEBUG
    Level const DEFAULT_LOG_LEVEL = DEBUG;
#else
    Level const DEFAULT_LOG_LEVEL = INFO;
#endif
    Level level_ = DEFAULT_LOG_LEVEL;
    int rank_ = kRANK_UNSET;

    /// Per-module log level overrides. std::less<> enables heterogeneous lookup
    /// so find(std::string_view) works without allocating a temporary std::string.
    std::map<std::string, Level, std::less<>> moduleLevels_;

    Logger(); // NOLINT(modernize-use-equals-delete)

    /// Attempt to detect the rank from environment variables.
    static int detectRank();

    /// Parse a level name string (case-insensitive) into a Level enum value.
    /// Returns true on success and writes to *outLevel; returns false on invalid input.
    static bool parseLevelName(std::string_view name, Level& outLevel);

    /// Parse the TLLM_LOG_LEVEL_BY_MODULE env var value.
    /// Format: "level:mod1,mod2;level:mod3,mod4"
    /// e.g.   "debug:runtime,kernel;info:plugin;warning:layers"
    static std::map<std::string, Level, std::less<>> parseModuleLevels(char const* envValue);

    static inline char const* getLevelName(Level const level)
    {
        switch (level)
        {
        case TRACE: return "V";
        case DEBUG: return "D";
        case INFO: return "I";
        case WARNING: return "W";
        case ERROR: return "E";
        }

        TLLM_THROW("Unknown log level: %d", level);
    }

    // ---- getPrefix variants -------------------------------------------------

    // No module, auto rank (for internal callers without __FILE__)
    inline std::string getPrefix(Level const level) const
    {
        if (rank_ >= 0)
        {
            return tensorrt_llm::common::fmtstr("%s [%s] [RANK %d] ", kPREFIX, getLevelName(level), rank_);
        }
        return tensorrt_llm::common::fmtstr("%s [%s] ", kPREFIX, getLevelName(level));
    }

    // No module, explicit rank
    static inline std::string getPrefix(Level const level, int const rank)
    {
        return tensorrt_llm::common::fmtstr("%s [%s] [RANK %d] ", kPREFIX, getLevelName(level), rank);
    }

    // With module, auto rank
    inline std::string getPrefix(Level const level, std::string_view module) const
    {
        if (rank_ >= 0)
        {
            return tensorrt_llm::common::fmtstr("%s [%s] [%-8.*s] [RANK %d] ", kPREFIX, getLevelName(level),
                static_cast<int>(module.size()), module.data(), rank_);
        }
        return tensorrt_llm::common::fmtstr(
            "%s [%s] [%-8.*s] ", kPREFIX, getLevelName(level), static_cast<int>(module.size()), module.data());
    }

    // With module, explicit rank
    static inline std::string getPrefix(Level const level, std::string_view module, int const rank)
    {
        return tensorrt_llm::common::fmtstr("%s [%s] [%-8.*s] [RANK %d] ", kPREFIX, getLevelName(level),
            static_cast<int>(module.size()), module.data(), rank);
    }
};

// ===========================================================================
// Template implementations — no source file (internal callers)
// ===========================================================================

template <typename... Args>
void Logger::log(Logger::Level const level, char const* format, Args const&... args)
{
    if (isEnabled(level))
    {
        auto const fmt = getPrefix(level) + format;
        auto& out = level_ < WARNING ? std::cout : std::cerr;
        if constexpr (sizeof...(args) > 0)
        {
            out << fmtstr(fmt.c_str(), args...);
        }
        else
        {
            out << fmt;
        }
        out << std::endl;
    }
}

template <typename... Args>
void Logger::log(Logger::Level const level, int const rank, char const* format, Args const&... args)
{
    if (isEnabled(level))
    {
        auto const fmt = getPrefix(level, rank) + format;
        auto& out = level_ < WARNING ? std::cout : std::cerr;
        if constexpr (sizeof...(args) > 0)
        {
            out << fmtstr(fmt.c_str(), args...);
        }
        else
        {
            out << fmt;
        }
        out << std::endl;
    }
}

// ===========================================================================
// Template implementations — with module tag (TLLM_LOG macros)
// ===========================================================================

template <typename... Args>
void Logger::log(Logger::Level const level, Module module, char const* format, Args const&... args)
{
    if (isEnabled(level, module.name))
    {
        auto const fmt = getPrefix(level, module.name) + format;
        auto& out = level_ < WARNING ? std::cout : std::cerr;
        if constexpr (sizeof...(args) > 0)
        {
            out << fmtstr(fmt.c_str(), args...);
        }
        else
        {
            out << fmt;
        }
        out << std::endl;
    }
}

template <typename... Args>
void Logger::log(Logger::Level const level, Module module, int const rank, char const* format, Args const&... args)
{
    if (isEnabled(level, module.name))
    {
        auto const fmt = getPrefix(level, module.name, rank) + format;
        auto& out = level_ < WARNING ? std::cout : std::cerr;
        if constexpr (sizeof...(args) > 0)
        {
            out << fmtstr(fmt.c_str(), args...);
        }
        else
        {
            out << fmt;
        }
        out << std::endl;
    }
}
} // namespace common

TRTLLM_NAMESPACE_END

#define TLLM_LOG(level, ...)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        auto* const logger = tensorrt_llm::common::Logger::getLogger();                                                \
        constexpr auto tllmModule_ = tensorrt_llm::common::Logger::extractModule(__FILE__);                            \
        if (logger->isEnabled(level, tllmModule_))                                                                     \
        {                                                                                                              \
            logger->log(level, tensorrt_llm::common::Logger::Module{tllmModule_}, __VA_ARGS__);                        \
        }                                                                                                              \
    } while (0)

#define TLLM_LOG_TRACE(...) TLLM_LOG(tensorrt_llm::common::Logger::TRACE, __VA_ARGS__)
#define TLLM_LOG_DEBUG(...) TLLM_LOG(tensorrt_llm::common::Logger::DEBUG, __VA_ARGS__)
#define TLLM_LOG_INFO(...) TLLM_LOG(tensorrt_llm::common::Logger::INFO, __VA_ARGS__)
#define TLLM_LOG_WARNING(...) TLLM_LOG(tensorrt_llm::common::Logger::WARNING, __VA_ARGS__)
#define TLLM_LOG_ERROR(...) TLLM_LOG(tensorrt_llm::common::Logger::ERROR, __VA_ARGS__)
#define TLLM_LOG_EXCEPTION(ex, ...) tensorrt_llm::common::Logger::getLogger()->log(ex, ##__VA_ARGS__)

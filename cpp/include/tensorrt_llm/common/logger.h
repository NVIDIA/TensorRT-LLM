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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <iostream>
#include <map>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace common
{

/// Map a raw module directory name to a short display abbreviation.
/// Every known module gets an explicit entry; unknown modules fall through as-is.
/// getPrefix() prints the module name with no padding.
constexpr std::string_view formatModule(std::string_view module)
{
    constexpr size_t kMinHashLength = 3;
    auto const length = module.size();
    if (length < kMinHashLength)
    {
        return module;
    }

    // Length and weighted first/third characters distinguish known modules.
    // Duplicate case values catch collisions; equality rejects unknown names.
#define TLLM_MODULE_HASH(LEN, FIRST, THIRD) (((LEN) << 6) + ((2 * ((FIRST) - 'a') - ((THIRD) - 'a')) & 63))

#define TLLM_MODULE_CASE(LEN, FIRST, THIRD, NAME, DISPLAY)                                                             \
    case TLLM_MODULE_HASH(LEN, FIRST, THIRD): return module == #NAME ? DISPLAY : module

    // Source modules (cpp/tensorrt_llm/*)
    // clang-format off
    switch (TLLM_MODULE_HASH(length, module[0], module[2]))
    {
    default: return module;
    TLLM_MODULE_CASE(13, 'b', 't', batch_manager,      "batchmgr");
    TLLM_MODULE_CASE( 6, 'c', 'm', common,             "common");
    TLLM_MODULE_CASE(18, 'c', 't', cutlass_extensions, "cutl_ext");
    TLLM_MODULE_CASE( 7, 'd', 'e', deep_ep,            "deep_ep");
    TLLM_MODULE_CASE( 9, 'd', 'e', deep_gemm,          "deepgemm");
    TLLM_MODULE_CASE( 8, 'e', 'e', executor,           "executor");
    TLLM_MODULE_CASE( 9, 'f', 'a', flash_mla,          "flashmla");
    TLLM_MODULE_CASE( 7, 'k', 'r', kernels,            "kernels");
    TLLM_MODULE_CASE( 6, 'l', 'y', layers,             "layers");
    TLLM_MODULE_CASE( 8, 'n', 'n', nanobind,           "nanobind");
    TLLM_MODULE_CASE( 7, 'r', 'n', runtime,            "runtime");
    TLLM_MODULE_CASE( 7, 't', 's', testing,            "testing");
    TLLM_MODULE_CASE( 4, 't', 'o', thop,               "thop");
    }
    // clang-format on

#undef TLLM_MODULE_CASE
#undef TLLM_MODULE_HASH
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

    /// Tag type that carries a module name.
    /// Used by the TLLM_LOG macros to disambiguate module names from format strings.
    struct Module
    {
        constexpr explicit Module(std::source_location const location = std::source_location::current())
            : name{Logger::extractModule(location.file_name())}
        {
        }

        constexpr explicit Module(std::string_view moduleName)
            : name{moduleName}
        {
        }

        std::string_view name;
    };

    static Logger* getLogger();

    /// Extract the first subdirectory after the *last* "tensorrt_llm/" from a source file path.
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
        return "others";
    }

    Logger(Logger const&) = delete;
    void operator=(Logger const&) = delete;

    // ---- log overloads WITHOUT source file (used by internal helpers) --------

    //! Log a message using a checked std::format string and the configured rank.
    //! Pass runtime message text as an argument to "{}".
    template <typename... Args>
    void log(Level const level, std::format_string<Args...> format, Args&&... args)
    {
        logImpl(level, Module{""}, std::nullopt, format, std::forward<Args>(args)...);
    }

    //! Log a message with an explicit rank.
    template <typename... Args>
    void log(Level const level, int const rank, std::format_string<Args...> format, Args&&... args)
    {
        logImpl(level, Module{""}, rank, format, std::forward<Args>(args)...);
    }

    // ---- log overloads WITH module tag (used by TLLM_LOG macros) -------------

    //! Log a message with module filtering and the configured rank.
    template <typename... Args>
    void log(Level const level, Module module, std::format_string<Args...> format, Args&&... args)
    {
        logImpl(level, module, std::nullopt, format, std::forward<Args>(args)...);
    }

    //! Log a message with module filtering and an explicit rank.
    template <typename... Args>
    void log(Level const level, Module module, int const rank, std::format_string<Args...> format, Args&&... args)
    {
        logImpl(level, module, rank, format, std::forward<Args>(args)...);
    }

    // ---- exception logging --------------------------------------------------

    void log(std::exception const& ex, Level level = Level::ERROR);

    [[nodiscard]] Level getLevel() const
    {
        return level_;
    }

    void setLevel(Level const level)
    {
        level_ = level;
        log(INFO, "Set logger level to {}", getLevelName(level));
    }

    /// Check whether logging is enabled at the given level (global check).
    [[nodiscard]] bool isEnabled(Level const level) const
    {
        return level_ <= level;
    }

    /// Check whether logging is enabled at the given level for a specific module.
    /// Per-module overrides (from TLLM_LOG_LEVEL_BY_MODULE) take precedence over the global level.
    [[nodiscard]] bool isEnabled(Level const level, std::string_view module) const
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
    void setRank(int rank)
    {
        rank_ = rank;
    }

    [[nodiscard]] int getRank() const
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

        TLLM_THROW("Unknown log level: {}", static_cast<int>(level));
    }

    template <typename... Args>
    void logImpl(
        Level const level, Module module, std::optional<int> rank, std::format_string<Args...> format, Args&&... args)
    {
        if (!isEnabled(level, module.name))
        {
            return;
        }

        if (!rank.has_value() && rank_ >= 0)
        {
            rank = rank_;
        }

        // FIXME: Remove printf compatibility once concurrent PRs have migrated to C++20 format strings.
        // std::format places arguments only through braces, so a format with arguments and no braces is printf-style.
        if constexpr (sizeof...(Args) > 0 && (kIsPrintfArg<Args> && ...))
        {
            // Format strings are string literals, so text is null-terminated.
            auto const text = format.get();
            if (text.find_first_of("{}") == std::string_view::npos)
            {
                auto const size = std::snprintf(nullptr, 0, text.data(), args...);
                std::string message(std::max(size, 0), '\0');
                std::snprintf(message.data(), message.size() + 1, text.data(), args...);
                write(level, module.name, rank, message);
                return;
            }
        }

        write(level, module.name, rank, std::format(format, std::forward<Args>(args)...));
    }

    //! Whether C varargs pass T to snprintf the way printf conversions expect.
    template <typename T>
    static constexpr bool kIsPrintfArg = std::is_arithmetic_v<std::decay_t<T>> || std::is_pointer_v<std::decay_t<T>>
        || std::is_null_pointer_v<std::decay_t<T>>;

    static std::string getPrefix(Level const level, std::string_view module, std::optional<int> rank)
    {
        if (!module.empty())
        {
            if (rank.has_value())
            {
                return std::format("{} [{}] [{}][RANK {}] ", kPREFIX, getLevelName(level), module, *rank);
            }
            return std::format("{} [{}] [{}] ", kPREFIX, getLevelName(level), module);
        }

        if (rank.has_value())
        {
            return std::format("{} [{}] [RANK {}] ", kPREFIX, getLevelName(level), *rank);
        }
        return std::format("{} [{}] ", kPREFIX, getLevelName(level));
    }

    static void write(Level const level, std::string_view module, std::optional<int> rank, std::string_view message)
    {
        auto const line = std::format("{}{}", getPrefix(level, module, rank), message);
        auto& out = level < WARNING ? std::cout : std::cerr;
        out << line << std::endl;
    }
};

//! Log a checked format string with module filtering and the configured rank.
template <typename... Args>
void log(Logger::Level const level, Logger::Module module, std::format_string<Args...> format, Args&&... args)
{
    Logger::getLogger()->log(level, module, format, std::forward<Args>(args)...);
}

//! Log a checked format string with module filtering and an explicit rank.
template <typename... Args>
void log(Logger::Level const level, Logger::Module module, int const rank, std::format_string<Args...> format,
    Args&&... args)
{
    Logger::getLogger()->log(level, module, rank, format, std::forward<Args>(args)...);
}

} // namespace common

TRTLLM_NAMESPACE_END

#define TLLM_LOG(level, ...) ::tensorrt_llm::common::log((level), ::tensorrt_llm::common::Logger::Module{}, __VA_ARGS__)

#define TLLM_LOG_TRACE(...) TLLM_LOG(tensorrt_llm::common::Logger::TRACE, __VA_ARGS__)
#define TLLM_LOG_DEBUG(...) TLLM_LOG(tensorrt_llm::common::Logger::DEBUG, __VA_ARGS__)
#define TLLM_LOG_INFO(...) TLLM_LOG(tensorrt_llm::common::Logger::INFO, __VA_ARGS__)
#define TLLM_LOG_WARNING(...) TLLM_LOG(tensorrt_llm::common::Logger::WARNING, __VA_ARGS__)
#define TLLM_LOG_ERROR(...) TLLM_LOG(tensorrt_llm::common::Logger::ERROR, __VA_ARGS__)
#define TLLM_LOG_EXCEPTION(ex, ...) tensorrt_llm::common::Logger::getLogger()->log(ex, ##__VA_ARGS__)

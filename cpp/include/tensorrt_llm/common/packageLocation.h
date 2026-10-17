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

#include <array>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <regex>
#include <string>
#include <system_error>
#include <utility>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace tensorrt_llm::common
{

//! \brief Find the directory that contains the `tensorrt_llm` Python package, given the path of one of its
//! shared libraries.
//!
//! Walks up from the directory of \p libraryPath and returns the first ancestor that has a
//! `tensorrt_llm/include` subdirectory. For a wheel install,
//! `<site-packages>/tensorrt_llm/libs/libtensorrt_llm.so` maps to `<site-packages>`; for an editable install,
//! `<repo>/tensorrt_llm/libs/libtensorrt_llm.so` maps to `<repo>`. This is the same directory that
//! `pip show tensorrt_llm` reports as `Location` or `Editable project location`. If \p libraryPath is a symlink
//! (e.g. a build that links instead of copying the libraries), the resolved path is searched as well.
//!
//! \return The package location, or std::nullopt if no ancestor within a few levels contains the package headers.
inline std::optional<std::filesystem::path> findPackageLocationFromLibraryPath(std::filesystem::path const& libraryPath)
{
    auto const searchAncestors = [](std::filesystem::path const& path) -> std::optional<std::filesystem::path>
    {
        constexpr int kMaxAncestorLevels = 4;
        std::error_code ec;
        auto dir = path.parent_path();
        for (int level = 0; level < kMaxAncestorLevels && !dir.empty(); ++level)
        {
            if (std::filesystem::is_directory(dir / "tensorrt_llm" / "include", ec))
            {
                return dir;
            }
            auto parent = dir.parent_path();
            if (parent == dir)
            {
                break;
            }
            dir = std::move(parent);
        }
        return std::nullopt;
    };

    std::error_code ec;
    auto const absolutePath = std::filesystem::absolute(libraryPath, ec);
    auto const& searchPath = ec ? libraryPath : absolutePath;
    if (auto location = searchAncestors(searchPath))
    {
        return location;
    }
    auto const resolvedPath = std::filesystem::weakly_canonical(searchPath, ec);
    if (!ec && resolvedPath != searchPath)
    {
        return searchAncestors(resolvedPath);
    }
    return std::nullopt;
}

//! \brief Find the `tensorrt_llm` package location from the shared library that this code is loaded from.
//!
//! This does not depend on which `pip` (if any) is first on `PATH`, so it also works in environments
//! created without pip, such as `uv venv`.
inline std::optional<std::filesystem::path> getPackageLocationFromLoadedLibrary()
{
#ifndef _WIN32
    Dl_info info{};
    if (dladdr(reinterpret_cast<void const*>(&getPackageLocationFromLoadedLibrary), &info) == 0
        || info.dli_fname == nullptr)
    {
        return std::nullopt;
    }
    return findPackageLocationFromLibraryPath(info.dli_fname);
#else
    return std::nullopt;
#endif
}

//! \brief Parse the output of `pip show tensorrt_llm` and return the package location.
//!
//! `Editable project location` is preferred over `Location` when both are present.
inline std::optional<std::filesystem::path> parsePipShowLocation(std::string const& pipShowOutput)
{
    std::regex const locationRegex("(Location|Editable project location): (.+)");
    std::optional<std::string> location;
    for (auto it = std::sregex_iterator(pipShowOutput.begin(), pipShowOutput.end(), locationRegex);
         it != std::sregex_iterator(); ++it)
    {
        if (it->str(1) == "Editable project location")
        {
            location = it->str(2);
            break;
        }
        if (!location)
        {
            location = it->str(2);
        }
    }
    if (!location)
    {
        return std::nullopt;
    }
    location->erase(location->find_last_not_of(" \n\r\t") + 1);
    if (location->empty())
    {
        return std::nullopt;
    }
    return std::filesystem::path(*location);
}

//! \brief Find the `tensorrt_llm` package location by running `pip show tensorrt_llm`.
inline std::optional<std::filesystem::path> getPackageLocationFromPip()
{
    char const* cmd = "pip show tensorrt_llm 2>/dev/null";
#ifdef _MSC_VER
    FILE* pipe = _popen(cmd, "r");
#else
    FILE* pipe = popen(cmd, "r");
#endif
    if (pipe == nullptr)
    {
        return std::nullopt;
    }
    std::array<char, 128> buffer;
    std::string result;
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
    {
        result += buffer.data();
    }
#ifdef _MSC_VER
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return parsePipShowLocation(result);
}

//! \brief Find the directory that contains the `tensorrt_llm` Python package (and its JIT headers).
//!
//! The location is derived from the loaded TensorRT-LLM shared library first; `pip show tensorrt_llm` is only
//! used as a fallback when the library is not inside a `tensorrt_llm` package (e.g. a C++-only build).
//! The result is computed once and cached.
inline std::optional<std::filesystem::path> getPackageLocation()
{
    static std::optional<std::filesystem::path> const location = []() -> std::optional<std::filesystem::path>
    {
        if (auto fromLibrary = getPackageLocationFromLoadedLibrary())
        {
            return fromLibrary;
        }
        return getPackageLocationFromPip();
    }();
    return location;
}

} // namespace tensorrt_llm::common

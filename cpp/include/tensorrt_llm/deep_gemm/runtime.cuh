/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "jit_utils.cuh"
#include "scheduler.cuh"

namespace deep_gemm::jit
{

/**
 * C++ implementation of the Runtime class from runtime.py
 * Loads and executes JIT-compiled kernels
 */
class Runtime
{
public:
    Runtime(std::string const& path, deep_gemm::GemmType gemm_type);
    ~Runtime();

    static bool isPathValid(std::string const& path);

    template <typename... Args>
    void operator()(Args&&... args)
    {
        // Load shared object if not already loaded
        if (!lib_)
        {
            std::filesystem::path libPath = std::filesystem::path(path_);
#ifdef _WIN32
            libPath /= "kernel.dll";
            lib_ = LoadLibraryA(libPath.string().c_str());
            if (!lib_)
            {
                throw std::runtime_error("Failed to load DLL: " + std::to_string(GetLastError()));
            }

            // Load launch function
            switch (gemm_type_)
            {
            case deep_gemm::GemmType::Normal:
                launchFuncNormal_ = reinterpret_cast<LaunchFuncNormal>(GetProcAddress((HMODULE) lib_, "launch"));
                if (!launchFuncNormal_)
                {
                    throw std::runtime_error("Failed to find launch function: " + std::to_string(GetLastError()));
                }
                break;
            case deep_gemm::GemmType::GroupedWithOffset:
                launchFuncGroupedWithOffset_
                    = reinterpret_cast<LaunchFuncGroupedWithOffset>(GetProcAddress((HMODULE) lib_, "launch"));
                if (!launchFuncGroupedWithOffset_)
                {
                    throw std::runtime_error("Failed to find launch function: " + std::to_string(GetLastError()));
                }
                break;
            case deep_gemm::GemmType::StridedBatched:
                launchFuncStridedBatched_
                    = reinterpret_cast<LaunchFuncStridedBatched>(GetProcAddress((HMODULE) lib_, "launch"));
                if (!launchFuncStridedBatched_)
                {
                    throw std::runtime_error("Failed to find launch function: " + std::to_string(GetLastError()));
                }
                break;
            default: throw std::runtime_error("Unsupported gemm type: " + gemm_type_to_string(gemm_type_));
            }
#else
            libPath /= "kernel.so";
            lib_ = dlopen(libPath.c_str(), RTLD_LAZY);
            if (!lib_)
            {
                throw std::runtime_error("Failed to load shared object: " + std::string(dlerror()));
            }

            // Load launch function
            switch (gemm_type_)
            {
            case deep_gemm::GemmType::Normal:
                launchFuncNormal_ = reinterpret_cast<LaunchFuncNormal>(dlsym(lib_, "launch"));
                if (!launchFuncNormal_)
                {
                    throw std::runtime_error("Failed to find launch function: " + std::string(dlerror()));
                }
                break;
            case deep_gemm::GemmType::GroupedWithOffset:
                launchFuncGroupedWithOffset_ = reinterpret_cast<LaunchFuncGroupedWithOffset>(dlsym(lib_, "launch"));
                if (!launchFuncGroupedWithOffset_)
                {
                    throw std::runtime_error("Failed to find launch function: " + std::string(dlerror()));
                }
                break;
            case deep_gemm::GemmType::StridedBatched:
                launchFuncStridedBatched_ = reinterpret_cast<LaunchFuncStridedBatched>(dlsym(lib_, "launch"));
                if (!launchFuncStridedBatched_)
                {
                    throw std::runtime_error("Failed to find launch function: " + std::string(dlerror()));
                }
                break;
            default: throw std::runtime_error("Unsupported gemm type: " + gemm_type_to_string(gemm_type_));
            }
#endif
        }

        // Call the launch function with the provided arguments
        switch (gemm_type_)
        {
        case deep_gemm::GemmType::Normal: callNormal(std::forward<Args>(args)...); break;
        case deep_gemm::GemmType::GroupedWithOffset: callGroupedWithOffset(std::forward<Args>(args)...); break;
        case deep_gemm::GemmType::StridedBatched: callStridedBatched(std::forward<Args>(args)...); break;
        default: throw std::runtime_error("Unsupported gemm type: " + gemm_type_to_string(gemm_type_));
        }
    }

private:
    using LaunchFuncNormal
        = void (*)(void*, int, void*, int, void*, int, float*, float*, uint32_t, int*, cudaStream_t, int, uint32_t);
    using LaunchFuncGroupedWithOffset = void (*)(
        void*, int, void*, int, void*, int, float*, float*, int64_t*, int64_t*, cudaStream_t, int, uint32_t, uint32_t);

    using LaunchFuncStridedBatched = void (*)(void*, uint64_t, uint64_t, void*, uint64_t, uint64_t, void*, uint64_t,
        uint64_t, float*, float*, uint32_t, uint32_t, cudaStream_t, int, uint32_t);

    std::string path_;
    void* lib_;
    deep_gemm::GemmType gemm_type_;
    LaunchFuncNormal launchFuncNormal_;
    LaunchFuncGroupedWithOffset launchFuncGroupedWithOffset_;
    LaunchFuncStridedBatched launchFuncStridedBatched_;

    // Helper method for Normal GEMM - checks for correct number of arguments
    template <typename... ArgsT>
    void callNormal(ArgsT&&... args)
    {
        constexpr size_t expected_args = 13;
        constexpr size_t actual_args = sizeof...(args);

        if constexpr (actual_args != expected_args)
        {
            throw std::invalid_argument(
                "Normal GEMM requires exactly 13 arguments, but " + std::to_string(actual_args) + " were provided");
        }
        else
        {
            launchFuncNormal_(std::forward<ArgsT>(args)...);
        }
    }

    // Helper method for GroupedWithOffset GEMM - checks for correct number of arguments
    template <typename... ArgsT>
    void callGroupedWithOffset(ArgsT&&... args)
    {
        constexpr size_t expected_args = 14;
        constexpr size_t actual_args = sizeof...(args);

        if constexpr (actual_args != expected_args)
        {
            throw std::invalid_argument("GroupedWithOffset GEMM requires exactly 14 arguments, but "
                + std::to_string(actual_args) + " were provided");
        }
        else
        {
            launchFuncGroupedWithOffset_(std::forward<ArgsT>(args)...);
        }
    }

    // Helper method for StridedBatched GEMM - checks for correct number of arguments
    template <typename... ArgsT>
    void callStridedBatched(ArgsT&&... args)
    {
        constexpr size_t expected_args = 16;
        constexpr size_t actual_args = sizeof...(args);

        if constexpr (actual_args != expected_args)
        {
            throw std::invalid_argument("StridedBatched GEMM requires exactly 16 arguments, but "
                + std::to_string(actual_args) + " were provided");
        }
        else
        {
            launchFuncStridedBatched_(std::forward<ArgsT>(args)...);
        }
    }
};

/**
 * C++ implementation of the RuntimeCache class from runtime.py
 * Caches Runtime instances by path
 */
class RuntimeCache
{
public:
    static RuntimeCache& getInstance();
    Runtime* operator[](std::string const& path);
    void set(std::string const& path, std::unique_ptr<Runtime> runtime);

private:
    // Private constructor for singleton pattern
    RuntimeCache() = default;

    // Delete copy constructor and assignment operator
    RuntimeCache(RuntimeCache const&) = delete;
    RuntimeCache& operator=(RuntimeCache const&) = delete;

    std::unordered_map<std::string, std::unique_ptr<Runtime>> cache_;
};

// Global function to access the singleton
RuntimeCache& getGlobalRuntimeCache();

} // namespace deep_gemm::jit

namespace deep_gemm::jit
{
// Runtime implementation
Runtime::Runtime(std::string const& path, deep_gemm::GemmType gemm_type)
    : path_(path)
    , gemm_type_(gemm_type)
    , lib_(nullptr)
{
    assert(isPathValid(path_));
}

Runtime::~Runtime()
{
    if (lib_)
    {
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(lib_));
#else
        dlclose(lib_);
#endif
    }
}

bool Runtime::isPathValid(std::string const& path)
{
    // Check if path exists and is a directory
    if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path))
    {
        return false;
    }

    // Check if all necessary files exist
#ifdef _WIN32
    std::string soName = "kernel.dll";
#else
    std::string soName = "kernel.so";
#endif
    std::vector<std::string> requiredFiles = {"kernel.cu", soName};
    for (auto const& file : requiredFiles)
    {
        if (!std::filesystem::exists(std::filesystem::path(path) / file))
        {
            return false;
        }
    }
    return true;
}

// RuntimeCache implementation
RuntimeCache& RuntimeCache::getInstance()
{
    static RuntimeCache instance;
    return instance;
}

Runtime* RuntimeCache::operator[](std::string const& path)
{
    // Check if already in cache
    auto it = cache_.find(path);
    if (it != cache_.end())
    {
        return it->second.get();
    }

    // Check if already compiled
    if (Runtime::isPathValid(path))
    {
        // Parse path to get gemm type
        std::string gemm_type_str = path.substr(path.find_last_of('_') + 1);
        deep_gemm::GemmType gemm_type;
        if (gemm_type_str == "Normal")
        {
            gemm_type = deep_gemm::GemmType::Normal;
        }
        else if (gemm_type_str == "GroupedWithOffset")
        {
            gemm_type = deep_gemm::GemmType::GroupedWithOffset;
        }
        else if (gemm_type_str == "StridedBatched")
        {
            gemm_type = deep_gemm::GemmType::StridedBatched;
        }
        else
        {
            throw std::runtime_error("Unsupported gemm type: " + gemm_type_str);
        }

        auto runtime = std::make_unique<Runtime>(path, gemm_type);
        Runtime* result = runtime.get();
        cache_[path] = std::move(runtime);
        return result;
    }

    return nullptr;
}

void RuntimeCache::set(std::string const& path, std::unique_ptr<Runtime> runtime)
{
    cache_[path] = std::move(runtime);
}

// Global function to access the RuntimeCache singleton
RuntimeCache& getGlobalRuntimeCache()
{
    return RuntimeCache::getInstance();
}

} // namespace deep_gemm::jit

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

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "jit_utils.cuh"
#include "runtime.cuh"
#include "scheduler.cuh"

#ifdef _WIN32
#include <windows.h>
#endif

namespace deep_gemm::jit
{

/**
 * C++ implementation of the Compiler class from compiler.py
 * Compiles CUDA kernels into shared libraries
 */
class Compiler
{
public:
    // Get singleton instance
    static Compiler& getInstance();

    // Build function
    Runtime* build(uint32_t const shape_n, uint32_t const shape_k, uint32_t const block_m, uint32_t const block_n,
        uint32_t const block_k, uint32_t const num_groups, uint32_t const num_stages, uint32_t const num_tma_multicast,
        deep_gemm::GemmType const gemm_type);

    // Helper functions
    std::filesystem::path getJitIncludeDir();
    std::string getNvccCompiler();
    std::filesystem::path getDefaultUserDir();
    std::filesystem::path getTmpDir();
    std::filesystem::path getCacheDir();
    std::string generateUniqueId();

private:
    // Private constructor for singleton pattern
    Compiler();

    // Delete copy constructor and assignment operator
    Compiler(Compiler const&) = delete;
    Compiler& operator=(Compiler const&) = delete;

    std::string generateKernel(uint32_t const shape_n, uint32_t const shape_k, uint32_t const block_m,
        uint32_t const block_n, uint32_t const block_k, uint32_t const num_groups, uint32_t const num_stages,
        uint32_t const num_tma_multicast, deep_gemm::GemmType const gemm_type);
};

// Global function to access the singleton
Compiler& getGlobalCompiler();

} // namespace deep_gemm::jit

namespace deep_gemm::jit
{

// Compiler implementation
Compiler::Compiler()
{
    // Create necessary directories
    std::filesystem::create_directories(getTmpDir());
    std::filesystem::create_directories(getCacheDir());
}

Compiler& Compiler::getInstance()
{
    static Compiler instance;
    return instance;
}

std::string Compiler::generateKernel(uint32_t const shape_n, uint32_t const shape_k, uint32_t const block_m,
    uint32_t const block_n, uint32_t const block_k, uint32_t const num_groups, uint32_t const num_stages,
    uint32_t const num_tma_multicast, deep_gemm::GemmType const gemm_type)
{
    // Create the kernel source code
    std::stringstream code;

    // Header
    code << "// DeepGEMM auto-generated JIT CUDA source file\n";
    code << "#include <cuda.h>\n";
    code << "#include <cuda_fp8.h>\n";
    code << "#include <cuda_runtime.h>\n";
    code << "#include <iostream>\n\n";

    // Include necessary headers
    code << "#include \"cutlass/cutlass.h\"\n";
    code << "#include \"deep_gemm/fp8_gemm.cuh\"\n\n";

    // Launch function with signature based on gemm type
    code << "extern \"C\" void launch(";

    switch (gemm_type)
    {
    case deep_gemm::GemmType::Normal:
        code << "void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_d, int ld_d, float* scales_a,\n"
             << "    float* scales_b, uint32_t shape_m, int* grouped_layout, cudaStream_t stream, int num_sms,\n"
             << "    uint32_t smem_size)\n";
        break;
    case deep_gemm::GemmType::GroupedWithOffset:
        code << "void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_d, int ld_d, float* scales_a,\n"
             << "    float* scales_b, int64_t* problem_m_offsets, int64_t* problem_m_padded_offsets, cudaStream_t "
                "stream, int num_sms,\n"
             << "    uint32_t smem_size, uint32_t max_shape_m_padded)\n";
        break;
    case deep_gemm::GemmType::StridedBatched:
        code << "void* mat_a, uint64_t ld_a, uint64_t stride_a, void* mat_b, uint64_t ld_b, uint64_t stride_b,\n"
             << "    void* mat_d, uint64_t ld_d, uint64_t stride_d, float* scales_a, float* scales_b, uint32_t "
                "num_problems,\n"
             << "    uint32_t shape_m, cudaStream_t stream, int num_sms, uint32_t smem_size)\n";
        break;
    default: throw std::runtime_error("Unsupported gemm type: " + gemm_type_to_string(gemm_type));
    }

    code << "{\n";
    code << "    using namespace deep_gemm;\n\n";

    // Template parameters
    code << "    // Templated args from JIT compilation\n";
    code << "    constexpr auto N = " << shape_n << ", K = " << shape_k << ";\n";
    code << "    constexpr auto BLOCK_M = " << block_m << ";\n";
    code << "    constexpr auto BLOCK_N = " << block_n << ";\n";
    code << "    constexpr auto BLOCK_K = " << block_k << ";\n";
    code << "    constexpr auto kNumGroups = " << num_groups << ";\n";
    code << "    constexpr auto kNumStages = " << num_stages << ";\n";
    code << "    constexpr auto kNumTMAMulticast = " << num_tma_multicast << ";\n\n";

    // GEMM type
    code << "    // Make a templated GEMM\n";
    code << "    using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, kNumGroups, kNumStages, kNumTMAMulticast, "
            "GemmType::"
         << gemm_type_to_string(gemm_type) << ">;\n\n";

    // Launch kernel
    code << "    // Launch kernel\n";
    switch (gemm_type)
    {
    case deep_gemm::GemmType::Normal:
        code << "    GemmType::runGemm(mat_a, ld_a, mat_b, ld_b, mat_d, ld_d, scales_a, scales_b, shape_m, "
                "grouped_layout, "
                "stream,\n"
             << "        num_sms, smem_size);\n";
        break;
    case deep_gemm::GemmType::GroupedWithOffset:
        code << "    GemmType::runGemm(mat_a, ld_a, mat_b, ld_b, mat_d, ld_d, scales_a, scales_b, problem_m_offsets, "
                "problem_m_padded_offsets, "
                "stream,\n"
             << "        num_sms, smem_size, max_shape_m_padded);\n";
        break;
    case deep_gemm::GemmType::StridedBatched:
        code << "    GemmType::runGemm(mat_a, ld_a, stride_a, mat_b, ld_b, stride_b, mat_d, ld_d, stride_d, scales_a, "
                "scales_b, num_problems, shape_m, "
                "stream,\n"
             << "        num_sms, smem_size);\n";
        break;
    default: throw std::runtime_error("Unsupported gemm type: " + gemm_type_to_string(gemm_type));
    }
    code << "}\n";
    // Debug print
    if (std::getenv("TRTLLM_DG_JIT_DEBUG"))
    {
        std::cout << "Generated code:\n" << code.str() << std::endl;
    }

    return code.str();
}

// Generate a unique ID for temporary directories to avoid collisions
std::string Compiler::generateUniqueId()
{
    // Use current time and random number to generate a unique ID
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_int_distribution<> distrib(0, 999999);

    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch().count();

    // Use the static random generator
    int random_value = distrib(gen);

    return std::to_string(value) + "_" + std::to_string(random_value);
}

Runtime* Compiler::build(uint32_t const shape_n, uint32_t const shape_k, uint32_t const block_m, uint32_t const block_n,
    uint32_t const block_k, uint32_t const num_groups, uint32_t const num_stages, uint32_t const num_tma_multicast,
    deep_gemm::GemmType const gemm_type)
{
    // Compiler flags
    std::vector<std::string> nvccFlags = {"-std=c++17", "-shared", "-O3", "--expt-relaxed-constexpr",
        "--expt-extended-lambda", "-gencode=arch=compute_90a,code=sm_90a",
        "--ptxas-options=--register-usage-level=10"
            + (std::getenv("TRTLLM_DG_PTXAS_VERBOSE") ? std::string(",--verbose") : std::string("")),
        "--diag-suppress=177,174,940"};

    std::vector<std::string> cxxFlags = {"-fPIC", "-O3", "-Wno-deprecated-declarations", "-Wno-abi"};

    std::string cxxFlagsStr = "--compiler-options=";
    for (size_t i = 0; i < cxxFlags.size(); ++i)
    {
        cxxFlagsStr += cxxFlags[i];
        if (i < cxxFlags.size() - 1)
        {
            cxxFlagsStr += ",";
        }
    }

    std::vector<std::string> flags = nvccFlags;
    flags.push_back(cxxFlagsStr);

    std::vector<std::filesystem::path> includeDirs = {getJitIncludeDir()};

    // Build signature - simplified, no MD5 calculation
    std::string name = "gemm_" + std::to_string(shape_n) + "_" + std::to_string(shape_k) + "_" + std::to_string(block_m)
        + "_" + std::to_string(block_n) + "_" + std::to_string(block_k) + "_" + std::to_string(num_groups) + "_"
        + std::to_string(num_stages) + "_" + std::to_string(num_tma_multicast) + "_" + gemm_type_to_string(gemm_type);
    std::filesystem::path path = getCacheDir() / name;

    // Check runtime cache or file system hit
    auto& runtimeCache = getGlobalRuntimeCache();
    Runtime* cachedRuntime = runtimeCache[path.string()];
    if (cachedRuntime != nullptr)
    {
        if (std::getenv("TRTLLM_DG_JIT_DEBUG"))
        {
            std::cout << "Using cached JIT runtime " << name << " during build" << std::endl;
        }
        return cachedRuntime;
    }

    // Write the code to a system temp directory with a unique ID to avoid multiprocess collisions
    std::filesystem::path tmpPath = getTmpDir() / (name + "_" + generateUniqueId());
    std::filesystem::create_directories(tmpPath);
    std::filesystem::path tmpSrcPath = tmpPath / "kernel.cu";

    // Write files
    std::ofstream srcFile(tmpSrcPath);
    std::string code = generateKernel(
        shape_n, shape_k, block_m, block_n, block_k, num_groups, num_stages, num_tma_multicast, gemm_type);
    srcFile << code;
    srcFile.close();

    // Compile into a shared object file
#ifdef _WIN32
    std::filesystem::path soPath = path / "kernel.dll";
    std::filesystem::path tmpSoPath = tmpPath / "kernel.dll";
#else
    std::filesystem::path soPath = path / "kernel.so";
    std::filesystem::path tmpSoPath = tmpPath / "kernel.so";
#endif

    // Create the target directory if it doesn't exist
    std::filesystem::create_directories(path);

    // Build command
    std::vector<std::string> command = {getNvccCompiler(), tmpSrcPath.string(), "-o", tmpSoPath.string()};
    command.insert(command.end(), flags.begin(), flags.end());

    for (auto const& dir : includeDirs)
    {
        command.push_back("-I" + dir.string());
    }

    // Print command if debug enabled
    if (std::getenv("TRTLLM_DG_JIT_DEBUG") || std::getenv("TRTLLM_DG_JIT_PRINT_NVCC_COMMAND"))
    {
        std::cout << "Compiling JIT runtime " << name << " with command: ";
        for (auto const& arg : command)
        {
            std::cout << arg << " ";
        }
        std::cout << std::endl;
    }

    // Execute command
    std::string cmd;
    for (auto const& arg : command)
    {
        cmd += arg + " ";
    }

    int returnCode = system(cmd.c_str());
    if (returnCode != 0)
    {
        throw std::runtime_error("Failed to compile " + tmpSrcPath.string());
    }

    // Copy the source and compiled files to the cache directory
    try
    {
        // Rename (atomic operation) to final locations
        std::filesystem::rename(tmpSrcPath, path / "kernel.cu");
        std::filesystem::rename(tmpSoPath, soPath);

        if (std::getenv("TRTLLM_DG_JIT_DEBUG"))
        {
            std::cout << "Successfully copied kernel files to cache directory: " << path.string() << std::endl;
        }
    }
    catch (std::exception const& e)
    {
        std::cerr << "Warning: Failed to copy kernel files to cache: " << e.what() << std::endl;
    }

    // Clean up temporary directory after successful compilation
    try
    {
        std::filesystem::remove_all(tmpPath);
    }
    catch (std::exception const& e)
    {
        std::cerr << "Warning: Failed to clean up temporary directory: " << e.what() << std::endl;
    }

    // Create runtime and cache it
    auto runtime = std::make_unique<Runtime>(path.string(), gemm_type);
    Runtime* result = runtime.get();
    runtimeCache.set(path.string(), std::move(runtime));

    return result;
}

std::filesystem::path Compiler::getJitIncludeDir()
{
    static std::filesystem::path includeDir;
    if (includeDir.empty())
    {
        // Command to execute
        char const* cmd = "pip show tensorrt_llm 2>/dev/null";

        // Buffer to store the output
        std::array<char, 128> buffer;
        std::string result;

// Open pipe to command
#ifdef _MSC_VER
        FILE* pipe = _popen(cmd, "r");
#else
        FILE* pipe = popen(cmd, "r");
#endif

        if (pipe)
        {
            // Read the output
            while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
            {
                result += buffer.data();
            }

// Close the pipe
#ifdef _MSC_VER
            _pclose(pipe);
#else
            pclose(pipe);
#endif

            // Parse the location using regex
            std::regex locationRegex("Location: (.+)");
            std::smatch match;

            if (std::regex_search(result, match, locationRegex) && match.size() > 1)
            {
                // Get the captured location, trimming any trailing whitespace
                std::string location = match.str(1);
                location.erase(location.find_last_not_of(" \n\r\t") + 1);

                // Set the include directory based on the package location
                includeDir = std::filesystem::path(location) / "tensorrt_llm" / "include";
            }
        }
    }
    return includeDir;
}

std::string Compiler::getNvccCompiler()
{
    static std::string compiler;
    if (compiler.empty())
    {
        // Check environment variable
        char const* envCompiler = std::getenv("TRTLLM_DG_NVCC_COMPILER");
        if (envCompiler)
        {
            compiler = envCompiler;
        }
        else
        {
            // Check CUDA_HOME
            char const* cudaHome = std::getenv("CUDA_HOME");
            if (cudaHome)
            {
                std::filesystem::path cudaPath(cudaHome);
#ifdef _WIN32
                compiler = (cudaPath / "bin" / "nvcc.exe").string();
#else
                compiler = (cudaPath / "bin" / "nvcc").string();
#endif
            }
            else
            {
// Default to system nvcc
#ifdef _WIN32
                compiler = "nvcc.exe";
#else
                compiler = "nvcc";
#endif
            }
        }
    }
    return compiler;
}

std::filesystem::path Compiler::getDefaultUserDir()
{
    static std::filesystem::path userDir;
    if (userDir.empty())
    {
        char const* cacheDir = std::getenv("TRTLLM_DG_CACHE_DIR");
        if (cacheDir)
        {
            userDir = cacheDir;
            std::filesystem::create_directories(userDir);
        }
        else
        {
#ifdef _WIN32
            char const* appData = std::getenv("APPDATA");
            if (appData)
            {
                userDir = std::filesystem::path(appData) / "tensorrt_llm";
            }
            else
            {
                userDir = std::filesystem::temp_directory_path() / "tensorrt_llm";
            }
#else
            char const* homeDir = std::getenv("HOME");
            if (homeDir)
            {
                userDir = std::filesystem::path(homeDir) / ".tensorrt_llm";
            }
            else
            {
                userDir = std::filesystem::temp_directory_path() / "tensorrt_llm";
            }
#endif
        }
    }
    return userDir;
}

std::filesystem::path Compiler::getTmpDir()
{
    return getDefaultUserDir() / "tmp";
}

std::filesystem::path Compiler::getCacheDir()
{
    return getDefaultUserDir() / "cache";
}

// Global function to access the Compiler singleton
Compiler& getGlobalCompiler()
{
    return Compiler::getInstance();
}

} // namespace deep_gemm::jit

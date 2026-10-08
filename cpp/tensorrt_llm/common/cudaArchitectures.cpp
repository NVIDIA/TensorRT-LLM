/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaArchitectures.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <algorithm>
#include <cuda_runtime_api.h>
#include <string>

// Comma-separated SM versions, defined for this source only by its CMakeLists.txt:
// TRTLLM_BUILT_CUDA_ARCHITECTURES lists the CMAKE_CUDA_ARCHITECTURES entries, and
// TRTLLM_SUPPORTED_CUDA_ARCHITECTURES lists every architecture CMAKE_CUDA_ARCHITECTURES may select.
#ifndef TRTLLM_BUILT_CUDA_ARCHITECTURES
#error "TRTLLM_BUILT_CUDA_ARCHITECTURES must be defined when compiling this file."
#endif
#ifndef TRTLLM_SUPPORTED_CUDA_ARCHITECTURES
#error "TRTLLM_SUPPORTED_CUDA_ARCHITECTURES must be defined when compiling this file."
#endif

TRTLLM_NAMESPACE_BEGIN

namespace common
{

namespace
{

bool containsArchitecture(std::vector<int> const& architectures, int smVersion)
{
    auto const isListed = [&architectures](int sm)
    { return std::find(architectures.begin(), architectures.end(), sm) != architectures.end(); };
    // SM 121 runs the SM 120 kernels; getSMVersion() reports it as SM 120.
    if (smVersion == 120 || smVersion == 121)
    {
        return isListed(120) || isListed(121);
    }
    return isListed(smVersion);
}

std::string joinArchitectures(std::vector<int> const& architectures)
{
    std::string joined;
    for (int const sm : architectures)
    {
        joined += (joined.empty() ? "" : ";") + std::to_string(sm);
    }
    return joined;
}

} // namespace

std::vector<int> const& getBuiltCudaArchitectures()
{
    static std::vector<int> const architectures{TRTLLM_BUILT_CUDA_ARCHITECTURES};
    return architectures;
}

bool isCudaArchitectureBuilt(int smVersion)
{
    return containsArchitecture(getBuiltCudaArchitectures(), smVersion);
}

std::vector<int> const& getSupportedCudaArchitectures()
{
    static std::vector<int> const architectures{TRTLLM_SUPPORTED_CUDA_ARCHITECTURES};
    return architectures;
}

bool isCudaArchitectureSupported(int smVersion)
{
    return containsArchitecture(getSupportedCudaArchitectures(), smVersion);
}

void checkCudaArchitectureSupported(int device)
{
    int major{0};
    int minor{0};
    check_cuda_error(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    int const smVersion = major * 10 + minor;
    if (isCudaArchitectureBuilt(smVersion))
    {
        return;
    }

    cudaDeviceProp prop{};
    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    std::string const built = joinArchitectures(getBuiltCudaArchitectures());
    if (isCudaArchitectureSupported(smVersion))
    {
        TLLM_THROW(
            "GPU %d (%s, compute capability %d.%d) is not included in this TensorRT-LLM build, which was built for "
            "CUDA architectures \"%s\". Rebuild TensorRT-LLM with %d included, e.g. with "
            "--cuda_architectures \"%s;%d\".",
            device, prop.name, major, minor, built.c_str(), smVersion, built.c_str(), smVersion);
    }
    auto const& supported = getSupportedCudaArchitectures();
    int const oldestSupported = *std::min_element(supported.begin(), supported.end());
    if (smVersion < oldestSupported)
    {
        TLLM_THROW(
            "GPU %d (%s, compute capability %d.%d) is too old for TensorRT-LLM, which requires compute capability "
            "%d.%d or newer.",
            device, prop.name, major, minor, oldestSupported / 10, oldestSupported % 10);
    }
    TLLM_THROW(
        "GPU %d (%s, compute capability %d.%d) is not supported by this version of TensorRT-LLM, which supports CUDA "
        "architectures \"%s\". Check for a newer TensorRT-LLM release that supports this GPU.",
        device, prop.name, major, minor, joinArchitectures(supported).c_str());
}

} // namespace common

TRTLLM_NAMESPACE_END

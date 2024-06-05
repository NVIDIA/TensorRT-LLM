/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_runtime_api.h>
#include <string>

#include "tensorrt_llm/common/cudaDriverWrapper.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace jit
{

class CubinObj
{
public:
    // Default constructor constructs an empty unusable CubinObj instance.
    CubinObj() = default;
    CubinObj(std::string const& content);
    CubinObj(void const* buffer, size_t buffer_size);
    void launch(dim3 gridDim, dim3 blockDim, CUstream hStream, void** kernelParams);

    size_t getSerializationSize() const noexcept;
    void serialize(void* buffer, size_t buffer_size) const noexcept;

private:
    void initialize(char const* content, char const* funcName);

    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;

    std::string mContent;

    CUmodule mModule;
    CUfunction mFunction;
    unsigned int mSharedMemBytes;
};

} // namespace jit
} // namespace kernels
} // namespace tensorrt_llm

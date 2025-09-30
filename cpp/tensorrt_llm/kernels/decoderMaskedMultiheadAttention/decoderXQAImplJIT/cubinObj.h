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
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImpl.h"

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
    // Constructs from raw cubin content.
    explicit CubinObj(std::string const& content);
    // Deserializes from a serialization buffer.
    CubinObj(void const* buffer, size_t buffer_size);

    CubinObj(CubinObj const& other);
    CubinObj& operator=(CubinObj const& other);

    // CubinObj can be move-constructed/assigned.
    CubinObj(CubinObj&& other);
    CubinObj& operator=(CubinObj&& other);
    ~CubinObj();

    // Should be called at least once before calling launch().
    void initialize();
    void launch(dim3 gridDim, dim3 blockDim, CUstream hStream, void** kernelParams) const;

    // It is safe to call getSerializeSize()/serialize() before calling initialize().
    size_t getSerializationSize() const noexcept;
    void serialize(void* buffer, size_t buffer_size) const noexcept;

    bool isInitialized() const
    {
        return mInitialized;
    }

    [[nodiscard]] XQAKernelType getKernelType() const
    {
        return mKernelType;
    }

private:
    [[nodiscard]] CUfunction kernel() const
    {
        return reinterpret_cast<CUfunction>(mKernel);
    }

    static constexpr char const* kFuncName = "kernel_mha";
    static constexpr char const* kSmemName = "smemSize";
    static constexpr char const* kKernelTypeName = "kernelType";
    // Constructors should populate mContent.
    std::string mContent;

    // Fields below are undefined prior to initialize() call.
    bool mInitialized;
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;
    CUlibrary mLibrary;
    CUkernel mKernel;
    unsigned int mSharedMemBytes;
    XQAKernelType mKernelType;
};

} // namespace jit
} // namespace kernels
} // namespace tensorrt_llm

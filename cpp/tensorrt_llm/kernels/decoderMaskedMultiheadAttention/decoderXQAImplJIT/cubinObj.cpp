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
#include "cubinObj.h"

#include "serializationUtils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplCommon.h"
#include <cuda_runtime_api.h>

namespace tensorrt_llm::kernels::jit
{

CubinObj::CubinObj(void const* buffer_, size_t buffer_size)
    : mInitialized(false)
{
    uint8_t const* buffer = static_cast<uint8_t const*>(buffer_);
    size_t remaining_buffer_size = buffer_size;
    uint32_t len = readFromBuffer<uint32_t>(buffer, remaining_buffer_size);
    mContent.resize(len);
    TLLM_CHECK(len <= remaining_buffer_size);
    memcpy(mContent.data(), buffer, len);
}

CubinObj::CubinObj(std::string const& content)
    : mContent(content)
    , mInitialized(false)
{
}

CubinObj::CubinObj(CubinObj const& other)
{
    // Only uninitialized CubinObj can be copy-constructed.
    TLLM_CHECK(!other.mInitialized);

    this->mContent = other.mContent;
    this->mInitialized = false;
}

CubinObj& CubinObj::operator=(CubinObj const& other)
{
    if (this == &other)
    {
        return *this;
    }

    // Only uninitialized CubinObj can be copy-assigned.
    TLLM_CHECK(!other.mInitialized);

    this->mContent = other.mContent;
    this->mInitialized = false;

    return *this;
}

CubinObj::CubinObj(CubinObj&& other)
{
    this->mContent = std::move(other.mContent);
    if (other.mInitialized)
    {
        this->mInitialized = true;
        this->mDriver = std::move(other.mDriver);
        this->mLibrary = other.mLibrary;
        this->mKernel = other.mKernel;
        this->mSharedMemBytes = other.mSharedMemBytes;
        this->mKernelType = other.mKernelType;

        other.mInitialized = false;
    }
    else
    {
        this->mInitialized = false;
    }
}

CubinObj& CubinObj::operator=(CubinObj&& other)
{
    if (this == &other)
    {
        return *this;
    }

    this->mContent = std::move(other.mContent);
    if (other.mInitialized)
    {
        this->mInitialized = true;
        this->mDriver = std::move(other.mDriver);
        this->mLibrary = other.mLibrary;
        this->mKernel = other.mKernel;
        this->mSharedMemBytes = other.mSharedMemBytes;
        this->mKernelType = other.mKernelType;

        other.mInitialized = false;
    }
    else
    {
        this->mInitialized = false;
    }
    return *this;
}

size_t CubinObj::getSerializationSize() const noexcept
{
    size_t result = sizeof(uint32_t) + mContent.size();
    // Make result multiples of 4.
    result = (result + 3) & ~3;
    return result;
}

void CubinObj::serialize(void* buffer_, size_t buffer_size) const noexcept
{
    size_t remaining_buffer_size = buffer_size;
    uint8_t* buffer = static_cast<uint8_t*>(buffer_);
    uint32_t len = mContent.size();
    writeToBuffer<uint32_t>(len, buffer, remaining_buffer_size);
    TLLM_CHECK(len <= remaining_buffer_size);
    memcpy(buffer, mContent.c_str(), len);
}

void CubinObj::launch(dim3 gridDim, dim3 blockDim, CUstream hStream, void** kernelParams) const
{
    TLLM_CHECK(mInitialized);
    CUlaunchAttribute pdlAttr;
    pdlAttr.id = CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    pdlAttr.value.programmaticStreamSerializationAllowed = (tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0);
    CUlaunchConfig const cfg{
        gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, mSharedMemBytes, hStream, &pdlAttr, 1};

    TLLM_CU_CHECK(mDriver->cuLaunchKernelEx(&cfg, kernel(), kernelParams, /*extra=*/nullptr));
}

void CubinObj::initialize()
{
    if (!mInitialized)
    {
        mDriver = tensorrt_llm::common::CUDADriverWrapper::getInstance();
        mLibrary = nullptr;
        TLLM_CU_CHECK(
            mDriver->cuLibraryLoadData(&mLibrary, mContent.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        TLLM_CHECK(mLibrary != nullptr);
        mKernel = nullptr;
        TLLM_CU_CHECK(mDriver->cuLibraryGetKernel(&mKernel, mLibrary, kFuncName));
        TLLM_CHECK(mKernel != nullptr);

        // Populate mSharedMemBytes and mKernelType.
        mSharedMemBytes = getGlobalVar<uint32_t>(mDriver, mLibrary, kSmemName, true).value();
        mKernelType = getGlobalVar<XQAKernelType>(mDriver, mLibrary, kKernelTypeName, true).value();

        TLLM_CHECK(mSharedMemBytes > 0);

        /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */
        if (mSharedMemBytes >= 46 * 1024)
        {
            CUdevice device;
            mDriver->cuCtxGetDevice(&device);
            TLLM_CU_CHECK(mDriver->cuKernelSetAttribute(
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, mSharedMemBytes, mKernel, device));
        }

        mInitialized = true;
    }
}

CubinObj::~CubinObj()
{
    if (mInitialized)
    {
        TLLM_CU_CHECK(mDriver->cuLibraryUnload(mLibrary));
        mInitialized = false;
    }
}

} // namespace tensorrt_llm::kernels::jit

/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tllmStreamReaders.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <cufile.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

// Non-GDS StreamReader

StreamReader::StreamReader(std::filesystem::path fp)
{
    mFile.open(fp.string(), std::ios::binary | std::ios::in);
    TLLM_CHECK_WITH_INFO(mFile.good(), std::string("Error opening engine file: " + fp.string()));
}

StreamReader::~StreamReader()
{
    if (mFile.is_open())
    {
        mFile.close();
    }
}

int64_t StreamReader::read(void* destination, int64_t nbBytes)
{
    if (!mFile.good())
    {
        return -1;
    }

    mFile.read(static_cast<char*>(destination), nbBytes);

    return mFile.gcount();
}

// StreamReader using GDS

GDSStreamReader::GDSStreamReader(std::filesystem::path const& filePath)
{
    auto const start_time = std::chrono::high_resolution_clock::now();
    initializeDriver();
    auto const elapsed_ms
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time);

    TLLM_LOG_INFO("GDS driver initialization time %lld ms", elapsed_ms);

    open(filePath);
}

bool GDSStreamReader::open(std::string const& filepath)
{
    if (!initializeDriver())
    {
        TLLM_LOG_INFO("Failed to initialize cuFile driver");
        return false;
    }

    int32_t const ret = ::open(filepath.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);

    if (ret < 0)
    {
        TLLM_LOG_INFO("Failed to open engine file");
        return false;
    }

    mFd = ret;
    mFileSize = lseek(mFd, 0, SEEK_END);
    lseek(mFd, 0, SEEK_SET);

    CUfileDescr_t fileDescr;
    memset((void*) &fileDescr, 0, sizeof(fileDescr));
    fileDescr.handle.fd = mFd;
    fileDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t gdsStatus = cuFileHandleRegister(&mFileHandle, &fileDescr);

    if (gdsStatus.err != CU_FILE_SUCCESS)
    {
        TLLM_LOG_INFO("Failed to cuFileHandleRegister");
        ::close(mFd);
        return false;
    }
    return true;
}

void GDSStreamReader::close()
{
    if (mFd >= 0)
    {
        ::close(mFd);
        mFd = -1;
    }
}

GDSStreamReader::~GDSStreamReader()
{
    if (mFileHandle)
    {
        cuFileHandleDeregister(mFileHandle);
        mFileHandle = nullptr;
    }

    if (mDriverInitialized)
    {
        cuFileDriverClose();
    }
}

bool GDSStreamReader::seek(int64_t offset, nvinfer1::SeekPosition where) noexcept
{
    switch (where)
    {
    case nvinfer1::SeekPosition::kSET: mCursor = offset; return true;
    case nvinfer1::SeekPosition::kCUR: mCursor += offset; return true;
    case nvinfer1::SeekPosition::kEND: mCursor = -offset; return true;
    default: return false;
    }
    return true;
}

int64_t GDSStreamReader::read(void* dest, int64_t bytes, cudaStream_t stream) noexcept
{
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, dest) != cudaSuccess)
    {
        TLLM_LOG_INFO("cudaPointerGetAttributes failed");
    }

    off_t destOffset = 0;
    void* destBase = dest;

    if (attributes.type == cudaMemoryTypeDevice)
    {
        CUdeviceptr cuDest = reinterpret_cast<CUdeviceptr>(dest);
        CUdeviceptr cuBufBase = 0;
        size_t cuBufSize = 0;

        cuMemGetAddressRange(&cuBufBase, &cuBufSize, cuDest);
        destOffset += cuDest - cuBufBase;
        destBase = reinterpret_cast<void*>(cuBufBase);
    }
    cuFileRead(this->mFileHandle, destBase, bytes, mCursor, destOffset);

    mCursor += bytes;
    return bytes;
}

void GDSStreamReader::reset()
{
    lseek(mFd, 0, SEEK_SET);
    mCursor = 0;
}

[[nodiscard]] bool GDSStreamReader::isOpen() const
{
    bool open = mFd >= 0;
    return open;
}

bool GDSStreamReader::initializeDriver()
{
    if (mDriverInitialized)
    {
        return true;
    }

    mCuFileLibHandle = dlopen("libcufile.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!mCuFileLibHandle)
    {
        TLLM_LOG_INFO("Failed to dlopen libcufile.so");
        return false;
    }

    // Load the required functions
    *reinterpret_cast<void**>(&cuFileDriverOpen) = dlsym(mCuFileLibHandle, "cuFileDriverOpen");
    *reinterpret_cast<void**>(&cuFileHandleRegister) = dlsym(mCuFileLibHandle, "cuFileHandleRegister");
    *reinterpret_cast<void**>(&cuFileHandleDeregister) = dlsym(mCuFileLibHandle, "cuFileHandleDeregister");
    *reinterpret_cast<void**>(&cuFileDriverClose) = dlsym(mCuFileLibHandle, "cuFileDriverClose");
    *reinterpret_cast<void**>(&cuFileRead) = dlsym(mCuFileLibHandle, "cuFileRead");

    if (!cuFileDriverOpen || !cuFileHandleRegister || !cuFileHandleDeregister || !cuFileDriverClose || !cuFileRead)
    {
        TLLM_LOG_INFO("Failed to dlsym libcufile.so");
        return false;
    }

    CUfileError_t gdsStatus = cuFileDriverOpen();
    if (gdsStatus.err != CU_FILE_SUCCESS)
    {
        TLLM_LOG_INFO("cuFileDriverOpen failed");
        return false;
    }

    mDriverInitialized = true;
    return true;
}

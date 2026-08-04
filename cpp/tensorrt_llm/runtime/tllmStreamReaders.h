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
#pragma once

#include <NvInferRuntime.h>

#include <cufile.h>
#include <filesystem>
#include <fstream>

class StreamReader final : public nvinfer1::IStreamReader
{
public:
    StreamReader(std::filesystem::path fp);

    virtual ~StreamReader();

    int64_t read(void* destination, int64_t nbBytes) final;

private:
    std::ifstream mFile;
};

class GDSStreamReader final : public nvinfer1::IStreamReaderV2
{
public:
    explicit GDSStreamReader(std::filesystem::path const& filePath);

    virtual ~GDSStreamReader();

    void close();

    [[nodiscard]] bool isOpen() const;

    bool open(std::string const& filepath);

    int64_t read(void* dest, int64_t bytes, cudaStream_t stream) noexcept final;

    void reset();

    bool seek(int64_t offset, nvinfer1::SeekPosition where) noexcept final;

private:
    bool initializeDriver();

    void* mCuFileLibHandle{};
    CUfileHandle_t mFileHandle{nullptr};
    bool mDriverInitialized{false};
    int32_t mFd{-1};
    int64_t mCursor{0};
    int64_t mFileSize{0};

    CUfileError_t (*cuFileDriverOpen)(){};
    CUfileError_t (*cuFileHandleRegister)(CUfileHandle_t*, CUfileDescr_t*){};
    CUfileError_t (*cuFileHandleDeregister)(CUfileHandle_t){};
    CUfileError_t (*cuFileDriverClose)(){};
    ssize_t (*cuFileRead)(CUfileHandle_t, void*, size_t, int64_t, int64_t){};
};

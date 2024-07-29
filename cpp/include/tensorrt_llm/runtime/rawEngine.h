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

#include "tensorrt_llm/common/assert.h"

#include <NvInferRuntime.h>
#include <filesystem>

namespace tensorrt_llm::runtime
{

class RawEngine
{
public:
    enum Type
    {
        FilePath,
        AddressWithSize,
        HostMemory
    };

    explicit RawEngine(std::filesystem::path enginePath) noexcept
        : mType(FilePath)
        , mEnginePath(std::move(enginePath))
    {
    }

    explicit RawEngine(void const* engineAddr, std::size_t engineSize) noexcept
        : mType(AddressWithSize)
        , mEngineAddr(engineAddr)
        , mEngineSize(engineSize)
    {
    }

    explicit RawEngine(nvinfer1::IHostMemory const* engineBuffer) noexcept
        : mType(HostMemory)
        , mEngineBuffer(engineBuffer)
    {
    }

    [[nodiscard]] Type getType() const
    {
        return mType;
    }

    [[nodiscard]] std::filesystem::path getPath() const
    {
        TLLM_CHECK(mType == FilePath);
        return mEnginePath;
    }

    [[nodiscard]] void const* getAddress() const
    {
        TLLM_CHECK(mType == AddressWithSize);
        return mEngineAddr;
    }

    [[nodiscard]] std::size_t getSize() const
    {
        TLLM_CHECK(mType == AddressWithSize);
        return mEngineSize;
    }

    [[nodiscard]] nvinfer1::IHostMemory const* getHostMemory() const
    {
        TLLM_CHECK(mType == HostMemory);
        return mEngineBuffer;
    }

private:
    Type mType;
    std::filesystem::path mEnginePath;

    struct
    {
        void const* mEngineAddr{};
        std::size_t mEngineSize{};
    };

    nvinfer1::IHostMemory const* mEngineBuffer{};
};

} // namespace tensorrt_llm::runtime

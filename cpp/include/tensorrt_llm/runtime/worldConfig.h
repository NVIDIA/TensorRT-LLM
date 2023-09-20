/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/common.h"

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{
class WorldConfig
{
public:
    static SizeType constexpr kDefaultGpusPerNode = 8;

    constexpr explicit WorldConfig(
        SizeType worldSize = 1, SizeType rank = 0, SizeType gpusPerNode = kDefaultGpusPerNode)
        : mSize{worldSize}
        , mRank{rank}
        , mGpusPerNode{gpusPerNode}
    {
    }

    [[nodiscard]] SizeType constexpr getSize() const noexcept
    {
        return mSize;
    }

    [[nodiscard]] SizeType constexpr getRank() const noexcept
    {
        return mRank;
    }

    [[nodiscard]] SizeType constexpr getGpusPerNode() const noexcept
    {
        return mGpusPerNode;
    }

    [[nodiscard]] SizeType constexpr getDevice() const noexcept
    {
        return mRank % mGpusPerNode;
    }

    static WorldConfig mpi(nvinfer1::ILogger& logger, SizeType gpusPerNode = kDefaultGpusPerNode);

    static WorldConfig mpi(SizeType gpusPerNode = kDefaultGpusPerNode);

private:
    SizeType mSize;
    SizeType mRank;
    SizeType mGpusPerNode;
};

} // namespace tensorrt_llm::runtime

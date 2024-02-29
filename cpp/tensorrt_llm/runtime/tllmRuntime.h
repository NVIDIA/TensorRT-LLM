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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <NvInferRuntime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace tensorrt_llm::runtime
{
class TllmRuntime
{
public:
    using TensorMap = StringPtrMap<ITensor>;

    explicit TllmRuntime(void const* engineData, std::size_t engineSize, nvinfer1::ILogger& logger);

    explicit TllmRuntime(nvinfer1::IHostMemory const& engineBuffer, nvinfer1::ILogger& logger)
        : TllmRuntime{engineBuffer.data(), engineBuffer.size(), logger}
    {
    }

    explicit TllmRuntime(void const* engineData, std::size_t engineSize);

    explicit TllmRuntime(nvinfer1::IHostMemory const& engineBuffer)
        : TllmRuntime{engineBuffer.data(), engineBuffer.size()}
    {
    }

    SizeType getNbContexts() const
    {
        return static_cast<SizeType>(mContexts.size());
    }

    nvinfer1::IExecutionContext& getContext(SizeType contextIndex) const
    {
        return *mContexts.at(contextIndex);
    }

    SizeType getNbProfiles() const
    {
        return static_cast<SizeType>(mEngine->getNbOptimizationProfiles());
    }

    nvinfer1::IExecutionContext& addContext(std::int32_t profileIndex);

    void clearContexts();

    void setInputTensors(SizeType contextIndex, TensorMap const& tensorMap);

    void setOutputTensors(SizeType contextIndex, TensorMap& tensorMap);

    bool executeContext(SizeType contextIndex) const;

    CudaStream const& getStream() const;

    BufferManager::CudaStreamPtr getStreamPtr()
    {
        return mStream;
    }

    nvinfer1::ICudaEngine& getEngine()
    {
        return *mEngine;
    }

    nvinfer1::ICudaEngine const& getEngine() const
    {
        return *mEngine;
    }

    BufferManager& getBufferManager()
    {
        return mBufferManager;
    }

    BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

private:
    BufferManager::CudaStreamPtr mStream;
    BufferManager mBufferManager;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    BufferManager::IBufferPtr mEngineBuffer;
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> mContexts;
    std::unique_ptr<ITensor> mDummyTensor;
};
} // namespace tensorrt_llm::runtime

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
#include "tensorrt_llm/runtime/layerProfiler.h"
#include "tensorrt_llm/runtime/rawEngine.h"
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

    explicit TllmRuntime(RawEngine const& rawEngine, nvinfer1::ILogger* logger, float gpuWeightsPercent = 1.0f,
        bool useShapeInference = true);

    SizeType32 getNbContexts() const
    {
        return static_cast<SizeType32>(mContexts.size());
    }

    nvinfer1::IExecutionContext& getContext(SizeType32 contextIndex) const
    {
        return *mContexts.at(contextIndex);
    }

    SizeType32 getNbProfiles() const
    {
        return static_cast<SizeType32>(mEngine->getNbOptimizationProfiles());
    }

    /// @brief If multiple TensorRT optimization profiles are built in the engine, this function selects the
    /// corresponding profile that is going to be used based on the runtime shape, for now, TensorRT-LLM only split
    /// multiple profiles on the num_tokens dimension, hence the profile index is selected based on which profile
    /// handles the actual num_tokens
    /// @return The index of the selected TensorRT optimization profile
    [[nodiscard]] SizeType32 getOptProfileId(int numTokens, std::vector<SizeType32> const& splitPoints) const
    {
        if (getNbProfiles() == 1)
        {
            return 0;
        }
        auto const it = std::lower_bound(splitPoints.begin(), splitPoints.end(), numTokens);
        auto const optProfileId = std::distance(splitPoints.begin(), it);
        return optProfileId;
    }

    nvinfer1::IExecutionContext& addContext(std::int32_t profileIndex);

    void clearContexts();

    void setInputTensors(SizeType32 contextIndex, TensorMap const& tensorMap);

    void setOutputTensors(SizeType32 contextIndex, TensorMap& tensorMap);

    bool executeContext(SizeType32 contextIndex) const;

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

    nvinfer1::IEngineInspector& getEngineInspector()
    {
        return *mEngineInspector;
    }

    nvinfer1::IEngineInspector const& getEngineInspector() const
    {
        return *mEngineInspector;
    }

    BufferManager& getBufferManager()
    {
        return mBufferManager;
    }

    BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

    void setLayerProfiler();
    bool hasLayerProfiler(SizeType32 contextId) const;
    std::string getLayerProfileInfo() const;
    void reportToProfiler(SizeType32 contextId);

private:
    BufferManager::CudaStreamPtr mStream;
    BufferManager mBufferManager;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    BufferManager::IBufferPtr mEngineBuffer;
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> mContexts;
    std::unique_ptr<ITensor> mDummyTensor;
    std::unique_ptr<nvinfer1::IEngineInspector> mEngineInspector;
    std::unique_ptr<LayerProfiler> mLayerProfiler;
    bool mUseShapeInference;
};
} // namespace tensorrt_llm::runtime

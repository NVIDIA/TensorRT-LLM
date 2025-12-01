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

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/layerProfiler.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <NvInferRuntime.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::runtime
{
class TllmRuntime
{
public:
    using TensorMap = StringPtrMap<ITensor>;

    explicit TllmRuntime(RawEngine const& rawEngine, nvinfer1::ILogger* logger, bool useGpuDirectStorage = false,
        float gpuWeightsPercent = 1.0f, bool useShapeInference = true);

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
    /// corresponding profile that is going to be used based on the runtime shape, for now, TensorRT LLM only split
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

    /// @brief Set input tensors from tensorMap for all contexts.
    /// @details The function can be used to set static input tensors for all iterations. If a tensor was set this way,
    /// it doesn't need to included in calls to setInputTensors anymore.
    void setStaticInputTensors(TensorMap const& tensorMap);

    /// @brief Set input tensors from tensorMap for context at contextIndex.
    /// @details The function expects that all input tensors (excluding the ones set by setStaticInputTensors) are
    /// contained in the tensorMap. If a tensor is missing, has a bad shape or type, it will throw.
    void setInputTensors(SizeType32 contextIndex, TensorMap const& tensorMap);

    /// @brief Set output tensors from tensorMap for context at contextIndex.
    /// @details The function expects that all output tensors are contained in the tensorMap. If a tensor is missing and
    /// shape inference is enabled, it will allocate the tensor on GPU and insert it into the tensorMap. Otherwise it
    /// will throw.
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
    void loadManagedWeights(RawEngine const& rawEngine, int localRank);
    void initializeUserBuffer(tensorrt_llm::runtime::WorldConfig const& world_config, SizeType32 maxBatchSize,
        SizeType32 maxBeamWidth, SizeType32 maxSequenceLength, SizeType32 hiddenSize,
        std::optional<SizeType32> maxNumTokens);

    bool isUserBufferEnabled() const
    {
        return mUserBufferEnabled;
    }

    void setCurrentBeamWidths(std::vector<SizeType32> const& beamWidth) noexcept
    {
        mCurrentBeamWidths = beamWidth;
    }

    [[nodiscard]] SizeType32 const& getCurrentBeamWidth() const noexcept
    {
        // At present, all requests of a batch must have the same beam width in one generation step (or they will not
        // be batched together). So, the beam widths in `mCurrentBeamWidths` are the same.
        // Corresponding changes must be done if Diverse-Beam-Width-Search (DBWS, requests with diverse beam width in
        // a batch in one generation step) is supported in the future.
        TLLM_CHECK_WITH_INFO(mCurrentBeamWidths.size() > 0, "`mCurrentBeamWidths` is empty.");
        bool const isEqual = std::all_of(mCurrentBeamWidths.begin(), mCurrentBeamWidths.end(),
            [&](int elem) { return elem == mCurrentBeamWidths.front(); });
        TLLM_CHECK_WITH_INFO(isEqual, "beam widths in `mCurrentBeamWidths` are not all equal.");
        return mCurrentBeamWidths.front();
    }

private:
    void cacheTensorNames();

    void setInputTensorsImpl(SizeType32 contextIndex, TensorMap const& tensorMap, bool throwOnMiss);

    void setUserBufferTensors(SizeType32 contextIndex, TensorMap& tensorMap);

    void printEngineInfo();

    void printContextInfo(SizeType32 contextIndex);

    // Tool functions for `printEngineInfo()`.
    static std::string shapeToString(nvinfer1::Dims64 const& dim)
    {
        std::string output("(");
        if (dim.nbDims == 0)
        {
            return output + ")";
        }
        for (int i = 0; i < dim.nbDims - 1; ++i)
        {
            output += std::to_string(dim.d[i]) + ", ";
        }
        output += std::to_string(dim.d[dim.nbDims - 1]) + ")";
        return output;
    }

    static std::string dataTypeToString(nvinfer1::DataType type)
    {
        switch (type)
        {
        case nvinfer1::DataType::kINT64: return "INT64";
        case nvinfer1::DataType::kINT32: return "INT32";
        case nvinfer1::DataType::kFLOAT: return "FP32";
        case nvinfer1::DataType::kBF16: return "BF16";
        case nvinfer1::DataType::kHALF: return "FP16";
        case nvinfer1::DataType::kBOOL: return "BOOL";
        case nvinfer1::DataType::kUINT8: return "UINT8";
        case nvinfer1::DataType::kINT8: return "INT8";
        case nvinfer1::DataType::kFP8: return "FP8";
        case nvinfer1::DataType::kINT4: return "INT4";
        case nvinfer1::DataType::kFP4: return "FP4";
        default: return "UNKNOWN";
        }
        return "";
    }

    static std::string alignText(
        std::string const& text, int const width, bool const bCenter = true, char const blank = ' ')
    {
        int textLen = text.size();
        int padLeft = 0;
        int padRight = 0;
        padLeft = bCenter ? (width - textLen) / 2 : 0;
        padRight = width - padLeft - textLen;
        return std::string(padLeft, blank) + text + std::string(padRight, blank);
    }

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
    TensorMap mManagedWeightsMap;
    // List of input tensor names.
    // Names of static tensors are removed from this list when setStaticInputTensors is called.
    std::vector<std::string> mInputTensorNames;
    std::vector<std::string> mOutputTensorNames;

    bool mUserBufferEnabled;
    // For Variable-Beam-Width-Search
    std::vector<SizeType32> mCurrentBeamWidths;
};
} // namespace tensorrt_llm::runtime

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
#include "tllmRuntime.h"
#include "common.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/common/safetensors.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tllmLogger.h"
#include "tllmStreamReaders.h"

#include "nlohmann/json.hpp"
#include <NvInferRuntime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace tensorrt_llm::runtime;
using TensorMap = StringPtrMap<ITensor>;

namespace
{
static_assert(std::is_signed<SizeType32>::value, "SizeType32 must be signed");

nvinfer1::Dims shapeToDims(std::vector<std::size_t> const& shape)
{
    TLLM_CHECK(shape.size() <= nvinfer1::Dims::MAX_DIMS);
    nvinfer1::Dims dims;
    auto constexpr dim_max = std::numeric_limits<ITensor::DimType64>::max();
    dims.nbDims = static_cast<std::int32_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        // shape[i] >= 0 because it has unsigned type. Check upper bound:
        TLLM_CHECK(shape[i] <= static_cast<std::size_t>(dim_max));
        dims.d[i] = static_cast<ITensor::DimType64>(shape[i]);
    }
    return dims;
}

std::vector<std::size_t> dimsToShape(nvinfer1::Dims const& dims)
{
    TLLM_CHECK(dims.nbDims >= 0);
    std::vector<std::size_t> shape(dims.nbDims);
    for (std::int32_t i = 0; i < dims.nbDims; ++i)
    {
        TLLM_CHECK(dims.d[i] >= 0);
        shape[i] = static_cast<std::size_t>(dims.d[i]);
    }
    return shape;
}

tensorrt_llm::runtime::TllmLogger defaultLogger{};

void setWeightStreaming(nvinfer1::ICudaEngine& engine, float const gpuWeightsPercent)
{
    if (gpuWeightsPercent < 1)
    {
        int64_t streamableSize = engine.getStreamableWeightsSize();
        int64_t budget = gpuWeightsPercent * streamableSize;
        TLLM_LOG_INFO("Set gpu weights percent to %f, which is %lld bytes. Valid range: %lld bytes - %lld bytes.",
            gpuWeightsPercent, budget, 0, streamableSize);
        engine.setWeightStreamingBudgetV2(budget);
    }
}

class LayerInfo
{
public:
    LayerInfo(std::optional<std::string> name, std::string type)
        : name(std::move(name))
        , type(std::move(type)){};
    std::optional<std::string> name;
    std::string type;
};

void assessLikelihoodOfRuntimeAllocation(
    nvinfer1::ICudaEngine const& engine, nvinfer1::IEngineInspector const& engineInspector)

{
    TLLM_LOG_INFO("Inspecting the engine to identify potential runtime issues...");
    auto const profilingVerbosity = engine.getProfilingVerbosity();
    if (profilingVerbosity != nvinfer1::ProfilingVerbosity::kDETAILED)
    {
        TLLM_LOG_INFO(
            "The profiling verbosity of the engine does not allow this analysis to proceed. Re-build the engine with "
            "'detailed' profiling verbosity to get more diagnostics.");
        return;
    }
    auto const* const layerTypeKey = "LayerType";
    auto const* const nameKey = "Name";
    auto const numLayers = engine.getNbLayers();
    TLLM_LOG_INFO("Model has %i layers.", numLayers);
    std::vector<SizeType32> indexes(numLayers);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::vector<std::optional<LayerInfo>> layerInfos(numLayers);
    std::transform(indexes.cbegin(), indexes.cend(), layerInfos.begin(),
        [&](SizeType32 const idx)
        {
            auto const* const layerInfo
                = engineInspector.getLayerInformation(idx, nvinfer1::LayerInformationFormat::kJSON);

            // Needs to be copied explicitly, see documentation of `getLayerInformation`.
            auto const layerInfoCopy = std::string(layerInfo);
            auto const jsonLayerInfo = nlohmann::json::parse(layerInfoCopy);
            auto const layerJsonType = jsonLayerInfo.type();
            if (layerJsonType != nlohmann::detail::value_t::object)
            {
                return std::optional<LayerInfo>{};
            }
            if (!jsonLayerInfo.contains(layerTypeKey))
            {
                return std::optional<LayerInfo>{};
            }
            auto const& typeJson = jsonLayerInfo.at(layerTypeKey);
            if (typeJson.type() != nlohmann::detail::value_t::string)
            {
                return std::optional<LayerInfo>{};
            }
            std::optional<std::string> name{};
            if (jsonLayerInfo.contains(nameKey))
            {
                auto const& nameJson = jsonLayerInfo.at(nameKey);
                auto const nameJsonType = nameJson.type();
                if (nameJsonType == nlohmann::detail::value_t::string)
                {
                    name = nameJson.get<std::string>();
                }
            }
            return std::make_optional(LayerInfo{name, typeJson.get<std::string>()});
        });
    auto const layersWithInfoEnd = std::partition(
        layerInfos.begin(), layerInfos.end(), [](std::optional<LayerInfo> const& info) { return info.has_value(); });
    if (layersWithInfoEnd == layerInfos.begin())
    {
        TLLM_LOG_INFO("Engine layer infos could not be parsed into useful information.");
        return;
    }
    auto const allocateLayersEnd = std::partition(layerInfos.begin(), layersWithInfoEnd,
        [](std::optional<LayerInfo> const& info) { return info.value().type == "allocate"; });
    auto numWarnings = 0;
    for (auto layerInfo = layerInfos.begin(); layerInfo != allocateLayersEnd; layerInfo++)
    {
        auto constexpr maxNumWarnings = 25;
        if (numWarnings < maxNumWarnings)
        {
            auto const layerName = layerInfo->value().name.value_or("");
            TLLM_LOG_WARNING(
                "Layer '%s' has type '%s', which could lead to large runtime memory allocations. Performance "
                "might be degraded and / or you might run out of memory.",
                layerName.c_str(), layerInfo->value().type.c_str());
        }
        numWarnings++;
    }
    if (numWarnings > 0)
    {
        TLLM_LOG_WARNING(
            "There were a total of %i layers with type 'allocate'. Some warnings might have been silenced to keep the "
            "output concise.",
            numWarnings);
    }
}

} // namespace

TllmRuntime::TllmRuntime(RawEngine const& rawEngine, nvinfer1::ILogger* logger, bool useGpuDirectStorage,
    float gpuWeightsPercent, bool useShapeInference)
    : mStream(std::make_shared<CudaStream>())
    , mBufferManager{mStream, true} // Ensure to trim the memory pool on destruction.
    , mRuntime{nvinfer1::createInferRuntime(static_cast<bool>(logger) ? *logger : defaultLogger)}
    , mUseShapeInference{useShapeInference}
    , mUserBufferEnabled{false}
{
    auto const startTime = std::chrono::high_resolution_clock::now();

    switch (rawEngine.getType())
    {
    case RawEngine::Type::FilePath:
    {
        if (useGpuDirectStorage)
        {
            TLLM_LOG_INFO("GDS is used to load the engine!");
            auto reader = GDSStreamReader(rawEngine.getPath());
            mEngine.reset(mRuntime->deserializeCudaEngine(reader));
        }
        else
        {
            auto reader = StreamReader(rawEngine.getPath());
            mEngine.reset(mRuntime->deserializeCudaEngine(reader));
        }
        break;
    }
    case RawEngine::Type::AddressWithSize:
        mEngine.reset(mRuntime->deserializeCudaEngine(rawEngine.getAddress(), rawEngine.getSize()));
        break;
    case RawEngine::Type::HostMemory:
        mEngine.reset(
            mRuntime->deserializeCudaEngine(rawEngine.getHostMemory()->data(), rawEngine.getHostMemory()->size()));
        break;
    default: TLLM_THROW("Unsupported raw engine type.");
    }

    auto const elapsedMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime);

    TLLM_LOG_INFO("Engine load time %lld ms", elapsedMs);

    TLLM_CHECK_WITH_INFO(mEngine != nullptr, "Failed to deserialize cuda engine.");
    mEngineInspector.reset(mEngine->createEngineInspector());
    assessLikelihoodOfRuntimeAllocation(*mEngine, *mEngineInspector);
    setWeightStreaming(getEngine(), gpuWeightsPercent);
    auto const devMemorySize = mEngine->getDeviceMemorySizeV2();
    mEngineBuffer = mBufferManager.gpu(devMemorySize);
    // Print context memory size for CI/CD to track.
    TLLM_LOG_INFO("[MemUsageChange] Allocated %.2f MiB for execution context memory.",
        static_cast<double>(devMemorySize) / 1048576.0);

    cacheTensorNames();
}

void TllmRuntime::cacheTensorNames()
{
    for (std::int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        auto const* const name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            mInputTensorNames.emplace_back(name);
        }
        else if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            mOutputTensorNames.emplace_back(name);
        }
    }
}

nvinfer1::IExecutionContext& TllmRuntime::addContext(std::int32_t profileIndex)
{
    TLLM_CHECK(0 <= profileIndex && profileIndex < mEngine->getNbOptimizationProfiles());
    mContexts.emplace_back(mEngine->createExecutionContextWithoutDeviceMemory());
    if (!mContexts.back())
    {
        if (mEngine->getStreamableWeightsSize() > 0)
        {
            TLLM_THROW("Failed to allocate memory for weights. Please try reducing --gpu_weights_percent.");
        }
        else
        {
            TLLM_THROW("Internal Error: Failed to create an execution context.");
        }
    }
    auto& context = *mContexts.back();
    context.setDeviceMemoryV2(mEngineBuffer->data(), static_cast<int64_t>(mEngineBuffer->getCapacity()));

    if (tensorrt_llm::common::Logger::getLogger()->isEnabled(tensorrt_llm::common::Logger::TRACE)
        && mContexts.size() == 1)
    {
        // Print engine information only once
        printEngineInfo();
    }

    context.setOptimizationProfileAsync(profileIndex, mStream->get());
    // If nvtx verbosity is DETAILED, print an info about potential perf overhead.
    if (context.getNvtxVerbosity() == nvinfer1::ProfilingVerbosity::kDETAILED)
    {
        TLLM_LOG_INFO(
            "The engine was built with kDETAILED profiling verbosity, which may result in small overheads at runtime.");
    }
    return context;
}

void TllmRuntime::printEngineInfo()
{
    auto& context = *(mContexts[0]);
    int const nIO = mEngine->getNbIOTensors();            // Count of input / output tensor
    int const nOP = mEngine->getNbOptimizationProfiles(); // Count of Optimization Profile
    std::size_t maxNameWidth = 0;
    std::size_t maxShapeWidth = 0;

    // Get information of engine input / output
    std::vector<std::string> tensorNameList{};
    tensorNameList.reserve(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        tensorNameList.emplace_back(mEngine->getIOTensorName(i));
    }
    std::vector<std::map<std::string, std::string>> tensorInfo(nIO);          // Tensor Information Vector
    std::vector<std::vector<std::vector<nvinfer1::Dims64>>> profileInfo(nIO); // Tensor Optimization Profile Vector
    for (int i = 0; i < nIO; ++i)
    {
        auto const& name = tensorNameList[i];
        char const* nameC{name.c_str()}; // name of C-style
        maxNameWidth = std::max(maxNameWidth, name.size());
        tensorInfo[i]["mode"] = mEngine->getTensorIOMode(nameC) == nvinfer1::TensorIOMode::kINPUT ? "I" : "O";
        tensorInfo[i]["location"]
            = mEngine->getTensorLocation(nameC) == nvinfer1::TensorLocation::kDEVICE ? "GPU" : "CPU";
        tensorInfo[i]["data_type"] = dataTypeToString(mEngine->getTensorDataType(nameC));
        tensorInfo[i]["build_shape"] = shapeToString(mEngine->getTensorShape(nameC));
        maxShapeWidth = std::max(maxShapeWidth, tensorInfo[i]["build_shape"].size());
        if (tensorInfo[i]["mode"] == "I")
        {
            std::vector<std::vector<nvinfer1::Dims64>> topPerTensor(nOP);
            for (int k = 0; k < nOP; ++k)
            {
                if (tensorInfo[i]["location"] == std::string("GPU"))
                {
                    std::vector<nvinfer1::Dims64> top(3);
                    top[0] = mEngine->getProfileShape(nameC, k, nvinfer1::OptProfileSelector::kMIN);
                    top[1] = mEngine->getProfileShape(nameC, k, nvinfer1::OptProfileSelector::kOPT);
                    top[2] = mEngine->getProfileShape(nameC, k, nvinfer1::OptProfileSelector::kMAX);
                    topPerTensor[k] = top;
                    maxShapeWidth = std::max(maxShapeWidth, shapeToString(top[2]).size());
                }
                else
                {
                    // Shape input tensor, not used in TRT-LLM support yet
                    std::vector<nvinfer1::Dims64> top(3);
                    int const nDim = mEngine->getTensorShape(nameC).nbDims;
                    nvinfer1::Dims64 tensorShape{nDim, {-1}};
                    int const* pos = nullptr;
                    pos = mEngine->getProfileTensorValues(nameC, k, nvinfer1::OptProfileSelector::kMIN);
                    std::copy(pos, pos + nDim, tensorShape.d);
                    top[0] = tensorShape;
                    pos = mEngine->getProfileTensorValues(nameC, k, nvinfer1::OptProfileSelector::kOPT);
                    std::copy(pos, pos + nDim, tensorShape.d);
                    top[1] = tensorShape;
                    pos = mEngine->getProfileTensorValues(nameC, k, nvinfer1::OptProfileSelector::kMAX);
                    std::copy(pos, pos + nDim, tensorShape.d);
                    top[2] = tensorShape;
                    topPerTensor[k] = top;
                }
            }
            profileInfo[i] = topPerTensor;
        }
        else
        {
            profileInfo[i] = std::vector<std::vector<nvinfer1::Dims64>>(nOP);
        }
    }
    // Set input shape to get output shape
    for (int k = 0; k < nOP; ++k)
    {
        for (int j = 0; j < 3; ++j) // Min, Opt, Max
        {
            for (int i = 0; i < nIO; ++i)
            {
                auto const& name = tensorNameList[i];
                char const* nameC = name.c_str();
                if (tensorInfo[i]["mode"] == "I")
                {
                    if (tensorInfo[i]["location"] == std::string("GPU"))
                    {
                        context.setInputShape(nameC, profileInfo[i][k][j]);
                    }
                    else
                    {
                        // Shape input tensor, not used in TRT-LLM support yet
                        context.setInputTensorAddress(nameC, profileInfo[i][k][j].d);
                    }
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(context.allInputDimensionsSpecified(), "Input dimensions not specified");
                    TLLM_CHECK_WITH_INFO(context.allInputShapesSpecified(), "Input shapes not specified");
                    if (tensorInfo[i]["location"] == std::string("GPU"))
                    {
                        profileInfo[i][k].push_back(context.getTensorShape(nameC));
                    }
                    else
                    {
                        // Shape input tensor, not used in TRT-LLM support yet
                        int const nDim = mEngine->getTensorShape(nameC).nbDims;
                        nvinfer1::Dims64 tensorShape{nDim, {}};
                        int const* pos = reinterpret_cast<int const*>(context.getTensorAddress(nameC));
                        std::copy(pos, pos + nDim, tensorShape.d);
                        profileInfo[i][k].push_back(tensorShape);
                    }
                }
            }
        }
    }

    // Print information of engine input / output
    std::string info;
    TLLM_LOG_TRACE("Information of engine input / output.");
    TLLM_LOG_TRACE(std::string(maxNameWidth + maxShapeWidth + 24, '='));
    info = alignText("Name", maxNameWidth) + "|I/O|Location|DataType|" + alignText("Shape", maxShapeWidth) + "|";
    TLLM_LOG_TRACE(info.c_str());
    TLLM_LOG_TRACE(std::string(maxNameWidth + maxShapeWidth + 24, '-'));
    for (int i = 0; i < nIO; ++i)
    {
        info = alignText(tensorNameList[i], maxNameWidth, false) + "|";
        info += alignText(tensorInfo[i]["mode"], 3) + "|";
        info += alignText(tensorInfo[i]["location"], 8) + "|";
        info += alignText(tensorInfo[i]["data_type"], 8) + "|";
        info += alignText(tensorInfo[i]["build_shape"], maxShapeWidth) + "|";
        TLLM_LOG_TRACE(info.c_str());
    }
    TLLM_LOG_TRACE(std::string(maxNameWidth + maxShapeWidth + 24, '='));
    // Print information of optimization profile
    TLLM_LOG_TRACE("Information of optimization profile.");
    for (int k = 0; k < nOP; ++k)
    {
        TLLM_LOG_TRACE("Optimization Profile %d:", k);
        TLLM_LOG_TRACE(std::string(maxNameWidth + maxShapeWidth * 3 + 4, '='));
        info = alignText("Name", maxNameWidth) + "|";
        info += alignText("Min", maxShapeWidth) + "|";
        info += alignText("Opt", maxShapeWidth) + "|";
        info += alignText("Max", maxShapeWidth) + "|";
        TLLM_LOG_TRACE(info.c_str());
        TLLM_LOG_TRACE(std::string(maxNameWidth + maxShapeWidth * 3 + 4, '-'));
        for (int i = 0; i < nIO; ++i)
        {
            auto const& top = profileInfo[i][k];
            info = alignText(tensorNameList[i], maxNameWidth, false) + "|";
            info += alignText(shapeToString(top[0]), maxShapeWidth) + "|";
            info += alignText(shapeToString(top[1]), maxShapeWidth) + "|";
            info += alignText(shapeToString(top[2]), maxShapeWidth) + "|";
            TLLM_LOG_TRACE(info.c_str());
        }
        TLLM_LOG_TRACE(std::string(maxNameWidth + maxShapeWidth * 3 + 4, '='));
    }
}

void TllmRuntime::printContextInfo(SizeType32 contextIndex)
{
    auto const& context = *(mContexts[contextIndex]);
    int const nIO = mEngine->getNbIOTensors(); // Count of input / output tensor
    std::size_t maxNameWidth = 0;
    std::size_t maxShapeWidth = 0;
    std::vector<std::tuple<std::string, bool, std::string>> tensorInfo(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        auto const name = std::string(mEngine->getIOTensorName(i));
        bool const isInput = mEngine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
        auto const shape = shapeToString(context.getTensorShape(name.c_str()));
        tensorInfo[i] = std::make_tuple(name, isInput, shape);
        maxNameWidth = std::max(maxNameWidth, name.size());
        maxShapeWidth = std::max(maxShapeWidth, shape.size());
        // Shape input tensor is not considered in TRT-LLM yet
    }

    TLLM_LOG_TRACE("Information of context input / output.");
    TLLM_LOG_TRACE("Using Optimization Profile: %d", contextIndex);
    TLLM_LOG_TRACE(std::string(maxNameWidth + maxShapeWidth + 6, '='));
    std::string info = alignText("Name", maxNameWidth) + "|I/O|" + alignText("Shape", maxShapeWidth) + "|";
    TLLM_LOG_TRACE(info.c_str());
    TLLM_LOG_TRACE(std::string(maxNameWidth + maxShapeWidth + 6, '-'));
    for (int i = 0; i < nIO; ++i)
    {
        auto const& [name, isInput, shape] = tensorInfo[i];
        info = alignText(name, maxNameWidth, false) + "|";
        info += alignText(isInput ? "I" : "O", 3) + "|";
        info += alignText(shape, maxShapeWidth) + "|";
        TLLM_LOG_TRACE(info.c_str());
    }
    TLLM_LOG_TRACE(std::string(maxNameWidth + maxShapeWidth + 6, '='));
}

void TllmRuntime::clearContexts()
{
    for (auto& context : mContexts)
    {
        context.reset();
    }
    mContexts.clear();
}

bool TllmRuntime::executeContext(SizeType32 contextIndex) const
{
    NVTX3_FUNC_RANGE();
    auto& context = getContext(contextIndex);
    auto res = context.enqueueV3(mStream->get());
    sync_check_cuda_error(mStream->get());
    return res;
}

void TllmRuntime::setInputTensorsImpl(SizeType32 contextIndex, TensorMap const& tensorMap, bool throwOnMiss)
{
    NVTX3_FUNC_RANGE();
    auto& context = getContext(contextIndex);
    for (auto const& name : mInputTensorNames)
    {
        auto const pos = tensorMap.find(name);
        if (pos == tensorMap.end())
        {
            if (throwOnMiss)
            {
                auto expectedShape = mEngine->getTensorShape(name.c_str());
                TLLM_THROW("Input tensor '%s' not found; expected shape: %s", name.c_str(),
                    ITensor::toString(expectedShape).c_str());
            }
            else
            {
                continue;
            }
        }

        auto const& tensor = pos->second;
        auto const tensorDtype = tensor->getDataType();
        auto const engineDtype = mEngine->getTensorDataType(name.c_str());
        // WAR: TRT does not support mixed FP8 and FP16 input, so engine expects FP16 tensors.
        TLLM_CHECK_WITH_INFO(tensorDtype == engineDtype
                || (tensorDtype == nvinfer1::DataType::kFP8 && engineDtype == nvinfer1::DataType::kHALF),
            "%s: expected type %d, provided type %d", name.c_str(), static_cast<std::int32_t>(engineDtype),
            static_cast<std::int32_t>(tensorDtype));

        auto tensorShape = tensor->getShape();

        // Change shape of `cache_indirection` for Variable-Beam-Width-Search
        // TODO: remove this hack if beamWidth of each request are passed into GptAttentionPlugin by input tensor
        if (name == "cache_indirection" && mCurrentBeamWidths.size() > 0)
        {
            SizeType32 const beamWidth = getCurrentBeamWidth();
            if (tensorShape.d[1] != beamWidth)
            {
                tensorShape.d[1] = beamWidth;
                TLLM_LOG_TRACE("Change shape of cache_indirection to %s", ITensor::toString(tensorShape).c_str());
            }
        }

        auto const setInputShapeSuccess = context.setInputShape(name.c_str(), tensorShape);
        if (!setInputShapeSuccess)
        {
            auto const minShape
                = mEngine->getProfileShape(name.c_str(), contextIndex, nvinfer1::OptProfileSelector::kMIN);
            auto const maxShape
                = mEngine->getProfileShape(name.c_str(), contextIndex, nvinfer1::OptProfileSelector::kMAX);

            TLLM_THROW("Tensor '%s' has invalid shape %s, expected in range min %s, max %s", name.c_str(),
                ITensor::toString(tensorShape).c_str(), ITensor::toString(minShape).c_str(),
                ITensor::toString(maxShape).c_str());
        }
        auto* const data = tensor->data();
        if (static_cast<bool>(data))
        {
            context.setInputTensorAddress(name.c_str(), data);
        }
        else
        {
            TLLM_CHECK_WITH_INFO(tensor->getSize() == 0, std::string("Invalid data for tensor: ") + name);
            // TensorRT runtime does not support nullptr.
            if (!mDummyTensor)
            {
                mDummyTensor = mBufferManager.gpu(ITensor::makeShape({1}));
            }
            context.setInputTensorAddress(name.c_str(), mDummyTensor->data());
        }
    }
}

void TllmRuntime::setStaticInputTensors(TensorMap const& tensorMap)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_FUNC_RANGE();

    TLLM_CHECK_WITH_INFO(getNbContexts() > 0, "Contexts should be created before calling setStaticInputTensors");
    for (auto contextIndex = 0; contextIndex < getNbContexts(); ++contextIndex)
    {
        setInputTensorsImpl(contextIndex, tensorMap, false);
    }

    // move static input tensor names to separate vector
    auto const begin = mInputTensorNames.begin();
    auto end = mInputTensorNames.end();
    for (auto const& [name, tensor] : tensorMap)
    {
        end = std::remove(begin, end, name);
    }
    mInputTensorNames.erase(end, mInputTensorNames.end());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TllmRuntime::setInputTensors(SizeType32 contextIndex, TensorMap const& tensorMap)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_FUNC_RANGE();
    setInputTensorsImpl(contextIndex, tensorMap, true);

    auto& context = getContext(contextIndex);
    if (mUseShapeInference)
    {
        NVTX3_SCOPED_RANGE(infer_shapes);
        char const* missing = nullptr;
        auto const nbMissing = context.inferShapes(1, &missing);
        if (nbMissing > 0)
        {
            TLLM_THROW("Input shape not specified: %s", missing);
        }
        else if (nbMissing < 0)
        {
            TLLM_THROW("Invalid input shape");
        }
    }

    {
        NVTX3_SCOPED_RANGE(final_checks);
        TLLM_CHECK_WITH_INFO(context.allInputDimensionsSpecified(), "Input dimensions not specified");
        TLLM_CHECK_WITH_INFO(context.allInputShapesSpecified(), "Input shapes not specified");
    }

    // Print shape of input / output tensors for the TRT engine
    if (tensorrt_llm::common::Logger::getLogger()->isEnabled(tensorrt_llm::common::Logger::TRACE))
    {
        printContextInfo(contextIndex);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TllmRuntime::setOutputTensors(SizeType32 contextIndex, TensorMap& tensorMap)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_FUNC_RANGE();
    if (isUserBufferEnabled())
    {
        // This function will identify the output tensors in the network that need to be bound as UB buffers
        // and bind the corresponding buffers to them based on their names.
        setUserBufferTensors(contextIndex, tensorMap);
    }

    auto& context = getContext(contextIndex);
    for (auto const& name : mOutputTensorNames)
    {
        auto const engineDtype = mEngine->getTensorDataType(name.c_str());
        auto const pos = tensorMap.find(name);
        if (pos != tensorMap.end())
        {
            auto const& tensor = pos->second;
            auto const tensorDtype = tensor->getDataType();
            // WAR: TRT does not support mixed FP8 and FP16 input, so engine expects FP16 tensors.
            TLLM_CHECK_WITH_INFO(tensorDtype == engineDtype
                    || (tensorDtype == nvinfer1::DataType::kFP8 && engineDtype == nvinfer1::DataType::kHALF),
                "%s: expected type %d, provided type %d", name.c_str(), static_cast<std::int32_t>(engineDtype),
                static_cast<std::int32_t>(tensorDtype));

            if (mUseShapeInference)
            {
                auto const dims = context.getTensorShape(name.c_str());
                tensor->reshape(dims);
            }
            context.setTensorAddress(name.c_str(), tensor->data());
        }
        else if (mUseShapeInference)
        {
            auto const dims = context.getTensorShape(name.c_str());
            auto tensor = ITensor::SharedPtr(mBufferManager.gpu(dims, engineDtype));
            tensorMap.insert(pos, std::make_pair(name, tensor));
            context.setTensorAddress(name.c_str(), tensor->data());
        }
        else
        {
            TLLM_THROW("Tensor %s is not found in tensorMap and shape inference is not allowed", name.c_str());
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TllmRuntime::setUserBufferTensors(SizeType32 contextIndex, TensorMap& tensorMap)
{
    auto startsWith = [](std::string const& str, std::string const& prefix) -> bool
    { return str.size() > prefix.size() && str.compare(0, prefix.size(), prefix) == 0; };
    std::string const prefix(tensorrt_llm::runtime::ub::tensor_prefix);
    auto& context = getContext(contextIndex);
    for (auto const& name : mOutputTensorNames)
    {
        auto const pos = tensorMap.find(name);
        if (pos != tensorMap.end() || !startsWith(name, prefix))
        {
            continue;
        }
        auto const engineDtype = mEngine->getTensorDataType(name.c_str());
        auto const dims = context.getTensorShape(name.c_str());
        void* ubBuffer = nullptr;
        if (name[prefix.size()] == '0')
        {
            ubBuffer = tensorrt_llm::runtime::ub::ub_get(0).addr;
        }
        else if (name[prefix.size()] == '1')
        {
            ubBuffer = tensorrt_llm::runtime::ub::ub_get(1).addr;
        }
        else if (name[prefix.size()] == '2')
        {
            ubBuffer = tensorrt_llm::runtime::ub::ub_get(2).addr;
        }
        else
        {
            TLLM_CHECK(false);
        }
        auto tensor = ITensor::SharedPtr(ITensor::wrap(ubBuffer, engineDtype, dims));
        tensorMap.insert(pos, std::make_pair(name, tensor));
        context.setTensorAddress(name.c_str(), ubBuffer);
    }
}

void TllmRuntime::initializeUserBuffer(tensorrt_llm::runtime::WorldConfig const& world_config, SizeType32 maxBatchSize,
    SizeType32 maxBeamWidth, SizeType32 maxSequenceLength, SizeType32 hiddenSize,
    std::optional<SizeType32> maxNumTokens)
{
    auto startsWith = [](std::string const& str, std::string const& prefix) -> bool
    { return str.size() > prefix.size() && str.compare(0, prefix.size(), prefix) == 0; };
    std::string const prefix(tensorrt_llm::runtime::ub::tensor_prefix);
    bool useNVFP4Model = false;
    for (auto const& name : mOutputTensorNames)
    {
        if (startsWith(name, prefix))
        {
            mUserBufferEnabled = true;
            if (name[prefix.size()] == '2')
            {
                useNVFP4Model = true;
                break;
            }
        }
    }
    if (!mUserBufferEnabled)
    {
        return;
    }
    // The hidden size returned by ModelConfig is the real hidden size divided by the TP size.
    auto const tpSize = world_config.getTensorParallelism();
    size_t const realHiddenSize = hiddenSize * tpSize;
    size_t const tokensNum = maxNumTokens.value_or(maxBatchSize * maxBeamWidth * maxSequenceLength);
    TLLM_CHECK(tokensNum > 0);
    size_t const elemNum = tokensNum * realHiddenSize;
    TLLM_LOG_INFO("[UserBuffer] MaxBatchSize %d, maxBeamWidth %d, maxSequenceLength %d, maxNumTokens %d, select %lu",
        maxBatchSize, maxBeamWidth, maxSequenceLength, maxNumTokens.has_value() ? maxNumTokens.value() : 0, tokensNum);
    tensorrt_llm::runtime::ub::ub_initialize(world_config);
    tensorrt_llm::runtime::ub::ub_allocate(elemNum * sizeof(half));
    tensorrt_llm::runtime::ub::ub_allocate(elemNum * sizeof(half));
    if (useNVFP4Model)
    {
        tensorrt_llm::runtime::ub::ub_allocate(elemNum * sizeof(uint8_t) / 16);
    }
}

CudaStream const& TllmRuntime::getStream() const
{
    return *mStream;
}

bool TllmRuntime::hasLayerProfiler(SizeType32 contextId) const
{
    return mContexts[contextId]->getProfiler() != nullptr;
}

void TllmRuntime::setLayerProfiler()
{
    mLayerProfiler = std::make_unique<LayerProfiler>();
    for (auto& context : mContexts)
    {
        context->setProfiler(mLayerProfiler.get());
        context->setEnqueueEmitsProfile(false);
    }
}

std::string TllmRuntime::getLayerProfileInfo() const
{
    TLLM_CHECK(mLayerProfiler);
    return mLayerProfiler->getLayerProfile();
}

void TllmRuntime::reportToProfiler(SizeType32 contextId)
{
    mContexts[contextId]->reportToProfiler();
}

void TllmRuntime::loadManagedWeights(RawEngine const& rawEngine, int localRank)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_FUNC_RANGE();
    auto& engine = getEngine();
    auto& manager = getBufferManager();
    if (rawEngine.getManagedWeightsMapOpt().has_value())
    {
        TLLM_LOG_DEBUG("Loading managed weights from raw engine");
        auto executorMap = rawEngine.getManagedWeightsMapOpt().value();
        for (auto const& [name, weight] : executorMap)
        {
            TLLM_LOG_DEBUG("Loading managed weight: %s", name.c_str());
            auto iTensor = tensorrt_llm::executor::detail::toITensor(weight);
            auto weightsDevice = std::shared_ptr<ITensor>{manager.copyFrom(*iTensor, MemoryType::kGPU)};
            mManagedWeightsMap.insert(std::make_pair(name, weightsDevice));
        }
    }
    else
    {
        TLLM_LOG_DEBUG("Loading managed weights from file");
        auto const enginePath = rawEngine.getPathOpt();
        TLLM_CHECK_WITH_INFO(enginePath.has_value(), "Engine path is not set.");
        auto weightPath
            = enginePath->parent_path() / ("rank" + std::to_string(localRank) + "_managed_weights.safetensors");
        auto managed_weights = common::safetensors::ISafeTensor::open(weightPath.string().c_str());
        for (auto const& name : managed_weights->keys())
        {
            TLLM_LOG_DEBUG("Loading managed weight: %s", name.c_str());
            auto const weight = managed_weights->getTensor(name.c_str());
            TLLM_CHECK(weight->dtype() == engine.getTensorDataType(name.c_str()));
            auto weightsDevice
                = std::shared_ptr<ITensor>{manager.allocate(MemoryType::kGPU, weight->trtDims(), weight->dtype())};
            manager.copy(weight->data(), *weightsDevice, MemoryType::kCPU);
            mManagedWeightsMap.insert(std::make_pair(name, weightsDevice));
        }
    }
    setStaticInputTensors(mManagedWeightsMap);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

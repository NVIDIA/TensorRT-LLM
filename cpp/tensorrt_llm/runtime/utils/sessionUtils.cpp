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

#include "sessionUtils.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

#include <algorithm>
#include <cassert>
#include <cstddef>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::runtime::utils
{
int initDevice(WorldConfig const& worldConfig)
{
    auto const device = worldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));
    return device;
}

// follows https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/common/sampleEngines.cpp
std::vector<uint8_t> loadEngine(std::string const& enginePath)
{
    std::ifstream engineFile(enginePath, std::ios::binary);
    TLLM_CHECK_WITH_INFO(engineFile.good(), std::string("Error opening engine file: " + enginePath));
    engineFile.seekg(0, std::ifstream::end);
    auto const size = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<uint8_t> engineBlob(size);
    engineFile.read(reinterpret_cast<char*>(engineBlob.data()), size);
    TLLM_CHECK_WITH_INFO(engineFile.good(), std::string("Error loading engine file: " + enginePath));
    return engineBlob;
}

std::vector<ITensor::SharedPtr> createBufferVector(TllmRuntime const& runtime, SizeType32 const indexOffset,
    SizeType32 const numBuffers, std::string const& prefix, MemoryType memType)
{
    auto const& manager = runtime.getBufferManager();
    auto const& engine = runtime.getEngine();

    std::vector<ITensor::SharedPtr> vector;

    for (SizeType32 i = indexOffset; i < indexOffset + numBuffers; ++i)
    {
        std::string name{prefix + std::to_string(i)};
        auto type = engine.getTensorDataType(name.c_str());
        vector.emplace_back(manager.emptyTensor(memType, type));
    }
    return vector;
}

std::vector<ITensor::SharedPtr> createBufferVector(
    TllmRuntime const& runtime, SizeType32 const numBuffers, MemoryType const memType, nvinfer1::DataType const dtype)
{
    auto const& manager = runtime.getBufferManager();

    std::vector<ITensor::SharedPtr> vector;

    for (SizeType32 i = 0; i < numBuffers; ++i)
    {
        vector.emplace_back(manager.emptyTensor(memType, dtype));
    }
    return vector;
}

void reshapeBufferVector(std::vector<ITensor::SharedPtr>& vector, nvinfer1::Dims const& shape)
{
    for (auto& buffer : vector)
    {
        buffer->reshape(shape);
    }
}

void assertNoVGQA(ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    auto [numKvHeadsPerLayerBegin, numKvHeadsPerLayerEnd] = modelConfig.getNumKvHeadsPerLayerLocalRange(
        worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank());
    TLLM_CHECK_WITH_INFO(std::all_of(numKvHeadsPerLayerBegin, numKvHeadsPerLayerEnd,
                             [firstNumKvHeads = *numKvHeadsPerLayerBegin](SizeType32 numKvHeads)
                             { return numKvHeads == firstNumKvHeads; }),
        "Deprecated session API does not support multiple cache pools, use the newer executor API instead");
}

std::vector<ITensor::SharedPtr> sliceBufferVector(
    std::vector<ITensor::SharedPtr> const& vector, SizeType32 const offset, SizeType32 const size)
{
    return transformVector(
        vector, [offset, size](auto const& buffer) { return std::shared_ptr{ITensor::slice(buffer, offset, size)}; });
}

void insertTensorVector(StringPtrMap<ITensor>& map, std::string const& key, std::vector<ITensor::SharedPtr> const& vec,
    SizeType32 indexOffset, std::vector<ModelConfig::LayerType> const& layerTypes, ModelConfig::LayerType type)
{
    if (layerTypes.empty())
    {
        for (std::size_t i = 0; i < vec.size(); ++i)
            map.insert_or_assign(key + std::to_string(indexOffset + i), vec[i]);
    }
    else
    {
        std::size_t vecIndex = 0;
        for (std::size_t i = 0; i < layerTypes.size(); ++i)
        {
            if (layerTypes[i] == type)
            {
                map.insert_or_assign(key + std::to_string(indexOffset + i), vec.at(vecIndex++));
            }
        }
    }
}

void insertTensorSlices(
    StringPtrMap<ITensor>& map, std::string const& key, ITensor::SharedPtr const& tensor, SizeType32 const indexOffset)
{
    auto const numSlices = tensor->getShape().d[0];
    for (SizeType32 i = 0; i < numSlices; ++i)
    {
        ITensor::SharedPtr slice = ITensor::slice(tensor, i, 1);
        slice->squeeze(0);
        map.insert_or_assign(key + std::to_string(indexOffset + i), slice);
    }
}

void printTensorMap(std::ostream& stream, StringPtrMap<ITensor> const& map)
{
    for (auto const& [name, tensor] : map)
    {
        stream << "Tensor name: " << name << '\n';
        stream << "Shape" << tensor->getShape() << '\n';
        stream << *tensor << '\n';
    }
}

void setRawPointers(ITensor& pointers, ITensor::SharedPtr const& input, int32_t pointersSlot, int32_t inputSlot)
{
    auto const pointersLength = static_cast<int32_t>(pointers.getSizeInBytes() / sizeof(void**));
    TLLM_CHECK_WITH_INFO(pointersSlot < pointersLength,
        tc::fmtstr("Pointer slot (%d) out of range [0,%d].", pointersSlot, pointersLength - 1));

    auto const inputSliced = ITensor::slice(input, inputSlot);
    auto pointersPtr = static_cast<void const**>(pointers.data());
    pointersPtr[pointersSlot] = inputSliced->data();
}

void setRawPointers(ITensor& pointers, ITensor::SharedPtr const& input)
{
    auto const& inputRows = input->getShape().d[0];
    auto const pointersLength = static_cast<int32_t>(pointers.getSizeInBytes() / sizeof(void**));
    TLLM_CHECK_WITH_INFO(inputRows == pointersLength,
        tc::fmtstr("Input dim 0 (" FMT_DIM ") does not match pointers length (%d).", inputRows, pointersLength));

    for (SizeType32 inputSlot = 0; inputSlot < inputRows; ++inputSlot)
    {
        setRawPointers(pointers, input, inputSlot, inputSlot);
    }
}

void tileBufferReplace(ITensor::SharedPtr& tensor, SizeType32 beamWidth, BufferManager& manager)
{
    if (tensor)
    {
        auto& stream = manager.getStream();
        auto shape = tensor->getShape();
        shape.d[0] *= beamWidth;
        auto tiledTensor = std::shared_ptr(manager.gpu(shape, tensor->getDataType()));
        kernels::tileTensor(*tiledTensor, *tensor, beamWidth, stream);
        stream.synchronize();
        tensor = tiledTensor;
    }
}

namespace
{
template <typename T>
void tileCpuBufferReplaceImpl(ITensor::SharedPtr& tensor, SizeType32 beamWidth)
{
    TLLM_CHECK(tensor != nullptr);
    TLLM_CHECK(tensor->getDataType() == TRTDataType<T>::value);
    auto shape = tensor->getShape();
    shape.d[0] *= beamWidth;

    ITensor::SharedPtr tiledTensor;
    switch (tensor->getMemoryType())
    {
    case MemoryType::kCPU: tiledTensor = std::shared_ptr(BufferManager::cpu(shape, tensor->getDataType())); break;
    case MemoryType::kPINNED: tiledTensor = std::shared_ptr(BufferManager::pinned(shape, tensor->getDataType())); break;
    case MemoryType::kPINNEDPOOL:
        tiledTensor = std::shared_ptr(BufferManager::pinnedPool(shape, tensor->getDataType()));
        break;
    default: TLLM_THROW("Tensor is not using CPU memory."); break;
    }
    auto const src = bufferCast<T>(*tensor);
    auto const dst = bufferCast<T>(*tiledTensor);
    TLLM_CHECK(tensor->getSize() * beamWidth == tiledTensor->getSize());
    for (size_t i = 0; i < tensor->getSize(); i++)
    {
        std::fill_n(dst + beamWidth * i, beamWidth, src[i]);
    }
    tensor = tiledTensor;
}
} // namespace

void tileCpuBufferReplace(ITensor::SharedPtr& tensor, SizeType32 beamWidth)
{
    if (tensor)
    {
        switch (tensor->getDataType())
        {
        case nvinfer1::DataType::kINT32: tileCpuBufferReplaceImpl<int32_t>(tensor, beamWidth); break;
        case nvinfer1::DataType::kFLOAT: tileCpuBufferReplaceImpl<float>(tensor, beamWidth); break;
        case nvinfer1::DataType::kHALF: tileCpuBufferReplaceImpl<half>(tensor, beamWidth); break;
        case nvinfer1::DataType::kINT8: tileCpuBufferReplaceImpl<int8_t>(tensor, beamWidth); break;
        case nvinfer1::DataType::kBOOL: tileCpuBufferReplaceImpl<bool>(tensor, beamWidth); break;
        case nvinfer1::DataType::kUINT8: tileCpuBufferReplaceImpl<uint8_t>(tensor, beamWidth); break;
        case nvinfer1::DataType::kINT64: tileCpuBufferReplaceImpl<int64_t>(tensor, beamWidth); break;
        default: TLLM_THROW("unsupported data type");
        }
    }
}

} // namespace tensorrt_llm::runtime::utils

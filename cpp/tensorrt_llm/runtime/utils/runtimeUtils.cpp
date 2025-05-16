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

#include "runtimeUtils.h"

#include "tensorrt_llm/common/assert.h"

#include <cassert>
#include <cstddef>

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

} // namespace tensorrt_llm::runtime::utils

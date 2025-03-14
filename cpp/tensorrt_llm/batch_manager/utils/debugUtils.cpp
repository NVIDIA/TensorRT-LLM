/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "debugUtils.h"

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace tensorrt_llm::batch_manager::utils
{
using executor::IterationType;
using runtime::ITensor;
using TensorPtr = runtime::ITensor::SharedPtr;
using TensorMap = runtime::ITensor::TensorMap;
using runtime::BufferManager;

namespace
{

fs::path getOutputPath(IterationType iterCounter, runtime::WorldConfig const& worldConfig)
{
    auto tmpPath = fs::temp_directory_path();
    return tmpPath / "tllm_debug"                                        //
        / ("PP_" + std::to_string(worldConfig.getPipelineParallelism())) //
        / ("TP_" + std::to_string(worldConfig.getTensorParallelism()))   //
        / ("iteration_" + std::to_string(iterCounter));
}

void dumpTensor(
    fs::path const& outputPath, std::string const& tensorName, ITensor const& tensor, BufferManager const& manager)
{
    fs::create_directories(outputPath);
    auto const outputFile = outputPath / (tensorName + ".npy");
    TLLM_LOG_INFO("Saving tensor '%s' to '%s'", tensorName.c_str(), outputFile.c_str());
    runtime::utils::saveNpy(manager, tensor, outputFile.string());
}

template <class TensorConsumer>
void forEachDebugTensor(std::vector<std::string> const& debugTensorNames, TensorMap const& inputMap,
    TensorMap const& outputMap, TensorConsumer tensorConsumer)
{
    for (auto const& debugTensorName : debugTensorNames)
    {
        auto foundTensor = false;
        for (auto const& tensorMap : {inputMap, outputMap})
        {
            auto tensorIt = tensorMap.find(debugTensorName);
            if (tensorIt != tensorMap.end())
            {
                auto const& [tensorName, tensor] = *tensorIt;
                tensorConsumer(tensorName, tensor);
                foundTensor = true;
            }
        }
        if (!foundTensor)
        {
            TLLM_LOG_WARNING("Debug tensor with name '%s' not found", debugTensorName.c_str());
        }
    }
}

template <class TensorConsumer>
void forEachTensor(executor::DebugConfig const& debugConfig, TensorPtr const& requestIds, TensorMap const& inputMap,
    TensorMap const& outputMap, TensorConsumer tensorConsumer)
{
    tensorConsumer(std::string("request_ids"), requestIds);

    if (debugConfig.getDebugTensorNames().empty())
    {
        if (debugConfig.getDebugInputTensors())
        {
            for (auto const& [tensorName, tensor] : inputMap)
            {
                tensorConsumer(tensorName, tensor);
            }
        }
        if (debugConfig.getDebugOutputTensors())
        {
            for (auto const& [tensorName, tensor] : outputMap)
            {
                tensorConsumer(tensorName, tensor);
            }
        }
    }
    else
    {
        forEachDebugTensor(debugConfig.getDebugTensorNames(), inputMap, outputMap, tensorConsumer);
    }
}

} // namespace

void dumpTensor(IterationType iterCounter, std::string const& tensorName, ITensor const& tensor,
    runtime::WorldConfig const& worldConfig, BufferManager const& manager)
{
    auto const outputPath = getOutputPath(iterCounter, worldConfig);
    dumpTensor(outputPath, tensorName, tensor, manager);
}

void dumpTensors(IterationType iterCounter, TensorMap const& tensorMap, runtime::WorldConfig const& worldConfig,
    BufferManager const& manager)
{
    auto const outputPath = getOutputPath(iterCounter, worldConfig);

    for (auto const& [tensorName, tensor] : tensorMap)
    {
        dumpTensor(outputPath, tensorName, *tensor, manager);
    }
}

void dumpDebugTensors(IterationType iterCounter, std::vector<std::string> const& debugTensorNames,
    TensorMap const& inputMap, TensorMap const& outputMap, runtime::WorldConfig const& worldConfig,
    BufferManager const& manager)
{
    auto dumpTensorFunc = [outputPath = getOutputPath(iterCounter, worldConfig), &manager](
                              std::string const& tensorName, TensorPtr const& tensor)
    { dumpTensor(outputPath, tensorName, *tensor, manager); };

    forEachDebugTensor(debugTensorNames, inputMap, outputMap, dumpTensorFunc);
}

void dumpIOTensors(executor::DebugConfig const& debugConfig, IterationType iterCounter, TensorPtr const& requestIds,
    TensorMap const& inputMap, TensorMap const& outputMap, runtime::WorldConfig const& worldConfig,
    BufferManager const& manager)
{
    auto dumpTensorFunc = [outputPath = getOutputPath(iterCounter, worldConfig), &manager](
                              std::string const& tensorName, TensorPtr const& tensor)
    { dumpTensor(outputPath, tensorName, *tensor, manager); };

    forEachTensor(debugConfig, requestIds, inputMap, outputMap, dumpTensorFunc);
}

TensorMap storeIOTensors(executor::DebugConfig const& debugConfig, TensorPtr const& requestIds,
    TensorMap const& inputMap, TensorMap const& outputMap, BufferManager const& manager)
{
    TensorMap tensors;

    auto storeTensor = [&tensors, &manager](std::string const& tensorName, TensorPtr const& tensor)
    {
        TensorPtr tensorCopy = manager.copyFrom(*tensor, tensor->getMemoryType());
        tensors.emplace(tensorName, tensorCopy);
    };

    forEachTensor(debugConfig, requestIds, inputMap, outputMap, storeTensor);

    return tensors;
}

template <typename T>
void writeBinArray(std::string const& filename, T const* tensor, const int64_t size, cudaStream_t stream)
{
    // write the tensor into a binary file. Can load from python by using
    // np.fromfile(filename, dtype=np_type), where np_type is np.float16 when T is half, and so on.
    TLLM_LOG_ERROR("%s start, size: %ld", __PRETTY_FUNCTION__, size);
    std::ofstream outfile(filename, std::ios::binary);
    TLLM_CHECK_WITH_INFO(
        outfile, tensorrt_llm::common::fmtstr("Failed to open file for writing: %s", filename.c_str()));

    std::vector<char> hostTensor(size * sizeof(T));
    tensorrt_llm::common::check_cuda_error(
        cudaMemcpyAsync(hostTensor.data(), tensor, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    tensorrt_llm::common::check_cuda_error(cudaStreamSynchronize(stream));
    // Write the tensor data
    outfile.write(reinterpret_cast<char const*>(hostTensor.data()), size * sizeof(T));

    outfile.close();
}

template void writeBinArray(std::string const& filename, half const* tensor, const int64_t size, cudaStream_t stream);
template void writeBinArray(std::string const& filename, float const* tensor, const int64_t size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void writeBinArray(
    std::string const& filename, __nv_bfloat16 const* tensor, const int64_t size, cudaStream_t stream);
#endif

} // namespace tensorrt_llm::batch_manager::utils

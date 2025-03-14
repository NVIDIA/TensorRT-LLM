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

#pragma once

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime
{
class TllmRuntime;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager::utils
{

/// @brief Dump tensor to an npy file.
/// @details Files are written to a temp directory determined by `std::filesystem::temp_directory_path`,
/// the exact paths are logged at INFO level.
void dumpTensor(executor::IterationType iterCounter, std::string const& tensorName, runtime::ITensor const& tensor,
    runtime::WorldConfig const& worldConfig, runtime::BufferManager const& manager);

/// @brief Dump each tensor in the `tensorMap` to an npy file.
/// @details Files are written to a temp directory determined by `std::filesystem::temp_directory_path`,
/// the exact paths are logged at INFO level.
void dumpTensors(executor::IterationType iterCounter, runtime::ITensor::TensorMap const& tensorMap,
    runtime::WorldConfig const& worldConfig, runtime::BufferManager const& manager);

/// @brief Dump each tensor in `debugTensorNames` to an npy file if it is found in one of the tensor maps.
/// @details Files are written to a temp directory determined by `std::filesystem::temp_directory_path`,
/// the exact paths are logged at INFO level.
void dumpDebugTensors(executor::IterationType iterCounter, std::vector<std::string> const& debugTensorNames,
    runtime::ITensor::TensorMap const& inputMap, runtime::ITensor::TensorMap const& outputMap,
    runtime::WorldConfig const& worldConfig, runtime::BufferManager const& manager);

/// @brief Dump engine IO tensors to files.
/// @details Files are written to a temp directory determined by `std::filesystem::temp_directory_path`,
/// the exact paths are logged at INFO level.
void dumpIOTensors(executor::DebugConfig const& debugConfig, executor::IterationType iterCounter,
    runtime::ITensor::SharedPtr const& requestIds, runtime::ITensor::TensorMap const& inputMap,
    runtime::ITensor::TensorMap const& outputMap, runtime::WorldConfig const& worldConfig,
    runtime::BufferManager const& manager);

/// @brief Store engine IO tensors in tensor map.
runtime::ITensor::TensorMap storeIOTensors(executor::DebugConfig const& debugConfig,
    runtime::ITensor::SharedPtr const& requestIds, runtime::ITensor::TensorMap const& inputMap,
    runtime::ITensor::TensorMap const& outputMap, runtime::BufferManager const& manager);

template <typename T>
void writeBinArray(std::string const& filename, T const* tensor, const int64_t size, cudaStream_t stream);

} // namespace tensorrt_llm::batch_manager::utils

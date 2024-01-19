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
#include "tensorrt_llm/runtime/iTensor.h"

#include <string>

namespace tensorrt_llm::runtime::utils
{

//! \brief Create new tensor from numpy file.
[[nodiscard]] ITensor::UniquePtr loadNpy(BufferManager& manager, const std::string& npyFile, const MemoryType where);

//! \brief Save tensor to numpy file.
void saveNpy(BufferManager& manager, ITensor const& tensor, const std::string& filename);

} // namespace tensorrt_llm::runtime::utils

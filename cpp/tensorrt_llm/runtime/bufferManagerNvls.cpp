/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/tllmBuffers.h"

namespace tensorrt_llm::runtime
{

BufferManager::ITensorPtr BufferManager::ipcNvls(
    std::set<int> ranks, tensorrt_llm::Dims dims, tensorrt_llm::DataType type)
{
    return std::make_unique<MulticastTensor>(dims, type, ranks);
}

} // namespace tensorrt_llm::runtime

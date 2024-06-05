/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <cstddef>
#include <cstdint>

#include "tensorrt_llm/common/assert.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace jit
{

template <typename T>
T readFromBuffer(uint8_t const*& buffer, size_t& remaining_buffer_size)
{
    TLLM_CHECK(sizeof(T) <= remaining_buffer_size);

    T result = *reinterpret_cast<T const*>(buffer);
    buffer += sizeof(T);
    remaining_buffer_size -= sizeof(T);
    return result;
}

template <typename T>
void writeToBuffer(T output, uint8_t*& buffer, size_t& remaining_buffer_size)
{
    TLLM_CHECK(sizeof(T) <= remaining_buffer_size);

    *reinterpret_cast<T*>(buffer) = output;
    buffer += sizeof(T);
    remaining_buffer_size -= sizeof(T);
}

} // namespace jit
} // namespace kernels
} // namespace tensorrt_llm

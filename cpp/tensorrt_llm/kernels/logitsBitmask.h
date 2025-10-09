/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeLogitsBitmask(
    T** logits, uint32_t const** bitmask, int32_t batchSize, int32_t vocabSizePadded, cudaStream_t stream);

template <typename T>
void invokeContiguousLogitsBitmask(T* logits, uint32_t const* bitmask, int32_t const* tokenMask, int32_t const* d2t,
    int32_t batchSize, int32_t vocabSizePadded, int32_t bitmaskSize, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm

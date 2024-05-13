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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm
{
namespace kernels
{

template <typename input_t>
size_t invokeComputeCumsumLastDimWorkspaceSize(int input_length);

template <typename input_t>
void invokeCumsumLastDim(int batch_size, int input_length, void const* __restrict__ input, void* __restrict__ output,
    void* workspace, size_t temp_storage_bytes, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeMinimaxM3SelectBlocks(float const* scores, int64_t headStride, int64_t blockStride, int64_t queryStride,
    int32_t const* nValidBlocks, int32_t* output, int32_t numKvHeads, int32_t numBlocks, int32_t totalQueries,
    int32_t initBlocks, int32_t localBlocks, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

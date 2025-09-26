/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <stdint.h>

namespace tensorrt_llm
{
namespace kernels
{

#ifdef __CUDACC__
#define ALIGN_256 __align__(256)
#else
#define ALIGN_256 alignas(256)
#endif

constexpr int WARP_SIZE = 32;
constexpr uint32_t WARP_MASK = 0xffffffff;

struct MoeEpWorldInfo
{
    int epSize;
    int epRank;
};

struct MoeExpertParallelInfo
{
    int expertCount = -1;
    int topK = 1;
};

} // namespace kernels
} // namespace tensorrt_llm

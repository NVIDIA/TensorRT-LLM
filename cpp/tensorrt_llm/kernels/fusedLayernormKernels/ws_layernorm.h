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
#include <cstdint>

namespace tensorrt_llm::kernels
{

struct WarpSpecializedCounters
{
    uint32_t tile_ctr;
    uint32_t cta_completion_ctr;
};

template <typename Param>
struct WarpSpecializedParam : public Param
{
    WarpSpecializedCounters* counters;
};

enum class SCALE_TYPE
{
    NONE,
    SCALAR,
    VECTOR,
    SIZE
};

template <typename T>
void invokeWSLayerNorm(WarpSpecializedParam<T> param, bool use_rms_norm, int ctas);

} // namespace tensorrt_llm::kernels

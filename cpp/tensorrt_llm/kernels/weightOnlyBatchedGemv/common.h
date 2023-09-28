/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace tensorrt_llm
{
namespace kernels
{
struct WeightOnlyParams
{
    const uint8_t* qweight;
    const half* scales;
    const half* zeros;
    const half* in;
    const half* bias;
    half* out;
    const int m;
    const int n;
    const int k;
    const int group_size;

    WeightOnlyParams(const uint8_t* _qweight, const half* _scales, const half* _zeros, const half* _in,
        const half* _bias, half* _out, const int _m, const int _n, const int _k, const int _group_size)
        : qweight(_qweight)
        , scales(_scales)
        , zeros(_zeros)
        , in(_in)
        , bias(_bias)
        , out(_out)
        , m(_m)
        , n(_n)
        , k(_k)
        , group_size(_group_size)
    {
    }
};
enum class WeightOnlyQuantType
{
    Int4b,
    Int8b
};
enum class WeightOnlyType
{
    PerChannel,
    GroupWise
};

struct WeightOnlyPerChannel;
template <int GS>
struct WeightOnlyGroupWise;

enum class WeightOnlyActivationType
{
    Gelu,
    Relu,
    Identity,
    InvalidType
};
} // namespace kernels
} // namespace tensorrt_llm

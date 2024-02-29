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
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#if defined(ENABLE_BF16)
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace tensorrt_llm
{
namespace kernels
{
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

enum class WeightOnlyActivationFunctionType
{
    Gelu,
    Relu,
    Identity,
    InvalidType
};

enum class WeightOnlyActivationType
{
    FP16,
    BF16
};

struct WeightOnlyParams
{
    // ActType is fp16 or bf16
    using ActType = void;
    using WeiType = uint8_t;

    const uint8_t* qweight;
    const ActType* scales;
    const ActType* zeros;
    const ActType* in;
    const ActType* act_scale;
    const ActType* bias;
    ActType* out;
    const int m;
    const int n;
    const int k;
    const int group_size;
    WeightOnlyQuantType quant_type;
    WeightOnlyType weight_only_type;
    WeightOnlyActivationFunctionType act_func_type;
    WeightOnlyActivationType act_type;

    WeightOnlyParams(const uint8_t* _qweight, const ActType* _scales, const ActType* _zeros, const ActType* _in,
        const ActType* _act_scale, const ActType* _bias, ActType* _out, const int _m, const int _n, const int _k,
        const int _group_size, const WeightOnlyQuantType _quant_type, const WeightOnlyType _weight_only_type,
        const WeightOnlyActivationFunctionType _act_func_type, const WeightOnlyActivationType _act_type)
        : qweight(_qweight)
        , scales(_scales)
        , zeros(_zeros)
        , in(_in)
        , act_scale(_act_scale)
        , bias(_bias)
        , out(_out)
        , m(_m)
        , n(_n)
        , k(_k)
        , group_size(_group_size)
        , quant_type(_quant_type)
        , weight_only_type(_weight_only_type)
        , act_func_type(_act_func_type)
        , act_type(_act_type)
    {
    }
};
} // namespace kernels
} // namespace tensorrt_llm

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
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace tensorrt_llm
{
namespace kernels
{
namespace weight_only
{
enum class KernelType
{
    W4A16,
    W4A8
};

struct Params
{
    void* act;
    void* act_scale;
    void* weight;
    void* scales;
    void* zeros;
    void* bias;
    void* out;
    float alpha;
    int m;
    int n;
    int k;
    int groupsize;
    KernelType type;

    Params(void* _act, void* _act_scale, void* _weight, void* _scales, void* _zeros, void* _bias, void* _out,
        float _alpha, int _m, int _n, int _k, int _groupsize, KernelType _type)
        : act(_act)
        , act_scale(_act_scale)
        , weight(_weight)
        , scales(_scales)
        , zeros(_zeros)
        , bias(_bias)
        , out(_out)
        , alpha(_alpha)
        , m(_m)
        , n(_n)
        , k(_k)
        , groupsize(_groupsize)
        , type(_type)
    {
    }
};
} // namespace weight_only
} // namespace kernels
} // namespace tensorrt_llm

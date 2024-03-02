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

#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/sm90/kernel.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/sm90/kernelLauncher.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace weight_only
{
#define DISPATCHER_FOR_M(target_m, CtaM, CtaN, Threads)                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (params.m == target_m)                                                                                      \
        {                                                                                                              \
            exec_kernel<Details, CtaM, CtaN, Threads, GroupSize, EnableActScale, EnableZero, EnableBias>(params, s);   \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0);

template <typename Details, int GroupSize, bool EnableActScale, bool EnableZero, bool EnableBias>
void dispatcher(Params& params, cudaStream_t s)
{
    // clang-format off
    DISPATCHER_FOR_M(1,  1,  8,  128);
    DISPATCHER_FOR_M(2,  2,  4,  128);
    DISPATCHER_FOR_M(3,  3,  16, 128);
    DISPATCHER_FOR_M(4,  4,  16, 128);
    DISPATCHER_FOR_M(5,  5,  16, 128);
    DISPATCHER_FOR_M(6,  6,  16, 128);
    DISPATCHER_FOR_M(7,  7,  16, 128);
    DISPATCHER_FOR_M(8,  8,  16, 128);
    DISPATCHER_FOR_M(9,  9,  8,  128);
    DISPATCHER_FOR_M(10, 10, 8,  128);
    DISPATCHER_FOR_M(11, 11, 8,  128);
    DISPATCHER_FOR_M(12, 12, 8,  128);
    DISPATCHER_FOR_M(13, 13, 8,  128);
    DISPATCHER_FOR_M(14, 14, 8,  128);
    DISPATCHER_FOR_M(15, 15, 8,  128);
    DISPATCHER_FOR_M(16, 16, 8,  128);
    // clang-format on
    throw std::runtime_error("unsupported m");
}

template <typename Details, int GroupSize>
void check_pointer(Params& params, cudaStream_t s)
{
    if (params.act_scale && params.zeros && params.bias)
    {
        dispatcher<Details, GroupSize, true, true, true>(params, s);
    }
    else if (params.act_scale && params.zeros && !params.bias)
    {
        dispatcher<Details, GroupSize, true, true, false>(params, s);
    }
    else if (params.act_scale && !params.zeros && params.bias)
    {
        dispatcher<Details, GroupSize, true, false, true>(params, s);
    }
    else if (!params.act_scale && params.zeros && params.bias)
    {
        dispatcher<Details, GroupSize, false, true, true>(params, s);
    }
    else if (!params.act_scale && !params.zeros && params.bias)
    {
        dispatcher<Details, GroupSize, false, false, true>(params, s);
    }
    else if (params.act_scale && !params.zeros && !params.bias)
    {
        dispatcher<Details, GroupSize, true, false, false>(params, s);
    }
    else if (!params.act_scale && params.zeros && !params.bias)
    {
        dispatcher<Details, GroupSize, false, true, false>(params, s);
    }
    else
    {
        dispatcher<Details, GroupSize, false, false, false>(params, s);
    }
}

template <typename Details>
void select_gs(Params& params, cudaStream_t s)
{
    if (params.groupsize == 64)
    {
        check_pointer<Details, 64>(params, s);
    }
    else if (params.groupsize == 128)
    {
        check_pointer<Details, 128>(params, s);
    }
}

void kernel_launcher(Params& params, cudaStream_t s)
{
    if (params.type == KernelType::W4A16)
    {
        select_gs<Fp16Details>(params, s);
    }
    else if (params.type == KernelType::W4A8)
    {
        select_gs<Fp8Details>(params, s);
    }
}
} // namespace weight_only
} // namespace kernels
} // namespace tensorrt_llm

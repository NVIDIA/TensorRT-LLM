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

#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/common.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/utility.h"

namespace tensorrt_llm
{
namespace kernels
{
template <WeightOnlyQuantType QType, typename WeightOnlyFlag, template <typename T> class ActOp, bool Zero, bool Bias,
    int N_PER_BLOCK, int BATCH, int BLOCK_SIZE>
struct WeightOnlyBatchedGemvKernelLauncher
{
    static void run(const WeightOnlyParams& params, cudaStream_t stream);
};

template <WeightOnlyQuantType QType, typename WeightOnlyFlag, template <typename T> class ActOp, int N_PER_BLOCK,
    int BATCH, int BLOCK_SIZE>
void select_zero_bias(const WeightOnlyParams& params, cudaStream_t stream)
{
    if (params.zeros && params.bias)
    {
        WeightOnlyBatchedGemvKernelLauncher<QType, WeightOnlyFlag, ActOp, true, true, N_PER_BLOCK, BATCH,
            BLOCK_SIZE>::run(params, stream);
    }
    else if (params.zeros && !params.bias)
    {
        WeightOnlyBatchedGemvKernelLauncher<QType, WeightOnlyFlag, ActOp, true, false, N_PER_BLOCK, BATCH,
            BLOCK_SIZE>::run(params, stream);
    }
    else if (!params.zeros && params.bias)
    {
        WeightOnlyBatchedGemvKernelLauncher<QType, WeightOnlyFlag, ActOp, false, true, N_PER_BLOCK, BATCH,
            BLOCK_SIZE>::run(params, stream);
    }
    else
    {
        WeightOnlyBatchedGemvKernelLauncher<QType, WeightOnlyFlag, ActOp, false, false, N_PER_BLOCK, BATCH,
            BLOCK_SIZE>::run(params, stream);
    }
}

template <WeightOnlyQuantType QType, typename WeightOnlyFlag, int N_PER_BLOCK, int BATCH, int BLOCK_SIZE>
void select_activation(WeightOnlyActivationType atype, const WeightOnlyParams& params, cudaStream_t stream)
{
    switch (atype)
    {
    case WeightOnlyActivationType::Gelu:
    {
        select_zero_bias<QType, WeightOnlyFlag, GeluActivation, N_PER_BLOCK, BATCH, BLOCK_SIZE>(params, stream);
        break;
    }
    case WeightOnlyActivationType::Relu:
    {
        select_zero_bias<QType, WeightOnlyFlag, ReluActivation, N_PER_BLOCK, BATCH, BLOCK_SIZE>(params, stream);
        break;
    }
    case WeightOnlyActivationType::Identity:
    {
        select_zero_bias<QType, WeightOnlyFlag, IdentityActivation, N_PER_BLOCK, BATCH, BLOCK_SIZE>(params, stream);
        break;
    }
    default:
    {
        throw std::runtime_error("Use unsupported activation");
        break;
    }
    }
}

template <typename WeightOnlyFlag, int N_PER_BLOCK, int BATCH, int BLOCK_SIZE>
void select_quant_type(
    WeightOnlyQuantType qtype, WeightOnlyActivationType atype, const WeightOnlyParams& params, cudaStream_t stream)
{
    if (qtype == WeightOnlyQuantType::Int4b)
    {
        select_activation<WeightOnlyQuantType::Int4b, WeightOnlyFlag, N_PER_BLOCK, BATCH, BLOCK_SIZE>(
            atype, params, stream);
    }
    else if (qtype == WeightOnlyQuantType::Int8b)
    {
        select_activation<WeightOnlyQuantType::Int8b, WeightOnlyFlag, N_PER_BLOCK, BATCH, BLOCK_SIZE>(
            atype, params, stream);
    }
    else
    {
        throw std::runtime_error("Unknown QuantType");
    }
}

template <int N_PER_BLOCK, int BATCH, int BLOCK_SIZE>
void select_groupwise_weight_only(WeightOnlyQuantType qtype, WeightOnlyType wtype, WeightOnlyActivationType atype,
    const WeightOnlyParams& params, cudaStream_t stream)
{
    if (wtype == WeightOnlyType::GroupWise && params.group_size == 64)
    {
        select_quant_type<WeightOnlyGroupWise<64>, N_PER_BLOCK, BATCH, BLOCK_SIZE>(qtype, atype, params, stream);
    }
    else if (wtype == WeightOnlyType::GroupWise && params.group_size == 128)
    {
        select_quant_type<WeightOnlyGroupWise<128>, N_PER_BLOCK, BATCH, BLOCK_SIZE>(qtype, atype, params, stream);
    }
    else
    {
        throw std::runtime_error("Only support groupwise weight only for gs=64/128");
    }
}

void weight_only_batched_gemv_launcher(WeightOnlyQuantType qtype, WeightOnlyType wtype, WeightOnlyActivationType atype,
    const WeightOnlyParams& params, cudaStream_t stream)
{
    if (wtype == WeightOnlyType::PerChannel)
    {
        if (qtype == WeightOnlyQuantType::Int4b)
        {
            switch (params.m)
            {
            case 1:
            {
                select_activation<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, 1, 1, 192>(atype, params, stream);
                break;
            }
            case 2:
            {
                select_activation<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, 2, 2, 128>(atype, params, stream);
                break;
            }
            case 3:
            {
                select_activation<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, 2, 3, 256>(atype, params, stream);
                break;
            }
            case 4:
            {
                select_activation<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, 4, 4, 256>(atype, params, stream);
                break;
            }
            default:
            {
                throw std::runtime_error("Weight only cuda kernel only supported bs <= 4");
                break;
            }
            }
        }
        else if (qtype == WeightOnlyQuantType::Int8b)
        {
            switch (params.m)
            {
            case 1:
            {
                select_activation<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, 2, 1, 256>(atype, params, stream);
                break;
            }
            case 2:
            {
                select_activation<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, 2, 2, 256>(atype, params, stream);
                break;
            }
            case 3:
            {
                select_activation<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, 2, 3, 256>(atype, params, stream);
                break;
            }
            case 4:
            {
                select_activation<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, 2, 4, 256>(atype, params, stream);
                break;
            }
            default:
            {
                throw std::runtime_error("Weight only cuda kernel only supported bs <= 4");
                break;
            }
            }
        }
    }
    else if (wtype == WeightOnlyType::GroupWise)
    {
        switch (params.m)
        {
        case 1:
        {
            select_groupwise_weight_only<2, 1, 256>(qtype, wtype, atype, params, stream);
            break;
        }
        case 2:
        {
            select_groupwise_weight_only<2, 2, 256>(qtype, wtype, atype, params, stream);
            break;
        }
        case 3:
        {
            select_groupwise_weight_only<2, 3, 128>(qtype, wtype, atype, params, stream);
            break;
        }
        case 4:
        {
            select_groupwise_weight_only<2, 4, 128>(qtype, wtype, atype, params, stream);
            break;
        }
        default:
        {
            throw std::runtime_error("Weight only cuda kernel only supported bs <= 4");
            break;
        }
        }
    }
}
} // namespace kernels
} // namespace tensorrt_llm

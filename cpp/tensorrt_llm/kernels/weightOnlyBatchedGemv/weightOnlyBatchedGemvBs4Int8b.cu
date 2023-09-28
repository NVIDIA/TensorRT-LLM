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

#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h"

namespace tensorrt_llm
{
namespace kernels
{

template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, GeluActivation,
    true, true, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, GeluActivation,
    true, false, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, GeluActivation,
    false, true, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, GeluActivation,
    false, false, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, ReluActivation,
    true, true, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, ReluActivation,
    true, false, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, ReluActivation,
    false, true, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, ReluActivation,
    false, false, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel,
    IdentityActivation, true, true, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel,
    IdentityActivation, true, false, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel,
    IdentityActivation, false, true, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel,
    IdentityActivation, false, false, 2, 4, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>, GeluActivation,
    true, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>, GeluActivation,
    true, false, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>, GeluActivation,
    false, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>, GeluActivation,
    false, false, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>, ReluActivation,
    true, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>, ReluActivation,
    true, false, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>, ReluActivation,
    false, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>, ReluActivation,
    false, false, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>,
    IdentityActivation, true, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>,
    IdentityActivation, true, false, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>,
    IdentityActivation, false, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<64>,
    IdentityActivation, false, false, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    GeluActivation, true, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    GeluActivation, true, false, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    GeluActivation, false, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    GeluActivation, false, false, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    ReluActivation, true, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    ReluActivation, true, false, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    ReluActivation, false, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    ReluActivation, false, false, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    IdentityActivation, true, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    IdentityActivation, true, false, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    IdentityActivation, false, true, 2, 4, 128>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, WeightOnlyGroupWise<128>,
    IdentityActivation, false, false, 2, 4, 128>;

} // namespace kernels
} // namespace tensorrt_llm

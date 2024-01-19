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

#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h"

namespace tensorrt_llm
{
namespace kernels
{

template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel,
    IdentityActivation, false, false, 1, 1, 192>;

template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
    IdentityActivation, true, true, 2, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
    IdentityActivation, true, false, 2, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
    IdentityActivation, false, true, 2, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
    IdentityActivation, false, false, 2, 1, 256>;

template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
    IdentityActivation, true, true, 2, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
    IdentityActivation, true, false, 2, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
    IdentityActivation, false, true, 2, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
    IdentityActivation, false, false, 2, 1, 256>;

} // namespace kernels
} // namespace tensorrt_llm

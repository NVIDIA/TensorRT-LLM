/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "fmhaRunnerParams.h"
#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

void runPrepareCustomMask(TllmGenFmhaRunnerParams const& runnerParams, FmhaKernelType kernelType, int32_t stepQ,
    int32_t stepKv, int32_t tileSizeQ, int32_t tileSizeKv, cudaStream_t stream);

//! Upper bound, in uint32 elements, of the custom mask one generation kernel configuration writes.
int64_t getCustomMaskSizeUpperBound(FmhaKernelType kernelType, int32_t batchSize, int32_t maxSeqLenQ,
    int32_t numHeadsQPerKv, int32_t stepQ, int32_t stepKv, int32_t tileSizeQ, int32_t tileSizeKv);

//! Custom mask capacity, in uint32 elements, that covers every generation custom-mask kernel configuration.
int64_t getCustomMaskCapacityUpperBound(int32_t maxBatchSize, int32_t maxSeqLenQ, int32_t numHeadsQPerKv);

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels

TRTLLM_NAMESPACE_END

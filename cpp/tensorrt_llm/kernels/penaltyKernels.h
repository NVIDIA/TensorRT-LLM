/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_fp16.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/penaltyTypes.h"

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
struct InvokeBatchApplyPenaltyParams
{
    T const* const* inputLogits;
    T* outputLogits;
    const T* biases;
    int* penaltyWorkspace;
    const int* penaltyWorkspacePrev;
    const float* temperatures;
    const float* repetitionPenalties;
    const float* presencePenalties;
    const float* frequencyPenalties;
    const bool accumulateVocab;
    const size_t batchSize;
    const int beamWidth;
    const int maxSeqLen;
    const size_t vocabSize;
    const size_t vocabSizePadded;
    const int** outputIdsPtr;
    const int** parentIdsPtr;
    const int* inputLengths;
    const int* sequenceLengths;
    const int* minLengths;
    const int* endIds;
    const int* batchSlots;
    cudaStream_t stream;
};

template <typename T>
void invokeBatchApplyPenalty(const InvokeBatchApplyPenaltyParams<T>& params);

} // namespace kernels
} // namespace tensorrt_llm

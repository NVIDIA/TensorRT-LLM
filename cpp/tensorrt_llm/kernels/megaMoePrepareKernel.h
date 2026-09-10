/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! Dtype of the token-selected expert index tensor consumed by invokeMegaMoePrepare.
enum class MegaMoePrepareExpertType
{
    INT32,
    INT64,
};

//! Dtype of the token-final scale tensor consumed by invokeMegaMoePrepare.
enum class MegaMoePrepareScaleType
{
    FP32,
    FP16,
    BF16,
};

//! Prepare DeepGEMM MegaMoE inputs by quantizing activations and copying routing metadata.
//!
//! Expected tensor-like arguments:
//! - input: [numTokens, hiddenSize] BF16 activations.
//! - tokenSelectedExperts: [numTokens, topK] INT32 or INT64 expert/slot ids.
//! - tokenFinalScales: [numTokens, topK] FP32, FP16, or BF16 routing scales.
//! - xOut: [>=numTokens, hiddenSize] FP8 E4M3 output activations.
//! - xSfOut: [>=numTokens, hiddenSize / 128] INT32 packed UE8M0 scales.
//! - topkIdxOut: [>=numTokens, topK] INT64 expert/slot ids.
//! - topkWeightsOut: [>=numTokens, topK] FP32 routing scales.
//!
//! hiddenSize must be divisible by 128. All pointers must refer to contiguous
//! CUDA buffers on the same device, and the kernel requires SM100 or newer.
void invokeMegaMoePrepare(void const* input, void const* tokenSelectedExperts, void const* tokenFinalScales, void* xOut,
    void* xSfOut, int64_t* topkIdxOut, float* topkWeightsOut, int numTokens, int hiddenSize, int topK,
    MegaMoePrepareExpertType expertType, MegaMoePrepareScaleType scaleType, int multiProcessorCount,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

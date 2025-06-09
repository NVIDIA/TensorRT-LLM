/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/quantization.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{

enum class FP4QuantizationSFLayout
{
    // Block scale factors are stored in swizzled layout for cutlass FP4 kernel. Scale factor
    // blocks are organized in 512-byte blocks in global memory, with each block having 128x4 FP8 values.
    // The SF matrix dimensions are therefore padded - rows to the nearest multiple of 128 and columns to
    // the nearest multiple of 4.
    //
    // The scale factor block rows map to data block rows in an interleaved pattern:
    // For a scale factor row 'i', it maps to data block row: (i % 4) * 32 + (i / 4)
    // Column 'j' in the scale factor block corresponds to scaling the j-th block in the data tensor.
    //
    // Please refer to https://nvbugs/4165523 for more details about the swizzled layout.
    SWIZZLED,
    // Block scale factors are stored in linear layout (row-major). This is used in some trtllm-gen kernels standard.
    LINEAR
};

#define PadUpFn(X, Y) ((X + Y - 1) / (Y) * (Y))

// totalCloumn should be in SFMatrix, not activation Matrix, so no sfVecSize needed.
inline int computeFP4SwizzledLayoutSFSize(int totalRow, int totalColumn)
{
    int paddedRow = PadUpFn(totalRow, 128);
    int paddedColumn = PadUpFn(totalColumn, 4);
    return paddedRow * paddedColumn;
}

inline int computeFP4LinearLayoutSFSize(int totalRow, int totalColumn)
{
    return totalRow * totalColumn;
}

namespace kernels
{

template <typename T>
void invokeQuantization(
    int8_t* dst, T const* src, int64_t const size, float const* scalePtr, cudaStream_t stream = 0, int maxGirdSize = 0);

template <typename T, typename QuantT>
void invokePerTokenQuantization(QuantT* dst, T const* src, int64_t const numRows, int64_t const numCols,
    float const* clampPtr, float* scalePtr, float* sumPtr, tensorrt_llm::common::QuantMode quantMode,
    cudaStream_t stream = 0);

template <typename T, int SF_VEC_SIZE = 16>
void invokeFP4Quantization(int m, int n, T const* input, float const* globalScale, int64_t* output, int32_t* SFOuput,
    bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream = 0);

template <typename T, int SF_VEC_SIZE = 16>
void invokeBatchedFP4Quantization(int b, int m, int n, T const* input, float const* globalScale, int64_t* output,
    int32_t* SFOuput, bool useUE8M0, int multiProcessorCount, cudaStream_t stream = 0);

void invokeNVFP4BlockScaleInterleave(
    int b, int m, int n, uint8_t const* SFIn, uint8_t* SFOutput, int multiProcessorCount, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tensorrt_llm

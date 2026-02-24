/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass/gemm_coord.h"
#include "tensorrt_llm/common/config.h"
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/**
 * @brief CUDA Graph compatible wrapper for grouped GEMM operations.
 *
 * This function accepts GPU pointers directly without any workspace for parameters,
 * making it fully compatible with CUDA Graph capture and replay.
 *
 * @param problemSizesPtr GPU pointer to array of cutlass::gemm::GemmCoord
 * @param problemCount Number of GEMM problems
 * @param ptrAGpu GPU pointer to array of A matrix pointers
 * @param ptrBGpu GPU pointer to array of B matrix pointers
 * @param ptrCGpu GPU pointer to array of C matrix pointers (can be nullptr)
 * @param ptrDGpu GPU pointer to array of D matrix pointers
 * @param isLoraIn Whether this is for LoRA input transformation
 * @param dataType Data type of the matrices
 * @param minKN Minimum K*N value for kernel selection
 * @param stream CUDA stream
 */
void cudaGraphGroupedGemm(cutlass::gemm::GemmCoord* problemSizesPtr, int problemCount, void** ptrAGpu, void** ptrBGpu,
    void** ptrCGpu, void** ptrDGpu, int64_t* ldaGpu, int64_t* ldbGpu, int64_t* ldcGpu, int64_t* lddGpu, bool isLoraIn,
    nvinfer1::DataType dataType, int minKN, cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr, cudaStream_t stream);

/**
 * @brief CUDA Graph compatible wrapper for split-K grouped GEMM operations.
 *
 * Similar to cudaGraphGroupedGemm but uses split-K algorithm for better
 * performance with certain problem sizes. No parameter workspace needed.
 */
void cudaGraphSplitKGroupedGemm(cutlass::gemm::GemmCoord* problemSizesPtr, int problemCount, void** ptrAGpu,
    void** ptrBGpu, void** ptrCGpu, void** ptrDGpu, int64_t* ldaGpu, int64_t* ldbGpu, int64_t* ldcGpu, int64_t* lddGpu,
    bool isLoraIn, nvinfer1::DataType dataType, int splitKSlices, int minKN,
    cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr, int64_t* splitKOffsetsGpu, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

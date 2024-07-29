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

#include "tensorrt_llm/runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tensorrt_llm::kernels::speculative_decoding
{

//! \brief Linearly packs accepted paths in memory according to the accceptedLengths and bestPathIds
//!
//! \param acceptedLengthsCumSum input buffer [maxBatchSize + 1], exclusive sum of accepted lengths
//! (indexed linearly in memory).
//! \param pathsOffsets input buffer [maxBatchSize * maxDraftLen], slices of accepted paths packed in memory
//! \param acceptedLengths input buffer [maxBatchSize], length of the data accepted tokens
//! \param bestPathIds input buffer [maxBatchSize], indices of the selected paths
//! \param paths input buffer [batchSize, numPaths, maxPathLen] if isPathsLinearBatchIdx else [maxBatchSize, numPaths,
//! maxPathLen], paths to restore sequences from outputIds and targetIds. Should be filled with -1 for everything that
//! is not path. \param batchSlots input buffer [batchSize], address map from local index to global index [0, batchSize]
//! -> [0, maxBatchSize] \param batchSize current batch size \param numPaths maximum number of tokens per step
//! configured in the system \param maxPathLen maximum sequence length of the sequence containing draft tokens \param
//! isPathsLinearBatchIdx \param stream stream
void invokePackAcceptedPaths(runtime::SizeType32* acceptedLengthsCumSum, runtime::SizeType32* pathsOffsets,
    runtime::SizeType32 const* acceptedLengths, runtime::SizeType32 const* bestPathIds,
    runtime::SizeType32 const* paths, runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize,
    runtime::SizeType32 numPaths, runtime::SizeType32 maxPathLen, bool isPathsLinearBatchIdx, cudaStream_t stream);

} // namespace tensorrt_llm::kernels::speculative_decoding

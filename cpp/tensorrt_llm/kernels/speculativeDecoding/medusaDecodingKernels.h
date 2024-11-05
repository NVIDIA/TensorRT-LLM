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

#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/speculativeDecoding/common.h"
#include "tensorrt_llm/runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tensorrt_llm::kernels::speculative_decoding
{

//! \brief assembles draft tokens to treeDraftIds from sourceDraftIds using indices of treeIds
//!
//! \param treeDraftIds output buffer [maxBatchSize, maxDecodingTokens-1], output draft tokens
//! scattered from sourceDraftIds according to treeIds111
//! \param sourceDraftIds input buffer [maxBatchSize, maxDecodingTokens], draft tokens saved leanearly after
//! sampling from Medusa heads with TopK.
//! \param treeIds input buffer [maxBatchSize, maxDecodingTokens-1], address map from sourceDraftIds to treeDraftIds
//! [0, unqiueDraftTokens] -> [0, maxDecodingTokens], where unqiueDraftTokens = sum(MedusaHeadsTopK)
//! unqiueDraftTokens <= maxDraftTokens
//! \param tokensPerStep input buffer [maxBatchSize], number of output draft tokens
//! \param batchSlots input buffer [maxBatchSize], address map from local index
//! to global index [0, batchSize] -> [0, maxBatchSize]
//! \param maxDecodingTokens maximum number of tokens per step configured in the system
//! \param batchSize current batch size
//! \param stream cuda stream
void scatterMedusaDraftTokens(runtime::TokenIdType* treeDraftIds, runtime::TokenIdType const* sourceDraftIds,
    runtime::SizeType32 const* treeIds, runtime::SizeType32 const* tokensPerStep, runtime::SizeType32 const* batchSlots,
    runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 batchSize, cudaStream_t stream);

} // namespace tensorrt_llm::kernels::speculative_decoding

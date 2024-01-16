/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#include <vector>

namespace tensorrt_llm::kernels::parallel_decoding
{

using IndexType = int;

void updateLinearKVCacheDraftTokenLocation(const int* seqAcceptedDraftTokenOffsets,
    const IndexType* packedAcceptedDraftTokensIndices, const int32_t* pastKeyValueLengths,
    int8_t* const* pastKeyValueList, int layerCount, int seqCount, int numKVHeads, int sizeInBytesPerKVHead,
    int rewindDraftTokenCount, int maxKVCacheLen, cudaStream_t stream);

void updateKVBlockArrayDraftTokenLocation(const int* seqAcceptedDraftTokenOffsets,
    const IndexType* packedAcceptedDraftTokensIndices, const int32_t* pastKeyValueLengths, int64_t* const* pointerArray,
    int layerCount, int seqCount, int numKVHeads, int sizeInBytesPerKVHead, int rewindDraftTokenCount,
    int maxKVCacheLen, int maxBlocksPerSeq, int tokensPerBlock, cudaStream_t stream);

} // namespace tensorrt_llm::kernels::parallel_decoding

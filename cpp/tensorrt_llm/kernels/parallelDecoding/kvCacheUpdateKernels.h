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

/*!
 * Update Linear KV cache using common rewind count.
 * @param seqAcceptedDraftTokenOffsets : Array of length seqCount + 1, like [0, 3, 5]
 * @param packedAcceptedDraftTokensIndices : Array of length seqAcceptedDraftTokenOffsets[seqCount], each value is in
 * range [0, maxDraftTokenCount - 1]
 * @param pastKeyValueLengths : Array of length seqCount, meaning how many tokens are already in KV cache
 * @param pastKeyValueList : Past key value list, which is the pointer array of each KVLinear cache.
 * @param layerCount : Count of layers
 * @param seqCount : Count of sequence
 * @param numKVHeads : Number of KV heads
 * @param sizeInBytesPerKVHead : Size of each KV head
 * @param rewindDraftTokenCount : Count to rewind
 * @param maxKVCacheLen : Maximum length of each KV cache
 * @param stream : CUDA stream to use.
 */
void updateLinearKVCacheDraftTokenLocationCommonRewind(const int* seqAcceptedDraftTokenOffsets,
    const IndexType* packedAcceptedDraftTokensIndices, const int32_t* pastKeyValueLengths,
    int8_t* const* pastKeyValueList, int layerCount, int seqCount, int numKVHeads, int sizeInBytesPerKVHead,
    int rewindDraftTokenCount, int maxKVCacheLen, cudaStream_t stream);

/*!
 * Update Block KV cache using common rewind count.
 * @param seqAcceptedDraftTokenOffsets : Array of length seqCount + 1, like [0, 3, 5]
 * @param packedAcceptedDraftTokensIndices : Array of length seqAcceptedDraftTokenOffsets[seqCount], each value is in
 * range [0, maxDraftTokenCount - 1]
 * @param pastKeyValueLengths : Array of length seqCount, meaning how many tokens are already in KV cache
 * @param pointerArray : Pointer array of each Block KV cache.
 * @param layerCount : Count of layers
 * @param seqCount : Count of sequence
 * @param numKVHeads : Number of KV heads
 * @param sizeInBytesPerKVHead : Size of each KV head
 * @param rewindDraftTokenCount : Count to rewind
 * @param maxKVCacheLen : Maximum length of each KV cache
 * @param maxBlocksPerSeq : Maximum blocks per sequence of Block KV cache.
 * @param tokensPerBlock : Tokens per block of Block KV cache
 * @param stream : CUDA stream to use.
 */
void updateKVBlockArrayDraftTokenLocationCommonRewind(const int* seqAcceptedDraftTokenOffsets,
    const IndexType* packedAcceptedDraftTokensIndices, const int32_t* pastKeyValueLengths, int64_t* const* pointerArray,
    int layerCount, int seqCount, int numKVHeads, int sizeInBytesPerKVHead, int rewindDraftTokenCount,
    int maxKVCacheLen, int maxBlocksPerSeq, int tokensPerBlock, cudaStream_t stream);

/*!
 * Update Linear KV cache using separate rewind count for each sequence.
 * @param seqAcceptedDraftTokenOffsets : Array of length seqCount + 1, like [0, 3, 5]
 * @param packedAcceptedDraftTokensIndices : Array of length seqAcceptedDraftTokenOffsets[seqCount], each value is in
 * range [0, maxDraftTokenCount - 1]
 * @param pastKeyValueLengths : Array of length seqCount, meaning how many tokens are already in KV cache
 * @param pastKeyValueList : Past key value list, which is the pointer array of each KVLinear cache.
 * @param layerCount : Count of layers
 * @param seqCount : Count of sequence
 * @param numKVHeads : Number of KV heads
 * @param sizeInBytesPerKVHead : Size of each KV head
 * @param rewindDraftTokenCounts : Pointer to an array of length seqCount, each element indicated the rewind count of
 * one sequence.
 * @param maxKVCacheLen : Maximum length of each KV cache
 * @param stream : CUDA stream to use.
 */
void updateLinearKVCacheDraftTokenLocationSeparateRewind(const int* seqAcceptedDraftTokenOffsets,
    const IndexType* packedAcceptedDraftTokensIndices, const int32_t* pastKeyValueLengths,
    int8_t* const* pastKeyValueList, int layerCount, int seqCount, int numKVHeads, int sizeInBytesPerKVHead,
    int* rewindDraftTokenCounts, int maxKVCacheLen, cudaStream_t stream);

/*!
 * Update Block KV cache using separate rewind count for each sequence.
 * @param seqAcceptedDraftTokenOffsets : Array of length seqCount + 1, like [0, 3, 5]
 * @param packedAcceptedDraftTokensIndices : Array of length seqAcceptedDraftTokenOffsets[seqCount], each value is in
 * range [0, maxDraftTokenCount - 1]
 * @param pastKeyValueLengths : Array of length seqCount, meaning how many tokens are already in KV cache
 * @param pointerArray : Pointer array of each Block KV cache.
 * @param layerCount : Count of layers
 * @param seqCount : Count of sequence
 * @param numKVHeads : Number of KV heads
 * @param sizeInBytesPerKVHead : Size of each KV head
 * @param rewindDraftTokenCounts : Pointer to an array of length seqCount, each element indicated the rewind count of
 * one sequence.
 * @param maxKVCacheLen : Maximum length of each KV cache
 * @param maxBlocksPerSeq : Maximum blocks per sequence of Block KV cache.
 * @param tokensPerBlock : Tokens per block of Block KV cache
 * @param stream : CUDA stream to use.
 */
void updateKVBlockArrayDraftTokenLocationSeparateRewind(const int* seqAcceptedDraftTokenOffsets,
    const IndexType* packedAcceptedDraftTokensIndices, const int32_t* pastKeyValueLengths, int64_t* const* pointerArray,
    int layerCount, int seqCount, int numKVHeads, int sizeInBytesPerKVHead, int* rewindDraftTokenCounts,
    int maxKVCacheLen, int maxBlocksPerSeq, int tokensPerBlock, cudaStream_t stream);

/*!
 * Update Linear KV cache using both common rewind and separate rewind count for each sequence. The common
 * rewindDraftTokenCommonCount and rewind count of each sequence in rewindDraftTokenSeparateAdjustments will be added
 * together for the final rewind count. It can save one add if both of them need to be used.
 * @param seqAcceptedDraftTokenOffsets : Array of length seqCount + 1, like [0, 3, 5]
 * @param packedAcceptedDraftTokensIndices : Array of length seqAcceptedDraftTokenOffsets[seqCount], each value is in
 * range [0, maxDraftTokenCount - 1]
 * @param pastKeyValueLengths : Array of length seqCount, meaning how many tokens are already in KV cache
 * @param pastKeyValueList : Past key value list, which is the pointer array of each KVLinear cache.
 * @param layerCount : Count of layers
 * @param seqCount : Count of sequence
 * @param numKVHeads : Number of KV heads
 * @param sizeInBytesPerKVHead : Size of each KV head
 * @param rewindDraftTokenCommonCount : Common token count to rewind
 * @param rewindDraftTokenSeparateAdjustments : Pointer to an array of length seqCount, each element indicated the
 * rewind adjustment for one sequence.
 * @param maxKVCacheLen : Maximum length of each KV cache
 * @param stream : CUDA stream to use.
 */
void updateLinearKVCacheDraftTokenLocation(const int* seqAcceptedDraftTokenOffsets,
    const IndexType* packedAcceptedDraftTokensIndices, const int32_t* pastKeyValueLengths,
    int8_t* const* pastKeyValueList, int layerCount, int seqCount, int numKVHeads, int sizeInBytesPerKVHead,
    int rewindDraftTokenCommonCount, int* rewindDraftTokenSeparateAdjustments, int maxKVCacheLen, cudaStream_t stream);

/*!
 * Update Block KV cache using both common rewind and separate rewind count for each sequence. The common
 * rewindDraftTokenCommonCount and rewind count of each sequence in rewindDraftTokenSeparateAdjustments will be added
 * together for the final rewind count. It can save one add if both of them need to be used.
 * @param seqAcceptedDraftTokenOffsets : Array of length seqCount + 1, like [0, 3, 5]
 * @param packedAcceptedDraftTokensIndices : Array of length seqAcceptedDraftTokenOffsets[seqCount], each value is in
 * range [0, maxDraftTokenCount - 1]
 * @param pastKeyValueLengths : Array of length seqCount, meaning how many tokens are already in KV cache
 * @param pointerArray : Pointer array of each Block KV cache.
 * @param layerCount : Count of layers
 * @param seqCount : Count of sequence
 * @param numKVHeads : Number of KV heads
 * @param sizeInBytesPerKVHead : Size of each KV head
 * @param rewindDraftTokenCommonCount : Common token count to rewind
 * @param rewindDraftTokenSeparateAdjustments : Pointer to an array of length seqCount, each element indicated the
 * rewind adjustment for one sequence.
 * @param maxKVCacheLen : Maximum length of each KV cache
 * @param maxBlocksPerSeq : Maximum blocks per sequence of Block KV cache.
 * @param tokensPerBlock : Tokens per block of Block KV cache
 * @param stream : CUDA stream to use.
 */
void updateKVBlockArrayDraftTokenLocation(const int* seqAcceptedDraftTokenOffsets,
    const IndexType* packedAcceptedDraftTokensIndices, const int32_t* pastKeyValueLengths, int64_t* const* pointerArray,
    int layerCount, int seqCount, int numKVHeads, int sizeInBytesPerKVHead, int rewindDraftTokenCommonCount,
    int* rewindDraftTokenSeparateAdjustments, int maxKVCacheLen, int maxBlocksPerSeq, int tokensPerBlock,
    cudaStream_t stream);

} // namespace tensorrt_llm::kernels::parallel_decoding
